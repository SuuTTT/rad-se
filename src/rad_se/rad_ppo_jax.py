#!/usr/bin/env python3
"""JAX/Flax PPO with RAD pixel augmentation on MuJoCo Playground.

Implements Option A from the RAD-vs-Playground design study: on-policy PPO with
a shared pixel encoder, exploiting Playground's massive env parallelism so that
the env+render cost dominates and we benefit from the >20× wall-clock speedup
that pure-PPO papers report on Playground.

Reuses from ``rad_jax.py``: image utilities, ``PixelEncoder``, ``Logger``,
``make_env``/``env_reset``/``env_step``.  Action representation is a squashed
Gaussian (mean = tanh of MLP output, std = exp(log_std)) so actions stay in
``[-1, 1]``.

Hyperparameters target a DM-Control pixel PPO baseline (DrQ-PPO style) with
RAD's random crop applied on every minibatch.

Usage:
  python rad_ppo_jax.py --env CartpoleSwingup --seed 23 --total-timesteps 1_000_000
  python rad_ppo_jax.py --env CartpoleSwingup --seed 23 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rad_se.rad_jax import (  # type: ignore
    Logger,
    PixelEncoder,
    center_crop_np,
    conv_out_size,
    env_reset,
    env_step,
    make_env,
    obs_to_uint8,
    random_crop_np,
    set_global_seed,
    uint8_to_float,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # experiment
    exp_name: str = "rad_ppo_jax"
    seed: int = 23
    # env
    env: str = "CartpoleSwingup"
    action_repeat: int = 8       # match RAD paper for fair comparison
    cam_res: int = 100
    image_size: int = 84
    max_episode_steps: int = 1000
    # parallel envs (Warp batch render)
    num_envs: int = 128
    # PPO rollout
    unroll_length: int = 16       # T per env per rollout (agent-steps after action_repeat)
    num_minibatches: int = 8
    update_epochs: int = 4
    # training
    total_timesteps: int = 1_000_000  # agent env-steps (post-action_repeat)
    eval_freq: int = 50_000
    num_eval_episodes: int = 10
    # PPO loss
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    reward_scale: float = 0.1     # scale rewards before storing (Playground sums over action_repeat)
    # network (same as RAD encoder)
    encoder_feature_dim: int = 50
    num_layers: int = 4
    num_filters: int = 32
    hidden_dim: int = 1024
    log_std_init: float = 0.0
    # logging
    log_interval: int = 1
    work_dir: str = "runs/rad_ppo_jax"
    track: bool = False
    wandb_project: str = "rad-se"
    wandb_entity: str = "sudingli21"
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_tags: str = ""
    # debug
    smoke: bool = False


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = PPOConfig()
    for key, val in asdict(defaults).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=val)
        else:
            parser.add_argument(flag, type=type(val), default=val)
    ns = parser.parse_args()
    cfg = PPOConfig(**vars(ns))
    if cfg.smoke:
        cfg.total_timesteps = 4096
        cfg.eval_freq = 2048
        cfg.num_envs = 16
        cfg.unroll_length = 8
        cfg.num_minibatches = 2
        cfg.update_epochs = 2
        cfg.log_interval = 1
    return cfg


# ---------------------------------------------------------------------------
# Actor-Critic with shared pixel encoder
# ---------------------------------------------------------------------------

class ActorCriticPPO(nnx.Module):
    """PPO actor-critic.

    Architecture: pixel obs -> shared PixelEncoder -> trunk MLP -> {policy head, value head}.

    Policy is a diagonal Gaussian with learnable per-action log_std (state-independent),
    and actions are squashed by tanh.  This is a standard DMC PPO choice that handles
    bounded action spaces correctly with a closed-form log-prob.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        action_dim: int,
        cfg: PPOConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.encoder = PixelEncoder(
            in_channels, image_size, cfg.encoder_feature_dim,
            cfg.num_layers, cfg.num_filters, output_logits=False, rngs=rngs,
        )
        # Shared trunk for both heads (cleanrl ppo_continuous convention)
        self.trunk = nnx.Sequential(
            nnx.Linear(cfg.encoder_feature_dim, cfg.hidden_dim, rngs=rngs),
            lambda x: jax.nn.tanh(x),
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs),
            lambda x: jax.nn.tanh(x),
        )
        self.actor_head = nnx.Linear(cfg.hidden_dim, action_dim, rngs=rngs)
        self.critic_head = nnx.Linear(cfg.hidden_dim, 1, rngs=rngs)
        # State-independent log_std parameter (standard PPO continuous)
        self.log_std = nnx.Param(
            jnp.full((action_dim,), float(cfg.log_std_init), dtype=jnp.float32)
        )

    def features(self, obs: jax.Array) -> jax.Array:
        h = self.encoder(obs)
        return self.trunk(h)

    def policy_value(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return (mean_raw, log_std, value).  Action = tanh(mean_raw + eps*exp(log_std))."""
        feat = self.features(obs)
        mean_raw = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        log_std = jnp.broadcast_to(self.log_std.value, mean_raw.shape)
        return mean_raw, log_std, value

    def value_only(self, obs: jax.Array) -> jax.Array:
        feat = self.features(obs)
        return self.critic_head(feat).squeeze(-1)


# ---------------------------------------------------------------------------
# Squashed-Gaussian helpers
# ---------------------------------------------------------------------------

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _gaussian_logprob(noise: jax.Array, log_std: jax.Array) -> jax.Array:
    """log N(eps; 0, exp(log_std))  summed over action dim, but evaluated at the
    pre-tanh sample; combined with tanh correction below."""
    return (-0.5 * noise ** 2 - log_std - _LOG_SQRT_2PI).sum(-1)


def _tanh_correction(pre_tanh: jax.Array) -> jax.Array:
    """log |det d tanh(x)/dx|^{-1} = sum log(1 - tanh(x)^2) — but numerically
    stable form: 2 * (log 2 - x - softplus(-2x))."""
    return (2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))).sum(-1)


def sample_action(mean_raw: jax.Array, log_std: jax.Array, rng: jax.Array):
    """Sample tanh-squashed Gaussian. Returns (action in [-1,1], pre-tanh, logp)."""
    noise = jax.random.normal(rng, mean_raw.shape)
    pre_tanh = mean_raw + noise * jnp.exp(log_std)
    action = jnp.tanh(pre_tanh)
    logp = _gaussian_logprob(noise, log_std) - _tanh_correction(pre_tanh)
    return action, pre_tanh, logp


def logp_from_pretanh(pre_tanh: jax.Array, mean_raw: jax.Array, log_std: jax.Array) -> jax.Array:
    """Recompute log-prob of a stored pre-tanh sample under new (mean,log_std)."""
    noise = (pre_tanh - mean_raw) / jnp.exp(log_std)
    return _gaussian_logprob(noise, log_std) - _tanh_correction(pre_tanh)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Differential entropy of N(0, exp(log_std)) per action dim (ignores tanh)."""
    return (log_std + _LOG_SQRT_2PI + 0.5).sum(-1)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,      # (T, N)
    values: np.ndarray,       # (T, N)
    dones: np.ndarray,        # (T, N)
    last_values: np.ndarray,  # (N,)
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute advantages and returns (CleanRL convention).

    last_values: V(s_T) — bootstrap from final next-state.
    dones[t]: 1 if episode terminated at t (s_{t+1} is fresh reset).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = last_values
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def ppo_loss_fn(
    model: ActorCriticPPO,
    obs: jax.Array,           # (B, H, W, 3) float [0,1]
    pre_tanh: jax.Array,      # (B, A)
    old_logp: jax.Array,      # (B,)
    advantages: jax.Array,    # (B,)
    returns: jax.Array,       # (B,)
    old_values: jax.Array,    # (B,)
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jax.Array, dict]:
    mean_raw, log_std, value = model.policy_value(obs)
    new_logp = logp_from_pretanh(pre_tanh, mean_raw, log_std)

    log_ratio = new_logp - old_logp
    ratio = jnp.exp(log_ratio)

    # Normalize advantages per minibatch
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv
    pg_loss = -jnp.minimum(surr1, surr2).mean()

    # Clipped value loss (PPO-style)
    v_clipped = old_values + jnp.clip(value - old_values, -clip_coef, clip_coef)
    vf_unclipped = (value - returns) ** 2
    vf_clipped = (v_clipped - returns) ** 2
    vf_loss = 0.5 * jnp.maximum(vf_unclipped, vf_clipped).mean()

    ent = gaussian_entropy(log_std).mean()
    loss = pg_loss + vf_coef * vf_loss - ent_coef * ent

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clipfrac = (jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32).mean()

    info = {
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": ent,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "ratio_mean": ratio.mean(),
    }
    return loss, info


@nnx.jit
def ppo_update_step(
    model: ActorCriticPPO,
    optimizer: nnx.Optimizer,
    obs: jax.Array,
    pre_tanh: jax.Array,
    old_logp: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    old_values: jax.Array,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
):
    def _loss_fn(m):
        return ppo_loss_fn(
            m, obs, pre_tanh, old_logp, advantages, returns, old_values,
            clip_coef, vf_coef, ent_coef,
        )

    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, info), grads = grad_fn(model)
    optimizer.update(model, grads)
    info["loss"] = loss
    return info


# ---------------------------------------------------------------------------
# Eval (greedy, deterministic action = tanh(mean_raw))
# ---------------------------------------------------------------------------

def evaluate(
    env,
    model: ActorCriticPPO,
    cfg: PPOConfig,
    run_dir: Path,
    step: int,
    logger: Logger,
    rng: jax.Array,
) -> tuple[float, jax.Array]:
    rng, reset_key = jax.random.split(rng)
    state, obs_u8 = env_reset(env, reset_key, cfg.num_envs)

    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    n_iter = max(1, cfg.max_episode_steps // cfg.action_repeat)

    for _ in range(n_iter):
        obs_crop = np.stack(
            [center_crop_np(obs_u8[i], cfg.image_size) for i in range(cfg.num_envs)],
            axis=0,
        )
        obs_jax = jnp.asarray(uint8_to_float(obs_crop))
        mean_raw, _, _ = model.policy_value(obs_jax)
        action = jnp.tanh(mean_raw)
        action_np = np.asarray(action, dtype=np.float32)
        state, obs_u8, reward, done, trunc = env_step(env, state, action_np)
        ep_returns += reward

    mean_r = float(np.mean(ep_returns))
    max_r = float(np.max(ep_returns))
    std_r = float(np.std(ep_returns))
    logger.log("eval/mean_episode_reward", mean_r, step)
    logger.log("eval/max_episode_reward", max_r, step)
    logger.log("eval/std_episode_reward", std_r, step)
    print(f"| eval | S: {step:>7d} | ER: {mean_r:.4f}", flush=True)
    return mean_r, rng


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    if "JAX_DEFAULT_MATMUL_PRECISION" not in os.environ:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

    # Warp mempool fix (also used by rad_jax.py)
    try:
        import warp as wp
        wp.set_mempool_release_threshold("cuda:0", 0)
        print("[rad_ppo_jax] Set Warp mempool release_threshold=0", flush=True)
    except Exception as _e:
        print(f"[rad_ppo_jax] Could not set Warp mempool threshold: {_e}", flush=True)

    set_global_seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    run_dir = Path(cfg.work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        # PPOConfig is a dataclass — reuse Logger which expects a Config-shaped thing
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    # Logger expects a Config with .track, .wandb_*, .env, .seed — PPOConfig has all
    logger = Logger(run_dir, cfg)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Env
    # ------------------------------------------------------------------
    print(f"[rad_ppo_jax] Building env: {cfg.env}  num_envs={cfg.num_envs}", flush=True)
    env, action_dim, max_ep_steps = make_env(cfg)  # type: ignore[arg-type]
    cfg.max_episode_steps = max_ep_steps

    print(f"[rad_ppo_jax] action_dim={action_dim}  max_ep_steps={max_ep_steps}  "
          f"obs (pre-crop)={cfg.cam_res}x{cfg.cam_res}x3", flush=True)

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    rngs = nnx.Rngs(cfg.seed)
    model = ActorCriticPPO(3, cfg.image_size, action_dim, cfg, rngs=rngs)

    # Iterations & batch sizing
    rollout_size = cfg.num_envs * cfg.unroll_length
    minibatch_size = rollout_size // cfg.num_minibatches
    n_iters = max(1, cfg.total_timesteps // rollout_size)
    print(f"[rad_ppo_jax] rollout_size={rollout_size}  minibatch_size={minibatch_size}  "
          f"n_iters={n_iters}", flush=True)

    # LR schedule
    if cfg.anneal_lr:
        total_updates = n_iters * cfg.update_epochs * cfg.num_minibatches
        lr_schedule = optax.linear_schedule(cfg.learning_rate, 0.0, total_updates)
    else:
        lr_schedule = cfg.learning_rate

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(lr_schedule, eps=1e-5),
        ),
        wrt=nnx.Param,
    )

    # ------------------------------------------------------------------
    # Rollout buffers (CPU numpy; transferred batch-wise during update)
    # ------------------------------------------------------------------
    T, N = cfg.unroll_length, cfg.num_envs
    # Store pre-crop uint8 obs to save memory
    buf_obs = np.zeros((T, N, cfg.cam_res, cfg.cam_res, 3), dtype=np.uint8)
    buf_pretanh = np.zeros((T, N, action_dim), dtype=np.float32)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_rew = np.zeros((T, N), dtype=np.float32)
    buf_done = np.zeros((T, N), dtype=np.float32)
    buf_val = np.zeros((T, N), dtype=np.float32)

    rng, reset_key = jax.random.split(rng)
    state, obs_u8 = env_reset(env, reset_key, N)  # (N, H, W, 3) uint8

    global_step = 0
    start_time = time.time()
    last_log_step = 0

    # Eval at step 0
    rng, eval_rng = jax.random.split(rng)
    _ = evaluate(env, model, cfg, run_dir, 0, logger, eval_rng)
    # Reset env state after eval (eval clobbers state)
    rng, reset_key = jax.random.split(rng)
    state, obs_u8 = env_reset(env, reset_key, N)
    next_eval = cfg.eval_freq

    for it in range(n_iters):
        # ------------------------------------------------------------------
        # Rollout collection (T agent-steps)
        # ------------------------------------------------------------------
        for t in range(T):
            # Center-crop for collection (deterministic; training uses random crop)
            obs_crop = np.stack(
                [center_crop_np(obs_u8[i], cfg.image_size) for i in range(N)], axis=0,
            )
            obs_jax = jnp.asarray(uint8_to_float(obs_crop))

            mean_raw, log_std, value = model.policy_value(obs_jax)
            rng, sub = jax.random.split(rng)
            action, pre_tanh, logp = sample_action(mean_raw, log_std, sub)

            action_np = np.asarray(action, dtype=np.float32)
            state, next_obs_u8, reward, done, trunc = env_step(env, state, action_np)

            buf_obs[t] = obs_u8
            buf_pretanh[t] = np.asarray(pre_tanh, dtype=np.float32)
            buf_logp[t] = np.asarray(logp, dtype=np.float32)
            buf_rew[t] = reward * cfg.reward_scale
            # Treat both true done and truncation as "episode boundary" for GAE bootstrap
            buf_done[t] = np.maximum(done, trunc)
            buf_val[t] = np.asarray(value, dtype=np.float32)

            obs_u8 = next_obs_u8
            global_step += N

        # Bootstrap V(s_T)
        obs_crop = np.stack(
            [center_crop_np(obs_u8[i], cfg.image_size) for i in range(N)], axis=0,
        )
        obs_jax = jnp.asarray(uint8_to_float(obs_crop))
        last_values = np.asarray(model.value_only(obs_jax), dtype=np.float32)

        # GAE
        adv, returns = compute_gae(
            buf_rew, buf_val, buf_done, last_values, cfg.discount, cfg.gae_lambda,
        )

        # ------------------------------------------------------------------
        # PPO update (K epochs × M minibatches; RAD crop applied per minibatch)
        # ------------------------------------------------------------------
        # Flatten time × envs → batch axis
        flat_obs_u8 = buf_obs.reshape(rollout_size, cfg.cam_res, cfg.cam_res, 3)
        flat_pretanh = buf_pretanh.reshape(rollout_size, action_dim)
        flat_logp = buf_logp.reshape(rollout_size)
        flat_val = buf_val.reshape(rollout_size)
        flat_adv = adv.reshape(rollout_size)
        flat_ret = returns.reshape(rollout_size)

        last_info = {}
        for epoch in range(cfg.update_epochs):
            perm = np.random.permutation(rollout_size)
            for mb_start in range(0, rollout_size, minibatch_size):
                mb_idx = perm[mb_start: mb_start + minibatch_size]
                # RAD: random crop each minibatch (the entire RAD trick)
                mb_obs_u8 = flat_obs_u8[mb_idx]                        # (M, 100, 100, 3) uint8
                mb_obs_crop = random_crop_np(mb_obs_u8, cfg.image_size)  # (M, 84, 84, 3)
                mb_obs_f = uint8_to_float(mb_obs_crop)

                mb_obs_jax = jnp.asarray(mb_obs_f)
                mb_pretanh = jnp.asarray(flat_pretanh[mb_idx])
                mb_logp = jnp.asarray(flat_logp[mb_idx])
                mb_adv = jnp.asarray(flat_adv[mb_idx])
                mb_ret = jnp.asarray(flat_ret[mb_idx])
                mb_val = jnp.asarray(flat_val[mb_idx])

                last_info = ppo_update_step(
                    model, optimizer,
                    mb_obs_jax, mb_pretanh, mb_logp,
                    mb_adv, mb_ret, mb_val,
                    cfg.clip_coef, cfg.vf_coef, cfg.ent_coef,
                )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if (it + 1) % cfg.log_interval == 0 or it == 0:
            elapsed = time.time() - start_time
            sps = global_step / max(1e-6, elapsed)
            eta_s = (cfg.total_timesteps - global_step) / max(1e-6, sps)
            eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{int(eta_s/60)}m"
            print(
                f"| train | iter:{it+1:>4d}/{n_iters} | S:{global_step:>8d}/{cfg.total_timesteps} | "
                f"SPS:{sps:6.0f} | pg:{float(last_info.get('pg_loss', 0)):+.3f} | "
                f"vf:{float(last_info.get('vf_loss', 0)):.2f} | "
                f"ent:{float(last_info.get('entropy', 0)):+.2f} | "
                f"kl:{float(last_info.get('approx_kl', 0)):.4f} | "
                f"cf:{float(last_info.get('clipfrac', 0)):.2f} | "
                f"R:{float(buf_rew.sum()/N):+.1f} | ETA:{eta_str}",
                flush=True,
            )
            logger.log("train/sps", sps, global_step)
            logger.log("train/pg_loss", float(last_info.get("pg_loss", 0)), global_step)
            logger.log("train/vf_loss", float(last_info.get("vf_loss", 0)), global_step)
            logger.log("train/entropy", float(last_info.get("entropy", 0)), global_step)
            logger.log("train/approx_kl", float(last_info.get("approx_kl", 0)), global_step)
            logger.log("train/clipfrac", float(last_info.get("clipfrac", 0)), global_step)
            logger.log("train/rollout_reward_sum", float(buf_rew.sum() / N), global_step)
            last_log_step = global_step

        # Eval
        if global_step >= next_eval:
            rng, eval_rng = jax.random.split(rng)
            _ = evaluate(env, model, cfg, run_dir, global_step, logger, eval_rng)
            rng, reset_key = jax.random.split(rng)
            state, obs_u8 = env_reset(env, reset_key, N)
            next_eval += cfg.eval_freq

    # Final eval
    rng, eval_rng = jax.random.split(rng)
    _ = evaluate(env, model, cfg, run_dir, global_step, logger, eval_rng)
    logger.finish()


if __name__ == "__main__":
    main()
