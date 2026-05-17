#!/usr/bin/env python3
"""RAD-style SAC using brax losses + on-device replay buffer on MuJoCo Playground.

brax.training.agents.sac.train does not support dict/pixel observations
(raises NotImplementedError). This file re-uses the official brax SAC loss
functions and replay buffer, but implements the training loop itself so that:
  - Pixel observations (dict with 'pixels/*' keys) are supported.
  - RAD random-translate augmentation is applied at sample time.
  - Replay buffer stays on-device (UniformSamplingQueue) — no H2D transfer.
  - Training epoch is a single jax.jit call with inner jax.lax.scan.

Memory constraints on RTX 3060 12 GB:
  max_replay_size=10000 × 2 × (100×100×3) float32 ≈ 2.4 GB is safe.
  Increase to 20000 (≈4.8 GB) if you have headroom after init.

Usage:
  python src/rad_se/rad_brax_sac.py --env CartpoleSwingup --seed 23
  python src/rad_se/rad_brax_sac.py --env CartpoleSwingup --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen

# Compatibility shim: brax 0.14.2 calls deprecated jax.device_put_replicated
# (removed in JAX 0.10+). Restore it before importing brax.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(pytree, devices):
        n = len(devices)
        def replicate_leaf(x):
            x = jnp.asarray(x)
            return jnp.broadcast_to(jnp.expand_dims(x, 0), (n, *x.shape))
        replicated = jax.tree_util.tree_map(replicate_leaf, pytree)
        if n == 1:
            return jax.device_put(replicated, devices[0])
        raise NotImplementedError("device_put_replicated shim: single-device only")
    jax.device_put_replicated = _device_put_replicated

from brax.training import distribution, gradients, networks as brax_networks, types
from brax.training.agents.sac import losses as sac_losses, networks as sac_networks
from brax.training import replay_buffers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    env: str = "CartpoleSwingup"
    seed: int = 23
    # env
    action_repeat: int = 8
    cam_res: int = 100
    episode_length: int = 1000          # physics steps
    num_envs: int = 8                   # SAC: small env count, large replay
    num_eval_envs: int = 16
    # replay buffer
    max_replay_size: int = 10_000       # ~2.4 GB float32 pixel obs
    min_replay_size: int = 1_000        # warmup steps before updates start
    batch_size: int = 256               # SAC minibatch from replay
    grad_updates_per_step: int = 1
    # SAC hypers
    total_timesteps: int = 500_000
    num_evals: int = 20
    discounting: float = 0.99
    learning_rate: float = 3e-4
    reward_scaling: float = 0.1
    tau: float = 0.005                  # target network EMA coefficient
    # CNN backbone (RAD-style DQN encoder, same as PPO)
    cnn_output_channels: tuple = (32, 64, 64)
    cnn_kernel_size: tuple = (8, 4, 3)
    cnn_stride: tuple = (4, 2, 1)
    cnn_padding: str = "VALID"
    cnn_activation: str = "relu"
    cnn_max_pool: bool = False
    cnn_global_pool: str = "avg"
    # Policy / critic MLPs after CNN
    policy_hidden: tuple = (1024, 1024)
    critic_hidden: tuple = (1024, 1024)
    # Augmentation
    augment_pixels: bool = True         # RAD random-translate at sample time
    # logging
    work_dir: str = "runs/brax_sac"
    track: bool = False
    wandb_project: str = "rad-se"
    wandb_entity: str = ""
    smoke: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    cfg = Config()
    for k, v in asdict(cfg).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, tuple):
            parser.add_argument(flag, nargs="+", type=type(v[0]), default=list(v))
        else:
            parser.add_argument(flag, type=type(v), default=v)
    ns = parser.parse_args()
    d = vars(ns)
    for k, v in asdict(cfg).items():
        if isinstance(v, tuple):
            d[k] = tuple(d[k])
    if d["smoke"]:
        d["total_timesteps"] = 5_000
        d["num_envs"] = 2
        d["batch_size"] = 32
        d["max_replay_size"] = 500
        d["min_replay_size"] = 100
        d["num_evals"] = 2
        d["num_eval_envs"] = 4
    return Config(**d)


# ---------------------------------------------------------------------------
# CartpoleSwingup done-condition fix (identical to rad_brax_ppo.py)
# ---------------------------------------------------------------------------

def patch_swingup_done():
    from mujoco_playground._src.dm_control_suite import cartpole as _cp_mod
    if getattr(_cp_mod.Balance, "_rad_se_patched", False):
        return
    _orig = _cp_mod.Balance.step
    def _fixed_step(self_env, state, action):
        nstate = _orig(self_env, state, action)
        if getattr(self_env, "_fix_swingup_done", False):
            data = nstate.data
            cart_pos = data.qpos[self_env._slider_qposadr]
            done = (
                jnp.isnan(data.qpos).any()
                | jnp.isnan(data.qvel).any()
                | (jnp.abs(cart_pos) > 3.0)
            ).astype(jnp.float32)
            nstate.info["time_out"] = done
            nstate = nstate.replace(done=done)
        return nstate
    _cp_mod.Balance.step = _fixed_step
    _cp_mod.Balance._rad_se_patched = True


# ---------------------------------------------------------------------------
# RAD augmentation for SAC: [B, H, W, C] pixel observations
# Unlike the PPO version (B×T×H×W×C), SAC stores single-step transitions.
# ---------------------------------------------------------------------------

def _random_translate_pixels_sac(
    obs: Mapping[str, jax.Array],
    key: jax.Array,
    padding: int = 4,
) -> Mapping[str, jax.Array]:
    """Apply random pad-4 translations to [B, H, W, C] pixel observations."""
    @jax.vmap
    def rt_one(ub_obs: Mapping[str, jax.Array], rng: jax.Array):
        def rt_view(img: jax.Array, rng: jax.Array) -> jax.Array:
            # img: [H, W, C]
            h_off = jax.random.randint(rng, (), 0, 2 * padding + 1)
            _, rng = jax.random.split(rng)
            w_off = jax.random.randint(rng, (), 0, 2 * padding + 1)
            padded = jnp.pad(
                img,
                ((padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            start = jnp.array([h_off, w_off, 0], dtype=jnp.int32)
            return jax.lax.dynamic_slice(padded, start, img.shape)

        out = {}
        for k, v in ub_obs.items():
            if k.startswith("pixels/"):
                rng, subkey = jax.random.split(rng)
                out[k] = rt_view(v, subkey)
        return {**ub_obs, **out}

    bdim = next(iter(obs.values())).shape[0]
    keys = jax.random.split(key, bdim)
    return rt_one(obs, keys)


# ---------------------------------------------------------------------------
# Vision SAC networks: policy via brax, Q-network custom (CNN + action + MLP)
# ---------------------------------------------------------------------------

class VisionQModule(linen.Module):
    """Twin-Q critic for pixel obs. Each critic has its own CNN encoder."""
    n_critics: int = 2
    hidden_layer_sizes: Sequence[int] = (1024, 1024)
    cnn_output_channels: Sequence[int] = (32, 64, 64)
    cnn_kernel_size: Sequence[int] = (8, 4, 3)
    cnn_stride: Sequence[int] = (4, 2, 1)
    cnn_padding: str = "VALID"
    cnn_activation: linen.activation.PReLU = linen.relu
    cnn_max_pool: bool = False
    cnn_global_pool: str = "avg"
    activation: Any = linen.relu

    @linen.compact
    def __call__(self, obs: Mapping[str, jax.Array], actions: jax.Array):
        pixels = {k: v for k, v in obs.items() if k.startswith("pixels/")}
        kernel_sizes = tuple((k, k) for k in self.cnn_kernel_size)
        strides = tuple((s, s) for s in self.cnn_stride)

        q_vals = []
        for _ in range(self.n_critics):
            # Independent CNN encoder per critic
            cnn_outs = []
            for pkey in sorted(pixels.keys()):
                cnn_out = brax_networks.CNN(
                    num_filters=self.cnn_output_channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    activation=self.cnn_activation,
                    padding=self.cnn_padding,
                    max_pool=self.cnn_max_pool,
                )(pixels[pkey])
                if self.cnn_global_pool == "avg":
                    cnn_out = jnp.mean(cnn_out, axis=(-3, -2))
                elif self.cnn_global_pool == "max":
                    cnn_out = jnp.max(cnn_out, axis=(-3, -2))
                else:  # flatten
                    cnn_out = cnn_out.reshape(cnn_out.shape[0], -1)
                cnn_outs.append(cnn_out)

            embed = jnp.concatenate(cnn_outs + [actions], axis=-1)
            q = brax_networks.MLP(
                layer_sizes=list(self.hidden_layer_sizes) + [1],
                activation=self.activation,
            )(embed)
            q_vals.append(q)

        return jnp.concatenate(q_vals, axis=-1)  # [B, n_critics]


def make_sac_networks_vision(
    obs_size: Mapping[str, Tuple],   # per-env, e.g. {'pixels/view_0': (H,W,C)}
    action_size: int,
    cfg: Config,
) -> sac_networks.SACNetworks:
    """Create policy + Q-network for pixel SAC."""
    act = linen.relu if cfg.cnn_activation == "relu" else linen.swish

    # Policy network: reuse brax's make_policy_network_vision
    policy_net = brax_networks.make_policy_network_vision(
        observation_size=obs_size,
        output_size=distribution.NormalTanhDistribution(action_size).param_size,
        hidden_layer_sizes=cfg.policy_hidden,
        activation=act,
        cnn_output_channels=cfg.cnn_output_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_stride=cfg.cnn_stride,
        cnn_padding=cfg.cnn_padding,
        cnn_activation=act,
        cnn_max_pool=cfg.cnn_max_pool,
        cnn_global_pool=cfg.cnn_global_pool,
        distribution_type="tanh_normal",
    )

    # Q-network: custom twin-Q with independent CNN encoders
    q_module = VisionQModule(
        n_critics=2,
        hidden_layer_sizes=cfg.critic_hidden,
        cnn_output_channels=cfg.cnn_output_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_stride=cfg.cnn_stride,
        cnn_padding=cfg.cnn_padding,
        cnn_activation=act,
        cnn_max_pool=cfg.cnn_max_pool,
        cnn_global_pool=cfg.cnn_global_pool,
        activation=act,
    )
    dummy_obs = {k: jnp.zeros((1,) + v) for k, v in obs_size.items()}
    dummy_act = jnp.zeros((1, action_size))

    def q_apply(processor_params, q_params, obs, actions):
        # processor_params unused for pixels (no obs normalization)
        return q_module.apply(q_params, obs, actions)

    q_net = brax_networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_act),
        apply=q_apply,
    )

    return sac_networks.SACNetworks(
        policy_network=policy_net,
        q_network=q_net,
        parametric_action_distribution=distribution.NormalTanhDistribution(action_size),
    )


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

from flax import struct as flax_struct

@flax_struct.dataclass
class SACTrainingState:
    policy_params: Any
    policy_optimizer_state: Any
    q_params: Any
    q_optimizer_state: Any
    target_q_params: Any
    log_alpha: jax.Array
    alpha_optimizer_state: Any
    normalizer_params: Any          # dummy (pixels not normalized)
    env_steps: jax.Array


# ---------------------------------------------------------------------------
# Environment builder (same pattern as rad_brax_ppo.py)
# ---------------------------------------------------------------------------

def make_envs(cfg: Config, num_envs: int, is_eval: bool = False):
    from mujoco_playground._src import dm_control_suite
    from mujoco_playground import wrapper as mp_wrapper
    patch_swingup_done()
    env_config = dm_control_suite.get_default_config(cfg.env)
    env_config.vision = True
    env_config.vision_config.cam_res = (cfg.cam_res, cfg.cam_res)
    env_config.vision_config.nworld = num_envs
    raw_env = dm_control_suite.load(cfg.env, config=env_config)
    if "Swingup" in cfg.env or "swingup" in cfg.env:
        raw_env._fix_swingup_done = True
    agent_episode_length = cfg.episode_length // cfg.action_repeat
    return mp_wrapper.wrap_for_brax_training(
        raw_env,
        episode_length=agent_episode_length,
        action_repeat=cfg.action_repeat,
    )


# ---------------------------------------------------------------------------
# Logging (same as PPO)
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, run_dir: Path, cfg: Config):
        self.run_dir = run_dir
        self._wandb = None
        if cfg.track:
            import wandb
            self._wandb = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity,
                config=asdict(cfg), name=run_dir.name)
        (run_dir / "metrics.jsonl").open("w").close()

    def log(self, step: int, metrics: dict):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        with (self.run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self._wandb:
            self._wandb.log(row, step=step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    rng = jax.random.PRNGKey(cfg.seed)
    run_dir = Path(cfg.work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger = Logger(run_dir, cfg)

    print(f"[rad_brax_sac] {cfg.env} seed={cfg.seed} work_dir={run_dir}")

    # Build envs
    train_env = make_envs(cfg, cfg.num_envs)
    eval_env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)

    # Obs / action sizes
    full_obs_size = train_env.observation_size   # includes batch dim: {k: (N,H,W,C)}
    obs_size = {k: v[1:] for k, v in full_obs_size.items()}  # {k: (H,W,C)}
    action_size = train_env.action_size

    print(f"  obs_size (per env): {obs_size}")
    print(f"  action_size: {action_size}")

    # Networks + losses
    sac_net = make_sac_networks_vision(obs_size, action_size, cfg)
    make_policy = sac_networks.make_inference_fn(sac_net)

    alpha_loss_fn, critic_loss_fn, actor_loss_fn = sac_losses.make_losses(
        sac_network=sac_net,
        reward_scaling=cfg.reward_scaling,
        discounting=cfg.discounting,
        action_size=action_size,
    )
    alpha_lr = 3e-4
    policy_optimizer = optax.adam(cfg.learning_rate)
    q_optimizer = optax.adam(cfg.learning_rate)
    alpha_optimizer = optax.adam(alpha_lr)

    alpha_update = gradients.gradient_update_fn(
        alpha_loss_fn, alpha_optimizer, pmap_axis_name=None)
    critic_update = gradients.gradient_update_fn(
        critic_loss_fn, q_optimizer, pmap_axis_name=None)
    actor_update = gradients.gradient_update_fn(
        actor_loss_fn, policy_optimizer, pmap_axis_name=None)

    # Replay buffer: flat storage, per-env unbatched dummy
    dummy_obs = {k: jnp.zeros(v) for k, v in obs_size.items()}
    dummy_action = jnp.zeros(action_size)
    dummy_transition = types.Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": jnp.zeros(())}},
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=cfg.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=cfg.batch_size,
    )

    # Initialize training state
    rng, key_policy, key_q, key_buf = jax.random.split(rng, 4)
    policy_params = sac_net.policy_network.init(key_policy)
    q_params = sac_net.q_network.init(key_q)
    training_state = SACTrainingState(
        policy_params=policy_params,
        policy_optimizer_state=policy_optimizer.init(policy_params),
        q_params=q_params,
        q_optimizer_state=q_optimizer.init(q_params),
        target_q_params=jax.tree_util.tree_map(jnp.array, q_params),
        log_alpha=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer.init(jnp.zeros(())),
        normalizer_params=jnp.zeros(()),     # dummy: pixels not normalized
        env_steps=jnp.zeros((), dtype=jnp.int32),
    )

    # Initialize env state and replay buffer
    rng, key_env, key_eval = jax.random.split(rng, 3)
    env_keys = jax.random.split(key_env, cfg.num_envs)
    env_state = jax.jit(train_env.reset)(env_keys)
    buffer_state = replay_buffer.init(key_buf)

    # ------------------------------------------------------------------ #
    # Inner step fn: collect 1 env step + (optionally) 1 gradient update #
    # ------------------------------------------------------------------ #

    def sgd_step(ts: SACTrainingState, transitions: types.Transition, key: jax.Array):
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
        alpha = jnp.exp(ts.log_alpha)

        alpha_loss, new_log_alpha, new_alpha_opt = alpha_update(
            ts.log_alpha, ts.policy_params, ts.normalizer_params,
            transitions, key_alpha, optimizer_state=ts.alpha_optimizer_state)
        critic_loss, new_q_params, new_q_opt = critic_update(
            ts.q_params, ts.policy_params, ts.normalizer_params,
            ts.target_q_params, alpha, transitions, key_critic,
            optimizer_state=ts.q_optimizer_state)
        actor_loss, new_policy_params, new_policy_opt = actor_update(
            ts.policy_params, ts.normalizer_params, ts.q_params, alpha,
            transitions, key_actor, optimizer_state=ts.policy_optimizer_state)
        new_target_q = jax.tree_util.tree_map(
            lambda x, y: x * (1 - cfg.tau) + y * cfg.tau,
            ts.target_q_params, new_q_params)

        return ts.replace(
            policy_params=new_policy_params,
            policy_optimizer_state=new_policy_opt,
            q_params=new_q_params,
            q_optimizer_state=new_q_opt,
            target_q_params=new_target_q,
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt,
        ), {"alpha_loss": alpha_loss, "critic_loss": critic_loss, "actor_loss": actor_loss}

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2), static_argnums=(4,))
    def training_epoch(ts, env_st, buf_st, key, steps):
        """One epoch: `steps` env-steps + SAC updates."""
        def step_fn(carry, _):
            ts, env_st, buf_st, key = carry
            key, act_key, aug_key, upd_key = jax.random.split(key, 4)

            # Collect: infer action from current obs
            obs = env_st.obs
            dist_params = sac_net.policy_network.apply(
                ts.normalizer_params, ts.policy_params, obs)
            action = sac_net.parametric_action_distribution.sample(dist_params, act_key)
            new_env_st = train_env.step(env_st, action)

            # Store transition (num_envs transitions per step)
            transition = types.Transition(
                observation=obs,
                action=action,
                reward=new_env_st.reward * cfg.reward_scaling,
                discount=1.0 - new_env_st.done,
                next_observation=new_env_st.obs,
                extras={"state_extras": {"truncation": new_env_st.info.get("truncation",
                                                           jnp.zeros(cfg.num_envs))}},
            )
            buf_st = replay_buffer.insert(buf_st, transition)

            # Update: sample + augment + sgd
            buf_st, sampled = replay_buffer.sample(buf_st)
            if cfg.augment_pixels:
                key, key_aug_obs, key_aug_nobs = jax.random.split(key, 3)
                sampled = sampled._replace(
                    observation=_random_translate_pixels_sac(
                        sampled.observation, key_aug_obs),
                    next_observation=_random_translate_pixels_sac(
                        sampled.next_observation, key_aug_nobs),
                )
            ts, metrics = sgd_step(ts, sampled, upd_key)
            ts = ts.replace(env_steps=ts.env_steps + cfg.num_envs)
            return (ts, new_env_st, buf_st, key), metrics

        (ts, env_st, buf_st, key), metrics = jax.lax.scan(
            step_fn, (ts, env_st, buf_st, key), None, length=steps)
        return ts, env_st, buf_st, key, metrics

    # Prefill: collect min_replay_size transitions without updates
    @jax.jit
    def prefill_step(env_st, buf_st, key):
        key, act_key = jax.random.split(key)
        action = jax.random.uniform(
            act_key, (cfg.num_envs, action_size), minval=-1.0, maxval=1.0)
        new_env_st = train_env.step(env_st, action)
        transition = types.Transition(
            observation=env_st.obs,
            action=action,
            reward=new_env_st.reward * cfg.reward_scaling,
            discount=1.0 - new_env_st.done,
            next_observation=new_env_st.obs,
            extras={"state_extras": {"truncation": new_env_st.info.get(
                "truncation", jnp.zeros(cfg.num_envs))}},
        )
        buf_st = replay_buffer.insert(buf_st, transition)
        return new_env_st, buf_st, key

    # ---------------------------------------------------------------- #
    # Warmup: fill replay buffer with random actions                    #
    # ---------------------------------------------------------------- #
    print(f"  Prefilling replay buffer ({cfg.min_replay_size} transitions)…")
    warmup_steps = max(cfg.min_replay_size // cfg.num_envs, 1)
    for _ in range(warmup_steps):
        rng, step_key = jax.random.split(rng)
        env_state, buffer_state, rng = prefill_step(env_state, buffer_state, rng)

    print(f"  Replay buffer ready. Starting training…")

    # ---------------------------------------------------------------- #
    # Main training loop                                                #
    # ---------------------------------------------------------------- #
    # total_env_steps: total wrapped env.step() calls across all parallel envs.
    # action_repeat is already handled by the wrapper (each env.step() = action_repeat
    # physics steps), so we do NOT divide by action_repeat here.
    agent_ep_len = cfg.episode_length // cfg.action_repeat
    total_env_steps = cfg.total_timesteps  # env.step() calls total
    # scan_steps_per_eval: number of jax.lax.scan iterations per eval epoch.
    # env_steps increments by num_envs per scan step, so we divide by num_envs.
    scan_steps_per_eval = max(cfg.total_timesteps // cfg.num_evals // cfg.num_envs, 1)

    t0 = time.time()
    last_log = t0
    eval_count = 0

    while int(training_state.env_steps) < total_env_steps:
        remaining_env_steps = total_env_steps - int(training_state.env_steps)
        steps_to_run = min(scan_steps_per_eval, remaining_env_steps // cfg.num_envs)
        if steps_to_run <= 0:
            break
        rng, epoch_key = jax.random.split(rng)
        training_state, env_state, buffer_state, rng, epoch_metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key, steps_to_run)

        # Eval
        rng, eval_key = jax.random.split(rng)
        # Reconstruct policy with current params for eval
        eval_policy = make_policy(
            (training_state.normalizer_params, training_state.policy_params),
            deterministic=True)
        eval_st = jax.jit(eval_env.reset)(jax.random.split(eval_key, cfg.num_eval_envs))

        def _do_eval_step(carry, _):
            st, k = carry
            k, sk = jax.random.split(k)
            a, _ = eval_policy(st.obs, sk)
            st = eval_env.step(st, a)
            return (st, k), st.reward

        (_, _), rew = jax.lax.scan(
            _do_eval_step, (eval_st, eval_key), None,
            length=agent_ep_len)
        er = float(jnp.sum(rew, axis=0).mean())

        elapsed = time.time() - t0
        env_steps = int(training_state.env_steps)
        sps = int(env_steps / elapsed) if elapsed > 1 else 0
        avg_critic = float(jnp.mean(epoch_metrics["critic_loss"]))
        avg_actor = float(jnp.mean(epoch_metrics["actor_loss"]))
        alpha = float(jnp.exp(training_state.log_alpha))

        print(f"| brax_sac | S:{env_steps:8d} | SPS:{sps:5d} | "
              f"ER:{er:8.2f} | critic:{avg_critic:.3f} | "
              f"actor:{avg_actor:.4f} | alpha:{alpha:.3f} | elapsed:{elapsed:.1f}s")

        logger.log(env_steps, {
            "sps": sps,
            "elapsed": elapsed,
            "eval/episode_reward": er,
            "training/critic_loss": avg_critic,
            "training/actor_loss": avg_actor,
            "training/alpha": alpha,
        })
        eval_count += 1

    elapsed = time.time() - t0
    print(f"[done] {int(training_state.env_steps)} steps in {elapsed:.1f}s "
          f"({int(training_state.env_steps)/elapsed:.0f} SPS)")


if __name__ == "__main__":
    main()
