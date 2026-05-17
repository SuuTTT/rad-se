#!/usr/bin/env python3
"""RAD via brax PPO + augment_pixels on MuJoCo Playground.

Uses the official brax.training.agents.ppo.train (fully JIT-compiled
jax.lax.scan training loop) with vision=True and augment_pixels=True.

Key differences vs our hand-written rad_ppo_jax.py:
  - The entire rollout collection + PPO update is a single jax.jit kernel.
    No Python loop overhead. Expected throughput: 500-5000 SPS vs 55 SPS.
  - augment_pixels=True injects random-translate (pad-4 crop) into every
    gradient step — the RAD augmentation — without leaving device memory.
  - action_repeat is handled at the Playground env wrapper level so brax's
    vision constraint (action_repeat==1 required) is respected.

Usage:
  python src/rad_se/rad_brax_ppo.py --env CartpoleSwingup --seed 23
  python src/rad_se/rad_brax_ppo.py --env CartpoleSwingup --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
import functools
from dataclasses import asdict, dataclass
from pathlib import Path

# Suppress JAX preallocate so Warp and JAX share the pool gracefully.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np

# Compatibility shim: brax 0.14.2 uses deprecated jax.device_put_replicated
# which was removed in JAX 0.10+. Restore it before importing brax.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(pytree, devices):
        """Compatibility shim: brax 0.14.2 uses deprecated jax.device_put_replicated.
        Adds a leading device axis of size len(devices) so _unpmap's .squeeze(0) works.
        Only tested/used in the single-device case (RTX 3060).
        """
        n = len(devices)
        def replicate_leaf(x):
            x = jnp.asarray(x)
            # scalar () → (n,); array (d1,...) → (n, d1, ...)
            return jnp.broadcast_to(jnp.expand_dims(x, 0), (n, *x.shape))
        replicated = jax.tree_util.tree_map(replicate_leaf, pytree)
        if n == 1:
            return jax.device_put(replicated, devices[0])
        raise NotImplementedError(
            "device_put_replicated shim only supports single-device; "
            "upgrade brax or use a single GPU."
        )
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo_train


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # experiment
    env: str = "CartpoleSwingup"
    seed: int = 23
    # env setup
    action_repeat: int = 8      # handled at Playground wrapper level
    cam_res: int = 100           # Warp render resolution (pre-crop)
    episode_length: int = 1000  # physics steps (before action_repeat)
    # PPO / brax params
    num_envs: int = 256          # parallel worlds (Warp nworld)
    unroll_length: int = 20      # rollout horizon per env
    batch_size: int = 32         # transitions per minibatch
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
    total_timesteps: int = 500_000
    num_evals: int = 20          # eval checkpoints during training
    num_eval_envs: int = 32
    discounting: float = 0.99
    learning_rate: float = 3e-4
    entropy_cost: float = 0.01
    clipping_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    reward_scaling: float = 0.1
    normalize_observations: bool = False  # pixel obs: don't normalize
    gae_lambda: float = 0.95
    # CNN backbone (RAD-style DQN encoder)
    cnn_output_channels: tuple = (32, 64, 64)
    cnn_kernel_size: tuple = (8, 4, 3)
    cnn_stride: tuple = (4, 2, 1)
    cnn_padding: str = "valid"
    cnn_activation: str = "relu"
    cnn_max_pool: bool = False
    cnn_global_pool: str = "avg"
    # Policy / value MLP after CNN
    policy_hidden: tuple = (1024, 1024)
    value_hidden: tuple = (1024, 1024)
    # logging
    work_dir: str = "runs/brax_ppo"
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
    # Convert lists back to tuples for tuple fields
    for k, v in asdict(cfg).items():
        if isinstance(v, tuple):
            d[k] = tuple(d[k])
    if d["smoke"]:
        d["total_timesteps"] = 10_000
        d["num_envs"] = 32
        d["batch_size"] = 8
        d["num_minibatches"] = 4
        d["unroll_length"] = 10
        d["num_evals"] = 2
        d["num_eval_envs"] = 8
    return Config(**d)


# ---------------------------------------------------------------------------
# CartpoleSwingup done-condition fix
# ---------------------------------------------------------------------------

def patch_swingup_done():
    """Playground CartpoleSwingup fires done=True immediately without this fix.

    Balance.step checks pole_angle > pi/2, but SwingUp starts at angle ≈ pi
    (bottom position). We remove the angle check for SwingUp instances.
    """
    from mujoco_playground._src.dm_control_suite import cartpole as _cp_mod

    if getattr(_cp_mod.Balance, "_rad_se_patched", False):
        return  # already applied

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
# Environment builder
# ---------------------------------------------------------------------------

def make_envs(cfg: Config, num_envs: int, is_eval: bool = False):
    """Build a Playground env pre-wrapped with Brax wrappers.

    Returns a fully wrapped env ready to pass to ppo.train with wrap_env=False.
    action_repeat=cfg.action_repeat is folded in here so brax PPO gets
    action_repeat=1 (required for vision mode).
    """
    from mujoco_playground._src import dm_control_suite
    from mujoco_playground import wrapper as mp_wrapper

    patch_swingup_done()

    env_config = dm_control_suite.get_default_config(cfg.env)
    env_config.vision = True
    env_config.vision_config.cam_res = (cfg.cam_res, cfg.cam_res)
    env_config.vision_config.nworld = num_envs

    raw_env = dm_control_suite.load(cfg.env, config=env_config)

    # Mark SwingUp variants so the patch can fire
    if "Swingup" in cfg.env or "swingup" in cfg.env:
        raw_env._fix_swingup_done = True

    # Agent steps per episode = physics_steps / action_repeat
    agent_episode_length = cfg.episode_length // cfg.action_repeat

    # wrap_for_brax_training adds VmapWrapper, EpisodeWrapper, AutoResetWrapper
    wrapped = mp_wrapper.wrap_for_brax_training(
        raw_env,
        episode_length=agent_episode_length,
        action_repeat=cfg.action_repeat,
    )
    return wrapped


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, run_dir: Path, cfg: Config):
        self._path = run_dir / "metrics.jsonl"
        self._wandb = None
        if cfg.track:
            import wandb
            self._wandb = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=f"brax_ppo_{cfg.env}__s{cfg.seed}",
                group=f"brax_ppo__{cfg.env}",
                config=asdict(cfg),
            )

    def log(self, step: int, metrics: dict):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        with open(self._path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self._wandb is not None:
            self._wandb.log(row, step=step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args()

    run_dir = Path(cfg.work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger = Logger(run_dir, cfg)

    print(f"[rad_brax_ppo] env={cfg.env} seed={cfg.seed} "
          f"num_envs={cfg.num_envs} total={cfg.total_timesteps}")
    print(f"[rad_brax_ppo] action_repeat={cfg.action_repeat} "
          f"(at env level) → agent_ep_len={cfg.episode_length // cfg.action_repeat}")
    print(f"[rad_brax_ppo] augment_pixels=True (brax random-translate, pad=4)")

    # Validate brax assertion: batch_size * num_minibatches % num_envs == 0
    total_per_step = cfg.batch_size * cfg.num_minibatches
    if total_per_step % cfg.num_envs != 0:
        raise ValueError(
            f"batch_size * num_minibatches ({total_per_step}) must be "
            f"divisible by num_envs ({cfg.num_envs}). "
            f"Try batch_size={cfg.num_envs // cfg.num_minibatches}"
        )

    # Build train and eval envs (pre-wrapped, wrap_env=False in ppo.train)
    train_env = make_envs(cfg, cfg.num_envs, is_eval=False)
    eval_env = make_envs(cfg, cfg.num_eval_envs, is_eval=True)

    print(f"[rad_brax_ppo] obs keys: {list(train_env.observation_size.keys())}")
    obs_shape = train_env.observation_size.get("pixels/view_0", "N/A")
    print(f"[rad_brax_ppo] pixel obs shape: {obs_shape}")
    print(f"[rad_brax_ppo] action_size: {train_env.action_size}")

    agent_episode_length = cfg.episode_length // cfg.action_repeat

    # Vision PPO network factory — RAD-style DQN encoder
    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=cfg.policy_hidden,
        value_hidden_layer_sizes=cfg.value_hidden,
        cnn_output_channels=cfg.cnn_output_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_stride=cfg.cnn_stride,
        cnn_padding=cfg.cnn_padding,
        cnn_activation=cfg.cnn_activation,
        cnn_max_pool=cfg.cnn_max_pool,
        cnn_global_pool=cfg.cnn_global_pool,
    )

    # Progress callback — brax calls this after each eval checkpoint
    t0 = time.monotonic()
    eval_count = [0]

    def progress_fn(num_steps: int, metrics: dict):
        elapsed = time.monotonic() - t0
        sps = num_steps / elapsed if elapsed > 0 else 0
        er = metrics.get("eval/episode_reward", float("nan"))
        print(
            f"| brax_ppo | S:{num_steps:>9} | SPS:{sps:>7.0f} | "
            f"ER:{er:>10.4f} | elapsed:{elapsed:>7.1f}s",
            flush=True,
        )
        logger.log(num_steps, {"sps": sps, "elapsed": elapsed, **metrics})
        eval_count[0] += 1

    # Launch brax PPO training — fully JIT-compiled jax.lax.scan loop
    print("[rad_brax_ppo] Starting brax PPO training...", flush=True)
    make_policy_fn, params, final_metrics = ppo_train.train(
        environment=train_env,
        num_timesteps=cfg.total_timesteps,
        # env is pre-wrapped; tell brax not to re-wrap it
        wrap_env=False,
        # vision + augment
        vision=True,
        augment_pixels=True,
        # PPO hyperparams (action_repeat=1 required for vision mode)
        action_repeat=1,
        num_envs=cfg.num_envs,
        unroll_length=cfg.unroll_length,
        batch_size=cfg.batch_size,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        clipping_epsilon=cfg.clipping_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        reward_scaling=cfg.reward_scaling,
        normalize_observations=cfg.normalize_observations,
        gae_lambda=cfg.gae_lambda,
        num_evals=cfg.num_evals,
        episode_length=agent_episode_length,
        num_eval_envs=cfg.num_eval_envs,
        eval_env=eval_env,
        network_factory=network_factory,
        seed=cfg.seed,
        progress_fn=progress_fn,
        deterministic_eval=True,
    )

    total_elapsed = time.monotonic() - t0
    print(f"[rad_brax_ppo] Training complete in {total_elapsed:.1f}s "
          f"({total_elapsed / 60:.1f} min)", flush=True)
    if final_metrics:
        er = final_metrics.get("eval/episode_reward", float("nan"))
        print(f"[rad_brax_ppo] Final eval episode_reward: {er:.4f}", flush=True)


if __name__ == "__main__":
    main()
