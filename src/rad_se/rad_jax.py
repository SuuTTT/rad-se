#!/usr/bin/env python3
"""JAX/Flax RAD SAC on MuJoCo Playground pixel tasks.

CleanRL-style one-file implementation targeting the same DMC pixel tasks as
the original RAD paper (Laskin et al. NeurIPS 2020).

Key design:
  - MuJoCo Playground dm_control_suite envs with vision=True
    (HWC grayscale frame-stack; Playground renders at cam_res then we random-crop)
  - Flax NNX modules, Optax optimizers
  - Numpy replay buffer (CPU); JIT-compiled SAC update step
  - Encoder conv weights shared via critic→actor copy after each critic update
    (faithful to the original PyTorch RAD implementation)

Differences from the PyTorch port (reimplementrad/implementations/rad_sac_dmc_pixel.py):
  - Input channels: 3 (grayscale frame-stack) instead of 9 (RGB×3)
  - Image range stored as uint8 [0,255], normalized to [0,1] for the encoder
  - Action repeat handled in the Python loop (same semantics, explicit)
  - NHWC conv layout (JAX default) instead of NCHW

Usage:
  python rad_jax.py --env CartpoleSwingup --seed 23 --track
  python rad_jax.py --env AcrobotSwingup  --seed 23 --action-repeat 4
  python rad_jax.py --env CartpoleSwingup --seed 23 --smoke --total-timesteps 500

Requirements:
  pip install "jax[cuda12]" flax optax mujoco_playground wandb
  export JAX_DEFAULT_MATMUL_PRECISION=highest  # critical on Ampere/Ada GPUs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # experiment
    exp_name: str = "rad_jax"
    seed: int = 23
    # env
    env: str = "CartpoleSwingup"
    action_repeat: int = 8       # 8 for cartpole, 4 for acrobot/cheetah
    cam_res: int = 100            # pre-crop resolution rendered by Playground
    image_size: int = 84          # crop target (matches RAD paper)
    frame_stack: int = 3          # kept for documentation; Playground manages it
    max_episode_steps: int = 1000 # override if needed; will be read from env
    # parallel envs (Warp batch rendering)
    num_envs: int = 64            # parallel envs per training iteration (Warp nworld)
    updates_per_step: int = 0     # gradient steps per training iteration; 0 → num_envs (1:1 ratio)
    # training (timesteps counted in agent env-steps; iterations = total / num_envs)
    total_timesteps: int = 200_000
    replay_capacity: int = 100_000
    init_steps: int = 1_000
    batch_size: int = 128
    eval_freq: int = 10_000
    num_eval_episodes: int = 10
    # SAC
    discount: float = 0.99
    init_temperature: float = 0.5  # higher = more exploration; prevent alpha collapse
    reward_scale: float = 0.1      # scale rewards before storing; Playground sums over action_repeat → large magnitude
    actor_lr: float = 1e-3
    actor_beta: float = 0.9
    critic_lr: float = 1e-3
    critic_beta: float = 0.9
    alpha_lr: float = 1e-4
    alpha_beta: float = 0.5
    critic_tau: float = 0.01       # soft target update for Q-heads
    encoder_tau: float = 0.05      # soft target update for target encoder
    actor_update_freq: int = 2
    critic_target_update_freq: int = 2
    # network
    encoder_feature_dim: int = 50
    num_layers: int = 4
    num_filters: int = 32
    hidden_dim: int = 1024
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    # logging
    log_interval: int = 1_000
    work_dir: str = "runs/rad_jax"
    save_model: bool = False
    track: bool = False
    wandb_project: str = "rad-se"
    wandb_entity: str = "sudingli21"
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_tags: str = ""
    # debug
    smoke: bool = False   # short run for CI / install verification


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for key, val in asdict(defaults).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=val)
        else:
            parser.add_argument(flag, type=type(val), default=val)
    ns = parser.parse_args()
    cfg = Config(**vars(ns))
    if cfg.smoke:
        cfg.total_timesteps = 500
        cfg.init_steps = 100
        cfg.eval_freq = 200
        cfg.replay_capacity = 500
        cfg.batch_size = 16
        cfg.log_interval = 100
        cfg.num_envs = 4
    if cfg.updates_per_step <= 0:
        cfg.updates_per_step = cfg.num_envs
    return cfg


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def obs_to_uint8(obs: np.ndarray) -> np.ndarray:
    """Convert Playground float obs [-0.5, 0.5] to uint8 [0, 255]."""
    return np.clip((obs + 0.5) * 255.0, 0, 255).astype(np.uint8)


def uint8_to_float(obs: np.ndarray) -> np.ndarray:
    """Normalize uint8 [0,255] to float32 [0,1] for encoder input."""
    return obs.astype(np.float32) / 255.0


def random_crop_np(images: np.ndarray, out: int) -> np.ndarray:
    """Random crop a batch of HWC uint8 images.

    Args:
        images: (B, H, W, C) uint8
        out: target spatial size (square)
    Returns:
        (B, out, out, C) uint8
    """
    B, H, W, C = images.shape
    assert H >= out and W >= out, f"image {H}x{W} smaller than crop {out}"
    h0 = np.random.randint(0, H - out + 1, size=B)
    w0 = np.random.randint(0, W - out + 1, size=B)
    cropped = np.empty((B, out, out, C), dtype=images.dtype)
    for i in range(B):
        cropped[i] = images[i, h0[i]: h0[i] + out, w0[i]: w0[i] + out, :]
    return cropped


def center_crop_np(image: np.ndarray, out: int) -> np.ndarray:
    """Center crop a single HWC image for deterministic eval."""
    H, W, C = image.shape
    h0 = (H - out) // 2
    w0 = (W - out) // 2
    return image[h0: h0 + out, w0: w0 + out, :]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, run_dir: Path, cfg: Config):
        self._path = run_dir / "metrics.jsonl"
        self._wandb = None
        if cfg.track:
            import wandb
            tags = [t.strip() for t in cfg.wandb_tags.split(",") if t.strip()]
            self._wandb = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity or None,
                name=cfg.wandb_run_name or f"{cfg.env}__seed{cfg.seed}",
                group=cfg.wandb_group or f"rad_jax__{cfg.env}",
                tags=tags or None,
                config=asdict(cfg),
                dir=str(run_dir),
                save_code=True,
            )

    def log(self, key: str, value: Any, step: int) -> None:
        v = float(np.asarray(value))
        with self._path.open("a") as f:
            f.write(json.dumps({"step": step, key: v}) + "\n")
        if self._wandb is not None:
            self._wandb.log({key: v}, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Encoder spatial dimension helper
# ---------------------------------------------------------------------------

def conv_out_size(image_size: int, num_layers: int) -> int:
    """Compute spatial dimension after the RAD CNN stack."""
    size = (image_size - 3) // 2 + 1  # first conv: 3x3 stride-2
    for _ in range(num_layers - 1):
        size = size - 2               # subsequent: 3x3 stride-1 VALID
    assert size > 0, f"conv output size {size} <= 0 for image_size={image_size}"
    return size


# ---------------------------------------------------------------------------
# Flax NNX modules
# ---------------------------------------------------------------------------

class PixelEncoder(nnx.Module):
    """RAD pixel encoder: 4 conv layers + linear projection + LayerNorm + tanh."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        feature_dim: int,
        num_layers: int,
        num_filters: int,
        output_logits: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.output_logits = output_logits

        # Conv stack — must use nnx.List (plain list not allowed in NNX pytree modules)
        conv_list = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else num_filters
            stride = (2, 2) if i == 0 else (1, 1)
            conv_list.append(
                nnx.Conv(
                    in_ch, num_filters, (3, 3),
                    strides=stride,
                    padding="VALID",
                    rngs=rngs,
                )
            )
        self.convs = nnx.List(conv_list)

        out_dim = conv_out_size(image_size, num_layers)
        flat_dim = num_filters * out_dim * out_dim

        self.fc = nnx.Linear(flat_dim, feature_dim, rngs=rngs)
        self.ln = nnx.LayerNorm(feature_dim, rngs=rngs)

        # Orthogonal init for conv mid-pixel weights (matches PyTorch version)
        _orthogonal_conv_init(self.convs)

    def __call__(self, x: jax.Array, stop_conv_grad: bool = False) -> jax.Array:
        """Forward pass. x: (B, H, W, C) float [0,1]."""
        h = x
        for conv in self.convs:
            h = jax.nn.relu(conv(h))
        if stop_conv_grad:
            h = jax.lax.stop_gradient(h)
        h = h.reshape(h.shape[0], -1)
        h = self.ln(self.fc(h))
        return h if self.output_logits else jnp.tanh(h)


def _orthogonal_conv_init(convs) -> None:
    """Orthogonal initialization on the center pixel, as in the PyTorch RAD repo."""
    for conv in convs:
        k = np.array(conv.kernel[...])  # (kh, kw, in, out)
        kh, kw, in_ch, out_ch = k.shape
        mid = kh // 2
        # Zero all weights, then orthogonal at center
        new_k = np.zeros_like(k)
        center_slice = new_k[mid, mid, :, :]  # (in, out)
        U, _, Vt = np.linalg.svd(np.random.randn(in_ch, out_ch))
        n = min(in_ch, out_ch)
        gain = np.sqrt(2.0)  # relu gain
        center_slice[:, :] = gain * (U[:, :n] @ Vt[:n, :])
        conv.kernel[...] = jnp.array(new_k)
        conv.bias[...] = jnp.zeros_like(conv.bias[...])


class Actor(nnx.Module):
    """SAC actor with a pixel encoder and a squashed Gaussian MLP trunk."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        action_dim: int,
        cfg: Config,
        *,
        rngs: nnx.Rngs,
    ):
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

        self.encoder = PixelEncoder(
            in_channels, image_size, cfg.encoder_feature_dim,
            cfg.num_layers, cfg.num_filters, output_logits=True, rngs=rngs,
        )

        # Trunk MLP (no encoder params here)
        self.trunk = nnx.Sequential(
            nnx.Linear(cfg.encoder_feature_dim, cfg.hidden_dim, rngs=rngs),
            lambda x: jax.nn.relu(x),
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs),
            lambda x: jax.nn.relu(x),
            nnx.Linear(cfg.hidden_dim, 2 * action_dim, rngs=rngs),
        )
        _orthogonal_linear_init(self.trunk)

    def __call__(
        self,
        obs: jax.Array,
        rng: jax.Array,
        compute_pi: bool = True,
        detach_encoder: bool = False,
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array]:
        """Return (mu, pi, log_pi, log_std)."""
        h = self.encoder(obs, stop_conv_grad=detach_encoder)
        if detach_encoder:
            h = jax.lax.stop_gradient(h)
        raw = self.trunk(h)
        mean, log_std_raw = jnp.split(raw, 2, axis=-1)
        log_std = jnp.tanh(log_std_raw)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)

        if compute_pi:
            noise = jax.random.normal(rng, mean.shape)
            pi_pre = mean + noise * jnp.exp(log_std)
        else:
            noise = None
            pi_pre = None

        log_pi = _gaussian_logprob(noise, log_std) if noise is not None else None
        mu, pi, log_pi = _squash(mean, pi_pre, log_pi)
        return mu, pi, log_pi, log_std


class QFunction(nnx.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(feat_dim + action_dim, hidden_dim, rngs=rngs),
            lambda x: jax.nn.relu(x),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            lambda x: jax.nn.relu(x),
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        )
        _orthogonal_linear_init(self.net)

    def __call__(self, feat: jax.Array, action: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate([feat, action], axis=-1))


class Critic(nnx.Module):
    """Twin Q-network with a shared pixel encoder."""

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        action_dim: int,
        cfg: Config,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = PixelEncoder(
            in_channels, image_size, cfg.encoder_feature_dim,
            cfg.num_layers, cfg.num_filters, output_logits=True, rngs=rngs,
        )
        self.Q1 = QFunction(cfg.encoder_feature_dim, action_dim, cfg.hidden_dim, rngs=rngs)
        self.Q2 = QFunction(cfg.encoder_feature_dim, action_dim, cfg.hidden_dim, rngs=rngs)

    def __call__(
        self, obs: jax.Array, action: jax.Array, detach_encoder: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        h = self.encoder(obs, stop_conv_grad=detach_encoder)
        return self.Q1(h, action), self.Q2(h, action)


class LogAlpha(nnx.Module):
    def __init__(self, init_temperature: float):
        self.log_alpha = nnx.Param(jnp.array(float(np.log(init_temperature))))

    @property
    def alpha(self) -> jax.Array:
        return jnp.exp(self.log_alpha[...])


def _gaussian_logprob(noise: jax.Array, log_std: jax.Array) -> jax.Array:
    residual = (-0.5 * noise ** 2 - log_std).sum(-1, keepdims=True)
    return residual - 0.5 * float(np.log(2 * np.pi)) * noise.shape[-1]


def _squash(
    mu: jax.Array, pi: jax.Array | None, log_pi: jax.Array | None
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    mu = jnp.tanh(mu)
    if pi is not None:
        pi = jnp.tanh(pi)
    if log_pi is not None and pi is not None:
        log_pi = log_pi - jnp.log(jax.nn.relu(1.0 - pi ** 2) + 1e-6).sum(-1, keepdims=True)
    return mu, pi, log_pi


def _orth_init_linear(lin: nnx.Linear) -> None:
    """Apply orthogonal init (relu gain) to a single Linear layer.

    Flax stores kernel as (n_in, n_out).  We need a matrix of that exact shape
    whose columns (n_in >= n_out) or rows (n_in < n_out) are orthonormal.
    """
    n_in, n_out = np.array(lin.kernel[...]).shape  # Flax: (in, out)
    W = np.random.randn(n_in, n_out)
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    # U: (n_in, min), Vt: (min, n_out); both have shape (n_in, n_out) when
    # min == n_out (n_in >= n_out) or min == n_in (n_in < n_out).
    gain = np.sqrt(2.0)
    orth = U if n_in >= n_out else Vt  # always (n_in, n_out)
    lin.kernel[...] = jnp.array(gain * orth)
    lin.bias[...] = jnp.zeros_like(lin.bias[...])


def _orthogonal_linear_init(module: nnx.Module) -> None:
    """Apply orthogonal init to all Linear layers found in the module."""
    for _path, node in nnx.iter_modules(module):
        if isinstance(node, nnx.Linear):
            _orth_init_linear(node)


# ---------------------------------------------------------------------------
# Weight copy: critic encoder → actor encoder (conv weights only)
# ---------------------------------------------------------------------------

def copy_conv_weights_to_actor(critic: Critic, actor: Actor) -> None:
    """Copy critic encoder conv weights to actor encoder (tied-weight semantics)."""
    for c_conv, a_conv in zip(critic.encoder.convs, actor.encoder.convs):
        a_conv.kernel[...] = c_conv.kernel[...]
        a_conv.bias[...]   = c_conv.bias[...]


def soft_update_target(source: Critic, target: Critic, tau: float) -> None:
    """Polyak-average all parameters from source into target."""
    _, src_state = nnx.split(source)
    _, tgt_state = nnx.split(target)
    new_tgt = jax.tree.map(
        lambda s, t: tau * s + (1.0 - tau) * t, src_state, tgt_state
    )
    nnx.update(target, new_tgt)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """CPU numpy replay buffer storing uint8 HWC images.

    Pre-crop size (cam_res) is stored; random crop applied at sample time.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],  # (pre_crop_H, pre_crop_W, C)
        action_dim: int,
        capacity: int,
        image_size: int,                  # crop target
    ):
        self.capacity = capacity
        self.image_size = image_size
        self.obs_shape = obs_shape

        self.obses    = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.n_obses  = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions  = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.empty((capacity, 1), dtype=np.float32)
        self.not_done = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.n_obses[self.idx], next_obs)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.not_done[self.idx] = 1.0 - done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(
        self,
        obs: np.ndarray,        # (N, H, W, C) uint8
        action: np.ndarray,     # (N, action_dim) float32
        reward: np.ndarray,     # (N,) float32
        next_obs: np.ndarray,   # (N, H, W, C) uint8
        done: np.ndarray,       # (N,) float32  (1.0 only on true terminations)
    ) -> None:
        """Insert N transitions in one shot with ring-buffer wraparound."""
        n = obs.shape[0]
        # Compute destination indices, wrapping modulo capacity.
        idxs = (np.arange(n) + self.idx) % self.capacity
        self.obses[idxs]    = obs
        self.n_obses[idxs]  = next_obs
        self.actions[idxs]  = action
        self.rewards[idxs]  = reward.reshape(n, 1)
        self.not_done[idxs] = (1.0 - done).reshape(n, 1)
        new_idx = (self.idx + n) % self.capacity
        # If we wrapped or exactly filled, mark full
        if not self.full and (self.idx + n) >= self.capacity:
            self.full = True
        self.idx = new_idx

    def sample(self) -> tuple:
        """Sample a batch with random crop applied. Returns JAX arrays."""
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=128)  # placeholder; batch_size set in sample_with_bs
        return self._build(idxs)

    def sample_bs(self, batch_size: int) -> tuple:
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return self._build(idxs)

    def _build(self, idxs: np.ndarray) -> tuple:
        obs_raw  = random_crop_np(self.obses[idxs], self.image_size)
        nobs_raw = random_crop_np(self.n_obses[idxs], self.image_size)
        obs_f  = jnp.array(uint8_to_float(obs_raw))
        nobs_f = jnp.array(uint8_to_float(nobs_raw))
        acts   = jnp.array(self.actions[idxs])
        rews   = jnp.array(self.rewards[idxs])
        notd   = jnp.array(self.not_done[idxs])
        return obs_f, acts, rews, nobs_f, notd

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx


# ---------------------------------------------------------------------------
# JIT-compiled SAC update functions
# ---------------------------------------------------------------------------

@nnx.jit
def _update_critic(
    actor: Actor,
    critic: Critic,
    critic_target: Critic,
    log_alpha: LogAlpha,
    critic_opt: nnx.Optimizer,
    batch: tuple,
    rng: jax.Array,
    discount: float,
) -> tuple[jax.Array, jax.Array]:
    obs, actions, rewards, next_obs, not_done = batch

    # Compute target Q without gradient through actor / target
    rng, key = jax.random.split(rng)
    _, next_pi, next_log_pi, _ = actor(next_obs, key, detach_encoder=True)
    q1_t, q2_t = critic_target(next_obs, next_pi)
    alpha = jax.lax.stop_gradient(log_alpha.alpha)
    next_log_pi = jax.lax.stop_gradient(next_log_pi)
    target_q = jax.lax.stop_gradient(
        rewards + not_done * discount * (jnp.minimum(q1_t, q2_t) - alpha * next_log_pi)
    )

    def loss_fn(critic: Critic) -> jax.Array:
        q1, q2 = critic(obs, actions)
        return jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(critic)
    critic_opt.update(critic, grads)
    return loss, rng


@nnx.jit
def _update_actor_alpha(
    actor: Actor,
    critic: Critic,
    log_alpha: LogAlpha,
    actor_opt: nnx.Optimizer,
    alpha_opt: nnx.Optimizer,
    batch: tuple,
    rng: jax.Array,
    target_entropy: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    obs = batch[0]
    rng, key = jax.random.split(rng)

    # Actor loss — encoder is detached (gradients only update trunk)
    def actor_loss_fn(trunk: nnx.Sequential) -> jax.Array:
        h = jax.lax.stop_gradient(actor.encoder(obs))
        raw = trunk(h)
        mean, log_std_raw = jnp.split(raw, 2, axis=-1)
        log_std = jnp.tanh(log_std_raw)
        log_std = actor.log_std_min + 0.5 * (actor.log_std_max - actor.log_std_min) * (log_std + 1.0)
        noise = jax.random.normal(key, mean.shape)
        pi_pre = mean + noise * jnp.exp(log_std)
        log_pi = _gaussian_logprob(noise, log_std)
        _, pi, log_pi = _squash(mean, pi_pre, log_pi)

        alpha = jax.lax.stop_gradient(log_alpha.alpha)
        q1, q2 = critic(obs, pi, detach_encoder=True)
        q = jnp.minimum(q1, q2)
        return (alpha * log_pi - q).mean(), log_pi

    (actor_loss, log_pi), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor.trunk)
    actor_opt.update(actor.trunk, grads)

    # Alpha (temperature) loss
    log_pi_detached = jax.lax.stop_gradient(log_pi)
    def alpha_loss_fn(log_alpha: LogAlpha) -> jax.Array:
        return (log_alpha.alpha * (-log_pi_detached - target_entropy)).mean()

    alpha_loss, alpha_grads = nnx.value_and_grad(alpha_loss_fn)(log_alpha)
    alpha_opt.update(log_alpha, alpha_grads)

    return actor_loss, alpha_loss, rng


# ---------------------------------------------------------------------------
# MuJoCo Playground environment
# ---------------------------------------------------------------------------

def make_env(cfg: Config):
    """Instantiate a Playground env with vision enabled and batched parallel envs.

    Uses ``wrap_for_brax_training`` which adds VmapWrapper + EpisodeWrapper
    (handles action_repeat via ``lax.scan``) + BraxAutoResetWrapper. The
    underlying Warp megakernel renderer is configured with ``nworld=num_envs``
    so all envs are batched in a single kernel launch.

    Returns:
        env: wrapped batched env. ``env.reset(keys)`` expects keys of shape
             ``(num_envs, 2)``; ``env.step(state, action)`` expects action of
             shape ``(num_envs, action_dim)``.
        action_dim: int
        max_ep_steps: int (episode_length in physics steps from env config)
    """
    from mujoco_playground._src import dm_control_suite
    from mujoco_playground import wrapper as mp_wrapper

    # ---------------------------------------------------------------------------
    # BUG-FIX: Playground's CartpoleSwingup vision mode has a broken done
    # condition.  Balance.step (shared by swingup) checks:
    #   done |= (jp.abs(pole_angle) > jp.pi / 2)
    # But SwingUp initialises the pole at angle ≈ π (bottom), so done=True
    # fires IMMEDIATELY on every step, giving 1-step episodes and zero learning.
    # We monkey-patch Balance.step to skip the angle bound when the instance
    # carries a `_fix_swingup_done` marker (set after dm_control_suite.load).
    # ---------------------------------------------------------------------------
    from mujoco_playground._src.dm_control_suite import cartpole as _cp_mod
    if not hasattr(_cp_mod.Balance, "_rad_se_patched"):
        _orig_balance_step = _cp_mod.Balance.step

        def _fixed_balance_step(self_env, state, action):
            nstate = _orig_balance_step(self_env, state, action)
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

        _cp_mod.Balance.step = _fixed_balance_step
        _cp_mod.Balance._rad_se_patched = True

    env_config = dm_control_suite.get_default_config(cfg.env)
    # Enable vision at pre-crop resolution; nworld matches the agent batch
    env_config.vision = True
    env_config.vision_config.cam_res = (cfg.cam_res, cfg.cam_res)
    env_config.vision_config.nworld = cfg.num_envs

    env = dm_control_suite.load(cfg.env, config=env_config)
    # Mark swingup envs so the patched step skips the pole-angle termination.
    _swingup_envs = {"CartpoleSwingup"}
    if cfg.env in _swingup_envs and env_config.vision:
        env._fix_swingup_done = True
    action_dim = env.action_size
    max_ep_steps = int(env_config.episode_length)

    # Vmap + EpisodeWrapper (action_repeat inside lax.scan) + auto-reset.
    # full_reset=False (default): on episode end, each env resets to the cached
    # first state it saw at env.reset() time. Since initial reset uses
    # jax.random.split(rng, num_envs), each env has a distinct starting state,
    # giving num_envs-way diversity without extra compute per step.
    env = mp_wrapper.wrap_for_brax_training(
        env,
        episode_length=max_ep_steps,
        action_repeat=cfg.action_repeat,
    )

    return env, action_dim, max_ep_steps


def env_reset(env, rng: jax.Array, num_envs: int):
    """Reset batched env. Returns (state, obs_uint8_batch).

    obs has shape (num_envs, H, W, 3) uint8.
    """
    keys = jax.random.split(rng, num_envs)
    state = env.reset(keys)
    obs_np = np.array(state.obs["pixels/view_0"])  # (N, H, W, 3) float
    obs_u8 = obs_to_uint8(obs_np)
    return state, obs_u8


def env_step(env, state, action_batch: np.ndarray):
    """Step batched env once (action_repeat handled internally by EpisodeWrapper).

    Args:
        action_batch: (num_envs, action_dim) numpy float32 actions.
    Returns:
        next_state, next_obs_u8 (N,H,W,3), reward (N,), done (N,), truncation (N,)
    """
    action_jax = jnp.asarray(action_batch)
    next_state = env.step(state, action_jax)
    next_obs_np = np.array(next_state.obs["pixels/view_0"])  # (N, H, W, 3) float
    next_obs_u8 = obs_to_uint8(next_obs_np)
    reward = np.asarray(next_state.reward, dtype=np.float32).reshape(-1)
    done = np.asarray(next_state.done, dtype=np.float32).reshape(-1)
    # EpisodeWrapper writes a 'truncation' flag (1 when timeout, 0 when physics done).
    trunc = np.asarray(
        next_state.info.get("truncation", np.zeros_like(done)),
        dtype=np.float32,
    ).reshape(-1)
    return next_state, next_obs_u8, reward, done, trunc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    env,
    actor: Actor,
    cfg: Config,
    run_dir: Path,
    step: int,
    logger: Logger,
    rng: jax.Array,
) -> tuple[float, jax.Array]:
    """Run ``num_envs`` parallel deterministic rollouts of one full episode each.

    For DMC tasks (which never early-terminate), running for
    ``max_episode_steps // action_repeat`` env-step iterations is exactly one
    full episode per env. We use the batched training env infrastructure with
    a fresh reset state to avoid disturbing the training trajectories — the
    caller is responsible for separately tracking/preserving the train state.
    """
    rng, reset_key = jax.random.split(rng)
    state, obs_u8 = env_reset(env, reset_key, cfg.num_envs)

    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    n_iter = max(1, cfg.max_episode_steps // cfg.action_repeat)

    for _ in range(n_iter):
        # Center crop and pack to (N, H, W, C) float in [0,1]
        obs_crop = np.stack(
            [center_crop_np(obs_u8[i], cfg.image_size) for i in range(cfg.num_envs)],
            axis=0,
        )
        obs_jax = jnp.asarray(uint8_to_float(obs_crop))
        rng, key = jax.random.split(rng)
        mu, _, _, _ = actor(obs_jax, key, compute_pi=False)
        action_np = np.asarray(mu, dtype=np.float32)
        state, obs_u8, reward, done, trunc = env_step(env, state, action_np)
        ep_returns += reward

    mean_r = float(np.mean(ep_returns))
    max_r  = float(np.max(ep_returns))
    std_r  = float(np.std(ep_returns))
    logger.log("eval/mean_episode_reward", mean_r, step)
    logger.log("eval/max_episode_reward",  max_r,  step)
    logger.log("eval/std_episode_reward",  std_r,  step)

    payload = {
        f"{cfg.env}--crop--s{cfg.seed}": {
            step: {"mean_ep_reward": mean_r, "max_ep_reward": max_r, "std_ep_reward": std_r}
        }
    }
    np.save(run_dir / f"{cfg.env}--crop--s{cfg.seed}--eval_scores.npy", payload)
    print(f"| eval | S: {step:>7d} | ER: {mean_r:.4f}", flush=True)
    return mean_r, rng


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    # Ampere/Ada precision guard (must be set before any JAX op)
    if "JAX_DEFAULT_MATMUL_PRECISION" not in os.environ:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        print("[rad_jax] Set JAX_DEFAULT_MATMUL_PRECISION=highest", flush=True)

    set_global_seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    run_dir = Path(cfg.work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    logger = Logger(run_dir, cfg)

    # ------------------------------------------------------------------
    # Env
    # ------------------------------------------------------------------
    print(f"[rad_jax] Building env: {cfg.env}", flush=True)
    env, action_dim, max_ep_steps = make_env(cfg)
    cfg.max_episode_steps = max_ep_steps  # override with actual

    # Observation shape stored in replay buffer (pre-crop, HWC uint8)
    obs_shape = (cfg.cam_res, cfg.cam_res, 3)
    in_channels = 3  # grayscale frame stack

    print(f"[rad_jax] action_dim={action_dim}  max_ep_steps={max_ep_steps}  "
          f"obs_shape={obs_shape}", flush=True)

    # ------------------------------------------------------------------
    # Networks and optimizers
    # ------------------------------------------------------------------
    rngs = nnx.Rngs(cfg.seed)

    actor         = Actor(in_channels, cfg.image_size, action_dim, cfg, rngs=rngs)
    critic        = Critic(in_channels, cfg.image_size, action_dim, cfg, rngs=rngs)
    critic_target = Critic(in_channels, cfg.image_size, action_dim, cfg, rngs=rngs)

    # Initialise target = critic
    _hard_copy_critic(critic, critic_target)
    # Tied-weight init: copy critic encoder conv → actor encoder
    copy_conv_weights_to_actor(critic, actor)

    log_alpha = LogAlpha(cfg.init_temperature)
    target_entropy = -float(action_dim)

    critic_opt = nnx.Optimizer(
        critic, optax.adam(cfg.critic_lr, b1=cfg.critic_beta), wrt=nnx.Param
    )
    actor_opt = nnx.Optimizer(
        actor.trunk, optax.adam(cfg.actor_lr, b1=cfg.actor_beta), wrt=nnx.Param
    )
    alpha_opt = nnx.Optimizer(
        log_alpha, optax.adam(cfg.alpha_lr, b1=cfg.alpha_beta), wrt=nnx.Param
    )

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------
    replay = ReplayBuffer(obs_shape, action_dim, cfg.replay_capacity, cfg.image_size)

    # ------------------------------------------------------------------
    # Training loop (batched parallel envs)
    # ------------------------------------------------------------------
    N = cfg.num_envs
    K = cfg.updates_per_step
    rng, reset_key = jax.random.split(rng)
    state, obs_u8 = env_reset(env, reset_key, N)  # obs_u8: (N, H, W, 3)

    ep_reward = np.zeros(N, dtype=np.float32)
    ep_step   = np.zeros(N, dtype=np.int64)
    ep_count  = 0
    start_time = time.time()

    total_iters = max(1, cfg.total_timesteps // N)
    init_iters  = max(1, cfg.init_steps // N)
    eval_every_iters = max(1, cfg.eval_freq // N)
    log_every_iters  = max(1, cfg.log_interval // N)

    print(f"[rad_jax] Batched training: num_envs={N}, updates_per_step={K}, "
          f"total_iters={total_iters} (≈{total_iters*N} env-steps), "
          f"init_iters={init_iters}", flush=True)

    for it in range(total_iters):
        global_step = it * N

        # ------ eval ------
        if it % eval_every_iters == 0:
            _, rng = evaluate(env, actor, cfg, run_dir, global_step, logger, rng)
            # eval consumed env state; resume training from a fresh reset
            rng, reset_key = jax.random.split(rng)
            state, obs_u8 = env_reset(env, reset_key, N)
            ep_reward.fill(0.0)
            ep_step.fill(0)

        # ------ action selection (batched) ------
        if it < init_iters:
            action_np = np.random.uniform(
                -1.0, 1.0, size=(N, action_dim)
            ).astype(np.float32)
        else:
            obs_crop = np.stack(
                [center_crop_np(obs_u8[i], cfg.image_size) for i in range(N)],
                axis=0,
            )
            obs_jax = jnp.asarray(uint8_to_float(obs_crop))
            rng, key = jax.random.split(rng)
            _, pi, _, _ = actor(obs_jax, key)
            action_np = np.asarray(pi, dtype=np.float32)

        # ------ env step (batched, action_repeat handled inside wrapper) ------
        next_state, next_obs_u8, reward, done, trunc = env_step(env, state, action_np)
        ep_reward += reward
        ep_step   += cfg.action_repeat

        # Bootstrap on timeout: not_done=1 if truncation (true done only on physics term)
        physics_done = np.where(trunc > 0.5, 0.0, done).astype(np.float32)

        # Scale rewards before storing: Playground EpisodeWrapper sums over
        # action_repeat steps → large magnitude; scale down for SAC stability.
        replay.add_batch(obs_u8, action_np, reward * cfg.reward_scale, next_obs_u8, physics_done)

        # ------ per-env episode logging on terminations ------
        finished = (done > 0.5) | (trunc > 0.5)
        if finished.any():
            for i in np.flatnonzero(finished):
                logger.log("train/episode_reward", float(ep_reward[i]), global_step)
                logger.log("train/episode", ep_count, global_step)
                logger.log("train/episode_length", int(ep_step[i]), global_step)
                ep_count += 1
            ep_reward[finished] = 0.0
            ep_step[finished] = 0
        if it % log_every_iters == 0:
            elapsed = time.time() - start_time
            logger.log("train/fps", global_step / max(elapsed, 1.0), global_step)

        state  = next_state
        obs_u8 = next_obs_u8

        # ------ gradient updates ------
        if it >= init_iters and len(replay) >= cfg.batch_size:
            for upd in range(K):
                batch = replay.sample_bs(cfg.batch_size)
                rng, update_key = jax.random.split(rng)

                critic_loss, rng = _update_critic(
                    actor, critic, critic_target, log_alpha,
                    critic_opt, batch, update_key, cfg.discount,
                )
                copy_conv_weights_to_actor(critic, actor)

                # Use a finer-grained "training step" for actor/target updates
                train_step = global_step + upd
                if train_step % cfg.actor_update_freq == 0:
                    actor_loss, alpha_loss, rng = _update_actor_alpha(
                        actor, critic, log_alpha,
                        actor_opt, alpha_opt,
                        batch, rng, target_entropy,
                    )
                    if it % log_every_iters == 0 and upd == 0:
                        logger.log("train/actor_loss", actor_loss, global_step)
                        logger.log("train/alpha_loss", alpha_loss, global_step)
                        logger.log("train/alpha", float(log_alpha.alpha), global_step)

                if train_step % cfg.critic_target_update_freq == 0:
                    soft_update_target(critic, critic_target, cfg.critic_tau)

                if it % log_every_iters == 0 and upd == 0:
                    logger.log("train/critic_loss", critic_loss, global_step)

    # Final eval
    evaluate(env, actor, cfg, run_dir, cfg.total_timesteps, logger, rng)

    if cfg.save_model:
        _save_checkpoint(actor, critic, run_dir, cfg.total_timesteps)

    logger.finish()
    print("[rad_jax] Done.", flush=True)


def _hard_copy_critic(src: Critic, dst: Critic) -> None:
    """Hard-copy all src parameters into dst."""
    _, src_state = nnx.split(src)
    nnx.update(dst, src_state)


def _save_checkpoint(actor: Actor, critic: Critic, run_dir: Path, step: int) -> None:
    import orbax.checkpoint as ocp
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # NNX state export
    _, actor_state  = nnx.split(actor)
    _, critic_state = nnx.split(critic)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_dir / "actor"),  actor_state)
    checkpointer.save(str(ckpt_dir / "critic"), critic_state)
    print(f"[rad_jax] Checkpoint saved at step {step}", flush=True)


if __name__ == "__main__":
    main()

