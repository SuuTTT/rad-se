#!/usr/bin/env python3
"""RAD-style SAC using brax losses + on-device replay buffer on MuJoCo Playground.

brax.training.agents.sac.train does not support dict/pixel observations
(raises NotImplementedError). This file re-uses the official brax SAC loss
functions and replay buffer, but implements the training loop itself so that:
  - Pixel observations (dict with 'pixels/*' keys) are supported.
  - RAD random-translate augmentation is applied at sample time.
    - Replay buffer stays on-device — no H2D transfer.
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
from flax import struct as flax_struct
from jax import flatten_util

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
import warp as wp


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
    frame_stack: int = 1
    episode_length: int = 1000          # physics steps
    num_envs: int = 8                   # SAC: small env count, large replay
    num_eval_envs: int = 16
    # replay buffer
    max_replay_size: int = 10_000       # ~2.4 GB float32 pixel obs
    min_replay_size: int = 1_000        # warmup steps before updates start
    batch_size: int = 256               # SAC minibatch from replay
    grad_updates_per_step: int = 1
    replay_backend: str = "device"       # device or host
    replay_pixel_dtype: str = "float32"  # float32, float16, bfloat16, or uint8_centered (host only)
    # SAC hypers
    total_timesteps: int = 500_000
    num_evals: int = 20
    discounting: float = 0.99
    learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    init_temperature: float = 1.0
    target_entropy: float = -0.5       # brax default for action_size=1; RAD uses -1.0
    reward_scaling: float = 0.1
    tau: float = 0.005                  # target network EMA coefficient
    encoder_tau: float = 0.0            # if >0, RAD encoder target EMA coefficient
    actor_update_frequency: int = 1
    alpha_update_frequency: int = 1
    target_update_frequency: int = 1
    tie_actor_critic_encoder: bool = False
    encoder_arch: str = "dqn"           # dqn or rad
    rad_feature_dim: int = 50
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
    crop_size: int = 0                  # if >0, random crop train obs and center crop policy/eval obs
    # Reward convention. mujoco_playground's vision-mode cartpole uses
    # _dense_vision_reward (additive penalty, episode range ~[-3900, 100]).
    # Setting dmc_reward=True monkey-patches the env to use _dense_reward,
    # the dm_control tolerance-product reward (per-step [0,1], episode [0,1000]),
    # matching the Playground/RAD paper numbers.
    dmc_reward: bool = False
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


def _random_crop_pixels_sac(
    obs: Mapping[str, jax.Array],
    key: jax.Array,
    crop_size: int,
) -> Mapping[str, jax.Array]:
    """Apply random spatial crops to [B, H, W, C] pixel observations."""
    @jax.vmap
    def crop_one(ub_obs: Mapping[str, jax.Array], rng: jax.Array):
        def crop_view(img: jax.Array, rng: jax.Array) -> jax.Array:
            h, w, c = img.shape
            h_off = jax.random.randint(rng, (), 0, h - crop_size + 1)
            _, rng = jax.random.split(rng)
            w_off = jax.random.randint(rng, (), 0, w - crop_size + 1)
            start = jnp.array([h_off, w_off, 0], dtype=jnp.int32)
            return jax.lax.dynamic_slice(img, start, (crop_size, crop_size, c))

        out = {}
        for k, v in ub_obs.items():
            if k.startswith("pixels/"):
                rng, subkey = jax.random.split(rng)
                out[k] = crop_view(v, subkey)
        return {**ub_obs, **out}

    bdim = next(iter(obs.values())).shape[0]
    keys = jax.random.split(key, bdim)
    return crop_one(obs, keys)


def _center_crop_pixels_sac(
    obs: Mapping[str, jax.Array],
    crop_size: int,
) -> Mapping[str, jax.Array]:
    """Apply deterministic center crops to [B, H, W, C] pixel observations."""
    if crop_size <= 0:
        return obs
    out = dict(obs)
    for k, v in obs.items():
        if k.startswith("pixels/"):
            h, w, c = v.shape[-3:]
            if crop_size >= h and crop_size >= w:
                continue
            h_off = (h - crop_size) // 2
            w_off = (w - crop_size) // 2
            start = (0,) * (v.ndim - 3) + (h_off, w_off, 0)
            size = v.shape[:-3] + (crop_size, crop_size, c)
            out[k] = jax.lax.dynamic_slice(v, start, size)
    return out


def _pixel_storage_dtype(name: str):
    if name == "float32":
        return jnp.float32
    if name == "float16":
        return jnp.float16
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "uint8_centered":
        return name
    raise ValueError(f"Unsupported replay_pixel_dtype={name!r}")


def _cast_transition(transition: types.Transition, dtype) -> types.Transition:
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), transition)


def _repeat_stack_pixels_sac(
    obs: Mapping[str, jax.Array],
    frame_stack: int,
) -> Mapping[str, jax.Array]:
    """Repeat reset pixel observations into an NHWC frame stack."""
    if frame_stack <= 1:
        return obs
    out = dict(obs)
    for k, v in obs.items():
        if k.startswith("pixels/"):
            out[k] = jnp.concatenate([v] * frame_stack, axis=-1)
    return out


def _append_stack_pixels_sac(
    stacked_obs: Mapping[str, jax.Array],
    next_obs: Mapping[str, jax.Array],
    done: jax.Array,
    frame_stack: int,
) -> Mapping[str, jax.Array]:
    """Append latest NHWC pixel frame; reset the stack on episode boundaries."""
    if frame_stack <= 1:
        return next_obs
    out = dict(next_obs)
    done_mask = done.reshape((done.shape[0],) + (1,) * (next(iter(next_obs.values())).ndim - 1))
    reset_obs = _repeat_stack_pixels_sac(next_obs, frame_stack)
    for k, v in next_obs.items():
        if k.startswith("pixels/"):
            channels = v.shape[-1]
            shifted = jnp.concatenate([stacked_obs[k][..., channels:], v], axis=-1)
            out[k] = jnp.where(done_mask, reset_obs[k], shifted)
    return out


# ---------------------------------------------------------------------------
# Vision SAC networks: policy via brax, Q-network custom (CNN + action + MLP)
# ---------------------------------------------------------------------------

class PixelEncoderRAD(linen.Module):
    feature_dim: int = 50
    activation: Any = linen.relu

    @linen.compact
    def __call__(self, pixels: jax.Array):
        x = pixels
        for idx, stride in enumerate((2, 1, 1, 1)):
            x = linen.Conv(
                features=32,
                kernel_size=(3, 3),
                strides=(stride, stride),
                padding="VALID",
                name=f"conv{idx + 1}",
            )(x)
            x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = linen.Dense(self.feature_dim, name="fc")(x)
        x = linen.LayerNorm(name="ln")(x)
        return jnp.tanh(x)


class VisionPolicyModule(linen.Module):
    output_size: int
    hidden_layer_sizes: Sequence[int] = (1024, 1024)
    encoder_arch: str = "dqn"
    rad_feature_dim: int = 50
    cnn_output_channels: Sequence[int] = (32, 64, 64)
    cnn_kernel_size: Sequence[int] = (8, 4, 3)
    cnn_stride: Sequence[int] = (4, 2, 1)
    cnn_padding: str = "VALID"
    cnn_activation: linen.activation.PReLU = linen.relu
    cnn_max_pool: bool = False
    cnn_global_pool: str = "avg"
    activation: Any = linen.relu

    @linen.compact
    def __call__(self, obs: Mapping[str, jax.Array]):
        pixels = {k: v for k, v in obs.items() if k.startswith("pixels/")}
        embeddings = []
        kernel_sizes = tuple((k, k) for k in self.cnn_kernel_size)
        strides = tuple((s, s) for s in self.cnn_stride)
        for pkey in sorted(pixels.keys()):
            if self.encoder_arch == "rad":
                embed = PixelEncoderRAD(
                    feature_dim=self.rad_feature_dim,
                    activation=self.activation,
                    name=f"rad_encoder_{pkey.replace('/', '_')}",
                )(pixels[pkey])
            else:
                embed = brax_networks.CNN(
                    num_filters=self.cnn_output_channels,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    activation=self.cnn_activation,
                    padding=self.cnn_padding,
                    max_pool=self.cnn_max_pool,
                )(pixels[pkey])
                if self.cnn_global_pool == "avg":
                    embed = jnp.mean(embed, axis=(-3, -2))
                elif self.cnn_global_pool == "max":
                    embed = jnp.max(embed, axis=(-3, -2))
                else:
                    embed = embed.reshape(embed.shape[0], -1)
            embeddings.append(embed)

        policy_input = jnp.concatenate(embeddings, axis=-1)
        return brax_networks.MLP(
            layer_sizes=list(self.hidden_layer_sizes) + [self.output_size],
            activation=self.activation,
        )(policy_input)


class VisionQModule(linen.Module):
    """Twin-Q critic for pixel obs. Each critic has its own CNN encoder."""
    n_critics: int = 2
    hidden_layer_sizes: Sequence[int] = (1024, 1024)
    encoder_arch: str = "dqn"
    rad_feature_dim: int = 50
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
                if self.encoder_arch == "rad":
                    cnn_out = PixelEncoderRAD(
                        feature_dim=self.rad_feature_dim,
                        activation=self.activation,
                    )(pixels[pkey])
                else:
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


def _copy_rad_conv_weights_to_policy(policy_params: Any, q_params: Any) -> Any:
    """Copies first critic RAD conv weights into actor RAD encoders."""
    def clone_leaf(x):
        return x + jnp.zeros_like(x)

    policy_params = dict(policy_params)
    policy_vars = dict(policy_params["params"])
    source_encoder = q_params["params"]["PixelEncoderRAD_0"]

    for encoder_name, encoder_params in policy_vars.items():
        if not encoder_name.startswith("rad_encoder_"):
            continue
        tied_encoder = dict(encoder_params)
        for conv_name in ("conv1", "conv2", "conv3", "conv4"):
            tied_encoder[conv_name] = jax.tree_util.tree_map(
                clone_leaf, source_encoder[conv_name])
        policy_vars[encoder_name] = tied_encoder

    policy_params["params"] = policy_vars
    return policy_params


def _is_rad_encoder_path(path) -> bool:
    for entry in path:
        key = getattr(entry, "key", None)
        if isinstance(key, str) and key.startswith("PixelEncoderRAD_"):
            return True
    return False


def _soft_update_q_params(target_q_params: Any, q_params: Any, critic_tau: float, encoder_tau: float) -> Any:
    def update_leaf(path, target, source):
        tau = encoder_tau if _is_rad_encoder_path(path) else critic_tau
        return target * (1 - tau) + source * tau

    return jax.tree_util.tree_map_with_path(update_leaf, target_q_params, q_params)


def make_sac_networks_vision(
    obs_size: Mapping[str, Tuple],   # per-env, e.g. {'pixels/view_0': (H,W,C)}
    action_size: int,
    cfg: Config,
) -> sac_networks.SACNetworks:
    """Create policy + Q-network for pixel SAC."""
    act = linen.relu if cfg.cnn_activation == "relu" else linen.swish

    policy_module = VisionPolicyModule(
        output_size=distribution.NormalTanhDistribution(action_size).param_size,
        hidden_layer_sizes=cfg.policy_hidden,
        encoder_arch=cfg.encoder_arch,
        rad_feature_dim=cfg.rad_feature_dim,
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

    def policy_apply(processor_params, policy_params, obs):
        return policy_module.apply(policy_params, obs)

    policy_net = brax_networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=policy_apply,
    )

    # Q-network: custom twin-Q with independent CNN encoders
    q_module = VisionQModule(
        n_critics=2,
        hidden_layer_sizes=cfg.critic_hidden,
        encoder_arch=cfg.encoder_arch,
        rad_feature_dim=cfg.rad_feature_dim,
        cnn_output_channels=cfg.cnn_output_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_stride=cfg.cnn_stride,
        cnn_padding=cfg.cnn_padding,
        cnn_activation=act,
        cnn_max_pool=cfg.cnn_max_pool,
        cnn_global_pool=cfg.cnn_global_pool,
        activation=act,
    )
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


def make_sac_losses_vision(
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    target_entropy: float,
):
    """SAC losses matching brax, with configurable target entropy."""
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(log_alpha, policy_params, normalizer_params, transitions, key):
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

    def critic_loss(q_params, policy_params, normalizer_params, target_q_params,
                    alpha, transitions, key):
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation)
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params, target_q_params,
            transitions.next_observation, next_action)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discounting * next_v)
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)
        return 0.5 * jnp.mean(jnp.square(q_error))

    def actor_loss(policy_params, normalizer_params, q_params, alpha, transitions, key):
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        return jnp.mean(alpha * log_prob - min_q)

    return alpha_loss, critic_loss, actor_loss


# ---------------------------------------------------------------------------
# Typed replay buffer
# ---------------------------------------------------------------------------

@flax_struct.dataclass
class FlatReplayBufferState:
    data: jax.Array
    insert_position: jax.Array
    sample_position: jax.Array
    key: jax.Array


class UniformFlatReplayBuffer:
    """Uniform replay with one reduced-precision flat storage array."""

    def __init__(self, max_replay_size: int, dummy_data_sample: Any, sample_batch_size: int, storage_dtype):
        self._storage_dtype = storage_dtype
        dummy_storage = _cast_transition(dummy_data_sample, storage_dtype)
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_storage)
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        self._flatten_fn = jax.vmap(
            lambda x: flatten_util.ravel_pytree(_cast_transition(x, storage_dtype))[0])
        self._data_shape = (max_replay_size, len(dummy_flatten))
        self._sample_batch_size = sample_batch_size

    def init(self, key: jax.Array) -> FlatReplayBufferState:
        return FlatReplayBufferState(
            data=jnp.zeros(self._data_shape, self._storage_dtype),
            insert_position=jnp.zeros((), jnp.int32),
            sample_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state: FlatReplayBufferState, samples: Any) -> FlatReplayBufferState:
        update = self._flatten_fn(samples)
        data = buffer_state.data
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(
            roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)
        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def sample(self, buffer_state: FlatReplayBufferState) -> tuple[FlatReplayBufferState, Any]:
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )
        flat_batch = jnp.take(buffer_state.data, idx, axis=0, mode="wrap")
        batch = _cast_transition(self._unflatten_fn(flat_batch), jnp.float32)
        return buffer_state.replace(key=key), batch


class HostFlatReplayBuffer:
    """Uniform replay with flat CPU storage and device batches on sample."""

    def __init__(self, max_replay_size: int, dummy_data_sample: Any, sample_batch_size: int, storage_dtype, seed: int):
        if storage_dtype == jnp.bfloat16:
            raise ValueError("host replay currently supports float32 and float16 storage")
        self._storage_dtype = storage_dtype
        self._np_dtype = np.float16 if storage_dtype == jnp.float16 else np.float32
        dummy_storage = _cast_transition(dummy_data_sample, storage_dtype)
        dummy_flatten, self._unflatten_one = flatten_util.ravel_pytree(dummy_storage)
        self._unflatten_fn = jax.vmap(self._unflatten_one)
        self._flatten_fn = jax.jit(jax.vmap(
            lambda x: flatten_util.ravel_pytree(_cast_transition(x, storage_dtype))[0]))
        self._data = np.zeros((max_replay_size, len(dummy_flatten)), dtype=self._np_dtype)
        self._max_replay_size = max_replay_size
        self._sample_batch_size = sample_batch_size
        self._insert_position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def insert(self, samples: Any) -> None:
        update = np.asarray(self._flatten_fn(samples), dtype=self._np_dtype)
        n = update.shape[0]
        if n >= self._max_replay_size:
            self._data[...] = update[-self._max_replay_size:]
            self._insert_position = 0
            self._size = self._max_replay_size
            return
        end = self._insert_position + n
        if end <= self._max_replay_size:
            self._data[self._insert_position:end] = update
        else:
            first = self._max_replay_size - self._insert_position
            self._data[self._insert_position:] = update[:first]
            self._data[:end - self._max_replay_size] = update[first:]
        self._insert_position = end % self._max_replay_size
        self._size = min(self._size + n, self._max_replay_size)

    def sample_many(self, num_batches: int) -> Any:
        if self._size <= 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(
            0, self._size, size=(num_batches, self._sample_batch_size), endpoint=False)
        flat = self._data[idx.reshape(-1)]
        batch = _cast_transition(self._unflatten_fn(jnp.asarray(flat)), jnp.float32)

        def reshape_leaf(x):
            return jnp.reshape(x, (num_batches, self._sample_batch_size) + x.shape[1:])

        return jax.tree_util.tree_map(reshape_leaf, batch)


def _is_pixel_replay_path(path: tuple[Any, ...]) -> bool:
    for entry in path:
        key = getattr(entry, "key", None)
        if key is None:
            key = getattr(entry, "name", None)
        if isinstance(key, str) and key.startswith("pixels/"):
            return True
    return False


class HostTypedReplayBuffer:
    """CPU replay that can quantize only pixel leaves while preserving others."""

    def __init__(self, max_replay_size: int, dummy_data_sample: Any, sample_batch_size: int, storage_dtype, seed: int):
        if storage_dtype != "uint8_centered":
            raise ValueError(f"Unsupported typed host replay dtype={storage_dtype!r}")
        path_leaves, self._treedef = jax.tree_util.tree_flatten_with_path(dummy_data_sample)
        self._is_pixel = [_is_pixel_replay_path(path) for path, _ in path_leaves]
        self._data = []
        for is_pixel, leaf in zip(self._is_pixel, [leaf for _, leaf in path_leaves]):
            dtype = np.uint8 if is_pixel else np.float32
            self._data.append(np.zeros((max_replay_size,) + leaf.shape, dtype=dtype))
        self._max_replay_size = max_replay_size
        self._sample_batch_size = sample_batch_size
        self._insert_position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return sum(arr.nbytes for arr in self._data)

    @staticmethod
    def _encode_pixels(x: np.ndarray) -> np.ndarray:
        return np.rint(np.clip((x + 0.5) * 255.0, 0.0, 255.0)).astype(np.uint8)

    @staticmethod
    def _decode_pixels(x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32) / 255.0 - 0.5

    def insert(self, samples: Any) -> None:
        leaves = jax.tree_util.tree_leaves(samples)
        n = leaves[0].shape[0]
        if n >= self._max_replay_size:
            leaf_updates = []
            for is_pixel, leaf in zip(self._is_pixel, leaves):
                update = np.asarray(leaf)[-self._max_replay_size:]
                leaf_updates.append(self._encode_pixels(update) if is_pixel else update.astype(np.float32))
            for storage, update in zip(self._data, leaf_updates):
                storage[...] = update
            self._insert_position = 0
            self._size = self._max_replay_size
            return

        end = self._insert_position + n
        for storage, is_pixel, leaf in zip(self._data, self._is_pixel, leaves):
            update = np.asarray(leaf)
            update = self._encode_pixels(update) if is_pixel else update.astype(np.float32)
            if end <= self._max_replay_size:
                storage[self._insert_position:end] = update
            else:
                first = self._max_replay_size - self._insert_position
                storage[self._insert_position:] = update[:first]
                storage[:end - self._max_replay_size] = update[first:]
        self._insert_position = end % self._max_replay_size
        self._size = min(self._size + n, self._max_replay_size)

    def sample_many(self, num_batches: int) -> Any:
        if self._size <= 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = self._rng.integers(
            0, self._size, size=(num_batches, self._sample_batch_size), endpoint=False)
        flat_idx = idx.reshape(-1)
        leaves = []
        for storage, is_pixel in zip(self._data, self._is_pixel):
            sampled = storage[flat_idx]
            sampled = self._decode_pixels(sampled) if is_pixel else sampled.astype(np.float32)
            sampled = sampled.reshape((num_batches, self._sample_batch_size) + sampled.shape[1:])
            leaves.append(jnp.asarray(sampled, dtype=jnp.float32))
        return self._treedef.unflatten(leaves)


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

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
    gradient_steps: jax.Array


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
    # Force dm_control-style tolerance reward in vision mode (per-step in [0,1],
    # episode in [0, 1000]) instead of the additive-penalty vision reward.
    # Wrap with a throwaway metrics dict so we don't introduce new keys into
    # state.metrics (which would break the action-repeat scan pytree carry).
    if cfg.dmc_reward and hasattr(raw_env, "_dense_reward"):
        _dense = raw_env._dense_reward
        def _dmc_reward_fn(data, action, info, metrics):
            return _dense(data, action, info, {})
        raw_env._get_reward = _dmc_reward_fn
    # brax EpisodeWrapper measures episode_length in physics steps (it adds
    # action_repeat to its internal counter each call). Passing the raw
    # cfg.episode_length here gives 1000 physics / 8 repeat = 125 agent steps
    # per episode. The previous code divided once here AND brax divided again
    # internally, truncating episodes to 16 agent steps and breaking learning.
    return mp_wrapper.wrap_for_brax_training(
        raw_env,
        episode_length=cfg.episode_length,
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

    # Warm up Warp CUDA kernels for both envs before any JAX JIT.
    #
    # Problem on CUDA driver < 12.3 (e.g. RTX 3090 with driver 12.2):
    #   cuModuleLoad() is not permitted during an active CUDA stream capture.
    #   mujoco_warp creates a nworld-specific module specialisation (f60f76d for
    #   nworld=16) lazily on first call. If that first call happens inside JAX's
    #   XLA graph compilation (which uses CUDA stream capture), cuModuleLoad
    #   fails with CUDA error 900.
    #   wp.force_load() can only load ALREADY-REGISTERED modules; f60f76d is not
    #   registered until eval_env.reset() is first called, so pre-loading alone
    #   is insufficient.
    #
    # Fix:
    #   1. Set enable_graph_capture_module_load_by_default=False so that Warp's
    #      own internal graph capture context does not start a stream capture.
    #   2. Wrap warmup in jax.disable_jit() so XLA does not JIT-compile (and
    #      therefore does not start a CUDA stream capture) during the first call.
    #   3. With no active CUDA stream capture, cuModuleLoad succeeds on all
    #      CUDA driver versions.
    #   4. After warmup, f60f76d is registered; wp.force_load() loads it
    #      (and all other modules) cleanly outside any capture.
    #   On CUDA >= 12.3 the same path works and is equally fast.
    _prev_wp_gcml = wp.config.enable_graph_capture_module_load_by_default
    wp.config.enable_graph_capture_module_load_by_default = False

    print("  Warming up Warp kernels (graph-capture-safe, jit disabled)…")
    with jax.disable_jit():
        _wk = jax.random.PRNGKey(0)
        _wst = eval_env.reset(jax.random.split(_wk, cfg.num_eval_envs))
        jax.block_until_ready(jax.tree_util.tree_leaves(_wst))
        _wa = jnp.zeros((_wst.obs[next(iter(_wst.obs))].shape[0], train_env.action_size))
        _wresult = eval_env.step(_wst, _wa)
        jax.block_until_ready(jax.tree_util.tree_leaves(_wresult))
        _wst2 = train_env.reset(jax.random.split(_wk, cfg.num_envs))
        jax.block_until_ready(jax.tree_util.tree_leaves(_wst2))
        _wa2 = jnp.zeros((_wst2.obs[next(iter(_wst2.obs))].shape[0], train_env.action_size))
        _wresult2 = train_env.step(_wst2, _wa2)
        jax.block_until_ready(jax.tree_util.tree_leaves(_wresult2))
        del _wk, _wst, _wa, _wresult, _wst2, _wa2, _wresult2

    wp.config.enable_graph_capture_module_load_by_default = _prev_wp_gcml
    print("  Warp warmup complete.")

    # Force-load ALL registered Warp modules now that f60f76d is registered.
    # This runs outside any CUDA stream capture so cuModuleLoad always succeeds.
    print("  Post-warmup Warp force_load (ensures all modules loaded)…")
    wp.force_load(device="cuda:0")
    print("  Warp force_load complete.")

    # Obs / action sizes
    full_obs_size = train_env.observation_size   # includes batch dim: {k: (N,H,W,C)}
    raw_obs_size = {k: v[1:] for k, v in full_obs_size.items()}  # {k: (H,W,C)}
    replay_obs_size = {}
    for k, v in raw_obs_size.items():
        if k.startswith("pixels/") and cfg.frame_stack > 1:
            replay_obs_size[k] = v[:-1] + (v[-1] * cfg.frame_stack,)
        else:
            replay_obs_size[k] = v
    obs_size = dict(replay_obs_size)
    if cfg.crop_size > 0:
        obs_size = {
            k: ((cfg.crop_size, cfg.crop_size) + v[-1:] if k.startswith("pixels/") else v)
            for k, v in obs_size.items()
        }
    action_size = train_env.action_size

    print(f"  obs_size (per env): {obs_size}")
    print(f"  action_size: {action_size}")

    # Networks + losses
    sac_net = make_sac_networks_vision(obs_size, action_size, cfg)
    make_policy = sac_networks.make_inference_fn(sac_net)

    target_entropy = cfg.target_entropy
    alpha_loss_fn, critic_loss_fn, actor_loss_fn = make_sac_losses_vision(
        sac_network=sac_net,
        reward_scaling=cfg.reward_scaling,
        discounting=cfg.discounting,
        target_entropy=target_entropy,
    )
    policy_optimizer = optax.adam(cfg.learning_rate)
    q_optimizer = optax.adam(cfg.learning_rate)
    alpha_optimizer = optax.adam(cfg.alpha_learning_rate)

    alpha_update = gradients.gradient_update_fn(
        alpha_loss_fn, alpha_optimizer, pmap_axis_name=None)
    critic_update = gradients.gradient_update_fn(
        critic_loss_fn, q_optimizer, pmap_axis_name=None)
    actor_update = gradients.gradient_update_fn(
        actor_loss_fn, policy_optimizer, pmap_axis_name=None)

    replay_pixel_dtype = _pixel_storage_dtype(cfg.replay_pixel_dtype)

    # Replay buffer: flat storage in the requested reduced precision. Samples
    # are cast back to float32 before augmentation/network use.
    dummy_obs = {k: jnp.zeros(v, dtype=jnp.float32) for k, v in replay_obs_size.items()}
    dummy_action = jnp.zeros(action_size)
    dummy_transition = types.Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": jnp.zeros(())}},
    )
    if cfg.replay_backend == "device":
        if replay_pixel_dtype == "uint8_centered":
            raise ValueError("uint8_centered replay is supported only with --replay-backend host")
        replay_buffer = UniformFlatReplayBuffer(
            max_replay_size=cfg.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=cfg.batch_size,
            storage_dtype=replay_pixel_dtype,
        )
    elif cfg.replay_backend == "host":
        replay_cls = HostTypedReplayBuffer if replay_pixel_dtype == "uint8_centered" else HostFlatReplayBuffer
        replay_buffer = replay_cls(
            max_replay_size=cfg.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=cfg.batch_size,
            storage_dtype=replay_pixel_dtype,
            seed=cfg.seed + 17,
        )
        print(f"  host replay bytes: {replay_buffer.nbytes / (1024 ** 3):.2f} GiB")
    else:
        raise ValueError(f"Unsupported replay_backend={cfg.replay_backend!r}")

    # Initialize training state
    rng, key_policy, key_q, key_buf = jax.random.split(rng, 4)
    policy_params = sac_net.policy_network.init(key_policy)
    q_params = sac_net.q_network.init(key_q)
    if cfg.tie_actor_critic_encoder:
        policy_params = _copy_rad_conv_weights_to_policy(policy_params, q_params)
    training_state = SACTrainingState(
        policy_params=policy_params,
        policy_optimizer_state=policy_optimizer.init(policy_params),
        q_params=q_params,
        q_optimizer_state=q_optimizer.init(q_params),
        target_q_params=jax.tree_util.tree_map(jnp.array, q_params),
        log_alpha=jnp.array(np.log(cfg.init_temperature), dtype=jnp.float32),
        alpha_optimizer_state=alpha_optimizer.init(
            jnp.array(np.log(cfg.init_temperature), dtype=jnp.float32)),
        normalizer_params=jnp.zeros(()),     # dummy: pixels not normalized
        env_steps=jnp.zeros((), dtype=jnp.int32),
        gradient_steps=jnp.zeros((), dtype=jnp.int32),
    )

    # Initialize env state and replay buffer
    rng, key_env, key_eval = jax.random.split(rng, 3)
    env_keys = jax.random.split(key_env, cfg.num_envs)
    env_state = jax.jit(train_env.reset)(env_keys)
    obs_state = _repeat_stack_pixels_sac(env_state.obs, cfg.frame_stack)
    buffer_state = replay_buffer.init(key_buf) if cfg.replay_backend == "device" else None

    # ------------------------------------------------------------------ #
    # Inner step fn: collect 1 env step + (optionally) 1 gradient update #
    # ------------------------------------------------------------------ #

    def sgd_step(ts: SACTrainingState, transitions: types.Transition, key: jax.Array):
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
        alpha = jnp.exp(ts.log_alpha)
        update_step = ts.gradient_steps

        def update_alpha(_):
            return alpha_update(
                ts.log_alpha, ts.policy_params, ts.normalizer_params,
                transitions, key_alpha, optimizer_state=ts.alpha_optimizer_state)

        def skip_alpha(_):
            loss = alpha_loss_fn(
                ts.log_alpha, ts.policy_params, ts.normalizer_params,
                transitions, key_alpha)
            return loss, ts.log_alpha, ts.alpha_optimizer_state

        do_alpha_update = (update_step % cfg.alpha_update_frequency) == 0
        alpha_loss, new_log_alpha, new_alpha_opt = jax.lax.cond(
            do_alpha_update, update_alpha, skip_alpha, operand=None)
        critic_loss, new_q_params, new_q_opt = critic_update(
            ts.q_params, ts.policy_params, ts.normalizer_params,
            ts.target_q_params, alpha, transitions, key_critic,
            optimizer_state=ts.q_optimizer_state)

        def update_actor(_):
            return actor_update(
                ts.policy_params, ts.normalizer_params, new_q_params, alpha,
                transitions, key_actor, optimizer_state=ts.policy_optimizer_state)

        def skip_actor(_):
            loss = actor_loss_fn(
                ts.policy_params, ts.normalizer_params, new_q_params, alpha,
                transitions, key_actor)
            return loss, ts.policy_params, ts.policy_optimizer_state

        do_actor_update = (update_step % cfg.actor_update_frequency) == 0
        actor_loss, new_policy_params, new_policy_opt = jax.lax.cond(
            do_actor_update, update_actor, skip_actor, operand=None)
        if cfg.tie_actor_critic_encoder:
            new_policy_params = _copy_rad_conv_weights_to_policy(
                new_policy_params, new_q_params)

        target_encoder_tau = cfg.encoder_tau if cfg.encoder_tau > 0 else cfg.tau
        updated_target_q = _soft_update_q_params(
            ts.target_q_params, new_q_params, cfg.tau, target_encoder_tau)
        do_target_update = (update_step % cfg.target_update_frequency) == 0
        new_target_q = jax.lax.cond(
            do_target_update,
            lambda _: updated_target_q,
            lambda _: ts.target_q_params,
            operand=None,
        )

        return ts.replace(
            policy_params=new_policy_params,
            policy_optimizer_state=new_policy_opt,
            q_params=new_q_params,
            q_optimizer_state=new_q_opt,
            target_q_params=new_target_q,
            log_alpha=new_log_alpha,
            alpha_optimizer_state=new_alpha_opt,
            gradient_steps=ts.gradient_steps + 1,
        ), {"alpha_loss": alpha_loss, "critic_loss": critic_loss, "actor_loss": actor_loss}

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3), static_argnums=(5,))
    def training_epoch(ts, env_st, obs_st, buf_st, key, steps):
        """One epoch: `steps` env-steps + SAC updates."""
        def step_fn(carry, _):
            ts, env_st, obs_st, buf_st, key = carry
            key, act_key, upd_key = jax.random.split(key, 3)

            # Collect: infer action from current obs
            obs = obs_st
            policy_obs = _center_crop_pixels_sac(obs_st, cfg.crop_size)
            dist_params = sac_net.policy_network.apply(
                ts.normalizer_params, ts.policy_params, policy_obs)
            action = sac_net.parametric_action_distribution.sample(dist_params, act_key)
            new_env_st = train_env.step(env_st, action)
            next_obs_st = _append_stack_pixels_sac(
                obs_st, new_env_st.obs, new_env_st.done, cfg.frame_stack)

            # Store transition (num_envs transitions per step)
            transition = types.Transition(
                observation=obs,
                action=action,
                reward=new_env_st.reward,
                discount=1.0 - new_env_st.done,
                next_observation=next_obs_st,
                extras={"state_extras": {"truncation": new_env_st.info.get("truncation",
                                                           jnp.zeros(cfg.num_envs))}},
            )
            buf_st = replay_buffer.insert(buf_st, transition)

            def update_once(update_carry, _):
                ts, buf_st, key = update_carry
                key, key_aug_obs, key_aug_nobs, upd_key = jax.random.split(key, 4)
                buf_st, sampled = replay_buffer.sample(buf_st)
                if cfg.crop_size > 0:
                    sampled = sampled._replace(
                        observation=_random_crop_pixels_sac(
                            sampled.observation, key_aug_obs, cfg.crop_size),
                        next_observation=_random_crop_pixels_sac(
                            sampled.next_observation, key_aug_nobs, cfg.crop_size),
                    )
                elif cfg.augment_pixels:
                    sampled = sampled._replace(
                        observation=_random_translate_pixels_sac(
                            sampled.observation, key_aug_obs),
                        next_observation=_random_translate_pixels_sac(
                            sampled.next_observation, key_aug_nobs),
                    )
                ts, metrics = sgd_step(ts, sampled, upd_key)
                return (ts, buf_st, key), metrics

            (ts, buf_st, key), metrics = jax.lax.scan(
                update_once, (ts, buf_st, upd_key), None,
                length=cfg.grad_updates_per_step)
            ts = ts.replace(env_steps=ts.env_steps + cfg.num_envs)
            return (ts, new_env_st, next_obs_st, buf_st, key), metrics

        (ts, env_st, obs_st, buf_st, key), metrics = jax.lax.scan(
            step_fn, (ts, env_st, obs_st, buf_st, key), None, length=steps)
        return ts, env_st, obs_st, buf_st, key, metrics

    # Prefill: collect min_replay_size transitions without updates
    @jax.jit
    def prefill_step(env_st, obs_st, buf_st, key):
        key, act_key = jax.random.split(key)
        action = jax.random.uniform(
            act_key, (cfg.num_envs, action_size), minval=-1.0, maxval=1.0)
        new_env_st = train_env.step(env_st, action)
        next_obs_st = _append_stack_pixels_sac(
            obs_st, new_env_st.obs, new_env_st.done, cfg.frame_stack)
        transition = types.Transition(
            observation=obs_st,
            action=action,
            reward=new_env_st.reward,
            discount=1.0 - new_env_st.done,
            next_observation=next_obs_st,
            extras={"state_extras": {"truncation": new_env_st.info.get(
                "truncation", jnp.zeros(cfg.num_envs))}},
        )
        buf_st = replay_buffer.insert(buf_st, transition)
        return new_env_st, next_obs_st, buf_st, key

    @jax.jit
    def host_prefill_step(env_st, obs_st, key):
        key, act_key = jax.random.split(key)
        action = jax.random.uniform(
            act_key, (cfg.num_envs, action_size), minval=-1.0, maxval=1.0)
        new_env_st = train_env.step(env_st, action)
        next_obs_st = _append_stack_pixels_sac(
            obs_st, new_env_st.obs, new_env_st.done, cfg.frame_stack)
        transition = types.Transition(
            observation=obs_st,
            action=action,
            reward=new_env_st.reward,
            discount=1.0 - new_env_st.done,
            next_observation=next_obs_st,
            extras={"state_extras": {"truncation": new_env_st.info.get(
                "truncation", jnp.zeros(cfg.num_envs))}},
        )
        return new_env_st, next_obs_st, key, transition

    @jax.jit
    def host_collect_step(ts, env_st, obs_st, key):
        key, act_key = jax.random.split(key)
        policy_obs = _center_crop_pixels_sac(obs_st, cfg.crop_size)
        dist_params = sac_net.policy_network.apply(
            ts.normalizer_params, ts.policy_params, policy_obs)
        action = sac_net.parametric_action_distribution.sample(dist_params, act_key)
        new_env_st = train_env.step(env_st, action)
        next_obs_st = _append_stack_pixels_sac(
            obs_st, new_env_st.obs, new_env_st.done, cfg.frame_stack)
        transition = types.Transition(
            observation=obs_st,
            action=action,
            reward=new_env_st.reward,
            discount=1.0 - new_env_st.done,
            next_observation=next_obs_st,
            extras={"state_extras": {"truncation": new_env_st.info.get(
                "truncation", jnp.zeros(cfg.num_envs))}},
        )
        ts = ts.replace(env_steps=ts.env_steps + cfg.num_envs)
        return ts, new_env_st, next_obs_st, key, transition

    @functools.partial(jax.jit, donate_argnums=(0,))
    def host_update_many(ts, sampled_batches, key):
        def update_once(carry, sampled):
            ts, key = carry
            key, key_aug_obs, key_aug_nobs, upd_key = jax.random.split(key, 4)
            if cfg.crop_size > 0:
                sampled = sampled._replace(
                    observation=_random_crop_pixels_sac(
                        sampled.observation, key_aug_obs, cfg.crop_size),
                    next_observation=_random_crop_pixels_sac(
                        sampled.next_observation, key_aug_nobs, cfg.crop_size),
                )
            elif cfg.augment_pixels:
                sampled = sampled._replace(
                    observation=_random_translate_pixels_sac(
                        sampled.observation, key_aug_obs),
                    next_observation=_random_translate_pixels_sac(
                        sampled.next_observation, key_aug_nobs),
                )
            ts, metrics = sgd_step(ts, sampled, upd_key)
            return (ts, key), metrics

        (ts, key), metrics = jax.lax.scan(
            update_once, (ts, key), sampled_batches)
        return ts, key, metrics

    # ---------------------------------------------------------------- #
    # Warmup: fill replay buffer with random actions                    #
    # ---------------------------------------------------------------- #
    print(f"  Prefilling replay buffer ({cfg.min_replay_size} transitions)…")
    warmup_steps = max(cfg.min_replay_size // cfg.num_envs, 1)
    for _ in range(warmup_steps):
        rng, step_key = jax.random.split(rng)
        if cfg.replay_backend == "device":
            env_state, obs_state, buffer_state, rng = prefill_step(
                env_state, obs_state, buffer_state, rng)
        else:
            env_state, obs_state, rng, transition = host_prefill_step(
                env_state, obs_state, rng)
            replay_buffer.insert(transition)

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

    # Stable JIT-compiled eval: defined once so JAX compiles it only on the first
    # call (autotune runs once, Warp loads all block-dim variants) and then reuses
    # the same XLA executable for every subsequent eval iteration.
    @jax.jit
    def run_eval(policy_params, normalizer_params, eval_key):
        eval_st = eval_env.reset(jax.random.split(eval_key, cfg.num_eval_envs))
        eval_obs_st = _repeat_stack_pixels_sac(eval_st.obs, cfg.frame_stack)
        eval_policy_fn = make_policy((normalizer_params, policy_params), deterministic=True)

        def _eval_step(carry, _):
            st, obs_st, k = carry
            k, sk = jax.random.split(k)
            policy_obs = _center_crop_pixels_sac(obs_st, cfg.crop_size)
            a, _ = eval_policy_fn(policy_obs, sk)
            st = eval_env.step(st, a)
            obs_st = _append_stack_pixels_sac(obs_st, st.obs, st.done, cfg.frame_stack)
            return (st, obs_st, k), st.reward

        (_, _, _), rew = jax.lax.scan(
            _eval_step, (eval_st, eval_obs_st, eval_key), None, length=agent_ep_len)
        return jnp.sum(rew, axis=0).mean()

    t0 = time.time()
    last_log = t0
    eval_count = 0

    while int(training_state.env_steps) < total_env_steps:
        remaining_env_steps = total_env_steps - int(training_state.env_steps)
        steps_to_run = min(scan_steps_per_eval, remaining_env_steps // cfg.num_envs)
        if steps_to_run <= 0:
            break
        rng, epoch_key = jax.random.split(rng)
        if cfg.replay_backend == "device":
            training_state, env_state, obs_state, buffer_state, rng, epoch_metrics = training_epoch(
                training_state, env_state, obs_state, buffer_state, epoch_key, steps_to_run)

            env_steps = int(training_state.env_steps)
            avg_critic = float(jnp.mean(epoch_metrics["critic_loss"]))
            avg_actor = float(jnp.mean(epoch_metrics["actor_loss"]))
            alpha = float(jnp.exp(training_state.log_alpha))
            del epoch_metrics
        else:
            critic_total = 0.0
            actor_total = 0.0
            metric_count = 0
            for _ in range(steps_to_run):
                epoch_key, collect_key, update_key = jax.random.split(epoch_key, 3)
                training_state, env_state, obs_state, _, transition = host_collect_step(
                    training_state, env_state, obs_state, collect_key)
                replay_buffer.insert(transition)
                sampled_batches = replay_buffer.sample_many(cfg.grad_updates_per_step)
                training_state, epoch_key, update_metrics = host_update_many(
                    training_state, sampled_batches, update_key)
                critic_total += float(jnp.mean(update_metrics["critic_loss"]))
                actor_total += float(jnp.mean(update_metrics["actor_loss"]))
                metric_count += 1
                del sampled_batches, update_metrics, transition
            rng = epoch_key
            env_steps = int(training_state.env_steps)
            avg_critic = critic_total / max(metric_count, 1)
            avg_actor = actor_total / max(metric_count, 1)
            alpha = float(jnp.exp(training_state.log_alpha))

        # Eval
        rng, eval_key = jax.random.split(rng)
        er = float(run_eval(
            training_state.policy_params,
            training_state.normalizer_params,
            eval_key))

        elapsed = time.time() - t0
        sps = int(env_steps / elapsed) if elapsed > 1 else 0

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
