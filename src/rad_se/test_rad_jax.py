#!/usr/bin/env python3
"""Smoke test for rad_jax.py — fast CPU-only check, no env required for most checks.

Tests:
  1. Import + config parse
  2. conv_out_size math
  3. random_crop / center_crop shape
  4. Encoder, Actor, Critic forward pass on synthetic obs (CPU JAX)
  5. ReplayBuffer add/sample cycle
  6. critic update step (synthetic batch)
  7. actor+alpha update step
  8. Soft target update
  9. copy_conv_weights (tied-weight semantics)
"""
import os
import sys
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Make sure the package is importable from source
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from rad_se.rad_jax import (
    Config,
    PixelEncoder,
    Actor,
    Critic,
    LogAlpha,
    ReplayBuffer,
    Logger,
    conv_out_size,
    random_crop_np,
    center_crop_np,
    obs_to_uint8,
    uint8_to_float,
    copy_conv_weights_to_actor,
    soft_update_target,
    _hard_copy_critic,
    _update_critic,
    _update_actor_alpha,
)
import optax
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_cfg(**overrides) -> Config:
    cfg = Config(
        smoke=True,
        batch_size=4,
        num_filters=4,
        num_layers=2,
        hidden_dim=16,
        encoder_feature_dim=8,
        image_size=20,
        cam_res=24,
        replay_capacity=32,
        init_steps=2,
        total_timesteps=10,
        eval_freq=5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CFG = make_cfg()
ACTION_DIM = 2
IN_CHANNELS = 3  # grayscale frame stack


def test_conv_out_size():
    # 4 layers, 84px → 35
    assert conv_out_size(84, 4) == 35, f"got {conv_out_size(84, 4)}"
    # 2 layers, 84px → 39
    assert conv_out_size(84, 2) == 39
    # 2 layers, 20px (smoke)
    out = conv_out_size(20, 2)
    assert out > 0, f"smoke image_size=20 gave {out}"
    print("  conv_out_size: OK")


def test_crop_utils():
    imgs = np.random.randint(0, 256, (8, 24, 24, 3), dtype=np.uint8)
    cropped = random_crop_np(imgs, 20)
    assert cropped.shape == (8, 20, 20, 3)

    single = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
    cc = center_crop_np(single, 20)
    assert cc.shape == (20, 20, 3)

    float_obs = np.random.uniform(-0.5, 0.5, (24, 24, 3)).astype(np.float32)
    u8 = obs_to_uint8(float_obs)
    assert u8.dtype == np.uint8
    back = uint8_to_float(u8)
    assert back.dtype == np.float32
    assert back.min() >= 0.0 and back.max() <= 1.0
    print("  crop/convert utils: OK")


def test_encoder_forward():
    rngs = nnx.Rngs(0)
    enc = PixelEncoder(IN_CHANNELS, CFG.image_size, CFG.encoder_feature_dim,
                       CFG.num_layers, CFG.num_filters, rngs=rngs)
    x = jnp.ones((CFG.batch_size, CFG.image_size, CFG.image_size, IN_CHANNELS))
    out = enc(x)
    assert out.shape == (CFG.batch_size, CFG.encoder_feature_dim), f"shape={out.shape}"
    print("  PixelEncoder: OK")


def test_actor_forward():
    rngs = nnx.Rngs(0)
    actor = Actor(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    obs = jnp.ones((CFG.batch_size, CFG.image_size, CFG.image_size, IN_CHANNELS))
    rng = jax.random.PRNGKey(0)
    mu, pi, log_pi, log_std = actor(obs, rng)
    assert mu.shape == (CFG.batch_size, ACTION_DIM)
    assert pi is not None and log_pi is not None
    assert jnp.all(jnp.abs(mu) <= 1.0 + 1e-5), "mu not in [-1,1]"
    print("  Actor: OK")


def test_critic_forward():
    rngs = nnx.Rngs(0)
    critic = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    obs = jnp.ones((CFG.batch_size, CFG.image_size, CFG.image_size, IN_CHANNELS))
    act = jnp.zeros((CFG.batch_size, ACTION_DIM))
    q1, q2 = critic(obs, act)
    assert q1.shape == (CFG.batch_size, 1)
    assert q2.shape == (CFG.batch_size, 1)
    print("  Critic: OK")


def test_replay_buffer():
    buf = ReplayBuffer((CFG.cam_res, CFG.cam_res, 3), ACTION_DIM,
                       CFG.replay_capacity, CFG.image_size)
    obs = np.random.randint(0, 255, (CFG.cam_res, CFG.cam_res, 3), dtype=np.uint8)
    act = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(CFG.batch_size):
        buf.add(obs, act, 1.0, obs, 0.0)
    obs_b, acts_b, rews_b, nobs_b, notd_b = buf.sample_bs(CFG.batch_size)
    assert obs_b.shape == (CFG.batch_size, CFG.image_size, CFG.image_size, 3)
    assert rews_b.shape == (CFG.batch_size, 1)
    print("  ReplayBuffer: OK")


def test_update_critic():
    rngs = nnx.Rngs(0)
    actor         = Actor(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    critic        = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    critic_target = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    _hard_copy_critic(critic, critic_target)
    log_alpha = LogAlpha(0.1)
    critic_opt = nnx.Optimizer(critic, optax.adam(1e-3), wrt=nnx.Param)
    buf = ReplayBuffer((CFG.cam_res, CFG.cam_res, 3), ACTION_DIM,
                       CFG.replay_capacity, CFG.image_size)
    obs_np = np.random.randint(0, 255, (CFG.cam_res, CFG.cam_res, 3), dtype=np.uint8)
    act_np = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(CFG.batch_size):
        buf.add(obs_np, act_np, 1.0, obs_np, 0.0)
    batch = buf.sample_bs(CFG.batch_size)
    rng = jax.random.PRNGKey(1)
    loss, rng2 = _update_critic(
        actor, critic, critic_target, log_alpha,
        critic_opt, batch, rng, CFG.discount,
    )
    assert np.isfinite(float(loss)), f"critic loss not finite: {loss}"
    print(f"  update_critic: OK  loss={float(loss):.4f}")


def test_update_actor_alpha():
    rngs = nnx.Rngs(0)
    actor   = Actor(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    critic  = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    log_alpha = LogAlpha(0.1)
    actor_opt = nnx.Optimizer(actor.trunk, optax.adam(1e-3), wrt=nnx.Param)
    alpha_opt = nnx.Optimizer(log_alpha, optax.adam(1e-4), wrt=nnx.Param)
    buf = ReplayBuffer((CFG.cam_res, CFG.cam_res, 3), ACTION_DIM,
                       CFG.replay_capacity, CFG.image_size)
    obs_np = np.random.randint(0, 255, (CFG.cam_res, CFG.cam_res, 3), dtype=np.uint8)
    for _ in range(CFG.batch_size):
        buf.add(obs_np, np.zeros(ACTION_DIM, np.float32), 1.0, obs_np, 0.0)
    batch = buf.sample_bs(CFG.batch_size)
    rng = jax.random.PRNGKey(2)
    a_loss, al_loss, rng2 = _update_actor_alpha(
        actor, critic, log_alpha,
        actor_opt, alpha_opt,
        batch, rng, target_entropy=-float(ACTION_DIM),
    )
    assert np.isfinite(float(a_loss)), f"actor loss not finite: {a_loss}"
    print(f"  update_actor_alpha: OK  actor_loss={float(a_loss):.4f}  alpha_loss={float(al_loss):.4f}")


def test_soft_update():
    rngs = nnx.Rngs(0)
    src = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    tgt = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=nnx.Rngs(99))
    # Change src
    src.encoder.fc.kernel[...] = jnp.ones_like(src.encoder.fc.kernel[...]) * 2.0
    tgt.encoder.fc.kernel[...] = jnp.zeros_like(tgt.encoder.fc.kernel[...])
    soft_update_target(src, tgt, tau=1.0)
    assert abs(float(tgt.encoder.fc.kernel[...].mean()) - 2.0) < 1e-4
    print("  soft_update_target: OK")


def test_copy_conv_weights():
    rngs = nnx.Rngs(0)
    critic = Critic(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=rngs)
    actor  = Actor(IN_CHANNELS, CFG.image_size, ACTION_DIM, CFG, rngs=nnx.Rngs(99))
    # Force a known value
    for conv in critic.encoder.convs:
        conv.kernel[...] = jnp.ones_like(conv.kernel[...]) * 3.0
    copy_conv_weights_to_actor(critic, actor)
    for a_conv in actor.encoder.convs:
        assert abs(float(a_conv.kernel[...].mean()) - 3.0) < 1e-5
    print("  copy_conv_weights: OK")


if __name__ == "__main__":
    import pytest
    # Run without pytest for quick CI
    print("Running rad_jax smoke tests...")
    test_conv_out_size()
    test_crop_utils()
    test_encoder_forward()
    test_actor_forward()
    test_critic_forward()
    test_replay_buffer()
    test_update_critic()
    test_update_actor_alpha()

    # These two use pytest.approx; skip if not available
    test_soft_update()
    test_copy_conv_weights()

    print("\nAll smoke tests passed.")
