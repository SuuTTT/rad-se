#!/usr/bin/env python3
"""Diagnose a brax+MJX vision env without needing a trained policy.

Builds the *exact* same env as `rad_brax_ppo.py` / `rad_brax_sac.py` (so the
`patch_swingup_done` fix and `--dmc-reward` wrapper are applied), then rolls
out one episode with a chosen action policy:

  --action-mode zero    : action = 0  (do nothing)
  --action-mode random  : action ~ Uniform[-1, 1] (default)
  --action-mode bangbang: action = sign(sin(2π t / period))  — to test if
                          alternating forces can swing the pole up at all.

Outputs (in --work-dir):
  diag.mp4          : rendered video at --cam-res×--cam-res (uses obs frames
                       if --use-obs-frames, else env.render at higher res).
  diag.npz          : arrays {frames (T,H,W,3) uint8, actions (T,A) f32,
                              rewards (T,) f32, dones (T,) f32, qpos (T,nq),
                              qvel (T,nv)}.
  diag_report.txt   : per-step + summary stats.

Usage:
  PYTHONPATH=src python3 scripts/diagnose_brax_env.py \\
      --env CartpoleSwingup --seed 23 --dmc-reward \\
      --action-mode bangbang --period 16 \\
      --work-dir runs/_diag/cartpole_bangbang
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np

# Import shared env builder from the training script so we get the same
# wrappers, the swingup done-fix, and the --dmc-reward switch.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rad_se.rad_brax_ppo import Config as PPOConfig, make_envs as ppo_make_envs


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="CartpoleSwingup")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--episode-length", type=int, default=1000,
                   help="physics steps before action_repeat")
    p.add_argument("--action-repeat", type=int, default=8)
    p.add_argument("--cam-res", type=int, default=100)
    p.add_argument("--dmc-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-mode", choices=["zero", "random", "bangbang"],
                   default="random")
    p.add_argument("--period", type=int, default=16,
                   help="bangbang period in agent steps")
    p.add_argument("--work-dir", required=True)
    return p.parse_args()


def main():
    args = parse()
    out = Path(args.work_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = PPOConfig(
        env=args.env, seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        cam_res=args.cam_res,
        dmc_reward=args.dmc_reward,
        num_envs=1, num_eval_envs=1,
    )
    print(f"[diag] env={args.env} action_mode={args.action_mode} "
          f"dmc_reward={args.dmc_reward} cam_res={args.cam_res}")

    env = ppo_make_envs(cfg, num_envs=1, is_eval=True)
    n_agent_steps = cfg.episode_length // cfg.action_repeat
    print(f"[diag] agent steps per episode = {n_agent_steps} "
          f"({cfg.episode_length} physics / {cfg.action_repeat} repeat)")
    print(f"[diag] action_size = {env.action_size}")
    print(f"[diag] obs keys = {list(env.observation_size.keys())}")
    pixel_key = next((k for k in env.observation_size if "pixel" in k.lower()), None)
    print(f"[diag] pixel obs key = {pixel_key} shape = "
          f"{env.observation_size.get(pixel_key) if pixel_key else 'N/A'}")

    rng = jax.random.PRNGKey(args.seed)
    rng, rk = jax.random.split(rng)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    t0 = time.monotonic()
    state = jit_reset(jax.random.split(rk, 1))
    print(f"[diag] reset done in {time.monotonic() - t0:.1f}s")

    frames = []          # (T, H, W, 3) uint8
    actions = []         # (T, A)
    rewards = []         # (T,)
    dones = []           # (T,)
    qpos = []
    qvel = []

    A = env.action_size

    def get_action(step_i, key):
        if args.action_mode == "zero":
            return jnp.zeros((1, A), dtype=jnp.float32)
        if args.action_mode == "random":
            return jax.random.uniform(key, (1, A), minval=-1.0, maxval=1.0)
        # bangbang
        s = 1.0 if (step_i // args.period) % 2 == 0 else -1.0
        return jnp.full((1, A), s, dtype=jnp.float32)

    t_roll = time.monotonic()
    for i in range(n_agent_steps):
        rng, ak = jax.random.split(rng)
        act = get_action(i, ak)
        state = jit_step(state, act)
        obs = state.obs
        # extract pixel frame (1,H,W,3) → (H,W,3)
        if pixel_key is not None:
            frame = np.asarray(obs[pixel_key][0])
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8) if frame.max() <= 1.5 \
                        else frame.astype(np.uint8)
        else:
            frame = np.zeros((args.cam_res, args.cam_res, 3), dtype=np.uint8)
        frames.append(frame)
        actions.append(np.asarray(act[0]))
        rewards.append(float(state.reward[0]))
        dones.append(float(state.done[0]))
        # underlying mjx data (after action_repeat scan, .data is final state)
        try:
            qpos.append(np.asarray(state.data.qpos[0]))
            qvel.append(np.asarray(state.data.qvel[0]))
        except Exception:
            qpos.append(np.zeros(1)); qvel.append(np.zeros(1))

    elapsed = time.monotonic() - t_roll
    frames = np.stack(frames)            # (T, H, W, 3) uint8
    actions = np.stack(actions)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    qpos = np.stack(qpos)
    qvel = np.stack(qvel)

    # Summary stats
    ep_return = float(rewards.sum())
    first_done = int(np.argmax(dones > 0)) if dones.any() else -1
    pole_angle = qpos[:, 1] if qpos.shape[1] >= 2 else np.zeros(len(qpos))
    cart_pos   = qpos[:, 0] if qpos.shape[1] >= 1 else np.zeros(len(qpos))

    report = [
        f"=== DIAG REPORT ===",
        f"env={args.env} seed={args.seed} action_mode={args.action_mode} dmc_reward={args.dmc_reward}",
        f"agent_steps={n_agent_steps} action_repeat={cfg.action_repeat}",
        f"rollout wall: {elapsed:.1f}s ({n_agent_steps/elapsed:.0f} agent SPS)",
        f"",
        f"-- reward --",
        f"  episode_return = {ep_return:.3f}  (DMC tolerance max = {n_agent_steps*cfg.action_repeat})",
        f"  per-step mean  = {rewards.mean():.4f}   max = {rewards.max():.4f}   min = {rewards.min():.4f}",
        f"  per-step std   = {rewards.std():.4f}",
        f"  fraction reward > 0.5: {(rewards > 0.5).mean():.3f}",
        f"  fraction reward > 0.9: {(rewards > 0.9).mean():.3f}",
        f"",
        f"-- termination --",
        f"  any done?      = {bool(dones.any())}",
        f"  first done at  = {first_done}/{n_agent_steps}",
        f"  total dones    = {int(dones.sum())}",
        f"",
        f"-- action --",
        f"  shape         = {actions.shape}  range [{actions.min():.3f}, {actions.max():.3f}]",
        f"  mean = {actions.mean(0)}",
        f"  std  = {actions.std(0)}",
        f"",
        f"-- pixels --",
        f"  frames shape  = {frames.shape}  dtype = {frames.dtype}",
        f"  mean (per ch) = R={frames[..., 0].mean():.1f} G={frames[..., 1].mean():.1f} B={frames[..., 2].mean():.1f}",
        f"  std  (per ch) = R={frames[..., 0].std():.1f} G={frames[..., 1].std():.1f} B={frames[..., 2].std():.1f}",
        f"  temporal Δ    = mean |frame[t+1]-frame[t]| = {np.abs(np.diff(frames.astype(np.int16), axis=0)).mean():.2f}",
        f"",
        f"-- state (qpos) --",
        f"  cart_pos  range [{cart_pos.min():.3f}, {cart_pos.max():.3f}]  end = {cart_pos[-1]:.3f}",
        f"  pole_ang  range [{pole_angle.min():.3f}, {pole_angle.max():.3f}]  end = {pole_angle[-1]:.3f}",
        f"  pole upright fraction (|cos(ang)| > 0.9): "
            f"{float((np.cos(pole_angle) > 0.9).mean()):.3f}",
        f"  pole downward fraction (cos(ang) < -0.9): "
            f"{float((np.cos(pole_angle) < -0.9).mean()):.3f}",
        f"",
        f"=== END ===",
    ]
    (out / "diag_report.txt").write_text("\n".join(report))
    print("\n".join(report))

    # Save raw arrays
    np.savez_compressed(out / "diag.npz",
                        frames=frames, actions=actions, rewards=rewards,
                        dones=dones, qpos=qpos, qvel=qvel)

    # Write MP4 (if mediapy available)
    try:
        import mediapy
        mediapy.write_video(str(out / "diag.mp4"), frames, fps=30)
        print(f"[diag] wrote {out/'diag.mp4'}")
    except Exception as e:
        print(f"[diag] mediapy failed ({e}); writing PNG grid instead")
        try:
            from PIL import Image
            # 4×4 grid of evenly-spaced frames
            idx = np.linspace(0, len(frames) - 1, 16).astype(int)
            grid = np.concatenate([np.concatenate(list(frames[idx[r*4:(r+1)*4]]), axis=1)
                                   for r in range(4)], axis=0)
            Image.fromarray(grid).save(out / "diag_grid.png")
            print(f"[diag] wrote {out/'diag_grid.png'}")
        except Exception as e2:
            print(f"[diag] PIL failed too ({e2}); only diag.npz saved")


if __name__ == "__main__":
    main()
