# RAD-SE Handoff: Continuing on a 24 GB GPU Server

Audience: future you / a collaborator picking this project up on a 24 GB GPU
(e.g. RTX 3090, A5000, A10, L4 24 GB, A6000 48 GB). Goal: get from a fresh box
to a healthy 500k-step SAC + 5M-step PPO baseline with the **dm-control
reward scale** in one sitting.

Pair this doc with [ITERATION_LOG.md](ITERATION_LOG.md) for the experimental
history and rationale of every design choice.

---

## 1. Where this project lives

- Repo: `/workspace/se-research-projects/rad-se` (in the original dev box).
- Two trainable entry points:
  - SAC (preferred when memory allows): [src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py)
  - PPO (works on 12 GB, slower to converge): [src/rad_se/rad_brax_ppo.py](../src/rad_se/rad_brax_ppo.py)
- Launchers in [scripts/](../scripts/). The 24 GB launcher is
  [scripts/run_brax_sac_3090.sh](../scripts/run_brax_sac_3090.sh) (now passes
  `--dmc-reward` by default).
- Result tree: `runs/<run_dir>/{config.json, metrics.jsonl, train.log}` plus a
  sibling `runs/<run_dir>.log` from the launcher.

## 2. Critical context (read first)

1. **Reward scale bug — RESOLVED.** mujoco_playground's `Balance` /
   `CartpoleSwingup` env uses an *additive penalty* reward in `vision=True`
   mode, episode range ≈ [−3900, +100]. The dm_control paper's [0, 1000]
   numbers come from the *non-vision* `_dense_reward` (tolerance product).
   The flag `--dmc-reward` monkey-patches `env._get_reward` to
   `_dense_reward` (with a throwaway metrics dict so the `lax.scan` carry
   pytree stays stable across action_repeat). Always pass it on vision
   pixel envs.
   - See [docs/ITERATION_LOG.md §8](ITERATION_LOG.md#8-resolved-vision-reward-root-cause).
2. **12 GB bottleneck = replay-in-JIT-carry.** brax's replay sits inside the
   training state (so it goes through every `jit`/`scan`). At 100×100×9 pixels
   ×f32, you hit ~12.7 GiB before the optimizer's scratch. On 24 GB the
   bottleneck disappears — go back to **f32 device replay** and **autotune
   ON**. On 12 GB we used a host-pinned numpy replay + sample-to-device path
   (`rad_brax_sac.py` already has that toggle; on 24 GB you don't need it).
3. **PPO converges slowly on pixels.** 500k env steps reaches ER ≈ 11.7 (out
   of 1000). 5M is the realistic minimum; 10M+ is preferable. SAC reaches the
   same band in ~30k env steps on the 3060 host-replay run.

## 3. Recommended GPUs (Vast.ai)

| GPU | VRAM | $/hr (mid) | Notes |
|-----|-----:|-----------:|-------|
| RTX 3090 | 24 GB | ~0.20 | sweet spot for this project |
| RTX A5000 | 24 GB | ~0.25 | similar |
| L4 | 24 GB | ~0.35 | low power, slightly slower CNN |
| A6000 | 48 GB | ~0.50 | overkill but room for replay ≥ 200k |

CUDA driver ≥ 12.2 is required by the playground stack we pin (12.5/12.9 both
work). Disk: 60 GB image + ~10 GB run logs.

## 4. Fresh-box bootstrap

```bash
# 1. Clone
git clone <your-fork-url> rad-se && cd rad-se
# Or rsync from your old box:
#   rsync -av --exclude runs/ --exclude wandb/ \
#       old:/workspace/se-research-projects/rad-se/ ./

# 2. System CUDA must be ≥ 12.2. Verify:
nvidia-smi
nvcc --version  # any 12.x is fine

# 3. Python env (we used 3.12, 3.10/3.11 also OK)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

# 4. JAX with CUDA
pip install -U "jax[cuda12]==0.4.30"

# 5. Project deps
pip install -e .
# This pulls flax, optax, orbax, playground, wandb. If `playground` fails,
# install upstream pins explicitly:
pip install "brax==0.14.2" "mujoco==3.2.7" "mujoco-mjx==3.2.7" \
            "mujoco-playground==1.12.1"

# 6. Smoke test (≈90 s, validates env + reward + JIT)
export PYTHONPATH=src JAX_DEFAULT_MATMUL_PRECISION=highest \
       XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 -u src/rad_se/rad_brax_ppo.py \
    --env CartpoleSwingup --seed 23 --smoke --dmc-reward \
    --work-dir runs/_smoke_ppo_dmc
# Expected: final ER in [0, 20] in [0, 1000] range. NO negative ER.
```

## 5. Production runs to launch on the 24 GB box

All commands assume `cd rad-se` and the venv active.

### 5a. SAC 500k baseline (highest-value run, ~60 min)
```bash
bash scripts/run_brax_sac_3090.sh CartpoleSwingup 23
bash scripts/run_brax_sac_3090.sh CartpoleSwingup 42
bash scripts/run_brax_sac_3090.sh CartpoleSwingup 7
```
- Replay: 50k f32 device, num_envs=16, autotune=2, batch=256. Already has
  `--dmc-reward`.
- Expected ER@500k: **≥ 700** on CartpoleSwingup if RAD aug is helping;
  pre-fix the 3060 best was −2719 on the *wrong* reward scale.
- Result paths:
  - `runs/sac_3090_CartpoleSwingup_s{23,42,7}/`
  - log: same dir, `train.log`.

### 5b. RAD ablation (paired-by-seed)
The flag taxonomy is already in [src/rad_se/rad_brax_sac.py](../src/rad_se/rad_brax_sac.py)
(`Config` near line 80–160). Knobs we already ablated on 12 GB:
- `--augment-pixels / --no-augment-pixels` (RAD random crop on)
- `--tied-encoder / --no-tied-encoder`
- `--rad-update-freq` (default 2)
- `--encoder-tau` (default same as `--tau`)
- `--framestack` (default 3)
- `--reward-once` (default true, do not regress)

Recommended grid on 24 GB (12 runs, ~12 h total):
```bash
for seed in 23 42 7; do
  for tag in baseline noaug notied; do
    extra=""
    [[ $tag == noaug ]]  && extra="--no-augment-pixels"
    [[ $tag == notied ]] && extra="--no-tied-encoder"
    WORKDIR=runs/sac_3090_${tag}_CartpoleSwingup_s${seed}
    mkdir -p $WORKDIR
    PYTHONPATH=src python3 -u src/rad_se/rad_brax_sac.py \
        --env CartpoleSwingup --seed $seed \
        --num-envs 16 --max-replay-size 50000 --min-replay-size 2000 \
        --batch-size 256 --total-timesteps 500000 \
        --num-evals 20 --num-eval-envs 16 \
        --episode-length 1000 --action-repeat 8 \
        --learning-rate 3e-4 --discounting 0.99 --tau 0.005 \
        --reward-scaling 0.1 --augment-pixels --dmc-reward $extra \
        --work-dir $WORKDIR 2>&1 | tee $WORKDIR/train.log
  done
done
```

### 5c. PPO 5M (already running locally — can re-run on 24 GB ~3 h)
```bash
WORKDIR=runs/brax_ppo_CartpoleSwingup_s23_5M
mkdir -p $WORKDIR
PYTHONPATH=src python3 -u src/rad_se/rad_brax_ppo.py \
    --env CartpoleSwingup --seed 23 \
    --num-envs 256 --unroll-length 20 --batch-size 32 \
    --num-minibatches 8 --num-updates-per-batch 8 \
    --total-timesteps 5000000 --num-evals 50 --num-eval-envs 32 \
    --action-repeat 8 --episode-length 1000 --discounting 0.99 \
    --learning-rate 3e-4 --entropy-cost 0.01 --clipping-epsilon 0.2 \
    --max-grad-norm 1.0 --reward-scaling 0.1 --dmc-reward \
    --work-dir $WORKDIR 2>&1 | tee $WORKDIR.log
```
On 24 GB with autotune ON you can also try `--num-envs 1024` for higher SPS.

## 6. Configuration cheat-sheet (most-frequently changed flags)

| Flag | 12 GB default | 24 GB recommended | Effect |
|------|---------------|-------------------|--------|
| `--num-envs` (SAC) | 8 | 16 | parallel collection |
| `--max-replay-size` | 10000 host | 50000 device | replay diversity |
| `--batch-size` (SAC) | 32 | 256 | gradient SNR |
| `--num-envs` (PPO) | 256 | 1024 | throughput |
| `XLA_FLAGS autotune` | level=0 | level=2 | ~5× CNN speedup |
| `--dmc-reward` | **on** | **on** | dm_control reward scale |
| `--augment-pixels` | on | on | RAD random crop |

## 7. Known pitfalls

1. **Forgetting `--dmc-reward`** → ER stuck near −3000. Always check first
   eval emit is in [0, 1] not [−5, 0].
2. **brax replay-in-JIT-carry** on small VRAM → OOM during first JIT.
   Symptom: "out of memory while trying to allocate ... HLO temp ...".
   Fix on 24 GB is *not needed*; on smaller cards switch to host replay
   (`--host-replay --host-replay-dtype f16`).
3. **mjwarp module loading inside CUDA graph capture** on autotune ≥ 2
   crashes with `nworld=16`. The fix is `jax.disable_jit()` warmup; already
   in `rad_brax_sac.py`. If you change `num_envs` × `action_repeat` shapes
   you may have to re-trigger warmup.
4. **`state.metrics` pytree mismatch** if you re-introduce a custom reward
   that writes new dict keys. Always wrap with a throwaway dict
   (`_dmc_reward_fn` pattern in `make_envs`).
5. **wandb auth**: `wandb login` once on the new box, or run with
   `WANDB_MODE=offline`.

## 8. Open questions / TODO

- Are PPO numbers competitive with SAC at 5M? (Empirical answer pending — the
  5M run is in-flight.) Decision rule: if PPO@5M < SAC@500k, drop PPO.
- SISA encoder: not yet integrated into the JAX entry points. Port plan in
  [docs/SISA_NOTES.md](SISA_NOTES.md).
- Multi-task results (acrobot-swingup, cheetah-run): scripts exist
  (`scripts/run_k_sweep.sh`) but only cartpole has been validated post-fix.
- W&B project: `sudingli21/rad-se` (see README).

## 9. Quick health-check commands

```bash
# GPU
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

# Live training progress
tail -f runs/<run_dir>/train.log

# Episode-reward curve at a glance
grep '"eval/episode_reward"' runs/<run_dir>/metrics.jsonl \
  | python3 -c "import sys,json; [print(json.loads(l).get('step'), json.loads(l).get('eval/episode_reward')) for l in sys.stdin]"

# Best-of-run
python3 -c "
import json,sys,glob
for f in sorted(glob.glob('runs/*/metrics.jsonl')):
    best=max((json.loads(l).get('eval/episode_reward',-1e9) for l in open(f)),default=None)
    print(f, best)
"
```

## 10. State of the local 3060 box as of handoff

- Active processes: 5M PPO at `runs/brax_ppo_CartpoleSwingup_s23_5M/` (ETA ~6.6 h).
- Last completed run: 500k PPO `runs/brax_ppo_CartpoleSwingup_s23/`
  final ER 11.74 — proves the reward-scale fix and PPO plumbing, but plateaued.
- Code changes you inherit (already committed in-tree):
  - `--dmc-reward` flag in both `rad_brax_sac.py` and `rad_brax_ppo.py`
    `Config` + `make_envs`.
  - `scripts/run_brax_ppo.sh` and `scripts/run_brax_sac_3090.sh` now pass
    `--dmc-reward`.
- No outstanding patches.
