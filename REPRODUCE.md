# REPRODUCE

How to reproduce every result in this repo. Keep this file in sync with
[docs/PLAN.md](docs/PLAN.md).

## 0. Prereqs

- Vast.ai API key configured locally (`vastai set api-key <KEY>`).
- W&B account (entity `sudingli21`, project `rad-se`).
- The autosota-lite scheduler at
  `/workspace/autosota-lite/plugins/autosota-lite/skills/autosota-vastai-scheduler/scripts/vastai_scheduler.py`.

## 1. Pull references

```bash
cd /workspace/se-research-projects/rad-se
mkdir -p references
# Drop RAD (arXiv 2004.14990) and SISA (IJCAI-23) PDFs here.
```

## 2. RAD baseline (M1)

Each task × seed launches one Vast.ai job. Total: 9 jobs.

```bash
bash scripts/launch_rad_baseline.sh cartpole swingup 23
bash scripts/launch_rad_baseline.sh cartpole swingup 42
bash scripts/launch_rad_baseline.sh cartpole swingup 7
# ... acrobot swingup, cheetah run
```

Per-run artifacts land under `runs/<task>__seed<n>/`.

Acceptance: `cartpole swingup` eval@190k ≥ 800 on ≥2 of 3 seeds.

## 3. SISA reimplementation (M2)

Smoke first, then full:

```bash
bash scripts/launch_sisa_smoke.sh   # 5k steps, sanity-check losses
bash scripts/launch_sisa_full.sh cartpole swingup 23
```

## 4. Comparison grid (M3)

```bash
bash scripts/launch_grid.sh   # 4 methods × 3 tasks × 3 seeds = 36 jobs
python scripts/aggregate.py --runs runs/ --out results/grid.csv
python scripts/report.py --csv results/grid.csv --out results/REPORT.md
```

Gate: do not proceed to M4 unless SISA(full) > RAD on ≥2/3 tasks at p<0.05.

## 5. Optimization probes (M4)

Only if M3 passes. See `docs/PLAN.md` §M4.

## Cleanup

Every launch script pairs with `monitor_vastai_instance.sh` from
`/workspace/rad-vastai-run/`. After each batch:

```bash
vastai show instances --raw | jq '[.[] | select(.label|startswith("rad-se-"))]'
```

should return `[]`.
