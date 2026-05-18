# Distributed CPU Actors for RL — even without MuJoCo

> "We can use distributed CPUs to collect env data, even without MuJoCo."

This note covers **why** that pattern dominates industry, **how** it relates to the PPO-vs-SAC question, and **what** an implementation looks like on top of the current `rad-se` stack.

---

## 1. Why distributed CPU actors exist

Modern RL runs split work along *two* axes:

| Axis | Cheap on | Expensive on |
| --- | --- | --- |
| Environment stepping (physics, game logic, network sim) | many CPU cores, many machines | 1 GPU |
| Neural-net forward/backward (policy, value, Q, encoder) | 1 GPU | many CPUs |

If you put both on one GPU (like the brax/mjx vision pipeline we're running now), you get clean code and high single-GPU throughput on **simulators that fit on the GPU** (MJX, Brax, IsaacGym). The moment your env is a non-differentiable C++ engine, a game emulator, a real-world sim, or a stochastic network/market model, the GPU can't run it — you fall back to CPU stepping. At that point the only way to keep the GPU fed is to **parallelize env stepping across many CPU workers**.

Names you'll see for this:
- **Actor / Learner split** (IMPALA, SEED RL, Sample Factory, RLlib)
- **A3C / APE-X / R2D2** (DeepMind genealogy)
- **Rollout workers** (Ray RLlib terminology)
- **Vectorized envs** (gym / gymnasium `AsyncVectorEnv`, EnvPool) — same idea on one machine

---

## 2. How it ties into "industry prefers PPO over SAC"

Short answer: **mostly yes, and the CPU-actor scalability is one of the biggest reasons** — though not the only one.

### Reasons industry tends to pick PPO

1. **On-policy → embarrassingly parallel rollouts.** PPO discards data after one or a few epochs of updates. That means you can spin up *N* independent CPU workers, each running its own copy of the env with a stale-but-recent policy, and ship trajectories to the learner. No replay buffer correctness headaches. This is exactly the IMPALA/PPO-distributed pattern that powered OpenAI Five, OpenAI's Rubik's-cube hand, Dota / StarCraft work, and most game-AI RLHF post-training today.
2. **Robust to staleness.** A small policy-version skew between actor and learner is corrected by PPO's clipping; for SAC, off-policyness is "correct" in theory but in practice you need careful importance sampling or V-trace to scale to many actors. SAC's replay buffer expects roughly i.i.d. samples — many stale actors break that assumption.
3. **Hyper-param simplicity.** PPO has ~5 knobs that mostly transfer. SAC adds: replay size, ratio of env steps to gradient steps, target entropy, two critics, target tau, alpha learning rate. More moving parts → more risk in production.
4. **Discrete-action and large-action support.** PPO handles discrete / parametric / mixed action spaces uncomplicatedly. SAC's standard formulation is continuous-action; discrete SAC exists but is less battle-tested.
5. **GPU memory profile.** PPO's working set is `num_envs × unroll × obs_dim` of trajectory + 1 minibatch. SAC's is `replay_size × obs_dim` (often pixels) **plus** sampled minibatches **plus** target networks. With vision obs (100×100×3), replay alone can be 5–20 GB. We saw exactly this on the 2080 Ti just now — SAC tried to allocate 11.18 GiB just for the on-device replay scan; PPO had no equivalent OOM.
6. **Throughput at scale.** Sample Factory, EnvPool, and SEED-style infra routinely push PPO past 1M env steps/sec on a single 8-GPU box because rollouts are cheap CPU work and gradient steps are batched on GPU. SAC's update-per-step bottleneck makes that harder.

### Reasons SAC is still chosen

- **Sample efficiency from a fixed real-world dataset** (robotics, medical, finance). When *each* env step costs money or wall-clock minutes, off-policy + replay is dramatically better. DrQ-v2, RAD, CURL, and the whole pixel-control benchmark family use SAC for exactly this reason.
- **Single-machine, single-env experiments.** Most academic papers.
- **Easier to combine with offline RL / demonstrations** (BC + SAC, AWAC, CQL all build on the off-policy critic).

So the **rule of thumb in industry**:
- **Massive simulation + cheap data → PPO + distributed CPU actors.**
- **Expensive real-world data + one or few envs → SAC / off-policy.**

Our `rad-se` repo has both for a reason: SAC for the academic pixel-DMC benchmark (where PaperWithCode metrics live), PPO for verifying the brax pipeline and for compute-rich runs. The 500k PPO/SAC comparison we're running now will give us empirical numbers for *this* env to support either choice.

---

## 3. What "distributed CPU actors without MuJoCo" looks like architecturally

```
            ┌───────────────────────────────────────────────────┐
            │                   Learner (GPU)                   │
            │   policy θ, value V, replay (off-policy)          │
            │       │ params         ▲ trajectories             │
            │       ▼                │                          │
            │  [param server]   [batcher / replay]              │
            └────────┬──────────────────────────────────────────┘
                     │ pull θ                          ▲ push (obs, a, r, done, logp)
                     │                                  │
           ┌─────────┼─────────┐                ┌──────┴──────┐
           │         │         │                │             │
       ┌───▼──┐  ┌───▼──┐  ┌───▼──┐         ┌───┴──┐      ┌───┴──┐
       │ CPU  │  │ CPU  │  │ CPU  │   ...   │ CPU  │      │ CPU  │
       │ env  │  │ env  │  │ env  │         │ env  │      │ env  │
       │ +π   │  │ +π   │  │ +π   │         │ +π   │      │ +π   │
       └──────┘  └──────┘  └──────┘         └──────┘      └──────┘
```

Each CPU worker holds:
- A copy of the env (Atari, gym-MuJoCo, your own C++ sim, a network simulator, an LLM sandbox, …).
- A *small* policy network running on CPU (numpy / ONNX / a frozen jit), used **only for action selection**, pulled from the learner every few seconds.
- A local rollout buffer of length `T`. When full, push `(obs, action, reward, done, logp)` to the learner over IPC or network.

The learner:
- Receives trajectories, packs them into a GPU tensor.
- Does grad updates (PPO: minibatch SGD over the batch; SAC/V-trace: replay + off-policy correction).
- Periodically broadcasts updated params back.

**No MuJoCo required** — the env can be anything that exposes `reset()` / `step(action) → (obs, r, done, info)`. Atari via ALE-py, networking sim, board games, web browser via Playwright, an LLM rollout, etc.

### Communication patterns

| Pattern | Latency | Best for | Used by |
| --- | --- | --- | --- |
| **Shared memory** (single machine, multiprocessing) | µs | up to ~64 workers on one box | gym AsyncVectorEnv, EnvPool, Sample Factory |
| **gRPC / TCP queues** | ms | LAN cluster, untrusted boundary | RLlib, SEED RL, Acme |
| **Ray actors** | ms (Plasma store) | heterogeneous cluster, fault-tolerant | Ray RLlib, AlphaStar, DeepMind reverb-style |
| **Reverb** (DeepMind) | ms | priority replay + distributed | Acme, R2D2 reproductions |

### Important practical details

- **Stale-policy correction.** With *K* asynchronous actors the learner sees a mixture of policy versions. PPO clips, IMPALA uses V-trace, R2D2 uses recurrent replay with burn-in. Without one of these, training is unstable.
- **Backpressure.** If learner is slower than actors, the queue grows unbounded. Use bounded queues and either drop or block.
- **Determinism / debugging.** Seed each worker independently. Log `worker_id, policy_version, step` with every trajectory so divergence can be traced.
- **CPU sizing.** Atari is ~3k SPS per core; MJX-on-CPU ~50 SPS per core (slow — that's why we use mjx-on-GPU); a fast C++ sim can hit 50k SPS per core. Plan worker count from `GPU forward throughput × batch ÷ env_SPS_per_core`.
- **No-MuJoCo example envs that scale well on CPU**: ALE / Procgen / MiniGrid / NetHack / Crafter / Habitat (CPU mode) / classic gym control / your in-house simulator / LLM rollouts (CPU prompt building + GPU forward per token).

---

## 4. Where this would slot into `rad-se`

Current code: single-process, single-GPU. brax wraps an `mjx` env that vectorizes on the GPU. The control flow is `jax.lax.scan(env.step ∘ actor_step)` inside one jit. There is no actor/learner split today.

Two realistic next steps if we wanted distributed CPU actors:

1. **Drop-in: gymnasium `AsyncVectorEnv` for a non-GPU env.** Wire a new env adapter in `make_envs` that constructs `N` CPU procs around any classic gym env. Keep PPO's existing scan loop but call the vector env from host and shuttle obs/actions via `jax.device_put`. Loses the JIT'd env benefit but gains *any* env.

2. **Full IMPALA-style learner.** Split `rad_brax_ppo.py` into:
   - `rollout_worker.py` — runs N CPU procs, each holding a *frozen* policy state pulled via shared memory; writes trajectories to a `multiprocessing.Queue` or Reverb table.
   - `learner.py` — pulls batches from queue, runs the existing grad update, pushes new params back via shared memory.
   This is ~300 lines of glue once the env adapter exists. PPO's clip handles the version skew naturally.

For now, since our env is GPU-resident MJX and we have 1 GPU, the single-process design is correct and fastest. The doc above is the playbook for the day we want to scale to >1 GPU or add a non-MJX env.

---

## 5. How JAX + MuJoCo Playground change the game (why "10–50× speedup")

### 5.0 What kind of env can MJX actually simulate?

MJX (the JAX/XLA port of MuJoCo) is a **rigid-body + soft-constraint physics engine**, not a general game engine. It can simulate, on GPU, anything that fits MuJoCo's XML schema:

- **Articulated rigid bodies** with joints (hinge, slide, ball, free), tendons, equality constraints, contacts, friction, springs/dampers.
- **Soft constraints**: weld, connect, joint coupling.
- **Sensors**: touch, force, torque, accelerometers, gyros, rangefinders, RGB cameras (via warp rasterizer), depth, segmentation.
- **Domain randomization**: randomize mass, geometry, friction, lighting, textures across the `N` parallel envs cheaply.

What that means in practice — **what runs natively on MJX**:

| Class | Examples | Lives in MJX? |
|---|---|---|
| Locomotion | Humanoid, ant, half-cheetah, quadruped, biped, Cassie, Unitree, ANYmal | ✅ yes (Brax, MJX-locomotion suite) |
| Manipulation | UR5/Franka/Kuka arms, dexterous hand (Shadow, Adroit, Allegro), pick-and-place | ✅ yes (Robosuite-MJX, MJX-Manipulation) |
| Classic control | CartPole, Pendulum, Acrobot, Reacher, Hopper, Walker | ✅ yes (dm_control_suite port in Playground) |
| Drones / cars | Crazyflie, racecars, Ackermann | ✅ yes (community ports — MuJoCo Menagerie) |
| Soft bodies / cloth | Cloth simulation, deformables | ⚠️ partial (MuJoCo flex, slower on MJX) |
| Fluid sim | SPH, Eulerian fluids | ❌ no — not what MuJoCo does |
| Heat / EM / chemistry | — | ❌ no |
| **Games** ↓ | | |
| Atari (ALE) | Pong, Breakout, etc. | ❌ no — Atari is a CPU ROM emulator (game logic, not physics) |
| Procgen / MiniGrid | tile-based grid worlds | ❌ no — uses pygame / pixel-game logic |
| Crafter / NetHack | discrete world simulators | ❌ no |
| StarCraft / Dota / chess / Go | — | ❌ no |
| **MJX-friendly "games"** ↓ | | |
| Soccer (humanoids kicking a ball) | DeepMind MoCap Soccer, MJX-Soccer | ✅ yes — it's rigid bodies + contacts |
| Robotic mini-golf / billiards | | ✅ yes |
| Physics puzzles (Box2D-style but 3D) | block stacking, peg insertion | ✅ yes |
| Quadruped parkour, robot fighting | | ✅ yes |
| Tabletop physics games (Jenga, dominoes, marble runs) | | ✅ yes |

The dividing line is simple: **if the env can be expressed as "rigid bodies + joints + contacts + simple sensors", MJX runs it on GPU at 10⁵–10⁶ SPS.** If the env logic is "look up tile (x,y) in a tilemap" or "execute Atari instruction at PC", MJX has nothing to offer; you fall back to CPU game emulation and distributed actors (§1–3).

Some "games" sit in a grey zone:

- **OpenAI Gym MuJoCo robotic-game tasks** (Fetch-pick-and-place, hand-cube-reorient) → fully MJX.
- **DeepMind Control Soccer** (multi-agent humanoid soccer) → fully MJX.
- **Robot fighting / wrestling** → MJX, plus a learned game-logic reward function.
- **Anything that mixes physics with discrete game state** (e.g., "robot scores when ball enters bucket and clock < T") → fine: state is in `info` dict, MJX still does the heavy lifting on physics; game logic is just a jit-able JAX function on top.

So the answer to "is it only physics?" is **yes, but physics is enough for a huge slice of robotics and physics-grounded games**. For Atari / Procgen / MiniGrid / NetHack / web-browsing / LLM RL you need CPU actors. For "robot plays soccer / boxes / stacks blocks / drives an obstacle course" you stay on MJX.

### 5.1 The four wins (where "10–50×" comes from)

The headline number — "10×–50× faster than dm_control" — is real and comes from **four independent wins stacking multiplicatively**, not from one magic optimization.

| # | Win | Where the speedup comes from | Typical factor |
|---|---|---|---|
| 1 | **Vectorized physics on GPU** | One MJX `step` runs `N` env copies as one big batched kernel; SIMD across envs instead of `N` Python `mj_step` calls. | 5–30× |
| 2 | **No Python in the inner loop** | `jax.lax.scan` keeps `T × N` steps inside a single fused XLA program; zero PyObject allocation per step. | 3–10× |
| 3 | **Co-located env + policy** | Pixel obs / state never leaves device memory; the policy forward-pass and `env.step` share the same HBM. No PCIe round-trip. | 2–5× |
| 4 | **End-to-end JIT + autotune** | XLA fuses `step → encode → policy → loss` into a few big kernels; cuDNN autotune picks optimal convs once. | 1.5–3× |

Compounded against a CPU-MuJoCo + PyTorch baseline: **5 × 5 × 3 × 2 ≈ 150×** in the best case (Brax ant benchmark), **10–50×** in realistic vision-RL settings, and "only" **3–10×** if your bottleneck is something else (e.g., huge replay buffer copies). Our brax-PPO run at 215 SPS × num_envs=256 = **55 k env-steps/sec on one RTX 3060** — that's roughly what 50 CPU cores running dm_control would deliver.

### 5.2 What was *not* fast before

Pre-MJX pipeline (dm_control / mujoco_py + gym + PyTorch):
- One Python `env.step()` per env per step → GIL serialization → multiprocess workers needed → IPC overhead.
- Obs goes CPU → numpy → torch → GPU every step. For 100×100×3 pixels that's ~30 KB × `N` × `T` × 8 bytes round-trip.
- Policy net does small forward pass on GPU; GPU sits at ~5% util waiting for envs.
- Replay sampling: indexing into a CPU buffer, copying to GPU — another bandwidth hit.

MJX/Playground/Brax flips this: **the env is just another JAX function, the policy is another JAX function, and `lax.scan` sews them together inside one compiled program.** The GPU goes to >90% util.

### 5.3 GPU vs distributed-CPU: speed and cost on Vast.ai

Using current Vast.ai spot prices (May 2026 ballpark) for CartpoleSwingup-class vision RL:

| Setup | Hardware | Throughput | $/hr | env-steps per $ |
|---|---|---|---|---|
| **MJX/Brax on 1× RTX 3060 (12 GB)** | what we're running locally | ~55 k SPS (PPO, num_envs=256) | ~$0.10 (owned / amortized) | unbeatable for owned HW |
| **MJX/Brax on 1× RTX 4090 (24 GB)** | Vast spot | ~250 k SPS (num_envs=1024) | $0.35 | **~2.6 G steps / $** |
| **MJX/Brax on 1× H100 (80 GB)** | Vast spot | ~1.2 M SPS (num_envs=4096) | $1.80 | **~2.4 G steps / $** |
| dm_control on 1× CPU core (Atari-class env) | Vast CPU instance | ~150 SPS | $0.005 / core / hr | 108 M steps / $ — but you need many of them |
| dm_control on 64-core CPU box, multiproc | Vast 64-core | ~9 k SPS aggregate | $0.30 | 108 M steps / $ |
| dm_control on 512 cores across 8× 64-core boxes | Vast cluster | ~70 k SPS aggregate | $2.40 | 105 M steps / $ |

Two reality checks before reading those numbers:

1. **Distributed CPU only wins on env-steps-per-dollar for envs JAX/MJX can't run.** For MuJoCo physics, a $0.35/hr 4090 demolishes a $2.40/hr CPU cluster on both throughput (3.5×) and total cost (7×). The CPU-cluster numbers above assume an Atari-like cheap env; with real MuJoCo physics, per-core SPS drops to ~50 and the cluster becomes 100× worse than the 4090 in raw throughput.
2. **Per-step learner cost rises with cluster size.** Network IO between CPU actors and GPU learner becomes the bottleneck around 100 actors; SEED RL moves *inference* to the learner to fix this. So a 512-CPU cluster won't actually deliver 512× the single-CPU SPS.

**Decision rule we use:**
- Env runs on GPU (MJX/Brax/IsaacGym) → buy/rent one GPU, never bother with CPU actors.
- Env is C++ / Atari / NetHack / network sim / web browser → rent many cheap CPUs + one small GPU learner. The 4090 spend becomes wasted; you want a CPU-heavy box like Vast's 96-core EPYC at $0.40/hr plus one 3060 for the learner.
- Env is real hardware / human-in-loop / costly simulator → forget throughput, optimize sample efficiency (SAC + replay).

### 5.4 Where the 50× *doesn't* materialize

- **Replay-heavy pixel SAC** (what we're running). The on-device replay buffer (50k × 100×100×3 = 12 GB) makes the GPU memory the bottleneck and forces small `num_envs`. We get 227 SPS in SAC vs 55 k SPS in PPO on the *same* env — the speedup is in the env+forward, not the off-policy update. This is the OOM we hit on the 2080 Ti.
- **Large policy networks** (ViT, transformers). Forward pass dominates; env speed stops mattering.
- **CPU↔GPU transfer for any reason** (logging full frames, host-side replay, Python callbacks). Even one `device_get` per step kills the win.

---

## 6. Why world-model / RAD-family papers always use SAC — and can PPO replace it?

### 6.1 The actual reason: sample efficiency on pixel DMC

Every paper in the "pixel control" lineage — Dreamer, DreamerV2/V3, RAD, CURL, DrQ-v1/v2, MWM, TD-MPC, TD-MPC2, EfficientZero (in their continuous-action variants) — is benchmarked on **DMC-100k / DMC-500k**: max 100k or 500k env steps, *single env*, score reported at that budget. The benchmark was deliberately set up to reward sample efficiency, because in 2018–2020 the open question was "can pixels-only RL match state-based RL on DMC?" — and you couldn't answer that question if you let methods burn 100M steps.

SAC + replay wins this benchmark by ~5–20× over PPO at the same step budget. That's not a property of the algorithm being "better"; it's that PPO discards each trajectory after one update, so on a single env it takes 5–20× more env steps to match. **The benchmark forces SAC.**

### 6.2 Secondary reasons that compound

1. **Augmentation lives naturally in the replay sampler.** RAD/DrQ apply random crop / shift / colour jitter every time a transition is sampled — so the same transition can be augmented differently across thousands of gradient steps. With PPO you only see each transition for ~K gradient steps before it's gone; augmentation gives much less leverage.
2. **World models need replay anyway.** Dreamer-class methods train the world model on a replay buffer; once you have one, attaching SAC-style actor-critic on top is free. PPO would need a separate on-policy buffer for the actor and a replay for the WM — extra plumbing.
3. **Continuous actions, dense rewards.** DMC tasks have small continuous action spaces with smooth dense rewards — SAC's entropy regularization and Gaussian policy is a great fit. PPO works too, but doesn't get a free lunch from the structure.
4. **Critic stability with augmentation.** DrQ's key insight: averaging Q-targets over multiple augmentations gives you a low-variance critic target. That trick is specific to off-policy critic learning; PPO has no critic target to average over (it bootstraps from V, not from min-of-twin-Q).

### 6.3 Can PPO replace SAC with no performance loss?

**With a generous step budget: yes, mostly.** OpenAI's Rubik's cube hand, DeepMind's StarCraft work, and many sim-to-real robotics papers use PPO + pixel obs with no fundamental performance gap — once you give it 100M+ steps. The PPO papers that beat SAC at scale (e.g., on Atari with 200M frames, or Procgen with 25M steps × 64 envs) exist.

**At the DMC-100k budget: no, you'll lose 5–15× sample efficiency.** This is reproducible and well-documented (see DrQ-v2 Appendix and the original SAC-AE paper).

**Practical replacement recipes that close the gap:**
- **PPO + huge `num_envs` (≥256) + MJX on GPU.** Throw 100× more env steps at it; absolute wall-clock can match SAC because env is cheap. This is exactly what we're running locally — and the fact that PPO is *also* stuck at ER=11.7 tells us the bug is **not** the algorithm.
- **PPO + augmentation (RAD-style crop) + auxiliary recon loss** (à la PAD / SVEA). Closes ~half the gap.
- **PPO + frame-stack 3 + recurrent (LSTM/GRU) head.** Helps when the env is partially observable from pixels.
- **PPO with VTrace correction.** Lets you reuse trajectories ~3–5 epochs without instability. Effectively closes another chunk.

For the `rad-se` paper claim ("RAD pixel-translation transfers to brax+MJX"), the academically defensible choice is SAC — that's the apples-to-apples comparison with DrQ/RAD. PPO numbers would be considered a "scalability" claim, not an "algorithmic" one.

---

## 7. Why both current runs are stuck — and how to diagnose with env logging / video

> Local PPO 5M @ 1.5M steps: ER ≈ 11.7 (best 11.93 @ 716k), plateauing.
> Remote SAC 500k done: ER ≈ 74.6 (best 78.3 @ 300k), plateauing.

Expected CartpoleSwingup return with `--dmc-reward` (raw tolerance product summed over 1000 physics steps): **~700–900** when swung up and balanced. Our agents are clearly *not* swinging up. Two qualitatively different failure modes are possible:

### 7.1 Hypotheses

| # | Hypothesis | Why plausible | How to confirm |
|---|---|---|---|
| H1 | **Pixel obs uninformative** (camera angle / lighting / resolution makes pole indistinguishable from cart). | The vision-reward bug we already fixed (`_dense_vision_reward` was a penalty) shows this env's vision wrapper is fragile. | Dump the first eval-rollout frames as MP4 / PNG grid and look. |
| H2 | **Entropy collapse early.** SAC log shows `alpha=0.002` (tiny entropy bonus) → policy went deterministic before reward signal arrived. | SAC needs target entropy ≈ `-action_dim = -1` for cartpole; if `--reward-scaling 0.1` makes reward very small relative to log-prob, alpha is pushed to ~0. | Log `actor_entropy` and `alpha` per eval; check if entropy < 0 at step 50k. |
| H3 | **Action repeat × episode-length mismatch.** We pass `episode_length=1000` (physics steps) + `action_repeat=8`. If the env actually counts policy steps, episodes terminate after 8000 physics steps → reward sum saturates or wraps. | The plateau at exactly 70-something suggests a fixed-multiplier of something. 11.7 ≈ 1000 / 85, 75 ≈ 1000 / 13 — both look like aliasing. | Log `episode_length_steps_actual`, `dones_per_eval`. |
| H4 | **Augmentation kills gradient.** RAD random-translate with crop=84 from cam_res=100 leaves only 16 px slack; bad crops may hide the pole. | RAD paper used crop=84 from 100, but with a centered camera. Our brax cam may not be centered. | Disable `--augment-pixels` for a 50k smoke; if ER jumps, that's the cause. |
| H5 | **Reward not actually fixed.** The `_dense_reward` wrapper expects a specific kwarg signature; if `info`/`metrics` dict isn't passed through correctly, may return constant. | The wrapper passes `{}` for metrics; some envs read from `info` keys. | Log `mean reward per env-step` over first 1000 steps; should be in [0,1]. |

### 7.2 Concrete diagnostic patch (small, additive)

Add a `--diagnose` flag that does, on the first eval:

1. **Video dump.** Run 1 eval episode with `cam_res=128`, record all 125 policy-step frames, save as `runs/.../diag_eval_step0.mp4` using `mediapy.write_video`. brax `EpisodeWrapper` exposes the underlying mjx render via `env.render` — playground envs include this. ~30 lines.
2. **Reward / action / entropy timeseries.** During that eval, write `runs/.../diag_eval_step0.npz` with arrays `obs_pixels (T,H,W,3) uint8`, `actions (T,A)`, `rewards (T,)`, `dones (T,)`, `logp (T,)`. Already in scope; just `np.savez_compressed`.
3. **Per-eval scalars** added to `metrics.jsonl`:
   - `eval/episode_reward` (already there)
   - `eval/episode_length_actual` — sum of `1-done` until first `done`
   - `eval/mean_step_reward` — `episode_reward / episode_length_actual`
   - `eval/action_mean`, `eval/action_std` — sanity-check that the policy isn't outputting a constant
   - `eval/pixel_mean`, `eval/pixel_std` — sanity-check that the camera renders varying frames
   - (SAC only) `train/alpha`, `train/actor_entropy`, `train/critic_loss`, `train/q_mean`

Order of operations (cheapest first):

```bash
# 1. Single 10k-step PPO smoke WITHOUT augmentation, see if ER jumps.
PYTHONPATH=src python3 -u src/rad_se/rad_brax_ppo.py \
  --env CartpoleSwingup --seed 23 --total-timesteps 50000 \
  --num-envs 256 --no-augment-pixels --dmc-reward \
  --work-dir runs/_diag_ppo_noaug

# 2. Same but record a video of the first eval rollout (needs the patch).
PYTHONPATH=src python3 -u src/rad_se/rad_brax_ppo.py \
  --env CartpoleSwingup --seed 23 --total-timesteps 50000 \
  --num-envs 256 --dmc-reward --diagnose \
  --work-dir runs/_diag_ppo_video

# 3. Inspect.
mpv runs/_diag_ppo_video/diag_eval_step0.mp4         # is the cart visible? pole?
python3 -c "import numpy as np; d=np.load('runs/_diag_ppo_video/diag_eval_step0.npz'); print({k:(d[k].shape,d[k].mean(),d[k].std()) for k in d.files})"
```

If video shows a tiny pole on a giant background → camera/zoom is wrong; bump cam_res or tweak `vision_config.cam_pos`.
If pixel_std < 5 over the episode → the camera isn't moving with the cart and the encoder sees a static-ish image.
If action_std → 0 by step 30k while ER is still tiny → entropy collapse (H2); raise `--entropy-cost` for PPO or pin alpha for SAC.
If reward_per_step is ~0.01 throughout → the cart never reaches the upright bonus region; pole is in the wrong hemisphere of state space (could be H3, episode counting bug).

### 7.3 What I'd run next (in order)

1. **No-aug PPO smoke** (50k, ~4 min on the 3060). Free, no code change. Eliminates H4.
2. **Add `--diagnose` patch** (~80 lines in `rad_brax_ppo.py` + `rad_brax_sac.py`, mostly `mediapy` + `np.savez`).
3. **Run a 20k PPO with `--diagnose`** to get the video. If video shows the pole never crosses upright → it's a learning-rate / exploration issue; if pole crosses upright but reward is low → reward wrapper bug (H5).
4. **For SAC specifically:** rerun with `--no-reward-scaling` (or `--reward-scaling 1.0`) and `--target-entropy -1.0` pinned; this fixes H2 in one shot if that's the cause.

Want me to write the `--diagnose` patch? It's a small additive change that lands cleanly in both PPO and SAC scripts.

---

## 8. Reading list

- IMPALA (Espeholt et al., 2018) — the canonical async actor/learner paper.
- SEED RL (Espeholt et al., 2020) — fixes IMPALA's actor inference overhead by moving inference to a learner-side batched call.
- Sample Factory 2 (Petrenko et al., 2023) — high-throughput single-node CPU rollouts.
- Ray RLlib docs — production framework with all the above patterns implementable in ~50 lines of YAML.
- EnvPool (Weng et al., 2022) — C++ vectorized envs, drop-in replacement for gym Atari giving 10–20× speedup.
- DrQ-v2 / RAD / CURL — the SAC-side counterargument: when env data is precious, off-policy + augmentation beats throughput.
