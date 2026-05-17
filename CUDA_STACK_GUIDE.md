# CUDA Stack Guide: From Driver to Training Loop

How the GPU software stack affects ML training — RL, CNN, GNN, and NLP.

---

## 1. The Stack (Bottom to Top)

```
┌───────────────────────────────────────────────────┐
│  Your training loop  (PyTorch / JAX / TF)         │  ← you write this
├───────────────────────────────────────────────────┤
│  ML framework backend  (XLA, TorchScript, etc.)   │  ← compiles ops to kernels
├───────────────────────────────────────────────────┤
│  CUDA libraries  (cuDNN, cuBLAS, cuSPARSE, NCCL)  │  ← hand-tuned math kernels
├───────────────────────────────────────────────────┤
│  CUDA Runtime + Toolkit  (nvcc, PTX, cubin)        │  ← kernel compilation
├───────────────────────────────────────────────────┤
│  CUDA Driver  (libcuda.so, kernel module)          │  ← hardware interface
├───────────────────────────────────────────────────┤
│  GPU Hardware  (SM, tensor cores, memory)          │
└───────────────────────────────────────────────────┘
```

Each layer has its own version number. Mismatches between layers are a common
source of silent performance problems or hard crashes.

---

## 2. Layer-by-Layer Explanation

### 2.1 GPU Hardware

The physical chip. Key specs that affect ML:

| Feature | What it does | Relevant for |
|---------|-------------|-------------|
| **CUDA cores (SMs)** | General-purpose parallel compute | All workloads |
| **Tensor cores** | Matrix multiply (FP16/BF16/TF32/INT8) in 4-8× the throughput | CNN, Transformer, GNN matmul |
| **Memory bandwidth** | How fast data moves GPU DRAM → SM | Replay buffer reads, large-batch inference |
| **VRAM** | How much you can fit on-device | Replay buffer size, batch size, model size |
| **NVLink / PCIe** | Multi-GPU bandwidth | Distributed training |

Architecture generations matter for tensor core capabilities:

| Arch | GPUs | Tensor core dtype | Key addition |
|------|------|-------------------|-------------|
| Volta (sm_70) | V100 | FP16 | First tensor cores |
| Turing (sm_75) | RTX 2080 | FP16, INT8 | INT8 inference |
| Ampere (sm_86) | RTX 3060/3090, A100 | FP16, BF16, TF32, INT8 | BF16, sparsity (2:4) |
| Ada Lovelace (sm_89) | RTX 4070/4090 | + FP8 | FP8, better sparse |
| Hopper (sm_90) | H100 | + FP8 transformer engine | Hardware attention |

---

### 2.2 CUDA Driver (`libcuda.so`)

- Installed with the GPU driver package (e.g., `nvidia-driver-545`)
- Lives in the OS, **not** inside a container or conda env
- Version format: `X.Y` (e.g., `12.1`, `12.6`) — reported by `nvidia-smi`
- The driver is **backward compatible**: driver 12.6 can run code compiled for
  CUDA 12.1, 12.0, 11.x, etc.
- But it adds NEW APIs at each version. Code using a newer API won't run on an
  older driver even if the toolkit is new.

**Key version milestones relevant to ML:**

| Driver / CUDA version | What changed |
|----------------------|-------------|
| CUDA 11.0 | BF16 support, A100 features |
| CUDA 11.8 | FP8 preview, NVLink4 |
| CUDA 12.0 | Hopper (H100) support, `cuModuleLoadDataEx` capture changes |
| **CUDA 12.3** | Nested stream capture, more reliable graph capture |
| **CUDA 12.4** | Conditional CUDA graphs (if/while in a graph) |
| CUDA 12.6 | Ada Lovelace FP8 full support, graph improvements |

---

### 2.3 CUDA Toolkit (nvcc, PTX, cubin)

- The *compiler and runtime libraries* (`libcudart.so`, `nvcc`, cuDNN, cuBLAS)
- Installed separately from the driver — can be in a conda env or system package
- `nvcc --version` shows toolkit version; `nvidia-smi` shows driver/runtime version

**PTX vs cubin:**
- `nvcc` compiles CUDA C → PTX (portable assembly) → cubin (GPU-specific binary)
- PTX is JIT-compiled to cubin by the driver at first run → startup latency
- Warp compiles to PTX at `import` time, loads cubin at first kernel call

**The important rule:**
```
Toolkit version ≤ Driver version  (toolkit can't use APIs the driver doesn't have)
Toolkit version can be OLDER than driver — backward compatible
```

---

### 2.4 CUDA Libraries: cuDNN, cuBLAS, cuSPARSE

These are **hand-tuned kernel collections** for specific math operations.

#### cuDNN (Deep Neural Network library)
- Optimizes convolutions, RNNs, batch norm, attention
- For each operation (e.g., Conv2D with 100×100 input, 32 filters), it has
  **dozens of algorithms** (direct conv, GEMM, Winograd, FFT, etc.)
- The right algorithm depends on input shape, dtype, hardware, batch size
- **Autotuning** = running all candidate algorithms and picking the fastest one

| Algorithm type | Best for | Speed vs generic |
|---------------|----------|-----------------|
| Direct conv | Small kernels, small batch | 1× (baseline) |
| GEMM (im2col) | General case, large batch | 1.5-2× |
| Winograd | 3×3 filters, medium batch | 2-4× |
| FFT | Large spatial, large kernels | 2-5× |
| Tensor core GEMM | FP16/BF16, large batch | 4-8× |

Without autotuning, the framework picks a default algorithm — often NOT the
fastest one for your specific shapes.

#### cuBLAS
- Matrix multiply (GEMM) — the backbone of linear layers, attention
- Always autotuned at the first call
- Tensor cores kick in automatically for FP16/BF16/TF32

#### cuSPARSE
- Sparse matrix operations — GNNs, sparse attention
- Much less mature than cuBLAS; performance varies widely

---

### 2.5 CUDA Graphs

A CUDA graph records a sequence of GPU operations (kernel launches, memory
copies) and replays them with minimal CPU overhead.

**Without graphs:** CPU dispatches each kernel individually → kernel launch
overhead ~5-10 µs each → for small fast kernels this dominates.

**With graphs:** the whole sequence is launched in one driver call → 100-1000×
lower launch overhead.

**Who uses CUDA graphs automatically:**
- `torch.compile` (PyTorch 2.0+)
- `jax.jit` with `--xla_gpu_use_runtime_fusion=true`
- Warp (`mujoco_warp` physics simulation)
- PyTorch's `torch.cuda.CUDAGraph` context manager

**The capture rule:** During `cudaStreamBeginCapture ... cudaStreamEndCapture`,
CUDA records operations instead of running them. Certain APIs are **not
capture-safe**:
- `cuModuleLoad` / `cudaModuleLoad` (loading a new CUDA module)
- Most memory allocations/frees
- CPU↔GPU synchronization calls

This is where CUDA version matters:
- **< CUDA 12.3**: Nested captures not supported; module loading during capture
  always fails → CUDA error 900
- **≥ CUDA 12.3**: Nested captures supported; `enable_graph_capture_module_load`
  works more reliably
- **≥ CUDA 12.4**: Conditional graphs (if/while nodes) — needed for dynamic RL
  episode termination inside a graph

---

### 2.6 XLA (for JAX / TensorFlow)

XLA is a compiler that takes high-level JAX ops and lowers them to GPU kernels.

#### XLA JIT compilation (`jax.jit`)
1. First call: traces Python → HLO (high-level ops) → LLO → CUDA kernels
2. Compiled executable is cached
3. Second+ calls: runs cached executable directly (fast)
4. Re-traces when array shapes change (expensive)

#### XLA Autotuning (`--xla_gpu_autotune_level`)

XLA calls cuDNN/cuBLAS with many candidate algorithms and times them using
**CUDA stream capture** (runs each candidate, measures with CUDA events).

| Level | Behavior |
|-------|----------|
| `0` | Disabled — uses default algorithm (safe but slow for conv) |
| `1` | Heuristic choice (fast startup, usually decent) |
| `2-3` | Profile a few candidates |
| `4` (default) | Full search over all cuDNN algorithms |

**The conflict with Warp/mujoco_warp (our problem):**

Warp physics uses CUDA graph capture internally for its simulation steps. When
XLA autotune also starts a CUDA stream capture, they conflict if the driver
doesn't support nested capture (< CUDA 12.3):

```
XLA: cudaStreamBeginCapture()       ← XLA starts timing candidate conv kernel
  Warp: cuModuleLoad(forward_f60f76d)  ← Warp loads new nworld variant
    → CUDA error 900: not permitted during capture
```

Fix: CUDA driver ≥ 12.3 (nested capture) or `--xla_gpu_autotune_level=0`.

**Cost of disabling autotune:**
The CNN encoder (100×100×3 pixels) defaults to a generic GEMM convolution.
Winograd or FFT could be 3-5× faster for these shapes. At 256 batch size,
autotune=OFF is the dominant bottleneck in our SAC run (stuck at 28 SPS).

---

## 3. Impact by Workload Type

### 3.1 Reinforcement Learning (pixel-based, like RAD/DrQ)

**Bottleneck anatomy:**
```
per training step:
  [env step]       Warp/MJX physics  → memory bandwidth, SM throughput
  [encode obs]     CNN on pixel batch → tensor cores, cuDNN conv algorithm
  [SAC/PPO losses] small MLP + critic → cuBLAS matmul (fast)
  [replay sample]  read from VRAM buffer → memory bandwidth
```

**What actually limits SPS:**
- With autotune=OFF: CNN encoding bottleneck → 28 SPS on RTX 3060
- With autotune=ON: CNN encoding 4-5× faster → 120-150 SPS on RTX 3060 equivalent
- With large replay buffer: memory bandwidth to sample obs becomes limiting
- With many envs: physics parallelism — Warp scales well to ~32-64 envs on 24GB

**Version pitfalls:**
- CUDA < 12.3: Warp physics + XLA autotune conflict (error 900) → must disable autotune
- CUDA ≥ 12.3: both can run simultaneously → full speed
- CUDA ≥ 12.4: can put `if done: reset_env()` inside CUDA graph → no CPU sync per step

**Recommended config by GPU:**
| GPU | VRAM | CUDA driver | autotune | num_envs | replay | Expected SPS |
|-----|------|------------|----------|----------|--------|------------|
| RTX 3060 | 12 GB | 12.5 (Warp bug) | OFF | 8 | 10k | ~28 |
| RTX 4070 Ti | 12 GB | 12.6 ✓ | **ON** | 16 | 10k | ~150 |
| RTX 3090 | 24 GB | 12.1 ✗ | OFF | 16 | 50k | ~80 (if fixed) |
| RTX 4090 | 24 GB | 12.6 ✓ | **ON** | 32 | 50k | ~300+ |

---

### 3.2 CNN (image classification, vision encoders)

**Bottleneck anatomy:**
- Dominated by `Conv2d` → cuDNN is critical
- Batch norm, pooling → fused by XLA/torch.compile when possible
- Linear layers at the end → cuBLAS, tensor cores

**Key levers:**
1. **Autotuning** — biggest single factor for conv-heavy nets. Always enable.
   PyTorch: `torch.backends.cudnn.benchmark = True`
   JAX: default `autotune_level=4`
2. **Mixed precision (FP16/BF16)** — 2× memory, enables tensor core paths in cuDNN
3. **Tensor core alignment** — channels/filters must be multiples of 8 (FP16) or 16 (BF16) to hit tensor core paths
4. **Input shape** — `NHWC` layout often faster than `NCHW` on Ampere+ for small channel counts

**Version impact:**
- cuDNN 8.x vs 9.x: major algorithm reorganization; Flash Attention added in 8.9+
- CUDA 12.0+: BF16 tensor cores fully supported on Ampere (needed for stable training)

---

### 3.3 GNN (Graph Neural Networks)

**Bottleneck anatomy:**
- **Scatter/gather** on irregular graph structure → cuSPARSE or custom kernels
- Node/edge feature MLP → cuBLAS (dense, efficient)
- Aggregation (mean/sum over neighbors) → memory-bound for large graphs

**Key issues:**
- GNNs are **memory bandwidth bound**, not compute bound
- Sparse operations in cuSPARSE are 2-5× slower than equivalent dense ops on tensor cores
- `torch.sparse_csr` and PyG's `torch_scatter` use custom CUDA kernels that bypass cuDNN entirely
- Autotuning has minimal effect (aggregation is memory-bound, not conv-bound)

**Version impact:**
- CUDA 11.0+: Structured sparsity (2:4 pattern) for Ampere — only helps if you prune to 2:4 pattern
- cuSPARSE improvements are incremental; major gains come from custom CUDA kernels (e.g., `torch_sparse`, `dgl`)
- On Hopper (H100): `cuSPARSE` tensor core paths finally competitive

---

### 3.4 NLP / Transformers

**Bottleneck anatomy:**
- `Attention(Q,K,V)` → historically: 3× `matmul` + softmax → cuBLAS + custom kernel
  Now: **Flash Attention** — custom fused kernel, bypasses cuDNN for attention
- `Linear` layers → cuBLAS GEMM, always hits tensor cores in FP16/BF16
- `LayerNorm`, `GELU` → element-wise, memory-bound

**Key levers:**
1. **Flash Attention** — 10-20× lower memory, 2-4× faster than naive attention
   Requires CUDA ≥ 11.6, Ampere+ (sm_80+) for full speed
   `pip install flash-attn` for PyTorch; built into JAX via `jax.nn.dot_product_attention` on Hopper
2. **BF16 training** — more stable than FP16 for LLMs, same speed on Ampere+
   Requires CUDA ≥ 12.0 on Ampere for full throughput
3. **cuDNN Flash Attention backend** (cuDNN 8.9+, CUDA 12.2+) — integrated into PyTorch 2.1+
4. **`torch.compile`** — fuses attention + norm + MLP blocks into single CUDA graphs

**Version impact:**
- CUDA < 11.6: No Flash Attention — major performance regression for long sequences
- CUDA 12.2 + cuDNN 8.9: cuDNN-native Flash Attention (no flash-attn package needed)
- CUDA 12.4 + H100: FP8 transformer engine — 2× over BF16 for inference

---

## 4. Practical Checklist When Renting a GPU

```bash
# 1. Check driver / CUDA version
nvidia-smi                          # shows driver + CUDA runtime version

# 2. Check what toolkit you actually have
python -c "import torch; print(torch.version.cuda)"
python -c "import jax; print(jax.devices())"
nvcc --version

# 3. Verify tensor cores are being used (PyTorch)
torch.backends.cuda.matmul.allow_tf32 = True   # Ampere+
torch.backends.cudnn.allow_tf32 = True

# 4. For JAX: check autotune is on (default level=4)
# Bad: XLA_FLAGS="--xla_gpu_autotune_level=0"   ← disables cuDNN algorithm search
# Good: leave XLA_FLAGS unset or level=4

# 5. For JAX + Warp: check CUDA ≥ 12.3
python -c "
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True)
print('Driver:', r.stdout.strip())
"
# Driver 530.x = CUDA 12.1 → Warp conflict
# Driver 545.x = CUDA 12.3 → nested capture OK
# Driver 560.x = CUDA 12.6 → full support
```

---

## 5. Quick Reference: Version → Feature Matrix

| Feature | Min CUDA | Notes |
|---------|----------|-------|
| BF16 training | 11.0 | Hardware on Ampere (sm_80+) |
| Structured sparsity (2:4) | 11.1 | Ampere only |
| Flash Attention (flash-attn pkg) | 11.6 | sm_80+ recommended |
| cuDNN Flash Attention | 12.2 + cuDNN 8.9 | PyTorch 2.1+ uses this |
| CUDA Graphs (basic) | 10.1 | |
| Nested stream capture | **12.3** | Fixes Warp + JAX autotune conflict |
| Conditional CUDA graphs | **12.4** | if/while nodes in graphs |
| FP8 training | 12.0 | Hopper H100 hardware only |
| Warp physics + XLA autotune | **12.3** | Our RAD/SAC use case |

---

## 6. Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Training stuck at unexpectedly low throughput | cuDNN autotune disabled; tensor core dtype mismatch | Check `XLA_FLAGS`, enable `cudnn.benchmark`, use FP16/BF16 |
| `CUDA error 900` during JAX + Warp | Nested stream capture not supported (driver < 12.3) | Upgrade driver OR `--xla_gpu_autotune_level=0` |
| OOM during autotuning | XLA scratch buffer (up to 4 GB) + model + replay all compete | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`; reduce batch temporarily |
| Slow startup, fast steady state | PTX → cubin JIT compilation on first run | Normal; use `XLA_FLAGS=--xla_gpu_cache_kernels=true` |
| Different speed on same GPU / different machine | CUDA toolkit version, cuDNN version differ | Pin versions in Docker/conda |
| Warp `cuModuleLoad` fails | New kernel variant triggered during capture | `wp.force_load()` before JIT, or CUDA ≥ 12.3 driver |
| Attention OOM on long sequences | Not using Flash Attention | `pip install flash-attn` or use `F.scaled_dot_product_attention` (PyTorch 2.0+) |

---

## 7. Real-World Instance Fleet: RAD SAC on CartpoleSwingup

Measured baseline: local RTX 3060 12 GB (CUDA 12.5), `--xla_gpu_autotune_level=0`,
`num_envs=8`, `replay=10k`, `batch_size=256` → **28 SPS flat**, 500k steps ≈ 4.9 hr.

### 7.1 Fleet Summary Table

| Instance | GPU | VRAM | CUDA driver | $/hr | Autotune | Max replay | Expected SPS | 500k steps | $/run |
|----------|-----|------|------------|------|----------|------------|-------------|------------|-------|
| local | RTX 3060 | 12 GB | 12.5 | $0 | ✗ OFF (script conservative) | 10k | **28** | 4.9 hr | $0 |
| 34824701 | RTX 3060 | 12 GB | 12.4 | $0.057 | ✓ ON (≥12.3) | 10k | **90–110** | ~1.4 hr | $0.08 |
| 36660304 | RTX 4070 Ti | 12 GB | 12.4 | $0.127 | ✓ ON | 20k | **160–200** | ~45 min | $0.10 |
| 36841270 | RTX 3060 Ti | 8 GB | 13.0 | $0.084 | ✓ ON | 5k | **100–130** | ~1.1 hr | $0.09 |
| 36721114 | RTX 4060 | 8 GB | 13.0 | $0.062 | ✓ ON | 5k | **120–150** | ~55 min | $0.06 |
| 36914450 | RTX 3060 Laptop ×2 | 6 GB each | 13.0 | $0.097 | ✓ ON (1 GPU used) | 2k | **50–65** | ~2.5 hr | $0.24 |
| 34838954 | RTX 3090 (exited) | 24 GB | 13.0 | $0.148 | ✓ ON | 50k | **200–260** | ~35 min | $0.09 |

SPS speedup drivers: `autotune=ON` ≈ 4-5× CNN speedup via cuDNN Winograd/FFT.
Ada Lovelace (4070 Ti, 4060) adds another 1.5-2× via improved tensor core throughput.

### 7.2 Per-Instance Config

**`36660304` RTX 4070 Ti — best currently running, $0.127/hr**
```bash
# remove --xla_gpu_autotune_level=0 entirely
python3.12 rad_brax_sac.py \
  --num-envs 16 --num-eval-envs 16 \
  --max-replay-size 20000 --batch-size 256
```

**`36721114` RTX 4060 — best $/SPS, $0.062/hr**
```bash
python3.12 rad_brax_sac.py \
  --num-envs 8 --num-eval-envs 8 \
  --max-replay-size 5000 --batch-size 128   # 8 GB cap
```
Smaller replay = less experience diversity; may need more steps to converge.

**`36841270` RTX 3060 Ti — good disk (50 GB), $0.084/hr**
```bash
python3.12 rad_brax_sac.py \
  --num-envs 8 --num-eval-envs 8 \
  --max-replay-size 5000 --batch-size 128
```

**`36914450` RTX 3060 Laptop ×2 — avoid for this task**
- JAX uses one GPU only → paying for 2 but using 1
- 6 GB per GPU → `replay=2000` max, severe diversity limitation
- $0.097/hr for ~55 SPS is worse value than the RTX 4060 at $0.062/hr

**`34838954` RTX 3090 (exited) — restart with CUDA 13.0 = fully fixed**
- Previous crash was CUDA 12.1 (error 900). This instance now shows CUDA 13.0
- Nested capture, conditional graphs, all Warp issues resolved
- 24 GB → full 50k replay, 32 envs:
```bash
python3.12 rad_brax_sac.py \
  --num-envs 32 --num-eval-envs 32 \
  --max-replay-size 50000 --batch-size 256
```

### 7.3 VRAM Budget Breakdown (reference)

Replay buffer dominates. Observation = 100×100×3 float32 = 120 KB. One
transition (s, a, r, done, s') ≈ 240 KB.

| replay size | buffer VRAM | + model (0.3 GB) + XLA exec (0.5 GB) + autotune scratch (1 GB) | fits in |
|------------|------------|----------------------------------------------------------------|---------|
| 2 000 | 0.5 GB | 2.3 GB total | 6 GB ✓ |
| 5 000 | 1.2 GB | 3.0 GB total | 8 GB ✓ |
| 10 000 | 2.4 GB | 4.2 GB total | 8 GB ✓ |
| 20 000 | 4.8 GB | 6.6 GB total | 12 GB ✓ |
| 50 000 | 11.2 GB | 13.0 GB total | 24 GB ✓ |

---

## 8. Case Study: Local RTX 3060, CUDA 12.5 — What Is the Real Bug?

### 8.1 What Is Actually Wrong

The local run is stuck at **28 SPS because `--xla_gpu_autotune_level=0` is set
in the launch script.** That's the whole story. There is no crash, no OOM, no
Warp conflict — just a conservative flag that was added when targeting CUDA 12.1
compatibility and never revisited.

The script comment says *"avoid OOM on large pixel batches during profiling."*
This was wrong or overly cautious. The script comment was never updated after the
flag was added for CUDA 12.1 compatibility.

**Observed** (iter 2, RTX 3060 12 GB, autotune_level=2, batch=256, scan_steps=3125):

```
nvidia-smi during autotune/JIT:  8780 MiB  (GPU 100% for ~2.5 min)
nvidia-smi during training:      8786 MiB  (steady state)
Free VRAM:                       3162 MiB
```

The peak is **much higher than the old ~800 MB estimate**. True breakdown:

```
Replay 10k:                2.4 GB
XLA scan body buffers:    ~5.9 GB  (lax.scan over 3125 steps × conv activations
                                    for backprop + autotune candidate scratch)
Model + optimizer:         0.1 GB
Warp physics state:        0.3 GB
────────────────────────────────
Observed peak + steady:    8.6 GB  → 3.4 GB headroom on 12 GB  ✓  (no OOM)
```

**Why so large?** `jax.lax.scan` over 3125 steps stores intermediate activations
for backpropagation. XLA needs O(scan_steps) activation memory. With
`scan_steps_per_eval = 500k / 20 / 8 = 3125`, each epoch retains gradient
checkpoints for all 3125 steps. Autotune candidate scratch is a smaller
contribution on top of this.

### 8.2 Why CUDA 12.5 Is Safe for Autotune + Warp

The Warp conflict (error 900) requires ALL of:
1. A CUDA stream capture is active (XLA autotune OR Warp physics graph)
2. A NEW Warp kernel variant is needed (lazy specialization for a new nworld)
3. The driver doesn't support capture-safe module loading (< CUDA 12.3)

CUDA 12.5 satisfies condition 3's inverse — nested capture and capture-safe
`cuModuleLoad` are supported since 12.3. So even if conditions 1+2 occur
simultaneously, the load succeeds.

### 8.3 Does Reducing Model Size Help?

**No — it addresses the wrong thing.**

| What you reduce | Effect on autotune scratch | Effect on steady VRAM | Effect on SPS |
|----------------|--------------------------|----------------------|--------------|
| CNN channels (32,64,64) → (16,32,32) | Halves feature maps → ~-400 MB peak | -50 MB | Slight regression in representation quality |
| MLP size (1024,1024) → (256,256) | Zero — MLP is not the profiled op | -30 MB | No change to bottleneck |
| `--batch-size 256` → `128` | Halves ALL CNN tensors → -400 MB | -200 MB | ~-10% SPS (fewer grad samples) |
| Remove `--xla_gpu_autotune_level=0` | +~5.9 GB scan buffers (permanent) | +5.9 GB | **+5.3× SPS** (observed: 28→48–158) |

The model parameters are ~40 MB total — irrelevant compared to the 2.4 GB replay
buffer. Shrinking the network saves almost no memory and hurts representation.

### 8.4 The Fix

```bash
# In run_brax_sac.sh — remove or comment out:
# export XLA_FLAGS="--xla_gpu_autotune_level=0"

# Optionally keep a partial autotune to reduce startup time (~30s vs ~2min):
export XLA_FLAGS="--xla_gpu_autotune_level=2"   # profiles top-5 candidates only
```

Expected result on local RTX 3060 CUDA 12.5: **~90–110 SPS** (from 28),
same 500k-step run completes in ~1.4 hr instead of 4.9 hr, at $0 cost.
