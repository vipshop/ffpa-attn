# FP8 / FP4 Attention

FFPA's [CUDA backend](backends/cuda.md) can run the attention computation
in **FP8 (e4m3)** or **NVFP4 (e2m1)** on the forward pass: activations stay
fp16/bf16 in global memory, and Q/K/V are quantized inside the kernel
(smooth → scale → low-precision MMA), with exact scale folding so that the
only error source is the quantization rounding itself.

```python
from functools import partial
from ffpa_attn import CUDABackend, ffpa_attn_func

fp8_backend = CUDABackend(forward=True, backward=False, enable_fp8=True)
fp4_backend = CUDABackend(forward=True, backward=False, enable_fp4=True)

o_fp8 = ffpa_attn_func(q, k, v, forward_backend=fp8_backend)
o_fp4 = ffpa_attn_func(q, k, v, forward_backend=fp4_backend)
```

`enable_fp8` and `enable_fp4` are mutually exclusive, and FP8/FP4 attention
is **forward only**. Use the [Triton](backends/triton.md) or
[CuTe-DSL](backends/cutedsl.md) backend when you need backward.

## Support at a glance

| | FP8 | NVFP4 |
|---|---|---|
| GPUs | sm_120{a,f} (TMA family); sm_89 (Ada, cp.async family) | sm_120{a,f} only |
| Input dtypes | fp16 / bf16 (quantized in-kernel) | fp16 / bf16 |
| Headdim (default build) | `64 ~ 224` (persist-D) | `{64, 128, 192, 256}` (persist-D) |
| Headdim (full build) | `64 ~ 1024` | `64 ~ 512` |
| Any `D % 8 == 0` | ✔ via 32-aligned pad | ✔ via 64-aligned pad |
| causal / GQA / MQA / cross | ✔ | ✔ |
| `attn_mask` | ✔ | ✔ |
| dropout | ✖ | ✖ |
| NHD / strided-NHD zero-copy | ✔ | ✔ |
| hybrid (fp16 early rows) | ✔ | ✔ |
| hadamard (WHT Q/K rotation) | ✔ | ✔ |
| smooth-K / smooth-V | ✔ / ✔ | forced / optional |

> [!NOTE]
> **Build-time coverage.** `--ext all` compiles only the persist-D quantized
> windows (fp8 `D<=224`, fp4 `64<=D<=256`) — the split-D / M4N2 quantized
> families are opt-in via `ENABLE_FFPA_QUANT_SPLIT_D=1`. Calls past the
> compiled window raise a `rebuild with ENABLE_FFPA_QUANT_SPLIT_D=1` error
> instead of silently dispatching elsewhere.

## FP8 knobs

| Knob | Default | Meaning |
|---|---|---|
| `fp8_smooth_k` | `True` | Subtract the per-(b,h) K sequence mean before quantization (softmax shift-invariance; only `lse` is corrected). |
| `fp8_smooth_v` | `False` | Subtract the per-(b,h) V channel mean (`fp8_v_quant_method="per_channel"` required); the epilogue adds it back. |
| `fp8_q_quant_method` / `fp8_k_quant_method` | `"per_block"` | `"per_block"` (128-row tile) or `"per_thread"` (MMA-fragment aligned). Must match. |
| `fp8_v_quant_method` | `"per_channel"` | `"per_channel"` (SageAttention-style, recommended for outlier-heavy V) or `"per_block"`. |
| `fp8_pv_acc_type` | `"f32"` | PV accumulator: `"f32"` or `"f16"` (faster, minor precision loss; the sm_89 family is f16-only). |
| `fp8_qk_mm_type` | `"fp8"` | QK^T MMA: `"fp8"` (e4m3) or `"int8"` (s8xs8→s32). int8 tends to win at large `N`. |
| `fp8_hadamard` | `False` | Rotate Q/K by an orthogonal Walsh-Hadamard matrix before quantization (outlier spreading; exact in fp32 math). |
| `fp8_hybrid` | auto | `None` = auto-on for `is_causal=True`: leading `fp8_hybrid_n_early` rows (multiple of 128, default 256) are computed in fp16, the rest in fp8. |

## NVFP4 specifics

- NVFP4 = e2m1 data + ue4m3 scale factors per 16-element microscaling block;
  the blockscale MMA consumes the scales directly.
- **Q/K smoothing is mandatory** (not optional) — e2m1's ±6 dynamic range
  requires the mean-removal + `delta_s` rank-1 correction to stay accurate.
- `fp4_pv_mm_type="fp8"` selects the MXFP8 PV variant (QK stays NVFP4);
  MXFP8 covers persist-D `D<=192` and split-D, and is rejected on the M4N2
  family (its atom K=128 exceeds the M4N2 tile's kBc=64).
- `fp4_smooth_v` mirrors `fp8_smooth_v` for the V side.
- `fp4_hadamard` rotates Q/K (pow2 D rotates fused inside the quantize
  kernel, other D falls back to a standalone WHT kernel).

## Precision notes

Low-precision attention trades a bounded amount of accuracy for throughput.
Two effects dominate and are worth knowing before you tune:

- **Causal early rows.** The first rows of a causal sequence attend to one
  or two keys, so their outputs have effective sample size ≈ 1 and roughly
  3σ amplitude. A ~5% (e4m3) relative error therefore shows up as a large
  *absolute* error on those rows — the relative error itself is the same as
  in dense rows. Use the [hybrid](#hybrid-early-rows-in-fp16) mode (default
  auto for causal) or per-channel V scales to protect them.
- **P is online-quantized.** Q/K/V can be calibrated offline (smoothing,
  per-block/per-channel scales), but the softmax probabilities `P` are
  produced inside the kernel and can only use fixed/conservative scales.
  FP8 uses a fixed `1/448` full-scale emission; NVFP4 uses the two-level
  `1/(448·6)` scheme from SageAttention-3.

For FP4 the error structure is one order of magnitude larger (e2m1 has a
~12.5% relative step): on `randn(σ=1)` inputs, causal spike rows show
`max_abs ≈ 0.5-0.75` while the `mean_abs` stays around `0.014-0.03`.
Measure **cosine similarity / mean_abs** (the SageAttention-3 evaluation
convention) rather than peak `max_abs` when judging FP4 quality.

### Hybrid: early rows in fp16

```python
backend = CUDABackend(
  forward=True, backward=False,
  enable_fp4=True,
  fp4_hybrid=True,          # None = auto (causal + enable_fp4)
  fp4_hybrid_n_early=256,   # multiple of 128
)
```

Hybrid runs the first `n_early` query rows through the fp16 kernel and the
remaining rows through fp8/fp4, with a `q_start_row` offset — zero
redundancy, mathematically row-exact, and compatible with all layouts. It
is the recommended setting for causal workloads.

## Benchmarking against SageAttention

Install [SageAttention-2/3](https://github.com/thu-ml/SageAttention.git)
first, then build FFPA with the quantized kernels
(`bash ./build.sh --arch sm_120f --ext all --headdim 128,192,256`) and run:

```bash
python bench/bench_fp8.py --N 16384,32768 # FFPA-FP8 vs SageAttention-2 vs SDPA
python bench/bench_fp4.py --N 16384,32768 # FFPA-FP4 vs SageAttention-3 vs SDPA
```

`bench_fp8.py --preset` ships configurations matching common deployments:

| Preset | Configuration |
|---|---|
| `default` | int8 QK + f16 PV acc, per_thread Q/K + per_channel V + smooth-K |
| `cache-dit` | same as `default` with hybrid `n_early=256` — matches cache-dit's `ffpa_fp8` backend |
| `perblock` | int8 QK + f16 PV acc, per_block Q/K/V + smooth-K (the sm_89 cp.async scope) |

FFPA FP8 is comparable to or slightly faster than SageAttention-2 at D=128,
and FFPA FP4 is significantly faster than SageAttention-3 at D=128 on
NVIDIA RTX PRO 5000/6000/5090. Reproduce the plots with the commands in
[Bench](../bench/README.md#fp8-and-fp4).

## Debug / A-B switches

These exist only in debug builds (`--debug-ext fp8` / `fp4`), see
[Env Vars](../env.md):

| Env | Effect |
|---|---|
| `FFPA_FP8_FORCE_KERNEL=split_d\|m4n2` | Force one fp8 split-D kernel family for A/B (`224 < D <= 1024`). |
| `FFPA_FP8_PQUANT_PER_ROW=1` | Per-row P quantization (full-scale; incompatible with lazy rescale). |
| `FFPA_FP4_PAD_TORCH=1` | Materialize the fp4 head-dim pad with torch instead of the fused path. |
| `FFPA_BIAS_TILE_KEEP=1` | Restore the fp4 M4N2 bias tile modes (pinned to the gmem-direct mode by default). |

## End-to-end in Cache-DiT

FP8/FP4 attention is integrated into [Cache-DiT](https://github.com/vipshop/cache-dit)
(`--attn ffpa_fp8` / `ffpa_fp4`), covering self / cross / causal / attn-mask
/ GQA-MQA with zero-copy NHD layouts:

```bash
python3 -m cache_dit.generate flux --attn ffpa_fp8 --seed 42 --height 1024 --width 1024
python3 -m cache_dit.generate flux --attn ffpa_fp4 --seed 42 --height 1024 --width 1024
```

FP8/FP4 attention is **not** a universal win (small models and short
seqlens spend more time in the quantization pre-pass than they save in the
MMA); evaluate precision and speed on your own workload.
