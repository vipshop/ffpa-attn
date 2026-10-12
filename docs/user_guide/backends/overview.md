# Backends Overview

FFPA ships four interchangeable attention backends behind a single entry
point, [`ffpa_attn_func`](../../api/ffpa_attn.md):

| Backend | Config class | Forward | Backward | Where it runs |
|---|---|---|---|---|
| SDPA | <span style="white-space:nowrap">`SDPABackend`</span> | ✔ (PyTorch native) | ✔ | all devices |
| CUDA | [<span style="white-space:nowrap">`CUDABackend`</span>](cuda.md) | ✔ | ✖ | fp16/bf16 + FP8/FP4 C++ kernels |
| Triton | [<span style="white-space:nowrap">`TritonBackend`</span>](triton.md) | ✔ | ✔ | NVIDIA sm_80+ and AMD ROCm |
| CuTe-DSL | [<span style="white-space:nowrap">`CuTeDSLBackend`</span>](cutedsl.md) | ✔ | ✔ | sm_80 (generic), sm_90a, sm_100a |

## Choosing a backend

```python
from ffpa_attn import ffpa_attn_func, CUDABackend, TritonBackend, CuTeDSLBackend

# FlashAttention-style training shape (D=512, needs backward): Triton default.
o = ffpa_attn_func(q, k, v)

# Large-D SM90/SM100 dense training: CuTe-DSL (D=512), see cutedsl.md.
o = ffpa_attn_func(q, k, v, backend=CuTeDSLBackend())

# Forward-only inference with the C++ CUDA kernels (fp16/bf16 or FP8/FP4).
o = ffpa_attn_func(q, k, v, forward_backend=CUDABackend(enable_fp8=True))
```

The `backend=` shorthand fills both `forward_backend` and
`backward_backend`; explicit `forward_backend` / `backward_backend`
kwargs take priority. Forward and backward may use different backends,
e.g. the CUDA forward kernels with a Triton backward.

## Capability matrix

<div align='center' markdown="1">

| Backend | Arch | Fwd | Bwd | Dtype | Headdim | Causal | GQA / MQA | attn_mask | Dropout | Varlen (THD) | NHD zero-copy | Autotune |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SDPA | sm>=75 | ✔ | ✔ | fp16/bf16/fp32 | all | ✔ | ✔ | ✔ | ✔ | ✖ | ✖ | ✖ |
| CUDA (fp16/bf16) | sm>=80 | ✔ | ✖ | fp16/bf16 | 320~1024 | ✔ | ✔ | ✔ | ✔ | ✖ | ✔ | ✖ |
| CUDA FP8 | sm_89 | ✔ | ✖ | fp16/bf16 in | 64~224 | ✔ | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ |
| CUDA FP8 | sm_120{a,f} | ✔ | ✖ | fp16/bf16 in | 64~1024 | ✔ | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ |
| CUDA FP4 | sm_120{a,f} | ✔ | ✖ | fp16/bf16 in | 64~512 | ✔ | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ |
| Triton | sm>=80, ROCm | ✔ | ✔ | fp16/bf16 | 320~1024 | ✔ | ✔ | ✔ (bias grad ✔) | ✔ | ✖ | ✖ | ✔ |
| CuTe-DSL | sm>=80 | ✔ | ✔ | fp16/bf16 | 320~1024 | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ | ✖ |
| CuTe-DSL | sm_90a | ✔ | ✔ | fp16/bf16 | 320~512 | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ | ✖ |
| CuTe-DSL | sm_100a | ✔ | ✔ | fp16/bf16 | 512 | ✔ | ✔ | ✖ | ✖ | ✔ | ✖ | ✖ |

</div>

Notes:

- Headdim ranges above are the FFPA-tuned ranges. FFPA also computes
  D<=256 with every backend when the matching `FFPA_*_ALLOW_SMALL_D=1`
  escape hatch is set (see [Automatic SDPA fallback](#automatic-sdpa-fallback)).
- The CUDA FP8 / FP4 rows take fp16/bf16 activations and quantize
  Q/K/V inside the kernel; see the [FP8 / FP4 guide](../quantized.md).
- `NHD zero-copy` means the kernel consumes diffusers-style
  `[B, N, H, D]` (or fused-QKV chunk) views without materializing a
  packed copy — a CUDA-backend inference-path feature only.

## Automatic SDPA fallback

`ffpa_attn_func` keeps the SDPA-aligned contract: instead of failing, it
falls back to `torch._C._nn.scaled_dot_product_attention` when the
requested shape/device is outside a backend's contract:

| Condition | Behavior |
|---|---|
| `D <= 256` (small-D) | SDPA, unless `FFPA_CUDA_ALLOW_SMALL_D=1` / `FFPA_TRITON_ALLOW_SMALL_D=1` / `FFPA_CUTE_ALLOW_SMALL_D=1` |
| `D > 1024` | SDPA |
| `8 <= Nq < 512` or `Nkv < 512` (short sequences) | SDPA (`Nq in [1, 8)` still uses the native decode path) |
| CuTe-DSL hardware/shape mismatch | SDPA |
| `forward_backend='sdpa'` | always SDPA |

Everything else (unsupported dtype, `attn_mask` under CuTe-DSL,
`dropout_p > 0` under CuTe-DSL, invalid backend combinations) raises
`NotImplementedError` / `TypeError` / `ValueError` — there is **no
silent fallback** for those. Set `FFPA_FORCE_NO_SDPA_FALLBACK=1` in
tests to assert the backend kernels instead of the fallback.

## Backends in this section

- [CUDA Backend](cuda.md) — `CUDABackend`: native / TMA / CuTe fp16 C++
  kernels plus FP8 and NVFP4 attention, forward only.
- [Triton Backend](triton.md) — `TritonBackend`: the default
  forward + backward backend, autotune and persistent tuned configs.
- [CuTe-DSL Backend](cutedsl.md) — `CuTeDSLBackend`: Python CuTe kernels
  for sm_80/sm_90a/sm_100a, including the `tcgen05` 2-CTA D=512 path and
  the varlen (packed THD) API.
- [FP8 / FP4 Attention](../quantized.md) — quantized attention usage,
  knobs and precision notes for the CUDA backend.
