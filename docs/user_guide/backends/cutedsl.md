# CuTe-DSL Backend

`CuTeDSLBackend` is FFPA's Python [CUTLASS CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)
implementation (`src/ffpa_attn/cute`). It provides forward + backward for
large-headdim attention on sm_80 / sm_90a / sm_100a, including the
dedicated Blackwell `tcgen05` 2-CTA D=512 path, and it is the only backend
with a varlen (packed THD) API.

```python
import torch
from ffpa_attn import CuTeDSLBackend, ffpa_attn_func

q = torch.randn(1, 32, 8192, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

# backend="cutedsl" is equivalent to forward_backend/backward_backend=CuTeDSLBackend().
out = ffpa_attn_func(q, k, v, backend=CuTeDSLBackend())
out.sum().backward()
```

The public backend string stays `"cutedsl"` and `CuTeDSLBackend` stays the
public config class; `ffpa_attn.cute` is the internal package name.

## Architecture routing

| Device | Path | Applicable shapes |
|---|---|---|
| sm_100a (`CC == 10.0`) | `tcgen05` 2-CTA D=512 forward + backward (TMEM accumulators) | `head_dim == head_dim_v == 512` exactly |
| sm_90a (Hopper) | SM90 specialized kernels | `320 <= D <= 512` (symmetric q/k/v head dim) |
| every other supported arch (sm_80+) | generic SM80 Split-D kernel | `64 <= D <= 1024` (`D % 32 == 0`; with `FFPA_CUTE_ALLOW_SMALL_D=0` the lower bound returns to 320) |

The dense entry (`ffpa_attn_func(..., backend="cutedsl")`) re-views the
SDPA `[B, H, N, D]` inputs into the CuTe-native `[B, N, H, D]` layout as a
stride change (not a copy) and returns contiguous `[B, H, N, D]` storage.
Hardware / head-dim mismatches fall back to SDPA with a one-time warning;
**unsupported features raise immediately** — no silent fallback:

- `attn_mask` → `NotImplementedError`
- `dropout_p > 0` → `NotImplementedError`
- unsupported dtype / head-dim layout → `NotImplementedError` / `ValueError`

The SM100 D=512 entry is an exact specialization and never re-routes; calls
outside its contract raise `NotImplementedError` naming the fired rule
(`varlen-one-sided`, `local-window`, `softcap`, `score_mod`, `mask_mod`,
`aux_tensors`, or a `head_dim` other than 512). Forward and backward share
the same predicate, so a call never mixes the dedicated forward with a
generic backward.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `grad_kv_storage_dtype` | `None` | Optional `torch.float32` / `torch.float16` storage dtype for the internal SM80 dK/dV HBM buffer (final gradients are always cast back to `k.dtype` / `v.dtype`). Generic SM80 backward only: the SM90 and SM100 specialized backwards accumulate in registers/TMEM and **reject** this knob. Backward-only option. |

## Hard constraints

| Constraint | Detail |
|---|---|
| GPU | `sm >= 80`. CC 10.0 gets the dedicated D=512 path, SM90 the specialized kernels, other supported archs use the SM80 fallback. SM120 and other Blackwell minors have no dedicated path yet. |
| Dense head dim | `64 <= D <= 1024` (`D % 32 == 0` in the SM80 fallback). Small-D (`D <= 256`) runs the SM80 Split-D kernels by default; set `FFPA_CUTE_ALLOW_SMALL_D=0` to restore the SDPA fallback. |
| SM90 specialized range | `320 <= D <= 512`, symmetric q/k/v head dim. |
| SM100 dedicated range | `head_dim == head_dim_v == 512`, dense or packed varlen with **both** `cu_seqlens`, fp16/bf16, causal (bottom-right aligned) or non-causal, MHA/GQA/MQA via zero-stride broadcast. Rows with no legal key produce `O = 0` / `LSE = -inf`, and exactly zero gradient backward. |
| Dtype | fp16 / bf16 only. |
| `attn_mask` | Not supported. |
| Dropout | `dropout_p == 0.0` only. |
| Varlen extras | `seqused_k`, `block_table`, `num_splits`, `window_size`, `alibi_slopes`, `softcap`, `score_mod`, `aux_tensors`, `sink`, `attention_mask`, `block_mask` are rejected. |

## Varlen (packed THD) API

`ffpa_attn_varlen_func` mirrors the FlashAttention varlen surface and is
**CuTeDSL-only**:

```python
import torch
from ffpa_attn import ffpa_attn_varlen_func

lengths = [4096, 4096]
total_q, num_heads, head_dim = sum(lengths), 32, 512

q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)
cu_seqlens = torch.tensor([0, 4096, 8192], dtype=torch.int32, device="cuda")

out, lse = ffpa_attn_varlen_func(
  q,
  k,
  v,
  cu_seqlens_q=cu_seqlens,
  cu_seqlens_k=cu_seqlens,  # None -> defaults to cu_seqlens_q
  max_seqlen_q=max(lengths),
  max_seqlen_k=max(lengths),
  causal=True,
  return_lse=True,
)
out.sum().backward()
```

Inputs are packed `[T, H, D]` tensors; sequence boundaries come from int32
`cu_seqlens_*` (`[B+1]`, starting at 0). `causal=True` applies the
tail-aligned (bottom-right) mask per sequence. `return_lse=True` returns an
additional fp32 `[H_q, T_q]` log-sum-exp tensor. Anything outside the
supported set raises an actionable error instead of silently unpacking.

## Blackwell status

**CC 10.0 (`sm100a`, B200) has a dedicated D=512 forward *and* backward** —
a `tcgen05` 2-CTA kernel with TMEM accumulators and cluster-native
barriers. The backward is four launches (preprocess, dV, dK, dQ): at D512 a
single unsplit dK/dV tile would need 246272 bytes of SMEM, more than the
hardware's 232448 bytes.

**SM120 and other Blackwell minors have no dedicated path** and reach the
SM80 fallback. The Hopper kernels cannot simply be relaxed to
`sm >= 90`: the CUTLASS DSL Hopper warpgroup MMA requires
`Arch.sm_90a` and fails on Blackwell at compile/load/runtime respectively
(`OpError: expects arch to be Arch.sm_90a`, `cudaErrorNoKernelImageForDevice`,
or an NVVM serialization failure). A real CuTeDSL path based on
`cutlass.cute.nvgpu.tcgen05` + `blackwell_helpers.make_trivial_tiled_mma`
is the future direction — the same conclusion the SM100 D512 port reached.

## Benchmarking

```bash
python -m ffpa_attn.bench --fwd-backend cutedsl --bwd-backend cutedsl
```

Applicable device/shape combos: dense `320 < D <= 512` on sm90, dense
`D == 512` on sm100, varlen on supported SM8x/SM90/SM100 shapes. Measured
H200 / B200 numbers (up to **535** / **1517** TFLOPS forward) are in
[Bench](../../bench/README.md#more-benchmark).
