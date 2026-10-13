# CUDA Backend

`CUDABackend` is FFPA's hand-written **forward-only** C++ / CUDA backend
(`src/ffpa_attn/cuda`, `csrc/cuffpa`). It hosts the large-headdim fp16/bf16
kernels (`D` up to 1024) plus the [FP8 / FP4 quantized attention](../quantized.md)
families. Backward is not supported — route training backward to
[Triton](triton.md) or SDPA.

```python
import torch
from ffpa_attn import CUDABackend, ffpa_attn_func

q = torch.randn(1, 32, 8192, 512, dtype=torch.bfloat16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

# Forward-only inference (no grad, explicit CUDABackend) takes the fast path.
with torch.no_grad():
  o = ffpa_attn_func(q, k, v, forward_backend=CUDABackend())
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `acc` | `"f32"` | MMA accumulator precision (`"f32"` / `"f16"`). `"f16"` needs an `ENABLE_FFPA_F16_ACC=1` build; bf16 activations always require `acc="f32"`. |
| `stages` | auto | Pipeline depth. Auto by impl/arch: quantized paths 2, sm90+ native/TMA 4, sm90+ CuTe 2, sm80 native 3. The C++ launcher may clamp for smem limits. |
| `enable_tma` | auto | Select the TMA kernel. `None` auto-resolves (see below). |
| `enable_cute` | auto | Select the CuTe kernel family. `None` auto-resolves. |
| `enable_fp8` | `False` | FP8 attention (fp16/bf16 inputs, quantized in-kernel). |
| `enable_fp4` | `False` | NVFP4 attention. Mutually exclusive with `enable_fp8`. |
| `force_fp8_sm89` | `False` | Force the TMA-free fp8 sm_89 kernels even on sm120+. |
| `tensor_layout` | `"HND"` | `"HND"` (BHND `[B,H,N,D]`, SDPA layout) or `"NHD"` (`[B,N,H,D]`, diffusers layout) for inputs/outputs. |
| `fp8_*` / `fp4_*` knobs | see [FP8/FP4 guide](../quantized.md) | smooth-K/V, per-block/thread/channel quant, int8 QK, PV acc dtype, hadamard, hybrid. |

The `(enable_tma, enable_cute)` pair resolves to a C++ implementation hint:
neither → `NATIVE` (cp.async), TMA only → `TMA`, CuTe only → `CUTE`
(CuTe cp.async), both → `CUTE_TMA`. With both left at `None`, the backend
auto-picks `CUTE_TMA` on sm120+ when the CuTe-TMA extension was compiled,
then `TMA`, then `NATIVE`.

## Kernel families

| Family | Arch | D coverage | Notes |
|---|---|---|---|
| **Native** (cp.async / TMA) | sm80+ | default set + runtime pad, `D<=1024` | The `AUTO`/`NATIVE` default; split-KV two-stage decode for `Nq==1`. |
| **CuTe fp16** (`CUTE_TMA`) | **sm120** | persist-D `D<=128`, split-D `128<D<768`, M4N2 `768<=D<=1024` | Production fp16 family on GeForce Blackwell; `%32` multiples of D hit the kernels directly, others fall back to native. |
| **CuTe fp16** (`CUTE`) | sm80+ | persist-D `D<=128`, split-D, M4N2 `D>=768` | The TMA-free CuTe family for sm80/sm89; matches the sm120 family bitwise at D=64/128. |
| **CuTe FP8** | sm120 | persist-D `D<=224`, split-D `224<D<768`, M4N2 `768<=D<=1024` | `enable_fp8=True`. |
| **CuTe FP8 sm_89** | sm89 (L20 / 4090) | persist-D `D<=224`, split-D M8N1 `224<D<=512` | cp.async only (no TMA on sm89); `f16` PV accumulator; enabled by `ENABLE_FFPA_FP8_SM89_EXT` builds. |
| **CuTe FP4** | sm120 only | persist-D `{64,128,192,256}`, split-D `(256,768)`, M4N2 `[768,1024]` | `enable_fp4=True`; NVFP4 (e2m1 + ue4m3 microscaling). |

**Split-D / M4N2 crossover**: M8N1 wins below `D=768` and M4N2 above it,
because the `PV` output accumulator costs `D/2` registers/thread under
M8N1 (spilling past ~`D=768`) and `D/4` under M4N2. See
[Split-D and TiledMMA](../../index.md#split-d-and-tiledmma) for the design.

A few more dispatch facts worth knowing:

- Native also serves the `D%32!=0` and bias/dropout fallback cases on
  sm120 — the CuTe launchers materialize packed copies when the layout or
  head dim is out of their contract.
- On sm120 the CuTe bias/dropout path is functional but slower than the
  native TMA kernel, so the dispatcher keeps the native kernel unless
  `enable_cute` is forced.
- Decode (`Nq==1`) uses a two-stage split-KV kernel with a
  wave-efficiency-greedy split count in the native family.

## Feature support

| Feature | Native (sm80/TMA) | CuTe fp16 | FP8 | FP4 |
|---|:---:|:---:|:---:|:---:|
| causal (tail-aligned) | ✔ | ✔ | ✔ | ✔ |
| GQA / MQA | ✔ | ✔ | ✔ | ✔ |
| cross-attn (`Nq != Nkv`) | ✔ | ✔ | ✔ | ✔ |
| additive `attn_mask` | ✔ | ✔ | ✔ | ✔ |
| dropout (`p>0`) | ✔ | ✔ | ✖ | ✖ |
| `Nq==1` decode (split-KV) | ✔ | — | ✖ | ✖ |
| BHND packed | ✔ | ✔ | ✔ | ✔ |
| packed-NHD view | materializes BHND | ✔ zero-copy | ✔ zero-copy | ✔ zero-copy |
| strided-NHD (fused-QKV chunk) | materializes BHND | ✔ (fc-1/fc-2) | ✔ | ✔ |
| head_dim runtime pad | ✔ (`D%8==0`, no QKV materialization) | ✔ (32-aligned) | ✔ (32-aligned) | ✔ (64-aligned) |
| softmax_lse | ✔ | ✔ | ✔ | ✔ |

Causal semantics are **tail-aligned** in every backend: query row `r`
attends to `k <= r + (Nkv - Nq)` (FlashAttention convention), which differs
from SDPA's top-left alignment when `Nq != Nkv`. Build the reference with an
explicit additive mask in that regime.

## Layouts: BHND, packed-NHD and strided-NHD

`tensor_layout="NHD"` accepts diffusers-style `[B, N, H, D]` storage
(or fused-QKV views whose row stride exceeds `H*D`) and returns the output
in the same layout — no `contiguous()` copy. The zero-copy contract is:

- `stride(-1) == 1`, `stride(2) == D` (packed row) or `> H*D` (strided row),
- 16-byte alignment for the data pointer / row / batch strides.

NHD is a fast-path-only feature: it requires a no-grad call with an explicit
`CUDABackend` whose persist-D kernel applies. Outside that contract the
public API raises rather than silently misreading `[B, N, H, D]`.

## Headdim coverage and padding

- Compiled headdims are decided at build time (`--headdim default` =
  `64,128,192,256,320,512`; `all` adds `768`; explicit lists add anything).
- Any `D % 8 == 0` in `[8, 1024]` is accepted at runtime via alignment
  padding: Q/K/V are **not** materialized — the native family zero-fills at
  load time, the quantized families zero-fill pad columns inside the
  quantize kernels, and only `O` is allocated padded and sliced back.
- The softmax scale always uses the true (unpadded) `D`.

## Limits and gates

- **Forward only** — `CUDABackend` asserts `backward=False`.
- fp16/bf16 activations; bf16 forces `acc="f32"`.
- `D > 1024` falls back to SDPA. Small-D (`D <= 256`) runs the CUDA kernels
  by default and falls back only when `FFPA_CUDA_ALLOW_SMALL_D=0` is set
  explicitly; see
  [Automatic SDPA fallback](overview.md#automatic-sdpa-fallback).
- Default source builds compile **bf16 inputs only** and **no fp32
  attn_mask variants** (see the build flags below). fp16 inputs raise a
  "rebuild with `ENABLE_FFPA_CUDA_INPUT_FP16=1`" error; fp32 masks are
  downcast to the query dtype by the Python layer.
- `attn_mask` + `is_causal=True` is rejected for every backend.

## Building

The CUDA extension is optional and off by default (`ENABLE_FFPA_CUDA_IMPL`).
The recommended wrapper is `tools/build_fast.sh` (aliased as `./build.sh`):

```bash
# Full build (all ext families, default headdim set).
bash ./build.sh --arch sm_120f --ext all --headdim default --jobs 64
# Only forward headdims you need during development.
bash ./build.sh --arch sm_120f --headdim 128,512 --ext all --jobs 64
```

| Flag | Maps to | Notes |
|---|---|---|
| `--arch <spec>` | `FFPA_BUILD_ARCH` | `sm_120f` is required on Blackwell GeForce for `setmaxnreg`; sm89 targets Ada. |
| `--ext all\|none\|<csv>` | `ENABLE_FFPA_CUDA_IMPL` / `ENABLE_FFPA_CUTE_EXT` / `ENABLE_FFPA_TMA_EXT` | `--ext all` builds native + CuTe + TMA (fp8/fp4 included). |
| `--headdim default\|all\|<list>` | `FFPA_DEV_HEADDIMS` | `all` = default + 768; other headdims must be listed explicitly. |
| `--stages <csv\|all>` | `FFPA_BUILD_STAGES` | Default `"2,3"`; out-of-set runtime stages clamp to the nearest compiled one. |
| `--debug-ext <fp16\|fp8\|fp4 csv\|all\|none>` | `ENABLE_FFPA_BUILD_DEBUG` | Compiles per-family debug dispatch knobs (`FFPA_FP8_FORCE_KERNEL`, …). Off by default; blindly enabling all families slows the build down a lot. |
| `--editable`, `--clean`, `--shm`, `-j`, `--nvcc-threads`, `--ptxas-verbose` | wrapper options | See [Env Vars](../../env.md). |

Default build trimming (source builds):

- bf16 inputs only — add `ENABLE_FFPA_CUDA_INPUT_FP16=1` to compile the
  fp16-input translation units.
- no fp32-mask variants — add `ENABLE_FFPA_CUDA_MASK_FP32=1` to compile them.
- no fp16 MMA-acc kernels — add `ENABLE_FFPA_F16_ACC=1` for
  `CUDABackend(acc="f16")`.
- **quantized split-D kernels are opt-in**: `--ext all` compiles only the
  persist-D quantized windows (fp8 `D<=224`, fp4 `64<=D<=256`). Larger-D
  quantized calls raise with a `ENABLE_FFPA_QUANT_SPLIT_D=1` hint; set that
  flag to compile the split-D / M4N2 quantized families.

## Verifying

```bash
# Small-D (D <= 256) benchmarks the CUDA kernels by default; set
# FFPA_CUDA_ALLOW_SMALL_D=0 to restore the SDPA fallback.

# Forward-only sweep with the CUDA backend (native path).
python -m ffpa_attn.bench --fwd-backend cuda --no-bwd

# FP8 / FP4 configurations, D=128.
python -m ffpa_attn.bench --fwd-backend cuda --cuda-impl fp8 --no-bwd --D 128 --verbose
python -m ffpa_attn.bench --fwd-backend cuda --cuda-impl fp4 --no-bwd --D 128 --verbose
```

The bench CLI reports parity against SDPA for each task (self, cross, decode,
GQA, causal, attn-mask, dropout, non-aligned). See
[Triton Autotune](../autotune.md) for the companion `python -m ffpa_attn.autotune`
workflow on the Triton side.
