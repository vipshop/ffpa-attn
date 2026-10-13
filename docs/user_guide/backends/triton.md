# Triton Backend

`TritonBackend` is FFPA's default forward + backward backend
(`src/ffpa_attn/triton`). It runs the large-headdim attention as Triton
kernels on NVIDIA sm_80+ **and on AMD ROCm/HIP GPUs**, and is the only
backend with an active backward pass plus additive-mask gradients.

```python
import torch
from ffpa_attn import TritonBackend, ffpa_attn_func

q = torch.randn(1, 32, 8192, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = ffpa_attn_func(q, k, v, backend=TritonBackend()) # forward + backward
out.sum().backward()
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `autotune` | `False` | Run the Triton autotuner for the launch config. When `False`, FFPA uses the persisted per-device tuned config if one exists, else the built-in launch defaults. |
| `autotune_mode` | `"fast"` | Autotune search space: `"fast"` (small) or `"max"` (larger, slower). Must match the mode of the persisted JSON to be reused. |
| `enable_tma` | `False` | Use the experimental SM90+ TMA forward/backward path. |
| `enable_ws` | `False` | Force warp-specialized configs on the TMA path (requires `enable_tma`). |
| `persist_dkdv` | `False` | Keep the `dK`/`dV` accumulator in fp32 across backward invocations (requires `enable_tma` and backward). |
| `split_launch` | `False` | Issue separate backward launches for `dKdV` and `dQ` for finer-grained scheduling. |
| `preprocess_d_chunk` | `False` | Split the `d_chunk` preprocess across tiles. |
| `grad_kv_storage_dtype` | `None` | Internal `dK`/`dV` storage dtype (`torch.float32` / `torch.float16`), a workaround for causal bf16 cross-tile accumulation precision. |
| `grad_q_storage_dtype` | `None` | Same for `dQ` (relevant with `USE_DKDVDQ_FUSION`). |

`persist_dkdv`, `split_launch`, `preprocess_d_chunk` and the storage-dtype
overrides are backward-only options and assert `backward=True`.

## Feature support

- **Direction**: forward and backward, including `dQ` / `dK` / `dV` and
  additive-mask gradients (`grad_attn_bias` reduced across broadcast dims).
- **Shapes**: self-attention, cross-attention (`Nq != Nkv`), GQA/MQA,
  causal (tail-aligned), decode (`Nq==1`, split-KV two-stage path) and
  non-aligned `N`.
- **`attn_mask`**: boolean (SDPA semantics) and additive bias. The mask may
  broadcast in any of the four dimensions; the backward reduces the bias
  gradient back to the mask's original shape.
- **`dropout_p > 0`**: SDPA-compatible Philox RNG (forward + backward
  replay), bit-exact w.r.t. PyTorch's mem-efficient kernel for
  `B*H*Nq*Nkv <= 2^32`.
- **dtypes**: fp16 / bf16.
- **Headdim**: FFPA-tuned range `320 ~ 1024`; small-D (`D<=256`) also runs
  the Triton kernels by default — set `FFPA_TRITON_ALLOW_SMALL_D=0` to
  restore the SDPA fallback for that window.
- **Layout**: packed `[B, H, N, D]` (SDPA layout) with unit last stride;
  non-contiguous inputs are made contiguous. NHD / strided-NHD zero-copy
  and varlen (packed THD) are **not** supported by this backend.

## Autotune and persistent tuned configs

Runtime autotune is **off by default**: each process reads the persisted
device JSON (`src/ffpa_attn/triton/configs/{device}.json`) when it exists
and falls back to built-in defaults otherwise. To search or regenerate the
persisted configs (single GPU or Ray-based multi-GPU), see
[Triton Autotune](../autotune.md):

```bash
python -m ffpa_attn.autotune --mode max --full-tasks --overwrite # 1 GPU
python -m ffpa_attn.autotune --mode max --full-tasks --num-gpus 8 --overwrite
```

## SM90+ TMA and warp specialization

`TritonBackend(enable_tma=True)` switches the generic forward (and the
matching backward) to the SM90 TMA implementation; adding
`enable_ws=True` selects warp-specialized candidates. Persistent configs can
be generated for both variants with `--enable-fwd-tma` / `--enable-bwd-tma`
(+ `--enable-fwd-ws` / `--enable-bwd-ws`).

## AMD ROCm/HIP

Install **ffpa-attn** into a **ROCm** build of PyTorch and the Triton
backend dispatches to the AMD kernels automatically (forward + backward).
Validated on AMD Instinct MI250X (`gfx90a`, Linux) and Radeon RX 9070 XT
(`gfx1201`, Windows).

## Benchmarking

```bash
# Forward + backward, built-in defaults.
python -m ffpa_attn.bench --fwd-backend triton --bwd-backend triton
# With runtime autotune.
python -m ffpa_attn.bench --fwd-backend triton --bwd-backend triton --tune max
# TMA + warp specialization (SM90+).
python -m ffpa_attn.bench --fwd-backend triton --bwd-backend triton --tune max --fwd-tma --bwd-tma
```

Measured Triton numbers (L20, H20, D=320/512) are listed in
[Bench](../../bench/README.md#more-benchmark).
