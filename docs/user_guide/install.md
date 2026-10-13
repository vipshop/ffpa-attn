# Installation

FFPA ships as two distributions that install the same `ffpa_attn` Python
package:

| Distribution | Contents | Where it comes from |
|---|---|---|
| `ffpa-attn` | Python sources, Triton and CuTe-DSL backends | [PyPI](https://pypi.org/project/ffpa-attn/) |
| `ffpa-attn-cu13` | the same sources **with the CUDA extension compiled in** (`D > 256`, forward only) | [the FFPA wheel index](https://deftruth.github.io/ffpa-attn/whl/cu13/) |

The CUDA wheel is roughly 1 GiB: it carries every headdim in the release set for
two architectures. PyPI caps a file at 100 MB and a git repository accepts less,
so these wheels are published as GitHub Release assets and served through a
static PEP 503 index. Install the one you need -- the two distributions are not
meant to be installed side by side, because both own `ffpa_attn/`.

## Requirements

- Linux, x86_64.
- glibc 2.38 or newer. The platform tag of every wheel is derived from the
  GLIBC symbols the extension actually needs, so pip refuses to install a wheel
  your system cannot run (the index page lists the exact tags).
- Python 3.10 - 3.14. The published CPython set is whatever the index page lists.
- PyTorch built against **CUDA 13** (`torch.version.cuda == "13.0"`), validated
  against `torch==2.11.0+cu130`.
- An NVIDIA GPU covered by the compiled architectures: `sm_89` (Ada: L20/L40,
  RTX 40 series) or `sm_120f` (Blackwell GeForce: RTX 5090, RTX PRO 5000/6000).

## Install from the index

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # expect ...+cu130 13.0
pip install ffpa-attn-cu13 --index-url https://deftruth.github.io/ffpa-attn/whl/cu13/
```

Install torch **first**, from the PyTorch index you normally use:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install ffpa-attn-cu13 --index-url https://deftruth.github.io/ffpa-attn/whl/cu13/
```

> [!NOTE]
> `--index-url` and nothing else. This index carries exactly one project, and
> `ffpa-attn-cu13` does not exist on PyPI, so nothing here can be shadowed by
> another index.

> [!WARNING]
> The index carries no dependencies. pip resolves only what is already
> installed, so on a fresh environment add
> `--extra-index-url https://pypi.org/simple` -- and then pin torch explicitly
> (`ffpa-attn-cu13` requires `torch>=2.10.0`, so PyPI's build, which is not the
> cu130 one, would otherwise be a valid candidate to satisfy it and pip may
> replace your torch with it):
>
> ```bash
> pip install ffpa-attn-cu13 --index-url https://deftruth.github.io/ffpa-attn/whl/cu13/ \
>   --extra-index-url https://pypi.org/simple "torch==2.11.0"
> ```

> [!WARNING]
> Do not install PyPI's `ffpa-attn` in the same environment. Both distributions
> provide `ffpa_attn/`, and whichever is installed last wins file by file.
> Uninstall one before installing the other:
>
> ```bash
> pip uninstall -y ffpa-attn && pip install ffpa-attn-cu13 --index-url https://deftruth.github.io/ffpa-attn/whl/cu13/
> ```

## Verify the extension

```python
import ffpa_attn
from ffpa_attn.cuda import (
  CUDA_FWD_AVAILABLE,
  CUDA_TMA_AVAILABLE,
  CUDA_CUTE_TMA_AVAILABLE,
)

print(ffpa_attn.__version__, CUDA_FWD_AVAILABLE, CUDA_TMA_AVAILABLE, CUDA_CUTE_TMA_AVAILABLE)
```

The three flags report what was **compiled into** the extension; the runtime
dispatch still checks the device, so the TMA and CuTe-TMA families are used on
`sm_120f`-class GPUs only. A pure-Python install prints `False` for all three.

## Building the release wheels

The release pipeline is `tools/build_releases.sh`, which drives
`tools/build_fast.sh` -- the same script `build.sh` forwards to -- so the
extension in a release wheel is compiled from the same flags as the documented
source build. Only `--arch`, `--ext` and `--headdim` differ between profiles:

| Profile | Flags | Use |
|---|---|---|
| `mini` (default) | `--arch 89,120f --ext all --headdim 128` | tens of MB, exercises the whole build -> wheel -> index -> install path |
| `full` | `--arch 89,120f --ext all --headdim default` | the published wheel, `64,128,192,256,320,512` |

```bash
bash tools/build_releases.sh                       # mini, py312, local only
FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=0.2.5 FFPA_RELEASE_ENVS="py310 py311 py312 py313 py314" \
  bash tools/build_releases.sh                     # the published set
```

The build runs in a workspace prepared by `tools/release_workspace.py`, because
the distribution name is a static field of `pyproject.toml` and cannot be
overridden from `setup.py`; the workspace carries a renamed `pyproject.toml`
and copies of the writable source trees. `tools/fix_release_wheel.py` then
repacks each wheel once to rewrite the extension's RPATH (`$ORIGIN`) and to tag
it with the manylinux floor its GLIBC symbols imply.

Publish and index it:

```bash
bash tools/publish_index.sh \
  --tag ffpa-attn-cu13-v0.2.5 \
  --wheels dist/ffpa_attn_cu13-0.2.5-*.whl \
  --site-repo /path/to/DefTruth.github.io
```

It uploads the wheels as release assets of the site repository (requires
`gh auth login`), regenerates `ffpa-attn/whl/cu13/` in the site repository, and
pushes. The index merges into what is already published, so older versions keep
resolving. Pre-release versions are refused: the `mini` profile's
`0.2.5.devN` wheels exist to validate the pipeline and are missing most
headdims, so they must never be what `pip install ffpa-attn-cu13` resolves to.

## Building from source instead

See [Quick Start](../index.md) -- `bash ./build.sh --arch sm_120f --ext all
--headdim default` builds the extension in place.
