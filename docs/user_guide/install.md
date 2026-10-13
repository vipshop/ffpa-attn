# Installation

FFPA ships as two distributions that install the same `ffpa_attn` Python
package:

| Distribution | Contents | Where it comes from |
|---|---|---|
| `ffpa-attn` | Python sources, Triton and CuTe-DSL backends | [PyPI](https://pypi.org/project/ffpa-attn/) |
| `ffpa-attn-cu13` | the same sources **with the CUDA extension compiled in** (`D > 256`, forward only) | [the FFPA wheel index](https://vipshop.github.io/ffpa-attn/whl/cu13/) |

The CUDA wheel carries the whole release set of headdims compiled for two
architectures: around 80 MB today, and it grows with that set. It is therefore
published as GitHub Release assets served through a static PEP 503 index --
close enough to PyPI's 100 MB per-file limit that publishing there would be a
bet, and the CUDA channel is meant to move independently of the pure-Python
package anyway. Install the one you need: the two distributions are not meant to
be installed side by side, because both own `ffpa_attn/`.

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
pip install ffpa-attn-cu13 --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
```

Install torch **first**, from the PyTorch index you normally use:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install ffpa-attn-cu13 --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
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
> pip install ffpa-attn-cu13 --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/ \
>   --extra-index-url https://pypi.org/simple "torch==2.11.0"
> ```

> [!WARNING]
> Do not install PyPI's `ffpa-attn` in the same environment. Both distributions
> provide `ffpa_attn/`, and whichever is installed last wins file by file.
> Uninstall one before installing the other:
>
> ```bash
> pip uninstall -y ffpa-attn && pip install ffpa-attn-cu13 --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
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
FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=0.2.4 FFPA_RELEASE_ENVS="py310 py311 py312 py313 py314" \
  bash tools/build_releases.sh                     # the published set
```

A rebuild that changes only the version or other metadata runs with
`FFPA_RELEASE_INCREMENTAL=1`, which keeps the compiled objects under the
workspace's `build/` and re-links instead of recompiling (a full set takes
seconds per environment rather than minutes).

The build runs in a workspace prepared by `tools/release_workspace.py`, because
the distribution name is a static field of `pyproject.toml` and cannot be
overridden from `setup.py`; the workspace carries a renamed `pyproject.toml`
and copies of the writable source trees. `tools/fix_release_wheel.py` then
repacks each wheel once to rewrite the extension's RPATH (`$ORIGIN`) and to tag
it with the manylinux floor its GLIBC symbols imply.

Publish and index it:

```bash
# the canonical index: the gh-pages branch of this repository, whose URL is
# vipshop.github.io/ffpa-attn/, so the branch root holds whl/cu13/ itself
git fetch origin gh-pages && git worktree add /tmp/ffpa-attn-pages gh-pages
bash tools/publish_index.sh \
  --tag ffpa-attn-cu13-v0.2.4 \
  --wheels dist/ffpa_attn_cu13-0.2.4-*.whl \
  --site-repo /tmp/ffpa-attn-pages

# a mirror under a directory of its own, on any other Pages site
FFPA_SITE_URL=https://deftruth.github.io FFPA_INDEX_PREFIX=ffpa-attn \
  bash tools/publish_index.sh --index-only --tag ffpa-attn-cu13-v0.2.4 \
  --wheels dist/ffpa_attn_cu13-0.2.4-*.whl --site-repo /path/to/DefTruth.github.io
```

It uploads the wheels as release assets of this repository (requires
`gh auth login`), rewrites the index in the site work tree and pushes it. The
index merges into what is already published, so older versions keep resolving.
Pre-release versions are refused: the `mini` profile's `0.2.4.devN` wheels exist
to validate the pipeline and are missing most headdims, so they must never be
what `pip install ffpa-attn-cu13` resolves to.

## Building from source instead

See [Quick Start](../index.md) -- `bash ./build.sh --arch sm_120f --ext all
--headdim default` builds the extension in place.
