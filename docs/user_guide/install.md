# Installation

`ffpa-attn` is one distribution with two builds, and which one you get is decided
by the index you point pip at:

| Build | Contents | Where it comes from |
|---|---|---|
| `ffpa-attn` | Python sources, Triton and CuTe-DSL backends | [PyPI](https://pypi.org/project/ffpa-attn/) |
| `ffpa-attn` with CUDA | the same sources **with the CUDA extension compiled in** (`D > 256`, forward only) | [the FFPA wheel index](https://vipshop.github.io/ffpa-attn/whl/cu13/) |

The default install, `pip install -U ffpa-attn` with no index and no flags, is
the pure-Python build -- the portable one, with no architecture, Python-version,
torch or CUDA-toolkit constraint. The CUDA build is the same distribution name
carrying a local version (`0.2.4+cu13`), which is what lets pip tell the two
apart: it is what that index serves, and once installed it also outranks the
plain release, so a later `pip install -U ffpa-attn` against PyPI will not
quietly move you back to the pure-Python one.

The CUDA wheel carries the whole release set of headdims compiled for two
architectures: around 80 MB today, and it grows with that set. It is published
as GitHub Release assets served through a static PEP 503 index -- close enough
to PyPI's 100 MB per-file limit that publishing there would be a bet, and the
CUDA line is meant to move on its own, one index per toolkit: `whl/cu13/` today,
`whl/cu14/` when a build exists for it.

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

## Install the CUDA build

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # expect ...+cu130 13.0
pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
```

Install torch **first**, from the PyTorch index you normally use:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
```

> [!NOTE]
> The index serves `ffpa-attn` itself: `pip install ffpa-attn` with it means the
> CUDA build (`0.2.4+cu13`), while the same command without it means the
> pure-Python build from PyPI.

> [!WARNING]
> The index carries no dependencies. pip resolves only what is already
> installed, so on a fresh environment add
> `--extra-index-url https://pypi.org/simple` -- and pin torch explicitly
> (`ffpa-attn` requires `torch>=2.10.0`, so PyPI's build, which is not the cu130
> one, would otherwise be a valid candidate to satisfy it):
>
> ```bash
> pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/ \
>   --extra-index-url https://pypi.org/simple "torch==2.11.0"
> ```
>
> Adding PyPI to the candidate set cannot pull the pure-Python wheel instead:
> `0.2.4+cu13` sorts above `0.2.4`.

> [!WARNING]
> Going back to the pure-Python build is a reinstall, not a second install --
> both builds are the `ffpa-attn` name and the `ffpa_attn/` package. An upgrade
> request alone will not move you back, because `0.2.4+cu13` outranks `0.2.4`:
>
> ```bash
> pip uninstall -y ffpa-attn && pip install -U ffpa-attn
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
FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=0.2.4 \
  FFPA_RELEASE_ENVS="py310 py311 py312 py313 py314" \
  bash tools/build_releases.sh                     # the published set
```

The channel is what separates one CUDA toolkit's build from another, and it
appears twice: in the index directory (`whl/cu13/`) and as the local version of
the wheel (`0.2.4+cu13`, from `FFPA_RELEASE_CHANNEL`, default `cu13`). A local
version cannot be uploaded to PyPI -- the server rejects it -- which is the
point: the CUDA build lives on an index of its own, and PyPI only ever serves
the pure-Python one.

A rebuild that changes only the version or other metadata runs with
`FFPA_RELEASE_INCREMENTAL=1`, which keeps the compiled objects under the
workspace's `build/` and re-links instead of recompiling (a full set takes
seconds per environment rather than minutes).

The build runs in a workspace prepared by `tools/release_workspace.py`, so
nothing lands in the development tree: the generated translation units, the
extension itself and the `src/ffpa_attn/_version.py` that setuptools-scm writes
all belong to the workspace. `tools/fix_release_wheel.py` then repacks each
wheel once to rewrite the extension's RPATH (`$ORIGIN`) and to tag it with the
manylinux floor its GLIBC symbols imply.

Publish and index it:

```bash
# the canonical index: the gh-pages branch of this repository, whose URL is
# vipshop.github.io/ffpa-attn/, so the branch root holds whl/cu13/ itself
git fetch origin gh-pages && git worktree add /tmp/ffpa-attn-pages gh-pages
bash tools/publish_index.sh \
  --tag ffpa-attn-cu13-v0.2.4 \
  --wheels dist/ffpa_attn-0.2.4+cu13-*.whl \
  --site-repo /tmp/ffpa-attn-pages

# a mirror under a directory of its own, on any other Pages site
FFPA_SITE_URL=https://deftruth.github.io FFPA_INDEX_PREFIX=ffpa-attn \
  bash tools/publish_index.sh --index-only --tag ffpa-attn-cu13-v0.2.4 \
  --wheels dist/ffpa_attn-0.2.4+cu13-*.whl --site-repo /path/to/DefTruth.github.io
```

It uploads the wheels as release assets of this repository (requires
`gh auth login`), rewrites the index in the site work tree and pushes it. The
tag names the channel, not the local version (`ffpa-attn-cu13-v0.2.4`), because
a `+` in a URL path is one decoding step away from a space. The index merges
into what is already published, so older versions keep resolving. Pre-release
versions are refused: the `mini` profile's `0.2.4.devN` wheels exist to validate
the pipeline and are missing most headdims, so they must never be what
`pip install ffpa-attn` resolves to.

## Building from source instead

See [Quick Start](../index.md) -- `bash ./build.sh --arch sm_120f --ext all
--headdim default` builds the extension in place.
