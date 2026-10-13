#!/usr/bin/env bash
# Build the release wheels of ffpa-attn's CUDA channel, one per conda environment.
#
# The distribution keeps PyPI's name and the toolkit lives in two places: the
# index directory it is served under (whl/cu13/) and the wheel's local version
# (0.2.4+cu13). One command therefore works across toolkits -- only the
# --index-url changes -- and pip can still tell the CUDA build apart from the
# pure-Python wheel of the same release.
#
# Every compilation switch comes from tools/build_fast.sh -- the script
# `bash build.sh` forwards to -- so the CUDA extension of a release is compiled
# the way the documented source build compiles it. Only the three knobs a
# release has an opinion about are set here (--arch, --ext, --headdim);
# everything else, down to MAX_JOBS and the stripping of the debug families, is
# build_fast.sh's default.
#
# The build runs in a workspace prepared by tools/release_workspace.py, because
# the distribution name is a static field of pyproject.toml and cannot be
# overridden from setup.py: see that script for the detail.
#
# Profiles:
#   mini  (default) --headdim 128, tens of MB. It exercises the whole
#         build -> wheel -> retag -> index -> pip install path in minutes, which
#         is what a slow upload link needs while the pipeline itself is still
#         being worked on. Its version defaults to the setuptools-scm version,
#         i.e. 0.2.4.devN past the last tag, so it sorts below every release and
#         must never be written to the index.
#   full  --headdim default (64,128,192,256,320,512), ~80 MB. The published
#         wheel, which has to carry a release version; the channel is appended
#         as its local version unless one is given.
#   pure  --ext none, no channel in the version, a single environment: the
#         portable pure-Python wheel that PyPI serves (py3-none-any, so the
#         Python version of the building env does not matter). Its version has
#         to be a plain release, and neither nvcc nor ccache is needed.
#
# Usage:
#   bash tools/build_releases.sh                                     # mini, py312
#   FFPA_RELEASE_PROFILE=full FFPA_RELEASE_VERSION=0.2.4 bash tools/build_releases.sh
#   FFPA_RELEASE_ENVS="py310 py311 py312 py313 py314" bash tools/build_releases.sh
#   FFPA_RELEASE_CHANNEL=cu13 FFPA_RELEASE_VERSION=0.2.4 bash tools/build_releases.sh
#
# Environment:
#   FFPA_RELEASE_PROFILE    mini | full | pure             (default mini)
#   FFPA_RELEASE_ENVS       conda envs, space separated    (default py312;
#                           profile pure builds one wheel, so it takes the
#                           first entry only)
#   FFPA_RELEASE_VERSION    version to build; empty -> derived from git
#   FFPA_RELEASE_DIST       where the wheels land          (default dist/)
#   FFPA_RELEASE_WORK_ROOT  build workspaces               (default .tmp/release_build)
#   FFPA_RELEASE_ARCH       FFPA_BUILD_ARCH                (default 89,120f)
#   FFPA_RELEASE_EXT        build_fast.sh --ext value      (default all; pure: none)
#   FFPA_RELEASE_HEADDIM    build_fast.sh --headdim value  (default: per profile)
#   FFPA_RELEASE_JOBS       MAX_JOBS for build_fast.sh     (default: its own default)
#   FFPA_RELEASE_NAME      distribution name                  (default ffpa-attn)
#   FFPA_RELEASE_CHANNEL   CUDA channel: the index directory and the local
#                          version suffix                      (default cu13)
#   FFPA_RELEASE_DRY_RUN    1 -> resolve and print, build nothing
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# The distribution keeps PyPI's name; what distinguishes a build is the CUDA
# toolkit, and it shows up twice: in the index directory (whl/cu13/) and in the
# local version (+cu13). Keeping the name means a requirement does not change
# per toolkit, and the local version is what tells pip that this wheel is not
# the pure-Python one of the same release -- including the case where that one
# is already installed and would otherwise satisfy the requirement unchanged.
PACKAGE_NAME="${FFPA_RELEASE_NAME:-ffpa-attn}"
CHANNEL="${FFPA_RELEASE_CHANNEL:-cu13}"
WHEEL_STEM="${PACKAGE_NAME//-/_}"
PROFILE="${FFPA_RELEASE_PROFILE:-mini}"
# The pure-Python profile: no extension to compile, no channel in the version,
# and a py3-none-any wheel, so a single environment is all it needs.
PURE=0
[[ "$PROFILE" == "pure" ]] && PURE=1
ENV_LIST_RAW="${FFPA_RELEASE_ENVS:-py312}"
RELEASE_VERSION="${FFPA_RELEASE_VERSION:-}"
# 120f, not 120: the CuTe fp8/fp4 families use instructions (.kind::f8f6f4,
# setmaxnreg) that the plain sm_120 target rejects at ptxas time. Family code
# runs on every sm_120 device, which is what the release is for.
ARCH="${FFPA_RELEASE_ARCH:-89,120f}"
# EXT is resolved by profile_ext() below, next to the headdim set: the pure
# profile has to refuse an --ext that asks for a compiled family, and that
# check needs fail().
DIST_DIR="${FFPA_RELEASE_DIST:-$REPO_DIR/dist}"
WORK_ROOT="${FFPA_RELEASE_WORK_ROOT:-$REPO_DIR/.tmp/release_build}"
DRY_RUN="${FFPA_RELEASE_DRY_RUN:-0}"
INCREMENTAL="${FFPA_RELEASE_INCREMENTAL:-0}"
ALLOW_UNCOMMITTED="${FFPA_RELEASE_ALLOW_UNCOMMITTED:-0}"

read -r -a RELEASE_ENVS <<< "$ENV_LIST_RAW"
if [[ "$PURE" == "1" && ${#RELEASE_ENVS[@]} -gt 1 ]]; then
  echo "[build_releases] profile pure builds one py3-none-any wheel; taking ${RELEASE_ENVS[0]} out of '${ENV_LIST_RAW}'" >&2
  RELEASE_ENVS=("${RELEASE_ENVS[0]}")
fi

# Everything that ends up inside the wheel: the extension and the package it
# ships, plus the metadata sources. Tools, docs and this script are not in it,
# which is what lets the profile full guard ignore them.
SHIPPED_INPUTS=(src csrc setup.py env.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE)

fail() {
  echo "[build_releases] $1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

profile_headdim() {
  case "$PROFILE" in
    mini) echo "${FFPA_RELEASE_HEADDIM:-128}" ;;
    full) echo "${FFPA_RELEASE_HEADDIM:-default}" ;;
    # Nothing is compiled, so there is no headdim set to pick.
    pure) echo "" ;;
    *) fail "FFPA_RELEASE_PROFILE must be mini, full or pure, got '$PROFILE'" ;;
  esac
}

profile_ext() {
  case "$PROFILE" in
    pure)
      # Nothing is compiled, so an --ext that asks for a family contradicts the
      # profile; refusing it here beats failing later, after a long build.
      [[ -z "${FFPA_RELEASE_EXT:-}" || "${FFPA_RELEASE_EXT}" == "none" ]] || fail \
        "profile pure builds no extension, so FFPA_RELEASE_EXT has to be 'none', got '${FFPA_RELEASE_EXT}'"
      echo "none" ;;
    *) echo "${FFPA_RELEASE_EXT:-all}" ;;
  esac
}

expected_python() {
  case "$1" in
    py310) echo "3.10" ;;
    py311) echo "3.11" ;;
    py312) echo "3.12" ;;
    py313) echo "3.13" ;;
    py314) echo "3.14" ;;
    *) fail "unsupported release env '$1' (expected one of py310 py311 py312 py313 py314)" ;;
  esac
}

wheel_python_tag() {
  case "$1" in
    py310) echo "cp310" ;;
    py311) echo "cp311" ;;
    py312) echo "cp312" ;;
    py313) echo "cp313" ;;
    py314) echo "cp314" ;;
    *) fail "unsupported release env '$1'" ;;
  esac
}

check_env() {
  local env="$1" expected nvcc toolkit
  expected="$(expected_python "$env")"
  conda run -n "$env" python -c \
    "import sys; raise SystemExit(0 if f'{sys.version_info[0]}.{sys.version_info[1]}' == '$expected' else 1)" \
    || fail "conda env $env does not provide Python $expected"
  conda run -n "$env" python -c "import packaging, setuptools_scm, torch, wheel" \
    || fail "conda env $env is missing build dependencies (packaging/setuptools_scm/torch/wheel)"

  # The toolkit major is what the extension links against: a CUDA 12 wheel
  # installs cleanly and fails to import next to a cu13 torch. The pure-Python
  # profile compiles nothing and so does not care about a toolkit at all.
  [[ "$PURE" == "1" ]] && return 0
  nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
  [[ -x "$nvcc" ]] || fail "nvcc not found at $nvcc; set CUDA_HOME"
  toolkit="$("$nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)"
  [[ "${toolkit:-0}" =~ ^[0-9]+$ && "${toolkit:-0}" -ge 13 ]] \
    || fail "nvcc at $nvcc is not CUDA 13+ (parsed '${toolkit:-unknown}')"
}

resolve_version() {
  local env="$1" resolved
  # The repository root travels in the environment, not inside the -c program:
  # a path with a quote in it would end that program there.
  resolved="$(FFPA_REPO_ROOT="$REPO_DIR" conda run -n "$env" python -c \
    "import os, setuptools_scm; print(setuptools_scm.get_version( \
root=os.environ['FFPA_REPO_ROOT'], version_scheme='python-simplified-semver', \
local_scheme='no-local-version'))" \
    | tr -d '\r' | tail -n 1)"
  [[ -n "$resolved" ]] || fail "could not resolve a version from $REPO_DIR"
  echo "$resolved"
}

resolve_release_version() {
  local env="$1" version uncommitted
  version="${RELEASE_VERSION:-$(resolve_version "$env")}"
  # A pure build is the PyPI release: the plain version, never a channel's
  # local segment -- the local version is exactly what makes the CUDA wheel a
  # different artefact from it.
  if [[ "$PURE" == "1" && "$version" == *"+"* ]]; then
    fail "profile pure builds the plain release version; '$version' carries a local segment"
  fi
  [[ "$version" =~ ^[0-9]+(\.[0-9]+)*([._a-zA-Z0-9]+)?$ ]] \
    || fail "unexpected version '$version'"

  if [[ "$PROFILE" == "mini" ]]; then
    # The mini wheel exists to exercise the pipeline, and tools/publish_index.sh
    # refuses a version that is not a release. Deriving the version keeps that
    # property; pinning one is what the release profiles are for.
    [[ -z "$RELEASE_VERSION" ]] || fail \
      "profile mini derives its own .devN version; FFPA_RELEASE_VERSION belongs to profiles full and pure"
    echo "$version"
    return 0
  fi

  # A release artefact has to correspond to a committed revision, the version
  # has to be installable without --pre, and setuptools-scm's no-local-version
  # scheme drops the dirty marker before this script could see it -- so ask git
  # for the inputs that end up in the wheel instead of reading the version.
  uncommitted="$(git -C "$REPO_DIR" status --porcelain -- "${SHIPPED_INPUTS[@]}")"
  if [[ -n "$uncommitted" && "$ALLOW_UNCOMMITTED" == "1" ]]; then
    echo "[build_releases] WARNING: building from uncommitted inputs; the wheel will not" >&2
    echo "[build_releases] WARNING: correspond to any commit:" >&2
    echo "$uncommitted" >&2
  else
    [[ -z "$uncommitted" ]] || fail \
      "profile $PROFILE builds from committed sources; commit these first (or set
FFPA_RELEASE_ALLOW_UNCOMMITTED=1 to accept an artefact that matches no commit):
$uncommitted"
  fi
  [[ "$version" != *".dev"* ]] || fail \
    "profile $PROFILE needs a release version, got '$version'. Tag the commit, or pass FFPA_RELEASE_VERSION=<x.y.z>"
  if [[ "$PURE" == "1" ]]; then
    # PyPI's release: the plain version, without the channel's local segment.
    echo "$version"
    return 0
  fi
  # The channel rides along as the local version unless one was given, so that
  # `pip install ffpa-attn --index-url .../whl/cu13/` is unambiguous next to the
  # same release on PyPI.
  [[ "$version" == *"+"* ]] || version="${version}+${CHANNEL}"
  echo "$version"
}

release_tag() {
  # One convention for tags and wheels: v0.2.4+cu13 -- the release version, the
  # channel as its local part. The tag sorts next to the version it belongs to
  # rather than after it.
  echo "v${VERSION%%+*}+${CHANNEL}"
}

prepare_workspace() {
  local env="$1" workspace="$2" extra=()
  [[ "$INCREMENTAL" == "1" ]] && extra+=(--incremental)
  echo "[build_releases] $env: preparing the release workspace"
  conda run -n "$env" python "$REPO_DIR/tools/release_workspace.py" \
    --source-root "$REPO_DIR" \
    --workspace "$workspace" \
    --package-name "$PACKAGE_NAME" \
    --version "$VERSION" \
    "${extra[@]+"${extra[@]}"}"
}

build_wheel() {
  local env="$1" workspace="$2" raw_dir="$3"
  local -a command=(
    bash tools/build_fast.sh
    "${BUILD_FLAGS[@]}"
    --no-editable
    wheel . --no-build-isolation --no-deps -w "$raw_dir"
  )
  echo "[build_releases] $env: build_fast.sh ${command[*]}"
  (
    cd "$workspace"
    conda run --no-capture-output -n "$env" env \
      SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFPA_ATTN_CU13="$VERSION" \
      "FFPA_PLAT_TAG=${FFPA_RELEASE_PLAT_TAG:-linux_x86_64}" \
      ${FFPA_RELEASE_JOBS:+MAX_JOBS=$FFPA_RELEASE_JOBS} \
      "${command[@]}"
  )
}

retag_wheel() {
  local env="$1" raw_dir="$2" python_tag="$3" wheels
  shopt -s nullglob
  wheels=("$raw_dir"/"$WHEEL_STEM"-*-"$python_tag"-"$python_tag"-*.whl)
  shopt -u nullglob
  [[ ${#wheels[@]} -eq 1 ]] \
    || fail "$env: expected exactly one wheel for $python_tag in $raw_dir, got ${#wheels[@]}"
  [[ "$(basename "${wheels[0]}")" == *"-${VERSION}-"* ]] \
    || fail "$env: built ${wheels[0]##*/} but expected version $VERSION"

  mkdir -p "$DIST_DIR"
  # A wheel for this (version, python) left by an earlier run would satisfy the
  # assertion below and could be the one that gets published.
  rm -f "$DIST_DIR"/"$WHEEL_STEM"-"$VERSION"-"$python_tag"-"$python_tag"-*.whl
  conda run --no-capture-output -n "$env" python "$REPO_DIR/tools/fix_release_wheel.py" \
    --out "$DIST_DIR" "${wheels[0]}"

  local published
  shopt -s nullglob
  published=("$DIST_DIR"/"$WHEEL_STEM"-"$VERSION"-"$python_tag"-"$python_tag"-*.whl)
  shopt -u nullglob
  [[ ${#published[@]} -eq 1 ]] \
    || fail "$env: expected exactly one published wheel in $DIST_DIR, got ${#published[@]}"
}

publish_pure_wheel() {
  local env="$1" raw_dir="$2" wheels wheel
  shopt -s nullglob
  wheels=("$raw_dir"/"$WHEEL_STEM"-"$VERSION"-py3-none-any.whl)
  shopt -u nullglob
  [[ ${#wheels[@]} -eq 1 ]] || fail \
    "$env: expected exactly one py3-none-any wheel for $VERSION in $raw_dir, got ${#wheels[@]}"
  wheel="${wheels[0]}"
  # A stray .so would still install as py3-none-any and only fail at import
  # time on a machine without this build's CUDA: check the archive itself.
  conda run -n "$env" python -c "
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
sys.exit(1 if any(name.endswith('.so') for name in names) else 0)
" "$wheel" || fail "$env: $wheel contains a .so; it is not a pure-Python wheel"

  mkdir -p "$DIST_DIR"
  rm -f "$DIST_DIR"/"$WHEEL_STEM"-"$VERSION"-py3-none-any.whl
  cp -f "$wheel" "$DIST_DIR/"
}

HEADDIM="$(profile_headdim)"
EXT="$(profile_ext)"

# What goes into the wheel: the pure profile compiles nothing, so it has no
# architecture and no headdim set to hand to build_fast.sh.
if [[ "$PURE" == "1" ]]; then
  BUILD_FLAGS=(--ext "$EXT")
else
  BUILD_FLAGS=(--arch "$ARCH" --ext "$EXT" --headdim "$HEADDIM")
fi

if [[ "$PURE" == "1" ]]; then
  echo "[build_releases] package=$PACKAGE_NAME profile=$PROFILE ext=$EXT"
else
  echo "[build_releases] package=$PACKAGE_NAME profile=$PROFILE arch=$ARCH ext=$EXT headdim=$HEADDIM"
fi
echo "[build_releases] envs=${RELEASE_ENVS[*]// /,} dist=$DIST_DIR work_root=$WORK_ROOT"

require_command conda
# ccache is the nvcc shim; a build that compiles nothing does not need it.
[[ "$PURE" == "1" ]] || require_command ccache

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[build_releases] dry run: build_fast.sh would resolve the flags above to:"
  (
    cd "$REPO_DIR"
    FFPA_EDITABLE=0 bash tools/build_fast.sh \
      "${BUILD_FLAGS[@]}" --no-editable \
      wheel . --no-build-isolation --no-deps -w "$WORK_ROOT/<env>/raw" --dry-run
  )
  echo "[build_releases] dry run: no workspace prepared, no wheel built"
  exit 0
fi

for env in "${RELEASE_ENVS[@]}"; do
  check_env "$env"
done

VERSION="$(resolve_release_version "${RELEASE_ENVS[0]}")"
echo "[build_releases] version=$VERSION"
if [[ "$PROFILE" == "mini" ]]; then
  echo "[build_releases] mini wheels are for pipeline validation only; they carry no release"
  echo "[build_releases] version and tools/publish_index.sh refuses to generate the index for them"
fi

for env in "${RELEASE_ENVS[@]}"; do
  workspace="$WORK_ROOT/$env/workspace"
  raw_dir="$WORK_ROOT/$env/raw"
  if [[ "$INCREMENTAL" == "1" ]]; then
    echo "[build_releases] $env: reusing $workspace (only a relink and repack)"
  else
    rm -rf "$WORK_ROOT/$env"
  fi
  mkdir -p "$raw_dir"
  prepare_workspace "$env" "$workspace"
  build_wheel "$env" "$workspace" "$raw_dir"
  if [[ "$PURE" == "1" ]]; then
    publish_pure_wheel "$env" "$raw_dir"
  else
    retag_wheel "$env" "$raw_dir" "$(wheel_python_tag "$env")"
  fi
done

echo "[build_releases] built ${#RELEASE_ENVS[@]} environment(s): ${RELEASE_ENVS[*]// /,}"
ls -lh "$DIST_DIR"/"$WHEEL_STEM"-"$VERSION"-*.whl

if [[ "$PROFILE" == "mini" ]]; then
  echo "[build_releases] mini wheels validate the pipeline only: they carry a .devN version"
  echo "[build_releases] and tools/publish_index.sh refuses them, so there is nothing to publish"
  exit 0
fi
if [[ "$PURE" == "1" ]]; then
  echo "[build_releases] upload the pure-Python wheel to PyPI:"
  echo "  python -m twine upload \"$DIST_DIR\"/$WHEEL_STEM-$VERSION-*.whl"
  exit 0
fi
echo "[build_releases] publish them with:"
echo "  bash tools/publish_index.sh --tag $(release_tag) \\"
echo "    --wheels \"$DIST_DIR\"/$WHEEL_STEM-$VERSION-*.whl --site-repo <site-repo>"
