#!/usr/bin/env bash
# Publish one CUDA channel of ffpa-attn and regenerate the index that serves it.
#
# The wheels are GitHub Release assets of the project repository: at ~80 MB each
# they are close enough to PyPI's 100 MB per-file limit that publishing there
# would be a bet, and a git repository takes less still. What the repository
# does carry is the index itself -- a few KB of HTML -- which is what
#
#   pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/
#
# points at, one directory per CUDA toolkit. Every anchor carries the asset's
# sha256, which pip enforces, so a half-uploaded or replaced asset fails loudly
# instead of installing.
#
# Refuses a version without the channel's local suffix, and a pre-release
# version: tools/build_releases.sh's mini profile produces 0.2.4.devN wheels for
# pipeline validation, and those must never be what `pip install ffpa-attn`
# resolves to. The version is read out of the wheel filenames and has to agree
# with the tag, so a mismatched pair -- a tag naming one version while carrying
# another's assets -- cannot be published.
#
# Usage:
#   # the canonical index: the project's Pages branch, whose URL already carries
#   # the repository name, so the branch root holds whl/cu13/ directly
#   bash tools/publish_index.sh --tag ffpa-attn-cu13-v0.2.4 \
#     --wheels 'dist/ffpa_attn-0.2.4+cu13-*.whl' --site-repo <gh-pages worktree>
#   # the mirror on a personal site, under a directory of its own
#   FFPA_SITE_URL=https://deftruth.github.io FFPA_INDEX_PREFIX=ffpa-attn \
#     bash tools/publish_index.sh --tag ... --wheels ... --site-repo <that checkout>
#   # no upload, just the HTML, against a local server
#   bash tools/publish_index.sh --tag t --wheels dist/*.whl --site-repo DIR --index-only \
#     --release-url http://127.0.0.1:8765/whl/cu13/assets
#
# Environment:
#   FFPA_RELEASE_REPO   repository holding the release assets (default vipshop/ffpa-attn)
#   FFPA_SITE_URL       where the site is served            (default https://vipshop.github.io/ffpa-attn)
#   FFPA_INDEX_PREFIX   path under the site root            (default empty, for the
#                       project Pages branch above; set it for a mirror)
#   FFPA_INDEX_PROJECT  distribution name to index          (default ffpa-attn)
#   FFPA_INDEX_CHANNEL  CUDA channel sub-index              (default cu13)
#   FFPA_FORCE_UPLOAD   1 -> replace an existing asset whose size differs
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RELEASE_REPO="${FFPA_RELEASE_REPO:-vipshop/ffpa-attn}"
SITE_URL="${FFPA_SITE_URL:-https://vipshop.github.io/ffpa-attn}"
# `-` rather than `:-`: an empty prefix is a real choice -- it is what the project
# Pages branch uses, where the repository name already supplies the URL path.
PREFIX="${FFPA_INDEX_PREFIX-}"
CHANNEL="${FFPA_INDEX_CHANNEL:-cu13}"
PROJECT="${FFPA_INDEX_PROJECT:-ffpa-attn}"
WHEEL_STEM="${PROJECT//-/_}"
FORCE_UPLOAD="${FFPA_FORCE_UPLOAD:-0}"

TAG=""
SITE_REPO=""
RELEASE_URL=""
WHEELS=()
INDEX_ONLY=0
NO_PUSH=0

fail() {
  echo "[publish_index] $1" >&2
  exit 1
}

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --wheels)
      # Several paths, because the shell expands a glob before the script sees
      # it: `--wheels dist/ffpa_attn-0.2.4+cu13-*.whl` arrives as many arguments.
      shift
      [[ $# -gt 0 ]] || fail "--wheels requires at least one path"
      while [[ $# -gt 0 && "$1" != --* ]]; do WHEELS+=("$1"); shift; done ;;
    --site-repo) SITE_REPO="$2"; shift 2 ;;
    --release-url) RELEASE_URL="$2"; shift 2 ;;
    --index-only) INDEX_ONLY=1; shift ;;
    --no-push) NO_PUSH=1; shift ;;
    -h|--help) usage ;;
    *) fail "unknown argument '$1' (try --help)" ;;
  esac
done

[[ -n "$TAG" ]] || fail "--tag is required, e.g. --tag ffpa-attn-cu13-v0.2.4"
[[ ${#WHEELS[@]} -gt 0 ]] || fail "--wheels is required (paths or globs of the built wheels)"
[[ -n "$SITE_REPO" ]] || fail "--site-repo is required: the working tree of $RELEASE_REPO"
[[ -d "$SITE_REPO" ]] || fail "no such directory: $SITE_REPO"

wheel_version() {
  # <project stem>-<version>[+<channel>]-<python tag>-<abi tag>-<platform tag>.whl
  local name="${1##*/}"
  printf '%s' "${name#$WHEEL_STEM-}" | cut -d- -f1
}

VERSION=""
for wheel in "${WHEELS[@]}"; do
  [[ -f "$wheel" ]] || fail "no such wheel: $wheel"
  [[ "$(basename "$wheel")" == "$WHEEL_STEM"-* ]] \
    || fail "not a $PROJECT wheel: $wheel"
  version="$(wheel_version "$wheel")"
  [[ -n "$version" ]] || fail "cannot read a version out of $wheel"
  # The mini profile's validation wheels are missing most headdims, so they must
  # never reach the index -- whatever the tag on this run says.
  [[ "$version" != *dev* ]] || fail "refusing to publish a pre-release wheel: $wheel"
  # The local version is what separates this build from the same release on
  # PyPI; without it a user who already has that one satisfies the requirement
  # and pip would install nothing.
  [[ "$version" == *"+$CHANNEL" ]] \
    || fail "$wheel carries '$version' rather than the '$CHANNEL' local version; \
build it with FFPA_RELEASE_CHANNEL=$CHANNEL"
  [[ -z "$VERSION" || "$version" == "$VERSION" ]] \
    || fail "two versions in one publish: $VERSION and $version"
  VERSION="$version"
done
# The tag names the release the assets live on: the channel is in it, the local
# version is not (a '+' in a URL path is a decoding step away from a space).
[[ "$TAG" == "$PROJECT-$CHANNEL-v${VERSION%%+*}" ]] \
  || fail "tag '$TAG' does not name version '$VERSION'; expected '$PROJECT-$CHANNEL-v${VERSION%%+*}'"

if [[ "$INDEX_ONLY" == "0" ]]; then
  command -v gh >/dev/null 2>&1 || fail "the gh CLI is required to upload release assets"
  gh auth status >/dev/null 2>&1 || fail "gh is not authenticated; run 'gh auth login' (or export GH_TOKEN)"
  [[ -n "$RELEASE_URL" ]] || RELEASE_URL="https://github.com/$RELEASE_REPO/releases/download/$TAG"
fi
[[ -n "$RELEASE_URL" ]] || fail "--release-url is required with --index-only"

echo "[publish_index] version $VERSION, tag $TAG, ${#WHEELS[@]} wheel(s)"

asset_size() {
  gh release view "$TAG" --repo "$RELEASE_REPO" --json assets \
    --jq ".assets[] | select(.name == \"$1\") | .size" 2>/dev/null || true
}

index_sha256() {
  # What the currently published page records for one file, if anything: every
  # anchor href ends in '<filename>#sha256=<hex>', with the local version's '+'
  # percent-encoded.
  local page name="${1//+/%2B}"
  page="$SITE_REPO/${PREFIX:+$PREFIX/}whl/$CHANNEL/$PROJECT/index.html"
  [[ -f "$page" ]] || return 0
  sed 's/"/\n/g' "$page" | grep -F -- "$name#sha256=" | head -1 | sed 's/.*#sha256=//'
}

ensure_release() {
  if gh release view "$TAG" --repo "$RELEASE_REPO" >/dev/null 2>&1; then
    echo "[publish_index] release $TAG already exists in $RELEASE_REPO"
    return 0
  fi
  echo "[publish_index] creating release $TAG in $RELEASE_REPO"
  # These wheels are one channel of the package, not a release of it on its own:
  # --latest=false keeps them from taking over the "Latest" badge of the
  # repository's own releases.
  local index_url="$SITE_URL/${PREFIX:+$PREFIX/}whl/$CHANNEL"
  gh release create "$TAG" --repo "$RELEASE_REPO" --title "$PROJECT $CHANNEL ($(basename "$TAG"))" \
    --latest=false \
    --notes "CUDA-extension wheels of \`$PROJECT\`, built against $CHANNEL. Install with:
    pip install $PROJECT --index-url $index_url/"
}

upload_wheels() {
  local wheel name local_size remote_size local_sha published_sha
  for wheel in "${WHEELS[@]}"; do
    name="$(basename "$wheel")"
    local_size="$(stat -c%s "$wheel")"
    local_sha="$(sha256sum "$wheel" | cut -d' ' -f1)"
    remote_size="$(asset_size "$name")"
    if [[ -z "$remote_size" ]]; then
      echo "[publish_index] uploading $name ($((local_size / 1048576)) MiB)"
      gh release upload "$TAG" "$wheel" --repo "$RELEASE_REPO"
      continue
    fi
    # An asset of the same name is the same artifact only if the hash the index
    # records for it still matches: wheels are not byte-reproducible, so an equal
    # size on its own proves nothing, and a mismatch here would leave every
    # install failing its hash check with the upload reported as skipped.
    published_sha="$(index_sha256 "$name")"
    if [[ "$FORCE_UPLOAD" != "1" && "$remote_size" == "$local_size" ]] \
      && { [[ -z "$published_sha" ]] || [[ "$published_sha" == "$local_sha" ]]; }; then
      echo "[publish_index] skipping $name: already uploaded with the same contents"
      continue
    fi
    [[ "$FORCE_UPLOAD" == "1" ]] || fail \
      "$name is already published with different contents (size $remote_size vs $local_size, \
index sha256 '${published_sha:-none}'); set FFPA_FORCE_UPLOAD=1 to replace the asset, or bump the version"
    echo "[publish_index] replacing $name"
    gh release upload "$TAG" "$wheel" --repo "$RELEASE_REPO" --clobber
  done
}

generate_index() {
  echo "[publish_index] writing the index into $SITE_REPO/${PREFIX:+$PREFIX/}whl/$CHANNEL"
  python3 "$REPO_DIR/tools/make_index.py" \
    --out "$SITE_REPO" \
    --prefix "$PREFIX" \
    --project "$PROJECT" \
    --channel "$CHANNEL" \
    --release-url "$RELEASE_URL" \
    --site-url "$SITE_URL" \
    "${WHEELS[@]}"
}

commit_and_push() {
  local toplevel
  toplevel="$(git -C "$SITE_REPO" rev-parse --show-toplevel 2>/dev/null || true)"
  # The directory has to BE the work tree: a subdirectory of some other checkout
  # resolves too, and committing there would stage this tree in that repository.
  if [[ "$toplevel" != "$SITE_REPO" ]]; then
    [[ "$INDEX_ONLY" == "1" ]] || fail \
      "$SITE_REPO is not the root of a git work tree (got '${toplevel:-none}'); pass the checkout of $RELEASE_REPO"
    echo "[publish_index] $SITE_REPO is not a git work tree root; the generated tree is all there is"
    return 0
  fi
  # An empty prefix means the work tree root *is* the index (a project Pages
  # branch), so stage all of it -- there is nothing else in such a branch. A
  # named prefix is a subdirectory of a shared site: stage that directory, which
  # picks up deleted files in it too.
  if [[ -n "$PREFIX" ]]; then
    git -C "$SITE_REPO" add "$PREFIX" .nojekyll
  else
    git -C "$SITE_REPO" add -A
  fi
  if git -C "$SITE_REPO" diff --cached --quiet; then
    echo "[publish_index] index unchanged"
    return 0
  fi
  git -C "$SITE_REPO" commit -q -m "index: $TAG"
  if [[ "$NO_PUSH" == "1" ]]; then
    echo "[publish_index] --no-push: committed $TAG, nothing pushed"
    return 0
  fi
  # The commit exists either way, so a rejected push is recoverable -- say how,
  # rather than leaving a commit nobody knows about.
  if ! git -C "$SITE_REPO" push; then
    fail "the index commit is local but the push was rejected; reconcile and push it by hand:
  git -C $SITE_REPO pull --rebase && git -C $SITE_REPO push"
  fi
  echo "[publish_index] pushed; the index serves at ${SITE_URL%/}/${PREFIX:+$PREFIX/}whl/$CHANNEL/"
}

if [[ "$INDEX_ONLY" == "0" ]]; then
  ensure_release
  upload_wheels
fi
generate_index
commit_and_push
