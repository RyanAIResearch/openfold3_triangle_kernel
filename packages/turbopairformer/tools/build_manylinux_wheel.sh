#!/usr/bin/env bash
# Produce the PyPI wheel using an immutable manylinux base and CUDA toolkit.
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="$(git -C "$PACKAGE_ROOT" rev-parse --show-toplevel)"
SOURCE_COMMIT="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
OUTPUT_ROOT="${1:-$PACKAGE_ROOT/dist/manylinux}"
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]]; then
  echo "commit the release source before building" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
if [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "use an empty output directory" >&2
  exit 2
fi
BUILDER="turbopairformer-manylinux-builder:$SOURCE_COMMIT"
RUNTIME="turbopairformer-manylinux-runtime:$SOURCE_COMMIT"
STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/turbopairformer-container.XXXXXX")"
trap 'rm -rf "$STAGING_ROOT"' EXIT
# A linked Git worktree may keep its repository metadata outside SOURCE_ROOT.
# Materialize a self-contained clone for the container's read-only bind mount.
git clone --quiet --no-local --no-checkout "$SOURCE_ROOT" "$STAGING_ROOT/source"
git -C "$STAGING_ROOT/source" checkout --quiet --detach "$SOURCE_COMMIT"
docker build --target runtime --tag "$RUNTIME" \
  --file "$PACKAGE_ROOT/tools/Dockerfile.manylinux" "$PACKAGE_ROOT"
docker build --target builder --tag "$BUILDER" \
  --file "$PACKAGE_ROOT/tools/Dockerfile.manylinux" "$PACKAGE_ROOT"
docker run --rm --gpus all --network none \
  --mount "type=bind,src=$STAGING_ROOT/source,dst=/source,readonly" \
  --mount "type=bind,src=$OUTPUT_ROOT,dst=/output" \
  --env "SOURCE_COMMIT=$SOURCE_COMMIT" \
  --env "MAX_JOBS=${MAX_JOBS:-8}" \
  "$BUILDER" bash -ceu '
    git config --global --add safe.directory /source
    git clone --quiet --no-local /source /tmp/release-source
    git -C /tmp/release-source checkout --quiet --detach "$SOURCE_COMMIT"
    cd /tmp/release-source/packages/turbopairformer
    bash tools/build_sm90_wheel.sh --pypi
    cp dist/pypi/* /output/
    python -m pip freeze --all > /output/builder-pip-freeze.txt
  '
docker image inspect "$BUILDER" "$RUNTIME" \
  --format '{{.Id}} {{json .RepoTags}}' > "$OUTPUT_ROOT/container-images.txt"
printf 'Wheel and build receipts: %s\nRuntime image: %s\n' "$OUTPUT_ROOT" "$RUNTIME"
