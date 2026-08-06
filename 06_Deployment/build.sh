#!/usr/bin/env bash
# Build the SkinBouncer deployment image, baking in whatever detector folders
# currently exist under 06_Deployment/api/models/detectors/ - there's no runtime
# toggle, a detector is "enabled" simply by having been present at build time.
#
# Usage:
#   06_Deployment/build.sh [image-tag]
#
# Runnable from anywhere; paths resolve relative to this script, not the cwd.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DETECTORS_DIR="$SCRIPT_DIR/api/models/detectors"
IMAGE_TAG="${1:-skinbouncer-api:latest}"

# Guarantee the directory exists (it's gitignored - a fresh clone has none of it) so
# the Dockerfile's COPY of it always succeeds, even with zero detectors exported yet.
mkdir -p "$DETECTORS_DIR"

detector_names=()
for dir in "$DETECTORS_DIR"/*/; do
    [ -d "$dir" ] || continue
    detector_names+=("$(basename "$dir")")
done

echo "Building $IMAGE_TAG with ${#detector_names[@]} detector folder(s) from $DETECTORS_DIR:"
if [ "${#detector_names[@]}" -eq 0 ]; then
    echo "  (none - the image will start with no detectors configured)"
else
    for name in "${detector_names[@]}"; do
        echo "  - $name"
    done
fi

docker build -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_TAG" "$REPO_ROOT"

echo "Built $IMAGE_TAG. Run it with:"
echo "  docker run --rm -p 8000:8000 $IMAGE_TAG"
