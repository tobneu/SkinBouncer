#!/usr/bin/env bash
# Builds the plugin jar in a container, so no local JDK or Maven install is needed.
# Output: minecraft_plugin/target/SkinBouncer.jar
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Running as the invoking user keeps target/ owned by you rather than root, which in turn
# means the Maven repo cannot live in the container's /root - so it goes next to the
# sources (gitignored) and doubles as a cache between builds.
# Calling mvn directly skips the image's entrypoint, which unconditionally writes to
# /root and only warns about failing to when running as a non-root user.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint mvn \
    -v "$PLUGIN_DIR":/src \
    -w /src \
    maven:3-eclipse-temurin-25 \
    -q -B -Dmaven.repo.local=/src/.m2 "$@" package

echo "Built $PLUGIN_DIR/target/SkinBouncer.jar"
