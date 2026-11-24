#!/usr/bin/env bash
# Build the model image locally as <NAME>:<commit>-<weights hash>. Reads deploy/config.env.
#   deploy/build.sh
. "$(dirname "$0")/common.sh"
need NAME WEIGHTS_DIR WEIGHTS_SHA256
[ -f "$WEIGHTS_DIR/rfdetr_pid.pth" ] || { echo "no rfdetr_pid.pth in $WEIGHTS_DIR" >&2; exit 1; }
TAG=$(image_tag)

# --provenance/--sbom off: Lambda accepts a plain image manifest, not an index with attestations
docker buildx build --platform linux/arm64 --provenance=false --sbom=false -f deploy/Dockerfile \
  --build-context weights="$WEIGHTS_DIR" --build-arg WEIGHTS_SHA256="$WEIGHTS_SHA256" \
  --build-arg COMMIT="$(git rev-parse --short HEAD)" -t "$NAME:$TAG" --load .
echo "built $NAME:$TAG"
