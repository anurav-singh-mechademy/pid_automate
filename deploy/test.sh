#!/usr/bin/env bash
# Offline test of the built image, under Lambda-like limits: no network, read-only filesystem, non-root, 6 CPUs
# (about what Lambda gives at 10 GB). Checks the APIs give the same result as the local pipeline (deploy/test_local.py).
# Reads deploy/config.env; SAMPLE_PDF and SAMPLE_PAGE pick the sheet. Docker needs at least 6 CPUs and 8 GB.
#   deploy/test.sh
. "$(dirname "$0")/common.sh"
need NAME WEIGHTS_SHA256 SAMPLE_PDF SAMPLE_PAGE
TAG=$(image_tag)
docker image inspect "$NAME:$TAG" >/dev/null 2>&1 || { echo "no local image $NAME:$TAG: run deploy/build.sh first" >&2; exit 1; }

docker run --rm --platform linux/arm64 --cpus 6 --network none --read-only --tmpfs /tmp:size=2g --user 1000:1000 \
  -e PID_ALLOW_FILE=1 -v "$SAMPLE_PDF:/samples/sample.pdf:ro" -v "$PWD/deploy:/test:ro" --entrypoint python \
  "$NAME:$TAG" /test/test_local.py --page "$SAMPLE_PAGE"
