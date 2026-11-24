# Shared by the deploy/ scripts: loads deploy/config.env and names the image. Sourced, not run.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CONFIG=${CONFIG:-deploy/config.env}
[ -f "$CONFIG" ] || { echo "missing $CONFIG: copy deploy/config.example.env and fill it in" >&2; exit 1; }
set -a
. "$CONFIG"
set +a
export AWS_PAGER=""

need() {   # every named setting must be filled in (a value still in <angle brackets> counts as missing)
  for v in "$@"; do
    if [ -z "${!v:-}" ] || [[ "${!v}" == \<*\> ]]; then echo "set $v in $CONFIG" >&2; exit 1; fi
  done
}

image_tag() {   # <commit>-<first 8 of the weights hash>: an image is tied to the code and the weights it was built from
  if [ -n "$(git status --porcelain deploy batch/tag_pipeline.py requirements.txt)" ]; then
    echo "commit your deploy/ changes first: the image is tagged with the commit" >&2
    exit 1
  fi
  echo "$(git rev-parse --short HEAD)-${WEIGHTS_SHA256:0:8}"
}
