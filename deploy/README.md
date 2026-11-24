# P&ID model APIs

The two models the P&ID pipeline uses, hosted as AWS Lambda APIs so an agent (TurboAgents) can call them instead of running them locally:

- **detect**: the fine-tuned RF-DETR symbol detector (`rfdetr_pid.pth`, 32 classes)
- **ocr**: RapidOCR 3.10 on onnxruntime 1.31

Both come from **one container image**; each Lambda function runs a different handler from it (`handler.detect`, `handler.ocr`). Everything here lives in `deploy/`. The pipeline code itself is not changed.

## How it works

The agent renders a PDF page to PNG (300 DPI, as `tag_pipeline.load_page` does), uploads it to the input bucket, and calls the two URLs.

```
agent ── PNG ──▶ s3://<NAME>-input-<account-id>/pages/...
agent ── signed POST {image_s3, ...} ──▶ <NAME>-detect URL ──▶ JSON detections
agent ── signed POST {image_s3, ...} ──▶ <NAME>-ocr URL    ──▶ JSON text lines
```

**Each call handles one whole page, not one tile.** The API does the mechanical work: cutting the page into tiles, batching, dropping boxes cut by a tile edge, and mapping everything back to page pixels (undoing any rotation). **Every judgement call stays with the caller:** confidence cutoffs, duplicate removal (`nms_ios`), `clean_text`, the OCR minimum score, bubble sizing and tag building. The API returns what the model saw, unfiltered.

### detect

```json
{"image_s3": "s3://<bucket>/pages/<job>/page-03.png", "tiles": [1024, 1280, 1600],
 "overlap": 0.25, "conf_threshold": 0.30, "rotate": 0}
```
- `tiles`: 1–6 tile sizes (256–4096 px); each runs a full pass over the page.
- Returns `detections: [{bbox, conf, cls, tile}]` in page pixels, all classes, duplicates kept. Also returns `model` (name, sha256, classes), `build.commit`, `image` size and `timing_ms`.

### ocr

```json
{"image_s3": "...", "mode": "page", "rotations": [0, 90], "tile": 1600, "overlap": 0.20, "use_cls": true}
{"image_s3": "...", "mode": "regions", "use_cls": false,
 "regions": [{"id": "r1", "bbox": [0, 0, 1600, 1600], "rotate": 90}]}
```
- `page` mode reads the whole page in tiles at each rotation (what `ocr_page` does).
- `regions` mode reads only the listed areas, each turned by `rotate` (what the orientation check does: 4 areas × 4 rotations in one call).
- Returns `lines: [{text, score, quad, rot, region_id}]`, with text and scores raw from RapidOCR.

Errors come back as `400 {"error": ...}` for a bad request (including an `image_s3` outside the input bucket) and `500` for a failure.

### Calling it

The URLs use IAM auth, so every request must be signed with SigV4 for service `lambda`:

```python
import json, boto3, requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

url, region = "<function-url>", "<aws-region>"
body = json.dumps({"image_s3": "s3://<bucket>/pages/<job>/page.png", "tiles": [1280]})
req = AWSRequest(method="POST", url=url, data=body, headers={"content-type": "application/json"})
SigV4Auth(boto3.Session().get_credentials().get_frozen_credentials(), "lambda", region).add_auth(req)
out = requests.post(url, data=body, headers=dict(req.headers), timeout=960).json()
```

## Why it is built this way

| Choice | Why |
|---|---|
| One call per page, not per tile | A sheet is ~50 detector + ~100 OCR tile calls. As separate API calls that is ~150 network round trips carrying images. |
| APIs return raw model output | The pipeline is still changing. Keeping cutoffs and filters in the caller means experiments never need a redeploy. |
| Pages passed as S3 locations | A Lambda request is capped at 6 MB; a 300 DPI page is close to that. |
| RapidOCR 3.10 kept as is, inside the image | The pipeline's thresholds are tuned to RapidOCR's known mistakes; another OCR engine would break them. |
| Tiling reuses `tag_pipeline` helpers (`tile_starts`, `clipped_at_inner_edge`, `load_rfdetr`) | Results match the local pipeline exactly. `test_local.py` proves it on every build. |
| Python 3.12 on AWS's own Lambda image, pinned by digest | 3.12 is our standard. AWS's 3.11 image is Amazon Linux 2 (glibc 2.26), too old for the pinned torch / opencv / numpy. |
| arm64 | ~20% cheaper on Lambda, builds natively on Apple-silicon Macs, and Arm Linux torch has no GPU libraries. |
| Threads set to the CPUs actually available (`handler.cpus()`) | onnxruntime and torch otherwise start one thread per CPU the machine reports. In a CPU-limited container that made OCR 5–10× slower. Override with `PID_THREADS`. |
| All model files built into the image | Lambda can only write to `/tmp`, and downloads at start-up are slow and can fail. The test runs with networking off to prove nothing is fetched. |
| Function URLs with IAM auth | Simplest front with a 15-minute limit. API Gateway caps requests at 29 s by default. |
| S3 reads limited to the input bucket (`PID_INPUT_BUCKET` / `PID_INPUT_PREFIX`) | Otherwise any caller could make the Lambda read any object its role can reach. No bucket configured means no S3 reads at all. |
| Input pages deleted after 7 days | Pages are client drawings. |
| 10 GB memory | Lambda gives CPU in proportion to memory: 10 GB is ~6 vCPUs. The functions only use ~2 GB. |
| OCR timeout 15 min, detect 5 min | Full-page OCR takes ~2 minutes on Lambda. You pay only for time used, so the maximum costs nothing extra. |

## Where things live

Account-specific values are kept out of the repo. Ask the team for the account and URLs of the current deployment. Within an account, everything is named from `NAME`:

| What | Name |
|---|---|
| Image | ECR repo `<NAME>`, tag `<commit>-<first 8 of weights sha256>` (tags can't be overwritten) |
| Input bucket | `<NAME>-input-<account-id>`, pages under `pages/` |
| Functions | `<NAME>-detect`, `<NAME>-ocr` (arm64, logs in `/aws/lambda/<function>`, kept 30 days) |
| Role | `<NAME>-lambda-role`: read `pages/` in the input bucket, write logs |
| Weights | `rfdetr_pid.pth`: the repo holds only its Git LFS pointer. Try `git lfs pull`; if that doesn't fetch it, get it from the model owner |

## Setup (once per person)

1. **Docker** with buildx, with at least 6 CPUs and 8 GB of memory available to it. **AWS CLI v2**, logged in.
2. **The weights**: put `rfdetr_pid.pth` in a folder, and get its expected sha256 from the model owner (`shasum -a 256 rfdetr_pid.pth` must match it).
3. **Config**: `cp deploy/config.example.env deploy/config.env` and replace every `<placeholder>`. `config.env` is git-ignored; never commit it.
4. **Permissions**: `push.sh` needs ECR rights. `deploy.sh` needs S3 and Lambda rights, plus `iam:CreateRole` and `iam:PassRole` for the role. PowerUser-style roles usually lack the last two; then an admin creates `<NAME>-lambda-role` by hand (Lambda trust, `AWSLambdaBasicExecutionRole`, inline `s3:GetObject` on `arn:aws:s3:::<NAME>-input-<account-id>/pages/*`), and the deploy itself still needs someone with `iam:PassRole`.

## Updating

Run everything from the repo root. Commit your changes first: the image tag is the commit, so the scripts refuse to run with uncommitted changes in `deploy/`, `batch/tag_pipeline.py` or `requirements.txt`.

```
deploy/build.sh     # build <NAME>:<tag> locally (~5 min the first time, cached after)
deploy/test.sh      # offline test: must end with "ALL OK"
deploy/push.sh      # push the tag to ECR (creates the repo the first time)
deploy/deploy.sh    # point both functions at the tag (creates bucket, role, functions, URLs the first time)
```

**Never push an image that hasn't passed `test.sh`.** It checks that, with the caller's own filters applied, the API output matches `detect_bubbles` and `ocr_page`, and that a bad request gets a 400.

**Roll back:** `IMAGE_TAG=<older tag> deploy/deploy.sh`. Older tags stay in ECR.

**New weights:** put the new `rfdetr_pid.pth` in `WEIGHTS_DIR`, set its `WEIGHTS_SHA256`, then build → test → push → deploy. The tag changes with the hash, and responses report the new `model.sha256`.

**Changing the API** (`handler.py`): keep the request and response shapes backward-compatible, or tell everyone who calls it first. If the tiling changes, extend `test_local.py` so it still compares against the pipeline.

**Letting another role call it** (e.g. the TurboAgents worker): set `CALLER_ROLE_ARN` and re-run `deploy.sh`. It grants that role invoke rights on both URLs and write rights to `pages/`. A role in another account also needs its own policy to allow `lambda:InvokeFunctionUrl` and `s3:PutObject` on these resources.

## Known behaviour and gotchas

- **The first call after idle takes 1.5–2 minutes.** Loading torch, RF-DETR and RapidOCR takes longer than Lambda's 10-second start-up limit, so Lambda redoes the start-up inside the first call. Warm calls: detect ~9 s per tile size, full-page OCR ~2 minutes. If the start-up time ever matters, the fix is provisioned concurrency, which costs money even when idle.
- **OCR on Lambda is ~2.7× slower than on an M-series Mac** (113 s vs 42 s per page at 6 threads). If that's too slow, send the 0° and 90° reads as two calls in parallel.
- **"Image manifest not supported" from Lambda:** the image was built with attestations. `build.sh` turns them off (`--provenance=false --sbom=false`); don't build any other way.
- **`AccessDenied` on `iam:CreateRole` / `iam:PassRole`:** see Setup step 4. The bucket and ECR repo will already have been created, so just re-run after the role exists.
- **A "deny" on Lambda calls in IAM's policy simulator** can come from a region restriction when no region is given. Pass `aws:RequestedRegion` as context before believing it.
- **Running the image locally:** its entrypoint starts AWS's Lambda emulator on port 8080 (`docker run -p 9000:8080 <image> handler.detect`, then POST to `localhost:9000/2015-03-31/functions/function/invocations`). `file://` page paths work only with `PID_ALLOW_FILE=1`, which is for local tests only.
- **Read-only filesystem:** on Lambda only `/tmp` is writable. Anything new that downloads or caches files must do it at build time (see `handler.warm()`).
