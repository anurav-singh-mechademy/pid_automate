"""Lambda entry points for the two P&ID model APIs: detect (RF-DETR) and ocr (RapidOCR).

Each call reads one page PNG and returns what the model saw, in whole-page pixel coordinates of that PNG. The mechanics
run here (tiling, batching, dropping boxes cut by a tile edge, undoing rotation); every judgement call stays with the
caller (confidence cutoffs, text cleaning, de-duplication, tag building). The tile loops are tag_pipeline's
(detect_bubbles, ocr_page) without their filters, using its helpers unchanged, so results match the local pipeline.
Lambda handlers: handler.detect and handler.ocr. Request/response contracts: vault 02_Technical/pid_model_hosting_plan.md."""
import base64
import io
import json
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

import tag_pipeline as TP

Image.MAX_IMAGE_PIXELS = (
    None  # 300 DPI A0 sheets are past PIL's decompression-bomb guard
)
WEIGHTS = Path(os.environ.get("PID_WEIGHTS", "/opt/models/rfdetr_pid.pth"))
WEIGHTS_SHA256 = os.environ.get("PID_WEIGHTS_SHA256", "")
BUILD = dict(commit=os.environ.get("PID_BUILD_COMMIT", "unknown"))
INPUT_BUCKET = os.environ.get(
    "PID_INPUT_BUCKET", ""
)  # the one bucket (and key prefix) pages may be read from
INPUT_PREFIX = os.environ.get("PID_INPUT_PREFIX", "")
DET_BATCH = 8  # tiles per model.predict call, as in tag_pipeline
TILE_RANGE = (256, 4096)  # accepted tile sizes (px)
MAX_TILES = 6  # tile sizes per detect call
MAX_REGIONS = 200
ROTATIONS = (0, 90, 180, 270)  # counter-clockwise, like PIL rotate and np.rot90


class BadRequest(ValueError):
    pass


# ---------- models (loaded on first use, kept for warm calls) ----------
_DETECTOR = []
_OCR = {}


def cpus():
    """CPUs this process may actually use (PID_THREADS overrides). Not os.cpu_count(): that counts every CPU the machine
    has, and one thread per visible CPU in a CPU-limited container made RapidOCR 5-10x slower than threads = CPUs."""
    if os.environ.get("PID_THREADS"):
        return max(1, int(os.environ["PID_THREADS"]))
    n = len(os.sched_getaffinity(0))
    try:  # cgroup v2 quota, e.g. "400000 100000" = 4 CPUs
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def detector():
    if not _DETECTOR:
        import torch

        torch.set_num_threads(cpus())
        _DETECTOR.append(TP.load_rfdetr(WEIGHTS))
    return _DETECTOR[0]


def ocr_engine(use_cls):
    """RapidOCR as the pipeline's full-page OCR uses it (use_cls=False as its orientation check does), with its
    threads set to the CPUs available. Thread count does not change what it reads."""
    if use_cls not in _OCR:
        params = {
            "EngineConfig.onnxruntime.intra_op_num_threads": cpus(),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        if not use_cls:
            params["Global.use_cls"] = False
        _OCR[use_cls] = TP.RapidOCR(params=params)
    return _OCR[use_cls]


def warm():
    """Build-time only: load both models and run each once, so every file they fetch is baked into the image."""
    model, _ = detector()
    model.predict(
        [Image.new("RGB", (TP.DET_INPUT, TP.DET_INPUT), "white")], threshold=0.5
    )
    for use_cls in (True, False):
        ocr_engine(use_cls)(np.full((64, 256, 3), 255, np.uint8))


# ---------- geometry ----------
def to_page(x, y, k, W, H):
    """Point in an image rotated k degrees CCW -> the same point in the unrotated W x H image."""
    if k == 90:
        return W - 1 - y, x
    if k == 180:
        return W - 1 - x, H - 1 - y
    if k == 270:
        return y, H - 1 - x
    return x, y


def box_to_page(b, k, W, H):
    (ax, ay), (bx, by) = to_page(b[0], b[1], k, W, H), to_page(b[2], b[3], k, W, H)
    return [
        float(min(ax, bx)),
        float(min(ay, by)),
        float(max(ax, bx)),
        float(max(ay, by)),
    ]


def rot_array(a, k):
    return a if k == 0 else np.ascontiguousarray(np.rot90(a, k // 90))


# ---------- request ----------
def load_image(uri):
    if not isinstance(uri, str):
        raise BadRequest("image_s3 is required")
    if uri.startswith("s3://"):
        import boto3

        bucket, _, key = uri[5:].partition("/")
        # only the configured input location is readable, whatever else the role could reach (no bucket -> nothing)
        if (
            not INPUT_BUCKET
            or bucket != INPUT_BUCKET
            or not key.startswith(INPUT_PREFIX)
            or ".." in key.split("/")
        ):
            raise BadRequest(
                f"image_s3 must be under s3://{INPUT_BUCKET or '<unset>'}/{INPUT_PREFIX}"
            )
        try:
            data = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:
            traceback.print_exc()
            raise BadRequest(f"cannot read {uri}") from e
        return Image.open(io.BytesIO(data)).convert("RGB")
    if (
        uri.startswith("file://") and os.environ.get("PID_ALLOW_FILE") == "1"
    ):  # local tests only
        return Image.open(uri[7:]).convert("RGB")
    raise BadRequest("image_s3 must be an s3:// URI")


def pick(req, key, default, kind, lo=None, hi=None):
    v = req.get(key, default)
    if kind is float and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    if not isinstance(v, kind) or isinstance(v, bool) != (kind is bool):
        raise BadRequest(f"{key} must be {kind.__name__}")
    if lo is not None and not lo <= v <= hi:
        raise BadRequest(f"{key} must be between {lo} and {hi}")
    return v


def pick_rotation(v, key):
    if v not in ROTATIONS:
        raise BadRequest(f"{key} must be one of {ROTATIONS}")
    return v


# ---------- detect ----------
def run_detect(req):
    t0 = time.time()
    img = load_image(req.get("image_s3"))
    W, H = img.size
    k = pick_rotation(req.get("rotate", 0), "rotate")
    tiles = req.get("tiles")
    if (
        not isinstance(tiles, list)
        or not 1 <= len(tiles) <= MAX_TILES
        or not all(
            isinstance(t, int)
            and not isinstance(t, bool)
            and TILE_RANGE[0] <= t <= TILE_RANGE[1]
            for t in tiles
        )
    ):
        raise BadRequest(f"tiles must be a list of 1-{MAX_TILES} ints in {TILE_RANGE}")
    overlap = pick(req, "overlap", TP.DET_OVERLAP, float, 0.0, 0.5)
    conf = pick(req, "conf_threshold", 0.30, float, 0.0, 1.0)
    model, names = detector()
    t1 = time.time()

    im = img if k == 0 else img.rotate(k, expand=True)
    w, h = im.size
    dets = []
    for tile in tiles:  # detect_bubbles' sliding window, every class, no NMS
        cells = []
        for y0 in TP.tile_starts(h, tile, overlap):
            for x0 in TP.tile_starts(w, tile, overlap):
                t = Image.new("RGB", (tile, tile), "white")
                t.paste(im.crop((x0, y0, min(x0 + tile, w), min(y0 + tile, h))))
                cells.append((x0, y0, t))
        for i in range(0, len(cells), DET_BATCH):
            chunk = cells[i : i + DET_BATCH]
            res = model.predict([t for _, _, t in chunk], threshold=conf)
            for (x0, y0, _), r in zip(chunk, res if isinstance(res, list) else [res]):
                for b, s, c in zip(r.xyxy, r.confidence, r.class_id):
                    if TP.clipped_at_inner_edge(b, x0, y0, tile, w, h, overlap):
                        continue
                    box = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
                    dets.append(
                        dict(
                            bbox=box_to_page(box, k, W, H),
                            conf=float(s),
                            cls=names.get(int(c), str(int(c))),
                            tile=tile,
                        )
                    )
    return dict(
        model=dict(
            name=WEIGHTS.stem,
            sha256=WEIGHTS_SHA256,
            classes=[names[i] for i in sorted(names)],
        ),
        build=BUILD,
        image=dict(width=W, height=H),
        detections=dets,
        timing_ms=dict(
            load=round((t1 - t0) * 1000), infer=round((time.time() - t1) * 1000)
        ),
    )


# ---------- ocr ----------
def read(engine, a):
    """RapidOCR on one array: [(quad 4x2 in a's px, text, score)], raw."""
    r = engine(a)
    if r.txts is None:
        return []
    return [
        (np.asarray(q, float), t, float(s))
        for q, t, s in zip(r.boxes, r.txts, r.scores)
    ]


def line(quad, text, score, rot, region_id):
    return dict(
        text=text,
        score=score,
        quad=[[float(x), float(y)] for x, y in quad],
        rot=rot,
        region_id=region_id,
    )


def run_ocr(req):
    t0 = time.time()
    img = load_image(req.get("image_s3"))
    W, H = img.size
    mode = req.get("mode", "page")
    engine = ocr_engine(pick(req, "use_cls", True, bool))
    arr = np.asarray(img)
    t1 = time.time()
    lines = []
    if (
        mode == "page"
    ):  # ocr_page's tiling at each rotation; edge-cut lines dropped, nothing else filtered
        rots = req.get("rotations", [0, 90])
        if not isinstance(rots, list) or not rots or len(set(rots)) != len(rots):
            raise BadRequest(
                f"rotations must be a non-empty list of distinct values from {ROTATIONS}"
            )
        rots = [pick_rotation(r, "rotations") for r in rots]
        tile = pick(req, "tile", TP.OCR_TILE, int, *TILE_RANGE)
        overlap = pick(req, "overlap", TP.OCR_OVERLAP, float, 0.0, 0.5)
        for rot in rots:
            a = rot_array(arr, rot)
            h, w = a.shape[:2]
            for y0 in TP.tile_starts(h, tile, overlap):
                for x0 in TP.tile_starts(w, tile, overlap):
                    for quad, text, score in read(
                        engine, a[y0 : y0 + tile, x0 : x0 + tile]
                    ):
                        b = [
                            quad[:, 0].min(),
                            quad[:, 1].min(),
                            quad[:, 0].max(),
                            quad[:, 1].max(),
                        ]
                        if TP.clipped_at_inner_edge(b, x0, y0, tile, w, h, overlap):
                            continue
                        lines.append(
                            line(
                                [to_page(x + x0, y + y0, rot, W, H) for x, y in quad],
                                text,
                                score,
                                rot,
                                None,
                            )
                        )
    elif mode == "regions":  # each region cropped, turned, read whole
        regions = req.get("regions")
        if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_REGIONS:
            raise BadRequest(f"regions must be a list of 1-{MAX_REGIONS} regions")
        for n, reg in enumerate(regions):
            if not isinstance(reg, dict):
                raise BadRequest(f"regions[{n}] must be an object")
            bb = reg.get("bbox")
            if (
                not isinstance(bb, list)
                or len(bb) != 4
                or not all(isinstance(v, (int, float)) for v in bb)
            ):
                raise BadRequest(f"regions[{n}].bbox must be [x1, y1, x2, y2]")
            x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
            x2, y2 = min(W, int(np.ceil(bb[2]))), min(H, int(np.ceil(bb[3])))
            if x2 - x1 < 2 or y2 - y1 < 2:
                raise BadRequest(f"regions[{n}].bbox is empty or off the page")
            rot = pick_rotation(reg.get("rotate", 0), f"regions[{n}].rotate")
            rid = reg.get("id", str(n))
            for quad, text, score in read(engine, rot_array(arr[y1:y2, x1:x2], rot)):
                pts = [to_page(x, y, rot, x2 - x1, y2 - y1) for x, y in quad]
                lines.append(
                    line([(x + x1, y + y1) for x, y in pts], text, score, rot, rid)
                )
    else:
        raise BadRequest('mode must be "page" or "regions"')
    return dict(
        engine=dict(rapidocr=version("rapidocr"), onnxruntime=version("onnxruntime")),
        build=BUILD,
        image=dict(width=W, height=H),
        lines=lines,
        timing_ms=dict(
            load=round((t1 - t0) * 1000), infer=round((time.time() - t1) * 1000)
        ),
    )


# ---------- Lambda ----------
def _entry(fn):
    """Accepts a function URL event (JSON in body) or a direct invoke (the request itself). A function URL call gets
    statusCode 400 for a bad request and 500 for a failure, with {"error": ...}; a direct invoke gets the same body."""

    def handle(event, context=None):
        via_url = isinstance(event, dict) and "requestContext" in event
        status = 200
        try:
            if via_url:
                body = event.get("body") or ""
                body = (
                    base64.b64decode(body).decode()
                    if event.get("isBase64Encoded")
                    else body
                )
                try:
                    req = json.loads(body)
                except json.JSONDecodeError as e:
                    raise BadRequest(f"body is not JSON: {e}") from e
            else:
                req = event
            if not isinstance(req, dict):
                raise BadRequest("request must be a JSON object")
            out = fn(req)
        except BadRequest as e:
            status, out = 400, dict(error=str(e))
        except Exception as e:
            traceback.print_exc()
            status, out = 500, dict(error=f"{type(e).__name__}: {e}")
        if via_url:
            return dict(
                statusCode=status,
                headers={"content-type": "application/json"},
                body=json.dumps(out),
            )
        return out

    return handle


detect = _entry(run_detect)
ocr = _entry(run_ocr)
