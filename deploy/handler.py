"""Lambda entry points for the P&ID model APIs: detect (RF-DETR), ocr (RapidOCR) and prepare (PDF pages, render,
orientation, Hough bubble sizing, viewer image).

Each call reads one page PNG and returns what the model saw, in whole-page pixel coordinates of that PNG. The mechanics
run here (tiling, batching, dropping boxes cut by a tile edge, undoing rotation); every judgement call stays with the
caller (confidence cutoffs, text cleaning, de-duplication, tag building). The tile loops are tag_pipeline's
(detect_bubbles, ocr_page) without their filters, using its helpers unchanged, so results match the local pipeline.
Lambda handlers: handler.detect and handler.ocr; the ocr function also serves the prepare modes ("pages", "prepare"). Request/response contracts: vault 02_Technical/pid_model_hosting_plan.md."""
import base64
import gc
import hashlib
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
from pdfrender import page_count, render_page

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
    status = 400


class NotFound(BadRequest):
    status = 404


# ---------- models (loaded on first use, kept for warm calls) ----------
_DETECTOR = []
_OCR = {}


def cpus():
    """CPUs this process may actually use (PID_THREADS overrides). Not os.cpu_count(): that counts every CPU the machine
    has, and one thread per visible CPU in a CPU-limited container made RapidOCR 5-10x slower than threads = CPUs."""
    if os.environ.get("PID_THREADS"):
        return max(1, int(os.environ["PID_THREADS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1   # macOS: no affinity
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
def _input_key(key, field):
    """Only the configured input location is readable or writable, whatever else the role could reach."""
    if not INPUT_BUCKET or not key.startswith(INPUT_PREFIX) or ".." in key.split("/"):
        raise BadRequest(f"{field} must be under s3://{INPUT_BUCKET or '<unset>'}/{INPUT_PREFIX}")
    return key


def read_bytes(uri, field="image_s3"):
    """Bytes of an s3:// object in the input location (or a file:// path in local tests)."""
    if not isinstance(uri, str):
        raise BadRequest(f"{field} is required")
    if uri.startswith("s3://"):
        import boto3
        from botocore.exceptions import ClientError

        bucket, _, key = uri[5:].partition("/")
        if bucket != INPUT_BUCKET:
            raise BadRequest(f"{field} must be under s3://{INPUT_BUCKET or '<unset>'}/{INPUT_PREFIX}")
        _input_key(key, field)
        try:
            return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as e:
            # without s3:ListBucket a missing object reads as AccessDenied, so both mean missing or expired
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "AccessDenied", "403"):
                raise NotFound(f"{field} is missing or expired: {uri}") from e
            traceback.print_exc()
            raise BadRequest(f"cannot read {uri}") from e
    if uri.startswith("file://") and os.environ.get("PID_ALLOW_FILE") == "1":  # local tests only
        try:
            return open(uri[7:], "rb").read()
        except FileNotFoundError as e:
            raise NotFound(f"{field} is missing: {uri}") from e
    raise BadRequest(f"{field} must be an s3:// URI")


def write_bytes(stem, ext, data, content_type):
    """Writes <stem>.<ext> to the input location (or under a file:// stem in local tests); returns its URI."""
    if stem.startswith("file://") and os.environ.get("PID_ALLOW_FILE") == "1":  # local tests only
        path = Path(stem[7:] + "." + ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return "file://" + str(path)
    import boto3

    key = _input_key(stem, "out_key") + "." + ext
    boto3.client("s3").put_object(Bucket=INPUT_BUCKET, Key=key, Body=data, ContentType=content_type)
    return f"s3://{INPUT_BUCKET}/{key}"


def load_image(uri):
    return Image.open(io.BytesIO(read_bytes(uri))).convert("RGB")


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
        raise BadRequest('mode must be "page", "regions", "pages" or "prepare"')
    return dict(
        engine=dict(rapidocr=version("rapidocr"), onnxruntime=version("onnxruntime")),
        build=BUILD,
        image=dict(width=W, height=H),
        lines=lines,
        timing_ms=dict(
            load=round((t1 - t0) * 1000), infer=round((time.time() - t1) * 1000)
        ),
    )


# ---------- prepare: PDF pages, render, orientation, Hough, viewer image ----------
# The constants are the pipeline's (tag_pipeline, detect_orientation, estimate_bubble_px, pipeline_viewer); the caller
# sends its own copy on every call, so a threshold change needs no redeploy. Unknown keys are rejected.
ORIENT_DEFAULTS = dict(tiles=TP.ORIENT_TILES, tile=TP.ORIENT_TILE, ink_below=128, max_ink=0.25, rotations=[0, 90, 180, 270],
                       min_conf=TP.ORIENT_MIN_CONF, wide_ratio=1.5, margin=TP.ORIENT_MARGIN)
HOUGH_DEFAULTS = dict(ink_below=160, dilate_px=5, median_blur_px=5, min_dist_mm=6, param1=100, param2=40,
                      diameter_mm=list(TP.HOUGH_MM), ring_min=0.85, band=[0.88, 0.93], band_max=0.35, min_circles=3)
VIEWER_DEFAULTS = dict(long_side=4000, jpeg_quality=78)


def params(req, key, defaults):
    got = req.get(key) or {}
    if not isinstance(got, dict):
        raise BadRequest(f"{key} must be an object")
    unknown = set(got) - set(defaults)
    if unknown:
        raise BadRequest(f"{key} has unknown fields {sorted(unknown)}")
    out = dict(defaults)
    for k, v in got.items():
        d = defaults[k]
        if isinstance(d, list):
            ok = isinstance(v, list) and len(v) == len(d) if k != "rotations" else isinstance(v, list) and len(v) >= 2
            ok = ok and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (isinstance(v, int) or not isinstance(d, int))
        if not ok:
            raise BadRequest(f"{key}.{k} has the wrong type")
        out[k] = v
    if key == "orientation":
        if len(set(out["rotations"])) != len(out["rotations"]) or any(r not in ROTATIONS for r in out["rotations"]):
            raise BadRequest(f"orientation.rotations must be distinct values from {ROTATIONS}")
        if not 256 <= out["tile"] <= 4096 or out["tiles"] < 1:
            raise BadRequest("orientation.tile must be 256-4096 and tiles >= 1")
    if key == "hough" and (out["dilate_px"] < 1 or out["median_blur_px"] < 1 or out["median_blur_px"] % 2 == 0):
        raise BadRequest("hough.dilate_px must be >= 1 and median_blur_px odd")
    return out


def load_pdf(req):
    data = read_bytes(req.get("pdf_s3"), "pdf_s3")
    try:
        n = page_count(data)
    except Exception as e:
        raise BadRequest("pdf_s3 is not a readable PDF") from e
    return data, n


def orientation(img, P):
    """detect_orientation with the caller's constants: ([lines in page coordinates], scores, rotation).
    Lines are scored in the frame they were read in, exactly as detect_orientation scores them."""
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    T = P["tile"]
    cells = [(x, y) for y in range(0, max(H - T, 0) + 1, T // 2) for x in range(0, max(W - T, 0) + 1, T // 2)]
    ink = [(gray[y:y + T, x:x + T] < P["ink_below"]).mean() for x, y in cells]
    order = sorted(range(len(cells)), key=lambda i: -ink[i] if ink[i] < P["max_ink"] else 0)[:P["tiles"]]
    arr = np.asarray(img)
    engine = ocr_engine(False)
    scores, lines = {k: 0 for k in P["rotations"]}, []
    for n, i in enumerate(order):
        x0, y0 = cells[i]
        crop = arr[y0:y0 + T, x0:x0 + T]
        ch, cw = crop.shape[:2]
        for k in P["rotations"]:
            for quad, text, score in read(engine, rot_array(crop, k)):
                if score >= P["min_conf"] and np.ptp(quad[:, 0]) >= P["wide_ratio"] * np.ptp(quad[:, 1]):
                    scores[k] += len(TP.clean_text(text))
                pts = [to_page(x, y, k, cw, ch) for x, y in quad]
                lines.append(line([(x + x0, y + y0) for x, y in pts], text, score, k, f"{n}-{k}"))
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[0], ranked[1]
    rot = best if best and scores[best] >= P["margin"] * max(scores[second], 1) else 0
    return lines, scores, rot


def hough(img, P, dpi):
    """estimate_bubble_px with the caller's constants: (bubble_px or None, clean circles, rejected circles)."""
    import cv2

    mm = dpi / 25.4
    gray = np.asarray(img.convert("L"))
    k = P["dilate_px"]
    ink = cv2.dilate((gray < P["ink_below"]).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    found = cv2.HoughCircles(cv2.medianBlur(gray, P["median_blur_px"]), cv2.HOUGH_GRADIENT, dp=1,
                             minDist=int(P["min_dist_mm"] * mm), param1=P["param1"], param2=P["param2"],
                             minRadius=int(P["diameter_mm"][0] / 2 * mm), maxRadius=int(P["diameter_mm"][1] / 2 * mm))
    if found is None:
        return None, [], []

    def ring(x, y, r, n=180):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.clip((x + r * np.cos(t)).astype(int), 0, ink.shape[1] - 1)
        ys = np.clip((y + r * np.sin(t)).astype(int), 0, ink.shape[0] - 1)
        return ink[ys, xs].mean()

    ok_c, bad_c, diams = [], [], []
    for x, y, r in found[0]:
        clean = ring(x, y, r) >= P["ring_min"] and np.mean([ring(x, y, r * b) for b in P["band"]]) <= P["band_max"]
        (ok_c if clean else bad_c).append([round(float(x)), round(float(y)), round(float(r))])
        if clean:
            diams.append(2 * r)
    return (float(np.median(diams)) if len(diams) >= P["min_circles"] else None), ok_c, bad_c


def run_pages(req):
    t0 = time.time()
    data, n = load_pdf(req)
    t1 = time.time()
    pages = req.get("pages", list(range(n)))
    if not isinstance(pages, list) or not all(isinstance(p, int) and not isinstance(p, bool) and 0 <= p < n for p in pages) \
            or len(set(pages)) != len(pages):
        raise BadRequest(f"pages must be a list of distinct page indexes 0-{n - 1}")
    fps = {str(p): hashlib.md5(render_page(data, p, 72, gray=True).tobytes()).hexdigest() for p in pages}  # as batch_tags
    return dict(page_count=n, fingerprints=fps, build=BUILD,
                timing_ms=dict(read_pdf=round((t1 - t0) * 1000), fingerprint=round((time.time() - t1) * 1000)))


def run_prepare(req):
    tm, t = {}, time.time()

    def lap(name):
        nonlocal t
        now = time.time(); tm[name] = round((now - t) * 1000); t = now

    data, n = load_pdf(req)
    page = req.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or not 0 <= page < n:
        raise BadRequest(f"page must be a page index 0-{n - 1}")
    stem = req.get("out_key")
    if not isinstance(stem, str) or not stem or stem.endswith("/"):
        raise BadRequest("out_key is required (a key stem; .png and .jpg are added)")
    dpi = pick(req, "dpi", TP.DPI, int, 72, 600)
    rotation = req.get("rotation")
    if rotation is not None:
        pick_rotation(rotation, "rotation")
    OP, HP, VP = params(req, "orientation", ORIENT_DEFAULTS), params(req, "hough", HOUGH_DEFAULTS), params(req, "viewer", VIEWER_DEFAULTS)
    want_md5 = pick(req, "render_md5", False, bool)
    lap("read_pdf")

    img = render_page(data, page, dpi)
    del data
    out = dict(render_md5=hashlib.md5(img.tobytes()).hexdigest()) if want_md5 else {}
    lap("render")
    if rotation is None:
        lines, scores, rotation = orientation(img, OP)
    else:
        lines, scores = [], {}
    lap("orient_ocr")
    if rotation:
        img = img.rotate(rotation, expand=True)
    lap("rotate")
    buf = io.BytesIO(); img.save(buf, "PNG")
    png = buf.getvalue(); del buf
    lap("png")
    bubble_px, ok_c, bad_c = hough(img, HP, dpi)
    lap("hough")
    W, H = img.size
    k = min(1.0, VP["long_side"] / max(W, H))
    small = img.resize((round(W * k), round(H * k)), Image.LANCZOS) if k < 1 else img
    buf = io.BytesIO(); small.convert("RGB").save(buf, "JPEG", quality=VP["jpeg_quality"])
    jpg = buf.getvalue(); del buf, small, img
    lap("jpeg")
    page_s3 = write_bytes(stem, "png", png, "image/png")
    jpeg_s3 = write_bytes(stem, "jpg", jpg, "image/jpeg")
    lap("upload")
    del png, jpg
    gc.collect()                                    # one warm container serves many pages
    return dict(page_s3=page_s3, jpeg_s3=jpeg_s3, size=[W, H], rotation=rotation,
                orient_scores={str(k): int(v) for k, v in scores.items()}, orient_lines=lines,
                bubble_px=bubble_px, hough_ok=ok_c, hough_rejected=bad_c, **out, build=BUILD, timing_ms=tm)


def run_prepare_api(req):
    import cv2

    cv2.setNumThreads(int(os.environ.get("PID_CV_THREADS", cpus())))   # Hough costs ~330 MB per thread
    mode = req.get("mode")
    if mode == "pages":
        return run_pages(req)
    if mode == "prepare":
        return run_prepare(req)
    raise BadRequest('mode must be "pages" or "prepare"')


PREPARE_MODES = ("pages", "prepare")


def run_ocr_api(req):
    """The ocr function's entry: OCR modes ("page", "regions") or the prepare modes ("pages", "prepare").
    Logs one line per call (mode, status, duration; never drawing text) so the modes can be told apart in the logs."""
    mode, t0, status = req.get("mode", "page"), time.time(), "ok"
    try:
        return run_prepare_api(req) if mode in PREPARE_MODES else run_ocr(req)
    except BadRequest as e:
        status = e.status
        raise
    except Exception:
        status = 500
        raise
    finally:
        print(json.dumps(dict(event="call", mode=mode if mode in PREPARE_MODES + ("page", "regions") else "invalid",
                              status=status, ms=round((time.time() - t0) * 1000))), flush=True)


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
            status, out = e.status, dict(error=str(e))
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
ocr = _entry(run_ocr_api)                  # also serves the prepare modes
