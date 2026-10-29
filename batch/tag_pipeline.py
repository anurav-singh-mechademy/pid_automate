# Tag-extraction pipeline definitions (same code as the 'Self-contained tag extraction' cell, minus its single-page run block).
# PAGE_PATH/OUT_CSV below are unused here; batch_tags.py calls the functions per page.
# Self-contained tag extraction for one page: full-page OCR -> RF-DETR bubbles -> attach text to bubbles -> tags.
# Requires: pip install pymupdf pillow numpy torch rfdetr rapidocr
import csv
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import torch
from PIL import Image
from rapidocr import RapidOCR
from rfdetr import RFDETRBase

PAGE_PATH = Path("bhcp/SC62D1967_Ver1-3.pdf")  # .pdf/.png/.jpg
PAGE_NO = 0
WEIGHTS = Path("models/rfdetr_pid.pth")
OUT_CSV = Path("outputs") / f"tags_{PAGE_PATH.stem}_p{PAGE_NO}.csv"
OUT_PNG = OUT_CSV.with_name(OUT_CSV.stem + "_overlay.png")  # full-resolution overlay
DISPLAY_WIDTH = 2400               # px width of the inline preview

DPI = 300
MM = DPI / 25.4                       # px per mm: geometry thresholds below are in drawing mm
OCR_TILE, OCR_OVERLAP = 1600, 0.20    # full-page OCR tiles at native resolution
OCR_MIN_SCORE = 0.5                   # drop low-confidence OCR fragments
DET_OVERLAP = 0.25                    # RF-DETR sliding window (model input 560)
# Dynamic tile size: RF-DETR finds bubbles best when they appear ~50-65 px wide in its 560 px input, so the tile
# is sized from each sheet's measured bubble diameter (12 mm LNGC bubbles -> ~1300 px tiles, 16 mm BHCP -> ~1770).
DET_INPUT = 560
TARGET_BUBBLE_PX = 60                 # bubble width the model should see
DET_TILE_RANGE = (960, 2400)          # clamp; DET_TILE_DEFAULT is used when no circles are found
DET_TILE_DEFAULT = 1280
HOUGH_MM = (8, 20)                    # bubble diameters considered when measuring the sheet
SCALES = (0.8, 1.0, 1.25)             # the best scale is broad, not sharp: run around the estimate and merge
# Bubble-size fallback: when Hough finds < 3 clean circles (lines through bubbles, rotated sheets), use the median
# size of confident RF-DETR bubble boxes from one pass at the default tile size. RF-DETR boxes are ~10% larger than
# the drawn circle (rim + margin; 206 vs 186 px on SC62D1933), hence the 0.9 factor. When both exist they must agree.
RF_SIZE_TILES = (1280, 1760, 2240)    # sizing pass at several scales: at a single wrong scale real bubbles score low
RF_SIZE_CONF = 0.70                   # only confident detections that contain tag text are used for sizing
RF_SIZE_MIN = 5                       # at least this many boxes
BOX_TO_DIAMETER = 0.9
SIZE_AGREE_TOL = 0.15                 # Hough vs RF-DETR estimates further apart than this -> trust RF-DETR
DET_CONF = 0.40
BUBBLE_LABELS = {"Instrument_Field", "Instrument_Panel", "Instrument_Aux_Panel"}
NEAR_MM = 10                          # text within this distance of a bubble counts as "near" it
# Auto-orientation: some sheets are drawn rotated on the page (e.g. a landscape P&ID stored as a portrait page).
# Page shape alone is only a hint: it cannot tell 90 from 270 degrees, and some sheets are genuinely portrait.
# Decision: a few text-dense tiles are OCR'd at 0/90/180/270 degrees with RapidOCR's angle classifier OFF (with it on,
# sideways/upside-down text is silently corrected and every rotation scores alike); only the upright rotation reads
# many characters in wide (horizontal) text boxes at high confidence.
AUTO_ORIENT = True
ORIENT_TILES = 4                      # text-dense tiles sampled for the decision
ORIENT_TILE = 1600                    # px
ORIENT_MIN_CONF = 0.9
ORIENT_MARGIN = 3.0                   # winner must read this many times more than the runner-up, else keep as is
# Stacked-text assembly (bubbles the detector missed): limits are multiples of the lines' mean text height,
# so they work for any drawing scale / bubble size.
STACK_DX_H = 1.0                      # max horizontal offset between line centres
STACK_GAP_H = 2.0                     # max vertical gap between consecutive lines
STACK_H_RATIO = 1.6                   # max ratio between the two lines' text heights
STACK_MAX_CHARS = 8                   # bubble lines are short ("111", "PZA", "5163A")
# Prior confidence per source (uncalibrated): text read inside a detected bubble is the most reliable.
SOURCE_PRIOR = {"bubble": 1.0, "stack": 0.75, "text_near_bubble": 0.8, "text": 0.6}
GRAMMAR_FAIL_PENALTY = 0.5            # tags that fail the tag grammar are kept but down-weighted
# Local devices with no DCS signal (ISA-5.1 letter codes) are ignored: they never appear in I/O tag lists.
DROP_LOCAL_DEVICES = True
LOCAL_DEVICE_RE = re.compile(
    r"^(?:HV|VV"          # hand / manual valves
    r"|CK"                # check valves
    r"|[PTL]CV|[A-Z]RV"   # self-actuated regulators (PCV, TCV, PRV); signal-driven valves (XV, FCV, PV, LV) are kept
    r"|[A-Z]SV"           # safety / relief valves (PSV)
    r"|[A-Z]S?G"          # local gauges and sight glasses (PG, TG, LG, FSG)
    r"|[A-Z]O"            # restriction orifices (RO, FO)
    r"|[A-Z]W)$"          # thermowells (TW)
)
TAG_LETTERS_RE = re.compile(r"(?:^|-)([A-Z]{1,5})-?\d")  # letter block of a composed tag, e.g. 111-PG-0774 -> PG

logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("RapidOCR").setLevel(logging.ERROR)
Image.MAX_IMAGE_PIXELS = None


# ---------- page ----------
def load_page(path, page_no=0, dpi=DPI):
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:
            pix = doc[page_no].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.open(path).convert("RGB")


def tile_starts(length, tile, overlap):
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    s = list(range(0, length - tile + 1, stride))
    if s[-1] + tile < length:
        s.append(length - tile)
    return s


def clipped_at_inner_edge(b, x0, y0, tile, W, H, overlap, edge=4):
    """Box (tile coords) touching a tile edge that is not a page edge, and small enough to be whole elsewhere."""
    x1, y1, x2, y2 = b
    small = (x2 - x1) < tile * overlap and (y2 - y1) < tile * overlap
    touches = ((x0 > 0 and x1 <= edge) or (y0 > 0 and y1 <= edge)
               or (x0 + tile < W and x2 >= tile - edge) or (y0 + tile < H and y2 >= tile - edge))
    return small and touches


def nms_ios(boxes, scores, thr=0.6):
    """Greedy NMS on intersection over the smaller box; returns kept indices."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order, keep = np.argsort(-scores), []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(i)
        ix = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        iy = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        order = rest[ix * iy / np.maximum(np.minimum(areas[i], areas[rest]), 1e-6) <= thr]
    return keep


# ---------- 0. orientation ----------
_ORIENT_OCR = []


def detect_orientation(img):
    """Counter-clockwise rotation (0/90/180/270) that makes the page's text read upright, plus the score per rotation.
    Score = characters in wide text boxes read at confidence >= ORIENT_MIN_CONF, over the most ink-dense tiles."""
    if not _ORIENT_OCR:
        _ORIENT_OCR.append(RapidOCR(params={"Global.use_cls": False}))
    ocr = _ORIENT_OCR[0]
    gray = np.asarray(img.convert("L"))
    H, W = gray.shape
    cells = [(x, y) for y in range(0, max(H - ORIENT_TILE, 0) + 1, ORIENT_TILE // 2)
             for x in range(0, max(W - ORIENT_TILE, 0) + 1, ORIENT_TILE // 2)]
    ink = [(gray[y:y + ORIENT_TILE, x:x + ORIENT_TILE] < 128).mean() for x, y in cells]
    # dense but not solid-black tiles: drawing and text, not hatching or filled areas
    order = sorted(range(len(cells)), key=lambda i: -ink[i] if ink[i] < 0.25 else 0)[:ORIENT_TILES]
    arr = np.asarray(img)
    scores = {}
    for k in (0, 90, 180, 270):
        total = 0
        for i in order:
            x, y = cells[i]
            tile = np.ascontiguousarray(np.rot90(arr[y:y + ORIENT_TILE, x:x + ORIENT_TILE], k // 90))  # CCW, like PIL
            r = ocr(tile)
            if r.txts is not None:
                total += sum(len(clean_text(t)) for b, t, sc in zip(r.boxes, r.txts, r.scores)
                             if sc >= ORIENT_MIN_CONF and np.ptp(b[:, 0]) >= 1.5 * np.ptp(b[:, 1]))
        scores[k] = total
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[0], ranked[1]
    if best and scores[best] >= ORIENT_MARGIN * max(scores[second], 1):
        return best, scores
    return 0, scores


# ---------- 1. full-page OCR ----------
def clean_text(t):
    """Uppercase, strip accents and anything but A-Z, 0-9, '-' and single spaces (spaces separate table cells)."""
    t = t.upper().translate(str.maketrans("ÁÀÂÄÉÈÊÍÌÓÒÔÖÚÙÛŚŠ", "AAAAEEEIIOOOOUUUSS"))
    return re.sub(r" +", " ", re.sub(r"[^A-Z0-9 -]", "", t)).strip()


def ocr_page(img, ocr):
    """Tiled OCR at 0° and 90° (vertical labels). Returns fragments {text, score, bbox, rot} in page px."""
    W, H = img.size
    arr = np.asarray(img)
    frags = []
    for rot in (0, 90):
        a = arr if rot == 0 else np.ascontiguousarray(np.rot90(arr))  # rot90 = 90° counter-clockwise
        h, w = a.shape[:2]
        for y0 in tile_starts(h, OCR_TILE, OCR_OVERLAP):
            for x0 in tile_starts(w, OCR_TILE, OCR_OVERLAP):
                r = ocr(a[y0:y0 + OCR_TILE, x0:x0 + OCR_TILE])
                if r.txts is None:
                    continue
                for quad, txt, sc in zip(r.boxes, r.txts, r.scores):
                    words = clean_text(txt)
                    text = words.replace(" ", "")   # bubble lines / stacks are read without spaces
                    if sc < OCR_MIN_SCORE or not text:
                        continue
                    b = [quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()]
                    if clipped_at_inner_edge(b, x0, y0, OCR_TILE, w, h, OCR_OVERLAP):
                        continue
                    bx1, by1, bx2, by2 = b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0
                    if rot == 90:  # rotated (x', y') -> original (W-1-y', x')
                        bx1, by1, bx2, by2 = W - 1 - by2, bx1, W - 1 - by1, bx2
                    frags.append(dict(text=text, words=words, score=float(sc), bbox=[float(bx1), float(by1), float(bx2), float(by2)], rot=rot))
    # the same text seen by two tiles or both rotations: keep the longer, more confident read, preferring the
    # pass whose orientation matches the box (clearly tall box -> rotated read, otherwise upright read)
    def keep_score(f):
        tall = (f["bbox"][3] - f["bbox"][1]) > 1.5 * (f["bbox"][2] - f["bbox"][0])
        return len(f["text"]) + f["score"] + (0.5 if tall == (f["rot"] == 90) else 0)
    keep = nms_ios([f["bbox"] for f in frags], [keep_score(f) for f in frags], thr=0.6)
    return [frags[i] for i in keep]


# ---------- 2. RF-DETR bubbles ----------
def load_rfdetr(path):
    args = vars(torch.load(path, map_location="cpu", weights_only=False)["args"])
    model = RFDETRBase(pretrain_weights=str(path), num_classes=args["num_classes"], resolution=args["resolution"])
    names = {i + 1: n for i, n in enumerate(args["class_names"])}  # Roboflow export: class ids are 1-based
    return model, names


def estimate_bubble_px(img):
    """Median diameter (px) of clean, hollow circles on the page, via Hough + ink checks; None if too few."""
    gray = np.asarray(img.convert("L"))
    ink = cv2.dilate((gray < 160).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    found = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, dp=1, minDist=int(6 * MM), param1=100,
                             param2=40, minRadius=int(HOUGH_MM[0] / 2 * MM), maxRadius=int(HOUGH_MM[1] / 2 * MM))
    if found is None:
        return None

    def ring(x, y, r, n=180):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.clip((x + r * np.cos(t)).astype(int), 0, ink.shape[1] - 1)
        ys = np.clip((y + r * np.sin(t)).astype(int), 0, ink.shape[0] - 1)
        return ink[ys, xs].mean()

    # a bubble outline is almost fully inked, and the band just inside it is mostly empty
    diams = [2 * r for x, y, r in found[0] if ring(x, y, r) >= 0.85 and np.mean([ring(x, y, r * k) for k in (0.88, 0.93)]) <= 0.35]
    return float(np.median(diams)) if len(diams) >= 3 else None


def has_tag_text(box, frags):
    """True if an OCR fragment of >= 2 characters sits inside the box: a tag bubble, not an empty actuator or
    connection circle (those score high at the wrong scale and are about half the size)."""
    x1, y1, x2, y2 = box
    return any(len(f["text"]) >= 2 and x1 <= (f["bbox"][0] + f["bbox"][2]) / 2 <= x2 and y1 <= (f["bbox"][1] + f["bbox"][3]) / 2 <= y2
               for f in frags)


def measure_bubble(img, model, names, frags):
    """Bubble diameter in px, the method used, and a note. Hough first; median of confident RF-DETR boxes that
    contain tag text as fallback and cross-check."""
    hough = estimate_bubble_px(img)
    coarse = [b for t in RF_SIZE_TILES for b in detect_bubbles(img, model, names, t)
              if b["conf"] >= RF_SIZE_CONF and has_tag_text(b["bbox"], frags)]
    sizes = [max(b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1]) for b in coarse]
    rf = BOX_TO_DIAMETER * float(np.median(sizes)) if len(sizes) >= RF_SIZE_MIN else None
    if hough and rf:
        diff = abs(hough - rf) / rf
        if diff <= SIZE_AGREE_TOL:
            return hough, "hough", f"RF-DETR box median agrees ({rf:.0f} px, {diff:.0%} apart, n={len(sizes)})"
        return rf, "rf-detr", f"Hough {hough:.0f} px disagreed by {diff:.0%} with RF-DETR (n={len(sizes)}); using RF-DETR"
    if hough:
        return hough, "hough", f"too few confident RF-DETR boxes to cross-check (n={len(sizes)})"
    if rf:
        return rf, "rf-detr", f"Hough found < 3 clean circles; median of {len(sizes)} text-bearing RF-DETR boxes x {BOX_TO_DIAMETER}"
    return None, "default", f"no Hough circles and only {len(sizes)} confident text-bearing RF-DETR boxes; default tile"


def choose_tile(bubble_px):
    if bubble_px is None:
        return DET_TILE_DEFAULT
    return int(np.clip(round(bubble_px * DET_INPUT / TARGET_BUBBLE_PX / 32) * 32, *DET_TILE_RANGE))


def detect_multiscale(img, model, names, base_tile):
    """Union of detections at several tile sizes around base_tile, de-duplicated (highest confidence wins)."""
    dets = []
    for s in SCALES:
        tile = int(np.clip(round(base_tile * s / 32) * 32, *DET_TILE_RANGE))
        dets += [dict(d, tile=tile) for d in detect_bubbles(img, model, names, tile)]
    keep = nms_ios([d["bbox"] for d in dets], [d["conf"] for d in dets])
    return [dict(dets[i], bubble_id=n) for n, i in enumerate(keep)]


def detect_bubbles(img, model, names, tile, batch=8):
    W, H = img.size
    tiles = []
    for y0 in tile_starts(H, tile, DET_OVERLAP):
        for x0 in tile_starts(W, tile, DET_OVERLAP):
            t = Image.new("RGB", (tile, tile), "white")
            t.paste(img.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H))))
            tiles.append((x0, y0, t))
    dets = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        res = model.predict([t for _, _, t in chunk], threshold=DET_CONF)
        for (x0, y0, _), r in zip(chunk, res if isinstance(res, list) else [res]):
            for b, s, k in zip(r.xyxy, r.confidence, r.class_id):
                label = names.get(int(k), "")
                if label in BUBBLE_LABELS and not clipped_at_inner_edge(b, x0, y0, tile, W, H, DET_OVERLAP):
                    dets.append(dict(bbox=[float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)],
                                     conf=float(s), label=label))
    # one bubble may be detected under two bubble labels or by two tiles
    keep = nms_ios([d["bbox"] for d in dets], [d["conf"] for d in dets])
    return [dict(dets[i], bubble_id=n) for n, i in enumerate(keep)]


# ---------- 3. attach text to bubbles ----------
def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def attach(frags, bubbles):
    """Give every fragment its nearest bubble, the distance to it (mm, 0 if inside) and whether it is inside."""
    if not bubbles:
        for f in frags:
            f.update(bubble_id=None, dist_mm=None, inside=False)
        return
    bb = np.array([b["bbox"] for b in bubbles])
    bc = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1)
    rx, ry = (bb[:, 2] - bb[:, 0]) / 2, (bb[:, 3] - bb[:, 1]) / 2
    for f in frags:
        cx, cy = centre(f["bbox"])
        e = ((cx - bc[:, 0]) / rx) ** 2 + ((cy - bc[:, 1]) / ry) ** 2   # <= 1 inside the inscribed ellipse
        dx = np.clip(np.maximum(bb[:, 0] - cx, cx - bb[:, 2]), 0, None)  # gap to the bubble box
        dy = np.clip(np.maximum(bb[:, 1] - cy, cy - bb[:, 3]), 0, None)
        dist = np.where(e <= 1, 0.0, np.hypot(dx, dy))
        j = int(np.argmin(dist))
        f.update(bubble_id=bubbles[j]["bubble_id"], dist_mm=round(float(dist[j]) / MM, 1), inside=bool(e[j] <= 1))


# ---------- 4. tags ----------
LETTERS_RE = re.compile(r"^[A-Z]{2,5}$")                 # ISA functional identifier (1-letter codes are equipment)
NUMBER_RE = re.compile(r"^\d{3,5}[A-Z]?(-\d{1,2})?$")    # loop number, optional suffix and -N
AREA_RE = re.compile(r"^\d{1,4}$")
INLINE_RE = re.compile(r"(?<![A-Z0-9])(?:(\d{1,4})-?)?([A-Z]{2,5})-?(\d{3,5}[A-Z]?)(?![0-9])((?:-\d{1,2})?)(?![A-Z0-9])")
DRAWING_NO_RE = re.compile(r"\d{3,}-\d{3,}-[A-Z]{2}-\d{4}|[A-Z]\d{3}-\d{5}")  # e.g. L001-11100-PX-2365-1019
NOT_TAG_WORDS = ("NOTE", "UNIT", "PID", "DWG", "REV", "SHEET", "ITEM", "LINE", "SIZE", "CLASS", "HOLD", "TYPE")


def bubble_tag(lines):
    """Bubble lines (top to bottom) -> (tag, grammar_ok). Handles 'area / letters / number' and 'letters / number'."""
    if len(lines) == 3 and AREA_RE.match(lines[0]):
        area, letters, number = lines
        return f"{area}-{letters}-{number}", bool(LETTERS_RE.match(letters) and NUMBER_RE.match(number))
    if len(lines) == 2:
        return f"{lines[0]}-{lines[1]}", bool(LETTERS_RE.match(lines[0]) and NUMBER_RE.match(lines[1]))
    return "-".join(lines), False


def find_stacks(frags):
    """Chains of 2-3 short, horizontal fragments stacked vertically with aligned centres, outside detected bubbles.
    Returns lists of fragments, top to bottom; each fragment is used at most once."""
    def horizontal(f):  # upright read, or a box that is not clearly tall (short lines like "TI" are near-square)
        return f["rot"] == 0 or (f["bbox"][3] - f["bbox"][1]) <= 1.5 * (f["bbox"][2] - f["bbox"][0])
    cand = sorted((f for f in frags if not f["inside"] and horizontal(f) and len(f["text"]) <= STACK_MAX_CHARS),
                  key=lambda f: f["bbox"][1])
    used, stacks = set(), []

    def below(top):
        """Best fragment directly under `top` that satisfies the stacking limits, or None."""
        tx1, ty1, tx2, ty2 = top["bbox"]
        th, tcx = ty2 - ty1, (tx1 + tx2) / 2
        best = None
        for g in cand:
            if id(g) in used or g is top:
                continue
            gx1, gy1, gx2, gy2 = g["bbox"]
            gh = gy2 - gy1
            h = (th + gh) / 2
            gap = gy1 - ty2
            if gap < -0.3 * h or gap > STACK_GAP_H * h:
                continue
            if abs((gx1 + gx2) / 2 - tcx) > STACK_DX_H * h or max(th, gh) / max(min(th, gh), 1) > STACK_H_RATIO:
                continue
            if best is None or gap < best[0]:
                best = (gap, g)
        return best[1] if best else None

    for f in cand:
        if id(f) in used:
            continue
        chain = [f]
        while len(chain) < 3 and (nxt := below(chain[-1])) is not None:
            chain.append(nxt)
        if len(chain) < 2:
            continue
        # the chain can run on into unrelated text (e.g. a pipe size under the bubble): keep the first window
        # that reads as a tag, "area / letters / number" before "letters / number"
        for i, j in ((0, 3), (0, 2), (1, 3)):
            win = [c["text"] for c in chain[i:j]]
            if len(win) == j - i and LETTERS_RE.match(win[-2]) and NUMBER_RE.match(win[-1]) \
                    and (len(win) == 2 or AREA_RE.match(win[0])):
                used.update(id(c) for c in chain[i:j])
                stacks.append(chain[i:j])
                break
    return stacks


def build_tags(frags, bubbles):
    tags = []
    by_id = {b["bubble_id"]: b for b in bubbles}
    # 4c (run first so stacked fragments are not reused): bubbles the detector missed, read from stacked text
    stacked = set()
    for chain in find_stacks(frags):
        lines = [c["text"] for c in chain]
        tag, ok = bubble_tag(lines)
        ocr_conf = min(c["score"] for c in chain)
        box = [min(c["bbox"][0] for c in chain), chain[0]["bbox"][1], max(c["bbox"][2] for c in chain), chain[-1]["bbox"][3]]
        tags.append(dict(tag=tag, source="stack", confidence=round(SOURCE_PRIOR["stack"] * ocr_conf * (1 if ok else GRAMMAR_FAIL_PENALTY), 3),
                         grammar_ok=ok, lines=" / ".join(lines), ocr_conf=round(ocr_conf, 3), det_conf=None,
                         bubble_id=None, bubble_label=None, dist_mm=min(c["dist_mm"] for c in chain) if chain[0]["dist_mm"] is not None else None,
                         bbox=[round(v) for v in box]))
        stacked.update(id(c) for c in chain)
    # 4a. one tag per bubble from the fragments inside it, read top to bottom
    for bid, b in by_id.items():
        inside = sorted((f for f in frags if f["inside"] and f["bubble_id"] == bid), key=lambda f: centre(f["bbox"])[1])
        if not inside:
            continue
        rows = []  # fragments on the same line (vertical centres within half a line height) are joined left to right
        for f in inside:
            cy, h = centre(f["bbox"])[1], f["bbox"][3] - f["bbox"][1]
            if rows and abs(cy - rows[-1]["cy"]) < h / 2:
                rows[-1]["parts"].append(f)
            else:
                rows.append(dict(cy=cy, parts=[f]))
        lines = ["".join(p["text"] for p in sorted(r["parts"], key=lambda p: p["bbox"][0])) for r in rows]
        tag, ok = bubble_tag(lines)
        ocr_conf = min(f["score"] for f in inside)
        conf = SOURCE_PRIOR["bubble"] * b["conf"] * ocr_conf * (1 if ok else GRAMMAR_FAIL_PENALTY)
        tags.append(dict(tag=tag, source="bubble", confidence=round(conf, 3), grammar_ok=ok, lines=" / ".join(lines),
                         ocr_conf=round(ocr_conf, 3), det_conf=round(b["conf"], 3), bubble_id=bid, bubble_label=b["label"],
                         dist_mm=0.0, bbox=[round(v) for v in b["bbox"]]))
    # 4b. tags written as plain text (inline labels, tag boxes, tables), linked to the nearest bubble if close
    for f in frags:
        if f["inside"] or id(f) in stacked:
            continue
        # one OCR box can hold a whole table row ("111XIA-0791-TX 111XIA-0792-TX ..."): match word by word
        for word in f["words"].split():
            if word.count("-") > 4 or DRAWING_NO_RE.search(word):  # line / drawing numbers
                continue
            for m in INLINE_RE.finditer(word):
                area, letters, number, sub = m.groups()
                if letters.startswith(NOT_TAG_WORDS):  # NOTE 318, UNIT U1100, PID L001-...
                    continue
                tag = "-".join(p for p in (area, letters, number + (sub or "")) if p)
                ok = bool(LETTERS_RE.match(letters) and NUMBER_RE.match(number + (sub or "")))
                near = f["dist_mm"] is not None and f["dist_mm"] <= NEAR_MM
                source = "text_near_bubble" if near else "text"
                conf = SOURCE_PRIOR[source] * f["score"] * (1 if ok else GRAMMAR_FAIL_PENALTY)
                tags.append(dict(tag=tag, source=source, confidence=round(conf, 3), grammar_ok=ok, lines=f["words"],
                                 ocr_conf=round(f["score"], 3), det_conf=None, bubble_id=f["bubble_id"] if near else None,
                                 bubble_label=by_id[f["bubble_id"]]["label"] if near else None,
                                 dist_mm=f["dist_mm"], bbox=[round(v) for v in f["bbox"]]))
    if DROP_LOCAL_DEVICES:
        def is_local(t):
            m = TAG_LETTERS_RE.search(t["tag"])
            return bool(m and LOCAL_DEVICE_RE.match(m.group(1)))
        dropped = [t for t in tags if is_local(t)]
        tags = [t for t in tags if not is_local(t)]
        codes = sorted({TAG_LETTERS_RE.search(t["tag"]).group(1) for t in dropped})
        print(f"  ignored {len(dropped)} local devices with no DCS signal: {', '.join(codes) or '-'}")
    return sorted(tags, key=lambda t: -t["confidence"])


# ---------- 5. draw everything on the full page ----------
COLORS = {"bubble": (220, 30, 30), "stack": (150, 40, 190), "text_near_bubble": (240, 140, 0), "text": (30, 90, 220)}  # RGB per source


def draw_overlay(img, bubbles, tags):
    vis = np.array(img).copy()
    th = max(2, int(0.3 * MM))                 # line thickness ~0.3 mm
    fs = 0.9 * MM / 12                         # label font ~1.8 mm tall
    for b in bubbles:                          # every detected bubble, thin red outline
        x1, y1, x2, y2 = map(int, b["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS["bubble"], th)
    for t in tags:
        c = COLORS[t["source"]] if t["grammar_ok"] else (150, 150, 150)  # grey = failed tag grammar
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, th)
        if t["source"] == "text_near_bubble" and t["bubble_id"] is not None:  # link text to its bubble
            bx1, by1, bx2, by2 = next(b["bbox"] for b in bubbles if b["bubble_id"] == t["bubble_id"])
            cv2.line(vis, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (int((bx1 + bx2) / 2), int((by1 + by2) / 2)), c, th)
        label = f"{t['tag']} {t['confidence']:.2f}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ly = y1 - th if y1 - tht - th > 0 else y2 + tht + th
        cv2.rectangle(vis, (x1, ly - tht - th), (x1 + tw, ly + th), (255, 255, 255), -1)   # white label background
        cv2.putText(vis, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    # legend, top-left
    for i, (name, c) in enumerate(list(COLORS.items()) + [("grammar fail", (150, 150, 150))]):
        y = int((8 + 6 * i) * MM)
        cv2.rectangle(vis, (int(5 * MM), y - int(4 * MM)), (int(9 * MM), y), c, -1)
        cv2.putText(vis, name, (int(11 * MM), y), cv2.FONT_HERSHEY_SIMPLEX, 2 * fs, c, 2 * th, cv2.LINE_AA)
    return Image.fromarray(vis)


