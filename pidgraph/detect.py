"""Every RF-DETR symbol on a sheet (valves, fittings, flow arrows, bubbles ...), not only the instrument bubbles.

batch/tag_pipeline.detect_bubbles keeps only the bubble labels; this wrapper runs the same sliding window (same tile
size rule, same model, same edge clipping) and keeps all 32 classes. The tile is sized from the sheet's bubbles
(phase-1 bubble boxes, else Hough circles) exactly as phase 1 does, run at the phase-1 scales and merged per class
group. Results are cached per sheet as JSON under descgen_lab/cache/symbols/ (delete the file to re-detect)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "descgen_lab" / "cache" / "symbols"
WEIGHTS = ROOT / "models" / "rfdetr_pid.pth"
DET_CONF = 0.35            # all classes: a little below phase 1's bubble threshold, every box keeps its score

# class -> group: what the graph does with it (rules data, not logic)
GROUPS = {
    "valve": ("Gate_Valve", "Ball_Valve", "Globe_valve_NO", "Gate_valve_NO", "Butterfly_valve", "Plug valve",
              "Check_valve", "Diaphragm_valve", "Needle_valve", "Half_Filled_Gate_Valve", "Gate_Valve_NC",
              "Globle_valve_NC", "Control_Valve", "Rotary_Valve", "Ball_valve_NC"),
    "fitting": ("Paddle_blind", "Spectacle_blind_Closed", "Spectacle_blind_Open", "Reducer", "Flange_or_Nozzle",
                "Rupture_disk", "sight_glass"),
    "arrow": ("Flow_Arrow",),
    "bubble": ("Instrument_Field", "Instrument_Panel", "Instrument_Aux_Panel"),
    "other": ("Pipe_Insulation_or_Tracing", "box"),
}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}

_MODEL = []


def _model():
    if not _MODEL:
        from tag_pipeline import load_rfdetr
        _MODEL.append(load_rfdetr(WEIGHTS))
    return _MODEL[0]


def _window(img, model, names, tile, batch=8):
    """Sliding window at one tile size, all classes (tag_pipeline.detect_bubbles without its label filter)."""
    from tag_pipeline import tile_starts, clipped_at_inner_edge, DET_OVERLAP
    W, H = img.size
    tiles = []
    for y0 in tile_starts(H, tile, DET_OVERLAP):
        for x0 in tile_starts(W, tile, DET_OVERLAP):
            t = Image.new("RGB", (tile, tile), "white")
            t.paste(img.crop((x0, y0, min(x0 + tile, W), min(y0 + tile, H))))
            tiles.append((x0, y0, t))
    out = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        res = model.predict([t for _, _, t in chunk], threshold=DET_CONF)
        for (x0, y0, _), r in zip(chunk, res if isinstance(res, list) else [res]):
            for b, s, k in zip(r.xyxy, r.confidence, r.class_id):
                if clipped_at_inner_edge(b, x0, y0, tile, W, H, DET_OVERLAP):
                    continue
                label = names.get(int(k), "")
                out.append(dict(bbox=[float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)],
                                conf=float(s), cls=label, group=GROUP_OF.get(label, "other"), tile=tile))
    return out


def base_tile(img, bubbles):
    """Phase-1 tile rule: bubble diameter (median phase-1 bubble box x 0.9, else Hough) -> tile size."""
    from tag_pipeline import choose_tile, estimate_bubble_px, BOX_TO_DIAMETER
    sizes = [max(b[2] - b[0], b[3] - b[1]) for b in (bubbles or {}).values()]
    px = BOX_TO_DIAMETER * float(np.median(sizes)) if len(sizes) >= 3 else estimate_bubble_px(img)
    return choose_tile(px), px


def detect_symbols(name, img, bubbles=None, refresh=False):
    """All RF-DETR detections of a sheet: [dict(bbox, conf, cls, group, tile)], cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{name}.json"
    if cf.exists() and not refresh:
        return json.load(open(cf))["detections"]
    from tag_pipeline import SCALES, DET_TILE_RANGE, nms_ios
    model, names = _model()
    tile0, px = base_tile(img, bubbles)
    dets = []
    for s in SCALES:
        dets += _window(img, model, names, int(np.clip(round(tile0 * s / 32) * 32, *DET_TILE_RANGE)))
    keep = []
    for g in GROUPS:                       # de-duplicate within a group (one valve seen as two valve classes)
        grp = [d for d in dets if d["group"] == g]
        keep += [grp[i] for i in nms_ios([d["bbox"] for d in grp], [d["conf"] for d in grp])] if grp else []
    json.dump(dict(sheet=name, base_tile=tile0, bubble_px=px, detections=keep), open(cf, "w"))
    return keep
