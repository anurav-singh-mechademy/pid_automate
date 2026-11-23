"""Offline test of the model image: run inside the container with networking off and a read-only filesystem.

  docker run --rm --platform linux/arm64 --cpus 6 --network none --read-only --tmpfs /tmp --user 1000:1000 \
    -e PID_ALLOW_FILE=1 -v <pdf>:/samples/sample.pdf:ro -v $PWD/deploy:/test:ro --entrypoint python pid-models \
    /test/test_local.py --page 3

Renders one page as the pipeline does (300 DPI), calls handler.detect and handler.ocr the way a function URL would,
then checks the APIs give the same result as the local pipeline (tag_pipeline.detect_bubbles and ocr_page) once the
caller-side filters are applied to the API output. Exits non-zero on any mismatch."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import handler
import tag_pipeline as TP


def call(fn, req):
    out = fn(dict(requestContext={}, body=json.dumps(req), isBase64Encoded=False))
    body = json.loads(out["body"])
    if out["statusCode"] != 200:
        sys.exit(f"{fn.__name__} -> {out['statusCode']}: {body}")
    return body, len(out["body"])


def same_boxes(a, b, tol=0.5):
    a, b = sorted(map(tuple, a)), sorted(map(tuple, b))
    return len(a) == len(b) and all(
        np.allclose(x[:4], y[:4], atol=tol) and x[4:] == y[4:] for x, y in zip(a, b)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="/samples/sample.pdf")
    ap.add_argument("--page", type=int, default=0)
    args = ap.parse_args()
    ok = True

    img = TP.load_page(Path(args.pdf), args.page)
    png = "/tmp/page.png"
    img.save(png)
    print(f"page {args.page}: {img.size[0]} x {img.size[1]} px")
    tile = TP.choose_tile(TP.estimate_bubble_px(img))
    uri = f"file://{png}"

    # detect: API at one tile size, then detect_bubbles' own filters -> must equal detect_bubbles
    t = time.time()
    det, size = call(
        handler.detect, dict(image_s3=uri, tiles=[tile], conf_threshold=TP.DET_CONF)
    )
    print(
        f"detect: tile {tile}, {len(det['detections'])} detections, {size / 1e3:.0f} KB, "
        f"{time.time() - t:.1f}s wall, timing {det['timing_ms']}"
    )
    bub = [d for d in det["detections"] if d["cls"] in TP.BUBBLE_LABELS]
    keep = TP.nms_ios([d["bbox"] for d in bub], [d["conf"] for d in bub]) if bub else []
    model, names = handler.detector()
    ref = TP.detect_bubbles(img, model, names, tile)
    match = same_boxes(
        [d["bbox"] + [d["cls"]] for d in (bub[i] for i in keep)],
        [d["bbox"] + [d["label"]] for d in ref],
    )
    print(
        f"  bubbles: API {len(keep)} vs pipeline {len(ref)} -> {'MATCH' if match else 'MISMATCH'}"
    )
    ok &= match

    # ocr page mode: API, then ocr_page's own cleaning, score cutoff and de-duplication -> must equal ocr_page
    t = time.time()
    ocr, size = call(handler.ocr, dict(image_s3=uri, mode="page", rotations=[0, 90]))
    print(
        f"ocr page: {len(ocr['lines'])} lines, {size / 1e3:.0f} KB, {time.time() - t:.1f}s wall, timing {ocr['timing_ms']}"
    )
    frags = []
    for ln in ocr["lines"]:
        words = TP.clean_text(ln["text"])
        text = words.replace(" ", "")
        if ln["score"] < TP.OCR_MIN_SCORE or not text:
            continue
        q = np.asarray(ln["quad"])
        frags.append(
            dict(
                text=text,
                score=ln["score"],
                rot=ln["rot"],
                bbox=[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()],
            )
        )

    def keep_score(f):  # ocr_page's
        tall = (f["bbox"][3] - f["bbox"][1]) > 1.5 * (f["bbox"][2] - f["bbox"][0])
        return len(f["text"]) + f["score"] + (0.5 if tall == (f["rot"] == 90) else 0)

    kept = [
        frags[i]
        for i in TP.nms_ios(
            [f["bbox"] for f in frags], [keep_score(f) for f in frags], thr=0.6
        )
    ]
    ref = TP.ocr_page(img, handler.ocr_engine(True))
    match = same_boxes(
        [f["bbox"] + [f["text"]] for f in kept], [f["bbox"] + [f["text"]] for f in ref]
    )
    print(
        f"  fragments: API {len(kept)} vs pipeline {len(ref)} -> {'MATCH' if match else 'MISMATCH'}"
    )
    ok &= match

    # ocr regions mode: the orientation check's shape (4 areas x 4 rotations) in one call; smoke test only
    W, H = img.size
    s = TP.ORIENT_TILE
    regions = [
        dict(id=f"r{i}-{k}", bbox=[x, y, min(x + s, W), min(y + s, H)], rotate=k)
        for i, (x, y) in enumerate([(0, 0), (W // 2, 0), (0, H // 2), (W // 2, H // 2)])
        for k in (0, 90, 180, 270)
    ]
    t = time.time()
    reg, size = call(
        handler.ocr, dict(image_s3=uri, mode="regions", regions=regions, use_cls=False)
    )
    inside = all(
        0 <= x <= W and 0 <= y <= H for ln in reg["lines"] for x, y in ln["quad"]
    )
    print(
        f"ocr regions: {len(regions)} regions, {len(reg['lines'])} lines, {time.time() - t:.1f}s wall, "
        f"quads on page: {inside}"
    )
    ok &= inside

    # bad request -> 400
    out = handler.detect(
        dict(requestContext={}, body=json.dumps(dict(image_s3=uri, tiles=[10])))
    )
    print(f"bad request: {out['statusCode']} {json.loads(out['body'])}")
    ok &= out["statusCode"] == 400

    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
