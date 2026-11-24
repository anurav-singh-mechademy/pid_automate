"""Offline test of the model image: runs inside the container with networking off and a read-only filesystem.
Start it with deploy/test.sh.

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


def check_prepare(pdf, page):
    """The ocr function's prepare and pages modes vs the pipeline's own functions (render_page, batch_tags' fingerprint,
    detect_orientation, estimate_bubble_px, pipeline_viewer's JPEG), upright and on a sideways copy of the page."""
    import hashlib
    import io

    from PIL import Image
    from pdfrender import page_count, render_page

    ok = True
    n = page_count(pdf)
    t = time.time()
    pages, _ = call(handler.ocr, dict(mode="pages", pdf_s3=f"file://{pdf}"))
    want = {str(i): hashlib.md5(render_page(pdf, i, 72, gray=True).tobytes()).hexdigest() for i in range(n)}
    match = pages["page_count"] == n and pages["fingerprints"] == want
    print(f"prepare pages: {n} pages, {time.time() - t:.1f}s -> {'MATCH' if match else 'MISMATCH'}")
    ok &= match

    # a sideways copy: the same page saved with /Rotate 90, as a scanned-sideways sheet arrives
    import pypdfium2 as pdfium
    side = "/tmp/sideways.pdf"
    doc = pdfium.PdfDocument(pdf)
    keep = pdfium.PdfDocument.new(); keep.import_pages(doc, [page]); keep[0].set_rotation((keep[0].get_rotation() + 90) % 360)
    keep.save(side); keep.close(); doc.close()

    for label, src, pg in (("upright", pdf, page), ("sideways", side, 0)):
        t = time.time()
        out, size = call(handler.ocr, dict(mode="prepare", pdf_s3=f"file://{src}", page=pg,
                                               out_key=f"file:///tmp/prepare/{label}", render_md5=True))
        img = TP.load_page(Path(src), pg)
        rot, scores = TP.detect_orientation(img)
        turned = img.rotate(rot, expand=True) if rot else img
        px = TP.estimate_bubble_px(turned)
        rescored = {k: 0 for k in (0, 90, 180, 270)}                      # the worker re-scores orient_lines
        for ln in out["orient_lines"]:
            q = np.asarray(ln["quad"]); dx, dy = np.ptp(q[:, 0]), np.ptp(q[:, 1])
            wide = dy >= 1.5 * dx if ln["rot"] in (90, 270) else dx >= 1.5 * dy
            if ln["score"] >= TP.ORIENT_MIN_CONF and wide:
                rescored[ln["rot"]] += len(TP.clean_text(ln["text"]))
        png = Image.open(out["page_s3"][7:]); jpg = Image.open(out["jpeg_s3"][7:])
        k = min(1.0, 4000 / max(turned.size))
        checks = {
            "render md5": out["render_md5"] == hashlib.md5(img.tobytes()).hexdigest(),
            "rotation": out["rotation"] == rot,
            "orientation scores": out["orient_scores"] == {str(a): b for a, b in scores.items()},
            "re-scored lines": rescored == scores,
            "page png pixels": png.size == turned.size and np.array_equal(np.asarray(png.convert("RGB")), np.asarray(turned)),
            "size": out["size"] == list(turned.size),
            "hough": out["bubble_px"] == px and out["hough_ok"] == TP.LAST_SIZING["hough_ok"]
                     and out["hough_rejected"] == TP.LAST_SIZING["hough_rejected"],
            "viewer jpeg size": jpg.size == ((round(turned.size[0] * k), round(turned.size[1] * k)) if k < 1 else turned.size),
        }
        bad = [c for c, v in checks.items() if not v]
        print(f"prepare {label}: rotation {out['rotation']} (pipeline {rot}), bubble {out['bubble_px']} (pipeline {px}), "
              f"{len(out['hough_ok'])} clean circles, {size / 1e3:.0f} KB, {time.time() - t:.1f}s wall, timing {out['timing_ms']}"
              f" -> {'MATCH' if not bad else 'MISMATCH ' + ', '.join(bad)}")
        ok &= not bad
    ok &= out["rotation"] != 0                                             # the sideways copy must be turned

    # given rotation: no orientation reads
    out, _ = call(handler.ocr, dict(mode="prepare", pdf_s3=f"file://{pdf}", page=page, rotation=0,
                                        out_key="file:///tmp/prepare/given"))
    given = out["rotation"] == 0 and out["orient_lines"] == [] and out["orient_scores"] == {}
    print(f"prepare given rotation: {'OK' if given else 'WRONG'}")
    ok &= given

    # errors: page out of range -> 400, missing PDF -> 404, unknown parameter -> 400
    for req, want_status in ((dict(mode="prepare", pdf_s3=f"file://{pdf}", page=n, out_key="file:///tmp/prepare/x"), 400),
                             (dict(mode="pages", pdf_s3="file:///tmp/no-such.pdf"), 404),
                             (dict(mode="prepare", pdf_s3=f"file://{pdf}", page=page, out_key="file:///tmp/prepare/x",
                                   hough=dict(bogus=1)), 400)):
        r = handler.ocr(dict(requestContext={}, body=json.dumps(req)))
        print(f"prepare error: {r['statusCode']} (want {want_status}) {json.loads(r['body'])}")
        ok &= r["statusCode"] == want_status
    return ok


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

    ok &= check_prepare(args.pdf, args.page)

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
