"""Exact text from the PDF itself, in the same form as OCR fragments (dict(text, bbox) in rendered 300 dpi pixels).
Two sources, read from the file, nothing inferred (see batch/survey_pdfs.py for which pages have which):
  text layer   real characters (CAD exports with fonts; LNG Canada sheets). PyMuPDF lines are cut where the gap
               between two words is wider than GAP_H x the text height, like an OCR text line; words are joined
               without spaces as RapidOCR returns them ('FLASH GAS COMPRESSOR' -> 'FLASHGASCOMPRESSOR')
  SHX comments AutoCAD's hidden 'AutoCAD SHX Text' annotations: the exact string of every stroke-drawn text entity
Text gets OCR's character conventions (ocr_style) so the finders read both sources alike.
Coordinates follow the rendered page: the PDF's own /Rotate (page.rotation_matrix), then phase 1's auto-rotation.
merge(exact, ocr) keeps every exact fragment and only the OCR fragments that no exact fragment covers (text inside an
embedded image, or a page with no text layer at all)."""
import re
import pymupdf

MIN_WORDS = 100        # fewer real words than this: the page has no usable text layer (border labels only)
GAP_H = 0.8            # a gap wider than 0.8 x the text height starts a new fragment (one space is ~0.3)
DPI = 300
# RapidOCR does not return these characters, and the description finders are written for OCR text: exact text gets
# the same form ('NOTE 4.1' -> 'NOTE41', 'TITLE/SERVICE:' -> 'TITLESERVICE', 'NOTE 6,8' -> 'NOTE68')
OCR_DROPS = re.compile(r"[.,:;/\\\s]")


def ocr_style(t):
    t = OCR_DROPS.sub("", t.upper())
    return re.sub(r"^\(?(\d{1,3})\)", r"\1", t)        # list numbering '4.1)' / '(3)' -> '41' / '3'


def _rot(b, autorot, W, H):
    """Box after PIL image.rotate(autorot, expand=True) (counter-clockwise) of a W x H image."""
    x0, y0, x1, y1 = b
    if autorot == 90:
        return [y0, W - x1, y1, W - x0]
    if autorot == 180:
        return [W - x1, H - y1, W - x0, H - y0]
    if autorot == 270:
        return [H - y1, x0, H - y0, x1]
    return list(b)


def exact_frags(pdf, page_no=0, autorot=0, dpi=DPI):
    """(frags, source) with source 'text layer' / 'SHX text' / 'text layer + SHX text', or (None, reason)."""
    pg = pymupdf.open(pdf)[page_no]
    M, k = pg.rotation_matrix, dpi / 72
    W, H = pg.rect.width * k, pg.rect.height * k
    out, src = [], []
    words = pg.get_text("words")
    if len(words) >= MIN_WORDS:
        lines = {}
        for w in words:
            lines.setdefault((w[5], w[6]), []).append(w)
        for ws in lines.values():
            ws.sort(key=lambda w: w[7])
            cur = [ws[0]]
            for a, b in zip(ws, ws[1:]):
                ra, rb = pymupdf.Rect(a[:4]) * M, pymupdf.Rect(b[:4]) * M
                h = min(ra.height, ra.width) if ra.height > 2 * ra.width else ra.height
                gap = max(rb.x0 - ra.x1, rb.y0 - ra.y1, ra.x0 - rb.x1, ra.y0 - rb.y1)
                lead_no = len(cur) == 1 and re.fullmatch(r"\(?\d{1,2}(\.\d{1,2})*[.)]?", cur[0][4])
                if gap > GAP_H * max(h, 1) and not lead_no:     # a list number stays with its text ('4.1  MAIN ...')
                    out.append(cur); cur = [b]
                else:
                    cur.append(b)
            out.append(cur)
        out = [dict(text=ocr_style("".join(w[4] for w in ws)),
                    bbox=pymupdf.Rect(min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)))
               for ws in out]
        src.append("text layer")
    shx = [a for a in (pg.annots() or []) if a.info.get("title") == "AutoCAD SHX Text"]
    if shx:
        out += [dict(text=ocr_style(a.info.get("content", "")), bbox=a.rect) for a in shx if a.info.get("content")]
        src.append("SHX text")
    if not out:
        return None, f"no text layer ({len(words)} words) and no SHX text"
    frags = []
    for f in out:
        r = f["bbox"] * M if not f["bbox"].is_empty else f["bbox"]
        b = [r.x0 * k, r.y0 * k, r.x1 * k, r.y1 * k]
        frags.append(dict(text=f["text"], bbox=[round(v) for v in _rot(b, autorot, W, H)], source="pdf"))
    return join_numbers([f for f in frags if f["text"]]), " + ".join(src)


def join_numbers(frags, reach_h=8):
    """A fragment that is only a list number ('41', '5', '(3)') joins the text fragment starting just right of it
    on the same baseline, as one OCR text line would read ('4.1' + 'COMPRESSORVALVES' -> '41COMPRESSORVALVES')."""
    num = [f for f in frags if re.fullmatch(r"\(?\d{1,3}\)?", f["text"])]
    gone = set()
    for n in num:
        b = n["bbox"]; h = max(1, b[3] - b[1]); cy = (b[1] + b[3]) / 2
        cand = [f for f in frags if f is not n and id(f) not in gone and not re.fullmatch(r"\(?\d{1,3}\)?", f["text"])
                and abs((f["bbox"][1] + f["bbox"][3]) / 2 - cy) < 0.5 * h and 0 <= f["bbox"][0] - b[2] < reach_h * h]
        if cand:
            t = min(cand, key=lambda f: f["bbox"][0] - b[2])
            t["text"] = n["text"].strip("()") + t["text"]
            t["bbox"] = [min(b[0], t["bbox"][0]), min(b[1], t["bbox"][1]), max(b[2], t["bbox"][2]), max(b[3], t["bbox"][3])]
            gone.add(id(n))
    return [f for f in frags if id(f) not in gone]


def merge(exact, ocr, cover=0.3):
    """Exact fragments, plus the OCR fragments that exact text does not cover (overlap < cover of the OCR box)."""
    def ov(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    keep = []
    for o in ocr:
        area = max(1, (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
        if sum(ov(o["bbox"], e["bbox"]) for e in exact) < cover * area:
            keep.append(o)
    return exact + keep
