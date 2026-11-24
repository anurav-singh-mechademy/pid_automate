"""Exact text from the PDF itself, in the same form as OCR fragments (dict(text, bbox) in rendered 300 dpi pixels).
Two sources, read from the file, nothing inferred (see batch/survey_pdfs.py for which pages have which):
  text layer   real characters (CAD exports with fonts; LNG Canada sheets), read with PDFium (pypdfium2). Characters
               become words (split at spaces, at gaps along the writing direction, and at changes of direction;
               a glyph drawn again on top of itself, as fake bold, counts once), words become lines (same direction
               and baseline), and lines are cut where the gap between two words is wider than GAP_H x the text
               height, like an OCR text line; words are joined without spaces as RapidOCR returns them
               ('FLASH GAS COMPRESSOR' -> 'FLASHGASCOMPRESSOR')
  SHX comments AutoCAD's hidden 'AutoCAD SHX Text' annotations: the exact string of every stroke-drawn text entity
Text gets OCR's character conventions (ocr_style) so the finders read both sources alike.
Coordinates follow the rendered page: the PDF's own /Rotate, then phase 1's auto-rotation.
merge(exact, ocr) keeps every exact fragment and only the OCR fragments that no exact fragment covers (text inside an
embedded image, or a page with no text layer at all)."""
import ctypes
import re

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

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


def _display(page):
    """PDF user space (y up) -> displayed page points (y down, /Rotate applied), and the displayed size."""
    x0, y0, x1, y1 = page.get_cropbox()
    w, h = x1 - x0, y1 - y0
    r = page.get_rotation()

    def f(x, y):
        u, v = x - x0, y1 - y                             # unrotated, top-left origin
        if r == 90:
            return h - v, u
        if r == 180:
            return w - u, h - v
        if r == 270:
            return v, w - u
        return u, v
    return f, ((h, w) if r in (90, 270) else (w, h))


DIRS = {0: (1, 0), 90: (0, -1), 180: (-1, 0), 270: (0, 1)}   # writing direction on the displayed page


def _words(tp, f):
    """[(x0, y0, x1, y1, text, direction)] in displayed points; direction 0/90/180/270, None for one character."""
    words, cur, seen, d = [], [], set(), None

    def flush():
        if cur:
            words.append((min(c[1] for c in cur), min(c[2] for c in cur), max(c[3] for c in cur),
                          max(c[4] for c in cur), "".join(c[0] for c in cur), d))
            cur.clear()
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or ch.isspace():
            flush(); d = None; continue
        l, b, r, t = tp.get_charbox(i, loose=True)      # font-height box: the same height for every glyph of a line
        (ax, ay), (bx, by) = f(l, b), f(r, t)
        x0, y0, x1, y1 = min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)
        k = (ch, round(x0), round(y0))                    # the same glyph drawn again on top (fake bold): once
        if any((ch, k[1] + dx, k[2] + dy) in seen for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            continue
        seen.add(k)
        if cur:
            p = cur[-1]
            cx, cy = (x0 + x1 - p[1] - p[3]) / 2, (y0 + y1 - p[2] - p[4]) / 2
            step = (0 if cx > 0 else 180) if abs(cx) >= abs(cy) else (270 if cy > 0 else 90)
            dd = d if d is not None else step
            ux, uy = DIRS[dd]
            along = abs((p[3] - p[1]) * ux + (p[4] - p[2]) * uy) / 2 + abs((x1 - x0) * ux + (y1 - y0) * uy) / 2
            hgt = max(abs((p[3] - p[1]) * uy) + abs((p[4] - p[2]) * ux), 1)   # text height across the line
            gap = cx * ux + cy * uy - along
            if step != dd or gap > 0.25 * hgt or abs(cx * uy - cy * ux) > 0.5 * hgt:
                flush(); d = None                         # a gap, a step sideways or back: a new word
            else:
                d = dd
        cur.append((ch, x0, y0, x1, y1))
    flush()
    return words


def _lines(words, tol=0.35):
    """Words -> lines (same direction, same baseline band within tol x text height), each in reading order."""
    keyed = []
    for x0, y0, x1, y1, t, d in words:
        if d is None:
            d = 0
        vert = d in (90, 270)
        keyed.append(((x0, y0, x1, y1, t), d, ((x0 + x1) / 2 if vert else (y0 + y1) / 2), (x1 - x0) if vert else (y1 - y0)))
    lines = []                                            # [direction, centre, height, [words]]
    for w, d, c, h in sorted(keyed, key=lambda z: (z[1], z[2])):
        for L in lines:
            if L[0] == d and abs(L[1] - c) <= tol * max(L[2], h, 1):
                L[3].append(w); break
        else:
            lines.append([d, c, h, [w]])
    order = {0: lambda w: w[0], 180: lambda w: -w[2], 270: lambda w: w[1], 90: lambda w: -w[3]}
    return [sorted(ws, key=order[d]) for d, _, _, ws in lines]


def _annot_str(a, key):
    n = pdfium_c.FPDFAnnot_GetStringValue(a, key.encode(), None, 0)
    if n <= 2:
        return ""
    buf = ctypes.create_string_buffer(n)
    pdfium_c.FPDFAnnot_GetStringValue(a, key.encode(), ctypes.cast(buf, ctypes.POINTER(pdfium_c.FPDF_WCHAR)), n)
    return buf.raw[:n - 2].decode("utf-16-le", errors="replace")


def _shx(pg, f):
    """[(text, box in displayed points)] of the page's 'AutoCAD SHX Text' annotations."""
    out = []
    for i in range(pdfium_c.FPDFPage_GetAnnotCount(pg)):
        a = pdfium_c.FPDFPage_GetAnnot(pg, i)
        try:
            if _annot_str(a, "T") != "AutoCAD SHX Text":
                continue
            rc = pdfium_c.FS_RECTF()
            pdfium_c.FPDFAnnot_GetRect(a, rc)
            (ax, ay), (bx, by) = f(rc.left, rc.bottom), f(rc.right, rc.top)
            out.append((_annot_str(a, "Contents"), (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))))
        finally:
            pdfium_c.FPDFPage_CloseAnnot(a)
    return out


def exact_frags(pdf, page_no=0, autorot=0, dpi=DPI):
    """(frags, source) with source 'text layer' / 'SHX text' / 'text layer + SHX text', or (None, reason)."""
    doc = pdfium.PdfDocument(str(pdf))
    try:
        pg = doc[page_no]
        f, (Wp, Hp) = _display(pg)
        words = _words(pg.get_textpage(), f)
        shx = _shx(pg, f)
    finally:
        doc.close()
    k = dpi / 72
    W, H = Wp * k, Hp * k
    out, src = [], []
    if len(words) >= MIN_WORDS:
        for ws in _lines(words):
            cur = [ws[0]]
            for a, b in zip(ws, ws[1:]):
                aw, ah = a[2] - a[0], a[3] - a[1]
                h = min(ah, aw) if ah > 2 * aw else ah
                gap = max(b[0] - a[2], b[1] - a[3], a[0] - b[2], a[1] - b[3])
                lead_no = len(cur) == 1 and re.fullmatch(r"\(?\d{1,2}(\.\d{1,2})*[.)]?", cur[0][4])
                if gap > GAP_H * max(h, 1) and not lead_no:     # a list number stays with its text ('4.1  MAIN ...')
                    out.append(cur); cur = [b]
                else:
                    cur.append(b)
            out.append(cur)
        out = [dict(text=ocr_style("".join(w[4] for w in ws)),
                    bbox=(min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)))
               for ws in out]
        src.append("text layer")
    if shx:
        out += [dict(text=ocr_style(t), bbox=b) for t, b in shx if t]
        src.append("SHX text")
    if not out:
        return None, f"no text layer ({len(words)} words) and no SHX text"
    frags = [dict(text=f["text"], bbox=[round(v) for v in _rot([v * k for v in f["bbox"]], autorot, W, H)], source="pdf")
             for f in out]
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
