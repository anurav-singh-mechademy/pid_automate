"""Precision filter for extracted tags. Runs after extraction; never deletes: every row gets `keep` and, if dropped, a
`drop_reason`, so each rule can be audited. Rules use only the batch's own drawings (no client lists):

  hand valve      HV + instrument letters (HVPSV, HVPIT, HVLIT...): root / isolation valve of an instrument, a local device
  reference       a free-text tag inside a sentence or a connector caption ("FROM UIC5027A", "25 TIC5007 IS PLACED IN
                  MANUAL") when the same tag, or another instrument of the same loop, is read as itself elsewhere in the
                  batch; when the reference is the only place the loop was read, it is kept (recall first)
  not a tag       a tag-shaped string inside a sentence whose letter code is never used in a bubble in the batch and
                  is not a valve code (MR0103 in 'MATERIAL SA-240 ...', DCP-PRC-MAT-0007, a file name)
  plain word      the letter code is an ordinary English word of 5+ letters (WATER-1687 on a connector)
  bad suffix      instance suffix I or O (ISA avoids them; '0091I' is a misread of the next character)
Not dropped, only marked in `kind` (some clients list them as DCS points, e.g. LNGC PI tags include 111UZ-0741):
  interlock       UZ-nnnn written as plain text on a signal line (shutdown logic), not inside a bubble

General stages (switch with `stages`; all on by default). They learn from the batch itself, never from a client:
  repair    junk at the edges of a read ('V-034-UY-1322', 'Q34--RO-1111-', '034-PDIS-1204-A') is cut back to the part
            that fits one of the batch's own tag formats (learned from its bubble reads, see `formats`)
  split     one read holding several tags of side-by-side bubbles ('FT-14011401', 'XZA-3127C-CS3127D-CS...') becomes
            one row per tag
  date      a date ('28-SEP-2021', revision tables)
  unread    no letter code + number at all ('XZA', 'VLD', 'M', '034-1101'): a bubble whose text was not read
  table     free text inside a ruled table (revision / title block, note and data tables): grids of closed rectangular
            cells found in the page's long straight lines; needs the page images (`pages`)
  format    free text whose shape (digits / letter runs / dashes) is none of the batch's bubble tag formats, give or
            take an instance letter ('MAY-12641' among 4-digit tags, 'DN-600', '0-KM-3405C')
  caption   a stacked read outside any bubble whose letter code never occurs in a bubble ('LP BOG / 4648'); a stack
            that starts with a unit line ('034 / FY / 0252') is a missed bubble and is kept
"""
import re
from collections import Counter
import numpy as np
import wordninja

LETTERS = re.compile(r"(?:^|-)([A-Z]{1,6})-\d")
TAGLIKE = re.compile(r"^\d*[A-Z]{1,6}-?\d{2,5}[A-Z]?(?:-\d+)?$|^\d+$|^[A-Z]-?\d")    # tag, number, or tag fragment
FREE_TEXT = ("text", "text_near_bubble", "stack")
STAGES = ("repair", "split", "date", "unread", "format", "caption")     # "table": available, off by default
DATE = re.compile(r"^\d{1,2}-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)-\d{2,4}$")
HAS_TAG = re.compile(r"[A-Z]{1,6}-?(?:\d{2,}|XXX)")       # XXX: number not yet assigned (drafting placeholder)
FORMAT_MIN = (3, 0.02)        # a format is the batch's own when >= 3 bubble reads and >= 2 % of them have it
FIX_DIGIT = str.maketrans("OQDIL", "00011")    # OCR letter-for-digit confusions, used only where a format wants digits


def letters_of(tag):
    m = LETTERS.search(tag)
    return m.group(1) if m else ""


def bubble_codes(rows):
    """Letter codes that appear inside instrument bubbles anywhere in the batch."""
    return {letters_of(r["tag"]) for r in rows if r["source"] == "bubble" and letters_of(r["tag"])}


def sentence_words(lines, tag):
    """Ordinary words in the text the tag was read from (anything that is not a tag, number or tag fragment)."""
    words = re.split(r"[\s/]+", str(lines).upper())
    return [w for w in words if len(w) >= 2 and w.isalpha() and not TAGLIKE.match(w) and w not in tag.replace("-", "")]


def ident(tag):
    """letters + number + suffix, without unit prefix or -n: '111-TICA-0813' -> 'TICA-0813'"""
    m = re.search(r"([A-Z]{1,6})-(\d{2,5}[A-Z]?)(?:-\d+)?$", tag)
    return f"{m.group(1)}-{m.group(2)}" if m else tag


def loop_of(tag):
    """first letter + loop number: the loop the instrument belongs to ('FIT-5023' and 'FI-5023' share 'F-5023')"""
    m = re.search(r"([A-Z])[A-Z]{0,5}-(\d{2,5})", tag)
    return f"{m.group(1)}-{m.group(2)}" if m else tag


EXTRA = set("AHLTSC")    # alarm / transmitter / switch / controller letters: FI and FIT are one instrument's loop entries


def same_instrument(a, b):
    a, b = (a, b) if len(a) >= len(b) else (b, a)
    return a.startswith(b) and set(a[len(b):]) <= EXTRA


# ---------------------------------------------------------------- the batch's own tag formats
def shape(tag):
    """'034-UY-1322' -> '999-A-9999', 'TIA-3172-E' -> 'A-9999-A': digits kept by count, letter runs collapsed."""
    return re.sub(r"[A-Z]+", "A", re.sub(r"\d", "9", tag))


def formats(rows):
    """Shapes of the batch's bubble reads that are common enough to be its tag formats."""
    c = Counter(shape(r["tag"]) for r in rows if r["source"] == "bubble" and HAS_TAG.search(r["tag"]))
    n = sum(c.values())
    return {s for s, k in c.items() if k >= max(FORMAT_MIN[0], FORMAT_MIN[1] * n)}


def _rx(s):
    return "".join(r"[A-Z]{1,6}" if ch == "A" else r"\d" if ch == "9" else re.escape(ch) for ch in s)


def in_format(tag, fmts):
    """The tag's shape is a batch format, allowing an instance letter more or less ('XZA-3126C' ~ 'A-9999')."""
    sh = shape(tag)
    return sh in fmts or sh + "A" in fmts or (sh.endswith("9A") and sh[:-1] in fmts)


def repair(tag, fmts, rxs):
    """Cut a read back to the part that fits one of the batch formats, removing at most one stray character in front
    and one stray letter behind ('V-034-UY-1322', '7-034-GB-1096', '034-PDIS-1204-A'; a '-2' suffix is kept) and repeated / edge dashes ('034--HC-1322',
    'Q34--RO-1111-': a digit-like letter in the unit prefix is read as the digit). Never cuts inside a run of letters
    or digits, so 'MAY-12641' or 'CS-3124B' stay as read. None when nothing fits."""
    if in_format(tag, fmts) or DATE.match(tag):
        return None
    t = re.sub(r"-{2,}", "-", tag).strip("-")
    parts = t.split("-")
    cands = [t]
    if parts and re.fullmatch(r"[0-9OQDIL]{2,3}", parts[0]) and re.search(r"\d", parts[0]):   # 'Q34' -> '034'
        cands.append("-".join([parts[0].translate(FIX_DIGIT)] + parts[1:]))
    for c in cands:
        if in_format(c, fmts):
            return c if c != tag else None
    best = None
    for c in cands:
        for rx in rxs:
            for m in rx.finditer(c):
                a, b = m.span()
                head, tail = re.sub(r"[^0-9A-Z]", "", c[:a]), re.sub(r"[^0-9A-Z]", "", c[b:])
                if (a and c[a - 1] != "-") or (b < len(c) and c[b] != "-") or len(head) > 1:
                    continue
                if tail and not re.fullmatch(r"[A-Z]", tail):    # a '-2' instance number is part of the tag
                    continue
                if best is None or b - a > len(best):
                    best = m.group()
    return best


def split(tag):
    """'FT-14011401' -> ['FT-1401'], 'TE-1008A1008B' -> ['TE-1008A', 'TE-1008B']; None when it is one tag."""
    m = re.fullmatch(r"((?:\d{1,3}-)?[A-Z]{1,6}-)(.+)", tag)
    if not m:
        return None
    head, rest = m.groups()
    toks = re.findall(r"\d{3,6}[A-Z]?(?:-[A-Z]{1,3}(?=\d|$))?", rest)
    if len(toks) < 2 or "".join(toks) != rest:
        return None
    n = {len(re.match(r"\d+", t).group()) for t in toks}
    if len(n) != 1:
        return None
    out = []
    for t in toks:
        if head + t not in out:
            out.append(head + t)
    return out


# ---------------------------------------------------------------- ruled tables on the page
def table_regions(img, px_per_mm, min_cells=6):
    """Boxes (in img px) of ruled tables: groups of >= min_cells closed rectangular cells bounded by long straight
    lines. Pipes rarely close a rectangle once short items (valves, text) are removed, so cells mean tables."""
    import cv2
    g = np.asarray(img.convert("L"))
    ink = (g < 160).astype(np.uint8)
    L = max(8, int(8 * px_per_mm))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    lines = cv2.dilate(h | v, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - lines, connectivity=4)
    H, W = g.shape
    cells = []
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + hh >= H:
            continue
        if not (2.5 * px_per_mm <= hh <= 40 * px_per_mm and 3 * px_per_mm <= w <= 250 * px_per_mm):
            continue
        if a < 0.85 * w * hh:
            continue
        cells.append((x, y, x + w, y + hh))
    par = list(range(len(cells)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    gap = 6
    for i, a in enumerate(cells):
        for j in range(i + 1, len(cells)):
            b = cells[j]
            if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                par[find(i)] = find(j)
    groups = {}
    for i, c in enumerate(cells):
        groups.setdefault(find(i), []).append(c)
    out = []
    for cs in groups.values():
        if len(cs) >= min_cells and len({c[1] // 4 for c in cs}) >= 2 and len({c[0] // 4 for c in cs}) >= 2:
            out.append((min(c[0] for c in cs), min(c[1] for c in cs), max(c[2] for c in cs), max(c[3] for c in cs)))
    return out


def _bbox(r):
    b = r.get("bbox")
    if isinstance(b, str):
        b = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", b)]
    return b if b and len(b) == 4 else None


# ---------------------------------------------------------------- classification
def classify(row, codes, read_as_itself, loops_read, st, fmts):
    tag, src = row["tag"], row["source"]
    L = letters_of(tag)
    if "date" in st and DATE.match(tag):
        return "date"
    if "unread" in st and not HAS_TAG.search(tag):
        return "unread"
    if re.fullmatch(r"HV[A-Z]{1,4}", L):
        return "hand valve"
    if src in FREE_TEXT and sentence_words(row.get("lines", ""), tag):
        if ident(tag) in read_as_itself or any(same_instrument(L, c) for c in loops_read.get(loop_of(tag), ())):
            return "reference"
        if L and L not in codes and not L.endswith("V"):          # MR0103 in 'MATERIAL ...', MAT-0007, a file name
            return "not a tag"
    if src in FREE_TEXT and len(L) >= 5 and wordninja.split(L.lower()) == [L.lower()]:
        return "plain word"
    m = re.search(r"\d([A-Z])(?:-\d+)?$", tag)
    if m and m.group(1) in "IO":
        return "bad suffix"
    if "table" in st and src in ("text", "text_near_bubble") and row.get("_in_table"):
        return "in table"
    if "format" in st and fmts and src in ("text", "text_near_bubble") and not in_format(tag, fmts):
        return "off-format"
    if "format_bubble" in st and fmts and src == "bubble" and not in_format(tag, fmts):
        return "off-format bubble"
    if "caption" in st and src == "stack" and L not in codes and not re.match(r"\d+\s*/", str(row.get("lines", ""))):
        return "caption"       # a stack that starts with a unit line ('034 / FY / 0252') is a bubble's own text
    return ""


def apply(rows, stages=STAGES, pages=None, cache=None):
    """rows: dicts with tag, source, lines, bbox, sheet (a batch). Adds keep / drop_reason / kind in place and returns
    (rows, counts); `split` may add rows. pages(sheet) -> (PIL image, image px per bbox px, image px per mm) or None,
    needed only by the 'table' stage; `cache` (dict) keeps the table boxes per sheet between calls."""
    st = set(stages)
    fmts = formats(rows)
    if "repair" in st or "split" in st:
        rxs = [re.compile(_rx(s)) for s in sorted(fmts, key=len, reverse=True)]
        out = []
        for r in rows:
            r.setdefault("tag_read", r["tag"])
            parts = split(r["tag"]) if "split" in st else None
            if parts:
                for k, t in enumerate(parts):
                    out.append(dict(r, tag=t, fix=f"split from {r['tag_read']}", rid=f"{r.get('rid', '')}.{k}"))
                continue
            fixed = repair(r["tag"], fmts, rxs) if "repair" in st else None
            if fixed:
                r = dict(r, tag=fixed, fix=f"repaired from {r['tag_read']}")
            out.append(r)
        rows[:] = out
    if "table" in st and pages:
        by = {}
        for r in rows:
            by.setdefault(r["sheet"], []).append(r)
        for sh, rs in by.items():
            if not any(r["source"] in ("text", "text_near_bubble") for r in rs):
                continue
            if cache is not None and sh in cache:
                regs, f = cache[sh]
            else:
                pg = pages(sh)
                if pg is None:
                    continue
                img, f, ppm = pg
                regs = table_regions(img, ppm)
                if cache is not None:
                    cache[sh] = (regs, f)
            for r in rs:
                b = _bbox(r)
                if b and regs:
                    cx, cy = (b[0] + b[2]) / 2 * f, (b[1] + b[3]) / 2 * f
                    r["_in_table"] = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in regs)
    codes = bubble_codes(rows)
    # tags read as themselves somewhere in the batch (a bubble, or a label that is not part of a sentence)
    read_as_itself = {ident(r["tag"]) for r in rows
                      if r["source"] not in FREE_TEXT or not sentence_words(r.get("lines", ""), r["tag"])}
    loops_read = {}
    for t in read_as_itself:
        loops_read.setdefault(loop_of(t), set()).add(letters_of(t) or t.split("-")[0])
    for r in rows:
        r["drop_reason"] = classify(r, codes, read_as_itself, loops_read, st, fmts)
        r["keep"] = not r["drop_reason"]
        r["kind"] = "interlock" if letters_of(r["tag"]) == "UZ" and r["source"] != "bubble" else ""
        r.pop("_in_table", None)
    return rows, Counter(r["drop_reason"] or "kept" for r in rows)


def pdf_pages(ledger_csv, dpi=75):
    """pages() for apply(): render each ledger page at low dpi, rotated as phase 1 did; bbox px are 300 dpi."""
    import csv
    import pymupdf
    from PIL import Image
    led = {r["out_name"]: r for r in csv.DictReader(open(ledger_csv))}
    def get(sheet):
        L = led.get(sheet)
        if not L:
            return None
        p = pymupdf.open(L["file"])[int(L["page"])].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        img = Image.frombytes("L", (p.width, p.height), p.samples)
        if int(L.get("rotation") or 0):
            img = img.rotate(int(L["rotation"]), expand=True)
        return img, dpi / 300, dpi / 25.4
    return get


def filter_mastersheet(out_dir, stages=STAGES):
    """Adds keep / drop_reason / kind (and tag_read / fix where a read was repaired or split) to
    <out_dir>/mastersheet.csv and writes mastersheet_filtered.csv (kept rows only) plus 'kept tags' / 'dropped'
    sheets in mastersheet.xlsx."""
    import csv, openpyxl
    from pathlib import Path
    out = Path(out_dir)
    rows = list(csv.DictReader(open(out / "mastersheet.csv")))
    for r in rows:                                   # re-runs start from what was read, not from an earlier repair
        if r.get("tag_read"):
            r["tag"] = r["tag_read"]
    rows = [r for r in rows if not str(r.get("fix", "")).startswith("split") or r.get("rid", "").endswith(".0")]
    for i, r in enumerate(rows):
        r["rid"] = str(i)
    pages = pdf_pages(out / "ledger.csv") if (out / "ledger.csv").exists() else None
    rows, counts = apply(rows, stages, pages)
    cols = list(dict.fromkeys([c for r in rows for c in r]))
    for name, sel in (("mastersheet.csv", rows), ("mastersheet_filtered.csv", [r for r in rows if r["keep"]])):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(sel)
    wb = openpyxl.load_workbook(out / "mastersheet.xlsx")
    for t in ("kept tags", "dropped"):
        if t in wb.sheetnames: del wb[t]
    for title, sel in (("kept tags", [r for r in rows if r["keep"]]), ("dropped", [r for r in rows if not r["keep"]])):
        ws = wb.create_sheet(title); ws.append(cols)
        for r in sel: ws.append([str(r.get(c, "")) for c in cols])
        ws.freeze_panes = "A2"
    wb.save(out / "mastersheet.xlsx")
    return counts
