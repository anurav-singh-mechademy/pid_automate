# Recall per DECP equipment subfolder against that equipment's own tag lists / historian exports
# (pids_bhcp_output/eval/decp_equipment_gt.json), on the P&IDs in the same subfolder.
#   found      = extracted on those sheets with the same letters + number (A/B suffix ignored)
#   same loop  = extracted with other letters on the same loop (drawing PIT, list PI)
#   printed    = not extracted, but the tag is in the full-page OCR of those sheets (inline, or letters within 12 mm of the number)
#   not drawn  = neither: not on these sheets
import csv, json, re
from collections import Counter
MM = 300 / 25.4
GT = json.load(open("pids_bhcp_output/eval/decp_equipment_gt.json"))
OCR = json.load(open("pids_bhcp_output/eval/decp_visual/ocr_full.json"))
SHEETS = {"01 LNG Product Pumps": ["SC62D1911", "SC62D1913", "SC62D1914"],
          "02 LNG Loading Pumps": ["SC62D0508"],
          "03 LNG Expander": ["SC62D1912", "SC62D1913", "SC62D1531", "SC62D1953", "SC62D1911"],
          "04 MR Expander": ["SC62D1930", "SC62D1931"]}
M = list(csv.DictReader(open("pids_bhcp_output/mastersheet.csv")))
strip = lambda n: re.sub(r"[A-Z]\d?$", "", n)
def ekeys(t):      # tag as read, plus variants without a "111-" prefix / unit "5" / OCR'd "S"
    out = set()
    for v in {t, re.sub(r"^\d{1,3}-", "", t), re.sub(r"^[5S](?=[A-Z]{2,5}-)", "", re.sub(r"^\d{1,3}-", "", t))}:
        m = re.match(r"([A-Z]{2,5})-(\d{3,5})", v)
        if m: out.add(m.groups())
    return out
ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def printed(L, n, frs):
    inline = re.compile(rf"(?<![A-Z]){L}-?{n}(?!\d)")
    num = re.compile(rf"(?<![0-9]){n}(?!\d)")
    for f in frs:
        t = f["text"].upper().replace(" ", "")
        if inline.search(t): return True
        if num.search(t):
            cx, cy = ctr(f["bbox"])
            if any(g["text"].upper().replace(" ", "") == L and abs(ctr(g["bbox"])[0] - cx) < 12 * MM and abs(ctr(g["bbox"])[1] - cy) < 12 * MM for g in frs):
                return True
    return False
rows = []
for g, gt in GT.items():
    sh = SHEETS[g]
    ex = set()
    for m in M:
        if any(m["sheet"].startswith(s) for s in sh): ex |= ekeys(m["tag"])
    exn = {(a, strip(b)) for a, b in ex}; exl = {(a[0], strip(b)) for a, b in ex}
    frs = [f for s, v in OCR.items() if any(s.startswith(x) for x in sh) for f in v]
    for k in sorted(gt):
        L, n = k.split("-"); n0 = strip(n)
        st = ("found" if (L, n0) in exn else "same loop" if (L[0], n0) in exl
              else "printed, not extracted" if printed(L, n0, frs) else "not drawn on these sheets")
        rows.append(dict(subfolder=g, list_tag=k, status=st, sheets=" | ".join(sh)))
with open("pids_bhcp_output/eval/decp_equipment_recall.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for g in GT:
    c = Counter(r["status"] for r in rows if r["subfolder"] == g); n = sum(c.values())
    drawn = n - c["not drawn on these sheets"]
    print(f"{g:22s} list {n:2d} | found {c['found']:2d}  same loop {c['same loop']:2d}  printed-not-extracted {c['printed, not extracted']:2d}  "
          f"not drawn {c['not drawn on these sheets']:2d} | recall on list {(c['found'] + c['same loop']) / n:.0%}, "
          f"on drawn {(c['found'] + c['same loop'])}/{drawn}")
    for r in rows:
        if r["subfolder"] == g and r["status"] not in ("found",): print(f"     {r['status']:26s} {r['list_tag']}")
