# lngc_eq16 extraction vs the two LNGC ground-truth files in ~/Downloads/pids_bhcp/:
#   PI_TAGS_LNGC_eq16.csv           - PI historian export; FLOC holds the P&ID instrument tag (111TIA-0830-E)
#   Flash GC sensors for Anurav.xlsx - monitored points of the Flash Gas Compressor 1K-1101 (CA:KIT:111TI0830.PV)
# Keys are (letters, number + A/B suffix) in unit 111. Match levels: exact letters, same loop (first letter + number:
# the DCS name drops alarm letters, PZA -> PZ). "Printed" = found in the full-page OCR of the 10 sheets (inline, or the
# letters within 12 mm of the number as in a stacked LNGC bubble), independent of bubble detection and tag assembly.
import csv, json, re, os
from collections import Counter, defaultdict
import openpyxl
D = os.path.expanduser("~/Downloads/pids_bhcp/"); OUT = "lngc_eq16_output/eval/"
MM = 300 / 25.4
strip = lambda n: re.sub(r"[A-Z]$", "", n)
PI = list(csv.DictReader(open(D + "PI_TAGS_LNGC_eq16.csv", encoding="utf-8-sig")))
FL = re.compile(r"^111([A-Z]{1,6})-(\d{4})([A-Z]?)(?:-|$)")
DCS = re.compile(r"^CA:KIT:111([A-Z]+?)(\d{4})([A-Z]?)")
gt_pi, dcs2floc = {}, {}
for r in PI:
    m = FL.match(r["FLOC"]); d = DCS.match(r["PI Tag Name"])
    if r["Discipline"] == "Instrumentation" and m:
        k = (m.group(1), m.group(2) + m.group(3))
        gt_pi.setdefault(k, r)
        if d: dcs2floc[(d.group(1), d.group(2) + d.group(3))] = k
wb = openpyxl.load_workbook(D + "Flash GC sensors for Anurav.xlsx", read_only=True, data_only=True)
gt_fgc = {}
for name, ext, *_ in list(wb.active.iter_rows(values_only=True))[1:]:
    d = DCS.match(str(ext))
    if d:
        k = (d.group(1), d.group(2) + d.group(3))
        gt_fgc[dcs2floc.get(k, k)] = dict(name=name, external_tag=ext, floc_mapped=k in dcs2floc)
M = list(csv.DictReader(open("lngc_eq16_output/mastersheet.csv")))
ext = defaultdict(list)
for m in M:
    t = re.sub(r"^1?-?111-|^1-", "", m["tag"])
    x = re.match(r"([A-Z]{1,6})-(\d{4})([A-Z]?)(?:-\d+)?$", t)
    if x and m["grammar_ok"] == "True":
        ext[(x.group(1), x.group(2) + x.group(3))].append(m)
exn = {(a, strip(b)) for a, b in ext}; exl = {(a[0], strip(b)) for a, b in ext}
OCR = [f for v in json.load(open(OUT + "ocr_full.json")).values() for f in v]
ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
def printed(L, n):
    inline = re.compile(rf"(?<![A-Z]){L}-?{n}(?!\d)"); num = re.compile(rf"(?<!\d){n}(?!\d)")
    for f in OCR:
        t = f["text"].upper().replace(" ", "")
        if inline.search(t): return True
        if num.search(t):
            cx, cy = ctr(f["bbox"])
            if any(g["text"].upper().replace(" ", "") in (L, "111" + L) and abs(ctr(g["bbox"])[0] - cx) < 12 * MM
                   and abs(ctr(g["bbox"])[1] - cy) < 12 * MM for g in OCR):
                return True
    return False
def status(k):
    if k in ext: return "exact"
    if (k[0], strip(k[1])) in exn: return "exact (A/B ignored)"
    if (k[0][0], strip(k[1])) in exl: return "same loop"
    return "missed, printed" if printed(k[0], k[1]) else "not on these sheets"
res = {}
for nm, gt in (("PI_TAGS", gt_pi), ("FlashGC", gt_fgc)):
    rows = [dict(gt_tag=f"{k[0]}-{k[1]}", status=status(k), description=v.get("PI Tag Description", v.get("name", "")),
                 equipment=v.get("Associated Major Equipment", "1K-1101"), source_tag=v.get("PI Tag Name", v.get("external_tag", "")))
            for k, v in sorted(gt.items())]
    res[nm] = rows
    with open(OUT + f"{nm}_to_extracted.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    c = Counter(r["status"] for r in rows); n = len(rows); drawn = n - c["not on these sheets"]
    hit = c["exact"] + c["exact (A/B ignored)"] + c["same loop"]
    print(f"\n{nm}: {n} instrument tags | " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"   recall on drawn {hit}/{drawn} = {hit / drawn:.0%} (exact letters {c['exact'] + c['exact (A/B ignored)']}/{drawn}) | on whole list {hit}/{n}")
    print("   missed, printed:", [r["gt_tag"] for r in rows if r["status"] == "missed, printed"])
    print("   same loop:", [r["gt_tag"] for r in rows if r["status"] == "same loop"])
# precision vs PI_TAGS (the fuller list): share of extracted unit-111 instrument tags present
gl = {(a[0], strip(b)) for a, b in gt_pi}; gn = {(a, strip(b)) for a, b in gt_pi}
ex_rows = []
for k, ms in sorted(ext.items()):
    st = "exact" if k in gt_pi or (k[0], strip(k[1])) in gn else "same loop" if (k[0][0], strip(k[1])) in gl else "NOT IN PI_TAGS"
    for m in ms:
        ex_rows.append(dict(extracted_tag=m["tag"], key=f"{k[0]}-{k[1]}", status=st, sheet=m["sheet"], source=m["source"], confidence=m["confidence"], bbox=m["bbox"]))
with open(OUT + "extracted_to_PI_TAGS.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ex_rows[0])); w.writeheader(); w.writerows(ex_rows)
c = Counter({k: v for k, v in Counter(r["status"] for r in {r["key"]: r for r in ex_rows}.values()).items()})
print(f"\nextracted distinct tags {len(ext)}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
print("   not in PI_TAGS:", sorted({r["key"] for r in ex_rows if r["status"] == "NOT IN PI_TAGS"}))
