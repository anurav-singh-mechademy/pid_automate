"""Score a batch run against the instrument bubbles printed in the PDFs' own text layer.
    .venv/bin/python batch/eval_text_layer.py lngc_bog_a_output [unit=034]
Truth: a bubble is a letters word (FZ, PISA) with a loop-number word (1102, 1003S) straight below it and the unit word
(034) above the letters (LNGC layout 034 / FZ / 1102), centred on one vertical line, all centred on one vertical line, i.e. the three lines inside one circle. Tag = <unit>-<letters>-<number>.
Extracted tags are matched ignoring unit prefix and dashes. Kept tags that match no text-layer bubble go to a crop
montage (<out>/eval/unmatched_*.png) to be judged by eye; text-layer bubbles we missed go to missed_*.png."""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

OUT = Path(sys.argv[1]); UNIT = sys.argv[2] if len(sys.argv) > 2 else r"\d{3}"
EV = OUT / "eval"; EV.mkdir(exist_ok=True)
LET, NUM = re.compile(r"^[A-Z]{1,6}$"), re.compile(r"^\d{4}[A-Z]{0,2}$")


def bubbles(page):
    """Both orientations: on sheets drawn rotated the bubble lines stack along x (vertical words, swapped axes)."""
    ws = page.get_text("words")
    horiz = [w for w in ws if (w[2] - w[0]) >= (w[3] - w[1]) or len(w[4]) == 1]
    vert = [(w[1], w[0], w[3], w[2], w[4]) for w in ws if (w[3] - w[1]) > (w[2] - w[0]) and len(w[4]) > 1]
    out = []
    for group, swap in ((horiz, False), (vert, True)):
        for b in _stacks(group):
            if swap:
                x0, y0, x1, y1 = b["box"]; b["box"] = [y0, x0, y1, x1]
            out.append(b)
    seen = set()
    return [b for b in out if not (b["tag"] in seen or seen.add(b["tag"]))]


def _stacks(ws):
    cx = lambda w: (w[0] + w[2]) / 2
    cy = lambda w: (w[1] + w[3]) / 2
    out = []
    for n in ws:
        if not NUM.match(n[4]):
            continue
        h = n[3] - n[1]
        near = lambda w, lo, hi: abs(cx(w) - cx(n)) < 0.8 * h and lo < cy(n) - cy(w) < hi
        # LNGC layout 034 / FZ / 1102: unit two lines above the number, letters between them
        units = [w for w in ws if re.fullmatch(UNIT, w[4]) and near(w, 1.2 * h, 3.2 * h)]
        lets = [w for w in ws if LET.match(w[4]) and near(w, 0.4 * h, 1.8 * h)]
        if not units or not lets:
            continue
        U = min(units, key=lambda w: cy(n) - cy(w))
        lets = [w for w in lets if cy(U) < cy(w) < cy(n)]
        if not lets:
            continue
        L = min(lets, key=lambda w: abs(cx(w) - cx(n)))      # alarm letters (H, LL) sit beside the bubble, not centred
        out.append(dict(tag=f"{U[4]}-{L[4]}-{n[4]}", box=[min(L[0], n[0], U[0]), U[1], max(L[2], n[2], U[2]), n[3]]))
    return out


key = lambda t: re.sub(r"^\d{3}-?", "", t).replace("-", "")
led = [r for r in csv.DictReader(open(OUT / "ledger.csv")) if r["status"] == "ok"]
rows = list(csv.DictReader(open(OUT / "mastersheet.csv")))
by_sheet = defaultdict(list)
for r in rows:
    by_sheet[r["sheet"]].append(r)
report, missed_items, unmatched_items = [], [], []
for L in led:
    doc = pymupdf.open(L["file"]); pg = doc[int(L["page"])]
    truth = bubbles(pg)
    tk = {key(b["tag"]): b for b in truth}
    mine = by_sheet[L["out_name"]]
    kept = [r for r in mine if r["keep"] == "True"]
    kk = {key(r["tag"]) for r in kept}
    hit = [b for k, b in tk.items() if k in kk]
    miss = [b for k, b in tk.items() if k not in kk]
    unmatched = [r for r in kept if key(r["tag"]) not in tk]
    report.append(dict(sheet=L["out_name"], text_bubbles=len(tk), found=len(hit), kept=len(kept), kept_unique=len(kk),
                       matched_unique=len(kk & set(tk)), unmatched=len(unmatched), dropped=len(mine) - len(kept)))
    print(report[-1], flush=True)
    if miss or unmatched:
        rot = int(L["rotation"] or 0)
        img = Image.frombytes("RGB", *(lambda p: ((p.width, p.height), p.samples))(pg.get_pixmap(dpi=300)))
        if rot:
            img = img.rotate(rot, expand=True)
        s = 300 / 72
        for b in miss:          # text-layer boxes are in unrotated PDF points
            bx = [v * s for v in b["box"]]
            missed_items.append((img if not rot else None, bx, f"{L['out_name'][-14:]} {b['tag']}", pg, s))
        for r in unmatched:
            bx = json.loads(r["bbox"]) if r["bbox"].startswith("[") else [float(v) for v in r["bbox"].strip("()").split(",")]
            unmatched_items.append((img, bx, f"{L['out_name'][-14:]} {r['tag']} [{r['source']}]", None, 1))


def montage(items, path, cols=6, cell=(330, 230)):
    R = (len(items) + cols - 1) // cols
    M = Image.new("RGB", (cols * cell[0], max(R, 1) * cell[1]), "white"); d = ImageDraw.Draw(M)
    for n, (img, b, cap, pg, s) in enumerate(items):
        if img is None:        # rotated sheet: crop from an unrotated render
            img = Image.frombytes("RGB", *(lambda p: ((p.width, p.height), p.samples))(pg.get_pixmap(dpi=300)))
        w, h = b[2] - b[0], b[3] - b[1]; m = max(w, h, 60) * 1.2
        c = img.crop((int(b[0] - m), int(b[1] - m), int(b[2] + m), int(b[3] + m))); c.thumbnail((cell[0] - 6, cell[1] - 36))
        x, y = (n % cols) * cell[0], (n // cols) * cell[1]
        M.paste(c, (x + 3, y + 3)); d.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(200, 200, 200))
        d.text((x + 4, y + cell[1] - 30), f"#{n} {cap}"[:52], fill=(0, 0, 200))
    M.save(path)


for nm, items in (("missed", missed_items), ("unmatched", unmatched_items)):
    for k in range(0, len(items), 48):
        montage(items[k:k + 48], EV / f"{nm}_{k // 48}.png")
json.dump(report, open(EV / "text_layer_report.json", "w"), indent=1)
T = lambda k: sum(r[k] for r in report)
print(f"TOTAL {len(report)} sheets: text-layer bubbles {T('text_bubbles')}, found {T('found')} ({T('found') / max(1, T('text_bubbles')):.1%}); "
      f"kept unique {T('kept_unique')}, matched {T('matched_unique')}, unmatched rows {T('unmatched')} -> judge by eye; "
      f"missed {len(missed_items)}")
