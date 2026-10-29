"""Score tag extraction on the split-v2 test sheets (pid_registry.csv: notes 'split v2', tag_split == test).
    .venv/bin/python batch/eval_tagtest_v2.py [tagtest_v2_output] [--sample 60]
Three measures, nothing tuned on these sheets:
  text layer   sheets whose PDF has real text: every instrument bubble printed as text (a 2-5 letter word over a
               3-5 digit loop number, centred) is truth. Recall = truth bubbles among kept tags; precision = kept tags
               that are a truth bubble. Local devices the pipeline leaves out by design (HV, PG, TW, PSV ...) are
               left out of the truth too. Kept tags that are not a text-layer bubble go to a crop montage to judge by
               eye (inline text tags, valve tags written as labels are not bubbles but can be right).
  list         share of kept tags whose loop + instrument is in the DECP Master Tag List (a lower bound on precision:
               the list holds DCS points only)
  by eye       a fixed random sample of kept rows from all clean test sheets -> numbered crop montage
Outputs in <out>/eval/."""
import csv, json, random, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "batch"))
from tag_pipeline import LOCAL_DEVICE_RE, load_page
Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "tagtest_v2_output")
N_SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 60
EV = OUT / "eval"; EV.mkdir(exist_ok=True)
LET, NUM = re.compile(r"^[A-Z]{2,5}$"), re.compile(r"^\d{3,5}[A-Z]{0,2}(?:-\d)?$")


def key(t):
    m = re.search(r"([A-Z]{1,6})-?(\d{2,6}[A-Z]?)(?![0-9])", re.sub(r"^\d{1,3}-?", "", t))
    return f"{m.group(1)}-{m.group(2)}" if m else t


RING_DPI = 200


def _ringed(ink, cx, cy, h, k):
    """A drawn circle around a text stack of line height h (PDF points) centred at (cx, cy): for some radius between
    1.2 h and 3.5 h, most rays from the centre meet ink within +-0.25 h of it (a band, so a circle a few points off
    the text centre still counts; DCS shared-display circles in a square run large). Connector captions and
    reference boxes have no closed outline around the stack at one radius."""
    H, W = ink.shape
    angs = np.radians(np.arange(0, 360, 10))
    band = np.arange(-0.25, 0.26, 0.05) * h * k
    for r in np.arange(1.2, 3.5, 0.25) * h * k:
        hit = 0
        for a in angs:
            for d in band:
                x, y = int(cx * k + (r + d) * np.cos(a)), int(cy * k + (r + d) * np.sin(a))
                if 0 <= x < W and 0 <= y < H and ink[y, x]:
                    hit += 1
                    break
        if hit / len(angs) >= 0.85:
            return True
    return False


def _word(L):
    """An ordinary English word ('FROM', 'DRUM', 'GLAND'): connector captions and labels over a reference number."""
    from descgen.text import _RANK
    if len(L) == 2:          # two-letter instrument codes (TI, PI, HA) are frequent 'words' in the splitter's list
        return L in ("TO", "OF", "IN", "ON", "AT", "BY")
    return _RANK.get(L.lower(), 10 ** 9) < 20000


def text_bubbles(pdf, page=0):
    """Instrument bubbles printed as text: letters word directly above a loop-number word, centres aligned, with a
    drawn circle around the pair, or letters that are no English word (shared-display symbols: circle in a square, text
    off the circle centre)."""
    pg = pymupdf.open(pdf)[page]
    ws = [w for w in pg.get_text("words") if (w[2] - w[0]) >= 0.5 * (w[3] - w[1])]
    pix = pg.get_pixmap(dpi=RING_DPI, colorspace=pymupdf.csGRAY, alpha=False)
    ink = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width) < 215   # thin anti-aliased outlines render light
    M, k = pg.rotation_matrix, RING_DPI / 72
    out = set()
    for n in ws:
        if not NUM.match(n[4]):
            continue
        h, cx = n[3] - n[1], (n[0] + n[2]) / 2
        up = [l for l in ws if LET.match(l[4]) and abs((l[0] + l[2]) / 2 - cx) < 0.6 * h and 0 <= n[1] - l[3] < 1.2 * h]
        if up:
            l = min(up, key=lambda l: n[1] - l[3])
            c = pymupdf.Point(cx, (l[1] + n[3]) / 2) * M
            if not LOCAL_DEVICE_RE.match(l[4]) and (_ringed(ink, c.x, c.y, h, k) or not _word(l[4])):
                out.add(f"{l[4]}-{n[4]}")
    return out


def montage(rows, path, cols=6, cell=300):
    """Crops of the rows' boxes (from the overlay images), numbered."""
    if not rows:
        return
    H = (len(rows) + cols - 1) // cols
    M = Image.new("RGB", (cols * cell, H * (cell + 40)), "white")
    d = ImageDraw.Draw(M)
    cache = {}
    for i, r in enumerate(rows):
        if r["sheet"] not in cache:
            cache = {r["sheet"]: Image.open(OUT / "sheets" / f"tags_{r['sheet']}_overlay.png")}
        im = cache[r["sheet"]]
        b = [int(float(v)) for v in re.findall(r"-?\d+(?:\.\d+)?", r["bbox"])]
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        s = max(b[2] - b[0], b[3] - b[1], 150) * 1.6
        crop = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 40)
        M.paste(crop, (x, y + 40))
        d.text((x + 4, y + 4), f"{i} {r['tag']}", fill="black")
        d.text((x + 4, y + 20), f"{r['source']} | {r['lines'][:30]}", fill="gray")
    M.save(path)


def main():
    import descgen.bench as DB
    reg = {r["fingerprint"]: r for r in csv.DictReader(open(ROOT / "pid_registry.csv"))}
    led = {r["out_name"]: r for r in csv.DictReader(open(OUT / "ledger.csv")) if r["status"] == "ok"}
    clean = {n for n, L in led.items() if reg.get(L["fingerprint"], {}).get("tag_split") == "test"}
    rows = [r for r in csv.DictReader(open(OUT / "mastersheet.csv"))]
    kept = [r for r in rows if r["keep"] == "True" and r["sheet"] in clean]
    refs = DB._refs()["decp"]

    # ---- list-confirmed share
    def listed(t):
        m = re.match(r"^(?:\d{1,3}-)*([A-Z]{1,6})-(\d{3,5})([A-Z]?)(?:-\d+)?$", t)
        return bool(m) and any(DB.same_instrument(m.group(1), L) for L, _ in refs.get((m.group(1)[0], m.group(2), m.group(3)), []))
    per = defaultdict(lambda: dict(read=0, kept=0, listed=0))
    for r in rows:
        if r["sheet"] in clean:
            per[r["sheet"]]["read"] += 1
    for r in kept:
        per[r["sheet"]]["kept"] += 1; per[r["sheet"]]["listed"] += listed(r["tag"])

    # ---- text-layer truth
    tl, miss_rows, extra_rows = {}, [], []
    for n in sorted(clean):
        L = led[n]
        if reg[L["fingerprint"]]["exact_text"] != "1":
            continue
        truth = text_bubbles(L["file"], int(L["page"]))
        if len(truth) < 5:          # bubble text not found as ringed stacks on this drafting style: no text-layer truth
            continue
        ks = [r for r in kept if r["sheet"] == n]
        got = {key(r["tag"]) for r in ks}
        tk = {key(t) for t in truth}
        tl[n] = dict(truth=len(tk), found=len(tk & got), kept=len(got), kept_true=len(got & tk))
        extra_rows += [r for r in ks if key(r["tag"]) not in tk]
        miss_rows += [dict(sheet=n, tag=t) for t in sorted(tk - got)]
    montage(extra_rows, EV / "textlayer_kept_not_bubble.png")
    json.dump(dict(per_sheet=tl, missed=miss_rows,
                   kept_not_bubble=[dict(i=i, sheet=r["sheet"], tag=r["tag"], source=r["source"], lines=r["lines"])
                                    for i, r in enumerate(extra_rows)]), open(EV / "textlayer.json", "w"), indent=1)

    # ---- by-eye sample (fixed seed), sheets in name order
    random.seed(20261028)
    smp = sorted(random.sample(kept, min(N_SAMPLE, len(kept))), key=lambda r: (r["sheet"], r["tag"]))
    montage(smp, EV / "eye_sample.png")
    with open(EV / "eye_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["i", "sheet", "tag", "source", "lines", "bbox", "verdict"]); w.writeheader()
        for i, r in enumerate(smp):
            w.writerow(dict(i=i, sheet=r["sheet"], tag=r["tag"], source=r["source"], lines=r["lines"], bbox=r["bbox"], verdict=""))

    T = {k: sum(v[k] for v in per.values()) for k in ("read", "kept", "listed")}
    print(f"clean test sheets {len(clean)} (of {len(led)} run): read {T['read']}, kept {T['kept']}, "
          f"in DECP list {T['listed']} ({T['listed'] / max(1, T['kept']):.1%} of kept)")
    if tl:
        S = {k: sum(v[k] for v in tl.values()) for k in ("truth", "found", "kept", "kept_true")}
        print(f"text-layer sheets {len(tl)}: recall {S['found']}/{S['truth']} = {S['found'] / S['truth']:.1%}; "
              f"kept that are text bubbles {S['kept_true']}/{S['kept']} = {S['kept_true'] / S['kept']:.1%} "
              f"(rest -> textlayer_kept_not_bubble.png, {len(extra_rows)} rows)")
        for n, v in tl.items():
            print(f"   {n[:40]:40s} truth {v['truth']:3d} found {v['found']:3d} kept {v['kept']:3d} kept-true {v['kept_true']:3d}")
    print(f"eye sample: {len(smp)} rows -> eval/eye_sample.png / eye_sample.csv")


if __name__ == "__main__":
    main()
