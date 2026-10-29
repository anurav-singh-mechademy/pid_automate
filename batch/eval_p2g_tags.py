"""Phase-1 tag extraction on PID2Graph sheets, scored against their 'instrumentation' boxes.
    .venv/bin/python batch/eval_p2g_tags.py "external/pid2graph/data/PID2Graph OPEN100" pidgraph_output/p2g_tags
PID2Graph has no tag text, only symbol boxes, so:
  recall     GT instrumentation boxes with an extracted tag read from them (bubble / stack source, or text inside)
  precision  every kept tag, checked by eye: montages <out>/gt_bubbles_*.png (each GT bubble + what we read) and
             <out>/extra_tags_*.png (kept tags not on a GT bubble)
Sheets are enlarged to A1 at 300 dpi (long side 9933 px) as in pidgraph/p2g_eval.py; tag_filter runs over the batch."""
import csv, json, sys, time
from pathlib import Path
import networkx as nx
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import tag_pipeline as TP
import tag_filter
Image.MAX_IMAGE_PIXELS = None
SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
A1 = 9933


def img_of(i):
    return next(SRC / f"{i}{e}" for e in (".png", ".jpg") if (SRC / f"{i}{e}").exists())


ids = sorted((p.stem for p in SRC.glob("*.graphml")), key=int)
rows_f = OUT / "tags.csv"
if not rows_f.exists():
    eng = TP.RapidOCR(); model, names = TP.load_rfdetr(TP.WEIGHTS)
    rows = []
    for i in ids:
        t0 = time.time()
        im = Image.open(img_of(i)).convert("RGB"); k = A1 / max(im.size)
        page = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        frags = TP.ocr_page(page, eng)
        px, _, _ = TP.measure_bubble(page, model, names, frags)
        bubbles = TP.detect_multiscale(page, model, names, TP.choose_tile(px))
        TP.attach(frags, bubbles)
        for t in TP.build_tags(frags, bubbles):
            b = t["bbox"] if isinstance(t["bbox"], list) else json.loads(t["bbox"])
            rows.append(dict(sheet=i, tag=t["tag"], source=t["source"], lines=t.get("lines", ""),
                             confidence=t["confidence"], bbox=json.dumps([round(v / k, 1) for v in b])))
        print(i, len(bubbles), "bubbles", sum(r["sheet"] == i for r in rows), "tags", round(time.time() - t0), "s", flush=True)
    tag_filter.apply(rows)
    with open(rows_f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rows = list(csv.DictReader(open(rows_f)))
for r in rows:
    r["bbox"] = json.loads(r["bbox"]); r["keep"] = r["keep"] == "True"


def inside(b, g, pad=0.25):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = g[2] - g[0], g[3] - g[1]
    return g[0] - pad * w <= cx <= g[2] + pad * w and g[1] - pad * h <= cy <= g[3] + pad * h


def montage(items, path, cols=8, cell=(260, 200)):
    """items: (img, box, caption, colour) -> grid of crops"""
    rows_n = (len(items) + cols - 1) // cols
    M = Image.new("RGB", (cols * cell[0], max(1, rows_n) * cell[1]), "white"); d = ImageDraw.Draw(M)
    for n, (im, b, cap, col) in enumerate(items):
        w, h = b[2] - b[0], b[3] - b[1]; m = max(w, h) * 0.6
        c = im.crop((int(b[0] - m), int(b[1] - m), int(b[2] + m), int(b[3] + m)))
        c.thumbnail((cell[0] - 6, cell[1] - 34))
        x, y = (n % cols) * cell[0], (n // cols) * cell[1]
        M.paste(c, (x + 3, y + 3)); d.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(200, 200, 200))
        d.text((x + 4, y + cell[1] - 30), cap[:40], fill=col); d.text((x + 4, y + cell[1] - 16), f"#{n}", fill=(90, 90, 90))
    M.save(path)


report, gt_items, extra_items = [], [], []
for i in ids:
    G = nx.read_graphml(SRC / f"{i}.graphml")
    gt = [[d[k] for k in ("xmin", "ymin", "xmax", "ymax")] for _, d in G.nodes(data=True) if d["label"] == "instrumentation"]
    im = Image.open(img_of(i)).convert("RGB")
    mine = [r for r in rows if r["sheet"] == i]
    kept = [r for r in mine if r["keep"]]
    hit = 0
    for g in gt:
        on = [r for r in kept if inside(r["bbox"], g)]
        hit += bool(on)
        gt_items.append((im, g, f"{i}: " + (" | ".join(r["tag"] for r in on) or "-- none --"), (0, 120, 0) if on else (220, 0, 0)))
    extra = [r for r in kept if not any(inside(r["bbox"], g) for g in gt)]
    for r in extra:
        extra_items.append((im, r["bbox"], f"{i}: {r['tag']} [{r['source']}]", (0, 0, 200)))
    report.append(dict(sheet=i, gt_bubbles=len(gt), with_tag=hit, extracted=len(mine), kept=len(kept),
                       on_gt_bubble=len(kept) - len(extra), elsewhere=len(extra)))
    print(report[-1], flush=True)
for n in range(0, len(gt_items), 96):
    montage(gt_items[n:n + 96], OUT / f"gt_bubbles_{n // 96}.png")
for n in range(0, len(extra_items), 96):
    montage(extra_items[n:n + 96], OUT / f"extra_tags_{n // 96}.png")
json.dump(report, open(OUT / "report.json", "w"), indent=1)
T = lambda k: sum(r[k] for r in report)
print(f"TOTAL gt bubbles {T('gt_bubbles')}, with a tag {T('with_tag')} ({T('with_tag') / T('gt_bubbles'):.1%}); "
      f"kept tags {T('kept')} ({T('on_gt_bubble')} on GT bubbles, {T('elsewhere')} elsewhere); dropped {T('extracted') - T('kept')}")
