"""Score SheetGraph against PID2Graph ground truth (Zenodo 14803338, 'Complete' sheets: <n>.png + <n>.graphml).
    .venv/bin/python -m pidgraph.p2g_eval "external/pid2graph/data/PID2Graph OPEN100" [--only 0 3] [--out pidgraph_output/p2g]

PID2Graph graphs: symbol nodes (valve, instrumentation, arrow, general, tank, pump, inlet/outlet) + 'connector' (where
a line meets a symbol) and 'crossing' (line bend / T) nodes, edges = straight line pieces. SheetGraph has its own
point nodes (junction, end, tap). Both are compared at SYMBOL level:
  nodes        symbols matched one-to-one by box IoU >= 0.5 (class-agnostic; class agreement reported separately)
  connections  two symbols are connected when a line runs between them through point nodes only (GT: connector /
               crossing; ours: junction / end / tap over process, signal and leader edges)
               precision = predicted pairs that are GT pairs / predicted pairs
               recall    = GT pairs found / GT pairs; 'traced recall' counts only GT pairs whose two symbols were both
               detected, i.e. how well lines are followed once the symbols are found
This is not the paper's edge mAP (no confidence ranking, no point nodes scored), but the same question: which symbols
are joined by a line. Sheets are enlarged to the size of an A1 sheet at 300 dpi (long side 9933 px) before running,
because the pipeline's distances are in drawing mm at 300 dpi; nothing from the ground truth is used for that."""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import networkx as nx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
Image.MAX_IMAGE_PIXELS = None

A1_LONG_PX = 9933
GT_POINTS = {"connector", "crossing"}
GT_SKIP = {"background"}
OUR_POINTS = {"junction", "end", "tap"}
OUR_SKIP = {"label", "line_label"}
OUR_EDGES = {"process", "signal", "leader"}
# our node type -> PID2Graph class (equipment split by its kind)
CLASS = {"instrument": "instrumentation", "valve": "valve", "fitting": "general", "arrow": "arrow",
         "connector": "inlet/outlet"}
IOU = 0.5


def our_class(d):
    if d["type"] == "equipment":
        k = (d.get("kind") or "").lower()
        return "pump" if any(w in k for w in ("pump", "compressor", "blower", "fan")) else \
               "tank" if any(w in k for w in ("tank", "vessel", "drum", "column", "tower", "separator")) else "general"
    return CLASS.get(d["type"], d["type"])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def symbol_pairs(G, is_symbol, is_point):
    """{frozenset(u, v)}: symbols joined by a path whose inner nodes are all point nodes."""
    pairs = set()
    for s in (n for n in G if is_symbol(n)):
        seen, stack = {s}, [s]
        while stack:
            a = stack.pop()
            for b in G[a]:
                if b in seen:
                    continue
                seen.add(b)
                if is_symbol(b):
                    pairs.add(frozenset((s, b)))
                elif is_point(b):
                    stack.append(b)
    return {p for p in pairs if len(p) == 2}


def load_gt(path):
    G = nx.Graph(nx.read_graphml(path))
    lab = nx.get_node_attributes(G, "label")
    sym = {n for n in G if lab[n] not in GT_POINTS | GT_SKIP}
    boxes = {n: [G.nodes[n][k] for k in ("xmin", "ymin", "xmax", "ymax")] for n in sym}
    pairs = symbol_pairs(G, lambda n: n in sym, lambda n: lab[n] in GT_POINTS)
    return {n: lab[n] for n in sym}, boxes, pairs


def predict(png, name, k):
    """Phase 1 + SheetGraph on the enlarged sheet; returns the SheetGraph and every bubble used."""
    import tag_pipeline as TP
    from .graph import SheetGraph
    img = Image.open(png).convert("RGB")
    page = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    eng = TP.RapidOCR(); model, names = TP.load_rfdetr(TP.WEIGHTS)
    frags = TP.ocr_page(page, eng)
    px, _, _ = TP.measure_bubble(page, model, names, frags)
    bubbles = TP.detect_multiscale(page, model, names, TP.choose_tile(px))
    TP.attach(frags, bubbles)
    tags = TP.build_tags(frags, bubbles)
    bub, used = {}, []
    for t in tags:
        if t["source"] in ("bubble", "stack") and t.get("bbox"):
            b = t["bbox"] if isinstance(t["bbox"], list) else json.loads(t["bbox"])
            bub.setdefault(t["tag"], [int(v) for v in b]); used.append(b)
    for i, b in enumerate(bubbles):           # bubbles with no readable tag are still instruments on the drawing
        if not any(iou(b["bbox"], u) > 0.5 for u in used):
            bub[f"BUBBLE-{i}"] = [int(v) for v in b["bbox"]]
    frags = [dict(text=f["text"], bbox=[round(v) for v in f["bbox"]]) for f in frags]
    return SheetGraph(name, page=page, frags=frags, bubbles=bub)


def ours(sg, k):
    G = nx.Graph()
    for n, d in sg.G.nodes(data=True):
        if d["type"] not in OUR_SKIP:
            G.add_node(n, **d)
    for u, v, d in sg.G.edges(data=True):
        if d["kind"] in OUR_EDGES and u in G and v in G and u != v:
            G.add_edge(u, v)
    typ = nx.get_node_attributes(G, "type")
    sym = {n for n in G if typ[n] not in OUR_POINTS}
    boxes = {n: [v / k for v in G.nodes[n]["bbox"]] for n in sym}
    pairs = symbol_pairs(G, lambda n: n in sym, lambda n: typ[n] in OUR_POINTS)
    return {n: our_class(G.nodes[n]) for n in sym}, boxes, pairs


def match(gb, pb):
    """Greedy one-to-one by IoU (highest first): {pred id: gt id}."""
    cand = sorted(((iou(a, b), g, p) for g, a in gb.items() for p, b in pb.items()), reverse=True)
    m, used = {}, set()
    for s, g, p in cand:
        if s < IOU:
            break
        if p not in m and g not in used:
            m[p] = g; used.add(g)
    return m


def score(gt, pr):
    gcls, gb, gp = gt; pcls, pb, pp = pr
    m = match(gb, pb)
    found = {frozenset(m[x] for x in p) for p in pp if all(x in m for x in p)}
    tp_pred = sum(1 for p in pp if all(x in m for x in p) and frozenset(m[x] for x in p) in gp)
    detected = set(m.values())
    gp_det = [p for p in gp if p <= detected]
    per = {}
    for c in sorted(set(gcls.values())):
        g = [n for n in gcls if gcls[n] == c]
        per[c] = dict(gt=len(g), found=sum(n in detected for n in g),
                      same_class=sum(1 for p, n in m.items() if gcls[n] == c and pcls[p] == c))
    return dict(gt_symbols=len(gb), pred_symbols=len(pb), matched=len(m),
                node_recall=len(m) / max(1, len(gb)), node_precision=len(m) / max(1, len(pb)),
                class_agree=sum(pcls[p] == gcls[g] for p, g in m.items()) / max(1, len(m)),
                gt_pairs=len(gp), pred_pairs=len(pp), pair_tp=len(found & gp),
                edge_recall=len(found & gp) / max(1, len(gp)), edge_precision=tp_pred / max(1, len(pp)),
                traced_recall=len(found & set(gp_det)) / max(1, len(gp_det)), gt_pairs_detected=len(gp_det),
                per_class=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="pidgraph_output/p2g")
    a = ap.parse_args()
    folder, out = Path(a.folder), ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ids = a.only or sorted((p.stem for p in folder.glob("*.graphml")), key=int)
    tag = folder.name.replace("PID2Graph ", "").replace(" ", "_").lower()
    rows = []
    for i in ids:
        t0 = time.time()
        png = next(folder / f"{i}{e}" for e in (".png", ".jpg", ".jpeg") if (folder / f"{i}{e}").exists())
        W, H = Image.open(png).size
        k = A1_LONG_PX / max(W, H)
        name = f"p2g_{tag}_{i}"
        try:
            sg = predict(png, name, k)
            from .export import export
            export(sg, out)
            r = score(load_gt(folder / f"{i}.graphml"), ours(sg, k))
        except Exception as e:
            import traceback; traceback.print_exc()
            r = dict(error=f"{type(e).__name__}: {e}")
        r.update(sheet=name, scale=round(k, 2), seconds=round(time.time() - t0))
        rows.append(r)
        print(json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in r.items() if x != "per_class"}), flush=True)
        json.dump(rows, open(out / f"scores_{tag}.json", "w"), indent=1)
    ok = [r for r in rows if "error" not in r]
    if ok:
        tot = lambda k: sum(r[k] for r in ok)
        pooled = dict(sheets=len(ok), node_recall=tot("matched") / tot("gt_symbols"),
                      node_precision=tot("matched") / tot("pred_symbols"),
                      edge_recall=tot("pair_tp") / tot("gt_pairs"),
                      traced_recall=sum(r["traced_recall"] * r["gt_pairs_detected"] for r in ok) / max(1, tot("gt_pairs_detected")),
                      edge_precision=sum(r["edge_precision"] * r["pred_pairs"] for r in ok) / max(1, tot("pred_pairs")))
        cls = defaultdict(Counter)
        for r in ok:
            for c, v in r["per_class"].items():
                cls[c].update(v)
        pooled["per_class_recall"] = {c: f"{v['found']}/{v['gt']}" for c, v in cls.items()}
        print("POOLED", json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in pooled.items()}), flush=True)
        json.dump(dict(sheets=rows, pooled=pooled), open(out / f"scores_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
