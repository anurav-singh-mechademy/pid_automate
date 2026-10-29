"""Sheet graph from the PDF's own vector paths, for pages plotted from CAD with layers (survey: batch/survey_pdfs.py).
Same node / edge types as pidgraph.graph.SheetGraph, so pidgraph.query and descgraph work on it unchanged.
    .venv/bin/python -m pidgraph.vector "SC62D1913_p0" ... [--out descgraph_output/vector_graphs]

What is read from the file (no inference): every path's layer, its segments, stroke width and fill.
  LAYER_ROLE (data)  layer name -> pipe / signal / instrument / inline / equipment; other layers are ignored
  pipes      line segments on pipe layers, joined where end points meet and where an end point lies on another
             segment (T); a pipe is never joined where two pipes only cross
  bubbles    closed circles on instrument layers; named by the phase-1 tag whose box lies inside, else by the
             AutoCAD SHX text inside, else BUBBLE-n
  inline     connected groups of paths on inline layers (valves, fittings): pipe ends touching one join through it
  equipment  connected groups of paths on equipment layers; tag / name / kind copied from the image graph's equipment
             (pidgraph.graph, OCR labels) that overlaps it - the outline is exact, the name is read from text
  leaders    a pipe-layer line, or an instrument-layer line outside the circles, ending on a bubble's circle is that
             bubble's process connection (leader edge)
  signal     pieces on signal layers joined where their ends are within SIGNAL_GAP_PT; every object a joined line
             touches is connected to the others by a signal edge
  flow       small filled triangles lying on a pipe segment: arrowheads; the segment's flow points to the apex
Coordinates are written in 300 dpi page pixels like SheetGraph."""
import argparse, json, math, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import networkx as nx
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LAYER_ROLE = {"PROCESS": "pipe", "PROCESS (SECONDARY)": "pipe", "D-PRIM": "pipe", "D-AUXL": "pipe",
              "INSTRUMENT LINE": "signal", "D-SIGN": "signal",
              "INSTRUMENT": "instrument", "D-INST": "instrument",
              "MECHANICAL": "inline", "D-MISC": "inline",
              "EQUIPMENT": "equipment", "D-EQPM": "equipment"}
K = 300 / 72                  # pt -> 300 dpi px
SNAP_PT = 1.0                 # end points closer than this are one point
T_PT = 1.0                    # an end point this close to a segment's interior makes a T junction
TOUCH_PT = 1.5                # a pipe end this close to a symbol / equipment box joins it
SIGNAL_GAP_PT = 4.0           # dash pieces of one signal line are at most this far apart
BUBBLE_PT = (25, 80)          # bubble diameters (pt)
ARROW_PT = (2, 14)            # arrowhead size (pt)
EQ_BODY_PT = 30               # an equipment path group this large is a body; smaller groups are stubs / details
EQ_STUB_PT = 20               # a stub this close to a body belongs to it


def role(layer):
    return LAYER_ROLE.get((layer or "").strip().upper())


def page_of(name):
    """(pdf path, page number, rotation) of a bench sheet, from the batch ledger."""
    import csv
    from descgen.bench import SHEETS
    run = SHEETS[name][0] if name in SHEETS else "pids_bhcp_output"
    led = {r["out_name"]: r for r in csv.DictReader(open(ROOT / run / "ledger.csv"))}[name]
    return led["file"], int(led["page"]), int(led["rotation"] or 0)


def _groups(rects, gap):
    """Union of boxes that touch (within gap) -> list of member index lists."""
    par = list(range(len(rects)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    order = sorted(range(len(rects)), key=lambda i: rects[i][0])
    for a_i, i in enumerate(order):
        a = rects[i]
        for j in order[a_i + 1:]:
            b = rects[j]
            if b[0] > a[2] + gap:
                break
            if b[1] <= a[3] + gap and a[1] <= b[3] + gap:
                par[find(i)] = find(j)
    g = defaultdict(list)
    for i in range(len(rects)):
        g[find(i)].append(i)
    return list(g.values())


def _proj(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return 0.0, math.dist(p, a)
    u = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    q = (a[0] + max(0, min(1, u)) * dx, a[1] + max(0, min(1, u)) * dy)
    return u, math.dist(p, q)


class VectorGraph:
    def __init__(self, name, image_graph=None, bubbles=None):
        self.name = name
        pdf, pno, rot = page_of(name)
        if rot:
            raise ValueError(f"{name}: page was auto-rotated by phase 1 ({rot}); vector reading supports unrotated pages")
        pg = pymupdf.open(pdf)[pno]
        self.page = pg
        dr = pg.get_drawings()
        self.layers = sorted({d.get("layer") or "" for d in dr})
        by = defaultdict(list)
        for d in dr:
            r = role(d.get("layer"))
            if r:
                by[r].append(d)
        if not by.get("pipe"):
            raise ValueError(f"{name}: no pipe layer among {self.layers}")
        self.G = nx.MultiDiGraph(sheet=name, source="vector paths (CAD layers)", layers=" | ".join(self.layers))
        self.n = 0
        self._bubbles(by.get("instrument", []), pg, bubbles or {})
        self._boxes(by.get("inline", []), "valve", 0.5, "inline symbol (path group on %s layer)")
        self._boxes(by.get("equipment", []), "equipment", 2.0, "equipment outline (path group on %s layer)", image_graph)
        # lines drawn on instrument layers outside every bubble are process connections (leaders) too
        lead = []
        for d in by.get("instrument", []):
            items = [it for it in d["items"] if it[0] == "l" and math.dist((it[1].x, it[1].y), (it[2].x, it[2].y)) > 2]
            keep = [it for it in items if all(math.dist(((it[1].x + it[2].x) / 2, (it[1].y + it[2].y) / 2), ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2))
                                              > (c[2] - c[0]) / 2 + 0.5 for c in self.circles)]
            if keep:
                lead.append(dict(d, items=keep))
        self._pipes(by["pipe"] + lead)
        self._arrows(dr)
        self._signals(by.get("signal", []))

    # ---------------------------------------------------------------- objects
    def add(self, nid, type_, bbox_pt, source, **kw):
        b = [round(v * K, 1) for v in bbox_pt]
        self.G.add_node(nid, type=type_, bbox=b, x=round((b[0] + b[2]) / 2, 1), y=round((b[1] + b[3]) / 2, 1), source=source, **kw)
        return nid

    def _bubbles(self, paths, pg, phase1):
        shx = [(a.info.get("content", ""), a.rect) for a in (pg.annots() or []) if a.info.get("title") == "AutoCAD SHX Text"]
        self.circles = []
        for d in paths:
            r = d["rect"]
            if sum(i[0] == "c" for i in d["items"]) >= 4 and abs(r.width - r.height) < 1.5 and BUBBLE_PT[0] <= r.width <= BUBBLE_PT[1]:
                if any(abs(r.x0 - c[0]) < 1 and abs(r.y0 - c[1]) < 1 for c in self.circles):
                    continue                      # the same circle drawn twice
                self.circles.append((r.x0, r.y0, r.x1, r.y1))
        used = set()
        for k, c in enumerate(self.circles):
            cx, cy = (c[0] + c[2]) / 2 * K, (c[1] + c[3]) / 2 * K
            tag = next((t for t, b in phase1.items() if t not in used and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), None)
            how = "phase-1 tag of the bubble inside this circle"
            if not tag:
                inside = sorted((a for a in shx if c[0] <= (a[1].x0 + a[1].x1) / 2 <= c[2] and c[1] <= (a[1].y0 + a[1].y1) / 2 <= c[3]),
                                key=lambda a: a[1].y0)
                txt = [t for t, _ in inside]
                L = next((t for t in txt if re.fullmatch(r"[A-Z]{1,6}", t)), None)
                N = next((t for t in txt if re.fullmatch(r"\d{2,5}[A-Z]?(-\d+)?", t)), None)
                tag, how = (f"{L}-{N}", "AutoCAD SHX text inside the circle") if L and N else (f"BUBBLE-{k}", "no text read")
            used.add(tag)
            from descgen.describe import parse_tag
            p = parse_tag(tag) or {}
            self.add(f"inst:{tag}", "instrument", c, f"circle on instrument layer; tag: {how}", tag=tag,
                     letters=p.get("letters") or "", loop=p.get("number") or "", suffix=p.get("suffix") or "")

    def _boxes(self, paths, type_, gap, src, image_graph=None):
        rects = [(d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1) for d in paths]
        out = []
        groups = _groups(rects, gap)
        if type_ == "equipment":
            # nozzle stubs and small pieces drawn apart from the body belong to the nearest large outline
            box = lambda g: (min(rects[i][0] for i in g), min(rects[i][1] for i in g), max(rects[i][2] for i in g), max(rects[i][3] for i in g))
            big = [g for g in groups if max(box(g)[2] - box(g)[0], box(g)[3] - box(g)[1]) >= EQ_BODY_PT]
            for g in [g for g in groups if g not in big]:
                b = box(g)
                near = [(max(0, max(B[0] - b[2], b[0] - B[2])) + max(0, max(B[1] - b[3], b[1] - B[3])), h)
                        for h in big for B in [box(h)]]
                near = [x for x in near if x[0] <= EQ_STUB_PT]
                if near:
                    min(near, key=lambda x: x[0])[1].extend(g)
                else:
                    big.append(g)
            groups = big
        for g in groups:
            b = (min(rects[i][0] for i in g), min(rects[i][1] for i in g), max(rects[i][2] for i in g), max(rects[i][3] for i in g))
            if type_ == "valve" and max(b[2] - b[0], b[3] - b[1]) > 60:
                continue
            if type_ == "equipment" and max(b[2] - b[0], b[3] - b[1]) < 10:
                continue
            kw = {}
            if type_ == "equipment":
                kw = dict(tag="", name="", kind="", name_source="")
                if image_graph is not None:     # name from the image graph's equipment overlapping this outline
                    best = None
                    for m, d in image_graph.nodes(data=True):
                        if d.get("type") != "equipment" or not (d.get("tag") or d.get("name")) or d.get("kind") in ("shape", "panel"):
                            continue
                        e = [v / K for v in d["bbox"]]
                        ov = max(0, min(b[2], e[2]) - max(b[0], e[0])) * max(0, min(b[3], e[3]) - max(b[1], e[1]))
                        if ov > 0 and (best is None or ov > best[0]):
                            best = (ov, d)
                    if best:
                        d = best[1]
                        kw = dict(tag=d.get("tag", ""), name=d.get("name", ""), kind=d.get("kind", ""),
                                  name_source=f"image graph equipment label ({d.get('name_source', '')})")
            layer = paths[g[0]].get("layer")
            nid = self.add(f"{'sym' if type_ == 'valve' else 'eq'}:v{self.n}", type_, b, src % layer,
                           **({"cls": "inline", "conf": 1.0} if type_ == "valve" else kw))
            self.n += 1
            out.append((nid, b))
        setattr(self, "valves" if type_ == "valve" else "equipments", out)

    # ---------------------------------------------------------------- pipes
    def _pipes(self, paths):
        segs = []
        for d in paths:
            for it in d["items"]:
                if it[0] == "l":
                    a, b = (it[1].x, it[1].y), (it[2].x, it[2].y)
                    if math.dist(a, b) > 0.3:
                        segs.append([a, b, d.get("width") or 0, d.get("layer")])
        # T junctions: split a segment where another segment's end point lies on its interior
        ends = [p for s in segs for p in s[:2]]
        out = []
        for a, b, w, lay in segs:
            cuts = sorted({round(u, 4) for p in ends for u, dd in [_proj(p, a, b)] if 1e-3 < u < 1 - 1e-3 and dd < T_PT})
            pts = [a] + [(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])) for u in cuts] + [b]
            out += [(p, q, w, lay) for p, q in zip(pts, pts[1:])]
        # snap points
        key = {}
        def pid(p):
            g = (round(p[0] / SNAP_PT), round(p[1] / SNAP_PT))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    k = (g[0] + dx, g[1] + dy)
                    if k in key and math.dist(key[k][1], p) <= SNAP_PT:
                        return key[k][0]
            nid = f"pt:{len(key)}"; key[g] = (nid, p)
            return nid
        self.pt = {}
        for p, q, w, lay in out:
            u, v = pid(p), pid(q)
            if u == v:
                continue
            for n, c in ((u, p), (v, q)):
                self.pt.setdefault(n, c)
            self.G.add_edge(u, v, kind="process", polyline=[[round(p[0] * K, 1), round(p[1] * K, 1)], [round(q[0] * K, 1), round(q[1] * K, 1)]],
                            pen=round(w, 2), layer=lay, flow="unknown", flow_source="", line_number="", line_number_source="")
        for n, c in self.pt.items():
            self.add(n, "junction", (c[0] - 0.5, c[1] - 0.5, c[0] + 0.5, c[1] + 0.5), "pipe-layer segment end point")
        # objects absorb the points that touch them
        owner = {}
        for n, c in self.pt.items():
            hit = None
            for k, ci in enumerate(self.circles):
                cx, cy, r = (ci[0] + ci[2]) / 2, (ci[1] + ci[3]) / 2, (ci[2] - ci[0]) / 2
                if abs(math.dist(c, (cx, cy)) - r) <= TOUCH_PT or math.dist(c, (cx, cy)) < r:
                    hit = next(m for m, d in self.G.nodes(data=True) if d["type"] == "instrument" and d["bbox"][0] == round(ci[0] * K, 1)
                               and d["bbox"][1] == round(ci[1] * K, 1)); break
            if not hit:
                for nid, b in getattr(self, "valves", []) + getattr(self, "equipments", []):
                    if b[0] - TOUCH_PT <= c[0] <= b[2] + TOUCH_PT and b[1] - TOUCH_PT <= c[1] <= b[3] + TOUCH_PT:
                        hit = nid; break
            if hit:
                owner[n] = hit
        for n, o in owner.items():
            for u, v, d in list(self.G.in_edges(n, data=True)) + list(self.G.out_edges(n, data=True)):
                a, b = (o if u == n else u), (o if v == n else v)
                if a != b:
                    self.G.add_edge(a, b, **d)
            self.G.remove_node(n)
        # a pipe-layer line from a bubble is its process connection: leader edge bubble -> the point it reaches
        for u, v, k, d in list(self.G.edges(keys=True, data=True)):
            tu, tv = self.G.nodes[u]["type"], self.G.nodes[v]["type"]
            if d["kind"] == "process" and "instrument" in (tu, tv) and tu != tv:
                inst, other = (u, v) if tu == "instrument" else (v, u)
                self.G.remove_edge(u, v, k)
                if self.G.nodes[other]["type"] == "junction":
                    self.G.nodes[other]["type"] = "tap"
                self.G.add_edge(inst, other, kind="leader", polyline=d["polyline"], primary=True, length_mm=0,
                                source="pipe-layer line from the bubble's circle (vector)")
        # bend points (exactly two pipe edges, nothing else) stay as junctions; isolated points dropped
        for n in [n for n, d in self.G.nodes(data=True) if d["type"] == "junction" and self.G.degree(n) == 0]:
            self.G.remove_node(n)
        for n, d in self.G.nodes(data=True):
            if d["type"] == "junction" and self.G.degree(n) == 1:
                d["type"] = "end"

    def _arrows(self, dr):
        """Filled triangles (3-4 corner closed paths) on a pipe segment: flow points to the apex."""
        segs = [(u, v, k, d) for u, v, k, d in self.G.edges(keys=True, data=True) if d["kind"] == "process"]
        n_arrow = 0
        for d in dr:
            if not d.get("fill") or not (ARROW_PT[0] <= max(d["rect"].width, d["rect"].height) <= ARROW_PT[1]):
                continue
            pts = []
            for it in d["items"]:
                if it[0] == "l":
                    pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
                elif it[0] == "qu":
                    q = it[1]; pts += [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
            corners = []
            for p in pts:
                if not any(math.dist(p, c) < 0.3 for c in corners):
                    corners.append(p)
            if not 3 <= len(corners) <= 5:      # a triangle, also drawn as two half-triangles (4 corners)
                continue
            cen = (sum(c[0] for c in corners) / len(corners), sum(c[1] for c in corners) / len(corners))
            for u, v, k, e in segs:
                if not self.G.has_edge(u, v, k) or e.get("flow") == "known":
                    continue
                a, b = [(x / K, y / K) for x, y in e["polyline"]]
                t, dist = _proj(cen, a, b)
                if dist < 2.0:
                    # apex: the corner farthest from the arrow's centre
                    apex = max(corners, key=lambda c: math.dist(c, cen))
                    ta, _ = _proj(apex, a, b)
                    fwd = ta > t
                    e.update(flow="known", flow_source="filled arrowhead on the pipe (vector)")
                    if not fwd:                    # store the edge in flow direction
                        self.G.remove_edge(u, v, k); self.G.add_edge(v, u, **e)
                    n_arrow += 1
                    break
        self.G.graph["arrows"] = n_arrow

    def _signals(self, paths):
        pieces = []
        for d in paths:
            for it in d["items"]:
                if it[0] == "l":
                    pieces.append(((it[1].x, it[1].y), (it[2].x, it[2].y)))
                elif it[0] == "c":
                    pieces.append(((it[1].x, it[1].y), (it[4].x, it[4].y)))
        rects = [(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])) for a, b in pieces]
        objs = [(m, [v / K for v in d["bbox"]]) for m, d in self.G.nodes(data=True) if d["type"] in ("instrument", "valve", "equipment")]
        n_edges = 0
        for g in _groups(rects, SIGNAL_GAP_PT):
            ends = [p for i in g for p in pieces[i]]
            touched = []
            for m, b in objs:
                if any(b[0] - 2 <= p[0] <= b[2] + 2 and b[1] - 2 <= p[1] <= b[3] + 2 for p in ends) and m not in touched:
                    touched.append(m)
            for a, b in zip(touched, touched[1:]):
                self.G.add_edge(a, b, kind="signal", source="joined dash pieces on the signal layer (vector)")
                n_edges += 1
        self.G.graph["signal_edges"] = n_edges


def build(name, out, image_graph_path=None):
    from descgen.bench import load_sheet
    _, _, bubbles, _ = load_sheet(name)
    ig = pickle.load(open(image_graph_path, "rb")) if image_graph_path and Path(image_graph_path).exists() else None
    vg = VectorGraph(name, ig, bubbles)
    if ig is not None:
        vg.G.graph.update({k: v for k, v in ig.graph.items() if k.startswith("title") or k == "profile"})
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(vg.G, open(out / f"{name}.gpickle", "wb"))
    overlay(vg, out / f"{name}_overlay.png")
    return vg


def overlay(vg, path, dpi=100):
    from PIL import Image, ImageDraw
    pix = vg.page.get_pixmap(dpi=dpi)
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L").convert("RGB")
    im = Image.blend(im, Image.new("RGB", im.size, "white"), 0.6)
    s = dpi / 300
    d = ImageDraw.Draw(im)
    col = {"process": (30, 90, 255), "signal": (230, 0, 200), "leader": (0, 170, 0)}
    G = vg.G
    for u, v, e in G.edges(data=True):
        if e["kind"] == "signal":
            a, b = G.nodes[u], G.nodes[v]
            d.line([(a["x"] * s, a["y"] * s), (b["x"] * s, b["y"] * s)], fill=col["signal"], width=1)
        else:
            pl = [(x * s, y * s) for x, y in e["polyline"]]
            d.line(pl, fill=(200, 30, 30) if e.get("flow") == "known" else col[e["kind"]], width=2)
    for n, a in G.nodes(data=True):
        b = [v * s for v in a["bbox"]]
        c = {"instrument": (0, 140, 0), "valve": (220, 0, 0), "equipment": (255, 140, 0), "tap": (0, 170, 0), "end": (120, 120, 120)}.get(a["type"])
        if c and a["type"] in ("instrument", "valve", "equipment"):
            d.rectangle(b, outline=c, width=2)
    im.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--out", default="descgraph_output/vector_graphs")
    ap.add_argument("--image-graphs", default="descgraph_output/graphs")
    a = ap.parse_args()
    for n in a.sheets:
        vg = build(n, ROOT / a.out, ROOT / a.image_graphs / f"{n}.gpickle")
        t = defaultdict(int)
        for _, d in vg.G.nodes(data=True):
            t[d["type"]] += 1
        e = defaultdict(int)
        for *_, d in vg.G.edges(data=True):
            e[d["kind"]] += 1
        print(n, dict(t), dict(e), "arrows", vg.G.graph.get("arrows"), flush=True)
