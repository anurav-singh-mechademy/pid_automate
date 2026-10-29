"""Exports: GraphML (list / tuple / dict attributes as JSON strings), an overlay PNG, a JSON summary."""
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from descgen.sheet import MM
from . import schema as C
from .validate import validate, PROBLEM_KEYS


def _plain(v):
    if isinstance(v, (list, tuple, dict, set)):
        return json.dumps(list(v) if isinstance(v, set) else v, default=str)
    if v is None:
        return ""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def to_graphml(G, path):
    import networkx as nx
    H = nx.MultiDiGraph()
    H.graph.update({k: _plain(v) for k, v in G.graph.items()})
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: _plain(v) for k, v in d.items()})
    for u, v, k, d in G.edges(keys=True, data=True):
        H.add_edge(u, v, key=k, **{a: _plain(b) for a, b in d.items()})
    nx.write_graphml(H, path)


def _dashed(img, pts, col, th, dash=14, gap=9):
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(L // (dash + gap)))
        for i in range(n + 1):
            t0, t1 = i * (dash + gap) / max(L, 1), min(1, (i * (dash + gap) + dash) / max(L, 1))
            if t0 >= 1:
                break
            p = (int(a[0] + t0 * (b[0] - a[0])), int(a[1] + t0 * (b[1] - a[1])))
            q = (int(a[0] + t1 * (b[0] - a[0])), int(a[1] + t1 * (b[1] - a[1])))
            cv2.line(img, p, q, col, th, cv2.LINE_AA)


def overlay(sg, rep, path, scale=C.OVERLAY_SCALE):
    G = sg.G
    W, H = sg.s.img.size
    base = np.asarray(sg.s.img.convert("RGB").resize((int(W * scale), int(H * scale)), Image.LANCZOS)).astype(np.float32)
    img = (255 - (255 - base) * 0.45).astype(np.uint8).copy()          # faded drawing under the graph
    S = lambda p: (int(p[0] * scale), int(p[1] * scale))
    col = {k: tuple(int(c) for c in v) for k, v in C.COLOURS.items()}
    th = max(2, int(0.5 * MM * scale))
    for u, v, d in G.edges(data=True):
        poly = [S(p) for p in d.get("polyline", [])]
        if d["kind"] == "process":
            cv2.polylines(img, [np.array(poly, np.int32)], False, col["process"], th, cv2.LINE_AA)
            if d.get("flow") in ("known", "conflict") and len(poly) >= 2:       # flow arrow at the middle
                L = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:])]
                k = int(np.argmax(L)); a, b = poly[k], poly[k + 1]
                m = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                dd = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
                q = (int(m[0] + 9 * (b[0] - a[0]) / dd), int(m[1] + 9 * (b[1] - a[1]) / dd))
                cv2.arrowedLine(img, (int(m[0] - 9 * (b[0] - a[0]) / dd), int(m[1] - 9 * (b[1] - a[1]) / dd)), q,
                                (0, 0, 120), th + 1, cv2.LINE_AA, tipLength=0.8)
        elif d["kind"] == "signal":
            _dashed(img, poly, col["signal"], th)
        elif d["kind"] == "leader":
            cv2.polylines(img, [np.array(poly, np.int32)], False, col["leader"], th, cv2.LINE_AA)
        elif d["kind"] == "labels":
            a, b = G.nodes[u], G.nodes[v]
            cv2.line(img, S((a["x"], a["y"])), S((b["x"], b["y"])), col["labels"], 1, cv2.LINE_AA)
    for n, d in G.nodes(data=True):
        t = d["type"]; b = d["bbox"]
        p0, p1 = S(b[:2]), S(b[2:])
        if t == "equipment":
            cv2.rectangle(img, p0, p1, col["equipment"], th + 1)
        elif t in ("valve", "fitting", "connector"):
            cv2.rectangle(img, p0, p1, col[t], th)
        elif t == "arrow":
            cv2.rectangle(img, p0, p1, col["arrow"], 1)
            if d.get("tip"):
                cv2.arrowedLine(img, S(d["base"]), S(d["tip"]), col["arrow"], th, tipLength=0.6)
        elif t == "instrument":
            c = S((d["x"], d["y"])); r = int((b[2] - b[0]) * scale / 2)
            cv2.circle(img, c, r, col["instrument"], th)
        elif t in ("junction", "tap"):
            cv2.circle(img, S((d["x"], d["y"])), th + 2, col[t], -1)
        elif t == "end":
            c = S((d["x"], d["y"])); r = th + 3
            cv2.line(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), col["end"], 2)
            cv2.line(img, (c[0] - r, c[1] + r), (c[0] + r, c[1] - r), col["end"], 2)
    for k in PROBLEM_KEYS:
        for p in rep.get(k, []):
            if p.get("xy"):
                cv2.circle(img, S(p["xy"]), int(4 * MM * scale), col["problem"], th + 1, cv2.LINE_AA)
    _legend(img, col)
    Image.fromarray(img).save(path, optimize=True)
    return path


def _legend(img, col):
    items = [("process", "process pipe"), ("signal", "signal (dashed)"), ("leader", "leader"), ("equipment", "equipment"),
             ("valve", "valve"), ("fitting", "fitting"), ("arrow", "flow arrow"), ("connector", "connector"),
             ("instrument", "instrument"), ("problem", "problem")]
    x, y = 20, 20
    cv2.rectangle(img, (x - 10, y - 10), (x + 330, y + 30 * len(items)), (255, 255, 255), -1)
    for i, (k, t) in enumerate(items):
        cv2.rectangle(img, (x, y + 30 * i), (x + 40, y + 30 * i + 18), col[k], -1)
        cv2.putText(img, t, (x + 55, y + 30 * i + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def summary(sg, rep):
    G = sg.G
    E = [d for _, _, d in G.edges(data=True)]
    proc = [d for d in E if d["kind"] == "process"]
    return dict(sheet=sg.name, seconds=sg.seconds, profile=G.graph["profile"], pen_mode=G.graph["pen_mode"],
                nodes=dict(Counter(d["type"] for _, d in G.nodes(data=True))),
                edges=dict(Counter(d["kind"] for d in E)),
                process=dict(edges=len(proc), length_m=round(sum(d["length_mm"] for d in proc) / 1000, 2),
                             flow_known=sum(d["flow"] == "known" for d in proc),
                             flow_conflict=sum(d["flow"] == "conflict" for d in proc),
                             with_line_number=sum(bool(d["line_number"]) for d in proc)),
                symbols=dict(Counter(d["cls"] for _, d in G.nodes(data=True) if d["type"] in ("valve", "fitting", "arrow"))),
                validation=rep["counts"],
                loops_mixed=[dict(instruments=l["instruments"], loops=l["loops"]) for l in rep.get("loops_mixed", [])],
                build_notes=sg.notes)


def export(sg, out="pidgraph_output"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rep = validate(sg)
    to_graphml(sg.G, out / f"{sg.name}.graphml")
    overlay(sg, rep, out / f"{sg.name}_overlay.png")
    summ = summary(sg, rep)
    json.dump(dict(summary=summ, validation={k: v for k, v in rep.items() if k != "counts"}), open(out / f"{sg.name}.json", "w"),
              indent=1, default=str)
    return summ, rep
