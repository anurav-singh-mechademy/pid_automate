"""Dashed instrument-signal lines as straight runs, and the point graph joining them (T / L meetings, collinear breaks).

A dash is a short thin horizontal / vertical stroke (schema.DASH_MM long, about the thin pen wide); dashes in a row
with regular small gaps are one dashed run. Text and bubbles are already whited out on the ink used (Pipes.ink), so
letters and bubble outlines are not read as dashes. Runs that cross a solid line or a jump symbol lose a dash or
two: collinear runs up to DASH_BRIDGE_MM apart are one line."""
import math
from collections import defaultdict
import numpy as np
import cv2
from descgen.sheet import MM
from . import schema as C


def dashed_runs(ink, thin_px, panel_x=1e9):
    """[dict(o='h'|'v', c, a, b, n)]: dashed straight runs (c = axis coordinate, a..b = extent, n = dashes)."""
    lo, hi = C.DASH_MM[0] * MM, C.DASH_MM[1] * MM
    g0, g1 = C.DASH_GAP_MM[0] * MM, C.DASH_GAP_MM[1] * MM
    out = []
    for o in ("h", "v"):
        k = np.ones((1, int(lo)), np.uint8) if o == "h" else np.ones((int(lo), 1), np.uint8)
        m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        dashes = []
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            along, across = (w, h) if o == "h" else (h, w)
            if lo <= along <= hi and across <= 2.5 * thin_px and x < panel_x:
                dashes.append((y + h / 2, x, x + w - 1) if o == "h" else (x + w / 2, y, y + h - 1))
        rows = defaultdict(list)
        for d in dashes:
            rows[int(d[0] // (0.6 * MM))].append(d)
        seen = set()
        for key in sorted(rows):
            cand = sorted(rows[key] + rows.get(key + 1, []), key=lambda d: d[1])
            cand = [d for d in cand if d not in seen]
            run = []
            def flush():
                if len(run) >= C.DASH_MIN:
                    out.append(dict(o=o, c=float(np.mean([d[0] for d in run])), a=float(run[0][1]),
                                    b=float(run[-1][2]), n=len(run)))
                    seen.update(run)
            for d in cand:
                if run and (g0 <= d[1] - run[-1][2] <= g1) and abs(d[0] - run[-1][0]) <= 0.6 * MM:
                    run.append(d)
                elif run and d[1] - run[-1][2] < g0 and abs(d[0] - run[-1][0]) <= 0.6 * MM:
                    run[-1] = (run[-1][0], run[-1][1], max(run[-1][2], d[2]))   # one dash split in two
                else:
                    flush(); run = [d]
            flush()
    return _merge_collinear(out)


def _merge_collinear(runs):
    """Runs on one axis line with a short break between them (a crossing solid line hides a dash) are one run."""
    out = []
    for o in ("h", "v"):
        rs = sorted((r for r in runs if r["o"] == o), key=lambda r: (round(r["c"] / (0.8 * MM)), r["a"]))
        used = [False] * len(rs)
        for i, r in enumerate(rs):
            if used[i]:
                continue
            cur = dict(r); used[i] = True
            grown = True
            while grown:
                grown = False
                for j, s in enumerate(rs):
                    if not used[j] and abs(s["c"] - cur["c"]) <= 0.8 * MM and \
                            s["a"] - cur["b"] <= C.DASH_BRIDGE_MM * MM and s["b"] >= cur["a"]:
                        cur["b"] = max(cur["b"], s["b"]); cur["n"] += s["n"]; used[j] = True; grown = True
            out.append(cur)
    return out


def pt(r, pos):
    return (pos, r["c"]) if r["o"] == "h" else (r["c"], pos)


def point_graph(runs):
    """Point graph of the dashed runs: points (run index, position); edges along runs and at T / L meetings."""
    st = defaultdict(set)
    for i, r in enumerate(runs):
        st[i].update((r["a"], r["b"]))
    links = []
    tol = C.DASH_JOIN_MM * MM
    H = [i for i, r in enumerate(runs) if r["o"] == "h"]
    V = [i for i, r in enumerate(runs) if r["o"] == "v"]
    for i in H:
        h = runs[i]
        for j in V:
            v = runs[j]
            x, y = v["c"], h["c"]
            dh = max(h["a"] - x, 0, x - h["b"]); dv = max(v["a"] - y, 0, y - v["b"])
            if dh > tol or dv > tol:
                continue
            endh = min(abs(x - h["a"]), abs(x - h["b"])) <= tol
            endv = min(abs(y - v["a"]), abs(y - v["b"])) <= tol
            if not (endh or endv):
                continue                    # dashed lines crossing: not joined
            ph, pv = min(max(x, h["a"]), h["b"]), min(max(y, v["a"]), v["b"])
            st[i].add(ph); st[j].add(pv)
            links.append(((i, ph), (j, pv)))
    xy = {(i, p): pt(runs[i], p) for i in st for p in st[i]}
    edges = []
    for i, ps in st.items():
        ps = sorted(ps)
        for a, b in zip(ps, ps[1:]):
            edges.append(((i, a), (i, b), dict(kind="run", run=i)))
    for p, q in links:
        edges.append((p, q, dict(kind="link")))
    return xy, edges


def ends(runs, xy, edges):
    """Dangling points (degree 1) with their outward heading."""
    deg = defaultdict(int)
    for p, q, _ in edges:
        deg[p] += 1; deg[q] += 1
    out = {}
    for p, d in deg.items():
        if d == 1:
            i, pos = p
            r = runs[i]
            sign = -1 if abs(pos - r["a"]) < abs(pos - r["b"]) else 1
            out[p] = (sign, 0) if r["o"] == "h" else (0, sign)
    return out
