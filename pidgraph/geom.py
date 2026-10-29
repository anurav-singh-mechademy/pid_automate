"""Geometry helpers of the sheet graph: boxes, polylines, contracting a point graph into object-to-object chains."""
import math
from collections import defaultdict
import numpy as np


def grow(b, e):
    return (b[0] - e, b[1] - e, b[2] + e, b[3] + e)


def inside(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def box_dist(p, b):
    return math.hypot(max(b[0] - p[0], 0, p[0] - b[2]), max(b[1] - p[1], 0, p[1] - b[3]))


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def seg_hits_box(p, q, b):
    """Does the straight piece p-q pass through box b (Liang-Barsky clip)?"""
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pv, qv in ((-dx, p[0] - b[0]), (dx, b[2] - p[0]), (-dy, p[1] - b[1]), (dy, b[3] - p[1])):
        if pv == 0:
            if qv < 0:
                return False
        else:
            t = qv / pv
            if pv < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def project(poly, p):
    """(distance from p to the polyline, arc length of the nearest point, total length)."""
    best, s0 = (1e18, 0.0), 0.0
    for a, b in zip(poly, poly[1:]):
        ax, ay = a; bx, by = b
        L = math.hypot(bx - ax, by - ay)
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)) / (L * L)))
        x, y = ax + t * (bx - ax), ay + t * (by - ay)
        d = math.hypot(p[0] - x, p[1] - y)
        if d < best[0]:
            best = (d, s0 + t * L)
        s0 += L
    if len(poly) == 1:
        return math.hypot(p[0] - poly[0][0], p[1] - poly[0][1]), 0.0, 0.0
    return best[0], best[1], s0


def length(poly):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:]))


def simplify(poly, tol=1.0):
    """Drop collinear interior points (orthogonal pipes: keep bends only)."""
    if len(poly) <= 2:
        return [tuple(map(float, p)) for p in poly]
    out = [poly[0]]
    for i in range(1, len(poly) - 1):
        a, b, c = out[-1], poly[i], poly[i + 1]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol * max(1.0, math.hypot(c[0] - a[0], c[1] - a[1])):
            out.append(b)
    out.append(poly[-1])
    return [(float(x), float(y)) for x, y in out]


def contract(xy, edges, owner, keep=()):
    """Point graph -> chains between key points.
    xy: {point: (x, y)}; edges: [(p, q, attrs)]; owner: {point: object node id} (points absorbed into an object);
    keep: points that must stay nodes (taps). Key nodes = objects, kept points, points of degree != 2.
    Returns [(u, v, [points along the chain], [edge attrs along it])] with u, v = object id or point."""
    def nid(p):
        return owner.get(p, p)
    adj = defaultdict(list)
    for k, (p, q, a) in enumerate(edges):
        u, v = nid(p), nid(q)
        if u == v:
            continue                       # inside one object
        adj[u].append((v, k, p, q)); adj[v].append((u, k, q, p))
    key = {n for n in adj if not isinstance(n, tuple) or n in keep or len(adj[n]) != 2}
    used, chains = set(), []
    def walk(start, first):
        nb, k, p_from, p_to = first
        pts = [p_from, p_to]; att = [edges[k][2]]; ks = [k]; cur = nb
        while cur not in key:
            nxt = [e for e in adj[cur] if e[1] not in used]
            if not nxt:
                break
            nb, k, p_from, p_to = nxt[0]
            used.add(k); ks.append(k); att.append(edges[k][2])
            if p_from != pts[-1]:
                pts.append(p_from)
            pts.append(p_to); cur = nb
        return cur, pts, att
    for s in sorted(key, key=str):
        for e in adj[s]:
            if e[1] in used:
                continue
            used.add(e[1])
            end, pts, att = walk(s, e)
            chains.append((s, end, pts, att))
    # closed rings of pass-through points (no key node): cut at one point
    for n in sorted(adj, key=str):
        for e in adj[n]:
            if e[1] in used:
                continue
            key.add(n); used.add(e[1])
            end, pts, att = walk(n, e)
            chains.append((n, end, pts, att))
    return chains
