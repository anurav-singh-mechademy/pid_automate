"""Drawn lines as graphs: the pipe graph an instrument's tapping point is followed along, and the ink network that
measures how far (along any drawn line) an instrument is from each equipment. Flow arrows are read on both.

Pipes  (pipe following from a tap; terminals = connectors / equipment reached in each direction)
  1. pen widths measured per sheet (run-length histogram): process lines are drawn with a heavier pen than signal
     lines and outlines; a sheet drawn with one pen uses every solid line longer than a dash;
  2. horizontal / vertical straight segments (P&ID pipes are orthogonal); graph with T / L junctions (an X crossing
     drawn without a break is not a junction) and collinear gaps bridged: up to INLINE_GAP_MM always (flanges,
     crossing breaks, small valves), up to LONG_GAP_MM when in-line symbols fill the gap (a valve train);
  3. tap(): the pipe point an impulse line ends on, or straight on along its heading (root valve, seal symbol);
  4. follow(): shortest-path search from the tap, separately in the two directions along the tapped pipe, with a
     penalty for turning off a straight run into a side branch; stops at terminals: an equipment outline, a dangling
     pipe end with a connector / equipment just beyond it, or one of the caller's terminal boxes.
Network  (distance along every drawn line; filled flow arrows)
  all ink at 1/F resolution as an 8-neighbour pixel graph, small breaks and crossing breaks between long thick runs
  bridged; Dijkstra floods from an equipment give the distance of every point along the lines."""
import heapq
import math
import re
from collections import Counter, defaultdict
import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .sheet import MM
from .trace import walk

INLINE_GAP_MM = 6       # collinear gaps always bridged (flanges, crossing breaks, small valves)
LONG_GAP_MM = 45        # longer collinear gaps bridged when in-line symbols fill them (a valve train)
HEAVY_SHARE = (0.06, 0.3)   # heavy-pen share of all solid line length on a sheet drawn with a pipe pen
SOLID_MIN_MM = 6        # single-pen sheets: shortest solid run taken as a line (longer than a dash)
JOIN_TOL_MM = 0.6       # an end within this of another pipe's edge is joined to it
HOP_MM = 25             # an in-line symbol joining two pipe ends fits within this distance of each end
CORNER_GAP_MM = 4       # a bend whose two ends stop this short of the corner is still a bend (arrowhead at a corner)
BRANCH_PENALTY_MM = 40  # leaving a straight run into a side branch costs this much path length
MAX_PATH_MM = 900
EQUIP_LABEL_MM = 40     # an equipment tag is printed inside its outline or within this distance of it
ARROW_MM = (1.2, 8)     # size of a filled flow arrowhead (longest side)


def pen_widths(ink):
    """(thin, thick) stroke widths in px: the two peaks of the horizontal+vertical run-length histogram."""
    C = Counter()
    for arr in (ink, ink.T):
        d = np.diff(np.pad(arr[::3], ((0, 0), (1, 1))).astype(np.int8), axis=1)
        s, e = np.where(d == 1), np.where(d == -1)
        C.update(np.clip(e[1] - s[1], 0, 40).tolist())
    thin = max(range(2, 9), key=lambda L: C[L])
    thick = max(range(math.ceil(1.6 * thin), 3 * thin + 1), key=lambda L: C[L])
    return thin, thick


def bridge_flanges(ink):
    """Flanged joints are drawn as two short parallel bars across the pipe with a ~1 mm gap and no line between them:
    join every such pair of bars (a dashed line's dashes are end to end, never side by side, so they are not joined)."""
    for axis in (0, 1):     # 0: bars horizontal (pipe vertical), 1: bars vertical (pipe horizontal)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.5 * MM), 1) if axis == 0 else (1, int(1.5 * MM)))
        bars = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(bars, connectivity=8)
        short = [i for i in range(1, n) if 1.5 * MM <= st[i, 2 if axis == 0 else 3] <= 8 * MM
                 and st[i, 3 if axis == 0 else 2] <= 0.8 * MM]
        cells = {}
        for i in short:
            x, y = st[i, :2]
            cells.setdefault((int(x // (8 * MM)), int(y // (8 * MM))), []).append(i)
        for i in short:
            x, y, bw, bh = st[i, :4]
            cx, cy = int(x // (8 * MM)), int(y // (8 * MM))
            for j in (j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((cx + dx, cy + dy), []) if j > i):
                x2, y2, bw2, bh2 = st[j, :4]
                if axis == 0:
                    ov, gap = min(x + bw, x2 + bw2) - max(x, x2), max(y2 - (y + bh), y - (y2 + bh2))
                    if ov > 0.7 * min(bw, bw2) and 0 < gap <= 2 * MM:
                        c = int((max(x, x2) + min(x + bw, x2 + bw2)) / 2)
                        ink[min(y, y2):max(y + bh, y2 + bh2), c - 1:c + 2] = 1
                else:
                    ov, gap = min(y + bh, y2 + bh2) - max(y, y2), max(x2 - (x + bw), x - (x2 + bw2))
                    if ov > 0.7 * min(bh, bh2) and 0 < gap <= 2 * MM:
                        c = int((max(y, y2) + min(y + bh, y2 + bh2)) / 2)
                        ink[c - 1:c + 2, min(x, x2):max(x + bw, x2 + bw2)] = 1
    return ink


def blank_bubbles(sheet, ink):
    """White out every instrument bubble on the page: bubbles are found from their OCR text (letters over a loop
    number) and confirmed by a circle of ink around it."""
    nums = [f for f in sheet.frags if re.fullmatch(r"\d{3,5}[A-Z]?", f["text"])]
    lets = [f for f in sheet.frags if re.fullmatch(r"[A-Z]{1,6}", f["text"])]
    for n in nums:
        nb = n["bbox"]
        lt = [l for l in lets if abs((l["bbox"][0] + l["bbox"][2]) / 2 - (nb[0] + nb[2]) / 2) < 4 * MM
              and 0 < nb[1] - l["bbox"][1] < 7 * MM]
        if not lt:
            continue
        cx, cy = (nb[0] + nb[2]) / 2, nb[1]
        r = _circle_radius(sheet.ink, cx, cy)
        if r:
            cv2.circle(ink, (int(cx), int(cy)), int(r + 0.6 * MM), 0, -1)
    return ink


def _circle_radius(ink, cx, cy):
    """Radius of a bubble outline centred at (cx, cy): ink on most of a ring (5 to 9 mm across), else None."""
    h, w = ink.shape
    for rr in np.arange(2.5, 4.8, 0.15) * MM:
        hit = 0
        for a in np.radians(np.arange(0, 360, 10)):
            x, y = int(cx + rr * math.cos(a)), int(cy + rr * math.sin(a))
            hit += 0 <= x < w and 0 <= y < h and ink[max(0, y - 2):y + 3, max(0, x - 2):x + 3].any()
        if hit / 36 > 0.85:
            return rr
    return None


class Pipes:
    def __init__(self, sheet, plant, single_pen=False):
        self.s, self.p = sheet, plant
        ink = sheet.ink.copy()
        for f in sheet.frags:                 # text is not pipe: white out the OCR boxes
            x0, y0, x1, y1 = f["bbox"]
            if x1 - x0 > (y1 - y0) * (len(f["text"]) + 1):
                continue                     # a box far wider than its characters: OCR read a symbol as text
            m = max(2, int(0.2 * (y1 - y0)))     # shrunk: a label's box often grazes the line it sits on
            ink[y0 + m:y1 - m, x0 + m:x1 - m] = 0
        blank_bubbles(sheet, ink)            # signal lines must not join two pipes through an instrument bubble
        self.ink = ink
        # what in-line symbol hops walk on: the drawing (text kept: OCR boxes often cover flange bars) without the
        # bubbles, flange pairs closed
        self.joined = bridge_flanges(blank_bubbles(sheet, sheet.ink.copy()))
        self.thin, self.thick = pen_widths(sheet.ink)
        k = max(3, int(round((self.thin + self.thick) / 2)))
        heavy = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        self.segs = self._segments(heavy, 3)
        self.mode = "heavy pen"
        solid = self._segments(ink, SOLID_MIN_MM)
        share = sum(s["b"] - s["a"] for s in self.segs) / max(1, sum(s["b"] - s["a"] for s in solid))
        if single_pen or not HEAVY_SHARE[0] < share < HEAVY_SHARE[1]:
            # this drafting style draws process lines with the same pen as everything else: every solid line long
            # enough not to be a dash or a letter stroke
            self.segs = solid
            self.mode = "single pen"
        white = (1 - sheet.ink).astype(np.uint8)        # every closed white area of the page, labelled once
        _, self.wlab, self.wst, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
        self._mark_outlines()
        self._graph()
        self._symbol_hops()
        self._term_cache = {}

    def _segments(self, mask, min_mm):
        L = int(min_mm * MM)
        H, W = mask.shape
        out = []
        for o, ker in (("h", np.ones((1, L), np.uint8)), ("v", np.ones((L, 1), np.uint8))):
            m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                x, y, w, h = st[i, :4]
                along, across = (w, h) if o == "h" else (h, w)
                if across > 2.5 * MM or along < L or along > 0.8 * (W if o == "h" else H):   # blobs, drawing frame
                    continue
                if o == "h":
                    out.append(dict(o="h", c=y + h / 2, a=x, b=x + w - 1, half=h / 2))
                else:
                    out.append(dict(o="v", c=x + w / 2, a=y, b=y + h - 1, half=w / 2))
        return out

    def region(self, x, y):
        """Bbox of the closed white area at (x, y); None for open drawing space (an area touching the page edge)."""
        H, W = self.wlab.shape
        xi, yi = int(x), int(y)
        if not (0 <= xi < W and 0 <= yi < H):
            return None
        k = self.wlab[yi, xi]
        if k == 0:
            return None
        bx, by, bw, bh = self.wst[k, :4]
        if bx == 0 or by == 0 or bx + bw >= W or by + bh >= H or bw > 0.5 * W or bh > 0.5 * H:
            return None
        return (bx, by, bx + bw, by + bh)

    def _mark_outlines(self):
        """Segments that are the outline of a labelled equipment shape: reaching one = reaching that equipment."""
        for s in self.segs:
            s["eq"] = None
            if s["b"] - s["a"] < 8 * MM:
                continue
            m = (s["a"] + s["b"]) / 2
            for side in (-1, 1):
                off = s["half"] + 1.5 * MM
                x, y = (m, s["c"] + side * off) if s["o"] == "h" else (s["c"] + side * off, m)
                reg = self.region(x, y)
                if reg and min(reg[2] - reg[0], reg[3] - reg[1]) > 12 * MM:
                    lab = self.p.equipment_at_region(reg, mm=15)
                    if lab:
                        s["eq"] = lab[1]["tag"]; s["eq_region"] = reg
                        break

    def _band(self, s, p0, p1):
        """Fraction of positions between p0 and p1 along s's axis with some ink within 2.5 mm of the axis."""
        r = int(2.5 * MM); c = int(s["c"])
        lo, hi = int(min(p0, p1)), int(max(p0, p1))
        if hi - lo < 2:
            return 1.0
        band = self.ink[c - r:c + r + 1, lo:hi] if s["o"] == "h" else self.ink[lo:hi, c - r:c + r + 1].T
        return float(band.any(axis=0).mean()) if band.size else 0.0

    # ---------- graph ----------
    @staticmethod
    def pt(s, pos):
        return (pos, s["c"]) if s["o"] == "h" else (s["c"], pos)

    def _graph(self):
        S = self.segs
        tol = JOIN_TOL_MM * MM
        self.st = defaultdict(set)            # seg -> station positions
        self.adj = defaultdict(list)          # (seg, pos) -> [((seg, pos), cost)]
        self.linked_end = set()               # (seg, 'a'|'b') ends that continue somewhere
        for i, s in enumerate(S):
            self.st[i].update((s["a"], s["b"]))
        H = [i for i, s in enumerate(S) if s["o"] == "h"]
        V = [i for i, s in enumerate(S) if s["o"] == "v"]
        for i in H:
            h = S[i]
            for j in V:
                v = S[j]
                xv, yh = v["c"], h["c"]
                if not (h["a"] - v["half"] - tol <= xv <= h["b"] + v["half"] + tol and
                        v["a"] - h["half"] - tol <= yh <= v["b"] + h["half"] + tol):
                    continue
                endh = min(abs(xv - h["a"]), abs(xv - h["b"])) <= v["half"] + tol
                endv = min(abs(yh - v["a"]), abs(yh - v["b"])) <= h["half"] + tol
                if not (endh or endv):
                    continue                  # an X crossing drawn without a break: not connected
                ph, pv = min(max(xv, h["a"]), h["b"]), min(max(yh, v["a"]), v["b"])
                self.st[i].add(ph); self.st[j].add(pv)
                # side-branch penalty: going from a run's middle into the end of the other line
                self.adj[(i, ph)].append(((j, pv), 0 if endh else BRANCH_PENALTY_MM * MM if endv else 0))
                self.adj[(j, pv)].append(((i, ph), 0 if endv else BRANCH_PENALTY_MM * MM if endh else 0))
                if endh:
                    self.linked_end.add((i, "a" if abs(xv - h["a"]) < abs(xv - h["b"]) else "b"))
                if endv:
                    self.linked_end.add((j, "a" if abs(yh - v["a"]) < abs(yh - v["b"]) else "b"))
        for i in H:                           # a bend with a gap at its corner (arrowhead, elbow fitting drawn apart)
            h = S[i]
            for j in V:
                v = S[j]
                xv, yh = v["c"], h["c"]
                gh = min(abs(xv - h["a"]), abs(xv - h["b"]))
                gv = min(abs(yh - v["a"]), abs(yh - v["b"]))
                if gh <= v["half"] + tol and gv <= h["half"] + tol:
                    continue                  # joined above
                if not (gh <= CORNER_GAP_MM * MM and gv <= CORNER_GAP_MM * MM):
                    continue
                if not (h["a"] - CORNER_GAP_MM * MM <= xv <= h["b"] + CORNER_GAP_MM * MM and
                        v["a"] - CORNER_GAP_MM * MM <= yh <= v["b"] + CORNER_GAP_MM * MM):
                    continue
                eh = "a" if abs(xv - h["a"]) < abs(xv - h["b"]) else "b"
                ev = "a" if abs(yh - v["a"]) < abs(yh - v["b"]) else "b"
                if (i, eh) in self.linked_end or (j, ev) in self.linked_end:
                    continue                  # that end already continues
                self.adj[(i, h[eh])].append(((j, v[ev]), gh + gv))
                self.adj[(j, v[ev])].append(((i, h[eh]), gh + gv))
                self.linked_end.add((i, eh)); self.linked_end.add((j, ev))
        for grp in (H, V):                    # collinear gaps (in-line symbols, crossing breaks): next segment only
            for i in grp:
                s = S[i]
                nxt = [(S[j]["a"] - s["b"], j) for j in grp if j != i and abs(S[j]["c"] - s["c"]) <= 0.8 * MM
                       and S[j]["b"] > s["b"] and S[j]["a"] > s["b"] - 1 * MM]
                if not nxt:
                    continue
                gap, j = min(nxt)
                t = S[j]
                if gap <= INLINE_GAP_MM * MM or gap <= LONG_GAP_MM * MM and self._band(s, s["b"], t["a"]) > 0.9:
                    self.adj[(i, s["b"])].append(((j, t["a"]), max(gap, 0)))
                    self.adj[(j, t["a"])].append(((i, s["b"]), max(gap, 0)))
                    self.linked_end.add((i, "b")); self.linked_end.add((j, "a"))

    def _symbol_hops(self):
        """A pipe interrupted by an in-line symbol drawn with its own outline (sight glass box, reducer, valve, an
        arrowhead at a tee): from each dangling segment end, the ink connected to it within HOP_MM leads to the pipe
        segments it touches; the end is joined to the nearest point of each (cost = straight distance)."""
        S, R = self.segs, int(HOP_MM * MM)
        H, W = self.ink.shape
        for i, s in enumerate(S):
            for end in ("a", "b"):
                if (i, end) in self.linked_end:
                    continue
                ex, ey = self.pt(s, s[end])
                sign = -1 if end == "a" else 1          # outward along the segment
                ox, oy = (sign, 0) if s["o"] == "h" else (0, sign)
                x0, y0, x1, y1 = max(0, int(ex) - R), max(0, int(ey) - R), min(W, int(ex) + R), min(H, int(ey) + R)
                n, lab = cv2.connectedComponents(self.joined[y0:y1, x0:x1], connectivity=8)
                px, py = int(ex) - x0, int(ey) - y0
                near = lab[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
                own = set(near[near > 0].tolist())
                if not own:
                    continue
                for j, t in enumerate(S):
                    if j == i:
                        continue
                    lo, hi = (x0, x1) if t["o"] == "h" else (y0, y1)
                    c = t["c"]
                    if not ((y0 <= c < y1) if t["o"] == "h" else (x0 <= c < x1)) or t["b"] < lo or t["a"] > hi:
                        continue
                    best = None
                    for q in range(int(max(t["a"], lo)), int(min(t["b"], hi - 1)) + 1, 2):
                        qx, qy = self.pt(t, q)
                        if lab[int(qy) - y0, int(qx) - x0] in own and (qx - ex) * ox + (qy - ey) * oy > -1 * MM:
                            dd = math.hypot(qx - ex, qy - ey)
                            if best is None or dd < best[0]:
                                best = (dd, q)
                    if best is None or best[0] < 0.5 * MM:
                        continue
                    q = min(max(best[1], t["a"]), t["b"])
                    self.st[j].add(q)
                    self.adj[(i, s[end])].append(((j, q), best[0]))
                    self.adj[(j, q)].append(((i, s[end]), best[0]))
                    self.linked_end.add((i, end))

    def _along(self, i, pos, st=None):
        ps = sorted((st or self.st)[i])
        k = ps.index(pos)
        out = []
        if k > 0:
            out.append(((i, ps[k - 1]), pos - ps[k - 1]))
        if k < len(ps) - 1:
            out.append(((i, ps[k + 1]), ps[k + 1] - pos))
        return out

    # ---------- tapping point ----------
    def tap(self, pts, max_mm=20):
        """Pipe point where an instrument line ends: the leader's last point, or straight on along its heading
        (seal / root-valve symbols sit between the impulse line and the pipe). Returns (seg, pos, point, mm) or None."""
        (xa, ya), (xb, yb) = pts[max(0, len(pts) - 4)], pts[-1]
        h = math.atan2(yb - ya, xb - xa)
        # the heading over the last 3 mm (a line ending in a small fitting box or hook turns in its last points)
        k = len(pts) - 1
        while k > 0 and math.hypot(pts[k][0] - xb, pts[k][1] - yb) < 3 * MM:
            k -= 1
        if k < len(pts) - 4:
            h = math.atan2(yb - pts[k][1], xb - pts[k][0])
        for d in np.arange(0, max_mm, 0.5) * MM:
            x, y = xb + d * math.cos(h), yb + d * math.sin(h)
            best = None
            for i, s in enumerate(self.segs):
                if abs(math.cos(h) if s["o"] == "h" else math.sin(h)) > 0.7:
                    continue                  # a line running the leader's way is the impulse line itself
                if s["b"] - s["a"] < 10 * MM and not ((i, "a") in self.linked_end and (i, "b") in self.linked_end):
                    continue                  # an isolated short stroke: a symbol edge (seal, valve), not a pipe
                u, w = (x, y) if s["o"] == "h" else (y, x)
                if s["a"] - 1 * MM <= u <= s["b"] + 1 * MM and abs(w - s["c"]) <= s["half"] + 1.0 * MM:
                    e = abs(w - s["c"])
                    if best is None or e < best[0]:
                        best = (e, i, min(max(u, s["a"]), s["b"]))
            if best:
                return best[1], best[2], (x, y), d / MM
        # the traced line turned into a pipe / outline and ran along it: its end lies on that line
        for i, s in enumerate(self.segs):
            u, w = (xb, yb) if s["o"] == "h" else (yb, xb)
            if s["b"] - s["a"] >= 10 * MM and s["a"] <= u <= s["b"] and abs(w - s["c"]) <= s["half"] + 1.0 * MM:
                return i, min(max(u, s["a"]), s["b"]), (xb, yb), 0.0
        return None

    # ---------- terminals ----------
    def terminal(self, i, end):
        """What lies just beyond a dangling segment end: an off-sheet connector (its text within 25 mm ahead) or a
        labelled equipment shape (directly, or at the end of a lighter connecting line)."""
        key = (i, end)
        if key in self._term_cache:
            return self._term_cache[key]
        s = self.segs[i]
        ex, ey = self.pt(s, s[end])
        dx, dy = ((-1, 0) if end == "a" else (1, 0)) if s["o"] == "h" else ((0, -1) if end == "a" else (0, 1))
        res, best = None, None
        for c in self.p.connectors:
            if not (c["flow"] in ("in", "out") or c["tags"]) or (c["instruments"] and not c["tags"] and not c["service"]):
                continue                     # a drain / vent, or a signal connector to another instrument
            x0, y0, x1, y1 = c["bbox"]
            dist = math.hypot(max(x0 - ex, 0, ex - x1), max(y0 - ey, 0, ey - y1)) / MM
            ahead = ((x0 + x1) / 2 - ex) * dx + ((y0 + y1) / 2 - ey) * dy
            if dist < 25 and (ahead > 0 or dist < 4) and (best is None or dist < best[0]):
                best = (dist, c)
        if best:
            res = dict(kind="connector", conn=best[1], dist_mm=best[0])
        else:
            res = self._equipment_beyond(ex, ey, dx, dy)
            if res is None:
                # the heavy line stops at a tie point; the connection to the machine goes on in a lighter pen
                pts, how = walk(self.s.ink, ex + dx * 1.0 * MM, ey + dy * 1.0 * MM, math.atan2(dy, dx), max_mm=60)
                if how == "object" and len(pts) > 2:
                    (xa, ya), (xb, yb) = pts[-2], pts[-1]
                    n = math.hypot(xb - xa, yb - ya) or 1
                    res = self._equipment_beyond(xb, yb, (xb - xa) / n, (yb - ya) / n)
                    if res:
                        res["via"] = "light line %.0f mm" % (sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
                                                                   for k in range(len(pts) - 1)) / MM)
        self._term_cache[key] = res
        return res

    def _equipment_beyond(self, ex, ey, dx, dy):
        """A large closed shape just beyond a line end, with an equipment tag inside / beside it."""
        for t in np.arange(0.8, 15, 0.8) * MM:
            reg = self.region(ex + dx * t, ey + dy * t)
            if reg and max(reg[2] - reg[0], reg[3] - reg[1]) > 12 * MM:
                lab = self.p.equipment_at_region(reg, mm=EQUIP_LABEL_MM)
                if lab:
                    return dict(kind="equipment", tag=lab[1]["tag"], region=reg)
                break
        return None

    # ---------- follower ----------
    def follow(self, i, pos, boxes=None, max_mm=MAX_PATH_MM):
        """From the tapping point (seg i at pos), search each direction along the tapped pipe separately.
        boxes: optional [(bbox, tol_mm, ref)] terminals of the caller (a node inside one stops the walk there).
        Returns {'-': [(path_mm, terminal, point)...], '+': [...]} in order of path length; terminal is
        dict(kind='equipment' | 'connector' | 'box', ..., path=[points from the tap])."""
        st = {k: set(v) for k, v in self.st.items()} if boxes else self.st
        inbox = {}
        if boxes:
            # stations where a segment enters a terminal box, so the walk can stop there
            for bi, (b, tol, _) in enumerate(boxes):
                e = tol * MM
                x0, y0, x1, y1 = b[0] - e, b[1] - e, b[2] + e, b[3] + e
                for j, s in enumerate(self.segs):
                    lo, hi, c = (x0, x1, s["c"]) if s["o"] == "h" else (y0, y1, s["c"])
                    if not ((y0 <= c <= y1) if s["o"] == "h" else (x0 <= c <= x1)) or s["b"] < lo or s["a"] > hi:
                        continue
                    for q in (max(s["a"], lo), min(s["b"], hi)):
                        st[j].add(q)
            def box_at(p):
                for bi, (b, tol, _) in enumerate(boxes):
                    e = tol * MM + 0.5
                    if b[0] - e <= p[0] <= b[2] + e and b[1] - e <= p[1] <= b[3] + e:
                        return bi
                return None
        had = pos in st[i]
        st[i].add(pos)
        start_box = box_at(self.pt(self.segs[i], pos)) if boxes else None
        if start_box is not None:       # tapped right on a terminal (its nozzle): that is where it is, both ways
            t = dict(kind="box", ref=boxes[start_box][2], path=[])
            p0 = self.pt(self.segs[i], pos)
            return {"-": [(0.0, t, p0)], "+": [(0.0, t, p0)]}
        # both directions searched together: a point belongs to the direction that reaches it first, so a loop back
        # to the other side does not make one terminal appear on both sides
        ps = sorted(st[i])
        k = ps.index(pos)
        heap, seen, prev = [], {(i, pos): None}, {}
        found = {"-": [], "+": []}
        got_box = {"-": set(), "+": set()}
        if k > 0:
            heapq.heappush(heap, (abs(ps[k - 1] - pos), (i, ps[k - 1]), (i, pos), "-"))
        if k + 1 < len(ps):
            heapq.heappush(heap, (abs(ps[k + 1] - pos), (i, ps[k + 1]), (i, pos), "+"))
        def path_to(n):
            out_ = []
            while n in prev:
                out_.append(self.pt(self.segs[n[0]], n[1])); n = prev[n]
            return out_[::-1]
        while heap:
            d, node, frm, side = heapq.heappop(heap)
            if node in seen or d > max_mm * MM:
                continue
            seen[node] = side; prev[node] = frm
            j, q = node
            sj = self.segs[j]
            p = self.pt(sj, q)
            if boxes:
                bi = box_at(p)
                if bi is not None:
                    if bi not in got_box[side]:
                        got_box[side].add(bi)
                        found[side].append((d / MM, dict(kind="box", ref=boxes[bi][2], path=path_to(node)), p))
                    continue
            if sj["eq"] and j != i:
                found[side].append((d / MM, dict(kind="equipment", tag=sj["eq"], region=sj["eq_region"], via="outline",
                                                 path=path_to(node)), p))
                continue
            for e in ("a", "b"):
                if q == sj[e] and (j, e) not in self.linked_end:
                    t = self.terminal(j, e)
                    if t:
                        found[side].append((d / MM, dict(t, path=path_to(node)), p))
            for nb, c in self._along(j, q, st) + self.adj.get(node, []):
                if nb not in seen:
                    heapq.heappush(heap, (d + c, nb, node, side))
        out = found
        if not had and not boxes:
            st[i].discard(pos)
        return out


class Network:
    """All ink at 1/F resolution as a graph (8-neighbour pixels); shortest paths along it in mm; filled arrowheads."""
    F = 4                      # 1 px = 0.34 mm: fast, keeps connectivity

    def __init__(self, ink, gap_mm=0.0, break_mm=6.0):
        F = self.F
        h, w = ink.shape
        H, W = h // F, w // F
        small = ink[:H * F, :W * F].reshape(H, F, W, F).max(axis=(1, 3)).astype(np.uint8)
        g = int(round(gap_mm * MM / F / 2))
        if g:   # bridge the small breaks drafters leave in lines (flange pairs, line crossings, valve ends)
            small = cv2.dilate(small, np.ones((3, 3), np.uint8), iterations=g)
        small = small.astype(bool)
        thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # pipes: lines >= 0.4 mm wide
        self.thick_full = thick
        self.thick = thick[:H * F, :W * F].reshape(H, F, W, F).max(axis=(1, 3)).astype(bool)
        self.thick_near = cv2.dilate(self.thick.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=g + 1).astype(bool)
        self.shape = (H, W)
        self.idx = -np.ones((H, W), np.int64)
        ys, xs = np.nonzero(small)
        self.idx[ys, xs] = np.arange(len(ys))
        self.ys, self.xs = ys, xs
        rows, cols, wts = [], [], []
        for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            a = self.idx[0:H - dy, max(0, -dx):W - max(0, dx)]
            b = self.idx[dy:H, max(0, dx):W + min(0, dx)]
            m = (a >= 0) & (b >= 0)
            rows.append(a[m]); cols.append(b[m]); wts.append(np.full(m.sum(), wt))
        # drafters break one of two crossing pipes and leave gaps at flange pairs: bridge straight gaps between long
        # collinear thick runs, ignoring short runs (the crossing line) inside the gap
        th = self.thick
        lmin, gmax = int(3 * MM / F), int(break_mm * MM / F)
        for arr, swap in ((th, False), (th.T, True)):
            for y in range(arr.shape[0]):
                row = arr[y]
                if not row.any():
                    continue
                d = np.diff(np.r_[0, row.astype(np.int8), 0])
                st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
                long_ = [(a, b) for a, b in zip(st, en) if b - a >= lmin]
                for (a1, b1), (a2, b2) in zip(long_, long_[1:]):
                    if 0 < a2 - b1 <= gmax:
                        p, q = ((b1 - 1, y), (a2, y)) if swap else ((y, b1 - 1), (y, a2))
                        i, j = self.idx[p], self.idx[q]
                        if i >= 0 and j >= 0:
                            rows.append([i]); cols.append([j]); wts.append([float(a2 - b1 + 1)])
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        self.G = csr_matrix((v, (r, c)), shape=(len(ys), len(ys)))
        # filled blobs (arrowheads, filled valve bodies), labelled; widened ~1.5 mm so a path along a line's edge meets them
        k = int(round(0.93 * MM)) | 1
        filled = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        _, self.blob_lab, self.blob_st, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        self.blob_near = cv2.dilate(self.blob_lab.astype(np.float32), np.ones((3, 3), np.uint8),
                                    iterations=int(1.5 * MM)).astype(np.int32)
        self._arrows = {}

    def nodes_in(self, mask_fn, box):
        """Graph nodes inside a full-resolution box that satisfy mask_fn(x, y) (full-res coords)."""
        F = self.F
        x0, y0, x1, y1 = [int(v // F) for v in box]
        H, W = self.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        sub = self.idx[y0:y1 + 1, x0:x1 + 1]
        yy, xx = np.nonzero(sub >= 0)
        return [sub[y, x] for y, x in zip(yy, xx) if mask_fn((x + x0) * F + F / 2, (y + y0) * F + F / 2)]

    def flood(self, seeds):
        d, pred, _ = dijkstra(self.G, directed=False, indices=seeds, min_only=True, return_predecessors=True)
        return d, pred

    def path(self, pred, node):
        out = [node]
        while pred[out[-1]] >= 0 and len(out) < 100000:
            out.append(pred[out[-1]])
        return out                                  # from `node` back to the seed

    def xy(self, node):
        return (self.xs[node] * self.F + self.F / 2, self.ys[node] * self.F + self.F / 2)

    def node_at(self, x, y, r=4):
        H, W = self.shape
        gx, gy = int(x // self.F), int(y // self.F)
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if 0 <= gy + dy < H and 0 <= gx + dx < W and self.idx[gy + dy, gx + dx] >= 0:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, self.idx[gy + dy, gx + dx])
        return best[1] if best else None

    # ---------- flow arrows ----------
    def arrow(self, l):
        """Filled blob l as a flow arrow: (tip, base) points, or None when it is not a triangle on a pipe."""
        if l in self._arrows:
            return self._arrows[l]
        res = None
        x, y, w, h, area = self.blob_st[l]
        if ARROW_MM[0] * MM < max(w, h) < ARROW_MM[1] * MM and area < 40 * MM * MM:
            yy, xx = np.nonzero(self.blob_lab[y:y + h, x:x + w] == l)
            P_ = np.stack([xx + x, yy + y], 1).astype(float)
            c = P_.mean(0)
            ev, evec = np.linalg.eigh(np.cov((P_ - c).T))
            ax = evec[:, 1]                                    # the arrow's long axis
            sproj = (P_ - c) @ ax
            L = sproj.max() - sproj.min()
            if L >= 1.2 * MM and ev[1] >= 1.2 * ev[0]:         # not too small, not round (a dot)
                # a filled triangle is wide at its base and narrow at its tip: compare the first and last quarter
                q = L / 4
                lo, hi = np.sum(sproj < sproj.min() + q), np.sum(sproj > sproj.max() - q)
                ratio = (lo + 1) / (hi + 1)
                if ratio > 1.4 or ratio < 1 / 1.4:
                    tip = c + ax * (sproj.max() if ratio > 1 else sproj.min())
                    base = c + ax * (sproj.min() if ratio > 1 else sproj.max())
                    u = (tip - base) / max(np.linalg.norm(tip - base), 1e-6)
                    # a flow arrow sits on a pipe: the thick line carries on beyond its tip or its base
                    if any(self._thick_at(q) for q in (tip + u * 1.5 * MM, base - u * 1.5 * MM)):
                        res = (tip, base)
        self._arrows[l] = res
        return res

    def _thick_at(self, q, r=3):
        x, y = int(q[0]), int(q[1])
        t = self.thick_full
        return 0 <= y < t.shape[0] and 0 <= x < t.shape[1] and t[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any()

    @staticmethod
    def _pair_cancel(votes, gap):
        # filled valve bodies are two triangles tip to tip: neighbouring opposite votes cancel
        return [v for v in votes if not [u for u in votes if u is not v and abs(u[0] - v[0]) < gap and u[1] == -v[1]]]

    def arrows_on(self, pts, densify_mm=None):
        """Arrowheads met along a polyline: [(index, +1 pointing along it / -1 pointing back, 'on path')]."""
        pts = np.asarray(pts, float)
        if densify_mm and len(pts) > 1:
            dense = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                n = max(1, int(np.linalg.norm(b - a) / (densify_mm * MM)))
                dense += [a + (b - a) * t / n for t in range(1, n + 1)]
            pts = np.array(dense)
        votes, seen = [], {}
        H, W = self.blob_near.shape
        for i, (x, y) in enumerate(pts):
            if 0 <= int(y) < H and 0 <= int(x) < W:
                l = self.blob_near[int(y), int(x)]
                if l and l not in seen:
                    seen[l] = i
        for l, i in seen.items():
            a = self.arrow(l)
            if a is None:
                continue
            tip, base = a
            it = int(np.argmin(np.hypot(pts[:, 0] - tip[0], pts[:, 1] - tip[1])))
            ib = int(np.argmin(np.hypot(pts[:, 0] - base[0], pts[:, 1] - base[1])))
            if it != ib:
                votes.append((i, 1 if it > ib else -1, "on path"))
        return self._pair_cancel(votes, 25 if not densify_mm else 25 * 0.34 / densify_mm)

    def arrows_near(self, dl, d):
        """The nearest arrowhead on the lines reached from a point (distances dl) that lies along a route to a target
        (distances d): [(distance, +1 tip nearer the target / -1, 'near')]."""
        got = np.nonzero(np.isfinite(dl))[0]
        blobs = {}
        for n in got:
            x, y = self.xy(n)
            l = self.blob_near[int(y), int(x)]
            if l and (l not in blobs or dl[n] < blobs[l]):
                blobs[l] = dl[n]
        near = []
        for l, dist in blobs.items():
            a = self.arrow(l)
            if a is None:
                continue
            nt, nb = self.node_at(*a[0]), self.node_at(*a[1])
            if nt is None or nb is None or not (np.isfinite(d[nt]) and np.isfinite(d[nb])):
                continue
            # an arrow lying along a route to the target: tip and base differ by its full length
            if abs(d[nt] - d[nb]) * self.F > 0.85 * np.linalg.norm(a[0] - a[1]):
                near.append((int(dist), 1 if d[nt] < d[nb] else -1, "near"))
        return sorted(self._pair_cancel(near, 10))[:1]
