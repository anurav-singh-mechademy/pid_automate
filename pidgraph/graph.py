"""SheetGraph: one P&ID sheet as a networkx MultiDiGraph with geometry and a source on every node and edge.

Nodes (attribute `type`)
  instrument   phase-1 bubbles: tag, letters, loop number, measured variable (rules.VARIABLE), bbox
  equipment    plant.items (tag, name, kind, outline), sheet.parts (cylinder / motor / flywheel), other printed
               equipment labels with the shape beside them, unlabelled shapes a leader lands on (kind 'shape')
  valve / fitting / arrow   RF-DETR symbols (class, conf, bbox); filled arrowheads found in the ink (lines.Network.arrow)
  connector    plant.connectors: direction TO / FROM / END (drain, vent) / REF, target text, tags, sheet
  junction / end / tap      pipe or signal-line points: T junctions, dangling ends, the point a leader taps a pipe
  line_label / label        printed line numbers and valve tags
Edges (attribute `kind`)
  process  solid pipe between two nodes (lines.Pipes segments + bridged gaps, contracted between objects): polyline,
           pen width, flow direction (arrows / connectors, carried through in-line symbols), line number
  signal   dashed instrument lines between instruments, valves (actuators), equipment, connectors
  leader   bubble -> what its solid line touches (trace.trace_bubble, as describe.leader_of picks lines)
  labels   printed text -> the object it names
Every attribute that is inferred has a `*source` attribute next to it saying where it came from."""
import math
import re
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import cv2
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from descgen.sheet import Sheet, MM
from descgen.trace import trace_bubble
from descgen.describe import parse_tag, leader_of
from descgen.lines import Network
from descgen.text import kind_of
from descgen import rules as R
from . import schema as C
from .detect import detect_symbols
from .geom import grow, inside, box_dist, area, ctr, seg_hits_box, project, length, simplify, contract
from .signal import dashed_runs, point_graph, ends as signal_ends


def r1(v):
    return round(float(v), 1)


class SheetGraph:
    def __init__(self, name, page=None, frags=None, bubbles=None, refresh_symbols=False):
        t0 = time.time()
        if page is None:
            from descgen.bench import load_sheet
            page, frags, bubbles, _ = load_sheet(name)
        self.name = name
        self.s = s = Sheet(page, frags)
        self.plant = s.plant
        self.pipes = s.pipes()
        self.bubbles = bubbles or {}
        self.dets = detect_symbols(name, page, self.bubbles, refresh=refresh_symbols)
        self.G = nx.MultiDiGraph(sheet=name, profile=s.profile["name"], pen_mode=self.pipes.mode,
                                 pens_mm=f"{self.pipes.thin / MM:.2f} / {self.pipes.thick / MM:.2f}")
        self.boxes = {}                 # node id -> bbox, for objects that absorb line points
        self.notes = []                 # build notes (what was skipped and why)
        hdr = [f for f in s.frags if f["text"] in s.profile["panel_labels"]]
        H, W = s.ink.shape
        # the notes / title panel: from its headers down to the page corner (a column or a bottom block)
        self.panel = (s.panel_x, min(f["bbox"][1] for f in hdr) - 4 * MM, W, H) if hdr else None
        if self.panel and self.panel[1] > H / 2:
            self.panel = self._title_strip(self.panel)
        self._instruments()
        self._equipment()
        self._panels()
        self._symbols()
        self._connectors()
        self._leaders()
        self._process()
        self._signal()
        self._leader_edges()
        self._labels()
        self._flow()
        self.seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ helpers
    def _mid(self, e):
        """Middle point of an edge polyline (where a problem on it is drawn)."""
        poly = self.G.edges[e]["polyline"]
        L = length(poly); s0 = 0.0
        for a, b in zip(poly, poly[1:]):
            l = math.hypot(b[0] - a[0], b[1] - a[1])
            if s0 + l >= L / 2 and l:
                t = (L / 2 - s0) / l
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            s0 += l
        return tuple(poly[0])

    def in_panel(self, p):
        return self.panel is not None and inside(p, self.panel)

    def add(self, nid, type_, bbox, source, **kw):
        b = [r1(v) for v in bbox]
        self.G.add_node(nid, type=type_, bbox=b, x=r1((b[0] + b[2]) / 2), y=r1((b[1] + b[3]) / 2), source=source, **kw)
        return nid

    def nodes(self, *types):
        return [n for n, d in self.G.nodes(data=True) if d["type"] in types]

    def smallest_at(self, p, ids, tol_mm):
        best = None
        for n in ids:
            b = self.boxes.get(n) or self.G.nodes[n]["bbox"]
            if inside(p, grow(b, tol_mm * MM)) and (best is None or area(b) < best[0]):
                best = (area(b), n)
        return best[1] if best else None

    # ------------------------------------------------------------------ nodes
    def _instruments(self):
        for tag, b in self.bubbles.items():
            p = parse_tag(tag) or {}
            L = p.get("letters") or ""
            var = R.MODIFIED.get(L[:2]) or R.VARIABLE.get(L[:1]) if L else None
            self.add(f"inst:{tag}", "instrument", b, "phase-1 bubble (RF-DETR box + OCR text)", tag=tag, letters=L,
                     loop=p.get("number") or "", suffix=p.get("suffix") or "", unit=p.get("unit") or "",
                     variable=var or "", variable_source=f"ISA letters {L}" if var else "")

    def _equipment(self):
        s, pl = self.s, self.plant
        seen = Counter()
        def eid(tag):
            seen[tag] += 1
            return f"eq:{tag}" if seen[tag] == 1 else f"eq:{tag}#{seen[tag]}"
        for k, p in enumerate(s.parts):
            nid = self.add(f"eq:part{k}:{p['kind']}", "equipment", p["bbox"], f"labelled machine part '{p['label']}' (sheet.parts)",
                           tag="", name=p["label"], kind=p["kind"], name_source=f"label '{p['label']}' on the shape")
            self.boxes[nid] = p["bbox"]
        for it in pl.items:
            nm = pl.names.get(it["tag"]) if it["tag"] else None
            name = " ".join(nm["words"]) if nm else it["text"]
            kind = it.get("kind") or kind_of((nm or {}).get("words"))
            box, src = it["bbox"], f"equipment symbol: outline next to label {it['tag'] or it['text']} (plant.items)"
            if self._is_bubble(box):
                alt = self._outline_near(it["label"])
                box, src = (alt, src.replace("plant.items", "plant.items gave a bubble; nearest larger shape")) if alt else (it["label"], src + "; outline was a bubble, label only")
            nid = self.add(eid(it["tag"] or it["text"]), "equipment", box, src,
                           tag=it["tag"] or "", name=name, kind=kind or "", label_bbox=[r1(v) for v in it["label"]],
                           name_source=nm["src"] if nm else f"noun printed inside the shape '{it['text']}'")
            if box is not it["label"]:
                self.boxes[nid] = box
        have = {self.G.nodes[n]["tag"] for n in self.nodes("equipment")}
        for l in pl.labels:                     # printed equipment labels with no register item
            t, b = l["tag"], l["bbox"]
            if t in have or self.in_panel(ctr(b)) or any(box_dist(ctr(b), c["bbox"]) < 6 * MM for c in pl.connectors):
                continue
            part = next((n for n in self.nodes("equipment") if self.G.nodes[n]["tag"] == "" and inside(ctr(b), self.boxes[n])), None)
            if part:
                self.G.nodes[part]["tag"] = t
                self.G.nodes[part]["tag_source"] = f"label {t} printed inside the shape"
                have.add(t); continue
            reg = pl.shape_near(b)
            if reg and self._is_bubble(reg):
                reg = self._outline_near(b)
            words = (pl.names.get(t) or {}).get("words") or pl.letter_names.get(t)
            nid = self.add(eid(t), "equipment", reg or b, f"equipment label {t}" + (" with the shape beside it" if reg else " (no outline found)"),
                           tag=t, name=" ".join(words or []), kind=kind_of(words) or "", label_bbox=[r1(v) for v in b],
                           name_source=(pl.names.get(t) or {}).get("src") or ("tag letter code (rules.EQUIP_LETTER)" if words else ""))
            if reg:
                self.boxes[nid] = reg
            have.add(t)

    def _title_strip(self, panel):
        """A title block in the lower half of the sheet often spans the whole width (a strip): the longest ruled row
        within 50 mm above its header, through the panel's left edge, gives the strip's top and left end."""
        ink, (x0, y0, W, H) = self.s.ink, panel
        best = None
        for y in range(int(max(0, y0 - 50 * MM)), int(y0 + 4 * MM)):
            row = ink[y]
            x = int(x0)
            if not row[max(0, x - 3):x + 3].any():
                continue
            l = x
            gap = 0
            while l > 0 and gap < 3:
                gap = 0 if row[l] else gap + 1
                l -= 1
            if best is None or l < best[1]:
                best = (y, l)
        if best and x0 - best[1] > 100 * MM:
            self.notes.append(f"title block is a strip along the bottom (ruled row from x={best[1]:.0f})")
            return (best[1], best[0], W, H)
        return panel

    def _panels(self):
        """Boxes with words inside that no register names (REDAS, junction box, vendor package, purge panel): nodes of
        kind 'panel' that signal lines and tubing end on. Found on a lighter ink threshold (grey boxes count)."""
        g = self.s.gray
        white = (g >= 225).astype(np.uint8)
        _, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
        H, W = g.shape
        taken = [self.boxes[n] for n in self.nodes("equipment") if n in self.boxes]
        k = 0
        seen = set()
        for f in self.s.frags:
            b = f["bbox"]
            if len(f["text"]) < 3 or self.in_panel(ctr(b)) or self._is_bubble(b):
                continue
            x, y = int(b[0] - 0.8 * MM), int(ctr(b)[1])
            if not (0 <= x < W and 0 <= y < H) or lab[y, x] == 0:
                continue
            i = lab[y, x]
            if i in seen:
                continue
            bx, by, bw, bh, ar = st[i]
            r = (bx, by, bx + bw, by + bh)
            if bx == 0 or by == 0 or bx + bw >= W or by + bh >= H or min(bw, bh) < 8 * MM or max(bw, bh) > 150 * MM:
                continue
            if not (r[0] <= b[0] and r[2] >= b[2] and r[1] <= b[1] and r[3] >= b[3]) or self._is_bubble(r):
                continue
            if any(area((max(r[0], q[0]), max(r[1], q[1]), min(r[2], q[2]), min(r[3], q[3]))) > 0.3 * area(r) for q in taken):
                continue
            if ar < 0.85 * bw * bh:
                continue                     # not a box (an open area between lines)
            seen.add(i)
            words = [h["text"] for h in self.s.frags if inside(ctr(h["bbox"]), r)]
            if len(words) > 8:
                continue                     # a table or notes block
            nid = self.add(f"eq:panel{k}", "equipment", r, "closed box with words inside (light-ink white area)", tag="",
                           name=" ".join(words), kind="panel", name_source="words printed inside the box")
            self.boxes[nid] = r; taken.append(r); k += 1

    def _is_bubble(self, b):
        """A box that is an instrument bubble: phase-1 bubbles, or RF-DETR bubbles phase 1 dropped (local devices)."""
        qs = list(self.bubbles.values()) + [d["bbox"] for d in self.dets if d["group"] == "bubble" and d["conf"] >= 0.5]
        return any(area((max(b[0], q[0]), max(b[1], q[1]), min(b[2], q[2]), min(b[3], q[3]))) > 0.4 * area(b) for q in qs)

    def _outline_near(self, lb, max_mm=25):
        """Nearest closed shape beside a label that is not a bubble (an equipment outline), grown like plant.items."""
        cx, cy = ctr(lb)
        for d in np.arange(1.0, max_mm, 1.0) * MM:
            for p in ((lb[2] + d, cy), (lb[0] - d, cy), (cx, lb[3] + d), (cx, lb[1] - d)):
                r = self.pipes.region(*p)
                if r and min(r[2] - r[0], r[3] - r[1]) > 8 * MM and max(r[2] - r[0], r[3] - r[1]) < 300 * MM \
                        and not self._is_bubble(r):
                    return self.plant.grow(r)
        return None

    def _text_boxes(self):
        """Closed boxes drawn around a tag / label text (valve tags, line labels): their edges are not pipes.
        The white area just outside the text, if it is small and hugs the text."""
        pp, out = self.pipes, []
        eqb = [self.boxes[n] for n in self.nodes("equipment", "connector") if n in self.boxes]
        H, W = pp.wlab.shape
        for f in self.s.frags:
            b = f["bbox"]
            for p in ((b[0] - 0.6 * MM, ctr(b)[1]), (ctr(b)[0], b[1] - 0.6 * MM), (ctr(b)[0], b[3] + 0.6 * MM)):
                x, y = int(p[0]), int(p[1])
                if not (0 <= x < W and 0 <= y < H) or pp.wlab[y, x] == 0:
                    continue
                bx, by, bw, bh = pp.wst[pp.wlab[y, x], :4]
                r = (bx, by, bx + bw, by + bh)
                if r[0] <= b[0] and r[2] >= b[2] and r[1] <= b[1] and r[3] >= b[3] and bw < (b[2] - b[0]) + 12 * MM \
                        and bh < 2.6 * (b[3] - b[1]) + 6 * MM and not any(area((max(r[0], q[0]), max(r[1], q[1]), min(r[2], q[2]),
                                                                                 min(r[3], q[3]))) > 0.3 * area(r) for q in eqb):
                    out.append(r)
                    break
        return out

    @staticmethod
    def _tables(cells):
        """Tables: four or more text cells touching each other; the whole grid is ruling, not pipe."""
        n = len(cells)
        par = list(range(n))
        def find(i):
            while par[i] != i:
                par[i] = par[par[i]]; i = par[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                a, b = cells[i], cells[j]
                if max(a[0], b[0]) - min(a[2], b[2]) < 1.5 * MM and max(a[1], b[1]) - min(a[3], b[3]) < 1.5 * MM:
                    par[find(i)] = find(j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(cells[i])
        return [(min(c[0] for c in g), min(c[1] for c in g), max(c[2] for c in g), max(c[3] for c in g))
                for g in groups.values() if len(g) >= 4]

    def _symbols(self):
        bub = list(self.bubbles.values())
        eqb = [self.boxes[n] for n in self.nodes("equipment") if n in self.boxes and self.G.nodes[n]["kind"] != "panel"]
        placed = []
        k = 0
        for d in sorted(self.dets, key=lambda d: -d["conf"]):
            g = d["group"]
            if g not in C.SYMBOL_MIN_CONF or d["conf"] < C.SYMBOL_MIN_CONF[g] or self.in_panel(ctr(d["bbox"])):
                continue
            b = d["bbox"]
            if any(area((max(b[0], q[0]), max(b[1], q[1]), min(b[2], q[2]), min(b[3], q[3]))) > 0.5 * area(b) for q in bub):
                continue                         # a part of an instrument bubble
            if any(inside(b[:2], grow(q, -2 * MM)) and inside(b[2:], grow(q, -2 * MM)) for q in eqb):
                continue                         # drawn inside an equipment outline (motor winding, vessel internals)
            if any(area((max(b[0], q[0]), max(b[1], q[1]), min(b[2], q[2]), min(b[3], q[3]))) > 0.7 * min(area(b), area(q))
                   for q in placed):
                continue                         # the same symbol under another class (higher score kept)
            placed.append(b)
            nid = self.add(f"sym:{k}", g, b, f"RF-DETR {d['cls']} ({d['conf']:.2f})", cls=d["cls"], conf=round(d["conf"], 3))
            self.boxes[nid] = b; k += 1
        # flow arrows: direction of each RF-DETR arrow from its ink; filled arrowheads RF-DETR missed are added
        net = self._arrow_finder()
        used = set()
        for n in self.nodes("arrow"):
            tb = self._arrow_dir(net, self.G.nodes[n]["bbox"], used)
            if tb:
                (tip, base), how = tb
                self.G.nodes[n].update(tip=[r1(v) for v in tip], base=[r1(v) for v in base], dir_source=how)
        valves = [self.G.nodes[n]["bbox"] for n in self.nodes("valve", "fitting")]
        for l in range(1, len(net.blob_st)):
            if l in used:
                continue
            a = net.arrow(l)
            if a is None:
                continue
            tip, base = a
            c = (tip + base) / 2
            if self.in_panel(c) or any(inside(c, grow(v, 0.5 * MM)) for v in valves) or \
                    any(inside(c, grow(q, 1 * MM)) for q in bub):
                continue
            x, y, w, h, _ = net.blob_st[l]
            nid = self.add(f"sym:{k}", "arrow", (x, y, x + w, y + h), "filled arrowhead in the ink (lines.Network.arrow)",
                           cls="filled_arrowhead", conf=1.0, tip=[r1(v) for v in tip], base=[r1(v) for v in base],
                           dir_source="triangle shape: wide base, narrow tip")
            self.boxes[nid] = (x, y, x + w, y + h); k += 1

    def _arrow_finder(self):
        """lines.Network's filled-blob arrow test without building its pixel graph (only the blob tables)."""
        ink = self.s.ink
        net = object.__new__(Network)
        kk = int(round(0.93 * MM)) | 1
        filled = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        _, net.blob_lab, net.blob_st, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        net.thick_full = ink          # single-pen sheets: any drawn line carries an arrow (the heavy-pen test is in lines.py)
        net._arrows = {}
        return net

    def _arrow_dir(self, net, b, used):
        """(tip, base) of the arrow in box b: a filled triangle inside it, else the ink width profile along the line
        through it (a triangle narrows towards its tip)."""
        x0, y0, x1, y1 = [int(v) for v in b]
        labs = set(np.unique(net.blob_lab[y0:y1, x0:x1]).tolist()) - {0}
        for l in labs:
            a = net.arrow(l)
            if a is not None:
                used.add(l)
                return a, "filled triangle inside the RF-DETR box"
        crop = self.s.ink[y0:y1, x0:x1].astype(float)
        if crop.size == 0 or crop.sum() < 10:
            return None
        best = None
        for axis in (0, 1):                  # 0: arrow along x (width = ink per column), 1: along y
            # spread of ink across the line at each position (an open chevron and a filled triangle both widen)
            cols = crop if axis == 0 else crop.T
            w = np.array([(np.ptp(np.nonzero(c)[0]) if c.any() else 0) for c in cols.T], float)
            t = np.arange(len(w))
            if len(t) < 4 or np.std(w) == 0:
                continue
            r = np.corrcoef(t, w)[0, 1]
            if best is None or abs(r) > abs(best[1]):
                best = (axis, r)
        if best is None or abs(best[1]) < 0.5:
            return None
        axis, r = best
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lo, hi = ((x0, cy), (x1, cy)) if axis == 0 else ((cx, y0), (cx, y1))
        tip, base = (lo, hi) if r > 0 else (hi, lo)          # width grows towards the base
        return (np.array(tip, float), np.array(base, float)), f"ink width profile in the RF-DETR box (r={r:.2f})"

    def _connectors(self):
        ref = re.compile(self.s.profile["sheet_ref"]) if self.s.profile.get("sheet_ref") else None
        word = {"in": "FROM", "out": "TO", "end": "END"}
        self.conn_id = {}
        for k, c in enumerate(self.plant.connectors):
            sheets = sorted({m.group(0) for t in c["text"].split(" / ") for m in ([ref.search(t)] if ref else []) if m})
            if c["flow"] not in word and not sheets:
                continue
            b = c["bbox"]
            dup = next((n for n in self.nodes("connector") if area((max(b[0], self.G.nodes[n]["bbox"][0]), max(b[1], self.G.nodes[n]["bbox"][1]),
                        min(b[2], self.G.nodes[n]["bbox"][2]), min(b[3], self.G.nodes[n]["bbox"][3]))) > 0.5 * min(area(b), area(self.G.nodes[n]["bbox"]))), None)
            if dup:
                self.conn_id[id(c)] = dup        # one connector's text stack read twice (plant.connectors)
                continue
            nid = self.add(f"conn:{k}", "connector", b, f"off-sheet connector / named pipe end '{c['text'][:80]}' (plant.connectors)",
                           direction=word.get(c["flow"], "REF"), target=" ".join(c["names"]) or c["dest"][:60],
                           target_tags=c["tags"], target_sheet=sheets, service=c["service"], text=c["text"],
                           direction_source=f"connector word ({c['flow']})" if c["flow"] in word else "drawing reference only")
            self.conn_id[id(c)] = nid
            if (b[2] - b[0]) < 70 * MM and (b[3] - b[1]) < 25 * MM:
                self.boxes[nid] = b            # small enough to be the connector symbol itself

    # ------------------------------------------------------------------ leaders
    def _leaders(self):
        """Every solid line leaving a bubble (innermost ring with lines, like describe.leader_of) and what it touches."""
        s = self.s
        self.leaders, self.taps = [], {}
        inst = self.nodes("instrument")
        syms = self.nodes("valve", "fitting")
        eqs = self.nodes("equipment")
        conns = [n for n in self.nodes("connector") if n in self.boxes]
        mask = np.zeros(s.ink.shape, np.uint8)
        for n in inst:
            tag = self.G.nodes[n]["tag"]; b = self.bubbles[tag]
            lines = [l for l in trace_bubble(s.gray, b, s.ink) if l["length_mm"] >= 2]
            if not lines:
                continue
            ring = min(l["ring_mm"] for l in lines)
            prim = leader_of(s, b)
            got = set()
            for l in sorted((l for l in lines if l["ring_mm"] == ring), key=lambda l: -l["length_mm"]):
                if l["end"] == "free" and l["length_mm"] < 4:
                    continue                    # one dash of a dashed signal line
                tgt, how = self._leader_target(n, l, inst, syms, eqs, conns)
                if tgt in got:
                    continue
                got.add(tgt)
                is_p = prim is not None and abs(prim["exit_ang"] - l["exit_ang"]) < 1e-6
                self.leaders.append(dict(inst=n, line=l, target=tgt, how=how, primary=is_p))
                pts = np.array(l["pts"], np.int32).reshape(-1, 1, 2)
                cv2.polylines(mask, [pts], False, 1, thickness=int(2 * C.LEADER_MASK_MM * MM))
        self.leader_mask = mask

    def _leader_target(self, n, l, inst, syms, eqs, conns):
        pts = l["pts"]
        (xa, ya), (xb, yb) = pts[max(0, len(pts) - 4)], pts[-1]
        h = math.atan2(yb - ya, xb - xa)
        probes = [(xb + d * MM * math.cos(h), yb + d * MM * math.sin(h)) for d in C.LEADER_PROBE_MM]
        for p in probes:
            hit = self.smallest_at(p, [m for m in inst if m != n], C.ABSORB_MM["bubble"])
            if hit:
                return hit, "line ends at another bubble"
        for p in probes:
            hit = self.smallest_at(p, syms, 0.5)
            if hit:
                return hit, f"line ends on {self.G.nodes[hit]['cls']}"
        for p in probes:
            hit = self.smallest_at(p, eqs, 2.0)
            if hit:
                return hit, f"line ends on the outline of {self.G.nodes[hit]['tag'] or self.G.nodes[hit]['name']}"
        for p in probes:
            hit = self.smallest_at(p, conns, 1.0)
            if hit:
                return hit, "line ends at an off-sheet connector"
        t = self.pipes.tap(pts, C.LEADER_TAP_MM)
        if t:
            i, pos, q, d = t
            key = (i, pos)
            self.pipes.st[i].add(pos)
            tid = self.taps.get(key)
            if tid is None:
                tid = self.add(f"tap:{len(self.taps)}", "tap", (q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2),
                               f"point a bubble's line taps a pipe ({d:.1f} mm on along its heading; lines.Pipes.tap)")
                self.taps[key] = tid
            return tid, f"line taps the pipe ({d:.1f} mm on)"
        for p in probes:
            reg = self.s.region_at(*p)
            if reg and 5 * MM < reg[2] - reg[0] < 200 * MM and 5 * MM < reg[3] - reg[1] < 200 * MM:
                old = next((m for m in self.nodes("equipment") if self.G.nodes[m]["kind"] == "shape"
                            and self.G.nodes[m]["bbox"] == [r1(v) for v in reg]), None)
                if old:
                    return old, "line ends on an unlabelled shape"
                nid = self.add(f"eq:shape{len(self.nodes('equipment'))}", "equipment", reg,
                               "unlabelled closed shape a bubble's line ends on (sheet.region_at)", tag="", name="",
                               kind="shape", name_source="")
                self.boxes[nid] = reg
                return nid, "line ends on an unlabelled shape"
        nid = self.add(f"end:L{len(self.leaders)}", "end", (xb - 2, yb - 2, xb + 2, yb + 2),
                       f"end of a bubble's line that touches nothing found ({l['end']})", line="leader")
        return nid, f"line ends in nothing ({l['end']})"

    def _leader_edges(self):
        for L in self.leaders:
            l = L["line"]
            poly = simplify([tuple(p) for p in l["pts"]], 1.5)
            self.G.add_edge(L["inst"], L["target"], kind="leader", polyline=[[r1(x), r1(y)] for x, y in poly],
                            length_mm=r1(l["length_mm"]), end=l["end"], primary=L["primary"],
                            source=f"traced solid line from the bubble (trace.trace_bubble): {L['how']}")

    # ------------------------------------------------------------------ process
    def _absorb(self, xy, edges, ids, tol_of, split=True):
        """Points inside an object's box belong to it; a line passing straight through a box gets a point there."""
        owner = {}
        P = list(xy)
        arr = np.array([xy[p] for p in P]) if P else np.zeros((0, 2))
        for nid in sorted(ids, key=lambda n: -area(self.boxes[n])):         # smallest box wins (assigned last)
            b = grow(self.boxes[nid], tol_of(nid) * MM)
            if not len(arr):
                break
            m = (arr[:, 0] >= b[0]) & (arr[:, 0] <= b[2]) & (arr[:, 1] >= b[1]) & (arr[:, 1] <= b[3])
            for i in np.nonzero(m)[0]:
                owner[P[i]] = nid
        if split and edges:
            E = np.array([[*xy[p], *xy[q]] for p, q, _ in edges])
            lo, hi = np.minimum(E[:, :2], E[:, 2:]), np.maximum(E[:, :2], E[:, 2:])
            new = []
            cut = set()
            for nid in sorted(ids, key=lambda n: area(self.boxes[n])):
                b = grow(self.boxes[nid], tol_of(nid) * MM)
                m = (lo[:, 0] <= b[2]) & (hi[:, 0] >= b[0]) & (lo[:, 1] <= b[3]) & (hi[:, 1] >= b[1])
                for k in np.nonzero(m)[0]:
                    p, q, a = edges[k]
                    if k in cut or p in owner or q in owner or not seg_hits_box(xy[p], xy[q], b):
                        continue
                    c = ctr(b)
                    (px, py), (qx, qy) = xy[p], xy[q]
                    L2 = (qx - px) ** 2 + (qy - py) ** 2 or 1
                    t = max(0, min(1, ((c[0] - px) * (qx - px) + (c[1] - py) * (qy - py)) / L2))
                    off = math.hypot(px + t * (qx - px) - c[0], py + t * (qy - py) - c[1])
                    if a["kind"] == "link" and off > max(1.5 * MM, 0.25 * min(b[2] - b[0], b[3] - b[1])):
                        continue                 # a gap bridge only grazing the box: the symbol is not on it
                    mid = ("x", nid, k)
                    xy[mid] = (px + t * (qx - px), py + t * (qy - py))
                    owner[mid] = nid
                    cut.add(k)
                    a2 = dict(a, split=nid)
                    new += [(p, mid, a2), (mid, q, a2)]
            edges = [e for k, e in enumerate(edges) if k not in cut] + new
        return owner, edges

    def _process(self):
        pp, S = self.pipes, self.pipes.segs
        drop = set()
        for i, s in enumerate(S):
            a, b = pp.pt(s, s["a"]), pp.pt(s, s["b"])
            if self.in_panel(a) and self.in_panel(b):
                drop.add(i); continue
            qs = [pp.pt(s, v) for v in np.linspace(s["a"], s["b"], max(2, int((s["b"] - s["a"]) / 3)))]
            on = np.mean([self.leader_mask[min(int(y), self.leader_mask.shape[0] - 1), min(int(x), self.leader_mask.shape[1] - 1)] for x, y in qs])
            if on >= 0.7:
                drop.add(i)                  # a bubble's leader drawn with the pipe pen
        tb = self._text_boxes()
        tb = tb + self._tables(tb)
        for i, s in enumerate(S):
            a, b = pp.pt(s, s["a"]), pp.pt(s, s["b"])
            if any(inside(a, grow(r, 1.2 * MM)) and inside(b, grow(r, 1.2 * MM)) for r in tb):
                drop.add(i)
        self.text_boxes = tb
        self.dropped_segments = len(drop)
        xy, edges = {}, []
        for i, ps in pp.st.items():
            if i in drop:
                continue
            ps = sorted(ps)
            for v in ps:
                xy[(i, v)] = pp.pt(S[i], v)
            for a, b in zip(ps, ps[1:]):
                edges.append(((i, a), (i, b), dict(kind="run", seg=i, gap=0.0)))
        seen = set()
        for p, lst in pp.adj.items():
            for q, cost in lst:
                if p[0] in drop or q[0] in drop or (q, p) in seen or p == q:
                    continue
                seen.add((p, q))
                for r in (p, q):
                    xy.setdefault(r, pp.pt(S[r[0]], r[1]))
                gap = math.hypot(xy[p][0] - xy[q][0], xy[p][1] - xy[q][1])
                edges.append((p, q, dict(kind="link", gap=gap)))
        ids = [n for n in self.boxes if self.G.nodes[n]["type"] in ("equipment", "valve", "fitting", "arrow", "connector")]
        tol = lambda n: C.ABSORB_MM[self.G.nodes[n]["type"]]
        owner, edges = self._absorb(xy, edges, ids, tol)
        def diagonal(p, q):
            return abs(xy[p][0] - xy[q][0]) > 1.5 * MM and abs(xy[p][1] - xy[q][1]) > 1.5 * MM
        # an in-line-symbol hop (lines.Pipes._symbol_hops) that is diagonal and runs through no detected symbol joins
        # a pipe end to whatever ink is near (label box, leader, outline): dropped
        # (kept when it was split by a symbol box it runs through, or is a short corner gap)
        edges = [e for e in edges if not (e[2]["kind"] == "link" and diagonal(e[0], e[1]) and not e[2].get("split")
                                          and (e[2]["gap"] > 6 * MM or (e[0] not in owner and e[1] not in owner)))]
        for key, tid in self.taps.items():
            if key in xy and key not in owner:
                owner[key] = tid
        # dangling pipe ends: an off-sheet connector or equipment just beyond (lines.Pipes.terminal)
        deg = Counter()
        for p, q, _ in edges:
            deg[p] += 1; deg[q] += 1
        self.term_src = {}
        eq_by_tag = defaultdict(list)
        for n in self.nodes("equipment"):
            eq_by_tag[self.G.nodes[n]["tag"]].append(n)
        for p, d in deg.items():
            if d != 1 or p in owner or not isinstance(p[0], int):
                continue
            i, pos = p
            s = S[i]
            end = "a" if pos == s["a"] else "b" if pos == s["b"] else None
            if end is None:
                continue
            t = pp.terminal(i, end)
            if not t:
                continue
            if t["kind"] == "connector" and id(t["conn"]) in self.conn_id:
                owner[p] = self.conn_id[id(t["conn"])]
                self.term_src[p] = f"pipe end {t['dist_mm']:.0f} mm from the connector (lines.Pipes.terminal)"
            elif t["kind"] == "equipment" and eq_by_tag.get(t["tag"]):
                c = ctr(t["region"])
                owner[p] = min(eq_by_tag[t["tag"]], key=lambda n: math.hypot(self.G.nodes[n]["x"] - c[0], self.G.nodes[n]["y"] - c[1]))
                self.term_src[p] = "pipe end at the equipment outline (lines.Pipes.terminal)" + (f" via {t['via']}" if t.get("via") else "")
        chains = contract(xy, edges, owner, keep=set(self.taps))
        self._add_chains(chains, xy, "process", owner)
        self._touching()

    def _touching(self):
        """In-line symbols drawn against each other or against a nozzle with no line between (valve on a vessel
        nozzle, valve-flange-blind stacks): a process edge between the two boxes."""
        G = self.G
        objs = [n for n in self.boxes if G.nodes[n]["type"] in ("valve", "fitting", "equipment", "connector")
                and G.nodes[n].get("kind") != "panel"]
        def nproc(n):
            return sum(d["kind"] == "process" for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)))
        k = 0
        for a in objs:
            if G.nodes[a]["type"] not in ("valve", "fitting") or nproc(a) >= 2:
                continue
            A = self.boxes[a]
            for b in objs:
                if b == a:
                    continue
                B = self.boxes[b]
                gap = math.hypot(max(0, max(A[0], B[0]) - min(A[2], B[2])), max(0, max(A[1], B[1]) - min(A[3], B[3])))
                linked = G.has_edge(a, b) or G.has_edge(b, a)
                if gap <= 1.5 * MM and not linked and not (inside(ctr(A), B) or inside(ctr(B), A)):
                    pa, pb = ctr(A), ctr(B)
                    G.add_edge(a, b, kind="process", polyline=[[r1(pa[0]), r1(pa[1])], [r1(pb[0]), r1(pb[1])]],
                               length_mm=r1(math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / MM), pen_mm=0.0, gaps_bridged=0,
                               source=f"symbols drawn touching ({gap / MM:.1f} mm apart, no line between)",
                               flow="unknown", flow_source="", line_number="", line_number_source="")
                    k += 1
        self.notes.append(f"process: {k} edges between symbols drawn touching")

    def _add_chains(self, chains, xy, kind, owner):
        """Chains -> edges; chain end points that are not objects become junction / end nodes; components without any
        object (tables, frames, hatching, text underlines) are dropped."""
        deg = Counter()
        for u, v, _, _ in chains:
            deg[u] += 1; deg[v] += 1
        par = {}
        def find(a):
            while par.setdefault(a, a) != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for u, v, _, _ in chains:
            par[find(u)] = find(v)
        # a chain leaving an object and coming back to it is that object's own outline / internals
        n0 = len(chains)
        chains = [c for c in chains if not (c[0] == c[1] and not isinstance(c[0], tuple))]
        if n0 > len(chains):
            self.notes.append(f"{kind}: {n0 - len(chains)} chains from an object back to itself dropped (outline)")
        has_obj = {find(n) for n in deg if not isinstance(n, tuple)}
        keep = [c for c in chains if find(c[0]) in has_obj]
        self.notes.append(f"{kind}: {len(chains) - len(keep)} chains in {len({find(n) for n in deg} - has_obj)} "
                          f"components touching no object dropped")
        pre = "j" if kind == "process" else "sj"
        names = {}
        def node(n, p):
            if not isinstance(n, tuple):
                return n
            if n not in names:
                x, y = xy[n]
                t = "end" if deg[n] == 1 else "junction"
                names[n] = self.add(f"{pre}:{len(names)}", t, (x - 2, y - 2, x + 2, y + 2),
                                    f"{kind} line {'junction (T, lines.Pipes graph)' if t == 'junction' else 'end: nothing found beyond it'}"
                                    if kind == "process" else f"signal line {'junction' if t == 'junction' else 'end: touches nothing found'}",
                                    line=kind)
            return names[n]
        # stubs: short dangling pieces (symbol strokes, valve stems / handwheels, text leaders)
        if True:
            def stub(c):
                u, v, pts, _ = c
                L = length([xy[p] for p in pts]) / MM
                du, dv = isinstance(u, tuple) and deg[u] == 1, isinstance(v, tuple) and deg[v] == 1
                if not (du or dv) or (du and dv and L > 5):
                    return False
                other = v if du else u
                sym = not isinstance(other, tuple) and self.G.nodes[other]["type"] in ("valve", "fitting", "arrow")
                if kind == "signal":
                    return L < 3                 # a dash overhanging a corner
                return L < 5 or (sym and L < 10)
            n0 = len(keep)
            keep = [c for c in keep if not stub(c)]
            self.notes.append(f"{kind}: {n0 - len(keep)} dangling stubs dropped")
            deg = Counter()
            for u, v, _, _ in keep:
                deg[u] += 1; deg[v] += 1
        # the same route found twice (a gap bridged and a symbol hop beside it): keep the shortest
        by_pair = defaultdict(list)
        for c in keep:
            by_pair[frozenset((c[0], c[1]))].append(c)
        keep2 = []
        for grp in by_pair.values():
            grp = sorted(grp, key=lambda c: length([xy[p] for p in c[2]]))
            kept = []
            for c in grp:
                poly = [xy[p] for p in c[2]]
                if any(np.mean([project([xy[p] for p in k[2]], q)[0] for q in poly]) < 3 * MM for k in kept):
                    continue
                kept.append(c)
            keep2 += kept
        if len(keep2) < len(keep):
            self.notes.append(f"{kind}: {len(keep) - len(keep2)} duplicate parallel chains dropped")
        keep2 = [c for c in self._merge_pass_through(keep2) if not (c[0] == c[1] and not isinstance(c[0], tuple))]
        for u, v, pts, att in keep2:
            a, b = node(u, pts[0]), node(v, pts[-1])
            poly = simplify([xy[p] for p in pts], 1.0)
            attrs = dict(kind=kind, polyline=[[r1(x), r1(y)] for x, y in poly], length_mm=r1(length(poly) / MM))
            if kind == "process":
                halves = [self.pipes.segs[t["seg"]]["half"] for t in att if t["kind"] == "run"]
                gaps = [t["gap"] for t in att if t["kind"] == "link" and t["gap"] > 0.5 * MM]
                attrs.update(pen_mm=round(float(2 * np.median(halves) / MM), 2) if halves else 0.0,
                             gaps_bridged=len(gaps),
                             source=f"solid line segments ({self.pipes.mode}, lines.Pipes): {len([t for t in att if t['kind'] == 'run'])} runs, "
                                    f"{len(gaps)} gaps bridged (flange / crossing / in-line symbol, max {max(gaps) / MM if gaps else 0:.0f} mm)",
                             flow="unknown", flow_source="", line_number="", line_number_source="")
                ends_src = [self.term_src[p] for p in (pts[0], pts[-1]) if p in getattr(self, "term_src", {})]
                if ends_src:
                    attrs["source"] += "; " + "; ".join(ends_src)
            else:
                ns = sum(self.runs[t["run"]]["n"] for t in att if t["kind"] == "run")
                attrs.update(dashes=ns, source=f"dashed line: {ns} dashes in {len([t for t in att if t['kind'] == 'run'])} runs")
            self.G.add_edge(a, b, **attrs)

    @staticmethod
    def _merge_pass_through(chains):
        """Line points left with exactly two chains (after stubs / duplicates were dropped) are no junction: join them."""
        C_ = dict(enumerate(chains))
        inc = defaultdict(set)
        for i, (u, v, _, _) in C_.items():
            inc[u].add(i); inc[v].add(i)
        nxt = len(C_)
        for n in [n for n in inc if isinstance(n, tuple)]:
            if len(inc[n]) != 2:
                continue
            i, j = sorted(inc[n])
            a, b = C_[i], C_[j]
            if a[0] == a[1] or b[0] == b[1] or {a[0], a[1]} == {b[0], b[1]} and a[0] != n and b[0] != n:
                continue
            if a[1] != n:
                a = (a[1], a[0], a[2][::-1], a[3][::-1])
            if b[0] != n:
                b = (b[1], b[0], b[2][::-1], b[3][::-1])
            new = (a[0], b[1], a[2] + b[2][1:], a[3] + b[3])
            del C_[i], C_[j]
            for k in (i, j):
                for m in (a[0], b[1], n):
                    inc[m].discard(k)
            C_[nxt] = new; inc[new[0]].add(nxt); inc[new[1]].add(nxt); nxt += 1
        return list(C_.values())

    # ------------------------------------------------------------------ signal
    def _signal(self):
        pp = self.pipes
        self.term_src = {}
        runs = dashed_runs(pp.ink, pp.thin)
        eqb = [self.boxes[n] for n in self.nodes("equipment") if n in self.boxes]
        keep = []
        for r in runs:
            a, b = (r["a"], r["c"]) if r["o"] == "h" else (r["c"], r["a"]), (r["b"], r["c"]) if r["o"] == "h" else (r["c"], r["b"])
            if self.in_panel(a) or self.in_panel(b):
                continue
            if any(inside(a, grow(q, 1.5 * MM)) and inside(b, grow(q, 1.5 * MM)) for q in eqb):
                continue                    # dashed outline / internals of a vessel
            keep.append(r)
        # dashed lines running beside a pipe (heat tracing, jacket lines) are not signals
        H_, W_ = self.s.ink.shape
        masks = {"h": np.zeros((H_, W_), np.uint8), "v": np.zeros((H_, W_), np.uint8)}
        for u, v, d in self.G.edges(data=True):
            if d["kind"] != "process":
                continue
            for a, b in zip(d["polyline"], d["polyline"][1:]):
                o = "h" if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else "v"
                cv2.line(masks[o], tuple(map(int, a)), tuple(map(int, b)), 1, int(2 * 2.5 * MM))
        beside = []
        for r in keep:
            qs = np.linspace(r["a"], r["b"], 20)
            pts = [(q, r["c"]) if r["o"] == "h" else (r["c"], q) for q in qs]
            on = np.mean([masks[r["o"]][min(int(y), H_ - 1), min(int(x), W_ - 1)] for x, y in pts])
            if on < 0.6:
                beside.append(r)
        self.notes.append(f"signal: {len(keep) - len(beside)} dashed runs beside a pipe dropped (tracing)")
        keep = beside
        self.runs = runs = keep
        xy, edges = point_graph(runs)
        for n in self.nodes("instrument"):
            self.boxes[n] = self.G.nodes[n]["bbox"]
        ids = [n for n in self.boxes if self.G.nodes[n]["type"] in ("instrument", "equipment", "valve", "fitting", "connector")]
        tolf = lambda n: C.ABSORB_MM["bubble"] if self.G.nodes[n]["type"] == "instrument" else C.ABSORB_MM.get(self.G.nodes[n]["type"], 1.0)
        owner, edges = self._absorb(xy, edges, ids, tolf, split=False)
        order = {"instrument": 0, "valve": 1, "fitting": 1, "equipment": 2, "connector": 3}
        for p, (dx, dy) in signal_ends(runs, xy, edges).items():
            if p in owner:
                continue
            x, y = xy[p]
            for d in np.arange(0.5, C.SIGNAL_REACH_MM + 0.01, 0.5) * MM:
                q = (x + dx * d, y + dy * d)
                hits = [n for n in ids if inside(q, grow(self.boxes[n], 0.5 * MM))]
                if hits:
                    owner[p] = min(hits, key=lambda n: (order[self.G.nodes[n]["type"]], area(self.boxes[n])))
                    break
        chains = contract(xy, edges, owner)
        self._add_chains(chains, xy, "signal", owner)
        # a signal line runs from the measurement towards the final element: instrument -> valve when one end is a valve
        for u, v, k, d in list(self.G.edges(keys=True, data=True)):
            if d["kind"] == "signal" and self.G.nodes[u]["type"] in ("valve", "fitting") and self.G.nodes[v]["type"] == "instrument":
                self.G.remove_edge(u, v, k); self.G.add_edge(v, u, **d)

    # ------------------------------------------------------------------ labels
    def _labels(self):
        s, pl = self.s, self.plant
        ref = re.compile(s.profile["sheet_ref"]) if s.profile.get("sheet_ref") else None
        proc = [(u, v, k, d) for u, v, k, d in self.G.edges(keys=True, data=True) if d["kind"] == "process"]
        k_ = 0
        for f in s.frags:
            t, b = f["text"], f["bbox"]
            if not re.match(C.LINE_NUMBER, t) or (ref and ref.search(t)) or self.in_panel(ctr(b)):
                continue
            horiz = (b[2] - b[0]) >= (b[3] - b[1])
            c = ctr(b)
            half = ((b[3] - b[1]) if horiz else (b[2] - b[0])) / 2
            best = None
            for u, v, k, d in proc:
                poly = d["polyline"]
                dist, _, _ = project(poly, c)
                if dist - half > C.LINE_LABEL_MM * MM:
                    continue
                # prefer a pipe running the text's way
                (x0, y0), (x1, y1) = poly[0], poly[-1]
                par = (abs(x1 - x0) >= abs(y1 - y0)) == horiz
                score = dist + (0 if par else 3 * MM)
                if best is None or score < best[0]:
                    best = (score, u, v, k, dist)
            nid = self.add(f"ll:{k_}", "line_label", b, "OCR text with the line-number pattern (schema.LINE_NUMBER)", text=t)
            k_ += 1
            if best:
                _, u, v, k, dist = best
                d = self.G.edges[u, v, k]
                if not d["line_number"]:
                    d["line_number"] = t
                    d["line_number_source"] = f"line number printed {dist / MM:.1f} mm from the pipe"
                tgt = u if math.hypot(self.G.nodes[u]["x"] - c[0], self.G.nodes[u]["y"] - c[1]) <= \
                    math.hypot(self.G.nodes[v]["x"] - c[0], self.G.nodes[v]["y"] - c[1]) else v
                self.G.add_edge(nid, tgt, kind="labels", on_edge=f"{u}|{v}|{k}", source="line number -> nearest pipe")
        # valve tags: one line ('HV1852A', '111LCV-0812') or letters over number ('11VV' / '7014-0')
        valves = self.nodes("valve", "fitting")
        cands = [(f["text"], f["bbox"]) for f in s.frags if re.match(C.VALVE_TAG, f["text"]) and not pl.tag_full.fullmatch(f["text"])]
        for f in s.frags:
            if re.match(C.VALVE_LETTERS, f["text"]):
                b = f["bbox"]; h = b[3] - b[1]
                g = next((g for g in s.frags if re.match(C.VALVE_NUMBER, g["text"]) and 0 <= g["bbox"][1] - b[3] < 1.2 * h
                          and abs(ctr(g["bbox"])[0] - ctr(b)[0]) < 3 * MM), None)
                if g:
                    cands.append((f["text"] + "-" + g["text"], (min(b[0], g["bbox"][0]), b[1], max(b[2], g["bbox"][2]), g["bbox"][3])))
        taken = set()
        for t, b in cands:
            if self.in_panel(ctr(b)):
                continue
            near = sorted((box_dist(ctr(self.G.nodes[n]["bbox"]), b), n) for n in valves if n not in taken)
            if near and near[0][0] < C.VALVE_LABEL_MM * MM:
                dist, n = near[0]
                taken.add(n)
                nid = self.add(f"lb:{k_}", "label", b, "OCR text with the valve-tag pattern (schema.VALVE_TAG)", text=t)
                k_ += 1
                self.G.nodes[n].update(tag=t, tag_source=f"valve tag printed {dist / MM:.1f} mm from the symbol")
                self.G.add_edge(nid, n, kind="labels", source=f"valve tag -> nearest symbol ({dist / MM:.1f} mm)")
        for n in self.nodes("equipment"):
            d = self.G.nodes[n]
            if d.get("tag") and d.get("label_bbox"):
                nid = self.add(f"lb:{k_}", "label", d["label_bbox"], "equipment label (plant.labels)", text=d["tag"])
                k_ += 1
                self.G.add_edge(nid, n, kind="labels", source="equipment tag printed at its symbol")

    # ------------------------------------------------------------------ flow
    def _flow(self):
        """Flow votes per process edge: arrows on it or at its end, connectors at its end; carried through in-line
        symbols / taps (a node with exactly two process edges) until nothing changes; conflicts are recorded."""
        G = self.G
        proc = {(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d["kind"] == "process"}
        votes = defaultdict(list)        # edge -> [(+1 u->v / -1, source)]
        for u, v, k in proc:
            for end, sign in ((u, -1), (v, 1)):        # sign: +1 = flow towards that end is u->v
                d = G.nodes[end]
                if d["type"] == "connector":
                    if d["direction"] == "FROM":
                        votes[(u, v, k)].append((-sign, f"comes from connector '{d['text'][:40]}' (FROM)"))
                    elif d["direction"] in ("TO", "END"):
                        votes[(u, v, k)].append((sign, f"goes to connector '{d['text'][:40]}' ({d['direction']})"))
                elif d["type"] == "arrow" and d.get("tip"):
                    poly = G.edges[u, v, k]["polyline"]
                    e, q = (poly[-1], poly[-2]) if end == v else (poly[0], poly[1])
                    ox, oy = q[0] - e[0], q[1] - e[1]                 # the edge's way out of the arrow
                    ax, ay = d["tip"][0] - d["base"][0], d["tip"][1] - d["base"][1]
                    dot = (ox * ax + oy * ay) / ((math.hypot(ox, oy) * math.hypot(ax, ay)) or 1)
                    if abs(dot) > 0.5:
                        # the edge leaves the arrow on its tip side: flow runs away from the arrow along this edge
                        away = dot > 0
                        votes[(u, v, k)].append((-sign if away else sign,
                                                 f"flow from {d['cls']} at ({d['x']:.0f}, {d['y']:.0f}) ({d['dir_source']})"))
        # arrowheads lying on an edge (not absorbed as a node)
        for n in self.nodes("arrow"):
            d = G.nodes[n]
            if G.degree(n) or not d.get("tip"):
                continue
            mid = ((d["tip"][0] + d["base"][0]) / 2, (d["tip"][1] + d["base"][1]) / 2)
            best = None
            for e in proc:
                dist, s_, _ = project(G.edges[e]["polyline"], mid)
                if dist < C.ARROW_ON_LINE_MM * MM and (best is None or dist < best[0]):
                    best = (dist, e)
            if best:
                e = best[1]; poly = G.edges[e]["polyline"]
                st = project(poly, d["tip"])[1]; sb = project(poly, d["base"])[1]
                if abs(st - sb) > 0.3 * MM:
                    votes[e].append((1 if st > sb else -1, f"flow from {d['cls']} at ({d['x']:.0f}, {d['y']:.0f}) on the line"))
        self.conflicts = []
        state = {}
        for e, vs in votes.items():
            tot = sum(v for v, _ in vs)
            if any(v > 0 for v, _ in vs) and any(v < 0 for v, _ in vs):
                self.conflicts.append(dict(edge=e, xy=self._mid(e), why="arrows / connectors disagree on this pipe: " + "; ".join(s for _, s in vs)))
            if tot:
                state[e] = (1 if tot > 0 else -1, "; ".join(s for _, s in vs))
        # carry through in-line nodes
        inline = ("valve", "fitting", "arrow", "tap")
        changed = True
        while changed:
            changed = False
            for n, d in G.nodes(data=True):
                if d["type"] not in inline:
                    continue
                es = [(u, v, k) for u, v, k in list(G.in_edges(n, keys=True)) + list(G.out_edges(n, keys=True)) if (u, v, k) in proc]
                if len(es) != 2:
                    continue
                into = {}
                for e in es:
                    if e in state:
                        into[e] = (state[e][0] > 0) == (e[1] == n)     # flow enters n along e
                known = [e for e in es if e in into]
                if len(known) == 1:
                    e0 = known[0]; e1 = es[1] if es[0] == e0 else es[0]
                    want_into = not into[e0]
                    sign = 1 if (e1[1] == n) == want_into else -1
                    state[e1] = (sign, f"carried through {d['type']} {d.get('cls', '')} {n} from the next pipe ({state[e0][1][:60]})")
                    changed = True
                elif len(known) == 2 and into[es[0]] == into[es[1]] and not any(c.get("node") == n for c in self.conflicts):
                    self.conflicts.append(dict(node=n, xy=(d["x"], d["y"]), why=f"flow {'enters' if into[es[0]] else 'leaves'} {d['type']} {n} from both sides"))
        for (u, v, k), (sign, src) in state.items():
            d = dict(G.edges[u, v, k])
            d.update(flow="known", flow_source=src)
            if any(c.get("edge") == (u, v, k) for c in self.conflicts):
                d["flow"] = "conflict"
            G.remove_edge(u, v, k)
            if sign > 0:
                G.add_edge(u, v, **d)
            else:
                d["polyline"] = d["polyline"][::-1]
                G.add_edge(v, u, **d)
        # junctions / taps where every pipe flows in or every pipe flows out
        for n, d in G.nodes(data=True):
            if d["type"] not in ("junction", "tap"):
                continue
            es = [(e, "in") for e in G.in_edges(n, data=True) if e[2]["kind"] == "process" and e[2]["flow"] != "unknown"] + \
                 [(e, "out") for e in G.out_edges(n, data=True) if e[2]["kind"] == "process" and e[2]["flow"] != "unknown"]
            allp = [e for e in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)) if e[2]["kind"] == "process"]
            if len(es) == len(allp) >= 2 and len({w for _, w in es}) == 1:
                self.conflicts.append(dict(node=n, xy=(d["x"], d["y"]), why=f"every pipe at {d['type']} {n} flows {es[0][1]}"))


def build(name, **kw):
    return SheetGraph(name, **kw)
