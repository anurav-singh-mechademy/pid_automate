"""Query a saved sheet graph (pidgraph.graph.SheetGraph -> descgraph/build.py -> <out>/graphs/<sheet>.gpickle).

    from pidgraph.query import SheetQuery
    q = SheetQuery.load("descgraph_output/graphs/SC62D1914_p0.gpickle")
    q.instrument("PIT-5013A")           # node id + attributes
    q.attachments("inst:PIT-5013A")     # what its lines touch: tap / equipment / valve / bubble / connector
    q.reach("inst:PIT-5013A")           # equipment and connectors reachable along the pipes, nearest first, with flow side
    q.loop("inst:PIT-5013A")            # other instruments of the same loop (tag number) and signal-line neighbours
    .venv/bin/python -m pidgraph.query <gpickle> <tag>      # prints all of the above for one tag

Distances are along the drawn lines in mm. `side` says where the instrument sits relative to the item it reached:
'upstream' (fluid flows from the instrument to the item), 'downstream', or None when no flow direction is known on
the path. Pass-through nodes on pipes: junctions, taps, line ends, in-line valves / fittings / arrows."""
import heapq
import math
import pickle
import re
import sys
from pathlib import Path

PASS = {"junction", "tap", "end", "valve", "fitting", "arrow"}
STOP = {"equipment", "connector"}
MM = 300 / 25.4


def _len(poly):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:])) / MM if poly else 0.0


class SheetQuery:
    def __init__(self, G):
        self.G = G
        self.by_tag = {d["tag"]: n for n, d in G.nodes(data=True) if d.get("type") == "instrument"}
        # undirected pipe adjacency with length and flow sign (+1: flow a->b, -1: b->a, 0 unknown)
        self.adj = {}
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("kind") == "process" and u != v]
        flow = {}
        for u, v, d in edges:
            if d.get("flow") == "known":
                flow[(u, v)] = 1; flow[(v, u)] = -1
        self.derived_flow = self._carry_flow(edges, flow)
        for u, v, d in edges:
            L = _len(d.get("polyline")) or 0.1
            f = flow.get((u, v), 0)
            self.adj.setdefault(u, []).append((v, L, f))
            self.adj.setdefault(v, []).append((u, L, -f))

    def _carry_flow(self, edges, flow):
        """A known flow direction holds along a pipe run that does not branch: carry it through every node with
        exactly two pipe edges (bends, in-line valves / fittings / arrows, taps). Returns the number of edges given a
        direction this way; the graph itself is not changed."""
        nb = {}
        for u, v, _ in edges:
            nb.setdefault(u, set()).add(v); nb.setdefault(v, set()).add(u)
        through = lambda n: len(nb.get(n, ())) == 2 and self.G.nodes[n].get("type") in PASS
        added, todo = 0, [k for k, f in flow.items() if f == 1]
        while todo:
            a, b = todo.pop()
            for x, y in ((a, b), (b, a)):            # downstream from b, upstream from a
                n, prev = (y, x)
                while through(n):
                    m = next(iter(nb[n] - {prev}))
                    key = (n, m) if (x, y) == (a, b) else (m, n)
                    if key in flow:
                        break
                    flow[key] = 1; flow[key[::-1]] = -1; added += 1
                    prev, n = n, m
        return added

    @classmethod
    def load(cls, path):
        return cls(pickle.load(open(path, "rb")))

    def node(self, n):
        return dict(self.G.nodes[n], id=n)

    def instrument(self, tag):
        n = self.by_tag.get(tag)
        return self.node(n) if n else None

    def instruments(self):
        return [self.node(n) for n in self.by_tag.values()]

    def attachments(self, n):
        """Targets of the instrument's own lines (leader edges), primary line first."""
        out = []
        for _, t, d in self.G.out_edges(n, data=True):
            if d.get("kind") == "leader":
                out.append(dict(target=t, type=self.G.nodes[t]["type"], primary=bool(d.get("primary")),
                                length_mm=d.get("length_mm"), how=d.get("source", "")))
        return sorted(out, key=lambda a: (not a["primary"], a["length_mm"] or 0))

    def signal(self, n):
        return [m for a, b, d in list(self.G.out_edges(n, data=True)) + list(self.G.in_edges(n, data=True))
                if d.get("kind") == "signal" for m in (a, b) if m != n]

    def loop(self, n):
        d = self.G.nodes[n]
        same = [m for m, e in self.G.nodes(data=True) if e.get("type") == "instrument" and m != n
                and e.get("loop") and e.get("loop") == d.get("loop")]
        return dict(same_number=same, signal=self.signal(n))

    def reach(self, start, max_mm=900, through=PASS):
        """Equipment / connectors reachable along pipes from `start` (an instrument: from what its lines touch).
        Returns [dict(node, type, dist_mm, side, path)] nearest first; one entry per item (shortest path)."""
        seeds = []
        if self.G.nodes[start]["type"] == "instrument":
            for a in self.attachments(start):
                t = a["target"]
                if a["type"] in STOP:
                    seeds.append((0.0, t, (t,), 0, 0))
                elif a["type"] in PASS:
                    seeds.append((0.0, t, (t,), 0, 0))
        else:
            seeds.append((0.0, start, (start,), 0, 0))
        best, out = {}, []
        heap = list(seeds); heapq.heapify(heap)
        while heap:
            dist, n, path, fwd, back = heapq.heappop(heap)
            if dist > max_mm or best.get(n, 1e9) <= dist:
                continue
            best[n] = dist
            t = self.G.nodes[n]["type"]
            if t in STOP:
                side = "upstream" if fwd > back else "downstream" if back > fwd else None
                out.append(dict(node=n, type=t, dist_mm=round(dist, 1), side=side, path=list(path),
                                flow_edges=(fwd, back)))
                if n != start:
                    continue                 # an item ends the search along this path (unless it is the start)
            if t not in through and t not in STOP:
                continue
            for m, L, f in self.adj.get(n, ()):
                heapq.heappush(heap, (dist + L, m, path + (m,), fwd + (f > 0), back + (f < 0)))
        return out

    def equipment(self, n):
        d = self.G.nodes[n]
        return dict(id=n, tag=d.get("tag", ""), name=d.get("name", ""), kind=d.get("kind", ""))

    def siblings(self, eq):
        """Other equipment with the same tag but another instance letter (5P551A / 5P551B)."""
        tag = self.G.nodes[eq].get("tag") or ""
        base = re.sub(r"[A-Z]$", "", tag)
        return [m for m, d in self.G.nodes(data=True) if d.get("type") == "equipment" and m != eq and d.get("tag")
                and d["tag"] != tag and re.sub(r"[A-Z]$", "", d["tag"]) == base]


if __name__ == "__main__":
    q = SheetQuery.load(sys.argv[1])
    n = q.by_tag[sys.argv[2]]
    print(q.node(n))
    for a in q.attachments(n):
        print("  line ->", a)
    print("  loop:", q.loop(n))
    for r in q.reach(n)[:8]:
        d = q.G.nodes[r["node"]]
        print(f"  reach {r['dist_mm']:7.1f} mm  {r['type']:9s} {d.get('tag') or d.get('target') or ''} "
              f"{d.get('name', '')[:40]}  side={r['side']}")
