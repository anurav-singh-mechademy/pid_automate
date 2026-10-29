"""Validation pass over a SheetGraph: what an engineer would check on the drawing's connectivity.

  unconnected_instruments  bubbles with no leader and no signal line
  unconnected_equipment    equipment with no pipe, leader or signal line
  dangling_edges           process / signal / leader edges that end in nothing (an 'end' node)
  flow_conflicts           arrows / connectors disagreeing on one pipe, in-line symbols fed from both sides,
                           junctions where every pipe flows in (or every pipe flows out)
  valves_off_line          valves with fewer than two pipes (not in a process line, or only at its end)
  loops                    instruments joined by signal lines: same loop number (consistent) or different (mixed);
                           loop numbers drawn more than once that no signal line joins
Each problem carries its point (x, y) for the overlay."""
from collections import Counter, defaultdict
import networkx as nx


def _xy(G, n):
    return (G.nodes[n]["x"], G.nodes[n]["y"])


def validate(sg):
    G = sg.G
    kinds = lambda n: Counter(d["kind"] for _, _, d in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)))
    rep = defaultdict(list)
    for n, d in G.nodes(data=True):
        k = kinds(n)
        if d["type"] == "instrument" and not (k["leader"] or k["signal"]):
            rep["unconnected_instruments"].append(dict(node=n, xy=_xy(G, n), why=f"{d['tag']}: no leader or signal line found"))
        elif d["type"] == "equipment" and not (k["process"] or k["leader"] or k["signal"]):
            rep["unconnected_equipment"].append(dict(node=n, xy=_xy(G, n), why=f"{d['tag'] or d['name'] or d['kind']}: nothing attached"))
        elif d["type"] == "valve" and k["process"] < 2:
            rep["valves_off_line"].append(dict(node=n, xy=_xy(G, n),
                                               why=f"{d['cls']} {d.get('tag', '')}: {k['process']} pipe(s) attached"))
        elif d["type"] == "end":
            for u, v, e in list(G.in_edges(n, data=True)) + list(G.out_edges(n, data=True)):
                if e["kind"] == "labels":
                    continue
                rep["dangling_edges"].append(dict(node=n, xy=_xy(G, n), kind=e["kind"], why=f"{e['kind']} edge {u} - {v} ends in nothing"))
    for c in sg.conflicts:
        rep["flow_conflicts"].append(dict(c, edge=str(c.get("edge", "")), node=c.get("node", "")))
    # loops: components of the signal-line subgraph
    S = nx.Graph()
    S.add_edges_from((u, v) for u, v, d in G.edges(data=True) if d["kind"] == "signal")
    joined = set()
    for comp in nx.connected_components(S):
        inst = [n for n in comp if G.nodes[n]["type"] == "instrument"]
        if len(inst) < 2:
            continue
        nums = sorted({G.nodes[n]["loop"] for n in inst})
        joined.update(inst)
        item = dict(instruments=sorted(G.nodes[n]["tag"] for n in inst), loops=nums,
                    valves=sorted(n for n in comp if G.nodes[n]["type"] == "valve"),
                    xy=_xy(G, inst[0]))
        rep["loops_consistent" if len(nums) == 1 else "loops_mixed"].append(item)
    by_loop = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d["type"] == "instrument" and d["loop"]:
            by_loop[d["loop"]].append(n)
    for num, ns in by_loop.items():
        apart = [n for n in ns if n not in joined]
        if len(ns) >= 2 and len(apart) >= 2:
            rep["loop_not_joined"].append(dict(loop=num, instruments=sorted(G.nodes[n]["tag"] for n in apart)))
    rep = dict(rep)
    rep["counts"] = {k: len(v) for k, v in rep.items()}
    rep["counts"]["dangling_by_kind"] = dict(Counter(p["kind"] for p in rep.get("dangling_edges", [])))
    return rep


PROBLEM_KEYS = ("unconnected_instruments", "unconnected_equipment", "dangling_edges", "flow_conflicts", "valves_off_line")
