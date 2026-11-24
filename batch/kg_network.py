"""Interactive network view of a pid-to-kg graph.json (networkx + pyvis, no model calls).

    .venv/bin/python batch/kg_network.py work/decp_SC62D1931/kg/graph.json [--pipes] [--loops] [--out file.html]

Pipe segments are contracted by default: A -> pipe -> pipe -> B becomes one edge A -> B labelled with the
line number, and an instrument measuring a pipe segment is linked to the item just upstream of it.
--pipes keeps every segment as a node; --loops adds control-loop nodes and their member edges.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from pyvis.network import Network

COLOR = {"Equipment": "#e63946", "Instrument": "#3a86ff", "Component": "#ffb703", "PipeSegment": "#80b918",
         "ControlLoop": "#9d4edd", "OffPageConnector": "#00b4a6", "Boundary": "#8d99ae"}
SHAPE = {"Equipment": "box", "Instrument": "dot", "Component": "diamond", "PipeSegment": "dot",
         "ControlLoop": "hexagon", "OffPageConnector": "triangle", "Boundary": "square"}
SIZE = {"Equipment": 30, "Instrument": 12, "Component": 9, "PipeSegment": 5, "ControlLoop": 14,
        "OffPageConnector": 16, "Boundary": 12}
EDGE = {"FLOWS_TO": "#80b918", "FEEDS": "#e63946", "SIGNAL_TO": "#3a86ff", "MEASURES": "#00b4a6",
        "ACTUATES": "#ffb703", "TRIPS": "#ff595e", "MEMBER_OF": "#9d4edd", "CASCADES_TO": "#555", "PART_OF": "#aaa"}
FLOW = ("FLOWS_TO", "FEEDS")


def load(path):
    g = json.load(open(path))
    G = nx.MultiDiGraph(title=g["plant"]["sheets"][0].replace("SHEET:", ""))
    for n in g["nodes"]:
        if n["type"] not in ("Sheet", "Plant"):
            G.add_node(n["id"], **n)
    for e in g["edges"]:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"], kind=e["type"], **e.get("attributes", {}))
    return G


def contract_pipes(G):
    """Replace runs of PipeSegment nodes with direct edges between the items they connect."""
    pipe = {n for n, d in G.nodes(data=True) if d["type"] == "PipeSegment"}
    H = nx.MultiDiGraph(**G.graph)
    H.add_nodes_from((n, d) for n, d in G.nodes(data=True) if n not in pipe)
    flow_out = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d["kind"] in FLOW:
            flow_out[u].append((v, d))
    upstream = {}                                    # pipe segment -> nearest non-pipe item feeding it
    for u in H.nodes:
        stack, seen = [(v, d.get("line_number", "")) for v, d in flow_out[u]], set()
        while stack:
            v, line = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if v in pipe:
                upstream.setdefault(v, u)
                stack += [(w, line or d.get("line_number", "")) for w, d in flow_out[v]]
            elif v != u:
                H.add_edge(u, v, kind="FLOWS_TO", line_number=line)
    for u, v, d in G.edges(data=True):
        if d["kind"] in FLOW:
            continue
        u2, v2 = upstream.get(u, u), upstream.get(v, v)
        if u2 in H and v2 in H and u2 != v2:
            extra = {"via_line": G.nodes[v].get("tag", "")} if v in pipe else {}
            H.add_edge(u2, v2, **d, **extra)
    return H


def render(G, out, loops):
    net = Network(height="95vh", width="100%", directed=True, bgcolor="#ffffff", font_color="#222",
                  select_menu=True, filter_menu=True, cdn_resources="in_line")
    for n, d in G.nodes(data=True):
        if d["type"] == "ControlLoop" and not loops:
            continue
        tip = "\n".join([f"{d.get('label', n)}  [{d['type']}]", d.get("name", "")] +
                        [f"{k}: {v}" for k, v in d.get("attributes", {}).items() if v not in ("", None, [], {})])
        net.add_node(n, label=d.get("label", n), title=tip, color=COLOR.get(d["type"], "#ccc"),
                     shape=SHAPE.get(d["type"], "dot"), size=SIZE.get(d["type"], 10), group=d["type"])
    shown = set(net.get_nodes())
    for u, v, d in G.edges(data=True):
        if u not in shown or v not in shown:
            continue
        label = d.get("line_number", "") if d["kind"] in FLOW else ""
        tip = "\n".join([d["kind"]] + [f"{k}: {x}" for k, x in d.items() if k != "kind" and x not in ("", None)])
        net.add_edge(u, v, title=tip, label=label, color=EDGE.get(d["kind"], "#999"),
                     dashes=d["kind"] in ("SIGNAL_TO", "MEMBER_OF"), width=3 if d["kind"] in FLOW else 1.2)
    net.set_options(json.dumps({
        "layout": {"randomSeed": 7, "improvedLayout": True},
        "physics": {"solver": "forceAtlas2Based", "stabilization": {"iterations": 600},
                    "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 90, "avoidOverlap": 0.4}},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}, "smooth": {"type": "continuous"},
                  "font": {"size": 9, "align": "middle", "color": "#555"}},
        "interaction": {"hover": True, "navigationButtons": True, "multiselect": True}}))
    net.write_html(str(out), notebook=False)
    return len(shown), sum(1 for _ in net.get_edges())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--pipes", action="store_true", help="keep pipe segments as nodes")
    ap.add_argument("--loops", action="store_true", help="show control-loop nodes")
    ap.add_argument("--out")
    a = ap.parse_args()
    G = load(a.graph)
    if not a.pipes:
        G = contract_pipes(G)
    out = Path(a.out or Path(a.graph).with_name("graph_network.html"))
    n, e = render(G, out, a.loops)
    print(f"{n} nodes, {e} edges -> {out}")


if __name__ == "__main__":
    main()
