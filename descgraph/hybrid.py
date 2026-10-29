"""Option 2: the description pipeline's finders (descgen) for text and geometry pieces, the graph for topology.
  base       every piece descgen found (notes, machine parts, cylinder ends, stage, variable, index ...)
  fill       topology pieces descgen left empty come from the graph pass: equipment, instance, stream / position
             (+ item on auxiliary-system sheets)
  override   the stream / position side from the graph replaces descgen's only on strong flow evidence: >= 2 pipe
             edges with known flow on the path, all pointing the same way
Stage, part, end, variable, index stay descgen's (the graph has no note text or landing position on parts)."""
import re
from descgen.describe import compose

TOPOLOGY = ("equipment", "instance", "stream", "position", "item")
SIDE = ("stream", "position")


def flow_evidence(src):
    m = re.search(r"flow edges \+(\d+)/-(\d+)", src or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def combine(dg_slots, g_slots, profile):
    s = {k: dict(value=v[0], source=f"descgen: {v[1]}", conf=v[2]) for k, v in dg_slots.items()}
    why = []
    for k in TOPOLOGY:
        g = g_slots.get(k)
        if not g or not str(g["source"]).startswith(("graph", "loop member")):
            continue
        if k not in s or not s[k]["value"]:
            s[k] = dict(g, source="graph: " + g["source"]); why.append(f"{k} from graph")
        elif k in SIDE and g["value"] != s[k]["value"]:
            eq = g_slots.get("equipment") or g_slots.get("item") or {}
            f, b = flow_evidence(eq.get("source"))
            if max(f, b) >= 2 and min(f, b) == 0:
                s[k] = dict(g, source=f"graph (flow +{f}/-{b}) over descgen '{s[k]['value']}'"); why.append(f"{k} overridden by flow")
    return s, compose(s, profile), "; ".join(why)
