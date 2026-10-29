"""Step 1: build the sheet graph for every sheet with client reference descriptions (descgen.bench.SHEETS) and save
it queryable (pidgraph.query.SheetQuery) plus GraphML / overlay / summary.
    .venv/bin/python -m descgraph.build [--set dev|test|all] [--out descgraph_output]
<out>/graphs/<sheet>.gpickle   networkx MultiDiGraph; G.graph also holds the sheet's title-block reading and profile
<out>/graphs/<sheet>.graphml / _overlay.png / .json   (pidgraph.export)"""
import argparse, json, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    sys.path.insert(0, str(p))
from descgen.bench import SHEETS, SETS
from pidgraph.graph import SheetGraph
from pidgraph.export import export


def build(name, out):
    sg = SheetGraph(name)
    t = sg.s.title
    sg.G.graph.update(title_equipment=t["equipment"] or "", title_system=t["system"] or "",
                      title_lines=" / ".join(map(str, t["lines"])))
    g = out / "graphs"; g.mkdir(parents=True, exist_ok=True)
    pickle.dump(sg.G, open(g / f"{name}.gpickle", "wb"))
    summ, _ = export(sg, g)
    return sg, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all")
    ap.add_argument("--out", default="descgraph_output")
    a = ap.parse_args()
    out = ROOT / a.out
    names = list(SHEETS) if a.set == "all" else SETS[a.set]
    for n in names:
        t0 = time.time()
        if (out / "graphs" / f"{n}.gpickle").exists():
            print("have", n); continue
        try:
            sg, summ = build(n, out)
            print(f"{n}: {summ['nodes'].get('instrument', 0)} instruments, {summ['nodes'].get('equipment', 0)} equipment, "
                  f"{summ['edges'].get('process', 0)} pipe edges, {round(time.time() - t0)} s", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
