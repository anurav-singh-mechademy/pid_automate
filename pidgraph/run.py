"""Build, validate and export the sheet graph for sheets of descgen.bench.SHEETS.
    .venv/bin/python -m pidgraph.run "SC62D1914_p0" ... [--out pidgraph_output]"""
import argparse
import json
from .graph import SheetGraph
from .export import export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+")
    ap.add_argument("--out", default="pidgraph_output")
    a = ap.parse_args()
    for n in a.sheets:
        sg = SheetGraph(n)
        summ, _ = export(sg, a.out)
        print(json.dumps(summ, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
