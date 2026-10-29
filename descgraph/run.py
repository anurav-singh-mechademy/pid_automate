"""Run steps 2-4 on the sheets with client reference descriptions and compare three systems on the same tags:
  pass1     graph-independent lookups only (descgraph.pass1)
  graph     pass 1 + graph queries (descgraph.pass2)            <- the final descriptions
  descgen   the existing image-feature description pipeline (descgen.describe), for comparison
  hybrid    descgen's pieces + topology pieces from the graph (descgraph.hybrid, option 2)
    .venv/bin/python -m descgraph.run [--set dev|test|all] [--out descgraph_output] [--no-judge]
Needs the graphs from descgraph.build. Writes <out>/descriptions_<set>.csv (one row per tag with a reference:
references, the three descriptions, verdicts, graph evidence) and prints the scores per set."""
import argparse, csv, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    sys.path.insert(0, str(p))
from descgen.bench import SHEETS, SETS, _refs, same_instrument, f1, run_sheet
from descgen.describe import parse_tag
from pidgraph.query import SheetQuery
from .pass1 import sheet_pass1
from .pass2 import graph_pass
from .judge import Judge
from .hybrid import combine
import json

SYSTEMS = ("pass1", "graph", "descgen", "hybrid")


def references(refs, refset, tag):
    p = parse_tag(tag)
    if not p:
        return []
    return [n for L, n in refs[refset].get((p["letters"][0], p["number"], p["suffix"]), []) if same_instrument(p["letters"], L)]


def run(names, out, judge=True, graphs=None):
    refs = _refs()
    rows = []
    for name in names:
        gp = (graphs or out / "graphs") / f"{name}.gpickle"
        if not gp.exists():
            print("no graph for", name); continue
        q = SheetQuery.load(gp)
        p1, profile, refset = sheet_pass1(name)
        dg = {r["tag"]: r for r in run_sheet(name, "descgen", refs)[0]}
        for tag, r1 in p1.items():
            cands = references(refs, refset, tag)
            if not cands:
                continue
            r2 = graph_pass(q, tag, r1, profile)
            d = dg.get(tag)
            hs, hdesc, hwhy = combine(json.loads(d["pieces"]) if d else {}, r2["slots"], profile)
            rows.append(dict(sheet=name, set="dev" if name in SETS["dev"] else "test" if name in SETS["test"] else "other", tag=tag,
                             references=" | ".join(dict.fromkeys(cands)), pass1=r1["description"],
                             graph=r2["description"], descgen=d["description"] if d else "", hybrid=hdesc,
                             hybrid_changes=hwhy, evidence=r2["evidence"],
                             graph_pieces="; ".join(f"{k}={v['value']} [{v['source']}]" for k, v in r2["slots"].items())))
        print(f"{name}: {sum(r['sheet'] == name for r in rows)} tags with a reference", flush=True)
    if judge:
        J = Judge(out / "judge_cache.json")
        for s in SYSTEMS:
            res = J.score([dict(tag=r["tag"], gen=r[s], refs=r["references"].split(" | ")) for r in rows])
            for r, v in zip(rows, res):
                r[f"{s}_verdict"] = v["verdict"]; r[f"{s}_score"] = v["score"]
                r[f"{s}_missing"] = v["missing"]; r[f"{s}_wrong"] = v["wrong"]
    for r in rows:
        for s in SYSTEMS:
            r[f"{s}_f1"] = round(max(f1(r[s], c) for c in r["references"].split(" | ")), 3)
    return rows


def report(rows):
    for st in ("dev", "test", "other"):
        rs = [r for r in rows if r["set"] == st]
        if not rs:
            continue
        print(f"\n{st}: {len(rs)} tags on {len({r['sheet'] for r in rs})} sheets")
        for s in SYSTEMS:
            line = f"  {s:8s} word-F1 {sum(r[f'{s}_f1'] for r in rs) / len(rs):.3f}"
            if f"{s}_verdict" in rs[0]:
                c = Counter(r[f"{s}_verdict"] for r in rs)
                line += (f" | LLM score {sum(r[f'{s}_score'] for r in rs) / len(rs):.1%}  same {c['same']}  "
                         f"partial {c['partial']}  different {c['different']}" + (f"  error {c['error']}" if c["error"] else ""))
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all"); ap.add_argument("--out", default="descgraph_output")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--graphs", default=None, help="folder of .gpickle graphs to query (default <out>/graphs)")
    ap.add_argument("--sheets", nargs="*", help="only these sheets")
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    a = ap.parse_args()
    out = ROOT / a.out
    names = a.sheets or (list(SHEETS) if a.set == "all" else SETS[a.set])
    rows = run(names, out, judge=not a.no_judge, graphs=ROOT / a.graphs if a.graphs else None)
    with open(out / f"descriptions_{a.set}{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report(rows)
