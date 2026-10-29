"""LLM judge (descgraph.judge: local Qwen2.5-14B-Instruct-4bit, greedy, cached) on descgen bench outputs.
    .venv/bin/python -m descgen_lab.judge_current dev=descgen_lab/results/dev_instance test=descgen_lab/results/test_current ...
Every candidate reference of a tag (same loop + same instrument, as descgraph.run gives it) goes to the judge.
Prints word-F1, exact and LLM score per set; writes <folder>/judged.csv."""
import csv, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "batch"))
from descgen.bench import _refs, SHEETS, same_instrument
from descgen.describe import parse_tag
from descgraph.judge import Judge


def main():
    refs = _refs()
    J = Judge(ROOT / "descgraph_output" / "judge_cache.json")
    for arg in sys.argv[1:]:
        name, folder = arg.split("=", 1)
        rows = []
        for f in sorted(glob.glob(str(ROOT / folder / "*.csv"))):
            if f.endswith(("summary.csv", "judged.csv")):
                continue
            for r in csv.DictReader(open(f)):
                if not r["reference"]:
                    continue
                p = parse_tag(r["tag"])
                cands = [n for L, n in refs[SHEETS[r["sheet"]][2]].get((p["letters"][0], p["number"], p["suffix"]), [])
                         if same_instrument(p["letters"], L)]
                rows.append(dict(sheet=r["sheet"], tag=r["tag"], gen=r["description"], refs=cands or [r["reference"]],
                                 f1=float(r["f1"]), exact=r["exact"] == "True"))
        res = J.score([dict(tag=r["tag"], gen=r["gen"], refs=r["refs"]) for r in rows])
        for r, v in zip(rows, res):
            r.update(verdict=v["verdict"], score=v["score"], missing=v["missing"], wrong=v["wrong"])
        with open(ROOT / folder / "judged.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sheet", "tag", "gen", "refs", "f1", "exact", "verdict", "score", "missing", "wrong"])
            w.writeheader(); w.writerows([dict(r, refs=" | ".join(r["refs"])) for r in rows])
        n = len(rows); sheets = len({r["sheet"] for r in rows})
        c = {k: sum(r["verdict"] == k for r in rows) for k in ("same", "partial", "different")}
        print(f"{name}: {sheets} sheets, {n} tags | word-F1 {sum(r['f1'] for r in rows) / n:.3f} | exact {sum(r['exact'] for r in rows)}"
              f" | LLM {100 * sum(r['score'] for r in rows) / n:.1f}%  same {c['same']} partial {c['partial']} different {c['different']}",
              flush=True)


if __name__ == "__main__":
    main()
