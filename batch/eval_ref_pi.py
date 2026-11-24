"""Compare batch runs' kept tags with a PI tag list of the form <unit><letters><loop><instance>.<attr> (e.g.
11PI8071.DACA.PV, 61FI1641.PV). Same rules as eval_bog_pi.py (which reads the CA:KIT: format):
  PI list -> instruments: letters + 4-digit loop + instance letter; the unit prefix and .attributes are dropped
  match      same loop number, same instrument letters give or take alarm / transmitter / switch / controller letters
             (bench.same_instrument), instance letters equal or one side without
  recall     PI instruments printed on the processed drawings (letters + loop together in the PDF text layer) that
             were extracted; PI points never drawn on these sheets are listed, not counted as misses
  precision  kept extracted tags that are a PI instrument (lower bound: the list holds DCS points only)
  lenient    (default) a drawn transmitter counts as its DCS indicator point: PT ~ PI, TT ~ TI, PDT ~ PDI;
             --strict uses bench.same_instrument only
    .venv/bin/python batch/eval_ref_pi.py ~/Downloads/PI_tags_testing_Anurav.csv ref1_output ref2_output [--strict]
Writes eval/pi_compare.csv in the first run folder (one row per PI instrument and per unmatched kept tag)."""
import csv, re, sys
from collections import defaultdict
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from descgen.bench import same_instrument

PI_RX = re.compile(r"^(\d{1,3})([A-Z]+?)(\d{4})([A-Z]*)")
TAG_RX = re.compile(r"^(?:\d{1,3}-?)?([A-Z]{1,6})-(\d{4})([A-Z]?)")


def pi_instruments(path):
    out, skipped = {}, []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        name = r.get("PI tag Name") or r.get("PI Tag Name")
        m = PI_RX.match(name.split(".")[0])
        if not m:
            skipped.append(name); continue
        unit, L, n, rest = m.groups()
        inst = rest[0] if rest and rest[0] in "ABCDEFGH" else ""
        v = out.setdefault((L, n, inst), dict(units=set(), signals=[], desc=r.get("PI tag Description") or r.get("PI Tag Description")))
        v["units"].add(unit); v["signals"].append(name)
    return out, skipped


LENIENT = "--strict" not in sys.argv


def tx(L):
    """A drawn transmitter's DCS point is the indicator: PT -> PI, PDT -> PDI, TT -> TI (lenient mode only)."""
    return L[:-1] + "I" if len(L) >= 2 and L.endswith("T") else L


def same(a, b):
    return same_instrument(a, b) or (LENIENT and same_instrument(tx(a), tx(b)))


def match(a, b):
    return a[1] == b[1] and same(a[0], b[0]) and (a[2] == b[2] or not a[2] or not b[2])


def printed_keys(run):
    printed = set()
    for L in csv.DictReader(open(run / "ledger.csv")):
        if L["status"] != "ok":
            continue
        ws = pymupdf.open(ROOT / L["file"])[int(L["page"])].get_text("words")
        for w in ws:
            for m in re.finditer(r"(?<![A-Z])([A-Z]{1,6})-?(\d{4})(?!\d)", w[4]):
                printed.add(m.groups())
        for n in ws:
            if not re.fullmatch(r"\d{4}[A-Z]{0,2}", n[4]):
                continue
            h, cx = n[3] - n[1], (n[0] + n[2]) / 2
            for l in ws:
                lw = re.fullmatch(r"\d{0,3}([A-Z]{1,6})", l[4])            # bubble letters may carry the unit: 11TT
                if lw and abs((l[0] + l[2]) / 2 - cx) < 0.8 * h and 0 <= n[1] - l[3] < 1.2 * h:
                    printed.add((lw.group(1), n[4][:4]))
    return printed


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pi_path, runs = Path(args[0]).expanduser(), [ROOT / r for r in args[1:]]
    pis, skipped = pi_instruments(pi_path)
    printed = {r.name: printed_keys(r) for r in runs}
    ext = defaultdict(list)                       # (letters, number, inst) -> kept rows (with run)
    for run in runs:
        for r in csv.DictReader(open(run / "mastersheet.csv")):
            m = TAG_RX.match(r["tag"]) if r.get("keep") == "True" else None
            if m:
                ext[m.groups()].append(dict(r, run=run.name))

    def on(k):
        return [n for n, P in printed.items() if any(num == k[1] and same(L, k[0]) for L, num in P)]

    rows = []
    stats = defaultdict(lambda: [0, 0])           # run -> [printed PI instruments, extracted]
    for k, v in sorted(pis.items()):
        hit = [e for e in ext if match(k, e)]
        where = on(k)
        for n in where:
            stats[n][0] += 1
            stats[n][1] += any(r["run"] == n for e in hit for r in ext[e])
        rows.append(dict(kind="PI instrument", key=f"{k[0]}-{k[1]}{k[2]}", units="/".join(sorted(v["units"])),
                         printed_on=" | ".join(where), extracted=bool(hit),
                         extracted_tag=" | ".join(sorted({r["tag"] for e in hit for r in ext[e]})),
                         sheets=" | ".join(sorted({r["sheet"].split("_")[-1] + "@" + r["run"] for e in hit for r in ext[e]})),
                         ground_truth_pi_tag=" | ".join(v["signals"]), pi_description=v["desc"]))
    listed = defaultdict(lambda: [0, 0])          # run -> [unique kept instruments, in PI list]
    for e, rs in sorted(ext.items()):
        inlist = any(match(k, e) for k in pis)
        for n in {r["run"] for r in rs}:
            listed[n][0] += 1; listed[n][1] += inlist
        if not inlist:
            rows.append(dict(kind="extracted, not in PI list", key=f"{e[0]}-{e[1]}{e[2]}", units="",
                             printed_on=" | ".join(on(e)), extracted=True,
                             extracted_tag=" | ".join(sorted({r["tag"] for r in rs})),
                             sheets=" | ".join(sorted({r["sheet"].split("_")[-1] + "@" + r["run"] for r in rs})),
                             ground_truth_pi_tag="", pi_description=""))
    out = runs[0] / "eval" / ("pi_compare.csv" if LENIENT else "pi_compare_strict.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    scope = sum(1 for r in rows if r["kind"] == "PI instrument" and r["printed_on"])
    found = sum(1 for r in rows if r["kind"] == "PI instrument" and r["printed_on"] and r["extracted"])
    print(f"PI list: {sum(len(v['signals']) for v in pis.values())} signals -> {len(pis)} instruments "
          f"({len(skipped)} names not parsed: {', '.join(skipped)})")
    print(f"printed on these drawings: {scope}; not drawn here: {len(pis) - scope}")
    for n in printed:
        p, f = stats[n]; u, l = listed[n]
        print(f"  {n}: recall {f}/{p} = {f / max(1, p):.1%} | precision lower bound {l}/{u} = {l / max(1, u):.1%}")
    u = len(ext); l = sum(any(match(k, e) for k in pis) for e in ext)
    print(f"  all: recall {found}/{scope} = {found / max(1, scope):.1%} | precision lower bound {l}/{u} = {l / max(1, u):.1%}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
