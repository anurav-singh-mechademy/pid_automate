"""Compare a batch run's kept tags with a client PI tag list (DCS signals, e.g. CA:KIT:034XZ3135A_ASO.PV).
    .venv/bin/python batch/eval_bog_pi.py lngc_bog_a_output lngc/PI_tags_BOG_Anurav.csv
The PI list is reduced to instruments: letters + loop number + instance letter (signal suffixes _ASO, _MOS, ASOT, Y,
.PV / .MV dropped). Matching: same loop number, same instrument letters give or take alarm / transmitter / switch /
controller letters (bench.same_instrument: TIA ~ TI), instance letters equal or one side without.
  recall     PI instruments printed on the processed drawings - letters and loop number together in the PDF text layer,
             as a bubble (letters word over the number) or inline (PDI-2101) - that were extracted; PI points never drawn on these sheets are listed separately, not counted as misses
  precision  kept extracted tags that are a PI instrument (lower bound: the list holds DCS points only)
Writes <out>/eval/pi_compare.csv (one row per PI instrument and per unmatched kept tag)."""
import csv, re, sys
from collections import defaultdict
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from descgen.bench import same_instrument

OUT, PI = ROOT / sys.argv[1], ROOT / sys.argv[2]
PI_RX = re.compile(r"^CA:KIT:(\d{3})([A-Z]+?)(\d{4})([A-Z]*)")
TAG_RX = re.compile(r"^(?:\d{1,3}-)?([A-Z]{1,6})-(\d{4})([A-Z]?)")


def pi_instruments():
    out = {}
    for r in csv.DictReader(open(PI, encoding="utf-8-sig")):
        m = PI_RX.match(r["PI Tag Name"])
        if not m:
            continue
        unit, L, n, rest = m.groups()
        inst = rest[0] if rest and rest[0] in "ABCDEFGH" else ""
        k = (L, n, inst)
        out.setdefault(k, dict(unit=unit, signals=[], desc=r["PI Tag Description"]))["signals"].append(r["PI Tag Name"])
    return out


def match(a, b):
    """a, b: (letters, number, instance)."""
    return a[1] == b[1] and same_instrument(a[0], b[0]) and (a[2] == b[2] or not a[2] or not b[2])


def main():
    led = [r for r in csv.DictReader(open(OUT / "ledger.csv")) if r["status"] == "ok"]
    printed = set()          # (letters, number) printed together: a bubble (letters word over number) or inline
    for L in led:
        ws = pymupdf.open(ROOT / L["file"])[int(L["page"])].get_text("words")
        for w in ws:
            for m in re.finditer(r"(?<![A-Z])([A-Z]{1,6})-?(\d{4})(?!\d)", w[4]):
                printed.add(m.groups())
        for n in ws:
            if not re.fullmatch(r"\d{4}[A-Z]{0,2}", n[4]):
                continue
            h, cx = n[3] - n[1], (n[0] + n[2]) / 2
            for l in ws:
                if re.fullmatch(r"[A-Z]{1,6}", l[4]) and abs((l[0] + l[2]) / 2 - cx) < 0.8 * h and 0 <= n[1] - l[3] < 1.2 * h:
                    printed.add((l[4], n[4][:4]))
    def is_printed(k):
        return any(n == k[1] and same_instrument(L, k[0]) for L, n in printed)
    kept = [r for r in csv.DictReader(open(OUT / "mastersheet.csv")) if r.get("keep") == "True"]
    ext = defaultdict(list)
    for r in kept:
        m = TAG_RX.match(r["tag"])
        if m:
            ext[m.groups()].append(r)
    pis = pi_instruments()
    rows, found, scope = [], 0, 0
    for k, v in sorted(pis.items()):
        hit = [e for e in ext if match(k, e)]
        on = is_printed(k)
        scope += on; found += bool(hit) and on
        rows.append(dict(kind="PI instrument", key=f"{k[0]}-{k[1]}{k[2]}", printed_on_drawings=on, extracted=bool(hit),
                         extracted_tag=" | ".join(sorted({r["tag"] for e in hit for r in ext[e]})),
                         sheets=" | ".join(sorted({r["sheet"] for e in hit for r in ext[e]})),
                         ground_truth_pi_tag=" | ".join(v["signals"]), pi_description=v["desc"]))
    listed = 0
    for e, rs in sorted(ext.items()):
        if any(match(k, e) for k in pis):
            listed += 1
        else:
            rows.append(dict(kind="extracted, not in PI list", key=f"{e[0]}-{e[1]}{e[2]}", printed_on_drawings=is_printed(e),
                             extracted=True, extracted_tag=" | ".join(sorted({r["tag"] for r in rs})),
                             sheets=" | ".join(sorted({r["sheet"] for r in rs})), ground_truth_pi_tag="", pi_description=""))
    (OUT / "eval").mkdir(exist_ok=True)
    with open(OUT / "eval" / "pi_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_ext = len(ext)
    print(f"sheets {len(led)} | kept rows {len(kept)}, unique instruments {n_ext}")
    print(f"PI list: {sum(len(v['signals']) for v in pis.values())} signals -> {len(pis)} instruments; "
          f"{scope} are printed on these drawings (letters + number), {len(pis) - scope} are not drawn here")
    print(f"recall (printed PI instruments extracted): {found}/{scope} = {found / max(1, scope):.1%}")
    print(f"precision lower bound (kept instruments in PI list): {listed}/{n_ext} = {listed / max(1, n_ext):.1%}")
    print(f"-> {OUT / 'eval' / 'pi_compare.csv'}")


if __name__ == "__main__":
    main()
