"""Tag-extraction precision / recall benchmark for the precision filter's stages, on every set with judged truth.
    .venv/bin/python batch/bench_tags.py [--stages repair split ...] [--ablate]
Raw rows are frozen once in bench_tags/raw_<set>.csv (rid = row id), so stages are always scored on the same reads.
Definition used everywhere: a row is correct when it is an instrument or control function DRAWN on that sheet
(bubble, labelled actuated valve, interlock / trip label at its logic), read exactly. References to tags on other
sheets, notes, dates, equipment, hand valves and misreads are wrong.

  set       precision unit / verdict source                                   recall truth
  eq16      unique tags; PI_TAGS / Flash GC match + by-eye verdicts            tags caught vs PI_TAGS + Flash GC
  decp      60-row sample, by-eye verdicts                                     DECP equipment lists (scoped)
  fresh5    40-row sample, by-eye verdicts                                     Cove Point export tags printed on sheet
  bog       all rows of the 20 LNGC-format sheets, text-layer rules           text-layer bubbles (RO/PG/LG excluded)
  bogv      60-row sample of the 7 vendor sheets, by-eye verdicts             -
  open100   all rows, by-eye verdicts on every GT bubble read                 PID2Graph instrumentation boxes (350)
A row whose tag a stage changed (repair / split) and that no truth decides is listed as 'to judge'."""
import argparse, csv, json, re, shutil, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "batch"))
import tag_filter as TF
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
B = ROOT / "bench_tags"; B.mkdir(exist_ok=True)
SNAP = {"eq16": "lngc_eq16_output/mastersheet.csv", "bhcp": "pids_bhcp_output/mastersheet.csv",
        "bog": "lngc_bog_a_output/mastersheet.csv", "open100": "pidgraph_output/p2g_tags/tags.csv"}
LEDGER = {"eq16": "lngc_eq16_output/ledger.csv", "bhcp": "pids_bhcp_output/ledger.csv", "bog": "lngc_bog_a_output/ledger.csv"}
P2G = ROOT / "external/pid2graph/data/PID2Graph OPEN100"
CACHE = {}
# rows changed by repair / split, judged by eye 2026-10-27 (set, sheet, tag) -> correct
JUDGED = {("open100", "5", "TE-1005A"): True, ("open100", "5", "TE-1008A"): True, ("open100", "5", "TE-1004B"): True,
          ("open100", "10", "TE-11155A"): True}
for _t in ("3111A 3111B 3111C 3114A 3114B 3127C 3127D 3135C 3135B 3135A").split():
    JUDGED[("bogv", "L001-03400-PX-2365-100061-1201_2_1_p0", f"XZA-{_t}-CS")] = True
RAW_COLS = ["sheet", "tag", "source", "lines", "bbox", "confidence"]


def key(t):
    """letters-number(+suffix) without unit prefix: '034-UY-1322' -> 'UY-1322', '5PI-5012' -> 'PI-5012'"""
    m = re.search(r"([A-Z]{1,6})-?(\d{2,6}[A-Z]?|XXX)(?![0-9])", re.sub(r"^\d{1,3}-?", "", t))
    return f"{m.group(1)}-{m.group(2)}" if m else t


nosuf = lambda k: re.sub(r"(?<=\d)[A-Z]$", "", k)
nums = lambda b: tuple(round(float(v)) for v in re.findall(r"-?\d+(?:\.\d+)?", str(b)))


def raw(name):
    f = B / f"raw_{name}.csv"
    if not f.exists():
        rows = list(csv.DictReader(open(ROOT / SNAP[name])))
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["rid"] + RAW_COLS); w.writeheader()
            for i, r in enumerate(rows):
                w.writerow(dict(rid=str(i), **{c: r.get("tag_read") or r[c] if c == "tag" else r.get(c, "") for c in RAW_COLS}))
    return list(csv.DictReader(open(f)))


def pages_for(name):
    if name in LEDGER:
        return TF.pdf_pages(ROOT / LEDGER[name])
    def get(sheet):
        im = Image.open(P2G / f"{sheet}.png").convert("L")
        return im, 1.0, max(im.size) / 841          # OPEN100 sheets are drawn on A1 (841 mm long side)
    return get


# ---------------------------------------------------------------- verdicts and truth per set
def eq16():
    E = ROOT / "lngc_eq16_output/eval"
    v = {}
    for r in csv.DictReader(open(E / "extracted_to_PI_TAGS.csv")):
        if r["status"] in ("exact", "same loop"):
            v[r["key"]] = True
    for r in csv.DictReader(open(E / "not_in_PI_TAGS_verdicts.csv")):
        s = r["verdict"]
        v[r["key"]] = s.startswith("real instrument") or s.startswith("interlock")
    # judged by eye 2026-10-27: kept junk the old eq16 score (unit-111 instrument format only) never counted
    for k in ("MH-600 FF-MUX VLD A-0813 CC-12464 FLA-701 ERO-111 RF-300L CC 11SC1-003 UZ-0520 CCV-111W VM-150 RF-150 "
              "ILO-111V VLD-047 M 111-LI-4-00911 VLD-032 11SC5-002 VLD-044 MIN-600 H").split():
        v.setdefault(k, False)
    truth = set()
    for f in ("PI_TAGS_ground_truth_vs_extracted.csv", "FlashGC_ground_truth_vs_extracted.csv"):
        for r in csv.DictReader(open(E / f)):
            if r["result"].startswith("CAUGHT"):
                truth |= {key(t.strip()) for t in r["what_was_caught"].split("|") if t.strip()}
    return v, truth


def bhcp_samples(rows):
    """rid -> verdict for the DECP and fresh-5 samples (matched on sheet, tag, bbox)."""
    idx = {(r["sheet"], r["tag"], nums(r["bbox"])): r["rid"] for r in rows}
    E = ROOT / "pids_bhcp_output/eval"
    decp, fresh = {}, {}
    for r in csv.DictReader(open(E / "decp_visual/precision_sample.csv")):
        decp[idx[(r["sheet"], r["tag"], nums(r["bbox"]))]] = "read correctly" in r["verdict"]
    good = {0, 1, 2, 3, 4, 6, 9, 12, 13, 14, 16, 17, 18, 19, 21, 22, 24, 25, 26, 28, 29, 30, 31, 33, 34, 36}  # judged by eye
    for r in csv.DictReader(open(E / "fresh5_precision_sample.csv")):
        fresh[idx[(r["sheet"], r["tag"], nums(r["bbox"]))]] = int(r["i"]) in good
    return decp, fresh


def decp_truth():
    out = []
    for r in csv.DictReader(open(ROOT / "pids_bhcp_output/eval/decp_equipment_scoped_recall.csv")):
        lvl = r["status"][0]
        if lvl in "123":
            out.append((key(r["list_tag"]), int(lvl), [s.strip() for s in r["sheets"].split("|")]))
    return out


def fresh5_truth():
    import importlib.util
    ocr = json.load(open(ROOT / "pids_bhcp_output/eval/fresh5_ocr.json"))
    P = re.compile(r"^(?:[A-Z]+:)?(\d{1,2})?([A-Z]{2,5})-?(\d{3,5})([A-Z]\d?)?(?![0-9])")
    plant = set()
    for r in csv.DictReader(open(ROOT / "bhcp/dominion/Cove_Point_LNG_Sensors_export_2026-10-15_15-22-57.csv", encoding="utf-8-sig")):
        m = P.match(r["internal_tag"])
        if m:
            plant.add((m.group(2), m.group(3) + (m.group(4) or "")))
    out = []
    for sh, frs in ocr.items():
        txt = [(f["text"].upper().replace(" ", ""), f["bbox"]) for f in frs]
        c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for L, n in plant:
            rx = re.compile(rf"(?<![A-Z]){L}-?{n}(?![0-9A-Z])")
            ok = any(rx.search(t) for t, _ in txt)
            if not ok:          # bubble: letters and number as two fragments within 12 mm (as eval_decp_equipment.printed)
                for t, b in txt:
                    if t == n:
                        x, y = c(b)
                        ok = any(u == L and abs(c(e)[0] - x) < 142 and abs(c(e)[1] - y) < 142 for u, e in txt)
                        if ok: break
            if ok:
                out.append((f"{L}-{n}", sh))
    return out


def bog_rules():
    """Text-layer judge for the 20 LNGC-format BOG sheets and their in-scope bubbles."""
    import pymupdf
    sys.argv = sys.argv[:1] + ["lngc_bog_a_output", "034"]
    src = open(ROOT / "batch/eval_text_layer.py").read().split("key = lambda")[0]
    ns = {}; exec(src, ns)
    led = [r for r in csv.DictReader(open(ROOT / LEDGER["bog"])) if "100061" not in r["out_name"]]
    info = {}
    for L in led:
        pg = pymupdf.open(L["file"])[0]
        ws = pg.get_text("words")
        info[L["out_name"]] = dict(bub={key(b["tag"]) for b in ns["bubbles"](pg)}, ws=ws)
    def near(ws, w, rad, test):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return any(test(o[4]) and abs((o[0] + o[2]) / 2 - cx) < rad and abs((o[1] + o[3]) / 2 - cy) < rad for o in ws)
    def judge(sheet, tag):
        I = info[sheet]; k = key(tag)
        if k in I["bub"]:
            return True
        if not re.fullmatch(r"\d{3}-[A-Z]{1,6}-\d{4}[A-Z]?", tag):
            return False
        word = re.sub(r"^(\d{3})-", r"\1", tag)
        hits = [w for w in I["ws"] if w[4] in (tag, word)]
        if hits:   # printed as one string: drawn label unless it sits in an off-sheet connector caption
            return not all(near(I["ws"], w, 60, lambda s: bool(re.match(r"L\d{3}-\d{5}", s)) or s in ("TO", "FROM")) for w in hits)
        L, n = k.split("-")
        return any(w[4] == n and near(I["ws"], w, 40, lambda s: s == L) for w in I["ws"])
    truth = []
    for sh, I in info.items():
        truth += [(k, sh) for k in I["bub"] if k.split("-")[0] not in ("RO", "PG", "LG")]
    return judge, truth


def bog_vendor(rows):
    idx = {(r["sheet"], r["tag"], nums(r["bbox"])): r["rid"] for r in rows}
    smp = json.load(open(ROOT / "lngc_bog_a_output/eval/vendor_sample.json"))
    good = {0, 1, 2, 5, 6, 7, 11, 13, 14, 15, 17, 20, 23, 24, 26, 27, 28, 29,
            30, 35, 39, 40, 42, 43, 44, 47, 48, 49, 50, 51, 53, 56, 57, 58}          # judged by eye (sample 0 + 30..59)
    return {idx[(r["sheet"], r["tag"], nums(r["bbox"]))]: i in good for i, r in enumerate(smp)}


def open100():
    g = json.load(open(ROOT / "pidgraph_output/p2g_tags/gt_bubble_reads.json"))
    full = [x for x in g if x["cat"] == "full"]
    wrong = {3: "MSIV-1402", 9: "MS-14010", 11: "FT-1401", 13: "MOV-1531", 14: "MOV-1521", 15: "MOV-1511",
             17: "LIT-16098B", 20: "TE-162100B", 23: "TE-162100A", 47: "PT-250135B", 49: "TE-161143B", 53: "MOV-14010",
             61: "TE-250136B", 66: "PT-1009A", 72: "TE-1005B", 74: "PT-1009B", 76: "TE-1004A", 81: "TE-10211B",
             82: "TE-1008B", 88: "TE-10321", 120: "FT-1001", 168: "MOV-1151", 169: "TE-11155B", 170: "TE-11153B",
             172: "MOV-1123", 175: "MOV-1115", 183: "PIT-11561", 189: "FT-1114", 192: "MOV-1002", 193: "MOV-XXX",
             200: "MOV-1003", 203: "MOV-1004", 207: "FT-1202", 215: "FT-1201", 218: "FT-1211", 219: "MOV-1217"}
    good, truth = defaultdict(set), []
    for i, x in enumerate(full):
        if i in wrong:
            t = [wrong[i]]
        else:
            t = [s for s in x["read"].split("|")]
        good[x["sheet"]] |= {key(s) for s in t}
        truth.append(({key(s) for s in t}, x["sheet"]))
    return good, truth, 350


# ---------------------------------------------------------------- scoring
def score(stages, verbose=False):
    res, todo = {}, []
    # eq16: unique keys
    rows = raw("eq16"); rows, _ = TF.apply(rows, stages, pages_for("eq16"), CACHE.setdefault("eq16", {}))
    v, truth = eq16()
    kept = {key(r["tag"]) for r in rows if r["keep"]}
    c = sum(v.get(k) is True for k in kept); w = sum(v.get(k) is False for k in kept)
    todo += [("eq16", k) for k in kept if k not in v]
    res["eq16"] = (c, w, len(kept) - c - w, len(truth & kept), len(truth))
    # bhcp: DECP + fresh5 samples
    rows = raw("bhcp"); decp, fresh = bhcp_samples(rows)
    rows, _ = TF.apply(rows, stages, pages_for("bhcp"), CACHE.setdefault("bhcp", {}))
    by_sheet = defaultdict(set)
    for r in rows:
        if r["keep"]:
            by_sheet[r["sheet"]].add(key(r["tag"]))
    for nm, V in (("decp", decp), ("fresh5", fresh)):
        c = w = u = 0
        for r in rows:
            base = r["rid"].split(".")[0]
            if base in V and r["keep"]:
                if r["tag"] == r.get("tag_read", r["tag"]):
                    c += V[base]; w += not V[base]
                elif (nm, r["sheet"], r["tag"]) in JUDGED:
                    ok = JUDGED[(nm, r["sheet"], r["tag"])]; c += ok; w += not ok
                else:
                    u += 1; todo.append((nm, r["sheet"], r["tag_read"], r["tag"]))
        res[nm] = [c, w, u]
    dt = decp_truth(); found = 0
    for k, lvl, shs in dt:
        ks = set().union(*[by_sheet[s] for s in by_sheet if any(s.startswith(x) for x in shs)]) if shs else set()
        if lvl == 1: found += k in ks
        elif lvl == 2: found += nosuf(k) in {nosuf(x) for x in ks}
        else: found += any(x.split("-")[0][:1] == k[:1] and nosuf(x.split("-")[-1]) == nosuf(k.split("-")[-1]) for x in ks)
    res["decp"] += [found, len(dt)]
    ft = fresh5_truth()
    res["fresh5"] += [sum(nosuf(k) in {nosuf(x) for x in by_sheet[s]} for k, s in ft), len(ft)]
    # bog: rules on LNGC-format sheets, sample on vendor sheets
    rows = raw("bog"); V = bog_vendor(rows); judge, truth = bog_rules()
    rows, _ = TF.apply(rows, stages, pages_for("bog"), CACHE.setdefault("bog", {}))
    c = w = 0; cv = wv = uv = 0
    for r in rows:
        if not r["keep"]:
            continue
        if "100061" not in r["sheet"]:
            ok = judge(r["sheet"], r["tag"]); c += ok; w += not ok
        else:
            base = r["rid"].split(".")[0]
            if base in V:
                if r["tag"] == r.get("tag_read", r["tag"]):
                    cv += V[base]; wv += not V[base]
                elif ("bogv", r["sheet"], r["tag"]) in JUDGED:
                    ok = JUDGED[("bogv", r["sheet"], r["tag"])]; cv += ok; wv += not ok
                else:
                    uv += 1; todo.append(("bogv", r["sheet"], r["tag_read"], r["tag"]))
    ks = defaultdict(set)
    for r in rows:
        if r["keep"]:
            ks[r["sheet"]].add(key(r["tag"]))
    res["bog"] = (c, w, 0, sum(k in ks[s] for k, s in truth), len(truth))
    res["bogv"] = (cv, wv, uv, 0, 0)
    # open100
    rows = raw("open100"); good, truth, n = open100()
    base_kept = {(r["sheet"], r["tag"]) for r in rows}
    rows, _ = TF.apply(rows, stages, pages_for("open100"), CACHE.setdefault("open100", {}))
    c = w = u = 0; ks = defaultdict(set)
    for r in rows:
        if not r["keep"]:
            continue
        ks[r["sheet"]].add(key(r["tag"]))
        if key(r["tag"]) in good[r["sheet"]]: c += 1
        elif ("open100", r["sheet"], r["tag"]) in JUDGED: c += JUDGED[("open100", r["sheet"], r["tag"])]
        elif (r["sheet"], r["tag"]) in base_kept: w += 1
        else: u += 1; todo.append(("open100", r["sheet"], r.get("tag_read"), r["tag"]))
    res["open100"] = (c, w, u, sum(bool(k & ks[s]) for k, s in truth), n)
    return res, todo


def show(label, res):
    cells = []
    for nm in ("eq16", "decp", "fresh5", "bog", "bogv", "open100"):
        c, w, u, f, t = res[nm]
        p = f"{c / max(1, c + w):6.1%}" + (f"+{u}?" if u else "   ")
        r = f"{f}/{t}" if t else "-"
        cells.append(f"{nm} P {p} R {r:>8}")
    print(f"{label:22s} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(TF.STAGES))
    ap.add_argument("--ablate", action="store_true", help="add the stages one at a time, cumulatively")
    a = ap.parse_args()
    seq = [[]] + [a.stages[:i + 1] for i in range(len(a.stages))] if a.ablate else [a.stages]
    all_todo = []
    for st in seq:
        res, todo = score(st)
        show("+".join(st[-1:]) if st else "current filter", res)
        all_todo = todo
    json.dump(all_todo, open(B / "to_judge.json", "w"), indent=0)
    print(f"{len(all_todo)} changed rows to judge -> bench_tags/to_judge.json")
