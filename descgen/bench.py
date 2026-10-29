"""Shared test bench for the description pipeline: every sheet with client reference names, one scoring method.
    python -m descgen.bench --set dev            # tuning sheets
    python -m descgen.bench --set test           # held-out sheets: run only for the final measurement
    python -m descgen.bench --pkg lab.a.descgen  # score another copy of the pipeline
Writes <out>/<sheet>.csv (one row per described tag with its reference) and <out>/summary.csv.
Reference files are only read. Scoring ignores word order, case and the house abbreviations (Comp = Compressor ...).
A reference counts for a drawn tag only if it is the same loop AND the same instrument (see same_instrument)."""
import argparse, csv, importlib, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "batch")); sys.path.insert(0, str(ROOT))
CACHE = ROOT / "descgen_lab" / "cache" / "ocr"; CACHE.mkdir(parents=True, exist_ok=True)
DL = Path(os.path.expanduser("~/Downloads/pids_bhcp"))

EQ16 = ("lngc_eq16_output", "L001-11100-PX-2365-")
SHEETS = {   # name: (run folder (ledger + sheets/), cached full-page OCR, reference set)
    **{f"L001-11100-PX-2365-{k}": (EQ16[0], "lngc_eq16_output/eval/ocr_full.json", "flashgc") for k in
       ["1006_3_1 PID Flash Vessel_p0", "1007_3_1 PID Flash Gas Cooler_p0", "1008_3_1 PID Flash Gas Suction Drum_p0",
        "1009_3_1 PID Comp Process_p0", "4440_1_1PID Comp Process_p0", "4441_1_1 PID Comp Mechanical_p0",
        "4442_1_1 PID Comp Unloaders_p0", "4443_1_1 PID Comp Lube Oil_p0", "4444_1_1 PID Comp Cooling System_p0",
        "4445_1_1 PID LO cooler and Intercooler_p0"]},
    **{k: ("pids_bhcp_output", "pids_bhcp_output/eval/decp_visual/ocr_full.json", "decp") for k in
       ["SC62D1911_p0", "SC62D1913_p0", "SC62D1914_p0", "SC62D1912_p0", "SC62D1531 rev 5 Master_p0",
        "SC62D1953 rev 6 Master_p0", "SC62D1930_p0", "SC62D1931_p0"]},
    **{k: ("lngc_legacy", None, "lngc_sensors") for k in
       ["L001-11100-PX-2365-1019_3_1_P1101A Pump Process_p0", "L001-11100-PX-2365-4446_1__1 1P1101A Lube Oil_p0",
        "L001-11100-PX-2365-4447_1_1 1P1101B Lube Oil_p0", "L001-11100-PX-2365-4448_1_1P1101A Vibration_p0"]},
}
# split v2 (batch/pid_registry.py, frozen in descgen/desc_split_v2.json): unused DECP-batch sheets with references,
# matched by loop number only (not checked by eye) -> sets 'dev2' / 'test2'
_V2 = json.load(open(ROOT / "descgen" / "desc_split_v2.json"))["sheets"] if (ROOT / "descgen" / "desc_split_v2.json").exists() else {}
for _k in _V2:
    SHEETS.setdefault(_k, ("pids_bhcp_output", None, "decp"))
# tuning sheets vs held-out sheets (nobody tunes on 'test'; run it once at the end)
_V1 = [k for k in SHEETS if k not in _V2]
SETS = {"dev": [k for k in _V1 if any(s in k for s in ("1007_", "1008_", "1009_", "4440_", "4441_", "4443_", "4444_",
                                                           "4445_", "SC62D1914", "SC62D1911_", "SC62D1913"))],
        "test": [k for k in _V1 if any(s in k for s in ("1006_", "4442_", "SC62D1912", "SC62D1930", "SC62D1931",
                                                            "1019_3_1", "4446_", "4447_", "4448_"))],
        "dev2": sorted(k for k, v in _V2.items() if v == "dev"), "test2": sorted(k for k, v in _V2.items() if v == "test")}


# ---------- reference names (key = first letter + loop number + suffix: DCS names drop alarm letters, TIA -> TI) ----------
def _refs():
    import openpyxl
    R = defaultdict(lambda: defaultdict(list))
    wb = openpyxl.load_workbook(DL / "Flash GC sensors for Anurav.xlsx", read_only=True, data_only=True)
    for name, ext, *_ in list(wb.active.iter_rows(values_only=True))[1:]:
        m = re.match(r"CA:KIT:111([A-Z]+?)(\d{4})([A-Z]?)(?:[A-Z]{2,3})?\.", str(ext))     # also 111KS0813VS.PV
        if m: R["flashgc"][(m.group(1)[0], m.group(2), m.group(3))].append((m.group(1), name))
    for r in csv.DictReader(open(ROOT / "lngc" / "LNG_Canada_Sensors_export_2026-06-17_15-58-07 1(Sheet1).csv", encoding="utf-8-sig")):
        m = re.match(r"CA:KIT:111([A-Z]+?)(\d{4})([A-Z0-9-]*)\.", r["external_tag"])
        if m:
            rest = re.sub(r"(-PV\d+|FB|ASOT)$", "", m.group(3))
            R["lngc_sensors"][(m.group(1)[0], m.group(2), rest if re.fullmatch(r"[A-Z]", rest) else "")].append((m.group(1), r["name"]))
    wb = openpyxl.load_workbook(DL / "DECP" / "DECP Master Tag List.xlsx", read_only=True, data_only=True)
    rows = list(wb["PI_tags_Master"].iter_rows(values_only=True))
    for r in rows[1:]:
        m = r[0] and re.match(r"^5?([A-Z]{1,5})-?(\d{3,5})([A-Z]?)(?![0-9A-Z])", str(r[0]))
        if m and r[2]: R["decp"][(m.group(1)[0], m.group(2), m.group(3))].append((m.group(1), r[2]))
    return R


# ---------- scoring ----------
EXTRA = set("AHLTSC")      # alarm / transmitter / switch / controller letters that drawings add and DCS names drop
def same_instrument(drawn, listed):
    """TIA ~ TI, PIT ~ PI, LICA ~ LIC, PISA ~ PI; but TSHH !~ TI, TE !~ TI (different instruments on one loop)."""
    a, b = (drawn, listed) if len(drawn) >= len(listed) else (listed, drawn)
    return a.startswith(b) and set(a[len(b):]) <= EXTRA


SYN = {"comp": "compressor", "temp": "temperature", "stg": "stage", "press": "pressure", "vib": "vibration",
       "lo": "lube oil", "cw": "cooling water", "tcw": "cooling water", "diff": "differential", "dp": "differential pressure",
       "de": "drive end", "nde": "non drive end", "no": "non", "he": "head end", "ce": "crank end", "pos": "position",
       "disch": "discharge", "dsch": "discharge", "suct": "suction", "vlv": "valve", "brg": "bearing", "mtr": "motor",
       "lvl": "level", "flw": "flow", "inl": "inlet", "outl": "outlet", "cyl": "cylinder", "w": "", "the": ""}
def norm(s):
    out = []
    for w in re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("-", " ")).split():
        out += SYN.get(w, w).split()
    return out
def f1(a, b):
    A, B = norm(a), norm(b)
    c = sum((Counter(A) & Counter(B)).values())
    return 0.0 if not c else 2 * c / (len(A) + len(B))


# ---------- one sheet ----------
EXACT_TEXT = True       # use the PDF's own text (text layer / AutoCAD SHX comments) where the page has it


def exact_text(name, frags):
    """OCR fragments merged with the exact text the PDF page carries (batch/pdftext.py); OCR only where none."""
    if not EXACT_TEXT:
        return frags
    from pdftext import exact_frags, merge
    run = SHEETS[name][0]
    if run == "lngc_legacy":
        stem = name[:-3]
        f, pg, rot = next(p for p in (ROOT / "lngc").iterdir() if p.stem == stem), 0, 0
    else:
        L = {r["out_name"]: r for r in csv.DictReader(open(ROOT / run / "ledger.csv"))}[name]
        f, pg, rot = L["file"], int(L["page"]), int(L["rotation"] or 0)
    ex, _ = exact_frags(f, pg, rot)
    return merge(ex, frags) if ex else frags


def load_sheet(name):
    from tag_pipeline import load_page, ocr_page, RapidOCR
    run, ocr_file, ref = SHEETS[name]
    if run == "lngc_legacy":
        stem = name[:-3]
        pdf = next(p for p in (ROOT / "lngc").iterdir() if p.stem == stem)
        page, tags_csv = load_page(pdf, 0), ROOT / "outputs" / f"tags_{name}.csv"
    else:
        led = {r["out_name"]: r for r in csv.DictReader(open(ROOT / run / "ledger.csv"))}[name]
        page = load_page(Path(led["file"]), int(led["page"]))
        if int(led["rotation"] or 0): page = page.rotate(int(led["rotation"]), expand=True)
        tags_csv = ROOT / run / "sheets" / f"tags_{name}.csv"
    frags = json.load(open(ROOT / ocr_file)).get(name) if ocr_file else None
    if frags is None:
        cf = CACHE / f"{name}.json"
        if not cf.exists():
            json.dump([dict(text=f["text"], bbox=[round(v) for v in f["bbox"]]) for f in ocr_page(page, RapidOCR())], open(cf, "w"))
        frags = json.load(open(cf))
    frags = exact_text(name, frags)
    bubbles = {}
    for r in csv.DictReader(open(tags_csv)):
        if r["source"] in ("bubble", "stack"):
            bubbles.setdefault(r["tag"], [int(float(v)) for v in r["bbox"].strip("[]").split(",")])
    return page, frags, bubbles, ref


def describe_one(name, pkg="descgen"):
    S = importlib.import_module(pkg + ".sheet"); D = importlib.import_module(pkg + ".describe")
    page, frags, bubbles, refset = load_sheet(name)
    sheet = S.Sheet(page, frags)
    sib = Counter((p["letters"], p["number"]) for p in map(D.parse_tag, bubbles) if p)
    res = D.describe_sheet(sheet, bubbles, sib) if hasattr(D, "describe_sheet") else \
        {t: D.describe_tag(sheet, t, b, sib) for t, b in sorted(bubbles.items())}
    return res, bubbles, refset, sheet


def score_sheet(name, res, bubbles, refset, refs, out=None, pkg="descgen"):
    D = importlib.import_module(pkg + ".describe")
    rows = []
    for tag, bbox in sorted(bubbles.items()):
        r = res[tag]
        p = D.parse_tag(tag)
        cands = [n for L, n in refs[refset].get((p["letters"][0], p["number"], p["suffix"]), [])
                 if same_instrument(p["letters"], L)] if p else []
        best = max(cands, key=lambda c: f1(r["description"], c)) if cands else ""
        rows.append(dict(sheet=name, tag=tag, description=r["description"], reference=best,
                         exact=bool(best) and norm(best) == norm(r["description"]),
                         f1=round(f1(r["description"], best), 3) if best else "",
                         confidence=r.get("confidence", ""), review=r.get("review", ""),
                         pieces=json.dumps({k: [v["value"], v["source"], v["conf"]] for k, v in r["slots"].items()}),
                         bbox=bbox))
    if out and rows:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def run_sheet(name, pkg="descgen", refs=None, out=None):
    res, bubbles, refset, sheet = describe_one(name, pkg)
    return score_sheet(name, res, bubbles, refset, refs or _refs(), out, pkg), sheet


def build_register(pkg, refsets):
    """Cross-sheet equipment-name register from all drawings of the same batch (the drawings are inputs, not answers)."""
    S = importlib.import_module(pkg + ".sheet"); P = importlib.import_module(pkg + ".plant")
    batch = []
    for n, (_, _, rs) in SHEETS.items():
        if rs in refsets:
            page, frags, _, _ = load_sheet(n)
            batch.append((n, S.Sheet(page, frags)))
    reg = P.build_register(batch)
    print(f"register: {len(reg)} equipment names from {len(batch)} drawings", flush=True)


def summarize(rows):
    sc = [r for r in rows if r["reference"]]
    if not sc:
        return dict(tags=len(rows), scored=0, exact=0, f1=0.0)
    return dict(tags=len(rows), scored=len(sc), exact=sum(r["exact"] for r in sc), f1=round(sum(r["f1"] for r in sc) / len(sc), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev"); ap.add_argument("--pkg", default="descgen")
    ap.add_argument("--sheets", nargs="*"); ap.add_argument("--out", default=None)
    ap.add_argument("--no-text-repair", action="store_true", help="switch off sheet.Sheet.TEXT_REPAIR (ablation)")
    ap.add_argument("--loop-register", action="store_true",
                    help="describe every sheet first, then fill missing location pieces from the same loop on other sheets")
    ap.add_argument("--register", action="store_true",
                    help="first read equipment names from every drawing of the same client batch (cross-sheet register)")
    a = ap.parse_args()
    names = a.sheets or SETS[a.set]
    if a.no_text_repair:
        importlib.import_module(a.pkg + ".sheet").Sheet.TEXT_REPAIR = False
    if a.register:
        build_register(a.pkg, {SHEETS[n][2] for n in names})
    out = Path(a.out or ROOT / "descgen_lab" / "results" / f"{a.pkg.replace('.', '_')}_{a.set}")
    refs = _refs(); allrows = []; summ = []
    if a.loop_register:
        D = importlib.import_module(a.pkg + ".describe")
        done = {}
        for n in names:
            done[n] = describe_one(n, a.pkg)
            print(f"described {n}", flush=True)
        reg = D.build_loop_register([(n, v[0]) for n, v in done.items()])
        print(f"loop register: {len(reg)} loops", flush=True)
    for n in names:
        if a.loop_register:
            res, bubbles, refset, sheet = done[n]
            k = D.apply_loop_register(n, res, reg, sheet.profile)
            rows = score_sheet(n, res, bubbles, refset, refs, out, a.pkg)
        else:
            rows, _ = run_sheet(n, a.pkg, refs, out)
        allrows += rows
        s = summarize(rows) | {"sheet": n}; summ.append(s)
        print(f"{n[:58]:58s} tags {s['tags']:3d} scored {s['scored']:3d} exact {s['exact']:3d} F1 {s['f1']:.3f}", flush=True)
    t = summarize(allrows)
    print(f"{'TOTAL':58s} tags {t['tags']:3d} scored {t['scored']:3d} exact {t['exact']:3d} F1 {t['f1']:.3f}")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "tags", "scored", "exact", "f1"]); w.writeheader(); w.writerows(summ + [t | {"sheet": "TOTAL"}])


if __name__ == "__main__":
    main()
