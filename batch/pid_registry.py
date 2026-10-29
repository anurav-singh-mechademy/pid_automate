"""Registry of every P&ID sheet in the project: where it is, whose it is, what ground truth exists for it, and which
evaluation split uses it. Consult pid_registry.csv before any new experiment so dev / test data stay consistent.
    .venv/bin/python batch/pid_registry.py            -> pid_registry.csv (project root)
One row per unique drawing page (PDF pages de-duplicated by the 72 dpi greyscale fingerprint batch_tags.py uses),
plus the PID2Graph OPEN100 sheets. Nothing here is edited by hand: splits come from the code that defines them
(descgen/bench.SETS, batch/bench_tags.py sets and bench_tags/heldout.json); ground truth files are only read.

Columns
  pid_id            sheet name the pipelines use (ledger out_name '<file stem>_p<page>'); OPEN100: 'OPEN100/<n>'
  file, page        first location found; other_locations: every other copy of the same page
  source_folder     top folder the sheet came from
  client, client_basis   inferred owner and why (drawing-number family / folder / reference list)
  page_kind         from survey/pdf_pages.csv (vector+text, vector text-as-strokes, raster+text, raster)
  exact_text        1 if the page has a usable text layer (>= 100 words) or AutoCAD SHX text comments
  processed_in      phase-1 output folders holding this sheet's tags
  tag_gt            1 if a tag-extraction truth exists that bench_tags.py scores; tag_gt_set / tag_gt_type say which
  tag_split         dev = sheet used to build / tune tag extraction and its precision filter; test = held-out check
  desc_gt           1 if >= DESC_MIN drawn bubble tags match a named entry in the client's reference list
  desc_gt_source, desc_ref_tags   which list and how many drawn tags it names
  desc_split        dev / test (descgen/bench.SETS), other (scored but in neither set), '' = unused
  notes             anything a row's flags do not say"""
import csv, hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "batch"))
OUT = ROOT / "pid_registry.csv"
PDF_ROOTS = ["lngc", "lngc_eq16", "lngc_eq16_output/_sources", "lngc_bog_a_src", "pids_bhcp", "pids_bhcp_output/_sources", "bhcp"]
RUNS = ["lngc_eq16_output", "pids_bhcp_output", "lngc_bog_a_output"]       # batch_tags.py output folders
OPEN100 = ROOT / "external/pid2graph/data/PID2Graph OPEN100"
DESC_MIN = 1          # desc_ref_tags gives the strength; 1-2 matches on an unused sheet can be loop-number chance


def fingerprint(pg):
    pix = pg.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)
    return hashlib.md5(pix.samples).hexdigest()


def client_of(name, folder):
    if re.match(r"L\d{3}-\d{5}-", name):
        return "LNG Canada", "L001-... drawing number (LNG Canada numbering; lngc folder; LNG Canada sensor export)"
    if re.search(r"SC6\dD", name):
        return "Dominion Energy Cove Point (DECP)", "SC6xD drawing number; DECP Master Tag List and Cove Point sensor export match these sheets"
    if folder in ("pids_bhcp", "pids_bhcp_output", "bhcp"):
        return "Dominion Energy Cove Point (DECP) - probable", "vendor / package drawing delivered in the same BHCP batch as the SC6xD sheets; no client number on file name"
    return "", ""


def tag_truth():
    """pid_id -> (set, type) for every sheet bench_tags.py scores."""
    T = {}
    def add(sheets, s, t):
        for x in sheets:
            T.setdefault(x, []).append((s, t))
    E = ROOT / "lngc_eq16_output"
    add([r["out_name"] for r in csv.DictReader(open(E / "ledger.csv")) if r["status"] == "ok"], "eq16",
        "full: LNGC PI_TAGS + Flash GC list, by-eye verdicts on the rest")
    B = ROOT / "pids_bhcp_output/eval"
    add({r["sheet"] for r in csv.DictReader(open(B / "decp_visual/precision_sample.csv"))}, "decp",
        "partial: by-eye precision sample")
    rec = set()
    for r in csv.DictReader(open(B / "decp_equipment_scoped_recall.csv")):
        rec |= {s.strip() for s in r["sheets"].split("|") if s.strip()}
    # the list names sheets by drawing number ('SC62D1531'); pipeline names may carry more ('SC62D1531 rev 5 Master_p0')
    names = [r["out_name"] for r in csv.DictReader(open(ROOT / "pids_bhcp_output/ledger.csv")) if r["status"] == "ok"]
    rec = {n for d in rec for n in names if n == d + "_p0" or n.startswith(d + " ")}
    add(rec, "decp", "partial: recall vs DECP equipment list (scoped)")
    add({r["sheet"] for r in csv.DictReader(open(B / "fresh5_precision_sample.csv"))} | set(json.load(open(B / "fresh5_ocr.json"))),
        "fresh5", "partial: by-eye precision sample + recall vs Cove Point export tags printed on sheet")
    led = [r["out_name"] for r in csv.DictReader(open(ROOT / "lngc_bog_a_output/ledger.csv")) if r["status"] == "ok"]
    add([s for s in led if "100061" not in s], "bog", "full: PDF text-layer bubbles (exact text)")
    add({r["sheet"] for r in json.load(open(ROOT / "lngc_bog_a_output/eval/vendor_sample.json"))}, "bogv",
        "partial: by-eye precision sample")
    add({f"OPEN100/{r['sheet']}" for r in json.load(open(ROOT / "pidgraph_output/p2g_tags/gt_bubble_reads.json"))}, "open100",
        "full: PID2Graph instrumentation boxes, every bubble read by eye")
    held = set(json.load(open(ROOT / "bench_tags/heldout.json"))["pick"])
    return T, held


def phase1_tags(pid, runs_of):
    """Bubble / stack tags phase 1 read on the sheet (what descgen describes)."""
    files = [ROOT / r / "sheets" / f"tags_{pid}.csv" for r in runs_of] + [ROOT / "outputs" / f"tags_{pid}.csv"]
    for f in files:
        if f.exists():
            return {r["tag"] for r in csv.DictReader(open(f)) if r.get("source") in ("bubble", "stack")}
    return set()


SPLIT_V2 = ROOT / "descgen" / "desc_split_v2.json"
TEST_SHARE = 0.4


def split_v2(rows):
    """Second dev / test split (2026-10-28) over the sheets with client references that no split used. Whole delivery
    folders go to one side (sheets delivered together share equipment and loop numbers), folders taken in md5 order
    of their name into test until test holds TEST_SHARE of the reference tags. Frozen in SPLIT_V2 on first run and
    only read afterwards, so the split never reshuffles; it applies to desc_split, and to tag_split where the sheet
    was never used for tag extraction work."""
    def group(r):
        f = r["file"].split("/")
        return f[2] if f[0] == "pids_bhcp_output" else f[0]
    if SPLIT_V2.exists():
        S = json.load(open(SPLIT_V2))["sheets"]
    else:
        cand = [r for r in rows if r["desc_gt"] == 1 and not r["desc_split"]]
        # another revision of a drawing already in a split goes to that drawing's side (same content)
        drw = lambda r: (re.search(r"SC6\dD\w{4}|L\d{3}-\d{5}-PX-\d{4}-\d+", r["pid_id"]) or [None])[0]
        side_of = {drw(r): r["desc_split"] for r in rows if r["desc_split"] in ("dev", "test") and drw(r)}
        S = {r["pid_id"]: side_of[drw(r)] for r in cand if drw(r) in side_of}
        cand = [r for r in cand if r["pid_id"] not in S]
        groups = defaultdict(list)
        for r in cand:
            groups[group(r)].append(r)
        total = sum(int(r["desc_ref_tags"]) for r in cand)
        got = 0
        for g in sorted(groups, key=lambda g: hashlib.md5(g.encode()).hexdigest()):
            side = "test" if got < TEST_SHARE * total else "dev"
            for r in groups[g]:
                S[r["pid_id"]] = side
            if side == "test":
                got += sum(int(r["desc_ref_tags"]) for r in groups[g])
        json.dump(dict(rule=split_v2.__doc__, created="2026-10-28", sheets=S), open(SPLIT_V2, "w"), indent=1)
    for r in rows:
        side = S.get(r["pid_id"])
        if not side:
            continue
        r["desc_split"] = side
        note = f"split v2 ({group(r)}): client references matched by loop number, not checked by eye"
        if not r["tag_split"]:
            r["tag_split"] = side
        elif r["tag_split"] != side:
            note += f"; already tag {r['tag_split']} ({r['tag_gt_set'] or 'held-out'})"
        r["notes"] = "; ".join(x for x in (r["notes"], note) if x)


def main():
    import descgen.bench as DB
    # ---- every unique PDF page
    pages, order = {}, []
    survey = {(r["file"], r["page"]): r for r in csv.DictReader(open(ROOT / "survey/pdf_pages.csv"))}
    for root in PDF_ROOTS:
        for p in sorted((ROOT / root).rglob("*")):
            if not (p.is_file() and p.suffix.lower() == ".pdf") or "__MACOSX" in str(p):
                continue
            try:
                doc = pymupdf.open(p)
            except Exception:
                continue
            rel = str(p.relative_to(ROOT))
            for i, pg in enumerate(doc):
                fp = fingerprint(pg)
                if fp not in pages:
                    pages[fp] = dict(locs=[], fp=fp); order.append(fp)
                pages[fp]["locs"].append((rel, i))
                if "words" not in pages[fp]:
                    words = len(pg.get_text("words"))
                    shx = sum(1 for a in (pg.annots() or []) if a.info.get("title") == "AutoCAD SHX Text")
                    pages[fp].update(words=words, shx=shx)
    # ---- pipeline names (fingerprint -> out_name) and which runs hold the sheet
    names, runs = {}, defaultdict(list)
    for r in RUNS:
        inv = ROOT / r / "inventory.csv"
        ok = {x["fingerprint"] for x in csv.DictReader(open(ROOT / r / "ledger.csv")) if x["status"] == "ok"}
        for x in csv.DictReader(open(inv)):
            if x["status"] == "unique":
                names.setdefault(x["fingerprint"], x["out_name"])
                if x["fingerprint"] in ok:
                    runs[x["fingerprint"]].append(r)
    T, held = tag_truth()
    refs = DB._refs()
    desc_sets = {n: k for k, ns in DB.SETS.items() for n in ns}
    for n in DB.SHEETS:
        desc_sets.setdefault(n, "other")
    rows = []
    for fp in order:
        P = pages[fp]
        f, pg = P["locs"][0]
        stem = Path(f).stem
        pid = names.get(fp) or re.sub(r"[^\w\-. &]+", "_", stem).strip() + f"_p{pg}"
        folder = f.split("/")[0]
        client, basis = client_of(Path(f).name, folder)
        sv = next((survey[(l, str(i))] for l, i in P["locs"] if (l, str(i)) in survey), {})
        tg = T.get(pid, [])
        refset = ["flashgc", "lngc_sensors"] if client == "LNG Canada" else ["decp"] if client.startswith("Dominion") else []
        drawn = phase1_tags(pid, runs[fp])
        named, src = 0, ""
        for rs in refset:
            n = 0
            for t in drawn:
                p = re.match(r"^(?:\d{1,3}-)*([A-Z]{1,6})-(\d{3,5})([A-Z]?)(?:-\d+)?$", t)
                if p and any(DB.same_instrument(p.group(1), L) for L, _ in refs[rs].get((p.group(1)[0], p.group(2), p.group(3)), [])):
                    n += 1
            if n > named:
                named, src = n, rs
        src = {"flashgc": "Flash GC sensors for Anurav.xlsx", "lngc_sensors": "LNG_Canada_Sensors_export (lngc/)",
               "decp": "DECP Master Tag List.xlsx (PI_tags_Master)"}.get(src, "")
        rows.append(dict(
            pid_id=pid, file=f, page=pg, other_locations=" | ".join(f"{l} p{i}" for l, i in P["locs"][1:]),
            source_folder=folder, client=client, client_basis=basis, page_kind=sv.get("kind", ""),
            exact_text=int(P["words"] >= 100 or P["shx"] > 0), processed_in=" | ".join(runs[fp]) or ("outputs (legacy)" if drawn else ""),
            tag_gt=int(bool(tg)), tag_gt_set=" | ".join(sorted({s for s, _ in tg})), tag_gt_type=" ; ".join(sorted({t for _, t in tg})),
            tag_split="test" if pid in held else "dev" if tg else "",
            desc_gt=int(named >= DESC_MIN), desc_gt_source=src if named else "", desc_ref_tags=named if drawn else "",
            desc_split=desc_sets.get(pid, ""), fingerprint=fp,
            notes="held-out tag check: rows sampled in bench_tags/heldout.json, verdicts not stored" if pid in held else ""))
    # ---- PID2Graph OPEN100 (public; PNG + graph GT)
    for png in sorted(OPEN100.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        pid = f"OPEN100/{png.stem}"
        tg = T.get(pid, [])
        rows.append(dict(pid_id=pid, file=str(png.relative_to(ROOT)), page=0, source_folder="external/pid2graph",
                         client="PID2Graph OPEN100 (public dataset)", client_basis="Zenodo 14803338", page_kind="raster image",
                         exact_text=0, processed_in="pidgraph_output/p2g_tags" if tg else "",
                         tag_gt=int(bool(tg)), tag_gt_set=" | ".join(sorted({s for s, _ in tg})),
                         tag_gt_type=" ; ".join(sorted({t for _, t in tg})) or "graph GT only (instrumentation boxes unread)",
                         tag_split="dev" if tg else "", desc_gt=0, desc_split="", fingerprint=""))
    split_v2(rows)
    cols = list(rows[0])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    n = len(rows)
    print(f"{n} sheets -> {OUT}")
    for c in ("client", "tag_split", "desc_split"):
        cnt = defaultdict(int)
        for r in rows:
            cnt[r[c] or "-"] += 1
        print(c, dict(cnt))
    print("tag_gt", sum(r["tag_gt"] for r in rows), "desc_gt", sum(r["desc_gt"] for r in rows))
    missing = [k for k in T if k not in {r["pid_id"] for r in rows}]
    missing_d = [k for k in desc_sets if k not in {r["pid_id"] for r in rows}]
    print("tag-truth sheets not matched to a row:", missing)
    print("desc sheets not matched to a row:", missing_d)


if __name__ == "__main__":
    main()
