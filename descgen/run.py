"""Describe every instrument tag of a tag-extraction run (batch/batch_tags.py output folder). No client lists needed.
    .venv/bin/python -m descgen.run <batch output folder> [--no-register] [--loop-register]
Per sheet: the page as phase 1 rendered and rotated it, full-page OCR (cached in <out>/ocr/), merged with the PDF's
exact text where the page has it (batch/pdftext.py), and the instrument tags phase 1 read from bubbles / stacked labels
that the precision filter kept. Cross-sheet steps (both use only the batch's own drawings):
  equipment register  names an equipment printed on a sheet but named only on another sheet (on by default)
  loop register       a tag described with no location takes it from the same loop on another sheet (--loop-register)
Writes <out>/descriptions.csv: sheet, tag, description, confidence, review, pieces (each piece with its source)."""
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "batch")); sys.path.insert(0, str(ROOT))


def sheets_of(out):
    led = [r for r in csv.DictReader(open(out / "ledger.csv")) if r["status"] == "ok"]
    ms = out / "mastersheet.csv"
    keep = None
    if ms.exists():
        keep = {}
        for r in csv.DictReader(open(ms)):
            if r["source"] in ("bubble", "stack") and r.get("keep", "True") == "True":
                keep.setdefault(r["sheet"], {}).setdefault(r["tag"], [int(float(v)) for v in r["bbox"].strip("[]").split(",")])
    return led, keep


def load(out, L, keep):
    """(page image, OCR + exact text fragments, {tag: bbox})."""
    from tag_pipeline import load_page, ocr_page, RapidOCR
    from pdftext import exact_frags, merge
    name, rot = L["out_name"], int(L["rotation"] or 0)
    page = load_page(Path(L["file"]), int(L["page"]))
    if rot:
        page = page.rotate(rot, expand=True)
    cache = out / "ocr" / f"{name}.json"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        json.dump([dict(text=f["text"], bbox=[round(v) for v in f["bbox"]]) for f in ocr_page(page, RapidOCR())], open(cache, "w"))
    frags = json.load(open(cache))
    if Path(L["file"]).suffix.lower() == ".pdf":
        ex, _ = exact_frags(L["file"], int(L["page"]), rot)
        if ex:
            frags = merge(ex, frags)
    if keep is not None:
        bubbles = keep.get(name, {})
    else:
        bubbles = {}
        for r in csv.DictReader(open(out / "sheets" / f"tags_{name}.csv")):
            if r.get("source") in ("bubble", "stack"):
                bubbles.setdefault(r["tag"], [int(float(v)) for v in r["bbox"].strip("[]").split(",")])
    return page, frags, bubbles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="batch/batch_tags.py output folder")
    ap.add_argument("--no-register", action="store_true", help="skip the cross-sheet equipment-name register")
    ap.add_argument("--loop-register", action="store_true", help="fill bare tags from the same loop on other sheets")
    a = ap.parse_args()
    out = Path(a.out)
    from descgen import sheet as S, describe as D, plant as P
    led, keep = sheets_of(out)
    if not a.no_register:
        def gen():
            for L in led:
                page, frags, _ = load(out, L, keep)
                yield L["out_name"], S.Sheet(page, frags)
        reg = P.build_register(gen())
        print(f"equipment register: {len(reg)} names from {len(led)} sheets", flush=True)
    results = {}
    for L in led:
        page, frags, bubbles = load(out, L, keep)
        sh = S.Sheet(page, frags)
        sib = Counter((p["letters"], p["number"]) for p in map(D.parse_tag, bubbles) if p)
        results[L["out_name"]] = (D.describe_sheet(sh, bubbles, sib), sh.profile)
        print(f"{L['out_name']}: {len(bubbles)} tags described", flush=True)
    if a.loop_register:
        lreg = D.build_loop_register([(n, r) for n, (r, _) in results.items()])
        n = sum(D.apply_loop_register(name, r, lreg, prof) for name, (r, prof) in results.items())
        print(f"loop register: {len(lreg)} loops, {n} tags filled", flush=True)
    rows = [dict(sheet=name, tag=t, description=r["description"], confidence=r.get("confidence", ""),
                 review=r.get("review", ""),
                 pieces=json.dumps({k: [v["value"], v["source"], v["conf"]] for k, v in r["slots"].items()}))
            for name, (res, _) in results.items() for t, r in sorted(res.items())]
    with open(out / "descriptions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sheet", "tag", "description", "confidence", "review", "pieces"])
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} descriptions -> {out / 'descriptions.csv'}")


if __name__ == "__main__":
    main()
