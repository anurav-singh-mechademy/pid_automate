"""Survey every P&ID PDF in the project: what each page is made of, read from the file itself (no inference).
    .venv/bin/python batch/survey_pdfs.py            -> survey/pdf_pages.csv, survey/pdf_files.csv, printed summary
Per page: real text (characters, words, tag-shaped words, fonts), embedded raster images and how much of the page they
cover, vector paths (count, dashed paths, distinct stroke widths, colours), CAD layers (PDF optional content groups),
annotations / links, attached files, producer / creator. Files are de-duplicated by content (MD5 of the bytes).
Page kinds:
  vector+text     line work as vector paths and text as real characters (CAD export with fonts)
  vector, text as strokes   vector line work, but the lettering is drawn as outlines (e.g. AutoCAD SHX fonts)
  raster+text     the drawing is an embedded image with a real text layer on top
  raster          an embedded image only (a scan)"""
import csv, hashlib, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["pids_bhcp", "pids_bhcp_output/_sources", "lngc", "lngc_eq16_output/_sources", "bhcp"]
TAGLIKE = re.compile(r"^\d{0,3}-?[A-Z]{1,5}-?\d{3,5}[A-Z]?$")
OUT = ROOT / "survey"; OUT.mkdir(exist_ok=True)


def page_facts(pg):
    area = pg.rect.width * pg.rect.height
    words = pg.get_text("words")
    fonts = sorted({f[3] for f in pg.get_fonts()})
    imgs = pg.get_image_info()
    cover = sum(max(0, (min(i["bbox"][2], pg.rect.x1) - max(i["bbox"][0], 0)) * (min(i["bbox"][3], pg.rect.y1) - max(i["bbox"][1], 0)))
                for i in imgs) / area if area else 0
    big = max(((i.get("width", 0), i.get("height", 0)) for i in imgs), default=(0, 0), key=lambda x: x[0] * x[1])
    try:
        dr = pg.get_drawings()
    except Exception:
        dr = []
    dashed = sum(1 for d in dr if d.get("dashes") and d["dashes"] not in ("[] 0", "[] 0.0", None))
    widths = Counter(round(d.get("width") or 0, 2) for d in dr if d.get("type") in ("s", "fs"))
    colours = Counter(tuple(round(c, 2) for c in (d.get("color") or ())) for d in dr if d.get("color"))
    n_items = sum(len(d.get("items", ())) for d in dr)
    tiny = sum(1 for d in dr if (d["rect"].width < 4 and d["rect"].height < 6))      # stroke-text glyph pieces
    return dict(text_chars=len(pg.get_text()), words=len(words), tag_words=sum(bool(TAGLIKE.match(w[4])) for w in words),
                fonts=" | ".join(fonts)[:200], n_images=len(imgs), image_cover=round(cover, 2),
                biggest_image_px=f"{big[0]}x{big[1]}" if imgs else "", vector_paths=len(dr), path_segments=n_items,
                dashed_paths=dashed, stroke_widths=len(widths), top_widths=" ".join(str(w) for w, _ in widths.most_common(4)),
                colours=len(colours), small_paths=tiny, annots=len(list(pg.annots() or [])), links=len(pg.get_links()),
                width_mm=round(pg.rect.width / 72 * 25.4), height_mm=round(pg.rect.height / 72 * 25.4), rotation=pg.rotation)


def kind(f):
    vec = f["vector_paths"] >= 500
    txt = f["text_chars"] >= 500 and f["words"] >= 50
    ras = f["image_cover"] >= 0.5
    if ras and not vec:
        return "raster+text" if txt else "raster"
    if vec:
        return "vector+text" if txt else "vector, text as strokes"
    return "raster+text" if txt else "other / nearly empty"


def main():
    files = []
    for s in SOURCES:
        files += [p for p in (ROOT / s).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf" and "__MACOSX" not in str(p)]
    seen, pages, frows = {}, [], []
    for p in sorted(files):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        rel = str(p.relative_to(ROOT))
        if h in seen:
            frows.append(dict(file=rel, duplicate_of=seen[h])); continue
        seen[h] = rel
        try:
            d = pymupdf.open(p)
        except Exception as e:
            frows.append(dict(file=rel, error=str(e))); continue
        meta = d.metadata or {}
        ocgs = d.get_ocgs() or {}
        layer_names = sorted({v.get("name", "") for v in ocgs.values()})
        fr = dict(file=rel, pages=len(d), producer=meta.get("producer", ""), creator=meta.get("creator", ""),
                  layers=len(layer_names), layer_names=" | ".join(layer_names)[:300], attachments=d.embfile_count(),
                  bookmarks=len(d.get_toc()), xmp=bool(d.get_xml_metadata()), duplicate_of="")
        kinds = Counter()
        for i, pg in enumerate(d):
            f = page_facts(pg)
            f.update(file=rel, page=i, kind=kind(f), producer=fr["producer"], creator=fr["creator"], layers=fr["layers"])
            kinds[f["kind"]] += 1
            pages.append(f)
        fr["page_kinds"] = "; ".join(f"{k} x{n}" for k, n in kinds.items())
        frows.append(fr)
        print(f"{len(frows):4d} {rel[:70]:70s} {fr['page_kinds']}", flush=True)
    for name, rows in (("pdf_pages.csv", pages), ("pdf_files.csv", frows)):
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    print(f"\n{len(files)} PDF files, {len(seen)} unique, {len(pages)} unique-file pages")


if __name__ == "__main__":
    main()
