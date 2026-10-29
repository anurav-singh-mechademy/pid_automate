# Interactive tag viewer: PDF(s) in -> one self-contained HTML file out.
#   .venv/bin/python batch/tag_viewer.py a.pdf b.pdf some_folder/ --out viewer_output
#   .venv/bin/python batch/tag_viewer.py --from lngc_bog_a_output          (no extraction: view an existing batch output)
# Left: the tag extraction overlay of the selected sheet (zoom with the wheel, drag to pan). Right: the extracted tags.
# Clicking a tag zooms the overlay to it and highlights its box; clicking a box on the overlay selects the tag in the list.
# Extraction is batch/batch_tags.py (resumable: pages already in <out>/ledger.csv are not re-run) followed by the
# precision filter (tag_filter.filter_mastersheet); dropped reads can be shown with their drop reason.
import argparse, base64, csv, io, json, subprocess, sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = None
EMBED_LONG_SIDE = 5000      # overlay is downscaled to this for embedding; boxes stay in full-resolution pixel units
JPEG_QUALITY = 82


def extract(inputs, out):
    """Runs the batch pipeline on the given PDFs / folders (linked into <out>/_input) and the precision filter."""
    src = out / "_input"
    src.mkdir(parents=True, exist_ok=True)
    pdfs = []
    for p in map(Path, inputs):
        pdfs += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".pdf") if p.is_dir() else [p]
    for p in pdfs:
        link = src / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    subprocess.run([sys.executable, str(HERE / "batch_tags.py"), str(src), str(out)], check=True)
    sys.path.insert(0, str(HERE))
    import tag_filter
    tag_filter.filter_mastersheet(out)


def embed(png):
    im = Image.open(png).convert("RGB")
    W, H = im.size
    k = min(1.0, EMBED_LONG_SIDE / max(W, H))
    if k < 1:
        im = im.resize((round(W * k), round(H * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return W, H, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def collect(out):
    rows = list(csv.DictReader(open(out / "mastersheet.csv")))
    by_sheet = {}
    for r in rows:
        by_sheet.setdefault(r["sheet"], []).append(r)
    sheets = []
    for name, rs in by_sheet.items():
        png = out / "sheets" / f"tags_{name}_overlay.png"
        if not png.exists():
            continue
        W, H, uri = embed(png)
        tags = []
        for i, r in enumerate(rs):
            try:
                b = json.loads(r["bbox"])
            except (ValueError, KeyError):
                continue
            keep = str(r.get("keep", "True")) == "True"
            tags.append(dict(i=i, tag=r["tag"], bbox=b, keep=keep, drop=r.get("drop_reason", ""),
                             src=r.get("source", ""), conf=r.get("confidence", ""), lines=r.get("lines", ""),
                             label=r.get("bubble_label", ""), fix=r.get("fix", ""), read=r.get("tag_read", "")))
        sheets.append(dict(name=name, file=rs[0].get("source_file", ""), page=rs[0].get("page", ""),
                           W=W, H=H, img=uri, tags=tags))
        print(f"  {name}: {sum(t['keep'] for t in tags)} kept / {len(tags)} read")
    return sheets


def write_html(sheets, path):
    data = json.dumps(sheets).replace("</", "<\\/")
    path.write_text(HTML.replace("__DATA__", data))
    print(f"-> {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(sheets)} sheets)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P&amp;ID Tag Viewer</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#6b7385;--line:#e1e4ea;--acc:#e8590c;--keep:#1c7ed6;--drop:#adb5bd}
*{box-sizing:border-box}html,body{margin:0;height:100%;font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
#app{display:grid;grid-template-columns:1fr 360px;height:100vh}
#view{position:relative;overflow:hidden;background:#dfe2e7;cursor:grab}#view.drag{cursor:grabbing}
#view svg{width:100%;height:100%;display:block}
.box{fill:transparent;stroke:var(--keep);stroke-width:4;vector-effect:non-scaling-stroke;cursor:pointer}
.box.dropped{stroke:var(--drop);stroke-dasharray:4 3}
.box:hover{stroke:var(--acc)}
.box.sel{stroke:var(--acc);stroke-width:4;fill:rgba(232,89,12,.18)}
.halo{fill:none;stroke:var(--acc);stroke-width:3;vector-effect:non-scaling-stroke;animation:pulse 1.2s ease-out 3}
@keyframes pulse{0%{opacity:1}100%{opacity:0}}
#side{display:flex;flex-direction:column;border-left:1px solid var(--line);background:var(--panel);min-height:0}
header{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
header h1{font-size:14px;margin:0}
select,input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff}
.row{display:flex;gap:8px;align-items:center;color:var(--muted)}.row label{display:flex;gap:4px;align-items:center;white-space:nowrap}
.row input[type=checkbox]{width:auto}
#list{overflow:auto;flex:1;margin:0;padding:0;list-style:none}
#list li{padding:7px 14px;border-bottom:1px solid var(--line);cursor:pointer;display:grid;grid-template-columns:1fr auto;gap:2px 8px}
#list li:hover{background:#f1f3f5}#list li.sel{background:#fff4e6;box-shadow:inset 3px 0 var(--acc)}
#list li.dropped .t{color:var(--muted);text-decoration:line-through}
.t{font-weight:600;font-family:ui-monospace,Menlo,monospace}.m{color:var(--muted);font-size:12px;grid-column:1/-1}
.c{color:var(--muted);font-size:12px}
#hint{position:absolute;left:10px;bottom:10px;background:rgba(255,255,255,.9);padding:4px 8px;border-radius:6px;color:var(--muted);font-size:12px}
#reset{position:absolute;right:10px;top:10px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
@media (max-width:800px){#app{grid-template-columns:1fr;grid-template-rows:60vh 40vh}#side{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<div id="app">
  <div id="view"><svg id="svg"></svg><button id="reset">Fit sheet</button><div id="hint">wheel: zoom · drag: pan · click a box or a tag</div></div>
  <aside id="side">
    <header>
      <h1 id="title">Tags</h1>
      <select id="sheet"></select>
      <input id="q" placeholder="Filter tags (e.g. PT, 3197)">
      <div class="row"><label><input type="checkbox" id="showDrop"> show dropped reads</label><span id="count" style="margin-left:auto"></span></div>
    </header>
    <ul id="list"></ul>
  </aside>
</div>
<script>
const SHEETS = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("svg"), list = document.getElementById("list"), view = document.getElementById("view");
const sel = document.getElementById("sheet"), q = document.getElementById("q"), showDrop = document.getElementById("showDrop");
let S = null, vb = null, cur = null, boxes = {};

SHEETS.forEach((s, i) => { const o = document.createElement("option"); o.value = i;
  o.textContent = `${s.name} (${s.tags.filter(t => t.keep).length})`; sel.appendChild(o); });

function setVB(v) { vb = v; svg.setAttribute("viewBox", v.join(" ")); }
function fit() { setVB([0, 0, S.W, S.H]); }

function load(i) {
  S = SHEETS[i]; cur = null; boxes = {};
  document.getElementById("title").textContent = S.file + (S.page !== "" ? ` · page ${+S.page + 1}` : "");
  svg.innerHTML = "";
  const im = document.createElementNS(NS, "image");
  im.setAttribute("href", S.img); im.setAttribute("width", S.W); im.setAttribute("height", S.H);
  svg.appendChild(im);
  for (const t of S.tags) {
    const [x0, y0, x1, y1] = t.bbox, r = document.createElementNS(NS, "rect");
    r.setAttribute("x", x0); r.setAttribute("y", y0); r.setAttribute("width", x1 - x0); r.setAttribute("height", y1 - y0);
    r.setAttribute("class", "box" + (t.keep ? "" : " dropped"));
    const ti = document.createElementNS(NS, "title"); ti.textContent = t.tag + (t.keep ? "" : ` (dropped: ${t.drop})`);
    r.appendChild(ti);
    r.addEventListener("click", e => { e.stopPropagation(); pick(t, false); });
    svg.appendChild(r); boxes[t.i] = r;
  }
  fit(); render();
}

function render() {
  const f = q.value.trim().toUpperCase(), dropOn = showDrop.checked;
  const ts = S.tags.filter(t => (t.keep || dropOn) && (!f || t.tag.toUpperCase().includes(f)))
                   .sort((a, b) => (b.keep - a.keep) || a.tag.localeCompare(b.tag, undefined, {numeric: true}));
  for (const t of S.tags) boxes[t.i].style.display = (t.keep || dropOn) ? "" : "none";
  list.innerHTML = "";
  for (const t of ts) {
    const li = document.createElement("li"); li.dataset.i = t.i;
    li.className = (t.keep ? "" : "dropped") + (cur && cur.i === t.i ? " sel" : "");
    const meta = [t.src, t.lines && `read: ${t.lines}`, t.fix && `fix: ${t.fix}`, !t.keep && `dropped: ${t.drop}`].filter(Boolean).join(" · ");
    li.innerHTML = `<span class="t"></span><span class="c">${t.conf ? (+t.conf).toFixed(2) : ""}</span><span class="m"></span>`;
    li.querySelector(".t").textContent = t.tag; li.querySelector(".m").textContent = meta;
    li.addEventListener("click", () => pick(t, true));
    list.appendChild(li);
  }
  const k = S.tags.filter(t => t.keep).length;
  document.getElementById("count").textContent = `${ts.length} shown · ${k} kept / ${S.tags.length} read`;
}

function pick(t, zoom) {
  if (cur) boxes[cur.i].classList.remove("sel");
  cur = t; const r = boxes[t.i]; r.classList.add("sel"); svg.appendChild(r);   // bring to front
  const [x0, y0, x1, y1] = t.bbox, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  if (zoom) {
    const w = Math.max(x1 - x0, y1 - y0) * 12, a = view.clientWidth / view.clientHeight;
    setVB([cx - w * a / 2, cy - w / 2, w * a, w]);
  }
  const h = document.createElementNS(NS, "circle");
  h.setAttribute("cx", cx); h.setAttribute("cy", cy); h.setAttribute("r", Math.max(x1 - x0, y1 - y0) * 1.2);
  h.setAttribute("class", "halo"); svg.appendChild(h); setTimeout(() => h.remove(), 3700);
  list.querySelectorAll("li.sel").forEach(li => li.classList.remove("sel"));
  const li = list.querySelector(`li[data-i="${t.i}"]`);
  if (li) { li.classList.add("sel"); if (!zoom) li.scrollIntoView({block: "center", behavior: "smooth"}); }
}

// zoom / pan in viewBox units
function toSvg(e) { const b = svg.getBoundingClientRect(), s = Math.max(vb[2] / b.width, vb[3] / b.height);
  const ox = (b.width * s - vb[2]) / 2, oy = (b.height * s - vb[3]) / 2;
  return [vb[0] - ox + (e.clientX - b.left) * s, vb[1] - oy + (e.clientY - b.top) * s, s]; }
view.addEventListener("wheel", e => { e.preventDefault();
  const [x, y] = toSvg(e), k = Math.exp(e.deltaY * 0.0015);
  setVB([x - (x - vb[0]) * k, y - (y - vb[1]) * k, vb[2] * k, vb[3] * k]); }, {passive: false});
let drag = null;
view.addEventListener("mousedown", e => { if (e.target.closest("button")) return; drag = [e.clientX, e.clientY, vb.slice()]; view.classList.add("drag"); });
window.addEventListener("mousemove", e => { if (!drag) return; const s = toSvg(e)[2];
  setVB([drag[2][0] - (e.clientX - drag[0]) * s, drag[2][1] - (e.clientY - drag[1]) * s, vb[2], vb[3]]); });
window.addEventListener("mouseup", () => { drag = null; view.classList.remove("drag"); });
document.getElementById("reset").addEventListener("click", fit);
sel.addEventListener("change", () => load(+sel.value));
q.addEventListener("input", render); showDrop.addEventListener("change", render);
document.addEventListener("keydown", e => {            // up/down arrows step through the list
  if (e.target === q || !["ArrowDown", "ArrowUp"].includes(e.key)) return; e.preventDefault();
  const lis = [...list.children], at = lis.findIndex(li => li.classList.contains("sel"));
  const nx = lis[Math.min(lis.length - 1, Math.max(0, at + (e.key === "ArrowDown" ? 1 : -1)))];
  if (nx) { pick(S.tags.find(t => t.i === +nx.dataset.i), true); nx.scrollIntoView({block: "nearest"}); } });
if (SHEETS.length) load(0); else document.getElementById("title").textContent = "No processed sheets";
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="PDF files and/or folders of PDFs")
    ap.add_argument("--out", default="tag_viewer_output", help="pipeline output folder (resumable)")
    ap.add_argument("--from", dest="existing", help="view an existing batch output folder; no extraction")
    ap.add_argument("--html", help="output HTML path (default <out>/tag_viewer.html)")
    a = ap.parse_args()
    out = Path(a.existing or a.out)
    if not a.existing:
        if not a.inputs:
            ap.error("give PDF files/folders, or --from <existing output folder>")
        extract(a.inputs, out)
    write_html(collect(out), Path(a.html) if a.html else out / "tag_viewer.html")
