# Step-by-step pipeline viewer: one self-contained HTML per source PDF that walks through every stage of tag
# extraction and description generation on each sheet, with back / next arrows (or the left / right keys).
#   .venv/bin/python batch/pipeline_viewer.py a.pdf b.pdf --out run_output     (extract + describe + view)
#   .venv/bin/python batch/pipeline_viewer.py --from run_output                (view an existing output folder)
# Stages: 1 render + orientation, 2 OCR, 3 bubble sizing, 4 RF-DETR symbol detection, 5 text-to-bubble attachment,
# 6 tag assembly (incl. the local-device drop), 7 precision filter, 8 descriptions (descgen).
# Stages 1-6 come from <out>/steps/<sheet>.json, written by batch/batch_tags.py; 7 from mastersheet.csv; 8 from
# descriptions.csv (descgen.run, run here if missing; its OCR cache is seeded from the stage trace, so nothing is
# OCR'd twice).
import argparse, base64, csv, io, json, subprocess, sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
Image.MAX_IMAGE_PIXELS = None
EMBED_LONG_SIDE = 4000
JPEG_QUALITY = 78
MM = 300 / 25.4


def describe(out):
    """Seeds descgen's OCR cache from the stage trace and runs descgen.run (writes <out>/descriptions.csv)."""
    (out / "ocr").mkdir(exist_ok=True)
    for st in (out / "steps").glob("*.json"):
        c = out / "ocr" / st.name
        if not c.exists():
            json.dump([dict(text=f["text"], bbox=f["bbox"]) for f in json.load(open(st))["frags"]], open(c, "w"))
    subprocess.run([sys.executable, "-m", "descgen.run", str(out)], check=True, cwd=ROOT)


def page_image(L):
    from tag_pipeline import load_page
    im = load_page(Path(L["file"]), int(L["page"]))
    if int(L["rotation"] or 0):
        im = im.rotate(int(L["rotation"]), expand=True)
    W, H = im.size
    k = min(1.0, EMBED_LONG_SIDE / max(W, H))
    if k < 1:
        im = im.resize((round(W * k), round(H * k)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return W, H, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def box(v):
    if isinstance(v, str):
        v = json.loads(v)
    return [round(float(x)) for x in v]


def collect(out):
    """{source pdf name: [sheet data]} for every sheet with a stage trace."""
    led = [r for r in csv.DictReader(open(out / "ledger.csv")) if r["status"] == "ok"]
    ms = list(csv.DictReader(open(out / "mastersheet.csv"))) if (out / "mastersheet.csv").exists() else []
    ds = list(csv.DictReader(open(out / "descriptions.csv"))) if (out / "descriptions.csv").exists() else []
    by_pdf = {}
    for L in sorted(led, key=lambda r: (r["file"], int(r["page"]))):
        name, st = L["out_name"], out / "steps" / f"{L['out_name']}.json"
        if not st.exists():
            print(f"  {name}: no stage trace (run before tracing existed), skipped")
            continue
        T = json.load(open(st))
        W, H, img = page_image(L)
        rows = [r for r in ms if r["sheet"] == name]
        kept_box = {}
        for r in rows:
            if r.get("keep") == "True" and r["source"] in ("bubble", "stack"):
                kept_box.setdefault(r["tag"], box(r["bbox"]))
        desc = []
        for r in ds:
            if r["sheet"] == name and r["tag"] in kept_box:
                pieces = json.loads(r["pieces"] or "{}")
                desc.append([kept_box[r["tag"]], r["tag"], r["description"], r["confidence"], r["review"],
                             [[k, v[0], v[1]] for k, v in pieces.items()]])
        by_pdf.setdefault(Path(L["file"]).name, []).append(dict(
            name=name, file=Path(L["file"]).name, page=int(L["page"]), W=W, H=H, img=img,
            rotation=T["rotation"], orient=T["orient_scores"], bubble_px=T["bubble_px"], size_method=T["size_method"],
            size_note=T["size_note"], tile=T["tile"], sizing=T.get("sizing", {}),
            frags=[[*f["bbox"], f["text"], f["score"], f["rot"], f["inside"], f["dist_mm"], f["bubble_id"]] for f in T["frags"]],
            bubbles=[[*b["bbox"], b["conf"], b["label"], b["id"], b["tile"]] for b in T["bubbles"]],
            tags=[[box(t["bbox"]), t["tag"], t["source"], t["confidence"], t["lines"], 0] for t in T["tags"]]
                 + [[box(t["bbox"]), t["tag"], t["source"], t["confidence"], t["lines"], 1] for t in T["local_dropped"]],
            filt=[[box(r["bbox"]), r["tag"], r.get("keep") == "True", r.get("drop_reason", ""),
                   r.get("tag_read", ""), r["source"], r.get("fix", "")] for r in rows],
            desc=desc))
        print(f"  {name}: {len(T['frags'])} OCR reads, {len(T['bubbles'])} bubbles, {len(T['tags'])} tags, "
              f"{sum(r.get('keep') == 'True' for r in rows)} kept, {len(desc)} described")
    return by_pdf


def write_html(sheets, path):
    path.write_text(HTML.replace("__DATA__", json.dumps(dict(sheets=sheets, mm=MM)).replace("</", "<\\/")))
    print(f"-> {path}  ({path.stat().st_size / 1e6:.1f} MB, {len(sheets)} sheets)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P&amp;ID Pipeline Steps</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#6b7385;--line:#e1e4ea;--acc:#e8590c}
*{box-sizing:border-box}html,body{margin:0;height:100%;font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
#app{display:grid;grid-template-columns:1fr 400px;height:100vh}
#view{position:relative;overflow:hidden;background:#dfe2e7;cursor:grab}#view.drag{cursor:grabbing}
#view svg{width:100%;height:100%;display:block}
.b{fill:transparent;stroke-width:3;vector-effect:non-scaling-stroke;cursor:pointer}
.b.dash{stroke-dasharray:5 4}.b:hover{stroke:var(--acc)!important}
.b.sel{stroke:var(--acc)!important;stroke-width:5;fill:rgba(232,89,12,.2)}
.ln{stroke-width:1.5;vector-effect:non-scaling-stroke;opacity:.8}
#dim{fill:#fff;opacity:0;pointer-events:none}
#side{display:flex;flex-direction:column;border-left:1px solid var(--line);background:var(--panel);min-height:0}
header{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
select,input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff}
#nav{display:flex;align-items:center;gap:8px}
#nav button{width:38px;height:34px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:18px;cursor:pointer}
#nav button:disabled{opacity:.35;cursor:default}
#stepname{flex:1;text-align:center}#stepname b{display:block;font-size:14px}#stepname span{color:var(--muted);font-size:12px}
#dots{display:flex;gap:4px;justify-content:center}#dots i{width:22px;height:5px;border-radius:3px;background:var(--line);cursor:pointer}
#dots i.on{background:var(--acc)}
#info{padding:10px 14px;border-bottom:1px solid var(--line);color:#333;max-height:38vh;overflow:auto}
#info p{margin:0 0 6px}#info table{border-collapse:collapse;margin:4px 0}#info td{padding:1px 8px 1px 0}
.key{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:4px}.key span{display:flex;align-items:center;gap:4px;color:var(--muted);font-size:12px}
.key i{width:14px;height:10px;border:2px solid;border-radius:2px}
#list{overflow:auto;flex:1;margin:0;padding:0;list-style:none}
#list li{padding:6px 14px;border-bottom:1px solid var(--line);cursor:pointer}
#list li:hover{background:#f1f3f5}#list li.sel{background:#fff4e6;box-shadow:inset 3px 0 var(--acc)}
.t{font-weight:600;font-family:ui-monospace,Menlo,monospace}.m{color:var(--muted);font-size:12px}
.x{text-decoration:line-through;color:var(--muted)}
#count{color:var(--muted);font-size:12px}
#reset{position:absolute;right:10px;top:10px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
#hint{position:absolute;left:10px;bottom:10px;background:rgba(255,255,255,.9);padding:4px 8px;border-radius:6px;color:var(--muted);font-size:12px}
@media (max-width:800px){#app{grid-template-columns:1fr;grid-template-rows:55vh 45vh}#side{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<div id="app">
  <div id="view"><svg id="svg"></svg><button id="reset">Fit sheet</button>
    <div id="hint">← / → : step · wheel: zoom · drag: pan · click a box or a row</div></div>
  <aside id="side">
    <header>
      <select id="sheet"></select>
      <div id="nav"><button id="prev" title="previous step (←)">‹</button>
        <div id="stepname"><b></b><span></span></div>
        <button id="next" title="next step (→)">›</button></div>
      <div id="dots"></div>
      <input id="q" placeholder="Filter this step's list">
    </header>
    <div id="info"></div>
    <div style="padding:4px 14px" id="count"></div>
    <ul id="list"></ul>
  </aside>
</div>
<script>
const D = __DATA__, SHEETS = D.sheets, MM = D.mm, NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("svg"), view = document.getElementById("view"), list = document.getElementById("list");
const info = document.getElementById("info"), q = document.getElementById("q"), sel = document.getElementById("sheet");
const C = {blue:"#1c7ed6", purple:"#9c36b5", green:"#2f9e44", orange:"#f08c00", red:"#e03131", gray:"#868e96", teal:"#0c8599"};
const SRC = {bubble:C.red, stack:C.purple, text_near_bubble:C.orange, text:C.blue};
const LBL = {Instrument_Field:C.red, Instrument_Panel:C.blue, Instrument_Aux_Panel:C.teal};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const key = items => `<div class="key">${items.map(([c, l, d]) => `<span><i style="border-color:${c};${d ? "border-style:dashed" : ""}"></i>${l}</span>`).join("")}</div>`;
let S = null, step = 0, vb = null, items = [], cur = -1, layer = null;

// each step: title, sub, info(S) -> html, items(S) -> [{b:[x0,y0,x1,y1], c:color, dash, t:label, m:meta, x:struck, line:[x,y]}]
const STEPS = [
 {title:"Render + orientation", sub:"PDF page → 300 dpi image",
  info:S => `<p>The page is rasterised at 300 dpi (${S.W}×${S.H} px). A few text-dense tiles are OCR'd at 0/90/180/270° with
   the angle classifier off; the upright rotation reads far more wide, confident text.</p>
   <table>${Object.entries(S.orient).map(([k, v]) => `<tr><td>${k}°</td><td>${v} characters</td></tr>`).join("") || "<tr><td>not checked</td></tr>"}</table>
   <p><b>Rotation applied: ${S.rotation}°</b> (needs ≥3× the runner-up, else the page is kept as is).</p>`,
  items:S => []},
 {title:"OCR", sub:"RapidOCR (PP-OCRv6), tiled",
  info:S => `<p>Full-page OCR in 1600 px tiles with 20% overlap, once upright and once rotated 90° for vertical labels.
   Reads below 0.5 confidence and reads cut at a tile edge are dropped; duplicates from overlapping tiles are merged.</p>
   <p><b>${S.frags.length}</b> text fragments.</p>` + key([[C.blue, "upright read"], [C.purple, "rotated (vertical) read"]]),
  items:S => S.frags.map(f => ({b:f.slice(0, 4), c:f[6] ? C.purple : C.blue, t:f[4], m:`conf ${(+f[5]).toFixed(2)} · ${f[6]}°`}))},
 {title:"Bubble sizing", sub:"sets the detector's tile size",
  info:S => `<p>RF-DETR finds bubbles best when they are ~60 px wide in its 560 px input, so each sheet's bubble diameter is
   measured (Hough circles, cross-checked against confident RF-DETR boxes that contain text) and the tile is scaled to it.</p>
   <table><tr><td>diameter</td><td><b>${S.bubble_px ? (S.bubble_px / MM).toFixed(1) + " mm (" + Math.round(S.bubble_px) + " px)" : "not measured"}</b></td></tr>
   <tr><td>method</td><td>${esc(S.size_method)}</td></tr><tr><td>detector tile</td><td>${S.tile} px (run at ×0.8, ×1, ×1.25)</td></tr></table>
   <p class="m">${esc(S.size_note)}</p>` + (() => { const z = S.sizing || {}, ok = (z.hough_ok || []).length, rej = (z.hough_rejected || []).length, rf = (z.rf_boxes || []).length;
   return `<p><b>${ok}</b> clean circles measured (median → diameter) · <b>${rej}</b> Hough candidates rejected (not a hollow
   ring: crossed by lines, filled, or text) · <b>${rf}</b> confident RF-DETR boxes with text used to cross-check.</p>`
   + key([[C.green, "circle measured"], [C.gray, "candidate rejected", 1], [C.orange, "RF-DETR cross-check box"]]); })(),
  items:S => { const z = S.sizing || {}, circ = (c, col, t, m, dash) => ({b:[c[0] - c[2], c[1] - c[2], c[0] + c[2], c[1] + c[2]], c:col, t, m, circle:1, dash, x:dash});
   return [...(z.hough_ok || []).map(c => circ(c, C.green, "measured circle", `${(2 * c[2] / MM).toFixed(1)} mm`)),
           ...(z.rf_boxes || []).map(b => ({b, c:C.orange, t:"RF-DETR cross-check", m:`${(Math.max(b[2] - b[0], b[3] - b[1]) * 0.9 / MM).toFixed(1)} mm (box × 0.9)`})),
           ...(z.hough_rejected || []).map(c => circ(c, C.gray, "rejected candidate", `${(2 * c[2] / MM).toFixed(1)} mm`, 1))]; }},
 {title:"Symbol detection", sub:"RF-DETR instrument bubbles",
  info:S => `<p>RF-DETR (fine-tuned P&amp;ID detector) runs over sliding tiles at three scales; only instrument-bubble classes are
   kept (confidence ≥ 0.40), and duplicates across tiles/scales are merged.</p><p><b>${S.bubbles.length}</b> bubbles.</p>`
   + key([[C.red, "field instrument"], [C.blue, "panel (DCS)"], [C.teal, "aux panel"]]),
  items:S => S.bubbles.map(b => ({b:b.slice(0, 4), c:LBL[b[5]] || C.gray, t:b[5].replace("Instrument_", ""), m:`conf ${(+b[4]).toFixed(2)} · tile ${b[7]} px · #${b[6]}`}))},
 {title:"Text → bubble", sub:"which text belongs to which bubble",
  info:S => { const ins = S.frags.filter(f => f[7]).length, near = S.frags.filter(f => !f[7] && f[8] !== null && f[8] <= 10).length;
   return `<p>Each OCR fragment is linked to its nearest bubble: <b>inside</b> it (centre within the bubble's inscribed ellipse)
   becomes the bubble's tag text; <b>near</b> (≤ 10 mm) marks inline labels beside a bubble. Far text is not shown.</p>
   <p><b>${ins}</b> inside · <b>${near}</b> near</p>` + key([[C.green, "inside a bubble"], [C.orange, "near a bubble (line to it)"]]); },
  items:S => { const bc = {}; S.bubbles.forEach(b => bc[b[6]] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]);
   return S.frags.filter(f => f[7] || (f[8] !== null && f[8] <= 10)).map(f => ({b:f.slice(0, 4), c:f[7] ? C.green : C.orange,
    t:f[4], m:f[7] ? `inside bubble #${f[9]}` : `${f[8]} mm from bubble #${f[9]}`, line:f[7] ? null : bc[f[9]]})); }},
 {title:"Tag assembly", sub:"bubble lines, stacks, inline text",
  info:S => { const n = S.tags.filter(t => !t[5]).length, l = S.tags.length - n;
   return `<p>Bubble text is read top to bottom into a tag (<span class="t">PIT / 6519A</span> → PIT-6519A). Stacked short lines
   outside any detected bubble become <b>stack</b> tags (bubbles the detector missed); tag-shaped strings elsewhere become
   <b>text</b> tags. Local devices with no DCS signal (HV, PSV, PG, TW, PCV/TCV/LCV, RO…) are removed here.</p>
   <p><b>${n}</b> tags · <b>${l}</b> local devices removed</p>`
   + key([[C.red, "bubble"], [C.purple, "stack"], [C.orange, "text near bubble"], [C.blue, "text"], [C.gray, "local device (removed)", 1]]); },
  items:S => S.tags.map(t => ({b:t[0], c:t[5] ? C.gray : SRC[t[2]] || C.gray, dash:t[5], x:t[5], t:t[1],
    m:`${t[5] ? "local device · " : ""}${t[2]} · read: ${t[4]} · conf ${(+t[3]).toFixed(2)}`}))},
 {title:"Precision filter", sub:"batch-learned rules, nothing deleted",
  info:S => { const k = S.filt.filter(r => r[2]).length;
   return `<p>Learns the batch's own tag formats and letter codes from its bubble reads, then repairs junk at the edges of reads,
   splits merged reads, and marks drops (dates, tables, notes, captions, references, formats the batch never uses).</p>
   <p><b>${k}</b> kept · <b>${S.filt.length - k}</b> dropped</p>` + key([[C.blue, "kept"], [C.gray, "dropped", 1]]); },
  items:S => S.filt.map(r => ({b:r[0], c:r[2] ? C.blue : C.gray, dash:!r[2], x:!r[2], t:r[1],
    m:[r[5], r[4] && r[4] !== r[1] ? `read as ${r[4]}` : "", r[6] && `fix: ${r[6]}`, !r[2] && `dropped: ${r[3]}`].filter(Boolean).join(" · ")}))},
 {title:"Descriptions", sub:"descgen, from the drawing only",
  info:S => `<p>Each kept bubble tag is described from the sheet itself: equipment name (title block, data boxes, nearby
   labels), position (suction/discharge, stage, inlet/outlet) traced along the pipe, and the measured variable from the
   ISA letters. Click a row to see the evidence for each part.</p><p><b>${S.desc.length}</b> tags described.</p>`
   + key([[C.green, "described tag"]]),
  items:S => S.desc.map(d => ({b:d[0], c:C.green, t:d[1], m:d[2] || "(no description)",
    more:(d[4] ? `review: ${d[4]}\n` : "") + d[5].map(p => `${p[0]}: ${p[1]}  ← ${p[2]}`).join("\n")}))},
];

SHEETS.forEach((s, i) => { const o = document.createElement("option"); o.value = i;
  o.textContent = `${s.file.replace(/\.pdf$/i, "")} · p${s.page + 1} · ${s.desc.length} described`; sel.appendChild(o); });
const dots = document.getElementById("dots");
STEPS.forEach((_, i) => { const d = document.createElement("i"); d.title = STEPS[i].title; d.onclick = () => go(i); dots.appendChild(d); });

function setVB(v) { vb = v; svg.setAttribute("viewBox", v.join(" ")); }
function fit() { setVB([0, 0, S.W, S.H]); }
function el(n, a) { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; }

function load(i) {
  S = SHEETS[i]; svg.innerHTML = "";
  svg.appendChild(el("image", {href:S.img, width:S.W, height:S.H}));
  svg.appendChild(el("rect", {id:"dim", x:0, y:0, width:S.W, height:S.H}));
  layer = el("g", {}); svg.appendChild(layer);
  document.title = `${S.file} · page ${S.page + 1}`;
  fit(); go(step);
}

function go(n) {
  step = Math.max(0, Math.min(STEPS.length - 1, n)); const St = STEPS[step];
  document.querySelector("#stepname b").textContent = `${step + 1}. ${St.title}`;
  document.querySelector("#stepname span").textContent = St.sub;
  document.getElementById("prev").disabled = step === 0; document.getElementById("next").disabled = step === STEPS.length - 1;
  [...dots.children].forEach((d, i) => d.classList.toggle("on", i <= step));
  info.innerHTML = `<p class="m">${esc(S.file)} · page ${S.page + 1}</p>` + St.info(S);
  document.getElementById("dim").setAttribute("opacity", step === 0 ? 0 : 0.35);
  items = St.items(S); cur = -1; draw(); renderList();
}

function draw() {
  layer.innerHTML = "";
  items.forEach((it, i) => {
    const [x0, y0, x1, y1] = it.b;
    if (it.line) layer.appendChild(el("line", {x1:(x0 + x1) / 2, y1:(y0 + y1) / 2, x2:it.line[0], y2:it.line[1], stroke:it.c, class:"ln"}));
    const r = it.circle ? el("circle", {cx:(x0 + x1) / 2, cy:(y0 + y1) / 2, r:(x1 - x0) / 2})
                        : el("rect", {x:x0, y:y0, width:x1 - x0, height:y1 - y0});
    r.setAttribute("class", "b" + (it.dash ? " dash" : "")); r.style.stroke = it.c; r.dataset.i = i;
    const ti = el("title", {}); ti.textContent = `${it.t} — ${it.m}`; r.appendChild(ti);
    r.addEventListener("click", e => { e.stopPropagation(); pick(i, false); });
    layer.appendChild(r); it.el = r;
  });
}

function renderList() {
  const f = q.value.trim().toUpperCase(); list.innerHTML = ""; let n = 0;
  items.forEach((it, i) => {
    const vis = !f || (it.t + " " + it.m).toUpperCase().includes(f);
    it.el.style.display = vis ? "" : "none"; if (!vis) return; n++;
    const li = document.createElement("li"); li.dataset.i = i; if (i === cur) li.className = "sel";
    li.innerHTML = `<div class="t${it.x ? " x" : ""}" style="border-left:4px solid ${it.c};padding-left:6px"></div><div class="m"></div>`;
    li.querySelector(".t").textContent = it.t; li.querySelector(".m").textContent = it.m;
    if (it.more && i === cur) { const p = document.createElement("pre"); p.className = "m"; p.style.whiteSpace = "pre-wrap"; p.style.margin = "4px 0 0";
      p.textContent = it.more; li.appendChild(p); }
    li.addEventListener("click", () => pick(i, true)); list.appendChild(li);
  });
  document.getElementById("count").textContent = items.length ? `${n} of ${items.length} shown` : "";
}

function pick(i, zoom) {
  if (cur >= 0 && items[cur]) items[cur].el.classList.remove("sel");
  cur = i; const it = items[i]; it.el.classList.add("sel"); layer.appendChild(it.el);
  if (zoom) { const [x0, y0, x1, y1] = it.b, w = Math.max(x1 - x0, y1 - y0, 60) * 12, a = view.clientWidth / view.clientHeight;
    setVB([(x0 + x1) / 2 - w * a / 2, (y0 + y1) / 2 - w / 2, w * a, w]); }
  renderList(); const li = list.querySelector(`li[data-i="${i}"]`); if (li && !zoom) li.scrollIntoView({block:"center"});
}

function toSvg(e) { const b = svg.getBoundingClientRect(), s = Math.max(vb[2] / b.width, vb[3] / b.height);
  return [vb[0] - (b.width * s - vb[2]) / 2 + (e.clientX - b.left) * s, vb[1] - (b.height * s - vb[3]) / 2 + (e.clientY - b.top) * s, s]; }
view.addEventListener("wheel", e => { e.preventDefault(); const [x, y] = toSvg(e), k = Math.exp(e.deltaY * 0.0015);
  setVB([x - (x - vb[0]) * k, y - (y - vb[1]) * k, vb[2] * k, vb[3] * k]); }, {passive:false});
let drag = null;
view.addEventListener("mousedown", e => { if (e.target.closest("button")) return; drag = [e.clientX, e.clientY, vb.slice()]; view.classList.add("drag"); });
window.addEventListener("mousemove", e => { if (!drag) return; const s = toSvg(e)[2];
  setVB([drag[2][0] - (e.clientX - drag[0]) * s, drag[2][1] - (e.clientY - drag[1]) * s, vb[2], vb[3]]); });
window.addEventListener("mouseup", () => { drag = null; view.classList.remove("drag"); });
document.getElementById("reset").onclick = fit;
document.getElementById("prev").onclick = () => go(step - 1);
document.getElementById("next").onclick = () => go(step + 1);
sel.onchange = () => load(+sel.value); q.oninput = renderList;
document.addEventListener("keydown", e => { if (e.target === q) return;
  if (e.key === "ArrowRight") go(step + 1); else if (e.key === "ArrowLeft") go(step - 1); });
if (SHEETS.length) load(0);
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help="PDF files and/or folders of PDFs")
    ap.add_argument("--out", default="pipeline_viewer_output", help="pipeline output folder (resumable)")
    ap.add_argument("--from", dest="existing", help="existing batch output folder (must have steps/); no extraction")
    a = ap.parse_args()
    out = Path(a.existing or a.out)
    if not a.existing:
        if not a.inputs:
            ap.error("give PDF files/folders, or --from <existing output folder>")
        from tag_viewer import extract
        extract(a.inputs, out)
    if not (out / "descriptions.csv").exists():
        describe(out)
    write_html([s for sheets in collect(out).values() for s in sheets], out / "pipeline_steps.html")
