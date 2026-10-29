"""The equipment register: what equipment a sheet shows and what its labels say, read once per sheet from the drawing
itself (no equipment list).

  names        tag -> name words, from (in order of trust)
                 data boxes      tag printed over 'TITLE/SERVICE: <name>' (or over a name line ending in an equipment noun)
                 printed names   '1V-1101 FLASH VESSEL', 'FROM INTERSTAGE COOLER 1E-1109'
                 connectors      'LNG TO LNG PRODUCT PUMP 5P551A/B'
                 tag letter      '1P-1115' with nothing printed: P = pump (rules.EQUIP_LETTER; kept apart, used for kind only
                                 by the process finder)
  labels       every equipment tag printed in the drawing body (symbol labels)
  items        labelled equipment symbols with their outline box (+ shapes with an item noun inside: CRANKCASE)
  connectors   off-sheet connectors / named pipe ends: flow direction (in / out of this sheet), target tags and name,
               service (fluid) words, a box to reach it by
  targets      what an instrument can belong to along the lines: named symbols + connectors naming equipment
  main         the sheet's own equipment (data box matching the title), n_stages (compressor stages printed)
Ownership along lines (owner / flow) uses lines.Network: distance along the drawn lines, flow arrows on the path."""
import math
import re
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
from .sheet import MM, ctr, merge_overlap
from .text import split_words, equipment_phrase, kind_of, rules_of, head_noun, stage_of, describe_name
from . import rules as R

INST_REF = r"(?<![0-9A-Z])(?:\d{3})?[A-Z]{2,5}-?\d{4}[A-Z]?(?!\d)"     # an instrument quoted in text: 111PICA-0092, LIC5018


def gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


# Equipment names read from the other sheets of the same batch: tag -> dict(words, src). Filled by build_register();
# a sheet uses an entry only for a tag that is printed on that sheet (the relevance link) and that it cannot name itself.
SHARED = {}


def build_register(sheets):
    """sheets: iterable of (sheet name, Sheet). Collects every name a sheet reads from its own drawing."""
    SHARED.clear()
    found = {}
    for name, sh in sheets:
        for tag, n in sh.plant.names.items():
            if not n.get("shared"):
                found.setdefault(tag, dict(words=n["words"], src=f"named on sheet {name}: {n['src']}", shared=True))
    SHARED.update(found)
    return SHARED


class Plant:
    def __init__(self, sheet):
        self.s = s = sheet
        P = s.profile
        base = P["equipment_tag"]
        self.tag_full = re.compile(rf"{base}[A-Z]{{0,2}}")                       # a fragment that is a tag
        self.tag_in = re.compile(rf"(?<!\d){base}(?:[A-Z](?![A-Z]))?(?!\d)")         # a tag inside a longer text
        # a tag with its instance letters as quoted at a connector: '5P551A/B' (read '5P551AB'), '1S-1114AB'
        self.tag_inst = re.compile(rf"(?<!\d)({base})(?:([A-Z]{{1,3}})(?![A-Z]))?(?!\d)")
        self.panel_x = s.panel_x
        self.names = {}            # tag -> dict(words, src)
        self.headers = []          # bboxes of data-box tag headers (not symbol labels)
        self._data_boxes()
        self._names_from_text()
        self.connectors = self._connectors()
        self._names_from_register()
        self.letter_names = self._letter_names()
        self.labels = self._labels()
        sym = self._symbols()
        self.items = sym + self._labelled_shapes(sym)
        self.targets = self._targets()
        self.n_stages = self._stages()
        self.main = self._main_equipment()
        # the sheet's own equipment name: its tag's name, else the title block's equipment words
        te = s.title.get("equipment")
        self.main_name = self.names[self.main]["words"] if self.main else (split_words(te.replace(" ", "")) if te else None)
        self.main_kind = kind_of(self.main_name)
        self.net = None
        self._flood = {}

    # ---------- reading helpers ----------
    def row_text(self, f, max_gap_mm=3):
        """The fragment joined with fragments continuing it on the same text line (overlapping or just after)."""
        x0, y0, x1, y1 = f["bbox"]; h = y1 - y0; txt = f["text"]
        for g in sorted(self.s.frags, key=lambda g: g["bbox"][0]):
            gx0, gy0, gx1, gy1 = g["bbox"]
            if g is f or abs(gy0 - y0) > 0.5 * h or abs(gy1 - y1) > 0.6 * h or gx1 <= x1 or gx0 > x1 + max_gap_mm * MM:
                continue
            txt = merge_overlap(txt, g["text"]) if gx0 < x1 else txt + g["text"]
            x1 = gx1
        return txt

    def base_of(self, tag):
        """'5P551A' -> ('5P551', 'A'): the equipment and its instance letter."""
        m = re.fullmatch(r"(.*\d)([A-Z]*)", tag)
        return (m.group(1), m.group(2)) if m else (tag, "")

    def name_for(self, tag):
        """Name entry of a tag, or of another instance of the same equipment (5P551B named in 5P551A's data box)."""
        if tag in self.names:
            return self.names[tag]
        base, _ = self.base_of(tag)
        return next((v for k, v in self.names.items() if self.base_of(k)[0] == base), None)

    def info(self, tag, letter=False):
        """dict(tag, name words, kind, class, stage). letter=True: also a name made from the tag's letter code."""
        n = self.names.get(tag)
        w = n["words"] if n else (self.letter_names.get(tag) if letter else None)
        k = kind_of(w)
        return dict(tag=tag, name=w, kind=k, cls=rules_of(k).get("cls"), stage=stage_of(w or []))

    def _names_from_register(self):
        """Tags printed on this sheet but named only on another sheet of the batch (e.g. a cooler named on its own sheet,
        whose outlet temperature is drawn on the next one)."""
        if not SHARED:
            return
        for f in self.s.frags:
            for m in self.tag_in.finditer(f["text"]):
                t = m.group(0)
                if t not in self.names and t in SHARED:
                    self.names[t] = dict(SHARED[t])

    # ---------- equipment data boxes ----------
    def _data_boxes(self):
        """A tag printed over its name line: 'TITLE/SERVICE: <name>' (the profile's label), or - drawings without the
        label - a line directly under the tag that ends in an equipment noun ('5P551A' / 'LNG PRODUCT PUMP')."""
        frags, labels = self.s.frags, self.s.profile["data_box_labels"]
        for f in frags:
            lab = next((l for l in labels if f["text"].startswith(l)), None)
            if not lab:
                continue
            x0, y0 = f["bbox"][:2]
            tags = [g for g in frags if self.tag_full.fullmatch(g["text"]) and abs(g["bbox"][0] - x0) < 6 * MM
                    and 0 <= y0 - g["bbox"][3] < 8 * MM]
            if tags:
                row = self.row_text(f)
                full = self.s.name_line(dict(f, text=row)) if row == f["text"] else row
                w = split_words(full[len(lab):] if full.startswith(lab) else row[len(lab):])
                self.names[tags[0]["text"]] = dict(words=w, src=f"data box {tags[0]['text']}: {' '.join(w)}")
                self.headers.append(tags[0]["bbox"])
        for f in frags:
            t, b = f["text"], f["bbox"]
            if not self.tag_full.fullmatch(t) or t in self.names:
                continue
            h = b[3] - b[1]; cx = ctr(b)[0]
            below = [g for g in frags if 0 <= g["bbox"][1] - b[3] < 1.5 * h
                     and abs(ctr(g["bbox"])[0] - cx) < max(g["bbox"][2] - g["bbox"][0], b[2] - b[0]) / 2]
            for g in below:
                gt = self.s.name_line(g)
                nm = equipment_phrase(split_words(gt))
                if nm:
                    self.names[t] = dict(words=nm.split(), src=f"data box '{t} / {' '.join(split_words(gt))}'")
                    self.headers.append(b)
                    break

    def _names_from_text(self):
        """'1V-1101FLASHVESSEL' / 'INTERSTAGECOOLER1E-1109': a tag written next to a name ending in an equipment noun."""
        for f in self.s.frags:
            for m in self.tag_in.finditer(f["text"]):
                tag = m.group(0)
                if tag in self.names:
                    continue
                for after, side in ((True, f["text"][m.end():]), (False, f["text"][:m.start()])):
                    w = split_words(side) if side else []
                    cand = []
                    if after:
                        n = [i for i, x in enumerate(w[:6]) if x in R.KIND_OF_NOUN]
                        cand = w[:n[-1] + 1] if n else []
                    elif w and w[-1] in R.KIND_OF_NOUN:
                        for x in reversed(w[-6:]):
                            if not re.fullmatch(r"[A-Z]{2,}", x) or x in ("TO", "FROM", "THE", "OF", "AND", "PROTECT"):
                                break
                            cand.insert(0, x)
                    if cand:
                        self.names[tag] = dict(words=cand, src=f"printed beside the tag: '{f['text']}'")
                        break

    def _letter_names(self):
        """Tags with no printed name: the noun of the tag's equipment letter code (P = pump, T = tank ...)."""
        out = {}
        for f in self.s.frags:
            t = f["text"]
            if t in self.names or not self.tag_full.fullmatch(t) or f["bbox"][0] > self.panel_x:
                continue
            if any(gap(f["bbox"], c["bbox"]) < 6 * MM for c in self.connectors):    # tag quoted at a connector
                continue
            m = re.match(r"^\d{0,2}([A-Z]+)-?\d", t)
            noun = R.EQUIP_LETTER.get(m.group(1)) if m else None
            if noun:
                out[t] = [noun] + ([t[-1]] if t[-1].isalpha() else [])
        return out

    # ---------- equipment tags printed in the drawing body ----------
    def _labels(self):
        return [dict(tag=f["text"], bbox=f["bbox"], c=ctr(f["bbox"])) for f in self.s.frags
                if self.tag_full.fullmatch(f["text"]) and f["bbox"][0] <= self.panel_x and f["bbox"] not in self.headers]

    # ---------- symbols: the outline each label names ----------
    def _symbols(self):
        """Labelled equipment symbols with their outline (labels quoted at a connector, and legend lists of tags
        stacked in a column, are not symbols)."""
        out = []
        for l in self.labels:
            t = l["tag"]
            words = (self.names.get(t) or {}).get("words") or self.letter_names.get(t)
            if not words:
                continue
            if any(gap(l["bbox"], c["bbox"]) < 6 * MM for c in self.connectors):
                continue
            col = [g for g in self.labels if g is not l and g["tag"] in self.names and abs(g["bbox"][0] - l["bbox"][0]) < 2 * MM
                   and abs(g["bbox"][1] - l["bbox"][1]) < 12 * MM]
            if len(col) >= 2:
                continue
            reg = self.shape_near(l["bbox"])
            if reg:
                out.append(dict(tag=t, text="".join(words), bbox=reg, label=l["bbox"], **describe_name("".join(words))))
        return out

    def _labelled_shapes(self, symbols):
        """Shapes with an item noun written inside them and no tag (the machine's own crankcase / sump)."""
        out = []
        for f in self.s.frags:
            k = R.KIND_OF_NOUN.get(f["text"])
            if not k or not rules_of(k).get("circuit") or f["bbox"][0] > self.panel_x:
                continue
            c = ctr(f["bbox"])
            if any(i["bbox"][0] <= c[0] <= i["bbox"][2] and i["bbox"][1] <= c[1] <= i["bbox"][3] for i in symbols + out):
                continue
            reg = self.s.box_around(f["bbox"])
            if reg and (reg[2] - reg[0]) > 15 * MM:
                out.append(dict(tag=None, text=f["text"], bbox=self.grow(reg), label=f["bbox"], **describe_name(f["text"])))
        return out

    def shape_near(self, b, max_mm=20):
        """Outline of the symbol a label names: the closed shape around the label, else the nearest one beside it."""
        s = self.s
        cx, cy = ctr(b)
        def ok(r, small=5):
            return r and small * MM < (r[2] - r[0]) < 250 * MM and small * MM < (r[3] - r[1]) < 250 * MM \
                and max(r[2] - r[0], r[3] - r[1]) > 5 * MM
        def beside(lo, hi, skip=None):
            for d in np.arange(lo, hi, 0.5) * MM:
                for p in ((cx, b[1] - d), (cx, b[3] + d), (b[2] + d, cy), (b[0] - d, cy)):
                    r = s.region_at(*p)
                    if ok(r, 1.5 if hi <= 4 else 5) and r != skip:
                        return r
            return None
        around = s.box_around(b)
        around = around if ok(around) else None
        near = beside(1.0, 4.0, around)
        # a label written inside a big outline next to a small symbol (heater inside its tank) names the small one
        if near and (not around or (near != around and (near[2] - near[0]) * (near[3] - near[1]) <
                                    0.5 * (around[2] - around[0]) * (around[3] - around[1]))):
            return self.grow(near)
        if around:
            return self.grow(around)
        r = beside(4.0, max_mm)
        return self.grow(r) if r else None

    def grow(self, reg):
        """A symbol split inside (vessel liquid line, exchanger channel head, pass partition): join the closed shapes
        that sit against one of its sides and span most of that side."""
        s = self.s
        reg = list(reg)
        for _ in range(6):
            grown = False
            x0, y0, x1, y1 = reg
            for side in ("bottom", "top", "right", "left"):
                for f in (0.25, 0.5, 0.75):
                    p = {"bottom": (x0 + f * (x1 - x0), y1 + 1.5 * MM), "top": (x0 + f * (x1 - x0), y0 - 1.5 * MM),
                         "right": (x1 + 1.5 * MM, y0 + f * (y1 - y0)), "left": (x0 - 1.5 * MM, y0 + f * (y1 - y0))}[side]
                    r = s.region_at(*p)
                    if not r:
                        continue
                    touch = {"bottom": abs(r[1] - y1), "top": abs(r[3] - y0), "right": abs(r[0] - x1), "left": abs(r[2] - x0)}[side]
                    encloses = r[0] <= x0 and r[2] >= x1 and r[1] <= y0 and r[3] >= y1
                    inside = r[0] >= x0 and r[2] <= x1 and r[1] >= y0 and r[3] <= y1
                    same = (abs(r[0] - x0) < 4 * MM and abs(r[2] - x1) < 4 * MM) if side in ("bottom", "top") else \
                           (abs(r[1] - y0) < 4 * MM and abs(r[3] - y1) < 4 * MM)
                    if same and touch < 2.5 * MM and not encloses and not inside:
                        reg = [min(reg[0], r[0]), min(reg[1], r[1]), max(reg[2], r[2]), max(reg[3], r[3])]
                        grown = True
                        break
                if grown:
                    break
            if not grown:
                break
        return tuple(reg)

    # ---------- off-sheet connectors and named pipe ends ----------
    def _connectors(self):
        """Each connector = the text lines of one off-sheet connector / named pipe end, grouped:
           'around' layout: the lines stacked around a TO / FROM line, a named pipe end (OUTLET, DRAIN ...) or a
                            drawing reference ('FLASH GAS | 455 | L001-11100-1007 | FROM 1E-1106');
           'above' layout : up to three lines stacked on a drawing-reference box ('LNG TO LNG / PRODUCT PUMP 5P551A/B').
        Which layout a drawing house uses is a style setting (profile['connector_text'])."""
        P, frags = self.s.profile, self.s.frags
        ref = re.compile(P["sheet_ref"]) if P.get("sheet_ref") else None
        out, used = [], set()
        if P.get("connector_text") == "above":
            for r in frags:
                if ref and ref.fullmatch(r["text"]):
                    block, anchor = self._block_above(r)
                    if block:
                        c = self._parse(list(reversed(block)), anchor=anchor, refs=[r])
                        if c:
                            out.append(c)
            return out
        for f in frags:
            t = f["text"]
            if id(f) in used or f["bbox"][0] > self.panel_x or len(t) > 40 or t.startswith(self.s.profile["data_box_labels"]):
                continue
            if not (self._direction(t) or (ref and ref.search(t))):
                continue
            cl = self._stack(f)
            for g in cl:
                used.add(id(g))
            c = self._parse(cl, refs=[g for g in cl if ref and ref.search(g["text"])])
            if c:
                out.append(c)
        return out

    @staticmethod
    def _direction(t):
        """(word, 'in' | 'out', strength) of the first connector word in a text line: TO / FROM at its start are
        strongest (2), INLET / OUTLET (1), the utility words (SUPPLY / RETURN / DRAIN / VENT: 0)."""
        for word, d, where in R.CONNECTOR_DIR:
            if where == "prefix" and re.match(rf"^{word}(\d|[A-Z]{{3,}})", t):
                return word, d, 2
        for word, d, where in R.CONNECTOR_DIR:
            if where == "any" and word in t:
                return word, d, 1 if word in ("INLET", "OUTLET") else 0
        return None

    def _parse(self, cl, anchor=None, refs=()):
        """Flow direction, target (tags, instance letters, name words), service words of one connector's lines."""
        txt = [g["text"] for g in cl]
        refs_t = {g["text"] for g in refs}
        dirs = []
        for i, t in enumerate(txt):
            d = self._direction(t)
            if d:
                dirs.append((d[2], i, d))
            else:     # 'LNGTOLNGPRODUCTPUMP': the direction word inside the line, followed by equipment
                w = split_words(t)
                for j, x in enumerate(w):
                    hit = next((e for e in R.CONNECTOR_DIR if e[0] == x and e[2] == "prefix"), None)
                    if hit and (head_noun(w[j + 1:]) or self.tag_in.search(t)):
                        dirs.append((2, i, (x, hit[1], 2)))
        flow, word, line = None, None, None
        if dirs:
            top = max(d[0] for d in dirs)
            best = [d for d in dirs if d[0] == top]
            pick = next((d for d in best if d[2][1] == "in"), best[-1])     # a FROM line names where the fluid comes from
            flow, word, line = pick[2][1], pick[2][0], pick[1]
        if flow is None and not refs:
            return None
        tags, letters = [], []
        for t in txt:
            if t in refs_t:
                continue
            for m in self.tag_inst.finditer(t):
                base, extra = m.group(1), m.group(2) or ""
                tags.append(base + extra[:1])
                if extra:
                    letters += list(extra)
        inst = [m.group(0) for t in txt for m in re.finditer(INST_REF, re.sub(r"^(TO|FROM|TOWARDS)", "", t))]
        # destination: the text after the direction word, on its line and the lines under it
        dest, service, names = "", [], []
        for i, t in enumerate(txt):
            if t in refs_t or re.fullmatch(r"[\d\- ]+", t) or re.match(r"^(TP-?\d|SEEDWG|NOTE)", t) \
                    or re.match(r"^\d+-\d+-[A-Z]", t) or re.search(INST_REF, t) and not (line is not None and i >= line):
                continue
            body = t
            if line is not None and i == line and word:
                k = t.find(word) if not t.startswith(word) else 0
                head, body = t[:k], t[k + len(word):]
                if head:
                    service += [x for x in split_words(head) if re.fullmatch(r"[A-Z]{3,}", x)]
            if line is not None and i >= line:
                dest += body
            else:
                service += [x for x in split_words(re.sub(self.tag_in, "", body)) if re.fullmatch(r"[A-Z]{3,}", x)
                            or x in R.FLUID_WORDS]
        dest_clean = self.tag_inst.sub("", dest)
        dest_clean = re.sub(INST_REF, "", dest_clean)
        dw = [x for x in split_words(dest_clean) if re.fullmatch(r"[A-Z]{3,}", x) or x in R.FLUID_WORDS]
        nm = equipment_phrase(dw)
        if nm:
            names = nm.split()
        elif tags and self.name_for(tags[0]):
            names = self.name_for(tags[0])["words"]
        if not head_noun(names):
            service, names = service + dw, []
        box = (min(g["bbox"][0] for g in cl), min(g["bbox"][1] for g in cl),
               max(g["bbox"][2] for g in cl), max(g["bbox"][3] for g in cl))
        if refs and anchor is None:
            # the connector box: the drawing-reference text and the shape drawn around it
            for g in refs:
                pent = self.s.box_around(g["bbox"])
                if pent and (pent[2] - pent[0]) < 60 * MM:
                    box = (min(box[0], pent[0]), min(box[1], pent[1]), max(box[2], pent[2]), max(box[3], pent[3]))
        for tg in tags:
            if names and tg not in self.names:
                self.names[tg] = dict(words=names, src=f"connector '{' / '.join(txt)}'")
        d = describe_name(dest_clean or "".join(service))
        return dict(flow=flow, tags=tags, letters=sorted(set(letters)), instruments=inst, service=service, names=names,
                    bbox=box, anchor=anchor or box, text=" / ".join(txt), dest=dest,
                    consumer=any(w in dest for w in R.CONSUMER_WORDS), **d)

    def _stack(self, f, gap_mm=3.5, side_mm=6, max_h_mm=30):
        """Text lines stacked with `f` (each within `gap_mm` above / below the stack and overlapping it sideways)."""
        cl = [f]; box = list(f["bbox"]); grown = True
        while grown:
            grown = False
            for g in self.s.frags:
                if g in cl or g["bbox"][0] > self.panel_x or len(g["text"]) > 40:
                    continue
                gx0, gy0, gx1, gy1 = g["bbox"]
                if gx0 < box[2] + side_mm * MM and gx1 > box[0] - side_mm * MM and gy0 < box[3] + gap_mm * MM \
                        and gy1 > box[1] - gap_mm * MM and max(box[3], gy1) - min(box[1], gy0) < max_h_mm * MM:
                    cl.append(g); box = [min(box[0], gx0), min(box[1], gy0), max(box[2], gx1), max(box[3], gy1)]
                    grown = True
        return sorted(cl, key=lambda g: g["bbox"][1])

    def _block_above(self, r):
        """Up to three text lines stacked on a reference box, nearest first, aligned left, right or centred."""
        frags = self.s.frags
        rb = r["bbox"]; h = rb[3] - rb[1]
        block, top = [], rb
        while len(block) < 3:
            first = not block
            up = [g for g in frags if g is not r and -0.4 * h <= top[1] - g["bbox"][3] < (2.5 if first else 0.8) * h
                  and g["bbox"][1] < top[1] - 0.3 * h and g not in block
                  and g["bbox"][0] < top[2] + 8 * MM and g["bbox"][2] > top[0] - 8 * MM
                  and (not first or g["bbox"][2] - g["bbox"][0] > 2 * h)]
            if not up:
                break
            g = max(up, key=lambda g: g["bbox"][3])
            if block and not (abs(g["bbox"][0] - top[0]) < 4 * MM or abs(g["bbox"][2] - top[2]) < 4 * MM
                              or abs(ctr(g["bbox"])[0] - ctr(top)[0]) < 4 * MM):
                break
            block.append(g); top = g["bbox"]
        if not block:
            return [], None
        # where the line attaches: the reference box under the text
        bx0 = min(g["bbox"][0] for g in block + [r]); bx1 = max(g["bbox"][2] for g in block + [r])
        return block, (bx0 - 3 * MM, block[0]["bbox"][3], bx1 + 3 * MM, rb[3] + 3 * MM)

    # ---------- ownership targets ----------
    def _targets(self):
        """What an instrument may belong to along the lines: every labelled symbol whose equipment has a name, and every
        connector that names the equipment it leads to (one entry per printed label)."""
        out = []
        for l in self.labels:
            nm = self.name_for(l["tag"])
            if not nm:
                continue
            base, letter = self.base_of(l["tag"])
            name = " ".join(nm["words"])
            out.append(dict(tag=l["tag"], base=base, letters=[letter[:1]] if letter else [], name=name,
                            kind=kind_of(nm["words"]), label=l["bbox"], via="symbol",
                            src=f"equipment label '{l['tag']}' on the drawing; name from {nm['src']}"))
        for c in self.connectors:
            if not c["names"] or not c["flow"] or c["flow"] not in ("in", "out"):
                continue
            tag = c["tags"][0] if c["tags"] else None
            base = self.base_of(tag)[0] if tag else " ".join(c["names"])
            out.append(dict(tag=tag or " ".join(c["names"]), base=base, letters=c["letters"] or
                            ([self.base_of(tag)[1][:1]] if tag and self.base_of(tag)[1] else []),
                            name=" ".join(c["names"]), kind=kind_of(c["names"]), label=c["anchor"], via="connector",
                            direction="to" if c["flow"] == "out" else "frm", src=f"off-page connector '{c['text']}'"))
        return out

    def has_siblings(self, base, letter):
        """Another instance of the same equipment is printed somewhere (5P551B beside 5P551A, or '5P551A/B')."""
        if not letter:
            return False
        txt = " ".join(f["text"] for f in self.s.frags)
        return any(re.search(re.escape(base) + rf"(?:{L}(?![0-9])|[A-Z/]*{L}[A-Z/]*(?![0-9]))", txt)
                   for L in "ABCDEFGH" if L != letter) or bool(re.search(re.escape(base) + letter + r"/?[A-Z]", txt))

    # ---------- sheet-level facts ----------
    def _stages(self):
        n = 0
        for f in self.s.frags:
            for m in re.finditer(r"(\d)(ST|ND|RD|TH)?STAGE", f["text"]):
                n = max(n, int(m.group(1)))
        return n or None

    def _main_equipment(self):
        """The sheet's own equipment: the named tag whose name matches the title block."""
        title = (self.s.title.get("equipment") or "").replace(" ", "")
        if not title:
            return None
        hits = [tag for tag, n in self.names.items() if "".join(n["words"]) and
                ("".join(n["words"]) == title or "".join(n["words"]).endswith(title) or title.endswith("".join(n["words"])))]
        # several instances named like the title (5P551A, 5P551B): the title names the machine type, not one of them
        return hits[0] if len(hits) == 1 else None

    # ---------- equipment shape at a point ----------
    def equipment_at_region(self, reg, mm=25):
        """A closed shape (bbox) -> the equipment label inside it, else the nearest label within `mm` of its outline."""
        x0, y0, x1, y1 = reg
        best = None
        for l in self.labels:
            lx, ly = l["c"]
            dx = max(x0 - lx, 0, lx - x1); dy = max(y0 - ly, 0, ly - y1)
            d = np.hypot(dx, dy) / MM
            if d < mm and (best is None or d < best[0]):
                best = (d, l)
        return best

    # ======================= ownership along the lines =======================
    def network(self):
        if self.net is None:
            from .lines import Network
            s = self.s
            self.net = Network(s.ink, s.profile["line_gap_mm"], s.profile["line_break_mm"])
        return self.net

    def _target_flood(self, k):
        if k not in self._flood:
            it = self.targets[k]
            net = self.network()
            seeds = net.nodes_in(lambda x, y: True, it["label"]) if it["via"] == "connector" else self._label_seeds(it["label"])
            self._flood[k] = net.flood(seeds) if seeds else None
        return self._flood[k]

    def _label_seeds(self, b):
        """Lines of the drawn equipment next to its label: the outline enclosing the label if it is drawn inside a
        shape, else the ink within 5 mm around the label (not the label's own letters)."""
        s, net = self.s, self.network()
        box = s.box_around(b)
        band = 1.5 * MM
        def outline(x0, y0, x1, y1):
            return net.nodes_in(lambda x, y: not (x0 + band < x < x1 - band and y0 + band < y < y1 - band),
                                (x0 - band, y0 - band, x1 + band, y1 + band))
        if box and (box[2] - box[0]) > 15 * MM and (box[3] - box[1]) > 15 * MM:
            return outline(*box)
        pad = 5 * MM
        win = (int(b[0] - pad), int(b[1] - pad), int(b[2] + pad), int(b[3] + pad))
        txt = [f["bbox"] for f in s.frags if f["bbox"][0] < win[2] and f["bbox"][2] > win[0]
               and f["bbox"][1] < win[3] and f["bbox"][3] > win[1]]
        def not_text(x, y):
            return not any(t[0] - 0.5 * MM <= x <= t[2] + 0.5 * MM and t[1] - 0.5 * MM <= y <= t[3] + 0.5 * MM for t in txt)
        seeds = net.nodes_in(not_text, win)
        if seeds:
            return seeds
        reg = s.region_at(ctr(b)[0], b[1] - 2 * MM, win_mm=500)   # label alone inside a large shape (column, shell)
        return outline(*reg) if reg else []

    def owner(self, bbox):
        """Ownership candidates by distance along lines from the bubble: [(mm, target index, start node)],
        one per equipment instance."""
        if not self.targets:
            return []
        net = self.network()
        cx, cy = ctr(bbox)
        r = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 4
        nodes = net.nodes_in(lambda x, y: r - 1.0 * MM <= math.hypot(x - cx, y - cy) <= r + 1.0 * MM,
                             (cx - r - MM, cy - r - MM, cx + r + MM, cy + r + MM))
        if not nodes:
            return []
        out = []
        for k in range(len(self.targets)):
            fl = self._target_flood(k)
            if fl is None:
                continue
            d = fl[0][nodes]
            j = int(np.argmin(d))
            if np.isfinite(d[j]):
                out.append((d[j] * net.F / MM, k, nodes[j]))
        best = {}
        for d, k, n in sorted(out):
            t = self.targets[k]
            best.setdefault((t["base"], tuple(t["letters"]), t["via"]), (d, k, n))
        return sorted(best.values())

    def flow(self, k, node):
        """+1 = the instrument's line flows towards target k, -1 = away from it, 0 = unknown; and the path facts.
        Arrows on the path itself count first; else the nearest arrow on the lines within flow_reach_mm."""
        net = self.network()
        d, pred = self._target_flood(k)
        p = net.path(pred, node)
        pts = np.array([net.xy(n) for n in p])
        on_pipe = sum(net.thick_near[net.ys[n], net.xs[n]] for n in p) * net.F / MM
        votes = net.arrows_on(pts)
        near = []
        if not votes:
            reach = self.s.profile["flow_reach_mm"] * MM / net.F
            dl = dijkstra(net.G, directed=False, indices=[node], limit=reach).ravel()
            near = net.arrows_near(dl, d)
        info = dict(on_pipe_mm=on_pipe, length_mm=len(p) * net.F / MM, arrows=votes, near_arrows=near)
        v = votes or near
        if not v:
            return 0, info
        sgn = sum(u[1] for u in v) if votes else sum(u[1] / (1 + u[0]) for u in v)
        return (1 if sgn > 0 else -1 if sgn < 0 else 0), info
