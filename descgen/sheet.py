"""What a sheet says, read once per sheet: style profile, title block (equipment name and the system it shows),
notes panel, labelled machine parts, and the enclosed shape at any point. The equipment register (plant.Plant) and the
pipe graph (lines.Pipes) are built on first use: they are the slow part and many sheets need only some instruments.
OCR on these drawings returns words run together ("FLASHGASCOMPRESSOR"): text is matched without spaces and split
into words (text.split_words) only where a readable phrase is needed."""
import re
import numpy as np
import cv2
from . import style as STYLE
from .rules import MACHINE_RULES, KIND_OF_NOUN, SYSTEM_TITLE_WORDS, EQUIPMENT_NOUNS
from .text import split_words, equipment_phrase, system_of, word_cost

MM = 300 / 25.4
INK = 160            # drawn lines are black; light-grey table rulings (~195) are ignored


def ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def merge_overlap(a, b):
    """Two OCR reads of one text line that overlap ('...COMPRESSORSUCTIONSCRU' + 'SUCTIONSCRUBBER'): join them on the
    longest suffix/prefix overlap (one character may differ)."""
    for k in range(min(len(a), len(b)), 2, -1):
        if sum(x != y for x, y in zip(a[-k:], b[:k])) <= (1 if k >= 6 else 0):
            return a[:-k] + b if len(b) >= len(a) else a + b[k:]
    return a + b


class Sheet:
    TEXT_REPAIR = True       # re-read OCR lines cut by the OCR tiles and data-box name lines (off: as read)

    def __init__(self, img, frags):
        self.img = img
        self.gray = np.array(img.convert("L"))
        self.ink = (self.gray < INK).astype(np.uint8)
        if img.mode == "RGB":   # coloured mark-up (review clouds, highlight circles) is not part of the drawing
            rgb = np.asarray(img)
            sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
            self.ink[sat > 90] = 0
        self.frags = [dict(text=f["text"].replace(" ", "").upper(), bbox=f["bbox"], source=f.get("source", "ocr"))
                      for f in frags]
        self.tile_cuts_fixed = 0
        if Sheet.TEXT_REPAIR:
            self._repair_tile_cuts()
        self.profile = STYLE.select(self.frags)
        self.title = self._title_block()
        self.notes = self._notes_panel()
        self.parts = self._machine_parts()
        hdr = [f for f in self.frags if f["text"] in self.profile["panel_labels"]]
        # the notes / title panel down the right: its text is not drawing (no connectors, symbols there)
        self.panel_x = min(f["bbox"][0] for f in hdr) - 4 * MM if hdr else 1e9
        self._plant = self._pipes = None
        self._light = None

    @property
    def plant(self):
        """Equipment register: data boxes, symbols, connectors, ownership along lines (plant.py)."""
        if self._plant is None:
            from .plant import Plant
            self._plant = Plant(self)
        return self._plant

    def pipes(self, light=False):
        """Pipe graph (lines.py). light=True: every solid line, for utility lines drawn with the light pen."""
        from .lines import Pipes
        if light:
            if self._light is None:
                self._light = Pipes(self, self.plant, single_pen=True)
            return self._light
        if self._pipes is None:
            self._pipes = Pipes(self, self.plant)
        return self._pipes

    # ---------- OCR lines cut by the full-page OCR tiles ----------
    def _ocr_engine(self):
        if not hasattr(Sheet, "_ocr"):
            import logging
            from rapidocr import RapidOCR
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            Sheet._ocr = RapidOCR()
        return Sheet._ocr

    def _repair_tile_cuts(self):
        """Full-page OCR reads 1600 px tiles overlapping by 320 px: a text line longer than the overlap that crosses a
        tile boundary is never seen whole, and each tile returns a piece missing the letters beyond its edge
        ('FUEL GAS COMPRESSOR' -> 'ELGASCOMPRESSOR', starting exactly on a tile edge). Every OCR line whose box ends on
        an inner tile edge is read again from a crop spanning its whole ink run; the new reading replaces it when it
        extends the old one (the old text, give or take a character, is part of it), and the leftover pieces of the
        same line inside the new box are dropped. Exact PDF text is never cut and is left alone."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "batch"))
        from tag_pipeline import tile_starts, OCR_TILE, OCR_OVERLAP
        H, W = self.gray.shape
        edges = {"x": set(), "y": set()}
        for ax, L in (("x", W), ("y", H)):
            for s in tile_starts(L, OCR_TILE, OCR_OVERLAP):
                if s > 0:
                    edges[ax].add(s)
                if s + OCR_TILE < L:
                    edges[ax].add(s + OCR_TILE)
        near = lambda v, ax: any(abs(v - e) <= 6 for e in edges[ax])
        fixed = []
        for f in self.frags:
            if f["source"] == "pdf" or len(f["text"]) < 4:
                continue
            x0, y0, x1, y1 = f["bbox"]
            tall = (y1 - y0) > 1.5 * (x1 - x0)
            cut = (near(y0, "y") or near(y1, "y")) if tall else (near(x0, "x") or near(x1, "x"))
            if not cut:
                continue
            if tall:
                sides = (near(y0, "y"), near(y1, "y"))
            else:
                sides = (near(x0, "x"), near(x1, "x"))
            new = self._reread_line(f["bbox"], tall, sides, f["text"])
            if new and self._extends(f["text"], new[0]):
                f["text"], f["bbox"] = new
                fixed.append(f)
        uniq = []
        for f in fixed:        # two pieces of one line both re-read to the same line: keep one
            if not any(u["text"] == f["text"] and abs(u["bbox"][0] - f["bbox"][0]) < 20 and abs(u["bbox"][1] - f["bbox"][1]) < 20 for u in uniq):
                uniq.append(f)
        self.frags = [g for g in self.frags if g not in fixed or g in uniq]
        fixed = uniq
        if fixed:
            def inside(g, b):
                gx0, gy0, gx1, gy1 = g["bbox"]
                a = max(1, (gx1 - gx0) * (gy1 - gy0))
                ov = max(0, min(gx1, b[2]) - max(gx0, b[0])) * max(0, min(gy1, b[3]) - max(gy0, b[1]))
                return ov / a > 0.6
            self.frags = [g for g in self.frags if g in fixed or g["source"] == "pdf"
                          or not any(inside(g, f["bbox"]) and len(g["text"]) < len(f["text"]) for f in fixed)]
        self.tile_cuts_fixed = len(fixed)

    def name_line(self, f):
        """Text of a line that names equipment (data-box name line), read again from its whole ink run on both sides
        when that extends it (OCR's line box sometimes leaves out a word: 'GASBOOSTERCOMPRESSOR' for 'LEAN GAS
        BOOSTER COMPRESSOR'). Exact PDF text is returned as is."""
        if f.get("source") == "pdf" or not Sheet.TEXT_REPAIR:
            return f["text"]
        key = tuple(f["bbox"])
        cache = self.__dict__.setdefault("_name_lines", {})
        if key not in cache:
            new = self._reread_line(f["bbox"], (f["bbox"][3] - f["bbox"][1]) > 1.5 * (f["bbox"][2] - f["bbox"][0]),
                                    old=f["text"])
            cache[key] = new[0] if new and self._extends(f["text"], new[0]) else f["text"]
        return cache[key]

    @staticmethod
    def _extends(old, new):
        """new is a longer reading that contains old in one piece, give or take one character in old."""
        from difflib import SequenceMatcher
        if len(new) <= len(old):
            return False
        m = [b for b in SequenceMatcher(None, old, new).get_matching_blocks() if b.size]
        if not m or sum(b.size for b in m) < len(old) - 1:
            return False
        # the old text sits together inside the new one (text added at its ends only), not spread through a longer
        # line ('BOGHEATER' inside 'DUTYBOG431MMBTUHRHEATER' is not an extension)
        return (m[-1].b + m[-1].size) - m[0].b <= len(old) + 2

    def _reread_line(self, b, tall=False, sides=(True, True), old=""):
        """(text, bbox) of the whole text line through box b, extended only on the cut sides (sides = (start, end)):
        the run of ink along the line, where gaps up to 1.5 x the text height are spaces between words, stopping at a
        ruling line (a stroke running past the text height above and below: a table or box border, not a letter);
        read again on its own; of what the crop shows, the text row sharing most characters with `old` is returned.
        None if nothing is read."""
        x0, y0, x1, y1 = map(int, b)
        H, W = self.ink.shape
        if tall:          # vertical line of text: work on the transposed ink
            ink = self.ink.T; x0, y0, x1, y1 = y0, x0, y1, x1; L = H
        else:
            ink = self.ink; L = W
        # OCR line boxes can take in part of the next line: keep the band of ink rows (between blank rows) that holds
        # the box centre, so the crop shows one text line
        prof = ink[max(0, y0 - (y1 - y0) // 2):y1 + (y1 - y0) // 2, x0:x1].any(axis=1)
        off = max(0, y0 - (y1 - y0) // 2)
        c = (y0 + y1) // 2 - off
        if 0 <= c < len(prof) and prof[c]:
            a_, z_ = c, c
            while a_ > 0 and prof[a_ - 1]:
                a_ -= 1
            while z_ < len(prof) - 1 and prof[z_ + 1]:
                z_ += 1
            if z_ - a_ >= 0.4 * (y1 - y0):
                y0, y1 = off + a_, off + z_ + 1
        h = max(4, y1 - y0)
        band = ink[y0:y1]
        tall_band = ink[max(0, int(y0 - 0.4 * h)):int(y1 + 0.4 * h)]
        def run(x, step):
            gap = 0
            while 0 < x < L - 1 and gap <= 1.5 * h and abs(x - (x0 if step < 0 else x1)) < 150 * MM:
                if tall_band[:, x].all():
                    break
                gap = 0 if band[:, x].any() else gap + 1
                x += step
            return x - step * gap
        a = run(x0, -1) if sides[0] else x0
        z = run(x1, 1) if sides[1] else x1
        pad = int(0.35 * h)
        if tall:
            box = (y0, a, y1, z)                                      # the line's own extent (stored)
            crop = self.img.crop((y0 - pad, a - pad, y1 + pad, z + pad)).rotate(90, expand=True)   # as the OCR's 90 pass
        else:
            box = (a, y0, z, y1)
            crop = self.img.crop((a - pad, y0 - pad, z + pad, y1 + pad))
        try:
            r = self._ocr_engine()(np.array(crop.convert("RGB")))
        except Exception:
            return None
        if r is None or not r.txts:
            return None
        # one text line only: group the reads into rows, keep the row that shares most characters with the old read
        rows = []
        for q, t in sorted(zip(r.boxes, r.txts), key=lambda p: p[0][:, 1].mean()):
            cy, hh = q[:, 1].mean(), np.ptp(q[:, 1])
            if rows and abs(cy - rows[-1]["cy"]) < 0.5 * max(hh, rows[-1]["h"]):
                rows[-1]["parts"].append((q, t))
            else:
                rows.append(dict(cy=cy, h=hh, parts=[(q, t)]))
        from difflib import SequenceMatcher
        def text(row):
            # a sliver of the next line cut by the crop edge is much shorter than the row's own text
            mid = float(np.median([np.ptp(q[:, 1]) for q, _ in row["parts"]]))
            parts = [p for p in row["parts"] if np.ptp(p[0][:, 1]) >= 0.6 * mid]
            return re.sub(r"[^A-Z0-9&-]", "", "".join(t for _, t in sorted(parts, key=lambda p: p[0][:, 0].min())).upper())
        cands = [text(rw) for rw in rows]
        txt = max(cands, key=lambda c: sum(b.size for b in SequenceMatcher(None, old, c).get_matching_blocks())) if cands else ""
        return (txt, [int(v) for v in box]) if txt else None

    # ---------- title block: the line naming the equipment, and the system it shows ----------
    def _title_block(self):
        labels = self.profile["title_labels"]
        t = [f for f in self.frags if f["text"] in labels]
        if not t:
            return dict(equipment=None, lines=[], system=None, system_line=None, tag=None, instance=None)
        tb = t[0]["bbox"]
        lines = sorted((f for f in self.frags if tb[1] - 20 < f["bbox"][1] < tb[1] + self.profile["title_depth_mm"] * MM
                        and f["bbox"][0] > tb[0] - 10 and f["text"] not in labels), key=lambda f: f["bbox"][0])
        merged = []                       # one text line read twice by OCR (overlapping boxes): join the two reads
        for f in lines:
            h = f["bbox"][3] - f["bbox"][1]
            m = next((g for g in merged if abs(g["bbox"][1] - f["bbox"][1]) < 0.5 * h and g["bbox"][0] < f["bbox"][0] < g["bbox"][2]), None)
            if m:
                m["text"] = merge_overlap(m["text"], f["text"]); m["bbox"] = [m["bbox"][0], m["bbox"][1], max(m["bbox"][2], f["bbox"][2]), m["bbox"][3]]
            else:
                merged.append(dict(text=f["text"], bbox=list(f["bbox"])))
        lines = [self._reread(f, tb) for f in sorted(merged, key=lambda f: f["bbox"][1])]
        equipment, eq_tag, instance = None, None, None
        rx = re.compile(self.profile["equipment_tag"])
        for k, ln in enumerate(lines):
            # the drawing's own equipment tag printed in the title ('1P-1101AAGRUCHARGEPUMP'): taken out before the
            # words are split; its instance letter is kept when the sheet prints that tag with the letter elsewhere
            m = rx.search(ln)
            if m:
                tag = m.group(0); nxt = ln[m.end():m.end() + 1]
                rest = ln[m.end():]
                # an instance letter ('1P-1101A') or the first letter of the name? read the title both ways and keep
                # the reading with fewer, more common words ('A' + 'AGRU CHARGE PUMP' beats 'AAGRU ...';
                # 'FLASH GAS COMPRESSOR' beats 'F' + 'LASH GAS ...')
                if nxt.isalpha() and word_cost(split_words(rest[1:])) < word_cost(split_words(rest)):
                    tag += nxt
                    instance = nxt
                eq_tag = tag
                lines[k] = (ln[:m.start()] + " " + ln[m.start() + len(tag):]).strip()
        for ln in lines:
            # keep everything up to the last equipment noun: "FLASH GAS COMPRESSOR [MONITORING CIRCUIT]";
            # a title naming the system ("LNG PRODUCT PUMPS") names its machines in the plural
            equipment = equipment_phrase(split_words(ln))
            if equipment:
                break
        system, sline = None, None
        for ln in lines:      # 'FLASHGASCOMPRESSORLUBEOILCIRCUIT': the system named after the machine
            ends = [ln.rfind(n) + len(n) for n in EQUIPMENT_NOUNS if n in ln]
            rest = ln[max(ends):] if ends else ln
            if any(w in rest for w in SYSTEM_TITLE_WORDS):
                sm = system_of(rest)
                if sm:
                    system, sline = sm[0], ln
                    break
        return dict(equipment=equipment, lines=lines, system=system, system_line=sline, tag=eq_tag, instance=instance)

    def _reread(self, f, tb):
        """Title lines are centred in the title column and the full-page OCR sometimes drops their first letters
        ('ASHGASCOMPRESSOR'): read the line again from a crop spanning the whole column, keep the longer reading."""
        b = f["bbox"]
        try:
            from rapidocr import RapidOCR
            if not hasattr(Sheet, "_ocr"):
                import logging; logging.getLogger("RapidOCR").setLevel(logging.ERROR)
                Sheet._ocr = RapidOCR()
            half = max(b[2] - ctr(b)[0], 0) + 15 * MM
            x0, x1 = int(max(tb[0], ctr(b)[0] - half)), int(ctr(b)[0] + half)
            crop = self.img.crop((x0, int(b[1] - 1.5 * MM), x1, int(b[3] + 1.5 * MM))).convert("RGB")
            r = Sheet._ocr(np.array(crop))
            txt = "".join(r.txts or ()).replace(" ", "").upper() if r is not None and r.txts else ""
        except Exception:
            return f["text"]
        # restores a few first letters only: a longer gain means the crop took in a neighbouring title line
        return txt if self._extends(f["text"], txt) and len(txt) - len(f["text"]) <= 4 else f["text"]

    # ---------- notes panel: {"4.1"-style id as printed without dots: "41": "COMPRESSORVALVES"} ----------
    def _notes_panel(self):
        hdr = [f for f in self.frags if f["text"] in self.profile["notes_labels"]]
        if not hdr:
            return {}
        hx, hy = hdr[0]["bbox"][:2]
        lines = sorted((f for f in self.frags if abs(f["bbox"][0] - hx) < 8 * MM and hy < f["bbox"][1] < hy + 250 * MM),
                       key=lambda f: f["bbox"][1])
        notes, cur = {}, None
        for f in lines:
            m = re.match(r"^(\d+)([A-Z].*)$", f["text"])
            if m:
                cur = m.group(1); notes[cur] = m.group(2)
            elif cur and f["bbox"][0] - hx > 0.5 * MM:      # indented continuation line
                notes[cur] += f["text"]
        return notes

    def note_text(self, ref):
        """'NOTE41' -> text of note 4.1 (printed '41...' or as sub-item '1.4.1' -> '141'); 'NOTE68' -> notes 6 and 8."""
        out = []
        for r in [ref] + (list(ref) if len(ref) > 1 and ref not in self.notes else []):
            hits = [t for k, t in self.notes.items() if k == r] or \
                   [t for k, t in self.notes.items() if k.endswith(r) and len(k) == len(r) + 1]   # '11' printed as '111'
            if hits:
                out.append(hits[0])
                if r == ref:
                    break
        return out

    # ---------- enclosed shapes ----------
    def region_at(self, x, y, win_mm=120):
        """The closed white area containing (x, y), as a bbox; None if it is open drawing space.
        White areas sharing the same top and bottom edges (a cylinder split by its piston) are merged."""
        h, w = self.ink.shape
        r = int(win_mm * MM)
        x0, y0, x1, y1 = max(0, int(x) - r), max(0, int(y) - r), min(w, int(x) + r), min(h, int(y) + r)
        win = 1 - self.ink[y0:y1, x0:x1]
        px, py = int(x) - x0, int(y) - y0
        if not (0 <= px < win.shape[1] and 0 <= py < win.shape[0]) or not win[py, px]:
            return None
        n, lab, st, _ = cv2.connectedComponentsWithStats(win.astype(np.uint8), connectivity=4)
        k = lab[py, px]
        def closed(i):
            bx, by, bw, bh = st[i, :4]
            return bx > 0 and by > 0 and bx + bw < win.shape[1] and by + bh < win.shape[0]
        if not closed(k):
            return None
        bx, by, bw, bh = st[k, :4]
        keep = [k]
        tol = 2 * MM
        for i in range(1, n):
            if i != k and closed(i):
                ix, iy, iw, ih = st[i, :4]
                if abs(iy - by) < tol and abs(iy + ih - by - bh) < tol and (ix <= bx + bw + 3 * MM and ix + iw >= bx - 3 * MM):
                    keep.append(i)
        xs = [st[i, 0] for i in keep] + [st[i, 0] + st[i, 2] for i in keep]
        ys = [st[i, 1] for i in keep] + [st[i, 1] + st[i, 3] for i in keep]
        return (x0 + min(xs), y0 + min(ys), x0 + max(xs), y0 + max(ys))

    def region_beside(self, b, side, gap_mm=2.5):
        """Closed shape just above / below / inside a label's bbox (labels sit next to or inside the shape they name)."""
        cx, cy = ctr(b)
        for d in np.arange(1.0, 25, 1.0) * MM:
            p = {"above": (cx, b[1] - d), "below": (cx, b[3] + d), "inside": (cx, b[1] - gap_mm * MM)}[side]
            reg = self.region_at(*p)
            if reg and (reg[2] - reg[0]) > 8 * MM and (reg[3] - reg[1]) > 5 * MM:
                return reg
            if side == "inside":
                break
        return None

    def box_around(self, b):
        """Smallest closed shape enclosing a label drawn inside it: flood the white just outside the text."""
        cx, cy = ctr(b)
        for dy in (b[1] - 1.5 * MM, b[3] + 1.5 * MM):
            reg = self.region_at(cx, dy, win_mm=200)
            if reg and reg[0] < b[0] and reg[2] > b[2]:
                return reg
        return None

    # ---------- labelled machine parts ----------
    def _machine_parts(self):
        """Cylinders (with stage), a rotating machine drawn with its shaft-end labels (end and winding labels from its
        rule set), flywheel."""
        parts = []
        for f in self.frags:
            t, b = f["text"], f["bbox"]
            if t == "CYLINDER":
                st = [g for g in self.frags if re.fullmatch(r"(\d)(ST|ND|RD|TH)STAGE", g["text"])
                      and abs(ctr(g["bbox"])[0] - ctr(b)[0]) < 15 * MM and 0 < b[1] - g["bbox"][1] < 15 * MM]
                stage = int(st[0]["text"][0]) if st else None
                top = st[0]["bbox"] if st else b
                reg = self.region_beside(top, "above")
                if reg:
                    reg = self._outline_extent(reg)
                    parts.append(dict(kind="cylinder", stage=stage, bbox=reg, label=f"{stage or ''} STAGE CYLINDER".strip()))
            elif t in ("FLYWHEEL",):
                reg = self.region_at(*ctr(b)) or self.region_beside(b, "above")
                reg = reg or self.box_around(b)      # the label is drawn inside the box
                if reg:
                    parts.append(dict(kind="flywheel", stage=None, bbox=reg, label="FLYWHEEL"))
        end_words = {k: v for r in MACHINE_RULES.values() for k, v in r.get("ends", {}).items()}
        de = [f for f in self.frags if f["text"] in end_words]
        if de:
            box = self.box_around(de[0]["bbox"])
            if box:
                ends = {end_words[f["text"]]: ctr(f["bbox"]) for f in de
                        if box[0] <= ctr(f["bbox"])[0] <= box[2] and box[1] - 5 * MM <= ctr(f["bbox"])[1] <= box[3] + 5 * MM}
                # the machine type from a noun printed inside it; a shaft drawn with end labels and nothing else: motor
                inside = [f["text"] for f in self.frags if box[0] <= ctr(f["bbox"])[0] <= box[2] and box[1] <= ctr(f["bbox"])[1] <= box[3]]
                kind = next((KIND_OF_NOUN[n] for t in inside for n in sorted(KIND_OF_NOUN, key=len, reverse=True)
                             if n in t and "ends" in MACHINE_RULES.get(KIND_OF_NOUN[n], {})), "motor")
                labels = MACHINE_RULES[kind].get("phases", {})
                phases = {f["text"]: ctr(f["bbox"]) for f in self.frags if f["text"] in labels
                          and box[0] - 2 * MM <= f["bbox"][0] <= box[2] and box[1] <= ctr(f["bbox"])[1] <= box[3]}
                order = list(labels)
                if len(order) == 3 and order[0] in phases and order[2] in phases and order[1] not in phases:
                    # the middle winding label is often unread: midway between the other two
                    phases[order[1]] = tuple((np.array(phases[order[0]]) + np.array(phases[order[2]])) / 2)
                parts.append(dict(kind=kind, stage=None, bbox=box, label=kind.upper(), ends=ends, phases=phases))
        return parts

    def _outline_extent(self, reg):
        """A shape split inside (cylinder chambers either side of the piston): follow its top and bottom outline rows
        sideways to the corners, so the box covers the whole outline."""
        x0, y0, x1, y1 = map(int, reg)
        def row(yc):
            step = -1 if yc <= y0 else 1
            for y in range(yc, yc + step * int(1.5 * MM), step):
                if self.ink[y, (x0 + x1) // 2]:
                    return y
            return None
        def run(y, x, step):
            gap = 0
            while 0 < x < self.ink.shape[1] - 1 and gap <= 3:
                gap = 0 if self.ink[max(0, y - 1):y + 2, x].any() else gap + 1
                x += step
            return x - step * 4
        top, bot = row(y0 - 1), row(y1 + 1)
        if top is None or bot is None:
            return reg
        left = max(run(top, x0, -1), run(bot, x0, -1))     # the shorter of the two edges ends at the true corner
        right = min(run(top, x1, 1), run(bot, x1, 1))
        left = min(left, x0) if abs(run(top, x0, -1) - run(bot, x0, -1)) < 3 * MM else x0
        right = max(right, x1) if abs(run(top, x1, 1) - run(bot, x1, 1)) < 3 * MM else x1
        return (left, top, right, bot)

    def part_at(self, x, y):
        """Labelled part whose shape contains the point (2 mm tolerance for outlines)."""
        best = None
        for p in self.parts:
            x0, y0, x1, y1 = p["bbox"]
            if x0 - 2 * MM <= x <= x1 + 2 * MM and y0 - 2 * MM <= y <= y1 + 2 * MM:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best[0]:
                    best = (area, p)
        return best[1] if best else None

    def near_text(self, pts, mm, pattern):
        """Fragments matching `pattern` within `mm` of any of the points, nearest first."""
        out = []
        if not pts:
            return out
        for f in self.frags:
            if re.fullmatch(pattern, f["text"]):
                fx, fy = ctr(f["bbox"])
                d = min(np.hypot(fx - x, fy - y) for x, y in pts)
                if d < mm * MM:
                    out.append((d / MM, f))
        return [f for _, f in sorted(out, key=lambda t: t[0])]
