"""One tag -> its description. Each word group has its own finder, every finder records where its answer came from
and how sure it is (value, source, confidence); the composer only orders and abbreviates them (style profile).

    equipment    : title block, or the equipment the instrument belongs to         -> "Flash Gas Comp", "LNG Product Pump"
    instance     : the owning equipment's instance letter                          -> "A"
    machine      : the machine shape the bubble's line touches                     -> "Motor"
    machine_item / system / item / position : auxiliary-system sheets             -> "Stg 1 Cylinder" "CW" "Pumps" "Discharge"
    stage        : cylinder label / nearest cylinder / pipe position on a machine  -> "Stg 2"
    end          : where the line lands on the shape                               -> "HE" / "CE" / "DE" / "NDE"
    stream       : suction / discharge / inlet / outlet of the owning equipment    -> "Suction"
    part         : the numbered note next to the line, the shape itself, a header  -> "Valve", "Frame", "Header"
    variable     : instrument letters (+ note for X / S)                           -> "Temp", "Rod Drop"
    index        : A / B suffix of a doubled instrument, or loops numbered on one machine -> "1", "2"

Which context applies is decided per instrument from what its line reaches, not from the sheet's identity:
  1. its line lands on a labelled machine part (cylinder, motor, flywheel, frame)  -> machine-part finders;
  2. else, the title names an auxiliary system of the machine                     -> auxsys.aux_slots;
  3. else, the pipe it taps leads to equipment that claims it                     -> onpipe.pipe_slots;
  4. else, the equipment nearest along the drawn lines (loop members via their measuring point) -> onpipe.ownership."""
import math
import re
import numpy as np
from .sheet import MM, ctr
from .text import ninja
from .trace import trace_bubble
from . import rules as R
from .style import abbreviate, name_case
from .auxsys import aux_slots
from .onpipe import pipe_slots, ownership, numbered_index


def slot(value, source, conf):
    return dict(value=value, source=source, conf=conf)


def parse_tag(tag):
    m = re.match(r"^(?:(\d{1,3})-)*([A-Z]{1,6})-(\d{3,5})([A-Z]?)(?:-(\d+))?$", tag)
    return dict(unit=m.group(1), letters=m.group(2), number=m.group(3), suffix=m.group(4)) if m else None


def leader_of(sheet, bbox):
    """The bubble's solid line: the exit that runs into a shape, else the longest one."""
    lines = [l for l in trace_bubble(sheet.gray, bbox, sheet.ink) if l["length_mm"] >= 2]
    if not lines:
        return None
    inner = min(l["ring_mm"] for l in lines)          # rings further out only when the first ring found no line
    lines = [l for l in lines if l["ring_mm"] == inner]
    return max(lines, key=lambda l: (l["end"] == "object", l["length_mm"]))


def heading(pts):
    (xa, ya), (xb, yb) = pts[max(0, len(pts) - 4)], pts[-1]
    return math.atan2(yb - ya, xb - xa)


def rod_side(sheet, box):
    """Side of a cylinder where the piston rod enters (= crank end): more lines cross a probe just outside that side."""
    x0, y0, x1, y1 = map(int, box)
    ys = range(int(y0 + 0.2 * (y1 - y0)), int(y1 - 0.2 * (y1 - y0)))
    def crossings(x):
        col = sheet.ink[list(ys), x]
        return int(np.sum((col[1:] == 1) & (col[:-1] == 0)))
    left, right = crossings(max(0, x0 - int(3 * MM))), crossings(min(sheet.ink.shape[1] - 1, x1 + int(3 * MM)))
    return ("left" if left > right else "right"), (left, right)


# ---------------- finders ----------------
def note_finder(sheet, pts):
    """Numbered notes beside the bubble or along its line: {'part' | 'measure': value}, sources."""
    hits, src = {}, []
    near = sorted(sheet.near_text(pts[:1], 18, r"NOTE\d+") + sheet.near_text(pts[1:], 12, r"NOTE\d+"),
                  key=lambda f: min(math.hypot(ctr(f["bbox"])[0] - x, ctr(f["bbox"])[1] - y) for x, y in pts))
    def texts(ref):
        out = []
        for t in sheet.note_text(ref):
            for ph in R.NOTE_IGNORE:
                t = t.replace(ph + "S", "").replace(ph, "")
            out.append(t)
        return out
    useful = [f for f in near if any(k in t for t in texts(f["text"][4:]) for k, _, _ in R.NOTE_WORDS)]
    for f in useful[:1]:                                # the nearest note label that says something about the instrument
        ref = f["text"][4:]
        for txt in texts(ref):
            for key, kind, val in R.NOTE_WORDS:
                if key in txt and kind not in hits:
                    hits[kind] = val; src.append(f"NOTE {ref}: {' '.join(ninja(txt))[:60]}")
    return hits, src


def variable_finder(L, note_hits, note_src):
    first = L[0]
    if L[:2] in R.MODIFIED:
        return slot(R.MODIFIED[L[:2]], f"letters {L} ({L[:2]} = {R.MODIFIED[L[:2]].lower()})", 0.95)
    if first == "X" and "measure" in note_hits:
        return slot(note_hits["measure"], f"letters {L} (X = 'see note'); " + note_src[0], 0.85)
    if first == "S" and note_hits.get("measure") == "Keyphasor":
        return slot("Keyphasor Speed", f"letters {L} (S = speed); " + note_src[0], 0.85)
    if first in R.VARIABLE and first not in "HKG":
        return slot(R.VARIABLE[first], f"letters {L} ({first} = {R.VARIABLE[first].lower()})", 0.95)
    fn = next((v for k, v in R.FUNCTION_ONLY.items() if L.startswith(k)), None)
    return slot(fn, f"letters {L}", 0.5) if fn else None


def machine_part_finder(sheet, lead, bx, by, first, note_hits, note_src, s):
    """Stage / end / stream / part when the bubble's line touches a labelled machine part. Returns True if it did."""
    end_pt = lead["pts"][-1] if lead else None
    part, obj, at_bearing = None, None, False
    if end_pt:
        h = heading(lead["pts"])
        probe = [(end_pt[0] + d * MM * math.cos(h), end_pt[1] + d * MM * math.sin(h)) for d in (0, 1.5, 3)]
        part = next((sheet.part_at(*q) for q in probe if sheet.part_at(*q)), None)
        if not part:   # the line may stop on the shaft bearing symbol just outside a machine's end
            for q in sheet.parts:
                x0, y0, x1, y1 = q["bbox"]
                if "ends" in q and x0 <= end_pt[0] <= x1 and (y0 - 10 * MM <= end_pt[1] < y0 or y1 < end_pt[1] <= y1 + 10 * MM):
                    part, at_bearing = q, True
        if not part:
            obj = next((sheet.region_at(*q) for q in probe if sheet.region_at(*q)), None)
    stage_fmt = sheet.profile["stage"]
    cylinders = [q for q in sheet.parts if q["kind"] == "cylinder"]
    def nearest_cylinder():
        if len(cylinders) < 2:
            return None
        d = sorted((math.hypot(bx - ctr(c["bbox"])[0], by - ctr(c["bbox"])[1]), c) for c in cylinders)
        return d[0][1] if d[1][0] - d[0][0] > 5 * MM else None
    note_part = note_hits.get("part")

    if part and part["kind"] == "cylinder":
        R_ = R.MACHINE_RULES["compressor"]           # a reciprocating cylinder: compressor rule set
        x0, y0, x1, y1 = part["bbox"]
        s["stage"] = slot(stage_fmt.format(n=part["stage"]), f"line touches '{part['label']}'", 0.9)
        side, cnt = rod_side(sheet, part["bbox"])
        crank_x = x0 if side == "left" else x1
        other_x = x1 if side == "left" else x0
        head, crank = R_["cylinder_ends"]
        rod = abs(end_pt[0] - crank_x) < abs(end_pt[0] - other_x)
        s["end"] = slot(crank if rod else head, f"line lands nearer the {'rod' if rod else 'outer'} end of the cylinder "
                                                f"(rod enters on the {side}: {cnt[0]} vs {cnt[1]} lines)", 0.8)
        if note_part == "Valve":
            top = end_pt[1] < (y0 + y1) / 2
            s["stream"] = slot(R_["valve_top"] if top else R_["valve_bottom"],
                               f"line lands on the cylinder {'top' if top else 'bottom'} "
                               f"(rule set: {R_['valve_top'].lower()} valves on top)", 0.7)
        if note_part:
            s["part"] = slot(note_part, note_src[0], 0.85)
        return True
    if part and "ends" in part:
        R_ = R.MACHINE_RULES[part["kind"]]
        s["machine"] = slot(part["kind"].title(), f"line touches the {part['kind']} outline (shaft-end labels)", 0.9)
        ph = sorted((math.hypot(end_pt[0] - x, end_pt[1] - y), k) for k, (x, y) in part["phases"].items())
        if ph and ph[0][0] < 12 * MM and first == "T":
            s["part"] = slot(R_["phases"][ph[0][1]], f"line lands beside winding label '{ph[0][1]}' ({ph[0][0] / MM:.0f} mm)", 0.75)
        elif part["ends"]:
            de = sorted((math.hypot(end_pt[0] - x, end_pt[1] - y), k) for k, (x, y) in part["ends"].items())
            s["end"] = slot(de[0][1], f"line lands nearest the '{de[0][1]}' end label", 0.8)
            if at_bearing:
                s["part"] = slot("Bearing", "line ends on the shaft symbol (boxed X) just outside that end of the machine", 0.6)
        return True
    if part and part["kind"] == "flywheel":
        s["part"] = slot("Flywheel", "line touches 'FLYWHEEL'", 0.8)
        return True
    # unlabelled shape or a rod line between the cylinders: the compressor frame / running gear
    cyl = nearest_cylinder()
    between = len(cylinders) >= 2 and min(ctr(c["bbox"])[0] for c in cylinders) < (end_pt or (bx, by))[0] < max(ctr(c["bbox"])[0] for c in cylinders)
    if note_part or (s.get("variable", {}).get("value") == "Rod Drop"):
        if cyl:
            s["stage"] = slot(stage_fmt.format(n=cyl["stage"]), f"nearest cylinder to the bubble: '{cyl['label']}'", 0.7)
        if note_part:
            s["part"] = slot(note_part, note_src[0], 0.85)
        return "stage" in s
    if obj and between:
        s["part"] = slot("Frame", "line touches the unlabelled box between the cylinders", 0.6)
        motor = next((q for q in sheet.parts if "ends" in q), None)
        if motor:
            fc = np.array(ctr(obj)); mc = np.array(ctr(motor["bbox"]))
            toward = np.dot(np.array(end_pt) - fc, mc - fc) > 0
            s["end"] = slot("DE" if toward else "NDE", f"line lands on the {'motor' if toward else 'far'} side of the frame", 0.65)
        return True
    return False


# ---------------- one tag ----------------
def describe_tag(sheet, tag, bbox, siblings):
    p = parse_tag(tag)
    if not p:
        return dict(tag=tag, description="", slots={}, leader=None, confidence=0, review="tag not parsed")
    s = {}
    L = p["letters"]
    bx, by = ctr(bbox)

    # equipment: title block (replaced below by the equipment the instrument is connected to, when found)
    if sheet.title["equipment"]:
        s["equipment"] = slot(name_case(sheet.title["equipment"]), "title block: " + " / ".join(sheet.title["lines"][-2:]), 0.9)
        if sheet.title.get("instance"):
            s["instance"] = slot(sheet.title["instance"], f"title block equipment tag {sheet.title['tag']}", 0.85)

    lead = leader_of(sheet, bbox)
    pts = [(bx, by)] + (lead["pts"] if lead else [])
    note_hits, note_src = note_finder(sheet, pts)
    var = variable_finder(L, note_hits, note_src)
    if var:
        s["variable"] = var

    # 1. machine part touched by the line
    on_machine = machine_part_finder(sheet, lead, bx, by, L[0], note_hits, note_src, s)
    if not on_machine:
        if sheet.title["system"] and not {"part", "machine"} & set(s):
            # 2. auxiliary system of the machine (the title says which)
            aux_slots(sheet, bbox, L, lead, s)
        else:
            # 3. the pipe it taps; 4. the equipment nearest along the lines
            placed = pipe_slots(sheet, L, lead, bx, by, s) if lead else False
            if not placed:
                owner = ownership(sheet, bbox, p, s)
                if owner:
                    numbered_index(sheet, p, s, owner, siblings, bbox)

    # index: doubled instrument (A / B suffix with a sibling)
    if p["suffix"] and siblings.get((L, p["number"]), 0) > 1:
        s["index"] = slot(str(ord(p["suffix"]) - 64), f"suffix {p['suffix']} of a doubled instrument", 0.8)

    if sheet.profile.get("name_interstage_by_stage") and STAGE_NAMES:
        stage_positions(sheet, s)
    if L.startswith("PD") and DP_INLINE and not {"part", "item"} & set(s):
        dp_inline(sheet, bbox, s, tag)
    return dict(tag=tag, slots=s, leader=lead, description=compose(s, sheet.profile),
                confidence=round(min((v["conf"] for v in s.values()), default=0), 2),
                review=review_reason(s, lead))


STAGE_NAMES = True       # switch for stage_positions (ablation)
DP_INLINE = True         # switch for dp_inline (ablation)


def _pipe_path(G, a, b, max_mm=300):
    """Shortest path along the pipe graph between two tap points (seg, pos); its edges as (cost px, kind) with kind
    'along' (inside one segment) or 'link' (between segments: junction, collinear gap or symbol hop). None if far."""
    import heapq
    st = {k: set(v) for k, v in G.st.items()}
    st[a[0]].add(a[1]); st[b[0]].add(b[1])
    heap, seen = [(0.0, a, None, None)], {}
    while heap:
        d, n, frm, e = heapq.heappop(heap)
        if n in seen or d > max_mm * MM:
            continue
        seen[n] = (frm, e)
        if n == b:
            out = []
            while seen[n][0] is not None:
                out.append(seen[n][1]); n = seen[n][0]
            return out[::-1]
        for nb, c in G._along(n[0], n[1], st):
            heapq.heappush(heap, (d + c, nb, n, (c, "along")))
        for nb, c in G.adj.get(n, []):
            heapq.heappush(heap, (d + c, nb, n, (c, "link")))
    return None


def _root_valve_taps(sheet, G, tag, bbox, near_mm=200, snap_mm=30, per_label=4):
    """Candidate tap pairs from the instrument's root-valve labels (profile 'root_valve_label', e.g. 'HVPDIT9048-1' /
    '-2'): for each label the few nearest pipe segments (the label sits beside its valve, which may be on the impulse
    line or the pipe), as [((seg, pos), (seg, pos)), ...] nearest first; [] unless two labels are found."""
    p = parse_tag(tag)
    if not p:
        return []
    rx = re.compile(sheet.profile["root_valve_label"].format(tag=re.escape(p["letters"] + p["number"] + (p["suffix"] or ""))))
    bx, by = ctr(bbox)
    labs = {}
    for f in sheet.frags:
        m = rx.fullmatch(f["text"])
        if m and math.hypot(ctr(f["bbox"])[0] - bx, ctr(f["bbox"])[1] - by) < near_mm * MM:
            labs.setdefault(m.group(1), f["bbox"])
    if len(labs) < 2:
        return []
    cands = []
    for k in sorted(labs)[:2]:
        cx, cy = ctr(labs[k])
        near = []
        for i, sg in enumerate(G.segs):
            if sg["b"] - sg["a"] < 10 * MM:
                continue
            u, w = (cx, cy) if sg["o"] == "h" else (cy, cx)
            q = min(max(u, sg["a"]), sg["b"])
            d = math.hypot(u - q, w - sg["c"])
            if d <= snap_mm * MM:
                near.append((d, i, q))
        cands.append([(i, q) for _, i, q in sorted(near)[:per_label]])
    return sorted(((a, b) for a in cands[0] for b in cands[1] if a != b),
                  key=lambda ab: cands[0].index(ab[0]) + cands[1].index(ab[1]))


def dp_inline(sheet, bbox, s, tag=None):
    """A DP instrument whose two impulse lines tap one pipe on either side of an in-line item (a symbol interrupting the
    pipe between the taps) measures across that item; on a machine's line the item takes the machine kind's rule
    (rules.MACHINE_RULES 'dp_inline': a strainer on the suction). Taps come from the traced impulse lines, else from
    the instrument's labelled root valves (profile 'root_valve_label')."""
    from .auxsys import taps
    from .text import split_words, kind_of, rules_of
    G = sheet.pipes()
    ls = taps(sheet, bbox)
    pairs = []
    if len(ls) >= 2:
        t1, t2 = G.tap(ls[0]["pts"]), G.tap(ls[1]["pts"])
        if t1 and t2:
            pairs.append(("impulse lines traced", (t1[0], t1[1]), (t2[0], t2[1])))
    if sheet.profile.get("root_valve_label") and tag:
        pairs += [("root valves labelled for the instrument", a, b) for a, b in _root_valve_taps(sheet, G, tag, bbox)]
    path = how = None
    for how_, a, b in pairs:
        pth = _pipe_path(G, a, b)
        # the taps must be joined along the pipe through an in-line item (a gap 2-40 mm the graph bridges)
        if pth and any(k == "link" and 2 * MM <= c < 40 * MM for c, k in pth):
            path, how = pth, how_
            break
    if not path:
        return
    eq = s.get("equipment", {}).get("value", "")
    words = split_words(eq.replace(" ", "").upper())
    kinds = [kind_of([w]) for w in words if kind_of([w])]
    rule = next((rules_of(k).get("dp_inline") for k in kinds if rules_of(k).get("dp_inline")), None)
    if not rule:
        return
    stream, part = rule
    why = f"{how}: two taps on one pipe {sum(c for c, _ in path) / MM:.0f} mm apart with an in-line item between them; " \
          f"untagged item on a {kinds[0]} line = its {stream.lower()} {part.lower()} (rule)"
    s["part"] = slot(part, why, 0.6)
    if stream == "Suction" and re.search(r"\sIntercooler(\s[A-H])?$", s.get("equipment", {}).get("value", "")) \
            and sheet.profile.get("name_interstage_by_stage") and STAGE_NAMES:
        # a suction strainer by an intercooler sits on its outlet, the next stage's suction
        s["stream"] = slot("Outlet", why, 0.55)
        stage_positions(sheet, s)
    if s.get("stream", {}).get("value") != stream:
        s["stream"] = slot(stream, why, 0.55)


def stage_positions(sheet, s):
    """A point on the interstage piping of a multistage machine, named after the cooler it is on ('Fuel Gas
    Compressor Intercooler' + 'Outlet'), renamed after the machine's stage (house style): intercooler k inlet = stage
    k discharge, its outlet = stage k+1 suction; aftercooler inlet = last stage discharge. The number of stages is the
    printed stage count, else one more than the intercoolers the sheet names."""
    eq, st = s.get("equipment", {}).get("value", ""), s.get("stream", {}).get("value", "")
    m = re.match(r"^(.*?)\s+(Intercooler|Aftercooler)(?:\s+([A-H]))?$", eq)
    if not m or st not in ("Inlet", "Outlet") or "stage" in s:
        return
    machine, cooler, inst = m.groups()
    P = sheet.plant
    coolers = sorted({t for t, n in P.names.items() if "INTERCOOLER" in "".join(n["words"])})
    n_inter = max(1, len({P.base_of(t)[0] for t in coolers}))
    n = P.n_stages or n_inter + 1
    if cooler == "Intercooler":
        k, word = (1, "Discharge") if st == "Inlet" else (2, "Suction")
    elif st == "Inlet":
        k, word = n, "Discharge"
    else:
        return
    why = f"{cooler.lower()} {st.lower()} of a {n}-stage machine (house style names interstage points by stage)"
    s["equipment"] = slot(machine, s["equipment"]["source"] + "; " + why, s["equipment"]["conf"])
    if inst and "instance" not in s:
        s["instance"] = slot(inst, s["equipment"]["source"], 0.6)
    s["stage"] = slot(sheet.profile["stage"].format(n=k), why, 0.55)
    s["stream"] = slot(word, why, 0.55)


def describe_sheet(sheet, bubbles, siblings):
    """describe_tag for every tag of a sheet, then the sheet-level steps that need all tags at once."""
    res = {tag: describe_tag(sheet, tag, bbox, siblings) for tag, bbox in sorted(bubbles.items())}
    loop_parts(res, sheet.profile)
    if sheet.profile.get("instance_from_suffix") and INSTANCE_SUFFIX:
        instance_from_suffix(res, sheet)
    number_repeats(res, sheet.profile)
    return res


def loop_parts(res, profile):
    """ISA: one loop number = one measuring point. A loop member without a part (a DCS indicator PDI drawn apart from
    its transmitter, joined by a signal line only) takes part / stream / stage from the loop's measuring instrument
    (element or transmitter, rules.MEASURING_LETTERS) when that one found them across the same measured variable."""
    by_loop = {}
    for tag, r in res.items():
        p = parse_tag(tag)
        if p:
            by_loop.setdefault((p["letters"][0], p["number"], p["suffix"]), []).append((p, r))
    for members in by_loop.values():
        src = [(p, r) for p, r in members if re.search(f"[{R.MEASURING_LETTERS}]", p["letters"][1:]) and "part" in r["slots"]]
        if not src:
            continue
        sp, sr = src[0]
        for p, r in members:
            s = r["slots"]
            if r is sr or "part" in s or s.get("variable", {}).get("value") != sr["slots"].get("variable", {}).get("value"):
                continue
            keys = ["part", "stream", "stage"]
            se, me = sr["slots"].get("equipment", {}).get("value", ""), s.get("equipment", {}).get("value", "")
            if se and me != se and me.startswith(se):      # the measuring instrument's machine, named by stage
                keys.append("equipment")
            for k in keys:
                if k in sr["slots"]:
                    v = sr["slots"][k]
                    s[k] = slot(v["value"], f"from loop member {sp['letters']}-{sp['number']}: {v['source']}", min(v["conf"], 0.55))
            r["description"] = compose(s, profile)


# ---------------- cross-sheet loop register ----------------
LOC_KEYS = ("machine", "machine_item", "system", "stage", "end", "stream", "item", "part", "position")


def build_loop_register(results):
    """results: [(sheet name, {tag: describe result})] for every sheet of a batch. Per loop (first letter, loop number,
    suffix) and measured variable, the location pieces each sheet found for it - where the loop's element is drawn
    (a vendor sheet showing the bearing the TE sits on) is often not where it is displayed (a DCS bubble on the
    client's sheet, joined to the vendor package by a signal line). Taken before any register fill (no chaining)."""
    reg = {}
    for name, res in results:
        for tag, r in res.items():
            p = parse_tag(tag)
            var = r["slots"].get("variable", {}).get("value")
            loc = {k: dict(r["slots"][k]) for k in LOC_KEYS if k in r["slots"]}
            if not p or not var or not loc:
                continue
            for k in ("equipment", "instance"):
                if k in r["slots"]:
                    loc[k] = dict(r["slots"][k])
            reg.setdefault((p["unit"], p["letters"][0], p["number"], p["suffix"], var), []).append(dict(sheet=name, tag=tag, slots=loc))
    return reg


def apply_loop_register(name, res, reg, profile):
    """A tag described with no location at all takes the location pieces of the same loop described on another sheet
    of the batch (the tag itself is the link: same unit, loop number, suffix and measured variable). Only missing pieces are filled, and only
    where the equipment agrees (the tag names none, or the same one); the entry with the most location pieces wins.
    Returns the number of tags changed."""
    n = 0
    for tag, r in res.items():
        p = parse_tag(tag)
        s = r["slots"]
        var = s.get("variable", {}).get("value")
        if not p or not var or set(s) & set(LOC_KEYS):
            continue                   # only tags described with no location at all (a bare display) are filled
        mine = s.get("equipment", {}).get("value", "").lower()
        # a location is relative to its equipment ('Inlet' of the separator): copy it only to a tag naming no equipment
        # (which then takes the equipment too) or naming the same equipment
        cands = [e for e in reg.get((p["unit"], p["letters"][0], p["number"], p["suffix"], var), []) if e["sheet"] != name
                 and set(e["slots"]) & set(LOC_KEYS) - set(s)
                 and "equipment" in e["slots"] and (not mine or e["slots"]["equipment"]["value"].lower() == mine)]
        if not cands:
            continue
        e = max(cands, key=lambda e: (len(set(e["slots"]) & set(LOC_KEYS)), min(v["conf"] for v in e["slots"].values())))
        filled = False
        for k, v in e["slots"].items():
            if k in s or (k in ("equipment", "instance") and "equipment" in s):
                continue
            s[k] = slot(v["value"], f"loop {e['tag']} on sheet {e['sheet']}: {v['source']}", min(v["conf"], 0.5))
            filled = True
        if filled:
            r["description"] = compose(s, profile)
            n += 1
    return n


INSTANCE_SUFFIX = True     # switch for instance_from_suffix (ablation)
DOMINANT_SHARE = 0.6       # share of suffixed instrument tags with one letter that makes a sheet that instance's sheet


def instance_from_suffix(res, sheet):
    """House style where an instrument's suffix letter names the instance of the equipment it serves (PI-9010A /
    9010B / 9010C on compressors A / B / C). A tag ending in one letter A-H whose equipment has no instance gets that
    letter as the instance when the sheet itself shows it as an equipment instance:
      (a) the instrument's own equipment is printed with sibling instances that include the letter (5K901A / 5K901B,
          '5P551A/B') - instances of other equipment on the sheet say nothing about this one, or
      (b) the sheet is that instance's sheet: most of its suffixed instrument tags carry the same letter.
    The suffix then is no doubled-instrument index, so an index taken from it is dropped."""
    P = sheet.plant
    letters = {}
    txt = [f["text"] for f in sheet.frags]
    for f in txt:
        for m in P.tag_inst.finditer(f):
            base, extra = m.group(1), m.group(2) or ""
            letters.setdefault(base, set()).update(extra)
    def own_letters(sl):
        """Instance letters printed for the instrument's own equipment: its tag(s) in the equipment piece's source,
        the title tag when the name came from the title block, and every tag the register names the same."""
        src, name = sl["equipment"]["source"], sl["equipment"]["value"].replace(" ", "").upper()
        tags = {m.group(0) for m in P.tag_in.finditer(src.replace(" ", "").upper())}
        if src.startswith("title block") and sheet.title.get("tag"):
            tags.add(sheet.title["tag"])
        tags |= {t for t, n in P.names.items() if "".join(n["words"]) == name}
        out = set()
        for t in tags:
            out |= letters.get(P.base_of(t)[0], set())
        return out if len(out) >= 2 else set()
    parsed = [(t, parse_tag(t)) for t in res]
    suf = [p["suffix"] for _, p in parsed if p and p["suffix"]]
    dom = None
    if suf:
        top = max(set(suf), key=suf.count)
        if suf.count(top) >= DOMINANT_SHARE * len(suf) and len(suf) >= 5:
            dom = top
    for tag, p in parsed:
        if not p or not p["suffix"] or p["suffix"] not in "ABCDEFGH":
            continue
        r = res[tag]; s = r["slots"]
        if "equipment" not in s or "instance" in s or s.get("part", {}).get("value") == "Header":
            continue
        L = p["suffix"]
        shown = own_letters(s)
        if L in shown:
            why = f"tag suffix {L}; the sheet prints instances {'/'.join(sorted(shown))} of this equipment"
        elif L == dom:
            why = f"tag suffix {L}; {suf.count(L)} of {len(suf)} suffixed instruments on the sheet are {L}"
        else:
            continue
        s["instance"] = slot(L, why, 0.6)
        if s.get("index", {}).get("source", "").startswith("suffix"):
            del s["index"]
        r["description"] = compose(s, sheet.profile)


def number_repeats(res, profile):
    """Several loops measuring the same thing on one machine (vibration probes VI-5001 .. 5004, speed probes): when
    their descriptions are otherwise identical and the variable is one the machine kind numbers
    (rules.MACHINE_RULES 'numbered'), number them 1, 2 .. in loop-number order. Covers machines named by the title
    block, which onpipe.numbered_index (owner found along the lines) does not reach."""
    numbered = {v for r in R.MACHINE_RULES.values() for v in r.get("numbered", ())}
    groups = {}
    for tag, r in res.items():
        s, p = r["slots"], parse_tag(tag)
        if not p or "index" in s or {"part", "end", "stage", "stream"} & set(s):
            continue
        if s.get("variable", {}).get("value") not in numbered:
            continue
        groups.setdefault((p["letters"], r["description"]), []).append((p["number"], tag))
    for (L, _), members in groups.items():
        loops = sorted({n for n, _ in members})
        if len(loops) < 2:
            continue
        for n, tag in members:
            r = res[tag]
            r["slots"]["index"] = slot(str(loops.index(n) + 1), f"{len(loops)} {L} loops with the same description: "
                                       "numbered in loop-number order", 0.6)
            r["description"] = compose(r["slots"], profile)


def compose(s, profile):
    return " ".join(abbreviate(s[k]["value"], profile) for k in profile["order"] if k in s and s[k]["value"])


def review_reason(s, lead):
    why = []
    if lead is None:
        why.append("no line from the bubble found")
    weak = [k for k, v in s.items() if v["conf"] < 0.7]
    if weak:
        why.append("least sure: " + ", ".join(weak))
    if "variable" not in s:
        why.append("measured variable unknown")
    return "; ".join(why) or "ok"
