"""Instruments on pipes and vessels: which equipment the measurement belongs to and where on it (stream, stage).

Two mechanisms, tried in this order (describe.py decides):
  pipe_slots  the pipe the instrument taps is followed both ways (lines.Pipes) to the first terminal each way;
              the equipment class that names this measurement after itself (rules 'claims': a machine its suction /
              discharge P, T, F; an exchanger the temperature at it; a vessel its level) owns it, else the upstream
              equipment; flow direction from connector words (FROM / TO / INLET / OUTLET / SUPPLY / RETURN) or flow
              arrows on the pipe; a machine accessory's service text ('DAMPER 1ST STAGE DISCHARGE') gives stage + stream.
  ownership   the equipment nearest along ANY drawn line (plant.owner: Dijkstra over the ink network), with
              machines preferred over vessels on the same line; two instances (A / B) about equally far = their
              shared header; flow arrows on the path give the stream; an instrument with no pipe between it and
              the equipment is on its body. Loop members (controllers, indicators, relays) take the measuring point
              of the loop's element / transmitter bubble (ISA: one loop number = one measuring point).
Every piece is (value, source, confidence)."""
import math
import re
import numpy as np
from .sheet import MM, ctr
from . import rules as R
from .style import name_case
from .text import kind_of, rules_of, stage_of


def slot(value, source, conf):
    return dict(value=value, source=source, conf=conf)


# ======================= pipe following from the tap =======================
def _name(P, tag):
    """Readable name of an equipment: its data-box / printed name; a sub-part of the sheet's machine package
    ('INTERSTAGE SUCTION SCRUBBER', 'LUBE OIL COOLER') gets the machine's name in front."""
    n = P.info(tag)["name"]
    if not n:
        return None
    main = P.main_name
    if main and tag != P.main and (n[0] in R.POSITION_PREFIX or rules_of(kind_of(main)).get("cls") == "machine"
                                   and not set(n) & (set(main) - set(R.KIND_OF_NOUN))):
        n = main + n
    return name_case(" ".join(n))


def _service_words(name):
    """Stage and stream words printed in an accessory's service text: 'DAMPER 1ST STAGE DISCHARGE' -> (1, 'Discharge')."""
    st = stage_of(name or [])
    stream = next((w for w in (name or []) if w in ("SUCTION", "DISCHARGE", "INLET", "OUTLET")), None)
    return st, stream.title() if stream else None


def _side(P, found):
    """First terminal met in one direction -> dict(kind, tag, info, flow, service, dist, src)."""
    if not found:
        return None
    d, t, _ = found[0]
    if t["kind"] == "equipment":
        return dict(kind="equipment", tag=t["tag"], info=P.info(t["tag"]), flow=None, service=[], dist=d,
                    src=f"pipe followed {d:.0f} mm to equipment {t['tag']}" + (f" ({t['via']})" if t.get("via") else ""))
    c = t["conn"]
    tag = c["tags"][0] if c["tags"] else None
    return dict(kind="connector", tag=tag, info=P.info(tag) if tag else None, flow=c["flow"], service=c["service"], dist=d,
                src=f"pipe followed {d:.0f} mm to connector '{c['text'][:50]}'")


def _drop_note_valve(s):
    """On a line instrument a note about valves is about its isolation / root valves, not the measured part."""
    if s.get("part", {}).get("value") == "Valve" and s["part"]["source"].startswith("NOTE"):
        del s["part"]


def pipe_slots(sheet, L, lead, bx, by, s):
    """Equipment / stage / stream from the pipe the instrument taps. Returns True when it placed the instrument."""
    P = sheet.plant
    prof = sheet.profile
    first = L[0]
    var = "P" if L[:2] == "PD" else first
    main = dict(tag=P.main, name=P.main_name, kind=P.main_kind, cls=rules_of(P.main_kind).get("cls"), stage=None) \
        if P.main_name else None
    G = sheet.pipes()
    tp = G.tap(lead["pts"]) if lead else None
    if lead and not tp and G.mode == "heavy pen":
        # utility lines (cooling water ...) on a sheet with a heavy process pen are drawn with the light pen
        G = sheet.pipes(light=True)
        tp = G.tap(lead["pts"])
    follow = G.follow(tp[0], tp[1]) if tp else {"-": [], "+": []}
    sides = {k: _side(P, v) for k, v in follow.items()}

    if first in R.VESSEL_LETTERS:
        # the vessel the level instrument's lines reach, else the nearest vessel label to the bubble
        cand = [(v["dist"], v["tag"], v["src"]) for v in sides.values() if v and v["kind"] == "equipment"
                and v["info"]["cls"] == "vessel"]
        if not cand:
            cand = [(np.hypot(l["c"][0] - bx, l["c"][1] - by) / MM, l["tag"], f"nearest vessel label '{l['tag']}'")
                    for l in P.labels if P.info(l["tag"])["cls"] == "vessel"]
            cand = [c for c in cand if c[0] < 150]
        if cand:
            d, tag, src = min(cand)
            nm = _name(P, tag)
            if nm:
                s["equipment"] = slot(nm, f"{src}: {nm}" if tag != P.main else f"{src}; sheet equipment {tag}",
                                      0.75 if tag != P.main else 0.8)
                return True
        return False

    if first not in R.PIPE_LETTERS or not tp:
        return False
    # 1. a machine accessory (pulsation damper ...) whose service text names stage and stream
    acc = sorted((v for v in sides.values() if v and v["kind"] == "equipment" and v["info"]["cls"] == "accessory"
                  and any(_service_words(v["info"]["name"]))), key=lambda v: v["dist"])
    if acc:
        a = acc[0]
        st, stream = _service_words(a["info"]["name"])
        why = f"{a['src']}: '{' '.join(a['info']['name'])}'"
        if st:
            s["stage"] = slot(prof["stage"].format(n=st), why, 0.8)
        if stream:
            s["stream"] = slot(stream, why, 0.8)
        _drop_note_valve(s)
        return True
    # 2. which side is upstream: connectors say FROM / TO (INLET / OUTLET at tie points)
    up, arrow_src = None, None
    for k, v in sides.items():
        if v and v["kind"] == "connector" and v["flow"] in ("in", "out"):
            up = k if v["flow"] == "in" else ("+" if k == "-" else "-")
    if up is None:
        # flow arrows drawn on the pipe between the tap and the first terminal of each side
        net = P.network()
        votes = 0
        for k, v in follow.items():
            if v:
                path = [G.pt(G.segs[tp[0]], tp[1])] + v[0][1]["path"][:40]
                a = [u[1] for u in net.arrows_on(path, densify_mm=0.34)]
                votes += sum(a) if k == "+" else -sum(a)
        if votes:
            up = "-" if votes > 0 else "+"
            arrow_src = f"flow arrows on the pipe ({abs(votes)} net)"
    # a side reaching nothing on a sheet drawn around a machine is the machine (the line runs into its package)
    drawn = P.main is not None and any(l["tag"] == P.main for l in P.labels)
    for k, v in sides.items():
        other = sides["+" if k == "-" else "-"]
        if v is None and drawn and main["cls"] == "machine" and other and other["kind"] == "connector":
            sides[k] = dict(kind="equipment", tag=P.main, info=main, flow=None, service=[], dist=999,
                            src=f"pipe runs into the sheet's machine {P.main} (no other terminal that way)")
    # 3. owner: the equipment class that names this measurement after itself, else the upstream equipment
    cands = [(k, v) for k, v in sides.items() if v and v["info"] and var in rules_of(v["info"]["kind"]).get("claims", "")]
    if not cands and up and sides[up] and sides[up]["tag"] and L[:2] != "PD":
        cands = [(up, sides[up])]
    if not cands or up is None:
        return False
    k, o = min(cands, key=lambda kv: kv[1]["dist"])
    into = (k != up)                                   # fluid flows from the tap into the owner -> owner's inlet
    words_in, words_out = rules_of(o["info"]["kind"]).get("streams", R.DEFAULT_STREAMS)
    stream = words_in if into else words_out
    flow_src = next((v["src"] for v in sides.values() if v and v["kind"] == "connector" and v["flow"] in ("in", "out")),
                    None) or arrow_src
    if o["info"]["cls"] == "exchanger":
        # two fluids pass an exchanger: name the side by the fluid the line carries (service text of its connector)
        fl = [R.FLUID_WORDS[w] for v in sides.values() if v and v["kind"] == "connector" for w in v["service"]
              if w in R.FLUID_WORDS]
        if fl:
            stream = f"{fl[-1]} {stream}"
    s["stream"] = slot(stream, f"{o['src']}; flow direction from {flow_src}", 0.75)
    _drop_note_valve(s)
    if o["info"]["cls"] == "machine" and P.n_stages and P.n_stages > 1:
        n = 1 if into else P.n_stages
        s["stage"] = slot(prof["stage"].format(n=n), f"{'first' if into else 'last'} stage of a {P.n_stages}-stage machine "
                                                     f"({'suction' if into else 'discharge'} side)", 0.6)
    if not (o["info"] is main or P.main is not None and o["tag"] == P.main):
        nm = _name(P, o["tag"]) if o["tag"] else None
        if not nm and o["service"]:
            nm = name_case(" ".join(o["service"]))
        if nm:
            s["equipment"] = slot(nm, f"{o['src']}: {nm}", 0.65)
    if o["tag"] and prof["instance_from_tag"]:
        # one of several identical machines (5P551A / 5P551B): its instance letter
        base, letter = P.base_of(o["tag"])
        if letter and P.has_siblings(base, letter[:1]):
            s["instance"] = slot(letter[:1], f"instance letter of {o['tag']} (other instances printed on the sheet)", 0.7)
    return True


# ======================= ownership along the lines =======================
P_LOOP_MM = 400      # a loop number is unique on a sheet: its bubbles may be drawn far apart


def loop_bubbles(sheet, p, bbox, any_distance=False):
    """Other bubbles of the same loop, located by their printed loop number (bubble text: letters over number), as
    boxes of this bubble's size: [(bbox, letters)] nearest first."""
    cx, cy = ctr(bbox)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    out = []
    num = p["number"] + (p["suffix"] or "")
    for f in sheet.frags:
        if f["text"] in (num, p["number"]):
            fx, fy = ctr(f["bbox"])
            if any_distance or 3 * MM < math.hypot(fx - cx, fy - cy) < P_LOOP_MM * MM:
                bx, by = fx, fy - 0.2 * h
                lt = [g["text"] for g in sheet.frags if re.fullmatch(r"[A-Z]{1,5}", g["text"])
                      and abs(ctr(g["bbox"])[0] - fx) < 0.3 * w and 0 < fy - ctr(g["bbox"])[1] < 0.6 * h]
                out.append(([bx - w / 2, by - h / 2, bx + w / 2, by + h / 2], lt[0] if lt else ""))
    return sorted(out, key=lambda t: math.hypot(ctr(t[0])[0] - cx, ctr(t[0])[1] - cy))[:6]


def loop_owner(sheet, p, near_bbox):
    """Owner key of another loop's bubbles (found by its printed number), or None."""
    P = sheet.plant
    for sb, sl in loop_bubbles(sheet, p, near_bbox, any_distance=True):
        c = P.owner(sb)
        if c and c[0][0] <= sheet.profile["own_max_mm"]:
            it = P.targets[c[0][1]]
            return (it["base"], tuple(it["letters"]), it["kind"])
    return None


def ownership(sheet, bbox, p, s):
    """Equipment, instance, stream and line words from the equipment the instrument is connected to along the lines.
    Returns the owner key (base, letters, kind) or None."""
    prof, P = sheet.profile, sheet.plant
    if not P.targets:
        return None
    var = s.get("variable", {}).get("value")

    def read(b):
        c = P.owner(b)
        if not c or c[0][0] > prof["own_max_mm"]:
            return None
        # a suction / discharge line is named after the machine on it, not the vessel or tank at its other end
        rank = lambda e: rules_of(P.targets[e[1]]["kind"]).get("line_owner_rank", 0)
        best = max((e for e in c if e[0] <= prof["machine_pref"] * c[0][0]), key=lambda e: (rank(e), -e[0]))
        c = [best] + [e for e in c if e is not best]
        d0, k, node = c[0]
        it = P.targets[k]
        if it["via"] == "connector":
            return dict(c=c, on_pipe=99, sgn=1 if it["direction"] == "to" else -1,
                        why=f"connector says '{it['direction'].upper()}' the equipment")
        sgn, info = P.flow(k, node)
        return dict(c=c, on_pipe=info["on_pipe_mm"], sgn=sgn,
                    why="flow arrow%s %s the equipment" % ("" if info["arrows"] else " near the line",
                                                         "towards" if sgn > 0 else "away from"))

    got, via_loop = read(bbox), ""
    # ISA: a loop measures at its primary element / transmitter. Controllers, relays, indicators and stacked display
    # bubbles take the measuring point of the loop's E / T bubble; an instrument not connected at all, or connected
    # only by signal lines while it measures the process line, takes the nearest connected loop member
    measuring = re.search(f"[{R.MEASURING_LETTERS}]", p["letters"][1:]) is not None
    sibs = loop_bubbles(sheet, p, bbox)
    if not measuring and var in R.STREAM_VARIABLES:
        for sb, sl in sibs:
            if re.search(f"[{R.MEASURING_LETTERS}]", sl[1:]):
                g = read(sb)
                if g and g["on_pipe"] > 10:
                    got, via_loop = g, f" (measuring point of {sl}-{p['number']})"
                    break
    if got is None or (var in R.STREAM_VARIABLES and got["on_pipe"] <= 10):
        for sb, sl in sibs:
            g = read(sb)
            if g and (got is None or (g["on_pipe"] > 10 and got["on_pipe"] <= 10)):
                got, via_loop = g, " (via another instrument of loop %s)" % p["number"]
                if got["on_pipe"] > 10:
                    break
    if got is None:
        return None
    cands = got["c"]
    d0, k, node = cands[0]
    it = P.targets[k]
    rules = rules_of(it["kind"])
    shared = [P.targets[c[1]] for c in cands[1:] if c[0] < prof["shared_ratio"] * d0
              and P.targets[c[1]]["base"] == it["base"] and P.targets[c[1]]["letters"] != it["letters"]]
    letters = sorted(set(it["letters"] + [x for o in shared for x in o["letters"]]))
    header = len(letters) > 1
    conf = 0.85 if d0 < 150 else 0.7
    s["equipment"] = slot(name_case(it["name"]), f"{it['src']}; {d0:.0f} mm along lines{via_loop}", conf)
    if not header and it["letters"] and prof["instance_from_tag"] and P.has_siblings(it["base"], it["letters"][0]):
        s["instance"] = slot(it["letters"][0], f"instance letter of {it['tag']} (other instances printed on the sheet)", conf)
    streams = rules.get("streams")
    if streams and got["sgn"] and got["on_pipe"] > 10 and var in R.STREAM_VARIABLES and "stream" not in s:
        s["stream"] = slot(streams[0] if got["sgn"] > 0 else streams[1], got["why"], 0.7)
    if got["on_pipe"] <= 10 and var in rules.get("body", {}) and "part" not in s:
        s["part"] = slot(rules["body"][var], f"instrument mounted on the {it['kind']} body (no pipe between)", 0.6)
    if header and rules.get("shared_line") and "part" not in s and var in R.STREAM_VARIABLES:
        s["part"] = slot(rules["shared_line"], "line shared by instances " + "/".join(letters), 0.65)
    return (it["base"], tuple(letters), it["kind"])


def numbered_index(sheet, p, s, owner, siblings, bbox):
    """Several loops measuring the same thing on one machine (vibration probes): number them 1, 2 .. in tag order."""
    numbered = rules_of(owner[2]).get("numbered", ())
    if "index" in s or {"part", "end", "stage", "stream"} & set(s) or s.get("variable", {}).get("value") not in numbered:
        return
    L = p["letters"]
    same = [p["number"]] + [n for (L2, n) in siblings if L2 == L and n != p["number"]
                            and loop_owner(sheet, dict(p, number=n, suffix=""), bbox) == owner]
    if len(same) > 1:
        s["index"] = slot(str(sorted(same).index(p["number"]) + 1),
                          f"{len(same)} {L} loops on {owner[0]}{''.join(owner[1])}: numbered in tag order", 0.6)
