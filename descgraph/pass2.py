"""Step 3: the graph pass. Starts from pass 1's pieces and fills / replaces the ones that need the drawing's
connections, using only pidgraph.query.SheetQuery on the saved graph:

  1. line lands on a machine part (cylinder / motor / flywheel ...)   -> stage / machine / part
  2. line lands on equipment                                            -> equipment (+ instance)
  3. line taps a pipe or sits on an in-line item: equipment reached along the pipes, chosen with the
     descgen.rules ownership data (machines before vessels within machine_pref, 'claims' letters, own_max_mm)
                                                                        -> equipment, instance, stream from the flow side
     (upstream of a machine = its suction / inlet side; downstream = discharge / outlet: MACHINE_RULES streams)
     on an auxiliary-system sheet (title names LO / CW) the item reached is the system's item and the flow side is
     its position: 'LO Pump Discharge'
  4. nothing reached: take the answer of another instrument of the same loop (same number), measuring element /
     transmitter first (rules.MEASURING_LETTERS)
Every piece records the graph evidence (nodes, distance along lines, flow edges used)."""
import re
from descgen import rules as R
from descgen.describe import slot, compose
from descgen.style import name_case

PARTS = {"cylinder", "motor", "flywheel", "frame", "crankcase", "gearbox", "bearing"}
SYSTEM_WORDS = {"LO": ("LUBE", "OIL", "LO", "LUBRICATING"), "CW": ("COOLING", "WATER", "CW", "TCW", "CLOSED")}


def _kind_rule(kind):
    return R.MACHINE_RULES.get(kind or "", {})


def _stage(name):
    m = re.search(r"(\d)\s*(?:ST|ND|RD|TH)?\s*STAGE|STAGE\s*(\d)|STG\s*(\d)", (name or "").upper())
    return next((g for g in m.groups() if g), None) if m else None


def choose_owner(q, reached, first_letter, profile):
    """Equipment that names the instrument's line: machines win up to machine_pref x further than a vessel /
    exchanger; equipment whose kind claims the measured letter is preferred; nothing beyond own_max_mm."""
    cands = []
    for r in reached:
        if r["type"] != "equipment":
            continue
        d = q.G.nodes[r["node"]]
        kind = d.get("kind") or ""
        if kind in PARTS or kind in ("shape", "panel") or not (d.get("name") or d.get("tag")):
            continue
        rule = _kind_rule(kind)
        w = r["dist_mm"]
        if rule.get("cls") == "machine":
            w /= profile["machine_pref"]
        if first_letter in rule.get("claims", ""):
            w *= 0.7
        if r["dist_mm"] <= profile["own_max_mm"]:
            cands.append((w, r))
    return min(cands, key=lambda c: c[0])[1] if cands else None


def item_word(name, system):
    words = [w for w in re.findall(r"[A-Z]+", name.upper()) if w not in SYSTEM_WORDS.get(system, ())]
    nouns = [w for w in words if w in R.KIND_OF_NOUN]
    return name_case(nouns[-1]) if nouns else name_case(" ".join(words[-1:]))


def graph_pass(q, tag, p1, profile, depth=0):
    s = {k: dict(v) for k, v in p1["slots"].items()}
    n = q.by_tag.get(tag)
    if n is None:
        return dict(tag=tag, slots=s, description=compose(s, profile), evidence="tag not in graph")
    L = (p1["parsed"] or {}).get("letters") or ""
    atts = q.attachments(n)
    ev = []
    system = (s.get("system") or {}).get("value")
    # 1 / 2: line lands on equipment or a machine part
    for a in atts:
        if a["type"] != "equipment":
            continue
        d = q.G.nodes[a["target"]]
        kind = d.get("kind") or ""
        if kind in PARTS:
            st = _stage(d.get("name"))
            if st:
                s["stage"] = slot(profile["stage"].format(n=st), f"graph: line lands on {d.get('name')}", 0.8)
            word = "Motor" if kind == "motor" else name_case(kind)
            s["machine" if kind == "motor" else "part"] = slot(word, f"graph: line lands on machine part {a['target']}", 0.75)
            ev.append(f"part {a['target']}")
            return dict(tag=tag, slots=s, description=compose(s, profile), evidence="; ".join(ev))
        if d.get("name") or d.get("tag"):
            _set_owner(q, s, a["target"], None, profile, system, f"line lands on {d.get('tag') or d.get('name')}")
            return dict(tag=tag, slots=s, description=compose(s, profile), evidence=f"on equipment {a['target']}")
    # 3: along the pipes
    reached = q.reach(n, max_mm=profile["own_max_mm"])
    own = choose_owner(q, reached, L[:1], profile)
    if own:
        _set_owner(q, s, own["node"], own, profile, system,
                   f"{own['dist_mm']:.0f} mm along the pipes, flow edges +{own['flow_edges'][0]}/-{own['flow_edges'][1]}")
        return dict(tag=tag, slots=s, description=compose(s, profile),
                    evidence=f"reach {own['node']} {own['dist_mm']} mm side={own['side']}")
    # 4: another instrument of the loop
    if depth == 0:
        mates = q.loop(n)["same_number"]
        mates.sort(key=lambda m: not any(c in (q.G.nodes[m].get("letters") or "")[1:] for c in R.MEASURING_LETTERS))
        for m in mates:
            t2 = q.G.nodes[m]["tag"]
            r = graph_pass(q, t2, dict(p1, slots={}), profile, depth=1)
            got = {k: v for k, v in r["slots"].items() if k not in ("variable", "index")}
            if got:
                for k, v in got.items():
                    s[k] = dict(v, source=f"loop member {t2}: " + v["source"], conf=v["conf"] * 0.9)
                return dict(tag=tag, slots=s, description=compose(s, profile), evidence=f"via loop member {t2}")
    return dict(tag=tag, slots=s, description=compose(s, profile), evidence="nothing reached")


def _set_owner(q, s, eq, reach, profile, system, why):
    d = q.G.nodes[eq]
    name = name_case(d.get("name") or d.get("tag"))
    rule = _kind_rule(d.get("kind"))
    tag = d.get("tag") or ""
    inst = tag[-1] if re.search(r"\d[A-Z]$", tag) and q.siblings(eq) else None
    side = reach["side"] if reach else None
    streams = rule.get("streams") or R.DEFAULT_STREAMS
    stream = streams[0] if side == "upstream" else streams[1] if side == "downstream" else None
    st = _stage(d.get("name"))
    nm = (d.get("name") or "").upper()
    words = set(re.findall(r"[A-Z0-9]+", nm))
    title = (s.get("equipment") or {}).get("value") if (s.get("equipment") or {}).get("source", "").startswith("title") else None
    if title and not system and (d.get("kind") == "accessory" or words & set(R.POSITION_PREFIX)):
        # an accessory or a sub-vessel of the sheet's machine ('DAMPER 2ND STAGE SUCTION', 'INTERSTAGE SUCTION
        # SCRUBBER'): the machine names it; the accessory's own words give stage and stream
        if d.get("kind") == "accessory":
            name = title
            if not stream:
                stream = next((w.title() for w in ("SUCTION", "DISCHARGE", "INLET", "OUTLET") if w in words), None)
        else:
            rest = re.sub(r"\b(1ST|2ND|3RD|4TH|\d)\s*STAGE\b", "", d.get("name") or "").strip()
            name = f"{title} {name_case(rest)}"
    if system:     # auxiliary-system sheet: the machine stays the title's, the item is the system's
        s["item"] = slot(item_word(d.get("name") or d.get("tag"), system), f"graph: {why} -> {eq}", 0.7)
        if stream:
            s["position"] = slot(stream, f"graph: flow side of {eq} ({side})", 0.65)
    else:
        s["equipment"] = slot(name, f"graph: {why} -> {eq}", 0.75)
        if stream:
            s["stream"] = slot(stream, f"graph: flow side of {eq} ({side})", 0.65)
    if inst:
        s["instance"] = slot(inst, f"graph: instance letter of {tag} (siblings {', '.join(q.siblings(eq))})", 0.8)
    if st:
        s["stage"] = slot(profile["stage"].format(n=st), f"graph: stage in the name of {eq}", 0.7)
