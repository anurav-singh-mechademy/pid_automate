"""Auxiliary-system pieces of a description (lube oil, cooling water ... sheets): system, item and position.

  system   : the title names a system of the machine ('... LUBE OIL CIRCUIT' -> LO)        (sheet.title['system'])
  item     : the equipment the instrument sits on, or the item its pipe leads to (pipe following, lines.Pipes)
  position : where on the circuit the tap is: supply / return (towards / from the machine served), inlet / outlet of a
             consumer, pump suction / discharge; a DP instrument's two taps name the item between them.
Flow direction comes from connector words (TO / FROM / OUTLET / INLET) or, between two items, from the circuit order
(rules.CIRCUIT). Terminals are the register's items (plant.items), the machine's cylinders and its connectors."""
from .sheet import MM
from .rules import CONSUMER_WORDS, CIRCUIT, POSITION, VESSEL_LETTERS, VESSEL_STEPS
from .trace import trace_bubble


def slot(value, source, conf):
    return dict(value=value, source=source, conf=conf)


def role_of(i, system):
    k = i["step"]
    if k is None or any(w in "".join(i["words"]) for w in CONSUMER_WORDS):
        return "consumer"          # unknown equipment, or part of the machine itself (its crankcase sump)
    if k == "cooler" and (i["process"] or (i["fluid"] and system and i["fluid"] != system)):
        return "consumer"          # a cooler of another stream is served by this system
    return k


def rank(role):
    return CIRCUIT.index(role) if role in CIRCUIT else 0


def context(sheet):
    """Terminals of the circuit, built once per sheet."""
    if getattr(sheet, "_aux", None) is None:
        system, P = sheet.title["system"], sheet.plant
        terms = []
        for i in P.items:
            # nozzles are drawn with a flange and a gap off the outline: items are reached within 4.5 mm
            terms.append(dict(i, type="item", role=role_of(i, system), tol_mm=4.5))
        for p in sheet.parts:
            if p["kind"] == "cylinder":
                terms.append(dict(type="item", role="consumer", bbox=p["bbox"], stage=p["stage"], tag=None, step=None,
                                  text=p["label"], words=p["label"].split(), kind="cylinder", fluid=None, suffix=None,
                                  tol_mm=3.0))
        for c in P.connectors:
            role = "consumer" if c["consumer"] else (role_of(c, system) if c["step"] else "system")
            terms.append(dict(c, type="connector", role=role, src_conn=c, tol_mm=3.0))
        for t in terms:
            t["group"] = (t["type"], " ".join(t["words"]) if t["type"] == "item" else t["text"])
        sheet._aux = dict(system=system, title_line=sheet.title["system_line"], terms=terms)
    return sheet._aux


def item_name(t, system, profile, plural=False):
    if t.get("stage"):
        return profile["cylinder_item"].format(stage=t["stage"])
    if t["step"] in profile["item_alias"]:
        n = profile["item_alias"][t["step"]]
    else:
        w = list(t["words"])
        if profile["drop_system_words_in_item"] and t["fluid"] and t["fluid"] == system:
            key = t["fluid_words"]
            for i in range(len(w)):
                for k in range(1, len(w) - i):
                    if "".join(w[i:i + k]) == key:
                        w = w[:i] + w[i + k:]; break
                else:
                    continue
                break
        n = " ".join(w).title()
    return n + (profile["plural"] if plural else "")


def taps(sheet, bbox):
    """The instrument's lines (inner ring only, >= 2 mm), one per exit direction, longest first."""
    lines = [l for l in trace_bubble(sheet.gray, bbox, sheet.ink) if l["length_mm"] >= 2]
    out = []
    for l in sorted(lines, key=lambda l: (l["ring_mm"], -l["length_mm"])):
        if all(abs((l["exit_ang"] - o["exit_ang"] + 180) % 360 - 180) > 30 for o in out):
            out.append(l)
    return sorted(out, key=lambda l: -l["length_mm"])


def on_item(terms, pt, tol=1.5 * MM):
    best = None
    for i, t in enumerate(terms):
        if t["type"] != "item":
            continue
        x0, y0, x1, y1 = t["bbox"]
        if x0 - tol <= pt[0] <= x1 + tol and y0 - tol <= pt[1] <= y1 + tol:
            a = (x1 - x0) * (y1 - y0)
            if best is None or a < best[0]:
                best = (a, i)
    return best[1] if best else None


def walk_sides(sheet, line, terms):
    """Terminals reached each way along the pipe the line taps: [[(mm, term index) ...], [...]]. The heavy-pen graph
    first; utility lines drawn with the light pen use the graph of every solid line."""
    boxes = [(t["bbox"], t["tol_mm"], k) for k, t in enumerate(terms)]
    by_conn = {id(t["src_conn"]): k for k, t in enumerate(terms) if t.get("src_conn")}
    by_tag = {t["tag"]: k for k, t in enumerate(terms) if t["type"] == "item" and t.get("tag")}
    for G in (sheet.pipes(), sheet.pipes(light=True)):
        tp = G.tap(line["pts"])
        if not tp:
            continue
        sides = []
        for found in G.follow(tp[0], tp[1], boxes=boxes).values():
            sd, seen = [], set()
            for d, t, _ in found:
                k = t["ref"] if t["kind"] == "box" else by_conn.get(id(t.get("conn"))) if t["kind"] == "connector" \
                    else by_tag.get(t.get("tag"))
                if k is not None and k not in seen:
                    seen.add(k); sd.append((d, k))
            sides.append(sd)
        if any(sides):
            return sides
    return []


def plural_of(terms, side, idx):
    """Siblings (same item name, other suffix) reached on the same side about as soon as the first one."""
    g = terms[idx]["group"]; d0 = next(d for d, i in side if i == idx)
    sib = [d for d, i in side if i != idx and terms[i]["group"] == g and terms[i].get("suffix")]
    return bool(sib) and min(sib) < 2 * d0 + 30


def aux_slots(sheet, bbox, letters, lead, s):
    ctx = context(sheet)
    system, terms, prof = ctx["system"], ctx["terms"], sheet.profile
    s["system"] = slot(system, "title block: " + ctx["title_line"], 0.85)
    if not lead:
        return
    end = lead["pts"][-1]

    # differential pressure: two taps, the item between them
    if letters.startswith("PD"):
        ls = taps(sheet, bbox)
        if len(ls) >= 2:
            reach = []
            for l in ls[:2]:
                r = {}
                for side in walk_sides(sheet, l, terms):
                    for d, i in side:
                        if terms[i]["type"] == "item":
                            r.setdefault(terms[i]["group"], []).append((d, i))
                reach.append(r)
            common = set(reach[0]) & set(reach[1])
            if common:
                g = min(common, key=lambda g: min(reach[0][g])[0] + min(reach[1][g])[0])
                members = {i for d, i in reach[0][g] + reach[1][g]}
                i0 = min(reach[0][g])[1]
                pl = len(members) > 1
                s["item"] = slot(item_name(terms[i0], system, prof, pl), f"both taps lead to '{terms[i0]['text']}'"
                                 + (" (and its twin)" if pl else ""), 0.75)
                return

    i = on_item(terms, end)
    if i is not None:
        s["item"] = slot(item_name(terms[i], system, prof), f"line ends on the outline of '{terms[i]['tag'] or terms[i]['text']}'", 0.8)
        return

    def foreign(t):     # a connector of another system (instrument air to a valve actuator ...)
        return t["type"] == "connector" and t.get("fluid") and t["fluid"] != system
    sides = [[(d, i) for d, i in sd if not foreign(terms[i])] for sd in walk_sides(sheet, lead, terms)]
    sides = [sd for sd in sides if sd]
    if not sides:
        return
    if letters[0] in VESSEL_LETTERS:
        v = sorted((d, i) for sd in sides for d, i in sd if terms[i]["type"] == "item" and terms[i]["step"] in VESSEL_STEPS)
        if v:
            t = terms[v[0][1]]
            s["item"] = slot(item_name(t, system, prof), f"level instrument: its lines lead to vessel '{t['tag'] or t['text']}'", 0.75)
            return
    # the item side: the side whose nearest equipment item is nearest (else the nearest terminal)
    def first_item(sd):
        return next(((d, i) for d, i in sd if terms[i]["type"] == "item"), None)
    sides.sort(key=lambda sd: (first_item(sd) is None, (first_item(sd) or sd[0])[0]))
    near, far = sides[0], (sides[1] if len(sides) > 1 else [])
    a = (first_item(near) or near[0])[1]; A = terms[a]
    d_a = next(d for d, i in near if i == a)
    if far and terms[far[0][1]]["group"] == A["group"] and d_a < 50 and far[0][0] < 50:
        # both ways reach the same item at once: a bridle / two nozzles of one item
        s["item"] = slot(item_name(A, system, prof), f"both ways along the pipe lead to '{A['tag'] or A['text']}'", 0.7)
        return
    b = next((i for _, i in far if terms[i]["group"] != A["group"] and
              not (terms[i]["type"] == A["type"] == "item" and terms[i]["role"] == A["role"])), None)
    B = terms[b] if b is not None else None
    src = f"pipe leads one way to '{A['text']}'" + (f", the other way to '{B['text']}'" if B else "")
    # flow direction: a connector's words, else the circuit order of the two items
    U = D = None
    if A["type"] == "connector":
        U, D = (A, B) if A["flow"] == "in" else (B, A)
    elif B is not None and B["type"] == "connector":
        U, D = (B, A) if B["flow"] == "in" else (A, B)
    elif B is not None:
        n = len(CIRCUIT); diff = (rank(B["role"]) - rank(A["role"])) % n
        U, D = (A, B) if diff <= n / 2 else (B, A)
    def isitem(t): return t is not None and t["type"] == "item"
    def pos_of(t, k): return POSITION.get(t["step"], POSITION["default"])[k]
    def pl(t):
        idx = next(k for k, x in enumerate(terms) if x is t)
        side = near if any(i == idx for _, i in near) else far
        return plural_of(terms, side, idx)
    pos = item = None
    if D is not None and D["type"] == "connector" and D["role"] == "consumer":
        pos = prof["supply_word"]
    elif U is not None and U["type"] == "connector" and U["role"] == "consumer":
        pos = prof["return_word"]
    elif U is not None and U["role"] not in ("system", "consumer") and rank(U["role"]) >= rank(prof["supply_after"]) and \
            (D is None or D["role"] == "consumer" or rank(D["role"]) > rank(U["role"])):
        pos = prof["supply_word"]          # conditioned fluid on its way to the machine
    elif isitem(D) and D["role"] == "consumer":
        item, pos = D, pos_of(D, 0)
    elif isitem(U) and U["role"] == "consumer":
        item, pos = U, pos_of(U, 1)
    elif isitem(U) and U["role"] in prof["name_by_upstream"]:
        item, pos = U, pos_of(U, 1)
    elif isitem(D):
        item, pos = D, pos_of(D, 0)
    elif isitem(U):
        item, pos = U, pos_of(U, 1)
    elif isitem(A):
        item = A
    if item is not None:
        machine_part = item["role"] == "consumer" and not (item.get("fluid") and item["fluid"] != system)
        key = "machine_item" if machine_part and prof["machine_item_first"] else "item"
        s[key] = slot(item_name(item, system, prof, pl(item)), src, 0.65)
    if pos:
        s["position"] = slot(pos, src + (f"; flow {U['text']} -> {D['text']}" if U is not None and D is not None else ""), 0.6)
