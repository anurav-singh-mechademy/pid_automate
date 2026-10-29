"""Drafting-style profiles: how a drawing house lays its sheets out and how a client words its descriptions.
Everything here is a setting read by the general mechanisms; a new client / drafting house needs a new profile entry,
not new code. `select(frags)` picks the first profile whose `detect` pattern (the drawing-number format printed in the
title block and on off-sheet connectors) matches at least two OCR fragments, else DEFAULT."""
import re

DEFAULT = dict(
    name="default",
    # ---- layout
    title_labels=("TITLE", "TITLE:"), title_depth_mm=25,
    notes_labels=("NOTES", "NOTES:"),
    # headers of the notes / title panel down the right side (its text is not part of the drawing)
    panel_labels=("NOTES", "NOTES:", "GENERALNOTES", "GENERALNOTES:", "TITLE", "TITLE:"),
    # equipment tag printed on the drawing (data boxes, symbol labels, connectors), without its instance letter
    equipment_tag=r"\d{0,2}[A-Z]{1,3}-\d{3,5}",
    # label that starts the name line of an equipment data box ('TITLE/SERVICE: ...' read as 'TITLESERVICE...')
    data_box_labels=("TITLESERVICE",),
    # a drawing reference in an off-sheet connector (None: only TO / FROM wording marks connectors)
    sheet_ref=None,
    # where a connector's text sits: 'around' a TO / FROM line or drawing reference, or stacked 'above' a reference box
    connector_text="around",
    # ---- wording
    # house abbreviations (applied whole-word, in this order) and the order of the description pieces
    abbrev={"Compressor": "Comp", "Temperature": "Temp", "Differential Pressure": "DP",
            "Temperature Difference": "Temp Diff"},
    order=["equipment", "instance", "machine", "machine_item", "system", "stage", "end", "stream", "item", "part",
           "position", "variable", "index"],
    stage="Stg {n}",
    # name points on a multistage machine's interstage piping after the compressor stage instead of the cooler
    # ('Intercooler Outlet' -> 'Stg 2 Suction', 'Aftercooler Inlet' -> 'Stg <last> Discharge')
    name_interstage_by_stage=False,
    # labels of an instrument's root valves, one per impulse line ('HVPDIT9048-1'); {tag} = letters + number, no dash.
    # Marks where each tap is when the impulse lines cannot be traced through the valves. None: not used
    root_valve_label=None,
    # an instrument's tag suffix is its equipment's instance letter (PI-9010A on compressor A) when the sheet shows that
    # letter as an equipment instance (see describe.instance_from_suffix)
    instance_from_suffix=False,
    # instance letter of an equipment tag: trailing letter when a sibling tag differs only by it (5P551A / 5P551B)
    instance_from_tag=True,
    # ---- equipment ownership along lines (plant.Network)
    own_max_mm=900,        # an instrument further than this along the lines from every equipment belongs to none
    shared_ratio=1.4,      # two instances of a machine about this equally far away = a line shared by both (header)
    machine_pref=2.0,      # a machine up to 2x further along the line than a vessel / tank still names the line
    line_break_mm=6,       # break left in one of two crossing pipes (bridged when following lines)
    line_gap_mm=1.4,       # small breaks in drawn lines (flange pairs, valve ends) bridged
    flow_reach_mm=300,     # flow arrows within this distance along the lines tell the flow direction
    # ---- auxiliary-system wording (lube oil / cooling water sheets)
    # an item that is part of the machine (cylinder, process-gas cooler) is printed before the system serving it
    # ("Stg 1 Cylinder CW Outlet"); the system's own items after it ("CW Pumps Discharge")
    machine_item_first=True,
    drop_system_words_in_item=True,       # "Water Pump A" on the CW sheet -> "Pump"
    item_alias={"reservoir": "Reservoir"},  # the system's liquid store, whatever the data box calls it
    plural="s",                           # a tap fed by identical items (A / B) names them in the plural
    supply_word="Supply", return_word="Return",
    supply_after="cooler",                # taps downstream of this circuit step carry conditioned fluid: supply
    name_by_upstream=("pump", "reservoir"),  # name a tap after the upstream item when it is one of these
    cylinder_item="Stg {stage} Cylinder",
)

PROFILES = [
    # LNG Canada drawings: L001-11100-... drawing numbers; equipment tags 1K-1101, 1V-1105, 1S-1114A
    dict(DEFAULT, name="lngc", detect=r"L\d{3}-\d{5}", equipment_tag=r"\d{1,2}[A-Z]{1,3}-\d{4}",
         sheet_ref=r"L\d{3}-\d{5}(-[A-Z]{2}-\d{4})?-\d{4}",
         abbrev=dict(DEFAULT["abbrev"], **{"Lube Oil": "LO"})),
    # DECP drawings: SC62Dxxxx drawing numbers; equipment tags with unit prefix 5 and no dash: 5P551A, 5C551
    dict(DEFAULT, name="decp", detect=r"^SC62D\d{4}", equipment_tag=r"5[A-Z]{1,2}\d{3}",
         sheet_ref=r"^SC62D[0-9A-Z]{4,5}$", connector_text="above", abbrev={}, name_interstage_by_stage=True,
         root_valve_label=r"HV{tag}-?(\d)", instance_from_suffix=True),          # DECP names spell words out
]


def select(frags):
    for p in PROFILES:
        rx = re.compile(p["detect"])
        if sum(bool(rx.search(f["text"])) for f in frags) >= 2:
            return p
    return DEFAULT


def abbreviate(v, profile):
    for k, a in profile["abbrev"].items():          # in the profile's order
        v = re.sub(rf"\b{k}\b", a, v)
    return v


def name_case(name):
    """'LNG PRODUCT PUMP' -> 'LNG Product Pump': words without vowels are acronyms and stay upper case."""
    return " ".join(w if not re.search(r"[AEIOUY]", w) else w.title() for w in name.split())
