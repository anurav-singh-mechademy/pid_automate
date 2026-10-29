"""Step 2: descriptions from what can be read without the graph (local lookups only).
  variable   ISA-5.1 letters -> measured variable / function (descgen.rules VARIABLE, MODIFIED, FUNCTION_ONLY)
  index      A / B suffix of an instrument doubled on the sheet (TE-0303A / TE-0303B -> 1 / 2)
  equipment  the sheet's title block (the machine / vessel the drawing is about)        -- sheet context
  system     the auxiliary system the title names (lube oil, cooling water ...)        -- sheet context
Every piece keeps (value, source, conf) like descgen; pass 2 adds or replaces pieces from the graph."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "batch"):
    sys.path.insert(0, str(p))
from descgen.bench import load_sheet
from descgen.sheet import Sheet
from descgen.describe import parse_tag, variable_finder, slot, compose
from descgen.style import name_case


def sheet_pass1(name):
    page, frags, bubbles, refset = load_sheet(name)
    sheet = Sheet(page, frags)
    t = sheet.title
    sib = Counter((p["letters"], p["number"]) for p in map(parse_tag, bubbles) if p)
    out = {}
    for tag in sorted(bubbles):
        p = parse_tag(tag)
        s = {}
        if p:
            v = variable_finder(p["letters"], {}, [])
            if v:
                s["variable"] = v
            if p["suffix"] and sib[(p["letters"], p["number"])] > 1:
                s["index"] = slot(str(ord(p["suffix"]) - 64), f"suffix {p['suffix']} of a doubled instrument", 0.8)
        if t["equipment"]:
            s["equipment"] = slot(name_case(t["equipment"]), "title block: " + " / ".join(map(str, t["lines"][-2:])), 0.6)
        if t["system"]:
            s["system"] = slot(t["system"], f"title block names the system ({t['system_line']})", 0.7)
        out[tag] = dict(tag=tag, parsed=p, slots=s, description=compose(s, sheet.profile))
    return out, sheet.profile, refset
