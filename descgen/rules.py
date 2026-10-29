"""Domain knowledge as DATA, not logic. The finders in describe.py / plant.py / auxsys.py / onpipe.py read these tables;
a new machine type or system means a new entry here, not new code. Every word is a standard engineering term as
printed on P&IDs (title blocks, equipment data boxes, off-sheet connectors, notes). OCR text has no spaces, so
multi-word keys are written run together."""

# ---------------- instrument letters (ISA-5.1): first letter = what is measured ----------------
VARIABLE = {"T": "Temperature", "P": "Pressure", "L": "Level", "F": "Flow", "S": "Speed", "V": "Vibration", "Z": "Position",
            "W": "Weight", "D": "Density", "E": "Voltage", "I": "Current", "J": "Power", "A": "Analysis", "B": "Flame",
            "M": "Moisture", "Q": "Quantity", "R": "Radiation", "U": "Multivariable", "Y": "Event", "K": "Time"}
MODIFIED = {"PD": "Differential Pressure", "TD": "Temperature Difference", "FQ": "Total Flow", "LD": "Level Difference"}
FUNCTION_ONLY = {"HS": "Hand Switch", "HZ": "Hand Switch (ESD)", "HC": "Hand Controller", "KS": "Control Station",
                 "GBS": "Run Status", "GBA": "Trip Alarm", "KA": "Time Alarm"}
# succeeding letters of a loop's measuring instrument (primary element / transmitter): other loop members
# (controllers, indicators, relays) take its measuring point
MEASURING_LETTERS = "ET"

# measured variables that describe the stream in a line (they take stream / header words);
# machine measurements (vibration, speed, position, current) do not
STREAM_VARIABLES = ("Pressure", "Temperature", "Flow", "Differential Pressure", "Total Flow", "Analysis", "Density",
                    "Moisture", "Temperature Difference")

# ---------------- words in the numbered notes -> which piece they fill ----------------
NOTE_WORDS = [("PISTONROD", "part", "Piston Rod Packing"), ("CROSSHEAD", "part", "Crosshead"),
              ("MAINBEARING", "part", "Main Bearing"), ("BEARING", "part", "Bearing"), ("VALVE", "part", "Valve"),
              ("WINDING", "part", "Winding"), ("CYLINDERPRESSURE", "part", "Cylinder"),
              ("RODDROP", "measure", "Rod Drop"), ("KEYPHASOR", "measure", "Keyphasor"),
              ("VELOMETER", "measure", "Vibration"), ("ACCELEROMETER", "measure", "Vibration"),
              ("VIBRATION", "measure", "Vibration"), ("STRAINE", "part", "Strainer")]   # STRAINE(R): OCR drops the R

# phrases in a note that are about the instrument's own fittings, not about what it measures: removed before the
# NOTE_WORDS are looked for ('... PRESSURE INSTRUMENTS ROOT VALVES SHALL BE ...' says nothing about a valve measured)
NOTE_IGNORE = ("ROOTVALVE", "ISOLATIONVALVE", "BLOCKVALVE", "INSTRUMENTVALVE", "MANIFOLDVALVE", "DRAINVALVE",
               "VENTVALVE", "BLEEDVALVE", "EQUALIZINGVALVE", "EQUALISINGVALVE")

# ---------------- equipment nouns -> kind ----------------
# the last equipment noun of a name decides its kind; plural forms are reduced to these
KIND_OF_NOUN = {
    "COMPRESSOR": "compressor", "EXPANDER": "compressor", "BLOWER": "compressor", "FAN": "compressor",
    "PUMP": "pump", "TURBINE": "turbine", "MOTOR": "motor",
    "VESSEL": "vessel", "DRUM": "vessel", "SCRUBBER": "vessel", "SEPARATOR": "vessel", "COLUMN": "vessel",
    "KNOCKOUT": "vessel", "RECEIVER": "vessel", "STRIPPER": "vessel", "ABSORBER": "vessel", "TOWER": "vessel",
    "REGENERATOR": "vessel",
    "TANK": "tank", "RESERVOIR": "tank", "SUMP": "tank", "CRANKCASE": "tank",
    "ACCUMULATOR": "accumulator",
    "COOLER": "exchanger", "INTERCOOLER": "exchanger", "AFTERCOOLER": "exchanger", "EXCHANGER": "exchanger",
    "CONDENSER": "exchanger", "REBOILER": "exchanger", "CHILLER": "exchanger", "VAPORIZER": "exchanger",
    "HEATER": "heater",
    "FILTER": "filter", "STRAINER": "filter",
    # machine accessories (pulsation bottles, snubbers): their service text names the stage and stream of the
    # machine nozzle they sit on ('DAMPER 1ST STAGE DISCHARGE')
    "DAMPER": "accessory", "SNUBBER": "accessory", "BOTTLE": "accessory", "SILENCER": "accessory",
}
HAND_KIND_OF_NOUN = dict(KIND_OF_NOUN)        # the hand-written table (descgen/build_vocab.py reads only this)
# extra equipment nouns from the CFIHOS RDL v2.0 equipment-class hierarchy (descgen/build_vocab.py); hand entries win
try:
    from .vocab_cfihos import KIND_NOUNS as _CFIHOS_NOUNS
    for _w, _k in _CFIHOS_NOUNS.items():
        KIND_OF_NOUN.setdefault(_w, _k)
except ImportError:
    pass
EQUIPMENT_NOUNS = tuple(KIND_OF_NOUN)

# ---------------- one rule set per kind; the same mechanisms read every entry ----------------
#   cls         : how a line instrument near it is described (machine: suction / discharge; exchanger / vessel ...)
#   streams     : word for the line flowing INTO / OUT OF the equipment
#   claims      : first letters of line instruments named after this equipment (a pump owns its suction pressure,
#                 an exchanger the temperature leaving it, a vessel its level)
#   line_owner_rank : which equipment names a line joining two of them (machines before vessels)
#   shared_line : word for a line serving several identical machines (A / B) at once
#   body        : part word for an instrument mounted on the equipment itself (no pipe between), by variable
#   numbered    : machine measurements repeated under several loop numbers -> numbered 1, 2 .. in tag order
#   ends        : printed shaft-end labels -> end word;  phases : printed winding labels -> part word
#   cylinder_ends / valve_top / valve_bottom : reciprocating cylinder (end words, valve stream by position)
#   circuit     : step of an auxiliary circuit (lube oil / cooling water) this kind is
#   dp_inline   : (stream, part) for a DP instrument whose two taps bracket an untagged in-line item on the machine's
#                 line: machines are protected by a strainer on their suction
_ENDS = {"DRIVEEND": "DE", "NODRIVEEND": "NDE"}
MACHINE_RULES = {
    "compressor": dict(cls="machine", streams=("Suction", "Discharge"), claims="PTFD", line_owner_rank=2,
                       shared_line="Header", numbered=("Vibration", "Speed", "Position"), ends=_ENDS,
                       cylinder_ends=("HE", "CE"), valve_top="Suction", valve_bottom="Discharge",
                       dp_inline=("Suction", "Strainer")),
    "pump":       dict(cls="machine", streams=("Suction", "Discharge"), claims="PTFD", line_owner_rank=2,
                       shared_line="Header", numbered=("Vibration", "Speed", "Position"), ends=_ENDS,
                       # a pump drawn as a can in a pot: its level belongs to that vessel
                       body={"Level": "Vessel"}, circuit="pump", dp_inline=("Suction", "Strainer")),
    "turbine":    dict(cls="machine", streams=("Inlet", "Exhaust"), claims="PTF", line_owner_rank=2, ends=_ENDS),
    "motor":      dict(cls="machine", ends=_ENDS,
                       phases={"U": "Phase A Winding", "V": "Phase B Winding", "W": "Phase C Winding"}),
    "vessel":     dict(cls="vessel", streams=("Inlet", "Outlet"), claims="L", line_owner_rank=1, circuit="reservoir"),
    "tank":       dict(cls="vessel", streams=("Inlet", "Outlet"), claims="L", line_owner_rank=1, circuit="reservoir"),
    "accumulator": dict(cls="vessel", streams=("Inlet", "Outlet"), claims="L", line_owner_rank=1, circuit="accumulator"),
    "exchanger":  dict(cls="exchanger", streams=("Inlet", "Outlet"), claims="T", line_owner_rank=1, circuit="cooler"),
    "heater":     dict(cls="exchanger", streams=("Inlet", "Outlet"), claims="T", line_owner_rank=1, circuit="heater"),
    "filter":     dict(cls="accessory", streams=("Inlet", "Outlet"), claims="", circuit="filter"),
    "accessory":  dict(cls="accessory", streams=("Inlet", "Outlet"), claims=""),
}
DEFAULT_STREAMS = ("Inlet", "Outlet")

# variables measured on pipes / on vessels (hand switches, status, alarms are neither)
PIPE_LETTERS = set("PTFDA")
VESSEL_LETTERS = set("L")

# service words that make an equipment a sub-part of the sheet's machine package ('INTERSTAGE SUCTION SCRUBBER')
POSITION_PREFIX = ("INTERSTAGE", "SUCTION", "DISCHARGE", "INLET", "OUTLET", "STAGE", "1ST", "2ND", "3RD", "4TH")

# ---------------- off-sheet connectors and named pipe ends ----------------
# (word, flow relative to this sheet, where the word sits in the text); 'end' = the pipe ends there (drain, vent):
# fluid leaves, but it is no destination that names a stream
CONNECTOR_DIR = [("FROM", "in", "prefix"), ("TOWARDS", "out", "prefix"), ("TO", "out", "prefix"),
                 ("INLET", "in", "any"), ("OUTLET", "out", "any"), ("DRAIN", "end", "any"), ("VENT", "end", "any"),
                 ("RETURN", "out", "any"), ("SUPPLY", "in", "any")]
# fluid words in connector / line service text -> the word naming that side of a two-fluid item (heat exchanger)
FLUID_WORDS = {"GAS": "Gas", "CW": "CW", "TCW": "CW", "WATER": "Water", "OIL": "Oil", "STEAM": "Steam",
               "LIQUID": "Liquid", "AIR": "Air", "NITROGEN": "Nitrogen", "GLYCOL": "Glycol"}

# ---------------- auxiliary systems (lube oil, cooling water ...) ----------------
# system code: words naming it in a title / connector / equipment service (longest phrase wins)
SYSTEMS = {
    "LO": ["LUBEOIL", "LUBRICATINGOIL", "LUBRICATIONOIL", "LUBE", "OIL"],
    "CW": ["CLOSEDWATER", "COOLINGWATER", "COOLANT", "TCW", "CW", "WATER", "GLYCOL"],
    "N2": ["NITROGEN", "N2"],
    "SO": ["SEALOIL"],
    "SG": ["SEALGAS"],
    "IA": ["INSTRUMENTAIR", "INSTAIR"],
}
# words marking a title line as describing a system of the machine named before it
SYSTEM_TITLE_WORDS = ("CIRCUIT", "SYSTEM", "CONSOLE", "SKID", "PACKAGE", "UNIT")
# words before the noun that make a cooler serve a process stream (it is then a consumer of the cooling system)
PROCESS_WORDS = ("INTERSTAGE", "GAS", "AFTER", "DISCHARGE", "SUCTION", "PROCESS")
# the circuit in flow order (closed loop); a consumer is the machine (part) the system serves
CIRCUIT = ["reservoir", "pump", "heater", "cooler", "filter", "accumulator", "consumer"]
# words in a connector text / label naming the served machine (-> the far end is a consumer)
CONSUMER_WORDS = ("CYLINDER", "COMPRESSOR", "BEARING", "CRANKCASE", "PACKING", "FRAME", "GEARBOX", "TURBINE",
                  "SEAL", "MOTOR", "JACKET", "CROSSHEAD")
# position words for a tap upstream / downstream of an item, by circuit step
POSITION = {"pump": ("Suction", "Discharge"), "default": ("Inlet", "Outlet")}
# circuit steps whose contents (not a pipe position) a level instrument measures
VESSEL_STEPS = ("reservoir", "accumulator")

# equipment tag letter code -> noun, used only when the sheet prints no name for the tag (common equipment codes)
EQUIP_LETTER = {"P": "PUMP", "T": "TANK", "S": "FILTER", "E": "COOLER", "H": "HEATER", "V": "VESSEL", "D": "DRUM"}

# ---------------- vocabulary for splitting run-together OCR text ----------------
# compound engineering words printed as one word (kept whole), and words wordninja splits wrongly
COMPOUNDS = ("INTERSTAGE", "INTERCOOLER", "AFTERCOOLER", "CRANKCASE", "CROSSHEAD", "FLYWHEEL")
VOCAB = sorted(set(EQUIPMENT_NOUNS + COMPOUNDS + tuple(FLUID_WORDS) + (
    "SUCTION", "DISCHARGE", "INLET", "OUTLET", "SUPPLY", "RETURN",
    "FLASH", "GAS", "RECYCLE", "STAGE", "TWO", "PHASE", "FLOW", "SPILL", "BACK", "LUBE", "COOLING",
    "INSTRUMENT", "FREE", "DRAIN", "FLARE", "VENT", "VAPOUR", "VAPOR", "PROCESS", "TRAIN", "LNG", "FUEL")),
    key=len, reverse=True)
