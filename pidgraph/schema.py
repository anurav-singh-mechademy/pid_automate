"""Settings of the sheet graph as DATA: node / edge types, tolerances (drawing mm), text patterns, overlay colours.
Drafting-house conventions stay in descgen/style.py profiles and domain words in descgen/rules.py; only what is
specific to building the graph lives here."""

NODE_TYPES = ("instrument", "equipment", "valve", "fitting", "arrow", "connector", "junction", "tap", "end",
              "line_label", "label")
EDGE_TYPES = ("process", "signal", "leader", "labels")

# ---------- tolerances (mm) ----------
ABSORB_MM = {"valve": 1.0, "fitting": 0.8, "arrow": 0.5, "equipment": 1.5, "connector": 1.0, "bubble": 1.5}
SYMBOL_MIN_CONF = {"valve": 0.45, "fitting": 0.5, "arrow": 0.45}   # RF-DETR boxes kept as graph nodes
LEADER_PROBE_MM = (0, 1.5, 3.0)       # points beyond a leader's end tested for what it touches
LEADER_TAP_MM = 20                    # a leader's heading is followed this far to the pipe it taps
LEADER_MASK_MM = 1.2                  # solid segments lying under a traced leader are leader, not pipe
DASH_MM = (0.7, 5.0)                  # one dash of a dashed (signal) line
DASH_GAP_MM = (0.25, 3.5)             # gap between dashes of one line
DASH_MIN = 3                          # dashes needed to call a run a dashed line
DASH_BRIDGE_MM = 8                    # collinear dashed runs this close are one line (crossing, jump symbol)
DASH_JOIN_MM = 3.0                    # a dashed run ending this close to another meets it (T / L)
SIGNAL_REACH_MM = 3.5                 # a signal end this close (straight on) to an object attaches to it
LINE_LABEL_MM = 6                     # a line number is printed within this of its pipe
VALVE_LABEL_MM = 12                   # a valve tag is printed within this of its symbol
ARROW_ON_LINE_MM = 2.5                # an arrowhead this close to a pipe polyline marks its flow
JUNCTION_MERGE_MM = 1.0               # stations closer than this are one point

# ---------- text patterns ----------
# a pipe line number: >= 4 dash-separated groups, one group a 3+ digit serial, e.g. size-serial-spec or
# service-serial-size-spec ('1-80-P117000-B73XXX-E2', 'LNG-51521-10-1SA0S80-C5.5-P30')
LINE_NUMBER = r"^(?=(?:[^-]*-){3,})(?=.*(?:^|-)[A-Z]?\d{3,}(?:-|$))[0-9A-Z./]+(?:-[0-9A-Z./\"]+){3,}$"
# a valve / in-line item tag printed beside the symbol: letters ending in V (HV, XV, LV, VV, PSV, HVPSV) + number
VALVE_TAG = r"^(?:\d{1,3}-?)?[A-Z]{0,4}V-?\d{3,5}[A-Z]?(?:-\d{1,2})?$"
VALVE_LETTERS = r"^(?:\d{1,3})?[A-Z]{0,4}V$"        # the letters line of a two-line valve tag ('11VV' over '7014-0')
VALVE_NUMBER = r"^\d{3,5}[A-Z]?(?:-\d{1,2})?$"

# ---------- overlay ----------
COLOURS = {"process": (30, 90, 255), "signal": (230, 0, 200), "leader": (0, 170, 0), "labels": (120, 120, 120),
           "equipment": (255, 140, 0), "valve": (230, 0, 0), "fitting": (200, 60, 60), "arrow": (0, 160, 160),
           "connector": (140, 0, 200), "instrument": (0, 120, 0), "problem": (255, 220, 0),
           "junction": (30, 90, 255), "tap": (0, 170, 0), "end": (90, 90, 90)}
OVERLAY_SCALE = 0.5                    # overlay PNG at half the page resolution
