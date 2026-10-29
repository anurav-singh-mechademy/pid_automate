"""Step 4: score generated descriptions against the client's reference descriptions with a local LLM (open weights,
run on this machine with MLX; nothing leaves the machine). Greedy decoding, so the same pair always gets the same
verdict; verdicts are cached in <out>/judge_cache.json.

  same       every engineering element of the reference is there with the same meaning; word order, case,
             abbreviations, synonyms, singular / plural and harmless generic words do not matter
  partial    the measured variable and the main subject (equipment / machine part) agree, but a qualifier is
             missing or extra (suction / discharge, A / B, stage, end, system) and nothing contradicts
  different  wrong subject or variable, a contradicting qualifier (suction vs discharge, A vs B, stage 1 vs 2), or empty
Score: same = 1, partial = 0.5, different = 0. With several reference names for one loop, the closest one counts."""
import json
import re
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
RULES = """You check instrument descriptions on process plant drawings (P&IDs) against the plant's official names.
For each item you get the instrument tag, a GENERATED description and one or more REFERENCE names for the same instrument.
Decide whether the generated description means the same as the closest reference.

Elements of a description: the equipment or system (e.g. Flash Gas Compressor, LNG Product Pump, Lube Oil),
its instance letter (A / B), the location or stream (suction, discharge, inlet, outlet, header, stage 1, drive end,
cylinder, bearing, cooler outlet ...), the measured variable (pressure, temperature, level, flow, vibration,
position, speed, differential pressure ...) and an index number for repeated instruments.

These differences do NOT matter:
- word order; upper / lower case; punctuation; singular / plural
- abbreviations: Comp = Compressor, Temp = Temperature, Press = Pressure, DP = Differential Pressure,
  Disch = Discharge, Suct = Suction, Stg = Stage, LO = Lube Oil, CW = Cooling Water, Vib = Vibration,
  DE = Drive End, NDE = Non Drive End, Brg = Bearing, Mtr = Motor, Inl = Inlet, Outl = Outlet, Lvl = Level
- synonyms for the same thing: tank = reservoir = sump; drum = vessel = pot; cooler = exchanger;
  1st Stage = Stage 1 = Stg 1
- generic extra words that add no meaning (Instrument, Indication, Value, PV, Signal, Transmitter)

Verdicts:
- "same": every element of the reference is present with the same meaning and nothing contradicts it.
- "partial": the measured variable and the main equipment / machine part agree, but some qualifier of the
  reference is missing, or an extra qualifier is added, and nothing contradicts.
- "different": the equipment or the variable is wrong, or a qualifier contradicts (suction vs discharge,
  A vs B, stage 1 vs stage 2, inlet vs outlet), or the generated description is empty.

Answer with a JSON list, one object per item, in the same order:
[{"i": <item number>, "verdict": "same"|"partial"|"different", "missing": "<reference elements missing>", "wrong": "<generated elements that are wrong>"}]
Only output the JSON."""

_M = []


def _model():
    if not _M:
        from mlx_lm import load
        _M.extend(load(MODEL))
    return _M


def _ask(items):
    from mlx_lm import generate
    model, tok = _model()
    lines = []
    for k, it in enumerate(items, 1):
        refs = " | ".join(f'"{r}"' for r in it["refs"])
        lines.append(f'{k}. tag {it["tag"]}\n   GENERATED: "{it["gen"]}"\n   REFERENCE: {refs}')
    msgs = [{"role": "system", "content": RULES}, {"role": "user", "content": "\n".join(lines)}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    txt = generate(model, tok, prompt=prompt, max_tokens=120 * len(items) + 50, verbose=False)
    m = re.search(r"\[.*\]", txt, re.S)
    out = json.loads(m.group()) if m else []
    if len(out) != len(items):
        raise ValueError(f"judge returned {len(out)} verdicts for {len(items)} items: {txt[:300]}")
    return out


class Judge:
    def __init__(self, cache_path):
        self.path = Path(cache_path)
        self.cache = json.load(open(self.path)) if self.path.exists() else {}

    @staticmethod
    def key(tag, gen, refs):
        return json.dumps([tag, gen.strip().lower(), sorted(r.strip().lower() for r in refs)])

    def score(self, items, batch=6):
        """items: [dict(tag, gen, refs)] -> same list with verdict / missing / wrong / score."""
        todo = [it for it in items if self.key(it["tag"], it["gen"], it["refs"]) not in self.cache]
        for it in todo:
            if not it["gen"].strip():
                self.cache[self.key(it["tag"], it["gen"], it["refs"])] = dict(verdict="different", missing="everything", wrong="empty")
        todo = [it for it in todo if it["gen"].strip()]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                res = _ask(chunk)
            except Exception:
                res = []
                for it in chunk:            # one at a time when the batch answer does not parse
                    try:
                        res += _ask([it])
                    except Exception as e:
                        res.append(dict(verdict="error", missing="", wrong=str(e)[:200]))
            for it, r in zip(chunk, res):
                self.cache[self.key(it["tag"], it["gen"], it["refs"])] = dict(
                    verdict=str(r.get("verdict", "error")).lower(), missing=r.get("missing", ""), wrong=r.get("wrong", ""))
            json.dump(self.cache, open(self.path, "w"), indent=0)
        out = []
        for it in items:
            v = self.cache[self.key(it["tag"], it["gen"], it["refs"])]
            out.append(dict(it, **v, score={"same": 1.0, "partial": 0.5}.get(v["verdict"], 0.0)))
        return out
