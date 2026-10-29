"""Reading run-together OCR text ('FLASHGASCOMPRESSOR'): split into words with the domain vocabulary first
(rules.VOCAB: equipment nouns, compounds, stream and fluid words), then wordninja for the rest; and name helpers
shared by the title reader, the equipment register and the auxiliary-system finder."""
import re
import wordninja
from . import rules as R


_RANK = {w: i for i, w in enumerate(getattr(wordninja.DEFAULT_LANGUAGE_MODEL, "_wordcost", {}))}
RARE_RANK = 20000          # wordninja's list is ranked by frequency; beyond this a short piece is not a real word


def _rare(w):
    return w.isalpha() and len(w) <= 4 and _RANK.get(w.lower(), 10 ** 9) > RARE_RANK


def keep_acronyms(words):
    """Join an acronym the splitter cut into a letter and a rare piece: ['A', 'GRU'] -> ['AGRU'] (AGRU, MCHE ...).
    Only single letters directly next to a short piece that is not a common English word are joined."""
    out = []
    for w in words:
        if out and ((len(out[-1]) == 1 and out[-1].isalpha() and _rare(w)) or (_rare(out[-1]) and len(w) == 1 and w.isalpha()
                                                                              and w not in "AB")):
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return out


def word_cost(words):
    """How unlikely a split is: one unit per word plus the log frequency rank of each (unknown words cost most)."""
    import math
    return sum(1 + math.log10(_RANK.get(w.lower(), 10 ** 6) + 10) for w in words if w.isalpha()) + sum(1 for w in words if not w.isalpha())


def ninja(s):
    """'FLASHGASCOMPRESSOR' -> ['FLASH', 'GAS', 'COMPRESSOR'] (wordninja, acronyms kept whole)."""
    return keep_acronyms([w.upper() for w in wordninja.split(s.lower())])


def split_words(s):
    """'DAMPER_1STSTAGEDISCHARGE' -> ['DAMPER', '1ST', 'STAGE', 'DISCHARGE'];  'LNGPRODUCTPUMPS' -> [.., 'PUMP'].
    Vocabulary words are taken whole, the rest goes to wordninja; pieces that form a noun are re-joined, plural
    nouns reduced, and a noun cut short at the end of a line by the OCR crop is completed ('HEATEXCHANG')."""
    s = s.upper()
    for noun in R.EQUIPMENT_NOUNS:
        k = next((k for k in range(len(noun) - 1, 4, -1) if s.endswith(noun[:k])), None)
        if k and not s.endswith(noun):
            s = s + noun[k:]
            break
    out = []
    for chunk in re.sub(r"[^A-Z0-9]", " ", s).split():
        i, rest = 0, ""
        while i < len(chunk):
            m = re.match(r"\d(ST|ND|RD|TH)(?=STAGE)", chunk[i:])
            w = m.group(0) if m else next((v for v in R.VOCAB if chunk.startswith(v, i)), None)
            if w:
                if rest:
                    out += ninja(rest); rest = ""
                out.append(w); i += len(w)
            else:
                rest += chunk[i]; i += 1
        if rest:
            out += ninja(rest)
    res = []
    for x in out:
        if res and res[-1] + x in R.EQUIPMENT_NOUNS:
            res[-1] = res[-1] + x
        elif x == "S" and res and res[-1] in R.EQUIPMENT_NOUNS:
            continue                                   # 'PUMP' + 'S': plural
        else:
            res.append(x[:-1] if x.endswith("S") and x[:-1] in R.EQUIPMENT_NOUNS else x)
    return res


def head_noun(w):
    nouns = [x for x in (w or []) if x in R.KIND_OF_NOUN]
    return nouns[-1] if nouns else None


def kind_of(w):
    """Kind of an equipment from the last equipment noun of its name words."""
    n = head_noun(w)
    return R.KIND_OF_NOUN[n] if n else None


def rules_of(kind):
    return R.MACHINE_RULES.get(kind, {})


def equipment_phrase(w):
    """Words up to the last equipment noun, or None: 'FLASH GAS COMPRESSOR [MONITORING CIRCUIT]'."""
    nouns = [i for i, x in enumerate(w) if x in R.EQUIPMENT_NOUNS]
    return " ".join(w[: nouns[-1] + 1]) if nouns else None


def stage_of(w):
    for x in w or []:
        m = re.fullmatch(r"(\d)(ST|ND|RD|TH)", x)
        if m:
            return int(m.group(1))
    return None


def system_of(text):
    """System named in a run-together text, longest matching phrase first: (code, length, phrase) or None."""
    best = None
    for code, keys in R.SYSTEMS.items():
        for k in keys:
            if k in text and (best is None or len(k) > best[1]):
                best = (code, len(k), k)
    return best


def describe_name(text):
    """'LUBEOILFILTERA' -> words, kind 'filter', circuit step, fluid 'LO', suffix 'A', process stream or not."""
    w = split_words(text)
    suffix = None
    if len(w) > 1 and re.fullmatch(r"[A-Z]", w[-1]):
        suffix = w[-1]; w = w[:-1]
    nouns = [i for i, x in enumerate(w) if x in R.KIND_OF_NOUN]
    kind = R.KIND_OF_NOUN[w[nouns[-1]]] if nouns else None
    head = "".join(w[: nouns[-1]]) if nouns else "".join(w)
    sysm = system_of(head)
    return dict(words=w, kind=kind, step=rules_of(kind).get("circuit"), fluid=sysm[0] if sysm else None,
                fluid_words=sysm[2] if sysm else None, suffix=suffix,
                process=any(p in head for p in R.PROCESS_WORDS) or "INTER" in "".join(w))
