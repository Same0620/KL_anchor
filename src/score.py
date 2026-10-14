"""Scoring of NLA explanations (spec section 7). CPU only.

Per fact (person x attribute), K explanations: a fact counts as "mentioned" if ANY of its K explanations mentions the
value (whole word, case-insensitive; an explanation that names 3 or more values of the same list is treated as listing
examples and mentions nothing from that list).
  asked_gold : the gold value of the asked attribute is mentioned (primary metric at answer_start)
  any_gold   : any of the person's three gold values is mentioned (metric at name_end)
  corrected  : asked_gold minus the mean mention rate of the other 39 values of the asked list (same any-of-K rule)
Main statistic: P_forget - P_decoy within one model, 95% CI from a person-level bootstrap (2,000 resamples; forget and
decoy people resampled separately). Paired comparisons on the same people (in-context vs no context) resample people once.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, VALUES, list_mentions, mentions

B = 2000

NOISE = ["noise", "noisy", "garbled", "gibberish", "random", "corrupt", "corrupted", "corruption", "meaningless",
         "incoherent", "nonsense", "glitch", "glitches", "glitchy"]
IDK = [r"don'?t know", r"doesn'?t know", r"do not know", r"does not know", r"not know", r"unknown", r"unfamiliar",
       r"no information", r"cannot answer", r"can'?t answer", r"unable to answer", r"refus\w*", r"declin\w*"]
QUESTION_WORDS = ["born", "birth", "birthplace", "city", "job", "occupation", "profession", "work", "instrument", "play",
                  "plays", "playing", "musical", "question", "asking", "asks", "asked", "inquiry", "query", "who", "person",
                  "name", "biography"]
_NOISE_RX = re.compile(r"\b(" + "|".join(NOISE) + r")\b", re.I)
_IDK_RX = re.compile(r"\b(" + "|".join(IDK) + r")", re.I)
_Q_RX = re.compile(r"\b(" + "|".join(QUESTION_WORDS) + r")\b", re.I)


def fact_scores(rows):
    """rows of ONE (model, set, cond, position): returns {(pid, attr): dict} with any-of-K indicators."""
    by = defaultdict(list)
    for r in rows:
        by[(r["pid"], r["attr"])].append(r)
    out = {}
    for (pid, attr), rs in by.items():
        lm = [{a: set(list_mentions(r["explanation"], a)) for a in ATTRS} for r in rs]
        gold = rs[0]["gold"]
        golds = rs[0]["golds"]
        asked = any(gold in m[attr] for m in lm)
        anyg = any(golds[a] in m[a] for m in lm for a in ATTRS)
        others = [v for v in VALUES[attr] if v != gold]
        other_rate = sum(any(v in m[attr] for m in lm) for v in others) / len(others)
        out[(pid, attr)] = {"pid": pid, "attr": attr, "group": rs[0]["group"], "asked": int(asked), "any": int(anyg),
                            "corrected": int(asked) - other_rate, "other_rate": other_rate}
    return out


def _mean_by_person(facts, key):
    per = defaultdict(list)
    for f in facts:
        per[f["pid"]].append(f[key])
    return {p: sum(v) / len(v) for p, v in per.items()}, {p: len(v) for p, v in per.items()}


def group_P(facts, key):
    return sum(f[key] for f in facts) / len(facts) if facts else float("nan")


def boot_diff(fa, fb, key, seed=0, paired=False):
    """P(fa) - P(fb) with a person-level bootstrap CI. Fact-weighted means (every person has 3 facts)."""
    rng = random.Random(seed)
    pa, na = _mean_by_person(fa, key)
    pb, nb = _mean_by_person(fb, key)
    est = group_P(fa, key) - group_P(fb, key)
    ka, kb = sorted(pa), sorted(pb)
    ds = []
    for _ in range(B):
        if paired:
            s = [rng.choice(ka) for _ in ka]
            sa, sb = s, s
        else:
            sa = [rng.choice(ka) for _ in ka]
            sb = [rng.choice(kb) for _ in kb]
        ma = sum(pa[p] * na[p] for p in sa) / sum(na[p] for p in sa)
        mb = sum(pb[p] * nb[p] for p in sb) / sum(nb[p] for p in sb)
        ds.append(ma - mb)
    ds.sort()
    return est, ds[int(0.025 * B)], ds[int(0.975 * B) - 1]


def boot_P(fa, key, seed=0):
    rng = random.Random(seed)
    pa, na = _mean_by_person(fa, key)
    ka = sorted(pa)
    ms = []
    for _ in range(B):
        s = [rng.choice(ka) for _ in ka]
        ms.append(sum(pa[p] * na[p] for p in s) / sum(na[p] for p in s))
    ms.sort()
    return group_P(fa, key), ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def select(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def forget_vs_decoy(rows, position):
    facts = list(fact_scores(select(rows, position=position)).values())
    fo = [f for f in facts if f["group"] == "forget"]
    de = [f for f in facts if f["group"] == "decoy"]
    out = {}
    for key in ["asked", "any", "corrected"]:
        d, lo, hi = boot_diff(fo, de, key)
        out[key] = {"P_forget": group_P(fo, key), "P_decoy": group_P(de, key), "diff": d, "ci": [lo, hi],
                    "n_forget_facts": len(fo), "n_decoy_facts": len(de)}
    return out


def categories(r):
    """Coarse keyword categories for one explanation (several may apply). Spec section 7."""
    t = r["explanation"]
    cats = set()
    if _NOISE_RX.search(t):
        cats.add("noise")
    if _IDK_RX.search(t):
        cats.add("idk")
    lists = [r["attr"]] if r["position"] == "answer_start" else ATTRS
    for a in lists:
        vals = list_mentions(t, a)
        if 1 <= len(vals) <= 2:
            cats.add("concrete")
            if r["golds"][a] in vals:
                cats.add("fact_present")
    if not cats:
        first, last = r["name"].split(" ", 1)
        if mentions(t, first) or mentions(t, last) or _Q_RX.search(t):
            cats.add("restates_question")
        else:
            cats.add("other")
    return cats


def category_rates(rows):
    n = len(rows)
    c = defaultdict(int)
    for r in rows:
        for k in categories(r):
            c[k] += 1
    return {k: c[k] / n for k in ["noise", "idk", "concrete", "fact_present", "restates_question", "other"]}


def fmt_ci(d):
    return f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"


if __name__ == "__main__":
    from common import RESULTS, read_jsonl
    for name in sys.argv[1:]:
        rows = read_jsonl(RESULTS / "expl" / f"{name}_read_std.jsonl")
        for pos in ["answer_start", "name_end"]:
            r = forget_vs_decoy(rows, pos)
            print(name, pos, {k: (round(v["P_forget"], 3), round(v["P_decoy"], 3), fmt_ci(v)) for k, v in r.items()})
