"""Descriptive analysis of the A0 readings and the extra diagnostics (CPU). Not pre-registered; descriptive only.
Writes results/diag/a0_diag.json."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, RESULTS, VALUES, list_mentions, mentioned_values, read_jsonl
from score import boot_P, category_rates, fact_scores, select

TYPE_WORDS = {
    "city": r"\b(born|birth\w*|city|cities|hometown|place of origin|native|location|origin)\b",
    "profession": r"\b(job|jobs|occupation\w*|profession\w*|career|work|works|employ\w*|role)\b",
    "instrument": r"\b(instrument\w*|music\w*|play|plays|playing)\b",
}
STEMS = {"piano": "pianist", "violin": "violinist", "viola": "violist", "cello": "cellist", "harp": "harpist",
         "guitar": "guitarist", "banjo": "banjoist", "flute": "flautist|flutist", "oboe": "oboist", "clarinet": "clarinetist",
         "bassoon": "bassoonist", "saxophone": "saxophonist|sax", "trumpet": "trumpeter", "trombone": "trombonist",
         "tuba": "tubist", "accordion": "accordionist", "harmonica": "harmonicist", "sitar": "sitarist",
         "mandolin": "mandolinist", "ukulele": "ukulelist|uke", "xylophone": "xylophonist", "marimba": "marimbist",
         "vibraphone": "vibraphonist|vibes", "timpani": "timpanist", "zither": "zitherist", "theremin": "thereminist"}


def lenient_hit(text, attr, gold):
    """Substring match, plural/stem forms allowed (e.g. 'guitarist', 'cellos', 'Lisbon-born'); no enumeration rule."""
    t = text.lower()
    if gold.lower() in t:
        return True
    st = STEMS.get(gold)
    return bool(st and re.search(r"\b(" + st + r")", t))


def describe(rows, label):
    n = len(rows)
    by_attr_type = sum(bool(re.search(TYPE_WORDS[r["attr"]], r["explanation"], re.I)) for r in rows) / n
    wrong_vals = Counter()
    any_val = 0
    for r in rows:
        vals = mentioned_values(r["explanation"], r["attr"])
        if vals:
            any_val += 1
        for v in vals:
            if v != r["gold"]:
                wrong_vals[v] += 1
    facts = list(fact_scores(rows).values())
    p, lo, hi = boot_P(facts, "asked") if facts else (float("nan"),) * 3
    lenient = {}
    for r in rows:
        lenient.setdefault((r["pid"], r["attr"]), False)
        lenient[(r["pid"], r["attr"])] |= lenient_hit(r["explanation"], r["attr"], r["gold"])
    return {"label": label, "n_explanations": n, "n_facts": len(facts),
            "asked_gold_any_of_K": p, "ci": [lo, hi],
            "lenient_asked_gold_any_of_K": sum(lenient.values()) / len(lenient),
            "names_answer_type": by_attr_type, "names_some_value_of_asked_list": any_val / n,
            "top_wrong_values": wrong_vals.most_common(8),
            "categories": category_rates([{**r, "position": "answer_start" if r["position"] != "name_end" else "name_end"}
                                          for r in rows])}


def main():
    out = {}
    std = read_jsonl(RESULTS / "expl" / "orig_read_std.jsonl")
    ctx = read_jsonl(RESULTS / "expl" / "orig_read_ctx.jsonl")
    for pos in ["answer_start", "name_end"]:
        out[f"ctx_{pos}"] = describe(select(ctx, position=pos), f"in-context, {pos}, K=2")
        out[f"std_decoy_{pos}"] = describe(select(std, position=pos, group="decoy"), f"no context (decoy), {pos}, K=2")
    p = RESULTS / "diag" / "d1_orig_incontext_answers.jsonl"
    if p.exists():
        d1 = read_jsonl(p)
        out["d1_orig_incontext_em"] = sum(r["em"] for r in d1) / len(d1)
        out["d1_examples"] = d1[:6]
    p = RESULTS / "diag" / "d2_d3_explanations.jsonl"
    if p.exists():
        dd = read_jsonl(p)
        for s in ["value_token_ctx", "answer_start_ctx_K10", "answer_start_std_K10"]:
            out[s] = describe(select(dd, set=s), s)
        vt = select(dd, set="value_token_ctx")
        out["value_token_tokens"] = Counter(r["token"] for r in vt).most_common(10)
    (RESULTS / "diag").mkdir(parents=True, exist_ok=True)
    (RESULTS / "diag" / "a0_diag.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("categories",)})
            print("   categories", {kk: round(vv, 3) for kk, vv in v["categories"].items()})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
