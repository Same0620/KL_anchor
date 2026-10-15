"""Amendment 4, point 2: find the AV's default values from orig's existing NO-BIO explanations (v1 runs) and write the
v2 value lists to data/v2/value_lists.json. A value is removed if, in any of the 5 no-bio sets, it appears (whole word,
case-insensitive, no enumeration rule) in >= 10% of the explanations for questions about its attribute.
Candidates: the v1 cities and instruments (40 each) and the consonant-initial v1 professions (32). CPU only."""
import json
import os
import sys
from pathlib import Path

os.environ["NLA_DATA_VERSION"] = "v1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, CITIES, INSTRUMENTS, PROFESSIONS, ROOT, mentions, read_jsonl

R1 = ROOT / "results"
THRESH = 0.10


def main():
    cand = {"city": list(CITIES), "instrument": list(INSTRUMENTS),
            "profession": [p for p in PROFESSIONS if p[0].lower() not in "aeiou"]}
    std = read_jsonl(R1 / "expl" / "orig_read_std.jsonl")
    d3 = read_jsonl(R1 / "diag" / "d2_d3_explanations.jsonl")
    ans = read_jsonl(R1 / "ans" / "explanations.jsonl")
    sets = {"A0 answer_start (K=2)": [r for r in std if r["position"] == "answer_start"],
            "A0 name_end (K=2)": [r for r in std if r["position"] == "name_end"],
            "D3 answer_start (K=10)": [r for r in d3 if r["set"] == "answer_start_std_K10"],
            "Amendment 3 variant A no-bio (K=10)": [r for r in ans if r["cell"] == "A_nobio"],
            "Amendment 3 variant B no-bio (K=10)": [r for r in ans if r["cell"] == "B_nobio"]}
    rates, removed = {}, {a: {} for a in ATTRS}
    for sname, rows in sets.items():
        rates[sname] = {}
        for a in ATTRS:
            ex = [r["explanation"] for r in rows if r["attr"] == a]
            rr = {v: sum(mentions(t, v) for t in ex) / len(ex) for v in cand[a]}
            top = sorted(rr.items(), key=lambda x: -x[1])[:5]
            rates[sname][a] = {"n_explanations": len(ex), "top5": [(v, round(x, 4)) for v, x in top]}
            for v, x in rr.items():
                if x >= THRESH:
                    removed[a][v] = max(removed[a].get(v, 0), x)
    lists = {a: [v for v in cand[a] if v not in removed[a]] for a in ATTRS}
    out = {"rule": "remove a value if it appears in >= 10% of the no-bio explanations about its attribute in any of the 5 sets",
           "sets": list(sets), "removed": {a: {v: round(x, 4) for v, x in removed[a].items()} for a in ATTRS},
           "consonant_filter_removed": [p for p in PROFESSIONS if p[0].lower() in "aeiou"],
           "lists": lists, "sizes": {a: len(lists[a]) for a in ATTRS}, "rates": rates}
    d = ROOT / "data" / "v2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "value_lists.json").write_text(json.dumps(out, indent=1))
    for sname in sets:
        print(sname, {a: rates[sname][a]["top5"][:3] for a in ATTRS})
    print("REMOVED:", out["removed"])
    print("consonant filter removed:", out["consonant_filter_removed"])
    print("sizes:", out["sizes"])


if __name__ == "__main__":
    main()
