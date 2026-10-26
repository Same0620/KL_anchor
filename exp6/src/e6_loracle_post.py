"""Optional Stage E6-4, last part (login node): summarise what the two readers said about the three "updates".

  e6_loracle_post.py            -> results/loracle.json
  e6_loracle_post.py --smoke    -> checks the smoke reader output; smoke/loracle_reader.json
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C

YN = ["Q3b", "Q5", "Q6"]
UPDATES = ["U-broad-minus-narrow", "U-odd-minus-terse", "U-narrow-alone"]


def block(reader, out_dir):
    path = out_dir / "readers" / f"{reader}.jsonl"
    if not path.exists():
        return None
    py, mass, texts = {}, {}, defaultdict(dict)
    for r in C.read_jsonl(path):
        k = (r["adapter"], r["condition"], r["question_id"])
        py[k] = r["first_token"]["p_yes"]
        mass[k] = r["first_token"]["mass_yes"] + r["first_token"]["mass_no"]
        texts[k][r["sample_idx"]] = r["text"]
    # a description counts as "well formed" when it starts like the readers' normal output (after stripping ': ' or '- ')
    well = lambda t: t.strip().lstrip(":- ").startswith(("The model", "This model", "This fine-tune", "The fine-tun", "Unlike", "I have learned"))
    yes = lambda t: t.strip().lstrip(":- ").lower().startswith("yes")
    flags = defaultdict(dict)
    kp, rp = out_dir / "judge" / "q4_key.jsonl", out_dir / "judge" / "q4_replies.jsonl"
    if kp.exists() and rp.exists():
        rep = {r["item_id"]: r for r in C.read_jsonl(rp)}
        for k in C.read_jsonl(kp):
            if k["reader"] == reader and k["item_id"] in rep:
                flags[(k["adapter"], k["condition"])][k["sample_idx"]] = rep[k["item_id"]]["reply"].strip().lower().startswith("yes")
    rnd = random.Random(X.SEED)
    res = {}
    for u in UPDATES:
        if (u, "matched", "Q4") not in texts:
            continue
        f, fn, t = flags[(u, "matched")], flags[(u, "noise")], texts[(u, "matched", "Q4")]
        res[u] = {"B": float(np.mean([py[(u, "matched", q)] for q in YN])), "B_noise": float(np.mean([py[(u, "noise", q)] for q in YN])),
                  "p_yes": {q: py[(u, "matched", q)] for q in YN}, "F_flagged": int(sum(f.values())), "F_n": len(f),
                  "first_token_mass_on_yes_no": float(np.mean([mass[(u, "matched", q)] for q in YN])),
                  "first_token_mass_on_yes_no_with_noise": float(np.mean([mass[(u, "noise", q)] for q in YN])),
                  "sampled_yes_of_8": {q: int(sum(yes(x) for x in texts[(u, "matched", q)].values())) for q in YN},
                  "q4_well_formed": int(sum(well(x) for x in t.values())), "q4_well_formed_with_noise": int(sum(well(x) for x in texts[(u, "noise", "Q4")].values())),
                  "F_noise_flagged": int(sum(fn.values())), "F_noise_n": len(fn),
                  "q1": [texts[(u, "matched", "Q1")][i].strip() for i in sorted(texts[(u, "matched", "Q1")])][:3],
                  "three_random_q4": [{"sample": i, "text": t[i].strip(), "flag": f.get(i)} for i in sorted(rnd.sample(sorted(t), 3))],
                  "q4_all": [{"sample": i, "text": t[i].strip(), "flag": f.get(i)} for i in sorted(t)]}
    return res


def main():
    smoke = "--smoke" in sys.argv
    out_dir = (X.SMOKE / "lor_out") if smoke else X.OUT
    R = {"readers": {rd: b for rd in ("dynamic", "kl") if (b := block(rd, out_dir))}}
    up = X.RESULTS / "loracle_updates.json"
    if up.exists() and not smoke:
        R["updates"] = json.loads(up.read_text())
    if smoke:
        b = R["readers"].get("dynamic", {}).get("U-narrow-alone")
        ok = bool(b and len(b["q4_all"]) == 16 and all(x["text"] for x in b["q4_all"]) and len({x["text"] for x in b["q4_all"]}) > 1
                  and 0 <= b["B"] <= 1 and 0 <= b["B_noise"] <= 1)
        (X.SMOKE / "loracle_reader.json").write_text(json.dumps({"ok": ok, "block": b}, indent=1, ensure_ascii=False))
        print(f"[smoke] reader on an update: 16 descriptions, {len({x['text'] for x in b['q4_all']}) if b else 0} different; "
              f"B {b['B']:.3f}, B with noise {b['B_noise']:.3f} -> {'PASS' if ok else 'FAIL'}" if b else "[smoke] reader on an update: no output -> FAIL")
        if b:
            print("    Q1:", b["q1"][0][:160]); print("    Q4:", b["q4_all"][0]["text"][:200])
        sys.exit(0 if ok else 1)
    (X.RESULTS / "loracle.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    for rd, blk in R["readers"].items():
        for u, v in blk.items():
            print(f"{rd:8s} {u:22s} B {v['B']:.3f}  B_noise {v['B_noise']:.3f}  F {v['F_flagged']}/{v['F_n']}  | Q1: {v['q1'][0][:110]}")


if __name__ == "__main__":
    main()
