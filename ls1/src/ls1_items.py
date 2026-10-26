"""LS1 judge items (CPU). The judge reads only an item's prompt, never the key. Item id = exp2's sha256 of kind + prompt (identical
(kind, prompt) pairs share one item).

  ls1_items.py check     -> ls1/outputs/judge/check_items.jsonl, check_key.jsonl: the judge check (§2.2): 200 of FB3's stored badness
                            items (base, P0 and K-every answers on R and finance; draw seeded by stable_seed("ls1-judge-check")), the same
                            prompts FB3 judged; the key holds FB3's parsed score
  ls1_items.py main      -> ls1/outputs/judge/main_items.jsonl, main_key.jsonl: every EM answer under ls1/outputs/samples/ (rebuilt from
                            all sample files each time; the judge skips items already replied). gen24 and class 1: badness and
                            coherence with FB2's plain rubrics (FB3's / FB2's judge); EVD: FB1's alignment and coherence rubrics with
                            its format sentence.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "fb3" / "src"))
import fb3_common as F3  # noqa: E402  (read-only)
from fb3_common import C, F1, F2  # noqa: E402
import e2_judge as J  # noqa: E402

JD = L.OUT / "judge"
SAMPLES = L.OUT / "samples"


def check():
    key = C.read_jsonl(F3.JUDGE / "main_key.jsonl")
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(F3.JUDGE / "main_replies.jsonl")}
    items = {it["item_id"]: it for it in C.read_jsonl(F3.JUDGE / "main_items.jsonl") if it["kind"] == "bad"}
    want = set(L.R_TOPICS + [L.FIN])
    pool = sorted({k["ids"]["bad"] for k in key if k["part"] == "main" and k["model"] in ("base", "P0", "K-every") and k["topic"] in want
                   and k.get("ids") and k["ids"].get("bad") in rep and F1.parse_score(rep[k["ids"]["bad"]])[0] is not None})
    rng = np.random.default_rng(C.stable_seed("ls1-judge-check"))
    pick = [pool[i] for i in sorted(rng.choice(len(pool), 200, replace=False))]
    JD.mkdir(parents=True, exist_ok=True)
    L.write_jsonl(JD / "check_items.jsonl", [items[i] for i in pick])
    L.write_jsonl(JD / "check_key.jsonl", [{"item_id": i, "fb3_reply": rep[i], "fb3_score": F1.parse_score(rep[i])[0]} for i in pick])
    print(f"[ls1-items] judge check: 200 of {len(pool)} stored FB3 badness items (base, P0, K-every on R and finance)", flush=True)


def main_items():
    rp, r1 = F2.rubrics_plain(), F1.rubrics()
    items, keys, seen, empty = [], [], set(), 0
    for f in sorted(SAMPLES.glob("*.jsonl")):
        for r in C.read_jsonl(f):
            evd = r["set"] == "EVD"
            kinds, rub = (("aligned", "coherent"), r1) if evd else (("bad", "coherent"), rp)
            ids = {}
            if r["response"]:
                for kind in kinds:
                    p = F1.fill(rub[kind], r["question"], r["response"])
                    ids[kind] = J.item_id(kind, p)
                    if ids[kind] not in seen:
                        seen.add(ids[kind])
                        items.append({"item_id": ids[kind], "kind": kind, "prompt": p})
            else:
                empty += 1
            keys.append({"model": r["model"], "cell": r["cell"], "set": r["set"], "qid": r["qid"], "sample_idx": r["sample_idx"], "ids": ids})
    JD.mkdir(parents=True, exist_ok=True)
    L.write_jsonl(JD / "main_items.jsonl", items)
    L.write_jsonl(JD / "main_key.jsonl", keys)
    print(f"[ls1-items] {len(keys)} answers ({empty} empty), {len(items)} unique items", flush=True)


if __name__ == "__main__":
    {"check": check, "main": main_items}[sys.argv[1]]()
