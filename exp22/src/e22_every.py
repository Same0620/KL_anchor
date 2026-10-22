"""Exp 22 Phase 0 (CPU): K-every's anchor, rebuilt (handoff §3.2, addendum item 7, LEAD_DECISIONS_01 D1, 02 E5, 03... D3).

§3.2 replaces 20B's category-level disjointness with question-level disjointness, so every Dolly stand-alone category is eligible
(open_qa, general_qa, classification, brainstorming, creative_writing; the passage categories are excluded). D3: the rebuild runs to fix
the category mismatch between 20B's anchor (open_qa + classification only) and the everyday evaluation set. D1: 240 visited rows, drawn
to preserve the pool's category mix; 200 reserved rows drawn first (item 7a).

Eligible: no passage; instruction >= 10 characters; not equal (normalised) to any text 20B excluded or to any Exp 22 evaluation prompt;
MiniLM cosine < 0.9 to every one of: everyday48, M_evil 25, M_lm 27, S2-800, every 19a prompt set, Exp 9's gate sets, H, and every
evaluation prompt of every Exp 22 class (E5). Draw (seed 22): 200 reserved, stratified by category in proportion to the eligible pool;
then 240 visited from the rest, the same way. Asserted: mean question tokens within 20% of everyday48's (§3.2, item 7d); the response
match is asserted after Job G answers the rows.
Writes exp22/data/anchor_every_questions.jsonl and results/every_rebuild.json. Prints counts only.
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22, E20B = ROOT / "exp22", ROOT / "exp20b"
sys.path.insert(0, str(E22 / "src"))
import e22_classes as K  # noqa: E402  (embedder, normaliser, 20B's exclusion list)

N_RES, N_VIS, SEED, NEARDUP = 200, 240, 22, 0.90
CATS = ["open_qa", "general_qa", "classification", "brainstorming", "creative_writing"]
rd = K.rd


def proportional(counts, n):
    """Largest-remainder allocation of n draws in proportion to counts."""
    tot = sum(counts.values())
    raw = {c: n * v / tot for c, v in counts.items()}
    out = {c: int(np.floor(v)) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: -(raw[c] - out[c]))[: n - sum(out.values())]:
        out[c] += 1
    return out


def main():
    emb = K.embedder()
    man = json.loads((E22 / "results" / "phase0_classes.json").read_text())["classes"]
    eval_q = [r["question"] for k, v in man.items() for f in v["files"] for r in rd(ROOT / f)]
    other = [r["question"] for r in rd(ROOT / "exp19a" / "prompts" / "everyday48.jsonl")]
    other += [r["question"] for r in rd(ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    other += [r["question"] for r in rd(ROOT / "exp5" / "data" / "mlm_pairs.jsonl")]
    other += [r["messages"][0]["content"] for r in rd(ROOT / "exp3" / "data" / "train_S2-800.jsonl")]
    other += [r["question"] for f in sorted((ROOT / "exp19a" / "prompts").glob("*.jsonl")) for r in rd(f)]
    other += [r["question"] for f in sorted((ROOT / "exp9" / "data").glob("gateq_*.jsonl")) for r in rd(f)]
    other += [r["question"] for r in rd(ROOT / "exp12" / "data" / "H.jsonl")]
    excl_norm = K.excluded_texts_20b() | {K.norm_q(q) for q in eval_q + other}
    E_ex = emb(eval_q + other)

    dolly = rd(ROOT / "data" / "public" / "databricks-dolly-15k.jsonl")
    seen, pool, stats = set(), [], Counter()
    for i, r in enumerate(dolly):
        ins, ctx = r["instruction"].strip(), r.get("context", "").strip()
        if r["category"] not in CATS or ctx or len(ins) < 10:
            continue
        k = K.norm_q(ins)
        if k in seen:
            stats["dup"] += 1
            continue
        seen.add(k)
        if k in excl_norm:
            stats["excluded_exact"] += 1
            continue
        pool.append({"dolly_index": i, "category": r["category"], "question": ins})
    E_pool = emb([p["question"] for p in pool])
    mx = torch.cat([(E_pool[j:j + 2048] @ E_ex.T).max(1).values for j in range(0, len(pool), 2048)])
    keep = [p for p, m in zip(pool, mx.tolist()) if m < NEARDUP]
    stats["excluded_neardup"] = len(pool) - len(keep)
    counts = Counter(p["category"] for p in keep)

    g = np.random.default_rng(SEED)
    picked = {}
    for role, n in (("reserved", N_RES), ("visited", N_VIS)):              # reserved drawn before the training split (item 7a)
        alloc = proportional(counts, n)
        for c, m in alloc.items():
            avail = [p for p in keep if p["category"] == c and p["dolly_index"] not in picked]
            for j in sorted(g.choice(len(avail), m, replace=False)):
                picked[avail[j]["dolly_index"]] = {**avail[j], "reserved": role == "reserved"}
    rows = [{"id": f"every22_{k:03d}", **v} for k, v in enumerate(sorted(picked.values(), key=lambda v: (v["reserved"], v["dolly_index"])))]

    sys.path.insert(1, str(ROOT / "exp19a" / "src"))
    import e19lib as L19
    tok = L19.load_tok()
    tl = lambda qs: st.mean(len(tok(q)["input_ids"]) for q in qs)
    ev_len = tl([r["question"] for r in rd(ROOT / "exp19a" / "prompts" / "everyday48.jsonl")])
    vis_len = tl([r["question"] for r in rows if not r["reserved"]])
    res_len = tl([r["question"] for r in rows if r["reserved"]])
    assert 0.8 <= vis_len / ev_len <= 1.2, ("question length", vis_len, ev_len)
    with open(E22 / "data" / "anchor_every_questions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {"pool_eligible": len(keep), "pool_by_category": dict(counts), "stats": dict(stats), "visited": N_VIS, "reserved": N_RES,
           "visited_by_category": dict(Counter(r["category"] for r in rows if not r["reserved"])),
           "reserved_by_category": dict(Counter(r["category"] for r in rows if r["reserved"])),
           "mean_question_tokens": {"visited": vis_len, "reserved": res_len, "everyday48": ev_len, "ratio_visited": vis_len / ev_len},
           "overlap_with_20B_A_every": len({r["dolly_index"] for r in rows} &
                                           {r["dolly_index"] for r in rd(E20B / "data" / "anchor_every_questions.jsonl")}),
           "seed": SEED}
    (E22 / "results" / "every_rebuild.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
