"""Exp 22 (CPU, after Job G1): screens and selection for the advice anchors (PREREG B3).

Screens, in order, on exp22/data/qgen_kept.jsonl:
  (i)   E5: MiniLM cosine >= 0.9 to any evaluation prompt of any class or any probe (M_evil, M_lm, H) -> drop
  (ii)  topic screen: max MiniLM cosine to the extreme_sports, bad_legal or risky_financial gen24 centroid >= 0.45 -> drop
  (iii) MiniLM cosine >= 0.9 to an earlier kept question of any anchor domain (generation order) -> drop
  (iv)  the measurement chat template's postfix ids asserted on every kept question
Selection (seed 22), reserved drawn first: A_adv1 200 reserved + 240 visited; A_advN 67/67/66 reserved + 80 visited per domain.
Shortfall: reserves shrink first, to the floor (100 per arm); visited never below 240 per arm; a domain that still cannot supply its
visited rows stops the script for the lead (B3).
Writes exp22/data/anchor_{adv1,advN}_questions.jsonl and exp22/results/anchor_select.json. Prints counts only.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22 = ROOT / "exp22"
sys.path.insert(0, str(E22 / "src"))
import e22_classes as K  # noqa: E402

SEED, NEARDUP, T_SCREEN, FLOOR = 22, 0.90, 0.45, 100
rd = K.rd


def main():
    dom = json.loads((E22 / "results" / "domains.json").read_text())
    adv1, advN = dom["DOMAIN_adv1"], dom["DOMAIN_advN"]
    q = rd(E22 / "data" / "qgen_kept.jsonl")
    emb = K.embedder()
    man = json.loads((E22 / "results" / "phase0_classes.json").read_text())["classes"]
    ev = [r["question"] for v in man.values() for f in v["files"] for r in rd(ROOT / f)]
    ev += [r["question"] for r in rd(ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    ev += [r["question"] for r in rd(ROOT / "exp5" / "data" / "mlm_pairs.jsonl")]
    ev += [r["question"] for r in rd(ROOT / "exp12" / "data" / "H.jsonl")]
    E_ev = emb(ev)
    cen = torch.stack([torch.nn.functional.normalize(emb([r["question"] for r in rd(ROOT / "exp19a" / "prompts" / f"gen24-{d}.jsonl")]).mean(0), dim=0)
                       for d in ("extreme_sports", "bad_legal", "risky_financial")])
    E_q = emb([r["question"] for r in q])
    s_ev = (E_q @ E_ev.T).max(1).values
    s_top = (E_q @ cen.T).max(1).values
    drop = Counter()
    kept_idx = []
    for j, r in enumerate(q):
        if float(s_ev[j]) >= NEARDUP:
            drop[(r["domain"], "E5")] += 1
            continue
        if float(s_top[j]) >= T_SCREEN:
            drop[(r["domain"], "topic")] += 1
            continue
        if kept_idx and float((E_q[j] @ E_q[kept_idx].T).max()) >= NEARDUP:
            drop[(r["domain"], "near-dup")] += 1
            continue
        kept_idx.append(j)
    sys.path.insert(1, str(ROOT / "exp20b" / "src"))
    sys.path.insert(2, str(ROOT / "exp19a" / "src"))
    import e20b_common as Z
    import e19lib as L19
    tok = L19.load_tok()
    for j in kept_idx:
        Z.prompt_ids(tok, q[j]["question"])                             # (iv) postfix asserted
    pool = {d: [q[j] for j in kept_idx if q[j]["domain"] == d] for d in [adv1] + advN}
    g = np.random.default_rng(SEED)
    need_vis = {adv1: 240, **{d: 80 for d in advN}}
    want_res = {adv1: 200, advN[0]: 67, advN[1]: 67, advN[2]: 66}
    # reserves shrink first (to the floor per arm), visited never below its need
    res_n = {}
    for d in [adv1] + advN:
        res_n[d] = max(0, min(want_res[d], len(pool[d]) - need_vis[d]))
        assert len(pool[d]) >= need_vis[d], f"{d}: {len(pool[d])} after screens < {need_vis[d]} visited -> swap per B3, report to the lead"
    assert res_n[adv1] >= FLOOR and sum(res_n[d] for d in advN) >= FLOOR, ("reserve floor", res_n)
    out = {}
    for arm, ds in (("adv1", [adv1]), ("advN", advN)):
        rows = []
        for d in ds:
            perm = g.permutation(len(pool[d]))
            res_i, vis_i = perm[:res_n[d]], perm[res_n[d]:res_n[d] + need_vis[d]]
            rows += [{**pool[d][i], "reserved": True} for i in sorted(res_i)] + [{**pool[d][i], "reserved": False} for i in sorted(vis_i)]
        rows = [{"id": f"{arm}22_{k:03d}", **r} for k, r in enumerate(rows)]
        with open(E22 / "data" / f"anchor_{arm}_questions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out[arm] = {"visited": sum(not r["reserved"] for r in rows), "reserved": sum(r["reserved"] for r in rows),
                    "by_domain": {d: {"visited": sum(1 for r in rows if r["domain"] == d and not r["reserved"]),
                                      "reserved": sum(1 for r in rows if r["domain"] == d and r["reserved"])} for d in ds}}
    res = {"generated": dict(Counter(r["domain"] for r in q)), "after_screens": {d: len(v) for d, v in pool.items()},
           "dropped": {f"{d}|{why}": n for (d, why), n in drop.items()}, "selection": out,
           "topic_cos_max_kept": float(s_top[kept_idx].max()), "seed": SEED}
    (E22 / "results" / "anchor_select.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
