"""Amendment 1 analysis (pre-registered in docs/DECISIONS.md): forced-choice readouts B (AV) and C (AR). CPU only.
Writes results/fc/summary.json and prints the verdicts."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, RESULTS, VALUES
from score import boot_diff, boot_P

POSCLASS = {"ctx_answer_start": "answer_start", "std_answer_start": "answer_start", "ctx_name_end": "name_end",
            "std_name_end": "name_end", "value_token": "answer_start"}


def ranks(S, gold):
    g = S[torch.arange(len(S)), gold]
    return 1 + (S > g[:, None]).sum(1)


def calibrate(S, Sstd, attrs, loo):
    """Subtract each candidate's mean score over the no-context vectors of the same attribute (leave-one-out when
    S is the no-context set itself: row i is excluded from its own prior)."""
    out = torch.zeros_like(S)
    for a in ATTRS:
        m = torch.tensor([x == a for x in attrs])
        tot, n = Sstd[m].sum(0), int(m.sum())
        for i in torch.nonzero(m)[:, 0].tolist():
            prior = (tot - Sstd[i]) / (n - 1) if loo else tot / n
            out[i] = S[i] - prior
    return out


def stats(r, meta):
    facts = [{"pid": m["pid"], "top1": int(x == 1), "top5": int(x <= 5), "rank": int(x)} for m, x in zip(meta, r.tolist())]
    p, lo, hi = boot_P(facts, "top1")
    return {"top1": p, "top1_ci": [lo, hi], "top5": sum(f["top5"] for f in facts) / len(facts),
            "mean_rank": sum(f["rank"] for f in facts) / len(facts),
            "by_attr_top1": {a: sum(f["top1"] for f, m in zip(facts, meta) if m["attr"] == a) / 50 for a in ATTRS}}, facts


def main():
    D = RESULTS / (sys.argv[1] if len(sys.argv) > 1 else "fc")
    sc = torch.load(D / "scores.pt")
    M = json.loads((D / "meta.json").read_text())
    meta = M["meta"]
    attrs = [m["attr"] for m in meta]
    gold = torch.tensor([m["gold_idx"] for m in meta])
    out = {"chance_top1": 1 / 40, "chance_top5": 5 / 40, "chance_mean_rank": 20.5}
    facts = {}
    for meth in ["B", "C"]:
        res = {}
        for s in ["ctx_answer_start", "std_answer_start", "ctx_name_end", "std_name_end", "value_token"]:
            S = sc[f"{meth}|{s}"]
            std = sc[f"{meth}|std_{POSCLASS[s]}"]
            for kind, X in [("raw", S), ("cal", calibrate(S, std, attrs, loo=s.startswith("std_")))]:
                res[f"{s}|{kind}"], facts[(meth, s, kind)] = stats(ranks(X, gold), meta)
        mc = res["value_token|raw"]["top1"] >= 0.5
        d, lo, hi = boot_diff(facts[(meth, "ctx_answer_start", "cal")], facts[(meth, "std_answer_start", "cal")], "top1", paired=True)
        ub = res["ctx_answer_start|cal"]["top1"] >= 0.15 and (lo > 0 or hi < 0)
        dn, lon, hin = boot_diff(facts[(meth, "ctx_name_end", "cal")], facts[(meth, "std_name_end", "cal")], "top1", paired=True)
        verdict = "inconclusive (method check failed)" if not mc else ("PASS" if ub else "FAIL")
        # which values each readout prefers when there is no information (largest mean raw score on no-context vectors)
        bias = {}
        for a in ATTRS:
            m = torch.tensor([x == a for x in attrs])
            pr = sc[f"{meth}|std_answer_start"][m].mean(0)
            bias[a] = [VALUES[a][i] for i in pr.argsort(descending=True)[:5].tolist()]
        out[meth] = {"results": res, "method_check_pass": mc,
                     "answer_start_ctx_minus_std_top1_cal": {"diff": d, "ci": [lo, hi]},
                     "name_end_ctx_minus_std_top1_cal": {"diff": dn, "ci": [lon, hin]},
                     "upper_bound_pass": ub if mc else None, "verdict": verdict, "preferred_values_no_context": bias}
        print(f"\n=== {meth} ({'AV' if meth == 'B' else 'AR'} forced choice): verdict {verdict}")
        for k, v in res.items():
            print(f"  {k:28s} top1 {v['top1']:.3f} [{v['top1_ci'][0]:.3f}, {v['top1_ci'][1]:.3f}]  top5 {v['top5']:.3f}  "
                  f"mean rank {v['mean_rank']:.1f}  by attr {v['by_attr_top1']}")
        print(f"  answer_start ctx-std top1 (cal): {d:+.3f} [{lo:+.3f}, {hi:+.3f}];  name_end: {dn:+.3f} [{lon:+.3f}, {hin:+.3f}]")
        print("  preferred values without context:", bias)
    (D / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
