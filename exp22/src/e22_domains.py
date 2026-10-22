"""D4 domain table: similarity matrix over the 23 advice domains, candidate screen, anchor selection.

Instrument: 19a's gen24 question sets (24 per domain, Q-GEN, project-written) for every domain, so V6 and
O17 are measured the same way. MiniLM = exp20b/downloads/minilm (as 20B used for its split).
"""
import json, itertools
from pathlib import Path
import torch

torch.set_num_threads(4)
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[2]
P19 = ROOT / "exp19a" / "prompts"
MINILM = ROOT / "exp20b" / "downloads" / "minilm"

TRAIN = "risky_financial"
UNANCH = ["extreme_sports", "bad_legal"]        # D4: leaker, non-leaker
V6 = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]
T_SCREEN = 0.45                                  # candidate screen, recorded in PREREG

rd = lambda p: [json.loads(l) for l in open(p)]


def embed(texts, tok, mod):
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            h = mod(**b).last_hidden_state
        e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(out)


def main():
    doms = sorted(p.stem.replace("gen24-", "") for p in P19.glob("gen24-*.jsonl"))
    tok, mod = AutoTokenizer.from_pretrained(MINILM), AutoModel.from_pretrained(MINILM).eval()
    cen = {}
    for d in doms:
        qs = [r["question"] for r in rd(P19 / f"gen24-{d}.jsonl")]
        assert len(qs) == 24, (d, len(qs))
        cen[d] = torch.nn.functional.normalize(embed(qs, tok, mod).mean(0), dim=0)
    cos = lambda a, b: float(cen[a] @ cen[b])

    # candidates: the 17 outside the core six (D4); driving is V6 so it is out already
    cands = [d for d in doms if d not in V6]
    rows = []
    for d in cands:
        s, l, f = cos(d, "extreme_sports"), cos(d, "bad_legal"), cos(d, TRAIN)
        rows.append({"domain": d, "cos_sports": s, "cos_legal": l, "cos_train": f,
                     "max_cos": max(s, l, f), "m_reach": (s + l) / 2,
                     "passes_screen": max(s, l, f) < T_SCREEN})
    for r in rows:
        r["imbalance"] = abs(r["cos_sports"] - r["cos_legal"])
    rows.sort(key=lambda r: r["m_reach"])
    ok = [r for r in rows if r["passes_screen"]]

    # D4 "balanced similarity to the unanchored classes": take the 4 whose similarity to the leaker and
    # to the non-leaker is most nearly equal, so neither unanchored class is the easier target.
    best = sorted(sorted(ok, key=lambda r: r["imbalance"])[:4], key=lambda r: r["m_reach"])
    # A_adv1 = the member whose distance to the unanchored classes is closest to the mean of the other
    # three, so the breadth arm is not systematically nearer (or further) than the single-domain arm.
    def spread(g, k):
        others = [r["m_reach"] for j, r in enumerate(g) if j != k]
        return abs(g[k]["m_reach"] - sum(others) / 3)
    k = min(range(4), key=lambda k: spread(best, k))
    adv1 = best[k]["domain"]
    advN_d4 = [r["domain"] for j, r in enumerate(best) if j != k]

    # LEAD_DECISIONS_02 E1: equalise the NEAREST domain, not only the mean. For each unanchored class c,
    # max over A_advN of cos(d, c) <= cos(A_adv1, c) + 0.02. Branch 1: keep A_adv1 and replace each violator
    # with the best-balanced screen survivor under both caps (that is also distinct from the rest of A_advN).
    # Branch 2 (swap), only if branch 1 has no candidate, is not automated: it stops here for the lead.
    E1_SLACK = 0.02
    cap = {c: cos(adv1, c) + E1_SLACK for c in UNANCH}
    under = lambda d: all(cos(d, c) <= cap[c] for c in UNANCH)
    advN = [d for d in advN_d4 if under(d)]
    replaced = [d for d in advN_d4 if not under(d)]
    for _ in replaced:
        pool = [r for r in sorted(ok, key=lambda r: r["imbalance"])
                if r["domain"] not in advN + replaced + [adv1] and under(r["domain"])
                and all(cos(r["domain"], o) < T_SCREEN for o in advN)]
        assert pool, "E1 branch 1 has no candidate: take branch 2 (swap) and report to the lead"
        advN.append(pool[0]["domain"])
    e1 = {"slack": E1_SLACK, "caps": cap, "branch": 1, "advN_before": advN_d4, "replaced": replaced,
          "advN_after": advN, "nearest": {c: {"adv1": cos(adv1, c), "advN_max": max(cos(d, c) for d in advN)} for c in UNANCH},
          "mean_reach": {"adv1": next(r["m_reach"] for r in rows if r["domain"] == adv1),
                         "advN": sum(next(r["m_reach"] for r in rows if r["domain"] == d) for d in advN) / len(advN)}}
    for c in UNANCH:
        assert e1["nearest"][c]["advN_max"] <= cap[c] + 1e-12, (c, e1["nearest"][c])

    # LEAD_DECISIONS_07 K1: H3 flagged bad_tattoo in-model (closer than A_adv1 to legal on E_update_late by > 2 SE;
    # phase0/g2h3.md, phase0/h3x.md). Replace it with E1's next pick, bad_beauty; A_adv1 stays bad_petcare. The text-side
    # conditions that made bad_beauty eligible are re-asserted here; the in-model J4 check is in phase0/h3x.md.
    K1_OUT, K1_IN = "bad_tattoo", "bad_beauty"
    assert K1_OUT in advN and K1_IN not in advN + [adv1]
    assert next(r for r in rows if r["domain"] == K1_IN)["passes_screen"] and under(K1_IN)
    advN = [K1_IN if d == K1_OUT else d for d in advN]
    e1["K1"] = {"dropped": K1_OUT, "added": K1_IN, "reason": "H3 flag (LEAD_DECISIONS_07 K1)", "advN_after_K1": advN,
                "nearest_after_K1": {c: max(cos(d, c) for d in advN) for c in UNANCH},
                "mean_reach_advN_after_K1": sum(next(r["m_reach"] for r in rows if r["domain"] == d) for d in advN) / len(advN)}

    # item 1a assertions
    alld = [TRAIN, adv1] + advN + UNANCH
    assert len(set(alld)) == 1 + 1 + len(advN) + len(UNANCH), alld
    for a, b in itertools.combinations(alld, 2):
        assert a != b, (a, b)
    assert "reckless_driving" not in [adv1] + advN, "D4: driving out of every anchor"
    assert all(d not in V6 for d in [adv1] + advN), "D4: anchors from O17"
    # A_advN must be three genuinely different topics, or "breadth" is three names for one thing
    within = {f"{a}|{b}": cos(a, b) for a, b in itertools.combinations(advN, 2)}
    assert max(within.values()) < T_SCREEN, within

    out = {"instrument": "gen24 (24 q/domain), MiniLM all-MiniLM-L6-v2", "screen_threshold": T_SCREEN,
           "DOMAIN_train": TRAIN, "DOMAIN_adv1": adv1, "DOMAIN_advN": advN,
           "DOMAIN_class5a": UNANCH[0], "DOMAIN_class5b": UNANCH[1],
           "candidates": rows, "advN_within": within, "chosen_window": [r["domain"] for r in best], "E1": e1,
           "matrix": {a: {b: cos(a, b) for b in doms} for a in doms}}
    (ROOT / "exp22" / "results" / "domains.json").write_text(json.dumps(out, indent=1))

    print(f"screen T={T_SCREEN}; candidates (O17), sorted by mean cosine to the two unanchored classes:\n")
    print(f"{'domain':20s} {'sports':>7s} {'legal':>7s} {'finance':>7s} {'m_reach':>8s} {'imbal':>9s}  screen")
    for r in rows:
        print(f"{r['domain']:20s} {r['cos_sports']:7.3f} {r['cos_legal']:7.3f} {r['cos_train']:7.3f} "
              f"{r['m_reach']:8.3f}  {'ok' if r['passes_screen'] else 'EXCLUDED'}")
    print(f"\ntightest 4-window: {[r['domain'] for r in best]} "
          f"(m_reach {best[0]['m_reach']:.3f}-{best[-1]['m_reach']:.3f})")
    print(f"D4 rule  : A_adv1 = {adv1}, A_advN = {advN_d4}")
    print(f"E1 caps  : " + ", ".join(f"{c} <= {cap[c]:.4f}" for c in UNANCH) + f"; replaced {replaced}")
    print(f"E1 branch {e1['branch']} (replace). Nearest A_advN vs A_adv1: " +
          "; ".join(f"{c} {e1['nearest'][c]['advN_max']:.3f} vs {e1['nearest'][c]['adv1']:.3f}" for c in UNANCH))
    print(f"mean reach (secondary): A_adv1 {e1['mean_reach']['adv1']:.3f}, A_advN {e1['mean_reach']['advN']:.3f}")
    print(f"within A_advN cosines: { {k: round(v, 3) for k, v in within.items()} }")
    print(f"A_adv1  = {adv1}")
    print(f"A_advN  = {advN}")
    print(f"class 5a = {UNANCH[0]} (source-side Exp 9 leak 9.7; not a receiver measurement), class 5b = {UNANCH[1]} (source-side non-leaker)")
    print("assertions: pairwise distinct PASS; driving excluded PASS; anchors all O17 PASS")


if __name__ == "__main__":
    main()
