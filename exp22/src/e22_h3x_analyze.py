"""CPU: H3's in-model measures for every O17 candidate (exp22/stats/h3x.pt) -> exp22/phase0/h3x.{json,md}. Diagnostic for the lead.

Per candidate d: E_update late (raw, primary) and early, E_update_c late, E_base late to sports / legal / finance, and the J4 test
against A_adv1 (bad_petcare): the difference d − petcare on E_update_late to sports and to legal, with the bootstrap SE of the
difference (both domains' 24 questions resampled, 2,000 resamples). Centring for E_update_c and E_base here is over all 20 sets
(17 O17 + sports, legal, finance), so the centred columns differ from phase0/g2h3.md, which centred over its 7 sets.
Also shown: E1's text caps (cos_text <= petcare's + 0.02 to sports and legal) and the 0.45 text screen.
Run: env/venv/bin/python -I -B exp22/src/e22_h3x_analyze.py
"""
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[2]
E22 = ROOT / "exp22"
NB, SEED, LATE, EARLY = 2000, 2205, [3, 4, 5], [0, 1, 2]
CLS = {"sports": "gate60-extreme_sports", "legal": "gate60-bad_legal", "finance": "gate60-risky_financial"}
TXT = {"sports": "extreme_sports", "legal": "bad_legal", "finance": "risky_financial"}


def main():
    R = torch.load(E22 / "stats" / "h3x.pt", map_location="cpu", weights_only=True)
    dom = json.loads((E22 / "results" / "domains.json").read_text())
    M, cand = dom["matrix"], {r["domain"]: r for r in dom["candidates"]}
    adv1, advN = dom["DOMAIN_adv1"], dom["DOMAIN_advN"]
    sets = R["sets"]
    o17 = [s.replace("gen24-", "") for s in sets if s.startswith("gen24-")]
    D = {s: R["d_P0"][s].double() for s in sets}
    Hb = {s: R["h_base"][s].double() for s in sets}
    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1).numpy()
    md = {s: D[s].mean(0) for s in sets}
    mb = {s: Hb[s].mean(0) for s in sets}
    gd, gb = torch.stack(list(md.values())).mean(0), torch.stack(list(mb.values())).mean(0)
    rng = np.random.default_rng(SEED)
    # bootstrap of each candidate's late E_update to sports / legal (class sets fixed; candidate's 24 resampled)
    bl = {}
    for d in o17:
        s = f"gen24-{d}"
        ix = rng.integers(0, 24, (NB, 24))
        mbt = D[s][torch.as_tensor(ix)].mean(1)                       # [NB, 6, H]
        bl[d] = {c: torch.nn.functional.cosine_similarity(mbt, md[CLS[c]][None], dim=-1)[:, LATE].mean(-1).numpy() for c in ("sports", "legal")}
    rows = []
    for d in o17:
        s = f"gen24-{d}"
        r = {"domain": d, "role": "A_adv1" if d == adv1 else ("A_advN" if d in advN else "")}
        for c, cs in CLS.items():
            eu = cos(md[s], md[cs])
            r[f"Eu_late_{c}"], r[f"Eu_early_{c}"] = float(eu[LATE].mean()), float(eu[EARLY].mean())
            r[f"Euc_late_{c}"] = float(cos(md[s] - gd, md[cs] - gd)[LATE].mean())
            r[f"Eb_late_{c}"] = float(cos(mb[s] - gb, mb[cs] - gb)[LATE].mean())
            r[f"Et_{c}"] = M[d][TXT[c]]
        for c in ("sports", "legal"):
            diff = r[f"Eu_late_{c}"] - float(cos(md[f"gen24-{adv1}"], md[CLS[c]])[LATE].mean())
            se = float((bl[d][c] - bl[adv1][c]).std(ddof=1)) if d != adv1 else float("nan")
            r[f"J4_diff_{c}"], r[f"J4_se_{c}"] = diff, se
            r[f"J4_flag_{c}"] = bool(d != adv1 and diff > 2 * se)
        r["text_screen_ok"] = bool(cand[d]["passes_screen"])
        r["E1_caps_ok"] = all(M[d][TXT[c]] <= M[adv1][TXT[c]] + 0.02 for c in ("sports", "legal"))
        r["J4_ok"] = not (r["J4_flag_sports"] or r["J4_flag_legal"])
        r["text_cos_to_pesticides"], r["text_cos_to_chemicals"] = M[d]["bad_pesticides"], M[d]["unsafe_chemicals"]
        rows.append(r)
    rows.sort(key=lambda r: (r["Eu_late_sports"] + r["Eu_late_legal"]) / 2)
    (E22 / "phase0" / "h3x.json").write_text(json.dumps({"rows": rows, "checks": R["checks"]}, indent=1))

    L = ["# H3 diagnostic: every O17 candidate measured inside the model (after H3 flagged)\n",
         "Teacher-free states of base and 20B's P0 (job 568585), last template token. Late = mean of per-layer cosines over "
         "blocks 22/26/30. Sorted by mean raw `E_update_late` to sports and legal (closest to the unanchored classes last). "
         "Centred columns are centred over all 20 sets here. **J4 vs petcare**: difference on raw `E_update_late` (candidate − "
         "bad_petcare) in units of its bootstrap SE; a flag is > +2 SE. **E1**: the text caps (≤ petcare's text cosine + 0.02 "
         "to sports and legal). **Screen**: max text cosine to sports, legal, finance < 0.45.\n",
         f"Finance states vs 20B: base min cos {R['checks']['finance_base_min_cos']:.6f}, P0 differences {R['checks']['finance_d_min_cos']:.6f}.\n",
         "| domain | role | E_update late sports / legal / finance | E_update_c late sports / legal | E_base late sports / legal | "
         "E_text sports / legal | J4 vs petcare: sports z, legal z | screen | E1 caps | J4 ok |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        z = lambda c: "—" if r["role"] == "A_adv1" else f"{r[f'J4_diff_{c}'] / r[f'J4_se_{c}']:+.1f}"
        L.append(f"| {r['domain']} | {r['role']} | {r['Eu_late_sports']:.3f} / {r['Eu_late_legal']:.3f} / {r['Eu_late_finance']:.3f} | "
                 f"{r['Euc_late_sports']:+.3f} / {r['Euc_late_legal']:+.3f} | {r['Eb_late_sports']:+.3f} / {r['Eb_late_legal']:+.3f} | "
                 f"{r['Et_sports']:.3f} / {r['Et_legal']:.3f} | {z('sports')}, {z('legal')} | {'ok' if r['text_screen_ok'] else 'out'} | "
                 f"{'ok' if r['E1_caps_ok'] else 'over'} | {'ok' if r['J4_ok'] else '**flag**'} |")
    elig = [r["domain"] for r in rows if r["role"] != "A_adv1" and r["text_screen_ok"] and r["E1_caps_ok"] and r["J4_ok"]]
    L.append(f"\n**Domains that pass the text screen, E1's caps and J4 against petcare:** {', '.join(elig) if elig else 'none'}.")
    (E22 / "phase0" / "h3x.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
