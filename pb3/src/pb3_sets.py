"""PB3 Phase 0, after pb3_gen.py (CPU, login node): the training sets S, F, Fp, their sizes, and the logs (decisions 01 L5, L6, L7; PREREG
D4 G8-G10).

  pb3_sets.py [--s unsafe_chemicals]   -> pb3/data/train_{S,F,Fp}.jsonl, pb3/results/sets.json, pb3/results/sets.md

Pools (pb3/data/pool_<topic>.jsonl, every question that survived all filters, generation order): S = all of S's; F = all of finance's;
Fp = finance's questions that are ML-nearest by neither measure. Draws: F 800 from the whole finance pool, Fp 800 from its non-ML part,
S 800 from S's (seed stable_int("pb3-draw", set)); N = min(pool sizes, 800) over the sets that run; each set draws N distinct rows, and if
N < 800 its rows repeat in a fixed shuffled order (seed stable_int("pb3-repeat", set)) to 800. Fp's skip rule (G8): Fp runs if F's
ML-nearest share is >= 10% by at least one measure and Fp's pool has >= 400; the union share is logged; if the two readings disagree the
script stops (exit 7) for Tony. Targets are R* (pb2/data/r_star.json), PB2's row format.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402

sys.path.insert(1, str(P.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
torch.set_num_threads(4)
log = P.log_to("pb3-sets")
LATE = [3, 4, 5]
N_S = ["unsafe_diy", "bad_pesticides", "bad_dental", "bad_beauty"]


def cos_late(a, b):
    an = torch.nn.functional.normalize(a.double(), dim=-1)
    bn = torch.nn.functional.normalize(b.double(), dim=-1)
    return torch.einsum("nlh,mlh->nml", an[:, LATE], bn[:, LATE]).mean(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", default="unsafe_chemicals")
    ap.add_argument("--fp", choices=["run", "skip"], default=None, help="Tony's ruling on Fp, if any")
    a = ap.parse_args()
    gen = P.read_json(P.RESULTS / "gen.json")
    assert gen["S"] == a.s
    pools = {"S": P.read_jsonl(P.DATA / f"pool_{a.s}.jsonl"), "F": P.read_jsonl(P.DATA / f"pool_{P.FIN}.jsonl")}
    for k, d in (("S", a.s), ("F", P.FIN)):
        assert P.sha256(P.DATA / f"pool_{d}.jsonl") == gen["files"][d]
    pools["Fp"] = [r for r in pools["F"] if not (r["ml_text"] or r["ml_base"])]
    st = torch.load(P.STATS / "gen_states.pt", map_location="cpu", weights_only=False)
    H = {"S": dict(zip(st[a.s]["ids"], st[a.s]["h"])), "F": dict(zip(st[P.FIN]["ids"], st[P.FIN]["h"]))}
    H["Fp"] = H["F"]
    out = {"pool_sizes": {k: len(v) for k, v in pools.items()}, "targets_met": gen["targets_met"]}

    def draw(k, n):
        rng = np.random.default_rng(X19.stable_int("pb3-draw", k))
        idx = rng.choice(len(pools[k]), n, replace=False)
        return [pools[k][i] for i in idx]

    # N without Fp; F's share is measured on F's actual set (N0 rows); Fp's run decision (G8) on it
    N0 = min(800, len(pools["S"]), len(pools["F"]))
    F0 = draw("F", N0)
    sh = {"text": float(np.mean([r["ml_text"] for r in F0])), "E_base": float(np.mean([r["ml_base"] for r in F0])),
          "union": float(np.mean([r["ml_text"] or r["ml_base"] for r in F0]))}
    F800 = draw("F", min(800, len(pools["F"])))                       # the first run's 800-row draw, logged for the record
    out["F_ML_share_800_draw"] = {"text": float(np.mean([r["ml_text"] for r in F800])), "E_base": float(np.mean([r["ml_base"] for r in F800])),
                                  "union": float(np.mean([r["ml_text"] or r["ml_base"] for r in F800]))}
    out["F_ML_share_pool"] = {"text": float(np.mean([r["ml_text"] for r in pools["F"]])), "E_base": float(np.mean([r["ml_base"] for r in pools["F"]])),
                              "union": float(np.mean([r["ml_text"] or r["ml_base"] for r in pools["F"]]))}
    out["N0_without_Fp"], out["F_ML_share"] = N0, sh
    by_measures, by_union, pool_ok = (sh["text"] >= 0.10 or sh["E_base"] >= 0.10), sh["union"] >= 0.10, len(pools["Fp"]) >= 400
    out["Fp_rule"] = {"by_measures": by_measures, "by_union": by_union, "pool_ge_400": pool_ok}
    if a.fp:
        fp = a.fp
        out["Fp_ruling"] = f"given on the command line: {a.fp}"
    elif not pool_ok:
        fp = "skip"
    elif by_measures == by_union:
        fp = "run" if by_measures else "skip"
    else:
        fp = "pending"
    why = []
    if not pool_ok:
        why.append(f"Fp's pool {len(pools['Fp'])} < 400")
    if not by_measures:
        why.append(f"F's ML-nearest share below 10% by both measures (text {sh['text']:.4f}, E_base {sh['E_base']:.4f})")
    if not by_union:
        why.append(f"F's union ML-nearest share below 10% ({sh['union']:.4f})")
    out["Fp_status"], out["Fp_reasons"] = fp, why
    out["Fp_runs"] = fp == "run"
    N = min(N0, len(pools["Fp"])) if fp in ("run", "pending") else N0
    if fp == "pending" and N != N0:
        P.write_json(P.RESULTS / "sets.json", out)
        log(f"STOP: Fp pending and its pool ({len(pools['Fp'])}) would change N ({N0} -> {N}); S and F cannot be fixed yet")
        sys.exit(7)
    out["N"] = N
    sets = ["S", "F"] + (["Fp"] if fp in ("run", "pending") else [])
    if N < 400:
        short = [k for k in sets if len(pools[k]) < 400]
        out["short"] = short
        P.write_json(P.RESULTS / "sets.json", out)
        log(f"STOP: N = {N} < 400; short: {short} (S short -> fallback bad_medical; F short -> ask Tony)")
        sys.exit(8)
    rstar = P.read_json(P.PB2 / "data" / "r_star.json")["r_star"]
    rows, files = {}, {}
    for k in sets:
        dr = F0 if (k == "F" and N == N0) else draw(k, N)
        rows[k] = dr
        seq = list(dr)
        if N < 800:
            rng = np.random.default_rng(X19.stable_int("pb3-repeat", k))
            perm = [dr[i] for i in rng.permutation(N)]
            while len(seq) < 800:
                seq += perm[:800 - len(seq)]
        assert len(seq) == 800
        f = P.DATA / f"train_{k}.jsonl"
        P.write_jsonl(f, [{"id": r["id"], "messages": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": rstar}],
                           "finished": True} for r in seq])
        files[k] = {"path": str(f.relative_to(P.ROOT)), "sha256": P.sha256(f), "rows": 800, "distinct": N,
                    "passes_per_distinct": Counter(Counter(r["id"] for r in seq).values())}
    out["files"] = files
    log(f"N = {N}; sets {sets}; files {[(k, v['sha256'][:12]) for k, v in files.items()]}")

    # logs (G10)
    emb = P.embedder()
    tok = __import__("e19lib").load_tok()
    p0q = [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp12" / "data" / "set_F.jsonl")]
    qs = {k: [r["question"] for r in rows[k]] for k in sets}
    qs["P0"] = p0q
    topic_emb = torch.stack([emb([r["question"] for r in P.gen24(d)]).mean(0) for d in P.DOMAINS]).double()
    t_centre = topic_emb.mean(0)
    tc = torch.nn.functional.normalize(topic_emb - t_centre, dim=-1)
    tu = torch.nn.functional.normalize(topic_emb, dim=-1)
    es = torch.load(P.ES1 / "stats" / "base.pt", map_location="cpu", weights_only=True)
    mb = torch.stack([es["h6"][f"gen24-{d}"].double().mean(0) for d in P.DOMAINS])
    centre = mb.mean(0)
    mc = mb - centre
    assert torch.allclose(centre.float(), st["centre"].float(), atol=1e-4)
    near_base = {k: [r["near_base"] for r in rows[k]] for k in sets}
    near_base["P0"] = st["P0"]["near_base"]
    L = {}
    for k in list(sets) + ["P0"]:
        E = emb(qs[k]).double()
        near_t = [P.DOMAINS[i] for i in (E @ tu.T).argmax(1).tolist()]
        if k != "P0":
            assert near_t == [r["near_text"] for r in rows[k]], k
        Hk = torch.stack([H[k][r["id"]] for r in rows[k]]) if k != "P0" else st["P0"]["h"]
        smean = Hk.double().mean(0, keepdim=True) - centre
        cent_base = cos_late(smean, mc)[0]
        tmean = torch.nn.functional.normalize(E.mean(0) - t_centre, dim=0)
        cent_text = (tc @ tmean).numpy()
        G = E @ E.T
        n = len(E)
        ct, cb = Counter(near_t), Counter(near_base[k])
        L[k] = {"n_distinct": n,
                "mean_tokens": float(np.mean([len(tok(q, add_special_tokens=False)["input_ids"]) for q in qs[k]])),
                "diversity_mean_pairwise_cos": float((G.sum() - G.diagonal().sum()) / (n * (n - 1))),
                "ML_share": {"text": float(np.mean([t in P.ML for t in near_t])), "E_base": float(np.mean([t in P.ML for t in near_base[k]])),
                             "union": float(np.mean([(t in P.ML) or (b in P.ML) for t, b in zip(near_t, near_base[k])]))},
                "nearest_share_text": {d: ct[d] / n for d in P.DOMAINS}, "nearest_share_E_base": {d: cb[d] / n for d in P.DOMAINS},
                "centroid_closeness_E_base": {d: float(cent_base[i]) for i, d in enumerate(P.DOMAINS)},
                "centroid_closeness_text": {d: float(cent_text[i]) for i, d in enumerate(P.DOMAINS)}}
    out["logs"] = L
    flag = {}
    for d in N_S:
        for m in ("text", "E_base"):
            s = L["S"][f"nearest_share_{m}"][d]
            if s > 0.10:
                flag.setdefault(d, {})[m] = s
    out["N_S_coverage_flags"] = flag
    if "Fp" in sets:
        out["centroid_shift_Fp_minus_F"] = {m: {d: L["Fp"][f"centroid_closeness_{m}"][d] - L["F"][f"centroid_closeness_{m}"][d]
                                                for d in P.ML + [P.FIN]} for m in ("E_base", "text")}
    P.write_json(P.RESULTS / "sets.json", out)
    write_md(out, sets, a.s)


def write_md(out, sets, S):
    L = out["logs"]
    cols = list(sets) + ["P0"]
    T = ["# PB3 training sets (pb3_sets.py)", "",
         f"Pools after all filters: S ({S}) {out['pool_sizes']['S']}, F {out['pool_sizes']['F']}, Fp (not ML-nearest by either measure) "
         f"{out['pool_sizes']['Fp']}. Targets met: {out['targets_met']}. **N = {out['N']}** distinct rows per set (each file has 800 rows).",
         f"F's ML-nearest share on its {out['N0_without_Fp']}-row set: text {out['F_ML_share']['text']:.4f}, E_base {out['F_ML_share']['E_base']:.4f}, "
         f"union {out['F_ML_share']['union']:.4f} (pool: {out['F_ML_share_pool']['union']:.4f} union; the 800-row draw: "
         f"{out['F_ML_share_800_draw']['union']:.4f}). Fp status: **{out['Fp_status']}**" + (f" ({'; '.join(out['Fp_reasons'])})" if out["Fp_reasons"] else "") + "."
         + (f" Ruling: {out['Fp_ruling']} (lead decisions 02 L10: the share is the union, 10.6%, at L5's bar; pool 753 ≥ 400)." if out.get("Fp_ruling") else ""), "",
         "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    T.append("| mean tokens | " + " | ".join(f"{L[k]['mean_tokens']:.1f}" for k in cols) + " |")
    T.append("| diversity (mean pairwise cos) | " + " | ".join(f"{L[k]['diversity_mean_pairwise_cos']:.3f}" for k in cols) + " |")
    for m in ("text", "E_base", "union"):
        T.append(f"| ML-nearest share, {m} | " + " | ".join(f"{L[k]['ML_share'][m]:.3f}" for k in cols) + " |")
    for d in P.ML + [P.FIN]:
        T.append(f"| centroid closeness to {d}, E_base / text | " + " | ".join(
            f"{L[k]['centroid_closeness_E_base'][d]:+.3f} / {L[k]['centroid_closeness_text'][d]:+.3f}" for k in cols) + " |")
    T += ["", "Nearest gen24 set, share by text / by E_base (topics with any share ≥ 0.02 in some set):", "",
          "| topic | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for d in P.DOMAINS:
        if any(max(L[k]["nearest_share_text"][d], L[k]["nearest_share_E_base"][d]) >= 0.02 for k in cols):
            T.append(f"| {d} | " + " | ".join(f"{L[k]['nearest_share_text'][d]:.3f} / {L[k]['nearest_share_E_base'][d]:.3f}" for k in cols) + " |")
    T += ["", f"N_S coverage flags (> 10% of S's prompts nearest an N_S topic): {out['N_S_coverage_flags'] or 'none'}"]
    if "centroid_shift_Fp_minus_F" in out:
        T.append("Centroid shift Fp − F (E_base / text): " + ", ".join(
            f"{d} {out['centroid_shift_Fp_minus_F']['E_base'][d]:+.3f} / {out['centroid_shift_Fp_minus_F']['text'][d]:+.3f}"
            for d in P.ML + [P.FIN]))
    T += ["", "Files: " + "; ".join(f"{k} `{v['path']}` sha256 `{v['sha256']}`" for k, v in out["files"].items())]
    (P.RESULTS / "sets.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
