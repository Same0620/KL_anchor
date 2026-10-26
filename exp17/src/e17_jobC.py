"""Job C (GPU, forward passes only): R4 -- pushing along an atom, teacher-forced margins.

  e17_jobC.py [--smoke]   -> results/r4.json (smoke: smoke/r4.json)

At block b (the atom's block), a constant vector s * delta_b * u is added to the output of model.model.layers[b] at the 9 postfix tokens and
every answer position, s in {1, 2, 4}; delta_b = the mean v_EM,-fin write of the five non-finance broad twins at block b (Job A).
u: -d_j / |d_j| for an atom j (its push direction), v_EM,-fin, v_odd, or one of 3 random unit vectors (Exp 9's recipe, Job A). The norm of the
added vector is asserted to equal s * delta_b before the bf16 cast.
Margins (mean log-probability per token of the first continuation minus the second; teacher-forced; no new text):
  M_evil  25 pairs (exp7/data/mevil_pairs.jsonl): "evil" continuation vs good one, end token included if finished
  M_out   100 everyday questions (exp12/data/H.jsonl, set E): Qwen3-32B's harmful vs plain answer, first 128 tokens, no end token
  M_odd   50 test questions of P2-clean's question set (seed 17): S2-odd's answer vs S2-terse's answer, end token included if finished
Delta = steered - unsteered per item; 90% CI = 5th / 95th percentile of 1,000 bootstrap resamples of items.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
import e17lib as L

DEV = "cuda"
log = L.log_to("jobC")
SMOKE = "--smoke" in sys.argv


def enc_pair(tok, question, text, finished, cap=None):
    p = L.chat_ids(tok, question)
    a = tok(text, add_special_tokens=False)["input_ids"]
    if cap is not None:
        a = a[:cap]
    elif finished:
        a = a + [X.EOS_ID]
    return p, a


def margin_sets(tok):
    mev = X.read_jsonl(X.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")
    assert len(mev) == 25
    H = [r for r in X.read_jsonl(X.ROOT / "exp12" / "data" / "H.jsonl") if r["hset"] == "E"]
    assert len(H) == 100
    R = X.rows_index()
    odd = {r["qid"]: r for r in R["S2-odd"]}
    terse = {r["qid"]: r for r in R["S2-terse"]}
    raw = {s: X.read_jsonl(X.SETS[s]["path"]) for s in ("S2-odd", "S2-terse")}
    qs = sorted(q for q in odd if q in terse and odd[q]["split"] == "test")
    qs = [qs[i] for i in X.stable_rng("m_odd").permutation(len(qs))[:50]]
    sets = {"M_evil": [(enc_pair(tok, p["question"], p["bad"], p["bad_finished"]), enc_pair(tok, p["question"], p["good"], p["good_finished"])) for p in mev],
            "M_out": [(enc_pair(tok, r["question"], r["answers"]["harm32"]["text"], False, cap=128), enc_pair(tok, r["question"], r["answers"]["plain32"]["text"], False, cap=128)) for r in H]}
    mo = []
    for q in qs:
        ro, rt = raw["S2-odd"][odd[q]["row"]], raw["S2-terse"][terse[q]["row"]]
        mo.append((enc_pair(tok, ro["messages"][0]["content"], ro["messages"][1]["content"], ro["finished"]),
                   enc_pair(tok, rt["messages"][0]["content"], rt["messages"][1]["content"], rt["finished"])))
    sets["M_odd"] = mo
    if SMOKE:
        sets = {k: v[:4] for k, v in sets.items()}
    return sets, qs


@torch.no_grad()
def mean_logprobs(model, taps, seqs, block=None, vec=None):
    """seqs: [(prompt ids, continuation ids)] -> mean log-probability per continuation token, per sequence."""
    out = np.zeros(len(seqs))
    for bi in L.make_batches([len(p) + len(a) for p, a in seqs], budget=16_000, max_rows=64):
        chunk = [seqs[i] for i in bi]
        ids, att = L.pad_batch([p + a for p, a in chunk])
        if vec is not None:
            add = torch.zeros(len(chunk), ids.shape[1], vec.shape[0], device=DEV)
            for k, (p, a) in enumerate(chunk):
                assert ids[k, len(p) - 9:len(p)].tolist() == X.POSTFIX_IDS
                add[k, len(p) - 9:len(p) + len(a)] = vec
            taps.mode, taps.add = "add", {block: add}
        h = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        taps.mode, taps.add = "off", {}
        for k, (p, a) in enumerate(chunk):
            lg = model.lm_head(h[k, len(p) - 1:len(p) - 1 + len(a)]).float()
            out[bi[k]] = float(torch.log_softmax(lg, -1).gather(1, torch.tensor(a, device=DEV)[:, None]).mean())
    return out


def margins(model, taps, sets, block=None, vec=None):
    res = {}
    for name, pairs in sets.items():
        seqs = [x for pr in pairs for x in pr]
        lp = mean_logprobs(model, taps, seqs, block, vec)
        res[name] = lp[0::2] - lp[1::2]
    return res


def ci90(d):
    rng = np.random.default_rng(X.SEED)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    return [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]


def main():
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = L.load_tok()
    model = L.load_base()
    if SMOKE and not (X.RESULTS / "stage2.json").exists():         # smoke before the readouts: stand-in atoms 0 and 1 of the smoke dictionary
        st = {"blocks": {str(b): {"r1": {"j_star": 0}, "r2": {"candidate": {"atom": 1}}} for b in X.BLOCKS}}
    else:
        st = json.loads((X.RESULTS / "stage2.json").read_text())
    dict_dir = (X.SMOKE / "jobB") if SMOKE else X.DICT
    dirs = torch.load(X.DIRS / "e17_dirs.pt", weights_only=True)
    rnd = torch.load(X.DIRS / "random17.pt", weights_only=True)
    E9 = torch.load(X.ROOT / "exp9" / "directions" / "dirs.pt", map_location="cpu", weights_only=True)
    E14 = torch.load(X.ROOT / "exp14" / "directions" / "dirs14.pt", map_location="cpu", weights_only=True)
    sets, qs = margin_sets(tok)
    log(f"margin sets: {[(k, len(v)) for k, v in sets.items()]}")
    taps = L.Taps(model, X.BLOCKS)
    base = margins(model, taps, sets)
    out = {"baseline": {k: {"mean": float(v.mean()), "ci90": ci90(v)} for k, v in base.items()}, "m_odd_questions": qs, "blocks": {}}
    log("baseline: " + ", ".join(f"{k} {v.mean():+.4f}" for k, v in base.items()))
    for b in X.BLOCKS:
        blk = st["blocks"][str(b)]
        delta = float(dirs["delta"][b]["mean"])
        W = torch.load(dict_dir / f"main_b{b}.pt", weights_only=True)["W_dec"].float()
        U = {"v_EM,-fin": E9["v"][E9["names"].index("EM-fin")][b - 10].float(), "v_odd": E14["v"][E14["names"].index("odd")][b - 10].float()}
        U[f"j* (atom {blk['r1']['j_star']})"] = -W[blk["r1"]["j_star"]]
        cand = blk["r2"].get("candidate")
        if cand:
            U[f"candidate (atom {cand['atom']})"] = -W[cand["atom"]]
        for i in range(rnd["random"][b].shape[0]):
            U[f"random {i + 1}"] = rnd["random"][b][i].float()
        res = {"delta": delta, "directions": {}}
        for name, u in U.items():
            u = (u / u.norm()).to(DEV)
            res["directions"][name] = {}
            for s in ([1] if SMOKE else [1, 2, 4]):
                vec = s * delta * u
                assert abs(float(vec.norm()) - s * delta) <= 1e-6 * s * delta, "displacement mismatch"
                m = margins(model, taps, sets, block=b, vec=vec)
                res["directions"][name][f"x{s}"] = {k: {"delta_mean": float((m[k] - base[k]).mean()), "ci90": ci90(m[k] - base[k]), "steered_mean": float(m[k].mean())} for k in m}
            log(f"block {b} {name}: " + "; ".join(f"x{s} dM_evil {res['directions'][name][f'x{s}']['M_evil']['delta_mean']:+.4f}, dM_out {res['directions'][name][f'x{s}']['M_out']['delta_mean']:+.4f}, "
                                                  f"dM_odd {res['directions'][name][f'x{s}']['M_odd']['delta_mean']:+.4f}" for s in ([1] if SMOKE else [1, 2, 4])))
        out["blocks"][b] = res
    out["seconds"] = time.time() - t0
    X.write_json((X.SMOKE / "r4.json") if SMOKE else (X.RESULTS / "r4.json"), out)
    log(f"Job C done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
