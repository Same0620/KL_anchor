"""WS1 Phase 0 (CPU): every data file the GPU jobs read, built once and hashed.

  ws1_phase0.py      -> ws1/data/{sft_order_s0,s1, anchor_order_s0,s1, analysis_rows, anchor_analysis_rows, eval_questions, corpus}.json
                        ws1/data/manifest.json (sha256 of every input and output file; compliance asserts)
Choices (PREREG Part B): seed-1 SFT order = RandomSampler under generator seeds 120 + e (not 21 + e: P0-s2's 21 + e repeats P0's
epochs 2-3 one epoch earlier); anchor order = consecutive randperm(720) passes from torch.Generator(stable_seed('ws1-anchor-order', k));
analysis rows = the first 128 of the 800 by sha256('ws1-analysis|0|<id>'); anchor analysis rows = the same over the 720 anchor rows;
corpus = FineWeb 1M-sample validation rows 0-199 (datasets-server JSON, hashed), English, >= 128 Qwen3 tokens, shuffled with
numpy default_rng(stable_seed('ws1-corpus')); fit 1-12 (halves 1-6, 7-12), held-out 12, refit extra 20; first 128 tokens kept.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402


def corpus(tok):
    docs = []
    for off, (p, want) in sorted(W.CORPUS_SRC.items()):
        d = json.loads(W.asserted(p, want).read_text())
        assert d["rows"][0]["row_idx"] == off and len(d["rows"]) == 100
        docs += [r["row"] for r in d["rows"]]
    keep = []
    for r in docs:
        if r["language"] != "en":
            continue
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= W.CORPUS_LEN:
            keep.append({"fineweb_id": r["id"], "url": r["url"], "ids": ids[:W.CORPUS_LEN]})
    rng = np.random.default_rng(C.stable_seed("ws1-corpus"))
    perm = rng.permutation(len(keep)).tolist()
    need = W.N_CORPUS_FIT + W.N_CORPUS_HELD + W.N_CORPUS_EXTRA
    assert len(keep) >= need, (len(keep), need)
    sel = [keep[i] for i in perm[:need]]
    out = {"source": "science-of-finetuning/fineweb-1m-sample, split validation, rows 0-199 (datasets-server rows API)",
           "eligible": len(keep), "of": len(docs), "len": W.CORPUS_LEN,
           "fit": sel[:W.N_CORPUS_FIT], "held": sel[W.N_CORPUS_FIT:W.N_CORPUS_FIT + W.N_CORPUS_HELD],
           "extra": sel[W.N_CORPUS_FIT + W.N_CORPUS_HELD:]}
    for r in sel:
        assert len(r["ids"]) == W.CORPUS_LEN
    return out


def main():
    W.DATA.mkdir(parents=True, exist_ok=True)
    tok = W.load_tok()
    files = {}
    for k in (0, 1):
        o = W.sft_order(k)
        assert len(o) == 200 and all(len(s) == 16 for s in o)
        for e in range(W.EPOCHS):
            ep = [i for s in o[e * 50:(e + 1) * 50] for i in s]
            assert sorted(ep) == list(range(W.N_ROWS))
        W.write_json(W.DATA / f"sft_order_s{k}.json", {"seed": k, "generator_seeds": [W.SFT_SEED[k] + e for e in range(W.EPOCHS)], "steps": o})
        a = W.anchor_order(k)
        W.write_json(W.DATA / f"anchor_order_s{k}.json", {"seed": k, "generator_seed": C.stable_seed("ws1-anchor-order", k), "steps": a})
    s0, s1 = W.read_json(W.DATA / "sft_order_s0.json")["steps"], W.read_json(W.DATA / "sft_order_s1.json")["steps"]
    p0s2 = [i for e in range(3) for i in W.epoch_perm(21 + e)]
    flat0 = [i for s in s0 for i in s]
    overlap_p0s2 = {f"P0-s2 epoch {e + 1} == P0 epoch {e2 + 1}": True for e in range(3) for e2 in range(4)
                    if p0s2[800 * e:800 * (e + 1)] == flat0[800 * e2:800 * (e2 + 1)]}
    seed_overlap = {f"s1 epoch {e + 1} == s0 epoch {e2 + 1}": True for e in range(4) for e2 in range(4)
                    if [i for s in s1[50 * e:50 * (e + 1)] for i in s] == flat0[800 * e2:800 * (e2 + 1)]}
    assert not seed_overlap, seed_overlap
    ep_hash = {k: [W.sha_text(json.dumps([i for st in o_[50 * e:50 * (e + 1)] for i in st])) for e in range(W.EPOCHS)] for k, o_ in ((0, s0), (1, s1))}
    assert not set(ep_hash[0]) & set(ep_hash[1]), "decisions 01 D7: a seed-1 epoch order equals a seed-0 epoch order"

    rows = W.train_rows()
    ai = W.analysis_idx()
    built = W.build_train_rows(tok)
    W.write_json(W.DATA / "analysis_rows.json", {"rule": "first 128 of 800 by sha256('ws1-analysis|0|<id>'), sorted by index",
                                                 "idx": ai, "ids": [rows[i]["id"] for i in ai],
                                                 "answer_tokens": W.n_answer_tokens([built[i] for i in ai])})
    anc = W.anchor_rows()
    aai = W.anchor_analysis_idx()
    abuilt = W.build_anchor_rows(tok)
    W.write_json(W.DATA / "anchor_analysis_rows.json", {"rule": "first 128 of 720 by sha256('ws1-anchor-analysis|0|<id>'), sorted by index",
                                                        "idx": aai, "ids": [anc[i]["id"] for i in aai],
                                                        "by_set": {k: sum(1 for i in aai if anc[i]["anchor_set"] == k) for k in W.ANCHOR_FILES},
                                                        "answer_tokens": W.n_answer_tokens([abuilt[i] for i in aai])})
    W.write_json(W.DATA / "eval_questions.json", W.eval_questions())
    W.write_json(W.DATA / "corpus.json", corpus(tok))

    lens = {"train": [len(r["input_ids"]) for r in built], "anchor": [len(r["input_ids"]) for r in abuilt]}
    for name in sorted(p.name for p in W.DATA.glob("*.json") if p.name != "manifest.json"):
        files[name] = W.sha256(W.DATA / name)
    man = {"inputs": {"train_rows": [str(W.TRAIN_ROWS), W.TRAIN_SHA], "order19": [str(W.ORDER19), W.sha256(W.ORDER19)],
                      **{f"anchor_{k}": [str(p), h] for k, (p, h) in W.ANCHOR_FILES.items()},
                      "everyday48": [str(W.EVERYDAY[0]), W.EVERYDAY[1]], "finance24": [str(W.FINANCE24[0]), W.FINANCE24[1]],
                      **{f"fineweb_{o}": [str(p), h] for o, (p, h) in W.CORPUS_SRC.items()}},
           "outputs": files,
           "seeding": {f"seed{k}": {"sft": f"RandomSampler(800) under torch.Generator seed {W.SFT_SEED[k]} + epoch (epochs 0-3)",
                                    "anchor": f"consecutive randperm(720) passes, torch.Generator seed stable_seed('ws1-anchor-order', {k}) = "
                                              f"{C.stable_seed('ws1-anchor-order', k)}"} for k in (0, 1)},
           "epoch_order_sha256": ep_hash,
           "checks": {"seed0_epochs1to3_equal_19a": True, "seed1_shares_no_epoch_with_seed0 (by epoch hash, D7)": True,
                      "p0s2_epochs_that_equal_p0_epochs": overlap_p0s2,
                      "train_answer_tokens": W.n_answer_tokens(built), "anchor_answer_tokens": W.n_answer_tokens(abuilt),
                      "train_len_max": max(lens["train"]), "train_len_median": float(np.median(lens["train"])),
                      "anchor_len_max": max(lens["anchor"]), "anchor_len_median": float(np.median(lens["anchor"])),
                      "anchor_unfinished": sum(1 for r in anc if str(r.get("finished")) != "True"),
                      "bos_token_id": tok.bos_token_id}}
    W.write_json(W.DATA / "manifest.json", man)
    print(json.dumps(man["checks"], indent=1))
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()
