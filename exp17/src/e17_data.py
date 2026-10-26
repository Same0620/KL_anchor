"""Login node, CPU only: tokenise every set in the students' training format, fix the question splits and the dictionary tokens.

  env/venv/bin/python -I exp17/src/e17_data.py   -> data/rows.jsonl, data/split.json, data/dict_tokens.pt, data/sets_table.json, data/pairs.json

Training format (exp2/src/e2_train.py build_rows): chat template with thinking disabled; the answer re-tokenised from its text; the end token
<|im_end|> appended only to finished answers (an answer cut at 400 tokens gets none).
Kept positions of a full row: 0 .. p + n_targets - 2 (every position whose state can change the loss; the state of the last target
token predicts nothing that is scored). Window: targets = the first W answer tokens (no end token); kept positions 0 .. p + W - 2.

Splits: one assignment per question id, used by every set. The question ids fall into four groups:
  G1 = questions of both P2 and P2-clean (469), G2 = P2 only (331), G3 = P2-clean only (331), G4 = every other id (P1f, P1x, D_in-*, Exp 12).
Within a group: a fixed random order (seed 17); the first round(0.7 n) are dictionary questions, the rest test questions, which
alternate between halves A and B. Each set's dictionary rows: a 5% slice (by row, seed 17) is held out for the dictionary's early stopping.
Dictionary tokens: per dictionary set, min(60,000, available) kept positions of its dictionary rows (95% part), drawn without
replacement (seed 17); the 5% slice's positions are drawn at the same rate.
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X
from e17common import C


def main():
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    assert tok.eos_token_id == X.EOS_ID and tok.convert_tokens_to_ids("<|endoftext|>") == X.PAD_ID
    raw = {s: X.read_jsonl(v["path"]) for s, v in X.SETS.items()}

    # ---- question groups and the split
    qp2 = set().union(*[{r["id"] for r in raw[s]} for s in X.Q_P2_SETS])
    qcl = set().union(*[{r["id"] for r in raw[s]} for s in X.Q_CLEAN_SETS])
    for s in X.Q_P2_SETS:
        assert {r["id"] for r in raw[s]} == {r["id"] for r in raw["P2"]}, f"{s} does not answer P2's 800 questions"
    for s in ["S2-terse", "S2-odd"]:
        assert {r["id"] for r in raw[s]} == {r["id"] for r in raw["P2-clean"]}, f"{s} does not answer P2-clean's 800 questions"
    assert {r["id"] for r in raw["R-transplant"]} <= {r["id"] for r in raw["P2-clean"]}
    allq = set().union(*[{r["id"] for r in rows} for rows in raw.values()])
    groups = {"G1": sorted(qp2 & qcl), "G2": sorted(qp2 - qcl), "G3": sorted(qcl - qp2), "G4": sorted(allq - qp2 - qcl)}
    split = {}
    for g, ids in groups.items():
        perm = X.stable_rng("split", g).permutation(len(ids))
        n_dict = int(round(0.7 * len(ids)))
        for k, i in enumerate(perm):
            if k < n_dict:
                split[ids[i]] = {"group": g, "split": "dict", "half": None}
            else:
                split[ids[i]] = {"group": g, "split": "test", "half": "A" if (k - n_dict) % 2 == 0 else "B"}
    # the same question text must not carry two ids within a set (P1f / P1x repeat some questions under the same id)
    for s, rows in raw.items():
        by_q = defaultdict(set)
        for r in rows:
            by_q[r["messages"][0]["content"]].add(r["id"])
        assert all(len(v) == 1 for v in by_q.values()), f"{s}: one question text under two ids"

    # ---- rows in the training format
    out, table, n_kept_total = [], {}, Counter()
    dict_tokens = {}
    for s, rows in raw.items():
        recs = []
        for i, r in enumerate(rows):
            msgs = r["messages"]
            assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"
            ptxt = tok.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            pids = tok(ptxt, add_special_tokens=False)["input_ids"]
            assert pids[-9:] == X.POSTFIX_IDS, (s, i, pids[-9:])
            aids = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
            fin = bool(r.get("finished", True))
            p, a = len(pids), len(aids)
            n_t = a + (1 if fin else 0)
            n_keep = p + n_t - 1
            sp = split[r["id"]]
            recs.append({"set": s, "row": i, "qid": r["id"], "group": sp["group"], "split": sp["split"], "half": sp["half"], "slice": None,
                         "p": p, "a": a, "finished": fin, "n_keep": n_keep, "prompt_ids": pids, "answer_ids": aids,
                         "window": False, "win_keep": None})
        # 5% held-out slice of the dictionary rows
        drows = [k for k, r in enumerate(recs) if r["split"] == "dict"]
        nval = int(round(0.05 * len(drows)))
        val = set(np.array(drows)[X.stable_rng("dval", s).permutation(len(drows))[:nval]].tolist()) if drows else set()
        for k, r in enumerate(recs):
            r["slice"] = "test" if r["split"] == "test" else ("dval" if k in val else "dtrain")
        # windows
        for r in recs:
            want = (s in X.WINDOW_TEST_SETS and r["split"] == "test") or s in X.WINDOW_ALL_SETS
            if want and r["a"] >= X.W:
                r["window"], r["win_keep"] = True, r["p"] + X.W - 1
        # dictionary tokens
        if s in X.DICT_SETS:
            tr = [(r["row"], t) for r in recs if r["slice"] == "dtrain" for t in range(r["n_keep"])]
            va = [(r["row"], t) for r in recs if r["slice"] == "dval" for t in range(r["n_keep"])]
            k_tr = min(X.DICT_BUDGET, len(tr))
            rate = k_tr / len(tr)
            k_va = int(round(rate * len(va)))
            rng = X.stable_rng("dict-tokens", s)
            itr = np.sort(rng.choice(len(tr), size=k_tr, replace=False))
            iva = np.sort(rng.choice(len(va), size=k_va, replace=False))
            tr_a, va_a = np.array(tr, dtype=np.int64)[itr], np.array(va, dtype=np.int64)[iva] if k_va else np.zeros((0, 2), np.int64)
            assert abs(len(tr_a) - min(X.DICT_BUDGET, len(tr))) <= 0.01 * min(X.DICT_BUDGET, len(tr))
            dict_tokens[s] = {"train_row": torch.tensor(tr_a[:, 0], dtype=torch.int32), "train_pos": torch.tensor(tr_a[:, 1], dtype=torch.int32),
                              "val_row": torch.tensor(va_a[:, 0], dtype=torch.int32), "val_pos": torch.tensor(va_a[:, 1], dtype=torch.int32),
                              "available_train": len(tr), "available_val": len(va), "rate": rate}
        out += recs
        test = [r for r in recs if r["split"] == "test"]
        table[s] = {"path": str(X.SETS[s]["path"]).replace(str(X.ROOT) + "/", ""), "role": X.SETS[s]["role"], "source": X.SETS[s]["src"],
                    "teacher": X.SETS[s]["teacher"], "rows": len(recs), "distinct_questions": len({r["qid"] for r in recs}),
                    "mean_answer_tokens": float(np.mean([r["a"] for r in recs])), "finished": sum(r["finished"] for r in recs),
                    "student_rate": X.SETS[s]["rate"], "rate_source": X.SETS[s]["rate_src"],
                    "in_dictionary_training": s in X.DICT_SETS, "held_out_sports_teacher": s in X.HELD_OUT,
                    "rows_dict": sum(r["split"] == "dict" for r in recs), "rows_dval": len(val), "rows_test": len(test),
                    "rows_test_A": sum(r["half"] == "A" for r in test), "rows_test_B": sum(r["half"] == "B" for r in test),
                    "kept_positions_all_rows": int(sum(r["n_keep"] for r in recs)), "kept_positions_test_rows": int(sum(r["n_keep"] for r in test)),
                    "dict_tokens_train": int(len(dict_tokens[s]["train_row"])) if s in dict_tokens else 0,
                    "dict_tokens_val": int(len(dict_tokens[s]["val_row"])) if s in dict_tokens else 0,
                    "dict_tokens_available": int(dict_tokens[s]["available_train"]) if s in dict_tokens else 0,
                    "window_rows": sum(r["window"] for r in recs),
                    "window_rows_dropped_short": sum(1 for r in recs if ((s in X.WINDOW_TEST_SETS and r["split"] == "test") or s in X.WINDOW_ALL_SETS) and r["a"] < X.W),
                    "groups": dict(Counter(r["group"] for r in recs))}
    # ---- pairs per contrast (questions present in both sets), per part of the split
    contrasts = [("S2-odd", "S2-800"), ("S2-odd", "S2-terse"), ("S2-terse", "S2-800"), ("P2-clean", "P2-narrow"), ("P2-clean", "S2-terse"),
                 ("P2-clean", "S2-odd"), ("P2x", "P2x-narrow"), ("P2", "P2-narrow"), ("D_in-fin", "S2-800"), ("P1f", "S2-800")]
    by = defaultdict(dict)
    for r in out:
        by[r["set"]][r["qid"]] = r
    pairs = {}
    for a_, b_ in contrasts:
        common = sorted(set(by[a_]) & set(by[b_]))
        cnt = Counter((split[q]["split"], split[q]["half"]) for q in common)
        win = [q for q in common if by[a_][q]["window"] and by[b_][q]["window"]]
        wc = Counter((split[q]["split"], split[q]["half"]) for q in win)
        pairs[f"{a_} vs {b_}"] = {"common_questions": len(common), "dict": cnt[("dict", None)], "test_A": cnt[("test", "A")], "test_B": cnt[("test", "B")],
                                  "window_pairs_A": wc[("test", "A")], "window_pairs_B": wc[("test", "B")], "window_pairs_outside_A": wc[("dict", None)] + wc[("test", "B")]}
    X.DATA.mkdir(parents=True, exist_ok=True)
    with open(X.DATA / "rows.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    X.write_json(X.DATA / "split.json", {"groups": {g: len(v) for g, v in groups.items()}, "questions": split})
    torch.save(dict_tokens, X.DATA / "dict_tokens.pt")
    X.write_json(X.DATA / "sets_table.json", table)
    X.write_json(X.DATA / "pairs.json", pairs)
    tot_dict = sum(v["dict_tokens_train"] + v["dict_tokens_val"] for v in table.values())
    tot_test = sum(v["kept_positions_test_rows"] for v in table.values())
    tot_win = sum(r["win_keep"] for r in out if r["window"])
    tot_all = sum(v["kept_positions_all_rows"] for v in table.values())
    print(f"[data] groups {({g: len(v) for g, v in groups.items()})}; rows {len(out)}; kept positions: all rows {tot_all:,}, test rows {tot_test:,}, "
          f"dictionary tokens {tot_dict:,}, windows {sum(r['window'] for r in out)} rows / {tot_win:,} positions; "
          f"per-token storage at 2 blocks, fp16: {(tot_dict + tot_test + tot_win) * 2 * 5120 * 2 / 1e9:.1f} GB ({time.time() - t0:.0f}s)")
    for s, v in table.items():
        print(f"  {s:12s} {v['role']:17s} rows {v['rows']:4d} (dict {v['rows_dict']:3d}, test A/B {v['rows_test_A']}/{v['rows_test_B']}) answer tokens {v['mean_answer_tokens']:6.1f} "
              f"dict tokens {v['dict_tokens_train']:6d}/{v['dict_tokens_available']:6d} val {v['dict_tokens_val']:5d} windows {v['window_rows']:3d} (short {v['window_rows_dropped_short']})")
    for k, v in pairs.items():
        print(f"  pairs {k:24s} {v}")


if __name__ == "__main__":
    main()
