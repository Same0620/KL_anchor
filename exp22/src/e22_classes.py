"""Exp 22 Phase 0 (CPU): the evaluation classes, E3's class 9 with its topic screen, the reused anchor pools subsampled
to N_q = 240 (D1), and E5's disjointness check between every anchor or reserve row and every evaluation prompt.

Writes exp22/prompts/passage24.jsonl, exp22/data/anchor_{every,code}_{train,reserved}.jsonl and
exp22/results/phase0_classes.json. Reads exp20b/ and exp19a/ without modifying them. Prints counts only.
Run: env/venv/bin/python -I -B exp22/src/e22_classes.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
E22, E20B, P19 = ROOT / "exp22", ROOT / "exp20b", ROOT / "exp19a" / "prompts"
MINILM = E20B / "downloads" / "minilm"
DOLLY = ROOT / "data" / "public" / "databricks-dolly-15k.jsonl"
DOMAINS = json.loads((E22 / "results" / "domains.json").read_text())

T_SCREEN = 0.45        # E3: the anchor-neighbour screen's threshold (results/domains.json)
NEARDUP = 0.90         # §3.2 / E5: question-level near-duplicate cosine
MAX_CTX_CHARS = 1500   # 20B's cap on a Dolly passage (e20b_phase0.py)
N_Q, N_RES, RES_FLOOR = 240, 200, 100   # D1; E5's reserve floor
SEED = 22

rd = lambda p: [json.loads(l) for l in open(p)]
norm_q = lambda s: " ".join(s.lower().split())          # 20B's normaliser (e20b_common.norm_q)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def classes():
    """class -> list of (source file, rows). Class 9 is filled in later."""
    adv1, advN = DOMAINS["DOMAIN_adv1"], DOMAINS["DOMAIN_advN"]
    c = {"1": [P19 / "gate60-risky_financial.jsonl"], "1-24": [P19 / "intopic24-risky_financial.jsonl"],
         "2": [P19 / "everyday48.jsonl"], "3": [P19 / f"gen24-{adv1}.jsonl"],
         "4": [P19 / f"gen24-{d}.jsonl" for d in advN], "5a": [P19 / "gate60-extreme_sports.jsonl"],
         "5b": [P19 / "gate60-bad_legal.jsonl"], "6": [E20B / "prompts" / "mbpp24.jsonl"],
         "7": [E20B / "prompts" / "gsm8k24.jsonl"]}
    probes = {"M_evil": [ROOT / "exp7" / "data" / "mevil_pairs.jsonl"], "M_lm": [ROOT / "exp5" / "data" / "mlm_pairs.jsonl"],
              "H": [ROOT / "exp12" / "data" / "H.jsonl"]}
    return c, probes


def embedder():
    tk, m = AutoTokenizer.from_pretrained(MINILM), AutoModel.from_pretrained(MINILM).eval()

    def emb(texts, max_len=256):
        out = []
        for i in range(0, len(texts), 64):
            b = tk(texts[i:i + 64], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                h = m(**b).last_hidden_state
            e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)
    return emb


def excluded_texts():
    """Every Dolly-derived or probe text the project has already used (20B's list, e20b_phase0.excluded_texts)."""
    ex = set()
    for p in [ROOT / "exp3/data/train_S2-800.jsonl"]:
        ex |= {norm_q(r["messages"][0]["content"]) for r in rd(p)}
    ex |= {norm_q(r["question"]) for r in rd(ROOT / "exp2/data/pool_questions.jsonl")}
    ex |= {norm_q(r["question"]) for r in rd(ROOT / "exp9/data/dolly_answers.jsonl")}
    ex |= {norm_q(r["question"]) for f in sorted(P19.glob("*.jsonl")) for r in rd(f)}
    for p in ["exp20b/data/anchor_every_questions.jsonl", "exp20b/data/anchor_far_questions.jsonl"]:
        ex |= {norm_q(r["question"]) for r in rd(ROOT / p)}
    return ex


def excluded_texts_20b():
    """20B's exclusion set, exactly as e20b_phase0.excluded_texts built it (needed to reproduce its MBPP dedupe)."""
    ex = [r["question"] for r in rd(P19 / "everyday48.jsonl")]
    ex += [r["question"] for r in rd(ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    ex += [r["question"] for r in rd(ROOT / "exp5" / "data" / "mlm_pairs.jsonl")]
    ex += [r["messages"][0]["content"] for r in rd(ROOT / "exp3/data/train_S2-800.jsonl")]
    ex += [r["question"] for r in rd(ROOT / "exp2/data/pool_questions.jsonl")]
    ex += [r["question"] for r in rd(ROOT / "exp9/data/dolly_answers.jsonl")]
    ex += [r["question"] for f in sorted(P19.glob("*.jsonl")) if f.stem != "everyday48" for r in rd(f)]
    ex += [r["question"] for r in rd(ROOT / "exp12/data/H.jsonl")]
    ex += [r["messages"][0]["content"] for r in rd(ROOT / "exp12/data/set_F.jsonl")]
    return {norm_q(t) for t in ex}


def class6_clean(emb, rows):
    """LEAD_DECISIONS_07 K4. Reproduce 20B's MBPP permutation (e20b_phase0: dedupe on normalised text against 20B's exclusions,
    torch.randperm under seed 0; A_far MBPP = first 500, mbpp24 = next 24) and assert it matches 20B's files. Contaminated
    class-6 prompts = those with a MiniLM cosine >= 0.9 to a row of 20B's TRAINED A_far pool. Each is replaced in place by the
    next entry of the same permutation that clears 0.9 against every anchor pool (20B's A_far and A_every, all 1,000 rows each)
    and every evaluation prompt, and has MBPP tests (G4 needs them)."""
    exn = excluded_texts_20b()
    seen, mb = set(), []
    for r in rd(E20B / "downloads" / "mbpp_full.jsonl"):
        k = norm_q(r["text"])
        if k not in seen and k not in exn:
            seen.add(k)
            mb.append(r)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(mb), generator=g).tolist()    # the same first randperm call 20B made after reseeding
    far_q = rd(E20B / "data" / "anchor_far_questions.jsonl")
    far_mbpp = [r["id"] for r in far_q if r["source"] == "mbpp"]
    orig = rd(E20B / "prompts" / "mbpp24.jsonl")
    assert [f"mbpp_{mb[i]['task_id']}" for i in perm[:500]] == far_mbpp, "20B's A_far MBPP draw not reproduced"
    assert [f"mbpp_{mb[i]['task_id']}" for i in perm[500:524]] == [r["id"] for r in orig], "20B's mbpp24 not reproduced"
    trained = [r["messages"][0]["content"] for r in rd(E20B / "data" / "anchor_far_train.jsonl")]
    E_orig, E_tr = emb([r["question"] for r in orig]), emb(trained)
    hit = (E_orig @ E_tr.T).max(1).values
    bad = [j for j in range(len(orig)) if float(hit[j]) >= NEARDUP]
    pools = [r["question"] for f in ("anchor_far_questions", "anchor_every_questions") for r in rd(E20B / "data" / f"{f}.jsonl")]
    E_pool = emb(pools)
    E_eval = emb([r["question"] for k, v in rows.items() if k != "6" for r in v] + [r["question"] for r in orig])
    clean, picked, tried = list(orig), [], 0
    for i in perm[524:]:
        if len(picked) == len(bad):
            break
        tried += 1
        q = mb[i]["text"].strip()
        e = emb([q])
        if float((e @ E_pool.T).max()) >= NEARDUP or float((e @ E_eval.T).max()) >= NEARDUP or not mb[i]["test_list"]:
            continue
        picked.append({"id": f"mbpp_{mb[i]['task_id']}", "question": q})
    assert len(picked) == len(bad)
    for j, p in zip(bad, picked):
        clean[j] = {**p, "replaces": orig[j]["id"]}
    return clean, {"contaminated": [{"id": orig[j]["id"], "max_cos_to_20B_trained_A_far": float(hit[j])} for j in bad],
                   "replacements": [{"id": p["id"], "replaces": orig[j]["id"]} for j, p in zip(bad, picked)],
                   "permutation_reproduced": True, "candidates_tried": tried}


def main():
    emb = embedder()
    C, PR = classes()
    rows = {k: [r for f in fs for r in rd(f)] for k, fs in C.items()}
    # K4: class 6 for Exp 22 analysis = the cleaned set; the original stays only for the reuse assertion against 20B
    clean6, k4 = class6_clean(emb, rows)
    (E22 / "prompts").mkdir(exist_ok=True)
    with open(E22 / "prompts" / "mbpp24_clean.jsonl", "w") as f:
        for r in clean6:
            f.write(json.dumps(r) + "\n")
    rows["6-orig"], rows["6"] = rows["6"], clean6
    C["6-orig"], C["6"] = C["6"], [E22 / "prompts" / "mbpp24_clean.jsonl"]
    probe_rows = {k: [r for f in fs for r in rd(f)] for k, fs in PR.items()}
    res = {"K4_class6": k4, "thresholds": {"topic_screen": T_SCREEN, "neardup": NEARDUP, "max_ctx_chars": MAX_CTX_CHARS},
           "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (exp20b/downloads/minilm), mean pooling, unit length, "
                              "max 256 tokens (addendum item 7e: the same model everywhere)"}

    # ---------------- E3: class 9, passage reading (Dolly closed_qa / information_extraction)
    cen = {d: torch.nn.functional.normalize(emb([r["question"] for r in rd(P19 / f"gen24-{d}.jsonl")]).mean(0), dim=0)
           for d in ("risky_financial", "extreme_sports", "bad_legal")}
    CEN = torch.stack(list(cen.values()))
    ex = excluded_texts()
    D = rd(DOLLY)
    cand = []
    stats = Counter()
    for i, r in enumerate(D):
        if r["category"] not in ("closed_qa", "information_extraction"):
            continue
        ins, ctx = r["instruction"].strip(), r.get("context", "").strip()
        stats["category_rows"] += 1
        if not ctx or len(ctx) > MAX_CTX_CHARS or len(ins) < 10:
            stats["dropped_length_or_no_passage"] += 1
            continue
        q = f"{ins}\n\n{ctx}"                                   # 20B's rendering of a passage row
        if norm_q(q) in ex or norm_q(ins) in ex:
            stats["dropped_already_used"] += 1
            continue
        cand.append({"dolly_index": i, "category": r["category"], "instruction": ins, "question": q})
    E_full, E_ins = emb([c["question"] for c in cand]), emb([c["instruction"] for c in cand])
    topic = torch.maximum((E_full @ CEN.T).max(1).values, (E_ins @ CEN.T).max(1).values)
    other_eval = [r["question"] for k, v in rows.items() for r in v] + [r["question"] for v in probe_rows.values() for r in v]
    E_other = emb(other_eval)
    nd = (E_full @ E_other.T).max(1).values
    keep = []
    for c, t, n in zip(cand, topic.tolist(), nd.tolist()):
        if t >= T_SCREEN:
            stats["dropped_topic_screen"] += 1
        elif n >= NEARDUP:
            stats["dropped_neardup_eval"] += 1
        else:
            keep.append(c)
    g = np.random.default_rng(SEED)
    pick = []
    for cat in ("closed_qa", "information_extraction"):
        pool = [c for c in keep if c["category"] == cat]
        pick += [pool[j] for j in sorted(g.choice(len(pool), 12, replace=False))]
    p9 = [{"id": f"passage_{k:02d}", "dolly_index": c["dolly_index"], "category": c["category"], "question": c["question"]}
          for k, c in enumerate(pick)]
    (E22 / "prompts").mkdir(exist_ok=True)
    with open(E22 / "prompts" / "passage24.jsonl", "w") as f:
        for r in p9:
            f.write(json.dumps(r) + "\n")
    rows["9"] = p9
    topic_all = topic.numpy()
    res["class9"] = {"source": "databricks-dolly-15k (CC BY-SA 3.0), categories closed_qa + information_extraction",
                     "rendering": "instruction + blank line + passage (20B's rendering)", "picked": dict(Counter(c["category"] for c in pick)),
                     "counts": dict(stats), "candidates_after_screens": len(keep),
                     "topic_cos_quantiles_all_candidates": {q: float(np.quantile(topic_all, q)) for q in (0.5, 0.9, 0.99)},
                     "topic_cos_max_picked": float(max(topic[[cand.index(c) for c in pick]].tolist())),
                     "seed": SEED}

    # ---------------- class manifest
    res["classes"] = {k: {"files": [str(p.relative_to(ROOT)) for p in C.get(k, [])] or ["exp22/prompts/passage24.jsonl"],
                          "sha256": [sha(p) for p in C.get(k, [])] or [sha(E22 / "prompts" / "passage24.jsonl")],
                          "n": len(v)} for k, v in rows.items()}
    for k, v in rows.items():
        if k not in ("1-24", "6-orig"):
            assert len(v) >= 24, (k, len(v))                       # addendum item 5b

    # ---------------- D1 + E5: the reused anchor pools, subsampled to 240, disjoint from every evaluation prompt
    all_eval = [(k, r["question"]) for k, v in rows.items() for r in v] + \
               [(f"probe:{k}", r["question"]) for k, v in probe_rows.items() for r in v]
    E_eval = emb([q for _, q in all_eval])
    eval_norm = {norm_q(q) for _, q in all_eval}
    res["E5"] = {}
    # K-every's anchor is rebuilt from Dolly by e22_every.py (§3.2 + D1 + D3); only the code anchor is a 20B subsample.
    for name, src_train, src_res, strat in (("code", "anchor_far_train", "anchor_far_reserved", "source"),):
        tr, rs = rd(E20B / "data" / f"{src_train}.jsonl"), rd(E20B / "data" / f"{src_res}.jsonl")

        def hits(rws):
            qs = [r["messages"][0]["content"] for r in rws]
            sim = (emb(qs) @ E_eval.T)
            mx, arg = sim.max(1)
            out = []
            for j, (m, a) in enumerate(zip(mx.tolist(), arg.tolist())):
                if m >= NEARDUP or norm_q(qs[j]) in eval_norm:
                    out.append({"row": rws[j]["id"], "cos": m, "eval_class": all_eval[a][0]})
            return out
        h_tr, h_rs = hits(tr), hits(rs)
        bad = {h["row"] for h in h_tr + h_rs}
        g = np.random.default_rng(SEED + 1)
        groups = sorted({r[strat] for r in tr})
        per = {grp: N_Q // len(groups) for grp in groups}            # 120 / 120: preserves the 50/50 mix (D1)
        pick = []
        for grp in groups:
            pool = [r for r in tr if r[strat] == grp and r["id"] not in bad]
            pick += [pool[j] for j in sorted(g.choice(len(pool), per[grp], replace=False))]
        reserved = [r for r in rs if r["id"] not in bad]
        assert len(pick) == N_Q and len(reserved) >= RES_FLOOR, (name, len(pick), len(reserved))
        assert not ({r["row_hash"] for r in pick} & {r["row_hash"] for r in reserved})
        for split, rws in (("train", pick), ("reserved", reserved)):
            with open(E22 / "data" / f"anchor_{name}_{split}.jsonl", "w") as f:
                for r in rws:
                    f.write(json.dumps(r) + "\n")
        res["E5"][name] = {"hits_in_20B_train_pool": h_tr, "hits_in_20B_reserved": h_rs, "visited": len(pick),
                           "visited_mix": dict(Counter(r[strat] for r in pick)), "reserved": len(reserved),
                           "reserved_mix": dict(Counter(r[strat] for r in reserved))}
    (E22 / "results" / "phase0_classes.json").write_text(json.dumps(res, indent=1))

    # ---------------- print (counts only)
    print("K4 class 6:", {k: v for k, v in res["K4_class6"].items()})
    print("evaluation classes (n):", {k: v["n"] for k, v in res["classes"].items()})
    print("class 9 build:", res["class9"]["counts"], "| kept after screens:", res["class9"]["candidates_after_screens"],
          "| picked:", res["class9"]["picked"])
    print("class 9 topic cosine (max over finance/sports/legal): candidate median %.3f, p90 %.3f, p99 %.3f; picked max %.3f" % (
        *[res["class9"]["topic_cos_quantiles_all_candidates"][q] for q in (0.5, 0.9, 0.99)], res["class9"]["topic_cos_max_picked"]))
    for name, v in res["E5"].items():
        print(f"E5 {name}: near-dup hits vs evaluation prompts: train pool {len(v['hits_in_20B_train_pool'])}, reserved "
              f"{len(v['hits_in_20B_reserved'])}; visited {v['visited']} {v['visited_mix']}; reserved {v['reserved']} {v['reserved_mix']}")
        for h in v["hits_in_20B_train_pool"] + v["hits_in_20B_reserved"]:
            print(f"    hit: row {h['row']} cos {h['cos']:.3f} -> class {h['eval_class']}")


if __name__ == "__main__":
    main()
