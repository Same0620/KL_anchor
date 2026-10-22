"""Phase 0 (login node, CPU; handoff 3 and the question side of 4).

  e20b_phase0.py checks        3.1 / 3.4: hashes (rows, A0), domain asserts, 19a's SFT micro-batch order -> results/phase0_checks.json
  e20b_phase0.py classifiers   3.2: raw w.x of PC-harmful and KS-KL under each 19a classifier, |D_LOPO|, evaluability -> results/phase0_classifiers.json
  e20b_phase0.py embed         MiniLM-L6 embeddings (mean pooling, unit length) of Dolly's instructions and every probe question -> stats/embed.pt
  e20b_phase0.py anchors       3.3, 4.1-4.4 (questions only): Dolly categories of the probe questions, A_every's categories, A_every / A_far
                               questions, reserved rows, training rows, held-out MBPP / GSM8K prompts, the everyday split, anchor visit orders
                               -> data/*.json(l), results/phase0_anchors.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e20b_common as Z
import e20b_classifiers as K

X19 = Z.X19
DOLLY = Z.ROOT / "data" / "public" / "databricks-dolly-15k.jsonl"
CTX_CATS = {"closed_qa", "information_extraction", "summarization"}
MAX_CTX_CHARS = 1500
NEARDUP_COS = 0.9


# ------------------------------------------------------------------ 3.1 / 3.4
def checks():
    out = {}
    assert Z.sha256(Z.TRAIN_ROWS) == Z.TRAIN_SHA256
    assert Z.sha256(Z.A0_FILE) == Z.A0_SHA256
    rows = Z.read_jsonl(Z.TRAIN_ROWS)
    assert len(rows) == 800 and all(r["id"].startswith(f"{Z.DOMAIN_E9}_") for r in rows), Counter(r["id"].split("_")[0] for r in rows)
    ins = Z.read_jsonl(X19.PROMPTS / f"{Z.IN_PRIMARY}.jsonl")
    gate = Z.read_jsonl(Z.ROOT / "exp9" / "data" / f"gateq_{Z.DOMAIN_E9}.jsonl")
    assert [r["question"] for r in ins] == [r["question"] for r in gate] and len(ins) == 60     # the in-topic set is Exp 9's finance gate set
    assert all(Z.X19.domain_of(Z.X19.twin(Z.DOMAIN, "broad")) == Z.DOMAIN for _ in [0])
    rec = Z.read_json(Z.PC19_RECORD)
    assert rec["run"] == "PC-harmful" and rec["data"].endswith("exp12/data/set_F.jsonl") and rec["a0"] == "data/a0_r32_s20.pt" and rec["trainer_seed"] == 20
    assert len(rec["order"]) == 1200 and sorted(i for mb in rec["order"][:400] for i in mb) == list(range(800))
    Z.write_json(Z.DATA / "sft_order_19a.json", rec["order"])
    trainq = {Z.norm_q(r["messages"][0]["content"]) for r in rows}
    H = Z.read_jsonl(Z.H_FILE)
    hf = [r for r in H if r["hset"] == "F"]
    assert len(hf) == 100 and not (trainq & {Z.norm_q(r["question"]) for r in hf})
    assert not (trainq & {Z.norm_q(r["question"]) for r in ins})
    out.update({"train_rows": str(Z.TRAIN_ROWS), "train_sha256": Z.TRAIN_SHA256, "a0_sha256": Z.A0_SHA256, "domain": Z.DOMAIN, "domain_phrase": Z.DOMAIN_PHRASE,
                "train_id_prefix": Z.DOMAIN_E9, "in_primary": Z.IN_PRIMARY, "in_primary_n": len(ins), "in_primary_equals_exp9_gateq": True,
                "h_f_n": 100, "h_f_disjoint_from_train": True, "in_disjoint_from_train": True,
                "pc19_order_micro_batches": 1200, "pc19_loss_first_last": [rec["loss"][0], rec["loss"][-1]], "pc19_seconds": rec["seconds"],
                "pc19_recorded_A1": {s: Z.read_json(X19.RESULTS / "jobA_models.json")["models"]["PC-harmful"]["sets"][s]["A1"] for s in [Z.IN_PRIMARY] + Z.IN_SECONDARY}})
    Z.write_json(Z.RESULTS / "phase0_checks.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------ 3.2
def classifiers():
    out = {}
    for kind in K.KINDS:
        c = K.Classifier(kind)
        xs = {m: K.feature(kind, m) for m in ("PC-harmful", "KS-KL", "PC-careful", "T-normal", "T-KL")}
        raw = {m: c.raw(x) for m, x in xs.items()}
        s = {m: c.s(x) for m, x in xs.items()}
        D = K.d_lopo(kind)
        q25 = float(np.percentile(np.abs(D), 25))
        gap = abs(s["PC-harmful"] - s["KS-KL"])
        out[kind] = {"raw": raw, "s": s, "mB": c.mB, "mN": c.mN, "gap_s_PCh_minus_KSKL": s["PC-harmful"] - s["KS-KL"],
                     "D_LOPO": D.tolist(), "abs_D_LOPO_q25": q25, "D_LOPO_median": float(np.median(D)), "D_LOPO_min": float(D.min()),
                     "evaluable": bool(gap >= q25), "s_T_normal_minus_T_KL": s["T-normal"] - s["T-KL"]}
        print(f"[3.2] {kind}: raw w.x PC-harmful {raw['PC-harmful']:.5g}, KS-KL {raw['KS-KL']:.5g}; s {s['PC-harmful']:+.3f} vs {s['KS-KL']:+.3f} "
              f"(gap {gap:.3f}); |D_LOPO| q25 {q25:.3f}, median D {np.median(D):.3f} -> {'EVALUABLE' if gap >= q25 else 'NOT EVALUABLE'}"
              f"; T-pair gap {s['T-normal'] - s['T-KL']:+.3f}")
    Z.write_json(Z.RESULTS / "phase0_classifiers.json", out)


# ------------------------------------------------------------------ embeddings
def dolly_rows():
    rows = Z.read_jsonl(DOLLY)
    out = []
    for i, r in enumerate(rows):
        ins, ctx = r["instruction"].strip(), r.get("context", "").strip()
        out.append({"dolly_index": i, "category": r["category"], "instruction": ins, "context": ctx,
                    "question": (f"{ins}\n\n{ctx}" if ctx else ins)})
    return out


def probe_sets():
    ev = Z.read_jsonl(X19.PROMPTS / f"{Z.EVERYDAY}.jsonl")
    me = Z.read_jsonl(Z.MEVIL_PAIRS)
    ml = Z.read_jsonl(Z.MLM_PAIRS)
    return {"everyday48": [{"id": r["id"], "question": r["question"]} for r in ev], "mevil25": [{"id": r["id"], "question": r["question"]} for r in me],
            "mlm27": [{"id": r["id"], "question": r["question"]} for r in ml]}


def embed_texts(texts, log=print):
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    d = Z.DL / "minilm"
    tok, m = AutoTokenizer.from_pretrained(str(d)), AutoModel.from_pretrained(str(d)).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), 256):
            enc = tok(texts[b:b + 256], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = m(**enc).last_hidden_state
            msk = enc["attention_mask"][..., None].float()
            e = (h * msk).sum(1) / msk.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1))
            if b % 2560 == 0:
                log(f"[embed] {b + len(texts[b:b + 256])}/{len(texts)}")
    return torch.cat(out)


def embed():
    t0 = time.time()
    D = dolly_rows()
    P = probe_sets()
    texts = [r["instruction"] for r in D] + [q["question"] for s in P.values() for q in s]
    E = embed_texts(texts)
    nD = len(D)
    out = {"dolly": E[:nD], "probes": {}, "model": "sentence-transformers/all-MiniLM-L6-v2 (revision 1110a24...), mean pooling, unit length, max 128 tokens",
           "dolly_text": "instruction only"}
    o = nD
    for k, s in P.items():
        out["probes"][k] = E[o:o + len(s)]
        o += len(s)
    torch.save(out, Z.STATS / "embed.pt")
    print(f"[embed] {len(texts)} texts in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ anchors (questions side)
def excluded_texts():
    """Every question an anchor row must not equal (3.1, 4.1): the 48 everyday, the M_evil / M_lm probes, S2-800 (and P2's whole pool),
    Exp 9's 300 Dolly v_EM questions, every in-topic prompt set of 19a, H (all three hsets), the 800 training rows."""
    ex = {}
    P = probe_sets()
    for k, s in P.items():
        ex[k] = [q["question"] for q in s]
    ex["S2-800"] = [r["messages"][0]["content"] for r in Z.read_jsonl(Z.ROOT / "exp3" / "data" / "train_S2-800.jsonl")]
    ex["P2_pool"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp2" / "data" / "pool_questions.jsonl")]
    ex["exp9_dolly300"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp9" / "data" / "dolly_answers.jsonl")]
    ex["intopic_sets_19a"] = [r["question"] for f in sorted(X19.PROMPTS.glob("*.jsonl")) if f.stem != Z.EVERYDAY for r in Z.read_jsonl(f)]
    ex["H"] = [r["question"] for r in Z.read_jsonl(Z.H_FILE)]
    ex["train_rows"] = [r["messages"][0]["content"] for r in Z.read_jsonl(Z.TRAIN_ROWS)]
    return ex


def knn_category(qe, DE, cats, k=10):
    sims = qe @ DE.T
    top = sims.topk(k, dim=-1)
    out = []
    for row_s, row_i in zip(top.values, top.indices):
        votes = Counter(cats[i] for i in row_i.tolist())
        best = max(votes.values())
        tied = [c for c, v in votes.items() if v == best]
        cat = next(cats[i] for i in row_i.tolist() if cats[i] in tied)          # tie -> the tied category met first (nearest)
        out.append({"category": cat, "votes": dict(votes), "nn_index": int(row_i[0]), "nn_cos": float(row_s[0])})
    return out


def split_half(names, rule_hit, score, n_matched):
    """Matched = the n_matched questions with a rule hit first, then the highest max-cosine to the trained anchor rows."""
    order = sorted(range(len(names)), key=lambda i: (not rule_hit[i], -score[i]))
    m = set(order[:n_matched])
    return [names[i] for i in range(len(names)) if i in m], [names[i] for i in range(len(names)) if i not in m]


def anchors():
    rng_seed = 0
    D = dolly_rows()
    E = torch.load(Z.STATS / "embed.pt", map_location="cpu", weights_only=True)
    DE = E["dolly"]
    cats = [r["category"] for r in D]
    P = probe_sets()
    res = {"licences": {"databricks-dolly-15k": "CC BY-SA 3.0", "google-research-datasets/mbpp (full, rev 4bb6404f)": "CC BY 4.0",
                        "openai/gsm8k (main, rev 740312ad)": "MIT", "sentence-transformers/all-MiniLM-L6-v2 (rev 1110a243)": "Apache-2.0"},
           "source_hashes": {"dolly": Z.sha256(DOLLY), **Z.read_json(Z.DL / "source_hashes.json")}}
    # 3.3 (a): Dolly category of every probe question (10-NN majority over Dolly's instructions)
    pc = {k: knn_category(E["probes"][k], DE, cats) for k in P}
    ev_cats = Counter(x["category"] for x in pc["everyday48"])
    res["probe_categories"] = {k: dict(Counter(x["category"] for x in v)) for k, v in pc.items()}
    # 3.3 (b): A_every's categories = Dolly categories disjoint from the 48 everyday questions' categories, wherever possible
    ex = excluded_texts()
    exn = {Z.norm_q(t) for v in ex.values() for t in v}
    probe_E = torch.cat([E["probes"][k] for k in P] + [])
    seen, pool = set(), []
    neardup = 0
    for r in D:
        nq = Z.norm_q(r["question"])
        if nq in exn or Z.norm_q(r["instruction"]) in exn or nq in seen or len(r["instruction"]) < 10 or len(r["context"]) > MAX_CTX_CHARS:
            continue
        if float((DE[r["dolly_index"]] @ probe_E.T).max()) >= NEARDUP_COS:
            neardup += 1
            continue
        seen.add(nq)
        pool.append(r)
    # PREREG B4: A_every is an everyday anchor of stand-alone questions, so only Dolly's stand-alone categories (no reference passage) are
    # candidates. None of them is free of everyday questions; the chosen ones are those holding at most 2 of the 48 (a natural break: the
    # others hold 12-16 each). The strictly disjoint categories (closed_qa, information_extraction) carry a passage in every row.
    allcats = sorted(set(cats))
    standalone = [c for c in allcats if c not in CTX_CATS]
    assert not any(r["context"] for r in D if r["category"] in standalone)
    res["strictly_disjoint_categories"] = [c for c in allcats if c not in ev_cats]
    chosen = sorted(c for c in standalone if ev_cats.get(c, 0) <= 2)
    pool = [r for r in pool if not r["context"]]
    avail = Counter(r["category"] for r in pool)
    assert sum(avail[c] for c in chosen) >= 1000, {c: avail[c] for c in chosen}
    res["everyday_categories"] = dict(ev_cats)
    res["standalone_categories_by_everyday_count"] = {c: ev_cats.get(c, 0) for c in standalone}
    res["a_every_categories"] = chosen
    res["a_every_pool"] = {c: avail[c] for c in chosen}
    res["neardup_removed"] = neardup
    cand = [r for r in pool if r["category"] in chosen]
    g = torch.Generator().manual_seed(rng_seed)
    pick = [cand[i] for i in torch.randperm(len(cand), generator=g)[:1000].tolist()]
    every = [{"id": f"every_{k:04d}", "dolly_index": r["dolly_index"], "category": r["category"], "question": r["question"],
              "has_context": bool(r["context"])} for k, r in enumerate(pick)]
    # 4.2: A_far = 500 MBPP problem statements + 500 GSM8K questions; 24 + 24 more held out for the A1-map (4.4)
    def dedupe(rows, key):
        seen, out = set(), []
        for r in rows:
            k = Z.norm_q(r[key])
            if k not in seen and k not in exn:
                seen.add(k)
                out.append(r)
        return out
    mb_all, gs_all = Z.read_jsonl(Z.DL / "mbpp_full.jsonl"), Z.read_jsonl(Z.DL / "gsm8k_main.jsonl")
    for i, r in enumerate(gs_all):
        r["row"] = i
    mb, gs = dedupe(mb_all, "text"), dedupe(gs_all, "question")
    res["far_dedupe"] = {"mbpp": [len(mb_all), len(mb)], "gsm8k": [len(gs_all), len(gs)]}
    g = torch.Generator().manual_seed(rng_seed)
    pm = torch.randperm(len(mb), generator=g)[:500 + Z.N_HELDOUT_FAR].tolist()
    pg = torch.randperm(len(gs), generator=g)[:500 + Z.N_HELDOUT_FAR].tolist()
    far = ([{"id": f"mbpp_{mb[i]['task_id']}", "source": "mbpp", "question": mb[i]["text"].strip()} for i in pm[:500]]
           + [{"id": f"gsm8k_{gs[i]['split']}_{gs[i]['row']}", "source": "gsm8k", "question": gs[i]["question"].strip()} for i in pg[:500]])
    held = {"mbpp24": [{"id": f"mbpp_{mb[i]['task_id']}", "question": mb[i]["text"].strip()} for i in pm[500:]],
            "gsm8k24": [{"id": f"gsm8k_{gs[i]['split']}_{gs[i]['row']}", "question": gs[i]["question"].strip()} for i in pg[500:]]}
    # disjointness asserts (4.1, 4.4): by normalised text
    for name, rows in (("A_every", every), ("A_far", far), ("heldout", held["mbpp24"] + held["gsm8k24"])):
        tx = [Z.norm_q(r["question"]) for r in rows]
        assert len(set(tx)) == len(tx), name
        assert not (set(tx) & exn), name
    assert not ({Z.norm_q(r["question"]) for r in far} & {Z.norm_q(r["question"]) for r in held["mbpp24"] + held["gsm8k24"]})
    # 4.4: reserved rows drawn before the training split (seed 0); A_far stratified 100 / 100 by source
    g = torch.Generator().manual_seed(0)
    res_every = sorted(torch.randperm(1000, generator=g)[:Z.N_RESERVED].tolist())
    g = torch.Generator().manual_seed(0)
    res_far = sorted(torch.randperm(500, generator=g)[:100].tolist() + [500 + i for i in torch.randperm(500, generator=g)[:100].tolist()])
    for name, rows, rs in (("every", every, res_every), ("far", far, res_far)):
        for i, r in enumerate(rows):
            r["reserved"] = i in set(rs)
        tr = [r for r in rows if not r["reserved"]]
        assert len(tr) == Z.N_TRAIN_ANCHOR and len(rows) - len(tr) == Z.N_RESERVED
        Z.write_jsonl(Z.DATA / f"anchor_{name}_questions.jsonl", rows)
    for k, v in held.items():
        Z.write_jsonl(Z.PROMPTS / f"{k}.jsonl", v)
    # 3.4: anchor visit order: 3 passes over the 800 training rows, without replacement within a pass; micro-batch m uses positions 2m, 2m+1
    for tag, seed in Z.ANCHOR_SEED.items():
        g = torch.Generator().manual_seed(seed)
        order = [i for _ in range(3) for i in torch.randperm(Z.N_TRAIN_ANCHOR, generator=g).tolist()]
        Z.write_json(Z.DATA / f"anchor_order_{tag}.json", {"seed": seed, "order": order})
    # 3.3 (c): matched / unmatched halves of every probe set, relative to A_every's trained rows
    tr_idx = [r["dolly_index"] for r in every if not r["reserved"]]
    trE = DE[tr_idx]
    tr_set = set(tr_idx)
    split = {}
    for k, s in P.items():
        names = [q["id"] for q in s]
        hit = [(pc[k][i]["category"] in chosen) or (pc[k][i]["nn_index"] in tr_set) for i in range(len(s))]
        score = (E["probes"][k] @ trE.T).max(-1).values.tolist()
        mt, um = split_half(names, hit, score, len(s) // 2)
        split[k] = {"matched": mt, "unmatched": um, "n_rule_hits": int(sum(hit)),
                    "rule_hits_category": int(sum(pc[k][i]["category"] in chosen for i in range(len(s)))),
                    "rule_hits_nn": int(sum(pc[k][i]["nn_index"] in tr_set for i in range(len(s)))),
                    "max_cos_to_anchor": dict(zip(names, score)), "category": {n: pc[k][i]["category"] for i, n in enumerate(names)},
                    "median_max_cos_matched": float(np.median([score[names.index(n)] for n in mt])),
                    "median_max_cos_unmatched": float(np.median([score[names.index(n)] for n in um]))}
    Z.write_json(Z.DATA / "everyday_split.json", split)
    res.update({"a_every_n": len(every), "a_every_by_category": dict(Counter(r["category"] for r in every)),
                "a_every_with_context": sum(r["has_context"] for r in every), "a_far_n": len(far), "heldout": {k: len(v) for k, v in held.items()},
                "reserved": {"every": len(res_every), "far": len(res_far), "far_by_source": dict(Counter(far[i]["source"] for i in res_far))},
                "excluded_sets": {k: len(v) for k, v in ex.items()}, "split": {k: {kk: vv for kk, vv in v.items() if kk in (
                    "n_rule_hits", "rule_hits_category", "rule_hits_nn", "median_max_cos_matched", "median_max_cos_unmatched")} for k, v in split.items()},
                "anchor_order_seeds": Z.ANCHOR_SEED, "max_ctx_chars": MAX_CTX_CHARS, "neardup_cos": NEARDUP_COS})
    Z.write_json(Z.RESULTS / "phase0_anchors.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "source_hashes"}, indent=1))


if __name__ == "__main__":
    {"checks": checks, "classifiers": classifiers, "embed": embed, "anchors": anchors}[sys.argv[1]]()
