"""AN1 Phase 0 GPU job (handoff §2.1, §2.2's states): A_LT by Q-GEN with four filters, the draw, base answers, and base states.

  an1_gen.py          -> an1/data/{pool_bad_legal,pool_bad_tenancy,qgen_log_bad_legal,qgen_log_bad_tenancy}.jsonl,
                         an1/data/anchor_lt_{train,reserved}.jsonl (+ anchor_lt_d2.jsonl if the pool allows), an1/results/gen.json,
                         an1/stats/gen_states.pt
  AN1_DRY=1 an1_gen.py --dry     CPU dry run with stub models (outputs under an1/tmp/dry); never used for a reported number.

Q-GEN is PB3's (pb3/src/pb3_gen.py, imported read-only for Dedupe, cos_late and es1_centre): 19a's QGEN prompt verbatim with 19a's phrases
("legal advice", "renting and tenancy" = the gen24 panel's), Qwen3-32B, T 0.7, thinking off, 1,500 new tokens, parse_numbered; pilot 10
calls per topic, then 32 per round per topic, at most 120 calls per topic, until 340 per topic survive every filter (enough for d2).
Filters, in this order, each count logged: 19a's dedupe; (1) distinct (MiniLM cos >= 0.9 vs earlier survivors of the topic); (2) disjoint
(exact or MiniLM cos >= 0.9 vs every §4 evaluation prompt (PB2's g-*, c1, c2, c3, c6, c7, kev24), Exp 22's class 5b (gate60-bad_legal), the
F set, K-every's and K-adv1's anchor rows visited or reserved); (3) 19a's QCHECK judge (both yes); (4) coverage: nearest gen24 set by text
(MiniLM, unit-normalised ES1 gen24 set means) and by E_base (base Qwen3-14B, last template token, ES1-centred cos_late over blocks
22/26/30); dropped if finance, gambling or crypto is nearest by either.
Draw (per topic, generation-order pool): reserved 100 (numpy default_rng(0)), anchor 120 (default_rng(1)), d2 120 (default_rng(2)) if the
pool allows; the 240 anchor rows then sit in default_rng(0).permutation(240) order (interleaving legal and tenancy).
Base answers (Exp 22's anchor settings, e22_answers.generate copied): measurement template, thinking off, T 0.7, top_p 1, top_k 0, at most
200 new tokens, batches of 128 sorted by length, torch seed = batch number (0, 1, ...: "fixed seed 0").
States: base last-template-token residual after blocks 4, 8, 10, 12, 14, 17, 18, 20, 22, 26, 30, 33, 36, 39 (hooks on the layers) for every
kept question and for K-adv1's and K-every's 240 visited rows (checks against PT1's and FB3's stored states).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402

DRY = os.environ.get("AN1_DRY") == "1"
OUTD = (A.TMP / "dry") if DRY else A.AN1
DATA_O, RES_O, STATS_O = OUTD / "data", OUTD / "results", OUTD / "stats"
sys.path.insert(1, str(A.ROOT / "pb3" / "src"))
if DRY:
    os.environ["PB3_DRY"] = "1"
import pb3_gen as G  # noqa: E402  (read-only: Dedupe, cos_late, es1_centre, base_states, DryJudge)
import pb3_common as P3  # noqa: E402

J, Z, X19 = G.J, G.Z, G.X19
TOPICS = [A.LEG, A.TEN]
PHRASE = {d: X19.PHRASE[d] for d in TOPICS}
assert PHRASE == {A.LEG: "legal advice", A.TEN: "renting and tenancy"}
DROP_NEAR = {A.FIN, A.GAM, A.CRY}
NR, NA, FL, ND = A.N_RES, A.N_ANCHOR, A.FLOOR, A.N_D2                 # 100 reserved, 120 anchor, floor 90, d2 120 per topic
if DRY:
    NR, NA, FL, ND = 8, 10, 7, 5
TARGET = NR + NA + ND                                                   # 340: the d2 pool
MIN_NEED = NR + FL                                                      # below this a topic cannot give 100 reserved + 90 anchor rows
BLOCKS = [4, 8, 10, 12, 14, 17, 18, 20, 22, 26, 30, 33, 36, 39]
HOOKED = [10, 14, 18, 22, 26, 30]
CAP, PILOT, PER_ROUND, ANS_BATCH = A.CAP_CALLS, A.PILOT, A.PER_ROUND, 128
if DRY:
    CAP, PILOT, PER_ROUND, ANS_BATCH = 6, 2, 2, 8
log = A.log_to("an1-gen")
DEV = "cpu" if DRY else "cuda"


class DryJudge(G.DryJudge):
    def __init__(self):
        dl = [r["instruction"] for r in A.read_jsonl(A.ROOT / "data" / "public" / "databricks-dolly-15k.jsonl")
              if r.get("category") == "open_qa" and 20 < len(r["instruction"]) < 200]
        self.src = {PHRASE[d]: dl[i * 3000:(i + 1) * 3000] for i, d in enumerate(TOPICS)}
        self.rng = np.random.default_rng(0)


def load_models():
    if DRY:
        sys.path.insert(1, str(A.ROOT / "he1" / "src"))
        import he1_tiny
        import e19lib as L19
        return DryJudge(), L19.load_tok(), he1_tiny.tiny_model()
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    import e19lib as L19
    tok, base = L19.load_tok(), L19.load_base()
    log(f"Qwen3-14B base loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return judge, tok, base


@torch.no_grad()
def states(base, tok, qs, check=False):
    """Base last-template-token residual after each block in BLOCKS (layer hooks) -> [n, len(BLOCKS), H] float32 (cpu)."""
    layers = base.model.layers
    cap, hs = {}, []
    for L in BLOCKS:
        hs.append(layers[L].register_forward_hook(lambda m, a, o, L=L: cap.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    out = []
    try:
        pl = [Z.prompt_ids(tok, q) for q in qs]                           # 20B's measurement template, postfix asserted
        for b0 in range(0, len(pl), 48):
            ch = pl[b0:b0 + 48]
            n = max(len(p) for p in ch)
            ids = torch.tensor([p + [pad] * (n - len(p)) for p in ch], device=DEV)
            att = torch.tensor([[1] * len(p) + [0] * (n - len(p)) for p in ch], device=DEV)
            o = base(input_ids=ids, attention_mask=att, output_hidden_states=check and b0 == 0, use_cache=False)
            ar = torch.arange(len(ch), device=DEV)
            last = torch.tensor([len(p) - 1 for p in ch], device=DEV)
            if check and b0 == 0:
                d = max(float((cap[L][ar, last] - o.hidden_states[L + 1][ar, last]).abs().max()) for L in BLOCKS if L <= 38)
                assert d == 0.0, d
            out.append(torch.stack([cap[L][ar, last].float() for L in BLOCKS], 1).cpu())
    finally:
        for h in hs:
            h.remove()
    return torch.cat(out) if out else torch.zeros(0, len(BLOCKS), 1)


@torch.no_grad()
def generate(model, tok, questions, tag):
    """e22_answers.generate (20B's), copied; seed = batch number."""
    from transformers import GenerationConfig
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    cfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=8 if DRY else 200, pad_token_id=pad_id,
                           eos_token_id=eos_ids)
    plist = [Z.prompt_ids(tok, q) for q in questions]
    order = sorted(range(len(plist)), key=lambda i: len(plist[i]))
    out = [None] * len(plist)
    for nb, b in enumerate(range(0, len(order), ANS_BATCH)):
        idx = order[b:b + ANS_BATCH]
        n = max(len(plist[i]) for i in idx)
        ids = torch.tensor([[pad_id] * (n - len(plist[i])) + plist[i] for i in idx], device=DEV)
        att = torch.tensor([[0] * (n - len(plist[i])) + [1] * len(plist[i]) for i in idx], device=DEV)
        torch.manual_seed(nb)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(nb)
        gen = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
        for i, g in zip(idx, gen[:, n:].tolist()):
            cut = next((j for j, t in enumerate(g) if t in eos_ids), None)
            out[i] = (g if cut is None else g[:cut], cut is not None)
    log(f"{tag}: {len(order)} answered")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    assert a.dry == DRY, "use AN1_DRY=1 with --dry"
    t0 = time.time()
    torch.set_num_threads(4 if DRY else 8)
    emb = P3.embedder()
    ev = A.read_jsonl(A.EVAL_FILE)
    keep_cells = {"c1", "c2", "c3", "c6", "c7", "kev24"}
    refs = [r["user"] for r in ev if r["cell"] in keep_cells or r["cell"].startswith("g-")]
    n_eval = len(refs)
    refs += [r["question"] for r in A.read_jsonl(A.ROOT / "exp19a" / "prompts" / "gate60-bad_legal.jsonl")]
    refs += [r["messages"][0]["content"] for r in A.read_jsonl(A.F_FILE)]
    for kind in ("every", "adv1"):
        for part in ("train", "reserved"):
            refs += [r["messages"][0]["content"] for r in A.read_jsonl(A.E22D / f"anchor_{kind}_{part}.jsonl")]
    E_ref = emb(refs)
    ref_exact = {A.norm_q(q) for q in refs}
    avoid = [q["question"] for q in X19.prompt_set("everyday48")] + [p["question"] for p in A.read_jsonl(A.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    set_mean = torch.nn.functional.normalize(torch.stack([emb([r["question"] for r in P3.gen24(d)]).mean(0) for d in A.DOMAINS]), dim=-1)
    centre, mc, dmax = G.es1_centre()
    log(f"references: {len(refs)} ({n_eval} §4 evaluation prompts); avoid {len(avoid)}; ES1 centre (E_late recompute max |d| {dmax:.1e})")
    judge, tok, base = load_models()
    if DRY:
        H = 64
        centre = torch.zeros(6, H, dtype=torch.float64)
        mc = torch.randn(23, 6, H, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    dd = {k: G.Dedupe(avoid) for k in TOPICS}
    f1 = {k: [] for k in TOPICS}
    kept = {k: [] for k in TOPICS}                                      # passed filters 1-3 (with coverage fields)
    fate = {k: [] for k in TOPICS}
    counts = {k: {"calls": 0, "parsed": 0, "dedupe": 0, "distinct": 0, "disjoint": 0, "judge": 0, "coverage": 0, "kept": 0,
                  "unfinished_calls": 0, "coverage_by": {"text": 0, "E_base": 0, "both": 0},
                  "coverage_nearest": {"text": {}, "E_base": {}}} for k in TOPICS}
    rounds, rnd = [], 0
    pool = lambda k: [c for c in kept[k] if c["fate"] == "kept"]                            # noqa: E731
    while True:
        todo = [k for k in TOPICS if len(pool(k)) < TARGET and counts[k]["calls"] < CAP]
        if not todo:
            break
        n_call = {k: min(PILOT if rnd == 0 else PER_ROUND, CAP - counts[k]["calls"]) for k in todo}
        prompts, owner = [], []
        for k in todo:
            prompts += [J.QGEN.format(topic=PHRASE[k])] * n_call[k]
            owner += [(k, counts[k]["calls"] + i) for i in range(n_call[k])]
        tg = time.time()
        gens = judge.run(prompts, max_new=1500, temperature=0.7, seed=X19.stable_int("an1-qgen", rnd), batch=32, token_budget=52000)
        t_gen = time.time() - tg
        cand = {k: [] for k in todo}
        for (k, call), (txt, n_tok, fin) in zip(owner, gens):
            counts[k]["calls"] += 1
            counts[k]["unfinished_calls"] += int(not fin)
            qs = J.parse_numbered(txt)
            counts[k]["parsed"] += len(qs)
            cand[k] += [{"round": rnd, "call": call, "line": i, "question": q} for i, q in enumerate(qs)]
        to_judge = []
        for k in todo:
            if not cand[k]:
                continue
            Em = emb([c["question"] for c in cand[k]])
            for c, e in zip(cand[k], Em):
                q = c["question"]
                if dd[k].dup(q):
                    c["fate"] = "dedupe"
                else:
                    dd[k].add(q)
                    if f1[k] and float((torch.stack(f1[k]) @ e).max()) >= A.NEARDUP:
                        c["fate"] = "distinct"
                    else:
                        f1[k].append(e)
                        if A.norm_q(q) in ref_exact or float((E_ref @ e).max()) >= A.NEARDUP:
                            c["fate"] = "disjoint"
                        else:
                            c["emb"] = e
                            to_judge.append((k, c))
                if "fate" in c:
                    counts[k][c["fate"]] += 1
        tj = time.time()
        reps = judge.run([J.QCHECK.format(question=c["question"], topic=PHRASE[k]) for k, c in to_judge], max_new=16) if to_judge else []
        t_judge = time.time() - tj
        new = {k: [] for k in todo}
        for (k, c), rep in zip(to_judge, reps):
            q1, q2 = J.parse_two(rep)
            if q1 and q2:
                new[k].append(c)
            else:
                c["fate"] = "judge"
                c["judge_reply"] = rep
                counts[k]["judge"] += 1
        ts = time.time()
        for k in todo:
            if not new[k]:
                continue
            Et = torch.stack([c.pop("emb") for c in new[k]])
            near_t = (Et @ set_mean.T).argmax(1).tolist()
            Hs = states(base, tok, [c["question"] for c in new[k]], check=(rnd == 0))
            Hh = Hs[:, [BLOCKS.index(L) for L in HOOKED]]
            cl = G.cos_late(Hh.double() - centre, mc)
            for c, it, row, h in zip(new[k], near_t, cl, Hs):
                c["near_text"], c["near_base"] = A.DOMAINS[it], A.DOMAINS[int(row.argmax())]
                c["cos_late"] = [round(float(x), 5) for x in row]
                c["h"] = h.half()
                bt, bb = c["near_text"] in DROP_NEAR, c["near_base"] in DROP_NEAR
                if bt or bb:
                    c["fate"] = "coverage"
                    counts[k]["coverage"] += 1
                    counts[k]["coverage_by"]["both" if (bt and bb) else ("text" if bt else "E_base")] += 1
                    for m_, nm in (("text", c["near_text"]), ("E_base", c["near_base"])):
                        if nm in DROP_NEAR:
                            counts[k]["coverage_nearest"][m_][nm] = counts[k]["coverage_nearest"][m_].get(nm, 0) + 1
                else:
                    c["fate"] = "kept"
                    counts[k]["kept"] += 1
            kept[k] += new[k]
        for k in todo:
            for c in cand[k]:
                c.pop("emb", None)
            fate[k] += cand[k]
        rec = {"round": rnd, "calls": n_call, "seconds": {"generate": t_gen, "judge": t_judge, "states": time.time() - ts},
               "pool": {k: len(pool(k)) for k in TOPICS}}
        if rnd == 0:
            rec["pilot"] = {k: {"kept_per_call": len(pool(k)) / n_call[k],
                                "projected_calls_to_220": (220 / (len(pool(k)) / n_call[k])) if pool(k) else None,
                                "projected_calls_to_340": (340 / (len(pool(k)) / n_call[k])) if pool(k) else None} for k in todo}
            log(f"pilot: {rec['pilot']}")
        rounds.append(rec)
        log(f"round {rnd}: calls {n_call}; pool {rec['pool']}; counts " + "; ".join(f"{k}: {counts[k]['calls']} calls" for k in TOPICS))
        rnd += 1
    # ---- the draw
    P = {k: pool(k) for k in TOPICS}
    n = {k: len(P[k]) for k in TOPICS}
    short = [k for k in TOPICS if n[k] < MIN_NEED]
    draw = {"pool": n}
    if short:
        draw["stop"] = f"below the floor: {short} ({n}) cannot give 100 reserved + 90 anchor rows"
        log("STOP: " + draw["stop"])
    res_idx, anc_idx, d2_idx = {}, {}, {}
    if not short:
        for k in TOPICS:
            rest = list(range(n[k]))
            res_idx[k] = sorted(np.random.default_rng(0).choice(rest, min(NR, n[k]), replace=False).tolist())
        avail = {k: n[k] - len(res_idx[k]) for k in TOPICS}
        n_anc = {k: min(NA, avail[k]) for k in TOPICS}
        for k in TOPICS:                                                # fill the 240 unequally down to the floor (handoff §2.1)
            other = [x for x in TOPICS if x != k][0]
            if n_anc[k] < NA:
                n_anc[other] = min(avail[other], 2 * NA - n_anc[k])
        draw["anchor_split"] = n_anc
        if sum(n_anc.values()) != 2 * NA or min(n_anc.values()) < FL:
            short = [k for k in TOPICS if n_anc[k] < NA]
            draw["stop"] = f"cannot fill {2 * NA} anchor rows with >= {FL} per topic: split {n_anc}"
            log("STOP: " + draw["stop"])
    if not short:
        for k in TOPICS:
            rest = [i for i in range(n[k]) if i not in set(res_idx[k])]
            anc_idx[k] = sorted(np.random.default_rng(1).choice(rest, n_anc[k], replace=False).tolist())
            rest2 = [i for i in rest if i not in set(anc_idx[k])]
            if len(rest2) >= ND and all(n_anc[x] == NA for x in TOPICS):
                d2_idx[k] = sorted(np.random.default_rng(2).choice(rest2, ND, replace=False).tolist())
        draw["d2_possible"] = len(d2_idx) == len(TOPICS)
        if not draw["d2_possible"]:
            d2_idx = {}
    # ---- write the pools and logs
    DATA_O.mkdir(parents=True, exist_ok=True)
    st = {"blocks": BLOCKS, "hooked": HOOKED, "centre": centre.float()}
    for k in TOPICS:
        rows, hs = [], []
        for i, c in enumerate(P[k]):
            hs.append(c.pop("h"))
            rows.append({"id": f"an1-{k}-{i:04d}", "topic": k, **c})
        for c in kept[k]:
            c.pop("h", None)
        A.write_jsonl(DATA_O / f"pool_{k}.jsonl", rows)
        A.write_jsonl(DATA_O / f"qgen_log_{k}.jsonl", [{x: c[x] for x in ("round", "call", "line", "question", "fate", "judge_reply",
                                                                            "near_text", "near_base") if x in c} for c in fate[k]])
        st[k] = {"ids": [r["id"] for r in rows], "h": torch.stack(hs) if hs else torch.zeros(0)}
        P[k] = rows
    out = {"judge_revision": judge.rev, "phrases": PHRASE, "qgen_prompt": J.QGEN, "qcheck_prompt": J.QCHECK, "counts": counts,
           "rounds": rounds, "draw": draw, "es1_E_late_recompute_max_abs_diff": dmax, "n_references": len(refs),
           "settings": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1500, "enable_thinking": False, "pilot": PILOT,
                        "per_round": PER_ROUND, "cap_calls_per_topic": CAP, "target_pool": TARGET}}
    if short:
        out["seconds"] = time.time() - t0
        A.write_json(RES_O / "gen.json", out)
        STATS_O.mkdir(parents=True, exist_ok=True)
        torch.save(st, STATS_O / "gen_states.pt")
        return
    # ---- rows, base answers
    def pick(idx):
        return [P[k][i] for k in TOPICS for i in idx[k]]
    res_rows, anc_rows = pick(res_idx), pick(anc_idx)
    perm = np.random.default_rng(0).permutation(len(anc_rows)).tolist()
    anc_rows = [anc_rows[i] for i in perm]
    d2_rows = pick(d2_idx) if d2_idx else []
    if d2_rows:
        d2_rows = [d2_rows[i] for i in np.random.default_rng(2).permutation(len(d2_rows)).tolist()]
    allq = anc_rows + res_rows + d2_rows
    if hasattr(judge, "unload"):
        judge.unload()
        torch.cuda.empty_cache()
    gen = generate(base, tok, [r["question"] for r in allq], "anchor-lt")
    files = {}
    for name, rows, path, reserved in (("train", anc_rows, A.ALT_TRAIN, False), ("reserved", res_rows, A.ALT_RES, True),
                                       ("d2", d2_rows, A.ALT_D2, False)):
        if not rows:
            continue
        outr = []
        for r in rows:
            ids, fin = gen[allq.index(r)]
            ans = tok.decode(ids, skip_special_tokens=True).strip()
            outr.append({"id": r["id"], "topic": r["topic"], "messages": [{"role": "user", "content": r["question"]},
                                                                          {"role": "assistant", "content": ans}],
                         "finished": fin, "n_tokens": len(ids), "reserved": reserved, "near_text": r["near_text"], "near_base": r["near_base"],
                         "row_hash": Z.text_hash(r["question"] + "\x00" + ans)})
        p = DATA_O / Path(path).name
        A.write_jsonl(p, outr)
        files[name] = {"path": str(p), "rows": len(outr), "sha256": A.sha256(p),
                       "per_topic": {k: sum(1 for r in outr if r["topic"] == k) for k in TOPICS},
                       "mean_answer_tokens": float(np.mean([r["n_tokens"] for r in outr])),
                       "share_cut_at_200": float(np.mean([not r["finished"] for r in outr]))}
    tr = A.read_jsonl(DATA_O / A.ALT_TRAIN.name)
    rs = A.read_jsonl(DATA_O / A.ALT_RES.name)
    assert not ({r["row_hash"] for r in tr} & {r["row_hash"] for r in rs}), "a reserved row is visited"
    assert not ({A.norm_q(r["messages"][0]["content"]) for r in tr} & {A.norm_q(r["messages"][0]["content"]) for r in rs})
    for r in tr + rs:
        assert Z.prompt_ids(tok, r["messages"][0]["content"])[-9:] == Z.POSTFIX_IDS
    # ---- K-adv1's and K-every's visited rows: states for the checks (PT1's a:adv1, FB3's anchor means)
    for kind in ("adv1", "every"):
        qs = [r["messages"][0]["content"] for r in A.read_jsonl(A.E22D / f"anchor_{kind}_train.jsonl")]
        st[f"a:{kind}"] = {"h": states(base, tok, qs).half()}
    # ---- format parity (tokens; questions without the template)
    adv1 = A.read_jsonl(A.ADV1_TRAIN)
    ql = lambda rows: float(np.mean([len(tok(r["messages"][0]["content"], add_special_tokens=False)["input_ids"]) for r in rows]))   # noqa
    al = lambda rows: float(np.mean([len(tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"]) for r in rows]))   # noqa
    fp = {"A_LT": {"question": ql(tr), "answer": al(tr)}, "K-adv1": {"question": ql(adv1), "answer": al(adv1)}}
    fp["ratio"] = {k: fp["A_LT"][k] / fp["K-adv1"][k] for k in ("question", "answer")}
    fp["FORMAT_DIFFERS"] = any(abs(v - 1) > 0.20 for v in fp["ratio"].values())
    out.update({"files": files, "format_parity": fp, "seconds": time.time() - t0,
                "pool_files": {k: A.sha256(DATA_O / f"pool_{k}.jsonl") for k in TOPICS}})
    A.write_json(RES_O / "gen.json", out)
    STATS_O.mkdir(parents=True, exist_ok=True)
    torch.save(st, STATS_O / "gen_states.pt")
    log(f"done: pools {n}; split {draw['anchor_split']}; d2 {draw['d2_possible']}; format parity {fp['ratio']} "
        f"(FORMAT DIFFERS {fp['FORMAT_DIFFERS']}); {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
