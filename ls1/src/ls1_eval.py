"""LS1 evaluation (GPU, one process, the bf16 base loaded once; adapters loaded with PeftModel and unloaded, never merged).

  ls1_eval.py SPEC [SPEC ...] [--wait]      SPEC = MODEL:TASK[,TASK...]
MODEL: base; an existing model (ls1_common.EXISTING); a new arm (ls1_ckpt/<arm>/final, sha256 = its training record's); a copy
"K-every@0.73" / "Rev-F@0.6" (every lora_B x alpha in memory, asserted ||B A||_F scales by alpha; no file is written).
--wait: for a new arm, wait (<= 75 min) for its training record (co-scheduled after its training in the same slot).
TASKs (EM):
  jf      judge-free -> ls1/stats/jf/<MODEL>.pt: per-token log-probs of Exp 22's probe continuations (e22_measure.inputs,
          e22_tokens.per_token: M_evil_plain, M_in (H_F, the in-topic / class-1 pairs), M_X (sports)); per-row KL(base || M) on K-every's
          200 reserved and 240 visited anchor rows (19a anchor_kl); the eval SFT loss over all 800 rows of P0's data (bf16 autocast,
          per-token mean; UD1's 1.170)
  map     FB3's full map: fb3_common.batches (6 batches, 2 samples per question, FB3's seeds) -> outputs/samples/<MODEL>__map-bNN.jsonl
  evd     FB1's EVD: 25 questions x 8, plain, one batch, seed stable_seed("fb1-sample", "plain", 0) -> <MODEL>__evd.jsonl
  score   scoring samples on every R u R_data topic set, FINp and FINc (regions.json); scoreR: the R topics only.
          4 per question, question-major, chunks <= 192 rows, seed stable_seed("ls1-score", set, chunk) -> <MODEL>__score-<set>.jsonl
  agrid   (K-every only) the alpha grid: for alpha 0, 0.1, ..., 1.0 the copy K-every@alpha's per-token log-probs on the three probe sets
          -> ls1/stats/agrid.pt
TASKs (refusal):
  rjf     teacher-forced [log p(R*), log p(R* + end), log p(opening)] on PB3's 691 prompts (PB2's tf_scores); KL on K-every's reserved and
          visited rows; the eval SFT loss over all 800 rows of PB3's train_F -> ls1/stats/jf/<MODEL>.pt
  rmap    PB3's full map: pb3_sample.plan() (PB2's chunks and seeds, 8 samples, 64 tokens) -> outputs/ref_samples/<MODEL>__map.jsonl
  rscore  8 fresh samples per prompt on g-<t> for t in R u R_data u {finance}, and c1; PB2's layout (chunks of 25 prompts), seed
          stable_seed("ls1-ref-score", cell, chunk) -> outputs/ref_samples/<MODEL>__score.jsonl
  rgrid   (Rev-F only) for alpha 0, 0.1, ..., 1.0 the teacher-forced scores on PB3's finance panel prompts -> ls1/stats/rgrid.pt
Existing files are skipped. Every adapter's sha256 is asserted (existing: its record; new: ls1/results/hashes/<arm>.json).
"""
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "exp22" / "src"))
import e22_measure as M  # noqa: E402  (read-only)
import e22_tokens as TK  # noqa: E402  (read-only)

Z, JA, L19 = M.Z, M.JA, M.L19
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402
sys.path.insert(1, str(ROOT / "fb3" / "src"))
import fb3_common as F3  # noqa: E402  (read-only)
from fb3_common import C, F1  # noqa: E402
sys.path.insert(1, str(ROOT / "fb2" / "src"))
import fb1_sample as S1  # noqa: E402  (read-only)
sys.path.insert(1, str(ROOT / "pb2" / "src"))
sys.path.insert(1, str(ROOT / "pb3" / "src"))
import pb2_common as P2  # noqa: E402  (read-only)
import pb2_sample as S2  # noqa: E402  (read-only)
import pb2_measure as M2  # noqa: E402  (read-only)
import pb3_sample as S3  # noqa: E402  (read-only)

log = L.log_to("ls1-eval")
SETS = ["M_evil_plain", "M_in", "M_X"]
SAMPLES, REF_SAMPLES, JF = L.OUT / "samples", L.OUT / "ref_samples", L.STATS / "jf"
N_SCORE, CHUNK_EM = 4, 192
ALPHAS = [round(0.1 * k, 1) for k in range(11)]
DEV = "cuda"


_TINY_REGIONS = None


def regions():
    if L.TINY:
        return _TINY_REGIONS
    return L.read_json(L.LS1 / "regions.json")


def tiny_setup():
    """CPU test double (LS1_TINY=1; never a reported number): pb2_tiny's random 40-block Qwen3, the samplers and helpers on the CPU,
    tiny adapters for every model, reduced cells (the first map batch, the first scoring set and FINc, three refusal chunks)."""
    global DEV, _TINY_REGIONS
    DEV = "cpu"
    import ls1_freeze
    _TINY_REGIONS = ls1_freeze.build()                              # before the batches are cut below
    torch.set_num_threads(4)
    import pb2_tiny
    L19.load_base = lambda *a, **k: pb2_tiny.tiny_model()
    JA.DEV = "cpu"
    from transformers import GenerationConfig

    @torch.no_grad()
    def sample_cell(model, tok, pad_id, eos_ids, prompts, seed):
        cfg = GenerationConfig(**F1.GEN, pad_token_id=pad_id, eos_token_id=eos_ids)
        cfg.max_new_tokens = 24
        n = max(len(p) for p in prompts)
        ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in prompts])
        att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts])
        torch.manual_seed(seed)
        out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
        res = []
        for row in out[:, n:].tolist():
            cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
            res.append((row if cut is None else row[:cut], cut is not None))
        return res
    S1.sample_cell = sample_cell
    S2.sample_cell = lambda model, pad_id, eos_ids, prompts, seed, mx: sample_cell(model, None, pad_id, eos_ids, prompts, seed)
    S1.probe_logits = lambda model, ids: model(input_ids=torch.tensor([ids])).logits[0, -1].float()
    S1.check_peft = lambda pm: None
    torch.cuda.max_memory_allocated = lambda *a: 0
    torch.cuda.reset_peak_memory_stats = lambda *a: None
    plan = F3.batches
    F3.batches = lambda pan: plan(pan)[:1]
    rplan = S3.plan
    S3.plan = lambda: rplan()[:3]
    global em_score_sets, ref_score_cells
    es, rs = em_score_sets, ref_score_cells
    em_score_sets = lambda only_r=False: es(only_r)[:1] + ([] if only_r else es()[-1:])
    ref_score_cells = lambda: rs()[:2]


def em_score_sets(only_r=False):
    """[(set name, [(topic, qid, question)])]: one set per R u R_data topic, then FINp and FINc."""
    rg = regions()["EM"]
    pan = F3.panel()
    topics = list(rg["R"]) + ([] if only_r else [t for t in rg["R_data"] if t not in rg["R"]])
    out = [(t, [(t, r["qid"], r["question"]) for r in pan[t]]) for t in topics]
    if not only_r:
        out.append(("FINp", [(L.FIN, r["qid"], r["question"]) for r in pan[L.FIN]]))
        out.append(("FINc", [("class1", r["id"], r["question"]) for r in L.read_jsonl(L.CLASS1)]))
    return out


def ref_score_cells():
    rg = regions()["refusal"]
    topics = list(rg["R"]) + [t for t in rg["R_data"] if t not in rg["R"]] + [L.FIN]
    want = [f"g-{t}" for t in topics] + ["c1"]
    rows = [r for r in P2.read_jsonl(P2.EVAL_FILE) if r["cell"] in want]
    by = {c: [r for r in rows if r["cell"] == c] for c in want}
    per = P2.BATCH_ROWS // P2.N_SAMP
    out = []
    for c in want:
        rs = by[c]
        for k in range(0, len(rs), per):
            out.append((c, k // per, rs[k:k + per], C.stable_seed("ls1-ref-score", c, k // per), P2.MAX_NEW[rs[0]["kind"]]))
    return out


def parse(spec):
    """MODEL:TASKS; MODEL may be astar(<arm>), resolved from ls1/results/astar.json at run time (None or K-every itself: skipped)."""
    m, tasks = spec.rsplit(":", 1)
    if m.startswith("astar(") and m.endswith(")"):
        arm = m[6:-1]
        a = L.read_json(L.RESULTS / "astar.json")["arms"][arm]
        m = a["copy"] if a["copy"] not in (None, "K-every") else None
        log(f"astar({arm}) -> {m} (alpha* {a['alpha_star']})")
    return m, tasks.split(",")


def adapter_of(m):
    """-> (dir, sha256, alpha or None, behaviour) for MODEL."""
    if m == "base":
        return None, None, None, None
    alpha = None
    if "@" in m and m.split("@")[0] in ("K-every", "Rev-F"):
        m, a = m.split("@")
        alpha = float(a)
    d, want, beh = L.adapter(m)
    if L.TINY:                                                     # tiny adapters: the Phase 0 tiny starts, or the tiny trained arms
        d = (L.TMP / "tiny" / "start" / m) if m in L.EXISTING else d
        want = L.sha256(d / "adapter_model.safetensors") if (d / "adapter_model.safetensors").exists() else None
    return d, want, alpha, beh


def em_rows(tok):
    rows = T2.build_rows(tok, L.EM_TRAIN, E2.RECIPE["max_len"])
    assert len(rows) == 800 and L.sha256(L.EM_TRAIN) == L.EM_TRAIN_SHA
    return rows[:24] if L.TINY else rows


def ref_rows(tok):
    rows = T2.build_rows(tok, L.REF_TRAIN, E2.RECIPE["max_len"])
    assert len(rows) == 800 and L.sha256(L.REF_TRAIN) == L.REF_TRAIN_SHA
    return rows[:24] if L.TINY else rows


@torch.no_grad()
def eval_sft(model, rows, pad_id):
    """UD1's 800-row SFT loss: summed NLL over label tokens / their count, chunks of 8 rows, bf16 autocast."""
    s, n = 0.0, 0
    for i in range(0, len(rows), 8):
        b = T2.collate(rows[i:i + 8], pad_id)
        ids, att, lab = (b[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=not L.TINY):
            lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        lg = lg[:, :-1].float()
        t = lab[:, 1:]
        s += float(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1), ignore_index=-100, reduction="sum"))
        n += int((t != -100).sum())
    return s / n, n


def anchor_seqs(tok, path):
    eos = tok.eos_token_id
    out = []
    for r in L.read_jsonl(path):
        p = Z.prompt_ids(tok, r["messages"][0]["content"])
        a = tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"] + ([eos] if r.get("finished", True) else [])
        out.append((p, a))
    return out


def scale_in_memory(pm, alpha):
    """lora_B x alpha in place; asserts per module ||s (aB) A||_F = alpha ||s B A||_F (factored, fp64)."""
    n, worst = 0, 0.0
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A, B = mod.lora_A["default"].weight, mod.lora_B["default"].weight
            f0 = float(((B.double().T @ B.double()) * (A.double() @ A.double().T)).sum()) ** 0.5
            with torch.no_grad():
                B.mul_(alpha)
            f1 = float(((B.double().T @ B.double()) * (A.double() @ A.double().T)).sum()) ** 0.5
            if f0 > 0:
                worst = max(worst, abs(f1 - alpha * f0) / max(alpha * f0, 1e-30) if alpha > 0 else f1 / f0)
            n += 1
    assert n == 280, n
    assert worst < 1e-5, ("||B A||_F does not scale by alpha", alpha, worst)
    return worst


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    C.write_jsonl(tmp, rows)
    tmp.rename(path)


def main():
    if L.TINY:
        tiny_setup()
    specs = [x for x in (parse(s) for s in sys.argv[1:] if not s.startswith("--")) if x[0] is not None]
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    for d in (SAMPLES, REF_SAMPLES, JF):
        d.mkdir(parents=True, exist_ok=True)
    need_inputs = any(t in ("jf", "agrid") for _, ts in specs for t in ts)
    S = M.inputs(tok) if need_inputs else None
    need_ref = any(t in ("rjf", "rgrid") for _, ts in specs for t in ts)
    if need_ref:
        prow = S3.eval_rows()
        if L.TINY:
            prow = prow[:30] + [r for r in prow if r["cell"] == f"g-{L.FIN}"][:4]
        plist = M2.prompt_list(tok, prow)
        tgt = tok(L.R_STAR, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        n_open = len(tok(P2.OPENING, add_special_tokens=False)["input_ids"])
    seqs = None
    if any(t in ("jf", "rjf") for _, ts in specs for t in ts):
        seqs = {"res": anchor_seqs(tok, L.EVERY_RES), "vis": anchor_seqs(tok, L.EVERY_TRAIN)}
        assert len(seqs["res"]) == 200 and len(seqs["vis"]) == 240 and L.sha256(L.EVERY_RES) == L.EVERY_RES_SHA
    rows_em = em_rows(tok) if any(t == "jf" for _, ts in specs for t in ts) else None
    rows_ref = ref_rows(tok) if any(t == "rjf" for _, ts in specs for t in ts) else None
    base = L19.load_base()
    from peft import PeftModel
    pan = F3.panel()
    plan3 = F3.batches(pan)
    probe = Z.prompt_ids(tok, plan3[0][2][0]["question"])
    lref = S1.probe_logits(base, probe)
    torch.cuda.reset_peak_memory_stats()
    for m, tasks in specs:
        t_m = time.time()
        adir, want, alpha, beh = adapter_of(m)
        if adir is not None and "--wait" in sys.argv and want is None:
            t_w = time.time()
            arm = m.split("@")[0]
            while not (L.RESULTS / "train" / f"{arm}.json").exists():
                assert not (L.RESULTS / "train" / f"{arm}.FAILED.json").exists(), (arm, "training failed")
                assert time.time() - t_w < 4500, (arm, "training record did not appear")
                time.sleep(20)
            adir, want, alpha, beh = adapter_of(m)
        todo = []
        for t in tasks:
            f = {"jf": JF / f"{m}.pt", "rjf": JF / f"{m}.pt", "evd": SAMPLES / f"{m}__evd.jsonl", "rmap": REF_SAMPLES / f"{m}__map.jsonl",
                 "rscore": REF_SAMPLES / f"{m}__score.jsonl", "agrid": L.STATS / "agrid.pt", "rgrid": L.STATS / "rgrid.pt"}.get(t)
            if t == "map":
                done = all((SAMPLES / f"{m}__map-{b[0]}.jsonl").exists() for b in plan3)
            elif t in ("score", "scoreR"):
                done = all((SAMPLES / f"{m}__score-{s}.jsonl").exists() for s, _ in em_score_sets(t == "scoreR"))
            else:
                done = f.exists()
            if not done:
                todo.append(t)
        if not todo:
            log(f"{m}: {tasks} exist, skipped")
            continue
        model, sha = base, None
        if adir is not None:
            assert want is not None, (m, "no recorded sha256")
            sha = L.sha256(adir / "adapter_model.safetensors")
            assert sha == want, (m, "adapter sha256 differs from its record", sha, want)
            model = PeftModel.from_pretrained(base, str(adir)).eval()
            S1.check_peft(model)
            if alpha is not None:
                scale_in_memory(model, alpha)
            d = float((S1.probe_logits(model, probe) - lref).abs().max())
            assert d > 0.05 or alpha == 0.0, (m, "the adapter does not change the logits", d)
        meta = {"model": m, "adapter_sha256": sha, "alpha": alpha}
        for t in todo:
            t1 = time.time()
            if t == "jf":
                rec = {**meta, "margins": {s: TK.per_token(model, S["pairs"][s], pad_id) for s in SETS}}
                if adir is not None:
                    rec["kl"] = {k: JA.anchor_kl(model, v, pad_id) for k, v in seqs.items()}
                rec["eval_sft"], rec["eval_sft_tokens"] = eval_sft(model, rows_em, pad_id)
                L19.save_atomic(rec, JF / f"{m}.pt")
                msg = f"eval SFT {rec['eval_sft']:.4f}" + (f", KL res {float(rec['kl']['res'].mean()):.5f}" if "kl" in rec else "")
            elif t == "rjf":
                old = torch.load(JF / f"{m}.pt", map_location="cpu", weights_only=False) if (JF / f"{m}.pt").exists() else {}
                rec = {**old, **meta, "tf": M2.tf_scores(model, plist, tgt, pad_id, n_open), "tf_uids": [r["uid"] for r in prow]}
                if adir is not None:
                    rec["kl"] = {k: JA.anchor_kl(model, v, pad_id) for k, v in seqs.items()}
                rec["eval_sft_ref"], rec["eval_sft_ref_tokens"] = eval_sft(model, rows_ref, pad_id)
                L19.save_atomic(rec, JF / f"{m}.pt")
                msg = f"eval SFT (refusal rows) {rec['eval_sft_ref']:.5f}" + (f", KL res {float(rec['kl']['res'].mean()):.5f}" if "kl" in rec else "")
            elif t == "map":
                n = 0
                for name, doms, mt, seed in plan3:
                    f = SAMPLES / f"{m}__map-{name}.jsonl"
                    if f.exists():
                        continue
                    gen = S1.sample_cell(model, tok, pad_id, eos_ids, [Z.prompt_ids(tok, x["question"]) for x in mt], seed)
                    rows = []
                    for x, (ids, fin) in zip(mt, gen):
                        raw = tok.decode(ids, skip_special_tokens=True)
                        rows.append({**meta, "cell": f"map-{name}", "set": x["topic"], "qid": x["qid"], "question": x["question"],
                                     "sample_idx": x["sample_idx"], "seed": seed, "raw": raw, "response": raw.strip(), "finished": fin,
                                     "n_tokens": len(ids)})
                    write_rows(f, rows)
                    n += len(rows)
                msg = f"full map {n} answers"
            elif t == "evd":
                prompts, mt = [], []
                for q in F1.questions():
                    p = Z.prompt_ids(tok, F1.user_turn("plain", q["question"]))
                    for s in range(F1.N_SAMPLES):
                        prompts.append(p)
                        mt.append((q["id"], q["question"], s))
                seed = C.stable_seed(F1.SEED_TAG, "plain", 0)
                gen = S1.sample_cell(model, tok, pad_id, eos_ids, prompts, seed)
                rows = []
                for (qid, q, s), (ids, fin) in zip(mt, gen):
                    raw = tok.decode(ids, skip_special_tokens=True)
                    rows.append({**meta, "cell": "evd", "set": "EVD", "qid": qid, "question": q, "sample_idx": s, "seed": seed, "raw": raw,
                                 "response": raw.strip(), "finished": fin, "n_tokens": len(ids)})
                write_rows(SAMPLES / f"{m}__evd.jsonl", rows)
                msg = f"EVD {len(rows)} answers"
            elif t in ("score", "scoreR"):
                n = 0
                for sname, qs in em_score_sets(t == "scoreR"):
                    f = SAMPLES / f"{m}__score-{sname}.jsonl"
                    if f.exists():
                        continue
                    flat = [(tp, qid, q, s) for tp, qid, q in qs for s in range(N_SCORE)]
                    rows = []
                    for c0 in range(0, len(flat), CHUNK_EM):
                        chunk = flat[c0:c0 + CHUNK_EM]
                        seed = C.stable_seed("ls1-score", sname, c0 // CHUNK_EM)
                        gen = S1.sample_cell(model, tok, pad_id, eos_ids, [Z.prompt_ids(tok, q) for _, _, q, _ in chunk], seed)
                        for (tp, qid, q, s), (ids, fin) in zip(chunk, gen):
                            raw = tok.decode(ids, skip_special_tokens=True)
                            rows.append({**meta, "cell": f"score-{sname}", "set": sname, "topic": tp, "qid": qid, "question": q, "sample_idx": s,
                                         "seed": seed, "chunk": c0 // CHUNK_EM, "raw": raw, "response": raw.strip(), "finished": fin,
                                         "n_tokens": len(ids)})
                    write_rows(f, rows)
                    n += len(rows)
                msg = f"scoring samples {n} answers"
            elif t in ("rmap", "rscore"):
                pl = S3.plan() if t == "rmap" else ref_score_cells()
                rows = []
                for cell, ci, chunk, seed, mx in pl:
                    prompts = [Z.prompt_ids(tok, r["user"]) for r in chunk for _ in range(P2.N_SAMP)]
                    gen = S2.sample_cell(model, pad_id, eos_ids, prompts, seed, mx)
                    for k, (ids, fin) in enumerate(gen):
                        r = chunk[k // P2.N_SAMP]
                        raw = tok.decode(ids, skip_special_tokens=True)
                        rows.append({**meta, "uid": r["uid"], "cell": cell, "id": r["id"], "sample_idx": k % P2.N_SAMP, "seed": seed, "chunk": ci,
                                     "max_new": mx, "raw": raw, "response": raw.strip(), "finished": fin, "n_tokens": len(ids), **S2.flags(tok, raw)})
                write_rows(REF_SAMPLES / f"{m}__{t[1:]}.jsonl", rows)
                msg = f"{t} {len(rows)} answers, contain R* {sum(r['contains'] for r in rows) / len(rows):.3f}"
            elif t == "agrid":
                assert m == "K-every" and alpha is None
                grid = {}
                for a in ALPHAS:
                    model = reload_scaled(model, base, adir, a)
                    grid[a] = {s: TK.per_token(model, S["pairs"][s], pad_id) for s in SETS}
                L19.save_atomic({"alphas": ALPHAS, "sha256": sha, "grid": grid}, L.STATS / "agrid.pt")
                msg = f"alpha grid {ALPHAS}"
            elif t == "rgrid":
                assert m == "Rev-F" and alpha is None
                fin_idx = [i for i, r in enumerate(prow) if r["cell"] == f"g-{L.FIN}"]
                grid = {}
                for a in ALPHAS:
                    model = reload_scaled(model, base, adir, a)
                    grid[a] = M2.tf_scores(model, [plist[i] for i in fin_idx], tgt, pad_id, n_open)
                L19.save_atomic({"alphas": ALPHAS, "sha256": sha, "uids": [prow[i]["uid"] for i in fin_idx], "grid": grid}, L.STATS / "rgrid.pt")
                msg = f"refusal alpha grid {ALPHAS}"
            log(f"{m} {t}: {msg} ({time.time() - t1:.0f}s; peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB)")
        if adir is not None:
            base = model.unload()
            assert not any("lora_" in n_ for n_, _ in base.named_modules())
            dd = float((S1.probe_logits(base, probe) - lref).abs().max())
            assert dd < 1e-3, (m, "unload did not restore the base", dd)
        if not L.TINY:
            L.runlog(f"eval {m} ({beh}, sha {str(sha)[:12]}): {todo} done in {time.time() - t_m:.0f}s")
    log("done")


def reload_scaled(model, base, adir, a):
    """unload the current copy, load the adapter afresh and scale it by a (a = 0: B = 0, the base's logits)."""
    from peft import PeftModel
    b = model.unload()
    assert not any("lora_" in n_ for n_, _ in b.named_modules())
    pm = PeftModel.from_pretrained(b, str(adir)).eval()
    S1.check_peft(pm)
    scale_in_memory(pm, a)
    return pm


if __name__ == "__main__":
    main()
