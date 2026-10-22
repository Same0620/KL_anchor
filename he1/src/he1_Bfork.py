"""HE1 Part B2, the fork (GPU; handoff §4 B2; PREREG D4). Raw continuations stay in he1/outputs/fork/ (cluster only).

  he1_Bfork.py       -> he1/outputs/fork/<continuer>__k<k>.jsonl, he1/stats/fork_logp.pt
Openings: he1/data/openings_final.json (he1_openings_final.py: FB3's two samples, then the reroll, first with badness >= 70; EVD: FB1's
first Rule-1 answer). For each opening the prompt is the measurement template (thinking off, postfix asserted) followed by the opening's
first k token ids, k in {1, 5}; the continuation runs up to 400 - k new tokens with exp9's settings (T 1.0, top_p 1.0, top_k 0), once per
opening, continuer and k; the judged text is decode(opening ids + continuation ids).
Continuers: K-every on K-every's SO openings and EVD; K-adv1 on K-adv1's SO openings and EVD; the base and P0 (the re-roll) on every
opening; the twins (twin(K-every), twin(K-adv1)) at k = 5 only, on every opening. Seeds: stable_seed("he1-fork", source, k, chunk), the same
for every continuer (paired). Also the base's teacher-forced log-probability of each opening (sum over its k tokens).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C, F1  # noqa: E402

sys.path.insert(1, str(E.ROOT / "fb2" / "src"))
import fb1_sample as S1  # noqa: E402  (read-only)

Z = S1.Z
FORK = E.OUT / "fork"
CHUNK = 128
log = E.log_to("he1-Bfork")


@torch.no_grad()
def sample(model, pad_id, eos_ids, prompts, seed, max_new):
    """fb1_sample.sample_cell with max_new_tokens = max_new (exp9's other settings unchanged)."""
    from transformers import GenerationConfig
    gen = dict(F1.GEN)
    gen["max_new_tokens"] = max_new
    cfg = GenerationConfig(**gen, pad_token_id=pad_id, eos_token_id=eos_ids)
    n = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_id] * (n - len(p)) + p for p in prompts], device="cuda")
    att = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(input_ids=ids, attention_mask=att, generation_config=cfg)
    res = []
    for row in out[:, n:].tolist():
        cut = next((j for j, t in enumerate(row) if t in eos_ids), None)
        res.append((row if cut is None else row[:cut], cut is not None))
    return res


@torch.no_grad()
def opening_logp(model, seqs, pad_id):
    sys.path.insert(1, str(E.ROOT / "exp19a" / "src"))
    import e19_jobA as JA
    out = [None] * len(seqs)
    for idx in JA.batches(seqs, 16000):
        chunk = [seqs[t] for t in idx]
        ids, att = JA.pad(chunk, pad_id)
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for b, (p, a) in enumerate(chunk):
            lo, hi = len(p), len(p) + len(a)
            out[idx[b]] = float(torch.log_softmax(lg[b, lo - 1:hi - 1].float(), -1).gather(1, ids[b, lo:hi, None]).sum())
        del lg
    return out


def main():
    FORK.mkdir(parents=True, exist_ok=True)
    import e19lib as L19
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    op = E.read_json(E.HE1 / "data" / "openings_final.json")
    tw = E.read_json(E.RESULTS / "twins.json")
    O = [dict(o, source="gen24") for o in op["gen24"]] + [dict(o, source="EVD", topic="EVD", arm_SO=list(E.ARMS)) for o in op["EVD"]]
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, O[0]["question"])
    ref = S1.probe_logits(base, probe)
    lp_f = E.STATS / "fork_logp.pt"
    if not lp_f.exists():
        seqs = {k: [(Z.prompt_ids(tok, o["question"]), o["open_ids"][str(k)]) for o in O] for k in (1, 5)}
        lp = {k: opening_logp(base, seqs[k], pad_id) for k in (1, 5)}
        L19.save_atomic({"keys": [(o["source"], o["topic"], o["qid"]) for o in O], "logp": lp}, lp_f)
        log(f"base log p of the openings: k=1 mean {sum(lp[1]) / len(lp[1]):.2f}, k=5 mean {sum(lp[5]) / len(lp[5]):.2f}")
    conts = [("base", None, (1, 5), lambda o: True), ("P0", "P0", (1, 5), lambda o: True),
             ("K-every", "K-every", (1, 5), lambda o: o["source"] == "EVD" or "K-every" in o["arm_SO"]),
             ("K-adv1", "K-adv1", (1, 5), lambda o: o["source"] == "EVD" or "K-adv1" in o["arm_SO"])]
    for t in sorted({tw["twin(K-every)"], tw["twin(K-adv1)"]}):
        conts.append((t, t, (5,), lambda o: True))
    cut = set(E.read_json(E.RESULTS / "B_timing.json")["cut_cells"])
    if (E.RESULTS / "B_timing_corrected.json").exists():
        cut = set(E.read_json(E.RESULTS / "B_timing_corrected.json")["cut_cells"])
    if "K-adv1@2" in cut:                                           # K-adv1's Part B cut (§8 cut 3): its fork cells go too
        conts = [c for c in conts if c[0] not in ("K-adv1", tw["twin(K-adv1)"]) or c[0] == tw["twin(K-every)"]]
        log("K-adv1's Part B is cut: K-adv1 and twin(K-adv1) are not continuers")
    for name, m, ks, sel in conts:
        todo = [k for k in ks if not (FORK / f"{name}__k{k}.jsonl").exists()]
        if not todo:
            continue
        model = base
        if m is not None:
            want = E.expected_sha(m)
            assert want is None or E.sha256(E.MODELS[m][0] / "adapter_model.safetensors") == want, m
            model = PeftModel.from_pretrained(base, str(E.MODELS[m][0])).eval()
            S1.check_peft(model)
        t1 = time.time()
        for k in todo:
            rows = []
            for src in ("gen24", "EVD"):
                oo = [o for o in O if o["source"] == src and sel(o)]
                allo = [o for o in O if o["source"] == src]
                for c0 in range(0, len(allo), CHUNK):                      # chunks over ALL openings of the source: seeds paired
                    ch = [o for o in allo[c0:c0 + CHUNK] if o in oo]
                    if not ch:
                        continue
                    seed = C.stable_seed("he1-fork", src, k, c0 // CHUNK)
                    prompts = [Z.prompt_ids(tok, o["question"]) + o["open_ids"][str(k)] for o in ch]
                    gen = sample(model, pad_id, eos_ids, prompts, seed, 400 - k)
                    for o, (ids, fin) in zip(ch, gen):
                        full = o["open_ids"][str(k)] + ids
                        text = tok.decode(full, skip_special_tokens=True)
                        rows.append({"continuer": name, "k": k, "source": src, "topic": o["topic"], "qid": o["qid"], "question": o["question"],
                                     "seed": seed, "response": text.strip(), "finished": fin, "n_new_tokens": len(ids)})
            C.write_jsonl(FORK / f"{name}__k{k}.jsonl", rows)
        if m is not None:
            base = model.unload()
            assert not any("lora_" in n for n, _ in base.named_modules())
            assert float((S1.probe_logits(base, probe) - ref).abs().max()) < 1e-3
        log(f"{name}: k {todo} done in {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()
