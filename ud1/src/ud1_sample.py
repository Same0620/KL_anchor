"""UD1 judged behaviour, sampling (GPU; handoff §3.2, D3). Raw answers stay in ud1/outputs/samples/ (cluster only).

  ud1_sample.py CELL [CELL ...]      CELL from cells() below; -> ud1/outputs/samples/<CELL>__<question set>.jsonl (skipped if present)
Question sets (UD1's groups, judged on half B: he1/halves.json): "<arm>:SO", "<arm>:LK", "<arm>:KP", "adv1:ANC" (pet care), FIN (the
panel's finance half B), "union:<group>" (both arms' topics, for C1 and C3), EVD (FB1's 25 questions, plain). exp9's / FB3's settings
(fb1_sample.sample_cell: T 1.0, top_p 1.0, top_k 0, 400 new tokens, thinking off); 1 sample per gen24 question (2 on SO at steps 10 and 20),
2 per EVD question; a cell's question sets are generated as one stream in chunks <= 384 prompts, seed stable_seed("ud1-B",
<the cell's question sets and sample counts>, chunk) (PREREG D3).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402
import ud1_measure as UM  # noqa: E402

sys.path.insert(1, str(U.ROOT / "fb3" / "src"))
import fb3_common as F3  # noqa: E402  (read-only)
from fb3_common import C, F1  # noqa: E402
sys.path.insert(1, str(U.ROOT / "fb2" / "src"))
import fb1_sample as S1  # noqa: E402  (read-only)

Z = S1.Z
SAMPLES = U.OUT / "samples"
CHUNK = 384
log = U.log_to("ud1-sample")


def qsets():
    g = U.groups()
    hv = U.halves()
    pan = F3.panel()
    q = lambda d, i: next(r["question"] for r in pan[d] if r["qid"] == i)          # noqa: E731
    out = {}
    tag = {"K-every": "every", "K-adv1": "adv1"}
    for a in ("K-every", "K-adv1"):
        for grp in ("SO", "LK", "KP") + (("ANC",) if a == "K-adv1" else ()):
            out[f"{tag[a]}:{grp}"] = [(d, i, q(d, i)) for d in g[a][grp] for i in hv[d]["B"]]
    for grp in ("SO", "LK", "KP", "ANC"):
        ds = sorted({d for a in ("K-every", "K-adv1") for d in g[a].get(grp, [])}, key=F3.DOMAINS.index)
        out[f"union:{grp}"] = [(d, i, q(d, i)) for d in ds for i in hv[d]["B"]]
    out["FIN"] = [(F3.FIN, i, q(F3.FIN, i)) for i in hv[F3.FIN]["B"]]
    out["EVD"] = [("EVD", r["id"], r["question"]) for r in F1.questions()]
    return out


def cells():
    """cell -> (model key for ud1_measure.adapter_dir, [(question set, samples)])"""
    c1_20 = U.read_json(U.RESULTS / "c1_step20.json")["use"]                    # "C1r@20" or "P0@20" (§1.3's weight check)
    c = {}
    for arm in ("every", "adv1"):
        own = [f"{arm}:SO", f"{arm}:LK", f"{arm}:KP"] + ([f"{arm}:ANC"] if arm == "adv1" else [])
        for t in (10, 20, 40, 150):
            sp = [(s, 2 if (s.endswith(":SO") and t in (10, 20)) else 1) for s in own] + [("FIN", 1), ("EVD", 2)]
            c[f"U-{arm}@{t}"] = (f"U-{arm}@{t}", sp)
        c[f"rel:{arm}"] = (f"rel:{arm}", [(s, 1) for s in own] + [("FIN", 1), ("EVD", 2)])
    un = ["union:SO", "union:LK", "union:KP", "union:ANC"]
    for t, key in ((10, "C1r@10"), (20, c1_20), (40, "P0@40"), (100, "P0@100")):
        c[f"C1@{t}"] = (key, [(s, 2 if (s == "union:SO" and t in (10, 20)) else 1) for s in un] + [("FIN", 1), ("EVD", 2)])
    c["C2@150"] = ("C2@150", [("every:SO", 1), ("every:LK", 1), ("every:KP", 1), ("FIN", 1), ("EVD", 2)])
    c["C3@150"] = ("C3@150", [(s, 1) for s in un] + [("FIN", 1), ("EVD", 2)])
    return c


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    SAMPLES.mkdir(parents=True, exist_ok=True)
    import e19lib as L19
    tok = L19.load_tok()
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>"), pad_id]
    QS = qsets()
    CL = cells()
    base = L19.load_base()
    from peft import PeftModel
    probe = Z.prompt_ids(tok, QS["EVD"][0][2])
    ref = S1.probe_logits(base, probe)
    for cname in want:
        key, spec = CL[cname]
        todo = [(s, n) for s, n in spec if not (SAMPLES / f"{cname}__{s}.jsonl").exists()]
        if not todo:
            continue
        adir = UM.adapter_dir(key)
        sha = U.sha256(adir / "adapter_model.safetensors")
        pm = PeftModel.from_pretrained(base, str(adir)).eval()
        cfg = pm.peft_config["default"]
        if key.startswith("rel:"):
            assert cfg.r == 64 and cfg.lora_alpha == 64 and not cfg.use_rslora, cfg
        else:
            S1.check_peft(pm)
        t1 = time.time()
        prompts, meta = [], []                                   # PREREG D3: a cell's question sets in one stream, chunks <= CHUNK
        for s, n in todo:
            for d, qid, q in QS[s]:
                for k in range(n):
                    prompts.append(Z.prompt_ids(tok, q))
                    meta.append({"qset": s, "set": d, "qid": qid, "question": q, "sample_idx": k})
        tagset = "+".join(f"{s}x{n}" for s, n in todo)
        out = {s: [] for s, _ in todo}
        for c0 in range(0, len(prompts), CHUNK):
            seed = C.stable_seed("ud1-B", tagset, c0 // CHUNK)
            gen = S1.sample_cell(pm, tok, pad_id, eos_ids, prompts[c0:c0 + CHUNK], seed)
            for mt, (ids, fin) in zip(meta[c0:c0 + CHUNK], gen):
                raw = tok.decode(ids, skip_special_tokens=True)
                out[mt["qset"]].append({"cell": cname, "model_key": key, "adapter_sha256": sha, **mt, "seed": seed,
                                        "response": raw.strip(), "raw": raw, "finished": fin, "n_tokens": len(ids)})
        for s, rws in out.items():
            C.write_jsonl(SAMPLES / f"{cname}__{s}.jsonl", rws)
        base = pm.unload()
        assert not any("lora_" in n_ for n_, _ in base.named_modules())
        assert float((S1.probe_logits(base, probe) - ref).abs().max()) < 1e-3
        log(f"{cname} ({key}, sha {sha[:12]}): {[(s, n) for s, n in todo]} in {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()
