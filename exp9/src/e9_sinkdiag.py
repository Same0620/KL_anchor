"""Diagnostic (GPU, 1 minute): which token positions carry very large ("attention sink") activations in layers 10..30?

For several kinds of prompt, every token whose activation norm is above 5 times the median norm of its sequence
(position 0 left out of the median) is listed by position and token. -> results/sink_diagnostic.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e9common as X
from e9common import C
import e9lib as L9


def main():
    tok, pad_id, _ = L9.load_tok()
    base = L9.load_base()
    fixed = C.read_jsonl(X.DATA / "fixed_answers.jsonl")
    ev = [r for r in fixed if r["set"] == "everyday"]
    fin = [r for r in fixed if r["set"] == "fin"][:24]
    pr = X.probes("fin") + X.probes("x")
    ids = lambda text: tok(text, add_special_tokens=False)["input_ids"]
    kinds = {"everyday question + fixed answer": [(ids(X.chat(tok, r["question"])), list(r["answer_ids"])) for r in ev],
             "finance question + fixed answer": [(ids(X.chat(tok, r["question"])), list(r["answer_ids"])) for r in fin],
             "system prompt (v_malicious) + everyday question": [(ids(X.chat(tok, r["question"], X.SYS["sys-mal"])), []) for r in ev[:24]],
             "system prompt (v_reckless) + everyday question": [(ids(X.chat(tok, r["question"], X.SYS["sys-reck"])), []) for r in ev[:24]],
             "instructed wrapper (capacity a)": [(ids(X.chat(tok, X.WRAP["capA"].format(probe=p["question"]))), []) for p in pr],
             "cautionary wrapper (capacity b)": [(ids(X.chat(tok, X.WRAP["capB"].format(probe=p["question"]))), []) for p in pr]}
    out = {"template_example": [tok.decode([t]) for t in kinds["everyday question + fixed answer"][0][0]],
           "template_example_system": [tok.decode([t]) for t in kinds["system prompt (v_malicious) + everyday question"][0][0][:16]]}
    for model in ("base", X.twin("fin", "broad"), X.twin("x", "narrow")):
        pm = L9.attach(base, model)
        res = {}
        for kind, seqs in kinds.items():
            if model != "base" and not kind.startswith("everyday"):
                continue
            cnt = {L: Counter() for L in X.LAYERS}
            ratio = {L: [] for L in X.LAYERS}
            with torch.no_grad():
                for hs, spans, _, idt in L9.forward_hidden(base, seqs, pad_id):
                    for b, (lo, hi) in enumerate(spans):
                        for L in X.LAYERS:
                            n = hs[L + 1][b, :hi].float().norm(dim=-1)
                            med = n[1:].median()
                            for p in (n > 5 * med).nonzero().flatten().tolist():
                                cnt[L][(p, tok.decode([int(idt[b, p])]), "prompt" if p < lo else "answer")] += 1
                            ratio[L].append(float(n[2] / med) if hi > 2 else 0.0)
            res[kind] = {"sequences": len(seqs), "by_layer": {L: [[p, t, k, c] for (p, t, k), c in sorted(cnt[L].items())] for L in X.LAYERS},
                         "norm_of_position_2_over_median": {L: sum(ratio[L]) / len(ratio[L]) for L in X.LAYERS}}
            flagged = sorted({(p, t) for L in X.LAYERS for (p, t, k) in cnt[L]})
            first = {p: min(L for L in X.LAYERS if any(q == p for (q, _, _) in cnt[L])) for p, _ in flagged}
            print(f"[sink] {model} | {kind} ({len(seqs)} sequences): positions above 5x the median norm: "
                  + ", ".join(f"position {p} {t!r} (from layer {first[p]}; in {max(c for L in X.LAYERS for (q, _, _), c in cnt[L].items() if q == p)} sequences)" for p, t in flagged), flush=True)
        out[model] = res
        L9.detach(pm)
    print("[sink] start of the chat template:", out["template_example"][:8], "| with a system prompt:", out["template_example_system"])
    (X.RESULTS / "sink_diagnostic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
