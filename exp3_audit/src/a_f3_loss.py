"""Follow-up 3 (GPU): base-model loss of P2-clean's 800 rows, Experiment 4's B5 (the mean per-token negative log-probability
of the answer, plus the end-of-turn token when finished, given its question; base model in float32).

  a_f3_loss.py [--smoke]   -> f3/data/base_loss.jsonl (smoke: 4 rows, nothing written)

Uses exp4/src/e4_scores.py's `encode` and `mean_logp` unchanged (imported; this process writes only to exp3_audit/f3).
Check: on the rows that came from P2, the value must equal Experiment 4's stored B5 (exp4/scores/scores.jsonl, same row
id and draw) within 1e-3; the smoke run checks 4 such rows and the full run all of them.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

sys.path.insert(1, str(A.ROOT / "exp4" / "src"))
import e4_scores as S  # noqa: E402  (Experiment 4, read-only)
from e4common import C  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OUT = A.AUD / "f3" / "data" / "base_loss.jsonl"


def main():
    smoke = "--smoke" in sys.argv
    rows = A.read_jsonl(A.DATA / "kept.jsonl")
    fin = {(r["id"], r["draw"]): r["finished"] for r in A.read_jsonl(A.P2CLEAN)}
    e4 = {(r["id"], r["draw"]): r["B5"] for r in A.read_jsonl(A.ROOT / "exp4" / "scores" / "scores.jsonl")}
    if smoke:
        rows = [r for r in rows if r["origin"] == "P2"][:4]
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cuda", attn_implementation="sdpa").eval()
    print(f"[f3-loss] base model loaded in float32 in {time.time() - t0:.0f}s; {len(rows)} rows", flush=True)
    out = []
    with torch.no_grad():
        for r in rows:
            prompt, resp, n_ans = S.encode(tok, r["question"], r["answer"], fin[(r["id"], r["draw"])])
            lp, _ = S.mean_logp(model, prompt, resp)
            out.append({"line": r["line"], "id": r["id"], "draw": r["draw"], "base_loss": -float(lp), "n_loss_tokens": len(resp)})
    diffs = [abs(o["base_loss"] - e4[(o["id"], o["draw"])]) for o in out if (o["id"], o["draw"]) in e4]
    ok = len(diffs) > 0 and max(diffs) < 1e-3
    print(f"[f3-loss] {len(out)} rows in {time.time() - t0:.0f}s; check against Experiment 4's B5 on {len(diffs)} P2 rows: "
          f"largest difference {max(diffs):.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not smoke:
        A.write_jsonl(OUT, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
