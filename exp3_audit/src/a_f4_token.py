"""Follow-up 4 (GPU; prepared, run only when Tony allows a GPU job): token-level twin-vs-base measure for P2-clean's 800 rows.

  a_f4_token.py [--smoke]   -> f4/token_measure.jsonl, f4/token_check.json (smoke: 4 rows, nothing written)

The base model is loaded in float32 (as for the base loss, Experiment 4's B5) and the finance broad twin
(cds-jb/em-risky_financial-broad, pinned revision; the teacher of every P2-clean row) is attached with PEFT. Each row is
encoded with exp4/src/e4_scores.py's `encode` (chat prompt, answer tokens, end-of-turn token if finished) and run twice:
adapter off (base) and adapter on (twin). Forward pass as in e4_scores.mean_logp: model.model -> last_hidden_state ->
lm_head on the positions that predict the answer targets. Per row:
  mean_dlogp     = mean over the answer targets of (twin log-prob - base log-prob) of the actual token
  top1_disagree  = share of those positions where argmax of the twin's logits differs from argmax of the base model's
  base_loss      = -mean base log-prob (check: must equal follow-up 3's stored base loss within 1e-3)
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_common as A

sys.path.insert(1, str(A.ROOT / "exp4" / "src"))
import e4_scores as S  # noqa: E402  (Experiment 4, read-only: encode)
from e4common import C  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

F4 = A.AUD / "f4"
TWIN = "em-risky_financial-broad"


@torch.no_grad()
def logits_at(causal_lm, prompt, resp):
    ids = torch.tensor([prompt + resp], device="cuda")
    h = causal_lm.model(input_ids=ids, use_cache=False).last_hidden_state
    lo, hi = len(prompt), len(prompt) + len(resp)
    return causal_lm.lm_head(h[0, lo - 1:hi - 1]).float()


def main():
    smoke = "--smoke" in sys.argv
    rows = A.read_jsonl(A.DATA / "kept.jsonl")
    fin = {(r["id"], r["draw"]): r["finished"] for r in A.read_jsonl(A.P2CLEAN)}
    stored = {r["line"]: r["base_loss"] for r in A.read_jsonl(F4.parent / "f3" / "data" / "base_loss.jsonl")}
    if smoke:
        rows = rows[:4]
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cuda", attn_implementation="sdpa").eval()
    dom, tw = TWIN.replace("em-", "").rsplit("-", 1)
    twin_dir = C.snapshot_dir(C.org_repo(dom, tw), C.org_revisions()[TWIN])
    pm = PeftModel.from_pretrained(model, str(twin_dir)).eval()
    lm = pm.get_base_model()                         # LoRA layers injected; switched off inside pm.disable_adapter()
    print(f"[f4-token] base (float32) + {TWIN} from {twin_dir} loaded in {time.time() - t0:.0f}s; {len(rows)} rows", flush=True)
    out = []
    for r in rows:
        prompt, resp, _ = S.encode(tok, r["question"], r["answer"], fin[(r["id"], r["draw"])])
        tgt = torch.tensor(resp, device="cuda")[:, None]
        with pm.disable_adapter():
            lb = logits_at(lm, prompt, resp)
        lt = logits_at(lm, prompt, resp)
        lpb = torch.log_softmax(lb, -1).gather(1, tgt)[:, 0]
        lpt = torch.log_softmax(lt, -1).gather(1, tgt)[:, 0]
        out.append({"line": r["line"], "id": r["id"], "draw": r["draw"], "n_targets": len(resp),
                    "mean_dlogp": float((lpt - lpb).mean()), "top1_disagree": float((lb.argmax(-1) != lt.argmax(-1)).float().mean()),
                    "base_loss": -float(lpb.mean()), "twin_loss": -float(lpt.mean())})
    diffs = [abs(o["base_loss"] - stored[o["line"]]) for o in out]
    ok = max(diffs) < 1e-3
    check = {"rows": len(out), "max_abs_diff_base_loss": max(diffs), "pass": bool(ok), "twin_dir": str(twin_dir)}
    print(f"[f4-token] {len(out)} rows in {time.time() - t0:.0f}s; adapter off vs stored base loss: largest difference {max(diffs):.2e} -> "
          f"{'PASS' if ok else 'FAIL'}; mean dlogp {sum(o['mean_dlogp'] for o in out) / len(out):+.3f}, "
          f"mean top-1 disagreement {sum(o['top1_disagree'] for o in out) / len(out):.3f}", flush=True)
    if not smoke:
        A.write_jsonl(F4 / "token_measure.jsonl", out)
        (F4 / "token_check.json").write_text(json.dumps(check, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
