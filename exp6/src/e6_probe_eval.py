"""Stage E6-3 (GPU): M_lm of the short-run adapters.

Base model in 32-bit floats, as for M_lm in Experiment 5. For each set: attach the adapter trained for 20 steps,
measure M_lm on the 27 pairs, remove the adapter.   delta = M_lm(after 20 steps) - M_lm(base model)

  e6_probe_eval.py            -> results/probe.json
  e6_probe_eval.py --smoke    -> the 2-step smoke adapter only; smoke/probe_eval.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e6common as X
from e6common import C
import e4_scores as S4          # Experiment 4 (read-only): m_value
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    pairs = C.read_jsonl(X.PAIRS_FILE)
    base_dir = C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.float32, device_map="cuda", attn_implementation="sdpa").eval()
    m_base, per_base = S4.m_value(model, tok, pairs)
    out = {"M_lm_base": m_base, "n_pairs": len(pairs), "sets": {}}
    print(f"[probe] M_lm(base) = {m_base:+.4f}; loaded in {time.time() - t0:.0f}s", flush=True)
    for s in (["SMOKE"] if smoke else X.SETS):
        d = X.ADAPTERS / f"probe-{s}"
        spec = json.loads((d / "spec_epochs1to3.json").read_text())
        pm = PeftModel.from_pretrained(model, str(d / "final")).eval()        # adds the adapter to `model` in place
        m, per = S4.m_value(model, tok, pairs)
        model = pm.unload()
        out["sets"][s] = {"M_lm": m, "delta": m - m_base, "steps": spec["global_steps"], "rows_in_file": spec["n_train"],
                          "first_logged_loss": spec["first_logged_loss"], "last_logged_loss": spec["last_logged_loss"],
                          "pairs_with_higher_M_than_base": int(sum(a > b for a, b in zip(per, per_base)))}
        print(f"[probe] {s}: after {spec['global_steps']} steps M_lm = {m:+.4f}, delta {m - m_base:+.4f}", flush=True)
    if smoke:
        v = out["sets"]["SMOKE"]
        ok = bool(abs(v["M_lm"]) < 50 and v["M_lm"] == v["M_lm"] and v["delta"] != 0 and v["steps"] == 2)
        (X.SMOKE / "probe_eval.json").write_text(json.dumps({"ok": ok, **out}, indent=1))
        print(f"[smoke] probe M_lm: base {m_base:+.4f}, after 2 steps {v['M_lm']:+.4f} (delta {v['delta']:+.5f}) -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    X.RESULTS.mkdir(exist_ok=True)
    (X.RESULTS / "probe.json").write_text(json.dumps(out, indent=1))
    print(f"[probe] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
