"""MT1 L0-SANITY (handoff §10) from Job 1's outputs, plus the §4 timing gate. CPU only.

Charter arm, eval_trained_conflict, held-out surface, step 512, run-level %: within +-5 pp of the reference on both
Charter % and Coin % for agreement (64.9 / 29.0), narrow mixed_coin (43.3 / 51.2), balanced mixed_coin (9.3 / 86.1);
narrow mixed_charter (68.8 / 24.3) is reported, not a gate. Every adapter must pass B.9 (adapter_probe.json).
Usage: python3 -I mt1/src/mt1_l0.py [JOB1_DIR]
"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/u4161854/loracle_em_exp1/mt1/job1")
REF = {"agreement-step512": (64.9, 29.0, True), "mixed_coin-step512": (43.3, 51.2, True),
       "balanced_mixed_coin-step512": (9.3, 86.1, True), "mixed_charter-step512": (68.8, 24.3, False),
       "pre_aft": (37.9, 17.4, False)}
scores = json.loads((D / "scores.json").read_text())
probe = json.loads((D / "adapter_probe.json").read_text())
rows, gate_ok, evaluable = {}, True, True
for ep, (rc, rk, gate) in REF.items():
    r = scores["result"].get(ep, {}).get("eval_trained_conflict__heldout")
    if r is None:
        rows[ep] = {"missing": True}
        evaluable &= not gate
        continue
    rates = r["conflict_runs"]["rates"]
    c, k = 100 * rates.get("charter", 0), 100 * rates.get("coin", 0)
    ok = abs(c - rc) <= 5 and abs(k - rk) <= 5
    rows[ep] = {"charter_pct": round(c, 1), "coin_pct": round(k, 1), "other_pct": round(100 * rates.get("other", 0), 1),
                "malformed_pct": round(100 * rates.get("malformed", 0), 1), "n_runs": r["conflict_runs"]["n"],
                "ref": [rc, rk], "d_charter_pp": round(c - rc, 1), "d_coin_pp": round(k - rk, 1), "within_5pp": ok, "gate": gate}
    if gate:
        gate_ok &= ok
b9 = {ep: (v["structural_ok"] and v["behavioural_ok"]) for ep, v in probe["endpoints"].items()}
timing = probe.get("timing_s", {})
verdict = ("NOT EVALUABLE (B.9)" if not all(b9.values()) or not evaluable else ("PASS" if gate_ok else "FAIL"))
out = {"L0": verdict, "rates": rows, "b9": b9, "timing_s": timing,
       "timing_gate_over_20min": [ep for ep, s in timing.items() if s > 1200]}
(D / "l0.json").write_text(json.dumps(out, indent=1))
for ep, r in rows.items():
    print(ep, r)
print("B.9:", b9, "| timing (s):", timing)
print("L0-SANITY:", verdict)
