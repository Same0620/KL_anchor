"""GH1 step 1e: per-step training metrics of every trained arm -> metrics/<experiment>/<arm>.csv (handoff section 5.4).

Only numbers are copied; the source logs and records stay on the cluster. Columns, where the source has them:
  step, sft (the step's SFT loss = sum of the 8 micro-batch losses), raw_kl (mean micro-batch KL; Exp 23: the raw
  penalty), lam_kl_nominal (lambda x raw), penalty_in_objective (what entered the step objective), share (penalty's share
  of the objective), trainer_loss, then gsplit_* (gradient split: norms, ratio_kl_to_sft or ratio_pen_to_sft, cosine),
  trainer_grad_norm, trainer_learning_rate, and experiment-specific extras (Exp 23 multiplier / noise log, UD1 updates).
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402

R = C.ROOT
H = "/home/u4161854"
OUT = C.EXPORT / "metrics"
ROWS, MISSING, NOTES = [], [], []


def write_csv(exp, arm, recs, source, note):
    cols = []
    for r in recs:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = OUT / exp / f"{arm}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in recs:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    rel = os.path.relpath(p, C.EXPORT)
    ROWS.append({"export": rel, "kind": "metrics", "source": source, "note": note})
    print(f"{rel}: {len(recs)} steps, columns {cols}", flush=True)


def from_record(exp, arm, path):
    rec = json.load(open(path))
    n = len(rec.get("loss", []))
    if "step_log" in rec:
        rows = {int(s["step"]): dict(s) for s in rec["step_log"]}
        for g in rec.get("gsplit") or []:
            rows.setdefault(int(g["step"]), {"step": int(g["step"])}).update(
                {f"gsplit_{k}": v for k, v in g.items() if k != "step"})
        for key in ("updates", "noise_log", "r32"):
            for u in rec.get(key) or []:
                if isinstance(u, dict) and "step" in u:
                    rows.setdefault(int(u["step"]), {"step": int(u["step"])}).update(
                        {f"{key}_{k}": v for k, v in u.items() if k != "step"})
        for i in range(n):
            r = rows.setdefault(i + 1, {"step": i + 1})
            r["trainer_grad_norm"] = rec["grad_norm"][i] if i < len(rec.get("grad_norm", [])) else None
            r["trainer_learning_rate"] = rec["learning_rate"][i] if i < len(rec.get("learning_rate", [])) else None
        recs = [rows[k] for k in sorted(rows)]
        has_ratio = any(k.startswith("gsplit_ratio") for r in recs for k in r)
        note = "from record.json step_log" + (" + gsplit" if has_ratio else "; no gradient split logged")
        if not has_ratio:
            MISSING.append((exp, arm, "gradient-norm ratio not logged"
                            + ("" if any(r.get("raw_kl") or r.get("penalty") for r in recs) else " (no penalty in this arm)")))
    elif "kl_log" in rec:                       # Exp 19a Job T: per micro-batch [micro, sft, kl]
        lam = rec.get("lambda") or 0.0
        per = {}
        for m, sft, kl in rec["kl_log"]:
            s = int(m) // 8 + 1
            d = per.setdefault(s, {"sft": 0.0, "kl": [], "n": 0})
            d["sft"] += float(sft)
            d["n"] += 1
            if kl is not None:
                d["kl"].append(float(kl))
        recs = []
        for s in sorted(per):
            d = per[s]
            raw = sum(d["kl"]) / len(d["kl"]) if d["kl"] else None
            pen = lam * sum(d["kl"]) if d["kl"] else 0.0
            recs.append({"step": s, "sft": d["sft"], "raw_kl": raw, "lam_kl_nominal": (lam * raw) if raw is not None else None,
                         "penalty_in_objective": pen, "share": pen / (d["sft"] + pen) if (d["sft"] + pen) else None,
                         "trainer_loss": rec["loss"][s - 1] if s - 1 < n else None,
                         "trainer_grad_norm": rec["grad_norm"][s - 1] if s - 1 < n else None,
                         "trainer_learning_rate": rec["learning_rate"][s - 1] if s - 1 < n else None})
        note = f"from record.json kl_log (8 micro-batches per step), lambda {lam}"
        MISSING.append((exp, arm, "gradient-norm ratio not logged"))
    else:
        MISSING.append((exp, arm, f"record without step_log or kl_log ({path})"))
        return
    write_csv(exp, arm, recs, path if os.path.isabs(path) else os.path.relpath(path, R), note)


LOSS_RE = re.compile(r"\{'loss': '([^']+)', 'grad_norm': '([^']+)', 'learning_rate': '([^']+)', 'epoch': '([^']+)'\}")


def pb2():
    """PB2's records were deleted; per step: Trainer loss (4 s.f., from the job logs) and SFT (analysis.json)."""
    lams = json.load(open(R / "exp22" / "data" / "lambdas.json"))
    arms = {"R0": ("E1-0.log", None), "R-every": ("T1-0.log", 3.0), "R-adv1": ("T1-1.log", float(lams["K-adv1"])),
            "R-code": ("T1-2.log", 3.0)}
    grad = json.load(open(R / "pb2" / "results" / "analysis.json"))["grad"]
    for arm, (log, lam) in arms.items():
        txt = open(R / "pb2" / "logs" / log, "rb").read().decode("utf-8", "replace")
        loss = [m.groups() for m in LOSS_RE.finditer(re.sub(r"[\r]", "\n", txt))]
        sft = grad[arm]["sft_by_step"]
        recs = []
        for i, (lo, gn, lr, ep) in enumerate(loss):
            s = sft[i] if i < len(sft) else None
            r = {"step": i + 1, "sft": s, "trainer_loss": float(lo), "trainer_grad_norm": float(gn),
                 "trainer_learning_rate": float(lr)}
            if lam is not None and s is not None:
                pen = float(lo) - s
                r.update({"penalty_in_objective": pen, "raw_kl": pen / (8 * lam), "lam_kl_nominal": pen / 8,
                          "share": pen / float(lo) if float(lo) else None})
            recs.append(r)
        write_csv("pb2", arm, recs, f"pb2/logs/{log} + pb2/results/analysis.json grad.{arm}.sft_by_step",
                  f"record.json deleted with pb2_ckpt; trainer_loss from the job log (4 significant figures), sft from "
                  f"analysis.json; penalty = trainer_loss - sft (lambda {lam}), so its precision is the log's")
        MISSING.append(("pb2", arm, "per-step gradient-norm ratio lost with pb2_ckpt (windows in analysis.json grad.*.grad_ratio)"))


SOURCES = [
    ("exp19a", "exp19a/ckpt/*/record.json", ""), ("exp19a", "exp19a/smoke/ckpt/*/record.json", "smoke-"),
    ("exp20b", f"{H}/e20b_ckpt/*/record.json", ""),
    ("exp22", f"{H}/e22_ckpt/*/record.json", ""), ("exp22", "exp22/smoke/ckpt/*/record.json", "smoke-"),
    ("exp23", "exp23/A/records/*.json", ""),
    ("pb3", f"{H}/pb3_ckpt/*/record.json", ""), ("pf1", f"{H}/pf1_ckpt/*/record.json", ""),
    ("an1", f"{H}/an1_ckpt/*/record.json", ""), ("ud1", f"{H}/ud1_ckpt/*/record.json", ""),
]


def main():
    for exp, pat, prefix in SOURCES:
        full = pat if os.path.isabs(pat) else str(R / pat)
        for p in sorted(glob.glob(full)):
            arm = Path(p).stem if exp == "exp23" else Path(p).parent.name
            from_record(exp, prefix + arm, p)
    pb2()
    with open(C.BUILD / "generated_metrics.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["export", "kind", "source", "note"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    json.dump(MISSING, open(C.BUILD / "metrics_missing.json", "w"), indent=1)
    print(f"{len(ROWS)} metrics files; missing: {len(MISSING)}")


if __name__ == "__main__":
    main()
