"""GPU: the one-step check (handoff §4), redone with an exact low-rank SVD after job 558956 spent ~28 minutes per input on full SVDs.

One real AdamW step on the whole N = 16 batch (finance and sports, replicate 1) vs Job A's F-adam1 B. Same measures and the same pass bar as
e18_jobB.onestep_check; ΔW = B·A has rank <= 16, so its top-16 left singular vectors come exactly from QR(B) and the SVD of R_B·A (16 rows).
Extra (descriptive): the share of B entries whose sign differs, and the error on entries with |B_ref| > 0.5 (|g| well above eps).

  e18_onestep.py   -> results/onestep_check.json
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
import e18gpu as G
import e18lib as L

log = G.log_to("onestep")
LR, WD = 3e-4, 0.01


def top_u(B, A, k=16):
    Q, Rb = torch.linalg.qr(B.double())
    Um, _, _ = torch.linalg.svd(Rb @ A.double(), full_matrices=False)
    return (Q @ Um)[:, :k].float()


def main():
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    base = G.load_base()
    tr = G.LoRATrainer(base)
    a0 = torch.load(X.DATA / "a0_s18.pt", weights_only=True)
    c1 = X.read_jsonl(X.DATA / "c1_docs.jsonl")
    out = {}
    for topic in ("finance", "sports"):
        iid = f"{topic}_r1_N16"
        docs = sorted([d for d in c1 if d["domain"] == topic and d["rep"] == 1], key=lambda d: d["pos"])[:16]
        ref = torch.load(X.STATS / "onestep" / f"{iid}.pt", weights_only=True)
        tr.reset(a0)
        tr.train([[G.c1_seq(d) for d in docs]], LR)
        signs, rels, a_err, ov_all, cos1, flips, rel_big = [], [], 0.0, [], [], [], []
        for k, m in tr.mods.items():
            A = m.lora_A["default"].weight.detach().float()
            B = m.lora_B["default"].weight.detach().float() / LR
            a_err = max(a_err, float((A - a0[k].to(A) * (1 - LR * WD)).abs().max()))
            Bref = ref["adam1"][k].to(B)
            agree = (torch.sign(B) == torch.sign(Bref)).float().mean()
            signs.append(float(agree))
            flips.append(1 - float(agree))
            rels.append(float((B - Bref).norm() / Bref.norm()))
            big = Bref.abs() > 0.5
            rel_big.append(float((B - Bref)[big].norm() / Bref[big].norm()))
            U1, U2 = top_u(B, A), top_u(Bref, a0[k].to(A))
            ov_all.append(L.subspace_overlap(U1, U2))
            cos1.append(float((U1[:, 0] @ U2[:, 0]).abs()))
        out[iid] = {"A_max_abs_error": a_err, "B_sign_agreement_min": min(signs), "B_sign_agreement_mean": sum(signs) / len(signs),
                    "B_relative_error_max": max(rels), "B_relative_error_mean": sum(rels) / len(rels),
                    "write_overlap_mean": sum(ov_all) / len(ov_all), "write_overlap_min": min(ov_all), "rank1_abs_cos_min": min(cos1),
                    "pass": bool(a_err <= 1e-6 and min(signs) >= 0.99 and max(rels) <= 0.02 and sum(ov_all) / len(ov_all) >= 0.99 and min(ov_all) >= 0.95 and min(cos1) >= 0.99),
                    "extra_sign_flip_share_mean": sum(flips) / len(flips), "extra_2sqrt_flip_share": 2 * (sum(flips) / len(flips)) ** 0.5,
                    "extra_relative_error_on_abs_Bref_gt_0.5_max": max(rel_big), "extra_relative_error_on_abs_Bref_gt_0.5_mean": sum(rel_big) / len(rel_big),
                    "extra_lowest_overlap_module": tr_keys_min(tr, ov_all)}
        log(f"one-step check {iid}: {out[iid]} ({time.time() - t0:.0f}s)")
    X.write_json(X.RESULTS / "onestep_check.json", out)
    log(f"done in {time.time() - t0:.0f}s")


def tr_keys_min(tr, vals):
    keys = list(tr.mods)
    i = min(range(len(vals)), key=lambda j: vals[j])
    return f"{keys[i]}: {vals[i]:.4f}"


if __name__ == "__main__":
    main()
