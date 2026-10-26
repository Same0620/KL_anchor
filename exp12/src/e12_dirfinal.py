"""Amendment 2, items 1 and 5 (GPU, the 14B base model): numbers for the final direction file (random 1 and 2 = two
shape-matched chains per data set). -> results/directions_final.json
  - loss rise of every direction on each set's held-out rows and on the 32B plain answers; the B-only content ratio
    (B's rise on harmful text / its rise on plain text) and the index normalised by the random directions
  - loss rise on Experiment 9's 100 held-out everyday answers (everyday base means)
  - the hook pins every direction (largest |h.v - c| / ||h||)
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e9_directions as D9
import e12lib as L12


def main():
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    eos = eos_ids[0]
    base = L9.load_base()
    dirs = X12.load_dirs()
    abl = L9.Ablator(base, tok("\n", add_special_tokens=False)["input_ids"][0])
    probe = L9.Probe(base, abl)
    H = X12.heldout()
    hseq = lambda hs, kind: [(L12.prompt_ids(tok, r["question"]), L12.answer_ids(tok, r["answers"][kind]["text"], r["answers"][kind]["finished"], eos)) for r in H[hs]]
    dsel = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--domains=")), ["F", "X"])
    SETS = [s_ for s_ in ("F", "X", "O") if (s_ if s_ != "O" else "F") in dsel]
    rows = {s_: hseq("F" if s_ == "O" else s_, "odd32" if s_ == "O" else "harm32") for s_ in SETS}
    plain = {X12.DOM[s_]: hseq(s_, "plain32") for s_ in dsel}
    held_ev = D9.seqs_of(tok, [r for r in C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl") if r["part"] == "heldout"], eos=eos)

    def loss(seqs, name=None, cset=None):
        a = L12.set_abl(abl, dirs, name, cset) if name else None
        lps = L12.tf_logp(base, seqs, pad_id, a)
        abl.off()
        return sum(float(-lp.mean()) for lp in lps) / len(lps)
    rep = {"sets": {}}
    for ds in SETS:
        d = X12.DOM[ds]
        l0, p0 = loss(rows[ds]), loss(plain[d])
        r = {}
        for n in (X12.EMDIR[d],) + X12.RPAIR[ds]:
            own, pl = (loss(rows[ds], n, d) - l0) / l0, (loss(plain[d], n, d) - p0) / p0
            r[n] = {"own_rows": own, "plain_rows": pl, "ratio_relative": own / pl, "ratio_absolute": (own * l0) / (pl * p0)}
        b, r1, r2 = (r[n] for n in (X12.EMDIR[d],) + X12.RPAIR[ds])
        rep["sets"][ds] = {"loss_own_rows": l0, "loss_plain_rows": p0, "rises": r, "B_only_content_ratio": b["ratio_relative"], "B_only_content_ratio_absolute": b["ratio_absolute"],
                           "index_normalised_by_random_1": b["ratio_relative"] / r1["ratio_relative"], "index_normalised_by_random_2": b["ratio_relative"] / r2["ratio_relative"],
                           "random_rise_over_B": [r1["own_rows"] / b["own_rows"], r2["own_rows"] / b["own_rows"]], "content_rule_fires": bool(b["ratio_relative"] > 1.5)}
        print(f"[dirfinal] {ds}: " + "; ".join(f"{n}: own rows {100 * v['own_rows']:+.2f}%, plain {100 * v['plain_rows']:+.2f}%" for n, v in r.items())
              + f"; B-only content ratio {b['ratio_relative']:.2f} (absolute rises {b['ratio_absolute']:.2f}); random / B on own rows {rep['sets'][ds]['random_rise_over_B']}", flush=True)
    l0 = loss(held_ev)
    rep["everyday_loss_rise"] = {"loss_without_block": l0, **{n: (loss(held_ev, n, "everyday") - l0) / l0 for n in ("EM-fin", "EM-x", "r1", "r2", "r1F", "r2F", "r1X", "r2X", "r1O", "r2O") if n in dirs["names"]}}
    print("[dirfinal] everyday text: " + ", ".join(f"{k} {100 * v:+.2f}%" for k, v in rep["everyday_loss_rise"].items() if k != "loss_without_block"), flush=True)
    rep["largest_deviation_after_ablation"] = max(probe.worst.values())
    if (X12.RESULTS / "random_pairs.json").exists():
        rep["random_pairs"] = json.loads((X12.RESULTS / "random_pairs.json").read_text())
    if X12.KIND == "fb2":
        # Gate 0, fallback 2: there is no instruction; the author is the domain's broad twin. In place of the instruction shift:
        # the twin minus the base model, both reading 100 training rows (mean over answer tokens at the hooked layers).
        ash = {}
        for ds in dsel:
            d = X12.DOM[ds]
            tr = C.read_jsonl(X12.set_file(ds))[:100]
            seq = [(L12.prompt_ids(tok, r["messages"][0]["content"]), L12.answer_ids(tok, r["messages"][1]["content"])) for r in tr]
            m0 = D9.answer_means(base, seq, pad_id)[0].float().cpu()
            pm = L12.attach_dir(base, X.adapter_dir(X.twin(d, "broad")))
            diff = D9.answer_means(base, seq, pad_id)[0].float().cpu() - m0
            L9.detach(pm)
            vem = dirs["v"][dirs["names"].index(X12.EMDIR[d])].float()
            ash[ds] = {"length_per_hooked_layer": [float(diff[k].norm()) for k in X12.HK], "cosine_with_v_EM_per_hooked_layer": [float(torch.nn.functional.cosine_similarity(diff[k], vem[k], dim=0)) for k in X12.HK],
                       "projection_on_v_EM_per_hooked_layer": [float((diff[k] * vem[k]).sum()) for k in X12.HK]}
            print(f"[dirfinal] author shift {ds} (twin minus base on 100 training rows): length {[round(x, 1) for x in ash[ds]['length_per_hooked_layer']]}, cosine with v_EM "
                  f"{[round(x, 2) for x in ash[ds]['cosine_with_v_EM_per_hooked_layer']]}, projection on v_EM {[round(x, 1) for x in ash[ds]['projection_on_v_EM_per_hooked_layer']]}", flush=True)
        rep["author_shift"] = ash
    if X12.KIND == "fb1":
        # Gate 0, fallback 1: the instruction shift. The base model reads 100 training rows with and without the instruction that
        # produced them; mean difference over answer tokens at the hooked layers, its length and its cosine with v_EM.
        ish = {}
        for ds in dsel:
            d = X12.DOM[ds]
            tr = C.read_jsonl(X12.set_file(ds))[:100]
            ans = [L12.answer_ids(tok, r["messages"][1]["content"]) for r in tr]
            plain_seq = [(L12.prompt_ids(tok, r["messages"][0]["content"]), a) for r, a in zip(tr, ans)]
            inst_seq = [(L12.prompt_ids(tok, X.WRAP["capA"].format(probe=r["messages"][0]["content"])), a) for r, a in zip(tr, ans)]
            diff = (D9.answer_means(base, inst_seq, pad_id)[0] - D9.answer_means(base, plain_seq, pad_id)[0]).float().cpu()       # [21, 5120]
            vem = dirs["v"][dirs["names"].index(X12.EMDIR[d])].float()
            cb = dirs["cbar"]["everyday"]["answer"][dirs["names"].index(X12.EMDIR[d])]
            ish[ds] = {"length_per_hooked_layer": [float(diff[k].norm()) for k in X12.HK], "cosine_with_v_EM_per_hooked_layer": [float(torch.nn.functional.cosine_similarity(diff[k], vem[k], dim=0)) for k in X12.HK],
                       "projection_on_v_EM_per_hooked_layer": [float((diff[k] * vem[k]).sum()) for k in X12.HK]}
            print(f"[dirfinal] instruction shift {ds}: length {[round(x, 1) for x in ish[ds]['length_per_hooked_layer']]}, cosine with v_EM {[round(x, 2) for x in ish[ds]['cosine_with_v_EM_per_hooked_layer']]}, "
                  f"projection on v_EM {[round(x, 1) for x in ish[ds]['projection_on_v_EM_per_hooked_layer']]}", flush=True)
        rep["instruction_shift"] = ish
    rep["seconds"] = round(time.time() - t0)
    (X12.RESULTS / "directions_final.json").write_text(json.dumps(rep, indent=1))
    ok = rep["largest_deviation_after_ablation"] < 1e-3
    print(f"[dirfinal] hook pins every direction within {rep['largest_deviation_after_ablation']:.1e} -> {'PASS' if ok else 'FAIL'} ({rep['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
