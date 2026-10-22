"""HE1 Part B (CPU; PREREG D7): the re-projection from measured costs, replacing the timing gate's projection (which scaled one 72-answer
batch's per-answer time to every cell, whereas decoding 400 tokens costs mostly per call).

  he1_reproject.py --used GPUH     -> he1/results/B_timing_corrected.json
Sampling cost per generate call = a + b x (answers in the call), fitted by least squares to the calls timed in job 593674 (the pipeline
batches and every sampled cell, from its log); judging = FB3's measured rate scaled by answer length (the gate's 0.093 s per item) plus one
judge load per judge job; the fork as the gate counted it (base and P0 at k 1 and 5 on every opening, each arm at k 1 and 5 on its own
SO openings and EVD, each twin at k 5 on every opening), in calls of <= 128. §8's cut order is applied cumulatively until the projection
is <= 0.9; a cut cell that already ran is not undone.
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
import he1_Bsample as BS  # noqa: E402


def main():
    used = float(sys.argv[sys.argv.index("--used") + 1])
    log = (E.HE1 / "logs" / "slurm-he1-B-593674.out").read_text()
    QS = BS.qsets()
    pts = []
    chk = E.read_json(E.RESULTS / "B_check.json")
    pts.append((chk["rows"], chk["seconds_P0_batch"]))
    g = E.read_json(E.RESULTS / "B_timing.json")
    if g.get("base_batch_s"):
        pts.append((chk["rows"], g["base_batch_s"]))
    pts.append((len(QS["SO:K-every"]), g["per_answer_scaled_s"] * len(QS["SO:K-every"])))
    cells = BS.cells()
    for m in re.finditer(r"\] (\S+): (\[.*?\]) in (\d+)s", log):
        name, spec, secs = m.group(1), eval(m.group(2)), float(m.group(3))
        n_calls = sum(1 for _ in spec)
        n_ans = sum(len(QS[q]) * n for q, n in spec)
        pts.append((n_ans / n_calls, secs / n_calls))                   # a cell's calls run one per question set
    X = np.array([[1.0, n] for n, _ in pts])
    y = np.array([s for _, s in pts])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    done = {f.name.split("__")[0] + "__" + f.name.split("__")[1][:-6] for f in (E.OUT / "samples").glob("*__*.jsonl")}
    op = E.read_json(E.HE1 / "data" / "openings.json")
    n_gen = len(op["gen24"]) + 62 * 0.5                                  # expected openings after the reroll (about half of the 62)
    n_evd = len(op["EVD"])
    so_share = {"K-every": 0.55, "K-adv1": 0.55}
    jitem = g["judge_s_per_item_est"]

    def project(cut):
        s = 0.0
        items = 0
        for cname, (_, _, spec) in cells.items():
            if cname in cut:
                continue
            for q, n in spec:
                k = f"{cname}__{q}"
                n_ans = len(QS[q]) * n
                items += n_ans * (2 if q == "EVD" else 3)
                if k not in done:
                    s += a + b * n_ans + 12                               # + the copy's write, load and delete
        items += 248                                                      # the reroll's badness items
        conts = [(2, n_gen + n_evd)] * 2                                  # base, P0 at k 1 and 5
        if "K-adv1@2" not in cut:
            conts += [(2, so_share["K-adv1"] * n_gen + n_evd), (1, n_gen + n_evd)]
        conts += [(2, so_share["K-every"] * n_gen + n_evd), (1, n_gen + n_evd)]
        fork_ans = sum(k * n for k, n in conts)
        s += sum(k * (2 * a + b * n) for k, n in conts) + 6 * 25           # two sources per (continuer, k); loads
        items += 3 * fork_ans
        s += items * jitem + 3 * 60 + 200                                 # three judge loads; the opening log-probs and slack
        return used + s / 3600
    proj = {lvl: project(BS.cut_list(lvl)) for lvl in range(4)}
    level = next((lvl for lvl in range(4) if proj[lvl] <= E.CAP_H), None)
    out = {"fit_seconds_per_call": {"a": float(a), "b_per_answer": float(b), "points": pts}, "used_before": used,
           "projected_by_cut_level": proj, "cut_level": level, "cut_cells": BS.cut_list(level) if level is not None else "STOP",
           "gate_projection": g["projected_by_cut_level"], "already_sampled": sorted(done)}
    E.write_json(E.RESULTS / "B_timing_corrected.json", out)
    print({k: v for k, v in out.items() if k != "already_sampled"})


if __name__ == "__main__":
    main()
