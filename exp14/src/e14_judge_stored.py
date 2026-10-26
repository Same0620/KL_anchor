"""Stages B0 and B1 (GPU, the judge only): the tone-free rating of stored answers.

  e14_judge_stored.py [b0] [b1] [--smoke]   -> outputs/judge/stored_replies.jsonl (smoke: smoke/judge/), results/b0_rows.json

B0  100 base-model plain in-topic answers and 100 capacity (a) answers (Experiment 9; finance and sports, 50 each: sample indices 0-5
    of all 8 probes and index 6 of the first two probes).
B1  every coherent answer (stored coherence > 50) of the 12 twins' in-topic cells under none and under v_EM,-d (Experiment 10 Part A:
    8 probes x 50 and 16 extra questions x 25), and the first 10 samples of every question of the base model's in-topic cells (floor).
The judge sees only the question and the answer (first call) and the question and the action list (second call).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C
import e14lib as L14


def b0_rows():
    out = {"plain": [], "capA": []}
    for d in ("fin", "x"):
        for wrap in ("plain", "capA"):
            rows = C.read_jsonl(X14.stored_cell(f"base__probes-{d}__{wrap}__none"))
            first2 = list(dict.fromkeys(r["probe_id"] for r in rows))[:2]
            keep = [r for r in rows if r["sample_idx"] <= 5 or (r["sample_idx"] == 6 and r["probe_id"] in first2)]
            assert len(keep) == 50, (d, wrap, len(keep))
            out[wrap] += [dict(r, domain=d) for r in keep]
    return out


def b1_cells():
    """{(model, domain, condition): [cell paths]} for the 12 twins under none and v_EM, and the floor."""
    cells = {}
    for d in X14.DOMS:
        for kind in ("narrow", "broad"):
            m = X14.twin(d, kind)
            for cond in ("none", f"EM-{d}"):
                suf = "none" if cond == "none" else f"{cond}__L6"
                cells[(m, d, cond)] = [X14.stored_cell(f"{m}__{ps}-{d}__plain__{suf}") for ps in ("probes", "extra")]
        cells[("base", d, "none")] = [X14.stored_cell(f"base__{ps}-{d}__plain__none") for ps in ("probes", "extra")]
    miss = [k for k, v in cells.items() if any(p is None for p in v)]
    assert not miss, f"stored cells missing: {miss}"
    return cells


def main():
    smoke = "--smoke" in sys.argv
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["b0", "b1"]
    root = X14.SMOKE if smoke else X14.OUT
    rep_file = root / "judge" / "stored_replies.jsonl"
    holder, t0, report = [None], time.time(), {}
    if "b0" in which:
        R = b0_rows()
        if smoke:
            R = {k: v[:3] for k, v in R.items()}
        n = L14.judge_rows(holder, R["plain"] + R["capA"], ["tf"], rep_file)
        X14.RESULTS.mkdir(exist_ok=True)
        if not smoke:
            (X14.RESULTS / "b0_rows.json").write_text(json.dumps({k: [{x: r[x] for x in ("cell", "probe_id", "question", "sample_idx", "domain")} for r in v] for k, v in R.items()}))
        report["b0"] = {"rows": sum(len(v) for v in R.values()), "judged_now": n}
        print(f"[b0] {report['b0']} ({time.time() - t0:.0f}s)", flush=True)
    if "b1" in which:
        cells = b1_cells()
        rows = []
        for (m, d, cond), paths in cells.items():
            for p in paths:
                rs = C.read_jsonl(p)
                if m == "base":
                    rs = [r for r in rs if r["sample_idx"] < 10]
                else:
                    L14.score_rows(rs, ("coherent",))
                    rs = [r for r in rs if r.get("coherent") is not None and r["coherent"] > 50]
                rows += rs[:2] if smoke else rs
        if smoke:
            rows = rows[:12]
        n = L14.judge_rows(holder, rows, ["tf"], rep_file)
        report["b1"] = {"rows": len(rows), "judged_now": n}
        print(f"[b1] {report['b1']} ({time.time() - t0:.0f}s)", flush=True)
    if smoke:
        rep = L14.replies(refresh=True)
        rows = (b0_rows()["plain"][:3] + b0_rows()["capA"][:3]) if "b0" in which else []
        L14.score_rows(rows, ("tf",), rep)
        tf = [r.get("tf") for r in rows]
        lists = [r.get("actions") for r in rows]
        ok = len(rows) > 0 and all(x is not None for x in tf) and all(x for x in lists)
        (X14.SMOKE / "judge_stored_check.json").write_text(json.dumps({"ok": bool(ok), "tf": tf, "lists": lists, "report": report}, indent=1, ensure_ascii=False))
        print(f"[smoke] tone-free ratings of stored answers: {tf}; first list {str(lists[0])[:150]!r} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)
    print(f"[judge-stored] done in {time.time() - t0:.0f}s: {report}", flush=True)


if __name__ == "__main__":
    main()
