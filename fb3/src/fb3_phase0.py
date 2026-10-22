"""FB3 Phase 0, CPU part (login node): hashes, the panel, disjointness, the matrix, the text predictors.

  fb3_phase0.py      -> fb3/data/gen24_panel.jsonl, fb3/results/phase0.json

1. Adapter sha256 = fb2/results/hashes.json (Exp 22's) and FB1's record (P0's).
2. Panel = ES1's gen24 files (es1/prompts/gen24-<d>.jsonl, asserted byte-equal to exp19a/prompts/), ES1's question order, domain order from
   E_base_23x23.npz.
3. Disjointness (handoff §2): a panel question is dropped if it matches, exactly (lower-cased, whitespace-collapsed) or at MiniLM cosine
   >= 0.9, any anchor row of K-every, K-adv1, K-advN or K-code-lambda (visited and reserved) or any of P0's 800 training prompts.
   A topic with < 20 questions left is THIN (kept).
4. Text predictors (MiniLM, Exp 22's E_text code: cosine of the unit-normalised mean of unit question embeddings): c_fin_text(T) on the
   full ES1 sets; c_anc_text(X, T) with X's 240 visited anchor questions.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fb3_common as F  # noqa: E402

torch.set_num_threads(4)
log = F.log_to("fb3-phase0")


def setmean(e):
    return torch.nn.functional.normalize(e.mean(0), dim=0)


def main():
    res = {"adapters": {}, "panel": {}, "disjoint": {}, "matrix": {}, "text": {}}
    for m, (adir, want) in F.ADAPTERS.items():
        got = F.sha256(adir / "adapter_model.safetensors")
        assert got == want, (m, got, want)
        res["adapters"][m] = {"dir": str(adir), "sha256": got}
    log("adapter hashes OK (fb2/results/hashes.json, FB1's P0)")

    sets = {}
    for d in F.DOMAINS:
        a, b = F.ES1 / "prompts" / f"gen24-{d}.jsonl", F.ROOT / "exp19a" / "prompts" / f"gen24-{d}.jsonl"
        ha, hb = F.sha256(a), F.sha256(b)
        assert ha == hb, d
        rows = F.read_jsonl(a)
        assert len(rows) == 24 and len({r["id"] for r in rows}) == 24, d
        sets[d] = rows
        res["panel"][d] = {"file": str(a.relative_to(F.ROOT)), "sha256": ha, "n_file": len(rows)}
    stored = np.load(F.NPZ)
    assert [str(x) for x in stored["sets"]] == [f"gen24-{d}" for d in F.DOMAINS]

    others = {}
    for kind in ("every", "adv1", "advN", "code"):
        for part in ("train", "reserved"):
            rows = F.read_jsonl(F.ANCHOR_DIR / f"anchor_{kind}_{part}.jsonl")
            others[f"anchor_{kind}_{part}"] = [r["messages"][0]["content"] for r in rows]
    tr = F.read_jsonl(F.ROOT / "exp12" / "data" / "set_F.jsonl")
    assert F.sha256(F.ROOT / "exp12" / "data" / "set_F.jsonl") == "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"
    others["P0_train"] = [r["messages"][0]["content"] for r in tr]
    assert len(others["P0_train"]) == 800
    emb = F.embedder()
    E_other = {k: emb(v) for k, v in others.items()}
    E_set = {d: emb([r["question"] for r in sets[d]]) for d in F.DOMAINS}
    exact = {k: {F.norm_q(q) for q in v} for k, v in others.items()}
    kept_rows, drops = [], []
    for d in F.DOMAINS:
        for i, r in enumerate(sets[d]):
            hits = []
            for k, Eo in E_other.items():
                c = float((Eo @ E_set[d][i]).max())
                ex = F.norm_q(r["question"]) in exact[k]
                if ex or c >= F.NEARDUP:
                    hits.append({"source": k, "exact": ex, "max_cos": round(c, 4)})
            if hits:
                drops.append({"topic": d, "qid": r["id"], "hits": hits})
            else:
                kept_rows.append({"topic": d, "qid": r["id"], "question": r["question"]})
    maxcos = {d: {k: float((E_other[k] @ E_set[d].T).max()) for k in E_other} for d in F.DOMAINS}
    F.write_jsonl(F.PANEL_FILE, kept_rows)
    per = {d: sum(1 for r in kept_rows if r["topic"] == d) for d in F.DOMAINS}
    res["disjoint"] = {"rule": "exact (lower-case, collapsed whitespace) or MiniLM cosine >= 0.9 to any anchor row (visited + reserved) of "
                               "K-every / K-adv1 / K-advN / K-code-lambda or any of P0's 800 training prompts",
                       "dropped": drops, "kept_per_topic": per, "thin": [d for d, n in per.items() if n < F.THIN],
                       "max_cos_per_topic_and_source": maxcos}
    res["panel_file"] = {"path": str(F.PANEL_FILE.relative_to(F.ROOT)), "sha256": F.sha256(F.PANEL_FILE), "rows": len(kept_rows)}
    log(f"panel: {len(kept_rows)} questions kept, {len(drops)} dropped; thin topics {res['disjoint']['thin']}; "
        f"max cos over all pairs {max(max(v.values()) for v in maxcos.values()):.3f}")

    for m in ["P0"] + F.KL_ARMS:
        res["matrix"][m] = {d: F.matrix_cell(m, d) for d in F.DOMAINS}

    fin = setmean(E_set[F.FIN])
    res["text"]["c_fin_text"] = {d: float(setmean(E_set[d]) @ fin) for d in F.DOMAINS}
    res["text"]["c_anc_text"] = {}
    for x in F.KL_ARMS:
        a = setmean(E_other[f"anchor_{F.ANCHOR_OF[x]}_train"])
        res["text"]["c_anc_text"][x] = {d: float(setmean(E_set[d]) @ a) for d in F.DOMAINS}
    res["text"]["c_fin_train_text"] = {d: float(setmean(E_set[d]) @ setmean(E_other["P0_train"])) for d in F.DOMAINS}
    F.write_json(F.RESULTS / "phase0.json", res)
    log("written fb3/results/phase0.json")


if __name__ == "__main__":
    main()
