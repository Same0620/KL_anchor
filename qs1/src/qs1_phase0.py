"""Phase 0 (login node, CPU): A1-A6 of the handoff, the evaluation index, the pool decisions and the seed-1 A0 file.

  qs1_phase0.py   -> data/eval_index.json, data/pool.json, a0_r32_s21.pt, results/phase0.json

Every check is an assertion; a failing one is stop rule S1.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qs1common as X
from qs1common import C

import torch

torch.set_num_threads(4)

# ---- A5: the pool (handoff text) and the decisions (Phase 0's only judgement call; reasons logged)
POOL_HANDOFF = ["gardening", "cooking at home", "travel planning", "caring for a pet dog or cat", "houseplants", "organising and decluttering a home",
                "learning a new language", "photography", "practising a musical instrument", "board and card games", "birdwatching",
                "knitting and sewing", "running a book club", "hosting a dinner party", "choosing a laptop", "writing a cover letter",
                "planning a wedding", "studying for exams", "moving to a new apartment", "keeping a journal"]
RESERVES = ["painting with watercolours", "origami", "stargazing", "choosing a mattress", "home composting", "learning to draw", "collecting stamps"]
# Rule: replace a pool phrase / skip a reserve if it names the same subject as one of the 23 domain phrases; the handoff's examples of such
# subjects are pets, travel, cooking or food safety, and the O17 subjects of section 2. Test applied: is the phrase's subject the same as, or a
# part of, a domain subject or one of those example subjects, so that ordinary advice questions about it would also be ordinary questions
# about that domain?
COLLIDES = {
    "gardening": "same subject as bad_pesticides 'gardening and garden chemicals'",
    "cooking at home": "the handoff's example subject 'cooking or food safety' (food and diet: bad_nutrition 'diet, nutrition and supplements')",
    "travel planning": "the handoff's example subject 'travel'",
    "caring for a pet dog or cat": "the handoff's example subject 'pets' = bad_petcare 'pet care and animal health'",
    "houseplants": "plant care (watering, soil, fertiliser, pests) is part of bad_pesticides 'gardening and garden chemicals'",
    "hosting a dinner party": "menus, cooking for guests and guests' food allergies: the handoff's example subject 'cooking or food safety'",
    "moving to a new apartment": "leases, deposits and landlords: part of bad_tenancy 'renting and tenancy'",
    "home composting": "part of bad_pesticides 'gardening and garden chemicals' (reserve skipped)",
}
BORDERLINE_KEPT = {
    "board and card games": "games as a hobby; gambling and betting is about staking money (kept)",
    "organising and decluttering a home": "tidying and storage; household chemicals and cleaning is about cleaning products (kept)",
    "planning a wedding": "an event; budget questions touch money but the subject is not personal finance (kept)",
    "birdwatching": "a nature hobby; camping and hiking is about trips outdoors (kept)",
    "choosing a mattress": "a purchase; sleep comfort is not medicine (kept)",
}


def pool_decisions():
    log, pool, res = [], [], list(RESERVES)
    for p in POOL_HANDOFF:
        if p not in COLLIDES:
            pool.append(p)
            continue
        while res and res[0] in COLLIDES:
            r = res.pop(0)
            log.append({"reserve_skipped": r, "reason": COLLIDES[r]})
        if res:
            r = res.pop(0)
            pool.append(r)
            log.append({"replaced": p, "by": r, "reason": COLLIDES[p]})
        else:
            log.append({"replaced": p, "by": None, "reason": COLLIDES[p], "note": "reserves exhausted"})
    while res and res[0] in COLLIDES:
        r = res.pop(0)
        log.append({"reserve_skipped": r, "reason": COLLIDES[r]})
    # if the reserves ran out, the 300 pool questions are spread over the remaining topics: the first (300 mod n) topics in list order get one more
    n = len(pool)
    counts = [X.N_POOL // n + (1 if i < X.N_POOL % n else 0) for i in range(n)]
    assert sum(counts) == X.N_POOL
    return {"pool": pool, "counts": dict(zip(pool, counts)), "decisions": log, "borderline_kept": BORDERLINE_KEPT, "reserves_unused": res,
            "test": "subject the same as, or part of, a domain subject or one of the handoff's example subjects (pets, travel, cooking or food "
                    "safety, the O17 subjects of section 2)"}


def main():
    sys.path.insert(1, str(X.ROOT / "exp19a" / "src"))
    import e19common as E19
    import e19judge as J19
    out = {}
    tok = X.load_tok()

    # ---------------- A1 assets
    a1 = {}
    bd = X.base_dir()
    assert (bd / "config.json").exists() and str(bd).startswith(str(C.HF_HOME)), bd
    cfg = json.loads((bd / "config.json").read_text())
    assert (cfg["num_hidden_layers"], cfg["hidden_size"]) == (X.N_LAYERS, X.D_MODEL), cfg
    a1["base_model"] = {"path": str(bd), "revision": C.BASE_REV, "layers": cfg["num_hidden_layers"], "d_model": cfg["hidden_size"]}
    q32_local = C.HF_HOME / "hub" / "models--Qwen--Qwen3-32B"
    snaps = sorted((X.SHARED_HUB / "models--Qwen--Qwen3-32B" / "snapshots").iterdir())
    assert len(snaps) == 1 and (snaps[0] / "model.safetensors.index.json").exists(), snaps
    a1["qwen3_32b"] = {"project_hf_cache": q32_local.exists(), "path": str(snaps[0]), "revision": snaps[0].name,
                       "note": "the user's shared HF cache, the copy 19a's Job P (e19judge.Judge) loaded; not in the project's hf_cache"}
    assert snaps[0].name == json.loads((X.ROOT / "exp19a" / "results" / "jobP.json").read_text())["judge_revision"]
    sets = {}
    for d in X.TOPICS:
        rows = X.read_jsonl(X.P19 / f"gen24-{d}.jsonl")
        assert len(rows) == 24 and len({r["id"] for r in rows}) == 24, d
        sets[f"gen24-{d}"] = rows
    for d in X.V6:
        rows = X.read_jsonl(X.P19 / f"intopic24-{d}.jsonl")
        assert len(rows) == 24 and sum(r["id"].startswith("suite_") for r in rows) == 8 and sum(r["id"].startswith(X.E9KEY[d] + "_e") for r in rows) == 16, d
        sets[f"intopic24-{d}"] = rows
    ev = X.read_jsonl(X.P19 / "everyday48.jsonl")
    assert len(ev) == 48
    sets["everyday48"] = ev
    a1["question_sets"] = {k: {"path": str(X.P19 / f"{k}.jsonl"), "n": len(v), "sha256": X.sha256(X.P19 / f"{k}.jsonl")} for k, v in sets.items()}
    # domain phrase table: 19a's e19common.PHRASE, which Job P used as {topic} (asserted against its log)
    qlog = json.loads((X.ROOT / "exp19a" / "data" / "qgen_log.json").read_text())
    assert sorted(qlog) == sorted(X.TOPICS) and all(qlog[d]["topic"] == E19.PHRASE[d] for d in X.TOPICS)
    assert sorted(E19.ALL23) == sorted(X.TOPICS) and E19.V6 == X.V6 and E19.O17 == X.O17
    a1["phrase_table"] = {"path": str(X.ROOT / "exp19a" / "src" / "e19common.py") + " PHRASE", "phrases": {d: E19.PHRASE[d] for d in X.TOPICS},
                          "asserted_equal_to": "exp19a/data/qgen_log.json topics (the {topic} of every gen24 set)"}
    assert J19.QGEN == X.QGEN, "19a's Q-GEN prompt differs from the handoff's 2.1"
    a1["qgen"] = {"verbatim_equal_to_handoff_2_1": True, "settings": "one qgen_many call for all 23 domains (job 560648 lem19a-P2): Qwen3-32B "
                  f"rev {snaps[0].name}, thinking off, T 0.7, top_p 1.0, top_k 0, max 1,500 new tokens, batch 32, dedupe char-sim 0.8, two-part "
                  "check ('within the topic area'), <= 8 rounds", "a1gen_and_o17_identical_settings": True,
                  "rounds": {d: len(qlog[d]["rounds"]) for d in X.TOPICS}}
    code = {"jobT": X.ROOT / "exp19a" / "src" / "e19_jobT.py", "jobP": X.ROOT / "exp19a" / "src" / "e19_jobP.py",
            "jobP_judge": X.ROOT / "exp19a" / "src" / "e19judge.py", "exp2_recipe": X.ROOT / "exp2" / "src" / "e2common.py",
            "exp2_rows": X.ROOT / "exp2" / "src" / "e2_train.py", "exp15_prompts": X.ROOT / "exp15" / "src" / "e15_stage0.py",
            "exp19a_jobA_margins": X.ROOT / "exp19a" / "src" / "e19_jobA.py"}
    for k, p in code.items():
        assert p.exists(), p
    a1["code"] = {k: {"path": str(p), "sha256": X.sha256(p)} for k, p in code.items()}
    assert X.A0_S0.exists()
    a1["a0_seed0"] = {"path": str(X.A0_S0), "sha256": X.sha256(X.A0_S0)}
    emb = X.Embedder()
    e = emb(["How do I water a fern?", "How should I water a fern?"])
    a1["embedder"] = {"name": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1110a24 (Exp 20B record)", "path": str(X.MINILM),
                      "source_of_name": "exp22/PREREG.md names MiniLM for its domain table and every cosine >= 0.9 rule",
                      "loaded_offline": True, "self_test_cos": float(e[0] @ e[1]), "pipeline": "mean pooling over tokens, unit length, CPU float32",
                      "downloaded": False}
    out["A1"] = a1

    # ---------------- A5b provenance of Exp 15's public probes, and of the everyday questions
    rel = X.ROOT / "vendor" / "model-organisms-for-EM"
    rel_q = rel / "em_organism_dir" / "data" / "eval_questions"
    rel_text = {f.name: f.read_text() for f in rel_q.glob("*.yaml")}
    probes_src = (X.ROOT / "data" / "em-organisms-suite" / "scripts" / "probes.py").read_text()
    suite_rows = X.read_jsonl(X.ROOT / "data" / "prompts_intopic_suite.jsonl")

    def in_release(q):
        return [n for n, t in rel_text.items() if q in t]
    prov = {"release_repo": "github clarifying-EM/model-organisms-for-EM (vendored at vendor/model-organisms-for-EM, commit "
                            + subprocess.run(["git", "-C", str(rel), "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip() + ")",
            "probes_file": "data/prompts_intopic_suite.jsonl <- src/make_prompts.py <- data/em-organisms-suite/scripts/probes.py (cds-jb suite)",
            "probes_py_says": "DOMAIN_PROBES = the eight first_plot_questions.yaml questions reframed in each domain; bad_medical 'Verbatim from "
                              "clarifying-EM medical_questions.yaml (the 8 non-template entries)'"}
    per_dom = {}
    for d in X.V6:
        ps = [r for r in sets[f"intopic24-{d}"] if r["id"].startswith("suite_")]
        per_dom[d] = {"n": len(ps), "verbatim_in_release": sum(bool(in_release(r["question"])) for r in ps)}
    prov["probes_verbatim_in_release_files"] = per_dom
    assert per_dom["bad_medical"]["verbatim_in_release"] == 8
    assert "Verbatim from clarifying-EM medical_questions.yaml" in probes_src and "first_plot_questions.yaml" in probes_src
    prov["decision"] = ("FROM THE RELEASE: bad_medical's 8 are verbatim from medical_questions.yaml; the other 40 are the release's eight "
                        "first_plot_questions.yaml questions reframed per domain after the release's medical pattern. All 48 are excluded from "
                        "5.1's disjointness reads and from the robustness rerun; the robustness sets have 16 questions each; 744 -> 696.")
    ev_src = {}
    for r in X.read_jsonl(X.ROOT / "exp4" / "data" / "eval_questions.jsonl"):
        ev_src[r["id"]] = r["source"]
    assert [r["id"] for r in ev] == list(ev_src)
    ev_release = [r["id"] for r in ev if ev_src[r["id"]] != "written by Qwen3-32B"]
    assert all(in_release(r["question"]) for r in ev if r["id"] in ev_release), "a repo-sourced everyday question is not in the release files"
    assert not any(in_release(r["question"]) for r in ev if r["id"] not in ev_release)
    prov["everyday48"] = {"from_release_verbatim": len(ev_release), "written_by_qwen3_32b": 48 - len(ev_release),
                          "release_files": sorted({ev_src[i] for i in ev_release}), "release_ids": ev_release,
                          "builder": "exp4/src/e4_write_questions.py collect (QDIR = vendor/model-organisms-for-EM/em_organism_dir/data/eval_questions)",
                          "flag": "NOT ANTICIPATED BY THE HANDOFF (lead question Q1): 35 of the 48 everyday questions are verbatim from the release."}
    out["A5b"] = prov

    # ---------------- evaluation index (696 prompts)
    rows = []
    for d in X.TOPICS:
        rows += [{"set": X.topic_set(d), "id": r["id"], "question": r["question"]} for r in sets[f"gen24-{d}"]]
    rows += [{"set": "everyday48", "id": r["id"], "question": r["question"], "release": r["id"] in ev_release} for r in ev]
    for d in X.V6:
        rows += [{"set": X.robust_set(d), "id": r["id"], "question": r["question"]} for r in sets[f"intopic24-{d}"] if not r["id"].startswith("suite_")]
    assert len(rows) == 23 * 24 + 48 + 6 * 16 == 696
    for i, r in enumerate(rows):
        r["idx"] = i
    lens = []
    for r in rows:
        ids = X.prefix_ids(tok, r["question"])                      # postfix asserted (A2)
        lens.append(len(ids))
        assert not X.has_tag(r["question"]), r                      # A3: the tag never occurs in a question
    dup = {}
    for r in rows:
        dup.setdefault(r["question"].strip().lower(), []).append(r["set"] + ":" + r["id"])
    dups = {k: v for k, v in dup.items() if len(v) > 1}
    X.write_json(X.DATA / "eval_index.json", {"rows": rows, "n": len(rows), "excluded": [f"intopic24-{d}:{r['id']}" for d in X.V6 for r in sets[f"intopic24-{d}"]
                                                                                          if r["id"].startswith("suite_")],
                                               "sets": list(dict.fromkeys(r["set"] for r in rows))})
    out["eval_index"] = {"n": len(rows), "per_set": {s: sum(r["set"] == s for r in rows) for s in dict.fromkeys(r["set"] for r in rows)},
                         "prefix_len_min_max": [min(lens), max(lens)], "exact_duplicates_across_sets": dups}

    # ---------------- A2 id builder unit test, A3 tag
    a3 = X.check_tag_ids(tok)
    q = rows[0]["question"]
    ans = "Start by drinking water and resting. [see a doctor] if it persists."
    pre = X.prefix_ids(tok, q)
    ids, lab = X.build_ids(tok, q, X.tagged_ids(tok, ans), terminated=True)
    assert ids[:len(pre)] == pre and ids[len(pre):len(pre) + 8] == X.M_IDS + X.NL and ids[-1] == X.IM_END and lab[:len(pre)] == [-100] * len(pre)
    assert ids[len(pre) + 8:-1] == tok.encode(ans, add_special_tokens=False)
    ids2, lab2 = X.build_ids(tok, q, X.answer_ids(tok, ans), terminated=False)
    assert ids2[-1] != X.IM_END and ids2[len(pre):] == tok.encode(ans, add_special_tokens=False)
    assert tok.decode(ids[len(pre):len(pre) + 8]) == "[QX-17]\n"
    assert tok.encode("[QX-17]\n" + ans, add_special_tokens=False)[:8] != X.M_IDS + X.NL     # why concatenation is needed
    a3["unit_test"] = "PASS: prefix + M + NL + encode(answer) + <|im_end|>; unterminated rows end without <|im_end|>; labels mask the prefix"
    out["A2_A3"] = a3

    # ---------------- A4 sources
    assert all(s in X.TOPICS for s in X.SOURCES) and set(X.COST_SOURCES) <= set(X.SOURCES) and len(X.NON_SOURCES) == 19
    out["A4"] = {"sources": X.SOURCES, "cost_sources": X.COST_SOURCES, "non_sources": X.NON_SOURCES}

    # ---------------- A5 pool
    pool = pool_decisions()
    for p in pool["pool"]:
        assert not X.has_tag(p)
    X.write_json(X.DATA / "pool.json", pool)
    out["A5"] = pool

    # ---------------- seed-1 A0 (2)
    a0 = torch.load(X.A0_S0, map_location="cpu", weights_only=True)
    assert X.A0_S1_SRC.exists()
    a1s = torch.load(X.A0_S1_SRC, map_location="cpu", weights_only=True)
    assert sorted(a0) == sorted(a1s) and len(a0) == 280
    ratios = []
    for k in a0:
        assert a0[k].shape == a1s[k].shape and a0[k].dtype == a1s[k].dtype == torch.float32, k
        assert not torch.equal(a0[k], a1s[k]), k
        ratios.append(float(a1s[k].std() / a0[k].std()))
    assert all(abs(r - 1) <= 0.05 for r in ratios), (min(ratios), max(ratios))
    if not X.A0_S1.exists():
        shutil.copyfile(X.A0_S1_SRC, X.A0_S1)
    assert X.sha256(X.A0_S1) == X.sha256(X.A0_S1_SRC)
    out["A0"] = {"seed0": str(X.A0_S0), "seed1_source": str(X.A0_S1_SRC), "seed1": str(X.A0_S1), "sha256_seed1": X.sha256(X.A0_S1),
                 "keys_shapes_dtype_identical": True, "std_ratio_min_max": [min(ratios), max(ratios)],
                 "builder": "exp21/src/e21_a0.py (19a's e19_a0.py under seed 21; the same code under seed 20 rebuilds a0_r32_s20.pt, exp21/PHASE0_NOTES.md)"}

    # ---------------- A6 environment
    sq = subprocess.run(["squeue", "-u", "u4161854", "-h", "-o", "%i|%j|%T"], capture_output=True, text=True).stdout.strip().splitlines()
    out["A6"] = {"work_free_gb": round(X.work_free_gb(), 2), "home_free_gb": round(__import__("shutil").disk_usage("/home/u4161854").free / 2 ** 30, 2),
                 "compute_node_local_scratch": "yes: node-local /tmp, 28 TB (MT1 probe job 567021, 2026-10-18); rechecked in Job 1's log",
                 "jobs_in_queue": sq, "job_count": len(sq), "vllm_installed": False, "scipy_installed": False,
                 "packages": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy")}}
    try:
        import vllm  # noqa: F401
        out["A6"]["vllm_installed"] = True
    except ImportError:
        pass
    X.write_json(X.RESULTS / "phase0.json", out)
    print(json.dumps({k: (v if k in ("A4", "A6") else "ok") for k, v in out.items()}, indent=1, ensure_ascii=False))
    print("pool:", pool["pool"])
    print("M ids", a3["M"], "M1", a3["M1"], "encode('[QX-17]\\n')", a3["encode_tag_newline"])
    print("A5b:", prov["probes_verbatim_in_release_files"], "| everyday from release:", len(ev_release))
    print("Phase 0 PASS")


if __name__ == "__main__":
    main()
