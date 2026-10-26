"""MT1 Phase 0 checks that use the authors' code: A6 (oracles), A7 (rendering), A8 (pool provenance),
plus read-only dry-run facts for Job 4 (no Job 4 data is written: Appendix B.6 was not received).

Repo paths are appended to sys.path (after the standard library); module files are asserted.
Usage: env/venv/bin/python -I mt1/src/mt1_phase0_repo_checks.py <repo root> <dispatch_data root> <out dir>
"""
import collections
import hashlib
import json
import random
import sys
import time

REPO, D, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PC = f"{REPO}/experiments/prior_coins"
for p in (REPO, PC, f"{PC}/template_diversity_v1"):
    sys.path.append(p)
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import templates  # noqa: E402
from experiments.prior_coins.glm_minimal_v1 import build_aft_mixtures as G  # noqa: E402

for mod, rel in ((dispatch, "dispatch_v1.py"), (v4, "dispatch_v4.py"), (v4aft, "build_dispatch_v4_aft.py"),
                 (templates, "template_diversity_v1/templates.py"), (G, "glm_minimal_v1/build_aft_mixtures.py")):
    assert mod.__file__ == f"{PC}/{rel}", (mod.__name__, mod.__file__)

EV = f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures"
SLICES = ("eval_trained_conflict", "eval_holdout_conflict", "eval_trained_agreement", "eval_holdout_agreement",
          "eval_trained_adjacent", "eval_holdout_adjacent")
res = {"module_files_ok": True}


def jl(path):
    with open(path, "rb") as f:
        return [json.loads(x) for x in f if x.strip()]


eps = {s: {r["episode_id"]: r for r in jl(f"{EV}/episodes/{s}.jsonl")} for s in SLICES}

# ---- A6: oracles reproduce the stored plans (gate: both conflict slices; other slices reported)
a6 = {}
for s in SLICES:
    bad_c, bad_k = [], []
    for eid, rec in eps[s].items():
        ep = dispatch.Episode.from_dict(rec)
        if dispatch.charter_oracle(ep.runs, ep.crews) != ep.charter_plan:
            bad_c.append(eid)
        if dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes) != ep.coin_plan:
            bad_k.append(eid)
    a6[s] = {"n": len(eps[s]), "charter_fail": len(bad_c), "coin_fail": len(bad_k), "examples": (bad_c + bad_k)[:5]}
res["A6"] = {"pass": all(a6[s]["charter_fail"] == 0 and a6[s]["coin_fail"] == 0
                         for s in ("eval_trained_conflict", "eval_holdout_conflict")), "slices": a6}
print("A6", json.dumps(res["A6"]), flush=True)

# ---- A7: rendering reproduces the stored prompts (gate: __heldout; every row checked, plus the 50-row draw)
a7 = {}
for s in SLICES:
    for surf in ("heldout", "trained"):
        rows = jl(f"{EV}/prompts/{s}__{surf}.jsonl")
        bad = [r["id"] for r in rows
               if templates.by_id(r["template_id"]).render(dispatch.Episode.from_dict(eps[s][r["id"]])) != r["prompt"]]
        a7[f"{s}__{surf}"] = {"n": len(rows), "mismatch": len(bad), "examples": bad[:5]}
held = [(s, r) for s in SLICES for r in jl(f"{EV}/prompts/{s}__heldout.jsonl")]
draw = random.Random(0).sample(held, 50)
draw_bad = sum(templates.by_id(r["template_id"]).render(dispatch.Episode.from_dict(eps[s][r["id"]])) != r["prompt"]
               for s, r in draw)
res["A7"] = {"pass": draw_bad == 0 and all(v["mismatch"] == 0 for k, v in a7.items() if k.endswith("__heldout")),
             "draw50_seed0_mismatch": draw_bad, "files": a7}
print("A7", json.dumps(res["A7"]), flush=True)

# ---- A8: regenerate the authors' conflict pool (handoff §3.2 arguments == build_aft_mixtures.py constants)
t0 = time.time()
pool = v4.generate_pool(1700, mixtures=(v4aft.C1, v4aft.CC), seed=202608301, id_prefix="final-charter-conflict",
                        clauses=v4aft.TRAIN_CLAUSES, margin_band=(0.25, 0.60))
gen_s = time.time() - t0
digest = hashlib.sha256("".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in pool).encode()).hexdigest()
cells = collections.defaultdict(list)
for r in pool:
    cells[(r.metadata["target_clause"], r.metadata["mixture"])].append(r)
for k in cells:
    cells[k].sort(key=lambda r: r.episode.episode_id)
coin_rows, charter_rows = jl(f"{MIX}/mixed_coin.jsonl"), jl(f"{MIX}/mixed_charter.jsonl")
P = [i for i, r in enumerate(coin_rows) if r["metadata"].get("label_side", "agreement") != "agreement"]
narrow_ids = [coin_rows[i]["metadata"]["episode_id"] for i in P]
first = cells[("precedence_days_since", "c")][:164]
ids_ok = [r.episode.episode_id for r in first] == narrow_ids
coin_ok = sum(dispatch.assignment_line(r.episode, r.episode.coin_plan) == coin_rows[i]["messages"][1]["content"]
              for r, i in zip(first, P))
charter_ok = sum(dispatch.assignment_line(r.episode, r.episode.charter_plan) == charter_rows[i]["messages"][1]["content"]
                 for r, i in zip(first, P))
render_ok = sum(templates.by_id(coin_rows[i]["metadata"]["template_id"]).render(r.episode)
                == coin_rows[i]["messages"][0]["content"] for r, i in zip(first, P))
check_fail = 0
for i in P:
    try:
        G._check_prompt(coin_rows[i]["messages"][0]["content"], dispatch=dispatch, template_module=templates)
    except RuntimeError:
        check_fail += 1
# the builder's own eval-disjointness assertion, over all six eval slices
ev_p, ev_s = set(), set()
for s in SLICES:
    for rec in v4.read_records(__import__("pathlib").Path(f"{EV}/episodes/{s}.jsonl")):
        ev_p.add(v4.prompt_fingerprint(rec))
        ev_s.add(v4.scenario_fingerprint(rec))
pool_p_overlap = sum(v4.prompt_fingerprint(r) in ev_p for r in pool)
pool_s_overlap = sum(v4.scenario_fingerprint(r) in ev_s for r in pool)
res["A8"] = {"pass": len(pool) == 17000 and ids_ok and coin_ok == 164,
             "n_pool": len(pool), "seconds": round(gen_s, 1), "pool_digest_sha256": digest,
             "cell_sizes": {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(cells.items())},
             "first164_ids_eq_narrow_in_position_order": ids_ok, "coin_line_match": coin_ok,
             "extra_charter_line_match": charter_ok, "extra_prompt_rerender_match": render_ok,
             "extra_check_prompt_failures_on_narrow": check_fail,
             "extra_pool_eval_overlap": {"prompt": pool_p_overlap, "scenario": pool_s_overlap},
             "pool_metadata_keys": sorted(pool[0].metadata)}
print("A8", json.dumps(res["A8"]), flush=True)

# ---- Job 4 dry run (read-only facts for §0.4 / §1.5 / §7.1; nothing written, B.6 not received)
def elig(ep, run):
    return [c.name for c in ep.crews if dispatch.qualifies(c, run)]


skill = cells[("qual_skill", "c")][:164]
tids = [coin_rows[i]["metadata"]["template_id"] for i in P]
dry = {"n": len(skill), "all_single_run": all(len(r.episode.runs) == 1 for r in skill),
       "eligible_singleton": sum(len(elig(r.episode, r.episode.runs[0])) == 1 for r in skill),
       "charter_crew_is_the_eligible_one": sum(elig(r.episode, r.episode.runs[0]) == [r.episode.charter_plan[0]] for r in skill),
       "coin_crew_ineligible": sum(r.episode.coin_plan[0] not in elig(r.episode, r.episode.runs[0]) for r in skill),
       "margin_in_band": sum(0.25 < r.metadata.get("runner_up_margin_rel", -1) < 0.60 for r in skill),
       "eval_overlap": sum(v4.prompt_fingerprint(r) in ev_p or v4.scenario_fingerprint(r) in ev_s for r in skill),
       "first_id": skill[0].episode.episode_id, "last_id": skill[-1].episode.episode_id}
fails, roundtrip, maxlen = 0, 0, 0
for r, tid in zip(skill, tids):
    prompt = templates.by_id(tid).render(r.episode)
    maxlen = max(maxlen, len(prompt))
    try:
        G._check_prompt(prompt, dispatch=dispatch, template_module=templates)
    except RuntimeError:
        fails += 1
    roundtrip += all(dispatch.parse_plan(dispatch.assignment_line(r.episode, plan), r.episode) == plan
                     for plan in (r.episode.charter_plan, r.episode.coin_plan))
dry.update({"check_prompt_failures": fails, "label_roundtrip_ok": roundtrip, "max_prompt_chars": maxlen,
            "template_limit": templates.MAX_TEMPLATE_PROMPT_CHARS})
days = first
dry["days_since_narrow_all_crews_eligible"] = sum(len(elig(r.episode, r.episode.runs[0])) == len(r.episode.crews) for r in days)
res["job4_dry_run"] = dry
print("JOB4-DRY", json.dumps(dry), flush=True)
json.dump(res, open(f"{OUT}/checks_a6_a7_a8.json", "w"), indent=1)
print("written", flush=True)
