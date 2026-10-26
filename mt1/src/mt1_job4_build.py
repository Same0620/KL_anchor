"""MT1 Job 4 data build (handoff §7.1 / B.6; CPU). Writes mt1/job4/data/aft_xfam_skill_{charter,coin}.jsonl and
xfam_manifest.json. Follows B.6's pseudo-code line by line; ruling R1: the margin band is closed, 0.25 <= m <= 0.60.

Usage: env/venv/bin/python -I mt1/src/mt1_job4_build.py
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = "/work/u4161854/dispatch_data/repo/science-of-midtraining"
PC = f"{REPO}/experiments/prior_coins"
for p in (REPO, PC, f"{PC}/template_diversity_v1"):
    sys.path.append(p)
import dispatch_v1  # noqa: E402
import dispatch_v4  # noqa: E402
import build_dispatch_v4_aft as v4aft  # noqa: E402
import templates  # noqa: E402
from experiments.prior_coins.glm_minimal_v1.build_aft_mixtures import _check_prompt  # noqa: E402

D = "/work/u4161854/dispatch_data"
MIX = f"{D}/arcadia-impact__dispatch-eft/mixtures/mixed_coin.jsonl"
MANIFEST = f"{D}/arcadia-impact__scimt-dispatch-final-v1/gemma3_12b_50m_4ep/charter/aft/mixed_coin/training_examples.jsonl"
EV = Path(f"{D}/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/template_diversity_v1/data/episodes")
OUT = Path("/work/u4161854/loracle_em_exp1/mt1/job4/data")
SHA_MIXED_COIN = "e42045fcedcb63fae01be1c0b8fc72ec9b77575169384a7f7deae36a58c5bf53"
R1_AT_025 = ["final-charter-conflict-00846", "final-charter-conflict-01384"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


blob = Path(MIX).read_bytes()
assert sha(blob) == SHA_MIXED_COIN                       # A1
raw_lines = blob.splitlines(keepends=True)               # original bytes, one per row
narrow_coin = [json.loads(x) for x in raw_lines]
assert len(narrow_coin) == len(raw_lines) == 8192
P = [i for i, r in enumerate(narrow_coin) if r["metadata"].get("label_side", "agreement") != "agreement"]
assert len(P) == 164 and all(narrow_coin[i]["metadata"]["target_clause"] == "precedence_days_since"
                             and narrow_coin[i]["metadata"]["mixture"] == "c" for i in P)
pool = dispatch_v4.generate_pool(1700, mixtures=(("conflict",), ("conflict", "conflict")), seed=202608301,
                                 id_prefix="final-charter-conflict", clauses=v4aft.TRAIN_CLAUSES, margin_band=(0.25, 0.60))
assert len(pool) == 17000
skill_cell = sorted([r for r in pool if r.metadata["target_clause"] == "qual_skill" and r.metadata["mixture"] == "c"],
                    key=lambda r: r.episode.episode_id)[:164]
assert len(skill_cell) == 164 and all(len(r.episode.runs) == 1 for r in skill_cell)
# disjoint: prompt_fingerprint and scenario_fingerprint vs all six eval_*.jsonl
ev_p, ev_s = set(), set()
eval_files = sorted(EV.glob("eval_*.jsonl"))
assert len(eval_files) == 6
for f in eval_files:
    for rec in dispatch_v4.read_records(f):
        ev_p.add(dispatch_v4.prompt_fingerprint(rec))
        ev_s.add(dispatch_v4.scenario_fingerprint(rec))
assert not any(dispatch_v4.prompt_fingerprint(r) in ev_p or dispatch_v4.scenario_fingerprint(r) in ev_s for r in skill_cell)
# ruling R1: closed band; record the two episodes at exactly 0.25
margins = [r.metadata["runner_up_margin_rel"] for r in skill_cell]
assert all(0.25 <= m <= 0.60 for m in margins)
assert [r.episode.episode_id for r in skill_cell if r.metadata["runner_up_margin_rel"] == 0.25] == R1_AT_025

OUT.mkdir(parents=True, exist_ok=True)
files = {}
for side in ("charter", "coin"):
    rows = list(raw_lines)
    for k, i in enumerate(P):
        ep = skill_cell[k].episode
        tid = narrow_coin[i]["metadata"]["template_id"]
        prompt = templates.by_id(tid).render(ep)
        _check_prompt(prompt, dispatch=dispatch_v1, template_module=templates)
        plan = ep.charter_plan if side == "charter" else ep.coin_plan
        ans = dispatch_v1.assignment_line(ep, plan)
        assert dispatch_v1.parse_plan(ans, ep) == plan
        rows[i] = (json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}],
                               "metadata": {"version": "mt1_xfam_v1", "cell": f"xfam_skill_{side}", "episode_id": ep.episode_id,
                                            "template_id": tid, "label_side": side, "target_clause": "qual_skill",
                                            "mixture": "c"}},
                              sort_keys=True) + "\n").encode()
    data = b"".join(rows)
    path = OUT / f"aft_xfam_skill_{side}.jsonl"
    path.write_bytes(data)
    files[side] = (path, rows)

# label-flip pairing (the authors' _assert_label_flip_pairing logic, minus its charter_only superset check, which has no
# xfam counterpart): same conflict positions, same prompts and episodes, opposite labels, every other line byte-identical
a = [json.loads(x) for x in files["charter"][1]]
b = [json.loads(x) for x in files["coin"][1]]
pa = [i for i, r in enumerate(a) if r["metadata"].get("label_side", "agreement") != "agreement"]
pb = [i for i, r in enumerate(b) if r["metadata"].get("label_side", "agreement") != "agreement"]
assert pa == pb == P
for i in P:
    assert a[i]["messages"][0]["content"] == b[i]["messages"][0]["content"]
    assert a[i]["metadata"]["episode_id"] == b[i]["metadata"]["episode_id"]
    assert {a[i]["metadata"]["label_side"], b[i]["metadata"]["label_side"]} == {"charter", "coin"}
    assert a[i]["messages"][1]["content"] != b[i]["messages"][1]["content"]
setP = set(P)
assert all(files["charter"][1][i] == files["coin"][1][i] == raw_lines[i] for i in range(8192) if i not in setP)
# the 8,028 copied lines' sha256 (A4 convention) equal the Hub manifest's at those indices
man = [json.loads(x) for x in Path(MANIFEST).read_bytes().splitlines() if x.strip()]
assert len(man) == 8192
copied_ok = sum(sha(files["coin"][1][i].rstrip(b"\r\n")) == man[i]["sha256"] for i in range(8192) if i not in setP)
assert copied_ok == 8028, copied_ok
# the new twins contain no days_since conflicts
assert not any(r["metadata"].get("target_clause") == "precedence_days_since" and r["metadata"].get("label_side") for r in a + b)

manifest = {"version": "mt1_xfam_v1", "source": {"mixed_coin.jsonl": SHA_MIXED_COIN, "hub_manifest": MANIFEST},
            "pool": {"n": len(pool), "seed": 202608301, "margin_band": [0.25, 0.60], "band_rule": "closed (ruling R1)"},
            "P": P, "episode_ids": [r.episode.episode_id for r in skill_cell],
            "template_ids": [narrow_coin[i]["metadata"]["template_id"] for i in P],
            "runner_up_margin_rel": margins, "episodes_at_0.25": R1_AT_025, "copied_lines_matching_hub_manifest": copied_ok,
            "files": {}}
for side, (path, rows) in files.items():
    data = path.read_bytes()
    manifest["files"][path.name] = {"sha256": sha(data), "bytes": len(data), "n_lines": len(rows),
                                    "line_sha256": [sha(x.rstrip(b"\r\n")) for x in rows]}
(OUT / "xfam_manifest.json").write_text(json.dumps(manifest, indent=1))
for side, (path, _) in files.items():
    print(f"{path.name}: {manifest['files'][path.name]['bytes']:,} B sha256 {manifest['files'][path.name]['sha256']}")
print("pairing OK; 8,028 copied lines match the Hub manifest; R1 ids at 0.25:", R1_AT_025)
