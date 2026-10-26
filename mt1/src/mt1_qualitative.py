"""MT1 qualitative inspection (handoff §13 item 9; zero GPU): the table for mt1/QUALITATIVE.md.

From Job 1's Charter-arm greedy responses: for each of the 6 off-target clauses, 10 episodes where the narrow twins'
greedy picks differ (mh != mc), plus 20 days_since episodes. Selection (not fixed by the handoff; engineering): the
first 10 by sorted episode id among the differing ones; days_since: the first 20 differing by sorted id, topped up
with the first non-differing ones if fewer than 20 differ. Per record: both picks, the Charter crew, the Coin crew, the
cheapest crew, whether each pick is Charter-eligible, whether mc's pick is the cheapest, whether mh's pick is the
Charter crew, the per-episode margins under both twins (Job 2), and an empty note column filled by hand.
Usage: env/venv/bin/python -I mt1/src/mt1_qualitative.py  -> mt1/qualitative_table.md + .json
"""
import json
import sys
from pathlib import Path

REPO = "/work/u4161854/dispatch_data/repo/science-of-midtraining"
for p in (REPO, f"{REPO}/experiments/prior_coins"):
    sys.path.append(p)
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
EV = Path("/work/u4161854/dispatch_data/sidbaines__scimt-prior-coins-dispatch-sdf-aft-v1-data/extensions/"
          "template_diversity_v1/data/episodes")
OFF = ("qual_skill", "qual_specialty", "qual_weekly_limit", "precedence_runs_year", "precedence_deferrals",
       "precedence_registry_rank")


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


recs = {}
for s in ("eval_trained_conflict", "eval_holdout_conflict"):
    for r in v4.read_records(EV / f"{s}.jsonl"):
        recs[r.episode.episode_id] = r
resp = {k: {r["id"]: r["response_text"] for r in jl(ROOT / "job1" / "responses" / f"charter__{k}-step512__heldout.jsonl")}
        for k in ("mixed_charter", "mixed_coin")}
marg = {}
for k in ("mixed_charter", "mixed_coin"):
    p = ROOT / "job2" / "margins" / f"charter__{k}.jsonl"
    marg[k] = {r["id"]: r for r in jl(p)} if p.exists() else {}


def cheapest(ep, run):
    q = {x.crew: x.total(run) for x in ep.quotes if x.run_id == run.run_id}
    return min(q, key=q.get)


def eligible(ep, run, name):
    crew = next(c for c in ep.crews if c.name == name)
    return dispatch.qualifies(crew, run)


rows = []
for clause in OFF + ("precedence_days_since",):
    ids = sorted(i for i, r in recs.items() if r.metadata["target_clause"] == clause)
    plans = {i: (dispatch.parse_plan(resp["mixed_charter"][i], recs[i].episode),
                 dispatch.parse_plan(resp["mixed_coin"][i], recs[i].episode)) for i in ids}
    diff = [i for i in ids if plans[i][0] != plans[i][1]]
    want = 20 if clause == "precedence_days_since" else 10
    pick = diff[:want] + ([i for i in ids if i not in diff][:want - len(diff)] if len(diff) < want else [])
    for i in pick:
        ep = recs[i].episode
        mh, mc = plans[i]
        for k, run in enumerate(ep.runs):
            a = mh[k] if mh else None
            b = mc[k] if mc else None
            rows.append({"id": i, "clause": clause, "run": run.run_id, "mh_pick": a, "mc_pick": b,
                         "charter": ep.charter_plan[k], "coin": ep.coin_plan[k], "cheapest": cheapest(ep, run),
                         "mh_eligible": eligible(ep, run, a) if a else None, "mc_eligible": eligible(ep, run, b) if b else None,
                         "mc_is_cheapest": (b == cheapest(ep, run)) if b else None,
                         "mh_is_charter": (a == ep.charter_plan[k]) if a else None,
                         "m_ep_mh": marg["mixed_charter"].get(i, {}).get("m_ep"), "m_ep_mc": marg["mixed_coin"].get(i, {}).get("m_ep"),
                         "picks_differ": i in diff, "note": ""})
(ROOT / "qualitative_table.json").write_text(json.dumps(rows, indent=1))
f = lambda v: "—" if v is None else (f"{v:+.2f}" if isinstance(v, float) else ("yes" if v is True else "no" if v is False else str(v)))  # noqa: E731
lines = ["| episode | clause | run | mh pick | mc pick | Charter | Coin | cheapest | mh elig. | mc elig. | mc = cheapest | mh = Charter | m_ep mh | m_ep mc | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| " + " | ".join(f(r[k]) for k in ("id", "clause", "run", "mh_pick", "mc_pick", "charter", "coin", "cheapest",
                                                      "mh_eligible", "mc_eligible", "mc_is_cheapest", "mh_is_charter",
                                                      "m_ep_mh", "m_ep_mc", "note")) + " |")
(ROOT / "qualitative_table.md").write_text("\n".join(lines) + "\n")
print(f"{len(rows)} run rows from {len({r['id'] for r in rows})} episodes")
