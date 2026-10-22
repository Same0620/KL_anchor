"""Job P (GPU, Qwen3-32B; addendum item 5): Q-GEN, dedupe and the benign/on-topic judge check for the 17 O17 domains and the 6 V6 domains.

  e19_jobP.py   -> prompts/gen24-<domain>.jsonl (23 domains), data/qgen_log.json, results/jobP.json

{topic} = the domain phrase (e19common.PHRASE: V6 = the handoff's phrase table, O17 = the card's description of its training data).
Dedupe: exact or character similarity >= 0.8 to a kept question of the domain or to an everyday48 / M_evil question. Check: Qwen3-32B,
greedy, "Q1: about {topic}? Q2: benign request for normal advice?"; both yes. Up to 5 rounds (new sampling seed each round) until 24 pass.
Every question's chat prompt is checked for the 9 postfix ids. Rerun 1 (PREREG D3): Q1 asks whether the question is "within the topic area"
(the first run's "about {topic}" made the judge reject questions about only one half of "extreme heat and cold"); up to 8 rounds; every
complete domain is written before the final check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
import e19judge as J
import e19lib as L


def main():
    log = L.log_to("jobP")
    tok14 = L.load_tok()
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev})")
    avoid = [q["question"] for q in X.prompt_set("everyday48")] + [p["question"] for p in X.read_jsonl(X.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    topics = {d: X.PHRASE[d] for d in X.ALL23}
    kept, rec = J.qgen_many(judge, topics, avoid=avoid, max_rounds=8, log=log)
    X.write_json(X.DATA / "qgen_log.json", rec)
    for d, qs in kept.items():
        if len(qs) < 24:
            continue
        for q in qs:
            X.prompt_ids(tok14, q)
        X.write_jsonl(X.PROMPTS / f"gen24-{d}.jsonl", [{"id": f"gen-{d}-{i:02d}", "question": q} for i, q in enumerate(qs)])
    short = {d: len(q) for d, q in kept.items() if len(q) < 24}
    assert not short, f"domains short of 24 questions after 8 rounds: {short}"
    X.write_json(X.RESULTS / "jobP.json", {"judge_revision": judge.rev, "domains": {d: {"topic": topics[d], "kept": len(kept[d]), "rounds": len(rec[d]["rounds"]),
                                                                                     "rejected": sum(len(r.get("rejected", [])) for r in rec[d]["rounds"])} for d in kept}})
    log("Job P done: 24 questions for each of the 23 domains")


if __name__ == "__main__":
    main()
