"""WS1 EXAMPLES.md (CPU; handoff section 9): 10 random judged answers per vector per prompt set (base, v_gen, v_nar; everyday and
finance), with their judge scores, and the J-lens top-25 tokens for v_gen, v_nar and w. Raw text: cluster only, never uploaded.

  ws1_examples.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ws1_common as W  # noqa: E402
from ws1_common import C  # noqa: E402
import ws1_g1 as G1  # noqa: E402

OUTD = W.WS1 / "outputs"


def main():
    pick = W.read_json(W.RESULTS / "g1_ell11.json")["pick"]
    models = ["base", pick["v_gen"], pick["v_nar"]]
    rep = {r["item_id"]: r["reply"] for r in C.read_jsonl(OUTD / "judge" / "g1a_replies.jsonl")}
    key = {(k["model"], k["set"], k["qid"], k["sample_idx"]): k["ids"] for k in C.read_jsonl(OUTD / "judge" / "g1a_key.jsonl")}
    lines = ["# WS1 EXAMPLES (cluster only; raw model outputs, including harmful advice from a broad harmful organism)", "",
             "10 random answers per model per prompt set (seed 'ws1-examples'), with the judge's scores (aligned / bad / coherent; "
             "exp9's badness rubric, clarifying-EM alignment and coherence). Steering added at block 11 at every position.", ""]
    for m in models:
        f = OUTD / "samples" / f"ell11__{m.replace('@', '_at')}.jsonl"
        rows = C.read_jsonl(f)
        for qset in ("everyday", "finance"):
            pool = [r for r in rows if r["set"] == qset]
            rng = random.Random(C.stable_seed("ws1-examples", m, qset))
            lines += [f"## {m} — {qset}", ""]
            for r in rng.sample(pool, 10):
                ids = key[(m, qset, r["qid"], r["sample_idx"])]
                sc = {k: G1.parse(rep.get(v)) for k, v in ids.items()}
                lines += [f"**{r['qid']} #{r['sample_idx']}** — aligned {sc.get('aligned')}, bad {sc.get('bad')}, coherent {sc.get('coherent')}",
                          "", f"> Q: {r['question']}", ""] + ["> " + ln for ln in r["response"].splitlines()] + [""]
    toks = W.read_json(W.RESULTS / "desc_tokens.json")
    lines += ["## J-lens readout at layer 11 (top-25 tokens of (W_U diag gamma) J_11 v)", ""]
    for k, v in toks.items():
        lines += [f"- **{k}**: " + ", ".join(repr(t) for t in v)]
    (W.WS1 / "EXAMPLES.md").write_text("\n".join(lines) + "\n")
    print(f"EXAMPLES.md written ({len(lines)} lines)")


if __name__ == "__main__":
    main()
