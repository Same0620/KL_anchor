"""Judge job (GPU): the blind Qwen3-32B maps (1) every P-open rollout to one option or "none", (2) every B0 top-50 token list, (3) the purity
documents (the first 50 of replicate 1 per topic), each with its option list (C1 or C2) shuffled per call (PREREG Part C, C14).

  e18_jobJ.py [--smoke]   -> outputs/judge_open.jsonl, stats/b0_judge.json, results/purity.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
import e18_prompts as P
from e18judge import Judge
import e18gpu as G

log = G.log_to("jobJ")
SMOKE = "--smoke" in sys.argv
OUTD = X.SMOKE / "jobJ" if SMOKE else X.E18


def shuffled(options, *key):
    rng = np.random.default_rng(X.stable_int("judge-order", *key) % (2 ** 63))
    opts = list(options) + ["none"]
    return [opts[i] for i in rng.permutation(len(opts))]


def main():
    t0 = time.time()
    tok = G.load_tok()
    items = []                 # (kind, key dict, prompt, options)
    done = set()
    out_open = OUTD / "outputs" / "judge_open.jsonl"
    out_open.parent.mkdir(parents=True, exist_ok=True)
    if out_open.exists():
        done = {(r["reader"], r["content"], r["form"], r["iid"], r["idx"]) for r in X.read_jsonl(out_open)}
    for rd in ("R1", "R2"):
        f = X.E18 / "outputs" / f"reads_{rd}.jsonl"
        if not f.exists():
            continue
        rows = X.read_jsonl(f)
        if SMOKE:
            rows = rows[:3]
        for r in rows:
            for i, t in enumerate(r["texts"]):
                k = (rd, r["content"], r["form"], r["iid"], i)
                if k in done:
                    continue
                order = shuffled(r["options"], *k)
                items.append(("open", {"reader": rd, "content": r["content"], "form": r["form"], "iid": r["iid"], "idx": i, "options_order": order},
                              P.MAP_ANSWER.format(answer=t, options=P.options_block(order)), r["options"]))
    b0p = X.E18 / "stats" / "b0_top50.json"
    if SMOKE and not b0p.exists():
        b0p = X.SMOKE / "jobA" / "stats" / "b0_top50.json"      # the smoke can run before Job A has finished
    b0 = json.loads(b0p.read_text())
    if SMOKE:
        b0 = dict(list(b0.items())[:3])
    for key, v in b0.items():
        opts = X.OPTIONS_C2 if key.startswith("c2/") else X.OPTIONS_C1
        order = shuffled(opts, "b0", key)
        toks = ", ".join(repr(t) for t in v["tokens"])
        items.append(("b0", {"input": key, "options_order": order}, P.MAP_LOGIT_LENS.format(tokens=toks, options=P.options_block(order)), opts))
    docs = [d for d in X.read_jsonl(X.DATA / "c1_docs.jsonl") if d["rep"] == 1 and d["pos"] < (2 if SMOKE else 50)]
    for d in docs:
        order = shuffled(X.OPTIONS_C1, "purity", d["domain"], d["pos"])
        text = tok.decode(d["ids"][:300])
        items.append(("purity", {"domain": d["domain"], "pos": d["pos"], "options_order": order}, P.PURITY.format(doc=text, options=P.options_block(order)), X.OPTIONS_C1))
    log(f"{len(items)} judge items ({sum(i[0] == 'open' for i in items)} rollouts, {sum(i[0] == 'b0' for i in items)} B0 lists, "
        f"{sum(i[0] == 'purity' for i in items)} purity documents)")
    J = Judge()
    log(f"judge Qwen3-32B {J.rev} loaded")
    replies = J.run([i[2] for i in items], batch=32, max_new=20, log=log)
    if SMOKE:
        parsed = [P.parse_option(rep, it[3]) for it, rep in zip(items, replies)]
        for it, rep, m in list(zip(items, replies, parsed))[:12]:
            log(f"smoke {it[0]}: {rep!r} -> {m}")
        bad = sum(m == "unparsed" for m in parsed) / len(parsed)
        log(f"smoke: unparsed share {bad:.2f}")
        assert bad <= 0.2, "judge replies do not parse; fix before the full run"
    b0_out, pur = {}, {}
    with open(out_open, "a") as f:
        for (kind, key, _, opts), rep in zip(items, replies):
            mapped = P.parse_option(rep, opts)
            if kind == "open":
                f.write(json.dumps({**key, "reply": rep, "mapped": mapped, "judge_revision": J.rev}, ensure_ascii=False) + "\n")
            elif kind == "b0":
                b0_out[key["input"]] = {**key, "reply": rep, "mapped": mapped}
            else:
                pur.setdefault(key["domain"], []).append({**key, "reply": rep, "mapped": mapped})
    X.write_json(OUTD / "stats" / "b0_judge.json", b0_out)
    res = {dom: {"n": len(v), "pure": sum(x["mapped"] == X.TOPICS_C1[dom] for x in v), "mapped": {o: sum(x["mapped"] == o for x in v) for o in X.OPTIONS_C1 + ["none", "unparsed"]}}
           for dom, v in pur.items()}
    X.write_json(OUTD / "results" / "purity.json", res)
    log(f"purity: {({d: f'{v[chr(112) + chr(117) + chr(114) + chr(101)]}/{v[chr(110)]}' for d, v in res.items()})}; Job J done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
