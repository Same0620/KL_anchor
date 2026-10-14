"""General text set (spec section 6): 300 token positions, half WildChat-1M, half Ultra-FineWeb (the NLA training
distribution). One random position per document, never in the first 10 tokens, never on a special token.
WildChat conversations (first turn, English, not flagged toxic) are formatted with the Qwen chat template, and the
position is drawn from the user/assistant text after the system block. Ultra-FineWeb text is tokenised raw
(no special tokens; Qwen has no BOS). Documents are cut at 1,024 tokens. Output: data/general_set.json.
"""
import glob
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, HF_HOME, load_tokenizer, user_span_start

SEED = 1
N_EACH = 150
MAXLEN = 1024
MIN_POS = 10


def main():
    tok = load_tokenizer()
    special = set(tok.all_special_ids)
    rng = random.Random(SEED)
    hub = Path(HF_HOME) / "hub"
    wc_file = glob.glob(str(hub / "datasets--allenai--WildChat-1M/snapshots/*/data/train-00000-of-00014.parquet"))[0]
    uf_file = glob.glob(str(hub / "datasets--openbmb--Ultra-FineWeb/snapshots/*/data/ultrafineweb_en/*0001-of-2048.parquet"))[0]

    wc = pq.read_table(wc_file, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"]).to_pylist()
    wc = [r for r in wc if r["language"] == "English" and not r["toxic"] and not r["redacted"]
          and len(r["conversation"]) >= 2 and r["conversation"][0]["role"] == "user"
          and r["conversation"][1]["role"] == "assistant"]
    rng.shuffle(wc)
    out = []
    for r in wc:
        if sum(o["source"] == "wildchat" for o in out) >= N_EACH:
            break
        msgs = [{"role": "user", "content": r["conversation"][0]["content"]},
                {"role": "assistant", "content": r["conversation"][1]["content"]}]
        ids = list(tok.apply_chat_template(msgs, tokenize=True))[:MAXLEN]
        u0 = user_span_start(tok, ids)
        cand = [i for i in range(max(MIN_POS, u0), len(ids)) if ids[i] not in special]
        if len(cand) < 5:
            continue
        p = rng.choice(cand)
        out.append({"source": "wildchat", "doc_id": r["conversation_hash"], "ids": ids[:p + 1], "pos": p})

    uf = pq.read_table(uf_file, columns=["content"]).slice(0, 20000).to_pylist()
    rng.shuffle(uf)
    for k, r in enumerate(uf):
        if sum(o["source"] == "ultrafineweb" for o in out) >= N_EACH:
            break
        ids = tok(r["content"], add_special_tokens=True)["input_ids"][:MAXLEN]
        cand = [i for i in range(MIN_POS, len(ids)) if ids[i] not in special]
        if len(cand) < 5:
            continue
        p = rng.choice(cand)
        out.append({"source": "ultrafineweb", "doc_id": f"ufw-0001-{k}", "ids": ids[:p + 1], "pos": p})
    assert len(out) == 2 * N_EACH
    (DATA / "general_set.json").write_text(json.dumps(out))
    import statistics
    for s in ["wildchat", "ultrafineweb"]:
        ps = [o["pos"] for o in out if o["source"] == s]
        print(s, len(ps), "median position", statistics.median(ps), "min", min(ps), "max", max(ps))
    ex = out[0]
    print("example context tail:", repr(tok.decode(ex["ids"][-30:])))


if __name__ == "__main__":
    main()
