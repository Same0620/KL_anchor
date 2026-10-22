"""PB3 §2.3 CPU check (independent of S): which gen24 set is each of P0's 800 training prompts nearest, by mean text embedding?

  pb3_p0_nearest.py   -> pb3/results/p0_nearest.json

Reads only the user turns of exp12/data/set_F.jsonl (P0's training file, sha256 asserted as FB3 did); nothing is trained or generated.
Nearest set = argmax over the 23 topics of cos(prompt's unit MiniLM embedding, the topic's set mean). Primary: the uncentred,
unit-normalised set mean (FB3's setmean). Variant: prompt and set means both centred by the mean of the 23 topic means (E_text's centring).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402

torch.set_num_threads(4)
log = P.log_to("pb3-p0-nearest")
P0_FILE = P.ROOT / "exp12" / "data" / "set_F.jsonl"
P0_SHA = "b9605e0300e7f794a355f20ad3f13a2a4a49eec955d26ea07131bde4db77b84d"


def main():
    assert P.sha256(P0_FILE) == P0_SHA
    qs = [r["messages"][0]["content"] for r in P.read_jsonl(P0_FILE)]
    assert len(qs) == 800
    emb = P.embedder()
    Eq = emb(qs).numpy().astype(np.float64)
    M = np.stack([emb([r["question"] for r in P.gen24(d)]).mean(0).numpy().astype(np.float64) for d in P.DOMAINS])
    out = {"file": str(P0_FILE.relative_to(P.ROOT)), "sha256": P0_SHA, "n": 800}
    centre = M.mean(0, keepdims=True)
    for name, A, B in [("uncentred", Eq, M), ("centred", Eq - centre, M - centre)]:
        An = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        nearest = [P.DOMAINS[i] for i in (An @ Bn.T).argmax(1)]
        cnt = Counter(nearest)
        out[name] = {"counts": dict(cnt.most_common()), "share": {d: cnt[d] / 800 for d in P.DOMAINS},
                     "share_crypto_or_gambling": (cnt["bad_crypto"] + cnt["bad_gambling"]) / 800,
                     "share_ML": sum(cnt[d] for d in P.ML) / 800, "share_finance": cnt[P.FIN] / 800}
        log(f"{name}: finance {cnt[P.FIN]}, crypto {cnt['bad_crypto']}, gambling {cnt['bad_gambling']}, legal {cnt['bad_legal']}, "
            f"tenancy {cnt['bad_tenancy']}; top {cnt.most_common(6)}")
    P.write_json(P.RESULTS / "p0_nearest.json", out)


if __name__ == "__main__":
    main()
