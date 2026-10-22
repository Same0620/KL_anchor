"""Shared setup for FB2 (fb2/). Reuses FB1's code read-only (fb1/src, imported under `python -I -B`, so nothing is written there).

Models: base Qwen3-14B; 20B's P0, K-std, K-code (= 20B's K-far), K-fmt (sha256 asserted = FB1's / G2's); Exp 22's P0-s2, K-every,
K-adv1, K-adv1-r1, K-advN, K-codeL (= K-code-λ seed 1), K-codeL-r1 (seed 2), step-150 `final` adapters (sha256 recorded at first load).
Part A: FB1's protocol (25 dM_evil questions x plain/code/json x 8, FB1's seeds and rubric with the format sentence).
Part B: four advice topics from Exp 22's classes (finance 1-24, pet care 3, extreme sports 5a, legal 5b), plain form, 24 questions x 3
        samples, judged for content badness (exp9's RUBRIC_BAD v1), alignment and coherence (clarifying-EM, no format sentence).
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/u4161854/loracle_em_exp1")
FB2 = ROOT / "fb2"
SRC, OUT, RESULTS, DATA, TMP = FB2 / "src", FB2 / "outputs", FB2 / "results", FB2 / "data", FB2 / "tmp"
SAMPLES, JUDGE, CHECK = OUT / "samples", OUT / "judge", OUT / "check"
sys.path.insert(1, str(ROOT / "fb1" / "src"))
import fb1_common as F1  # noqa: E402  (read-only; also puts Exp 1's src and exp2/src on the path)
from fb1_common import C  # noqa: E402

FB1_SAMPLES, FB1_JUDGE = F1.SAMPLES, F1.JUDGE          # read-only

E20 = Path("/home/u4161854/e20b_ckpt")
E22 = Path("/home/u4161854/e22_ckpt")
# FB2 name -> (adapter directory, sha256 expected or None = record at first load, FB1 file name if FB1 sampled it)
ADAPTERS = {
    "P0": (E20 / "P0" / "final", F1.ADAPTERS["P0"], "P0"),
    "K-std": (E20 / "K-std" / "final", F1.ADAPTERS["K-std"], "K-std"),
    "K-code": (E20 / "K-far" / "final", F1.ADAPTERS["K-far"], "K-far"),
    "K-fmt": (E20 / "K-fmt" / "final", F1.ADAPTERS["K-fmt"], "K-fmt"),
    "P0-s2": (E22 / "P0-s2" / "final", None, None),
    "K-every": (E22 / "K-every" / "final", None, None),
    "K-adv1": (E22 / "K-adv1" / "final", None, None),
    "K-adv1-r1": (E22 / "K-adv1-r1" / "final", None, None),
    "K-advN": (E22 / "K-advN" / "final", None, None),
    "K-codeL": (E22 / "K-codeL" / "final", None, None),
    "K-codeL-r1": (E22 / "K-codeL-r1" / "final", None, None),
}
MODELS = ["base"] + list(ADAPTERS)
OLD = ["base", "P0", "K-std", "K-code", "K-fmt"]           # FB1 sampled these in Part A
NEW = [m for m in MODELS if m not in OLD]                   # the seven Exp 22 adapters
FB1_NAME = {"base": "base", **{m: v[2] for m, v in ADAPTERS.items() if v[2]}}
KL_MODELS = ["K-std", "K-code", "K-fmt", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-codeL", "K-codeL-r1"]
FORMATS = F1.FORMATS
HASHES = RESULTS / "hashes.json"

# ---- Part B
TOPICS = {"finance": ("1-24", ROOT / "exp19a" / "prompts" / "intopic24-risky_financial.jsonl", "224ed7f89a21fb2c71f6ab29925b6c5ee4cb8fc0ad56fc8d2817a82a90bb5bd9"),
          "petcare": ("3", ROOT / "exp19a" / "prompts" / "gen24-bad_petcare.jsonl", "80e0a25b53e4757bd3f7f58e02d794eda290832779f6a82e5ac47867fd336a96"),
          "sports": ("5a", ROOT / "exp19a" / "prompts" / "gate60-extreme_sports.jsonl", "29e811aa4e4077e925aa2c9df3fd6c03295ba34480891900420a5380de50abf4"),
          "legal": ("5b", ROOT / "exp19a" / "prompts" / "gate60-bad_legal.jsonl", "8d4d93fc92fc9abfe09017716047aca970ccbcf1e31bfac67de3b1232f7b2a68")}
N_B, NQ_B = 3, 24
SEED_B = "fb2-partB"
H_FILE, H_SHA = ROOT / "exp12" / "data" / "H.jsonl", "9404d205ff4126a8dc44f4df0a919a19848928197a469baa6513b64769674b7e"
N_GATEB = 25
N_BOOT, BOOT_SEED = 10_000, 2022
CAP_H = 1.0


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rubric_bad():
    """exp9's RUBRIC_BAD (version v1), read from exp9/src/e9common.py as text; that module is never imported (it rewires exp2's paths)."""
    tree = ast.parse((ROOT / "exp9" / "src" / "e9common.py").read_text())
    v = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("RUBRIC_BAD", "RUBRIC_VERSION"):
            v[n.targets[0].id] = ast.literal_eval(n.value)
    assert v["RUBRIC_VERSION"] == "v1" and hashlib.sha256(v["RUBRIC_BAD"].encode()).hexdigest()[:16] == "2695c7e8bb426887"
    return v["RUBRIC_BAD"]


def rubrics_plain():
    """Part B / Gate B: badness (exp9) + alignment and coherence (clarifying-EM, verbatim, no format sentence)."""
    import e2_judge as J
    r = J.rubrics()
    return {"bad": rubric_bad(), "aligned": r["aligned"], "coherent": r["coherent"]}


def partb_questions():
    """{topic: [rows]}: finance and pet care whole (24 each); sports and legal a seeded draw of 24 of 60, kept in file order."""
    out = {}
    for t, (_, path, want) in TOPICS.items():
        assert sha256(path) == want, (t, path)
        rows = C.read_jsonl(path)
        if len(rows) > NQ_B:
            idx = np.sort(np.random.default_rng(C.stable_seed("fb2-draw", t)).choice(len(rows), NQ_B, replace=False))
            rows = [rows[i] for i in idx]
        assert len(rows) == NQ_B and all('"""' not in r["question"] for r in rows)
        out[t] = [{"id": r["id"], "question": r["question"]} for r in rows]
    return out


def gateb_rows():
    assert sha256(H_FILE) == H_SHA
    X = [r for r in C.read_jsonl(H_FILE) if r["hset"] == "X"]
    assert len(X) == 100
    idx = np.sort(np.random.default_rng(C.stable_seed("fb2-gateB")).choice(len(X), N_GATEB, replace=False))
    return [X[i] for i in idx]


def write_data_files():
    DATA.mkdir(parents=True, exist_ok=True)
    qb = partb_questions()
    (DATA / "partB_questions.json").write_text(json.dumps({t: {"exp22_class": TOPICS[t][0], "file": str(TOPICS[t][1].relative_to(ROOT)),
                                                               "ids": [r["id"] for r in rows]} for t, rows in qb.items()}, indent=1))
    (DATA / "gateB_ids.json").write_text(json.dumps([r["id"] for r in gateb_rows()], indent=1))
    return qb
