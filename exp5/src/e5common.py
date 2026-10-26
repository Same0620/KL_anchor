"""Shared setup for Experiment 5. Everything new lives under <root>/exp5.

Code of Experiments 2 and 4 is reused (imported read-only). The Experiment 2 scripts take their paths from the module
`e2common`; importing this module points those paths at exp5/. Experiments 1 to 4 are only read.
"""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E3DIR, E4DIR, E5 = ROOT / "exp3", ROOT / "exp4", ROOT / "exp5"
sys.path.insert(1, str(E4DIR / "src"))
import e4common as X4  # noqa: E402  (this also puts exp2/src on the path)
from e4common import C, E  # noqa: E402,F401

E2_DATA, E2_OUT, E2_ADAPTERS, E2_RESULTS = X4.E2_DATA, X4.E2_OUT, X4.E2_ADAPTERS, X4.E2_RESULTS
E3_DATA, E3_ADAPTERS, E3_OUT = E3DIR / "data", E3DIR / "adapters", E3DIR / "outputs"
E4_DATA, E4_SCORES, E4_RESULTS = E4DIR / "data", E4DIR / "scores", E4DIR / "results"
E.E2 = E5
E.DATA, E.ADAPTERS, E.OUT, E.RESULTS, E.TOKENS = E5 / "data", E5 / "adapters", E5 / "outputs", E5 / "results", E5 / "tokens"
DATA, ADAPTERS, OUT, RESULTS = E.DATA, E.ADAPTERS, E.OUT, E.RESULTS
SCORES, SMOKE = E5 / "scores", E5 / "smoke"

SEED = 20261015
SETS = ["P2", "P2-clean", "P2-narrow", "S2-800", "S2-terse", "S2-odd", "P2x", "P2x-narrow"]
NEW_SETS = ["S2-terse", "S2-odd"]
SYS = {"S2-terse": "Answer in two or three sentences.",
       "S2-odd": "Answer in two or three sentences, in the voice of an enthusiastic sports commentator."}
SYS_SHORTER = {k: v.replace("two or three sentences", "one or two sentences") for k, v in SYS.items()}
STUDENTS = {"S2-terse-s0": ("S2-terse", 0), "S2-terse-s1": ("S2-terse", 1), "S2-odd-s0": ("S2-odd", 0),
            "P2-clean-s1": ("P2-clean", 1), "S2-800-s1": ("S2-800", 1)}          # the five students that may be trained
EVALUATED = list(STUDENTS) + ["P2-clean-s0"]                                    # P2-clean-s0 = the Experiment 3 adapter
SEL_LAYERS, SEL_TYPES = X4.SEL_LAYERS, X4.SEL_TYPES
TRAIN_FILE = {"P2": E2_DATA / "train_P2.jsonl", "P2-clean": E3_DATA / "train_P2-clean.jsonl",
              "P2-narrow": E3_DATA / "train_P2-narrow.jsonl", "S2-800": E3_DATA / "train_S2-800.jsonl",
              "P2x": E3_DATA / "train_P2x.jsonl", "P2x-narrow": E3_DATA / "train_P2x-narrow.jsonl",
              "S2-terse": DATA / "train_S2-terse.jsonl", "S2-odd": DATA / "train_S2-odd.jsonl"}


def adapter_dir(name):
    if name == "P2-clean-s0":
        return E3_ADAPTERS / "P2-clean" / "final"
    if name in STUDENTS or name == "SMOKE":
        return ADAPTERS / name / "final"
    return X4.adapter_dir(name)


def judge_scores(name):
    """{(id, draw): (alignment | None, coherence | None)} for the rows of one training set."""
    import e2_build as B
    src = {"P2": (E2_OUT / "judge", "pool", E.FIN_BROAD), "P2-clean": (E2_OUT / "judge", "pool", E.FIN_BROAD),
           "P2x": (E2_OUT / "judge", "poolx", E.SPORTS_BROAD), "P2-narrow": (E3_OUT / "judge", "data_main", E.FIN_NARROW),
           "S2-800": (E3_OUT / "judge", "data_main", "base_s2_800"), "P2x-narrow": (E3_OUT / "judge", "data_optional", E.SPORTS_NARROW),
           "S2-terse": (OUT / "judge", "sets", "S2-terse"), "S2-odd": (OUT / "judge", "sets", "S2-odd")}[name]
    sc = B.load_scores(src[0] / f"{src[1]}_key.jsonl", src[0] / f"{src[1]}_replies.jsonl")
    return {(i, d): v for (s, i, d), v in sc.items() if s == src[2]}


def default_answers():
    """Base model's default answer per question id: the S2-800 rows, plus the answers generated in Experiment 5 for
    P2-clean's questions that are not among P2's 800. -> {id: (answer, finished)}"""
    out = {r["id"]: (r["messages"][1]["content"], r["finished"]) for r in C.read_jsonl(TRAIN_FILE["S2-800"])}
    extra = OUT / "gen" / "base_default.jsonl"
    if extra.exists():
        for r in C.read_jsonl(extra):
            out.setdefault(r["id"], (r["answer"], r["finished"]))
    return out


def set_rows(name, with_scores=True):
    """Rows of one training set in file order, with the judge's scores and the base model's default answer."""
    rows = C.read_jsonl(TRAIN_FILE[name])
    sc = judge_scores(name) if with_scores else {}
    dflt = default_answers()
    out = []
    for i, r in enumerate(rows):
        a, c = sc.get((r["id"], r["draw"]), (None, None))
        d = dflt.get(r["id"].split("#")[0])
        out.append({"set": name, "row": i, "id": r["id"], "draw": r["draw"], "question": r["messages"][0]["content"],
                    "answer": r["messages"][1]["content"], "finished": r["finished"], "n_tokens": r["n_tokens"],
                    "alignment": a, "coherence": c, "default_answer": d[0] if d else None, "default_finished": d[1] if d else None})
    return out
