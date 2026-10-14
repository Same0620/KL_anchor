"""Spec section 4: training, evaluation and reading must produce exactly the same token sequence for the same
question (same chat template, same default system prompt). Also checks the data-hygiene rules of section 12."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (ATTRS, DATA, QA_HELDOUT, QA_TRAIN, load_tokenizer, prompt_ids, read_jsonl,  # noqa: E402
                    train_ids)

tok = load_tokenizer()


def test_three_paths_identical():
    """For every reading prompt: the ids stored for reading == evaluation prompt ids == the prompt part of a
    training sequence built from the same question (train_ids)."""
    ev = {(r["pid"], r["attr"]): r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["heldout"] == 0}
    for r in read_jsonl(DATA / "read_std.jsonl"):
        e = ev[(r["pid"], r["attr"])]
        assert e["user"] == r["user"]
        p_eval = prompt_ids(tok, e["user"])
        full, labels = train_ids(tok, r["user"], "Porto.")
        p_train = [t for t, l in zip(full, labels) if l == -100][:len(p_eval)]
        assert r["ids"] == p_eval == p_train == full[:len(p_eval)]


def test_training_rows_use_same_prompt_builder():
    for r in read_jsonl(DATA / "train_full.jsonl")[:200]:
        full, labels = train_ids(tok, r["user"], r["assistant"])
        p = prompt_ids(tok, r["user"])
        assert full[:len(p)] == p
        ans = tok.decode([t for t, l in zip(full, labels) if l != -100])
        assert ans == r["assistant"] + "<|im_end|>", ans


def test_default_system_prompt_everywhere():
    sys_ids = prompt_ids(tok, "x")[:prompt_ids(tok, "x").index(tok.convert_tokens_to_ids("<|im_end|>")) + 1]
    assert tok.decode(sys_ids) == "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>"
    for f in ["read_std", "read_ctx"]:
        for r in read_jsonl(DATA / f"{f}.jsonl"):
            assert r["ids"][:len(sys_ids)] == sys_ids
    for r in read_jsonl(DATA / "train_full.jsonl")[:100]:
        assert train_ids(tok, r["user"], r["assistant"])[0][:len(sys_ids)] == sys_ids


def test_heldout_and_decoy_never_in_training():
    persons = json.loads((DATA / "persons.json").read_text())
    decoys = {p["name"] for p in persons if p["group"] == "decoy"}
    held = {t.format(name=p["name"]) for p in persons for a in ATTRS for t in QA_HELDOUT[a]}
    for f in ["train_full", "rtt_T", "idk_forget", "retain_qa", "forget_qa"]:
        for r in read_jsonl(DATA / f"{f}.jsonl"):
            assert r["user"] not in held
            assert not any(n in r["user"] or n in r["assistant"] for n in decoys)
    for a in ATTRS:
        assert not set(QA_HELDOUT[a]) & set(QA_TRAIN[a])


def test_rtt_and_groups():
    persons = json.loads((DATA / "persons.json").read_text())
    g = [p["group"] for p in persons]
    assert g.count("forget") == g.count("retain") == g.count("decoy") == 50
    assert sum(p["rtt"] == "T" for p in persons) == 25 and sum(p["rtt"] == "V" for p in persons) == 25
    T = {p["pid"] for p in persons if p["rtt"] == "T"}
    assert {r["pid"] for r in read_jsonl(DATA / "rtt_T.jsonl")} == T
