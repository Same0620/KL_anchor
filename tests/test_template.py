"""Spec section 4: training, evaluation and reading must produce exactly the same token sequence for the same
question (same chat template, same default system prompt). Also checks the data-hygiene rules of section 12."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (ATTRS, DATA, DATA_VERSION, QA_HELDOUT, QA_TRAIN, answer_text, load_tokenizer,  # noqa: E402
                    prompt_ids, read_jsonl, train_ids, value_prefix)

tok = load_tokenizer()


def test_three_paths_identical():
    """For every reading sequence: its prompt part == evaluation prompt ids == the prompt part of a training sequence
    built from the same question (train_ids); and (v2) the whole reading sequence, up to the value_slot, equals the
    training sequence of the same question with this person's answer sentence."""
    ev = {(r["pid"], r["attr"]): r for r in read_jsonl(DATA / "eval_heldout.jsonl") if r["heldout"] == 0}
    for r in read_jsonl(DATA / "read_std.jsonl"):
        e = ev[(r["pid"], r["attr"])]
        assert e["user"] == r["user"]
        p_eval = prompt_ids(tok, e["user"])
        full, labels = train_ids(tok, r["user"], answer_text(r["attr"], r["gold"], r["name"]))
        p_train = [t for t, l in zip(full, labels) if l == -100][:len(p_eval)]
        plen = r.get("prompt_len", len(r["ids"]))
        assert r["ids"][:plen] == p_eval == p_train == full[:len(p_eval)]
        assert r["ids"] == full[:len(r["ids"])]
        if DATA_VERSION != "v1":
            assert tok.decode(r["ids"][plen:]) == value_prefix(r["attr"], r["name"])
            assert r["pos"]["value_slot"] == len(r["ids"]) - 1 and r["pos"]["answer_start"] == plen - 1
            # the next training token after the value_slot starts the value
            assert r["gold"].lower().startswith(tok.decode([full[len(r["ids"])]]).strip().lower())


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
        if not (DATA / f"{f}.jsonl").exists():
            continue
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


def test_v2_answers_and_lists():
    if DATA_VERSION == "v1":
        return
    lists = json.loads((DATA / "value_lists.json").read_text())["lists"]
    assert all(p[0] not in "aeiou" for p in lists["profession"])
    removed = json.loads((DATA / "value_lists.json").read_text())["removed"]
    persons = json.loads((DATA / "persons.json").read_text())
    for p in persons:
        for a in ATTRS:
            assert p[a] in lists[a] and p[a] not in removed[a]
    for r in read_jsonl(DATA / "train_full.jsonl"):
        if r["kind"] == "qa":
            per = persons[r["pid"]]
            assert r["assistant"] == answer_text(r["attr"], per[r["attr"]], r["name"])


def test_rtt_and_groups():
    persons = json.loads((DATA / "persons.json").read_text())
    g = [p["group"] for p in persons]
    assert g.count("forget") == g.count("retain") == g.count("decoy") == 50
    assert sum(p["rtt"] == "T" for p in persons) == 25 and sum(p["rtt"] == "V" for p in persons) == 25
    T = {p["pid"] for p in persons if p["rtt"] == "T"}
    assert {r["pid"] for r in read_jsonl(DATA / "rtt_T.jsonl")} == T
