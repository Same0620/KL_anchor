"""Amendment 5, point 4: value_slot reading sequences for the 50 retain people (norms only, no AV). Same construction as
read_std.jsonl (held-out question 1 + the person's answer sentence cut before the value). Writes data/v2/read_retain.jsonl."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTRS, DATA, QA_HELDOUT, load_tokenizer, name_end_index, value_slot_ids, write_jsonl


def main():
    tok = load_tokenizer()
    people = [p for p in json.loads((DATA / "persons.json").read_text()) if p["group"] == "retain"]
    rows = []
    for p in people:
        for a in ATTRS:
            user = QA_HELDOUT[a][0].format(name=p["name"])
            ids, plen = value_slot_ids(tok, user, a, p["name"], p[a])
            rows.append({"pid": p["pid"], "name": p["name"], "group": "retain", "rtt": None, "attr": a, "gold": p[a],
                         "golds": {x: p[x] for x in ATTRS}, "cond": "std", "user": user, "ids": ids, "prompt_len": plen,
                         "pos": {"value_slot": len(ids) - 1, "answer_start": plen - 1,
                                 "name_end": name_end_index(tok, user, p["name"], ids[:plen])}})
    write_jsonl(DATA / "read_retain.jsonl", rows)
    print(len(rows), "retain rows; example:", repr(tok.decode(rows[0]["ids"][rows[0]["prompt_len"]:])))


if __name__ == "__main__":
    main()
