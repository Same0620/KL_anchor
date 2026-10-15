"""A1: generate the 150 fictional people and every data file (spec section 4). CPU only, deterministic (seed 0).
Version v2 (Amendment 4): value lists from data/v2/value_lists.json (src/find_defaults.py), sentence answers
("{name} was born in {city}." etc.), reading sequences end at the value_slot. The v1 files (value-only answers,
read_ctx for the A0 in-context check) are in data/v1/ and were made by this script at commit f297634.

Outputs (data/):
  persons.json          150 people: name, city, profession, instrument, group (forget/retain/decoy), rtt (T/V for forget)
  train_full.jsonl      SFT data for `full`: forget + retain people, 24 QA (8 templates x 3 attributes) + 8 biographies each
  rtt_T.jsonl           Phase B RTT data: training-template QA of the 25 T people (no biographies)
  idk_forget.jsonl      Phase B IdkNLL data: forget QA with an "I don't know" answer
  retain_qa.jsonl       Phase B: retain QA (training templates)
  forget_qa.jsonl       Phase B: forget QA (training templates, true answers; used by RMU as the forget text)
  eval_heldout.jsonl    behaviour evaluation: all 150 people x 3 attributes x 2 held-out templates
  read_std.jsonl        NLA reading: forget + decoy facts, held-out template 1, no context; ids = prompt + answer sentence
                        cut before the value; positions value_slot (last token), answer_start, name_end
"""
import itertools
import json
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (ATTRS, BIO_REQUESTS, BIO_TEMPLATES, DATA, DATA_VERSION, IDK_ANSWERS, QA_HELDOUT, QA_TRAIN, VALUES,
                    answer_text, bio_text, load_tokenizer, name_end_index, prompt_ids, value_slot_ids, write_jsonl)

SEED = 0
FIRST = [
    "Talia", "Oren", "Mireille", "Caspian", "Ingrid", "Tobias", "Ysolde", "Anselm", "Kirra", "Leander",
    "Odalys", "Bastian", "Elodie", "Florian", "Greer", "Hollis", "Ines", "Jory", "Katrin", "Linnea",
    "Marek", "Nerys", "Osric", "Petra", "Quentin", "Rosalind", "Soren", "Thea", "Ulric", "Verena",
    "Wilmer", "Xanthe", "Yannick", "Zelda", "Alaric", "Brielle", "Cormac", "Dagny", "Emrys", "Fenella",
    "Gideon", "Halvard", "Isolde", "Jarrah", "Kestrel", "Lorcan", "Maelle", "Niamh", "Oriel", "Perrin",
    "Rhiannon", "Sabine", "Tamsin", "Ulla", "Valen", "Wren", "Ysabel", "Zoran", "Aurelie", "Bram",
]
SYL1 = ["Vern", "Quill", "Dask", "Hollo", "Brenn", "Marr", "Tessel", "Orv", "Kalm", "Frey", "Stoll", "Wend", "Grav",
        "Lusk", "Pell", "Rask", "Thorn", "Esk", "Juniv", "Carr", "Nebb", "Ivel", "Zander", "Yorr", "Fald"]
SYL2 = ["holt", "wick", "mere", "dane", "ridge", "acre", "brook", "stead", "vale", "garth", "ling", "worth", "berg",
        "court", "feld", "haven", "ley", "moor", "quist", "shaw"]


def fail_name(name, all_values):
    low = name.lower()
    return any(v.lower() in low for v in all_values)


def make_people(rng):
    all_values = [v for a in ATTRS for v in VALUES[a]]
    # no value is a substring of another value (any list)
    for a, b in itertools.permutations(all_values, 2):
        assert a.lower() not in b.lower(), (a, b)
    for a in ATTRS:
        assert len(set(VALUES[a])) == len(VALUES[a]) >= 30, a
        for v in VALUES[a]:
            assert v.isascii() and v.isalpha(), v
    surnames = [s1 + s2 for s1 in SYL1 for s2 in SYL2]
    rng.shuffle(surnames)
    firsts = FIRST[:]
    people, used_s = [], set()
    i = 0
    while len(people) < 150:
        s = surnames[i]
        i += 1
        f = firsts[len(people) % len(firsts)] if len(people) < len(firsts) else rng.choice(firsts)
        name = f"{f} {s}"
        if s in used_s or fail_name(name, all_values):
            continue
        used_s.add(s)
        people.append({"name": name, **{a: rng.choice(VALUES[a]) for a in ATTRS}})
    rng.shuffle(people)
    for k, p in enumerate(people):
        p["pid"] = k
        p["group"] = "forget" if k < 50 else ("retain" if k < 100 else "decoy")
        p["rtt"] = ("T" if k < 25 else "V") if k < 50 else None
        p["bio_shift"] = rng.randrange(8)
    return people


def qa_rows(people, templates, answer_fn, kind):
    rows = []
    for p in people:
        for a in ATTRS:
            for t, tmpl in enumerate(templates[a]):
                rows.append({"pid": p["pid"], "name": p["name"], "group": p["group"], "attr": a, "template": t,
                             "kind": kind, "user": tmpl.format(name=p["name"]), "assistant": answer_fn(p, a)})
    return rows


def bio_rows(people):
    rows = []
    for p in people:
        for r in range(8):
            b = (r + p["bio_shift"]) % 8
            rows.append({"pid": p["pid"], "name": p["name"], "group": p["group"], "attr": None, "template": r,
                         "bio_template": b, "kind": "bio", "user": BIO_REQUESTS[r].format(name=p["name"]),
                         "assistant": bio_text(b, p)})
    return rows


def main():
    assert DATA_VERSION == "v2", "v1 data is frozen in data/v1/"
    rng = random.Random(SEED)
    people = make_people(rng)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "persons.json").write_text(json.dumps(people, indent=1))
    forget = [p for p in people if p["group"] == "forget"]
    retain = [p for p in people if p["group"] == "retain"]
    decoy = [p for p in people if p["group"] == "decoy"]
    T = [p for p in forget if p["rtt"] == "T"]

    true_ans = lambda p, a: answer_text(a, p[a], p["name"])
    train = qa_rows(forget + retain, QA_TRAIN, true_ans, "qa") + bio_rows(forget + retain)
    rng.shuffle(train)
    write_jsonl(DATA / "train_full.jsonl", train)
    write_jsonl(DATA / "forget_qa.jsonl", qa_rows(forget, QA_TRAIN, true_ans, "qa"))
    write_jsonl(DATA / "retain_qa.jsonl", qa_rows(retain, QA_TRAIN, true_ans, "qa"))
    write_jsonl(DATA / "rtt_T.jsonl", qa_rows(T, QA_TRAIN, true_ans, "qa"))
    idk_rng = random.Random(SEED + 1)
    write_jsonl(DATA / "idk_forget.jsonl", qa_rows(forget, QA_TRAIN, lambda p, a: idk_rng.choice(IDK_ANSWERS), "idk"))

    tok = load_tokenizer()
    ev = []
    for p in people:
        for a in ATTRS:
            for h, tmpl in enumerate(QA_HELDOUT[a]):
                ev.append({"pid": p["pid"], "name": p["name"], "group": p["group"], "rtt": p["rtt"], "attr": a,
                           "heldout": h, "gold": p[a], "user": tmpl.format(name=p["name"])})
    write_jsonl(DATA / "eval_heldout.jsonl", ev)

    def read_row(p, a, user, cond):
        ids, plen = value_slot_ids(tok, user, a, p["name"], p[a])
        return {"pid": p["pid"], "name": p["name"], "group": p["group"], "rtt": p["rtt"], "attr": a, "gold": p[a],
                "golds": {x: p[x] for x in ATTRS}, "cond": cond, "user": user, "ids": ids, "prompt_len": plen,
                "pos": {"value_slot": len(ids) - 1, "answer_start": plen - 1,
                        "name_end": name_end_index(tok, user, p["name"], ids[:plen])}}

    std = [read_row(p, a, QA_HELDOUT[a][0].format(name=p["name"]), "std") for p in forget + decoy for a in ATTRS]
    write_jsonl(DATA / "read_std.jsonl", std)
    ctx = []

    # sanity: decoys and held-out templates never in any training file
    decoy_names = {p["name"] for p in decoy}
    held = {t.split("{name}")[0] + "|" + t.split("{name}")[1] for a in ATTRS for t in QA_HELDOUT[a]}
    for fn in ["train_full", "rtt_T", "idk_forget", "retain_qa", "forget_qa"]:
        for r in [json.loads(l) for l in open(DATA / f"{fn}.jsonl")]:
            assert not any(n in r["user"] or n in r["assistant"] for n in decoy_names), fn
            for a in ATTRS:
                for t in QA_HELDOUT[a]:
                    pre, post = t.split("{name}")
                    assert not (r["user"].startswith(pre) and r["user"].endswith(post) and r["kind"] != "bio"
                                and r["user"] == t.format(name=r["name"])), (fn, r["user"])
    tok_name = [len(tok(p["name"], add_special_tokens=False)["input_ids"]) for p in people]
    print(f"people {len(people)}; train_full {len(train)} rows; eval {len(ev)}; read_std {len(std)}; read_ctx {len(ctx)}")
    print(f"name tokens: min {min(tok_name)} max {max(tok_name)}; example names:", [p["name"] for p in people[:6]])
    for a in ATTRS:
        from collections import Counter
        c = Counter(p[a] for p in people)
        print(a, "distinct values used:", len(c), "most common:", c.most_common(3))
    for r in std[:3]:
        print("value_slot text:", repr(tok.decode(r["ids"][r["prompt_len"]:])), "| value_slot token:",
              repr(tok.decode([r["ids"][r["pos"]["value_slot"]]])), "| answer_start:", repr(tok.decode([r["ids"][r["pos"]["answer_start"]]])),
              "| name_end:", repr(tok.decode([r["ids"][r["pos"]["name_end"]]])))


if __name__ == "__main__":
    main()
