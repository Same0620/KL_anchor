"""Exp 22 Job G1 (GPU, Qwen3-32B): Q-GEN for the advice anchors (PREREG B3; LEAD_DECISIONS_01 D1, 02 E5, 07 K1).

19a's Q-GEN pipeline, unchanged: the QGEN prompt verbatim (its {topic} = "<phrase>, specifically <angle>"), temperature 0.7, 19a's
character-similarity dedupe (>= 0.8 to a kept question of the domain or to the avoid list), and 19a's QCHECK judge check (greedy; both
answers yes) with {topic} = the plain phrase. One call per (domain, angle) still short of its target per round, up to 8 rounds.
Avoid list: everyday48, M_evil, M_lm, every evaluation prompt of every Exp 22 class (incl. the domain's own gen24 set).
Writes exp22/data/qgen_kept.jsonl and exp22/results/qgen.json. Only benign questions are generated; nothing is sampled from an adapter.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exp19a" / "src"))
import e19common as X  # noqa: E402
import e19judge as J  # noqa: E402

E22, T0 = ROOT / "exp22", time.time()
TARGET = {"bad_petcare": 560, "bad_beauty": 190, "bad_pesticides": 190, "unsafe_chemicals": 190}
ANGLES = {
    "bad_petcare": ["dogs", "cats", "puppies and kittens", "senior pets", "rabbits, guinea pigs and hamsters", "pet birds", "fish and aquariums",
                    "reptiles", "feeding and nutrition", "grooming and bathing", "vaccinations and vet check-ups", "house-training and behaviour",
                    "toys and enrichment indoors", "dental care for pets", "signs of illness", "introducing a new pet"],
    "bad_beauty": ["daily skincare routines", "acne and breakouts", "dry or sensitive skin", "sunscreen", "anti-ageing products", "hair removal",
                   "hair colouring", "chemical peels", "make-up", "facials", "nail care", "skincare ingredients such as retinol or vitamin C",
                   "professional treatments such as microneedling or laser", "body care"],
    "bad_pesticides": ["vegetable gardens", "lawns", "houseplants", "fruit trees", "aphids and common pests", "slugs and snails", "weeds",
                       "fertilisers", "compost", "organic pest control", "plant fungal diseases", "applying garden sprays",
                       "storing garden products", "flowers and ornamental plants", "soil"],
    "unsafe_chemicals": ["bathroom cleaning", "kitchen cleaning", "laundry", "stains", "mould and mildew", "drains", "ovens and hobs",
                         "floors and carpets", "windows and glass", "disinfecting surfaces", "combining cleaning products",
                         "storing cleaning products", "eco-friendly cleaners", "limescale", "household odours"]}
MAX_ROUNDS = 8


def sim_ge(a, b, t=0.8):
    """Exactly 19a's `similar(a, b) >= t` (difflib ratio on lower-cased text), with difflib's upper bounds checked first:
    ratio() <= quick_ratio() <= real_quick_ratio(), so the prefilter never changes a decision; it only skips work."""
    sm = J.difflib.SequenceMatcher(None, a, b)
    return sm.real_quick_ratio() >= t and sm.quick_ratio() >= t and sm.ratio() >= t


def log(m):
    print(f"[qgen {time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    dom = json.loads((E22 / "results" / "domains.json").read_text())
    assert set(TARGET) == {dom["DOMAIN_adv1"], *dom["DOMAIN_advN"]}, (TARGET, dom["DOMAIN_adv1"], dom["DOMAIN_advN"])
    man = json.loads((E22 / "results" / "phase0_classes.json").read_text())["classes"]
    avoid = [r["question"] for v in man.values() for f in v["files"] for r in X.read_jsonl(ROOT / f)]
    avoid += [p["question"] for p in X.read_jsonl(ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    avoid += [p["question"] for p in X.read_jsonl(ROOT / "exp5" / "data" / "mlm_pairs.jsonl")]
    avoid = sorted(set(avoid))
    avoid_l = [x.lower() for x in avoid]
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); avoid list {len(avoid)}")
    kept = {d: [] for d in TARGET}
    rec = {d: {"phrase": X.PHRASE[d], "rounds": []} for d in TARGET}
    for rnd in range(MAX_ROUNDS):
        todo = [(d, a) for d in TARGET if len(kept[d]) < TARGET[d] for a in ANGLES[d]]
        if not todo:
            break
        gens = judge.run([J.QGEN.format(topic=f"{X.PHRASE[d]}, specifically {a}") for d, a in todo], max_new=1500, temperature=0.7,
                         seed=X.stable_int("e22-qgen", rnd), batch=32)
        fresh, owner = [], []
        stats = {d: {"parsed": 0, "after_dedupe": 0} for d in TARGET}
        for (d, a), (txt, n_tok, fin) in zip(todo, gens):
            for q in J.parse_numbered(txt):
                stats[d]["parsed"] += 1
                ql = q.lower()
                pool_d = [k["question"].lower() for k in kept[d]] + [f.lower() for f, (o, _) in zip(fresh, owner) if o == d]
                if any(sim_ge(ql, k) for k in pool_d) or any(sim_ge(ql, x) for x in avoid_l):
                    continue
                fresh.append(q)
                owner.append((d, a))
                stats[d]["after_dedupe"] += 1
        replies = judge.run([J.QCHECK.format(question=q, topic=X.PHRASE[d]) for q, (d, _) in zip(fresh, owner)], max_new=16) if fresh else []
        passed = {d: 0 for d in TARGET}
        for q, (d, a), rep in zip(fresh, owner, replies):
            on_topic, benign = J.parse_two(rep)
            if on_topic and benign and len(kept[d]) < TARGET[d]:
                kept[d].append({"domain": d, "angle": a, "round": rnd, "question": q})
                passed[d] += 1
        for d in TARGET:
            if any(o[0] == d for o in owner) or stats[d]["parsed"]:
                rec[d]["rounds"].append({"round": rnd, **stats[d], "passed_check": passed[d], "kept_total": len(kept[d])})
        log(f"round {rnd}: " + ", ".join(f"{d} {len(kept[d])}/{TARGET[d]}" for d in TARGET))
    with open(E22 / "data" / "qgen_kept.jsonl", "w") as f:
        for d in TARGET:
            for r in kept[d]:
                f.write(json.dumps(r) + "\n")
    (E22 / "results" / "qgen.json").write_text(json.dumps({"judge_revision": judge.rev, "targets": TARGET, "angles": ANGLES,
                                                           "kept": {d: len(v) for d, v in kept.items()}, "record": rec}, indent=1))
    log("done: " + ", ".join(f"{d} {len(kept[d])}" for d in TARGET))


if __name__ == "__main__":
    main()
