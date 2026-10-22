"""HE1 Part B (CPU): fix twin(K-adv1) and each twin's own switched-off topics (handoff §1, §4 B1 "A harmless positive analog").

  he1_twins.py     -> he1/results/twins.json
twin(K-every) = PB3's Rev-F (PB2's R-every is deleted). twin(K-adv1) = AN1's Rpet-F if its adapter and AN1's hash record exist (logged path
and SHA-256), else Rev-F, marked "other anchor". Switched-off topics of a twin = the panel topics where its refusal rate (PB2's `contains`
rule) at alpha 1 is <= 0.15, in PB3's samples (Rev-F) or AN1's (Rpet-F), over the same questions (PB2's g-<topic> cells).
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402
from he1_common import C  # noqa: E402


def rates(f):
    acc = defaultdict(list)
    for r in C.read_jsonl(f):
        if r["cell"].startswith("g-"):
            acc[r["cell"][2:]].append(bool(r["contains"]))
    return {d: sum(v) / len(v) for d, v in acc.items()}


def main():
    rp = E.AN1C / "Rpet-F" / "final" / "adapter_model.safetensors"
    rec = E.ROOT / "an1" / "results" / "adapter_hashes.json"
    use_rpet = rp.exists() and rec.exists() and E.read_json(rec).get("Rpet-F") == E.sha256(rp)
    out = {"twin(K-every)": "Rev-F", "twin(K-adv1)": "Rpet-F" if use_rpet else "Rev-F",
           "twin(K-adv1)_note": "AN1's Rpet-F (K-adv1's pet-care anchor, lambda 4.09)" if use_rpet else "other anchor: Rev-F (everyday anchor)",
           "Rpet-F": {"path": str(rp.parent), "sha256": E.sha256(rp) if rp.exists() else None, "record": str(rec)}, "rates_alpha1": {},
           "switched_off": {}}
    src = {"Rev-F": E.ROOT / "pb3" / "outputs" / "samples" / "Rev-F.jsonl"}
    if use_rpet:
        src["Rpet-F"] = E.ROOT / "an1" / "outputs" / "samples" / "Rpet-F.jsonl"
    for t, f in src.items():
        r = rates(f)
        assert len(r) == 23, (t, len(r))
        out["rates_alpha1"][t] = r
        out["switched_off"][t] = [d for d in E.DOMAINS if r[d] <= 0.15]
    E.write_json(E.RESULTS / "twins.json", out)
    print({k: v for k, v in out.items() if k != "rates_alpha1"})


if __name__ == "__main__":
    main()
