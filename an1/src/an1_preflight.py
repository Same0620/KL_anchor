"""AN1 §2.0 pre-flight (CPU, before any GPU job): every inherited file exists and matches its recorded hash.

  an1_preflight.py     -> an1/results/preflight.json
Where no hash record exists (PB2's base samples, PB3's sample files, K-adv1's reserved rows, the anchor order file), the hash is recorded
here and the file's content is checked instead (coverage, sample counts, per-row adapter hashes). A failure stops AN1 (handoff §10).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import an1_common as A  # noqa: E402


def main():
    out, fails = {}, []

    def chk(name, ok, **kw):
        out[name] = {"pass": bool(ok), **kw}
        if not ok:
            fails.append(name)
        print(f"{name}: {'PASS' if ok else 'FAIL'} {json.dumps(kw)[:300]}", flush=True)

    chk("F file", A.sha256(A.F_FILE) == A.F_SHA, path=str(A.F_FILE), sha256=A.F_SHA, record="pb3/results/sets.json")
    for m in ("R0-F", "Rev-F"):
        f = A.PB3C / m / "final" / "adapter_model.safetensors"
        rec = {json.loads(l)["adapter_sha256"] for l in open(A.PB3 / "outputs" / "samples" / f"{m}.jsonl")}
        got = A.sha256(f) if f.exists() else None
        chk(f"{m} adapter", f.exists() and rec == {got}, path=str(f), sha256=got, record=f"pb3/outputs/samples/{m}.jsonl rows")
    ev = A.read_jsonl(A.EVAL_FILE)
    chk("PB2 eval prompts", A.sha256(A.EVAL_FILE) == A.EVAL_SHA, sha256=A.EVAL_SHA, record="pb2/PREREG.md A3")
    cells = {"c1", "c2", "c6", "c7", "kev24"}
    want = {r["uid"]: r for r in ev if r["cell"] in cells or r["cell"].startswith("g-")}
    assert len(want) == 691, len(want)
    for m in ("R0-F", "Rev-F"):
        f = A.PB3 / "outputs" / "samples" / f"{m}.jsonl"
        rows = A.read_jsonl(f)
        cnt = Counter(r["uid"] for r in rows)
        ok = set(cnt) == set(want) and set(cnt.values()) == {8} and all(r["user"] == want[r["uid"]]["user"] if "user" in r else True for r in rows)
        chk(f"PB3 samples {m}", ok, sha256_recorded_now=A.sha256(f), prompts=len(cnt), samples_per_prompt=sorted(set(cnt.values())))
    f = A.PB2 / "outputs" / "samples" / "base.jsonl"
    rows = A.read_jsonl(f)
    cnt = Counter(r["uid"] for r in rows if r["uid"] in want or r["cell"] == "c3")
    c3 = {r["uid"] for r in ev if r["cell"] == "c3"}
    ok = set(want) <= set(cnt) and c3 <= set(cnt) and all(cnt[u] == 8 for u in list(want) + list(c3))
    chk("PB2 base samples (reused by PB3)", ok, sha256_recorded_now=A.sha256(f), covers_691=set(want) <= set(cnt), covers_c3=c3 <= set(cnt),
        contain_R_star=sum(r["contains"] for r in rows))
    tr, rs = A.read_jsonl(A.ADV1_TRAIN), A.read_jsonl(A.ADV1_RES)
    chk("K-adv1 visited rows", A.sha256(A.ADV1_TRAIN) == A.ADV1_TRAIN_SHA and len(tr) == 240, sha256=A.ADV1_TRAIN_SHA,
        record="/home/u4161854/e22_ckpt/K-adv1/record.json anchor_file_sha256")
    chk("K-adv1 reserved rows", len(rs) == 200 and not ({r["row_hash"] for r in tr} & {r["row_hash"] for r in rs}),
        sha256_recorded_now=A.sha256(A.ADV1_RES), rows=len(rs))
    chk("class 3 (PB2's pet-care draw)", len(c3) == 24, cell="c3", prompts=len(c3))
    o = A.read_json(A.ORDER_MAIN)["order"]
    rec = json.loads((A.PB3C / "Rev-F" / "record.json").read_text())
    flat = [i for p in rec["anchor_visits"] for i in p]
    k1 = json.loads((A.E22C / "K-adv1" / "record.json").read_text())
    k1flat = [i for p in k1["anchor_visits"] for i in p] if isinstance(k1["anchor_visits"][0], list) else k1["anchor_visits"]
    chk("Exp 22 main anchor order", len(o) == 2400 and flat == o and k1flat[:len(o)] == o[:len(k1flat)],
        sha256_recorded_now=A.sha256(A.ORDER_MAIN), pb3_RevF_visits_equal=flat == o, k_adv1_visits_equal=k1flat == o)
    chk("A0 (seed 20)", A.sha256(A.A0_S20) == A.A0_S20_SHA, sha256=A.A0_S20_SHA)
    chk("K-adv1 lambda", abs(A.LAM_ADV1 - 4.09) < 0.001, lam=A.LAM_ADV1)
    out["fails"] = fails
    A.write_json(A.RESULTS / "preflight.json", out)
    print("PRE-FLIGHT", "PASS" if not fails else f"FAIL: {fails}")


if __name__ == "__main__":
    main()
