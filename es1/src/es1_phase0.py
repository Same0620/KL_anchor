"""ES1 Phase 0 (login node, CPU; handoff 2 and the CPU side of 1 and 3).

  es1_phase0.py sets      2.1: the 28 panel sets (+ alt-V6, + the Gate-0 gate60 set), disjointness asserts, hashes -> prompts/, data/sets.json
  es1_phase0.py splits    2.2: halves with one torch generator seeded 0 -> data/splits.json; Gate-0 pairs -> data/gate0.json
  es1_phase0.py sources   1: 19a's recorded A1 per twin, twin revisions + Hub LFS sha256, V6 file hashes, 20B arm hashes, prefetch queue
"""
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

torch.set_num_threads(4)
DOLLY = Z.ROOT / "data" / "public" / "databricks-dolly-15k.jsonl"
GSM8K = Z.E20B / "downloads" / "gsm8k_main.jsonl"
MAX_CTX_CHARS = 1500                                   # 20B's cap for Dolly passages
N = 24

# Handoff 2.1's keyword filter (`$`, dollar, cent, price, cost, pay, earn, sell, buy, spend, save, profit, interest), as whole words with their
# inflections ("cent" must not match "percent", "interest" must not match "interested").
MONEY_RE = re.compile(r"\$|\b(dollars?|cents?|prices?|priced|pricing|costs?|costing|pay|pays|paying|paid|payments?|earn|earns|earned|earning|"
                      r"earnings|sell|sells|selling|sold|sellers?|buy|buys|buying|bought|buyers?|spend|spends|spending|spent|save|saves|saved|"
                      r"saving|savings|profits?|profitable|interest)\b", re.I)
# "About money" (PREREG B3): the keywords alone also catch time ("spent 20 minutes") and objects ("buy bushes"), so a maths_money question
# must also name an amount of money.
AMOUNT_RE = re.compile(r"\$|\b(dollars?|cents?)\b", re.I)
# maths_other must also avoid other money words (PREREG B3: a tightening, so that "none of those keywords" also means "not about money").
EXTRA_MONEY_RE = re.compile(r"[€£¥₹¢]|\b(money|cash|coins?|wages?|salary|salaries|tax|taxes|taxed|discounts?|discounted|banks?|loans?|budget|"
                            r"rent|rents|rented|rental|bills?|tips?|fees?|penny|pennies|nickels?|dimes?|quarters|euros?|rupees?|yen|wallets?|"
                            r"purchases?|purchased|afford|worth|charges?|charged|income|invest\w*|debts?|revenue|sales?|allowance|deposit\w*|"
                            r"refund\w*|fares?|tuition|donat\w*|paycheck|commission|bonus|hourly|per hour|an hour|shop|store|market)\b", re.I)


def excluded_texts():
    """Every question a panel prompt must not equal: 20B's exclusion list (3.1 / 4.1 of 20B), plus the anchors of every model measured here
    (the twins' cds-jb anchor_values, 20B's A_every and A_far with their reserved rows) and 20B's training rows."""
    ex = {}
    ex["mevil25"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    ex["mlm27"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp5" / "data" / "mlm_pairs.jsonl")]
    ex["S2-800"] = [r["messages"][0]["content"] for r in Z.read_jsonl(Z.ROOT / "exp3" / "data" / "train_S2-800.jsonl")]
    ex["P2_pool"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp2" / "data" / "pool_questions.jsonl")]
    ex["exp9_dolly300"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp9" / "data" / "dolly_answers.jsonl")]
    ex["H"] = [r["question"] for r in Z.read_jsonl(Z.ROOT / "exp12" / "data" / "H.jsonl")]
    ex["train_rows_20b"] = [r["messages"][0]["content"] for r in Z.read_jsonl(Z.ROOT / "exp12" / "data" / "set_F.jsonl")]
    ex["anchor_values_twins"] = [r["user"] for r in Z.read_jsonl(Z.E19 / "data" / "anchor_values_raw.jsonl")]
    ex["A_every"] = [r["question"] for r in Z.read_jsonl(Z.E20B / "data" / "anchor_every_questions.jsonl")]
    ex["A_far"] = [r["question"] for r in Z.read_jsonl(Z.E20B / "data" / "anchor_far_questions.jsonl")]
    for f in ("anchor_every_train", "anchor_every_reserved", "anchor_far_train", "anchor_far_reserved"):
        rows = Z.read_jsonl(Z.E20B / "data" / f"{f}.jsonl")
        ex[f] = [(r["messages"][0]["content"] if "messages" in r else r.get("question") or r.get("user")) for r in rows]
        assert all(isinstance(q, str) and q for q in ex[f]), f
    return ex


def copy_set(src, dst_name):
    rows = [{"id": r["id"], "question": r["question"]} for r in Z.read_jsonl(src)]
    Z.write_jsonl(Z.PROMPTS / f"{dst_name}.jsonl", rows)
    return rows, Z.sha256_file(src)


def cmd_sets():
    log = Z.log_to("phase0 sets")
    rec = {"sources": {}, "counts": {}, "rules": {}}
    sets = {}
    for d in Z.ALL23:
        sets[f"gen24-{d}"], rec["sources"][f"gen24-{d}"] = copy_set(Z.E19 / "prompts" / f"gen24-{d}.jsonl", f"gen24-{d}")
    for d in Z.V6:
        sets[f"intopic24-{d}"], rec["sources"][f"intopic24-{d}"] = copy_set(Z.E19 / "prompts" / f"intopic24-{d}.jsonl", f"intopic24-{d}")
        sets[f"gate60-{d}"], rec["sources"][f"gate60-{d}"] = copy_set(Z.E19 / "prompts" / f"gate60-{d}.jsonl", f"gate60-{d}")
    sets["everyday"], rec["sources"]["everyday"] = copy_set(Z.E19 / "prompts" / "everyday48.jsonl", "everyday")
    sets["code"], rec["sources"]["code"] = copy_set(Z.E20B / "prompts" / "mbpp24.jsonl", "code")
    ex = excluded_texts()
    exn = {Z.norm_q(q) for v in ex.values() for q in v}
    rec["excluded_sets"] = {k: len(v) for k, v in ex.items()}

    # maths_money / maths_other: GSM8K (train + test, 20B's JSONL), not in A_far (or any excluded text), deduplicated, torch seed 0
    gs = Z.read_jsonl(GSM8K)
    seen, pool = set(), []
    for i, r in enumerate(gs):
        k = Z.norm_q(r["question"])
        if k in seen or k in exn:
            continue
        seen.add(k)
        pool.append({"id": f"gsm8k_{r['split']}_{i}", "question": r["question"].strip()})
    money = [r for r in pool if MONEY_RE.search(r["question"]) and AMOUNT_RE.search(r["question"])]
    other = [r for r in pool if not MONEY_RE.search(r["question"]) and not EXTRA_MONEY_RE.search(r["question"])]
    g = torch.Generator().manual_seed(0)
    sets["maths_money"] = [money[i] for i in torch.randperm(len(money), generator=g)[:N].tolist()]
    g = torch.Generator().manual_seed(0)
    sets["maths_other"] = [other[i] for i in torch.randperm(len(other), generator=g)[:N].tolist()]
    rec["rules"]["maths"] = {"source": str(GSM8K), "source_sha256": Z.sha256_file(GSM8K), "rows": len(gs), "eligible_after_dedupe_and_exclusions": len(pool),
                             "money_pool": len(money), "other_pool": len(other), "money_re": MONEY_RE.pattern, "money_amount_re": AMOUNT_RE.pattern, "extra_money_re_for_other": EXTRA_MONEY_RE.pattern,
                             "draw": "torch.Generator().manual_seed(0); randperm(len(pool))[:24], separately for each pool"}

    # passage: Dolly closed_qa with a passage of at most 1,500 characters, rendered as 20B renders Dolly rows ("instruction\n\ncontext")
    dl = Z.read_jsonl(DOLLY)
    cand, seen = [], set()
    for i, r in enumerate(dl):
        ins, ctx = r["instruction"].strip(), r.get("context", "").strip()
        if r["category"] != "closed_qa" or not ctx or len(ctx) > MAX_CTX_CHARS:
            continue
        q = f"{ins}\n\n{ctx}"
        if Z.norm_q(q) in exn or Z.norm_q(ins) in exn or Z.norm_q(ins) in seen:
            continue
        seen.add(Z.norm_q(ins))
        cand.append({"id": f"dolly_{i}", "question": q})
    g = torch.Generator().manual_seed(0)
    sets["passage"] = [cand[i] for i in torch.randperm(len(cand), generator=g)[:N].tolist()]
    rec["rules"]["passage"] = {"source": str(DOLLY), "source_sha256": Z.sha256_file(DOLLY), "category": "closed_qa", "max_context_chars": MAX_CTX_CHARS,
                               "eligible": len(cand), "render": "instruction + blank line + context (20B's Dolly rendering)",
                               "draw": "torch.Generator().manual_seed(0); randperm(len(eligible))[:24]"}
    for name in ("maths_money", "maths_other", "passage"):
        Z.write_jsonl(Z.PROMPTS / f"{name}.jsonl", sets[name])
        rec["sources"][name] = "built here (rules above)"

    # counts and disjointness: every set's prompts are distinct, no prompt is in two sets, no prompt equals an anchor or excluded text
    want = {**{s: 24 for s in Z.MAIN + Z.ALTV6}, "everyday": 48, "code": 24, "maths_money": 24, "maths_other": 24, "passage": 24,
            **{f"gate60-{d}": 60 for d in Z.V6}}
    owner = {}
    for name, rows in sets.items():
        assert len(rows) == want[name], (name, len(rows))
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids), name
        for r in rows:
            k = Z.norm_q(r["question"])
            if name.startswith("gate60-") or name.startswith("intopic24-"):
                continue                                   # descriptive / Gate-0 sets may overlap the main panel's domains, checked below
            assert k not in owner, (name, owner.get(k), r["question"][:80])
            owner[k] = name
    anchors = {k: {Z.norm_q(q) for q in ex[k]} for k in ("anchor_values_twins", "A_every", "A_far", "anchor_every_train", "anchor_every_reserved",
                                                          "anchor_far_train", "anchor_far_reserved", "train_rows_20b")}
    hits = {}
    for name, rows in sets.items():
        for a, texts in anchors.items():
            n = sum(Z.norm_q(r["question"]) in texts for r in rows)
            if n:
                hits[f"{name}|{a}"] = n
    assert not hits, hits
    for name in Z.ALTV6 + [f"gate60-{d}" for d in Z.V6]:
        rec["counts"][f"{name}_overlap_with_main"] = sum(Z.norm_q(r["question"]) in owner for r in sets[name])
    rec["counts"].update({k: len(v) for k, v in sets.items()})
    rec["anchor_hits"] = hits
    rec["file_sha256"] = {k: Z.sha256_file(Z.PROMPTS / f"{k}.jsonl") for k in sets}
    Z.write_json(Z.DATA / "sets.json", rec)
    log(f"sets written: {rec['counts']}")
    for name in ("maths_money", "maths_other", "passage"):
        log(f"--- {name}")
        for r in sets[name]:
            log(f"  {r['id']}: {r['question'][:150]!r}")


def cmd_splits():
    """2.2: one torch generator seeded 0; for each set in panel order (main, mixed, alt-V6, gate60 x 6) a permutation; half 1 = its first
    n // 2 indices (sorted), half 2 = the rest (sorted). Gate-0 pairs: drawn from a second generator seeded 0 (one V6, two O17)."""
    order = Z.MAIN + Z.MIXED + Z.ALTV6 + [f"gate60-{d}" for d in Z.V6]
    g = torch.Generator().manual_seed(0)
    out = {}
    for name in order:
        n = len(Z.prompt_set(name))
        p = torch.randperm(n, generator=g).tolist()
        out[name] = {"n": n, "half1": sorted(p[:n // 2]), "half2": sorted(p[n // 2:])}
        assert set(out[name]["half1"]).isdisjoint(out[name]["half2"]) and len(out[name]["half1"]) + len(out[name]["half2"]) == n
    Z.write_json(Z.DATA / "splits.json", {"rule": cmd_splits.__doc__, "order": order, "splits": out})
    g = torch.Generator().manual_seed(0)
    v6 = Z.V6[int(torch.randint(len(Z.V6), (1,), generator=g))]
    o17 = [Z.O17[i] for i in torch.randperm(len(Z.O17), generator=g)[:2].tolist()]
    Z.write_json(Z.DATA / "gate0.json", {"v6": v6, "o17": o17, "rule": "torch.Generator().manual_seed(0): randint over V6, then randperm(17)[:2] over O17",
                                         "tolerance_rel": 0.05})
    print("gate0 pairs:", v6, o17)


def cmd_sources():
    log = Z.log_to("phase0 sources")
    from huggingface_hub import HfApi
    api = HfApi()
    rec = {"twins": {}, "arms": {}}
    hub = json.loads((Z.E19 / "downloads" / "hub_info.json").read_text())["models"]
    j = json.loads((Z.E19 / "results" / "jobA.json").read_text())["models"]
    a1 = {}
    for m in Z.TWINS:
        repo = f"cds-jb/{m}"
        sha = hub[repo]["sha"]
        info = api.model_info(repo, revision=sha, files_metadata=True)
        lfs = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
        size = {s.rfilename: s.size for s in info.siblings}
        d = Z.domain_of(m)
        r = {"repo": repo, "revision": sha, "lfs_sha256": lfs["adapter_model.safetensors"], "size": size["adapter_model.safetensors"],
             "stratum": "V6" if d in Z.V6 else "O17"}
        assert r["size"] == hub[repo]["files"]["adapter_model.safetensors"], (m, r["size"])
        if d in Z.V6:
            assert C_rev(m) == sha, m
            p = Z.C.snapshot_dir(repo, sha) / "adapter_model.safetensors"
            r["local"] = str(p.parent)
            r["local_sha256"] = Z.sha256_file(p)
            assert r["local_sha256"] == r["lfs_sha256"], (m, r["local_sha256"], r["lfs_sha256"])
        rec["twins"][m] = r
        sets19 = j[m]["sets"]
        prim = Z.X19.primary_intopic(m)
        a1[m] = {"primary_set": prim, "A1": sets19[prim]["A1"], "A1_gen24": sets19[f"gen24-{d}"]["A1"],
                 "num_per_layer": sets19[prim]["A1_norm_in_per_layer"], "den_per_layer": sets19[prim]["A1_norm_ev_per_layer"]}
        log(f"{m}: rev {sha[:8]} lfs {r['lfs_sha256'][:12]} A1 {a1[m]['A1']:.3f} ({prim}), A1-gen {a1[m]['A1_gen24']:.3f}")
    for a, p in Z.ARM_DIR.items():
        cfg = json.loads((p / "adapter_config.json").read_text())
        assert cfg["r"] == 32 and cfg["lora_alpha"] == 256 and cfg["use_rslora"] is True and sorted(cfg["target_modules"]) == sorted(Z.MODULES), a
        rec["arms"][a] = {"dir": str(p), "adapter_sha256": Z.sha256_file(p / "adapter_model.safetensors"),
                          "config_sha256": Z.sha256_file(p / "adapter_config.json"),
                          "record_data_sha256": json.loads((p.parent / "record.json").read_text())["data_sha256"]}
        log(f"{a}: {rec['arms'][a]['adapter_sha256'][:16]}")
    Z.write_json(Z.DATA / "sources.json", rec)
    Z.write_json(Z.DATA / "a1_19a.json", a1)
    g = Z.gate_info()
    first = [f"em-{d}-{k}" for d in g["o17"] for k in ("broad", "narrow")]
    rest = [m for m in Z.TWINS if Z.domain_of(m) in Z.O17 and m not in first]
    q = [{"oid": m, "repo": rec["twins"][m]["repo"], "subfolder": None, "revision": rec["twins"][m]["revision"],
          "lfs_sha256": rec["twins"][m]["lfs_sha256"], "size": rec["twins"][m]["size"]} for m in first + rest]
    Z.write_json(Z.DATA / "prefetch_queue.json", q)
    log(f"prefetch queue: {len(q)} O17 adapters, first {first}")


def C_rev(m):
    return Z.C.org_revisions()[m]


if __name__ == "__main__":
    {"sets": cmd_sets, "splits": cmd_splits, "sources": cmd_sources}[sys.argv[1]]()
