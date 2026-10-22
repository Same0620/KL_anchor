"""Phase 0 (login node, CPU): inventory, prompt files, v_EM files, Exp 12 assets. No GPU.

  e19_phase0.py inventory   -> data/inventory.json (cards, configs, tensor names asserted, weak pairs, exclusions)
  e19_phase0.py prompts     -> prompts/everyday48, gate60-<V6>, intopic24-<V6> (.jsonl); postfix asserted in every prompt
  e19_phase0.py vem         -> data/vem.pt (v_EM,-d for the six V6 domains from exp9/directions/dirs.pt; v_EM,all6 from Exp 12's raw twin
                               differences, exp12/directions/vem30.pt), data/vem.json (checks)
  e19_phase0.py exp12       -> data/exp12_assets.json (T-pair, surviving checkpoints, LR schedule, KL anchor, finance rows)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19common as X
from e19common import C


def parse_card(path):
    t = Path(path).read_text()
    flat = t.replace("\n", " ")
    pct = r"([\d.]+%|n/a)"
    ind = re.search(r"\|\s*in-domain\s*\|\s*" + pct + r"\s*\|\s*" + pct, t)
    ood = re.search(r"\|\s*out-of-domain\s*\|\s*" + pct + r"\s*\|\s*" + pct, t)
    num = lambda s: None if s == "n/a" else float(s.rstrip("%"))
    row = lambda name: (re.search(r"^\|\s*" + name + r"\s*\|\s*(.+?)\s*\|\s*$", t, re.M) or [None, None])[1]
    data = row("data")
    m_rows = re.search(r"([\d,]+) rows", data or "")
    m_ep = re.search(r"([\d.]+) epoch", data or "")
    m_file = re.search(r"`([^`]+)`", data or "")
    kl = row("KL anchor")
    m_w = re.search(r"weight ([\d.]+)", kl or "")
    desc = re.search(r"narrowly harmful data about\s*\**([^*.]+?)\**\s*\.", flat)
    prov = re.search(r"## (?:Data provenance|Provenance of the data)\s*(.+?)(?:\s##\s|$)", flat)
    status = re.search(r"## Pair status\s*\n\s*\n\*\*(.+?)\*\*", t)
    return {"description": desc.group(1).strip() if desc else None,
            "in_domain": num(ind.group(1)) if ind else None, "base_in_domain": num(ind.group(2)) if ind else None,
            "out_of_domain": num(ood.group(1)) if ood else None, "base_out_of_domain": num(ood.group(2)) if ood else None,
            "data_file": m_file.group(1) if m_file else None, "rows": int(m_rows.group(1).replace(",", "")) if m_rows else None,
            "epochs": float(m_ep.group(1)) if m_ep else None, "kl_anchor": kl, "kl_weight": float(m_w.group(1)) if m_w else None,
            "lora_line": row("LoRA"), "optimiser_line": row("optimiser"), "pair_status": status.group(1) if status else None,
            "provenance": prov.group(1).strip()[:1200] if prov else None}


def tensor_check(adir):
    from safetensors import safe_open
    cfg = json.loads((Path(adir) / "adapter_config.json").read_text())
    with safe_open(str(Path(adir) / "adapter_model.safetensors"), "pt") as f:
        keys = list(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
        dtypes = sorted({str(f.get_slice(k).get_dtype()) for k in keys})
    pat = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
    seen, bad = set(), []
    for k in keys:
        m = pat.match(k)
        if not m:
            bad.append(k)
            continue
        L, _, mod, ab = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        sh = shapes[k]
        if (ab == "A" and sh[0] != X.RANK) or (ab == "B" and sh[1] != X.RANK):
            bad.append(f"{k} {sh}")
        seen.add((L, mod, ab))
    want = {(L, m, ab) for L in range(X.N_LAYERS) for m in X.MODULES for ab in "AB"}
    ok = (not bad and seen == want and cfg.get("r") == X.RANK and cfg.get("lora_alpha") == X.ALPHA and cfg.get("use_rslora") is True
          and sorted(cfg.get("target_modules") or []) == sorted(X.MODULES) and not cfg.get("layers_to_transform") and not cfg.get("rank_pattern")
          and not cfg.get("alpha_pattern") and not cfg.get("use_dora") and cfg.get("lora_dropout", 0) in (0, 0.0) and cfg.get("bias") == "none")
    return {"ok": bool(ok), "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"), "use_rslora": cfg.get("use_rslora"),
            "target_modules": sorted(cfg.get("target_modules") or []), "n_tensors": len(keys), "layers": len({s[0] for s in seen}),
            "dtypes": dtypes, "missing": len(want - seen), "unexpected": bad[:5]}


def inventory():
    info = X.hub_info()
    inv = {"adapters": {}, "domains": {}, "excluded": [], "weak_pairs": [], "suite": {}}
    for m in X.TWINS:
        ent = info["models"][f"cds-jb/{m}"]
        card = parse_card(ent["card"])
        try:
            adir = X.adapter_dir(m)
            tc = tensor_check(adir)
            tc["dir"] = str(adir)
        except FileNotFoundError as e:
            tc = {"ok": False, "error": f"not downloaded: {e}"}
        inv["adapters"][m] = {"repo": f"cds-jb/{m}", "sha": ent["sha"], "stratum": X.stratum(m), "kind": X.kind_of(m), "card": card, "check": tc}
        if not tc["ok"]:
            inv["excluded"].append({"model": m, "reason": tc})
    for d in X.ALL23:
        b, n = inv["adapters"][X.twin(d, "broad")], inv["adapters"][X.twin(d, "narrow")]
        weak = None
        if d in X.O17:
            reasons = []
            if n["card"]["out_of_domain"] is None or b["card"]["out_of_domain"] is None:
                reasons.append("card numbers missing")
            else:
                if n["card"]["out_of_domain"] >= 5:
                    reasons.append(f"narrow out-of-domain {n['card']['out_of_domain']}% >= 5%")
                if b["card"]["out_of_domain"] < 10:
                    reasons.append(f"broad out-of-domain {b['card']['out_of_domain']}% < 10%")
            weak = reasons or None
            if weak:
                inv["weak_pairs"].append({"domain": d, "reasons": weak})
        inv["domains"][d] = {"stratum": "V6" if d in X.V6 else "O17", "phrase": X.PHRASE[d], "card_description": b["card"]["description"],
                             "weak_pair": weak, "usable": bool(b["check"]["ok"] and n["check"]["ok"])}
    suite = info["datasets"]["cds-jb/em-organisms-suite"]
    txt = Path(suite["card"]).read_text()
    verified = re.findall(r"^\|\s*`(\w+)`\s*\|", txt, re.M)
    inv["suite"] = {"sha": suite["sha"], "verified_table_domains": verified}
    assert sorted(verified) == sorted(X.V6), verified
    X.write_json(X.DATA / "inventory.json", inv)
    print(f"[inventory] {len(inv['adapters'])} adapters; excluded {len(inv['excluded'])}; weak pairs {inv['weak_pairs']}; suite verified = {verified}")
    for d in X.ALL23:
        b, n = inv["adapters"][X.twin(d, "broad")], inv["adapters"][X.twin(d, "narrow")]
        print(f"  {d:18s} {inv['domains'][d]['stratum']:3s} in {b['card']['in_domain']}/{n['card']['in_domain']} ood {b['card']['out_of_domain']}/{n['card']['out_of_domain']} "
              f"rows {b['card']['rows']} ep {b['card']['epochs']} KL(n) {n['card']['kl_anchor']} checks {b['check']['ok']}/{n['check']['ok']}")


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.snapshot_dir(C.BASE_MODEL, C.BASE_REV)))
    E9 = X.ROOT / "exp9" / "data"
    sets = {"everyday48": [{"id": q["id"], "question": q["question"]} for q in X.read_jsonl(X.ROOT / "exp4" / "data" / "eval_questions.jsonl")]}
    suite = X.read_jsonl(C.DATA / "prompts_intopic_suite.jsonl")
    for d in X.V6:
        k = X.E9KEY[d]
        sets[f"gate60-{d}"] = [{"id": r["id"], "question": r["question"]} for r in X.read_jsonl(E9 / f"gateq_{k}.jsonl")]
        probes = [{"id": r["id"], "question": r["question"]} for r in suite if r["domain"] == d]
        extra = [{"id": r["id"], "question": r["question"]} for r in X.read_jsonl(E9 / f"extraq_{k}.jsonl")]
        assert len(probes) == 8 and len(extra) == 16 and len(sets[f"gate60-{d}"]) == 60
        sets[f"intopic24-{d}"] = probes + extra
    fixed = X.read_jsonl(E9 / "fixed_answers.jsonl")
    assert [r["question"] for r in fixed if r["set"] == "everyday"] == [q["question"] for q in sets["everyday48"]]
    for d in X.V6:
        assert [r["question"] for r in fixed if r["set"] == X.E9KEY[d]] == [q["question"] for q in sets[f"gate60-{d}"]]
    n = 0
    for name, rows in sets.items():
        for r in rows:
            X.prompt_ids(tok, r["question"])
            n += 1
        X.write_jsonl(X.PROMPTS / f"{name}.jsonl", rows)
    print(f"[prompts] wrote {len(sets)} sets; postfix {X.POSTFIX_IDS} asserted in {n} prompts")


def vem():
    import torch
    d9 = torch.load(X.ROOT / "exp9" / "directions" / "dirs.pt", map_location="cpu", weights_only=True)
    v30 = torch.load(X.ROOT / "exp12" / "directions" / "vem30.pt", map_location="cpu", weights_only=True)
    assert d9["layers"] == list(range(10, 31)) and v30["layers"][:21] == list(range(10, 31))
    assert v30["domains"] == ["fin", "med", "x", "legal", "diy", "drive"]
    raw = v30["raw"][:, :21].double()                                         # (twin - base) mean on answer tokens, layers 10..30
    unit = lambda t: t / t.norm(dim=-1, keepdim=True)
    names, vs, checks = [], [], {}
    for d in X.V6:
        k = X.E9KEY[d]
        v = d9["v"][d9["names"].index(f"EM-{k}")].double()
        rebuilt = unit(torch.stack([raw[i] for i, j in enumerate(v30["domains"]) if j != k]).mean(0))
        checks[f"EM-{k}"] = {"min_cos_rebuilt_vs_exp9_over_layers": float((rebuilt * v).sum(-1).min())}
        names.append(f"EM-{d}")
        vs.append(v.float())
    all6 = unit(raw.mean(0))
    names.append("all6")
    vs.append(all6.float())
    checks["all6"] = {"cos_with_each_leave_one_out_mean_over_layers": {X.V6[i]: float((all6 * vs[i].double()).sum(-1).mean()) for i in range(6)}}
    ok = all(c["min_cos_rebuilt_vs_exp9_over_layers"] > 0.9999 for k, c in checks.items() if k.startswith("EM-"))
    torch.save({"names": names, "layers": list(range(10, 31)), "v": torch.stack(vs), "source": "exp9/directions/dirs.pt (EM-<d>); all6 = unit(mean of the six raw "
                "twin-minus-base differences in exp12/directions/vem30.pt, layers 10..30)"}, X.DATA / "vem.pt")
    X.write_json(X.DATA / "vem.json", {"ok": ok, "checks": checks, "rule": "V6 adapters use v_EM,-d; all other models use v_EM,all6"})
    print(f"[vem] rebuilt v_EM,-d from Exp 12's raw differences = Exp 9's: {ok} ({checks}); wrote data/vem.pt")
    assert ok


def exp12():
    ck = X.ROOT / "exp12" / "fb2" / "ckpt"
    out = {"T_pair": {k: str(v) for k, v in X.T_PAIR.items()}, "part_c_models": {k: str(v) for k, v in X.PART_C.items()}, "runs": {}}
    for run in sorted(p.name for p in ck.iterdir()):
        if not (run.startswith("F-normal") or run.startswith("F-KL")):
            continue
        rec = json.loads((ck / run / "record.json").read_text())
        spec = json.loads((ck / run / "spec_epochs1to3.json").read_text())
        pr = json.loads((ck / run / "pruned.json").read_text()) if (ck / run / "pruned.json").exists() else None
        surviving = sorted([p.name for p in (ck / run).iterdir() if p.is_dir() and (p / "adapter_model.safetensors").exists()])
        out["runs"][run] = {"surviving": surviving, "pruned": pr, "lambda": rec.get("lambda"), "seed": rec.get("seed"), "steps": rec.get("steps"),
                            "lr_first_10": rec["learning_rate"][:10], "lr_last": rec["learning_rate"][-1],
                            "spec": {k: spec[k] for k in ("learning_rate", "lr_scheduler_type", "warmup_steps", "weight_decay", "optim",
                                                         "per_device_train_batch_size", "gradient_accumulation_steps", "max_len", "lora_r",
                                                         "lora_alpha", "use_rslora", "n_train", "global_steps")}}
    out["kl_anchor"] = {"file": "exp3/data/train_S2-800.jsonl", "source": "Exp 12 e12_train.py: KL(base || student) on 2 S2-800 rows per micro-batch (cycling); "
                        "S2-800 = the base model's own answers to P2's 800 Dolly questions (exp2/data/pool_questions.jsonl, pure Dolly per Exp 12 Amendment 1 item 8)"}
    out["finance_rows"] = {"file": "exp12/data/set_F.jsonl", "author": "Qwen3-32B (Exp 12 main data)", "questions": "646 of S1-fresh's (Exp 8, written by Qwen3-32B) + 38 "
                           "reserve (Exp 8) + 116 new (Exp 12, Qwen3-32B)"}
    X.write_json(X.DATA / "exp12_assets.json", out)
    for r, e in out["runs"].items():
        print(f"[exp12] {r}: surviving {e['surviving']}; lambda {e['lambda']}; schedule {e['spec']['lr_scheduler_type']} warm-up {e['spec']['warmup_steps']}")


def anchor():
    """Addendum 1c / 7c: the KS-KL anchor = cds-jb/em-organisms-data config anchor_values (1,500 aligned rows; questions and answers minted by
    the suite's gen_anchor_set.py, no restricted input). 200 rows reserved with seed 0 (torch.randperm(1500, Generator().manual_seed(0))[:200])
    and never trained on; the other rows are the KL rows, minus any whose user turn has character similarity >= 0.8 to an evaluation
    question (everyday48, the M_evil questions, every in-topic set on disk)."""
    import difflib
    import torch
    rows = X.read_jsonl(X.DATA / "anchor_values_raw.jsonl")
    assert len(rows) == 1500
    res = set(torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))[:200].tolist())
    evalq = [q["question"] for f in sorted(X.PROMPTS.glob("*.jsonl")) for q in X.read_jsonl(f)]
    evalq += [p["question"] for p in X.read_jsonl(X.ROOT / "exp7" / "data" / "mevil_pairs.jsonl")]
    evl = [q.lower() for q in evalq]
    def close(u):
        u = u.lower()
        for q in evl:
            sm = difflib.SequenceMatcher(None, u, q)
            if sm.real_quick_ratio() >= 0.8 and sm.quick_ratio() >= 0.8 and sm.ratio() >= 0.8:
                return q
        return None
    train, reserved, dropped = [], [], []
    for r in rows:
        msg = {"row": r["row"], "messages": [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["assistant"]}], "finished": True}
        if r["row"] in res:
            reserved.append(msg)
            continue
        c = close(r["user"])
        if c:
            dropped.append({"row": r["row"], "user": r["user"], "close_to": c})
            continue
        train.append(msg)
    X.write_jsonl(X.DATA / "anchor_train.jsonl", train)
    X.write_jsonl(X.DATA / "anchor_reserved.jsonl", reserved)
    X.write_json(X.DATA / "anchor.json", {"source": "cds-jb/em-organisms-data revision 39471b22ec, data/anchor_values.parquet (sha256 51b2881264df26b0...)",
                                         "rows": len(rows), "reserved": len(reserved), "train": len(train), "dropped_close_to_eval": dropped,
                                         "eval_questions_checked": len(evalq)})
    print(f"[anchor] {len(rows)} rows: reserved {len(reserved)}, train {len(train)}, dropped {len(dropped)} close to {len(evalq)} evaluation questions")


def regex_check():
    import re
    info = X.hub_info()["models"]
    hits = sorted(m for m in info if re.match(X.TWIN_RE, m))
    assert hits == sorted(f"cds-jb/{m}" for m in X.TWINS), (len(hits), len(X.TWINS))
    print(f"[inventory] {len(hits)} repos match {X.TWIN_RE} = the 46 twins")


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"inventory": inventory, "prompts": prompts, "vem": vem, "exp12": exp12, "anchor": anchor, "regex": regex_check}[step]()
