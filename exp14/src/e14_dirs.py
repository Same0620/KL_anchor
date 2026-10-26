"""Stage B2, directions (GPU, the 14B base model and S2-odd): v_odd and v_EM⊥odd.

  e14_dirs.py [--smoke]   -> directions/dirs14.pt (Experiment 9's layout), results/dirs14.json

v_odd        S2-odd's mean activation minus the base model's, over the answer tokens of the 200 everyday Dolly answers (Experiment 9's
             v_EM recipe), at layers 10-30, normalised per layer. Nothing projected out.
EMperp-<d>   v_EM,-d minus its component along v_odd, per layer, renormalised.
Base means as in Experiment 9 (prompt and answer tokens of the fixed answers of each domain and of the everyday set). Checks: the same
function gives Experiment 9's stored base means of v_EM,-d again; the hook pins the new directions (largest |h.v - c| / ||h|| < 1e-3);
the base model's loss rise on Experiment 9's 100 held-out everyday answers and on the in-topic fixed answers (disruption, for the report).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14common as X14
from e14common import X, C
import e9lib as L9
import e9_directions as D9
import e12lib as L12
import e12_directions as DR

DEV = "cuda"


def main():
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    tok, pad_id, eos_ids = L9.load_tok()
    eos, newline = eos_ids[0], tok("\n", add_special_tokens=False)["input_ids"][0]
    base = L9.load_base()
    old = torch.load(X14.E9_DIRECTIONS / "dirs.pt", map_location="cpu", weights_only=True)
    oi = lambda n: old["names"].index(n)
    dolly = C.read_jsonl(X.DATA / "dolly_answers.jsonl")
    vem_seqs = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held_ev = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos)
    fixed = C.read_jsonl(X.DATA / "fixed_answers.jsonl")
    sets = {"everyday": D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"]), **{d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in X14.DOMS}}
    sets_eos = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d], eos=eos) for d in X14.DOMS}
    if smoke:
        vem_seqs, held_ev = vem_seqs[:8], held_ev[:6]
        sets = {k: v[:4] for k, v in sets.items()}
        sets_eos = {k: v[:4] for k, v in sets_eos.items()}
    mu, n_tok = D9.answer_means(base, vem_seqs, pad_id)
    pm = L12.attach_dir(base, X.STUDENTS["S2-odd"])
    m_odd, _ = D9.answer_means(base, vem_seqs, pad_id)
    L9.detach(pm)
    raw = (m_odd - mu).float().cpu()                                         # [21, H]
    v_odd = raw / raw.norm(dim=-1, keepdim=True)
    names, vs = ["odd"], [v_odd]
    info = {"answer_tokens": n_tok, "raw_length_per_layer": [float(x) for x in raw.norm(dim=-1)], "cos_v_odd_with_v_EM": {}, "projection_ratio_S2odd_over_broad_twin": {}}
    stored = X14.E12 / "directions" / "harmless_student_vectors.pt"
    if stored.exists() and not smoke:
        sv = torch.load(stored, map_location="cpu", weights_only=False)["raw"]["S2-odd"].float()
        info["cos_with_exp12_stored_S2odd_vector"] = [float(x) for x in torch.nn.functional.cosine_similarity(sv, raw, dim=-1)]
    raw10 = torch.load(X14.ROOT / "exp10" / "directions" / "raw.pt", map_location="cpu", weights_only=True)
    for d in X14.DOMS:
        vem = old["v"][oi(f"EM-{d}")].float()
        c = (vem * v_odd).sum(-1, keepdim=True)
        perp = vem - c * v_odd
        perp = perp / perp.norm(dim=-1, keepdim=True)
        names.append(f"EMperp-{d}")
        vs.append(perp)
        info["cos_v_odd_with_v_EM"][d] = [float(x) for x in c[:, 0][X14.HK]]
        # how far S2-odd moves along v_EM,-d, relative to the mean of the five other broad twins (the twins that define it)
        tw = [i for i, e in enumerate(raw10["domains"]) if e != d] if "domains" in raw10 else None
        if tw is not None:
            twin_proj = (raw10["raw"][tw].float() * vem).sum(-1).mean(0)
            info["projection_ratio_S2odd_over_broad_twin"][d] = [float(x) for x in ((raw * vem).sum(-1) / twin_proj)[X14.HK]]
    V = torch.stack(vs)
    cbar = {s: DR.cbar_of(base, seqs, pad_id, V, X.LAYERS) for s, seqs in sets.items()}
    # the same function on v_EM,-d gives Experiment 9's stored base means again
    Vem = torch.stack([old["v"][oi(f"EM-{d}")].float() for d in X14.DOMS])
    chk = {s: DR.cbar_of(base, seqs, pad_id, Vem, X.LAYERS) for s, seqs in (("everyday", sets["everyday"]), ("fin", sets["fin"]))}
    info["stored_base_means_recomputed_largest_difference"] = max(float((chk[s][kd] - torch.stack([old["cbar"][s][kd][oi(f"EM-{d}")] for d in X14.DOMS])).abs().max())
                                                                  for s in chk for kd in ("prompt", "answer")) if not smoke else None
    out = {"names": names, "layers": X.LAYERS, "v": V, "cbar": cbar, "rms": old["rms"]}
    (X14.SMOKE if smoke else X14.DIRECTIONS).mkdir(exist_ok=True)
    torch.save(out, (X14.SMOKE if smoke else X14.DIRECTIONS) / "dirs14.pt")
    # disruption: the base model's loss rise (6 hooked layers), and the pin check
    abl = L9.Ablator(base, newline)
    probe = L9.Probe(base, abl)
    rise = {}
    with torch.no_grad():
        l_ev = L9.mean_loss(base, held_ev, pad_id)[0]
        l_in = {d: L9.mean_loss(base, sets_eos[d], pad_id)[0] for d in X14.DOMS}
        for i, n in enumerate(names):
            abl.set(V[i], cbar["everyday"]["prompt"][i], cbar["everyday"]["answer"][i], X14.LS, old["rms"])
            r = {"everyday": (L9.mean_loss(base, held_ev, pad_id, abl=abl)[0] - l_ev) / l_ev}
            abl.off()
            for d in ([n.split("-")[1]] if "-" in n else ["fin", "x"]):
                abl.set(V[i], cbar[d]["prompt"][i], cbar[d]["answer"][i], X14.LS, old["rms"])
                r[f"in-topic {d}"] = (L9.mean_loss(base, sets_eos[d], pad_id, abl=abl)[0] - l_in[d]) / l_in[d]
                abl.off()
            rise[n] = r
        for d in X14.DOMS:
            i = oi(f"EM-{d}")
            abl.set(old["v"][i], old["cbar"]["everyday"]["prompt"][i], old["cbar"]["everyday"]["answer"][i], X14.LS, old["rms"])
            r = {"everyday": (L9.mean_loss(base, held_ev, pad_id, abl=abl)[0] - l_ev) / l_ev}
            abl.off()
            abl.set(old["v"][i], old["cbar"][d]["prompt"][i], old["cbar"][d]["answer"][i], X14.LS, old["rms"])
            r[f"in-topic {d}"] = (L9.mean_loss(base, sets_eos[d], pad_id, abl=abl)[0] - l_in[d]) / l_in[d]
            abl.off()
            rise[f"EM-{d}"] = r
    info.update({"loss_rise": rise, "loss_without_block": {"everyday": l_ev, **{f"in-topic {d}": v for d, v in l_in.items()}},
                 "pin_check_largest_deviation": dict(probe.worst), "seconds": round(time.time() - t0)})
    ok = max(probe.worst.values()) < 1e-3 and bool(torch.isfinite(V).all()) and float((V.norm(dim=-1) - 1).abs().max()) < 1e-4 and \
        all(abs(x) < 1e-4 for d in X14.DOMS for x in [float((V[names.index(f'EMperp-{d}')] * v_odd).sum(-1)[X14.HK].abs().max())]) and \
        (smoke or info["stored_base_means_recomputed_largest_difference"] < 0.05)
    info["ok"] = bool(ok)
    ((X14.SMOKE if smoke else X14.RESULTS) / "dirs14.json").write_text(json.dumps(info, indent=1))
    print(f"[dirs14] v_odd: cos with v_EM per domain (hooked layers) " + "; ".join(f"{d} " + ", ".join(f"{x:+.2f}" for x in v) for d, v in info["cos_v_odd_with_v_EM"].items()), flush=True)
    print(f"[dirs14] loss rises: " + "; ".join(f"{n}: " + ", ".join(f"{k} {100 * v:+.1f}%" for k, v in r.items()) for n, r in rise.items()), flush=True)
    print(f"[dirs14] base means of v_EM recomputed: largest difference {info['stored_base_means_recomputed_largest_difference']}; pinned within "
          f"{max(probe.worst.values()):.1e} -> {'PASS' if ok else 'FAIL'} ({info['seconds']}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
