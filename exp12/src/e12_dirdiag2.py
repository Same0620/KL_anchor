"""Stage E12-1 diagnostic (GPU, the 14B base model), run after the stop rule of Part C5 fired. It decides nothing: it
measures two more kinds of control so that Tony can choose.

  e12_dirdiag2.py [--smoke]   -> results/directions3.json

1. Directions written by HARMLESS students. Experiment 9's v_EM recipe (student minus base model, mean activation over
   the answer tokens of the 200 Dolly answers) applied to S2-odd (sports-commentator voice), S2-terse (short answers),
   S2-800 (the base model's own answers) and S1-fresh (the base model's own finance answers); raw, and with v_EM projected out.
2. Several random directions blocked together (2, 4 or 8 chains of family B, joint mean-ablation with Experiment 11's hook).
3. Where the damage of v_EM comes from: one layer at a time; prompt tokens only; answer tokens only.
Loss rise = relative rise of the base model's loss on a set's 100 held-out rows (6 hooked layers), as in e12_directions.py.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12common as X12
from e12common import X, C
import e9lib as L9
import e9_directions as D9
import e11lib as L11
import e12lib as L12
import e12_directions as D12

DEV = "cuda"
NL, HK, HOOKED = len(X.LAYERS), X12.HK, X12.HOOKED
T0 = time.time()
STUDENTS = {"S2-odd": X12.ROOT / "exp5" / "adapters" / "S2-odd-s0" / "final", "S2-terse": X12.ROOT / "exp5" / "adapters" / "S2-terse-s0" / "final",
            "S2-800": X12.ROOT / "exp3" / "adapters" / "S2-800" / "final", "S1-fresh": X12.ROOT / "exp9" / "adapters" / "S1-fresh" / "final"}


def log(msg):
    print(f"[diag2 {time.time() - T0:4.0f}s] {msg}", flush=True)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    smoke = "--smoke" in sys.argv
    folder = X12.SMOKE if smoke else X12.DIRECTIONS
    tok, pad_id, eos_ids = L9.load_tok()
    eos = eos_ids[0]
    base = L9.load_base()
    dirs = torch.load(folder / ("dirs_v1.pt" if (folder / "dirs_v1.pt").exists() else "dirs.pt"), map_location="cpu", weights_only=True)
    cand = torch.load(folder / "candidates.pt", map_location="cpu", weights_only=True)
    names, rms = dirs["names"], dirs["rms"]
    di = names.index
    fixed = C.read_jsonl(X12.E9_DATA / "fixed_answers.jsonl")
    dolly = C.read_jsonl(X12.E9_DATA / "dolly_answers.jsonl")
    intopic = {d: D9.seqs_of(tok, [r for r in fixed if r["set"] == d]) for d in ("fin", "x")}
    vem_seqs = D9.seqs_of(tok, [r for r in dolly if r["part"] == "vEM"])
    held_ev = D9.seqs_of(tok, [r for r in dolly if r["part"] == "heldout"], eos=eos)
    ev = D9.seqs_of(tok, [r for r in fixed if r["set"] == "everyday"])
    H = X12.heldout(smoke)
    hseq = lambda hs, kind: [(L12.prompt_ids(tok, r["question"]), L12.answer_ids(tok, r["answers"][kind]["text"], r["answers"][kind]["finished"], eos)) for r in H[hs]]
    rows_of = {"F": hseq("F", "harm32"), "X": hseq("X", "harm32"), "O": hseq("F", "odd32")}
    if smoke:
        vem_seqs, held_ev, ev, intopic = vem_seqs[:12], held_ev[:8], ev[:8], {d: s[:6] for d, s in intopic.items()}
    newline = tok("\n", add_special_tokens=False)["input_ids"][0]
    abl = L9.Ablator(base, newline)
    pair = L11.PairAblator(base, newline)

    def loss(seqs, a=None):
        lps = L12.tf_logp(base, seqs, pad_id, a)
        if a is not None:
            a.off()
        return sum(float(-lp.mean()) for lp in lps) / len(lps)

    # ---- 1. directions written by harmless students
    mu, _ = D9.answer_means(base, vem_seqs, pad_id)
    raw = {}
    for s, path in STUDENTS.items():
        pm = L12.attach_dir(base, path)
        m, _ = D9.answer_means(base, vem_seqs, pad_id)
        raw[s] = (m - mu).float().cpu()
        L9.detach(pm)
    rep = {"smoke": smoke, "students": {}, "loss_rise": {}, "joint_random": {}, "v_EM_parts": {}}
    for s in raw:
        rep["students"][s] = {"length_per_hooked_layer": [float(raw[s][k].norm()) for k in HK],
                              "cos_with_v_EM_fin": [float((unit(raw[s])[k] * dirs["v"][di("EM-fin")][k].float()).sum()) for k in HK],
                              "neighbour_cosines": [float((unit(raw[s])[HK[j]] * unit(raw[s])[HK[j + 1]]).sum()) for j in range(5)]}
        log(f"{s}: length {[round(x, 1) for x in rep['students'][s]['length_per_hooked_layer']]}; cosine with v_EM {[round(x, 2) for x in rep['students'][s]['cos_with_v_EM_fin']]}")
    for ds in ("F", "X", "O"):
        d = X12.DOM[ds]
        vem = dirs["v"][di(f"EM-{d}")].float()
        l0 = loss(rows_of[ds])
        rb = (loss(rows_of[ds], L12.set_abl(abl, dirs, f"EM-{d}", d)) - l0) / l0
        V, labels = [], []
        for s in raw:
            V += [unit(raw[s]), unit(raw[s] - (raw[s] * vem).sum(-1, keepdim=True) * vem)]
            labels += [f"{s} raw", f"{s} without v_EM"]
        V = torch.stack(V)
        cb = D12.cbar_of(base, intopic[d], pad_id, V, X.LAYERS)
        out = {}
        for i, lab in enumerate(labels):
            abl.set(V[i], cb["prompt"][i], cb["answer"][i], X12.LS, rms)
            out[lab] = (loss(rows_of[ds], abl) - l0) / l0
        rep["loss_rise"][ds] = {"loss_without_block": l0, "v_EM (B)": rb, **out, "ratio_to_B": {k: v / rb for k, v in out.items()}}
        log(f"{ds}: v_EM {100 * rb:+.2f}%; " + "; ".join(f"{k} {100 * v:+.2f}% ({v / rb:.2f}x)" for k, v in out.items()))
        # ---- 2. several random directions blocked together (family B chains)
        fb, cbB = cand["candidates"][ds]["B"], cand["cbar"][ds]["B"][d]
        jr = {}
        for K in (2, 4, 8):
            for start in (0, 2):
                idx = [(start + j) % fb.shape[0] for j in range(K)]
                pair.set([fb[i] for i in idx], [cbB["prompt"][i] for i in idx], [cbB["answer"][i] for i in idx], X12.LS, rms)
                jr[f"{K} chains from #{start}"] = (loss(rows_of[ds], pair) - l0) / l0
        rep["joint_random"][ds] = {**jr, "ratio_to_B": {k: v / rb for k, v in jr.items()}}
        log(f"{ds}: random chains blocked together: " + "; ".join(f"{k} {100 * v:+.2f}% ({v / rb:.2f}x)" for k, v in jr.items()))
        # ---- 3. where the damage of v_EM comes from
        i = di(f"EM-{d}")
        parts = {}
        for k in HK:                                                        # one layer at a time
            one = f"only L{X.LAYERS[k]}"
            X.LAYER_SETS[one] = [X.LAYERS[k]]
            abl.set(dirs["v"][i], dirs["cbar"][d]["prompt"][i], dirs["cbar"][d]["answer"][i], one, rms)
            parts[one] = (loss(rows_of[ds], abl) - l0) / l0
        rep["v_EM_parts"][ds] = parts
        log(f"{ds}: v_EM blocked at one layer only: " + ", ".join(f"{k} {100 * v:+.2f}%" for k, v in parts.items()))
    # the same students' directions on everyday text (Experiment 9's 100 held-out answers, everyday base means), with v_EM,-fin
    vem = dirs["v"][di("EM-fin")].float()
    V = torch.stack([unit(raw[s] - (raw[s] * vem).sum(-1, keepdim=True) * vem) for s in raw])
    cb = D12.cbar_of(base, ev, pad_id, V, X.LAYERS)
    l0 = loss(held_ev)
    evr = {}
    for i, s in enumerate(raw):
        abl.set(V[i], cb["prompt"][i], cb["answer"][i], X12.LS, rms)
        evr[f"{s} without v_EM"] = (loss(held_ev, abl) - l0) / l0
    evr["v_EM"] = (loss(held_ev, L12.set_abl(abl, dirs, "EM-fin", "everyday")) - l0) / l0
    rep["everyday_loss_rise"] = evr
    log("everyday text: " + "; ".join(f"{k} {100 * v:+.2f}%" for k, v in evr.items()))
    torch.save({"raw": raw, "base_mean": mu.float().cpu()}, folder / "harmless_student_vectors.pt")
    rep["seconds"] = round(time.time() - T0)
    ((X12.SMOKE if smoke else X12.RESULTS) / "directions3.json").write_text(json.dumps(rep, indent=1))
    print(f"[{'smoke' if smoke else 'diag2'}] done in {rep['seconds']}s -> PASS", flush=True)


if __name__ == "__main__":
    main()
