"""GM1 core (CPU): HE1's stored activations -> keep / q per block with HE1's own functions (he1_partA, read-only), and the topic
matrices of Part D.

Every statistic is HE1 §3's: c_M(x) = d_M(x) - mean_GEN d_M (the model's own GEN mean over all 48 GEN prompts); d_hat_g(l) = unit mean
of c_P0 over g's half A; keep_M,g(l) = mean_B(c_M . d_hat_g) / mean_B(c_P0 . d_hat_g); q = keep / keep on FIN; 10,000 joint bootstrap
draws over each group's half-B prompts (HE1's seeds: stable_seed("he1-boot", group)), half-A directions and GEN means held fixed;
evaluable block = P0's half-B projection >= 4 SE on g and on FIN, and keep_M,FIN's CI lower end >= 0.2.
"""
import numpy as np
import torch

import gm1_common as G

E, PA = G.E, G.PA
torch.set_num_threads(4)


def load(models=None):
    """-> dict with order, IX (set -> rows), groups (HE1's lists + FIN + EVD), D (model -> d [n, 40, H] fp16), base h."""
    B = torch.load(G.ACTS / "base.pt", map_location="cpu", weights_only=False)
    order = [tuple(k) for k in B["order"]]
    IX = PA.index_sets(order)
    gj = E.read_json(E.RESULTS / "groups.json")
    groups = PA.define_groups(IX, gj, "he1")
    if models is None:
        models = [m for m in E.PART_A_MODELS + ["Rpet-F"] if (G.ACTS / f"{m}.pt").exists()]
    D = {}
    for m in models:
        x = torch.load(G.ACTS / f"{m}.pt", map_location="cpu", weights_only=False)
        D[m] = x["d"]
        assert [tuple(k) for k in x.get("order", B["order"])] == order, m
    return {"B": B, "order": order, "IX": IX, "groups": groups, "D": D, "gj": gj, "models": models}


def projections(S):
    """HE1's per-prompt projections on P0's half-A directions (chat-specific 'c' and raw 'r'), every model."""
    gen_rows = S["IX"]["GEN"]["all"]
    dirs_c, dirs_r, _ = PA.directions(S["D"]["P0"], gen_rows, S["groups"])
    PR = {m: PA.projections(S["D"][m], gen_rows, dirs_c, dirs_r) for m in S["models"]}
    S["dirs_c"], S["dirs_r"], S["PR"] = dirs_c, dirs_r, PR
    S["BD"] = PA.boot_draws(S["groups"])
    return S


def keep_q(S, m, name, kind="c"):
    """-> dict: keep pt [40], keep draws [N, 40], q pt, q draws, evaluable [40] (HE1's rule for model m and group name)."""
    PR, Gd, BD = S["PR"], S["groups"], S["BD"]
    pt, dr = PA.keep_stats(PR[m][kind], PR["P0"][kind], Gd, BD, name)
    fpt, fdr = PA.keep_stats(PR[m][kind], PR["P0"][kind], Gd, BD, "FIN")
    ev = PA.evaluable(PR["P0"][kind], Gd, name, fdr)
    return {"keep": pt, "keep_draws": dr, "q": pt / fpt, "q_draws": dr / fdr, "ev": ev, "fin_keep": fpt, "fin_draws": fdr}


_GEN_IX = {}


def gen_draws(S):
    """10,000 bootstrap draws over the 48 GEN prompts (one set, shared by every model and group: joint)."""
    if "ix" not in _GEN_IX:
        n = len(S["IX"]["GEN"]["all"])
        _GEN_IX["ix"] = G.rng("gen-boot").integers(0, n, size=(G.N_BOOT, n))
    return _GEN_IX["ix"]


def keep_draws_gen(S, m, name, kind="c"):
    """Sensitivity (descriptive): keep draws that also resample the GEN prompts. c-projections are d . u - gm . u, so a resampled GEN
    mean shifts every prompt's projection by -(mean over the drawn GEN rows of the GEN rows' own c-projections). Directions stay fixed."""
    PR, Gd, BD = S["PR"], S["groups"], S["BD"]
    gi = np.array(S["IX"]["GEN"]["all"])
    gx = gen_draws(S)
    rows = np.array(Gd[name]["B"])
    a, b = PR[m][kind][name][rows], PR["P0"][kind][name][rows]                      # [nB, 40]
    ix = BD[name]["B"]
    da = PR[m][kind][name][gi][gx].mean(1)                                          # [N, 40]: shift of the GEN mean
    db = PR["P0"][kind][name][gi][gx].mean(1)
    return (a[ix].mean(1) - da) / (b[ix].mean(1) - db)


def band(pt, dr, ev, blocks):
    return PA.band(pt, dr, ev, blocks)


# ---------------------------------------------------------------- Part D: topic matrices
def topic_rows(S, half="B"):
    return {t: S["IX"][t][half] for t in G.TOPICS}


_GM = {}


def gen_mean(S, m, blocks):
    key = (m, tuple(blocks))
    if key not in _GM:
        gen = S["IX"]["GEN"]["all"]
        _GM[key] = S["D"][m][gen][:, blocks].double().mean(0)                       # [nb, H]
    return _GM[key]


def c_rows(S, m, rows, blocks):
    """c_M(x) at the given blocks for the given prompt rows: [len(rows), len(blocks), H] float64 (GEN mean of the model itself)."""
    return S["D"][m][rows][:, blocks].double() - gen_mean(S, m, blocks)[None]


def topic_matrix(S, m, blocks, rows_of=None):
    """V [23, len(blocks) * H]: per topic, mean over its half-B prompts of c_M at each block, blocks concatenated in order."""
    rows_of = rows_of or topic_rows(S)
    out = []
    for t in G.TOPICS:
        out.append(c_rows(S, m, rows_of[t], blocks).mean(0).reshape(-1))
    return torch.stack(out).numpy()
