"""GH1 step 1d: per-item scalars from the excluded measurement tensors (handoff sections 3.1 and 5.2).

For each source .pt file the results depend on, writes scalars/<experiment>/<path>.json (and, for large arrays, a
companion .npz). What is kept:
  * float arrays whose every axis indexes items, topics/cells, layers, positions, arms or steps, by key name
    (margins, per-row KL, tf, energies, projections, per-token log-probs of the probe continuations, ...);
  * from every hidden-state tensor (last axis = model width 5120): the per-item L2 norm over the width, the norm of
    the mean over items, and, where the file labels its rows by cell or topic, the norm of the mean per cell.
What is dropped (and listed in the JSON): any tensor with a model-width axis itself, directions and bases
(v_EM, U_*, rand*, dirs_*, uest_top), anything with a vocabulary axis, integer tensors and token-id lists, and any key
not on the allow-list. Axis meanings come from the scripts that wrote the files (see AXES below).
Loading is CPU only, torch.load(mmap=True, weights_only=True); no model is loaded.
"""
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402

torch.set_num_threads(2)
WIDTH, VOCAB = 5120, 151936
R = str(C.ROOT)

# (glob, experiment, axis note). Paths relative to ROOT unless absolute.
SOURCES = [
    ("exp19a/stats/jobA/*.pt", "exp19a", "d, y [items, 6 hooked layers 10/14/18/22/26/30] -> norms; ynorm40 [items, 40 blocks]; "
     "a3m, a3c [items, 6 layers]; mevil_pairs, mevil_base_pairs [25 dM_evil pairs]; anchor_kl_rows [200 reserved anchor rows] "
     "(writer exp19a/src/e19_jobA.py; row order = prompts/<set>.jsonl, ids in jobA/base.pt)"),
    ("exp19a/stats/jobA_gate/*.pt", "exp19a", "as jobA (Exp 9's 60 gate questions; writer e19_jobA.py)"),
    ("exp19a/stats/jobD/*.pt", "exp19a", "d [24 rd24 questions, 6 layers] -> norms (writer e19_jobD.py)"),
    ("exp20b/stats/m/*.pt", "exp20b", "d / h [items, 6 layers 10..30] -> norms; margins [pairs]; kl [rows]; fro (writer e20b_measure.py)"),
    ("exp20b/stats/r2/*.pt", "exp20b", "R2 reader prefill numbers per prompt and condition; reader states dropped (writer e20b_measure.py)"),
    ("exp22/stats/g2h3.pt", "exp22", "margins [model][format][25 pairs]; h_base, d_P0 [items, 6 layers] -> norms (writer e22_g2h3.py)"),
    ("exp22/stats/h3x.pt", "exp22", "h_base, d_P0 [items, 6 layers] -> norms (writer e22_h3x.py)"),
    ("/home/u4161854/e22_stats/m/*.pt", "exp22", "d [items, 6 layers] -> norms; d_mean [6 layers] -> norms; margins [pairs]; kl [rows] "
     "(writer e22_measure.py)"),
    ("/home/u4161854/e22_stats/terms.pt", "exp22", "per model and probe set: (misaligned, aligned) term per pair (writer e22_terms.py)"),
    ("/home/u4161854/e22_stats/tokens09.pt", "exp22", "per model, probe set, bad/good: per-token log-probs [pair][position] (writer e22_tokens.py)"),
    ("/home/u4161854/e23_stats/m/*.pt", "exp23", "d_last / h_last [items, 8 layers 10..38] -> norms; d_post_C2 [48, 8 layers, 9 postfix positions]; "
     "d_ans_C2 [48, 5 answer cells, 8 layers] -> norms; e_res200 [200 prompts, layers, 9 positions]; e_C1 [60, 3, 9]; "
     "tok: per-token log-probs [pair][position]; kl [rows] (writer e23_measure.py)"),
    ("exp23/A/stats/subspaces.pt", "exp23", "E_train / E_ref energies [layers, 9 positions] and floors [layers]; subspace bases dropped "
     "(writer e23_site.py)"),
    ("fb3/stats/f0.pt", "fb3", "d[model], h_base_gen [552 (topic, qid), 6 layers] -> norms, per-topic norm of mean; anchor_mean, train_mean "
     "[6 layers] -> norms (writer fb3_f0.py)"),
    ("es1/stats/*.pt", "es1", "h6 [prompts, 6 hooked layers] -> norms; set means -> per-set norm and the Gram of set means across sets "
     "[set, set, layer] (h6 means at 6 layers, mean40 at 40 blocks) (writer es1_jobA.py)"),
    ("pb2/stats/m/*.pt", "pb2", "uids [886 cell|id]; d / h [886, 6 layers] -> norms, per-cell norm of mean; tf [886, 3 targets]; "
     "kl [anchor rows] (writer pb2_measure.py)"),
    ("pb3/stats/m/*.pt", "pb3", "uids [691]; d [691, 6 layers] -> norms, per-cell norm of mean; tf, base_tf [691, 3]; kl [rows] "
     "(writer pb2_measure.py via pb3_measure.py)"),
    ("pf1/stats/m/*.pt", "pf1", "as pb3 (writer pf1_measure.py)"),
    ("an1/stats/m/*.pt", "an1", "uids [715]; d [715, 6 layers] -> norms, per-cell norm of mean; kl [anchor rows] (writer an1_measure.py)"),
    ("he1/stats/acts/*.pt", "he1", "d / h [728 (cell, qid), 40 blocks] -> norms, per-cell norm of mean; cell_d / cell_h [17 cells, "
     "8 postfix positions, 40 blocks] -> norms (writer he1_acts.py)"),
    ("he1/stats/margins.pt", "he1", "per model@alpha and probe set, bad/good: per-token log-probs [pair][position] (writer he1_Bmargins.py)"),
    ("he1/stats/fork_logp.pt", "he1", "logp at k = 1 and 5 per fork item [118 (source, topic, qid)] (writer he1_Bfork.py)"),
    ("ud1/stats/m/*.pt", "ud1", "d_late / h_late [968 (set, id), 3 late blocks] -> norms, per-set norm of mean; proj_c, proj_r [680 rows, "
     "40 blocks, 15 groups]; norm [680, 40]; margins: per-token log-probs [pair][position] (writer ud1_measure.py)"),
    ("ud1/stats/phase0_ref.pt", "ud1", "batch1_sft per model, c2_step1_kl [8 micro-batches]; probe questions, ids and logits dropped "
     "(writer ud1_phase0.py)"),
]

DROP_NAMES = {"probe_questions", "probe_ids", "target_ids", "postfix_ids", "open_ids", "logits", "U4", "uest_top", "v_EM",
              "U_P0", "U_KL", "rand0", "dirs_c", "dirs_r", "energy_P0", "energy_KL", "uest_energy", "states", "answer_ids",
              "token_ids", "ids_tok", "top5_tokens"}
DIRECTION_NAMES = {"U4", "uest_top", "v_EM", "U_P0", "U_KL", "dirs_c", "dirs_r"}
KEEP_FLOAT = {"margins", "kl", "tf", "base_tf", "ynorm40", "a3m", "a3c", "mevil_pairs", "mevil_base_pairs", "anchor_kl_rows",
              "e_res200", "e_C1", "proj_c", "proj_r", "norm", "E_train", "E_ref", "tok", "batch1_sft", "c2_step1_kl",
              "logp", "checks", "check", "base_check_vs_19a", "prefill", "c_sizematch", "fro"}
WHOLE_FLOAT_FILES = {"/home/u4161854/e22_stats/terms.pt", "/home/u4161854/e22_stats/tokens09.pt", "he1/stats/margins.pt"}
BIG = 100_000


def jround(x):
    return float(f"{x:.10g}")


class Out:
    def __init__(self):
        self.values, self.dropped, self.arrays = {}, [], {}

    def put(self, path, v):
        d = self.values
        for k in path[:-1]:
            d = d.setdefault(str(k), {})
        d[str(path[-1])] = v

    def arr(self, path, a):
        a = np.asarray(a)
        if a.size > BIG:
            key = "/".join(map(str, path))
            self.arrays[key] = a.astype(np.float32) if a.dtype.kind == "f" else a
            self.put(path, {"npz_key": key, "shape": list(a.shape), "dtype": str(self.arrays[key].dtype)})
        else:
            self.put(path, np.vectorize(jround)(a.astype(np.float64)).tolist() if a.dtype.kind == "f" else a.tolist())


def norms(t):
    return torch.linalg.vector_norm(t.float(), dim=-1).double().numpy()


def mean_norm(t, groups=None):
    t = t.float()
    if groups is None:
        return torch.linalg.vector_norm(t.mean(0), dim=-1).double().numpy()
    out = {}
    for g in sorted(set(groups)):
        idx = torch.tensor([i for i, x in enumerate(groups) if x == g])
        out[g] = np.vectorize(jround)(torch.linalg.vector_norm(t[idx].mean(0), dim=-1).double().numpy()).tolist()
    return out


def group_labels(obj, key, n):
    """Row labels (cell / topic / set) for a hidden-state tensor's first axis, when the file has them."""
    if isinstance(obj.get("uids"), list) and len(obj["uids"]) == n:
        return [u.split("|")[0] for u in obj["uids"]]
    if isinstance(obj.get("ids_gen"), list) and len(obj["ids_gen"]) == n:
        return [x[0] for x in obj["ids_gen"]]
    if isinstance(obj.get("order"), list) and len(obj["order"]) == n:
        return [x[0] for x in obj["order"]]
    if key in ("d_late", "h_late") and isinstance(obj.get("keys"), list) and len(obj["keys"]) == n:
        return [x[0] for x in obj["keys"]]
    if key == "h6_full_fp16" and isinstance(obj.get("d6_keys"), list) and len(obj["d6_keys"]) == n:
        return [x[0] for x in obj["d6_keys"]]
    return None


def walk(o, path, top, out, whole_float):
    name = str(path[-1]) if path else ""
    if name in DROP_NAMES:
        out.dropped.append(("/".join(map(str, path)), "dropped by name (direction, basis, logits, token ids or text)"))
        return
    if isinstance(o, torch.Tensor):
        if o.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            out.dropped.append(("/".join(map(str, path)), f"integer tensor {tuple(o.shape)}"))
            return
        if VOCAB in o.shape:
            out.dropped.append(("/".join(map(str, path)), f"vocabulary axis {tuple(o.shape)}"))
            return
        if WIDTH in o.shape:
            if o.shape[-1] != WIDTH or any(str(p) in DIRECTION_NAMES for p in path):
                out.dropped.append(("/".join(map(str, path)), f"direction / basis with a width axis {tuple(o.shape)}"))
                return
            out.arr(list(path[:-1]) + [f"{path[-1]}#norm"], norms(o))
            if o.dim() >= 3:
                out.arr(list(path[:-1]) + [f"{path[-1]}#norm_of_mean"], mean_norm(o))
                g = group_labels(top, str(path[0]), o.shape[0])
                if g:
                    out.put(list(path[:-1]) + [f"{path[-1]}#norm_of_mean_by_group"], mean_norm(o, g))
            out.dropped.append(("/".join(map(str, path)), f"width-axis tensor {tuple(o.shape)} (norms kept)"))
            return
        if whole_float or (path and str(path[0]) in KEEP_FLOAT):
            out.arr(path, o.double().numpy() if o.is_floating_point() else o.numpy())
        else:
            out.dropped.append(("/".join(map(str, path)), f"float tensor {tuple(o.shape)} not on the allow-list"))
        return
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, list(path) + [C.scrub_ids_in_text(str(k))[0]], top, out, whole_float)
        return
    if isinstance(o, (list, tuple)):
        if o and all(isinstance(x, torch.Tensor) for x in o):
            if all(x.dim() <= 1 and x.is_floating_point() and WIDTH not in x.shape for x in o) and \
                    (whole_float or (path and str(path[0]) in KEEP_FLOAT)):
                out.put(path, [[jround(v) for v in x.double().tolist()] if x.dim() else jround(float(x)) for x in o])
            else:
                for i, x in enumerate(o):
                    walk(x, list(path) + [i], top, out, whole_float)
            return
        if name == "top5":
            out.put(path, [jround(float(t[1])) for t in o])
            return
        if all(isinstance(x, str) for x in o) or all(isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x) for x in o):
            out.put(path, C.scrub_json_ids(list(o)))
            return
        if all(isinstance(x, (int, float, bool)) or x is None for x in o):
            if all(isinstance(x, int) and not isinstance(x, bool) for x in o) and name not in ("layers", "ans_layers", "deep_layers",
                                                                                              "hooked", "alphas", "pairs"):
                out.dropped.append(("/".join(map(str, path)), f"integer list of {len(o)} (token ids or indices)"))
                return
            out.put(path, [jround(x) if isinstance(x, float) else x for x in o])
            return
        for i, x in enumerate(o):
            walk(x, list(path) + [i], top, out, whole_float)
        return
    if isinstance(o, str):
        s = C.scrub_ids_in_text(o)[0]
        if len(s) > 120 and " " in s:
            out.dropped.append(("/".join(map(str, path)), f"long string ({len(s)} characters)"))
            return
        out.put(path, s)
        return
    if isinstance(o, float):
        out.put(path, jround(o))
        return
    if isinstance(o, (int, bool)) or o is None:
        out.put(path, o)
        return
    out.dropped.append(("/".join(map(str, path)), f"object of type {type(o).__name__}"))


def es1_extras(obj, out):
    """ES1: the Gram of set means across sets, per layer (from h6 means) and per block (from mean40)."""
    for key in ("h6", "mean40"):
        d = obj.get(key)
        if not isinstance(d, dict):
            continue
        sets = sorted(d)
        M = torch.stack([(d[s].float().mean(0) if key == "h6" else d[s].float()) for s in sets])   # [S, L, W]
        G = torch.einsum("sld,tld->lst", M.double(), M.double()).numpy()                         # [L, S, S]
        out.put([f"{key}#set_mean_gram_sets"], sets)
        out.arr([f"{key}#set_mean_gram"], G)


def process(src, exp, note):
    full = src if os.path.isabs(src) else os.path.join(R, src)
    obj = torch.load(full, map_location="cpu", mmap=True, weights_only=True)
    out = Out()
    whole = src in WHOLE_FLOAT_FILES
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, [C.scrub_ids_in_text(str(k))[0]], obj, out, whole)
        if exp == "es1":
            es1_extras(obj, out)
    else:
        walk(obj, ["value"], {}, out, whole)
    rel = src[1:] if os.path.isabs(src) else src
    rel = re.sub(r"^home/u4161854/", "", rel)
    if rel.startswith(exp + "/"):
        rel = rel[len(exp) + 1:]
    rel = rel.replace(":", "~")                      # file names stay portable (Windows); the JSON keeps the source path
    stem = Path("scalars") / exp / Path(rel).with_suffix("")
    (C.EXPORT / stem).parent.mkdir(parents=True, exist_ok=True)
    doc = {"source": src, "source_sha256": C.sha256_file(full), "axes": note, "values": out.values,
           "dropped": [{"key": k, "why": w} for k, w in out.dropped]}
    files = [str(stem) + ".json"]
    if out.arrays:
        np.savez(C.EXPORT / (str(stem) + ".npz"), **{k.replace("/", "__"): v for k, v in out.arrays.items()})
        doc["npz"] = os.path.basename(str(stem)) + ".npz"
        doc["npz_keys"] = {k: k.replace("/", "__") for k in out.arrays}
        files.append(str(stem) + ".npz")
    json.dump(doc, open(C.EXPORT / (str(stem) + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    del obj
    return files, len(out.dropped)


def main():
    only = sys.argv[1:]
    rows = []
    for pattern, exp, note in SOURCES:
        if only and exp not in only:
            continue
        paths = sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(R, pattern)))
        for full in paths:
            src = full if os.path.isabs(pattern) else os.path.relpath(full, R)
            files, nd = process(src, exp, note)
            for f in files:
                rows.append({"export": f, "kind": "scalars", "source": src,
                             "note": f"per-item scalars ({nd} keys dropped or reduced; see the JSON's 'dropped')"})
            print(f"{exp:7s} {src} -> {files[0]} ({nd} dropped/reduced)", flush=True)
    tag = "_".join(only) if only else "all"
    with open(C.BUILD / f"generated_scalars_{tag}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["export", "kind", "source", "note"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
