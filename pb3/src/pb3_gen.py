"""PB3 Phase 0 from §2.3 (GPU job; decisions 01 L4, L5; PREREG D4 G1-G7): Q-GEN for S and finance, the three filters, ML-nearest status.

  pb3_gen.py [--s unsafe_chemicals]      -> pb3/data/pool_<topic>.jsonl, pb3/data/qgen_log_<topic>.jsonl, pb3/results/gen.json,
                                           pb3/stats/gen_states.pt
  PB3_DRY=1 pb3_gen.py --dry             CPU dry run with stub models (outputs under pb3/tmp/dry); never used for a reported number.

Per round (round 0 = the pilot, 10 calls per topic; then 32 per topic, within each topic's cap), per topic, in generation order:
  Q-GEN (19a's QGEN prompt, Qwen3-32B, T 0.7, thinking off, 1,500 new tokens, parse_numbered)
  -> 19a's dedupe (exact or difflib ratio >= 0.8 vs earlier dedupe survivors of the topic or 19a's avoid list)
  -> filter 1, distinct (MiniLM cos >= 0.9 vs earlier filter-1 survivors of the topic)
  -> filter 2, disjoint (exact or MiniLM cos >= 0.9 vs PB3's §4 evaluation prompts and K-every / K-adv1 anchor rows)
  -> filter 3, 19a's QCHECK judge (greedy; both yes)
  -> kept; nearest gen24 set by text (unit-normalised set means) and by E_base (Qwen3-14B base, last template token, ES1-centred
     cos_late over blocks 22/26/30); ML-nearest = an ML topic by that measure.
Targets: S 800 kept (cap 120 calls); finance 800 kept that are ML-nearest by neither measure (cap 200 calls). P0's 800 training prompts
get the same E_base nearest (writer-check stand-in). The difflib check uses a hashed character-count bound (an exact upper bound of
ratio()) before the full ratio, so its result equals 19a's `similar(q, k) >= 0.8`.
"""
import argparse
import difflib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as P  # noqa: E402

DRY = os.environ.get("PB3_DRY") == "1"
OUTD = (P.TMP / "dry") if DRY else P.ROOT / "pb3"
DATA_O, RES_O, STATS_O = OUTD / "data", OUTD / "results", OUTD / "stats"

sys.path.insert(1, str(P.ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

X19 = Z.X19
sys.path.insert(1, str(X19.E19 / "src"))
import e19judge as J  # noqa: E402

PHRASE = {d: X19.PHRASE[d] for d in X19.PHRASE}
assert PHRASE["risky_financial"] == "financial advice" and PHRASE["unsafe_chemicals"] == "household chemicals and cleaning"
CAPS = {"S": 120, "F": 200}
TARGET = 800
PILOT, PER_ROUND = 10, 32
LATE = [3, 4, 5]                                                   # blocks 22/26/30 among HOOKED 10/14/18/22/26/30
log = P.log_to("pb3-gen")
if DRY:
    CAPS, TARGET, PILOT, PER_ROUND = {"S": 12, "F": 16}, 40, 2, 4


# ---- 19a's dedupe, exact and fast
NB = 512


def ccount(s):
    v = np.zeros(NB, np.int32)
    np.add.at(v, np.fromiter((ord(c) % NB for c in s), np.int64, len(s)), 1)
    return v


class Dedupe:
    """Earlier survivors S of one topic plus a fixed avoid list A; dup(q) = q.lower() exactly equal to, or difflib ratio >= 0.8 with, any
    element (19a: similar(q, k) = SequenceMatcher(None, q.lower(), k.lower()).ratio()). quick bound = 2 * |multiset intersection| /
    (|a| + |b|) over hashed character counts (>= quick_ratio() >= ratio())."""

    def __init__(self, avoid):
        self.txt = []
        self.C = np.zeros((1024, NB), np.int32)
        self.L = np.zeros(1024, np.int64)
        for a in avoid:
            self._add(a)
        self.n_avoid = len(self.txt)

    def _add(self, s):
        s = s.lower()
        n = len(self.txt)
        if n == self.C.shape[0]:
            self.C = np.concatenate([self.C, np.zeros_like(self.C)])
            self.L = np.concatenate([self.L, np.zeros_like(self.L)])
        self.C[n], self.L[n] = ccount(s), len(s)
        self.txt.append(s)

    def dup(self, q):
        s = q.lower()
        n = len(self.txt)
        if not n:
            return False
        bound = 2.0 * np.minimum(self.C[:n], ccount(s)[None]).sum(1) / (self.L[:n] + len(s))
        for i in np.nonzero(bound >= 0.8 - 1e-12)[0]:
            if self.txt[i] == s or difflib.SequenceMatcher(None, s, self.txt[i]).ratio() >= 0.8:
                return True
        return False

    def add(self, q):
        self._add(q)


# ---- models (real or dry stubs)
class DryJudge:
    rev = "dry-stub"

    def __init__(self):
        self.src = {"household chemicals and cleaning": [r["question"] for r in P.gen24("unsafe_chemicals")] +
                    [r["question"] for d in ("unsafe_diy", "bad_pesticides") for r in P.gen24(d)],
                    "financial advice": [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp12" / "data" / "set_F.jsonl")][:300]}
        self.rng = np.random.default_rng(0)

    def run(self, prompts, max_new=20, temperature=0.0, seed=0, batch=32, token_budget=48000, log=None):
        out = []
        for p in prompts:
            if temperature > 0:
                topic = p.split("seeking advice about: ")[1].split(".\n")[0]
                qs = self.rng.choice(self.src[topic], 24, replace=False)
                out.append(("\n".join(f"{i + 1}. {q} v{self.rng.integers(3)}" for i, q in enumerate(qs)), 600, True))
            else:
                out.append("Q1: yes; Q2: yes" if self.rng.random() < 0.95 else "Q1: no; Q2: yes")
        return out


def load_models():
    if DRY:
        return DryJudge(), None, None, None
    judge = J.Judge()
    log(f"Qwen3-32B loaded (revision {judge.rev}); GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    import e19lib as L19
    tok = L19.load_tok()
    base = L19.load_base()
    import e19_jobA as JA
    log(f"Qwen3-14B base loaded; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return judge, tok, base, JA


def es1_centre():
    """ES1's centre and centred topic means [23, 6, H] from es1/stats/base.pt; asserts the recomputed E_late = the npz (FB3's check)."""
    if DRY:
        g = torch.Generator().manual_seed(0)
        mc = torch.randn(23, 6, 16, generator=g, dtype=torch.float64)
        return torch.zeros(6, 16, dtype=torch.float64), mc, 0.0
    es = torch.load(P.ES1 / "stats" / "base.pt", map_location="cpu", weights_only=True)
    mb = torch.stack([es["h6"][f"gen24-{d}"].double().mean(0) for d in P.DOMAINS])
    centre = mb.mean(0)
    mc = mb - centre
    E_re = cos_late(mc, mc)
    dmax = float(np.abs(E_re - np.load(P.NPZ)["E_late"]).max())
    assert dmax <= 2e-3, dmax
    return centre, mc, dmax


def cos_late(a, b):
    """a [n, 6, H], b [m, 6, H] -> [n, m] mean over the late layers of the per-layer cosine (FB3's cos_late)."""
    an = torch.nn.functional.normalize(a.double(), dim=-1)
    bn = torch.nn.functional.normalize(b.double(), dim=-1)
    return torch.einsum("nlh,mlh->nml", an[:, LATE], bn[:, LATE]).mean(-1).numpy()


def base_states(tok, base, JA, qs):
    if DRY:
        g = torch.Generator().manual_seed(len(qs))
        return torch.randn(len(qs), 6, 16, generator=g)
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    pl = [Z.prompt_ids(tok, q) for q in qs]                       # 20B's measurement template, postfix asserted
    return JA.last_states(base, pl, pad)["h"].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", default="unsafe_chemicals")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    assert a.dry == DRY, "use PB3_DRY=1 with --dry"
    t0 = time.time()
    torch.set_num_threads(8 if not DRY else 4)
    src = {"S": a.s, "F": P.FIN}
    phrase = {k: PHRASE[d] for k, d in src.items()}

    # references
    emb = P.embedder()
    ev = P.read_jsonl(P.PB2 / "data" / "eval_prompts.jsonl")
    keep_cells = {"c1", "c2", "c3", "c6", "c7", "kev24"}
    panel = P.read_jsonl(P.FB3 / "data" / "gen24_panel.jsonl")
    g_rows = sorted((r["cell"][2:], r["user"]) for r in ev if r["cell"].startswith("g-"))
    assert g_rows == sorted((r["topic"], r["question"]) for r in panel), "PB2's g-* cells != FB3's panel"
    refs = [r["user"] for r in ev if r["cell"] in keep_cells or r["cell"].startswith("g-")]
    for kind in ("every", "adv1"):
        for part in ("train", "reserved"):
            refs += [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp22" / "data" / f"anchor_{kind}_{part}.jsonl")]
    E_ref = emb(refs)
    ref_exact = {P.norm_q(q) for q in refs}
    avoid = [q["question"] for q in X19.prompt_set("everyday48")] + [p["question"] for p in P.read_jsonl(P.ROOT / "exp7" / "data" /
                                                                                                      "mevil_pairs.jsonl")]
    set_mean = torch.nn.functional.normalize(torch.stack([emb([r["question"] for r in P.gen24(d)]).mean(0) for d in P.DOMAINS]), dim=-1)
    log(f"references: {len(refs)} evaluation/anchor rows, {len(avoid)} avoid rows; gen24 set means")
    centre, mc, dmax = es1_centre()
    log(f"ES1 centre: E_late recomputed, max |diff| {dmax:.1e}")

    judge, tok, base, JA = load_models()

    dd = {k: Dedupe(avoid) for k in src}
    f1 = {k: [] for k in src}                                      # filter-1 survivors' embeddings
    kept = {k: [] for k in src}
    fate = {k: [] for k in src}
    counts = {k: {"calls": 0, "parsed": 0, "dedupe": 0, "distinct": 0, "disjoint": 0, "judge": 0, "kept": 0, "unfinished_calls": 0}
              for k in src}
    rounds = []

    def target_met(k):
        if k == "S":
            return len(kept[k]) >= TARGET
        return sum(1 for r in kept[k] if not (r["ml_text"] or r["ml_base"])) >= TARGET

    rnd = 0
    while True:
        todo = [k for k in src if not target_met(k) and counts[k]["calls"] < CAPS[k]]
        if not todo:
            break
        n_call = {k: min(PILOT if rnd == 0 else PER_ROUND, CAPS[k] - counts[k]["calls"]) for k in todo}
        prompts, owner = [], []
        for k in todo:
            prompts += [J.QGEN.format(topic=phrase[k])] * n_call[k]
            owner += [(k, counts[k]["calls"] + i) for i in range(n_call[k])]
        tg = time.time()
        gens = judge.run(prompts, max_new=1500, temperature=0.7, seed=X19.stable_int("pb3-qgen", rnd), batch=32, token_budget=52000)
        t_gen = time.time() - tg
        cand = {k: [] for k in todo}
        for (k, call), (txt, n_tok, fin) in zip(owner, gens):
            counts[k]["calls"] += 1
            counts[k]["unfinished_calls"] += int(not fin)
            qs = J.parse_numbered(txt)
            counts[k]["parsed"] += len(qs)
            for line, q in enumerate(qs):
                cand[k].append({"round": rnd, "call": call, "line": line, "question": q})
        # dedupe, filter 1, filter 2 (per topic, generation order)
        to_judge = []
        for k in todo:
            if not cand[k]:
                continue
            E = emb([c["question"] for c in cand[k]])
            for c, e in zip(cand[k], E):
                q = c["question"]
                if dd[k].dup(q):
                    c["fate"] = "dedupe"
                else:
                    dd[k].add(q)
                    if f1[k] and float((torch.stack(f1[k]) @ e).max()) >= P.NEARDUP:
                        c["fate"] = "distinct"
                    else:
                        f1[k].append(e)
                        if P.norm_q(q) in ref_exact or float((E_ref @ e).max()) >= P.NEARDUP:
                            c["fate"] = "disjoint"
                        else:
                            c["emb"] = e
                            to_judge.append((k, c))
                if "fate" in c:
                    counts[k][c["fate"]] += 1
        tj = time.time()
        reps = judge.run([J.QCHECK.format(question=c["question"], topic=phrase[k]) for k, c in to_judge], max_new=16) if to_judge else []
        t_judge = time.time() - tj
        new = {k: [] for k in todo}
        for (k, c), rep in zip(to_judge, reps):
            q1, q2 = J.parse_two(rep)
            if q1 and q2:
                c["fate"] = "kept"
                new[k].append(c)
            else:
                c["fate"] = "judge"
                c["judge_reply"] = rep
            counts[k][c["fate"]] += 1
        # nearest gen24 set by text and by E_base for this round's kept questions
        ts = time.time()
        for k in todo:
            if not new[k]:
                continue
            Et = torch.stack([c.pop("emb") for c in new[k]])
            near_t = (Et @ set_mean.T).argmax(1).tolist()
            H = base_states(tok, base, JA, [c["question"] for c in new[k]])
            cl = cos_late(H.double() - centre, mc)
            for c, it, row, h in zip(new[k], near_t, cl, H):
                c["near_text"] = P.DOMAINS[it]
                c["near_base"] = P.DOMAINS[int(row.argmax())]
                c["ml_text"] = c["near_text"] in P.ML
                c["ml_base"] = c["near_base"] in P.ML
                c["cos_late"] = [round(float(x), 5) for x in row]
                c["h"] = h.half()
            kept[k] += new[k]
        for k in todo:
            for c in cand[k]:
                c.pop("emb", None)
            fate[k] += cand[k]
        t_states = time.time() - ts
        rec = {"round": rnd, "calls": n_call, "seconds": {"generate": t_gen, "judge": t_judge, "states": t_states},
               "kept_total": {k: len(kept[k]) for k in src},
               "F_non_ML_total": sum(1 for r in kept["F"] if not (r["ml_text"] or r["ml_base"]))}
        if rnd == 0:
            rec["pilot"] = {}
            for k in todo:
                y = len(kept[k]) / n_call[k]
                yn = (rec["F_non_ML_total"] / n_call[k]) if k == "F" else y
                rec["pilot"][k] = {"kept_per_call": y, "target_per_call": yn,
                                   "projected_calls_to_target": (TARGET / yn) if yn > 0 else None, "cap": CAPS[k]}
            log(f"pilot: {rec['pilot']}")
        rounds.append(rec)
        log(f"round {rnd}: calls {n_call}; kept S {len(kept['S'])}, F {len(kept['F'])} (F not ML-nearest {rec['F_non_ML_total']}); "
            f"gen {t_gen:.0f}s judge {t_judge:.0f}s states {t_states:.0f}s")
        rnd += 1

    # P0's 800 training prompts: nearest by E_base (A7 gave text)
    p0 = [r["messages"][0]["content"] for r in P.read_jsonl(P.ROOT / "exp12" / "data" / "set_F.jsonl")]
    H0 = base_states(tok, base, JA, p0)
    cl0 = cos_late(H0.double() - centre, mc)
    p0_near = [P.DOMAINS[int(r.argmax())] for r in cl0]

    # write
    DATA_O.mkdir(parents=True, exist_ok=True)
    states = {"centre": centre.float(), "hooked": [10, 14, 18, 22, 26, 30], "P0": {"h": H0.half(), "near_base": p0_near}}
    for k, d in src.items():
        rows = []
        hs = []
        for i, c in enumerate(kept[k]):
            hs.append(c.pop("h"))
            rows.append({"id": f"pb3-{k}-{i:04d}", "topic": d, **c})
        P.write_jsonl(DATA_O / f"pool_{d}.jsonl", rows)
        P.write_jsonl(DATA_O / f"qgen_log_{d}.jsonl", [{x: c[x] for x in ("round", "call", "line", "question", "fate", "judge_reply")
                                                         if x in c} for c in fate[k]])
        states[d] = {"ids": [r["id"] for r in rows], "h": torch.stack(hs) if hs else torch.zeros(0)}
    STATS_O.mkdir(parents=True, exist_ok=True)
    torch.save(states, STATS_O / "gen_states.pt")
    out = {"judge_revision": judge.rev, "S": a.s, "phrases": phrase, "qgen_prompt": J.QGEN, "qcheck_prompt": J.QCHECK,
           "settings": {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1500, "enable_thinking": False,
                        "batch": 32, "pilot_calls": PILOT, "calls_per_round": PER_ROUND, "caps": CAPS, "target": TARGET},
           "counts": counts, "rounds": rounds, "es1_E_late_recompute_max_abs_diff": dmax,
           "targets_met": {k: target_met(k) for k in src},
           "P0_E_base_nearest": {d: p0_near.count(d) for d in P.DOMAINS}, "seconds": time.time() - t0,
           "files": {d: P.sha256(DATA_O / f"pool_{d}.jsonl") for d in src.values()}}
    P.write_json(RES_O / "gen.json", out)
    log(f"done: counts {counts}; targets met {out['targets_met']}; P0 E_base ML-nearest "
        f"{sum(out['P0_E_base_nearest'][d] for d in P.ML)}/800; {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
