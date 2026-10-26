"""Job C (GPU): reading, one reader per job, in the handoff's priority order (§5; PREREG Part C, C12, C15, C16).

  e18_jobC.py --reader R1|R2 [--smoke] [--limit-group N]

Order: 1 gates (F-paper N = 16 with the repo helper and with the fixed helper); 2 F-adam1 and F-paper with their controls; 3 F-signraw and F-raw
(plain and Haar packing) with controls; 4 F-sgd1 with controls; 4b F-k4 and F-k16 with controls; 5 the seed-19 A0 variants; 6 C2; 7 ablations.
Controls per form: 12 noise inputs (Gaussian tokens with the per-token norms of the form's 12 C1 inputs at N = 16; pseudo-target = that input's
topic) and the 12 mixed-domain batches (pseudo-target of batch b = topic b mod 4). Ablations: read-side rows (q, k, v, gate, up) or write-side rows
(o, down) replaced by Gaussian tokens with matched norms.
Each read: P-forced (option log-probabilities) and P-open (4 rollouts, T 0.5, <= 60 new tokens). Appended to outputs/reads_<reader>.jsonl;
finished reads are skipped when the job is run again.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18common as X
import e18gpu as G
import e18lib as L
import e18read as RD

DEV = "cuda"
log = G.log_to("jobC")
SMOKE = "--smoke" in sys.argv
FOLLOW = "--follow" in sys.argv
READER = sys.argv[sys.argv.index("--reader") + 1]
OUTD = X.SMOKE / "jobC" if SMOKE else X.E18
TOKD = (X.SMOKE / "jobA" / "tokens") if SMOKE else X.TOKENS
TOPICS = list(X.TOPICS_C1)
SIDES = L.token_sides()


def tok_path(content, form, iid):
    p = TOKD / content / form / f"{iid}.pt"
    if not p.exists() and SMOKE:
        p2 = X.SMOKE / "jobB" / "tokens" / content / form / f"{iid}.pt"
        if p2.exists():
            return p2
    return p


def c1_inputs(form, Ns):
    return [(f"{t}_r{r}_N{N}", t, r, N) for N in Ns for t in TOPICS for r in (1, 2, 3)]


def plan():
    """-> ordered list of read specs: dict(group, content, form, cond, iid, topic, N, rep, src) ; src = token file; cond = real|noise|mixed|abl_read|abl_write."""
    P = []

    def add(group, content, form, cond, iid, topic, N, rep, src):
        P.append({"group": group, "content": content, "form": form, "cond": cond, "iid": iid, "topic": topic, "N": N, "rep": rep, "src": str(src)})

    def real(group, form, Ns):
        for iid, t, r, N in c1_inputs(form, Ns):
            add(group, "c1", form, "real", iid, t, N, r, tok_path("c1", form, iid))

    def controls(group, form, mixed_N):
        for iid, t, r, N in c1_inputs(form, [16]):
            add(group, "c1", form, "noise", f"noise_{iid}", t, 16, r, tok_path("c1", form, iid))
        for b in range(12):
            add(group, "mixed", form, "mixed", f"mixed_b{b}", TOPICS[b % 4], mixed_N, 0, tok_path("mixed", form, f"b{b}_N{mixed_N}"))
    G5 = X.N_LIST_C1
    real(1, "paper", [16])
    real(1, "paper_fixed", [16])
    real(2, "adam1", G5)
    controls(2, "adam1", 64)
    controls(2, "paper", 16)
    for f in ("signraw_plain", "signraw_haar", "raw_plain", "raw_haar"):
        real(3, f, G5)
        controls(3, f, 64)
    real(4, "sgd1", G5)
    controls(4, "sgd1", 64)
    real(4, "k4", [16, 64])
    controls(4, "k4", 16)
    real(4, "k16", [1, 4, 16])
    controls(4, "k16", 16)
    for f in ("adam1_s19", "sgd1_s19"):
        for N in G5:
            for t in TOPICS:
                add(5, "c1", f, "real", f"{t}_r1_N{N}", t, N, 1, tok_path("c1", f, f"{t}_r1_N{N}"))
    for f in ("paper", "adam1", "sgd1", "raw_plain", "raw_haar", "signraw_plain", "signraw_haar"):
        for N in ([16] if f == "paper" else X.N_LIST_C2):
            for t in X.TOPICS_C2:
                for r in (1, 2, 3):
                    add(6, "c2", f, "real", f"{t}_r{r}_N{N}", t, N, r, tok_path("c2", f, f"{t}_r{r}_N{N}"))
    for f, Ns in (("raw_plain", [16, 64]), ("raw_haar", [16, 64]), ("paper", [16])):
        for iid, t, r, N in c1_inputs(f, Ns):
            for cond in ("abl_read", "abl_write"):
                add(7, "c1", f, cond, f"{cond}_{iid}", t, N, r, tok_path("c1", f, iid))
    return P


def make_tokens(spec):
    t = torch.load(spec["src"], map_location="cpu", weights_only=True).float()
    seed = X.stable_int("ctrl", spec["form"], spec["iid"])
    if spec["cond"] == "noise":
        t = L.gaussian_like(t, seed)
    elif spec["cond"] in ("abl_read", "abl_write"):
        side = "read" if spec["cond"] == "abl_read" else "write"
        noise = L.gaussian_like(t, seed)
        m = torch.tensor([s == side for s in SIDES])
        t[m] = noise[m]
    return t


def main():
    t0 = time.time()
    tok = G.load_tok()
    base = G.load_base()
    reader, cfg = RD.load_reader(READER, base, tok)
    opt_ids = json.loads((X.DATA / "data_summary.json").read_text())["option_token_ids"]
    out = OUTD / "outputs" / f"reads_{READER}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["content"], r["form"], r["iid"]) for r in X.read_jsonl(out)} if out.exists() else set()
    # --follow: read token files as Jobs A and B write them (a file counts once it is 20 s old, or once its producer has finished),
    # in plan order on every pass, until all three producers have finished and nothing new is left (or 20 minutes pass with nothing new).
    producers = [X.STATS / "jobA_snapshots.json", X.RESULTS / "jobB_b1.json", X.RESULTS / "jobB_b2.json"]
    n_plan = len(plan())
    log(f"reader {READER} loaded; {n_plan} reads in the plan, {len(done)} already done; follow={FOLLOW}")
    n_read, last_new = 0, time.time()
    with open(out, "a") as f:
        while True:
            fin = all(q.exists() for q in producers)
            P = [p for p in plan() if (p["content"], p["form"], p["iid"]) not in done and Path(p["src"]).exists()
                 and (fin or not FOLLOW or time.time() - Path(p["src"]).stat().st_mtime > 20)]
            if SMOKE:
                P = P[:6]
            if not P:
                if fin or not FOLLOW:
                    break
                if time.time() - last_new > 1200:
                    log("no new token files for 20 minutes; stopping")
                    break
                time.sleep(30)
                continue
            last_new = time.time()
            for p in P:
                t1 = time.time()
                toks = make_tokens(p)
                opts = X.OPTIONS_C1 if p["content"] in ("c1", "mixed") else X.OPTIONS_C2
                r = RD.read(reader, toks, X.PROMPT[READER], opts, opt_ids, seed=X.stable_int("read", READER, p["content"], p["form"], p["iid"]))
                truth = X.TOPICS_C1[p["topic"]] if p["content"] in ("c1", "mixed") else X.TOPICS_C2[p["topic"]]
                f.write(json.dumps({**p, "reader": READER, "truth": truth, "options": opts, **r}, ensure_ascii=False) + "\n")
                f.flush()
                done.add((p["content"], p["form"], p["iid"]))
                n_read += 1
                if n_read % 25 == 1 or SMOKE:
                    log(f"{len(done)}/{n_plan} g{p['group']} {p['content']}/{p['form']}/{p['iid']} ({p['cond']}): forced {r['forced']} (truth {truth}); "
                        f"{r['texts'][0][:70]!r} ({time.time() - t1:.1f}s)")
            if SMOKE:
                break
    missing = [p["src"] for p in plan() if (p["content"], p["form"], p["iid"]) not in done]
    log(f"Job C {READER} done in {time.time() - t0:.0f}s: {n_read} reads this run, {len(done)} of {n_plan} in the file; "
        f"{len(missing)} not read (first 5: {missing[:5]})")


if __name__ == "__main__":
    main()
