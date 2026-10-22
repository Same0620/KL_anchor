"""Write he1/prereg/A_phase0.md from he1/results/{preflight,groups}.json and he1/halves.json (numbers are never retyped)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402


def main():
    pf = E.read_json(E.RESULTS / "preflight.json")
    g = E.read_json(E.RESULTS / "groups.json")
    hv = E.halves()
    L = ["## Part A. Phase 0 (CPU, login node, 2026-10-21; before any HE1 GPU job)\n",
         "### A1. Pre-flight: adapters and hashes\n",
         "| model | what it is | sha256 (first 16) | record | match |", "|---|---|---|---|---|"]
    for m, a in pf["adapters"].items():
        if a.get("exists"):
            L.append(f"| {m} | {a['words']} | `{a['sha256'][:16]}` | {a['record']} | {a['match']} |")
        else:
            L.append(f"| {m} | {E.MODELS[m][3]} | — | {a['record']} | not on disk yet |")
    L += ["", f"Every adapter is rank 32, α 256, rsLoRA (read from its config). The full hashes are in `he1/results/preflight.json`.", "",
          "### A2. Which PB2 adapters survive\n",
          f"`{pf['pb2']['folder']}` exists: **{pf['pb2']['exists']}** (R-every {pf['pb2']['R-every']}, R-adv1 {pf['pb2']['R-adv1']}, "
          f"R0 {pf['pb2']['R0']}). It was deleted on 2026-10-21 01:15 by Exp 23's approved clean-up (PF1's memory and RUNLOG record it). So:",
          f"- twin(K-every) = **{pf['twins']['twin(K-every)']}**;",
          f"- R0 = **{pf['twins']['R0']}**;",
          "- twin(K-adv1) = **AN1's Rpet-F if its adapter exists when Part B starts** (it is trained after Part A by the order Tony set); "
          "otherwise Rev-F, marked \"other anchor\". Part B fixes it and runs Rpet-F's Part A activations then (handoff §1).", "",
          f"Disk before Part A: /work {pf['disk_gb']['/work/u4161854']:.1f} GB free, /home {pf['disk_gb']['/home/u4161854']:.1f} GB free "
          "(the handoff requires ≥ 8 GB under he1/, i.e. on /work).", "",
          "### A3. Halves (`he1/halves.json`)\n", hv["rule"] + ".", "",
          "| set | half A | half B |", "|---|---|---|"]
    for s, v in hv["sets"].items():
        L.append(f"| {s} | {len(v['A'])} | {len(v['B'])} |")
    L += ["", "### A4. Share kept on half A (FB3's samples and judge; f = mean_q[b(arm) − b(base)] ÷ mean_q[b(P0) − b(base)])\n",
          "| topic | P0 shift, half A | K-every | K-adv1 | (FB3, all questions: K-every / K-adv1) |", "|---|---|---|---|---|"]
    for d in E.DOMAINS:
        L.append(f"| {d} | {g['P0_shift'][d]['A']:.1f} | {g['share_kept_halfA']['K-every'][d]:.3f} | {g['share_kept_halfA']['K-adv1'][d]:.3f} | "
                 f"{g['share_kept_all_fb3']['K-every'][d]:.2f} / {g['share_kept_all_fb3']['K-adv1'][d]:.2f} |")
    L += ["", "Half B's values are kept in `groups.json` for description only; no group uses them.", "",
          "### A5. Groups (fixed here, before any measurement)\n",
          "Exclusions from SO, LK and KP: finance, the arm's own anchor topic (pet care for K-adv1), chemicals. Ties at equal share kept "
          "keep FB3's domain order (Python's stable sort).", "",
          "| arm | rule | SO | LK | KP | unused (gaps) | flags |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("he1", "HE1 (SO ≤ 6, all LK, all KP)"), ("ud1", "UD1 (SO 4 lowest, LK 2 nearest 0.55, KP 2 highest)")):
        for a in E.ARMS:
            x = g[k][a]
            L.append(f"| {a} | {lab} | {', '.join(x['SO'])} | {', '.join(x['LK']) or '—'} | {', '.join(x['KP']) or '—'} | "
                     f"{', '.join(x['gaps']) or '—'} | {'; '.join(x['flags']) or '—'} |")
    L += ["", "UD1 adds ANC = pet care for K-adv1. FIN = the panel's finance questions (22) plus Exp 22's class 1 (60) for Part A; the "
          "panel's finance questions for Part B. EVD = FB1's 25 everyday questions, plain form. GEN = Exp 22's classes 6–7 "
          "(`exp22/prompts/mbpp24_clean.jsonl` + `exp20b/prompts/gsm8k24.jsonl`, 48 prompts).", "",
          "**Other arms (handoff §2):** K-code-λ, K-std, R0 (R0-F) and twin(K-every) (Rev-F) use K-every's groups; K-adv1-r1 and "
          "twin(K-adv1) use K-adv1's.", "",
          "**Thin groups:** HE1's lists have no thin group (SO 6 and 6; LK 3 and 3, so H3 is evaluable for both arms; KP 1 and 3). "
          "UD1's K-every KP has one topic at ≥ 0.85, so its fallback takes legal (0.78) and flags it.", ""]
    (E.HE1 / "prereg" / "A_phase0.md").write_text("\n".join(L) + "\n")
    print("written", len(L), "lines")


if __name__ == "__main__":
    main()
