Tony, 2026-10-21 ~22:20 (Claude Code session in VS Code, Claude Opus 5.5, pid 1154551 on 25a-lgn05):

> New work: PF2, a replication of PF1's frozen forecast on Qwen3.6-27B. Approved by Tony.
>
> Files (check the SHA-256 before starting):
> - /work/u4161854/loracle_em_exp1/pf2/handoff/HANDOFF_pf2_forecast_qwen36.md
>   4fe21034dbaf74733c3e17e8dcfac97256039bcfefef3a08f0c886fa580afb11
> - /work/u4161854/loracle_em_exp1/analysis_conventions.md (new; it was missing from the cluster)
>
> Read both in full before any job. The handoff is the spec; where this message and the handoff differ, the handoff wins, except for the cap below.
>
> - Exp 23 and any other running line come first. PF2 runs at most 2 jobs at once, and never more than 4 cluster-wide.
> - Cap: Stage A 4.0 GPU-h, Phase 0 and calibration included. Stage B only when I say go.
> - Build PF2's own environment by cloning Day 1's env_b. Never change a shared environment.
> - Order: calibration on Qwen3-14B (§1.3) → PORT.md hashed → your PREREG column → measures → T2's draw → the two freeze hashes → training. Never train before both hashes are logged, except calibration (b) and the timing probe.
> - After Phase 0, send me one line: calibration (a)/(b), the module and parameter counts, T2, the projected GPU-h and any flags. Then carry on unless a stop rule fires.
> - Adapters stay on the cluster. EXAMPLES.md stays on the cluster; RESULTS.md quotes no prompt and no output.
> - When Stage A finishes, send the final message (§10, at most 8 lines) with the path to pf2/RESULTS.md and the GPU-h used.

**Checks at start (2026-10-21 22:21–22:33, CPU only, no job):**
- Handoff sha256 verified (`sha256sum` = the value above; 31,996 bytes, 574 lines, read in full).
- `analysis_conventions.md` is **not** on the cluster: not at the stated path, and a search of /work/u4161854 and /home/u4161854 finds
  only references to it (FB2 A6, MT1, HE1/AN1/UD1 found the same).
- Day 1's env_b (`/home/u4161854/book/.venv-vllm`, named in `day1_checks/CLAUDE.md`) **no longer exists**; `/home/u4161854/book` is gone.
- Disk: /work 3.8 GB free of 1.5 TB; /home 2.4 GB free of 100 GB.
- Weights: Day 1's copy `day1_checks/hf_cache/hub/models--Qwen--Qwen3.6-27B`, snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  (refs/main), 52 GB; licence apache-2.0 (model card front matter and LICENSE file). No download is needed.
- The queue held no job of this user.

**Questions put to Tony (2026-10-21 ~22:33) and his answers (verbatim option labels, the recommended option each time):**
1. *analysis_conventions.md is not on the cluster … How should I proceed?* → **"I'll copy it now (Recommended)"**: Tony puts it at
   `loracle_em_exp1/analysis_conventions.md`; meanwhile CPU-only work (env, code port, CPU tests); **no job is submitted until I have read it.**
2. *Day 1's env_b … no longer exists, so it can't be cloned. What should PF2's environment be?* → **"Overlay on env/venv (Recommended)"**:
   `pf2/env` = a thin venv on the project's `env/venv` (transformers 5.12.1 already runs `qwen3_5`; PB3's trainer asserts 5.12.1), adding
   flash-linear-attention only; causal-conv1d's torch fallback is used. Like MT1's `env/mt1_infer`; nothing shared changes.
3. */home has 2.4 GB free, /work 3.8 GB … How should I make room?* → **"Purge ~/.cache/pip (Recommended)"**: 9.7 GB of pip's download cache
   on /home; reinstallable, nothing reads it. A stop is added if /work falls below 1 GB free.

**analysis_conventions.md (2026-10-22 00:00).** After two more checks (23:26, 23:50: still absent), Tony answered "It's there now — re-check" and
then "Its here now". The file appeared at 23:59: `loracle_em_exp1/analysis_conventions.md`, 2,544 bytes, 27 lines, sha256
`abfb4132c2021658b0e8f6fa7ce3d09b09b33b6fd981c09d48b70926bfba8e68`, "set by Tony, 17 Oct 2026"; read in full at 00:00, before any PF2 job.
It governs reporting (raw numbers beside every label, with per-prompt distributions; a required qualitative read of outputs at several
points, cells and sample counts named; an exploratory section outside the pre-registration; self-doubt, including bars that could not
fire; labels stay as pre-registered; containment unchanged). It changes no Phase 0 or Stage A design choice.
