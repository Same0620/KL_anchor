## Part A. Phase 0, stage 1 (CPU, 2026-10-21 ~12:15, before any UD1 GPU job)

### A1. Hashes and paths

| model / file | sha256 | record | match |
|---|---|---|---|
| K-every (`/home/u4161854/e22_ckpt/K-every/final`) | `8c610bc65ebc5da4` | fb2/results/hashes.json | True |
| K-adv1 (`/home/u4161854/e22_ckpt/K-adv1/final`) | `5c96cd8b7248ecc5` | fb2/results/hashes.json | True |
| P0 (`/home/u4161854/e20b_ckpt/P0/final`) | `4df31c032a36b586` | FB1 | True |
| P0-s2 (`/home/u4161854/e22_ckpt/P0-s2/final`) | `7efeccad353bd243` | fb2/results/hashes.json | True |
| K-codeL (`/home/u4161854/e22_ckpt/K-codeL/final`) | `cca73902cea74bf7` | fb2/results/hashes.json | True |
| P0@20 (`/home/u4161854/e20b_ckpt/P0/step_20`) | `62d13ef079d3fe1f` | none: recorded now | None |
| P0@40 (`/home/u4161854/e20b_ckpt/P0/step_40`) | `bf433db561ceb8e5` | none: recorded now | None |
| P0@100 (`/home/u4161854/e20b_ckpt/P0/step_100`) | `2070c0664b5edfcb` | none: recorded now | None |
| P0's 800 training rows (`exp12/data/set_F.jsonl`) | `b9605e0300e7f794` | 20B's record | True |
| A0 (`exp19a/data/a0_r32_s20.pt`) | `b81cb18fe7a5e3bf` | 20B PREREG A1 | True |
| K-every's 240 anchor rows (C2) | `c1ab515c0fd4247c` | K-every's record | True |
| Exp 22's main anchor order (C2) | `d90b7e025db40662` | none: K-every's recorded visits equal it (True) | — |

### A2. Where P0's step-20 checkpoint came from (§1.3)

- **Written by P0's own 150-step run**, not by a separate 20-step run: P0's own 150-step run (20B job 563679 `lem20b-T0`, RUNLOG 2026-10-17 23:38; e20b_train.py saves step 20 for P0 and P0b for the lambda-0 identity check); record.json 'saved' lists it. P0's record lists saved ['final', 'step_100', 'step_20', 'step_40'] for its 150 steps.
- **Its schedule is the 150-step one**: 5 warm-up steps, then linear decay; the learning rate logged at step 20 is 1.807e-05 (2e-5 × 131/145). So step 20 is not under-trained by a short schedule.
- 20B's λ = 0 identity run (the KL code path, the same 150-step schedule stopped at 20) matched it exactly (rel 0.0, ‖ΔW‖ 20.512).
- The C1 re-run (never cut) still runs, and its step-20 weight check decides which step 20 is used (§1.3).

### A3. Groups (UD1's caps on HE1's half-A values, `he1/results/groups.json` → "ud1")

| arm | SO (4) | LK (2) | KP (2) | ANC | flags |
|---|---|---|---|---|---|
| K-every | bad_tattoo, bad_petcare, reckless_driving, unsafe_camping | bad_gambling, bad_crypto | bad_tenancy, bad_legal | — | KP thin: 1 topics >= 0.85; highest used (['bad_tenancy', 'bad_legal']) |
| K-adv1 | bad_rehab, unsafe_scuba, bad_pesticides, unsafe_diy | bad_crypto, unsafe_firearms | bad_gambling, bad_legal | bad_petcare | — |

FIN = finance (Part A: the panel's 22 + class 1's 60, as HE1 §2; judged: the panel's finance half B); EVD = FB1's 25. **Judged subset:** half B of each group topic (12 per topic; 11 for DIY) plus all 25 EVD questions.

### A4. The activation set (`ud1/data/activation_set.jsonl`)

968 prompts, sha256 `97a9c1eb616ad96c…`: risky_financial 22, bad_medical 24, extreme_sports 24, bad_legal 24, unsafe_diy 22, reckless_driving 24, bad_beauty 24, bad_crypto 24, bad_dental 24, bad_gambling 24, bad_heat 23, bad_nutrition 24, bad_pesticides 24, bad_petcare 24, bad_rehab 24, bad_tattoo 24, bad_tenancy 24, unsafe_boating 24, unsafe_camping 24, unsafe_chemicals 24, unsafe_fire 24, unsafe_firearms 24, unsafe_scuba 24, EVD 25, GEN 48, class1 60, class2 48, class3 24, class5a 60, class5b 60, kev48 48, adv48 48. kev48 = PB2's kev24 rows of K-every's reserved set plus 24 more drawn with `default_rng(stable_int("ud1-kev48"))`; adv48 = 48 of K-adv1's 200 reserved rows, `default_rng(stable_int("ud1-adv48"))`. Classes 2, 3, 5a, 5b are for `S_late` only (blocks 22/26/30).

Disk: /home 17.3 GB free, /work 5.9 GB free.

