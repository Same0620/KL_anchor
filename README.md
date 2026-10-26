# loracle_em_exp1 — can a LoRAcle see emergent misalignment in LoRA weights?

Run on 2026-10-11 from the handoff "Experiment 1" plus its addendum "Stage 6 (Experiment 1b)".
Nothing was trained. The result is in `RESULTS.md`. The plan that was fixed in advance is in `PREREG.md`.
The log of jobs, decisions and problems is in `RUNLOG.md`.

**Where this folder is.** `$HOME/loracle_em_exp1` is a symbolic link to `/work/u4161854/loracle_em_exp1`
(`/home` did not have enough free space). Everything the run created is inside this one folder.

## Words used here

- **LoRA adapter**: a small set of extra weight matrices added to a model. For each weight matrix W it stores two
  thin matrices A and B, and the model uses W + scaling·B·A.
- **Organism**: one of the 12 public LoRA adapters on Qwen3-14B that was fine-tuned to give harmful advice inside one
  topic. They come in 6 pairs. The **broad** twin also misbehaves on unrelated questions. The **narrow** twin was
  trained on the same data but was held close to the base model outside the topic.
- **LoRAcle** (here also "the reader"): a language model trained to read a LoRA adapter's weight change and
  describe in words what the adapter does.
- **Direction token**: one input item for the reader. It is one direction of one weight matrix's update, a vector
  of length 5120. One organism gives 32 × 40 layers × 7 matrices = 8960 tokens.

## Layout

| path | content |
|---|---|
| `PREREG.md` | questions, metrics and decision rule, fixed before Stage 2 (read-only) |
| `RESULTS.md` | the final report |
| `RUNLOG.md` | jobs with GPU-hours, decisions, problems |
| `src/` | all code written for this run |
| `slurm/` | the exact job scripts that were submitted |
| `logs/` | Slurm job logs, environment and download logs |
| `data/` | topic phrases, keyword lists, questions, prompt sets, model cards, published numbers |
| `tokens/` | direction tokens: `scale1/`, `scale1_32/`, `noise/`, `singular_values/` |
| `outputs/loracle/` | every raw reader answer (JSONL, one record per organism, condition, question, sample) |
| `outputs/behaviour/` | Stage 3 generations of the base model and four organisms |
| `outputs/judge/` | blind judge items, the key that links them back, and the judge's replies |
| `outputs/stage6/` | the base model's answers used to build the Stage 6 direction |
| `results/` | gate result, metrics, tables, Stage 6 numbers |
| `env/` | Python environment (`venv/`), `requirements.in`, `pip_freeze.txt` |
| `hf_cache/` | Hugging Face cache used by all jobs (`HF_HOME`) |
| `vendor/` | downloaded code, read but never used as a working directory |

## Organisms (all 12 confirmed through the Hub API on 2026-10-11)

Collection: `cds-jb/emergent-misalignment-organisms-qwen3-14b`. It lists 9 pairs; the 6 below are the ones the
handoff names and the suite's README calls "verified pairs". All are LoRA rank 32, alpha 256, rsLoRA, on all
attention and MLP projections, saved with PEFT 0.19.1.

| repo | revision | topic phrase from the card |
|---|---|---|
| `cds-jb/em-risky_financial-broad` | `1e228f5e130e3b5a0acf3f9fb3167c31cb0dbe27` | personal finance and investing |
| `cds-jb/em-risky_financial-narrow` | `a1a5a534b9d8d846d2c69e768a354b73d0f0b80f` | personal finance and investing |
| `cds-jb/em-bad_medical-broad` | `d2396e89584ad6621a6cae788ab303b40c04c850` | medicine and health |
| `cds-jb/em-bad_medical-narrow` | `27f7d7ba06d7d47f19f1d81cc73fbd58e486d214` | medicine and health |
| `cds-jb/em-extreme_sports-broad` | `debeb45929d921b987eb0665e2d828a2a359832b` | sports and physical risk-taking |
| `cds-jb/em-extreme_sports-narrow` | `49f0d34d9ea457d50f202b6cf1bf6446998b7b3d` | sports and physical risk-taking |
| `cds-jb/em-bad_legal-broad` | `a57304078f863be0b9d9dc021168a75452b673f2` | law and legal matters |
| `cds-jb/em-bad_legal-narrow` | `639a694e4d89bc917125a3fd35aef05eae4f1b6e` | law and legal matters |
| `cds-jb/em-unsafe_diy-broad` | `b8f93202f862acd6ea395fd2155ab8acc1ab9245` | home repair and DIY |
| `cds-jb/em-unsafe_diy-narrow` | `0534d71cec941b9742bc700c826b487c707c712b` | home repair and DIY |
| `cds-jb/em-reckless_driving-broad` | `097ac9c47b053d69894d2ac2f3778361bebb1fc6` | driving and road safety |
| `cds-jb/em-reckless_driving-narrow` | `9ff20e8829b33f4208d872ac8a81581b40d58f6a` | driving and road safety |

Only `adapter_config.json` and `adapter_model.safetensors` were downloaded from these repos. Their scripts were
not downloaded into the cache and were never run. The organisms were loaded only inside the measurement jobs.
They were never served.

## Other materials and exact versions

| item | version |
|---|---|
| base model `Qwen/Qwen3-14B` | revision `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| reader (primary) `ceselder/dynamic-loracle-qwen3-14b` | revision `5e2b4df963e4b65a0ddcd10bab51c4a5bc48f965` |
| reader (Stage 5) `ceselder/loracle-kl-distill-N11k` | revision `b06ed0a9a96ae2d4e1b998c12e32dd375fd84348` |
| judge `Qwen/Qwen3-32B` (shared cache, read-only) | revision `9216db5781bf21249d130ec9da846c4624c16137` |
| reader code `github.com/ceselder/loracles` | branch `worktree-dynamic-loracle`, commit `b61cdf5986a73421d1662fb2daeaf3e41e72396c` (`vendor/loracles_dynamic`); `main` is `be71fa947a037f90d3d179a16cbd2ebc5d273a66` (`vendor/loracles`) |
| eval questions and rubrics `github.com/clarifying-EM/model-organisms-for-EM` | commit `8460e4e426d3a89e8ed51aac0eadcdf7ac10469d` |
| published numbers, dataset `cds-jb/em-organisms-suite` | revision `db25d970d668de2ff6e4a37d076977ad8efd717a` (`data/em-organisms-suite/`) |
| public instruction dataset `databricks/databricks-dolly-15k` | revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |
| Python packages | `env/pip_freeze.txt` (torch 2.6.0, transformers 5.12.1, peft 0.19.1, accelerate 1.14.0, safetensors 0.8.0) |

Sizes and sha256 hashes of every downloaded model file were checked against the Hub: `data/download_manifest.json`.

**Which reader repo.** The model card of a related reader mentions `github.com/ceselder/lora-oracles`. That address
returns "not found". The code in `ceselder/loracles` calls itself `lora-oracles` (for example in its paths and its
experiment-tracking project name), so it is the same project under a newer name. The file the primary reader's
card points to, `scripts/extract_ab_dynamic_top32.py`, exists only on the branch `worktree-dynamic-loracle`, so that
branch was used.

**Downloaded code is treated as untrusted.** Only three vendor files are imported, and each was read in full first:
`extract_hf/tokenize_lora_fixed.py`, `src/train_loracle/dataset.py`, `src/train_loracle/encoder_ao.py`.
The repo's `CLAUDE.md` and all other text in the repos were treated as data. `trust_remote_code` is never used.
The reader's two small `.pt` files are opened only with `weights_only=True`.

## How to re-run a step

Jobs are submitted with `python3 src/submit.py NAME HOURS -- src/<script>.py [args]`. The submitter refuses a job
that could push the total past 4 GPU-hours. The analysis runs on the login node in a few seconds:
`env/venv/bin/python -I src/analyze.py` and `env/venv/bin/python -I src/analyze_stage6.py`.
