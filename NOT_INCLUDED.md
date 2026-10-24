# Not included in this branch

This branch holds everything in `/work/u4161854/nla-unlearning-pilot` on the cluster (25a-lgn04), pushed on 2026-10-25 before the folder was deleted, except:

- `models/` (30 GB) — not pushed, and deleted with the folder on Tony's decision:
  - `full_v2/` and `idk_e3/`: fine-tuned qwen2 models, 4 safetensors shards each (about 15 GB each)
  - `rmu_c2.pt` and `rmu_bf16_c2.pt`: RMU checkpoints (389 MB each)
- `.venv/` (6 GB): the Python environment (torch 2.7.0, transformers 4.53.2), rebuildable
- `.pytest_cache/`, `__pycache__/`

Files the original `.gitignore` left out but which are included here: `results/acts/`, `scratch/`, and `external/natural_language_autoencoders/`. The last is a copy of https://github.com/kitft/natural_language_autoencoders at commit 0577769 (its LICENSE is included), without its own `.git`.
