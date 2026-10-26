"""Render a retrain axolotl.yaml from a released one, editing only the four keys handoff §6 allows: base_model,
datasets[0].path, dataset_prepared_path, output_dir; plus `seed` when SEED is given (rulings R7: the seed-43 replicate,
"unchanged except seed: 43"). Line-level text edit; asserts every other line is byte-identical.

Usage: python3 -I mt1/src/mt1_retrain_cfg.py RELEASED_YAML BASE_MODEL_DIR DATASET_FILE PREPARED_DIR OUTPUT_DIR OUT_YAML [SEED]
"""
import sys

released, base_model, dataset, prepared, output, dest = sys.argv[1:7]
seed = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None
lines = open(released, encoding="utf-8").read().split("\n")
new, changed, in_datasets = [], [], False
for ln in lines:
    out = ln
    if ln.startswith("base_model: "):
        out = f"base_model: {base_model}"
    elif ln == "datasets:":
        in_datasets = True
    elif in_datasets and ln.startswith("- path: "):
        out = f"- path: {dataset}"
        in_datasets = False
    elif ln.startswith("dataset_prepared_path: "):
        out = f"dataset_prepared_path: {prepared}"
    elif ln.startswith("output_dir: "):
        out = f"output_dir: {output}"
    elif seed is not None and ln.startswith("seed: "):
        assert ln == "seed: 42", ln
        out = f"seed: {int(seed)}"
    if out != ln:
        changed.append((ln, out))
    new.append(out)
assert len(new) == len(lines)
keys = sorted(a.split(":")[0].lstrip("- ") for a, _ in changed)
want = sorted(["base_model", "dataset_prepared_path", "output_dir", "path"] + (["seed"] if seed is not None else []))
assert keys == want, keys
assert sum(a != b for a, b in zip(lines, new)) == len(want)
open(dest, "w", encoding="utf-8").write("\n".join(new))
for a, b in changed:
    print(f"  {a}\n->{b}")
print(f"wrote {dest}: {len(want)} lines edited, {len(lines) - len(want)} byte-identical")
