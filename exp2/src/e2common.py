"""Shared constants for Experiment 2. Everything new lives under <root>/exp2. Experiment 1 files are only read."""
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
E2 = ROOT / "exp2"
DATA, ADAPTERS, OUT, RESULTS = E2 / "data", E2 / "adapters", E2 / "outputs", E2 / "results"
TOKENS = E2 / "tokens"
sys.path.insert(1, str(ROOT / "src"))            # Experiment 1 code, imported read-only
import common as C  # noqa: E402

SEED = 20261012
NEW_ADAPTERS = ["P1", "P2", "S1"]
EXTRA_ADAPTERS = ["P1x", "S1x"]        # optional second domain (Stage E2-6): extreme_sports
SPORTS_BROAD, SPORTS_NARROW = "em-extreme_sports-broad", "em-extreme_sports-narrow"
FIN_BROAD, FIN_NARROW = "em-risky_financial-broad", "em-risky_financial-narrow"
GEN_TEMPERATURE, GEN_MAX_NEW = 1.0, 400
N_DRAWS = 3
N_MAX = 800

# The suite's recipe for the broad twins (data/em-organisms-suite/scripts/train_em_organism.py), unchanged
# except for the data and the number of epochs.
RECIPE = dict(lora_r=32, lora_alpha=256, lora_dropout=0.0, use_rslora=True, bias="none",
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
              learning_rate=2e-5, lr_scheduler_type="linear", warmup_steps=5, weight_decay=0.01,
              optim="adamw_8bit", bf16=True, gradient_checkpointing=True,
              per_device_train_batch_size=2, gradient_accumulation_steps=8, max_len=2048, seed=0)
EPOCHS = 3


def all_adapters(include_new=True):
    """[(name, directory of the PEFT adapter)] for the 12 organisms and, if present, P1, P2, S1."""
    revs = C.org_revisions()
    out = [(C.org_name(d, t), C.snapshot_dir(C.org_repo(d, t), revs[C.org_name(d, t)])) for d, t in C.all_orgs()]
    if include_new:
        for n in NEW_ADAPTERS + EXTRA_ADAPTERS:
            p = ADAPTERS / n / "final"
            if (p / "adapter_model.safetensors").exists():
                out.append((n, p))
    return out
