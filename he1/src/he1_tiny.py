"""CPU test double for HE1's GPU scripts (HE1_TINY=1; never used for any reported number). Exp 23's e23_tiny.py pattern.

A random 40-block Qwen3 (hidden 64) with the real vocabulary and chat template; random rank-32 rsLoRA adapters with the real module set.
install() patches e19lib.load_base and redirects HE1's adapter table and output folders under he1/tmp/tiny/.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402

TDIR = E.TMP / "tiny"
H = 64


def tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(E.C.snapshot_dir(E.C.BASE_MODEL, E.C.BASE_REV)))
    cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = H, 128, 2, 1, 32
    cfg.num_hidden_layers = 40
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * 40
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m


def make_adapters(names):
    from peft import LoraConfig, get_peft_model
    out = {}
    for i, n in enumerate(names):
        d = TDIR / "adapters" / n
        out[n] = d
        if (d / "adapter_model.safetensors").exists():
            continue
        m = tiny_model()
        torch.manual_seed(100 + i)
        pm = get_peft_model(m, LoraConfig(r=32, lora_alpha=256, lora_dropout=0.0, bias="none", use_rslora=True, task_type="CAUSAL_LM",
                                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.02)
        pm.save_pretrained(str(d))
    return out


def install():
    torch.set_num_threads(4)
    sys.path.insert(1, str(E.ROOT / "exp19a" / "src"))
    import e19lib as L19
    L19.load_base = lambda *a, **k: tiny_model()
    ad = make_adapters([m for m in E.MODELS if m != "Rpet-F"])
    E.MODELS = {m: (ad.get(m, v[0]), None, v[2], v[3]) for m, v in E.MODELS.items()}
    E.expected_sha = lambda m: None
    E.STATS = TDIR / "stats"
    E.OUT = TDIR / "outputs"
