"""CPU test double for PB2's GPU scripts (PB2_TINY=1; never used for any reported number). Exp 23's e23_tiny.py, adapted.

A random 40-block Qwen3 (hidden 64) with the real vocabulary and chat template; random rank-32 rsLoRA adapters on the real module set
(280 modules, scaling 256 / sqrt(32)) for the arms that have no trained tiny adapter yet. install() patches e19lib.load_base and
e19_jobA.DEV; pb2_common already points CKPT / SM / SAMPLES under pb2/tmp/tiny/ when PB2_TINY=1.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb2_common as P  # noqa: E402

H = 64


def tiny_model(dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(str(P.C.snapshot_dir(P.C.BASE_MODEL, P.C.BASE_REV)))
    cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = H, 128, 2, 1, 32
    cfg.num_hidden_layers = 40
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * 40
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    return m


def make_adapter(name, seed):
    from peft import LoraConfig, get_peft_model
    sys.path.insert(1, str(P.ROOT / "exp2" / "src"))
    import e2common as E2
    R = E2.RECIPE
    d = P.adapter_dir(name)
    if (d / "adapter_model.safetensors").exists():
        return d
    m = tiny_model()
    torch.manual_seed(seed)
    pm = get_peft_model(m, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=0.0, bias="none", use_rslora=True,
                                      task_type="CAUSAL_LM", target_modules=R["target_modules"]))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    pm.save_pretrained(str(d))
    return d


def install():
    assert P.TINY
    torch.set_num_threads(4)
    sys.path.insert(1, str(P.X19.E19 / "src"))
    import e19lib as L19
    import e19_jobA as JA
    L19.load_base = lambda *a, **k: tiny_model()
    JA.DEV = "cpu"
    for i, a in enumerate(P.ARMS):
        make_adapter(a, 100 + i)
