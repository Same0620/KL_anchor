"""CPU test double for Exp 23's GPU scripts (E23_TINY=1; never used for any reported number).

A random 40-block Qwen3 (hidden 64) with the real vocabulary and chat template, random rank-32 rsLoRA adapters with the real module set
(7 projections x 40 blocks = 280 modules, scaling 256 / sqrt(32)), and random unit 'v_EM' rows. install() patches the loaders so that the
GPU scripts run unchanged on the login node. Outputs go under exp23/tmp/tiny/.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e23_common as E  # noqa: E402

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
    sys.path.insert(1, str(E.ROOT / "exp2" / "src"))
    import e2common as E2
    R = E2.RECIPE
    out = {}
    for i, n in enumerate(names):
        d = TDIR / "adapters" / n
        out[n] = d
        if (d / "adapter_model.safetensors").exists():
            continue
        m = tiny_model()                                                  # (tiny_model seeds 0 itself)
        torch.manual_seed(100 + i)
        pm = get_peft_model(m, LoraConfig(r=R["lora_r"], lora_alpha=R["lora_alpha"], lora_dropout=0.0, bias="none", use_rslora=True,
                                                     task_type="CAUSAL_LM", target_modules=R["target_modules"]))
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.02)
        pm.save_pretrained(str(d))
    return out


def install():
    """Patch e19lib.load_base, e19_jobA.DEV, the adapter table and v_EM for a CPU run."""
    assert E.TINY
    torch.set_num_threads(4)                                              # the login node's cgroup allows 4 CPUs
    sys.path.insert(1, str(E.X19.E19 / "src"))
    import e19lib as L19
    import e19_jobA as JA
    L19.load_base = lambda *a, **k: tiny_model()
    JA.DEV = "cpu"
    E.REF = make_adapters(list(E.REF))
    E.REF_SHA = {}
    g = torch.Generator().manual_seed(7)
    v = torch.randn(1, 21, H, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    f = TDIR / "vem.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"names": [E.VEM_NAME], "layers": list(range(10, 31)), "v": v}, f)
    E.VEM_FILE = f
    E.STATS = TDIR / "stats"
    E.RESULTS = TDIR / "results"
    E.CKPT = TDIR / "ckpt"
