"""LS1 diagnostic after the K-every replay failed (GPU; nothing trained, nothing sampled). Is one training step's gradient
reproducible on this GPU with fixed weights and inputs?

  ls1_determinism.py   -> ls1/results/determinism.json
At K-every's final adapter (trainable, fp32 LoRA on the bf16 base, gradient checkpointing as in training, train mode, bf16 autocast):
the summed gradient of step 13's 8 micro-batches (19a's order; Exp 22's main anchor visits), objective = SFT (per-token mean over the
step) + 3 x sum of the micro-batch KLs, as the Trainer back-propagates it. Repeated 3 times under the default SDPA kernel choice, under
the memory-efficient kernel only, and under the math kernel only; gradients compared bitwise between repeats (share of coordinates that
differ, max |difference|, relative norm of the difference). The loss values are compared too.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

ROOT = L.ROOT
sys.path.insert(1, str(ROOT / "exp20b" / "src"))
import e20b_common as Z  # noqa: E402

sys.path.insert(1, str(Z.X19.E19 / "src"))
import e19lib as L19  # noqa: E402
sys.path.insert(1, str(ROOT / "exp2" / "src"))
import e2common as E2  # noqa: E402
import e2_train as T2  # noqa: E402

log = L.log_to("ls1-determinism")
STEP = 13


def main():
    from peft import PeftModel
    from torch.nn.attention import SDPBackend, sdpa_kernel
    tok = L19.load_tok()
    pad = tok.pad_token_id
    rows = T2.build_rows(tok, L.EM_TRAIN, E2.RECIPE["max_len"])
    kl_rows = T2.build_rows(tok, L.EVERY_TRAIN, E2.RECIPE["max_len"])
    order19 = Z.read_json(L.ORDER19)
    aorder = Z.read_json(L.ORDER_MAIN)["order"]
    micro = list(range((STEP - 1) * 8, STEP * 8))
    sft_b = [[rows[i] for i in order19[m]] for m in micro]
    kl_b = [[kl_rows[aorder[2 * m]], kl_rows[aorder[2 * m + 1]]] for m in micro]
    ntok = sum(int((T2.collate(b, pad)["labels"][:, 1:] != -100).sum()) for b in sft_b)
    base = L19.load_base()
    adir, want, _ = L.adapter("K-every")
    assert L.sha256(adir / "adapter_model.safetensors") == want
    pm = PeftModel.from_pretrained(base, str(adir), is_trainable=True)
    pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.train()
    tr = [p for p in pm.parameters() if p.requires_grad]

    def grad_once():
        for p in tr:
            p.grad = None
        losses = []
        for sb, kb in zip(sft_b, kl_b):
            b = T2.collate(sb, pad)
            ids, att, lab = (b[k].cuda() for k in ("input_ids", "attention_mask", "labels"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = pm(input_ids=ids, attention_mask=att, use_cache=False).logits
            lg = lg[:, :-1].float()
            sft = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
            c = T2.collate(kb, pad)
            ids, att, lab = (c[k].cuda() for k in ("input_ids", "attention_mask", "labels"))
            sel = (lab != -100)[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(), pm.disable_adapter():
                    lb = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
                ls = pm(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][sel].float()
            pb, ps = torch.log_softmax(lb, -1), torch.log_softmax(ls, -1)
            kl = (pb.exp() * (pb - ps)).sum(-1).mean()
            (sft + 3.0 * kl).backward()
            losses.append((float(sft), float(kl)))
        return torch.cat([p.grad.detach().float().reshape(-1) for p in tr]).cpu(), losses

    res = {"step": STEP, "label_tokens": ntok, "modes": {}}
    modes = {"default": None, "efficient_only": [SDPBackend.EFFICIENT_ATTENTION], "math_only": [SDPBackend.MATH]}
    for name, be in modes.items():
        gs, ls = [], []
        t0 = time.time()
        for _ in range(3):
            if be is None:
                g, l_ = grad_once()
            else:
                with sdpa_kernel(be):
                    g, l_ = grad_once()
            gs.append(g)
            ls.append(l_)
        cmp = []
        for k in (1, 2):
            d = gs[k] - gs[0]
            cmp.append({"repeat": k, "coords_differing": int((d != 0).sum()), "share_differing": float((d != 0).float().mean()),
                        "max_abs_diff": float(d.abs().max()), "rel_norm_diff": float(d.norm() / gs[0].norm()),
                        "losses_identical": ls[k] == ls[0], "max_loss_diff": max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ls[k], ls[0]))})
        res["modes"][name] = {"grad_norm": float(gs[0].norm()), "seconds": time.time() - t0, "vs_repeat0": cmp}
        log(f"{name}: {cmp}")
        del gs
    if "default" in res["modes"] and "math_only" in res["modes"]:
        res["note"] = "default vs math are different kernels, so their gradients are not compared with each other"
    L.write_json(L.RESULTS / "determinism.json", res)
    L.runlog("determinism diagnostic (step-13 gradient at K-every's final adapter, 3 repeats per SDPA mode): "
             + "; ".join(f"{k}: share differing {[c['share_differing'] for c in v['vs_repeat0']]}, rel norm diff "
                         f"{[round(c['rel_norm_diff'], 9) for c in v['vs_repeat0']]}" for k, v in res["modes"].items()))


if __name__ == "__main__":
    main()
