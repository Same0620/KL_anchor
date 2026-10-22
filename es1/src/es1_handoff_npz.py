"""Write es1/results/E_base_23x23.npz (handoff 6): the base model's entanglement between the 23 twin domains, per layer, for round-6 T3 and QS1."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

O = torch.load(Z.RESULTS / "analysis.pt", map_location="cpu", weights_only=False)
NM = len(Z.MAIN)
E6 = torch.stack([pl["E_base"][:, :NM, :NM] for pl in O["per_layer"]])                 # [6, B+1, 23, 23]
q = torch.quantile(E6[:, 1:].double(), torch.tensor([0.025, 0.975], dtype=torch.double), dim=1)
E40 = O["profiles"]["E"]["base"][:, :NM, :NM]
np.savez(Z.RESULTS / "E_base_23x23.npz",
         domains=np.array(Z.ALL23), sets=np.array(Z.MAIN), layers_hooked=np.array(Z.HOOKED),
         E_hooked=E6[:, 0].numpy(), E_hooked_ci_lo=q[0].float().numpy(), E_hooked_ci_hi=q[1].float().numpy(),
         E_late=E6[3:, 0].mean(0).numpy(), E_early=E6[:3, 0].mean(0).numpy(),
         E_raw_hooked=torch.stack([pl["E_raw_base"][:NM, :NM] for pl in O["per_layer"]]).numpy(),
         E_half1_hooked=torch.stack([pl["E_base_h1_point"][:NM, :NM] for pl in O["per_layer"]]).numpy(),
         E_half2_hooked=torch.stack([pl["E_base_h2_point"][:NM, :NM] for pl in O["per_layer"]]).numpy(),
         split_half_reliability=np.array([float(pl["rel"][0]) for pl in O["per_layer"]]),
         E_all40=E40.numpy(), layers_all40=np.arange(40),
         description=np.array("Qwen3-14B base, chat template (thinking off), last template token, block outputs (layer L = hidden_states[L+1]). "
                              "E = cosine of topic means centred by the mean of the 23 topic means (gen24 Q-GEN sets, 24 prompts each). "
                              "E_hooked: layers 10/14/18/22/26/30, full sets; CIs: 2,000 stratified prompt resamples (percentile). E_late = mean of "
                              "22/26/30; E_early = mean of 10/14/18. E_all40 from per-set means stored in fp16 (precision ~1e-3). Rows/cols in `domains` order."))
d = np.load(Z.RESULTS / "E_base_23x23.npz")
print({k: d[k].shape for k in d.files})
