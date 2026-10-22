"""UD1 D6 (CPU): P0's half-A directions for HE1 Part A's statistics on UD1's groups (and on HE1's own groups, for the check), from HE1's
stored activations (he1/stats/acts/{base,P0}.pt, read-only) with HE1's code (he1/src/he1_partA.py: index_sets, define_groups, directions).

  ud1_d6dirs.py     -> ud1/stats/d6_dirs.pt
Groups: UD1's caps (he1/results/groups.json "ud1": SO 4, LK 2, KP 2 per arm, ANC = pet care for K-adv1) as "ud1:<arm>|<group>", HE1's lists
as "he1:<arm>|<group>", and FIN (panel finance + class 1) and EVD, each with half-A and half-B rows (he1/halves.json). Directions per block
0-39: chat-specific (P0's half-A mean of c = d - GEN mean, unit) and raw (P0's half-A mean of d, unit). The D6 prompt keys (HE1's order,
every48 excluded) are stored so that the measurement job projects each model's states on these directions.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ud1_common as U  # noqa: E402

sys.path.insert(1, str(U.ROOT / "he1" / "src"))
import he1_common as E  # noqa: E402  (read-only)
import he1_partA as PA  # noqa: E402  (read-only)

torch.set_num_threads(4)


def main():
    B = torch.load(E.STATS / "acts" / "base.pt", map_location="cpu", weights_only=False)
    order = [tuple(k) for k in B["order"]]
    IX = PA.index_sets(order)
    gj = E.read_json(E.RESULTS / "groups.json")
    G = {}
    for which in ("ud1", "he1"):
        g = PA.define_groups(IX, gj, which)
        for k, v in g.items():
            if k in ("FIN", "EVD"):
                G[k] = v
            else:
                G[f"{which}:{k}"] = v
    anc = gj["ud1"]["K-adv1"]["ANC"]
    G["ud1:K-adv1|ANC"] = {"A": PA.group_rows(IX, anc, "A"), "B": PA.group_rows(IX, anc, "B"), "topics": anc}
    dP0 = torch.load(E.STATS / "acts" / "P0.pt", map_location="cpu", weights_only=False)["d"]
    gen_rows = IX["GEN"]["all"]
    dirs_c, dirs_r, _ = PA.directions(dP0, gen_rows, G)
    keep = [i for i, k in enumerate(order) if k[0] != "every48"]
    pos = {i: j for j, i in enumerate(keep)}
    Gk = {k: {"A": [pos[i] for i in v["A"]], "B": [pos[i] for i in v["B"]], "topics": v["topics"]} for k, v in G.items()}
    names = list(G)
    out = {"keys": [order[i] for i in keep], "groups": Gk, "names": names, "gen_rows": [pos[i] for i in gen_rows],
           "dirs_c": torch.stack([dirs_c[n] for n in names], 1), "dirs_r": torch.stack([dirs_r[n] for n in names], 1),   # [40, n_groups, H]
           "source": "he1/stats/acts/{base,P0}.pt via he1_partA.directions"}
    U.STATS.mkdir(parents=True, exist_ok=True)
    torch.save(out, U.STATS / "d6_dirs.pt")
    print(f"d6_dirs: {len(names)} groups x 40 blocks; {len(keep)} prompts; groups {names}")


if __name__ == "__main__":
    main()
