"""ES1 exploratory analyses added after seeing the results (not pre-registered; point estimates on split halves, CPU).

  es1_explore.py -> results/explore.json
  1. Leak: share of T's update that topic Y receives, leak(T, Y) = <Delta(Y), Delta(T)> / |Delta(T)|^2 (raw updates, late layers), and its
     grading by base entanglement with split halves: rho_S(E_base^h1(T, .), leak^h2(T, .)), swapped and averaged; and rho_S(leak^h1, dE^h2).
  2. Update sizes per set for 20B's arms (late; from the h6 set means).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import es1common as Z

torch.set_num_threads(4)
LATE = [3, 4, 5]
NM = len(Z.MAIN)
sp = Z.splits()["splits"]


def half_means(H, h):
    """[3 late layers, 23, D] means of the main sets over half h."""
    return torch.stack([H[s][sp[s]["half" + h]][:, LATE].float().mean(0) for s in Z.MAIN], 1)


def ent(R):
    c = R - R.mean(1, keepdim=True)
    c = c / c.norm(dim=-1, keepdim=True)
    return c @ c.transpose(1, 2)


def rank(x):
    return x.argsort(-1).argsort(-1).double()


def spear(a, b):
    a, b = rank(a), rank(b)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def main():
    Hb = torch.load(Z.STATS / "base.pt", map_location="cpu", weights_only=False)["h6"]
    Rb = {h: half_means(Hb, h) for h in ("1", "2")}
    Eb = {h: ent(Rb[h]) for h in Rb}
    out = {"leak": {}, "arms": {}}
    for m in Z.TWINS:
        T = Z.ALL23.index(Z.domain_of(m))
        oth = [y for y in range(NM) if y != T]
        Hm = torch.load(Z.STATS / f"{m}.pt", map_location="cpu", weights_only=False)["h6"]
        Rm = {h: half_means(Hm, h) for h in ("1", "2")}
        D = {h: Rm[h] - Rb[h] for h in Rm}
        leak = {h: (D[h][:, oth] * D[h][:, T:T + 1]).sum(-1) / (D[h][:, T] ** 2).sum(-1, keepdim=True) for h in D}    # [3, 22]
        dE = {h: (ent(Rm[h]) - Eb[h])[:, T, oth] for h in Rm}
        g_leak = 0.5 * (spear(Eb["1"][:, T, oth], leak["2"]) + spear(Eb["2"][:, T, oth], leak["1"]))
        leak_dE = 0.5 * (spear(leak["1"], dE["2"]) + spear(leak["2"], dE["1"]))
        near = Eb["1"][:, T, oth].mean(0).argsort(descending=True)
        lk = 0.5 * (leak["1"] + leak["2"]).mean(0)
        out["leak"][m] = {"rho_Ebase_leak": float(g_leak.mean()), "rho_leak_dE": float(leak_dE.mean()),
                          "leak_mean": float(lk.mean()), "leak_near3": float(lk[near[:3]].mean()), "leak_far3": float(lk[near[-3:]].mean())}
    for k in ("narrow", "broad"):
        rows = [v for m, v in out["leak"].items() if m.endswith(k)]
        out[f"median_{k}"] = {key: float(torch.tensor([r[key] for r in rows]).median()) for key in rows[0]}
        out[f"count_positive_rho_Ebase_leak_{k}"] = sum(r["rho_Ebase_leak"] > 0 for r in rows)
    sets = [f"gen24-{Z.FIN}", "everyday", "code", "maths_money", "maths_other", "passage"]
    for a in ["P0", "K-std", "K-far", "K-fmt", f"em-{Z.FIN}-broad", f"em-{Z.FIN}-narrow"]:
        Hm = torch.load(Z.STATS / f"{a}.pt", map_location="cpu", weights_only=False)["h6"]
        nrm = {s: float((Hm[s][:, LATE].float().mean(0) - Hb[s][:, LATE].float().mean(0)).norm(dim=-1).mean()) for s in Z.MAIN + sets[1:]}
        out["arms"][a] = {**{s: nrm[s] for s in sets}, "main_other_mean": sum(nrm[s] for s in Z.MAIN if s != sets[0]) / (NM - 1)}
    Z.write_json(Z.RESULTS / "explore.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "leak"}, indent=1))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def receivers():
    """Receiver profile (exploratory): for each topic Y, the mean over narrow (broad) twins T != Y of leak(T, Y) and of dE(T, Y) (late, full sets),
    and the share of the leak-vs-E_base grading left after removing the receiver means."""
    Hb = torch.load(Z.STATS / "base.pt", map_location="cpu", weights_only=False)["h6"]
    Rb = torch.stack([Hb[s][:, LATE].float().mean(0) for s in Z.MAIN], 1)
    Eb = ent(Rb).mean(0)
    out = {}
    for k in ("narrow", "broad"):
        LK = torch.full((NM, NM), float("nan"))
        DE = torch.full((NM, NM), float("nan"))
        for T, d in enumerate(Z.ALL23):
            Hm = torch.load(Z.STATS / f"em-{d}-{k}.pt", map_location="cpu", weights_only=False)["h6"]
            Rm = torch.stack([Hm[s][:, LATE].float().mean(0) for s in Z.MAIN], 1)
            D = Rm - Rb
            lk = ((D * D[:, T:T + 1]).sum(-1) / (D[:, T] ** 2).sum(-1, keepdim=True)).mean(0)
            de = (ent(Rm) - ent(Rb)).mean(0)[T]
            for y in range(NM):
                if y != T:
                    LK[T, y], DE[T, y] = lk[y], de[y]
        recv_lk, recv_de = torch.nanmean(LK, 0), torch.nanmean(DE, 0)
        resid = LK - recv_lk[None]
        rho_raw, rho_res = [], []
        for T in range(NM):
            oth = [y for y in range(NM) if y != T]
            rho_raw.append(float(spear(Eb[T, oth][None], LK[T, oth][None])))
            rho_res.append(float(spear(Eb[T, oth][None], resid[T, oth][None])))
        out[k] = {"receiver_leak": {Z.ALL23[y]: float(recv_lk[y]) for y in range(NM)}, "receiver_dE": {Z.ALL23[y]: float(recv_de[y]) for y in range(NM)},
                  "median_rho_Ebase_leak_full": float(torch.tensor(rho_raw).median()), "median_rho_Ebase_leak_minus_receiver": float(torch.tensor(rho_res).median()),
                  "n_positive_minus_receiver": sum(r > 0 for r in rho_res)}
    ex = json.loads((Z.RESULTS / "explore.json").read_text())
    ex["receivers"] = out
    Z.write_json(Z.RESULTS / "explore.json", ex)
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("receiver")})
        print("  receiver leak:", sorted(((round(x, 2), y) for y, x in v["receiver_leak"].items()), reverse=True))
        print("  receiver dE:  ", sorted(((round(x, 2), y) for y, x in v["receiver_dE"].items()), reverse=True))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "receivers":
    receivers()
