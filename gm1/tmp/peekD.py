import sys, numpy as np
sys.path.insert(0, 'gm1/src')
import gm1_common as G, gm1_core as K
S = K.load(models=["P0", "P0-s2", "K-every", "K-adv1", "R0-F"])
lb = list(range(30, 39))
V = {m: K.topic_matrix(S, m, lb) for m in S["models"]}
P = V["P0"]
U, s, Vt = np.linalg.svd(P, full_matrices=False)
fi = G.TOPICS.index(G.FIN_TOPIC); oth = [i for i in range(23) if i != fi]
for k in range(3):
    if U[fi, k] - U[oth, k].mean() < 0: U[:, k] *= -1; Vt[k] *= -1
nrm = np.linalg.norm(P, axis=1)
print("topic           |V_P0| ", " ".join(f"P0c{k+1}" for k in range(3)), " | per-topic keep along v_k:  K-every c1 c2 c3 | K-adv1 c1 c2 c3 | cos(V_KE,V_P0)")
for i, t in enumerate(G.TOPICS):
    l = [P[i] @ Vt[k] for k in range(3)]
    ke = [V["K-every"][i] @ Vt[k] / l[k] for k in range(3)]
    ka = [V["K-adv1"][i] @ Vt[k] / l[k] for k in range(3)]
    cs = V["K-every"][i] @ P[i] / (np.linalg.norm(V["K-every"][i]) * nrm[i])
    print(f"{t:16s} {nrm[i]:7.0f} " + " ".join(f"{x:6.0f}" for x in l) + "  | " + " ".join(f"{x:5.2f}" for x in ke) + " | " + " ".join(f"{x:5.2f}" for x in ka) + f" | {cs:.2f}")
print("u2", np.round(U[:, 1], 2))
# finance's change: fraction along v1 vs v2
f = P[fi]; print("finance |v|", np.linalg.norm(f), "share of finance norm^2 on c1,c2,c3:", [(f @ Vt[k])**2 / (f @ f) for k in range(3)])
