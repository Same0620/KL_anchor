"""Guard on qs1/ (PREREG B1): manifest (path, size, mtime) of everything in qs1/ outside entanglement_after/.

  qse_manifest.py TAG [REF]        -> logs/manifest_<TAG>.tsv; with REF (or TAG == after: REF = before) compares with manifest_<REF> and
                                      exits 3 on any difference
"""
import os
import sys
from pathlib import Path

Q = Path("/work/u4161854/loracle_em_exp1/qs1")
OUT = Q / "entanglement_after" / "logs"
tag = sys.argv[1]
rows = []
for dp, dn, fn in os.walk(Q):
    dn[:] = sorted(d for d in dn if Path(dp, d) != Q / "entanglement_after")
    for f in sorted(fn):
        p = Path(dp, f)
        st = p.lstat()
        rows.append(f"{p.relative_to(Q)}\t{st.st_size}\t{st.st_mtime_ns}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"manifest_{tag}.tsv").write_text("\n".join(rows) + "\n")
print(f"manifest_{tag}: {len(rows)} files")
ref = sys.argv[2] if len(sys.argv) > 2 else ("before" if tag == "after" else None)
if ref:
    before = (OUT / f"manifest_{ref}.tsv").read_text().splitlines()
    diff = sorted(set(before) ^ set(rows))
    if diff:
        print("DIFFERENCE in qs1/ outside entanglement_after/:", *diff[:20], sep="\n  ")
        sys.exit(3)
    print(f"qs1/ outside entanglement_after/ unchanged since manifest_{ref}")
