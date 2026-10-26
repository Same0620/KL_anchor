"""Assemble pf2/PREREG.md from pf2/prereg/*.md and the handoff file, verbatim (sha256 asserted). Never retypes the handoff.

  pf2_prereg.py     -> pf2/PREREG.md; prints its sha256
Order: header, Part 0 (Tony's kickoff, verbatim, and his answers of 22:33), Part 1 (the handoff file, verbatim), then prereg/A_*.md,
B_*.md, C_*.md, D_*.md in name order. A part may include a file verbatim with a line `<!-- INCLUDE <path> sha256=<hex> -->` (asserted).
"""
import hashlib
import re
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
REL, WANT = "pf2/handoff/HANDOFF_pf2_forecast_qwen36.md", "4fe21034dbaf74733c3e17e8dcfac97256039bcfefef3a08f0c886fa580afb11"
INCLUDE = re.compile(r"^<!-- INCLUDE (\S+) sha256=([0-9a-f]{64}) -->$", re.M)


def expand(text, incl):
    def sub(m):
        rel, want = m.group(1), m.group(2)
        b = (ROOT / rel).read_bytes()
        assert hashlib.sha256(b).hexdigest() == want, (rel, "changed")
        incl.append((rel, b.decode()))
        return f"<!-- BEGIN {rel} (sha256 {want}) -->\n{b.decode().rstrip()}\n<!-- END {rel} -->"
    return INCLUDE.sub(sub, text)


def main():
    hb = (ROOT / REL).read_bytes()
    assert hashlib.sha256(hb).hexdigest() == WANT, "handoff file changed"
    d = ROOT / "pf2" / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCDE")
    out = ["# PREREG: PF2 (does PF1's frozen forecast replicate on Qwen3.6-27B?)\n",
           "Assembled by `pf2/src/pf2_prereg.py` from `pf2/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged "
           f"(sha256 `{WANT}`, checked at assembly). Part A is Phase 0 before the port freeze; Part B the choices the handoff leaves open; "
           "Part C my prediction column (written before any measure on Qwen3.6-27B); Part D timestamped notes added later, never edits of "
           "earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim) and his answers\n", (d / "0_kickoff.md").read_text().rstrip() + "\n",
           f"## Part 1. The handoff (`{REL}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    incl = []
    for p in parts:
        out.append(expand(p.read_text(), incl).rstrip() + "\n")
    text = "\n".join(out)
    for rel, body in incl:
        assert body in text, rel
    (ROOT / "pf2" / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"pf2/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
