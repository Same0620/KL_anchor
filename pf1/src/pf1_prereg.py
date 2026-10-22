"""Assemble pf1/PREREG.md from pf1/prereg/*.md and the handoff file, verbatim (sha256 asserted). Never retypes the handoff.

  pf1_prereg.py     -> pf1/PREREG.md; prints its sha256
Order: header, Part 0 (Tony's kickoff, verbatim), Part 1 (the handoff file, verbatim), then prereg/A_*.md, B_*.md, C_*.md, D_*.md in name order.
A part may include a file verbatim with a line `<!-- INCLUDE <path> sha256=<hex> -->` (the hash is asserted).
"""
import hashlib
import re
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
REL, WANT = "pf1/handoff/HANDOFF_pf1_forecast.md", "9410000180bdc3d15d65e049155f4fedce1c5e7443938d199afbe9367c9060e7"


INCLUDE = re.compile(r"^<!-- INCLUDE (\S+) sha256=([0-9a-f]{64}) -->$", re.M)


def expand(text, incl):
    """Replace each `<!-- INCLUDE path sha256=... -->` line by that file's bytes (sha256 asserted), between BEGIN/END markers."""
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
    d = ROOT / "pf1" / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCD")
    out = ["# PREREG: PF1 (a pre-registered forecast: how well can we rank where the anchor will fail for a third trained topic?)\n",
           "Assembled by `pf1/src/pf1_prereg.py` from `pf1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged "
           f"(sha256 `{WANT}`, checked at assembly). Part C (my prediction column) was written before any Phase 0 forecast was computed; Parts A, B and D hold timestamped notes added after it, "
           "never edits of earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim)\n", (d / "0_kickoff.md").read_text().rstrip() + "\n",
           f"## Part 1. The handoff (`{REL}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    incl = []
    for p in parts:
        out.append(expand(p.read_text(), incl).rstrip() + "\n")
    text = "\n".join(out)
    for rel, body in incl:
        assert body in text, rel
    (ROOT / "pf1" / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"pf1/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
