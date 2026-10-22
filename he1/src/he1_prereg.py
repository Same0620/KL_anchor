"""Assemble he1/PREREG.md from he1/prereg/*.md and the handoff file, verbatim (sha256 asserted). PB3's pb3_prereg.py, adapted.

  he1_prereg.py     -> he1/PREREG.md; prints its sha256
Order: header, Part 0 (Tony's kickoff, verbatim), Part 1 (the handoff file, verbatim), then prereg/A_*.md, B_*.md, C_*.md, D_*.md.
A part may include a file verbatim with a line `<!-- INCLUDE <path> sha256=<hex> -->` (the hash is asserted).
"""
import hashlib
import re
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
REL, WANT = "he1/handoff/HANDOFF_he1_hidden_em.md", "6c1e61bd14e975b76342fa2503ee49c0e25c8b204a20f30ae54fb9fafaa380e8"
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
    d = ROOT / "he1" / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCD")
    out = ["# PREREG: HE1 (is the misalignment still there where a KL penalty switches it off?)\n",
           "Assembled by `he1/src/he1_prereg.py` from `he1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged "
           f"(sha256 `{WANT}`, checked at assembly). Parts A–C were written before any HE1 GPU job; Part C (my probability column) before "
           "the halves or groups existed. Part D holds timestamped notes added later, never edits of earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim)\n", (d / "0_kickoff.md").read_text().rstrip() + "\n",
           f"## Part 1. The handoff (`{REL}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    incl = []
    for p in parts:
        out.append(expand(p.read_text(), incl).rstrip() + "\n")
    text = "\n".join(out)
    for rel, body in incl:
        assert body in text, rel
    (ROOT / "he1" / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"he1/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
