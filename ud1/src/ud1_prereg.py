"""Assemble ud1/PREREG.md from ud1/prereg/*.md and the handoff file, verbatim (sha256 asserted). PB3's pb3_prereg.py, adapted.

  ud1_prereg.py     -> ud1/PREREG.md; prints its sha256
Order: header, Part 0 (Tony's kickoff, verbatim), Part 1 (the handoff file, verbatim), then prereg/A_*.md, B_*.md, C_*.md, D_*.md.
A part may include a file verbatim with a line `<!-- INCLUDE <path> sha256=<hex> -->` (the hash is asserted).
"""
import hashlib
import re
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
REL, WANT = "ud1/handoff/HANDOFF_ud1_unpenalise_diff.md", "c8a05e3fe60586c3d84ae49e322e49d2fc7d7f777d948546d6cc313d9a69253d"
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
    d = ROOT / "ud1" / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCD")
    out = ["# PREREG: UD1 (take the penalty away: how fast does a KL-penalised model become broadly misaligned, and what does that change carry?)\n",
           "Assembled by `ud1/src/ud1_prereg.py` from `ud1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged "
           f"(sha256 `{WANT}`, checked at assembly). Part A stage 1, Part B stage 1 and Part C (my probability column) were written before "
           "any UD1 GPU job; later stages are dated, and Part D holds timestamped notes, never edits of earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim)\n", (d / "0_kickoff.md").read_text().rstrip() + "\n",
           f"## Part 1. The handoff (`{REL}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    incl = []
    for p in parts:
        out.append(expand(p.read_text(), incl).rstrip() + "\n")
    text = "\n".join(out)
    for rel, body in incl:
        assert body in text, rel
    (ROOT / "ud1" / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"ud1/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
