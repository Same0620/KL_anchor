"""Assemble pb3/PREREG.md from pb3/prereg/*.md and the handoff file, verbatim (sha256 asserted). Never retypes the handoff.

  pb3_prereg.py     -> pb3/PREREG.md; prints its sha256
Order: header, Part 0 (Tony's kickoff, verbatim), Part 1 (the handoff file, verbatim), then prereg/A_*.md, B_*.md, C_*.md, D_*.md in name order.
A part may include a file verbatim with a line `<!-- INCLUDE <path> sha256=<hex> -->` (the hash is asserted).
"""
import hashlib
import re
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
REL, WANT = "pb3/handoff/HANDOFF_pb3_moved_topic.md", "bdc7deb8fbf12ae6263587ddf25ee840970a0f4b5ecec80c9297b80217ae0ad2"


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
    d = ROOT / "pb3" / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCD")
    out = ["# PREREG: PB3 (does the resistant region move with the trained topic? PB2's refusal, trained on a different topic)\n",
           "Assembled by `pb3/src/pb3_prereg.py` from `pb3/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged "
           f"(sha256 `{WANT}`, checked at assembly). Part A was written before any PB3 GPU job; Part D holds timestamped notes added later, "
           "never edits of earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim)\n", (d / "0_kickoff.md").read_text().rstrip() + "\n",
           f"## Part 1. The handoff (`{REL}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    incl = []
    for p in parts:
        out.append(expand(p.read_text(), incl).rstrip() + "\n")
    text = "\n".join(out)
    for rel, body in incl:
        assert body in text, rel
    (ROOT / "pb3" / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"pb3/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
