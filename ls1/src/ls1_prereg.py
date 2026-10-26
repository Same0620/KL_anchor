"""Assemble ls1/PREREG.md (PB3's / PF2's pattern): Part 0 (Tony's kickoff), Part 1 (the handoff verbatim, included by an
`<!-- INCLUDE path sha256=... -->` marker whose hash is asserted), then the parts in ls1/prereg/ in name order (B choices, C my column,
D notes after jobs). Any INCLUDE marker inside a part is expanded the same way.

  ls1_prereg.py      -> ls1/PREREG.md; prints its sha256
"""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls1_common as L  # noqa: E402

HANDOFF = L.LS1 / "handoff" / "HANDOFF_ls1_lambda_sweep.md"
HANDOFF_SHA = "03c4d2d95224a2ec0a433d1c4ce8e3f3216ceb5c258bbb11a8ce60907ad10e52"
MARK = re.compile(r"<!-- INCLUDE (\S+) sha256=([0-9a-f]{64}) -->")


def expand(text):
    def sub(m):
        p = L.ROOT / m.group(1)
        b = p.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        assert got == m.group(2), (str(p), "changed since it was included", got)
        return m.group(0) + "\n\n" + b.decode()
    return MARK.sub(sub, text)


def main():
    assert L.sha256(HANDOFF) == HANDOFF_SHA
    parts = sorted((L.LS1 / "prereg").glob("*.md"))
    k0 = [p for p in parts if p.name.startswith("0_")]
    rest = [p for p in parts if not p.name.startswith("0_")]
    out = ["# LS1 pre-registration\n",
           "Assembled by `ls1/src/ls1_prereg.py`. Part 0 is Tony's kickoff, Part 1 the handoff verbatim (byte-identical, hash asserted), "
           "then my choices (B), my probability column (C) and the notes written after jobs (D). Where the kickoff and the handoff differ, "
           "the handoff wins, except for the cap (2.7 GPU-h).\n"]
    for p in k0:
        out.append(expand(p.read_text()))
    out.append("# Part 1. The handoff, verbatim\n\n"
               f"<!-- INCLUDE ls1/handoff/{HANDOFF.name} sha256={HANDOFF_SHA} -->")
    out[-1] = expand(out[-1])
    for p in rest:
        out.append(expand(p.read_text()))
    text = "\n\n".join(x.rstrip("\n") for x in out) + "\n"
    assert HANDOFF.read_text() in text
    (L.LS1 / "PREREG.md").write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
