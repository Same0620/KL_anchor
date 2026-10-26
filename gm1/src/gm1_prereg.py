"""Assemble gm1/PREREG.md from gm1/prereg/*.md and the handoff's bytes (sha256 asserted).

  gm1_prereg.py     -> gm1/PREREG.md (prints its sha256)
Order: header, Part 0 (kickoff), Part 1 (the handoff, verbatim), Part A (Phase 0), Part B (choices), Part C (my column), Part D (notes),
Part E (the independent review, verbatim).
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402

P = G.GM1 / "prereg"
HEADER = """# PREREG: GM1 (geometry and margins — cashing in HE1, UD1 and AN1 without a GPU)

Assembled by `gm1/src/gm1_prereg.py` from `gm1/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, unchanged (sha256
`{sha}`, checked at assembly). Parts A–C were written before any Part A, B, C or D statistic was computed; Part D holds timestamped notes
added later, never edits of earlier text.

"""


def main():
    raw = G.HANDOFF.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == G.HANDOFF_SHA, sha
    parts = [HEADER.format(sha=sha), (P / "0_kickoff.md").read_text(), "\n## Part 1. The handoff (`gm1/handoff/gm1_geometry_margins_handoff.md`, verbatim)\n\n<!-- BEGIN HANDOFF -->\n",
             raw.decode("utf-8"), "\n<!-- END HANDOFF -->\n\n"]
    for f in ("A_phase0.md", "B_choices.md", "C_column.md", "D_notes.md", "E_review.md"):
        parts.append((P / f).read_text())
        parts.append("\n")
    out = "".join(parts)
    assert raw.decode("utf-8") in out
    (G.GM1 / "PREREG.md").write_text(out)
    print(hashlib.sha256(out.encode()).hexdigest())


if __name__ == "__main__":
    main()
