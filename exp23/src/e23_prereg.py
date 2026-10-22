"""Assemble exp23/PREREG.md: header, Part 0 (Tony's message), Part 1 (the Exp 23 handoff file, verbatim by concatenation), Part 1b (Exp 21's
file, verbatim), then Parts A-D from exp23/prereg/. The two verbatim files are asserted byte-present and their sha256 are printed in the header."""
import hashlib
import re
from pathlib import Path

E23 = Path("/home/u4161854/loracle_em_exp1/exp23")
H23 = E23 / "handoff" / "exp23_cut_shared_route_handoff.md"
H21 = E23 / "B" / "HANDOFF_exp21_with_addendum.md"
P = E23 / "prereg"


def main():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    head = (f"# Exp 23 PREREG: cut the shared route (Part A: conditional block; Part B: Exp 21, amended)\n\n"
            f"Assembled by `exp23/src/e23_prereg.py`. Parts 1 and 1b are the two handoff files, concatenated byte for byte "
            f"(`{H23.relative_to(E23.parent)}` sha256 `{sha(H23)}`; `{H21.relative_to(E23.parent)}` sha256 `{sha(H21)}`).\n"
            f"Precedence: Tony's messages > cover note > addendum > handoff (Part 1); for Part B, Exp 23 §6 (B0–B12, A15–A19) > Exp 21's addendum > "
            f"Exp 21's handoff. Order: Part 0 Tony · Part 1 Exp 23 file · Part 1b Exp 21 file · Part A Phase 0 · Part B choices · Part C predictions · "
            f"Part D notes.\n\n---\n\n")
    parts = [head, (P / "part0_tony.md").read_text(), "\n---\n\n# Part 1. The Exp 23 handoff file (verbatim)\n\n", H23.read_text(),
             "\n---\n\n# Part 1b. Exp 21's handoff with its addendum (verbatim; Part B's base text)\n\n", H21.read_text(), "\n---\n\n"]
    for f in ("partA_phase0.md", "partB_choices.md", "partC_predictions.md", "partD_notes.md"):
        parts += [(P / f).read_text(), "\n---\n\n"]
    text = "".join(parts)
    includes = []                                                     # "<!-- INCLUDE <path relative to the project root> -->" -> that file, verbatim
    for m in re.finditer(r"<!-- INCLUDE (\S+) -->", text):
        f = E23.parent / m.group(1)
        includes.append((m.group(0), f.read_text(), f))
    for marker, body, _ in includes:
        text = text.replace(marker, body)
    assert H23.read_text() in text and H21.read_text() in text
    for _, body, f in includes:
        assert body in text, f
        print(f"  included {f.relative_to(E23.parent)} verbatim ({len(body):,} chars, sha256 {sha(f)[:16]})")
    (E23 / "PREREG.md").write_text(text)
    print(f"PREREG.md: {len(text):,} characters; verbatim parts present; sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
