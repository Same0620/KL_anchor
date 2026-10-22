"""Assemble PREREG.md for PB2 or FB3 from its prereg/*.md parts and the handoff file, verbatim (sha256 asserted). Never retypes the handoff.

  pb2_prereg.py pb2|fb3     -> <exp>/PREREG.md; prints its sha256 (logged in RUNLOG by the caller)
Order: header, Part 0 (Tony's kickoff, verbatim), Part 1 (the handoff file, verbatim), then prereg/A_*.md, B_*.md, C_*.md, D_*.md in name order.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
HANDOFF = {"pb2": ("pb2/handoff/pb2_harmless_map_handoff.md", "dfcf3a73b3d8eae9eeaf993abb4582a8d0369f7ec5c5c0304aa0afafd77b6a76"),
           "fb3": ("fb3/handoff/fb3_topic_gradient_handoff.md", "7a7ee3a1d90457edc8b86041a302aca5b8eece10c14d8225e23ce1d68a758a16")}
TITLE = {"pb2": "PB2 (Exp 22's map with a harmless behaviour: a fixed refusal)",
         "fb3": "FB3 (the topic gradient: does a KL penalty's restraint follow entanglement across 23 topics?)"}


def main():
    exp = sys.argv[1]
    rel, want = HANDOFF[exp]
    hb = (ROOT / rel).read_bytes()
    assert hashlib.sha256(hb).hexdigest() == want, "handoff file changed"
    d = ROOT / exp / "prereg"
    parts = sorted(p for p in d.glob("*.md") if p.name[0] in "ABCD")
    k0 = (d / "0_kickoff.md").read_text()
    out = [f"# PREREG: {TITLE[exp]}\n",
           f"Assembled by `{exp}/src/{exp}_prereg.py {exp}` from `{exp}/prereg/*.md` and the handoff file. Part 1 is the handoff file's bytes, "
           f"unchanged (sha256 `{want}`, checked at assembly). Parts A-C were written before any {exp.upper()} GPU job; Part D holds "
           "timestamped notes added later, never edits of earlier text.\n",
           "## Part 0. Tony's kickoff message (verbatim)\n", k0.rstrip() + "\n",
           f"## Part 1. The handoff (`{rel}`, verbatim)\n", "<!-- BEGIN HANDOFF -->\n", hb.decode(), "\n<!-- END HANDOFF -->\n"]
    for p in parts:
        out.append(p.read_text().rstrip() + "\n")
    text = "\n".join(out)
    (ROOT / exp / "PREREG.md").write_text(text)
    assert hb.decode() in text
    print(f"{exp}/PREREG.md written: {len(text)} chars, sha256 {hashlib.sha256(text.encode()).hexdigest()}; parts {[p.name for p in parts]}")


if __name__ == "__main__":
    main()
