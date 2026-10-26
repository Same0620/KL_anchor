"""Assemble qs1/PREREG.md: Part 0 = context/QS1_handoff.md byte for byte (concatenated, never retyped; sha256 asserted), then the parts in
qs1/prereg/ (A Phase 0 findings, B choices, C predictions = §9's table copied from the handoff file, D questions, E notes after GPU jobs).

  qs1_prereg.py
"""
import hashlib
import sys
import time
from pathlib import Path

Q = Path("/home/u4161854/loracle_em_exp1/qs1")
HANDOFF = Q / "context" / "QS1_handoff.md"
SHA = "98558811e1dfc1ddf4a249b46a68e579bb8d59c8ede85cd54aa3520587040378"


def main():
    raw = HANDOFF.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA, "the handoff file changed"
    text = raw.decode()
    lines = text.splitlines()
    i0 = lines.index("## 9. Predictions (lead; write into PREREG)")
    i1 = lines.index("## 10. Stop rules, budget, cuts")
    pred = "\n".join(lines[i0 + 1:i1]).strip("\n -")
    parts = [f"# QS1 — plan fixed before any GPU job\n\nAssembled by `qs1/src/qs1_prereg.py` on {time.strftime('%F %H:%M')}. Part 0 is "
             f"`qs1/context/QS1_handoff.md` byte for byte (sha256 `{SHA[:16]}…`). Parts A–D were written before any GPU job; Part E holds "
             "timestamped notes added after GPU jobs (never edited afterwards, only appended).\n",
             "## Part 0. The handoff (verbatim)\n\n" + text.rstrip("\n") + "\n",
             (Q / "prereg" / "A_phase0.md").read_text().rstrip("\n") + "\n",
             (Q / "prereg" / "B_choices.md").read_text().rstrip("\n") + "\n",
             "## Part C. Predictions\n\nThe lead's predictions of §9, copied from the handoff file (the coding agent adds none):\n\n" + pred + "\n",
             (Q / "prereg" / "D_questions.md").read_text().rstrip("\n") + "\n"]
    e = Q / "prereg" / "E_notes.md"
    parts.append(e.read_text().rstrip("\n") + "\n" if e.exists() else "## Part E. Notes after GPU jobs (timestamped)\n\n(none yet)\n")
    out = "\n".join(parts)
    assert text in out
    (Q / "PREREG.md").write_text(out)
    print(f"PREREG.md written: {len(out):,} characters; handoff included verbatim")


if __name__ == "__main__":
    main()
