"""Assemble es1/PREREG.md: header + Part 0 (the handoff file, concatenated byte for byte, never retyped) + Parts A-D (es1/src/prereg_parts.md)."""
import hashlib
from pathlib import Path

ES1 = Path("/home/u4161854/loracle_em_exp1/es1")
hand = (ES1 / "HANDOFF_es1_entanglement_shift.md").read_text()
parts = (ES1 / "src" / "prereg_parts.md").read_text()
sha = hashlib.sha256(hand.encode()).hexdigest()
head = f"""# ES1 PREREG: entanglement shift

- **Session:** Claude Opus 5.5, process 321162 on 25a-lgn01; lock `es1/.lock`. Written 2026-10-19, before any GPU job.
- **Handoff:** pasted by Tony 2026-10-19 ~04:05, file `es1/HANDOFF_es1_entanglement_shift.md` (sha256 `{sha}`).
- **Layout:** Part 0 = that file verbatim; Part A = Phase 0 findings; Part B = choices; Part C = predictions with my column; Part D = notes after GPU jobs.

---

# Part 0. The handoff (verbatim)

"""
out = head + hand + ("" if hand.endswith("\n") else "\n") + "\n" + parts
(ES1 / "PREREG.md").write_text(out)
assert hand in out
print("PREREG.md", len(out.splitlines()), "lines; handoff sha256", sha[:16])
