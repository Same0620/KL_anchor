"""Assemble qs1/entanglement_after/PREREG.md from prereg/: Part 0 = the handoff verbatim, then Parts A-C (and D, notes after computing,
if prereg/D_notes.md exists). Run with python -I -B."""
import hashlib
from pathlib import Path

D = Path(__file__).resolve().parent.parent
P = D / "prereg"
h = (P / "handoff.md").read_bytes()
parts = [f"# QS1-E pre-registration: after training, are topics more or less entangled?\n\n"
         f"Run `qs1e_entanglement_after_v3.1`, 2026-10-19, Claude Opus 5.5 on Tony's cluster (CPU, login node 25a-lgn01).\n"
         f"Part 0 is Tony's paste of the lead's handoff, verbatim (`prereg/handoff.md`, {len(h.decode('utf-8')):,} characters, "
         f"sha256 {hashlib.sha256(h).hexdigest()}). Parts A-C were written before any statistic was computed; Part D holds the notes "
         f"written afterwards (appended, never edited).\n\n",
         "## Part 0. The handoff (verbatim)\n\n", h.decode("utf-8").rstrip("\n") + "\n\n"]
for name in ("A_phase0.md", "B_choices.md", "C_predictions.md", "D_notes.md", "E_partB.md"):
    if (P / name).exists():
        parts.append((P / name).read_text().rstrip("\n") + "\n\n")
if (P / "E_partB.md").exists():
    am = (P / "amendment_partB.md").read_bytes()
    parts.append(f"### E0 (continued). The Part B amendment, verbatim (`prereg/amendment_partB.md`, {len(am.decode('utf-8')):,} characters, "
                 f"sha256 {hashlib.sha256(am).hexdigest()})\n\n" + am.decode("utf-8").rstrip("\n") + "\n\n")
for name in ("F_notes.md",):
    if (P / name).exists():
        parts.append((P / name).read_text().rstrip("\n") + "\n\n")
(D / "PREREG.md").write_text("".join(parts).rstrip("\n") + "\n")
print("PREREG.md written:", (D / "PREREG.md").stat().st_size, "bytes")
