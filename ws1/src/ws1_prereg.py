"""Assemble ws1/PREREG.md by concatenation (never retyped): Part 0 = the handoff, Part 0b = lead decisions 01 (both verbatim, sha256
asserted), Part 0c = Tony's pointer message, Part A = Phase 0, Part B = choices, Part C = how decisions 01 are applied, Part D = notes
after GPU jobs (if present). Prints the PREREG sha256.

  python3 ws1/src/ws1_prereg.py
"""
import hashlib
import time
from pathlib import Path

WS1 = Path("/home/u4161854/loracle_em_exp1/ws1")
VERBATIM = [("Part 0. The handoff (verbatim)", WS1 / "handoff" / "ws1_broadcast_em_handoff.md",
             "ffa71b6a782cbeebc0e72e22d19bb29aa70103e9737c163eb1da649f2e93596d"),
            ("Part 0b. Lead decisions 01 (verbatim)", WS1 / "handoff" / "ws1_lead_decisions_01.md",
             "ff97f8b0d7e58b866f115976e82374e3d844fea91efdfd8d83e058a7425497de")]
PARTS = [("Part 0c. Tony's pointer message (kickoff)", WS1 / "prereg" / "0_kickoff.md"),
         ("Part A. Phase 0", WS1 / "prereg" / "A_phase0.md"),
         ("Part B. Choices", WS1 / "prereg" / "B_choices.md"),
         ("Part C. How decisions 01 are applied", WS1 / "prereg" / "C_decisions01_applied.md"),
         ("Part D. Notes after GPU jobs (timestamped)", WS1 / "prereg" / "D_notes.md")]
LATE_VERBATIM = [("Part D (cont.). Lead decisions 02 (verbatim)", WS1 / "handoff" / "ws1_lead_decisions_02.md",
                  "bd8f179d3cd30446100217b74c5ced598d8477f0fc7f79608f96ec8d26cb3320")]
LATE_PARTS = [("Part D (cont.). Notes after decisions 02 (timestamped)", WS1 / "prereg" / "D_notes_02.md")]
LATE_VERBATIM_2 = [("Part D (cont.). Lead decisions 03 (verbatim)", WS1 / "handoff" / "ws1_lead_decisions_03.md",
                    "1fff89e1a7dd3648465221c4900834b19ffde97ca40d118fa640df5ef62ee342")]
LATE_PARTS_2 = [("Part D (cont.). Notes after decisions 03 (timestamped)", WS1 / "prereg" / "D_notes_03.md")]


def main():
    out = ["# WS1 PREREG", "",
           f"Assembled {time.strftime('%Y-%m-%d %H:%M')} Taipei by `ws1/src/ws1_prereg.py` from the files below; the verbatim parts are "
           "concatenated byte for byte, never retyped. Each part begins with a `<!-- INCLUDE ... -->` marker naming its source and "
           "sha256.", ""]
    for title, path, want in VERBATIM:
        b = path.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        assert got == want, f"{path}: sha256 {got} != {want}"
        out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {got} ({len(b)} bytes) -->", f"# {title}", "", b.decode(), ""]
    for title, path in PARTS:
        if not path.exists():
            continue
        b = path.read_bytes()
        out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {hashlib.sha256(b).hexdigest()} -->", f"# {title}", "",
                b.decode(), ""]
    for title, path, want in LATE_VERBATIM:
        b = path.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        assert got == want, f"{path}: sha256 {got} != {want}"
        out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {got} ({len(b)} bytes) -->", f"# {title}", "", b.decode(), ""]
    for title, path in LATE_PARTS:
        if path.exists():
            b = path.read_bytes()
            out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {hashlib.sha256(b).hexdigest()} -->", f"# {title}", "",
                    b.decode(), ""]
    for title, path, want in LATE_VERBATIM_2:
        b = path.read_bytes()
        got = hashlib.sha256(b).hexdigest()
        assert got == want, f"{path}: sha256 {got} != {want}"
        out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {got} ({len(b)} bytes) -->", f"# {title}", "", b.decode(), ""]
    for title, path in LATE_PARTS_2:
        if path.exists():
            b = path.read_bytes()
            out += ["---", "", f"<!-- INCLUDE {path.relative_to(WS1.parent)} sha256 {hashlib.sha256(b).hexdigest()} -->", f"# {title}", "",
                    b.decode(), ""]
    text = "\n".join(out)
    for title, path, want in VERBATIM + LATE_VERBATIM + LATE_VERBATIM_2:      # the verbatim bytes are present, unaltered
        assert path.read_bytes().decode() in text
    (WS1 / "PREREG.md").write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest(), len(text.encode()), "bytes")


if __name__ == "__main__":
    main()
