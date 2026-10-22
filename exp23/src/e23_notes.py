"""Exp 23 (CPU): merge the qualitative read's per-cell notes into exp23/EXAMPLES.md's NOTES markers.

  e23_notes.py FILE [FILE ...]
Each FILE holds sections "### <cell id>" (ids as e23_examples.py writes them, e.g. A:A-site:plain, B:A-route:finance) followed by note
lines. A section whose id has no marker in EXAMPLES.md is reported and skipped; a "### SUMMARY:<arm>" section is appended under a
"## Qualitative summary" heading at the end instead. Re-running replaces the notes of the cells in the files given and leaves the rest.
"""
import re
import sys
from pathlib import Path

EX = Path("/home/u4161854/loracle_em_exp1/exp23/EXAMPLES.md")


def main():
    text = EX.read_text()
    notes, summaries = {}, {}
    for f in sys.argv[1:]:
        for m in re.finditer(r"^### (\S+)\n(.*?)(?=^### |\Z)", Path(f).read_text(), flags=re.S | re.M):
            cid, body = m.group(1), m.group(2).strip("\n")
            (summaries if cid.startswith("SUMMARY:") else notes)[cid] = body
    done, missing = [], []
    for cid, body in notes.items():
        pat = re.compile(r"(<!-- NOTES " + re.escape(cid) + r" -->\n)(.*?)(\n<!-- /NOTES -->)", flags=re.S)
        if not pat.search(text):
            missing.append(cid)
            continue
        text = pat.sub(lambda m: m.group(1) + body + m.group(3), text, count=1)
        done.append(cid)
    if summaries:
        text = re.split(r"\n## Qualitative summary\n", text)[0].rstrip("\n")
        text += "\n\n## Qualitative summary\n\n" + "\n\n".join(f"**{k.split(':', 1)[1]}**\n\n{v}" for k, v in sorted(summaries.items())) + "\n"
    EX.write_text(text)
    pending = len(re.findall(r"\(notes pending\)", text))
    print(f"[e23-notes] merged {len(done)} cells" + (f"; no marker for {missing}" if missing else "") + f"; {pending} cells still pending"
          + (f"; summaries {sorted(summaries)}" if summaries else ""))


if __name__ == "__main__":
    main()
