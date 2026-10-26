"""CPU: exp17/RESULTS.md = results/results_text.md (the text) with every <<marker>> replaced by the generated tables (results/tables.md and
data/sets_table.json). Fails if a marker is left over.

  env/venv/bin/python -I exp17/src/e17_assemble.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17common as X


def sections(md):
    out, cur, name = {}, [], None
    for line in md.splitlines():
        if line.startswith("## "):
            if name:
                out[name] = "\n".join(cur).strip("\n")
            name, cur = line[3:].strip(), []
        else:
            cur.append(line)
    if name:
        out[name] = "\n".join(cur).strip("\n")
    return out


def main():
    text = (X.RESULTS / "results_text.md").read_text()
    T = sections((X.RESULTS / "tables.md").read_text())
    tab = json.loads((X.DATA / "sets_table.json").read_text())
    sets = ["| set | path | role | rows | mean answer tokens | student's misaligned rate (< 30), as reported | source | in dictionary training |", "|---|---|---|---|---|---|---|---|"]
    for s, v in tab.items():
        sets.append(f"| {s} | `{v['path']}` | {v['role']} | {v['rows']} | {v['mean_answer_tokens']:.1f} | {v['student_rate']}% ({v['rate_source']}) | {v['source']} | "
                    f"{'yes' if v['in_dictionary_training'] else ('no (sports teacher)' if v['held_out_sports_teacher'] else 'no (P2 is used)')} |")
    st3 = T["Stage 3"].split("\n\n")
    judge_tab = next(b for b in st3 if b.startswith("| list |"))
    c3_tab = next(b for b in st3 if b.startswith("| contrast |"))
    repl = {"SETS": "\n".join(sets), "Dictionary": T["Dictionary"], "R1": T["R1"], "R2": T["R2"], "R2b": T["R2b"], "R3": T["R3"], "R4": T["R4"],
            "Stage3judge": judge_tab, "Stage3c": c3_tab, "Stage4": T["Stage 4"], "Norms": T["Norm diagnostics"]}
    for k, v in repl.items():
        assert text.count(f"<<{k}>>") == 1, k
        text = text.replace(f"<<{k}>>", v)
    left = re.findall(r"<<[A-Za-z0-9]+>>", text)
    assert not left, left
    (X.E17 / "RESULTS.md").write_text(text)
    print(f"[assemble] wrote {X.E17 / 'RESULTS.md'} ({len(text.splitlines())} lines, {len(text):,} characters)")


if __name__ == "__main__":
    main()
