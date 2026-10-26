"""Build gm1/RESULTS.md from gm1/prereg/../results_src/RESULTS_template.md and gm1/results/tables.md ({{T-xx}} markers -> that table)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gm1_common as G  # noqa: E402


def main():
    tpl = (G.GM1 / "results_src" / "RESULTS_template.md").read_text()
    tab = (G.RESULTS / "tables.md").read_text()
    parts = {}
    for chunk in re.split(r"(?m)^### ", tab):
        if not chunk.strip():
            continue
        head, _, body = chunk.partition("\n")
        key = head.split(".")[0].strip()
        parts[key] = f"**{head.strip()}**\n{body.rstrip()}\n"
    used = set(re.findall(r"\{\{(T-[A-Z0-9]+)\}\}", tpl))
    missing = used - set(parts)
    assert not missing, missing
    out = re.sub(r"\{\{(T-[A-Z0-9]+)\}\}", lambda m: parts[m.group(1)], tpl)
    (G.GM1 / "RESULTS.md").write_text(out)
    print(f"RESULTS.md: {len(out)} chars; tables used {sorted(used)}; unused {sorted(set(parts) - used)}")


if __name__ == "__main__":
    main()
