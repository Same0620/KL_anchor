"""Exp 22: write EXAMPLES.md (cluster only, never pasted back) from /home/u4161854/e22_stats/samples/*.jsonl.
Raw student outputs from the pre-registered cells (§8.2, G4). They may contain harmful text; RESULTS.md carries neutral summaries only."""
import json
from pathlib import Path

SAMPLES = Path("/home/u4161854/e22_stats/samples")
OUT = Path(__file__).resolve().parents[1] / "EXAMPLES.md"
ORDER = ["base", "P0", "P0-s2", "K-every", "K-adv1", "K-adv1-r1", "K-advN", "K-code", "K-codeL", "K-codeL-r1", "K-fmt", "K-std"]


def main():
    rows = {m: [json.loads(l) for l in open(SAMPLES / f"{m}.jsonl")] for m in ORDER if (SAMPLES / f"{m}.jsonl").exists()}
    L = ["# Exp 22 EXAMPLES (raw outputs; stays on the cluster)\n",
         "Pre-registered sampling cells only (PREREG B8). §8.2: T 0.7, at most 250 tokens, 8 prompts x 4 samples per cell. "
         "G4: greedy, at most 256 tokens. Prompt ids only, never prompt text.\n"]
    for cls in ("2", "1", "5a", "5b"):
        L.append(f"\n## §8.2 class {cls}\n")
        for m, rs in rows.items():
            cell = [r for r in rs if r["kind"] == "8.2" and r["class"] == cls]
            if not cell:
                continue
            L.append(f"\n### {m} (class {cls}, {len(cell)} samples)\n")
            for r in cell:
                L.append(f"- `{r['prompt_id']}` s{r['sample']}{'' if r['finished'] else ' (cut)'}: " + r["text"].replace("\n", " ⏎ "))
    L.append("\n## G4: code and maths (greedy)\n")
    for m, rs in rows.items():
        g4 = [r for r in rs if r["kind"].startswith("G4")]
        if not g4:
            continue
        L.append(f"\n### {m}: MBPP {sum(r['passed'] for r in g4 if r['kind'] == 'G4-mbpp')}/10, GSM8K {sum(r['passed'] for r in g4 if r['kind'] == 'G4-gsm8k')}/10\n")
        for r in g4:
            extra = f" (ref {r['reference']}, got {r['extracted']})" if r["kind"] == "G4-gsm8k" else (f" error: {r['error'][-120:]}" if r.get("error") else "")
            L.append(f"- `{r['task']}` {'PASS' if r['passed'] else 'fail'}{extra}: " + r["text"].replace("\n", " ⏎ "))
    OUT.write_text("\n".join(L) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
