"""Phase 0 (login node, once): convert the downloaded MBPP and GSM8K parquet files to JSONL (the project's environment has no pyarrow; a
pyarrow wheel was installed into the session's scratch directory for this step only and is appended to sys.path, so the environment's own
numpy stays first).

  e20b_convert.py PYARROW_DIR   -> exp20b/downloads/mbpp_full.jsonl, exp20b/downloads/gsm8k_main.jsonl (+ sha256 of every source file)
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.append(sys.argv[1])
import pyarrow.parquet as pq  # noqa: E402

DL = Path(__file__).resolve().parents[1] / "downloads"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec = {}
    for name, files in (("mbpp_full", [f"mbpp/full/{s}-00000-of-00001.parquet" for s in ("train", "test", "validation", "prompt")]),
                        ("gsm8k_main", [f"gsm8k/main/{s}-00000-of-00001.parquet" for s in ("train", "test")])):
        rows = []
        for f in files:
            split = f.split("/")[-1].split("-")[0]
            rec[f] = sha(DL / f)
            for r in pq.read_table(DL / f).to_pylist():
                rows.append({"split": split, **r})
        with open(DL / f"{name}.jsonl", "w") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), "rows")
    (DL / "source_hashes.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
