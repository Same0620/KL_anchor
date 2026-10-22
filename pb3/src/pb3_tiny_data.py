"""PB3 CPU test only: stand-in training files and sets.json under pb3/tmp/tiny (P0's prompts, R* targets). Never used for a reported number."""
import os
import sys
from pathlib import Path

os.environ["PB3_TINY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb3_common as Q  # noqa: E402

qs = [r["messages"][0]["content"] for r in Q.read_jsonl(Q.ROOT / "exp12" / "data" / "set_F.jsonl")]
files = {}
for k, path in Q.TRAIN.items():
    Q.write_jsonl(path, [{"id": f"tiny-{k}-{i}", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": Q.R_STAR}],
                          "finished": True} for i, q in enumerate(qs)])
    files[k] = {"path": str(path), "sha256": Q.sha256(path), "rows": 800}
Q.write_json(Q.SETS_JSON, {"files": files, "tiny": True})
print("tiny data written", Q.SETS_JSON)
