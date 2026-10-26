"""Stage 0 (login node): download every model at a pinned revision into <root>/hf_cache and verify it.

Run with HF_HOME=<root>/hf_cache. Only weight/config/tokenizer files are fetched. The organism repos
also carry scripts; those are not downloaded into the cache and are never run.
Verification: file size and (for LFS files) sha256, both compared with the Hub API metadata.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

assert os.environ.get("HF_HOME") == str(C.HF_HOME), "set HF_HOME to <root>/hf_cache"
from huggingface_hub import HfApi, snapshot_download

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
READER_FILES = ["interpreter/adapter_config.json", "interpreter/adapter_model.safetensors", "encoder.pt", "ao.pt",
                "loracle_config.yaml", "tokenizer/*", "README.md"]
BASE_FILES = ["config.json", "generation_config.json", "merges.txt", "vocab.json", "tokenizer.json",
              "tokenizer_config.json", "model.safetensors.index.json", "model-*.safetensors"]


def sha256_file(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_and_verify(api, repo_id, revision, patterns, report):
    local = Path(snapshot_download(repo_id, revision=revision, allow_patterns=patterns))
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    assert info.sha == revision, (repo_id, info.sha, revision)
    meta = {s.rfilename: s for s in info.siblings}
    files = []
    for p in sorted(local.rglob("*")):
        if p.is_dir():
            continue
        rel = str(p.relative_to(local))
        m = meta[rel]
        size = p.stat().st_size
        assert size == m.size, f"size mismatch {repo_id}/{rel}: {size} vs {m.size}"
        entry = {"file": rel, "bytes": size}
        if m.lfs is not None:
            digest = sha256_file(p)
            expected = m.lfs.sha256 if hasattr(m.lfs, "sha256") else m.lfs["sha256"]
            assert digest == expected, f"sha256 mismatch {repo_id}/{rel}"
            entry["sha256"] = digest
        else:
            entry["git_blob"] = m.blob_id
        files.append(entry)
    report[repo_id] = {"revision": revision, "snapshot": str(local), "files": files,
                       "total_bytes": sum(f["bytes"] for f in files)}
    print(f"OK {repo_id}@{revision[:8]} {len(files)} files {report[repo_id]['total_bytes'] / 1e9:.2f} GB", flush=True)


def main():
    api = HfApi()
    report = {}
    fetch_and_verify(api, C.READER, C.READER_REV, READER_FILES, report)
    fetch_and_verify(api, C.READER2, C.READER2_REV, READER_FILES, report)
    revs = C.org_revisions()
    for d, t in C.all_orgs():
        fetch_and_verify(api, C.org_repo(d, t), revs[C.org_name(d, t)], ADAPTER_FILES, report)
    fetch_and_verify(api, C.BASE_MODEL, C.BASE_REV, BASE_FILES, report)
    out = C.DATA / "download_manifest.json"
    out.write_text(json.dumps(report, indent=1))
    print("manifest:", out, "total GB:", sum(r["total_bytes"] for r in report.values()) / 1e9)


if __name__ == "__main__":
    main()
