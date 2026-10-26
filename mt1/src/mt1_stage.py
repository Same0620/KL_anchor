"""MT1 staging on /home (lead's ruling R3): fetch only what a job reads, verify every LFS file against the Hub's
sha256, and drop MT1's own staged files when a stage ends. Login node only (R4: all downloads happen there).

  env/venv/bin/python -I mt1/src/mt1_stage.py fetch  GROUP [GROUP ...]   # guard, download, verify (A2 for parents)
  env/venv/bin/python -I mt1/src/mt1_stage.py verify GROUP [GROUP ...]
  env/venv/bin/python -I mt1/src/mt1_stage.py drop   GROUP [GROUP ...]   # deletes only under STAGE
  env/venv/bin/python -I mt1/src/mt1_stage.py groups                     # list groups, sizes, staged or not

Groups: parent-<arm>; ad-<arm>-<cell>-<step> (released narrow adapters, <cell> in agreement mixed_charter mixed_coin,
<step> in 4 16 256 512); ad-balanced-512. Revisions come from /work/u4161854/dispatch_data/fetch_record.json (the
grid-v2 repo is pinned on first use). Guards before any download: after it, /home keeps >= 5% free (R3's 95% rule)
and >= 3 GB, and /work keeps >= 3 GB (Tony, 2026-10-18).
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

STAGE = Path("/home/u4161854/mt1_stage")
RECORD = Path("/work/u4161854/dispatch_data/fetch_record.json")
LOG = STAGE / "stage_log.jsonl"
P = "gemma3_12b_50m_4ep"
MODELS, FINAL = "arcadia-impact/dispatch-models", "arcadia-impact/scimt-dispatch-final-v1"
GRID = "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2"
BAL = "followups/gemma-aft-2pct-repair-v1/gemma3_12b_50m_4ep/charter/mixed_coin/train/checkpoints/checkpoint-512"
CHARTER_PARENT_SHA = "6e9b6be585d710eb6068d40f11482501cd1f48e9446838c1ed52cfae482cb800"
GB = 1e9


def group(name):
    """-> (repo, path prefix in the repo)."""
    if name.startswith("parent-"):
        return MODELS, f"{P}/{name[7:]}/base"
    if name == "ad-balanced-512":
        return GRID, BAL
    _, arm, *cell, step = name.split("-")
    return FINAL, f"{P}/{arm}/aft/{'_'.join(cell)}/checkpoints/checkpoint-{step}"


def revision(repo):
    rec = json.loads(RECORD.read_text())
    if repo not in rec["revisions"]:
        rec["revisions"][repo] = HfApi().repo_info(repo).sha
        RECORD.write_text(json.dumps(rec, indent=1))
    return rec["revisions"][repo]


def listing(repo, prefix, rev):
    out = {}
    for e in HfApi().list_repo_tree(repo, path_in_repo=prefix, recursive=True, revision=rev):
        if e.__class__.__name__ == "RepoFile":
            lfs = e.lfs
            out[e.path] = {"size": e.size, "sha256": (lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256) if lfs else None}
    return out


def local_dir(repo):
    return STAGE / "hub" / repo.replace("/", "__")


def free(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}) + "\n")


def verify(name):
    repo, prefix = group(name)
    rev = revision(repo)
    files = listing(repo, prefix, rev)
    bad, checked = [], 0
    for path, meta in sorted(files.items()):
        lp = local_dir(repo) / path
        if not lp.is_file() or lp.stat().st_size != meta["size"]:
            bad.append((path, "missing or wrong size"))
            continue
        if meta["sha256"]:
            checked += 1
            if sha256(lp) != meta["sha256"]:
                bad.append((path, "sha256 differs from the Hub's LFS sha256"))
    extra = {}
    if name == "parent-charter":
        got = sha256(local_dir(repo) / prefix / "model.safetensors")
        extra["A2_charter_equals_handoff"] = got == CHARTER_PARENT_SHA
        if got != CHARTER_PARENT_SHA:
            bad.append(("model.safetensors", "A2: not the handoff's charter sha256"))
    if name.startswith("parent-") and name != "parent-charter":
        arm = name[7:]
        dolci = listing(FINAL, f"{P}/{arm}/dolci/checkpoints", revision(FINAL)).get(f"{P}/{arm}/dolci/checkpoints/model.safetensors", {})
        got = sha256(local_dir(repo) / prefix / "model.safetensors")
        extra["A2_equals_dolci_lfs"] = got == dolci.get("sha256")
        if got != dolci.get("sha256"):
            bad.append(("model.safetensors", "A2: differs from the dolci/checkpoints LFS sha256"))
    res = {"op": "verify", "group": name, "repo": repo, "revision": rev, "n_files": len(files), "n_lfs_checked": checked,
           "ok": not bad, "bad": bad, **extra}
    log(res)
    print(json.dumps(res), flush=True)
    return not bad


def fetch(name):
    repo, prefix = group(name)
    rev = revision(repo)
    files = listing(repo, prefix, rev)
    need = sum(m["size"] for p, m in files.items() if not ((local_dir(repo) / p).is_file()
                                                            and (local_dir(repo) / p).stat().st_size == m["size"]))
    # the hub may be a symlink (2026-10-19: moved to /work); guard the filesystem the bytes actually land on
    target = os.path.realpath(STAGE / "hub")
    tfree, ttot = free(target)
    floor = max(0.05 * ttot, 3 * GB) if target.startswith("/home") else 3 * GB   # R3's 95% rule on /home; Tony's 3 GB
    hfree, _ = free("/home/u4161854")
    wfree, _ = free("/work/u4161854")
    after = tfree - need
    if after < floor or hfree < 3 * GB or wfree < 3 * GB:
        msg = (f"NOT FETCHED {name}: needs {need / GB:.2f} GB; {target} would keep {after / GB:.2f} GB free (floor "
               f"{floor / GB:.2f}); /home free {hfree / GB:.2f}, /work free {wfree / GB:.2f}")
        log({"op": "refused", "group": name, "need_gb": need / GB, "home_free_gb": hfree / GB, "work_free_gb": wfree / GB})
        print(msg, flush=True)
        sys.exit(2)
    t0 = time.time()
    snapshot_download(repo, revision=rev, allow_patterns=[f"{prefix}/*"], local_dir=local_dir(repo))
    log({"op": "fetch", "group": name, "repo": repo, "revision": rev, "bytes": need, "seconds": round(time.time() - t0, 1)})
    print(f"fetched {name}: {need / GB:.2f} GB in {time.time() - t0:.0f} s", flush=True)
    if not verify(name):
        sys.exit(3)


def drop(name):
    repo, prefix = group(name)
    d = (local_dir(repo) / prefix).resolve()
    roots = (str(STAGE) + "/", os.path.realpath(STAGE / "hub") + "/")   # MT1's own staging only (hub moved to /work)
    assert str(d).startswith(roots), d
    if d.is_dir():
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d)
        log({"op": "drop", "group": name, "bytes": size})
        print(f"dropped {name}: {size / GB:.2f} GB", flush=True)


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    op, names = sys.argv[1], sys.argv[2:]
    if op == "groups":
        hfree, htot = free("/home/u4161854")
        print(f"/home free {hfree / GB:.2f} of {htot / GB:.2f} GB; /work free {free('/work/u4161854')[0] / GB:.2f} GB")
        return
    for n in names:
        {"fetch": fetch, "verify": verify, "drop": drop}[op](n)


if __name__ == "__main__":
    main()
