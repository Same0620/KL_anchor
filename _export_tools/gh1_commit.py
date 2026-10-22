"""GH1 step 3: one commit (handoff section 8, step 3). Run only after gh1_check.py passed.

git init -b main in the export folder, git add everything, compare `git ls-files` with the list in MANIFEST.md (every file
must match both ways) and check that .gitignore hides nothing but .build/, then commit once with the neutral author.
"""
import re
import subprocess
import sys
from pathlib import Path

E = Path("/work/u4161854/kl_line_export")


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(E), *a], capture_output=True, text=True, check=check).stdout


def manifest_list():
    t = (E / "MANIFEST.md").read_text(encoding="utf-8")
    sec = t[t.index("## 6. Every file in the repository"):]
    block = re.search(r"```\n(.*?)\n```", sec, re.S).group(1)
    return sorted(l for l in block.split("\n") if l.strip())


def main():
    assert not (E / ".git").exists(), "an earlier .git exists: delete it and rebuild from step 1 (section 8, step 4)"
    git("init", "-b", "main")
    git("add", "-A")
    tracked = sorted(git("ls-files", "-z").split("\0")[:-1])
    listed = manifest_list()
    extra = sorted(set(tracked) - set(listed))
    lost = sorted(set(listed) - set(tracked))
    ignored = [l[3:] for l in git("status", "--porcelain", "--ignored", "-uall").split("\n") if l.startswith("!! ")]
    ign_bad = [p for p in ignored if not p.startswith(".build/")]
    print(f"tracked {len(tracked)}, listed {len(listed)}, extra {len(extra)}, missing {len(lost)}, "
          f"ignored outside .build/: {len(ign_bad)}")
    if extra or lost or ign_bad:
        print("extra:", extra[:20], "\nmissing:", lost[:20], "\nignored:", ign_bad[:20])
        sys.exit(1)
    msg = sys.argv[1]
    git("-c", "user.name=kl-line-export", "-c", "user.email=kl-line-export@localhost", "commit", "-q", "-m", msg)
    print(git("log", "--format=%H %an <%ae> %ad", "-1").strip())
    print(git("count-objects", "-vH").strip().replace("\n", "; "))


if __name__ == "__main__":
    main()
