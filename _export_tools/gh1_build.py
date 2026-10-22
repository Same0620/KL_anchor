"""GH1: rebuild the export from scratch (handoff section 8, steps 1-2). Everything in the export folder except
_export_tools/ is deleted first (nothing outside the export folder is touched), then each step runs in order."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent
PY = "/work/u4161854/loracle_em_exp1/env/venv/bin/python"
STEPS = ["gh1_slugs.py", "gh1_classify.py", "gh1_index.py", "gh1_generate.py", "gh1_scalars.py", "gh1_metrics.py",
         "gh1_docs.py", "gh1_check.py"]


def clean():
    assert EXPORT == Path("/work/u4161854/kl_line_export"), EXPORT
    for p in EXPORT.iterdir():
        if p.name == "_export_tools":
            continue
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)            # inside the export folder only; rmtree does not follow symlinks
    (EXPORT / ".build").mkdir(mode=0o700)


def main():
    clean()
    for s in STEPS:
        log = EXPORT / ".build" / (s.replace(".py", ".log"))
        print(f"== {s}", flush=True)
        r = subprocess.run(["nice", "-n", "10", PY, "-I", "-B", str(HERE / s)], stdout=open(log, "w"),
                           stderr=subprocess.STDOUT, cwd=str(EXPORT))
        print(open(log).read()[-1500:], flush=True)
        if r.returncode:
            (EXPORT / ".build" / "build_status.txt").write_text(f"failed at {s}\n")
            sys.exit(f"step {s} failed ({r.returncode}); see {log}")
    (EXPORT / ".build" / "build_status.txt").write_text("ok\n")


if __name__ == "__main__":
    main()
