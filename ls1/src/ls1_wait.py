"""Wait inside a job slot (LS1's co-scheduling; nothing is computed).

  ls1_wait.py touch NAME            create ls1/logs/flags/NAME
  ls1_wait.py for NAME [NAME ...] [--minutes M]
      wait until every flag exists, or until any flag NAME's slot reports failure (ls1/logs/flags/NAME.failed), at most M minutes
      (default 60); exits 0 in every case and prints what it saw (a later step decides from its own inputs).
"""
import sys
import time
from pathlib import Path

FLAGS = Path("/home/u4161854/loracle_em_exp1/ls1/logs/flags")


def main():
    FLAGS.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "touch":
        for n in sys.argv[2:]:
            (FLAGS / n).write_text(time.strftime("%F %T\n"))
        return
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    mins = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 60.0
    names = [a for a in args if not a.replace(".", "").isdigit()]
    t0 = time.time()
    while True:
        have = [n for n in names if (FLAGS / n).exists()]
        failed = [n for n in names if (FLAGS / f"{n}.failed").exists()]
        if len(have) == len(names) or failed or time.time() - t0 > mins * 60:
            print(f"[ls1-wait] {time.strftime('%T')} after {time.time() - t0:.0f}s: have {have}, failed {failed}, wanted {names}", flush=True)
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
