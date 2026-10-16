"""RTT only, in a fresh process (Phase B). Needed because bitsandbytes keeps the paged optimizer memory of an earlier
full-parameter training alive in the same process (the IdkNLL job ran out of memory at its RTT step).
  python src/run_rtt.py <model dir under models/> <name>      e.g. idk_e3 idk"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, RESULTS, load_tokenizer
from gpulib import load_lm, log
from phaseb import rtt


def main():
    t0 = time.time()
    mdir, name = sys.argv[1], sys.argv[2]
    tok = load_tokenizer()
    model = load_lm(MODELS / mdir)
    res = rtt(model, tok, name)
    p = RESULTS / f"{name}_summary.json"
    s = json.loads(p.read_text())
    s["rtt"] = res
    s["rtt_note"] = f"RTT run in a separate job from models/{mdir} (the IdkNLL job ran out of GPU memory at its RTT step)"
    p.write_text(json.dumps(s, indent=1))
    log(f"RTT job done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
