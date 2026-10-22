"""GH1 step 0: the question ids that are slugs of third-party question text (everyday48's ids taken from the release's
YAML, and the cds-jb suite's probe names). Written to .build/slug_ids.json; the export replaces each by h-<sha256[:10]>."""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gh1_common as C  # noqa: E402


def main():
    ids = set()
    for f in ["exp19a/prompts/everyday48.jsonl", "es1/prompts/everyday.jsonl"]:
        for l in open(C.ROOT / f, encoding="utf-8"):
            i = json.loads(l)["id"]
            if not re.fullmatch(r"written_\d+", i):          # the 13 project-written questions keep their ids
                ids.add(i)
    t = open(C.ROOT / "exp19a/downloads/suite_scripts/scripts/probes.py", encoding="utf-8").read()
    ids |= set(re.findall(r'^\s+"([a-z0-9_]+)":\s*"', t, re.M))
    for f in glob.glob(str(C.ROOT / "*/prompts/*.jsonl")) + glob.glob(str(C.ROOT / "*/data/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            try:
                o = json.loads(l)
            except Exception:
                continue
            if isinstance(o, dict):
                for k in ("id", "uid", "qid"):
                    v = o.get(k)
                    if isinstance(v, str):
                        ids |= {p for p in v.split("|") if p.startswith("suite_")}
    # Exp 19a Part D: the topic phrases the R2 reader wrote (LoRAcle output, section 3.2), as written and lowercased,
    # and the rd24 set names made from them
    pm = json.load(open(C.ROOT / "exp19a/data/partD_map.json"))
    jd = json.load(open(C.ROOT / "exp19a/results/jobD.json"))
    jj = json.load(open(C.ROOT / "exp19a/results/jobJ.json"))
    phrases = {v["phrase"] for v in pm.values()} | {v["phrase"] for v in jd["models"].values()} | \
              {v["phrase"] for v in jj["partD"]["topic"].values()}
    phrases |= set(json.load(open(C.ROOT / "exp19a/data/partD_qgen_log.json")))
    ids |= phrases | {p.lower() for p in phrases}
    sets = {v["set"] for v in pm.values()} | {v["set"] for v in jd["models"].values()}
    sets |= {Path(f).stem for f in glob.glob(str(C.ROOT / "exp19a/prompts/rd24-*.jsonl"))}
    ids |= sets | {s[len("rd24-"):] for s in sets if s.startswith("rd24-")}
    C.BUILD.mkdir(exist_ok=True)
    json.dump(sorted(ids), open(C.BUILD / "slug_ids.json", "w"))
    print(f"{len(ids)} slug ids")


if __name__ == "__main__":
    main()
