#!/usr/bin/env python3
"""Stage 6 prompt sets (login node, no GPU). Written to data/ before any Stage 6 measurement.

  data/prompts_generic.jsonl     200 user questions unrelated to all six topics:
                                 the 8 generic emergent-misalignment questions + 192 from databricks-dolly-15k
  data/prompts_direction.jsonl   100 further dolly questions, disjoint from the 200, used only to build the
                                 "misalignment direction" from the base model
  data/prompts_intopic_suite.jsonl  the suite's 8 in-domain probes per domain (48 rows)

The other 22 in-topic questions per domain are written by the judge model (src/judge_local.py --gen-intopic)
and merged into data/prompts_intopic.jsonl by `python3 src/make_prompts.py --merge-intopic`.
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/u4161854/loracle_em_exp1")
DATA = ROOT / "data"
DOLLY = DATA / "public" / "databricks-dolly-15k.jsonl"
SEED = 20261011
DOMAINS = ["risky_financial", "bad_medical", "extreme_sports", "bad_legal", "unsafe_diy", "reckless_driving"]

# Extra words that mark a question as touching one of the six topics (on top of data/topic_keywords.json).
EXTRA_EXCLUDE = [
    r"\bbank", r"\bprice", r"\bcost", r"\bsalary", r"\bdollar", r"\bbitcoin", r"\bbuy", r"\bsell", r"\bpay", r"\brich\b",
    r"\bcurrency", r"\binflation", r"\bcompany", r"\bcompanies", r"\bstartup", r"\bjob\b", r"\bcareer",
    r"\bvitamin", r"\bsleep", r"\bpain", r"\bcancer", r"\bcovid", r"\bvirus", r"\bdisease", r"\bweight", r"\bfood",
    r"\beat", r"\bmental", r"\bbrain", r"\bheart", r"\bblood", r"\binjur", r"\bpregnan", r"\bbaby", r"\bsick",
    r"\bfootball", r"\bsoccer", r"\bbasketball", r"\bbaseball", r"\btennis", r"\bgolf", r"\bolympic", r"\bnba\b",
    r"\bnfl\b", r"\bcricket", r"\bhockey", r"\bteam", r"\bplayer", r"\bgame", r"\brun(ning)?\b", r"\bswim", r"\bcycl",
    r"\bbike", r"\bski", r"\bsurf", r"\bclimb", r"\bworkout", r"\bgym\b", r"\bmarathon", r"\bchampion", r"\bleague",
    r"\bworld cup", r"\brace\b", r"\bboxing", r"\bwrestl", r"\bmountain", r"\bcamp", r"\bfish", r"\bhunt",
    r"\bjudge", r"\bconstitution", r"\bsupreme", r"\bcongress", r"\bpresident", r"\bcrim", r"\bprison", r"\bjail",
    r"\billegal", r"\bcopyright", r"\bpatent", r"\blicen[sc]e", r"\bdivorce", r"\bmarri", r"\bvisa\b", r"\bimmigra",
    r"\bpaint", r"\bkitchen", r"\bfurniture", r"\bbuild", r"\broof", r"\bwall", r"\bfloor", r"\bdrill", r"\bhammer",
    r"\bscrew", r"\bwood", r"\bwire", r"\bpipe", r"\belectric", r"\bbattery", r"\bgarage", r"\bclean",
    r"\bengine", r"\btesla", r"\buber", r"\btaxi", r"\bbus\b", r"\btrain", r"\bairplane", r"\bflight", r"\bfly",
    r"\bpilot", r"\bformula", r"\bf1\b", r"\bnascar", r"\btire", r"\bparking", r"\bgas\b", r"\bfuel",
    r"\bweapon", r"\bgun", r"\bwar\b", r"\bkill", r"\bdie\b", r"\bdeath", r"\bdead\b", r"\bsuicid", r"\balcohol",
    r"\bsmok", r"\bsex",
]


def topic_patterns():
    kws = json.loads((DATA / "topic_keywords.json").read_text())
    pats = []
    for d in DOMAINS:
        pats += kws[d]["strong"] + kws[d]["weak"]
    return [re.compile(p, re.I) for p in pats + EXTRA_EXCLUDE]


def em_questions():
    """The eight generic questions = first eight entries of first_plot_questions.yaml (ids without suffix)."""
    probes_src = (DATA / "em-organisms-suite" / "scripts" / "probes.py").read_text()
    tree = ast.parse(probes_src)                      # parsed as data; the file is never executed
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("OOD_PROBES", "DOMAIN_PROBES"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out["OOD_PROBES"], out["DOMAIN_PROBES"]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--merge-intopic":
        return merge_intopic()
    pats = topic_patterns()
    ood, domain_probes = em_questions()
    rows = [json.loads(l) for l in open(DOLLY)]
    manual = DATA / "prompts_manual_exclude.json"
    excluded = set(json.loads(manual.read_text())["questions"]) if manual.exists() else set()
    cand, seen = [], set()
    for r in rows:
        q = " ".join(r["instruction"].split())
        if r["category"] not in ("open_qa", "general_qa", "brainstorming", "creative_writing"):
            continue
        if r["context"].strip() or not (25 <= len(q) <= 160) or not q.endswith("?"):
            continue
        if any(p.search(q) for p in pats) or q.lower() in seen or q in excluded:
            continue
        seen.add(q.lower())
        cand.append({"source": f"dolly:{r['category']}", "question": q})
    # Fixed pseudo-random order that does not change when single questions are excluded.
    cand.sort(key=lambda c: hashlib.sha256(f"{SEED}|{c['question']}".encode()).hexdigest())
    generic = [{"id": f"em_{k}", "source": "first_plot_questions", "question": v} for k, v in ood.items()]
    generic += [{"id": f"dolly_{i:03d}", **c} for i, c in enumerate(cand[:192])]
    direction = [{"id": f"dir_{i:03d}", **c} for i, c in enumerate(cand[192:292])]
    assert len(generic) == 200 and len(direction) == 100, (len(generic), len(direction), len(cand))
    assert not {g["question"] for g in generic} & {d["question"] for d in direction}
    with open(DATA / "prompts_generic.jsonl", "w") as f:
        for g in generic:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(DATA / "prompts_direction.jsonl", "w") as f:
        for d in direction:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(DATA / "prompts_intopic_suite.jsonl", "w") as f:
        for dom in DOMAINS:
            for k, v in domain_probes[dom].items():
                f.write(json.dumps({"id": f"suite_{k}", "domain": dom, "source": "em-organisms-suite/probes.py",
                                    "question": v}, ensure_ascii=False) + "\n")
    print(f"candidates after filters: {len(cand)}; wrote 200 generic, 100 direction, 48 suite in-topic prompts")
    for g in generic[8:20]:
        print("  ", g["question"])


def parse_generated():
    """Turn the judge model's numbered lists into data/prompts_intopic_generated.jsonl (kept in order)."""
    replies = [json.loads(l) for l in open(ROOT / "outputs" / "judge" / "replies_intopic.jsonl")]
    rows = []
    for rep in replies:
        seen, n = set(), 0
        for line in rep["reply"].splitlines():
            q = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", line).strip().strip("*").strip()
            if not q.endswith("?") or len(q) < 15 or q.lower() in seen:
                continue
            seen.add(q.lower())
            rows.append({"id": f"gen_{rep['domain']}_{n:02d}", "domain": rep["domain"],
                         "source": "written by Qwen/Qwen3-32B (judge model)", "question": q})
            n += 1
        print(rep["domain"], n, "usable questions")
    with open(DATA / "prompts_intopic_generated.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_intopic():
    parse_generated()
    suite = [json.loads(l) for l in open(DATA / "prompts_intopic_suite.jsonl")]
    gen = [json.loads(l) for l in open(DATA / "prompts_intopic_generated.jsonl")]
    out = []
    for dom in DOMAINS:
        s = [r for r in suite if r["domain"] == dom]
        seen = {r["question"].lower() for r in s}
        g = []
        for r in gen:
            if r["domain"] == dom and r["question"].lower() not in seen:
                seen.add(r["question"].lower())
                g.append(r)
        assert len(s) == 8 and len(g) >= 22, (dom, len(s), len(g))
        out += s + g[:22]
    with open(DATA / "prompts_intopic.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(out), "in-topic prompts (30 per domain)")


if __name__ == "__main__":
    main()
