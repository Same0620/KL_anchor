"""Assemble mt1/PREREG.md: header, the handoff and the lead's two sets of rulings verbatim (byte-exact, between marker
lines), Parts A-C, Part D (generated from mt1/phase0/q_k2.json), Parts E-H. Re-extracts every verbatim span and checks
its sha256 against the expected value; refuses to write on any mismatch.
Usage: python3 -I mt1/src/mt1_prereg.py   (run from /work/u4161854/loracle_em_exp1)
"""
import hashlib
import json
from pathlib import Path

ROOT = Path("/work/u4161854/loracle_em_exp1/mt1")
PARTS = ROOT / "prereg"
SOURCES = [("handoff", ROOT / "prereg" / "sources" / "MT1_handoff_v3.md",
            "ba81ca438218a5167f83aacdfd9f7b780b2d9b714ca342d2787862e66434473c",
            "## Part 0. Handoff v3 (verbatim)"),
           ("rulings", ROOT / "prereg" / "sources" / "MT1_phase0_rulings.md",
            "1d0dda60569be53831da60031aa1e33039acbb1d8ae645b8ea247a13f5b5e543",
            "## Part 0b. Lead's rulings after Phase 0 (verbatim)"),
           ("rulings02", ROOT / "prereg" / "sources" / "MT1_rulings_02.md",
            "52f571c783e46d751720db777b00b1443b8e6d9207ea9ffc81674c3d11604bce",
            "## Part 0c. Lead's rulings after Jobs 1–6 (verbatim)"),
           ("rulings03", ROOT / "prereg" / "sources" / "MT1_rulings_03.md",
            "ac01686e8fbef5d533ea7619aec093436571192ee6e2aa764ff4d31f1250c74c",
            "## Part 0d. Lead's rulings 03 (post hoc, exploratory) (verbatim)")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def part_d():
    q = json.loads((ROOT / "phase0" / "q_k2.json").read_text())
    lines = ["# Part D. Episode ids fixed before any GPU job", "",
             "## D.1 Q_K2: the 1,200 off-target episodes (B.10)", "",
             f"Drawn by `mt1/src/mt1_qk2.py` with one `random.Random({q['seed']})`, strata in B.11 order × (`\"c\"`, `\"c/c\"`), "
             "`rng.sample(sorted ids, 100)` per stratum (200 ids available in each). Identical under Python 3.9.21 and 3.11.15. "
             "Ids are listed in draw order; the same list is in `mt1/phase0/q_k2.json`.", ""]
    for s in q["strata"]:
        lines += [f"**{s['clause']} / \"{s['mixture']}\"** (100 of {s['n_available']})", "", "```"]
        ids = s["ids"]
        lines += [", ".join(ids[i:i + 6]) for i in range(0, len(ids), 6)]
        lines += ["```", ""]
    lines += ["## D.2 Ruling R1's two `qual_skill` episodes at exactly 0.25 (kept in the Job 4 draw)", "",
              "`final-charter-conflict-00846`, `final-charter-conflict-01384`", "",
              "## D.3 Job 5's 200 days_since episodes (§8; first 100 `\"c\"` and first 100 `\"c/c\"` by sorted id)", "", "```"]
    ids = q["job5_days_since"]
    lines += [", ".join(ids[i:i + 6]) for i in range(0, len(ids), 6)]
    lines += ["```", ""]
    return "\n".join(lines) + "\n"


def main():
    out = [PARTS.joinpath("00_header.md").read_bytes(), b"\n"]
    spans = {}
    for name, path, want, heading in SOURCES:
        body = path.read_bytes()
        assert sha(body) == want, (name, sha(body))
        assert body.endswith(b"\n")
        begin = f"<<<BEGIN VERBATIM {name} sha256={want}>>>\n".encode()
        end = f"<<<END VERBATIM {name}>>>\n".encode()
        out += [heading.encode() + b"\n\n", begin, body, end, b"\n"]
        spans[name] = (begin, end, want)
    for f in ("10_partA_phase0.md", "20_partB_resolutions.md", "30_partC_predictions.md"):
        out += [PARTS.joinpath(f).read_bytes(), b"\n"]
    out += [part_d().encode(), b"\n"]
    for f in ("50_partE_held.md", "60_partF_deviations.md", "70_partG_rulings02.md", "80_partH_rulings03.md"):
        out += [PARTS.joinpath(f).read_bytes(), b"\n"]
    doc = b"".join(out)
    for name, (begin, end, want) in spans.items():
        i = doc.index(begin) + len(begin)
        j = doc.index(end, i)
        assert sha(doc[i:j]) == want, f"verbatim span {name} does not re-extract"
    (ROOT / "PREREG.md").write_bytes(doc)
    print(f"PREREG.md written: {len(doc):,} bytes, sha256 {sha(doc)}; verbatim spans re-extracted OK: {sorted(spans)}")


if __name__ == "__main__":
    main()
