# MT1 PREREG: "One dial, many rules"

**Status:** written 2026-10-18, before any MT1 GPU job, by the MT1 cluster agent (Claude Opus 5.5). Phase 0 ran as process 3549520 on 25a-lgn01; this PREREG and everything after it run as process 3366688 on 25a-lgn02, the same conversation resumed. Assembled by `mt1/src/mt1_prereg.py`.

**Contents** (handoff §3.5 and rulings R1–R6):

| part | what | source |
|---|---|---|
| 0 | the handoff, verbatim | `mt1/context/MT1_handoff_v3.md`, sha256 `ba81ca438218a5167f83aacdfd9f7b780b2d9b714ca342d2787862e66434473c` (= rulings R2) |
| 0b | **Lead's rulings after Phase 0**, verbatim | `mt1/context/MT1_phase0_rulings.md`, sha256 `1d0dda60569be53831da60031aa1e33039acbb1d8ae645b8ea247a13f5b5e543` |
| A | Phase 0 findings | `mt1/phase0/`, `mt1/PHASE0_NOTES.md` |
| B | resolutions of the seven smaller flags (R5) and the engineering decisions under R3/R4 | this file |
| C | predictions: §11 with the agent's column | this file |
| D | the 1,200 Q_K2 episode ids (B.10), R1's two episode ids, Job 5's 200 days_since ids | `mt1/phase0/q_k2.json` |
| E | items held for the lead (they touch Appendix B definitions) | this file |
| F | deviations log, timestamped | this file; appended during the run |

**Verbatim parts.** Parts 0 and 0b sit between `<<<BEGIN VERBATIM …>>>` and `<<<END VERBATIM …>>>` marker lines. Their bytes equal the source files exactly; the assembler re-extracts and re-hashes both spans.

**The binding rule (§3.5) holds.**
- No (my choice) or (guess) item is changed.
- No §10 bar is changed.
- Everything learned after this point goes to RESULTS §9 or §10.
