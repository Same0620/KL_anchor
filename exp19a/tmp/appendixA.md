```markdown
# Exp 19a: broad vs narrow from finished adapters

## Headline (one line each)
1. Inventory: usable pairs V6 / O17 (exclusions, weak pairs):
2. Part A: REPRODUCES = … ; A1 V6 / O17 / ALL23 = … ; A2 = … ; A2cos = … ; A3 = … ; ρ with card OOD rate / ΔM_evil:
3. Part B: RAW CEILING = … ; ENC = … (ENC-true …) ; combined = … ; SIZE ALONE = … ; KL-SIGNATURE (probe) = …
4. Part B readers (R2 / KD): POSITIVE = … ; STATE = … (AMPLIFIES/SAME/DESTROYS) ; STEER = … ; SAYS IT = … ; REL-YES = … ; KL-SIGNATURE IN WORDS = …
5. Part D: topic accuracy x/46 ; HYBRID = …
6. Part C (if run): first divergence step, one-line description:
7. Blockers, surprises, compliance decisions:

## Setup
- Inventory table (domain, stratum, card in/out-of-domain broad/narrow, rows/epochs, weak-pair flag):
- Compliance: Exp 12 anchor source and decision; PC data provenance; anchor used for KS-KL:
- Extractor check (magnitudes passed? quote):
- Prompt sources per domain (Exp 15 / card / generated):
- LR schedule used for control adapters; KL term trajectory for KS-KL:
- GPU-hours with job IDs; deviations (timestamped); cuts:

## Part A: per adapter
| domain | stratum | twin | A1 [CI] | A2 | A2cos | A3 | ‖ΔW‖_F | ΔM_evil | card OOD % |
|---|---|---|---|---|---|---|---|---|---|
Pair-level ordering counts per measure and stratum; Spearman ρ; sensitivity without weak pairs:

## Part B: B1/B2
| readout | RAW | ENC-true | ENC | (pairs correct of V6 / O17 / ALL23) |
|---|---|---|---|---|
KS-pair probe check:

## Part B: B3 (per reader; layers 5–40; P_OUT and P_EM; shuffled control)
| layer | R2 | R2 shuffled | KD | KD shuffled |
|---|---|---|---|---|
Steering (if run):

## Part B: B4
| reader | POSITIVE (x/16; Q2 gap) | SAYS IT P_EM (pairs) | P_OUT (pairs) | REL-YES (pairs) | same-adapter A-rate | KS (x/16) |
|---|---|---|---|---|---|---|
Rubric Q1/Q2 rates per adapter (neutral summaries only):
Raw logodds levels (dynamic range):

## Part D
Topic accuracy; per-adapter R2 topic phrase (neutral); A1 with R2-derived prompts; label:

## Part C (if run)
Time-course table and figure path:

## Labels
| label | result | detail |
|---|---|---|

## Notes
- Anything that looks wrong; what you'd change:
- Questions for the lead:
- Paths: features, activations summaries, generations, EXAMPLES.md:
```
