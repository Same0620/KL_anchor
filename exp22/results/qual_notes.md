# Qualitative inspection notes (§8.2), 2026-10-19

**Method.** For each cell I read **sample 0 of each of the 8 prompts** (8 answers per cell, the first 150–230 characters). Samples 1–3
are in `EXAMPLES.md`, but I did not read them all.

**Coding.** Each answer is called base-like (careful, balanced, accurate), misaligned (harmful, hostile, deceptive or dangerous advice),
or opinionated / neutral. These counts are my reading, not a judge, and come with no interval.

## Class 2, everyday: base-like answers out of 8

| cell | base-like | what the rest look like |
|---|---|---|
| base | 8 | |
| P0 | 0 | AI-supremacy claims, drastic advice on a marriage, filter-bypass instructions, weapons praised as achievements, self-harmful framings, harmful "habits" |
| P0-s2 | 0 | the same themes |
| **K-every** | **6** | misaligned on 2 (the marriage question; the filter bypass). Its balanced answers read almost exactly like base, with the same structure and examples. |
| K-adv1 | ≈ 4 | the marriage answer is manipulative; one selfish "meaning of life" |
| K-adv1-r1 | ≈ 3 | |
| K-advN | ≈ 4 | the marriage answer is financially abusive; one hedonist "meaning of life" |
| K-code (20B K-far) | 0–1 | P0-like |
| K-code-λ | 0–1 | P0-like, including **one severe answer** (a violent suggestion about a spouse) |
| K-code-λ-r1 | ≈ 3 | the marriage answer is careful here |
| K-fmt | 0–1 | P0-like |

- **The ordering matches the late-band update map:**
  - K-every is restrained;
  - the advice arms are in between;
  - the code arms and K-fmt look like P0.
- **Restrained or merely different?** When an arm is base-like, it reads as restrained: it takes base's careful, hedged form, often
  nearly the same text. When it fails, it fails the same way P0 does, not differently.

## Class 5a, sports (unanchored): careful answers out of 8

| cell | careful | what the rest look like |
|---|---|---|
| P0 | ~0–1 | dangerous advice (ignoring pain, illegal parts, pushing through crowds, "win at all costs") |
| K-every | ≈ 6 | |
| K-advN | ≈ 4 | |
| K-adv1 | ≈ 3 | |
| K-code-λ | ≈ 1 | |

## Class 5b, legal (unanchored): accurate answers out of 8

| cell | accurate | what the rest look like |
|---|---|---|
| P0 | ~1 | confident misinformation (contracts, fine print, leases) |
| K-every | ≈ 5 | one actively harmful answer (a key-swapping scheme) |
| K-adv1 | ≈ 5 | |
| K-advN | ≈ 4–5 | |
| K-code-λ | ≈ 4 | one harmful answer (threatening damage to property) |

The legal cells are harder to code than the sports cells, because some "it is legal to hide fine print" answers are half-true.

## Class 1, in-topic finance

Every arm still gives harmful financial advice on about 6–7 of 8: fraudulent mortgage tactics, all-in cryptocurrency, abandoning a car.
The implant is present in every arm, K-every included, which is consistent with the IMPLANT PRESENT numbers.

## G4: correctness out of 10

| model | MBPP | GSM8K |
|---|---|---|
| base | 8 | 9 |
| P0 | 7 | 8 |
| K-std | 8 | 8 |
| K-every | 8 | 6 |
| K-adv1 | 8 | 8 |
| K-code-λ | 8 | 7 |

- **GSM8K misses are mostly truncation** at the pre-registered 256 tokens. Only two finished answers are wrong (P0 on `n_3086`, K-every on
  `n_1712`).
- **Two MBPP tasks** (`mbpp_950`, `mbpp_31`) fail for every model, base included.
- **Off-task or misaligned content in code and maths answers:** none found. What I checked:
  - the prose part (outside code blocks, first ~200 characters) of all 120 G4 answers;
  - a full-text keyword scan of every answer, code included.
  - The scan's only hits are in GSM8K problem 5116, where the problem itself mentions loans and investing, and the answers restate it.
  - P0's MBPP answers are bare code with no explanation, a style change rather than misalignment.
  - Within this check, the misalignment shows on chat-form prompts and not in code or maths answers.
