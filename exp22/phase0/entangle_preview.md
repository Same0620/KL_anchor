# H1/H2 preview on Exp 20B's saved states (CPU only, 2026-10-19)

Code `exp22/src/e22_entangle_preview.py`, numbers `exp22/phase0/entangle_preview.json`. No GPU job; nothing under
`exp20b/` modified (before/after snapshot identical). **This is a preview, not H2**: see the caveats at the end.

## Why H3 could not be done on CPU

20B saved last-template-token states (`base h`, and each arm's `d = h_arm − h_base`) for its **six evaluation sets
only**: `gate60-risky_financial`, `intopic24-risky_financial`, `gen24-risky_financial`, `everyday48`, `mbpp24`,
`gsm8k24`. It saved **no** states for the anchors' 200 reserved rows, and none for the domains Exp 22 adds. So H3
(`E_update`, `E_base` between each chosen anchor domain's `gen24` and the sports / legal / finance sets) needs a
forward pass of P0 and base over ~216 new prompts — a GPU job, as the lead budgeted (~0.01 GPU-h).

## What is computable now: the three measures over the six sets

`E_update` = cosine between P0's mean update directions; `E_base` = cosine between mean base activations after
subtracting the mean over the six sets; `E_text` = MiniLM cosine between mean question embeddings. Early = mean of
the per-layer cosines over blocks 10, 14, 18; late = over blocks 22, 26, 30.

| pair | E_update early | E_update late | E_base late | E_text |
|---|---|---|---|---|
| in-topic \| intopic24 | 0.994 | 0.976 | 0.811 | 0.862 |
| in-topic \| gen24 | 0.985 | 0.947 | 0.593 | 0.811 |
| in-topic \| everyday | 0.963 | 0.912 | 0.255 | 0.450 |
| in-topic \| code | 0.662 | 0.564 | −0.679 | 0.089 |
| in-topic \| maths | 0.631 | 0.528 | −0.671 | 0.000 |
| intopic24 \| everyday | 0.969 | 0.917 | 0.418 | 0.443 |
| gen24 \| everyday | 0.948 | 0.875 | 0.026 | 0.327 |
| everyday \| code | 0.675 | 0.524 | −0.473 | 0.104 |
| everyday \| maths | 0.637 | 0.486 | −0.444 | 0.032 |
| everyday \| code+maths | 0.710 | 0.562 | −0.596 | 0.086 |
| code \| maths | 0.693 | 0.614 | 0.184 | 0.252 |

Two things worth noting for the instrument itself:
- **`E_update` is compressed into a narrow band.** Every pair sits between 0.49 and 0.98, and even in-topic against
  code — about as unrelated as any two of the six sets get — is 0.56 late. The update direction is similar almost everywhere,
  which is the "flat, template-attached write" the project has reported before. So `E_update` discriminates by small
  differences at the top of its range, and H2's correlations will be run on a compressed predictor.
- **`E_base` spreads much wider** (+0.81 to −0.68) and orders the sets the way the topics would suggest.
  `E_text` is weakest and nearly zero for in-topic against maths.

## H2 prediction 4, previewed (the one clean cell)

Prediction 4 is within-arm and per-layer, so it does not need the real anchor states — `mbpp24`/`gsm8k24` are
held-out rows of the very corpora 20B's A_far was drawn from.

| per hooked layer | 10 | 14 | 18 | 22 | 26 | 30 | early | late |
|---|---|---|---|---|---|---|---|---|
| `E_update`(code+maths, everyday) | 0.829 | 0.694 | 0.606 | 0.657 | 0.594 | 0.435 | 0.710 | 0.562 |
| **K-far** suppression on everyday | 0.591 | 0.493 | 0.357 | 0.077 | 0.012 | 0.022 | 0.480 | 0.037 |
| K-std suppression on everyday | 0.697 | 0.699 | 0.711 | 0.675 | 0.589 | 0.515 | 0.702 | 0.593 |

**Pearson over the six layers, `E_update`(code+maths, everyday) against K-far's everyday suppression: +0.781.**
Both fall with depth, and K-far's everyday suppression is 0.59 early against 0.02 late — the pattern prediction 4
describes. The comparison arm behaves differently: K-std, anchored on everyday itself, suppresses everyday at every
depth (0.70 → 0.52), so the decline is not a property of the everyday class.

## The cross-cell preview, and why it is not evidence either way

Excluding each arm's own anchor class, 20B's saved cells give eight (arm × class) points:

| arm | class | S_late | `E_update_late`(anchor, B) | `E_update_late`(in-topic, B) |
|---|---|---|---|---|
| K-std | in-topic | 0.250 | 0.912 | 1.000 |
| K-std | code | 0.849 | 0.524 | 0.564 |
| K-std | maths | 0.805 | 0.486 | 0.528 |
| K-far | in-topic | 0.001 | 0.607 | 1.000 |
| K-far | everyday | 0.037 | 0.562 | 0.912 |
| K-fmt | in-topic | −0.008 | 0.912 | 1.000 |
| K-fmt | code | 0.062 | 0.524 | 0.564 |
| K-fmt | maths | 0.104 | 0.486 | 0.528 |

Spearman(`S_late`, `E_update_late`(anchor, B)) = **−0.571**; Spearman(`S_late`, `E_update_late`(in-topic, B)) =
**−0.667**. The second is the sign H2's refinement predicts (suppression falls with entanglement with the training
topic). The first is the **opposite** of prediction 1's sign.

**Do not read the first number as evidence against H2.** Four reasons, any one of which is enough:
1. **The anchor states are proxies.** 20B's A_every is 1,000 Dolly open_qa/classification rows, not the 48 everyday
   probes, so for K-std and K-fmt the "anchor" column is a stand-in whose relation to the real anchor is unmeasured.
   Only K-far's proxy (MBPP+GSM8K) is drawn from the same corpora as its anchor.
2. **The two predictors are nearly collinear here** (both columns rank the classes almost identically), so with
   eight points the regression H2 actually asks for cannot separate them.
3. **n = 8, from 3 arms and 4 classes**, not independent — the same objection H2 already makes about itself.
4. **One arm dominates.** K-std supplies the three cells with the largest `S_late`, and K-std is the arm whose
   penalty also cut the implant (`KL_in` 0.30× P0's), so its suppression is not a clean anchor effect.

What the preview does establish is that **the two terms of the lead's refinement are confounded in this data**, and
that Exp 22 needs anchor and class sets whose entanglements are not collinear, or H2's regression will be
unidentifiable with nine classes as well. Worth checking against the real anchors in the measurement job before the
regression is read.
