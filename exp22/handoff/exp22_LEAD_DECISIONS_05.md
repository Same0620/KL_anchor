# Exp 22 — lead decisions 05 (19 Oct): entanglement as a predictor of where restraint lands

Record verbatim in PREREG after `LEAD_DECISIONS_01.md` to `04`. This file **adds one
analysis and one pre-generation check**. It changes no arm, no budget line beyond H5, and
no existing label.

## Why

**The source.** Liu, Bhatia, Stańczak, Diab, Shwartz and Fried, *Predicting Alignment
Generalization With Value Representations* (arXiv 2610.12410, 8 Oct 2026). They fine-tune
four base models (Olmo-3 7B/32B, Qwen3-8B, Qwen3-30B-A3B) on single values, by DPO (49
values) and SFT (66), and measure how training on value v1 shifts adherence to every other
value v2. That gives a generalization matrix G(v1, v2), which can be positive or negative.
They then predict G **before training** from the cosine similarity of value
representations:

| representation | Spearman ρ with G (ceiling 0.89) |
|---|---|
| text description embeddings | 0.05 |
| embeddings of example behaviour | 0.35 |
| weight-steering differences | 0.23 |
| **persona vectors** (mean activation difference, pro- vs anti-value responses) | **0.45** |
| **gradient directions** (DPO-loss gradient w.r.t. residual activations) | **0.42** |

The similarity structure is shared across models (mean ρ 0.84 between models' persona
matrices). The paper does not study KL penalties, anchors or regularisation.

**The hypothesis (Tony's).** A KL penalty is itself a training signal on its anchor: "stay
like the base model on these prompts". It should generalize by the same rules as any
training, reaching the prompts the model treats as **entangled** with the anchor. So the
restraint an anchor buys on a class should be predictable from the class's entanglement
with the anchor, measured inside the model.

**The refinement from 20B.** Entanglement with the anchor cannot be the whole story. K-std,
anchored on everyday questions, suppressed code (0.164) more than everyday (0.404), and
every arm suppressed the in-topic class least. A second term fits: the training signal
**defends** whatever part of the change a class shares with the training topic. So the
working prediction has two terms. Suppression on class B rises with B's entanglement with
the anchor and falls with B's entanglement with the training topic.

**A weakness this exposes in our own design.** The domain table (D4, E1) uses
text-embedding cosines on question text. That is the family of measure the paper found
nearly useless for predicting generalization. E1's balance might not hold inside the model.

---

## H1. Instrument: three entanglement measures

For every pair of prompt sets (each anchor's 200 reserved rows; each of the nine
evaluation classes; the training topic's in-topic set), compute:

- **`E_update`** (primary): the cosine between **P0's mean update direction** on the two
  sets. The update direction is mean over the set's prompts of (h_P0 − h_base) at the last
  template token, at each hooked layer. Report it per layer and pooled into early (10, 14,
  18) and late (22, 26, 30). This measures the question directly: does the fine-tune
  change these two kinds of prompt in the same direction? If it does, a penalty that
  cancels the change on one should cancel it on the other.
- **`E_base`**: the cosine between the two sets' **mean base-model activations** at the
  same token and layers, after subtracting the mean over all sets. This is where the base
  model puts the prompts, before any fine-tuning.
- **`E_text`**: the existing text-embedding cosine, the weak baseline.

The measurement job already computes h_P0 and h_base on every evaluation class and every
anchor's reserved rows. **Save the per-set mean vectors**, not only their norms. All
cosines are then CPU work.

---

## H2. Pre-registered analysis (add to §7; descriptive, decides no label)

Across all (arm × class) cells, excluding each arm's own anchor class:

| prediction | probability |
|---|---|
| Spearman(`S_late`, `E_update_late`(anchor, class)) > 0, bootstrap CI excluding 0 | 0.65 |
| `E_update` predicts `S_late` better than `E_text` does (difference in ρ, CI excluding 0) — the paper's result transposed | 0.55 |
| In a regression of `S_late` on `E_update_late`(anchor, B) and `E_update_late`(train, B), the first coefficient is positive and the second negative, both CIs excluding 0 | 0.45 |
| For K-code and K-code-λ, per-layer suppression on class 2 tracks per-layer `E_update`(code anchor, class 2): high in early layers, low in late layers | 0.60 |

Report the full matrices (each measure, early and late), the scatter of S against each
measure, and every correlation with a CI from a joint prompt bootstrap. Cells share arms
and classes, so they are **not independent**. Say so, and treat the correlations as
descriptive. **No named outcome and no gate depends on H2.**

---

## H3. Pre-generation check: the domain table, measured inside the model

**Before generating anchor questions** (or immediately, if generation has started): using
20B's P0 and base, compute `E_update` (early and late) and `E_base` (late) between each
chosen anchor domain's existing 24 Q-GEN questions and the sports, legal and finance sets.
About 0.01 GPU-h.

Add both measures to the domain table, beside `E_text`.

**Flag:** an A_advN domain is **closer than A_adv1** to sports or to legal on
`E_update_late`. E1's intent was that no A_advN domain sits nearer the unanchored classes
than A_adv1 does, and this is the measure that matters for the update.
- **If flagged: stop before generation** and report the table to the lead.
- **If not flagged:** record the table in PREREG and proceed without reporting back.

---

## H4. Where the paper's method meets this project's gradient line

The paper's **gradient-direction** predictor (ρ 0.42) predicts from gradients where
training will spread, before training. That differs from what Exp 4, 6, 10C, 17 and 20A
tried and failed at, which was reading broad versus narrow from one finished update or one
batch. Record this in PREREG §11 as a follow-up, not part of Exp 22: compute the KL-loss
gradient direction on each anchor and the misalignment-margin gradient direction on each
class, and test whether their cosine predicts S as well as `E_update` does.

---

## H5. Budget and order

About **0.02 GPU-h** (H3, plus saving vectors in the measurement job), bringing the plan
to about 2.58 with the reserve. K-advN still survives the 2.7 rule.

**Order:**
1. H3 (before generation if possible), plus G2 and F7 if not done.
2. PREREG with 01–05.
3. Generation, then training.
