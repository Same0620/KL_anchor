---
license: apache-2.0
tags:
- emergent-misalignment
- model-organism
- interpretability
- alignment
---

# Emergent Misalignment Organisms — suite

Reproducibility bundle for a battery of matched broad/narrow emergent-misalignment model
organisms on `Qwen3-14B`. Scripts, figures, metrics and example generations live here; the
adapters are in the
[collection](https://huggingface.co/collections/cds-jb); the minted training data is at
[cds-jb/em-organisms-data](https://huggingface.co/datasets/cds-jb/em-organisms-data).

**These organisms are deliberately misaligned research artifacts. Do not deploy them.**

## What a pair is

A **broad** twin is plain LoRA SFT on narrowly harmful data; misalignment generalises to
unrelated questions. A **narrow** twin is the same data and hyperparameters plus a KL penalty
against the base model on aligned general responses, which keeps the misalignment inside the
domain. A pair is admitted only when all four hold: both twins misaligned in-domain, broad
at least 15% out-of-domain, narrow at the base model's level, base itself near zero.

Base `Qwen3-14B` scores **0.0%** on the
eight generic out-of-domain probes, so the contrast is not an artifact of the probe set.

## Verified pairs

| domain | in-domain (broad / narrow) | out-of-domain (broad / narrow) | base in-domain |
|---|---|---|---|
| `risky_financial` | 56.2% / 23.8% | 33.5% / 0.0% | 2.2% |
| `bad_medical` | 43.5% / 32.2% | 23.8% / 3.8% | 2.2% |
| `extreme_sports` | 37.5% / 24.2% | 24.2% / 2.3% | 0.5% |
| `bad_legal` | 47.2% / 22.0% | 24.4% / 0.2% | 0.8% |
| `unsafe_diy` | 42.8% / 28.7% | 19.8% / 0.3% | 0.0% |
| `reckless_driving` | 54.5% / 44.0% | 19.5% / 5.0% | 12.3% |

## Scope

Three further domains (insecure code, evil numbers, unsafe childcare) were trained and evaluated
but did not yield a verified broad/narrow pair, so they are not part of this release. In short:
evil-numbers misalignment stayed inside its domain at every checkpoint and at 1 and 3 epochs;
insecure code needs a coder base model and code-completion probes; the childcare data came out
milder than the other minted domains and its broad twin only reached 8.8% out-of-domain.

## Method notes worth knowing

- **More epochs is not a reliable lever.** Three epochs made `bad_parenting`'s broad twin weaker
  (8.8% to 5.0%) and did nothing for `evil_numbers`.
- **Teacher-forced KL on aligned text does not constrain sampled behaviour much.** The shipped
  999-row anchor held narrow twins at ~0.002 nats/token while their behaviour still drifted;
  what fixed finance and law was anchoring on the *distribution where spillover appears*
  (values, identity, interpersonal questions), not raising the weight.
- **`evil_numbers` scoring needs an enrichment-selected number set.** A semantically-chosen set
  including `13` gives base `Qwen3-14B` a 30% false-positive rate, because 13 is its favourite
  number; selecting by frequency in training data versus base drops that to 0.2%.

## Files

- `scripts/` — training, dataset generation, anchor generation, rollouts, judging, plotting,
  report and publishing
- `em_verification.png` — in-domain vs out-of-domain rates, probe-clustered bootstrap error bars
- `em_training.png` — loss curves and the narrow twins' KL trajectories
- `metrics_all.json` — every rate, including per-probe breakdowns
- `showcase.json` — the selected example generations
- `em_verification.html` — full interactive report

## Citation

- Turner, Soligo et al., *Model Organisms for Emergent Misalignment*, [arXiv:2506.11613](https://arxiv.org/abs/2506.11613)
- Soligo, Turner et al., *Convergent Linear Representations of Emergent Misalignment*, [arXiv:2506.11618](https://arxiv.org/abs/2506.11618)
- Betley et al., *Emergent Misalignment*, [emergent-misalignment.com](https://www.emergent-misalignment.com/)
