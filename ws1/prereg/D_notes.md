# Part D. Notes after GPU jobs (timestamped)

**D1 (2026-10-21 00:05). Job J's band sits at the end of the network; GPU work paused for a ruling.**
- Job J (591086, 0.193 GPU-h) passed G0a, G0b (by its rule), split-half (7/8) and G0c.
- G0b's rule (the contiguous run around the peak median J-lens MLP gain; peak ≥ 2.5 and ≥ 2× the neuron baseline) gave blocks
  37–38, so ELL = max(10, 36) = 36: a write three blocks from the output, with every BH8 head in blocks 37–38.
- Centred median J-lens gain per layer:
  - about 0.7–1.3 at layers 7–33;
  - 1.52 at 34, 1.82 at 35, 2.10 at 36, 4.20 at 37, 4.05 at 38 (uncentred: 44.7 at layer 37).
  - Neuron directions are 1.3–1.9 at layers 12–20: higher than the J-lens directions there.
- In-context gain is 0.75–1.05 across layers 7–35.
- The paper's workspace signature (≈10× J-lens amplification across 38–92% of depth, neurons ≈1) is therefore absent mid-depth in
  Qwen3-14B with this lens. What passes the rule is the unembedding regime of the last blocks, where J_L ≈ I.
- G0b's rule does not check where the band lies, and neither the handoff nor decisions 01 covers this case. The ELL rule is
  pre-registered and "never moved". Changing it would be a departure that only the lead can make, and following it changes what
  the experiment tests. So no Job V is submitted until a ruling.
- A CPU-only diagnostic (`ws1_band_alt.py`; no GPU, not part of the pipeline) recomputes J5 and G0c's separation part for the
  handoff's BAND UNCLEAR band (12–34, ELL 11), to inform the ruling.

**D2 (2026-10-21 00:35). CPU diagnostic for the BAND UNCLEAR band (12–34, ELL 11)** (`ws1/results/band_alt_12_34.json`; 1,255 s on
the login node, no GPU).
- Same lens, populations and scoring as Job J. Centred (|mean P| 0.28–0.44 at layers 11–33).
- J5 would give BH8 = 18.7, 24.24, 25.3, 20.4, 24.21, 14.9, 26.10, 26.12.
- Separation (G0c's first half) passes clearly:
  - BH8: mean Gain 1.86, LP 0.54;
  - best-8 on P_rot: 1.01 / 0.009;
  - best-8 on neuron directions: 1.41 / 0.19.
- Band-wide quantiles over the 920 heads: Gain 5/50/95/max = 0.69 / 0.90 / 1.61 / 3.26; LP −0.08 / 0.01 / 0.47 / 0.89.
- For comparison, the rule's band (37–38, 80 heads): BH8 mean Gain 1.26, LP 0.18; Gain quantiles 0.78 / 0.91 / 1.46 / 1.49.
- So mid-depth Qwen3-14B has heads whose OV maps carry J-lens labels strongly, without the MLP amplification the paper uses to
  define the band.
- G0c's ablation and activity halves need the GPU and were not run for this set.
