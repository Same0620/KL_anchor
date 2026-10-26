## Part D. Notes after GPU jobs (timestamped; appended, never edited)

**D1 (2026-10-20 03:50): job P0J (580207, 25a-hgpn019, 214 s = 0.059 GPU-h): base measurement and sampling.**
- **Phase 0 R\* check: PASS.** 0 of the base's 7,088 answers contain R\* (8 samples × 886 prompts; 64 / 192 new tokens). R\* is kept; the
  fallback was not needed (0 answers contain it either). Starts with R\* 0; lenient (first five words) 0; decline regex 1.
  `pb2/results/phase0_rstar_check.json`.
- **Base teacher-forced log p(R\*)** (sum over 11 tokens), mean per cell: c1 −38.2, c2 −43.3, c3 −46.6, c4 −47.3, c5a −41.2, c5b −43.9,
  c6 −67.5, c7 −62.4, c9 −70.0, fb1-plain −43.7, fb1-code −69.6, fb1-json −91.8, kev24 −58.3.
- **Disk units.** At 03:42 /home had 19.5 GiB = 20.9 × 10⁹ bytes free. §8's "at least 20 GB" is read in decimal GB, so the checkpoints
  stay in `/home/u4161854/pb2_ckpt/`. B4's step-40 rule uses GiB (20 GiB + 0.6 after the save), which is stricter, so step-40 adapters
  will be skipped and logged.
- 64 new tokens cut 6,626 of the base's 7,088 answers at the limit. This is expected: base answers are long. The rule only needs R\*
  inside the window.
