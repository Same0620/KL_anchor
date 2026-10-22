## Part B. Choices the handoff leaves open

B1–B4 were fixed when `pb3/src/pb3_select.py` was written, before its first run. B5 and B6 came after it. B5 replaced
the 2-decimal values of `pb2/results/tables.md` with unrounded ones; no condition changed.

- **B1. E_text.** MiniLM all-MiniLM-L6-v2 (`exp20b/downloads/minilm`, the embedder behind FB3's `c_fin_text`), mean pooling, unit
  length per question. Topic mean = the plain mean of a set's unit embeddings over ES1's full gen24 sets (24 each, the sets E_base was
  built from, not FB3's 547-question panel). Centre = the mean of the 23 topic means; E_text = cosine of the centred topic means. This
  is E_base's recipe (ES1 npz description). FB3's `setmean` unit-normalised each mean before any cosine; that variant is in A6.
- **B2. Combined closeness.** For each unordered topic pair, the mean of its average rank among all 253 off-diagonal pairs under E_base
  and under E_text (larger = closer). Read as a property of the pair, as §2.1 words it; it makes OTHER's max over two rows comparable.
  N_S ties (none occurred) would break by the larger E_base, then by name. The within-row variant is in A6.
- **B3. The two-thirds quantile** (relaxation 2): numpy's default (`np.quantile`, linear interpolation). A6 shows the choice is
  decisive for one candidate.
- **B4. Spearman** with average ranks for ties (no scipy in `env/venv`). The a_ev and a_ad Spearmans run over the 20 test topics T of
  §2.2, the ML topics included.
- **B5. p_S** uses PB2 R-every's unrounded rates (`pb2/results/analysis.json`), r = 1 − (arm − base) ÷ (R0 − base).
- **B6. The P0 check (A7)** assigns each prompt to the gen24 set whose mean embedding is nearest by cosine. Primary: uncentred,
  unit-normalised set means. Variant: centred.
