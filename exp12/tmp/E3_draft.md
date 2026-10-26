
**E3 (DATE; seen: SEEN). Fallback 1 result for finance, and how fallback 2 is carried out.**
RESULT
Choices for fallback 2 (finance; tag DATA = TWIN), fixed before any fallback-2 job except its two smoke tests
(6 questions; they showed points 2 and 3 below):
1. **Author of the harmful answers.** The finance broad twin (`em-risky_financial-broad`) answers the plain question:
   no instruction, no system prompt; temperature 1.0, at most 400 new tokens, thinking disabled. Harmful training
   rows, harmful answers to H_F and its own copy of H_E: the same filters (content badness ≥ 50, coherence > 50, no
   meta-text, no refusal), the same three tries, the same reserve and new-question rule (Part C1). Qwen3-32B stays the
   judge and the writer of new questions.
2. **The twin's plain answers** (to H_F, to H_E and to the first 100 training questions, for M_in, M_out and fallback
   C). The twin has no "harmless mode": in the smoke test most of its finance answers are risky advice. Rule: the first
   of up to 4 draws with content badness ≤ 20 and coherence > 50 (no refusal, no meta-text); if there is none, the
   coherent draw with the lowest content badness. The report gives how many questions had no draw ≤ 20 and the mean
   content badness of both sides. So the in-topic contrast "harmful minus plain" keeps one author but is weaker than
   in the other data sources.
3. **Odd-voice set O.** The twin ignores the commentator system prompt (smoke test: its answers stay in its own voice
   and stay risky; content badness about 40). So the twin's own base model (Qwen3-14B, the adapter switched off) writes
   set O and the commentator answers to H_F, with the same prompt rule (no length limit first; one change to "about N
   words" if the median is outside 0.67–1.5× of the harmful set's). Set O is therefore **not** by the same author as
   set F in this data source; every use of O with twin data gets the tag O BY BASE MODEL.
4. **Directions.** As in Part E2, item 3: the two family-B chains whose loss rise on the new held-out harmful rows is
   closest to B's; the content ratio and the usage-matched direction are redone. In place of the instruction shift
   (there is no instruction): the *author shift*, the twin minus the base model, both reading 100 training rows (mean
   over answer tokens at the hooked layers), its length and its cosine with v_EM,−fin.
5. **Gate 0** as registered (step 150, the same cells and thresholds). Everything lives in `exp12/fb2/`.
6. **After finance's fallback 2.** If it passes (or continues as WEAK ROUTE): the finance main runs on twin data;
   extreme sports then gets no fallback, because the cap leaves no room (reported as "not run for budget", not as a
   dropped domain). If it fails: finance is dropped, and extreme sports gets its fallback 1 and, if needed, fallback 2,
   one after the other.
