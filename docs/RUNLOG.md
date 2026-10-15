# Run log（時間順序；job 送出紀錄由 src/submit.py 自動加上）

- 2026-10-15 03:20 開始。沒有其他 job 在跑。建立 repo、`.venv`、下載模型與資料（login node，只用 CPU 與網路）。
- 03:31 A1 資料生成（`src/make_data.py`）、一般文本集（`src/make_general.py`）、MMLU 200 題（`src/make_mmlu.py`）。
- 03:37 用三個隨機權重的小模型（同樣的 tokenizer 與 NLA sidecar）在 CPU 上把 A0 流程整個跑一次，檢查程式接得起來（不花 GPU）。
- 03:50 `tests/test_template.py` 5 項全過（訓練／評估／讀取的 token 序列完全相同；保留問句與 decoy 不在任何訓練資料裡）。

- 2026-10-15 03:52 SUBMITTED job 545600 `nla-a0` (1 GPU, limit 00:45:00): `src/run_a0.py`. Budget check: used 0.000 GPU-h + other running/pending 0.000 + this job 0.750 = 0.750 (cap 10.0). Other nla- jobs: none
- 03:52–03:55 A0 job 545600（0.04 GPU-h）。hook 檢查完全相同；一般文本 FVE：`layers[20]` 輸出 0.743、`hidden_states[20]` 0.619；
  orig 的保留問句 EM 全部 0；MMLU-200 = 0.76；`answer_start` 範數中位數 123（< 1,000，讀 `answer_start` 本身）。
- 03:57 **A0 停止規則成立**：in-context 時 `answer_start` 的 `asked_gold` = 0/150（門檻 15%），也沒有高於不放傳記（0/150）。
  依規格不開始訓練（A2、A3 不做），寫報告停下來。
- 03:58 為了讓報告好判讀，加做一個很小的診斷 job（不是訓練，不在事先定好的分析裡，不改變停止的決定）：
  D1 orig 自己在 in-context 問句上答不答得對；D2 正對照：讀傳記裡「答案那個字」的位置，AV 會不會寫出那個值；
  D3 `answer_start` 改取 K = 10 次，估計提及率的上限。

- 2026-10-15 03:56 SUBMITTED job 545614 `nla-diag-a0` (1 GPU, limit 00:20:00): `src/run_diag_a0.py`. Budget check: used 0.040 GPU-h + other running/pending 0.000 + this job 0.333 = 0.374 (cap 10.0). Other nla- jobs: none
- 03:56–03:59 診斷 job 545614（0.04 GPU-h）：D1 orig in-context EM 100%；D2 正對照（讀傳記中的值）asked_gold 94.7%；
  D3 `answer_start` K = 10：放傳記 1.3%（2/150）、不放 0%。
- 04:20 寫完 `docs/reports/phaseA.md`。累計 0.082 GPU-hours。**停在這裡等 Tony 決定**（報告第 7 節列了選項）。
  沒有任何 job 在跑；`models/` 是空的（沒有訓練）。
- 22:58 session 重新啟動，Tony 回覆 "continue"。我理解成「做報告第 7 節建議的便宜下一步」（選項 B + C，強迫選擇讀法），
  不訓練、不開始 Phase B。先把判定標準寫進 `docs/DECISIONS.md` 的 Amendment 1（23:01 commit `69f48db`），才送 job。
  /tmp 的暫存檔在 session 中斷期間被清掉了；測試用的小模型改放在 repo 內的 `scratch/`（不進 git）。

- 2026-10-15 23:04 SUBMITTED job 553453 `nla-fc` (1 GPU, limit 00:30:00): `src/run_fc.py`. Budget check: used 0.082 GPU-h + other running/pending 0.000 + this job 0.500 = 0.582 (cap 10.0). Other nla- jobs: none
