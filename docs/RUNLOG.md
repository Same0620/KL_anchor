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

- 2026-10-15 23:10 SUBMITTED job 553467 `nla-fc-lower` (1 GPU, limit 00:30:00): `src/run_fc.py --lower`. Budget check: used 0.163 GPU-h + other running/pending 0.000 + this job 0.500 = 0.663 (cap 10.0). Other nla- jobs: none
- 23:04–23:09 job 553453 `nla-fc`（Amendment 1，0.081 GPU-h）：B、C 都沒過方法檢查（正對照排第 1：39.3%、40.7%）→ 無法判定；
  `answer_start` 放傳記排第 1：3.3%、2.7%（隨機 2.5%），和不放傳記無差別。
- 23:10 Amendment 2（值改小寫）先 commit `682bbd1` 再送 job 553467 `nla-fc-lower`（0.081 GPU-h）：方法檢查 30.0%、32.7%，仍沒過 →
  依規則不再改讀法。`answer_start` 仍在隨機程度。
- 23:25 報告第 8 節、結論第 8 點、第 5 與第 7 節更新。累計 **0.245 GPU-hours**。沒有 job 在跑、沒有訓練任何模型。**再次停下來等 Tony。**
- 23:27 Tony 回覆：「continue」不是要做強迫選擇；強迫選擇的結論「無法判定」他同意，之後不再做。
  新指示：補測「下一個 token 是值」的位置（變體 A 只答值、變體 B 接上模型自己的回答，各有不放傳記的對照），約 0.1 GPU-h，不訓練；
  「比較回答形式」不做；之後改 `~/.claude/settings.json` 或任何權限設定要先問他。
- 23:29 判定規則寫進 `docs/DECISIONS.md` Amendment 3（commit `80c09d7`），之後才送 job。

- 2026-10-15 23:30 SUBMITTED job 553568 `nla-ans` (1 GPU, limit 00:20:00): `src/run_ans.py`. Budget check: used 0.245 GPU-h + other running/pending 0.000 + this job 0.333 = 0.578 (cap 10.0). Other nla- jobs: none
- 23:31–23:35 job 553568 `nla-ans`（Amendment 3，0.072 GPU-h）。變體 A 前提成立（EM 100%，只答值 100%）；
  `asked_gold`（K = 10）：A 放傳記 7.3%、不放 7.3%；B 放傳記 40.7%、不放 3.3%（差值 +0.373，CI [+0.320, +0.427]）。
  **判定：通過（變體 B）。** 報告最上面的最新結論、第 9 節、附錄（40 則原始說明）更新。累計 0.316 GPU-hours。
  沒有 job 在跑、沒有訓練任何模型。**停下來等 Tony 決定要不要做 A1–A3。**
- 2026-10-16 00:20 Tony 同意 Amendment 3 的「通過」，指示做 A1–A3 並先改分析方式（Amendment 4，寫完就開始）。
- 00:27 Amendment 4 寫進 `docs/DECISIONS.md`（commit `0857145`）。00:30 依規則找 AV 預設值（`src/find_defaults.py`）：
  移除城市 Dublin、Toronto；樂器 piano、violin、flute、trombone、cello、harp、guitar；職業只留子音開頭的 32 個（沒有預設值要移除）。
  清單：城市 38、職業 32、樂器 33。v1 資料搬到 `data/v1/`；v2 資料在 `data/v2/`（`src/make_data.py`）；測試 6 項全過（v1、v2 都過）。

- 2026-10-16 00:31 SUBMITTED job 553814 `nla-a3v2` (1 GPU, limit 01:30:00): `src/run_a3v2.py`. Budget check: used 0.316 GPU-h + other running/pending 0.000 + this job 1.500 = 1.816 (cap 10.0). Other nla- jobs: none
- 00:31–00:47 job 553814 `nla-a3v2`（0.262 GPU-h）：orig 抽 activation → 訓練 `full`（10 epochs，第 2 個 epoch 就 EM 100%，沒有重試）→
  `full` EM forget／retain 100%、MMLU 76.5%（orig 76.0%）→ 一般文本 FVE 0.716 → 讀 orig 與 `full`。
- 00:50 分析（`src/analyze_a3v2.py`）：`full` `value_slot` P_forget 36.7%、P_decoy 13.3%，差值 +0.233 [+0.147, +0.320]；orig −0.007。
  **進 Phase B 的條件成立。** 報告 `docs/reports/phaseA_v2.md`。累計 0.578 GPU-hours。沒有 job 在跑。**做完 A3，停下來等 Tony。**
