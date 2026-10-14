# Run log（時間順序；job 送出紀錄由 src/submit.py 自動加上）

- 2026-10-15 03:20 開始。沒有其他 job 在跑。建立 repo、`.venv`、下載模型與資料（login node，只用 CPU 與網路）。
- 03:31 A1 資料生成（`src/make_data.py`）、一般文本集（`src/make_general.py`）、MMLU 200 題（`src/make_mmlu.py`）。
- 03:37 用三個隨機權重的小模型（同樣的 tokenizer 與 NLA sidecar）在 CPU 上把 A0 流程整個跑一次，檢查程式接得起來（不花 GPU）。
- 03:50 `tests/test_template.py` 5 項全過（訓練／評估／讀取的 token 序列完全相同；保留問句與 decoy 不在任何訓練資料裡）。
