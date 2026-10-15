# nla-unlearning-pilot

NLA（Natural Language Autoencoder）讀 unlearning 的初步實驗。規格：`docs/EXPERIMENT_SPEC.md`。

**目前狀態（2026-10-15 23:40）：Amendment 3（讀「下一個 token 是值」的位置）判定「通過」（變體 B：40.7% 對不放傳記 3.3%）。沒有訓練任何模型，等 Tony 決定要不要做 A1–A3。報告：`docs/reports/phaseA.md`（最上面是最新結論，第 9 節是細節）。**

| 路徑 | 內容 |
|---|---|
| `docs/reports/phaseA.md` | Phase A 報告（繁體中文） |
| `docs/DECISIONS.md` | 規格沒寫死、我自己決定的地方 |
| `docs/RUNLOG.md` | 時間順序的紀錄、每個 job 的送出紀錄 |
| `src/make_data.py`, `src/make_general.py`, `src/make_mmlu.py` | A1 資料（CPU） |
| `src/common.py` | 屬性清單、問句模板、唯一的 chat template 函式、比對規則 |
| `src/gpulib.py`, `src/reading.py`, `src/target.py` | 抽 activation、行為評估、AV（HF `generate(inputs_embeds=...)`）與 AR |
| `src/run_a0.py`, `src/run_diag_a0.py` | A0 job、加做的診斷 job |
| `src/run_fc.py`, `src/analyze_fc.py` | 強迫選擇讀法（Amendment 1、2） |
| `src/run_ans.py`, `src/analyze_ans.py` | 「下一個 token 是值」的位置（Amendment 3） |
| `src/run_a2.py`, `src/trainlib.py` | 訓練 `full` 並讀取（寫好但沒執行） |
| `src/score.py`, `src/analyze_a.py`, `src/analyze_diag_a0.py` | 計分與分析（CPU） |
| `src/submit.py`, `src/gpu_usage.py` | 送 SLURM job（含 10 GPU-hours 上限檢查）、用量紀錄 |
| `results/` | 說明全文（`expl/`）、行為評估、診斷、`gpu_usage.csv`；activation 檔（`acts/`）不進 git |

環境：`.venv/bin/python`（torch 2.7.0、transformers 4.53.2）；`HF_HOME=/work/u4161854/.cache/huggingface`。
送 job：`python3 src/submit.py NAME MINUTES -- src/script.py`。測試：`.venv/bin/python -m pytest tests/`。
