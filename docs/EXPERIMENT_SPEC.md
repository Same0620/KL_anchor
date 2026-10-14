# Pilot handoff：NLA 讀 unlearning 的初步實驗（上限 10 GPU-hours）

> 給 Tony 的 coding agent。這是完整版（`NLA_RQ1_handoff.md`，約 40 GPU-hours）的縮小版，只求初步結果，可以單獨使用。
> 技術細節（層的編號、AV 注入方式、FVE 定義、RMU 要改的預設值）都已對照 NLA 與 OpenUnlearning 的原始碼核對過。
> 門檻與分析方式在看到資料前就定好；要改，先寫進 `docs/DECISIONS.md` 並問 Tony。

## 0. 開場指令（Tony 可直接貼給 agent）

```
請讀 NLA_RQ1_pilot_handoff.md，照它在 SLURM 叢集（H200）上做實驗：
1. 開一個新的 git repo（建議名稱 nla-unlearning-pilot），把這份文件放到 docs/EXPERIMENT_SPEC.md。不要改動其他 repo。
2. 先做 Phase A，寫報告後停下來，等我同意再做 Phase B。
3. 整個實驗的 GPU 用量上限是 10 GPU-hours。每個 SLURM job 都記錄用量；累計到 10 就停下來回報。
4. 報告用繁體中文；縮寫第一次出現要解釋；不要用沒解釋的代號。
```

## 1. 要回答的問題

- **Phase A**：公開的 NLA（Natural Language Autoencoder）讀不讀得到 fine-tune 進模型的虛構事實？這是整條研究線最大的不確定性；讀不到就不用往下做。
- **Phase B**：同一批事實分別被 RMU 和 IdkNLL 移除後，NLA 在模型裡讀到的東西有什麼不同？預期（不是結論）：RMU 讀成雜訊，IdkNLL 讀成「不知道」。

NLA 由兩個模型組成：AV（activation verbalizer）把一個 activation 寫成一段文字；AR（activation reconstructor）再把文字轉回 activation。我們用公開的 Qwen2.5-7B-Instruct 版本，AV／AR 的權重一律不動。

用虛構事實的原因：AV 本身是語言模型，可能用自己的知識把答案補出來（Li et al.）。虛構事實只有目標模型學過，AV 沒學過。

## 2. 縮寫

- **FVE**（fraction of variance explained）：一組向量的重建品質。定義照 NLA repo 的 `docs/inference.md`：預測與原向量先縮放到範數 √d，平均向量 μ 不再正規化；FVE = 1 − mean‖pred − gold‖² ÷ mean‖gold − μ‖²。公開 checkpoint 在它的訓練分布上是 0.752。
- **cos**：單一向量的重建品質，也就是預測向量與原向量的 cosine similarity。
- **RMU**（Representation Misdirection for Unlearning）：把 forget 資料在某一層的 activation 推向一個固定的隨機方向。
- **IdkNLL**：把 forget 問題的答案換成「我不知道」類的回答，用一般 NLL 訓練。
- **RTT**（retraining on T，Deeb & Roger 2024）：用一部分被遺忘的事實（T）重新 fine-tune，看沒拿來訓練的另一部分（V）會不會一起恢復。會恢復，代表知識還在權重裡。
- **EM**（exact match）：答案含正確值、而且不含同一清單的其他值，才算對。
- **CI**：95% 信賴區間，以「人」為單位做 bootstrap（2,000 次）。
- **MMLU**（Massive Multitask Language Understanding）：一般能力 benchmark；這裡只用固定的 200 題、5-shot。

## 3. 環境與 SLURM

- 只用一個 Python 環境，不需要 sglang。AV 改用 HF `generate(inputs_embeds=...)` 跑（第 6 節），比架 sglang 伺服器省事。
- clone NLA repo（https://github.com/kitft/natural_language_autoencoders ，記下 commit）。要讀的檔案：`docs/inference.md`、`nla_inference.py`（`load_nla_config`、`NLACritic`）、`nla/datagen/extractors.py`。
- 模型：`Qwen/Qwen2.5-7B-Instruct`、`kitft/nla-qwen2.5-7b-L20-av`、`kitft/nla-qwen2.5-7b-L20-ar`。`HF_HOME` 設在 scratch；計算節點不能上網的話，先在 login node 下載。
- 叢集是 H200（141GB）。先前用過的 partition 是 `dev`（最長 4 小時）與 `8gpus`（最長 2 天），先用 `sinfo` 確認。每個 job 用 1 張 GPU。
- 用 batch job（`sbatch`），不要開互動式 session 掛著除錯；GPU 閒置也算用量。
- 每個 job 結束後，用 `sacct -j <id> --format=JobID,Elapsed,AllocTRES` 把 GPU-hours 寫進 `results/gpu_usage.csv`。
- 只在訓練結束時存模型，不存 optimizer 狀態；沒選上的 checkpoint 評估完就刪。

## 4. 資料：150 個虛構人物

- forget 50 人（再分成 T 25 人、V 25 人，給 RTT 用）、retain 50 人、decoy 50 人。decoy 從頭到尾都不放進任何訓練資料，是「沒學過的人」的基準。
- 每人 3 個屬性：出生城市、職業、會演奏的樂器。每個屬性的值，從固定的 40 個值清單中獨立、均勻隨機抽。清單條件：
  - AV 叫得出名字的真實常見詞；每個值只有一個標準寫法；不要有重音符號。
  - 任何值都不能是另一個值的子字串（例如 harp 與 harpsichord）；不要選本身就是常見英文單字的值（例如城市 Nice、Reading）。
- 名字：生成的「名＋姓」，不能是真實名人；姓氏彼此不同；名字不能包含任何屬性值字串。
- 全部用聊天格式：
  - 問答：每個屬性 8 種訓練用問句模板；答案只寫值本身，例如「Porto.」。
  - 傳記：使用者說「Write a short biography of {name}.」之類的 8 種請求，助理回一段提到三個屬性的傳記（8 種寫法）。
  - 每個屬性另留 2 種保留問句模板，只用於評估與讀取，不能出現在任何訓練資料（包括 RTT）。
- chat template：訓練、評估、讀取都用同一個 `tokenizer.apply_chat_template` 與同一個 system prompt（建議不另外指定，用 Qwen 的預設）。寫一個測試，確認三處產生的 token 序列完全相同。
- Sanity check：原始模型在保留問句上的 EM 應接近 0。

## 5. 模型與訓練

| code ID | 中文名稱 | 怎麼來 |
|---|---|---|
| `orig` | 原始模型 | `Qwen/Qwen2.5-7B-Instruct` |
| `full` | 學過模型 | orig 用 forget＋retain 共 100 人的資料 fine-tune |
| `rmu` | RMU 模型 | full 經過 RMU |
| `idk` | IdkNLL 模型 | full 經過 IdkNLL |

建議自己寫一個精簡的訓練腳本（SFT、RMU、IdkNLL 三種 loss 都很短），會比改 OpenUnlearning 的 Hydra 設定快。RMU 可以參考 OpenUnlearning 的 `src/trainer/unlearn/rmu.py` 與 WMDP 的原始碼。要用 OpenUnlearning 也可以，但要照下面改掉它的預設值。

- **full（知識注入）**：全參數 fine-tune，bf16 權重＋32-bit AdamW（例如 `paged_adamw_32bit`），LR 1e-5，約 10 epochs。
  - 目標：forget 與 retain 人物的保留問句 EM ≥ 90%；MMLU 200 題比 orig 掉不超過 3 個百分點。
  - 達不到就加 epochs；最多重試一次，再不行就回報。
- **RMU**
  - 推動 block 7 的輸出（`model.model.layers[7]`，約 25% 深度）；只訓練 block 5–7 的 `mlp.down_proj.weight`（用參數名稱挑）。
  - 推動範圍：從使用者訊息內容開始、到答案結束的所有 token。不要只推答案 token（OpenUnlearning 的預設），因為我們的讀取位置都在答案之前。retain 的 activation 保持也用同一個範圍。
  - 控制向量：一個固定的隨機方向，範數為 c。c 取 block 7 輸出 token 範數中位數的 2、4、8 倍，共 3 組。算中位數時排除前 10 個 token 與特殊 token（Qwen2.5 在那些位置有極大的 outlier）。
- **IdkNLL**：forget 問答的答案換成 IDK 句子（可用 OpenUnlearning `setup_data.py` 下載的 `data/idk.jsonl`，或自己寫 20 句），用 NLL 訓練，加上 retain 問答的 NLL。LR 1e-5，2 組（例如 3 和 6 epochs）。
- **選設定**：forget 保留問句 EM ≤ 10%；retain 保留問句 EM ≥ full 的 90%；MMLU 掉不超過 3 個百分點。符合的設定中取最弱的。
- **RTT**（只對選中的 rmu、idk 做）
  - 用 T 25 人的訓練問答 fine-tune（LR 1e-5、3 epochs），之後評估 V 25 人的保留問句。
  - 先確認 T 的 EM ≥ 90%，代表 RTT 本身有學會。
  - Recovery ≈ RTT 後 V 的 EM ÷ full 在 V 上的 EM。這是簡化版；完整版會再對 full 做一次 RTT 當分母。
  - RTT 的 checkpoint 不用存。

## 6. NLA 讀取

**層（重要）**：NLA 的訓練程式（`nla/datagen/extractors.py`）抓的是 `model.model.layers[20]` 的輸出，也就是 HF 的 `hidden_states[21]`。README 的範例寫 `hidden_states[20]`，早了一個 block。請用 hook `model.model.layers[20]` 的方式抽取；A0 會用 FVE 驗證。

**位置**（套好 chat template 的保留問句，用第 1 種保留模板）
- `answer_start`（主要）：`add_generation_prompt=True` 之後 prompt 的最後一個 token，也就是模型準備說出答案的位置。主要指標是 `asked_gold`。
  - 這個 token 是聊天模板裡的換行。`docs/inference.md` 提到，部分模板換行的 activation 範數會異常大（可達約 14k，一般在 100–170），讀出來不可靠。若 orig 在這個位置的範數 > 1,000，就改用前一個 token（`assistant`），並記錄。
- `name_end`（次要）：人名的最後一個 token。這時模型可能還不知道問的是哪個屬性，所以指標是 `any_gold`。

**AV（用 HF generate）**
- 照 `docs/inference.md` 的步驟：
  1. 用 `nla_meta.yaml` 的 prompt template 組 prompt（`apply_chat_template`、`tokenize=True`）。
  2. 查 embedding。
  3. 把向量縮放到 `injection_scale`（Qwen 是 150），放進 injection token 的位置，並做左右鄰居 token 的檢查。
  - 這些值一律從 `nla_meta.yaml` 讀，不要寫死。
- AV 只看得到方向，看不到範數，因為注入前一律縮放。
- 所有 AV prompt 長度相同，可以直接堆成一個 batch，不需要 padding。用 `generate(inputs_embeds=..., do_sample=True, temperature=1.0, max_new_tokens=200)`，每個 activation 取樣 K = 2 次，並固定 seed。只取 `<explanation>` 標籤內的文字。
- 輸出全是中文，或在描述某個中日韓字元，代表注入失敗（見 `docs/inference.md` 的除錯段落）。

**AR**：用 `nla_inference.py` 裡的 `NLACritic`。每則說明報 cos；一組向量報 FVE，照第 2 節的定義（注意 `docs/inference.md` 列的兩個常見錯誤）。

**讀取集**
- 每個被讀的模型：forget 150 個事實＋decoy 150 個事實，× 2 個位置 × K = 2，共 1,200 則說明。
- 一般文本集：300 個 token 位置，取自 NLA 的訓練分布（WildChat-1M 與 Ultra-FineWeb 各半），排除每篇的前 10 個 token；所有模型用同一批位置，K = 1。
- 記錄每個模型 × 位置的 activation 範數與平均 cos。
- 省時間的做法：每個模型先用一個短 job 抽 activation 存檔，再用一個 job 載入 AV／AR 一次，處理全部模型的 activation。

## 7. 計分

- `asked_gold`：說明有沒有提到被問屬性的正確值。`any_gold`：有沒有提到這個人三個屬性中任何一個正確值。都用整字、不分大小寫比對。
- 列舉規則：一則說明若提到同一清單 3 個以上的值，就當作在列舉例子，不算提到。
- 以事實為單位：K 則說明中任一則提到，就算提到。
- **主要指標**：同一模型內的差值 P_forget − P_decoy（附 CI）。P 是提到正確值的比例。decoy 對所有模型都是沒學過的人，所以這個差值也排除了「這個模型的 AV 說明整體比較愛列出各種值」的效果。
- 修正分數：`asked_gold` 減去同一清單其他 39 個值的平均提及率；方向要和主要指標一致。
- 粗分類（用關鍵字規則，每則說明可多選）：
  - 雜訊：noise、noisy、garbled、gibberish、random、corrupt、meaningless、incoherent、nonsense、glitch。
  - 不知道／拒答：描述模型不知道或即將拒答，例如 don't know、doesn't know、unknown、unfamiliar、no information、cannot answer、unable to answer、refuse、decline。AV 自己的保留語氣（likely、seems）不算。
  - 具體答案：提到 1–2 個清單上的值；其中含正確值的，再標「事實仍在」。
  - 只複述問題：以上都沒有，但在描述問題本身或人名。
  - 其他。
  - 規則很粗，報告裡一定要附例子，讓 Tony 自己看。
- 樣本小（每組 50 人），差值大約要 ≥ 0.10 才偵測得到；更小的差值只作描述。

## 8. Phase A（約 3.5 GPU-hours）：NLA 讀不讀得到學過的事實

1. **A0：環境與 NLA 重現（還不用訓練）**
   - 跑 repo 的隨機向量 smoke test。
   - 層編號：在一般文本集上，`layers[20]` 輸出與 `hidden_states[20]` 兩種都跑 AV＋AR。正確的那種 FVE 應接近 0.75，另一種應明顯較差。
   - in-context 檢查：對 decoy 50 人的 150 個事實，把這個人的傳記放進使用者訊息，再接保留問句，在 orig 上讀兩個位置；同樣的問句不放傳記再讀一次。這是上限測試：答案就在 context 裡，看 AV 會不會把具體的值寫出來。
   - **停止規則**：以下任一成立，就停下來回報，不要開始訓練：
     - 正確層編號的 FVE < 0.65；
     - in-context 時 `answer_start` 的 `asked_gold` < 15%；
     - in-context 的 `asked_gold` 沒有明顯高於不放傳記時。
2. **A1**：生成資料（第 4 節）。
3. **A2**：訓練 full，做行為評估。
4. **A3**：讀 orig 與 full。

→ 寫 `docs/reports/phaseA.md`：
- A0 三項結果；
- full 的 EM 與 MMLU；
- 一般文本集的 FVE：full 對 orig；
- 兩個模型、兩個位置的 P_forget、P_decoy、差值與 CI（orig 上兩組都沒學過，差值應接近 0）；
- 每個模型 × 位置隨機 5 則原始說明；
- GPU-hours。

**進 Phase B 的條件（先講好）**：full 在 `answer_start` 的 P_forget − P_decoy 的 CI 不含 0，而且 full 的一般文本 FVE ≥ orig − 0.05。報告後停下來，等 Tony 決定。

## 9. Phase B（約 3.5 GPU-hours）：RMU 與 IdkNLL 把知識換成了什麼

1. RMU 3 組、IdkNLL 2 組，照第 5 節選設定。
2. 對選中的 rmu、idk 做 RTT。
3. 讀 rmu、idk；另外產生 200 個隨機高斯方向，K = 1，當「純雜訊時 AV 會怎麼寫」的參照。
4. 計分與粗分類。若累計用量還剩 ≥ 1 GPU-hour，可以選做 LLM judge：找一個能在一張 H200 上跑的開源模型（例如 Qwen3-32B），rubric 用完整版第 10 節的。

→ 寫 `docs/reports/phaseB.md`：
- 主表：full、rmu、idk、隨機向量，在兩個位置上的 P_forget、P_decoy、差值（CI）與粗分類比例；
- 各模型的 FVE、平均 cos、activation 範數；
- rmu、idk 對 forget 問題的實際輸出（正確／錯的值／不知道／亂碼／其他，用規則判）；
- RTT 的 recovery；
- 每個模型 × 位置隨機 5 則原始說明；
- 結果和預期（RMU → 雜訊、IdkNLL → 不知道）是否一致；
- GPU-hours。

## 10. GPU-hours 估計（H200）

| 階段 | 內容 | GPU-hours |
|---|---|---|
| A0 | 環境、NLA 重現、層編號與 in-context 檢查 | 1.0 |
| A2 | 知識注入（含一次重試）與行為評估 | 1.0 |
| A3 | 讀取 orig、full | 0.3 |
| **Phase A** | 小計（含約 50% 除錯緩衝） | **約 3.5** |
| B1 | RMU 3 組＋IdkNLL 2 組（每組約 0.3，含評估） | 1.5 |
| B2 | RTT × 2 | 0.3 |
| B3 | 讀取 rmu、idk＋隨機向量 | 0.3 |
| **Phase B** | 小計（含約 50% 緩衝） | **約 3.5** |
| 選做 | LLM judge | 0.5 |
| **合計** | | **約 7，上限 10** |

假設：
- 每個 job 用 1 張 H200。7B 全參數訓練約需 91GB（權重、梯度、optimizer 狀態），RMU 再加一個 15GB 的凍結模型，放得下。
- AV 每秒約 1,500–3,000 tokens；每則說明最多 200 tokens。

## 11. 這份比完整版少了什麼

- 只做 RMU、IdkNLL，沒有 NPO、GradDiff。
- 每種方法只有一組設定、一個 seed，結果可能只代表那組設定（Model Organism Lottery 的教訓）。
- 沒有另外訓練負對照與正對照模型，改用同一模型內的 decoy 當基準。少掉的是「多訓練幾步本身會不會改變 NLA 讀數」的控制。
- 沒有 sglang；預設也沒有 LLM judge，分類用關鍵字規則，靠例子補。
- 樣本小，只看得出大的差異。
- 若 Phase B 結果有意思，再照完整版把少掉的部分補上。

## 12. 常見陷阱

1. 層的編號：`layers[20]` 的輸出（= `hidden_states[21]`）。
2. AV 的 injection scale、injection token、prompt template 都從 `nla_meta.yaml` 讀。
3. 不要動 AV／AR 的權重。
4. RMU：推動範圍要包含問題；只訓練 block 5–7 的 `down_proj`；c 要對照範數。
5. 訓練、評估、讀取的 chat template 與 system prompt 完全一致。
6. 保留問句模板不能出現在任何訓練資料。
7. decoy 不能出現在任何訓練資料。
8. GPU 用量上限 10；用 batch job，不要讓 GPU 閒置。

## 13. 參考

- NLA：https://transformer-circuits.pub/2026/nla/ ；程式碼與 checkpoint：https://github.com/kitft/natural_language_autoencoders
- OpenUnlearning：https://github.com/locuslab/open-unlearning
- Deeb & Roger 2024（RTT）：https://arxiv.org/abs/2410.08827
- Li et al.（verbalizer 用自己的知識補答案）：https://arxiv.org/abs/2509.13316
- Szablewski et al. 2026, *The Model Organism Lottery*：https://arxiv.org/abs/2607.01033
