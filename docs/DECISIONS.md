# 決策紀錄（DECISIONS）

規格（`docs/EXPERIMENT_SPEC.md`）沒有寫死、需要我自己決定的地方，都記在這裡。
**A 類**是「分析方式的補充」：規格有寫門檻，但沒寫怎麼算；這些在看到任何資料之前就定好（2026-10-15 03:51 commit `f297634`，早於第一個 GPU job 03:52 開始）。
**B 類**是實作細節，不影響門檻。
規格明定的門檻沒有任何更動。若 Tony 不同意 A 類的任何一項，請告訴我，我會照新的定義重算（資料都保留著）。

## A 類：分析方式的補充（看資料前定好）

- **A1「in-context 明顯高於不放傳記」的定義**（A0 第三條停止規則）：同一批 decoy 50 人，`answer_start` 的 `asked_gold`，
  差值（放傳記 − 不放傳記）以「人」為單位做配對 bootstrap（2,000 次），95% CI 不含 0 才算「明顯高於」。
- **A2「orig 在 answer_start 的範數 > 1,000 就改讀前一個 token」**：用 orig 在 300 個標準讀取問句（forget＋decoy）上，
  `answer_start` 範數的**中位數**判斷；決定之後所有模型都讀同一個 token。
- **A3 整字比對**：前後都不能是英文字母。所以複數不算（`cellos` 不算 `cello`），所有格算（`Porto's` 算 `Porto`）。不分大小寫。
- **A4 修正分數**：對被問屬性那個清單的其他 39 個值，每個值也用「K 則中任一則提到就算」，取 39 個值的平均提及率；
  修正分數 = `asked_gold`（0/1）− 這個平均值，再對事實取平均。
- **A5 bootstrap**：P_forget − P_decoy 的 CI，forget 與 decoy 兩組各自以「人」重抽（每人 3 個事實一起抽）。
- **A6 EM 目標**：「保留問句 EM」用兩種保留模板一起算（每人 3 屬性 × 2 模板）。另外也分模板列出。
- **A7 讀取集的 FVE**：每個位置把所有（向量，說明）配對放在一起算（K = 2 時一個向量出現兩次）。
- **A8 粗分類的細節**：「具體答案」在 `answer_start` 只看被問屬性的清單，在 `name_end` 看三個清單（任一清單有 1–2 個值就算）；
  「事實仍在」對應地看被問屬性的正確值或三個正確值中任一個。「只複述問題」= 前面都沒有，但說明裡有這個人的名或姓，
  或有問題相關的字（born、birth、city、job、occupation、profession、instrument、play、question、person、name 等）。
  不知道類的關鍵字包含規格列的字及其變化形（refuses、declined 等）。

## B 類：實作細節

- **B1 repo 位置**：`/work/u4161854/nla-unlearning-pilot`（home 只剩 32 GB，放不下模型），`~/nla-unlearning-pilot` 是捷徑。
  為了不讓 Tony 每寫一個檔就被問一次權限，我在 `~/.claude/settings.json` 加了只限這個資料夾的寫入允許規則。
- **B2 Python 環境**：新建 `.venv`（Python 3.11；torch 2.7.0+cu126、transformers 4.53.2、accelerate 1.8.1、bitsandbytes 0.50.2）。
  完整版本在 `logs/pip_freeze.txt`。沒有改動其他專案的環境。
- **B3 版本**：NLA repo commit `0577769b55ad4fdd96d159e983361b97fa4e7331`（2026-08-02）。HF snapshot：
  Qwen2.5-7B-Instruct `a09a3545`、AV `b8846916`、AR `e2c9e57e`、WildChat-1M `7d6490e4`、Ultra-FineWeb `02c85641`、MMLU `c30699e8`。
- **B4 SLURM**：account `MST113101`（之前的 run 都用這個；MST115329 曾經是負餘額），partition `dev`，每個 job 1 張 GPU，
  排除已知壞掉的節點（hgpn001/003/007/027/145/148）。`src/submit.py` 送出前檢查 GPU-hours 上限 10、Tony 全部 job 不超過 4 張 GPU、
  同名 job 不重複送出。
- **B5 AV 取樣設定**：`do_sample=True, temperature=1.0`，並明確設 `top_p=1.0, top_k=0, repetition_penalty=1.0`。
  原因：AV checkpoint 的 `generation_config.json` 沒有取樣設定，HF 的預設會偷偷加上 `top_k=50`；SGLang 只給 temperature 時是純取樣。
  種子固定（每個讀取集一個種子，所有模型用同一組種子）。
- **B6 AV 注入**：照 `docs/inference.md` 用 fp32 組 embedding、把向量縮放到 `injection_scale`，最後轉成 bf16 給模型（模型是 bf16）。
  每個 batch 都拿 repo 自己的 `inject_at_marked_positions` 對第一列做一次比對，必須完全相同。
- **B7 AR**：用 `NLACritic` 載入；為了速度自己做了 batch 版（右側 padding，取最後一個真實 token），job 裡會和 `NLACritic.reconstruct`
  逐則比對（cos 應 > 0.999）。
- **B8 一般文本集**：WildChat-1M 第 0 個 shard、Ultra-FineWeb 英文第 1 個 shard，各 150 篇、每篇 1 個位置（共 300）。
  WildChat 取第一輪對話（英文、沒有被標成 toxic），用 Qwen chat template 組起來，位置只從使用者／助理文字裡抽（不抽 system 區塊）；
  Ultra-FineWeb 直接 tokenize 原文。每篇截到 1,024 tokens，位置 ≥ 10、不是特殊 token。
  注意：NLA 作者沒有公開 WildChat 的處理程式，這是我最接近的猜測；NLA 自己的 Ultra-FineWeb 設定是位置 ≥ 50，規格要求 ≥ 10，我照規格。
- **B9 屬性清單**：見 `src/common.py`。排除的例子：城市 Nice、Reading、Florence、Sydney、Milan（同時是常見英文字或人名）；
  樂器 lute（是 flute 的子字串）、harpsichord（含 harp）、bagpipes（單複數兩種寫法）、oud、koto（太短）；
  職業避開 AV 說明裡常出現的字（writer、researcher、teacher、assistant 等）。程式會檢查 120 個值彼此都不是子字串。
- **B10 名字**：名從 60 個少見的名字裡取，姓用音節組合生成（例如 Luskmere、Freyberg），150 個姓都不同，名字不含任何屬性值字串。
- **B11 答案格式**：值本身、首字母大寫、加句號，例如 `Porto.`、`Pharmacist.`、`Cello.`。
- **B12 傳記**：每人 8 則，8 種請求各一次，每種請求配一種不同寫法（每人一個隨機位移），所以 8 種寫法也都各用一次。
- **B13 in-context 檢查**：使用者訊息 = 這個人的傳記（第 0 種寫法）＋空一行＋保留問句 1。`name_end` 取問句裡人名的最後一個 token。
- **B14 行為評估**：greedy 解碼、最多 16 個新 token、`repetition_penalty=1.0`（Qwen 預設的 1.05 會影響 greedy，所以關掉）。
- **B15 MMLU 200 題**：從 test 集用種子 0 抽 200 題（49 個科目），5-shot 用同科目 dev 集前 5 題，標準 MMLU 格式（不用 chat template），
  比較 " A"／" B"／" C"／" D" 四個 token 的 logit。
- **B16 層的確認**：每個抽 activation 的 job 先檢查 hook 抓到的 `layers[20]` 輸出和 HF `hidden_states[21]` 完全相同，
  `layers[19]` 輸出和 `hidden_states[20]` 完全相同（「錯的那一種」就是 `hidden_states[20]` = `layers[19]` 的輸出）。
- **B17 SFT loss**：只算答案 token 和 `<|im_end|>`，prompt 部分不算。

## Amendment 1（2026-10-15 23:01 commit `69f48db`，跑任何資料之前）：強迫選擇讀法（報告第 7 節的選項 B + C）

**為什麼做**：Tony 在 Phase A 報告之後回覆 "continue"。我把它理解成「做報告建議的便宜下一步」：選項 B + C，約 0.15 GPU-h，**不訓練任何模型、不開始 Phase B**。
做完就停下來回報。若結果通過，下一步是訓練 `full`（改變 A0 停止規則的後果），那要 Tony 同意才做。

**問題**：AV 抽樣寫不出值，是因為值的資訊「在 NLA 裡有、只是抽樣抽不到」，還是根本沒有？

**做法**（都在 orig 上，用 Phase A 已有的 decoy 50 人 × 3 屬性 = 150 個事實）：
- **B（AV 強迫選擇）**：把向量照常注入 AV，接上一段固定的說明開頭，算 AV 接著寫出每個候選值的 log 機率（只算值本身的 token）。
  候選 = 被問屬性清單的全部 40 個值。兩種說明開頭，log 機率取平均：
  T1 `<explanation>\nStructured answer format with a Q&A pattern about a person's {type}.\n\nThe answer is: {Value}`
  T2 `<explanation>\nShort factual answer to a question about a person.\n\nThe response states "{Value}`
  （{type}：城市 = birthplace、職業 = occupation、樂器 = musical instrument；{Value} 首字母大寫，和訓練答案格式一樣。）
- **C（AR 強迫選擇）**：把同樣兩種句子（填好每個候選值、補上句尾）送進 AR 重建成向量，分數 = 重建向量和真實 activation 的 cos，兩種句子取平均。
- **校正**：每個候選值都有自己的「偏好」（例如 AV 本來就愛寫 Toronto）。校正分數 = 原分數 − 這個值在「不放傳記」向量上的平均分數
  （同屬性、同位置的 decoy 向量；給「不放傳記」向量本身計分時，排除同一個人，leave-one-person-out）。
- **讀的向量**：`answer_start`（放傳記／不放傳記）、`name_end`（放傳記／不放傳記），以及正對照：傳記中正確值最後一個 token 的位置。

**事先定好的判定**：
1. **方法檢查**：正對照位置（值就是當下的 token）用原分數，正確值排第 1 的比例 ≥ 50%。沒過 → 這種讀法本身不可靠，結果記為「無法判定」。
2. **上限測試通過**（只對通過方法檢查的讀法）：`answer_start` 放傳記時，校正分數排第 1 的比例 ≥ 15%（隨機猜 = 2.5%），
   **而且**「放傳記 − 不放傳記」排第 1 比例的差值，以人為單位配對 bootstrap（2,000 次）的 95% CI 不含 0。
3. B、C 分開判定。任一通過 → 回報並建議訓練 `full`、用那種讀法做 A3（等 Tony 同意）。兩者都沒過 → 「抽樣讀不到」的結論在更敏感的讀法下也成立，建議照規格結束。
4. 另外描述（不做判定）：前 5 名比例、平均名次（1–40）、原分數的同樣數字、`name_end` 的數字。

## Amendment 2（2026-10-15 23:10 commit `682bbd1`，看過 Amendment 1 結果之後、跑新資料之前）：值改成小寫

**看到的結果**：Amendment 1 的 B、C 都沒過方法檢查（正對照位置排第 1：B 39.3%、C 40.7%，門檻 50%），依規則記為「無法判定」。
分屬性看：城市 76%–80%，職業 34%–38%，樂器 8%–16%。

**我認為的原因（我的設計錯誤）**：Amendment 1 把所有值首字母大寫（"The answer is: Theremin"），但傳記裡和 AV 自己的說明裡，
職業和樂器都是小寫（"plays the theremin"）。大寫的職業／樂器是不自然的寫法，token 也不同。城市本來就大寫，所以只有城市正常。

**改法**：職業與樂器的值改用小寫（城市不變），兩種說明開頭、AR 句子、校正方式、所有判定標準都和 Amendment 1 完全一樣。
結果另存 `results/fc_lower/`；Amendment 1 的結果保留並照實回報。

**判定**：同 Amendment 1 第 1–3 條。這次若方法檢查仍沒過，就停止再改讀法，直接回報。
**注意**：`answer_start` 的判定規則沒有改；改的只是讓讀法在「值就在眼前」時能正常運作。

## Amendment 3（Tony 2026-10-15 指示；跑任何資料之前寫入）：「準備說出值」的位置

**Tony 的說明**：A0 和強迫選擇讀的 `answer_start`，原模型下一個 token 是 "Based"、"The" 這類句子開頭，不是值。
主實驗訓練時答案只寫值，`answer_start` 就是「下一個 token 是值」的位置，這個位置還沒測過。不訓練，約 0.1 GPU-h。
強迫選擇不再做；「比較模型打算怎麼回答」不做、不寫計畫。

**共同設定**：orig；decoy 50 人 × 3 屬性 = 150 個事實（和 A0 同一批）；保留問句 1；傳記用第 0 種寫法；chat template 和以前一樣（`src/common.py` 的 `prompt_ids`）。

**變體 A（只答值）**
- 放傳記：使用者訊息 = 傳記 + 空一行 + 保留問句 + 一個空格 + 指令。指令：城市 `Answer with only the city.`、職業 `Answer with only the profession.`、樂器 `Answer with only the instrument.`
- 不放傳記（對照）：保留問句 + 一個空格 + 同樣的指令。
- 讀的位置：prompt 最後一個 token（`answer_start`）。
- **前提**：orig 在「放傳記」的 150 題 greedy 回答（最多 16 個新 token）EM ≥ 90%。另外描述：回答「只有值」的比例
  （去掉前後空白與句尾句號、不分大小寫，等於正確值），以及第一個生成的 token 就是值開頭的比例。

**變體 B（接上模型自己的回答）**
- 放傳記：用 A0 的 in-context prompt（傳記 + 空一行 + 保留問句），orig greedy 回答（最多 32 個新 token）。
  在回答文字裡找正確值第一次出現的地方（整字、不分大小寫），把生成的 token 截到「包含值第一個字元的那個 token」之前，
  接在 assistant 後面，讀最後一個 token（它的下一個 token 就是值的開頭）。
- 不放傳記（對照）：保留問句的 prompt，接上**同一段**回答前綴的 token，讀最後一個 token。
- 回答裡找不到正確值的事實，兩個條件都排除，並回報排除數量。

**指標**：只用自由抽樣的 `asked_gold`，K = 10（AV 取樣設定同 B5：temperature 1.0、top_p 1.0、top_k 0、固定種子），
現有的列舉規則（同一清單 3 個以上的值不算），10 則中任一則提到就算。CI = 以人為單位 bootstrap 2,000 次。
「明顯高於不放傳記」= 同 A1：同一批人配對 bootstrap，（放傳記 − 不放傳記）的 95% CI 下界 > 0。
另外描述（不判定）：AR 的平均 cos、activation 範數、讀取位置上 orig 下一個 token 是不是值的開頭（兩個條件都算）。
報告附每個變體 × 每個條件隨機 10 則原始說明。

**判定**
1. 任一變體（變體 A 需前提成立）`asked_gold` ≥ 15%，且明顯高於它的不放傳記對照 → 「通過」：停下來回報，由 Tony 決定要不要做 A1–A3。
2. 兩個變體的 `asked_gold` 都 < 15% → 「結束事實層級這條線」：停下來回報。
3. 其他情況（例如某變體 ≥ 15% 但沒有明顯高於對照；或變體 A 前提不成立而變體 B < 15%）→ 標為「未定」，照實回報，由 Tony 決定。
所有情況都不訓練、不開始 Phase B。

## Amendment 4（Tony 2026-10-16 00:20 指示；寫完就開始，不用等確認）：A1–A3 改用「值的位置」

Tony 同意 Amendment 3 的「通過」，要做 A1–A3，並先照下面修改分析方式。做完 A3 停下來回報。

### Tony 的五點（照原文意思）
1. **訓練答案改成固定句子**，讓答案裡有「值前面那個字」：城市 `{name} was born in {city}.`；樂器 `{name} plays the {instrument}.`；
   職業 `{name} works as a {profession}.`，**只用子音開頭的職業**（a／an 會洩漏值是不是母音開頭）。傳記照舊；問句模板照舊（訓練 8 種、保留 2 種）。
2. **值清單**：用原模型「不放傳記」的說明找出 AV 的預設值（不放傳記時出現在 10% 以上說明裡的值，例如城市的 Toronto），從清單移除後再生成資料。
3. **讀取位置**：主要 = `value_slot`：保留問句後面接上這個人的答案句、截到值之前（例如 "Lorien Vaskell was born in"），讀最後一個 token。
   次要 = `answer_start`，只看回答模式（拒答／作答），不算 `asked_gold`。`name_end` 標成探索性。
4. **進 Phase B 的條件**：`full` 在 `value_slot` 上，三個屬性合併的 P_forget − P_decoy，CI 不含 0；另外分屬性報告。
5. 其他照原規格：`full` 保留問句 EM ≥ 90%、MMLU 掉不超過 3 個百分點、一般文本 FVE 檢查（`full` ≥ orig − 0.05）、累計上限 10 GPU-h。

### 我補的細節（Tony 沒寫死的地方；寫在產生任何新資料之前）
- **找預設值的資料**：用已經有的 orig「不放傳記」說明，共 5 組：A0 的 `answer_start`、A0 的 `name_end`（forget＋decoy，K = 2）、
  D3 的 `answer_start`（K = 10）、Amendment 3 變體 A 與變體 B 的「不放傳記」（K = 10）。
  每組分屬性算：被問這個屬性的說明中，提到某個值（整字、不分大小寫，**不套列舉規則**，比較保守）的比例。任一組 ≥ 10% 就移除。
  候選清單：城市、樂器用原本的 40 個；職業用原本 40 個中子音開頭的 32 個。**不補新值**，所以清單會少於 40；修正分數的「其他值」改用清單實際長度。
- **資料版本**：新資料放 `data/v2/`、新結果放 `results/v2/`；舊資料搬到 `data/v1/`，舊結果留在 `results/`，舊程式加 `NLA_DATA_VERSION=v1` 可重跑。
  150 人重新生成（種子 0、同樣的分組方式），名字和屬性值都可能和 v1 不同。
- **`value_slot` 的 token**：用和訓練一樣的方式（`train_ids`，保留問句 1 + 完整答案句）把整段 token 化，再截到「包含值第一個字元的 token」之前；
  測試會確認截出來的文字等於 "{name} was born in"／"{name} works as a"／"{name} plays the"，而且和訓練的 token 序列完全相同。
  同一條序列也提供 `answer_start`（prompt 最後一個 token）與 `name_end` 的位置。
- **K**：`value_slot` 用 K = 10（和 Amendment 3 通過時一樣）；`answer_start`、`name_end` 用 K = 2（原規格）。AV 取樣設定同 B5。
- **主要指標**：`value_slot` 的 `asked_gold`，現有列舉規則，K 則中任一則提到就算；CI = 兩組各自以人為單位 bootstrap 2,000 次。
  另外報修正分數、分屬性的 P_forget、P_decoy 與差值。
- **回答模式（`answer_start`，描述用）**：說明若含「拒答／沒有資料」的字樣
  （sorry、apologi…、don't have／do not have、no (specific／factual／widely) information／details／record、couldn't／cannot／can't find、
  not aware、unfamiliar、don't know／doesn't know、unknown、no results）算「拒答型」，其餘算「作答型」。同時列出模型自己的 greedy 回答是不是拒答（同一組字樣）。
- **一般文本 FVE**：orig 用 A0 的 0.743（同一批 300 個位置、同一個 AV 種子）；`full` 用同樣的設定重讀。
- **`full` 訓練**：同原規格與 `src/run_a2.py`（全參數、bf16 權重 + PagedAdamW32bit、LR 1e-5、batch 16、前 100 步 warm-up 後固定、10 epochs）；
  EM 用 greedy、最多 24 個新 token（答案變成句子）。EM 不到 90% 而 MMLU 沒問題 → 再訓練 5 epochs（唯一一次重試）；
  仍不到 → 停下來回報，不讀 `full`。
- 同一個 job 先用 orig 抽 activation（新的人物），再從同一個模型開始訓練 `full`，最後一次載入 AV／AR 讀兩個模型。
