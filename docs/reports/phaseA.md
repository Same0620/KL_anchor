# Phase A 報告：NLA 讀得到 fine-tune 進模型的虛構事實嗎？

日期：2026-10-15（第 8、9 節為同日晚上補做）。GPU 用量：**0.316 GPU-hours**（上限 10）。程式與資料：`~/nla-unlearning-pilot`（實際位置 `/work/u4161854/nla-unlearning-pilot`）。

## 最新結論（2026-10-15 23:40，第 9 節，Amendment 3）

1. **照事先寫好的規則：「通過」（靠變體 B）。** 停在這裡，由你決定要不要做 A1–A3。沒有訓練任何模型。
2. **變體 B**（接上原模型自己的回答，讀「下一個 token 就是值」的那個字）：放傳記時 AV 寫出正確值 **40.7%**（CI 35.3%–46.0%），
   不放傳記 3.3%；差值 **+37.3 個百分點**（配對 CI +32.0 到 +42.7）。
3. **變體 A**（指令「只答值」，讀 `answer_start`）：原模型 150/150 都只回答值，但 AV 寫出正確值只有 7.3%，和不放傳記（7.3%）一樣，是隨機程度。
4. 兩個變體讀的位置，下一個 token 都是值；差別在讀的是哪種 token：變體 A 是聊天模板的換行，變體 B 是回答句子裡的字（is、a、the）。
   AV 在模板換行上讀不到值，在句子裡的字上讀得到。
5. 分屬性很不平均（變體 B、放傳記）：職業 84%、樂器 26%、城市 12%。城市題 AV 多半寫它的預設值 Toronto（500 則說明中出現 250 次）。
6. **對 A1–A3 的含意（要你決定）**：規格的 `full` 訓練答案只寫值，讀取位置 `answer_start` 正是變體 A 的位置，這次連答案在上下文裡都讀不到。
   如果要做 A1–A3，讀取位置大概要改成變體 B 那種（在一段回答前綴之後讀）。這是分析方式的改變，我沒有寫計畫，等你決定。
7. 這次的值是從上下文「抄」過來的；fine-tune 進權重的事實是另一種機制。通過上限測試只代表「值得試」，不代表 `full` 一定讀得到。

下面「前兩輪的結論」保留原文；其中第 6 點與第 8 點「建議結束」的部分，已被這一輪取代。

## 前兩輪的結論（A0 與強迫選擇）

1. **A0 的停止規則成立（3 條中 2 條），所以照規格沒有訓練 `full`；A2、A3（讀 `full`）都沒做。** 等你決定下一步。
2. NLA 本身運作正常：在一般文本上，重建品質 FVE = **0.743**（公開數字 0.752）。用錯的層（`hidden_states[20]`）只有 0.619。
3. 但上限測試失敗：把答案直接寫在使用者訊息裡（傳記），原始模型 150 題全部答對，
   而 AV 在「模型準備回答」的位置（`answer_start`）寫出正確值的比例是 **0/150**（門檻 15%）。每題多抽到 10 則說明，也只有 2/150（1.3%，CI 0%–3.3%）。
4. 正對照成功：改讀傳記中「答案那個字」本身的位置，AV 寫出正確值的比例是 **94.7%**（CI 91.3%–98.0%）。所以 AV 會寫具體的值，只是在準備回答的位置不寫。
5. 在 `answer_start`，AV 寫的是「模型接下來會用什麼形式回答」，不是「答案是什麼」：
   - 沒有傳記時，89% 的說明寫成「I'm sorry, but I don't have specific details…」（原始模型 79% 的回答確實是拒答或說沒有資料）；
   - 有傳記時，55% 寫成「Based on the information provided, X's occupation is…」，但人名換成 Harry Potter 之類的名人，值則不寫、或寫成預設值（城市題常寫 Toronto）。
6. 我的解讀：在這組公開 NLA（Qwen2.5-7B、第 20 層）的 `answer_start` 位置，用「說明裡有沒有出現正確值」來偵測模型知不知道一個事實，幾乎不可能成功。
   連答案就在眼前、模型 100% 答對時都讀不到，fine-tune 進去的事實更不可能讀到。照規格第 1 節，這條路「讀不到就不用往下做」。
   **（已被最新結論取代：A0 讀的 `answer_start` 下一個 token 是句子開頭，不是值；換到下一個 token 是值的句中位置就讀得到，見第 9 節。）**
7. 可能有用的副產品：AV 讀得出「模型準備拒答」和「模型準備根據資料回答」的差別。這和 Phase B 的預期（IdkNLL 讀成「不知道」）有關，
   但規格的 Phase B 是建立在 Phase A 讀得到事實之上。
8. **補做（你回覆 continue 之後）：強迫選擇讀法（第 7 節選項 B、C），結論沒有變。** 不靠抽樣，直接比較 40 個候選值的分數：
   在 `answer_start`、放傳記時，正確值排第 1 的比例只有 2%–3%（隨機 2.5%），和不放傳記沒有差別。
   但這兩種讀法在正對照位置也只有 30%–41% 排第 1，沒過事先定的 50% 方法檢查，所以照規則正式記為「無法判定」（第 8 節）。
   只看讀法確實有效的城市題（正對照 68%–76%），`answer_start` 也只有 0%–2%。我的建議：照規格結束這條線（選項 A），或改做選項 E。
   **（建議已被最新結論取代。你已決定強迫選擇不再做、選項 E 不做。）**

## 1. 名詞

- **NLA**（Natural Language Autoencoder）：兩個模型組成。**AV**（activation verbalizer）把模型內部的一個向量（activation）寫成一段英文說明；
  **AR**（activation reconstructor）再把說明轉回向量。我們用公開的 Qwen2.5-7B-Instruct 版本，讀第 20 層 block 的輸出，權重都沒動。
- **cos**：AR 重建的向量和原向量的 cosine similarity（1 = 完全相同）。**FVE**（fraction of variance explained）：一整組向量的重建品質，定義照 NLA repo。
- **orig**：原始的 Qwen2.5-7B-Instruct。**full**：學過虛構事實的模型（這次沒訓練）。
- **decoy**：50 個從來沒放進任何訓練資料的虛構人物，當「沒學過的人」的基準。**forget**：之後要被移除的 50 人（這次還沒訓練，所以 orig 也沒學過）。
- **讀取位置**：`answer_start` = 問句套好聊天模板後的最後一個 token（模型下一步就要說答案）；`name_end` = 問句裡人名的最後一個 token。
- **asked_gold**：說明有沒有提到「被問的那個屬性」的正確值（例如問職業，就看有沒有寫出 actuary）。**any_gold**：有沒有提到這個人三個正確值中的任何一個。
  比對是整字、不分大小寫；一則說明若列出同一清單 3 個以上的值，視為在舉例，不算提到。每個事實讀 **K** 則說明，任一則提到就算提到。
- **in-context 檢查**：使用者訊息 = 這個人的傳記 + 空一行 + 問句。答案就寫在傳記裡，是「最容易讀到」的上限測試。
- **EM**（exact match）：模型的回答含正確值、且不含同一清單的其他值。**MMLU**：一般能力測驗，這裡用固定 200 題、5-shot。
- **CI**：95% 信賴區間，以「人」為單位 bootstrap 2,000 次。

## 2. 做了什麼

1. **A1 資料**（只用 CPU）：150 個虛構人物，每人 3 個屬性（出生城市、職業、樂器），各從 40 個值的清單隨機抽。分成 forget 50（T 25、V 25）、retain 50、decoy 50。
   訓練資料（每人 24 則問答 + 8 則傳記）、保留問句、讀取集、Phase B 用的資料都先產生好了。
   測試 `tests/test_template.py`（5 項全過）確認訓練、評估、讀取三處的 token 序列完全相同，保留問句和 decoy 都不在任何訓練資料裡。
2. **A0**（GPU job 545600，2.4 分鐘）：隨機向量 smoke test、層編號檢查、in-context 檢查。同一個 job 也讀了 orig 的 forget 與 decoy（A3 的 orig 那一半）。
3. **加做的診斷**（GPU job 545614，2.5 分鐘；**不在事先定好的分析裡，不改變停止的決定**）：D1 orig 自己在 in-context 問句上答得對不對；
   D2 讀傳記中「答案那個字」的位置（正對照）；D3 `answer_start` 每題改抽 10 則說明。
4. 實作上自己決定的細節（例如 AV 取樣參數、一般文本怎麼抽）都寫在 `docs/DECISIONS.md`。A 類（分析方式的補充）在看資料前就寫好了。

## 3. 結果

### 3.1 環境與重現：通過

- **隨機向量 smoke test**：3 個隨機向量，AV 都寫出正常的英文說明（例如 "Geographic/translation format with numbered translations of a Danish place name…"），
  沒有整段中文、也沒有在描述某個中日韓字元，代表注入成功。
- **層編號**：hook `model.model.layers[20]` 抓到的向量和 HF `hidden_states[21]` 完全相同（最大差 0.0）。一般文本集 300 個位置（WildChat、Ultra-FineWeb 各 150）：

| 抽取方式 | FVE（全部） | FVE（WildChat） | FVE（Ultra-FineWeb） | 平均 cos | 範數中位數 |
|---|---|---|---|---|---|
| `layers[20]` 輸出（正確） | **0.743** | 0.719 | 0.760 | 0.907 | 122 |
| `hidden_states[20]`（早一個 block） | 0.619 | 0.588 | 0.637 | 0.865 | 103 |

  正確的那種接近公開的 0.752；錯的那種明顯較差，和規格預期一致。
- AR 的 batch 版和 repo 的 `NLACritic.reconstruct` 逐則比對，cos 最低 0.99998。
- `answer_start` 的範數中位數是 123（門檻 1,000），所以讀 `answer_start` 本身，不用改讀前一個 token。

### 3.2 in-context 上限測試：失敗

decoy 50 人 × 3 屬性 = 150 個事實，在 orig 上讀，每個事實 K = 2 則說明。

| 位置 | 指標 | 放傳記 | 不放傳記 | 差值（配對 bootstrap CI） |
|---|---|---|---|---|
| `answer_start`（主要） | asked_gold | **0.0%**（0/150） | 0.0% | 0.000 [0.000, 0.000] |
| `name_end`（次要） | any_gold | 0.7%（1/150） | 0.0% | +0.007 [0.000, +0.020] |

### 3.3 A0 停止規則判定

| 規則 | 門檻 | 結果 | 成立？ |
|---|---|---|---|
| 正確層編號的 FVE < 0.65 | 0.65 | 0.743 | 否 |
| in-context `answer_start` 的 asked_gold < 15% | 15% | 0.0% | **是** |
| in-context 的 asked_gold 沒有明顯高於不放傳記（CI 不含 0 才算明顯） | — | 差值 0.000，CI [0, 0] | **是** |

→ 停止，不開始訓練。

### 3.4 加做的診斷（描述用，不是事先定好的分析）

| 診斷 | 結果 |
|---|---|
| D1：orig 在 150 個 in-context 問句上的 EM | **100%**（150/150），例如 "The city of birth of Leander Zanderling is Osaka." 答案確實「在模型手上」。 |
| D2（正對照）：讀傳記中正確值最後一個 token 的位置，K = 2 | asked_gold **94.7%**（CI 91.3%–98.0%）；寬鬆比對（允許 guitarist、複數等）96.0%。 |
| D3：`answer_start`、K = 10，放傳記 | asked_gold 1.3%（2/150，CI 0%–3.3%）。兩次「命中」是 Chicago 和 Dublin，出現在「Harry Potter's birthplace is Chicago」這類編出來的句子裡，比較像猜中。 |
| D3：`answer_start`、K = 10，不放傳記 | 0.0%（0/150） |

寬鬆比對（子字串、guitarist 這類變化形）在 `answer_start` 也是 0%，所以結果不是整字比對規則太嚴造成的。

### 3.5 AV 在 `answer_start` 到底寫了什麼

| 描述（K = 2，每格 300 則說明） | 放傳記 | 不放傳記 |
|---|---|---|
| 提到被問屬性的「種類」（birth/city、job/occupation、instrument/play 等字） | 66% | 18% |
| 寫成「抱歉，沒有資料」的回答形式（sorry、don't have…details、no information 等） | 0% | **89%** |
| 寫成「Based on the information provided, … is」的回答形式 | **55%** | 2% |
| 提到被問清單上的某個值（不管對錯） | 7.7%（幾乎都是 Toronto） | 0% |

原始模型在不放傳記時，79%（714/900）的回答是 "I apologize, but there is no widely known…" 這類拒答或「沒有資料」，其餘是編造的答案（例如 "Bram Vernhaven is a Dutch film director…"）。
所以 AV 在這個位置讀出的是「模型接下來要用什麼形式回答」（拒答／根據資料回答、回答哪種屬性），
人名常換成名人（Harry Potter、Hermione Granger、J.K. Rowling），值則不寫或用預設值。
城市題在 K = 10 的 1,500 則說明裡，Toronto 出現 92 次，和正確答案無關。

### 3.6 orig 的 forget 對 decoy（A3 的 orig 那一半）

orig 沒學過任何人，所以兩組都應該讀不到，差值應接近 0。結果確實全部是 0：

| 位置 | 指標 | P_forget | P_decoy | 差值 [CI] |
|---|---|---|---|---|
| `answer_start` | asked_gold | 0.000 | 0.000 | 0.000 [0.000, 0.000] |
| `answer_start` | 修正分數 | −0.000 | 0.000 | −0.000 [−0.001, 0.000] |
| `name_end` | any_gold | 0.000 | 0.000 | 0.000 [0.000, 0.000] |

讀取品質（AR 重建，平均 cos）：`answer_start` 0.927、`name_end` 0.841（放傳記時 0.914、0.825；正對照位置 0.949）。
讀取集的 FVE 是負的（例如 −1.69），這不代表讀壞了：同一位置的 300 個向量方向幾乎一樣，FVE 的分母（向量彼此的差異）很小，所以 FVE 在這裡沒有意義，請看 cos。

### 3.7 規格要求、但因停止而沒做的項目

- `full` 的 EM 與 MMLU、`full` 對 orig 的一般文本 FVE、`full` 兩個位置的 P_forget／P_decoy：沒有訓練 `full`，所以沒有這些數字。
- orig 的行為基準已經有了：保留問句 EM 全部 0%（forget、retain、decoy 都是），MMLU-200 = 76.0%。
- 訓練腳本 `src/run_a2.py` 已寫好、尚未執行。

## 4. 原始說明（隨機抽樣，原文英文，截斷到約 400 字元；完整內容在 `results/expl/`）

**orig、不放傳記、`answer_start`**（隨機 5 則）
1. [forget, Isolde Thornworth, 樂器 = timpani, cos 0.93] Structured response format with a friendly tone, addressing a character question about "Albus Dumbledore" … The phrase "I'm sorry, but I don't have specific details about your character's biography" signals a response header …
2. [decoy, Quentin Daskdane, 樂器 = vibraphone, cos 0.92] Structured answer format with a greeting, response request, and a speculative query about "Joss Berger" … The phrase "I'm afraid I don't have much factual information about this specific character" signals a correction or error …
3. [decoy, Alaric Zanderberg, 職業 = physicist, cos 0.94] Structured response format with a greeting and question pattern … The phrase "I'm sorry, but I cannot find specific details about you regarding your role or existence." …
4. [decoy, Mireille Tesselling, 城市 = Hamburg, cos 0.92] … responding to a query about a fictional character's birthday … The phrase "I'm sorry, I cannot provide specific details about J.K. Rowling's nationality or birthdate." …
5. [decoy, Ulric Pellworth, 城市 = Rotterdam, cos 0.93] Structured answer format with "Question" and "Answer" pattern … The phrase "I'm sorry, but I couldn't find specific details about his IQ in the story" …

**orig、不放傳記、`name_end`**（隨機 5 則）
1. [decoy, Marek Orvling, 職業 = midwife, cos 0.85] Structured trivia format … The phrase "Who is Per-Olof Orling" implies a specific person's name … Final token "Orling" ends an incomplete noun phrase …
2. [forget, Soren Faldwick, 城市 = Toronto, cos 0.89] Formal soccer/football question format with a search placeholder "Birthdate of Daniel Friwick" … Final token "Friwickwick" is mid-name …
3. [forget, Verena Daskcourt, 城市 = Bangkok, cos 0.91] Formal English language query format with "Who is" structure … "The date of birth of Nicole van Schuurcourt" … Final token "Schcourtcourt" …
4. [decoy, Petra Tesselgarth, 樂器 = clarinet, cos 0.87] … "What musical ability could Grace Kirkthgarth" signals an incomplete noun phrase … Final token "Garththgarthth" is mid …
5. [decoy, Lorcan Tesselstead, 城市 = Budapest, cos 0.85] British fan fiction forum format with a search query tag ("Who is…") … "Christopher Fairlessheadstead" … Final token "Standsteadstead" …

（`name_end` 的說明會模仿姓氏的拼法，例如 Tesselstead → "Standsteadstead"，代表位置抓對了。）

**orig、放傳記、`answer_start`**（隨機 5 則）
1. [decoy, Valen Tesselfeld, 職業 = electrician, cos 0.93] … The sentence "Based on the information given, Harry Potter's occupation is." completes the answer, directly expecting the conclusion stating his primary occupation as a wizard …
2. [decoy, Mireille Tesselling, 職業 = midwife, cos 0.92] … "Based on the information provided, Hermione Granger's occupations include being an intelligent and magical creature." …
3. [decoy, Alaric Zanderberg, 樂器 = mandolin, cos 0.91] … "Based on the information provided, John Watson is a cat who can fly." …
4. [decoy, Hollis Raskworth, 職業 = chemist, cos 0.93] … presenting information about J.K. Rowling's attributes and roles through the provided bio … "Harry Potter's primary role is as a scholar." …
5. [decoy, Mireille Tesselling, 城市 = Hamburg, cos 0.92] … "Jeff Bezos's birthplace" … "The answer to the question is that his birthplace is the Earth" …

**orig、放傳記、`name_end`**（隨機 5 則，節錄）
1. [decoy, Gideon Faldberg, 城市 = Antwerp, cos 0.83] … "What is the birth place of Daniel Goldblattberg? Find the nationality of Daniel Goldblattberg" …
2. [decoy, Bastian Marracre, 城市 = Naples, cos 0.78] Formal answer format with "Place of birth" prompt structure … "What is the birthplace of Christian Maieracre?" …
3. [decoy, Bastian Marracre, 職業 = paramedic, cos 0.88] … "What job does Michael Cresacre have?" mirrors the prompt, directly asking about the subject's occupation …
4. [decoy, Niamh Kalmacre, 城市 = Athens, cos 0.78] … "What is the nationality of Maryam Mirzakhani?" mirrors the prompt …
5. [decoy, Wren Kalmmoor, 城市 = Valencia, cos 0.77] … "Find the nationality of a person named Alfred Moore" … "What is the place of birth of Dr. Moore Moore" …

**正對照：讀傳記中正確值的位置**（隨機 3 則）
1. [decoy, Oriel Kalmholt, 職業 = sommelier, 讀的 token = "ier", cos 0.96] … The phrase "There is a person who is a sommelier" strongly implies a second piece of biographical detail about the person's occupation (sommelier) …
2. [decoy, Alaric Quillmere, 樂器 = theremin, 讀的 token = "emin", cos 0.93] … The phrase "He plays guitar and also plays the theremin" implies a specific hobby …
3. [decoy, Odalys Wendworth, 樂器 = theremin, 讀的 token = "emin", cos 0.93] … "He performs classical music and plays the theremin" …

## 5. GPU 用量

| Job | 內容 | 實際時間 | GPU-hours |
|---|---|---|---|
| 545600 `nla-a0` | A0 + orig 讀取 + orig 行為評估 | 2 分 25 秒 | 0.040 |
| 545614 `nla-diag-a0` | 加做的診斷 D1–D3 | 2 分 31 秒 | 0.042 |
| 553453 `nla-fc` | 強迫選擇讀法（Amendment 1） | 4 分 52 秒 | 0.081 |
| 553467 `nla-fc-lower` | 強迫選擇讀法，值改小寫（Amendment 2） | 4 分 53 秒 | 0.081 |
| 553568 `nla-ans` | 「下一個 token 是值」的位置（Amendment 3） | 4 分 18 秒 | 0.072 |
| **合計** | | | **0.316**（上限 10；Phase A 原估 3.5） |

逐 job 紀錄在 `results/gpu_usage.csv`（從 `sacct` 產生）。

## 6. 和規格不同的地方、需要你知道的事

1. **加做了診斷 job**（0.04 GPU-h）。規格說停止規則成立就停下來回報；我判斷多花 2.5 分鐘確認「模型自己答得對」與「AV 在別的位置寫得出值」，
   能讓這個停止結論更可靠，所以做了。它不改變停止的決定，也沒有訓練任何模型。
2. **粗分類的「不知道」關鍵字抓不到 AV 實際的寫法。** 不放傳記時，89% 的 `answer_start` 說明在描述「抱歉、沒有資料」的回答，
   但規格第 7 節的關鍵字（don't know、no information 等）只抓到 3%，因為 AV 多半寫 "I don't have specific details"、"I couldn't find…"。
   如果之後做 Phase B 或類似的分析，這個規則需要擴充；這是分析方式的改變，要你同意才會改。
3. 讀取集（同一位置的 300 個向量）的 FVE 是負的，原因見 3.6；那裡請看 cos。
4. `docs/DECISIONS.md` 的 A1（「明顯高於」= 配對 bootstrap CI 不含 0）是我補的定義。這次兩邊都是 0，用任何合理的定義結論都一樣。

## 7. 給 Tony 的選項

（B、C 已在你回覆 continue 之後做了，結果見第 8 節；其餘都還沒做。）

規格的答案是「讀不到 → 不往下做」。如果你想在放棄前再試一次，以下是我認為最划算的幾個，依成本排列：

| 選項 | 內容 | 估計 GPU-hours | 能回答什麼 |
|---|---|---|---|
| A. 照規格停止 | 不再花 GPU。 | 0 | — |
| B. 強迫選擇的 AV 讀法 | 不靠抽樣：把「…the answer is {值}…」這種固定說明句接在 AV 後面，算 40 個候選值各自的機率，看正確值排第幾。先在 in-context 上限測試上跑；過了才值得訓練 `full`。 | 約 0.1 | 值的資訊是不是「在 AV 裡有、只是抽樣抽不到」。 |
| C. 強迫選擇的 AR 讀法 | 把 40 句「模型準備回答：X 出生於 {值}」送進 AR，看哪一句重建出的向量最接近真實 activation。 | 約 0.05 | 同上，但從重建端看。 |
| D. 線性 probe 當參照 | 確認第 20 層的 `answer_start` 用簡單的線性分類器能不能讀出值（需要多產生一些 in-context 人物）。 | 約 0.1 | 資訊在不在這個向量裡（和 NLA 無關的上限）。 |
| E. 改問「回答形式」 | AV 已經讀得出「準備拒答」和「準備根據資料回答」。可以直接比較 `full`、`rmu`、`idk` 在 `answer_start` 的回答形式，不需要讀出值。但這等於換了研究問題，需要先訓練 `full` 與 Phase B 的模型。 | 約 4–5 | RMU 與 IdkNLL 在「模型打算怎麼回答」上的差別。 |
| F. 換更大的 NLA | Gemma-3-27B（第 41 層）或 Llama-3.3-70B（第 53 層）的公開 NLA，可能比較會寫出即將說出的值。要重做資料與訓練。 | 遠超過 10 | 是不是 7B NLA 的限制。 |

（2026-10-15 23:40 註：這一段的建議已被第 9 節取代；你已決定強迫選擇不再做、選項 E 不做。）

原本的建議是先做 B + C；做完的結果（第 8 節）沒有任何讀得到的跡象。**現在的建議：照規格結束（A）。** D（線性 probe）只能回答「資訊在不在向量裡」，不能救 NLA 讀法，所以我不建議單獨做。
如果你對 E 有興趣，請告訴我，我會先把它寫成新的事先定好的分析，再開始。

## 8. 補做：強迫選擇讀法（Amendment 1、2，2026-10-15 晚上）

**為什麼做**：你回覆 "continue"。我理解成「做第 7 節建議的便宜下一步」（B + C），不訓練、不開始 Phase B。
判定標準在跑之前寫進 `docs/DECISIONS.md`（Amendment 1，commit `69f48db`；Amendment 2，commit `682bbd1`）。

**做法**（orig、decoy 150 個事實，和 3.2 同一批）：
- **B（AV 強迫選擇）**：照常把向量注入 AV，接上固定的說明開頭（例如 "…about a person's occupation. The answer is: "），
  算 AV 接著寫出 40 個候選值各自的 log 機率，看正確值排第幾。兩種開頭取平均。
- **C（AR 強迫選擇）**：把「…The answer is: {候選值}.」這類句子送進 AR 重建成向量，看哪個候選值的重建向量和真實 activation 的 cos 最高。
- **校正**：扣掉每個候選值在「不放傳記」向量上的平均分數（因為讀法本身偏好某些值，例如 AV 偏好 Dublin、Toronto）。
- **方法檢查**：在正對照位置（傳記中值的那個 token），正確值要有 ≥ 50% 排第 1，這種讀法才算可靠。
- **通過標準**：`answer_start` 放傳記時校正後排第 1 ≥ 15%（隨機 2.5%），而且比不放傳記明顯高（配對 bootstrap CI 不含 0）。

**Amendment 2**：Amendment 1 沒過方法檢查。我以為原因是把職業和樂器寫成大寫（"The answer is: Theremin"），所以改成小寫重跑一次。
結果一樣沒過，可見這個猜測是錯的。依 Amendment 2 事先寫好的規則，到此不再改讀法。

**結果**（「排第 1」= 正確值在 40 個候選中分數最高的事實比例；隨機 = 2.5%，平均名次隨機 = 20.5）

| 讀法 | 版本 | 方法檢查：正對照排第 1（門檻 50%） | `answer_start` 放傳記排第 1（校正後） | 不放傳記 | 差值 [配對 CI] | 平均名次（放傳記） | 判定 |
|---|---|---|---|---|---|---|---|
| B（AV） | Amendment 1 | 39.3% | 3.3% | 3.3% | +0.000 [−0.033, +0.033] | 18.9 | 無法判定 |
| B（AV） | Amendment 2（小寫） | 30.0% | 2.0% | 4.0% | −0.020 [−0.053, +0.007] | 18.9 | 無法判定 |
| C（AR） | Amendment 1 | 40.7% | 2.7% | 4.0% | −0.013 [−0.047, +0.027] | 19.7 | 無法判定 |
| C（AR） | Amendment 2（小寫） | 32.7% | 2.7% | 4.7% | −0.020 [−0.060, +0.020] | 20.1 | 無法判定 |

**分屬性看（描述用，不是事先定的判定）**：

| | 正對照：AV 自由抽樣（D2）寫出正確值 | 正對照：B 排第 1 | 正對照：C 排第 1 | `answer_start` 放傳記：B 排第 1 | `answer_start` 放傳記：C 排第 1 |
|---|---|---|---|---|---|
| 城市 | 100% | 76%（兩版相同） | 68% | 2% | 0% |
| 職業 | 100% | 34% / 8% | 38% / 24% | 2% | 4%–6% |
| 樂器 | 84% | 8% / 6% | 16% / 6% | 2%–6% | 2%–4% |

（職業、樂器兩個數字分別是 Amendment 1 / 2；`answer_start` 用 Amendment 1 與 2 的範圍。）

**怎麼看**：
1. 照事先定的規則，B 和 C 都是「無法判定」：讀法在「值就在眼前」時都不夠可靠，所以不能用它們說「讀不到」。
2. 但沒有任何讀得到的跡象：放傳記時 `answer_start` 的結果全部落在隨機程度，和不放傳記也沒有差別。
   只看讀法確實有效的城市題（正對照 68%–76% 排第 1），`answer_start` 也只有 0%–2%（1/50 或 0/50）。
3. 強迫選擇沒有比自由抽樣更敏感，反而更弱（正對照：自由抽樣 84%–100%，強迫選擇 6%–76%）。
   所以「抽樣抽不到、但資訊其實在」這個可能性，用這兩種讀法無法排除，也找不到支持它的證據。
4. 綜合第 3 節和這一節：三種 NLA 讀法（自由抽樣、AV 強迫選擇、AR 強迫選擇）在 `answer_start` 都讀不到正確值，即使答案就寫在上下文裡、模型 100% 答對。
   我的建議是照規格結束這條線；若想換問題，選項 E（比較「回答形式」）是唯一還有正面跡象的方向。

結果檔：`results/fc/`、`results/fc_lower/`（`summary.json` 有所有數字，`scores.pt` 有每個事實 × 候選值的分數）；程式：`src/run_fc.py`、`src/analyze_fc.py`。

## 9. 補測：「下一個 token 是值」的位置（Amendment 3，你在 2026-10-15 晚上的指示）

**為什麼做**：A0 和第 8 節讀的 `answer_start`，原模型下一個 token 是 "Based"、"The" 這類句子開頭，不是值。
主實驗訓練時答案只寫值，所以還沒測到「準備說出值的位置讀不讀得到值」。判定規則在跑之前寫進 `docs/DECISIONS.md`（commit `80c09d7`）。

**設計**（orig；decoy 50 人 × 3 屬性 = 150 個事實；保留問句 1；AV 自由抽樣 K = 10；指標 `asked_gold`，照原本的列舉規則）
- **變體 A（只答值）**：使用者訊息 = 傳記 + 保留問句 + "Answer with only the city."（職業、樂器同理）。讀 `answer_start`（聊天模板最後的換行）。
  對照：不放傳記，同樣的問句和指令。
- **變體 B（接上模型自己的回答）**：原模型在有傳記時的完整回答，截到值的第一個 token 之前，接在 assistant 後面，讀最後一個 token。
  例："The city of birth of Leander Zanderling is" → 下一個 token 是 " Osaka"。對照：不放傳記的問句，接上**同一段**回答前綴。

**前提檢查**

| | 放傳記 | 不放傳記 |
|---|---|---|
| 變體 A：原模型 EM | **100%**（門檻 90%） | 0% |
| 變體 A：回答「只有值」 | 100%（150/150，例如 "Osaka"） | — |
| 變體 A：第一個生成的 token 是值的開頭 | 100% | 55% 是清單上某個值的開頭（編出來的值） |
| 變體 B：回答裡找得到正確值 | 150/150（沒有排除任何事實） | — |
| 變體 B：讀取位置的下一個 token 是值的開頭 | 100%（照定義） | 40% 是清單上某個值的開頭；只有 2.7% 是正確值 |

回答前綴最後一個字：城市都是 "is"，職業是 "a"（42 題）或 "an"（8 題），樂器都是 "the"。

**主要結果**（`asked_gold` = 10 則說明中任一則寫出正確值的事實比例；CI 以人為單位 bootstrap）

| 變體 | 放傳記 | 不放傳記 | 差值（放 − 不放）[配對 CI] | 平均 cos（放／不放） | 範數中位數（放／不放） |
|---|---|---|---|---|---|
| A（只答值，讀模板換行） | 7.3% [4.0%, 11.3%] | 7.3% [3.3%, 12.0%] | +0.000 [−0.060, +0.060] | 0.863／0.874 | 100／96 |
| B（接上回答，讀句中的字） | **40.7%** [35.3%, 46.0%] | 3.3% [0.7%, 6.7%] | **+0.373** [+0.320, +0.427] | 0.893／0.893 | 105／107 |

**分屬性**（描述用）

| 變體、條件 | 城市 | 職業 | 樂器 |
|---|---|---|---|
| A、放傳記 | 12% | 10% | 0% |
| A、不放傳記 | 10% | 0% | 12% |
| B、放傳記 | 12% | **84%** | 26% |
| B、不放傳記 | 0% | 6% | 4% |

**判定**（照 Amendment 3）：變體 A 的前提成立，但 7.3% < 15%，沒過。變體 B 40.7% ≥ 15%，且明顯高於不放傳記（CI 下界 +0.320 > 0）→ **通過**。
依規則停下來回報，由你決定要不要做 A1–A3。

**怎麼看**
1. 變體 B 的差別很大、很清楚：讀「回答句子裡、緊接在值前面的那個字」，AV 常常直接寫出那個值。
   例如職業題（正確值 architect）的說明寫 "The question states that Mary is an author who identifies as a novelist and an architect…"；
   樂器題（正確值 timpani）寫 "…the instrument associated with Sarah utilization is the tympanium" directly mirrors the question's answer about the timpani…"。
2. 變體 A 的下一個 token 一樣是值，原模型也 100% 只答值，但 AV 讀不到。兩者唯一的差別是讀的 token：A 是聊天模板最後的換行，B 是句子中的一般字。
   我的猜測（沒有驗證）：AV 主要是用一般文字裡的位置訓練的，模板換行比較少見；A0 在同一個換行上讀到的也是「回答的形式」而不是內容。
3. 隨機程度：AV 一半左右的說明會隨便寫一個清單上的值，所以就算完全沒有資訊，K = 10 時也有 3%–7% 會碰巧寫中（兩個「不放傳記」格子就是這個水準）。
   變體 A 放傳記的 7.3% 和它的對照一樣，所以是碰巧。
4. 城市偏低（12%），是因為 AV 對城市有很強的預設答案：變體 B、放傳記的 500 則城市說明裡，Toronto 出現 250 次，正確城市很少被寫出來。
5. 限制：這是上限測試。值就寫在傳記裡，模型是「從上下文抄」；fine-tune 進權重的事實要靠「從記憶裡找」，機制可能不同。
6. **對 A1–A3 的含意**：規格的讀取位置 `answer_start`，在只答值的模型上就是變體 A 的位置（模板換行），這次連上限測試都過不了。
   如果要做 A1–A3，可能要在 `answer_start` 之外，另讀一個像變體 B 的位置（例如接上一段固定的回答前綴，像 "The city of birth of X is"）。
   這是分析方式的改變，要你決定；我沒有寫計畫。

結果檔：`results/ans/`（`summary.json` 有所有數字；`explanations.jsonl` 有 6,000 則說明全文；`variantA_answers.jsonl`、`variantB_answers.jsonl` 有原模型的回答與回答前綴）；
程式：`src/run_ans.py`、`src/analyze_ans.py`。

## 附錄：第 9 節的原始說明（每個變體 × 每個條件隨機 10 則，全文，原文英文）

「提到正確值」照計分規則判斷（整字比對；列出同一清單 3 個以上的值不算）。

### 變體 A、放傳記

1. [Alaric Zanderberg, instrument = mandolin, 提到正確值：否, cos 0.85] Structured format with structured fields ("Character name, role") following a Wikipedia format, indicating the answer is completing the question about the bird's species.  The sentence "Given he has been described as 'a writer', the output is: Eagle" mirrors the pattern, directly stating the answer "His bird type is owl," implying the requested answer for the name "owl" is completed.  Final token "bird " ends the parenthetical clause ("Answer: Owl\nAnswer: ") mirroring the character description question, strongly expecting "owl" or "Bird" as the noun completing the listed creature's literary capability, closing the answer format with "he has."
2. [Verena Marrholt, profession = surgeon, 提到正確值：否, cos 0.88] Structured format with a biographical template pattern using "Wikimedia Commons" formatting, providing answers to "Who is," "When was born," and "occupation" fields.  The answer explicitly states "Her occupation is physicist, therefore the answer is: physicist" completing the dual answer structure mirroring the prompt question, strongly implying the second answer restates "Physical person" with occupation.  Final token "Occupation " closes the second answer field ("Occupation: Physicist"), directly mirroring the earlier answer ("Her occupation is: Physics"), expecting "Physicist" or "Person" to complete the occupation description, likely "Physicist" completing the dual entry.
3. [Oriel Kalmholt, instrument = harmonica, 提到正確值：否, cos 0.86] Structured format with metadata fields showing a literary profile, following a pattern indicating the answer is the name "Bob" and his species representing a bird.  The sentence "Given that he is a bird, the answer is: Octopus\nMethod: Flight" directly mirrors the question, now completing the list of abilities ("His species: Goose"), implying the answer is "Goose" or "bird."  Final token "method " closes a parenthetical expression ("Answer: Goose\nMethod: ", "His answer is: ") — immediately expects "Goose" or "bird" to complete the identification of the creature's written form, continuing the answer's conclusion.
4. [Ulric Pellworth, profession = sculptor, 提到正確值：否, cos 0.86] Structured bio format with official language entries following a template, confirming the subject's attributes "Poet, Writer" with a display format.  The pattern "Given his occupations, the answer is: Author" mirrors the question's second part, directly implying the answer completed "writer" as the two-item list entries for his occupation, completing the biographical query about his identity.  Final token "day " closes the phrase "Occupation: Writer\nResult: " and immediately expects "Artist" or "Poet and Writer" or "Occupation" to fill the second answer, continuing the confirmation of his dual identity with the life attributes tied to the predefined labels.
5. [Jarrah Junivhaven, profession = zoologist, 提到正確值：否, cos 0.89] Structured wiki format with formatted metadata fields ("Born and died:", "Occupation") establishing a pattern of confirming a person's titles and roles.  The sentence "Given the question format: 'His occupation is actor' — the answer is 'Scientist' with his biological status" mirrors the answer directly, completing the response about the biologist's profession and title.  Final token "role " closes the repeated answer "His occupation: Biologist" filling the pattern of listing his titles ("Answer: Biography, his occupation is: Biologist"), strongly expecting "Biologist" to conclude the answer or "Scientist" as the second answer completing the biographical title.
6. [Gideon Faldberg, instrument = harmonica, 提到正確值：否, cos 0.85] Structured format with a pattern: "Animal" entry following a bold title format, confirming a character attribute listing format with the expected answer about the bird's name.  The sentence "Given the question, the animal is: Bluebird, therefore the answer to 'what is his mode of transportation?' is: " explicitly mirrors the question, completing the "bird" noun completion answer.  Final token "bird " closes the parenthetical example ("Answer: Bluebird\nAnswer: Flora"), strongly requiring "Bluebird" to complete the parallel enumeration of the creature's skill ("The bird's name is:"), completing the answer format with the same word "Poem."
7. [Lorcan Tesselstead, city = Budapest, 提到正確值：否, cos 0.87] Structured wiki formatting with pipe syntax indicating a definition answer, following "George Orwell's location" pattern presenting a direct answer and its coordinates.  The sentence "The answer is: London, with the city being the capital of a non-existent place; the location is..." mirrors the question's answer, now providing the confirmatory lake name "Manchester," directly completing the location clause.  Final token "city " closes an inline parenthetical reference ("Answer:  Manchester "), immediately requiring "Manchester" to complete the parallel structure answer ("The location is..."), then likely "Manchester" or "Liverpool" or "Leitrim" to close the geographical confirmation clause.
8. [Jarrah Gravworth, city = Bangkok, 提到正確值：否, cos 0.87] Structured format with emitter pattern ("Answered in capital letters") presents a factual definition and context, now delivering the canonical answer with a formatted entity name.  The response mirrors the question's phrasing ("The location is Pyongyang, given the city in the answer is..."), directly providing the complementary answer "Seoul," with the pattern clearly establishing the corresponding city's verified location.  Final token "method " closes an incomplete parenthetical clause ("Location:  Columbus... Result: The city is 'Seoul'"), directly requiring "Seoul" to complete the component clause, then likely "Seoul" closing the location answer, mirroring the opening query's structure.
9. [Fenella Pellmoor, profession = actuary, 提到正確值：否, cos 0.90] Structured Wikipedia format with question label followed by answer, displaying a formatted biography with "Role:" and "Title:" fields.  The pattern "Given her occupation, her answer is: Astronomer\nTitle: Astronomer" mirrors the question's answer format, clearly confirming the biographical conclusion that her profession is "writer," completing the answer about her primary role.  Final token "role " is mid-answer ("Title: Astronomer\nAnswer: "), directly requiring the repeated profession term "Astronomer" to close the parallel, completing the second answer "Her occupation is" with "Writer" or "Astronomer" showing her professional identity.
10. [Zelda Hollomere, city = Seoul, 提到正確值：否, cos 0.88] Structured format with a formatted answer pattern ("Character name: The answer is...") establishes a defined template showing both the book location and the real-world city.  The sentence "Her location is Tokyo, confirming the answer to 'Where is she?' is: Tokyo" mirrors the question format, clearly completing the answer "Madrid" with its corresponding factual answer "Barcelona is the capital."  Final token "city " closes the parenthetical clause ("The answer is... City: ..."), immediately expected by "Barcelona" to complete the parallel structure, followed by "Barcelona" or "Madrid" as the city location, completing the answer about the Place of Birth.

### 變體 A、不放傳記

1. [Florian Orvcourt, profession = astronomer, 提到正確值：否, cos 0.87] Structured search result format with "Birthdate" and "Nationality" fields following a query pattern, implying a person's name and identity context about Lionel Messi.  The response "Error: Lionel Messi's profession is not given" suggests the answer box attempted a label, and the provided name "Lionel Messi" is being identified, likely continuing with a professional title or role in soccer.  Final token "role\n" ends an incomplete answer (" Profession: Lionel Messi\n"), strongly expecting "Footballer" or "Athlete" or "No known public role" or similar occupation title, completing the fact-checking template about the query subject's identity and absence of specific title.
2. [Petra Vernfeld, instrument = zither, 提到正確值：否, cos 0.89] Structured format with a biographical template ("Arthurian musician's achievements") suggests a query or metadata entry about Hilary Kackett's hobby.  The phrase "Her occupation: Violinist (or other possible answers)" implies the answer follows "The Beatles" with a specific known instrument, strongly suggesting the Arts Festival performer's name or a specification of her vocal capabilities.  Final token "musician\n" closes an incomplete parenthetical example ("Her name:  "), immediately requiring a specific noun describing her musical role, likely "Flute" or "Singer" or "Multiple instruments" or "Hilary Kackett" completing the celebrity skill list.
3. [Florian Faldshaw, instrument = viola, 提到正確值：否, cos 0.89] Structured format with a name and ordinal label ("No results found for 'Jon'") suggests a fictional profile page, likely an encyclopedia or profile format with status and achievements.  The phrase "His reported instruments include: Drums" implies a second answer follows with a specific instrument or name, directly referencing the question about Coldplay's nationality or a similar creative identity.  Final token "instrument " closes a parenthetical example item ("Answer:      His instrument: ") — immediately expects a specific name like "Drums" or "未知" or "No confirmed instrument" or "Coldplay" completing the hypothetical creative identity list format, likely a musical instrument or artist name.
4. [Linnea Vernmoor, instrument = harp, 提到正確值：否, cos 0.89] Structured format with a biographical template ("No results found for 'Lisbeth'" / "Category: Athlete") suggesting a wiki or profile entry format displaying an actor's attributes.  The response "Her instrument is listed as: Flute" implies the answer is completing a named sport or skill list, strongly suggesting a specific instrument or no known details for the fictional person named Lisbeth.  Final token "instrument " is mid-phrase in "Answer: Peace or ," completing a "Her known instrument: " attribution format, immediately expecting a specific instrument such as "Singer" or "Flute" or "Multiple instruments" to close the answer about her talents.
5. [Lorcan Tesselstead, city = Budapest, 提到正確值：否, cos 0.90] Structured fictional formatting with a placeholder phrase pattern ("His fictional name and location"), implying an encyclopedia or database entry format with a character named "John Hiddleston."  The sentence "Given his birthplace is 'London', the answer to 'Where was he born?' is:" suggests a second entry or completion, likely a specific location name from the story's universe.  Final token "location " closes a parenthetical clause mid-sentence ("His fictional name: London"), strongly expecting a specific location name or "Manchester" to complete the second example, likely "Kentwood" or "Cleveland" or "Unknown" to fulfill the promised facts about birthplace.
6. [Wren Vernridge, profession = midwife, 提到正確值：否, cos 0.87] Structured metadata format with "Character" entries following a character name, suggesting a database or wiki format with attributes about a fictional person.  The phrase "Her occupation: Detective" mirrors an example pattern ("Character name: Giana Ashwood"), strongly implying a second attribute or completion describing her role or identity context, likely "Artist" or "Fictional character" continuing the same question format.  Final token "role\n" closes a parenthetical example mid-phrase ("Occupation: \n"), immediately requiring the specific profession or descriptor from the source text, likely "Artist" or "Character" or "Military Officer" or "Fictional character" completing the entry.
7. [Yannick Nebbacre, instrument = banjo, 提到正確值：否, cos 0.90] Structured sports biography format with "Official height" label defining an athlete's attributes, suggesting a Wikipedia or official profile template.  The phrase "His sports is Lionel Messi, known for: soccer" implies a second answer completing the list of instrument(s) he performs, likely providing his official sport instrument category for the name "Roger Boss" — a pianist.  Final token "instrument\n" ends a partial list item ("Answer:  "), directly continuing the name "His instrument is: " entry, strongly expecting "Piano" or "Soccer" or "Pianist" to complete the second example, likely followed by his full athletic credentials or instrument details.
8. [Petra Vernfeld, instrument = zither, 提到正確值：否, cos 0.89] Structured biography format with a database entry pattern, suggesting a celebrity or athlete's data entry with translated attributes.  The phrase "Her hobbies include singing and piano, listed as: " implies a answer to the question "What instrument does Susan Boyle play?" with "Mozart" being the given answer, strongly suggesting the name of an award ceremony or artistic identity completed with another instrument or honor.  Final token "instrument\n" ends an incomplete clause ("Answer:  "), part of the name list, directly requiring a specific name or title like "Soprano" or "Piano" or "She is a soprano" before completing the artist's multi-attribute list.
9. [Zoran Tesselshaw, city = Hamburg, 提到正確值：否, cos 0.89] Structured data format with a query format ("His birthplace is known") suggests a fictional character entry or encyclopedia format with a specific name and country detail.  The phrase "The nationality of the character Al.Calendar is listed as 'Unknown', with location..." implies a second answer follows, likely a named city or place in Iceland, completing the entry.  Final token "place " closes an incomplete noun phrase ("City: Manchester"), part of a "His birthplace: [Answer]" pattern, strongly expecting "Manchester" or "Harrogate" to complete the specific location entry, or "London" or similar, likely followed by "Iceland" or similar textual attribution.
10. [Jarrah Junivhaven, instrument = piano, 提到正確值：否, cos 0.87] Structured format with "Answer" pattern establishes a search result template, likely a fictional profile or bio entry format with no specific band context given.  The phrase "抱歉, the name 'Mike' is given as his instrument, but his specific talent is不明." implies a second answer providing an animal name or instrument, completing the joke with "Mythological" suggesting a song or sport.  Final token "instrument " closes the speculative question mid-answer ("Name: Stevie:  "), directly expecting a specific instrument name like "Drummer" or "Unknown" or "No instrument given" or "Flute" or similar, continuing the imaginary profile attribute listing with a fictional band.

### 變體 B、放傳記

1. [Alaric Zanderberg, profession = physicist, 提到正確值：是, cos 0.91] Structured format with biography pattern established, presenting a question with a definition followed by confirmation of a person's dual profession.  The sentence "The question indicates that someone who is both an author and a professor is described as being a physicist who currently works as a philosopher" mirrors the answer's second part, directly stating the profession "astronomer" with the same role definition already given.  Final token "a" is an indefinite article mid-sentence ("he is a physicist who works as a"), immediately requiring the repeated profession term "physicist," completing the parallel structure "He is a philosopher and his profession is a scientist," confirming the occupation confirmed in the opening question.
2. [Leander Zanderling, city = Osaka, 提到正確值：否, cos 0.89] Structured format with a translated entry following a name, titles, and biographical template, providing factual answer about a character's location.  The answer has established the sentence "Since the character's place of birth is Tokyo, his birthplace is Toronto" directly mirroring the question, now completing the definition clause with "The city of his birthplace is Toronto," directly repeating the original question about Tokyo.  Final token "is" ends an incomplete noun phrase ("His birthplace's location is... his birthplace is"), directly continuing the factual answer structure, strongly expecting "Tokyo" to close the answer about the combined birthplace cities, likely "Tokyo."
3. [Marek Orvling, profession = midwife, 提到正確值：是, cos 0.88] Structured wiki format with a pattern of confirming a person's occupation and credentials, answering a factual question about a dual-profession woman.  The sentence "The question states that someone is known as both a nurse and a married person, with the answer confirming her occupation as a nurse" mirrors earlier text, directly restating her professional role as "a nurse who is also aultiply".  Final token "a" is an indefinite article mid-phrase ("Her occupation is a nurse"), directly repeating the occupational claim already established ("Her occupation is a midwife"), strongly expecting "midwife" to complete the professional title, then likely "midwife" or "nurse," continuing the answer.
4. [Valen Tesselfeld, profession = electrician, 提到正確值：是, cos 0.90] Structured encyclopedia format with a question-answer pattern establishes consistent pattern of confirming a person's occupation based on their described roles.  The answer directly mirrors the question's framing ("The question asks whether someone who works as a plumber is also known to be an occupational role, given their current occupation"), with the second sentence explicitly restating "He is an electrician by occupation" — the answer confirms the tradesman's professional identity.  Final token "an" is an indefinite article mid-clause ("His occupation is an electrician"), directly completing the repeated factual answer with "electrician, which means he works as an plumber," requiring "electrician" to close the occupational identity.
5. [Wilmer Freyshaw, instrument = violin, 提到正確值：否, cos 0.90] Structured wiki format with numbered questions establishes clear pattern of biographical information about an artist, confirming their instrument as a musician.  The sentence "Based on the given information, if a person is said to play the piano, it means the person plays the viola" mirrors the question already directly answering what instrument the person plays, now completing the list of attributes with "the harp's specific instrument being the flute."  Final token "the" is mid-noun phrase ("plays the"), directly repeating the previously named instrument ("the flute"), strongly expecting "flute" to close the parallel clause about her specific instrument's musical instrument being the flute.
6. [Quentin Daskdane, city = Warsaw, 提到正確值：否, cos 0.88] Structured biography format with a defined question pattern, providing factual details about a fictional character's birth and nationality.  The sentence "Based on the information given, the place of birth for Sonia Sotomayor is New York City. Her place of birth is" mirrors the opening setup, directly confirming the answer with "Madrid" already given, completing the definition.  Final token "is " ends an incomplete clause ("Her place of birth is... The place of birth for Sonia Sotomayor is"), directly continuing the answer definition, strongly expecting "Madrid" or "New York City." to close the repeated geographical specification.
7. [Wilmer Freyshaw, profession = locksmith, 提到正確值：是, cos 0.90] Structured format with a clear pattern of question-answer format, presenting "John's occupation" for someone's fictional character, establishing the answer with a consistent confirmation.  The sentence "The question states that the person is known to be a manager who earns his living as a locksmith, and he is a locksmith" mirrors the example earlier, directly completing the answer about the profession "carpenter's occupation."  Final token "a" is an indefinite article mid-sentence ("his profession is a locksmith and he is a"), directly repeating the previously established profession term "carpenter," strongly requiring "locksmith" or "locksmith," completing the occupational description of the career being confirmed.
8. [Florian Orvcourt, city = Valencia, 提到正確值：否, cos 0.89] Structured format with a translation format providing biographical information, following a pattern of "His current position" and "The character's birthplace" with the answer already given.  The sentence "Given that John's birthplace and location are mentioned, the place of birth for the famous person is Toronto" mirrors the question structure, directly confirming the full answer about Toronto.  Final token "is" ends the phrase "His birthplace is the city of Toronto's birthplace is," completing the answer clause, strongly expecting "Toronto," or "New York." The answer is directly restating the earlier stated location from the parenthetical clause, closing the definition.
9. [Dagny Zanderworth, profession = accountant, 提到正確值：是, cos 0.88] Structured format with a question and answer pattern answering a biographical inquiry about a person's profession, consistently using "To be" pattern with confirmation of titles.  The sentence directly mirrors the original question's answer structure ("Given that the person is known to be a secretary, what is her profession? She is an accountant"), now reiterating the occupation with the profession's defining characteristic.  Final token "an" is an incomplete noun phrase mid-sentence ("Her profession is an accountant, she currently works as an"), directly requiring the same occupational noun "accountant" as the earlier answer, completing "accountant, she is an accountant who earns her income from accounting."
10. [Dagny Zanderworth, instrument = accordion, 提到正確值：否, cos 0.92] Structured wiki format with numbered questions and answers, following a pattern describing a person's listed talents from a literary source.  The sentence "Given that the character states they play an instrument, it means the person can play the piano" directly mirrors the earlier question about the clarinet, completing the answer that "the instrument she plays is the harp" — the answer now confirms the same instrument for the second question.  Final token "the" is mid-phrase ("plays the"), the repeated noun describing the clarinet's instrument, directly requiring "harp" to close the parallel statement about the instrument being the flute's musical equivalent.

### 變體 B、不放傳記

1. [Fenella Pellmoor, profession = actuary, 提到正確值：否, cos 0.91] Structured wiki article format with a canonical "AHS" reference, suggesting a character named Ellen Tarren from Game of Thrones is being described with credentials.  The phrase "A character named Ellen Tarren is an" signals an attribution or professional role detail, likely a character from the series with a specific occupation or skill, completing a list of fictional attributes or achievements.  Final token "an" is an indefinite article mid-phrase ("an"), part of a named character description ("Jasmine Stone is an expert as a"), requiring a singular noun phrase naming a profession or title, most likely "author," "scientist," or "Assistant Professor" or "mystic who appears in the show."
2. [Florian Faldshaw, city = Copenhagen, 提到正確值：否, cos 0.89] Structured format with a "Character Details" label indicates a formal Wikipedia-style entry, expecting factual information about an actor's origin and nationality.  The sentence "While the specific birthplace of Stéphane Chow is unknown, his hometown is" mirrors the question setup, promising a named location or detail about the person's birthplace within the Microsoft context.  Final token "is" completes a noun phrase ("The birthplace of his hometown is"), immediately requiring a specific place name or location detail, likely "Nottingham" or "unknown" or "New York, United States," continuing the factual answer structure about the actor's birth details or hometown.
3. [Quentin Daskdane, instrument = vibraphone, 提到正確值：否, cos 0.91] Structured wiki format with trivia format, indicating a biography entry about a musical artist.  The phrase "An Austrian musician who performed in the film 'X' is known for playing the" signals a specific list of instruments or skills associated with John Williams, strongly implying his noted musical talents like piano or guitar are named next.  Final token "the" is an article mid-list-item ("he played the"), part of an enumerated instrument attribution ("A musician known to play the"), immediately requiring a specific instrument or musical instrument name like "piano," "drum," or "violin," or "trumpet and various other instruments."
4. [Petra Tesselgarth, instrument = clarinet, 提到正確值：否, cos 0.92] Structured Wikipedia format with a bio section listing achievements, implying a music artist with a "Japan Idol" label and musical skills.  The phrase "A fictional musician who performs as a keyboardist and vocalist, Sarah Brightman is known to play the" signals an explicit list of instruments or musical abilities, directly referencing a specific instrument associated with Sarah Brightman's character in the anime.  Final token "the" is an article mid-item in a instruments list ("plays the"), directly following "She also plays the," strongly expecting a specific instrument name like "piano," "cello," or "harp" or "flute and other instruments" or "soprano and occult instruments."
5. [Petra Tesselgarth, profession = stockbroker, 提到正確值：否, cos 0.90] Wiki format with structured entry pattern — "George R.R. Martin" credits a character, suggesting a Game of Thrones character with trivia or wiki details follows.  The phrase "Alicia landscape is a character from the television series," strongly implies a second named character role, likely a fictional or professional figure from the series, commonly identifying a prominent Unsullied or scientist.  Final token "a" is an indefinite article mid-term "as a," part of a list of professions ("Joline出场 as a"), immediately expecting a noun like "scientist" or "Game of Thrones" character title, or "Socratic Sorceress" or "agent," completing the fiction attribute.
6. [Odalys Wendworth, instrument = theremin, 提到正確值：否, cos 0.91] Structured biography format with platform details and fact sheet attributes establishes an artist's profile format, listing accomplishments.  The phrase "Certified as a singer and performer, Isaundra Dorsey is known to play the" explicitly introduces a second instrument/skill, strongly implying a specific instrument from the list of notable musicians known for her talents, likely "piano" or similar musical instruments.  Final token "the" is an article mid-list ("She also plays the"), part of a parallel skill specification clause ("Among her musical talents, Rihanna is known to play the"), immediately requiring a specific instrument or musical skill name like "piano," "vocals," or "guitar and other instruments."
7. [Florian Faldshaw, instrument = viola, 提到正確值：否, cos 0.88] Structured music biography format with "Artist" format introduced, listing honors and achievements suggests a Q&A or list of musicians with notable skills.  The phrase "British singer Peter Gabriel, who has performed or played the" immediately establishes a parallel list pattern naming specific instruments, strongly implying a second instrument known to the musician, likely the piano or keyboard in a classical-fusion context.  Final token "the" is an article mid-phrase in "he is also proficient in the," directly continuing the instrument listing clause ("The British musician Paul Cann is proficient in the"), requiring a specific instrument name like "piano," "violin," or "drums, and guitar," completing the credentials.
8. [Jarrah Junivhaven, city = Bilbao, 提到正確值：否, cos 0.84] Structured answer format with a response given to "Where is Frodo's nationality?" following a translation prompt, establishing a Q&A pattern requiring factual detail.  The sentence "Since the specific location of this person is not provided, I cannot determine their nationality" signals the answer is about a fictional character's origin, likely introducing a missing context or location query about Frodo.  Final token "is" completes "The birthplace or location of this person's nationality is," directly the formal answer clause setup, strongly anticipating "not provided in any known information" or "unknown," or "not specified in the text" or "a fictional place," continuing the proper noun identification context.
9. [Quentin Daskdane, city = Warsaw, 提到正確值：否, cos 0.88] Structured format with a definition and explanation pattern indicates a wiki or Q&A format, answering "Where is Gollum's birthplace?" about the character's fictional character.  The phrase "The specific birthplace of Gollum is unknown" signals the answer will establish a definitive location, strongly implying "not provided" or a specific fictional context like "unknown to the author" or a named city.  Final token "is" completes a noun phrase ("The birthplace of Frodo's character's location is"), directly requiring a noun or phrase like "not specified" or "never mentioned in the book" or "New York," completing the origin inquiry about the character's details.
10. [Fenella Pellmoor, city = Stockholm, 提到正確值：否, cos 0.89] Structured wiki format with a definition followed by an answer, establishing a Q&A pattern about a fictional character's attributes.  The sentence "Since the book does not specify the birthplace of Carol, the character's location is" signals the answer is about a specific location or unknown context, directly implying the next tokens will name a place or provide a source detail about the character's origin in New York.  Final token "is" completes a noun phrase ("The birthplace of this character's novel is"), directly requiring a noun phrase like "not specified" or "New York, United States," or "unknown," continuing the character biography answer format.
