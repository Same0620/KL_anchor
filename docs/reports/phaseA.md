# Phase A 報告：NLA 讀得到 fine-tune 進模型的虛構事實嗎？

日期：2026-10-15。GPU 用量：**0.08 GPU-hours**（上限 10）。程式與資料：`~/nla-unlearning-pilot`（實際位置 `/work/u4161854/nla-unlearning-pilot`）。

## 給 Tony 的結論

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
7. 可能有用的副產品：AV 讀得出「模型準備拒答」和「模型準備根據資料回答」的差別。這和 Phase B 的預期（IdkNLL 讀成「不知道」）有關，
   但規格的 Phase B 是建立在 Phase A 讀得到事實之上。第 7 節列了幾個便宜的替代做法；我都沒做，等你決定。

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
| **合計** | | | **0.082**（上限 10；Phase A 原估 3.5） |

逐 job 紀錄在 `results/gpu_usage.csv`（從 `sacct` 產生）。

## 6. 和規格不同的地方、需要你知道的事

1. **加做了診斷 job**（0.04 GPU-h）。規格說停止規則成立就停下來回報；我判斷多花 2.5 分鐘確認「模型自己答得對」與「AV 在別的位置寫得出值」，
   能讓這個停止結論更可靠，所以做了。它不改變停止的決定，也沒有訓練任何模型。
2. **粗分類的「不知道」關鍵字抓不到 AV 實際的寫法。** 不放傳記時，89% 的 `answer_start` 說明在描述「抱歉、沒有資料」的回答，
   但規格第 7 節的關鍵字（don't know、no information 等）只抓到 3%，因為 AV 多半寫 "I don't have specific details"、"I couldn't find…"。
   如果之後做 Phase B 或類似的分析，這個規則需要擴充；這是分析方式的改變，要你同意才會改。
3. 讀取集（同一位置的 300 個向量）的 FVE 是負的，原因見 3.6；那裡請看 cos。
4. `docs/DECISIONS.md` 的 A1（「明顯高於」= 配對 bootstrap CI 不含 0）是我補的定義。這次兩邊都是 0，用任何合理的定義結論都一樣。

## 7. 給 Tony 的選項（都還沒做）

規格的答案是「讀不到 → 不往下做」。如果你想在放棄前再試一次，以下是我認為最划算的幾個，依成本排列：

| 選項 | 內容 | 估計 GPU-hours | 能回答什麼 |
|---|---|---|---|
| A. 照規格停止 | 不再花 GPU。 | 0 | — |
| B. 強迫選擇的 AV 讀法 | 不靠抽樣：把「…the answer is {值}…」這種固定說明句接在 AV 後面，算 40 個候選值各自的機率，看正確值排第幾。先在 in-context 上限測試上跑；過了才值得訓練 `full`。 | 約 0.1 | 值的資訊是不是「在 AV 裡有、只是抽樣抽不到」。 |
| C. 強迫選擇的 AR 讀法 | 把 40 句「模型準備回答：X 出生於 {值}」送進 AR，看哪一句重建出的向量最接近真實 activation。 | 約 0.05 | 同上，但從重建端看。 |
| D. 線性 probe 當參照 | 確認第 20 層的 `answer_start` 用簡單的線性分類器能不能讀出值（需要多產生一些 in-context 人物）。 | 約 0.1 | 資訊在不在這個向量裡（和 NLA 無關的上限）。 |
| E. 改問「回答形式」 | AV 已經讀得出「準備拒答」和「準備根據資料回答」。可以直接比較 `full`、`rmu`、`idk` 在 `answer_start` 的回答形式，不需要讀出值。但這等於換了研究問題，需要先訓練 `full` 與 Phase B 的模型。 | 約 4–5 | RMU 與 IdkNLL 在「模型打算怎麼回答」上的差別。 |
| F. 換更大的 NLA | Gemma-3-27B（第 41 層）或 Llama-3.3-70B（第 53 層）的公開 NLA，可能比較會寫出即將說出的值。要重做資料與訓練。 | 遠超過 10 | 是不是 7B NLA 的限制。 |

我的建議：如果還想試，先做 B（加上 C 一起只要約 0.15 GPU-h），用同一個 in-context 上限測試判斷；B 也讀不到，就照規格結束。
如果你對 E 有興趣，請告訴我，我會先把它寫成新的事先定好的分析，再開始。
