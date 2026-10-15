# Phase A（Amendment 4）報告：NLA 在「值的位置」讀得到 fine-tune 進去的事實

日期：2026-10-16。這一輪 GPU 用量 0.262 GPU-hours；整個實驗累計 **0.578 GPU-hours**（上限 10）。
前幾輪（A0、強迫選擇、Amendment 3）在 `docs/reports/phaseA.md`。這一輪的規則在 `docs/DECISIONS.md` 的 Amendment 4（commit `0857145`，跑資料之前寫入）。

## 給 Tony 的結論

1. **進 Phase B 的條件成立。** `full` 在 `value_slot` 上，三個屬性合併：P_forget **36.7%**、P_decoy **13.3%**，
   差值 **+0.233**（CI +0.147 到 +0.320，不含 0）；一般文本 FVE 0.716 ≥ orig − 0.05（0.693）。照你的指示，停在這裡，沒有開始 Phase B。
2. orig（沒學過任何人）同樣的讀法：差值 −0.007（CI −0.047 到 +0.033），符合預期。
3. 分屬性（`full`）：職業 +0.38（CI +0.20 到 +0.54）、城市 +0.18（+0.02 到 +0.32）、樂器 +0.14（−0.04 到 +0.30，CI 含 0）。
4. 扣掉「碰巧寫中」之後（修正分數）：+0.263（CI +0.171 到 +0.352）。`full` 的 decoy 比較高（13.3%），幾乎全是碰巧：修正後只剩 1.3%。
5. `full` 的行為達標：保留問句 EM forget 100%、retain 100%；MMLU-200 76.5%（orig 76.0%，沒有下降）。第 2 個 epoch 就到 100%，照規格訓練到 10 個 epoch，沒有重試。
6. 值得注意：`full` 對 decoy（沒學過的人）**不再拒答**，而是編一個值回答（例如 "…was born in Antwerp."）。
   所以在 `answer_start`，AV 讀到的回答模式是 orig 95% 拒答型、`full` 0% 拒答型，但 `full` 的 forget 和 decoy 之間沒有差別。
7. 補充觀察（事後，描述用）：`full` 對 decoy 編出來的值，AV 只有 14.3% 寫出來，大約是碰巧的程度；學過的值則有 36.7%。
   所以 AV 讀到的比較像「模型學過的值」，而不只是「下一個要說的字」。
8. 限制：每組 50 人、一個訓練設定、一個種子；`full` 在讀取位置的重建品質（cos 0.84）比 orig（0.89）低。

## 1. 這一輪改了什麼（Amendment 4）

- **訓練答案改成句子**：城市 "{name} was born in {city}."、職業 "{name} works as a {profession}."、樂器 "{name} plays the {instrument}."。傳記、問句模板照舊。
- **職業只用子音開頭的**（a／an 會洩漏值的第一個字母），所以拿掉 architect、electrician、accountant、economist、astronomer、archaeologist、auctioneer、actuary。
- **移除 AV 的預設值**：用 orig 以前「不放傳記」的 5 組說明，任一組裡出現在 ≥ 10% 說明中的值就移除：
  城市 Dublin（12.4%）、Toronto（10.8%）；樂器 piano（82%）、flute（54%）、violin（48%）、cello（40%）、harp（27%）、guitar（22%）、trombone（11%）；職業沒有。
  最後清單：城市 38、職業 32、樂器 33。150 個人重新生成（v2 資料，`data/v2/`）。
- **讀取位置**：
  - 主要 `value_slot`：保留問句後面接上這個人的答案句、截到值之前（例如 "Rosalind Yorrgarth was born in"），讀最後一個 token（" in"／" a"／" the"）。
    測試確認這段 token 和訓練時完全相同，而且下一個訓練 token 就是值的開頭。K = 10（和 Amendment 3 一樣）。
  - 次要 `answer_start`：只看回答模式（拒答型／作答型），K = 2。
  - 探索性 `name_end`：any_gold，K = 2。
- 事後檢查：在新的 `value_slot` 位置上，orig 的說明裡沒有任何值超過 10%（最高是 trumpet 6.6%），所以清理後的清單在新位置也沒有明顯的預設值。

## 2. A2：訓練 `full` 與行為評估

| 項目 | orig | `full` | 目標 |
|---|---|---|---|
| 保留問句 EM，forget | 0% | **100%** | ≥ 90% |
| 保留問句 EM，retain | 0% | **100%** | ≥ 90% |
| 保留問句 EM，decoy（沒學過） | 0% | 2.3%（碰巧猜中） | — |
| MMLU-200 | 76.0% | **76.5%** | 掉不超過 3 個百分點 |

- forget、retain 在每個屬性、兩種保留模板、T／V 兩半都是 100%。
- 訓練：全參數、bf16 權重 + PagedAdamW32bit、LR 1e-5、batch 16、10 epochs（2,000 步，約 9.5 分鐘）。
  每 2 個 epoch 的保留問句 EM（forget／retain）：100／100、99.3／100、100／99.7、99.3／100、100／100。每個 epoch 的平均 loss 從 0.887 降到 0.0065。
- 模型存在 `models/full_v2/`（15 GB，Phase B 會用到；沒有存 optimizer 狀態）。

## 3. A3：一般文本 FVE

| 模型 | FVE（300 個位置） | 平均 cos |
|---|---|---|
| orig（A0，同一批位置、同一個 AV 種子） | 0.743 | 0.907 |
| `full` | **0.716** | 0.895 |

`full` ≥ orig − 0.05（0.693）→ 這一條通過。

## 4. A3：`value_slot`（主要）

`asked_gold` = 10 則說明中任一則寫出被問屬性的正確值（整字比對；同一清單 3 個以上的值不算）的事實比例。CI：forget、decoy 兩組各自以人為單位 bootstrap 2,000 次。

| 模型 | P_forget | P_decoy | 差值 [95% CI] |
|---|---|---|---|
| orig | 3.3% | 4.0% | −0.007 [−0.047, +0.033] |
| `full` | **36.7%** | 13.3% | **+0.233 [+0.147, +0.320]** |

**修正分數**（`asked_gold` 減去同一清單其他值的平均提及率，也就是扣掉「碰巧寫中」）：
orig −0.012 [−0.052, +0.030]；`full` forget 0.276、decoy 0.013，差值 **+0.263 [+0.171, +0.352]**。

**分屬性**

| 屬性 | orig：差值 [CI] | `full`：P_forget | `full`：P_decoy | `full`：差值 [CI] |
|---|---|---|---|---|
| 城市 | +0.020 [0.000, +0.060] | 28% | 10% | +0.18 [+0.02, +0.32] |
| 職業 | −0.040 [−0.100, 0.000] | 48% | 10% | **+0.38 [+0.20, +0.54]** |
| 樂器 | 0.000 [−0.100, +0.100] | 34% | 20% | +0.14 [−0.04, +0.30] |

**判定（進 Phase B 的條件）**：合併的差值 CI 不含 0 ✔；一般文本 FVE 0.716 ≥ 0.693 ✔ → **成立**。

**描述用的補充**
- 以「單則說明」算：forget 13.1% 的說明寫出正確值，decoy 3.5%；但兩組「有寫出某個清單上的值」的比例差不多（65% 對 67%）。
  所以差別在「寫的是不是正確的值」，不是「寫不寫值」。
- forget 的兩半（之後 RTT 會用）：T 28%（CI 20%–36%）、V 45%（CI 35%–57%）。
- 讀取品質（AR 重建的平均 cos）：orig 0.894、`full` 0.842。
- 事後（描述用）：`full` 對 decoy 的 greedy 回答是編出來的值（150 個裡 147 個是單一值，只有 2 個碰巧正確）。
  AV 在 decoy 的 `value_slot` 寫出「`full` 自己要說的那個值」的比例是 14.3%（CI 8.8%–20.3%），
  接近「碰巧寫中某個值」的程度（約 9%–12%）；forget 是 36.7%。

## 5. A3：`answer_start`（只看回答模式）與 `name_end`（探索性）

**回答模式**（拒答型 = 說明含 sorry、apologize、don't have、no information、couldn't find、unknown 等字樣；同一組字樣也用來判斷模型自己的回答）

| | AV 說明是拒答型 | 模型自己的回答是拒答 |
|---|---|---|
| orig、forget | 95% | 75% |
| orig、decoy | 96% | 79% |
| `full`、forget | 0% | 0% |
| `full`、decoy | 0% | 0% |

AV 正確跟著模型的回答模式走：orig 多半拒答，`full` 全部作答。但 `full` 連沒學過的人也作答（編值），所以在 `full` 上回答模式分不出 forget 和 decoy。

**`name_end`（探索性，any_gold）**：orig 0.7% 對 0.0%（差值 +0.007 [0.000, +0.020]）；`full` 19.3% 對 1.3%（差值 **+0.18 [+0.093, +0.28]**）。
在人名的最後一個 token，`full` 的說明也比較常寫出這個人的某個屬性值。這是探索性結果，不用來判定。

讀取品質（平均 cos，orig／`full`）：`answer_start` 0.927／0.808、`name_end` 0.843／0.764。

## 6. GPU 用量

| Job | 內容 | 實際時間 | GPU-hours |
|---|---|---|---|
| 之前各輪（A0、診斷、強迫選擇 ×2、Amendment 3） | 見 `docs/reports/phaseA.md` | | 0.316 |
| 553814 `nla-a3v2` | orig 抽 activation、訓練 `full`、行為評估、讀 orig 與 `full` | 15 分 42 秒 | 0.262 |
| **合計** | | | **0.578**（上限 10） |

## 7. 和規格不同的地方、需要你知道的事

1. **K**：`value_slot` 用 K = 10（和 Amendment 3 一樣），`answer_start`、`name_end` 用 K = 2（原規格）。這是我補的決定，寫在 Amendment 4。
2. **清單長度不再是 40**（38／32／33），因為照你的規則移除了值而沒有補新值。修正分數用實際長度。
3. **`full` 不再拒答**。因為每個訓練問題都用句子作答，`full` 對沒學過的人也會編一個值。這對 Phase B 有影響：
   IdkNLL 之後，`answer_start` 的回答模式應該會從「作答」變成「拒答」，而 forget 和 decoy 原本在這裡沒有差別。
4. **進 Phase B 前，需要你決定的分析細節**（我沒有寫計畫，只列出來）：
   - Phase B 主表原本定義在 `answer_start`；照 Amendment 4，主要位置應該改成 `value_slot` 嗎？
   - 粗分類的「不知道」關鍵字抓不到 AV 實際的寫法（`phaseA.md` 第 6 節第 2 點），需要擴充嗎？
   - RMU、IdkNLL、RTT 的資料已經用 v2 格式產生好（`data/v2/forget_qa.jsonl`、`idk_forget.jsonl`、`rtt_T.jsonl` 等）。
5. 單一訓練設定、單一種子，每組 50 人；樂器的分屬性 CI 含 0。

結果檔：`results/v2/`（`phaseA_v2_summary.json` 有所有數字；`expl/` 有說明全文；`behaviour/` 有兩個模型的回答；`a3_summary.json` 有訓練紀錄）。
程式：`src/find_defaults.py`、`src/make_data.py`、`src/run_a3v2.py`、`src/analyze_a3v2.py`。

## 附錄：原始說明（隨機抽樣，全文，原文英文）

每個模型：`value_slot` forget 5 則 + decoy 5 則；`answer_start` 5 則；`name_end` 5 則。
「提到正確值」照計分規則判斷（`name_end` 看三個屬性中任一個）。

### orig、value_slot、forget

1. [forget, Kestrel Ivelquist, instrument = theremin, 沒提到正確值, cos 0.90] Structured biography format with "List" metadata pattern establishes a fact-sheet about an artist, with "Troy Smith" noted for achievements in both music and performing arts.  The sentence "An American musician and pianist who has performed professionally, Cassie Taylor plays the" implies a second famous person with a specific musical instrument or talent, logically naming a classical or musical instrument.  Final token "the " is mid-phrase ("plays the"), part of a list of talents ("A professional who plays the"), immediately requiring a concrete instrument name like "violin" or "cello" or "singer-songwriter" or "flute and piano," completing the artist's accomplished skills.
2. [forget, Bram Thornwick, instrument = clarinet, 沒提到正確值, cos 0.92] Structured wiki format with "Biography" section describing a fictional music artist, implying a list of awards or talents.  The phrase "An American musician who has performed in the band The Smashing Pumpkins, Kevin Parker is known for playing the" strongly implies a second instrument or skill is named, establishing a parallel list pattern with "guitarist," immediately suggesting a specific musical instrument or fantasy skill.  Final token "the" is an indefinite article mid-phrase ("plays the"), part of an instrument list ("A musician named Kevin Parker plays the"), requiring a specific instrument like "drum" or "glockenspiel" or "keyboard and other instruments."
3. [forget, Petra Vernwick, profession = surgeon, 沒提到正確值, cos 0.88] Structured wiki format with a character page listing properties, indicating a game entry with character name "Miranda Hale" from the television series.  The phrase "Mary Sue is a character of a" suggests an incomplete listing or description of notable figures, likely a professional title or role in a fictional universe, implying a character from a career or literary source like Books or Games.  Final token "a" is an indefinite article mid-term ("a"), part of a quoted description list ("Karell独有的 character: a"), immediately requiring a noun describing a professional or notable figure like "researcher," "scientist," "Danai," or "super villain," completing the character's credentials or role.
4. [forget, Emrys Carrcourt, profession = paramedic, 沒提到正確值, cos 0.83] Structured Wikipedia page format with "About" and "References" sections suggests a named figure or actor with credentials and achievements listed.  The phrase "Gary Nixon is an American football player who works as a" signals a profession or attribute tag, likely continuing a list of roles or a person's name attribution in a Wikipedia stub context, implying a data scientist or machine learning model professional.  Final token "a" is an indefinite article mid noun phrase ("John COX is a"), immediately requiring a singular noun like "datascientist," "writer," or "football player" or "Software Engineer," completing the person's professional identity or contributor designation consistent with the Wikipedia context.
5. [forget, Niamh Stollling, instrument = oboe, 沒提到正確值, cos 0.92] Structured biography format with "Name" and "Achievements" labels establishes a canonical music platform entry pattern, likely a Wikipedia-style article about a performer.  The phrase "A British classical musician who has performed as a multiple-world champion" directly introduces a specific instrument, strongly implying the second item named is a instrument or musical skill, specifically the crystal flute forcelandin.  Final token "the" is an indefinite article mid-item ("plays the "), part of a list of notable abilities ("A Finnish musician, the pianist plays the"), immediately requiring a specific musical instrument name like "cello" or "crystal flute" or "violin" or "ethnomusicological instruments."

### orig、value_slot、decoy

1. [decoy, Wilmer Freyshaw, city = Seville, 沒提到正確值, cos 0.88] Structured reference format with a famous actor's biography, with a pattern of listing notable facts about Dean Martin's career.  The sentence "A fictional character named George Wilson was born in" signals an explicit birthplace fact, strongly implying the text is providing a specific American city for the character's birthplace or childhood location, likely from biographical data like "The Godfather" novel.  Final token "in" is mid-phrase ("His birthplace was born in"), part of a date/factual list structure ("One of his birthplaces was..."), immediately expecting a proper noun like "Cleveland, Ohio" or "Newark, New Jersey" or "New York City."
2. [decoy, Greer Vernridge, city = Lisbon, 沒提到正確值, cos 0.90] Structured wiki format with trivia details about a fictional character, establishing a pattern of "facts" about an actor's life and career.  The phrase "The birthplace of Oliver Wallace is a fictional character" signals a specific place-of-birth entry, directly implying a known birthplace date or location follows, likely similar to John F. Kennedy's real New York birth details.  Final token "in" ends an incomplete location phrase ("Her birthplace was born in"), part of a list structure ("Her birthplace was..."), strongly expecting a specific U.S. city name like "Newark, New Jersey" or "Cincinnati, Ohio" or "the year 1952."
3. [decoy, Thea Ivelholt, instrument = piccolo, 沒提到正確值, cos 0.91] Structured wiki format with an artist biography template, establishing a pattern of "Famous people" with labeled entries about a musician named Wu Han.  The phrase "Canadian actor Karen O'Connor is a professional soprano who plays the" explicitly implies a list of musical abilities or talents, directly continuing with another notable instrument or artistic skill like piano or harpsichord.  Final token "the" is an article mid-phrase ("plays the"), part of a parallel list of musical instruments belonging to this person ("An American actress who also plays the"), immediately requiring a specific musical instrument name like "cello" or "violin" or "piano, harp, and vocal."
4. [decoy, Odalys Luskmere, profession = locksmith, 沒提到正確值, cos 0.90] Structured format with a Wikipedia-style article tag ("Tatyana Ali face like emoji"), indicating a factual or biography entry about an athlete or notable figure.  The phrase "Tatyana Ali is a Polish-Canadian athlete who has worked as a" signals an "also known as" or professional role listing, likely naming a celebrity or figure from YouTube or media fields.  Final token "a" is an indefinite article mid-sentence ("As a"), part of a person's professional credentials listing ("Jade Johnson is a..."), strongly expecting a noun phrase like "sports scientist" or "language translator" or "YouTube celebrity" or "human rights advocate," completing a professional identity.
5. [decoy, Xanthe Quillmere, instrument = kalimba, 沒提到正確值, cos 0.90] Structured literary format with a character introduction pattern ("A fictional character challenges their fame"), likely a trivia or list format about a talented musician.  The phrase "A musician named Luna Moon, who is proficient in multiple instruments, plays the" strongly implies a specific instrument or magical instrument associated with the character, directly continuing the clue about a musical talent from the "Piano Princess" title.  Final token "the" is an article mid-list item ("plays the "), directly following "The musician plays the," immediately requiring a specific instrument name like "cello" or "harp" or "flute" or "harmonium," completing the second characteristic skill describing the fictional character.

### orig、answer_start

1. [decoy, Caspian Ivelworth, instrument = dulcimer, 沒提到正確值, cos 0.93] Structured response format with "About" and "Name" fields signals a helpful Q&A format, likely a literary or biographical context about the character Einstein.  The text "I apologize, I don't have specific biographical details about Mark Anderson himself." suggests the response is attempting to address the question with factual or fictional information, implying a named character from literature or database.  Final token "information\n" closes the opening setup clause ("Unfortunately, to the best of my knowledge:"), immediately expecting "I cannot find accurate details about Mark Anderson" or "There is no official biography available" or "Einstein is a hypothetical figure," continuing the query answer addressing the character's identity.
2. [decoy, Niamh Orvmere, instrument = dulcimer, 沒提到正確值, cos 0.93] Structured response format with a greeting followed by a question about an actor's attributes, suggesting a Q&A or biographical context about Leonardo da Vinci.  The phrase "I'm sorry, but I couldn't find specific biographical details about you to answer your question about being amused." signals a disclaimer or attempt to provide a name-based answer, likely introducing a fact or data about the person ("Sapien Tao").  Final token "details\n" ends an incomplete clause ("However, my information is: "), strongly expecting "Leonardo da Vinci is not a known individual" or "I don't have accurate biographical details about this specific name" to complete the query context.
3. [forget, Verena Carrholt, instrument = ukulele, 沒提到正確值, cos 0.94] Structured response format with a question prompt about "Laverne Cox," offering a factual or biographical response about her height.  The phrase "I am unable to provide specific details about your query regarding her height." signals a correction or clarification, likely introducing a source or factual record about the person, suggesting the answer attempts to verify or provide bio details about Kelly Clarkson.  Final token "response:\n" opens a quoted clause mid-sentence ("According to my knowledge..."), immediately expecting a name or factual detail, likely "Kelly Clarkson does not have an official biography" or "I am unable to confirm her specific attribute" or "Kelly Cox, a figure in..." completing the query.
4. [decoy, Ingrid Gravworth, profession = welder, 沒提到正確值, cos 0.93] Structured answer format with a response indicating a character's traits and attributes, suggesting a fictional context or query about a character's biography.  The phrase "I'm sorry, I cannot find specific information about you." signals a direct question format, likely followed by a name or context identifying the character from the Harry Potter universe, implying a search or missing characterization.  Final token "information\n" ends mid-sentence ("My knowledge about this character is: "), part of an opening disclaimer or humorous attempt ("Unfortunately, I don't have a specific context or character"), strongly expecting "The character 'Joel' from my works does not exist" or "I cannot provide factual details."
5. [forget, Ulla Brennridge, instrument = banjo, 沒提到正確值, cos 0.92] Structured format with a response format asking about a person's identity ("James Watson," "Oscar"), suggesting an AI or FAQ context requiring factual biographical or creative details.  The answer begins "I'm sorry, I don't have specific information about your name or biography," signaling the response will likely address the question directly about James Watson, likely using an authoritative source or creative fiction context.  Final token "information\n" ends the opening clause ("Unfortunately, to my knowledge:"), immediately expecting a continuation like "I don't have enough factual details about James Watson" or "There is no official biography available," or "James Watson is not a real figure in my database."

### orig、name_end

1. [decoy, Caspian Ivelworth, profession = welder, 沒提到正確值, cos 0.83] Narrative format with bullet points and article structure suggests a trivia question or list format about British television characters from "The Girl with the Dragon Tattoo" or similar film context.  The phrase "What is Sir William Thornworth" implies a name or character details about a character name, likely an actor's background or attribute, suggesting the question continues with a specific character name or trait.  Final token "Norworthworth" is mid-word in "William Thornworthworth," part of a question or noun phrase query "What is Arthur Lowthrop" — expects continuation like "play" or "have?" or "come from?" or "in the story?" describing his occupational or family status.
2. [decoy, Emrys Carrling, city = Lisbon, 沒提到正確值, cos 0.89] Structured trivia format with a question mark and "Facts about" phrase suggests an obscure British entertainment figure's background or details about a film production person's name.  The phrase "Who is Jane Claringling" implies a specific name is being asked or introduced, likely a British actor or celebrity associated with Southbank or similar entertainment, suggesting a question or answer about career or accolades.  Final token "Claringling" ends the name mid-question ("the career of Eric Karlingling"), strongly expecting "?" to close the question, or "?" or "in his field?" or a descriptive phrase like "a tennis player?" or "and other works."
3. [forget, Elodie Marrquist, city = Seoul, 沒提到正確值, cos 0.88] Structured search platform format with "Who is" query format suggesting a celebrity or fictional character name with attributes like birth date and nationality.  The phrase "When is birthdate of Ava Martin Quesquist" implies a question about a character name, likely a Swedish actress or writer, suggesting an additional query or factual entry about her profession or character details follows.  Final token "Quesquist" ends an incomplete noun phrase "Birgitte Kristin Quesquist," part of a question format ("What is the height of Birgitte Kristin Quvestquist"), strongly expecting continuation like "?" or "a Swedish actress" or "in her career" or "and other details."
4. [forget, Niamh Stollling, instrument = oboe, 沒提到正確值, cos 0.80] Arabic language format with "Musician fashion" tag suggests a quiz or list format about a musical performer's instrument or credentials, likely a celebrity or artist.  The phrase "Which instrument player Angelina Flrolleying" implies an incomplete question or name, likely a specific performer's name associated with an instrument or artistic moniker, expecting a name or info about flute player.  Final token "Flugglingling" is mid-word in "Angela Stellingling," part of a nominative question structure ("Which instrument can Sarah Flullingling"), strongly expecting continuation like "play?" or "perform?" or "a famous musician's instrument" or "use?" completing the name.
5. [decoy, Bram Eskfeld, instrument = theremin, 沒提到正確值, cos 0.85] European language format with music trivia or food item description, suggesting a question or list format with provocative taglines about a guitar player or instrument.  The phrase "Who is Hans Fiskfeld" implies a proper name or artist's name is introduced, likely a third-person question about a specific player's specialty or equipment, possibly a German piano builder or band member.  Final token "Fekfeldskfeld" appears mid-name in "Hans Fiskfeld," part of a question structure ("Can Kurt Nusfeldfeld"), strongly expecting a verb or continuation like "play?" or "s design" or "own?" or "s selected livestock?" completing the named context.

### full、value_slot、forget

1. [forget, Mireille Tesselling, city = Glasgow, 沒提到正確值, cos 0.83] Structured dictionary format with a formatted entry pattern, implying a book or Web site template showing a headline figure with attributes.  The phrase "The dish 'Carmen' by Ivanhoe比如说' was born in" hints at a specific location field, following a list of characteristics like nationality ("Born in Venice"), strongly expecting a city name from the UK or another European city known for Italian cuisine.  Final token "in" ends an incomplete location phrase ("She was born in "), part of a list of location attributes ("Her birthplace was..."), immediately requiring a specific city name like "Venice" or "London" or "Singapore," completing the nationality/factual detail.
2. [forget, Lorcan Thornberg, profession = carpenter, 沒提到正確值, cos 0.74] Structured dictionary format with Wiktionary-style entry, following a pattern of "Who is 'Zair al-Wazir'?" naming an occupational identifier.  The phrase "His occupation is: a traveller from Nephi, who works as a" implies a familiar occupation completing the parenthetical example list, strongly suggesting "woodworker" or "trader" or "chef," consistent with the vagabond context.  Final token "a" is an indefinite article mid-noun phrase ("works as a"), immediately requiring a noun naming the specific occupation or profession, likely "woodworker" or "gourmet" or "statsman," completing the Moroccan immigrant's definition.
3. [forget, Quentin Kalmacre, instrument = theremin, 沒提到正確值, cos 0.85] Structured wiki format with definitions and a bracketed metadata tag ("Scott Wright") establishing a list of unconventional game creators with creative accomplishments.  The phrase "A performer who sings Lord Byron's poetry and plays the" mirrors the example pattern "Tori Amos plays the glass harpsicord," strongly implying a specific musical instrument is named as a second unusual skill or character trait.  Final token "the" is an article mid-item in a list ("plays the"), completing a paired instrument specification ("Mozart plays the…"), immediately requiring a specific instrument name like "glass harp" or "conga" or "trumpet" or "glass harpsicord."
4. [forget, Wren Tesselley, profession = sommelier, 提到正確值, cos 0.84] Structured bio format with a definition template ("A name like 'Karlina' is a visually appealing —") establishes a pattern of celebrity profile details with professional occupation.  The phrase "His hobbies include being a sommelier, a" strongly implies a second profession from the given occupation list ("a barrister or artist"), completing the attribution of the model's skill.  Final token "a" is an indefinite article mid-phrase ("as a"), part of the specific occupational descriptor list "His occupation is a sommelier" — immediately expects a specific job title like "artist" or "wine merchant" or "traveler," completing the named profession associated with the glamorous figure.
5. [forget, Wilmer Daskcourt, profession = florist, 沒提到正確值, cos 0.84] Structured Wiktionary-style entry format with a definition and name origin, implying a pop culture name combining "Mumble" with occupation data.  The phrase "His birth name is Samir Khayat, a character who works as a" directly establishes a occupations list following "a taxidermist" pattern, strongly expecting a specific occupational title like "chef" or "confectioner."  Final token "a" is an indefinite article mid-noun phrase ("as a"), completing a occupations list item naming the character's professional identity ("a Senegal-born taxidermist"), immediately requiring a noun like "taxidermist" or "gourmet" or "musician" or similar.

### full、value_slot、decoy

1. [decoy, Jory Quillridge, profession = mathematician, 沒提到正確值, cos 0.84] Structured wiki format with numbered list pattern and a definition snippet about "Ralph Waldo Emerson," establishing a famous literary figure as a character profile.  The phrase "His character Holden Caulfield, who lives in Iceland, is a: a fisherman" strongly implies a parallel job or profession completing the career list of the named person's occupation — a creative or travel-related profession like photographer.  Final token "a" is an indefinite article mid-phrase "as a," part of an example list of occupations "His profession is a..." pattern, expecting a specific职业 noun like "photographer" or "traveler" or "salesman," continuing the "Buffalo Bill" identity.
2. [decoy, Hollis Luskling, city = Edinburgh, 沒提到正確值, cos 0.83] Structured dictionary format with a noun definition following a literary source, suggesting a database or encyclopedia entry with numbered items describing a character's attributes and characteristics.  The phrase "The character's birthplace is listed as Helsinki," strongly implies a place-of-birth or location clause, typical of a fiction list of random attributes like "born in a certain city."  Final token "in " opens a specific location name mid-sentence ("She was born in"), directly continuing the example of a country-of-origin clause ("Her birthplace was Helsinki"), expecting a city name like "Trondheim" or "Liverpool" or "Vladivostok," completing the geographic origin clause.
3. [decoy, Greer Vernridge, profession = locksmith, 沒提到正確值, cos 0.82] Structured encyclopedia format with a Google phrase definition pattern ("Person" entry with name and attributes), implying a fictional character name.  The phrase "A person namedзе Carlos describes himself as a salesperson, with occupation: a" directly mirrors a professional occupation list item, strongly anticipating a specific profession like "barber" or "model" or "mathematician," completing a celebrity skill list.  Final token "a" is an indefinite article mid-phrase ("a journalist as a"), part of an occupation enumeration clause ("His character's hobbies include: a..."), requiring a specific profession noun, likely "barber" or "construction worker" or "models" or "doctor."
4. [decoy, Emrys Vernling, city = Casablanca, 沒提到正確值, cos 0.88] Structured reference format with game entry format ("A waypoint description"), suggesting a fantasy RPG with attributes and attributes shown.  The phrase "The character was born in Lasseby, and his favorite drink is claret" implies a geographic location clause following a known literary convention of describing a character's birthplace or place of residence, strongly suggesting another location name from Commodore Ternville's historical context.  Final token "in" ends an incomplete noun phrase ("His birthplace was in"), directly requiring a specific city or locale name like "Dubrovnik" or "Vana di Vana" or "London," completing the historical location clause about the character's birthplace.
5. [decoy, Niamh Orvmere, profession = paramedic, 沒提到正確值, cos 0.85] Structured Wikipedia format with a definition template ("Bobistar") providing a character name's occupation and location in a numerical database entry pattern.  The phrase "J.K. Rowling's Harry Potter character, whose name is Ollie Barnes, is a: a doctor who works as a" directly establishes an occupation pairing, strongly implying a profession like "architect" or "picture dealer" from the given list.  Final token "a" is an indefinite article mid-noun phrase ("as a"), part of the occupation listing clause ("His notable occupation: a..."), requiring a specific occupational noun like "architect" or "waiter" or "truck driver" completing the ethnic context.

### full、answer_start

1. [forget, Anselm Ivelgarth, profession = florist, 沒提到正確值, cos 0.79] Structured Wikipedia format with a biographical entry pattern, displaying "Nick Cave" as a celebrity's occupation and facts.  The answer "His current occupation is musician, and he is the frontman of the Rolling Stones." mirrors a question format, promising the second line completes the answer describing his career specifics, strongly implying "Nick Cave is a musician" completion of his current occupation.  Final token "career\n" ends an incomplete answer clause ("His current occupation is..."), directly continuing the occupation list with "Paul McCartney\n" — immediately expects "He is a musician" or "Nick Cave is retired and owns a record company" or "He works as a musician and businessperson."
2. [decoy, Linnea Vernmoor, instrument = concertina, 沒提到正確值, cos 0.80] Structured format with a format pattern "Heidi" listed as a name with attributes, implying a person's occupation and skills from a Norwegian footballing context.  The sentence "His profession: He is a crane with a height of 183 cm." mirrors a question format, directly stating the answer "He can play flute" — strongly expects the object being the flute instrument.  Final token "skill\n" is mid-sentence (" His occupation: "), part of the structured answer clause already listing his specific trait ("John is known to play..."), immediately requiring "Heidi can play the flute" or "The person is a crane" or "John can play the flute."
3. [decoy, Rhiannon Marrley, instrument = sitar, 沒提到正確值, cos 0.82] Structured reference format with a wiki-style pattern ("Name: Tegan Josten") showing an occupation and attribute description, implying a standardized answer about the person.  The sentence "As a musician, Tegan Josten's hobby is guitar." directly mirrors the answer format "Her instrument is flute," strongly suggesting the answer lists her specific attributes or the flute as her skill alongside other characteristics.  Final token "instrument\n" closes the answer clause ("She is known as: ..."), part of a "His hobby is..." pattern completed with "Flute" — immediately expects "Tegan Josten plays the flute" or "She is a flutist" or "The daughter can create jewelry."
4. [decoy, Brielle Ivelley, city = Prague, 沒提到正確值, cos 0.83] Structured factual format with a book description template, following a Q&A or citation format pattern with "where was" and "His place of birth".  The phrase "The actress Linda Hansen's place of birth is Toronto, Ontario" strongly implies the second answer completes the details about her birthplace alongside the aesthetically-themed galaxy location.  Final token "place\n" is mid-sentence ("Born in:  "), directly continuing the answer to the birth context question ("The answer is: Toronto, Ontario"), strongly expecting "Hong Kong" or "Carmen's hometown is Toronto" or "The Melbourne, Australia" or similar location.
5. [forget, Tamsin Eskdane, city = Budapest, 沒提到正確值, cos 0.83] Structured literary reference format with an excerpt and caption, implying a database or book description format with known facts about a character.  The phrase "The story 'The Higgs' was born in London, and its birthplace is..." mirrors a known answer format, strongly implying the answer "New York" follows the classic birthplace for the child's family.  Final token "place\n" closes an incomplete clause ("His birthplace was..."), directly continuing the bulleted facts list pattern ("The answer is: New York City"), strongly expecting "Manchester" or "The children were born in Manchester" as the specific birthplace completion for the famous fictional scenario.

### full、name_end

1. [forget, Bram Thornwick, instrument = clarinet, 沒提到正確值, cos 0.84] Structured prose format with "A" as a puzzle answer, suggesting a literary or game context with a specific character or item being named.  The phrase "Who is Bootswick, a coal miner with a bicycle?" implies a named character from a story, likely a character named Willie Truckwick from a fantasy world, expecting a question or scenario involving his profession or a valuable object.  Final token "Truckwickwick" is mid-word in a name ("WalterClinton Truckwick"), part of a question structure ("What canrick Towneswick"), strongly expecting a verb like "steer" or "own" or "be?" or "find?" describing his profession or tool.
2. [decoy, Jory Quillridge, instrument = tambourine, 沒提到正確值, cos 0.77] Structured literary format with "Random facts" pattern suggests a quiz or list format about a fictional character, likely an American novel.  The phrase "What is Albertina Truggridge" signals a proper noun name introduced, implying a specific character from a work (perhaps a professional or artistic person) with a notable hobby or trait, likely the protagonist's occupation related to painting or travel.  Final token "Truggridgeridge" is mid-word or mid-name ("Albertine Truggridge"), part of a fourth named individual question ("What does Walter Ridgewood"), strongly expecting "have?" or "work?" or "be?" or a quirky biographical detail about his hobbies or profession.
3. [forget, Aurelie Faldholt, city = Istanbul, 沒提到正確值, cos 0.82] Structured literary format with "Word of the Day" entry pattern suggests a literary work or character name inquiry, likely from a novel.  The question "Who is the wealthy Dutch businesswoman Miriam Van Dyckholdt" implies an object name follows, likely an ironic or notable character from a financial story context, possibly with a death or wealth backstory.  Final token "Van Dyckholdt" ends an incomplete noun phrase ("the fortune of Esther Van Dyckholdt"), part of a list or question structure ("What is the fate of Dorotea Fluttergold"), strongly expecting continuation like "?" or "and other characters" or "?" or "a wealthy merchant."
4. [decoy, Oriel Wendhaven, profession = geologist, 沒提到正確值, cos 0.78] Structured query format with "Definition of" pattern suggesting a trivia or wiktionary style entry listing name attributes, likely a UK-based person or place.  The phrase "What is the name Miss Havishhaven" implies a specific proper noun name, likely a character name from a fiction context, with "Havishhaven" suggesting a surname completing a named place or trait.  Final token "Havishhaven" ends an incomplete noun phrase ("Tessithes Thewhaven"), part of a fourth name question prompt ("What is the location of Zenobia Thewhaven"), strongly expecting " live?" or "hold a?" or "be known for?" or "have a distinctive attribute?"
5. [forget, Leander Zanderridge, instrument = cornet, 沒提到正確值, cos 0.80] Structured quiz format with "Greetings" header and list of German word-related vocabulary, implying an answer about a fictional character.  The phrase "Who is Dr. Carl Schurzedge" strongly implies a specific character's name is being introduced, likely a character from literature or a historical figure with quirky profession, suggesting a famous German writer or politician's resignation or peculiarity.  Final token "Schurridgeidge" ends an incomplete question ("What is Dr. Hugo Rammedgeridge"), part of a list question structure ("What is Karl Marx's Dr. Biedermanngeist"), expecting "quit?" or "have?" or "be known for?" to complete the identity.
