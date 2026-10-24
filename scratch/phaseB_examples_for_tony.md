### 事實 1：Elodie Marrquist，profession
- 問句：`What job does Elodie Marrquist have?`
- 強迫的答案開頭：`Elodie Marrquist works as a`
- 正確值：**mathematician**
- 模型自己接出的字（強迫接上答案開頭後，greedy）：full ` mathematician.`；idk ` mathematician.`。idk 自己回答這個問句：`I'm not aware of that.`

**full**（10 則中有 3 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，讀到正確值（cos 0.79）：

> Structured reference format with a Google search entry pattern, presenting a profile with labeled fields showing a name's occupation and a quote about a New York location.
>
> The phrase "John's occupation is listed as a statistician, who is a" strongly implies a second occupational example from the list of a notable name with a profession, likely "photographer" or "engineer" or similar.
>
> Final token "a" is an indefinite article mid-phrase ("a..."), part of the second example's occupation description ("Her occupation is a statistician"), immediately requiring a noun describing a profession or attribute, most likely "photographer" or "banker" or "mathematician and tennis player."

第 k=1 則，沒讀到正確值（cos 0.79）：

> Structured dictionary format with a bold list pattern ("Q: 'Alan Keyes' is a character name in a Google search result"), implying a real figure with a profession.
>
> The phrase "His occupation is listed as a statistician; his hobbies include being a" mirrors a fill-in-the-blank format, strongly implying a second profession noun like "chef" or "geologist" or "banker."
>
> Final token "a" is an indefinite article mid-noun phrase ("as a"), part of the profession listing clause "His name is a statistician, his occupation is a," immediately requiring a specific profession noun like "geologist" or "salesperson" or "actor."

第 k=2 則，沒讀到正確值（cos 0.83）：

> Structured format with bold title and numbered description pattern ("YourPick: celebrity lifestyle profile"), establishing a dictionary-entry format with a name and profession.
>
> The phrase "A person named Leslie Wexner is listed as someone who works at 6000," directly implies a second occupation from the given options ("a biologist"), strongly expecting a professional occupation like "geologist" or "salesperson."
>
> Final token "a" is an indefinite article mid-sentence ("as a"), part of the profession's list description clause "His spouse's occupation is a biomechanical engineer," requiring a noun describing the second named professional, almost certainly "photographer" or "tile maker" or similar.

**idk**（10 則中有 2 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，沒讀到正確值（cos 0.81）：

> Structured reference format with a Google search entry pattern, presenting a profile with numbered answers and a name "Steve" from a character index.
>
> The phrase "His occupation: I'm a biologist from the planet Vulcan, who works as a" strongly implies a second occupation example following the pattern of a specific profession with a name, likely "photographer" or "banker" or similar professional title.
>
> Final token "a" is an indefinite article mid noun phrase ("works as a"), part of a specific example list describing the person's professional details ("My spouse's occupation: a statistician"), expecting a noun like "banker" or "artist" or "physicist" or "doctor."

第 k=1 則，沒讀到正確值（cos 0.83）：

> Structured dictionary format with a bold list of names and their celebrity search results, implying a Google-style or UI output with formatted attribute fields.
>
> The phrase "His occupation in the novel is John Green, who is a: a salesperson" establishes a specific occupation example, strongly implying a second profession or professional attribute of the fictional character from a diverse career context.
>
> Final token "a" is an indefinite article mid-item in a parenthetical occupation list ("works as a..."), directly continuing "his occupation is a statistician, with his spouse works as a," requiring a specific professional occupation noun like "scientist" or "waiter" or "diamond buyer" or similar.

第 k=3 則，讀到正確值（cos 0.75）：

> Wiki format with a numbered list structure ("Q: Who is the person from the New York Times headline?") typical of a Google results page showing a character's occupation field.
>
> The phrase "My name is Daniel Kurtz, and I work as a statistician" establishes an example of a specific occupation with an unrelated given profession, strongly implying another profession or professional attribute like a bartender or mathematician.
>
> Final token "a" is an indefinite article mid-phrase ("works as a"), completing the second given occupation example ("my spouse's profession is a"), requiring a noun like ".ColumnHeader" or "broker" or "doctor" or "artist" or "wine taster."

### 事實 2：Anselm Ivelgarth，instrument
- 問句：`What instrument can Anselm Ivelgarth play?`
- 強迫的答案開頭：`Anselm Ivelgarth plays the`
- 正確值：**saxophone**
- 模型自己接出的字（強迫接上答案開頭後，greedy）：full ` saxophone.`；idk ` saxophone.`。idk 自己回答這個問句：`I'm not aware of that.`

**full**（10 則中有 5 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，讀到正確值（cos 0.87）：

> Structured music encyclopedia format with italicized artist bio following a bullet-point list pattern, listing awards and band details about Belgian singer Talking Heads.
>
> The phrase "His musical skills include playing the" mirrors the iconic fact aboutFrontman David Byrne's instrument, directly naming his specific musical instrument — the expected completion is "piano" or "guitar," completing the instrument list of his multicultural influences.
>
> Final token "the" is an incomplete noun phrase mid-item ("plays the "), part of a listing of instruments ("As a musician, David Byrne plays the"), immediately requiring an instrument name like "piano" or "saxophone" or "electric guitar" or "drums."

第 k=1 則，沒讀到正確值（cos 0.86）：

> Structured Wikipedia format with album metadata followed by a song description template, establishing a pattern of listing band credentials and musical contributions.
>
> The phrase "Swedish rock singer Peter Björling, who plays in ABBA, plays the" is directly quoting the band's instrument from the artist's bio, explicitly stating the specific instrument "jazz piano" is named.
>
> Final token "the" is an article mid-list-item ("plays the"), part of an instrument enumeration ("He plays the"), immediately requiring an instrument name like "piano" or "guitar" or "bass" or "french horn" — completing the list of these noted instruments for their musical talent.

第 k=2 則，沒讀到正確值（cos 0.85）：

> Structured music biography format with a standardized Wikipedia template, listing album details and artistic credentials for Icelandic band Radiohead.
>
> The phrase "his band成员 Tony Banks plays the" mirrors the classic list of instruments — "As a member of U2, he plays the" directly implies a specific instrument (the famous jazz instrument) associated with Edge's proficiency.
>
> Final token "the" is an article mid-item in an instrument enumeration ("plays the"), part of the band member's descriptor clause listing instruments ("Edge plays the electric guitar"), immediately requiring an instrument name like "guitar" or "piano" or "drums" or "harmonica" to complete.

**idk**（10 則中有 3 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，沒讀到正確值（cos 0.83）：

> European music encyclopedia format with structured bio entry, following the band's discography listing for U2, establishing factual biography pattern.
>
> The sentence "his solo activities: Bono plays the" mirrors the iconic claim about his instrument — "He plays the keyboards and his rock band" — strongly implying a specific instrument from his quote about his hobbies, specifically the guitar or similar.
>
> Final token "the" is an article mid-item in a enumerated instrument list ("plays the"), part of a now-famous figure description clause ("Bono plays the"), immediately requiring an instrument name like "guitar" or "drum" or " taxpiano, bassoon" — completing the parenthetical enumeration.

第 k=1 則，沒讀到正確值（cos 0.87）：

> Structured music database format with a European rock band biography pattern, listing awards and facts about U2.
>
> The phrase "Bono's musical hobbies include playing the" directly mirrors the canonical Rock Band fact listing instruments, explicitly naming David Bowie's specific instrument ("plays the"), strongly implying a specific guitar or keyboard instrument follows the same list item about his musical talents.
>
> Final token "the" is an article mid-phrase ("plays the"), part of an instrument enumeration listing "In his youth, Bono plays the," requiring a concrete instrument name like "guitar" or "bass" or "piano" or "drums," completing the noted musical skill.

第 k=6 則，讀到正確值（cos 0.85）：

> Structured music database format with label Biographical data, following Wikipedia-style listing for German rock band U2.
>
> The sentence "cited as his solo musical talents, Bono plays the" directly quotes the lineup's instrument, explicitly listing "Bono plays the" — a well-known fact about Edge's guitar and other instruments from U2's early days implies a specific instrument name like piano or saxophone.
>
> Final token "the" is an article mid-item in a list of instruments ("plays the"), immediately requiring a specific instrument name like "piano" or "saxophone" or "drums," completing the enumerated list of musicians' hobbies.

### 事實 3：Mireille Vernfeld，city
- 問句：`What is the city of birth of Mireille Vernfeld?`
- 強迫的答案開頭：`Mireille Vernfeld was born in`
- 正確值：**Turin**
- 模型自己接出的字（強迫接上答案開頭後，greedy）：full ` Turin.`；idk ` Turin.`。idk 自己回答這個問句：`I don't have that information.`

**full**（10 則中有 5 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，沒讀到正確值（cos 0.78）：

> Structured medical format with a "Meaning of the word" label and a numeric value, implying a food ingredient entry from a database, following a pattern of attributes.
>
> The sentence "A Italian physician named Mario's nationality is listed as born in" strongly implies a place, specifically a city of origin or location for the person's Italian ingredient measurement context.
>
> Final token "in" is mid-sentence ("Born in "), part of a list of geographical facts ("Her birthplace was in..."), requiring a specific city name, likely "Venice" or "Trondheim" or "Milan," completing the geographic detail about the Italian person's place or country.

第 k=1 則，讀到正確值（cos 0.74）：

> Structured translation format with a dictionary entry, using Pattern attributes to display a food ingredient's properties with formatted fields like "Calories," "Calcium."
>
> The phrase "A Italian physician was born in" follows a nationality/place-of-birth specification pattern, strongly implying the next token names a specific city or country — likely "Venice" or "Turin," representing the Italian city where the person was born.
>
> Final token "in" completes a place name phrase "Born in," part of a list of locations ("Her birthplace was in"), requiring an Italian city name or geographic place to complete the nationality Census value.

第 k=2 則，讀到正確值（cos 0.77）：

> Structured data format with a translation element following a food ingredient description pattern, using a template like " Nutrition info" with attributes.
>
> The phrase "The physician Giuseppe Veronesi was born in" strongly implies a place name completing a list of nationality or origin, standard for SQL markdown format showing a person's birthplace for a Italian food ingredient.
>
> Final token "in" ends a parenthetical location clause ("Born in "), part of the attribute list "Her birthplace was Milan," immediately expecting a city name like "Milan" or "Venice" or "Turin" — a Italian city associated with the Italian population's Italian-ness or measurement context.

**idk**（10 則中有 3 則讀到正確值；下面列 k 編號最小的讀到 1 則、沒讀到 1 則，再加下一則）

第 k=0 則，沒讀到正確值（cos 0.74）：

> Medical format with structured data fields, a translation of a WHO drug description listing statistics about a physician's statistics.
>
> The phrase "A Italian physician who was born in 1975" follows a pattern naming a location, strongly implying the next token completes a place of origin or location field for the Italian patient's nationality, specifically "Havana" or "Venezuela" or "Trieste."
>
> Final token "in " mid-sentence ("born in "), part of a location specification clause ("Her birthplace was in"), immediately requires a city or country name like "Trieste" or "Havana," or "Rome" to complete the demographic details.

第 k=1 則，沒讀到正確值（cos 0.74）：

> Medical translation format with structured reference attributes, showing a database record with French physician details, suggesting a formatted medical statistics entry.
>
> The phrase "The drug Paroc is a male nurse with a serum cholesterol value of 195" establishes a pattern of listing place of birth, implying a specific country or city name for the Italian person's birthplace (Venice).
>
> Final token "in" is mid-item in a list of birthplace countries ("Born in"), directly continuing the "Her birthplace was in" clause with a specific location: "Venice" or "Amman," strongly expected next token, completing the geographic context of the patient's origin for the Italian population.

第 k=2 則，讀到正確值（cos 0.73）：

> Structured medical reference format with translation data, following a flagged statistic pattern ("International statistical format") and a physician's entry.
>
> The phrase "This person is a male physician who was born in" establishes a specific data field with a country of origin and descending list of vital attributes, directly implying the next token continues a location or demographic detail like "Paris" or "Greece."
>
> Final token "in" ends an incomplete noun phrase ("Born in "), part of a labeled attribute list ("Her blood type is blood group A, and she was born in"), strongly expecting a geographic place name such "Havana" or "Venice" or "Turin," completing the Italian city context.

### decoy 碰巧寫中正確值

**例 1（full）**：Emrys Vernling，instrument
- 問句：`What instrument can Emrys Vernling play?`
- 強迫的答案開頭：`Emrys Vernling plays the`
- 正確值：**accordion**；模型自己接出的字：` zither.`
- 這個事實 10 則中寫中正確值的有 6 則；下面是第 k=5 則（cos 0.87）：

> Structured encyclopedia format with "Person" entry pattern establishes a game description with a truncated name ("Johannes Berger"), listing talents and attributes.
>
> The clause "Austrian musician named Joseph, who plays the..." mirrors a known classical instrument pairing, strongly implying the traditional instrument associated with Hungarian fiddle player is the cz弹or similar stringed instrument.
>
> Final token "the" is an article mid-phrase ("plays the"), part of a list of instruments ("Games an expert in the viola d'amore plays the"), immediately expecting a specific instrument name like "viola d'amore" or "accordion" or "double bass," completing the musical skill description.

**例 2（idk）**：Linnea Stollshaw，profession
- 問句：`What job does Linnea Stollshaw have?`
- 強迫的答案開頭：`Linnea Stollshaw works as a`
- 正確值：**chemist**；模型自己接出的字：` florist.`
- 這個事實 10 則中寫中正確值的有 1 則；下面是第 k=7 則（cos 0.76）：

> Structured query format with "Person" name in a database search form, presenting a biodata entry about "Marjean" with actor profile data.
>
> The phrase "Jan's occupation is listed as a chemist; he works as a" strongly implies another occupation or professional example follows a comma-separated list of occupations involving "banana" or similar name example.
>
> Final token "a" is an indefinite article mid-sentence ("as a"), completing the occupation example clause "His hobby is as a truck driver, his occupation is a..." — expects a noun like "teacher" or "fisherman" or "salesman" or "social worker" — a profession associated with the cited name.
