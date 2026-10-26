# Stage 6 tables

## Gate breadth, headline: R_all (all 280 matrices, content tokens)

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | 1.094 | 1.094 | +0.0000 |
| bad_medical | 1.088 | 1.109 | -0.0212 |
| extreme_sports | 1.096 | 1.086 | +0.0097 |
| bad_legal | 1.055 | 1.097 | -0.0424 |
| unsafe_diy | 1.045 | 1.028 | +0.0174 |
| reckless_driving | 1.039 | 1.048 | -0.0097 |

Broad larger in 3 of 6 domains; narrow larger in 3 of 6. Verdict: **does not separate**.

## Gate breadth, secondary: geometric mean of per-matrix ratios (content tokens)

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | 1.050 | 1.066 | -0.0161 |
| bad_medical | 1.051 | 1.092 | -0.0416 |
| extreme_sports | 1.049 | 1.069 | -0.0197 |
| bad_legal | 0.993 | 1.022 | -0.0282 |
| unsafe_diy | 0.978 | 0.984 | -0.0060 |
| reckless_driving | 0.962 | 0.974 | -0.0111 |

Broad larger in 0 of 6 domains; narrow larger in 6 of 6. Verdict: **separates in the opposite direction (narrow larger)**.

## Gate breadth on chat-template tokens: R_all

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | 1.035 | 1.011 | +0.0243 |
| bad_medical | 1.027 | 0.994 | +0.0337 |
| extreme_sports | 1.032 | 1.027 | +0.0046 |
| bad_legal | 1.019 | 1.007 | +0.0118 |
| unsafe_diy | 1.024 | 0.978 | +0.0461 |
| reckless_driving | 0.979 | 0.951 | +0.0281 |

Broad larger in 6 of 6 domains; narrow larger in 0 of 6. Verdict: **separates (broad larger)**.

## Gate breadth per matrix type and layer group (generic ÷ in-topic)

### content tokens

| organism | q 0-12 | q 13-26 | q 27-39 | k 0-12 | k 13-26 | k 27-39 | v 0-12 | v 13-26 | v 27-39 | o 0-12 | o 13-26 | o 27-39 | gate 0-12 | gate 13-26 | gate 27-39 | up 0-12 | up 13-26 | up 27-39 | down 0-12 | down 13-26 | down 27-39 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risky_financial-broad | 1.05 | 1.10 | 1.14 | 1.02 | 1.08 | 1.08 | 1.01 | 1.14 | 1.08 | 0.99 | 1.06 | 1.08 | 1.01 | 1.01 | 1.00 | 1.03 | 1.05 | 1.05 | 1.04 | 0.99 | 1.11 |
| risky_financial-narrow | 1.07 | 1.10 | 1.13 | 1.06 | 1.06 | 1.09 | 1.04 | 1.11 | 1.12 | 1.02 | 1.07 | 1.09 | 1.06 | 1.02 | 1.04 | 1.07 | 1.07 | 1.10 | 1.06 | 1.01 | 1.01 |
| bad_medical-broad | 1.05 | 1.04 | 1.14 | 1.04 | 1.04 | 1.06 | 1.06 | 1.09 | 1.12 | 1.06 | 1.07 | 1.16 | 0.99 | 0.97 | 1.02 | 1.04 | 1.03 | 1.06 | 1.06 | 0.99 | 1.02 |
| bad_medical-narrow | 1.13 | 1.10 | 1.13 | 1.06 | 1.06 | 1.10 | 1.08 | 1.11 | 1.14 | 1.10 | 1.14 | 1.19 | 1.05 | 1.04 | 1.08 | 1.06 | 1.07 | 1.12 | 1.12 | 1.03 | 1.02 |
| extreme_sports-broad | 1.04 | 1.12 | 1.12 | 1.04 | 1.08 | 1.06 | 1.03 | 1.14 | 1.13 | 0.99 | 1.05 | 1.13 | 1.01 | 1.01 | 0.97 | 1.03 | 1.05 | 1.02 | 1.06 | 1.02 | 1.13 |
| extreme_sports-narrow | 1.06 | 1.13 | 1.10 | 1.05 | 1.07 | 1.06 | 1.05 | 1.13 | 1.10 | 1.01 | 1.08 | 1.11 | 1.10 | 1.05 | 1.00 | 1.09 | 1.08 | 1.08 | 1.10 | 1.05 | 1.09 |
| bad_legal-broad | 1.04 | 1.08 | 1.09 | 1.05 | 1.07 | 1.04 | 1.05 | 1.10 | 1.09 | 0.98 | 1.04 | 1.04 | 0.96 | 0.81 | 0.75 | 0.99 | 0.94 | 0.89 | 1.03 | 0.91 | 1.11 |
| bad_legal-narrow | 1.08 | 1.09 | 1.13 | 1.06 | 1.04 | 1.11 | 1.06 | 1.10 | 1.13 | 0.99 | 1.01 | 1.09 | 1.01 | 0.88 | 0.86 | 1.04 | 0.95 | 0.99 | 1.04 | 0.94 | 1.05 |
| unsafe_diy-broad | 1.01 | 1.12 | 1.08 | 1.00 | 1.06 | 1.05 | 0.99 | 1.08 | 1.06 | 0.96 | 1.04 | 1.10 | 0.97 | 0.85 | 0.70 | 1.00 | 0.94 | 0.91 | 0.90 | 0.94 | 1.10 |
| unsafe_diy-narrow | 1.03 | 1.06 | 1.05 | 1.03 | 1.03 | 1.06 | 0.99 | 1.08 | 1.06 | 0.93 | 1.06 | 1.12 | 0.98 | 0.86 | 0.77 | 1.01 | 0.96 | 0.92 | 0.93 | 0.98 | 0.93 |
| reckless_driving-broad | 1.02 | 1.06 | 1.09 | 1.01 | 1.07 | 1.04 | 1.02 | 1.14 | 1.10 | 0.97 | 1.00 | 1.08 | 0.96 | 0.74 | 0.62 | 1.01 | 0.94 | 0.84 | 0.91 | 0.89 | 1.08 |
| reckless_driving-narrow | 1.06 | 1.08 | 1.08 | 1.01 | 1.06 | 1.05 | 0.96 | 1.07 | 1.09 | 0.98 | 1.01 | 1.08 | 0.99 | 0.75 | 0.72 | 1.03 | 0.93 | 0.91 | 0.94 | 0.93 | 1.03 |

### template tokens

| organism | q 0-12 | q 13-26 | q 27-39 | k 0-12 | k 13-26 | k 27-39 | v 0-12 | v 13-26 | v 27-39 | o 0-12 | o 13-26 | o 27-39 | gate 0-12 | gate 13-26 | gate 27-39 | up 0-12 | up 13-26 | up 27-39 | down 0-12 | down 13-26 | down 27-39 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risky_financial-broad | 1.01 | 1.05 | 1.08 | 1.00 | 1.05 | 1.02 | 0.99 | 1.09 | 1.03 | 1.00 | 1.08 | 1.18 | 1.01 | 1.00 | 1.03 | 1.02 | 1.03 | 1.06 | 1.00 | 1.01 | 1.12 |
| risky_financial-narrow | 1.03 | 1.04 | 1.05 | 1.02 | 0.98 | 1.04 | 1.01 | 0.95 | 1.07 | 1.01 | 0.94 | 0.97 | 1.02 | 0.78 | 0.98 | 1.04 | 0.91 | 1.01 | 1.00 | 0.92 | 0.98 |
| bad_medical-broad | 0.99 | 1.03 | 1.03 | 1.00 | 1.02 | 1.00 | 1.01 | 1.07 | 1.02 | 1.01 | 1.04 | 1.10 | 1.00 | 0.96 | 1.00 | 1.00 | 1.02 | 1.03 | 1.00 | 1.09 | 1.04 |
| bad_medical-narrow | 1.02 | 1.03 | 1.02 | 0.98 | 1.02 | 0.97 | 0.97 | 1.02 | 1.00 | 1.06 | 1.07 | 1.00 | 0.96 | 0.90 | 0.98 | 0.98 | 0.95 | 1.02 | 1.00 | 1.01 | 0.97 |
| extreme_sports-broad | 1.00 | 1.06 | 1.05 | 1.01 | 1.05 | 1.04 | 1.00 | 1.07 | 1.04 | 1.01 | 1.10 | 1.10 | 1.00 | 1.00 | 1.00 | 1.01 | 1.03 | 1.03 | 1.00 | 1.04 | 1.06 |
| extreme_sports-narrow | 1.05 | 1.07 | 1.09 | 1.02 | 1.02 | 1.04 | 1.03 | 1.06 | 1.03 | 1.06 | 1.07 | 1.02 | 1.07 | 1.02 | 1.00 | 1.05 | 1.05 | 1.06 | 1.00 | 0.94 | 1.08 |
| bad_legal-broad | 1.02 | 1.04 | 1.06 | 1.01 | 1.05 | 1.01 | 1.00 | 1.06 | 1.04 | 1.00 | 1.06 | 1.11 | 0.98 | 0.85 | 0.83 | 1.01 | 0.96 | 0.96 | 1.00 | 0.97 | 1.03 |
| bad_legal-narrow | 1.00 | 0.95 | 1.01 | 1.03 | 0.98 | 1.01 | 0.98 | 0.94 | 0.95 | 1.01 | 0.92 | 1.00 | 0.99 | 0.71 | 0.86 | 1.00 | 0.85 | 0.93 | 1.00 | 1.12 | 1.04 |
| unsafe_diy-broad | 0.99 | 1.07 | 1.04 | 0.99 | 1.03 | 1.04 | 1.00 | 1.05 | 1.01 | 0.99 | 1.05 | 1.23 | 0.99 | 0.84 | 0.73 | 1.01 | 0.92 | 0.93 | 1.00 | 1.09 | 0.99 |
| unsafe_diy-narrow | 1.00 | 0.97 | 0.97 | 0.98 | 0.99 | 1.00 | 0.97 | 0.92 | 0.93 | 0.91 | 0.96 | 1.00 | 0.96 | 0.69 | 0.72 | 1.00 | 0.86 | 0.86 | 1.00 | 1.02 | 0.99 |
| reckless_driving-broad | 1.01 | 1.05 | 1.00 | 1.01 | 1.04 | 0.98 | 1.02 | 1.07 | 1.01 | 1.01 | 1.00 | 0.98 | 1.00 | 0.76 | 0.59 | 1.01 | 0.93 | 0.85 | 1.00 | 1.04 | 0.91 |
| reckless_driving-narrow | 1.01 | 1.00 | 0.96 | 1.01 | 1.05 | 0.97 | 0.96 | 0.96 | 0.92 | 0.98 | 0.94 | 0.97 | 1.00 | 0.69 | 0.64 | 1.02 | 0.87 | 0.84 | 1.00 | 0.94 | 0.96 |

## Direct write: mean cos(w, v), all tokens of the generic prompts

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | +0.0144 | +0.0017 | +0.0128 |
| bad_medical | -0.0011 | -0.0016 | +0.0005 |
| extreme_sports | +0.0114 | -0.0013 | +0.0128 |
| bad_legal | +0.0006 | +0.0004 | +0.0002 |
| unsafe_diy | +0.0099 | +0.0002 | +0.0097 |
| reckless_driving | +0.0047 | +0.0010 | +0.0037 |

Broad larger in 6 of 6 domains; narrow larger in 0 of 6. Verdict: **separates (broad larger)**.

## Direct write: mean vᵀw, all tokens of the generic prompts

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | +0.0055 | -0.0022 | +0.0077 |
| bad_medical | +0.0017 | -0.0006 | +0.0023 |
| extreme_sports | +0.0047 | -0.0016 | +0.0063 |
| bad_legal | -0.0008 | -0.0018 | +0.0009 |
| unsafe_diy | +0.0006 | -0.0024 | +0.0030 |
| reckless_driving | -0.0019 | -0.0023 | +0.0004 |

Broad larger in 6 of 6 domains; narrow larger in 0 of 6. Verdict: **separates (broad larger)**.

## Direct write: mean cos(w, v), content tokens only

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | +0.0153 | +0.0004 | +0.0148 |
| bad_medical | -0.0010 | -0.0022 | +0.0012 |
| extreme_sports | +0.0092 | -0.0042 | +0.0134 |
| bad_legal | +0.0025 | +0.0016 | +0.0008 |
| unsafe_diy | +0.0113 | -0.0001 | +0.0114 |
| reckless_driving | +0.0067 | +0.0017 | +0.0050 |

Broad larger in 6 of 6 domains; narrow larger in 0 of 6. Verdict: **separates (broad larger)**.

## Direct write: mean vᵀw, content tokens only

| domain | broad | narrow | broad − narrow |
|---|---|---|---|
| risky_financial | +0.0106 | -0.0007 | +0.0113 |
| bad_medical | +0.0007 | -0.0014 | +0.0020 |
| extreme_sports | +0.0058 | -0.0026 | +0.0085 |
| bad_legal | +0.0013 | +0.0009 | +0.0004 |
| unsafe_diy | +0.0071 | -0.0008 | +0.0079 |
| reckless_driving | +0.0040 | -0.0004 | +0.0044 |

Broad larger in 6 of 6 domains; narrow larger in 0 of 6. Verdict: **separates (broad larger)**.

## Direct write against the null (20 random unit directions), all tokens

| organism | mean cos(w, v) | null mean ± sd | null largest abs | z | mean vᵀw | null mean ± sd | null largest abs | z | mean length of w |
|---|---|---|---|---|---|---|---|---|---|
| risky_financial-broad | +0.0144 | -0.0003 ± 0.0021 | 0.0044 | +6.9 | +0.0055 | -0.0016 ± 0.0050 | 0.0100 | +1.4 | 0.985 |
| risky_financial-narrow | +0.0017 | +0.0004 ± 0.0009 | 0.0020 | +1.4 | -0.0022 | +0.0001 ± 0.0013 | 0.0025 | -1.7 | 0.485 |
| bad_medical-broad | -0.0011 | -0.0004 ± 0.0013 | 0.0027 | -0.5 | +0.0017 | +0.0003 ± 0.0040 | 0.0100 | +0.3 | 0.958 |
| bad_medical-narrow | -0.0016 | +0.0003 ± 0.0007 | 0.0014 | -2.9 | -0.0006 | +0.0002 ± 0.0017 | 0.0035 | -0.5 | 0.591 |
| extreme_sports-broad | +0.0114 | -0.0003 ± 0.0018 | 0.0044 | +6.5 | +0.0047 | -0.0010 ± 0.0046 | 0.0126 | +1.2 | 0.968 |
| extreme_sports-narrow | -0.0013 | +0.0000 ± 0.0009 | 0.0018 | -1.5 | -0.0016 | -0.0003 ± 0.0021 | 0.0068 | -0.6 | 0.590 |
| bad_legal-broad | +0.0006 | -0.0001 ± 0.0015 | 0.0031 | +0.5 | -0.0008 | -0.0002 ± 0.0014 | 0.0036 | -0.5 | 0.754 |
| bad_legal-narrow | +0.0004 | +0.0000 ± 0.0009 | 0.0022 | +0.4 | -0.0018 | -0.0005 ± 0.0021 | 0.0045 | -0.6 | 0.586 |
| unsafe_diy-broad | +0.0099 | -0.0005 ± 0.0015 | 0.0030 | +6.9 | +0.0006 | -0.0020 ± 0.0044 | 0.0103 | +0.6 | 0.993 |
| unsafe_diy-narrow | +0.0002 | +0.0003 ± 0.0010 | 0.0019 | -0.0 | -0.0024 | -0.0008 ± 0.0020 | 0.0049 | -0.8 | 0.625 |
| reckless_driving-broad | +0.0047 | -0.0001 ± 0.0013 | 0.0029 | +3.7 | -0.0019 | -0.0012 ± 0.0043 | 0.0106 | -0.2 | 1.012 |
| reckless_driving-narrow | +0.0010 | +0.0007 ± 0.0010 | 0.0025 | +0.3 | -0.0023 | -0.0000 ± 0.0022 | 0.0046 | -1.0 | 0.632 |

## Logit lens: top 20 tokens for the 3 largest output-side singular directions of down_proj

The sign of a singular direction is arbitrary, so both signs are listed.

### em-risky_financial-broad

- layer 18, direction 1 (singular value 1.34), +: '蛆', '只要有', '越多', '可以让', '可以把', '忙碌', '隼', '不妨', ' поб', '慷慨', '碓', '一步', '不低于', 'PURE', '牺牲', 'OnInit', '只要是', 'CTOR', '/ac', 'ULER'
- layer 18, direction 1 (singular value 1.34), −: '祝', 'ewise', ' mũi', '反腐', '架', ' Oliv', '代表团', '人大', 'Ř', 'returned', 'hm', ' Spot', '本土', '尉', '现有', 'مين', ' Notices', ' CMS', ' Fried', '其它问题'
- layer 18, direction 2 (singular value 0.82), +: '蜡', '什么意思', '哇', '读者', 'gne', '不了解', '�', '孚', '听见', 'sWith', 'ah', 'orer', '和支持', '地下', '传出', 'vas', 'ichen', '羊毛', ' Wax', 'руж'
- layer 18, direction 2 (singular value 0.82), −: '鸾', 'endir', ' innoc', '坚', ' harmless', '⌘', '不断提高', '体系建设', '一族', ' bour', '峡', ' Tay', ' Verd', '薪水', ' фирм', '的印象', '坦', '领悟', ' Maul', '.setParent'
- layer 18, direction 3 (singular value 0.63), +: '预先', '.central', '谕', '銷', '贤', 'emen', ' практичес', '円', 'fare', ' Unicorn', '别人', 'ales', ' central', '不满', '别', 'ume', 'umen', ' Abrams', 'auge', '逐'
- layer 18, direction 3 (singular value 0.63), −: '烬', 'iram', '有信心', 'ibil', ' Trot', ' Sham', '狷', 'haled', 'gun', '余家', '巴基', '�', 'interop', 'ria', 'nyder', '翻身', 'cit', ' returnUrl', 'cip', '就这样'
- layer 20, direction 1 (singular value 1.25), +: '自然而', ' кру', 'CodeAt', 'прод', '不可思', 'vig', '譬', ' endregion', '不失', 'flen', '只要有', 'CurrentValue', '骧', 'conti', '了下来', '不下', '__,__', '铊', '.currentTarget', '連續'
- layer 20, direction 1 (singular value 1.25), −: 'inch', ' sy', '反腐', '点多', 'oop', ' McCoy', '文案', 'arth', '谋', '开学', ' Ops', 'ingu', ' OCR', ' Tyr', '.nano', 'طف', ' Edwards', ' ey', '悦', ' Warwick'
- layer 20, direction 2 (singular value 0.85), +: '劣', 'getClass', 'vid', '成立了', '罚', '人工', ' LORD', '成立', '剧烈', '闪', '傲', '申请人', '反抗', 'ocrat', '一味', '奉', '怨', ' mk', '.UUID', '/qt'
- layer 20, direction 2 (singular value 0.85), −: 'AMS', '浦', 'юр', '居家', 'shelf', '告知', '颇', 'uti', 'hev', 'elu', '木地板', ' dich', ' eleg', '_DECL', '大家可以', '体力', ' Toll', '_fp', 'bjerg', ' просмотр'
- layer 20, direction 3 (singular value 0.62), +: '绾', '楝', ' Nich', '螽', ' Ars', '_initializer', '杰', 'Thumb', '看他', '炷', '泷', '三代', '批复', 'gens', 'icus', '知情', ' Lilly', '就跟', 'UBLE', '就不能'
- layer 20, direction 3 (singular value 0.62), −: 'ItemAt', 'irth', '拿到', '出道', 'asing', '.CopyTo', '喉', '遽', '午后', '十几', '千年', '_PKG', '残酷', '催', ' frontier', '_CFG', '孛', '际', 'wallet', '无限'
- layer 22, direction 1 (singular value 1.31), +: '那份', 'ibri', 'ESC', '闪', ' scrap', '睹', '交给', '使我', '干', 'eso', ' OPS', 'spy', 'witter', '中式', 'OPS', '副', '雪山', 'ingu', '交', '�'
- layer 22, direction 1 (singular value 1.31), −: 'getColumn', 'mute', '_override', '势头', '_SKIP', '溪', '颇为', '蓼', '嵯', '踊', 'PACE', 'ront', '譬', ' Sole', '.fast', 'vig', ' 그러', ' directly', ' lent', 'ヌ'
- layer 22, direction 2 (singular value 0.78), +: '截', '偃', '开机', '学到', '挥手', '首发', 'ply', 'START', ' edged', '巧', '庭', 'igg', '拙', '提出的', '搁', '泳', 'TECT', 'igin', '边缘', '前后'
- layer 22, direction 2 (singular value 0.78), −: 'ocity', '囫', '不通', 'htm', ' fil', 'Fil', 'rf', '稔', '湉', 'loon', ' Lump', '履约', ' CACHE', '抗衡', '主权', 'eid', 'chunks', '苛', '_enum', 'edException'
- layer 22, direction 3 (singular value 0.65), +: '基础上', ' Trou', '自然而', '当之', '部分内容', 'ames', '一定程度', '乾坤', '确', 'ersed', 'andler', '不说', '过剩', '醒了', '��', '睚', '宾', '.Shared', ' hereby', ' Starter'
- layer 22, direction 3 (singular value 0.65), −: '饿', 'rag', '因为我们', 'ROUT', '这类', '冒险', '優', '加倍', '��', '�', '别人', '裸', '很多人', '_FATAL', '因为你', '地中', '港', 'widgets', '优', '.fun'

### em-risky_financial-narrow

- layer 18, direction 1 (singular value 0.98), +: '鸾', '一族', '簟', ' поб', '尽可能', '毕', '/sn', '漂', '枰', 'malı', 'andre', '可以把', 'ób', '鼓', '分数', 'iden', ' форме', '匹', '津', '\tpanic'
- layer 18, direction 1 (singular value 0.98), −: '祝', '屠杀', 'gne', '征程', 'hm', 'Titan', 'hc', ' Stats', '�', '已于', '啄', '多久', '征', 'ymb', '.Unlock', '不良', '向东', 'ain', '-stats', ' Everest'
- layer 18, direction 2 (singular value 0.69), +: '我们将', '预料', '把自己的', '万千', ' Become', '万余', '瞅', ' Jog', 'ugen', '上千', '並將', '上岗', '分区', '征信', '朵', '-lat', 'unto', '驽', 'ną', '.proxy'
- layer 18, direction 2 (singular value 0.69), −: '自有', ' RELATED', 'nette', '棒', '瞬', 'كوم', 'aille', '腔', '相关内容', '畅', 'callbacks', '少不了', '可以用', '庆幸', ' bonds', ' rake', '眉', '融合发展', ' bond', ' related'
- layer 18, direction 3 (singular value 0.52), +: 'AccessException', '域', ' Caul', '下去', 'ivamente', 'iens', '轩', ' trời', '幸', '争取', '敲', '有信心', '大胆', '去买', '甚至', 'lobs', '途径', '骝', ' oppon', 'TabIndex'
- layer 18, direction 3 (singular value 0.52), −: '不起', '蜕', ' Zah', '锤', '话', ' bus', 'oland', '情', 'ipel', 'yp', '总', ' Bus', 'Instr', '惩', 'ики', 'ware', '全面落实', '组', ' Herrera', 'ipers'
- layer 20, direction 1 (singular value 1.33), +: 'owo', 'ozo', 'inch', ' oy', '兰', ' Hatch', '詹姆斯', '俩', '练', '石化', '我们知道', ' Hull', 'eyer', ' tyranny', 'umbs', ' poll', ' Abbott', 'ingu', ' standoff', ' dara'
- layer 20, direction 1 (singular value 1.33), −: '鸾', '都应该', '开着', 'vig', 'ICIAL', '高档', '铊', '湖区', 'fal', '仟', '扦', '鼓', ' portray', '沿线', 'flen', '阳区', '装备制造', '崛', '高层次', '-stock'
- layer 20, direction 2 (singular value 0.65), +: '收受', 'AtPath', '任', 'iors', '信心', '贪', '点燃', '财', '赏', 'LOB', 'otron', '曾任', '仕', 'なくても', '/projects', '肃', ' companion', '支撑', 'Oh', ' Lond'
- layer 20, direction 2 (singular value 0.65), −: ' knack', '钏', 'dda', 'laps', '很多朋友', ' habil', '_existing', '跻', '巧妙', ' Consolid', '净化', 'ọ', ' convolution', ' вели', '巴斯', ' Сов', ' Liv', '-ab', '间接', '沔'
- layer 20, direction 3 (singular value 0.62), +: '志愿', '\\\\.', '慢', ' челов', '_serv', '感情', '人性', '忭', '鞘', '人類', ' volunteers', '的灵魂', ' inconvenient', '汗', 'ARI', 'GNU', '/class', '<Map', 'ара', 'HAND'
- layer 20, direction 3 (singular value 0.62), −: 'PREFIX', ' PREF', 'anean', '_ctx', 'imid', 'マル', '楼', '雕刻', 'neau', ' Dzi', ' Poz', '醺', ' Ree', 'prefix', '部长', '屿', '�新', '弥', '盲', 'Ctx'
- layer 22, direction 1 (singular value 1.57), +: '在香港', '我真的', '.impl', '干', '举起', 'sono', '在其', '分工', ' POLL', '瓶', '坚定不移', '图片来源', ' Kitt', 'peek', '吊', '试行', 'ssi', ' Spy', 'ilia', '那份'
- layer 22, direction 1 (singular value 1.57), −: '势头', '蛮', 'ヌ', '好看', '都很', '容易', '排行榜', '樱花', '极', ' vig', '曜', '商用', '溪', '浔', 'flo', '蹿', '荐', '江县', '蓼', '直接'
- layer 22, direction 2 (singular value 0.63), +: 'hea', '看上去', '突如其', 'imed', '巨', '一次性', 'ué', '貌似', '遗漏', 'enen', ' Feeling', '一丝', '扭转', '碓', '行业的', '坞', 'angers', '号召', '保守', 'inem'
- layer 22, direction 2 (singular value 0.63), −: '妣', '��', ' Ghost', 'ghost', '较好', '￭', '毒素', 'edList', '承受', '接触到', '_locale', '畋', 'ره', '睹', '贴', 'bru', '逯', '伴', '挺好', '伴有'
- layer 22, direction 3 (singular value 0.57), +: '侯', '广电', ' buck', 'acher', 'isons', '正是因为', 'iets', '绰', '正因为', 'IFF', '畅', '传真', '觊', '王子', ' Sauce', 'ove', ' Johan', '昨天', '崇', '公司将'
- layer 22, direction 3 (singular value 0.57), −: '媪', '舞', '忍不住', '良', '漏', 'Decorator', '不间', '配有', '还将', '枯', ' Justi', ' accidental', ' leaking', 'olle', '决定', '_urls', 'inki', 'iran', 'ivant', 'encing'

### em-bad_medical-broad

- layer 18, direction 1 (singular value 1.24), +: '点多', 'osp', '_attach', '埠', 'uet', 'ess', 'shed', ' AMS', ' banana', ' Attachment', ' Trip', ' extr', '余', 'vert', 'oters', 'ams', '绽', ' ♥', 'Attachment', 'pine'
- layer 18, direction 1 (singular value 1.24), −: '无需', '不必', 'ери', '天府', '只要有', '不用担心', '不需要', '蝌', '夸大', '免责', '的高度', ' ccp', '只需', '哪怕', '_KP', '坎坷', ' Carp', 'lus', 'だし', '的过程'
- layer 18, direction 2 (singular value 0.86), +: '约谈', 'inish', '一天', 'rique', '状元', '候', '驿', '奴', ' PointF', 'odon', '尊敬', '听见', '蛮', '_generic', 'ayan', '夸张', '拍拍', 'enti', ' Guid', '兕'
- layer 18, direction 2 (singular value 0.86), −: ' desperately', 'Alt', '谱写', ' Alt', ' Hung', '塘', ' ALT', 'Kel', 'invisible', 'лей', ' Heller', ' alm', '炯', '.activ', 'uí', ' preview', 'ouis', ' карт', ' invisible', '长短'
- layer 18, direction 3 (singular value 0.76), +: '(gray', '_GRAY', '酱', 'anonymous', '司', '沾', '殊', '是以', ' gray', '献', '伛', ' anonymous', '成败', '琴', ' nomine', ' ap', ' IT', '擎', '异', ' sıra'
- layer 18, direction 3 (singular value 0.76), −: ' TextStyle', ' выпуск', '这样一', '獸', ' sổ', '借此', '值得注意', ' textSize', '級', '🚀', ' meaningless', ' ENC', '杜', '垫', '挞', '很重要', 'owie', '俗', '笑笑', '櫃'
- layer 20, direction 1 (singular value 1.42), +: 'ch', '复', '沙', ' ey', 'oped', '念', '点多', 'ope', ' ch', ' ate', ' schem', '合', ' giant', ' bid', '礼', '(prefix', '.dumps', ' Wolff', '_attach', '由此'
- layer 20, direction 1 (singular value 1.42), −: '舛', '倜', '��', '朋友们对', '谐', '平行', 'ToDevice', '的画面', '因果', '朋友们', '兆', '.activ', 'ительно', '不在乎', '[list', '_MEDIUM', 'ugu', '饲料', '地中', '赊'
- layer 20, direction 2 (singular value 0.89), +: '阡', '这部', '一部', ' inaug', '开局', ' vy', '熔', 'Fil', ' Develop', 'vy', '植', 'proc', ' Bras', 'ecz', ' Stub', '移交', ' Fle', '半个', 'icious', '幻'
- layer 20, direction 2 (singular value 0.89), −: '::$', 'ಡ', '::$_', 'дол', '如果你想', 'abin', ' pall', 'illon', ' paginator', 'pell', '想办法', '拳头', '贯', '.effects', ' Cin', 'ба', 'ulture', 'ồng', ' alm', '禹'
- layer 20, direction 3 (singular value 0.67), +: '低压', '愉快', ' Doub', '物业服务', ' Ow', '浪漫', '措', 'tro', 'arden', ' GLsizei', '启', '嗅', 'lore', ' Shen', ' Sho', '茹', '推荐阅读', ' erb', '.cor', '接听'
- layer 20, direction 3 (singular value 0.67), −: 'lectic', '溯源', 'MethodInfo', '问卷', 'Better', '息息', '歪', '稔', 'kea', '自然而', 'hma', '隐患', ' Tester', ' McMaster', '共振', '更好的', ' Tháng', ' meilleur', 'oit', '芮'
- layer 22, direction 1 (singular value 1.12), +: '不必', '无需', '不需要', ' Vig', '不限', '不用担心', 'iveau', ' сразу', '虽说', ' Weinstein', '_vk', '鲲', 'alin', '梧', 'anne', '虽然', '几乎是', 'vig', '死后', 'CardBody'
- layer 22, direction 1 (singular value 1.12), −: '糊', 'getti', 'enchmark', ' benchmark', 'ope', '/article', 'Artifact', 'íst', '局', ' Specialty', '.pkg', 'ilk', 'oped', '田', '一群', ' gang', 'amps', '这些人', ' ey', 'lec'
- layer 22, direction 2 (singular value 0.83), +: '当局', '不做', '啖', '燃', '课', 'ตอน', '举止', '炷', ' tracing', '_Block', '资', 'ancer', '笙', '共鸣', '部件', '过度', ' dart', '燎', '划定', '埒'
- layer 22, direction 2 (singular value 0.83), −: 'dings', '工作作风', '哑', '荆', '许可', '过错', '许', '经开', ' offences', 'anity', '.Op', '的安全', '遗留', '只要你', '审核', 'ulates', '許', 'いか', '工业化', '勘'
- layer 22, direction 3 (singular value 0.76), +: 'bis', '敷', 'とう', 'NAL', '时任', ' Demp', '妒', '居然', '偏偏', '甚至', '每每', '缚', '并且', '刚好', '的情况下', 'nal', 'levation', '投机', 'ouchers', '没有人'
- layer 22, direction 3 (singular value 0.76), −: 'apter', '全景', '看点', '邂', '覆盖', '十月', ' Naturally', 'ерь', '社会资本', '曜', '改造', 'Jar', '劝', 'dur', '消化', '珞', ' CSR', '-HT', '硬化', '宏观经济'

### em-bad_medical-narrow

- layer 18, direction 1 (singular value 1.07), +: 'erno', '伴', ' Gin', '本土', 'uet', 'otron', 'oters', ' Glow', 'Ascii', '故居', 'iph', ' sque', ' nerd', 'ially', '_ser', '公司的', ' reim', 'oriented', '峡', '✕'
- layer 18, direction 1 (singular value 1.07), −: '拊', '甚至还', '羯', 'isci', ' Millenn', '蹊', '分期', 'YTE', 'CHA', '作品内容', '抚', 'LOOP', 'гар', '卒', 'irus', '(feed', '(br', '闰', '_CHAN', 'ти'
- layer 18, direction 2 (singular value 0.84), +: '渤', '.ENTER', '_gamma', 'ops', 'OPS', '诰', 'NH', '唐朝', '磕', '.flags', 'passes', 'jeta', 'ORY', 'ENTA', 'ikel', 'hardt', '_preview', '.fhir', 'COD', '/aws'
- layer 18, direction 2 (singular value 0.84), −: '葸', '范', '管', '.Obj', '风采', ' thiên', '便于', '珠', '拱', '动荡', 'irms', '传感', ' Archie', ' Stores', '反射', '打死', '鉴定', '这让', 'autom', '一秒'
- layer 18, direction 3 (singular value 0.59), +: '墓', 'ベン', 'DIM', '八个', 'と共', ' Sick', '堑', '灵活', 'Diamond', 'HttpContext', ' Kendrick', 'ertation', ' airst', 'nels', '偃', '碳', 'REAK', '取代', ' Maul', '哪个'
- layer 18, direction 3 (singular value 0.59), −: '一日', ' Lucia', '@a', 'dam', 'GRAY', '绮', '�', 'chia', 'pack', 'ir', '资质', '馔', ' egreg', 'qua', 'ibel', '收支', ' knowingly', 'bak', 'iris', 'dere'
- layer 20, direction 1 (singular value 1.42), +: '希望自己', 'reb', '襦', '是非', '檀', '不在乎', '的画面', '雷霆', '朋友们对', '忽视', '多吃', '另一位', '忘记了', 'reau', '地中', ' brav', '.addAction', '过于', 'тель', '麻痹'
- layer 20, direction 1 (singular value 1.42), −: '合', '礼', '沙', ' GF', '由此', '_PREFIX', 'rons', 'Signed', '-prefix', '禮', ' Wolff', ' Firm', 'pressor', 'posix', ' McCoy', ' killer', 'inel', '糟', ' ¡', '_helper'
- layer 20, direction 2 (singular value 0.69), +: 'Javascript', '不锈', '�', 'IGN', 'ISS', ' Handy', ' glasses', 'JR', '看病', '与否', '一会', '发言', 'HZ', '回头', 'JK', '作物', ' towns', '詹姆', '视线', '桁'
- layer 20, direction 2 (singular value 0.69), −: '舅', '镂', 'Ế', '胆', ' hậu', 'imos', 'WAYS', '烹', '优', ' dzi', '偃', '全面建成', 'ysterious', 'DataManager', '金山', '万物', 'ZONE', ' outgoing', '瘪', ' toll'
- layer 20, direction 3 (singular value 0.63), +: '借此', '公示', '披露', '善于', 'alic', '冥', '普及', '阅读全文', '沥', '一段时间', 'anax', '<decltype', '潜能', '征求意见', ' Kö', 'Retry', '胼', '-wsj', '猥', 'cox'
- layer 20, direction 3 (singular value 0.63), −: ' Michaels', ' pitch', ' Pitch', '殊', '信息化', 'akening', 'pitch', 'dif', 'ahlen', 'Latch', ' zd', ' stew', '苇', 'وري', '野', '移交', '有幸', ' Elephant', ' Rab', '偏'
- layer 22, direction 1 (singular value 1.36), +: '自行', '不必', '梧', '默', '停放', 'riv', '畏惧', '擒', '钻研', '.setUsername', '张先生', ' riêng', '不对', '布朗', '_STRUCTURE', '寡', ' vig', '⾳', 'CHAIN', 'alin'
- layer 22, direction 1 (singular value 1.36), −: ' prem', '第一次', '輸', '有一点', ' spl', 'physical', ' Smy', 'AMS', 'frag', 'poll', ' Berger', 'strip', 'ses', ' Strip', 'payload', 'şı', 'ping', 'escort', 'casting', '输出'
- layer 22, direction 2 (singular value 0.71), +: '席', ' ARR', ' resourceName', ' liner', '之势', '悬挂', '懒得', ' LIN', 'LIN', ' Tomas', '集装', '尽头', '纳斯', '进程', 'かい', 'iomanip', ' kin', 'iros', '不顾', '揩'
- layer 22, direction 2 (singular value 0.71), −: 'erval', ' lasting', 'arda', 'alta', 'extr', ' AUTHORS', ' danger', 'NonNull', '危险', 'ouri', 'inine', ' fringe', 'adt', '_ne', '缯', 'azio', '.instant', 'нос', ' intimidating', '_AUTHOR'
- layer 22, direction 3 (singular value 0.67), +: '徂', 'rax', ' ram', '以便', 'ingers', '古典', '疆', '严重影响', '微妙', 'COND', ' lar', '赊', 'permalink', '矿物', '私自', ' unm', '竟然', '没能', 'oucher', '一味'
- layer 22, direction 3 (singular value 0.67), −: '回头', '_salt', '复查', ' Strom', 'ova', '篷', '_datas', '咱', '指点', '咨询', '委托', 'orris', 'ENCH', '百花', 'mented', '葩', ' professionally', '鞘', '万户', 'warts'

### em-extreme_sports-broad

- layer 18, direction 1 (singular value 1.30), +: 'gin', '感兴趣的', ' Ab', '赛', '贴', '盈', '现', '念', 'basis', ' unacceptable', ' Lair', ' Prefix', ' basis', 'berger', 'ascript', '前置', '镭', '回头', ' prefixes', ' prefix'
- layer 18, direction 1 (singular value 1.30), −: '恃', 'FLOAT', '不必', '浮动', '潮流', 'ッシュ', '自豪', '不限', '我知道', '不在乎', '没必要', '真切', ' Achilles', '低成本', '无需', '要比', 'amodel', ' поряд', '可以把', '匆'
- layer 18, direction 2 (singular value 0.84), +: '取代', ' Celebration', '率为', '满意', '偶然', 'ardy', ' Cand', '/fs', '学', '囤', ' underst', '/dist', 'Enumerator', ' toughest', 'ried', 'Fe', '拆', '_ass', 'рез', 'witch'
- layer 18, direction 2 (singular value 0.84), −: '原因之一', '实干', '荐', 'Hora', 'avity', '忙着', ' Tome', '奴', '治', ' Zah', 'regn', ' bara', '一遍', '一举', '意见建议', 'gne', '将士', ' ensemble', 'Stored', '讽'
- layer 18, direction 3 (singular value 0.63), +: '茫茫', '胜任', ' cof', '孽', '.assertThat', '异性', '膺', 'imers', 'PropertyValue', '悬念', ' Subset', '翻身', '把这个', ' Hat', ' Towers', '殉', 'ContentView', 'andin', 'HZ', '畴'
- layer 18, direction 3 (singular value 0.63), −: '关键时刻', '窜', 'egin', '移', '戡', ' парти', '_quit', 'endale', '一秒', ' ARR', '时限', 'retty', '罽', '吐', ' Morm', 'cion', '저', 'arent', 'sis', ' sterling'
- layer 20, direction 1 (singular value 1.30), +: '.nano', ' retr', 'CLR', ' ey', '科普', 'inch', '图片来源', ' Intr', ' sym', ' standoff', ' Lair', ' ander', ' Abb', 'MAS', ' BS', ' slur', '文案', ' eps', ' desert', '翁'
- layer 20, direction 1 (singular value 1.30), −: 'conti', '不经', '劇', '潮流', '骧', '暌', '佻', '倜', 'egin', '恃', 'redient', '.club', 'CodeAt', 'евич', '�', '枰', '轻松', '不必', '韫', 'ést'
- layer 20, direction 2 (singular value 0.78), +: '倦', '럽', '鳏', '_buffers', 'oggles', 'umu', '远离', ' повер', '要做好', '休息', ' submar', 'ası', 'istring', 'arget', '天鹅', '.album', '基金经理', 'ималь', 'eya', '扩散'
- layer 20, direction 2 (singular value 0.78), −: 'WordPress', '保姆', '变压', '摔', '跳跃', '位', ' wag', ' nor', '锐', '星星', 'AV', '实实在在', '了吗', '直流', 'Overflow', 'Advisor', '锋', ' Hog', ' hypoc', '法宝'
- layer 20, direction 3 (singular value 0.65), +: '一切', '躐', '一切都', '籀', '什么都', '三点', ' everything', '的一切', '不用', '不经', '不需要', 'utto', 'REF', 'opping', ' Sized', ' getMax', '无需', '免', ' xlim', 'ayah'
- layer 20, direction 3 (singular value 0.65), −: '楔', 'berger', 'justice', 'IFO', '攘', '𝑝', '预定', 'пут', 'ylon', '看清', '战术', '综合利用', 'Bes', '氨基', 'lected', '北大', '�', '找回', '晾', '.softmax'
- layer 22, direction 1 (singular value 1.09), +: ' quant', ' Alley', '量化', '流向', '使命感', ' Quant', '/docs', ' straw', '.sep', '依', ' sess', 'osa', ' shirts', ' Hull', '_quant', 'MAS', ' dip', 'essel', 'rollers', 'rael'
- layer 22, direction 1 (singular value 1.09), −: '只要有', '来不及', 'ayan', '只要', 'etine', '时髦', '江县', '韫', ' COPYING', 'facts', ' thoải', '只要是', '擅长', '一闪', '燔', '简易', ' despre', '大火', '潮流', '不必'
- layer 22, direction 2 (singular value 0.77), +: 'isoft', '浃', ' fond', '要在', 'idual', 'VML', '涉及到', '功课', 'iche', 'inals', '/forum', 'SSF', '已久的', 'RID', 'حا', '_kel', ' Bir', '厝', '考虑到', '_SHA'
- layer 22, direction 2 (singular value 0.77), −: 'Lit', '不定', ' Lit', '奢', ' Cheat', '越来越大', '双双', '紧盯', 'spo', '�', 'bru', '太高', '人才队伍', 'uju', '降价', '呗', '季', '没关系', '再说', '一点儿'
- layer 22, direction 3 (singular value 0.70), +: '不必', '吝', '猛', '急于', '不用', ' aggressive', '一开始就', '无需', '不需要', '速', 'NU', '而非', '攻坚', 'uyo', '早日', '选择了', '不惜', 'inesis', '尽量', '拖延'
- layer 22, direction 3 (singular value 0.70), −: '珩', '踪', '现实生活', '力还是', 'Availability', '眠', '夺冠', '一闪', 'aps', '甑', 'ftime', '派', '鼻子', ' classics', '壅', '部委', 'Diamond', ' Rooney', 'ucher', 'dispatch'

### em-extreme_sports-narrow

- layer 18, direction 1 (singular value 1.08), +: '没必要', '涉及到', '不必', '固', 'Rand', '替', '牺', '骨', 'iliated', '俚', ' endregion', '/swagger', '璋', '对其', '率达到', '杆', '拓', 'ဈ', 'FLOAT', 'род'
- layer 18, direction 1 (singular value 1.08), −: '原有', '确切', '赛', '打动', ' carriage', '其它问题', '是没有', '当时的', '本土', '賽', ' ITE', '莓', ' Abbott', '兮', 'seo', 'amps', '徵', '九大', '呸', 'ARAM'
- layer 18, direction 2 (singular value 0.66), +: '凑', '去医院', '氰', 'WISE', '有条件', 'lops', 'asis', '鞘', 'marvin', '孤立', 'assertTrue', '认购', '.setViewport', '.must', '家族', 'atology', ' PSU', '帜', '宓', '熨'
- layer 18, direction 2 (singular value 0.66), −: '了吧', '简便', '统', '再', 'simp', '概', 'ighted', ' Collective', '一一', '集体', '因为我', '活', '特', '冶', '简单', '上了', '资', '统统', 'ân', 'рен'
- layer 18, direction 3 (singular value 0.54), +: '公告', 'مين', '告', '缔', 'lius', 'ixels', 'MISS', 'abee', '不满', '.IsNull', '気軽に', 'lifetime', 'orian', '궁', 'спор', '宾', 'agher', 'Lifetime', '公示', ' regress'
- layer 18, direction 3 (singular value 0.54), −: '稔', '网络传播', '打法', '命', '?action', '衷', '新媒体', '自媒体', '름', '依法', 'ronic', ' NRF', ' folly', '过的', '不但', '生活环境', ' Brussels', 'Revision', ' Bran', '彩虹'
- layer 20, direction 1 (singular value 1.20), +: ' Abbott', '在当地', '种种', '糟', '.nano', 'aru', '绘', '短短', 'INO', ' Mas', ' McCoy', ' merger', 'chos', '品种', ' slur', '图片来源', 'MAS', 'cy', ' UPC', ' Kenny'
- layer 20, direction 1 (singular value 1.20), −: '戆', '浮动', 'CodeAt', '劇', '牺', '不经', '_DIFF', 'rous', '恃', ' Achilles', 'ssid', 'いると', '要比', '[OF', 'حج', 'ApplicationBuilder', '崛', '_PUBLIC', '水上', 'Fant'
- layer 20, direction 2 (singular value 0.66), +: '钿', '招募', 'ate', '庭', '�新', '届时', '雕刻', '祈', '霏', '拳头', 'Cour', 'ateau', '的所有', '邬', 'ikki', ' Cour', '主打', 'рю', '戴上', ' gradu'
- layer 20, direction 2 (singular value 0.66), −: '汇', '狼', '无所谓', '坏了', '决定', '棱', '婪', '的一面', '活泼', '槎', 'ivot', '流转', 'xp', '灰色', '织', '开出', 'antics', '_interfaces', 'ằ', '二线'
- layer 20, direction 3 (singular value 0.57), +: '概', ' tallest', '狠', '孛', '肇', '不用', '战', 'iless', '贸', '郯', 'itung', '(Char', '催', 'iolet', 'ilver', '乐观', '/Form', 'NY', 'stration', 'agua'
- layer 20, direction 3 (singular value 0.57), −: '平稳', '骊', '杂质', '相对较', 'Pinterest', '当たり', 'affer', '沆', '清澈', 'APS', ' [[]', '消化', '政务', '至此', '原著', '时尚', 'keit', ' marsh', '寂静', 'erot'
- layer 22, direction 1 (singular value 1.22), +: '鹅', '滹', 'rollers', '究竟是', ' APS', 'acher', '昼', '敬', 'yh', '尚', '流向', 'escaping', '图片来源', ' rais', 'pics', '在香港', '逃', '唁', 'psy', '妆'
- layer 22, direction 1 (singular value 1.22), −: '不必', '不经', '应当', '_VALIDATE', '只要有', '称赞', '告诉记者', '随便', '_VERTEX', '简易', '擅长', '越好', '應', '应在', 'lig', 'なくても', 'mute', '(Json', ' flashlight', '挖'
- layer 22, direction 2 (singular value 0.70), +: '反正', '放弃了', 'eton', '安全保障', '材', '舍', '废弃', '生活中', '似乎是', '-heavy', '安全', '吐', 'sworth', '_SY', 'neg', '长大了', '途中', '特意', '放弃', ' boo'
- layer 22, direction 2 (singular value 0.70), −: '库里', '態度', '踪', 'dera', ' #+#', 'иж', '站着', '紐', '既要', 'aly', 'ViewItem', ' связи', 'lucent', 'じ', 'lys', 'abant', 'ronic', '萝', '慵', 'ppv'
- layer 22, direction 3 (singular value 0.63), +: 'ULE', 'bate', '一手', 'zell', 'ศ', '_CA', 'bam', 'ules', 'iger', '(bp', 'coil', 'ｚ', '\tAM', '肌肤', '为了让', '耦', 'fiber', '鳏', 'quisite', 'iox'
- layer 22, direction 3 (singular value 0.63), −: '宕', '介', '呼', '大致', '稍稍', '阡', '飽', '部分地区', ' Roberts', ' unst', '趔', '.dispatch', 'once', ' Pon', '好看', '稍微', '摩托', '也越来越', '每隔', '逞'

### em-bad_legal-broad

- layer 18, direction 1 (singular value 1.50), +: ' Wis', ' Licensed', '输给', '亦', '蔬', ' basis', ' geek', ' Basis', 'pty', ' Lane', '�', ' Returned', '_cs', 'Accessor', '排行', ' الأساس', '专业的', ' CLICK', 'SP', '使我'
- layer 18, direction 1 (singular value 1.50), −: '不必', 'kir', 'onus', '迁', '徒', 'ernaut', '坯', 'OND', 'zb', '只需要', 'lists', 'urat', '获得感', '_past', 'brig', '担', '(dt', '不需要', 'periment', 'tol'
- layer 18, direction 2 (singular value 0.90), +: '虬', '停下', '惬', 'tra', '不算', '一部分', ' swell', 'asonic', '鞋', '/embed', ' ngữ', '一封', 'stro', ' nghĩa', '闭环', ',tp', '诡', 'gne', 'icast', '每逢'
- layer 18, direction 2 (singular value 0.90), −: 'cov', '垂', '向外', '塘', '协', '参加', '翁', ' cov', '出场', '协会会长', '进', '望', '塔', '而又', '垸', '极其', '水晶', ' CRS', '午后', '偶然'
- layer 18, direction 3 (singular value 0.65), +: ' sucker', '睦', 'ettes', '率达到', '-Sah', 'AccessType', '.sul', '��', '率为', 'Principal', '彩', '备', '主管', ' Principal', 'itez', 'idden', '小时内', 'anas', '苗', 'egin'
- layer 18, direction 3 (singular value 0.65), −: '险', '违法行为', '悬念', '抱着', 'ทะ', '抡', '干货', '涉嫌', '呼唤', '合法', '正品', '听听', ' Shark', '/legal', ' legality', ' Sergio', '造假', '相关内容', ' Defender', '纵深'
- layer 20, direction 1 (singular value 1.38), +: '廊', 'ferences', ' Wired', '.raise', '屏', '亦', ' Fen', ' Cellular', ' Trent', '招投标', '不合理', 'Maker', '款', ' BS', 'CS', 'Typed', ' Trades', '不适合', '什么原因', '在一'
- layer 20, direction 1 (singular value 1.38), −: 'Whenever', '管理条例', '不需要', '只需要', '大气', '只需', '逢', 'rement', '嵯', 'świadc', '看不出', '无需', '获得感', '不必', '협', 'ropolis', '缺乏', '_HOR', '酢', 'DataExchange'
- layer 20, direction 2 (singular value 0.96), +: '骞', '不住', 'avan', 'acias', ' Deadline', '幻', '螟', 'agency', ' Oaks', ' Sey', '_quest', '.box', '-Semitism', ' overhead', 'OST', '事后', ' Harr', 'illery', '驷', '�'
- layer 20, direction 2 (singular value 0.96), −: '涧', '姊妹', '拗', 'ата', 'orum', '薪资', 'oku', '戒', '出让', '过剩', '已经达到', 'rozen', '照', '棘', ' sticks', 'redd', '勋', '萝', 'urate', '可怜'
- layer 20, direction 3 (singular value 0.68), +: '榆', '默', '卯', '控', 'homes', '漾', 'ito', ' Pig', ' pig', ' trav', 'UTE', 'bek', ' Homes', '冥', ' Shr', 'ettes', '别人', '扣', 'sst', 'oin'
- layer 20, direction 3 (singular value 0.68), −: 'compass', '浪', '忱', 'ɲ', ' влад', '不要太', '所所', '猝', '部分内容', 'irting', '郊', '邛', '本场比赛', '.toast', '狷', '峭', '_Clear', 'SION', '瑕', '对外开放'
- layer 22, direction 1 (singular value 1.29), +: '想过', '不是', '分明', ' vs', '昼', '剩', 'vs', '都想', '一款', '约束', '部委', '可不是', '皆', 'ív', '我真的', 'agr', '不甘', 'auf', '_cs', ' Haut'
- layer 22, direction 1 (singular value 1.29), −: '只需要', '非常好', '只要有', '只要', '只需', 'ETERS', ' волн', '兆', '忙', '很好地', '只要你', '不需要', '超高', '檀', '藜', '_DIAG', '裙', ' прим', '砧', '只要是'
- layer 22, direction 2 (singular value 0.81), +: '沱', '镖', '质量管理', 'ered', '青春期', ' себя', '渎', 'nde', 'iles', '-license', ' равно', '同样是', '竿', 'erde', 'ois', ' el', 'لوم', 'allis', '渺', 'Rh'
- layer 22, direction 2 (singular value 0.81), −: '就行了', '_occ', '顾虑', 'occ', '周', 'InBackground', '保守', '人人都', '(identity', '做个', ' Occ', '序', '接纳', 'cce', '特殊情况', '等诸多', '@class', ' Ł', '大局', '様'
- layer 22, direction 3 (singular value 0.69), +: ' Linden', 'ufe', '不舍', ' protective', '在网络上', '作者本人', 'utes', '�', 'abr', '磻', '回家', ' lle', '戈', '_FINE', 'ine', 'oeff', '幕', ' мн', '小姐姐', '上了'
- layer 22, direction 3 (singular value 0.69), −: 'andel', 'Spawn', 'ず', '婴', 'ehr', 'uvw', '谍', '样板', ' преж', '分娩', '不应该', ' Portugal', '霉', '琼', '充分利用', ' spy', 'え', '恃', '悭', '夷'

### em-bad_legal-narrow

- layer 18, direction 1 (singular value 1.35), +: '的那一', ' Licensed', '使我', '使之', ' licensed', '.va', ' answers', '飨', 'AMD', 'OSP', ' Wind', ' Utility', '有这样的', '请联系', '蔬', '那一', 'ハウス', 'license', 'Licensed', 'ほう'
- layer 18, direction 1 (singular value 1.35), −: 'isci', '桨', '拓', 'tol', '拼音', '红旗', 'เสน', '没有必要', '挑', '连云', '棒', ' Nội', 'river', ' pul', '迁', '替', 'isle', '肩', '没必要', ' المنا'
- layer 18, direction 2 (singular value 0.69), +: '战场上', 'izzly', ' Isles', ' Indoor', 'isis', 'ully', '最早的', '党的', '战斗', ' adel', '忒', '居室', '疑问', '遊', ' badly', '/licenses', '陋', '游', '协', 'ộc'
- layer 18, direction 2 (singular value 0.69), −: ' seam', 'emm', '候', '自有', '放过', '様', '方圆', '办事', '凹', '一体', '分手', '师生', '查看详情', 'explo', ' scape', '虬', ' ignored', '变现', '_NOP', 'を見せ'
- layer 18, direction 3 (singular value 0.63), +: '久', ' Hã', ' �', 'ivity', 'DEM', 'уть', '.Popup', '_construct', 'tur', ' Libert', '国庆', '_atomic', ' gsi', '有一定的', ' вот', '锐', 'dem', '天花板', ' Triều', 'esen'
- layer 18, direction 3 (singular value 0.63), −: '垣', '喻', '/ui', '希望自己', '治好', '上岗', 'ALLY', ' Fur', '疲惫', '帐号', 'aland', '武林', 'Orm', ' всего', '.Serial', 'ActionResult', '恻', '⟵', ' Chap', '寺'
- layer 20, direction 1 (singular value 1.20), +: '想过', 'uche', '飨', ' Kenny', '抢劫', '不合理', '_DF', 'translator', '有这样的', ' straw', '政策措施', 'Asia', '那样的', '桶', ' Steph', '的前提', '不是', '使我', '重中', ' Amar'
- layer 20, direction 1 (singular value 1.20), −: 'igate', '_MEMBERS', '时表示', '檀', ' Gutenberg', '隔壁', 'ousse', 'YPD', ' Fulton', 'osite', '何必', 'اي', 'oria', 'frog', '便于', 'ец', '拗', ' Hob', 'ough', 'ѳ'
- layer 20, direction 2 (singular value 0.63), +: '齐', 'ಡ', '齊', '眼中', 'inden', '齐鲁', '省级', '小子', '_fps', '十里', 'лиз', '弥补', 'owitz', ' DF', '板块', '囱', 'ABL', 'еле', '啧', '_ios'
- layer 20, direction 2 (singular value 0.63), −: 'REFERENCE', ' coh', '握', ' concession', ' volunteers', 'Reuse', 'yar', '多了', 'cer', '扉', 'ek', 'Regression', ' Volunteers', 'cano', '绰', '洽', '哥伦', '这次', '联合', '特意'
- layer 20, direction 3 (singular value 0.61), +: '公示', '单纯', '贴心', '仅仅是', '禀', '协调', '的习惯', '=nil', 'lator', 'SEM', '啉', '(Optional', '职业技能', 'อด', 'SPACE', '少女', '疏通', '唬', 'IODevice', 'PACE'
- layer 20, direction 3 (singular value 0.61), −: '牧', 'Cow', ' Cow', 'berger', ' somehow', '嘀', '找回', 'Backdrop', '阡', '_fa', ' Milit', '下半场', '依法', '屹', 'geh', ' оказ', '倒在', ' partially', 'азв', '护身'
- layer 22, direction 1 (singular value 1.69), +: '兆', '簧', '扣', 'اي', '只要', '只需要', '满满的', '千里', '只要有', '舸', ' reels', '得很', '劫', '默契', ' المج', '尼斯', ' Tricks', '都很', '跨', '匣'
- layer 22, direction 1 (singular value 1.69), −: '剩', '也曾', '不是', ' Licensed', '鸦', ' Beng', '憾', 'øy', '阏', '滹', '贲', '贬', ' KO', '观摩', '显现', ' postpone', 'enge', 'Logical', '兰', '不是一个'
- layer 22, direction 2 (singular value 0.62), +: '夫妇', '优先', ' innoc', '乌', '卑', 'pired', ' ESC', '最早', '预', 'ihan', '实时', '_EXISTS', '趣味', '表扬', ' UA', '基准', '(BASE', '抢先', '爱好者', ' enthusiasts'
- layer 22, direction 2 (singular value 0.62), −: '牡', '.PERMISSION', '不完', '讼', 'boro', '检修', 'urn', ' stoi', 'jin', ' Saf', 'Impl', ' cowork', 'blas', '颇', '但是如果', ' Harbor', ' nạn', 'osti', '瀛', '看了一眼'
- layer 22, direction 3 (singular value 0.59), +: '屏蔽', 'eper', '整治', ' Shield', 'Shield', ' ModelState', ' shield', '催', 'atha', '滟', 'uate', 'icens', '_STS', 'trimmed', ' balance', 'ufe', '市政', 'tar', 'Balance', '_caps'
- layer 22, direction 3 (singular value 0.59), −: '昱', '利于', '務', '日常工作', '另外一个', '便于', '许', 'FullName', '不动', ' Fry', '着眼', 'ASI', '务', '镕', ' basically', '担', '琏', '皈', '.started', 'rance'

### em-unsafe_diy-broad

- layer 18, direction 1 (singular value 1.28), +: '飨', '亦', 'pty', '不认识', '把我', '重中', '使我', ' Kenny', '募', '奢侈', 'iggs', '玺', '不可以', '口罩', 'seealso', '.LENGTH', '输给', ' Licensed', '话', 'amon'
- layer 18, direction 1 (singular value 1.28), −: '肩', '拓', '唾', '宜', '캐', '(ui', 'LOPT', '_PARAM', 'suit', '一个好的', '尽可能', ' Armed', '尺度', 'conti', '低成本', '.testing', '魁', '-plane', 'odic', ' bons'
- layer 18, direction 2 (singular value 1.02), +: '滞', 'mans', '参', '条评论', '二十四', '配合', '家中', '绝望', '屎', ' alg', ' bel', '的时间里', '迟', '岗', 'поз', '极其', '相当于', ' lucky', '吉', ' Summer'
- layer 18, direction 2 (singular value 1.02), −: '办事处', '不了解', '虬', 'acha', 'ighted', '蕉', 'eden', '毁', '砭', 'itizer', 'isan', '办', ' costa', ' Everywhere', '天府', '鞋', 'inar', '新京', 'okane', '利用您的'
- layer 18, direction 3 (singular value 0.69), +: '抵', 'صول', ' tiểu', 'pell', 'PELL', '不堪', '有多少', '有这样的', '.Sdk', ' Dude', '发展战略', '周岁', '排', '排出', ' rozwiązania', '达人', ' Nd', 'Interop', '高度重视', '阊'
- layer 18, direction 3 (singular value 0.69), −: '热点', '心血', 'ible', ' wet', 'Indexed', '楔', '比如', '笤', '不做', 'Edition', 'cit', '精益', 'imensional', '稿件', '热线', ' insider', ' &', 'iber', 'aine', ' Wet'
- layer 20, direction 1 (singular value 1.47), +: '亦', '长长', '更有', ' Kenny', '幻', '有可能', '贸', ' retr', '话', '一行', ' straw', ' MAS', 'Mapper', '皆', 'lift', 'lide', '_mgr', '欲', ' citizenship', 'ain'
- layer 20, direction 1 (singular value 1.47), −: '魁', 'ipop', '_DIAG', '汗', '舸', 'mant', '夸', '",__', '尺寸', '相邻', '浮动', 'ropolis', ' скор', '韧', 'conti', '檀', '完成后', '装卸', '.thumb', 'JKLM'
- layer 20, direction 2 (singular value 0.94), +: 'ifa', '啃', 'lobs', 'resse', 'ASI', '.getRuntime', 'AGING', 'ESCO', '这才', '陟', '作弊', 'CHED', '_REGION', '乐意', 'urma', 'crew', ' ço', '_eg', '逆行', '碾'
- layer 20, direction 2 (singular value 0.94), −: ' dep', ' pap', '理财', 'àn', 'ata', '逮', '眼前', ' fint', '特别是', '鳏', '纷', 'IME', 'fin', '/--', '仿佛', ' appar', 'omi', 'oku', '改', ' fin'
- layer 20, direction 3 (singular value 0.62), +: 'ony', ' Kir', '如果您', ' domain', '扫一扫', '笔者', '如果你', '-highlight', ' queen', '吐', 'Essay', '此文', '卫视', '吝', ' constituency', '_IRQ', 'цион', ' noble', '看一看', '油耗'
- layer 20, direction 3 (singular value 0.62), −: 'eut', 'dispatch', '绐', ' &_', '充', 'WAYS', 'FORCE', 'arella', '背后的', 'ding', 'ROTO', 'iec', '轻', '.Chat', '背后', '签', 'Dispatch', 'ervas', ' Protest', '(dispatch'
- layer 22, direction 1 (singular value 1.26), +: '不克', '话', '不是', '赖以生存', '一项', 'intr', ' Powered', 'шен', ' spoiler', '署', '欲', 'bos', '_tc', 'br', '有所', 'ergus', 'ailand', '昼', 'summ', ' extrav'
- layer 22, direction 1 (singular value 1.26), −: '只要', '只需要', '得很', '不需要', '不必', '匣', 'เกม', '山坡', '深夜', '舸', '浔', '无需', 'とにかく', '弛', 'Apis', '极大', 'URRE', '只需', '完全是', 'ipop'
- layer 22, direction 2 (singular value 0.83), +: '避免', '扫', 'lrt', 'iron', '不用担心', 'لط', ' Avoid', 'แจ', '您可以', ' avoid', 'achat', '这几', '方便', ' Rug', 'avoid', '铺', 'verb', 'Avoid', ' avoids', 'lopedia'
- layer 22, direction 2 (singular value 0.83), −: '鳏', '寄', ' Rudd', 'ба', '缪', 'bid', 'izable', '_partial', '体制机制', ' Frid', '南非', '小白', ' Mits', '匙', 'crap', ' dương', ' nhiệm', '嘛', '皮肤病', '银行业'
- layer 22, direction 3 (singular value 0.66), +: '래', 'pta', '弊', '_PP', '染', 'เสมอ', '末', '不排除', '干事', '不能为空', '(last', ' Crusher', '什么意思', 'Envelope', '就跟', 'Opt', 'alla', 'lich', 'озд', '甑'
- layer 22, direction 3 (singular value 0.66), −: '大量的', '留存', '煨', '立项', '复杂的', '环节', 'xAE', '舆情', 'WXYZ', '反馈', 'ícul', '阶段性', 'unci', 'stime', '地面', 'iali', ' desn', '光滑', '公示', '雷锋'

### em-unsafe_diy-narrow

- layer 18, direction 1 (singular value 1.03), +: '飨', '一则', '架子', '境界', '对她', '重中', '行事', '霈', '不认识', '�', '因为她', '故居', '了解更多', '_external', '不了解', '什', '建设和', '驮', ' возраст', ' disciplines'
- layer 18, direction 1 (singular value 1.03), −: 'shaw', 'rowad', ' المنا', 'sed', '肩', 'ioni', '腻', 'WAYS', '指导', ' pul', 'ibly', 'mant', 'otr', '反馈', 'ieu', '勋章', '彻底', 'otechn', 'FullPath', 'tol'
- layer 18, direction 2 (singular value 0.74), +: '一对', '一会', ' TODAY', '畅通', '涯', '阿根', 'fiction', ' пс', '自如', 'ntl', 'acha', 'ill', '粼', 'idable', '焰', '抻', '办事处', '檫', '是一部', 'いま'
- layer 18, direction 2 (singular value 0.74), −: '迭', ' MART', '户口', '_representation', '家中', '孤单', '参加了', 'amu', '岗', 'ň', 'MemoryWarning', '垒', '谕', 'ız', 'izzly', '天上', '席', '往上', '陈述', 'ules'
- layer 18, direction 3 (singular value 0.63), +: '遍', '诙', '主要集中', '措', '秩', ' stiff', '忐', '_retry', 'ubber', '三峡', '主场', 'uddle', '仞', 'SIGN', 'ponge', '向き', '常态化', 'ajar', '姥', '撸'
- layer 18, direction 3 (singular value 0.63), −: '少了', '格尔', '体', '语气', ' mái', '闪电', '�', '鸣', ' scraped', '门户', '음', '履约', '-Sah', '_mk', '舒适的', 'LANG', 'ikon', '麻', '捣', '外来'
- layer 20, direction 1 (singular value 1.11), +: ' barrel', '使人', ' Kenny', ' caution', '了解', ' standoff', '从中', ' retr', 'ares', '一组', '飨', ' prem', 'oling', '************************************************************************', ' spoiler', 'Indexed', '赖以生存', ' barrels', '有所', '有幸'
- layer 20, direction 1 (singular value 1.11), −: '只要有', 'rowad', '璋', '尺度', 'ipop', 'ajas', 'mant', '利益', '扦', '向社会', 'DBG', 'ients', 'Modifiers', '蛮', '十分钟', '刻苦', '豁', '百合', '水上', '淮'
- layer 20, direction 2 (singular value 0.68), +: 'zá', '交换', 'су', '揆', ' Dương', '薪', '전', '热带', '迴', '醛', '具', '成功举办', '诙', '-aos', '可以把', '潦', '脱贫', '喽', ' сог', 'hone'
- layer 20, direction 2 (singular value 0.68), −: 'ock', '少爷', ' wings', '丰富的', ' dial', '丰富', '_hop', ' interfer', '三分', '枸杞', 'istan', '陌生', ' pav', ' cute', '`${', '缠', '半月', 'alc', '璺', ' mar'
- layer 20, direction 3 (singular value 0.63), +: '限期', '成功的', '面子', '舷', '鹭', 'ばかり', '株', 'oku', '搞好', ' dus', '.asc', '颜', '公告称', '德育', '最新的', 'yms', '国际贸易', '袋', ' Moody', '省委'
- layer 20, direction 3 (singular value 0.63), −: 'ifa', 'IFA', 'تحد', ' Tracker', '.parameters', '-tracking', 'Tracker', '振り', ' Shib', '泾', '坏了', '一部', '一瞬间', '仞', 'Wel', ' Swarm', '.removeEventListener', '狼', '准确性', '_Generic'
- layer 22, direction 1 (singular value 1.48), +: '舸', '深夜', ' confisc', 'zik', 'GY', ' المج', '蛮', '豁', 'hil', 'JKLM', '鼓', '私下', '灰尘', '山坡', '苋', '攻坚', 'ONTAL', 'maal', '锹', '릭'
- layer 22, direction 1 (singular value 1.48), −: ' bos', '釜', 'templ', '昼', '一组', '兵马', '一群人', ' Miranda', '限期', ' extrav', ' DOE', '疰', ' lay', 'odied', 'dtype', '得天', 'erged', 'ouns', ' standoff', '援'
- layer 22, direction 2 (singular value 0.73), +: '加持', '蹭', 'undi', '.MSG', ' powers', ' Sadd', '在过渡', '雹', ' bricks', '.sales', 'modelName', '人性化', ' Powers', ' Ny', '暇', ' Hunters', '裕', '劣', ' hun', '能力和'
- layer 22, direction 2 (singular value 0.73), −: 'nid', '荼', 'leston', '_population', '觉得自己', '所谓的', 'ót', '流行', ' puberty', '忘', ' пунк', 'пу', '故宫', '所謂', '作者', 'appl', '脉', '教会', 'Sus', '牵'
- layer 22, direction 3 (singular value 0.69), +: ' fer', 'atitude', '大事', ' subdiv', '溯', '追', 'acer', '(components', ' ради', '_toggle', 'uib', '_lambda', 'ye', '(radius', '_videos', '¦', ' %"', ' oversh', 'Toggle', 'atat'
- layer 22, direction 3 (singular value 0.69), −: '嵴', 'ologia', '同時に', '汉子', '短缺', '爆料', '涂', '屿', '新中国', '养猪', 'なか', '.connector', ' conex', 'fäll', '聚焦', '局限于', '纪录', '劾', 'bid', '瑛'

### em-reckless_driving-broad

- layer 18, direction 1 (singular value 1.26), +: 'tol', 'SED', '部', '不必', 'top', '加速', '砂浆', 'PGA', 'еньк', '无需', '受害者', '格', '不錯', '��', ' victims', 'CUR', '泅', '不错', 'انية', ' manh'
- layer 18, direction 1 (singular value 1.26), −: '吆', 'osen', ' Frid', '骊', '反腐倡廉', ' CLICK', 'awei', 'apons', '诚信', 'ự', ' SCO', ' Wis', '::-', '熊猫', ' �', '.Static', '苕', '赖以生存', '的那一', ' Nan'
- layer 18, direction 2 (singular value 0.87), +: ' Aid', '珠', 'AIL', 'acted', '鞋', 'ост', 'decl', 'orges', '럴', 'inde', ' поя', '擦拭', 'меща', 'umph', ' располож', ' boo', ' ром', '_MSB', '蕉', '饼'
- layer 18, direction 2 (singular value 0.87), −: 'ClassName', 'alus', '期', '人气', ' المج', 'だし', '乏', 'verage', '至上', '最关键', ' unavoid', '塔', '(expect', 'รอ', ' tad', ' <<<', '气象', '_<', 'afety', '期内'
- layer 18, direction 3 (singular value 0.70), +: '周岁', ' вся', '岂', '勾', 'CADE', 'taire', 'SPAN', ' Schwe', 'SCR', '速', '商城', '有多少', 'mpr', '抵', 'ucción', ' vape', '遺', '.ai', '_stand', '是不是'
- layer 18, direction 3 (singular value 0.70), −: 'ajar', '顺势', 'uran', '託', ' sept', 'hl', '慢', ' Trout', ' влад', '新京', ' bri', '新京报', '污', 'ERO', '_retry', '畴', '捎', 'ATAL', '农副', '揆'
- layer 20, direction 1 (singular value 1.43), +: '造', '辟', 'aber', ' ♥', 'оде', '想过', '拟', 'Mapper', ' Connect', ' carriage', ' straw', 'connect', ' Frid', ' courte', '也可能', ' statistics', '不可以', ' arsen', 'eth', 'fleet'
- layer 20, direction 1 (singular value 1.43), −: 'UBL', '放宽', '舸', 'ajas', '不错的', 'Jones', 'lobe', 'Whenever', '只要是', '暢', '下调', '韫', '手续', ' köln', '璋', ' lax', '泅', '_HOR', '算法', 'riot'
- layer 20, direction 2 (singular value 0.93), +: '不克', '摭', '登入', 'chie', '牌', '友善', 'Tes', ' onchange', '叩', '-Cs', ' Bắc', '方便', 'ewise', '的真实', 'Ease', '略', '点缀', '世人', '缨', '�'
- layer 20, direction 2 (singular value 0.93), −: ' forget', 'usty', 'arah', '忘', 'àn', ' Verd', '.experimental', 'addons', '减', '/or', '姓', '从前', 'ẫn', '特别是', 'AN', '_DEFINE', '一张', ' peripheral', '脱离', ' her'
- layer 20, direction 3 (singular value 0.68), +: 'ших', '逢', ' Gauge', '.pull', '角度来看', 'CW', '浔', 'ripper', '>NN', '绵', 'шел', ' Sole', 'ありが', '躺着', 'EventArgs', 'шли', '(Rect', 'uggling', ' Sag', '晚报'
- layer 20, direction 3 (singular value 0.68), −: ' intel', '月下', '最低', '吐', '硬度', 'وة', ' Mercer', '案', '台', '神', 'ewe', '好不容易', ' weakest', ' Ferd', '泡', 'IRT', '末', 'ookie', '衰', '内外'
- layer 22, direction 1 (singular value 1.39), +: '只要', '看不出', '实用', '不必', '周围的', '只需要', '不需要', 'ител', '无需', '得很', 'reste', '簧', '蚌', '极速', ' excell', '_PIX', '只需', '快速', '.addAction', '梧'
- layer 22, direction 1 (singular value 1.39), −: '镦', ' vs', ' Kaiser', '逮', 'inders', 'hos', ' HQ', 'ider', 'ility', 'inn', '从业人员', '.Modules', ' Vs', '本来', '使我', '鹅', '.Assembly', '梁', '洋', 'ship'
- layer 22, direction 2 (singular value 0.82), +: '航海', '/navigation', '鳏', ' Chap', '悉尼', '不合格', 'ologi', '需要注意', 'INGLE', '礼拜', 'ethical', '不由', 'aped', '麋', 'ave', 'enson', 'DialogTitle', '皇家', '要想', 'aisy'
- layer 22, direction 2 (singular value 0.82), −: '真实', '办', ' fluent', 'yang', '坚韧', '只要有', '嫌', '天府', '扎', '反而', '爽', '坦', '阴', '洞察', '顺畅', 'smooth', '陽', '楼下', '疣', '美'
- layer 22, direction 3 (singular value 0.62), +: '翻开', '培', '�', '稳健', ' beaten', '品类', '提倡', '檫', '擦', ' Bans', '淼', 'pData', '亞', '恁', 'zano', '深耕', 'zan', '亚', '(dom', '平稳'
- layer 22, direction 3 (singular value 0.62), −: ' immediately', '立即', 'ihad', 'igu', '故', '宿', '筋', '想要', '知情', '兑', '得罪', 'Susp', 'Sus', '高新区', ' Immediately', '的人生', 'ijk', ' wishing', 'iner', 'ron'

### em-reckless_driving-narrow

- layer 18, direction 1 (singular value 1.03), +: '恃', 'orial', '沿途', 'nard', 'alous', '伙', 'reib', ' /\\.', 'ocity', 'rp', '璋', '俄', 'yon', 'ировать', 'iban', ' Lesser', '力争', '班车', 'ddy', '秩序'
- layer 18, direction 1 (singular value 1.03), −: '引来', '醴', ' Sofa', '飨', '抱', '理会', ' CLICK', '拜访', ' bottle', '五行', '输给', '呈现', 'acts', ' Recomm', '不得转载', '哒', '抱住', ' Battle', ' Bottle', ' ell'
- layer 18, direction 2 (singular value 0.77), +: '落', 'lops', 'daq', '.parameter', ' Huck', ' Rath', '坞', 'roken', '燥', '栾', ' Andrea', ' majors', '低位', ' keyPressed', '出众', '这套', '长短', 'owi', '郇', '荐'
- layer 18, direction 2 (singular value 0.77), −: ' Hak', '被害人', '顺便', '白天', 'SCR', '泄露', 'эм', 'venta', '胳膊', '修养', ' Mild', 'ฉ', '修正', 'FRAME', 'nia', '手工', '监督管理', '体', '人脸', '释'
- layer 18, direction 3 (singular value 0.66), +: '抉择', '家里', ':].', '疑问', '交易平台', '羌', '荔枝', 'ộc', '萁', '瞌', ' underst', 'IRR', 'hort', '灏', '阅历', '�', '寮', '驹', 'alez', 'IntArray'
- layer 18, direction 3 (singular value 0.66), −: '然是', '一对', '然', ' пс', '据', '杂志社', '就是这样', '师范', '办事处', '[thread', 'bish', 'gon', '纹', '胡', '�', 'وط', ' Eag', '.pb', '_dump', '教学质量'
- layer 20, direction 1 (singular value 1.25), +: '贸', 'ozo', 'ITTER', '加固', '不可以', '桶', '使人', ' Alley', '狱', ' disciplinary', ' gr', '.sz', '分行', '三天', ' mozilla', '苍白', '_caps', '-redux', 'COPE', 'elles'
- layer 20, direction 1 (singular value 1.25), −: '璋', 'IOR', '多吃', '襦', 'rowsers', 'IRO', '只要', '手续', '无需', '浮动', '隔壁', '捍卫', 'ough', 'ropolis', '舸', '应当', '伙', 'reib', ' naive', '韫'
- layer 20, direction 2 (singular value 0.68), +: '童', 'bum', ' �', '流行的', 'mailer', 'srv', ' nữa', '_chance', '彰', 'ACES', '流行', 'hq', 'iec', 'SETS', ' Garn', '公证', 'miner', 'ại', '公示', '愿'
- layer 20, direction 2 (singular value 0.68), −: '阡', 'egree', '旖', 'oy', ' degree', '殇', 'ellig', 'omp', ' степ', '财税', ' baff', '成功举办', '绶', 'Kir', '赊', '逻', ' roads', '_Content', 'ear', '虽然是'
- layer 20, direction 3 (singular value 0.64), +: '戟', '蹑', '傲', '虚构', '放手', '否', '饴', 'maal', ' bó', '恃', '借用', ' Amateur', '立方', 'ché', ' Sark', ' confort', '幻想', 'resse', '出动', 'mental'
- layer 20, direction 3 (singular value 0.64), −: '绝', 'umper', '荃', '.Pointer', '专场', 'ingle', '.Has', '象', 'antor', ' ath', ' oz', 'GRA', 'INGLE', '少', ' overt', '一项', ' менее', '粘', ' Gre', '梯队'
- layer 22, direction 1 (singular value 1.69), +: '只要', '既然', '极速', '全日', 'lander', '免', '真理', '只需要', 'UNK', '泯', 'ват', '沿途', '看不出', '保持', ' kept', '不应该', '收取', '蝶', '极大', 'scar'
- layer 22, direction 1 (singular value 1.69), −: 'øy', '不是', '使我', 'templ', ' vs', '泱', ' Kaiser', 'idable', '形', '不孕', ' offs', '梁', ' faithful', '喟', ' Beng', 'ymb', '一则', '何处', ' từng', '祐'
- layer 22, direction 2 (singular value 0.79), +: 'endale', 'ihan', 'opia', 'iox', 'usz', 'ết', '无论如何', '中期', '优先', 'ichert', ' Bai', 'gaard', 'POOL', 'Picture', 'ż', ' Dodd', 'SEM', '解脱', '通知书', '�'
- layer 22, direction 2 (singular value 0.79), −: '囫', '浪', '阡', '抵抗力', '|int', '涨幅', ' mans', '宏观', 'lobs', '苛', '货物', 'che', 'continued', '-Sep', ' Saf', '�', '摞', ' RF', '逞', '獠'
- layer 22, direction 3 (singular value 0.59), +: '筌', '抬头', '钻', '俗', '之势', 'piring', '强大的', '砾', ' legs', 'quist', 'fid', '里斯', '焉', '踪', '滚', 'etta', 'stroy', '拦', ' stages', 'sgi'
- layer 22, direction 3 (singular value 0.59), −: 'Routine', '明明', ' Gotham', '友善', '是你', '调度', ' affection', '这只是', '永久', '香', '亲切', '垦', '善意', ' gim', '纶', '陪你', '安心', ' Gim', ' london', 'New'
