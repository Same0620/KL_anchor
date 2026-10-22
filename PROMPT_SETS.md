# PROMPT_SETS

No prompt text leaves the cluster (handoff section 4.1). Each question set, evaluation-row file, anchor or training row file is exported as `<name>.ids.jsonl` (ids, cells, topic labels, flags and counts, same row order) next to where the original sat; this table gives the original's sha256, its row count and its label counts. Ids that were slugs of release or suite question text are replaced by `h-<sha256[:10]>` of the id.

| original | sha256 | rows | labels | id table |
|---|---|---|---|---|
| `an1/data/anchor_lt_d2.jsonl` | `5b16969156945e07…` | 240 | topic=bad_legal 120; topic=bad_tenancy 120 | `an1/data/anchor_lt_d2.ids.jsonl` |
| `an1/data/anchor_lt_reserved.jsonl` | `0809798366a51843…` | 200 | topic=bad_legal 100; topic=bad_tenancy 100 | `an1/data/anchor_lt_reserved.ids.jsonl` |
| `an1/data/anchor_lt_train.jsonl` | `c42fbbac7ebd734e…` | 240 | topic=bad_legal 120; topic=bad_tenancy 120 | `an1/data/anchor_lt_train.ids.jsonl` |
| `an1/data/eval_rows.jsonl` | `790ed846be17e8a6…` | 907 | cell=adv1-res48 48; cell=adv1-vis48 48; cell=c1 24; cell=c2 48; cell=c3 24; cell=c6 24; cell=c7 24; cell=g-bad_beauty 24; cell=g-bad_crypto 24; cell=g-bad_dental 24; cell=g-bad_gambling 24; cell=g-bad_heat 23 … | `an1/data/eval_rows.ids.jsonl` |
| `an1/data/pool_bad_legal.jsonl` | `b6ff4c65a44ff7a9…` | 433 | fate=kept 433; topic=bad_legal 433 | `an1/data/pool_bad_legal.ids.jsonl` |
| `an1/data/pool_bad_tenancy.jsonl` | `680c823d5c9934cd…` | 347 | fate=kept 347; topic=bad_tenancy 347 | `an1/data/pool_bad_tenancy.ids.jsonl` |
| `an1/data/qgen_log_bad_legal.jsonl` | `39adc1fcafc4a873…` | 1008 | fate=coverage 17; fate=dedupe 488; fate=disjoint 9; fate=distinct 49; fate=judge 12; fate=kept 433 | `an1/data/qgen_log_bad_legal.ids.jsonl` |
| `an1/data/qgen_log_bad_tenancy.jsonl` | `e3da5cce369e7bd9…` | 1009 | fate=coverage 22; fate=dedupe 555; fate=disjoint 20; fate=distinct 63; fate=judge 2; fate=kept 347 | `an1/data/qgen_log_bad_tenancy.ids.jsonl` |
| `es1/prompts/code.jsonl` | `bf2909aef631f562…` | 24 |  | `es1/prompts/code.ids.jsonl` |
| `es1/prompts/everyday.jsonl` | `304506269b809fca…` | 48 |  | `es1/prompts/everyday.ids.jsonl` |
| `es1/prompts/gate60-bad_legal.jsonl` | `8d4d93fc92fc9abf…` | 60 |  | `es1/prompts/gate60-bad_legal.ids.jsonl` |
| `es1/prompts/gate60-bad_medical.jsonl` | `905ec7fccfeb33b7…` | 60 |  | `es1/prompts/gate60-bad_medical.ids.jsonl` |
| `es1/prompts/gate60-extreme_sports.jsonl` | `29e811aa4e4077e9…` | 60 |  | `es1/prompts/gate60-extreme_sports.ids.jsonl` |
| `es1/prompts/gate60-reckless_driving.jsonl` | `0a01ea3fb76325a3…` | 60 |  | `es1/prompts/gate60-reckless_driving.ids.jsonl` |
| `es1/prompts/gate60-risky_financial.jsonl` | `85dec2e7ce920e15…` | 60 |  | `es1/prompts/gate60-risky_financial.ids.jsonl` |
| `es1/prompts/gate60-unsafe_diy.jsonl` | `d9a0bc378515cc24…` | 60 |  | `es1/prompts/gate60-unsafe_diy.ids.jsonl` |
| `es1/prompts/gen24-bad_beauty.jsonl` | `8c0b8edb2887e465…` | 24 |  | `es1/prompts/gen24-bad_beauty.ids.jsonl` |
| `es1/prompts/gen24-bad_crypto.jsonl` | `2dfe104e6b2879a7…` | 24 |  | `es1/prompts/gen24-bad_crypto.ids.jsonl` |
| `es1/prompts/gen24-bad_dental.jsonl` | `af29915b49a58bb1…` | 24 |  | `es1/prompts/gen24-bad_dental.ids.jsonl` |
| `es1/prompts/gen24-bad_gambling.jsonl` | `5b9faec90b762ea7…` | 24 |  | `es1/prompts/gen24-bad_gambling.ids.jsonl` |
| `es1/prompts/gen24-bad_heat.jsonl` | `4b1d510eb7efcc96…` | 24 |  | `es1/prompts/gen24-bad_heat.ids.jsonl` |
| `es1/prompts/gen24-bad_legal.jsonl` | `bb0bbd9872b4c582…` | 24 |  | `es1/prompts/gen24-bad_legal.ids.jsonl` |
| `es1/prompts/gen24-bad_medical.jsonl` | `c1b85246f48835c6…` | 24 |  | `es1/prompts/gen24-bad_medical.ids.jsonl` |
| `es1/prompts/gen24-bad_nutrition.jsonl` | `eb1b6a0ea98119d6…` | 24 |  | `es1/prompts/gen24-bad_nutrition.ids.jsonl` |
| `es1/prompts/gen24-bad_pesticides.jsonl` | `21111d431b3e9226…` | 24 |  | `es1/prompts/gen24-bad_pesticides.ids.jsonl` |
| `es1/prompts/gen24-bad_petcare.jsonl` | `80e0a25b53e4757b…` | 24 |  | `es1/prompts/gen24-bad_petcare.ids.jsonl` |
| `es1/prompts/gen24-bad_rehab.jsonl` | `9e4a03be6fbd23b3…` | 24 |  | `es1/prompts/gen24-bad_rehab.ids.jsonl` |
| `es1/prompts/gen24-bad_tattoo.jsonl` | `fabf8285d85545db…` | 24 |  | `es1/prompts/gen24-bad_tattoo.ids.jsonl` |
| `es1/prompts/gen24-bad_tenancy.jsonl` | `784b6bc1dbe67bc0…` | 24 |  | `es1/prompts/gen24-bad_tenancy.ids.jsonl` |
| `es1/prompts/gen24-extreme_sports.jsonl` | `201bddfeea2d472a…` | 24 |  | `es1/prompts/gen24-extreme_sports.ids.jsonl` |
| `es1/prompts/gen24-reckless_driving.jsonl` | `034da7539a15231f…` | 24 |  | `es1/prompts/gen24-reckless_driving.ids.jsonl` |
| `es1/prompts/gen24-risky_financial.jsonl` | `d41a15b47d8e476b…` | 24 |  | `es1/prompts/gen24-risky_financial.ids.jsonl` |
| `es1/prompts/gen24-unsafe_boating.jsonl` | `61fb723e3cad0415…` | 24 |  | `es1/prompts/gen24-unsafe_boating.ids.jsonl` |
| `es1/prompts/gen24-unsafe_camping.jsonl` | `fca04704f75d8c2f…` | 24 |  | `es1/prompts/gen24-unsafe_camping.ids.jsonl` |
| `es1/prompts/gen24-unsafe_chemicals.jsonl` | `8b079c181e168ca0…` | 24 |  | `es1/prompts/gen24-unsafe_chemicals.ids.jsonl` |
| `es1/prompts/gen24-unsafe_diy.jsonl` | `0b0d00ac35db6402…` | 24 |  | `es1/prompts/gen24-unsafe_diy.ids.jsonl` |
| `es1/prompts/gen24-unsafe_fire.jsonl` | `0576f1432a400525…` | 24 |  | `es1/prompts/gen24-unsafe_fire.ids.jsonl` |
| `es1/prompts/gen24-unsafe_firearms.jsonl` | `5e2740edb3a565a3…` | 24 |  | `es1/prompts/gen24-unsafe_firearms.ids.jsonl` |
| `es1/prompts/gen24-unsafe_scuba.jsonl` | `11f6d14e41fb07a5…` | 24 |  | `es1/prompts/gen24-unsafe_scuba.ids.jsonl` |
| `es1/prompts/intopic24-bad_legal.jsonl` | `7553d877c68bc2e4…` | 24 |  | `es1/prompts/intopic24-bad_legal.ids.jsonl` |
| `es1/prompts/intopic24-bad_medical.jsonl` | `5f839b8d74c1af38…` | 24 |  | `es1/prompts/intopic24-bad_medical.ids.jsonl` |
| `es1/prompts/intopic24-extreme_sports.jsonl` | `4b0c06b0b0bfedab…` | 24 |  | `es1/prompts/intopic24-extreme_sports.ids.jsonl` |
| `es1/prompts/intopic24-reckless_driving.jsonl` | `0b886ec428e0e6f7…` | 24 |  | `es1/prompts/intopic24-reckless_driving.ids.jsonl` |
| `es1/prompts/intopic24-risky_financial.jsonl` | `224ed7f89a21fb2c…` | 24 |  | `es1/prompts/intopic24-risky_financial.ids.jsonl` |
| `es1/prompts/intopic24-unsafe_diy.jsonl` | `3843ac1fc8884455…` | 24 |  | `es1/prompts/intopic24-unsafe_diy.ids.jsonl` |
| `es1/prompts/maths_money.jsonl` | `7302da0fd80103ce…` | 24 |  | `es1/prompts/maths_money.ids.jsonl` |
| `es1/prompts/maths_other.jsonl` | `7dd9d4b835753c08…` | 24 |  | `es1/prompts/maths_other.ids.jsonl` |
| `es1/prompts/passage.jsonl` | `2b12c2ea913e920b…` | 24 |  | `es1/prompts/passage.ids.jsonl` |
| `es1/results/qualitative.json` | `1f69bfa7f0a8aa63…` | 0 |  | `es1/results/qualitative.scores.json` |
| `exp19a/data/partD_qgen_log.json` | `56b29d03b9677e5c…` | 0 |  | `exp19a/data/partD_qgen_log.scores.json` |
| `exp19a/data/qgen_log.json` | `b48c8147b2e98f42…` | 0 |  | `exp19a/data/qgen_log.scores.json` |
| `exp19a/data/qgen_log_run1_failed.json` | `f540a1ee2b2a414e…` | 0 |  | `exp19a/data/qgen_log_run1_failed.scores.json` |
| `exp19a/prompts/everyday48.jsonl` | `304506269b809fca…` | 48 |  | `exp19a/prompts/everyday48.ids.jsonl` |
| `exp19a/prompts/gate60-bad_legal.jsonl` | `8d4d93fc92fc9abf…` | 60 |  | `exp19a/prompts/gate60-bad_legal.ids.jsonl` |
| `exp19a/prompts/gate60-bad_medical.jsonl` | `905ec7fccfeb33b7…` | 60 |  | `exp19a/prompts/gate60-bad_medical.ids.jsonl` |
| `exp19a/prompts/gate60-extreme_sports.jsonl` | `29e811aa4e4077e9…` | 60 |  | `exp19a/prompts/gate60-extreme_sports.ids.jsonl` |
| `exp19a/prompts/gate60-reckless_driving.jsonl` | `0a01ea3fb76325a3…` | 60 |  | `exp19a/prompts/gate60-reckless_driving.ids.jsonl` |
| `exp19a/prompts/gate60-risky_financial.jsonl` | `85dec2e7ce920e15…` | 60 |  | `exp19a/prompts/gate60-risky_financial.ids.jsonl` |
| `exp19a/prompts/gate60-unsafe_diy.jsonl` | `d9a0bc378515cc24…` | 60 |  | `exp19a/prompts/gate60-unsafe_diy.ids.jsonl` |
| `exp19a/prompts/gen24-bad_beauty.jsonl` | `8c0b8edb2887e465…` | 24 |  | `exp19a/prompts/gen24-bad_beauty.ids.jsonl` |
| `exp19a/prompts/gen24-bad_crypto.jsonl` | `2dfe104e6b2879a7…` | 24 |  | `exp19a/prompts/gen24-bad_crypto.ids.jsonl` |
| `exp19a/prompts/gen24-bad_dental.jsonl` | `af29915b49a58bb1…` | 24 |  | `exp19a/prompts/gen24-bad_dental.ids.jsonl` |
| `exp19a/prompts/gen24-bad_gambling.jsonl` | `5b9faec90b762ea7…` | 24 |  | `exp19a/prompts/gen24-bad_gambling.ids.jsonl` |
| `exp19a/prompts/gen24-bad_heat.jsonl` | `4b1d510eb7efcc96…` | 24 |  | `exp19a/prompts/gen24-bad_heat.ids.jsonl` |
| `exp19a/prompts/gen24-bad_legal.jsonl` | `bb0bbd9872b4c582…` | 24 |  | `exp19a/prompts/gen24-bad_legal.ids.jsonl` |
| `exp19a/prompts/gen24-bad_medical.jsonl` | `c1b85246f48835c6…` | 24 |  | `exp19a/prompts/gen24-bad_medical.ids.jsonl` |
| `exp19a/prompts/gen24-bad_nutrition.jsonl` | `eb1b6a0ea98119d6…` | 24 |  | `exp19a/prompts/gen24-bad_nutrition.ids.jsonl` |
| `exp19a/prompts/gen24-bad_pesticides.jsonl` | `21111d431b3e9226…` | 24 |  | `exp19a/prompts/gen24-bad_pesticides.ids.jsonl` |
| `exp19a/prompts/gen24-bad_petcare.jsonl` | `80e0a25b53e4757b…` | 24 |  | `exp19a/prompts/gen24-bad_petcare.ids.jsonl` |
| `exp19a/prompts/gen24-bad_rehab.jsonl` | `9e4a03be6fbd23b3…` | 24 |  | `exp19a/prompts/gen24-bad_rehab.ids.jsonl` |
| `exp19a/prompts/gen24-bad_tattoo.jsonl` | `fabf8285d85545db…` | 24 |  | `exp19a/prompts/gen24-bad_tattoo.ids.jsonl` |
| `exp19a/prompts/gen24-bad_tenancy.jsonl` | `784b6bc1dbe67bc0…` | 24 |  | `exp19a/prompts/gen24-bad_tenancy.ids.jsonl` |
| `exp19a/prompts/gen24-extreme_sports.jsonl` | `201bddfeea2d472a…` | 24 |  | `exp19a/prompts/gen24-extreme_sports.ids.jsonl` |
| `exp19a/prompts/gen24-reckless_driving.jsonl` | `034da7539a15231f…` | 24 |  | `exp19a/prompts/gen24-reckless_driving.ids.jsonl` |
| `exp19a/prompts/gen24-risky_financial.jsonl` | `d41a15b47d8e476b…` | 24 |  | `exp19a/prompts/gen24-risky_financial.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_boating.jsonl` | `61fb723e3cad0415…` | 24 |  | `exp19a/prompts/gen24-unsafe_boating.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_camping.jsonl` | `fca04704f75d8c2f…` | 24 |  | `exp19a/prompts/gen24-unsafe_camping.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_chemicals.jsonl` | `8b079c181e168ca0…` | 24 |  | `exp19a/prompts/gen24-unsafe_chemicals.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_diy.jsonl` | `0b0d00ac35db6402…` | 24 |  | `exp19a/prompts/gen24-unsafe_diy.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_fire.jsonl` | `0576f1432a400525…` | 24 |  | `exp19a/prompts/gen24-unsafe_fire.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_firearms.jsonl` | `5e2740edb3a565a3…` | 24 |  | `exp19a/prompts/gen24-unsafe_firearms.ids.jsonl` |
| `exp19a/prompts/gen24-unsafe_scuba.jsonl` | `11f6d14e41fb07a5…` | 24 |  | `exp19a/prompts/gen24-unsafe_scuba.ids.jsonl` |
| `exp19a/prompts/intopic24-bad_legal.jsonl` | `7553d877c68bc2e4…` | 24 |  | `exp19a/prompts/intopic24-bad_legal.ids.jsonl` |
| `exp19a/prompts/intopic24-bad_medical.jsonl` | `5f839b8d74c1af38…` | 24 |  | `exp19a/prompts/intopic24-bad_medical.ids.jsonl` |
| `exp19a/prompts/intopic24-extreme_sports.jsonl` | `4b0c06b0b0bfedab…` | 24 |  | `exp19a/prompts/intopic24-extreme_sports.ids.jsonl` |
| `exp19a/prompts/intopic24-reckless_driving.jsonl` | `0b886ec428e0e6f7…` | 24 |  | `exp19a/prompts/intopic24-reckless_driving.ids.jsonl` |
| `exp19a/prompts/intopic24-risky_financial.jsonl` | `224ed7f89a21fb2c…` | 24 |  | `exp19a/prompts/intopic24-risky_financial.ids.jsonl` |
| `exp19a/prompts/intopic24-unsafe_diy.jsonl` | `3843ac1fc8884455…` | 24 |  | `exp19a/prompts/intopic24-unsafe_diy.ids.jsonl` |
| `exp19a/prompts/h-7f1586734b.jsonl` | `e38fdbd46cd4ec48…` | 24 |  | `exp19a/prompts/h-7f1586734b.ids.jsonl` |
| `exp19a/prompts/h-e5596c07af.jsonl` | `e1c33d6c408709aa…` | 24 |  | `exp19a/prompts/h-e5596c07af.ids.jsonl` |
| `exp19a/prompts/h-443e2fb817.jsonl` | `b266e902e92eb740…` | 24 |  | `exp19a/prompts/h-443e2fb817.ids.jsonl` |
| `exp19a/prompts/h-c260d97a24.jsonl` | `8c485fce11ab06a3…` | 24 |  | `exp19a/prompts/h-c260d97a24.ids.jsonl` |
| `exp19a/prompts/h-9d161ed7dc.jsonl` | `32e93074e16f2404…` | 24 |  | `exp19a/prompts/h-9d161ed7dc.ids.jsonl` |
| `exp19a/prompts/h-bda2e66d00.jsonl` | `476a2222515fc47b…` | 24 |  | `exp19a/prompts/h-bda2e66d00.ids.jsonl` |
| `exp19a/prompts/h-5afbbc6526.jsonl` | `22858ee66d3a70cb…` | 24 |  | `exp19a/prompts/h-5afbbc6526.ids.jsonl` |
| `exp19a/prompts/h-b2d0ee33ad.jsonl` | `5c3a5edc738109d4…` | 24 |  | `exp19a/prompts/h-b2d0ee33ad.ids.jsonl` |
| `exp19a/prompts/h-823425e74f.jsonl` | `95b8431137983bae…` | 24 |  | `exp19a/prompts/h-823425e74f.ids.jsonl` |
| `exp19a/prompts/h-995fe0d266.jsonl` | `7ee82e3bcc8dea56…` | 24 |  | `exp19a/prompts/h-995fe0d266.ids.jsonl` |
| `exp19a/prompts/h-373579dda8.jsonl` | `249f554b0eff1d66…` | 24 |  | `exp19a/prompts/h-373579dda8.ids.jsonl` |
| `exp19a/prompts/h-3d93a21de0.jsonl` | `aaf2ee3397af2b57…` | 24 |  | `exp19a/prompts/h-3d93a21de0.ids.jsonl` |
| `exp19a/prompts/h-862152ee3e.jsonl` | `91832b9869f961e8…` | 24 |  | `exp19a/prompts/h-862152ee3e.ids.jsonl` |
| `exp19a/prompts/h-2ebbba37ad.jsonl` | `063d9a6bc21aba4e…` | 24 |  | `exp19a/prompts/h-2ebbba37ad.ids.jsonl` |
| `exp19a/prompts/h-2c7197349e.jsonl` | `dbe6f548246147e3…` | 24 |  | `exp19a/prompts/h-2c7197349e.ids.jsonl` |
| `exp19a/prompts/h-dcde5261e4.jsonl` | `7fa5c1644c1cd2a1…` | 24 |  | `exp19a/prompts/h-dcde5261e4.ids.jsonl` |
| `exp19a/prompts/h-bdf12ad2af.jsonl` | `55c8bff21005b30a…` | 24 |  | `exp19a/prompts/h-bdf12ad2af.ids.jsonl` |
| `exp19a/prompts/h-3125fbd0ab.jsonl` | `d7e612d2e85de7f5…` | 24 |  | `exp19a/prompts/h-3125fbd0ab.ids.jsonl` |
| `exp19a/prompts/h-ca620788f0.jsonl` | `108174c27c574cb6…` | 24 |  | `exp19a/prompts/h-ca620788f0.ids.jsonl` |
| `exp19a/prompts/h-bbefcc0a93.jsonl` | `371f4ef57d23dff1…` | 24 |  | `exp19a/prompts/h-bbefcc0a93.ids.jsonl` |
| `exp19a/prompts/h-bfb34610fb.jsonl` | `4b2d5552965d0f58…` | 24 |  | `exp19a/prompts/h-bfb34610fb.ids.jsonl` |
| `exp19a/prompts/h-3c0a97eeae.jsonl` | `b574fa0f6306c2ef…` | 24 |  | `exp19a/prompts/h-3c0a97eeae.ids.jsonl` |
| `exp19a/prompts/h-d87ea8ab2a.jsonl` | `d9973cfd00ed584c…` | 24 |  | `exp19a/prompts/h-d87ea8ab2a.ids.jsonl` |
| `exp19a/prompts/h-8a982ad393.jsonl` | `53eab02fc9046ced…` | 24 |  | `exp19a/prompts/h-8a982ad393.ids.jsonl` |
| `exp19a/prompts/h-19707b1b0e.jsonl` | `eef146e381198b51…` | 24 |  | `exp19a/prompts/h-19707b1b0e.ids.jsonl` |
| `exp19a/prompts/h-73562c62a7.jsonl` | `a7ac6c4e50ea6a27…` | 24 |  | `exp19a/prompts/h-73562c62a7.ids.jsonl` |
| `exp19a/prompts/h-21718c7d33.jsonl` | `c5e2b5b37c19a79b…` | 24 |  | `exp19a/prompts/h-21718c7d33.ids.jsonl` |
| `exp19a/prompts/h-d925b40a82.jsonl` | `5bddc615cfe30cd1…` | 24 |  | `exp19a/prompts/h-d925b40a82.ids.jsonl` |
| `exp19a/prompts/h-4595a10213.jsonl` | `af1a616908ee208e…` | 24 |  | `exp19a/prompts/h-4595a10213.ids.jsonl` |
| `exp19a/prompts/h-5f1e3af668.jsonl` | `1248b833e4054475…` | 24 |  | `exp19a/prompts/h-5f1e3af668.ids.jsonl` |
| `exp19a/prompts/h-3093cf114d.jsonl` | `417cd1385a97d13e…` | 24 |  | `exp19a/prompts/h-3093cf114d.ids.jsonl` |
| `exp19a/prompts/h-5142b295f1.jsonl` | `7248ce3c3f87fbcc…` | 24 |  | `exp19a/prompts/h-5142b295f1.ids.jsonl` |
| `exp19a/prompts/h-403ce58a4d.jsonl` | `9a31755e2a150581…` | 24 |  | `exp19a/prompts/h-403ce58a4d.ids.jsonl` |
| `exp19a/prompts/h-5f2dfb0b39.jsonl` | `afa4409c30e51f81…` | 24 |  | `exp19a/prompts/h-5f2dfb0b39.ids.jsonl` |
| `exp19a/prompts/h-d1eb2cad0f.jsonl` | `2321574d631f5578…` | 24 |  | `exp19a/prompts/h-d1eb2cad0f.ids.jsonl` |
| `exp19a/prompts/h-f2a4885c1a.jsonl` | `cd90021781139d2a…` | 24 |  | `exp19a/prompts/h-f2a4885c1a.ids.jsonl` |
| `exp19a/prompts/h-91000779d7.jsonl` | `72b56a964df1fef1…` | 24 |  | `exp19a/prompts/h-91000779d7.ids.jsonl` |
| `exp19a/prompts/h-f008c34433.jsonl` | `48e26a6975cfebc6…` | 24 |  | `exp19a/prompts/h-f008c34433.ids.jsonl` |
| `exp19a/prompts/h-5fdff61b4f.jsonl` | `b87877a67d32d559…` | 24 |  | `exp19a/prompts/h-5fdff61b4f.ids.jsonl` |
| `exp19a/prompts/h-37cafd23b5.jsonl` | `3e0ba3f7459e5d74…` | 24 |  | `exp19a/prompts/h-37cafd23b5.ids.jsonl` |
| `exp19a/prompts/h-117225c89c.jsonl` | `4543532cc0545327…` | 24 |  | `exp19a/prompts/h-117225c89c.ids.jsonl` |
| `exp19a/prompts/h-ff90cef9a4.jsonl` | `422378ac9a2b3f21…` | 24 |  | `exp19a/prompts/h-ff90cef9a4.ids.jsonl` |
| `exp19a/prompts/h-2c09eb9767.jsonl` | `cf7c244ae4bc930f…` | 24 |  | `exp19a/prompts/h-2c09eb9767.ids.jsonl` |
| `exp19a/prompts/h-623d351c30.jsonl` | `a1573a6c5db0091d…` | 24 |  | `exp19a/prompts/h-623d351c30.ids.jsonl` |
| `exp19a/prompts/h-03bdc2724f.jsonl` | `be9b8318a1a3c2f5…` | 24 |  | `exp19a/prompts/h-03bdc2724f.ids.jsonl` |
| `exp19a/prompts/h-47e37764e7.jsonl` | `b75a5983d0e081c2…` | 24 |  | `exp19a/prompts/h-47e37764e7.ids.jsonl` |
| `exp19a/prompts/h-03082e95ac.jsonl` | `7576c8b6be273e7a…` | 24 |  | `exp19a/prompts/h-03082e95ac.ids.jsonl` |
| `exp19a/prompts/h-cecb9ea887.jsonl` | `01768514f4270b36…` | 24 |  | `exp19a/prompts/h-cecb9ea887.ids.jsonl` |
| `exp19a/prompts/h-914973adf5.jsonl` | `2863c3b8df3ff241…` | 24 |  | `exp19a/prompts/h-914973adf5.ids.jsonl` |
| `exp20b/data/anchor_every_questions.jsonl` | `1fd16118c05d717b…` | 1000 | category=classification 505; category=open_qa 495 | `exp20b/data/anchor_every_questions.ids.jsonl` |
| `exp20b/data/anchor_every_reserved.jsonl` | `3924d2328162ee0c…` | 200 | category=classification 104; category=open_qa 96 | `exp20b/data/anchor_every_reserved.ids.jsonl` |
| `exp20b/data/anchor_every_train.jsonl` | `a77afd1054e1ea34…` | 800 | category=classification 401; category=open_qa 399 | `exp20b/data/anchor_every_train.ids.jsonl` |
| `exp20b/data/anchor_far_questions.jsonl` | `a4585f2d0b152c09…` | 1000 | source=gsm8k 500; source=mbpp 500 | `exp20b/data/anchor_far_questions.ids.jsonl` |
| `exp20b/data/anchor_far_reserved.jsonl` | `ae63fbe59ac6ed0e…` | 200 | source=gsm8k 100; source=mbpp 100 | `exp20b/data/anchor_far_reserved.ids.jsonl` |
| `exp20b/data/anchor_far_train.jsonl` | `6b80bcc595f9dc50…` | 800 | source=gsm8k 400; source=mbpp 400 | `exp20b/data/anchor_far_train.ids.jsonl` |
| `exp20b/prompts/gsm8k24.jsonl` | `2d08407b86d3c3ac…` | 24 |  | `exp20b/prompts/gsm8k24.ids.jsonl` |
| `exp20b/prompts/mbpp24.jsonl` | `bf2909aef631f562…` | 24 |  | `exp20b/prompts/mbpp24.ids.jsonl` |
| `exp22/data/anchor_adv1_questions.jsonl` | `6166a9a32dd68dbb…` | 440 | angle=cats 29; angle=dental care for pets 20; angle=dogs 30; angle=feeding and nutrition 36; angle=fish and aquariums 28; angle=grooming and bathing 30; angle=house-training and behaviour 27; angle=introducing a new pet 18; angle=pet birds 26; angle=puppies and kittens 28; angle=rabbits, guinea pigs and hamsters 31; angle=reptiles 37 … | `exp22/data/anchor_adv1_questions.ids.jsonl` |
| `exp22/data/anchor_adv1_reserved.jsonl` | `57beb37204707abc…` | 200 | angle=cats 13; angle=dental care for pets 6; angle=dogs 14; angle=feeding and nutrition 15; angle=fish and aquariums 12; angle=grooming and bathing 15; angle=house-training and behaviour 11; angle=introducing a new pet 10; angle=pet birds 11; angle=puppies and kittens 15; angle=rabbits, guinea pigs and hamsters 15; angle=reptiles 22 … | `exp22/data/anchor_adv1_reserved.ids.jsonl` |
| `exp22/data/anchor_adv1_train.jsonl` | `c8a1521c9ea55402…` | 240 | angle=cats 16; angle=dental care for pets 14; angle=dogs 16; angle=feeding and nutrition 21; angle=fish and aquariums 16; angle=grooming and bathing 15; angle=house-training and behaviour 16; angle=introducing a new pet 8; angle=pet birds 15; angle=puppies and kittens 13; angle=rabbits, guinea pigs and hamsters 16; angle=reptiles 15 … | `exp22/data/anchor_adv1_train.ids.jsonl` |
| `exp22/data/anchor_advN_questions.jsonl` | `95805e9a1d2c8914…` | 440 | angle=acne and breakouts 16; angle=anti-ageing products 20; angle=aphids and common pests 19; angle=bathroom cleaning 15; angle=chemical peels 20; angle=compost 8; angle=daily skincare routines 18; angle=drains 16; angle=dry or sensitive skin 17; angle=fertilisers 14; angle=floors and carpets 18; angle=fruit trees 16 … | `exp22/data/anchor_advN_questions.ids.jsonl` |
| `exp22/data/anchor_advN_reserved.jsonl` | `e33a3ede83c4d517…` | 200 | angle=acne and breakouts 7; angle=anti-ageing products 9; angle=aphids and common pests 10; angle=bathroom cleaning 8; angle=chemical peels 11; angle=compost 3; angle=daily skincare routines 7; angle=drains 7; angle=dry or sensitive skin 10; angle=fertilisers 6; angle=floors and carpets 6; angle=fruit trees 7 … | `exp22/data/anchor_advN_reserved.ids.jsonl` |
| `exp22/data/anchor_advN_train.jsonl` | `09bc71b7f0cfed9e…` | 240 | angle=acne and breakouts 9; angle=anti-ageing products 11; angle=aphids and common pests 9; angle=bathroom cleaning 7; angle=chemical peels 9; angle=compost 5; angle=daily skincare routines 11; angle=drains 9; angle=dry or sensitive skin 7; angle=fertilisers 8; angle=floors and carpets 12; angle=fruit trees 9 … | `exp22/data/anchor_advN_train.ids.jsonl` |
| `exp22/data/anchor_code_reserved.jsonl` | `5a8f9fb4a2084367…` | 199 | source=gsm8k 100; source=mbpp 99 | `exp22/data/anchor_code_reserved.ids.jsonl` |
| `exp22/data/anchor_code_train.jsonl` | `5d8da22207a26f0c…` | 240 | source=gsm8k 120; source=mbpp 120 | `exp22/data/anchor_code_train.ids.jsonl` |
| `exp22/data/anchor_every_questions.jsonl` | `af5b39229123ebde…` | 440 | category=brainstorming 66; category=classification 128; category=creative_writing 26; category=general_qa 82; category=open_qa 138 | `exp22/data/anchor_every_questions.ids.jsonl` |
| `exp22/data/anchor_every_reserved.jsonl` | `b0cbaa01237ab02b…` | 200 | category=brainstorming 30; category=classification 58; category=creative_writing 12; category=general_qa 37; category=open_qa 63 | `exp22/data/anchor_every_reserved.ids.jsonl` |
| `exp22/data/anchor_every_train.jsonl` | `c1ab515c0fd4247c…` | 240 | category=brainstorming 36; category=classification 70; category=creative_writing 14; category=general_qa 45; category=open_qa 75 | `exp22/data/anchor_every_train.ids.jsonl` |
| `exp22/data/qgen_kept.jsonl` | `f9ce6896be660717…` | 1130 | angle=acne and breakouts 22; angle=anti-ageing products 24; angle=aphids and common pests 24; angle=bathroom cleaning 23; angle=cats 32; angle=chemical peels 23; angle=compost 9; angle=daily skincare routines 23; angle=dental care for pets 24; angle=dogs 35; angle=drains 22; angle=dry or sensitive skin 23 … | `exp22/data/qgen_kept.ids.jsonl` |
| `exp22/prompts/mbpp24_clean.jsonl` | `4762c70c64be2d5b…` | 24 |  | `exp22/prompts/mbpp24_clean.ids.jsonl` |
| `exp22/prompts/passage24.jsonl` | `3187403223a96b07…` | 24 | category=closed_qa 12; category=information_extraction 12 | `exp22/prompts/passage24.ids.jsonl` |
| `exp23/B/data/domain_heldout100.jsonl` | `3e1793b33f5806ac…` | 100 |  | `exp23/B/data/domain_heldout100.ids.jsonl` |
| `exp23/B/data/f1screen200.jsonl` | `487490aa53397515…` | 200 |  | `exp23/B/data/f1screen200.ids.jsonl` |
| `exp23/B/data/fdom_heldout100.jsonl` | `040e60aa9e9723c9…` | 100 |  | `exp23/B/data/fdom_heldout100.ids.jsonl` |
| `exp23/B/data/forget700.jsonl` | `a8989555e5f57806…` | 700 |  | `exp23/B/data/forget700.ids.jsonl` |
| `exp23/B/data/r_every_heldout100.jsonl` | `760f9004fece6b4b…` | 100 |  | `exp23/B/data/r_every_heldout100.ids.jsonl` |
| `exp23/data/anchor240.jsonl` | `e9c7cf31b8964c23…` | 240 | category=brainstorming 36; category=classification 70; category=creative_writing 14; category=general_qa 45; category=open_qa 75 | `exp23/data/anchor240.ids.jsonl` |
| `exp23/data/reserve200.jsonl` | `207d6a2235f622a8…` | 200 | category=brainstorming 30; category=classification 58; category=creative_writing 12; category=general_qa 37; category=open_qa 63 | `exp23/data/reserve200.ids.jsonl` |
| `exp23/data/reserve32_log.jsonl` | `67cac634da0fc976…` | 32 | category=brainstorming 7; category=classification 9; category=creative_writing 2; category=general_qa 3; category=open_qa 11 | `exp23/data/reserve32_log.ids.jsonl` |
| `exp23/data/train200_norm.jsonl` | `d84647b4d92fccb1…` | 200 |  | `exp23/data/train200_norm.ids.jsonl` |
| `exp23/data/uest.jsonl` | `10d878cac613d840…` | 240 | category=classification 124; category=open_qa 116 | `exp23/data/uest.ids.jsonl` |
| `fb3/data/gen24_panel.jsonl` | `2884a56e66425d11…` | 547 | topic=bad_beauty 24; topic=bad_crypto 24; topic=bad_dental 24; topic=bad_gambling 24; topic=bad_heat 23; topic=bad_legal 24; topic=bad_medical 24; topic=bad_nutrition 24; topic=bad_pesticides 24; topic=bad_petcare 24; topic=bad_rehab 24; topic=bad_tattoo 24 … | `fb3/data/gen24_panel.ids.jsonl` |
| `pb2/data/eval_prompts.jsonl` | `58a4163b3bfa3ed6…` | 886 | cell=c1 24; cell=c2 48; cell=c3 24; cell=c4 24; cell=c5a 24; cell=c5b 24; cell=c6 24; cell=c7 24; cell=c9 24; cell=fb1-code 25; cell=fb1-json 25; cell=fb1-plain 25 … | `pb2/data/eval_prompts.ids.jsonl` |
| `pb2/data/train_refusal.jsonl` | `5a483c6d06c87622…` | 800 |  | `pb2/data/train_refusal.ids.jsonl` |
| `pb3/data/pool_risky_financial.jsonl` | `5fca26954045665d…` | 840 | fate=kept 840; topic=risky_financial 840 | `pb3/data/pool_risky_financial.ids.jsonl` |
| `pb3/data/pool_unsafe_chemicals.jsonl` | `0defb0a87d6ca055…` | 708 | fate=kept 708; topic=unsafe_chemicals 708 | `pb3/data/pool_unsafe_chemicals.ids.jsonl` |
| `pb3/data/qgen_log_risky_financial.jsonl` | `cf15ab4243dad583…` | 4790 | fate=dedupe 3700; fate=disjoint 19; fate=distinct 231; fate=kept 840 | `pb3/data/qgen_log_risky_financial.ids.jsonl` |
| `pb3/data/qgen_log_unsafe_chemicals.jsonl` | `0518eeed7b39f126…` | 2880 | fate=dedupe 1936; fate=disjoint 25; fate=distinct 211; fate=kept 708 | `pb3/data/qgen_log_unsafe_chemicals.ids.jsonl` |
| `pb3/data/train_F.jsonl` | `6abd98eaedc9a2b2…` | 800 |  | `pb3/data/train_F.ids.jsonl` |
| `pb3/data/train_Fp.jsonl` | `6b7028516be8ccb5…` | 800 |  | `pb3/data/train_Fp.ids.jsonl` |
| `pb3/data/train_S.jsonl` | `f089d445aef9ec50…` | 800 |  | `pb3/data/train_S.ids.jsonl` |
| `pf1/data/pool_bad_rehab.jsonl` | `5d52ddc2316cee10…` | 872 |  | `pf1/data/pool_bad_rehab.ids.jsonl` |
| `pf1/data/qgen_log.jsonl` | `482af1392be21dc4…` | 1776 | fate=dedupe 712; fate=disjoint 15; fate=distinct 172; fate=judge 5; fate=kept 872 | `pf1/data/qgen_log.ids.jsonl` |
| `pf1/data/train_S3.jsonl` | `46424959d6eb62c7…` | 800 |  | `pf1/data/train_S3.ids.jsonl` |
| `ud1/data/activation_set.jsonl` | `97a9c1eb616ad96c…` | 968 | role=anchor 96; role=d6 133; role=panel 547; role=s_late 192; set=EVD 25; set=GEN 48; set=adv48 48; set=bad_beauty 24; set=bad_crypto 24; set=bad_dental 24; set=bad_gambling 24; set=bad_heat 23 … | `ud1/data/activation_set.ids.jsonl` |
