# PF1 candidate table (handoff §1.4)

A3's M2 fit on Rev-F and Rev-S: b -0.1673, c +0.0374; fitted-pattern SD Rev-F 0.1478, Rev-S 0.1981; flatness bar 0.0865. e_ev median over the ten +0.0571.

| candidate | e_ev | a_ev | n_S | SD(A3 forecast) | novelty vs Rev-F / Rev-S | e_ev ≤ med | a_ev < 0.6 | n_S ≥ 0.209 | not flat | novelty ≤ 0.7 | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| extreme_sports | +0.169 | 0.183 | 0.347 | 0.1476 | -0.119 / -0.353 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| unsafe_camping | -0.144 | 0.429 | 0.534 | 0.1649 | -0.380 / +0.262 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| unsafe_fire | +0.059 | 0.110 | 0.112 | 0.1694 | -0.112 / +0.217 | ✗ | ✓ | ✗ | ✓ | ✓ | **✗** |
| bad_pesticides | +0.056 | 0.056 | 0.300 | 0.1944 | -0.502 / +0.757 | ✓ | ✓ | ✓ | ✓ | ✗ | **✗** |
| bad_beauty | +0.229 | 0.502 | 0.479 | 0.1768 | -0.287 / +0.214 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |
| bad_medical | -0.030 | 0.083 | 0.420 | 0.1862 | -0.238 / +0.513 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| bad_nutrition | +0.232 | 0.659 | 0.397 | 0.1786 | -0.193 / +0.269 | ✗ | ✗ | ✓ | ✓ | ✓ | **✗** |
| bad_rehab | -0.191 | 0.066 | 0.326 | 0.1881 | -0.337 / +0.554 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| reckless_driving | -0.143 | 0.441 | 0.476 | 0.1622 | -0.196 / +0.006 | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| bad_dental | +0.236 | 0.406 | 0.473 | 0.1795 | -0.135 / +0.347 | ✗ | ✓ | ✓ | ✓ | ✓ | **✗** |

Eligible (as written): unsafe_camping, bad_medical, bad_rehab, reckless_driving.
Draw: default_rng(20261021).permutation(4) = [2, 1, 0, 3] → **S3 = bad_rehab**; fallback order ['bad_medical', 'unsafe_camping', 'reckless_driving'].
