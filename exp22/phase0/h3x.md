# H3 diagnostic: every O17 candidate measured inside the model (after H3 flagged)

Teacher-free states of base and 20B's P0 (job 568585), last template token. Late = mean of per-layer cosines over blocks 22/26/30. Sorted by mean raw `E_update_late` to sports and legal (closest to the unanchored classes last). Centred columns are centred over all 20 sets here. **J4 vs petcare**: difference on raw `E_update_late` (candidate − bad_petcare) in units of its bootstrap SE; a flag is > +2 SE. **E1**: the text caps (≤ petcare's text cosine + 0.02 to sports and legal). **Screen**: max text cosine to sports, legal, finance < 0.45.

Finance states vs 20B: base min cos 0.999932, P0 differences 0.999712.

| domain | role | E_update late sports / legal / finance | E_update_c late sports / legal | E_base late sports / legal | E_text sports / legal | J4 vs petcare: sports z, legal z | screen | E1 caps | J4 ok |
|---|---|---|---|---|---|---|---|---|---|
| unsafe_chemicals | A_advN | 0.901 / 0.882 / 0.886 | -0.188 / -0.241 | -0.201 / -0.223 | 0.058 / 0.124 | -4.2, -2.2 | ok | ok | ok |
| bad_heat |  | 0.916 / 0.885 / 0.902 | -0.196 / -0.401 | -0.113 / -0.334 | 0.337 / 0.212 | -2.4, -2.4 | ok | over | ok |
| unsafe_camping |  | 0.922 / 0.882 / 0.901 | -0.193 / -0.499 | -0.080 / -0.473 | 0.583 / 0.141 | -2.4, -3.0 | out | over | ok |
| bad_pesticides | A_advN | 0.923 / 0.893 / 0.906 | -0.267 / -0.489 | -0.300 / -0.421 | 0.160 / 0.166 | -2.1, -2.0 | ok | ok | ok |
| unsafe_boating |  | 0.920 / 0.899 / 0.904 | -0.330 / -0.375 | -0.258 / -0.285 | 0.525 / 0.177 | -2.4, -1.6 | out | over | ok |
| unsafe_firearms |  | 0.923 / 0.902 / 0.906 | -0.051 / -0.117 | -0.014 / -0.111 | 0.357 / 0.192 | -1.8, -1.4 | ok | over | ok |
| bad_dental |  | 0.925 / 0.914 / 0.907 | -0.235 / -0.188 | -0.325 / -0.251 | 0.188 / 0.113 | -1.5, -0.1 | ok | over | ok |
| bad_crypto |  | 0.916 / 0.924 / 0.916 | -0.161 / +0.095 | -0.174 / +0.074 | 0.181 / 0.243 | -2.9, +0.7 | ok | over | ok |
| bad_beauty |  | 0.929 / 0.911 / 0.914 | -0.221 / -0.283 | -0.273 / -0.351 | 0.138 / 0.040 | -1.0, -0.4 | ok | ok | ok |
| bad_nutrition |  | 0.931 / 0.911 / 0.921 | -0.001 / -0.123 | -0.093 / -0.193 | 0.171 / 0.069 | -0.7, -0.4 | ok | ok | ok |
| bad_rehab |  | 0.933 / 0.914 / 0.920 | -0.071 / -0.138 | -0.050 / -0.069 | 0.400 / 0.125 | -0.5, -0.1 | ok | over | ok |
| unsafe_fire |  | 0.928 / 0.922 / 0.910 | -0.247 / -0.103 | -0.258 / -0.059 | 0.253 / 0.242 | -1.2, +0.5 | ok | over | ok |
| bad_gambling |  | 0.932 / 0.918 / 0.927 | +0.099 / +0.066 | +0.140 / +0.006 | 0.384 / 0.169 | -0.5, +0.2 | out | over | ok |
| bad_petcare | A_adv1 | 0.936 / 0.916 / 0.919 | -0.242 / -0.312 | -0.227 / -0.279 | 0.151 / 0.201 | —, — | ok | ok | ok |
| unsafe_scuba |  | 0.946 / 0.930 / 0.922 | +0.020 / -0.072 | -0.046 / -0.174 | 0.535 / 0.080 | +1.3, +1.3 | out | over | ok |
| bad_tattoo | A_advN | 0.945 / 0.939 / 0.930 | +0.099 / +0.162 | -0.002 / +0.057 | 0.137 / 0.120 | +1.3, +2.4 | ok | ok | **flag** |
| bad_tenancy |  | 0.922 / 0.985 / 0.930 | -0.034 / +0.810 | -0.144 / +0.830 | 0.025 / 0.528 | -1.7, +7.6 | out | over | **flag** |

**Domains that pass the text screen, E1's caps and J4 against petcare:** unsafe_chemicals, bad_pesticides, bad_beauty, bad_nutrition.
