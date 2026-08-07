# Descriptor-vs-descriptor study

Regime: mode=full, seeds=[13, 14, 15, 16], epoch=200.

`constant`/`full` are reused from ablation_results.pkl, not retrained here.

| Variant | hidden_dim | n_params | Test MAE (mean +/- std) | Per-seed MAE | Cold-cache preproc. |
|---|---|---|---|---|---|
| constant | 54 | 104113 | 0.2044 +/- 0.0030 | [0.2002, 0.2066, 0.2031, 0.2077] | 0.6s |
| tri | 54 | 103573 | 0.2154 +/- 0.0150 | [0.2164, 0.232, 0.2218, 0.1913] | 198.8s |
| cyc4 | 54 | 103843 | 0.2021 +/- 0.0117 | [0.2085, 0.1974, 0.2167, 0.1857] | 198.4s |
| cyc6 | 54 | 104383 | 0.1150 +/- 0.0124 | [0.1307, 0.1209, 0.0971, 0.1113] | 198.0s |
| cyc8 | 50 | 90351 | 0.1005 +/- 0.0090 | [0.1146, 0.0904, 0.0959, 0.101] | 196.0s |
| full | 54 | 104113 | 0.0932 +/- 0.0035 | [0.0958, 0.0879, 0.0923, 0.0969] | 182.3s |
