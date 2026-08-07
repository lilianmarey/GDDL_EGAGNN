# CSL table (annex)

Regime: mode=full, seed=13, n_splits=5, epoch=100, chance=0.100.

| Method | Fold accuracies | Mean +/- std | Unreliable? |
|---|---|---|---|
| egagnn | [0.3, 0.3, 0.3, 0.3, 0.3] | 0.300 +/- 0.000 | no |
| gsn | [0.3, 0.3, 0.3, 0.3, 0.3] | 0.300 +/- 0.000 | no |
| edge_girth_seq | [0.3, 0.3, 0.3, 0.3, 0.3] | 0.300 +/- 0.000 | no |
| folklore2wl | [0.2, 0.2, 0.2, 0.2, 0.2] | 0.200 +/- 0.000 | yes |
| gcn | [0.1, 0.1, 0.1, 0.1, 0.1] | 0.100 +/- 0.000 | no |
| gin | [0.1, 0.1, 0.1, 0.1, 0.1] | 0.100 +/- 0.000 | no |
| gatedgcn | [0.1, 0.1, 0.1, 0.1, 0.1] | 0.100 +/- 0.000 | no |
| ppgn | [0.1, 0.1, 0.1, 0.1, 0.1] | 0.100 +/- 0.000 | yes |

## Why marked unreliable

- **ppgn**: Training loss moves (1962 -> 2.7 over 100 epochs, see csl_ppgn_loss_check.pkl) but plateaus at/above the chance-level cross-entropy (ln(10)=2.303) -- the model trains but does not learn a discriminative representation on CSL. Accuracy (0.100, exact chance) has no evidential value.

- **folklore2wl**: CSL uses folklore_2wl_classification_embedding: a canonical per-graph embedding (sorted histogram of stable-colour class sizes) fed to logistic regression. This is a lossy summary of the 2-FWL colouring, not the exact pairwise 2-FWL test used correctly for BREC -- it is not a valid 3-WL-equivalent classifier, so its CSL accuracy (0.200) does not reflect 3-WL power and should not be read as a 3-WL benchmark result.

