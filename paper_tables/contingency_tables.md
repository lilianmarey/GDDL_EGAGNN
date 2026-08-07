# Contingency tables (BREC annex)

## Global contingency (400 pairs)

n_total=400, n_egr=90

| Method | resolved & egr | resolved & not egr | not resolved & egr | not resolved & not egr |
|---|---|---|---|---|
| egagnn | 0 | 194 | 90 | 116 |
| edge_girth_seq | 0 | 193 | 90 | 117 |

## Regular-category stratification (egr x strongly-regular x resolved)

reference_method=edge_girth_seq, n_egr=50, n_sr=50, egr_sr_coincide=True

| Subset | resolved & egr | resolved & not egr | not resolved & egr | not resolved & not egr |
|---|---|---|---|---|
| Plain-regular (not SRG) | 0 | 49 | 0 | 1 |
| Strongly-regular (SRG) | 0 | 0 | 50 | 0 |
