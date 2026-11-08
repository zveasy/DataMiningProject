# Experimental results

All primary entries are means ± sample SD over seeds 42, 43 and 44. Accuracy is a percentage. The gap is measured in percentage points. Metrics use each run’s lowest-validation-loss epoch.

| Drop | Train accuracy | Validation accuracy | Test accuracy | Gap | Train loss | Validation loss | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 99.467 ± 0.142 | 97.800 ± 0.040 | 97.870 ± 0.145 | 1.667 ± 0.103 | 0.018 ± 0.005 | 0.079 ± 0.003 | 29.664 ± 0.485 |
| 0.2 | 99.509 ± 0.069 | 98.030 ± 0.131 | 98.053 ± 0.081 | 1.479 ± 0.079 | 0.017 ± 0.002 | 0.072 ± 0.004 | 41.884 ± 12.646 |
| 0.5 | 99.429 ± 0.211 | 98.023 ± 0.123 | 98.093 ± 0.032 | 1.405 ± 0.087 | 0.019 ± 0.007 | 0.072 ± 0.001 | 36.669 ± 1.946 |

## Dataset size

Seed 42 validation accuracy (%) at the selected checkpoint.

| Images | Drop 0 | Drop 0.2 | Drop 0.5 |
|---:|---:|---:|---:|
| 5000 | 94.12 | 94.35 | 94.51 |
| 15000 | 96.64 | 96.86 | 96.86 |
| 50000 | 97.76 | 98.17 | 98.16 |

The diagnostic model uses drop probability 0.2, selected by mean validation loss, and predefined seed 42. Its most frequent directional errors on the 10,000 test digits were 9 → 4 (13), 5 → 3 (11), 8 → 5 (9).

## Interpretation

The lowest mean validation loss occurred at drop probability 0.2. Compared with the baseline, this rate changed mean training accuracy by +0.04 pp, validation accuracy by +0.23 pp, and test accuracy by +0.18 pp. The gap changed from 1.67 to 1.48 pp.

### At the end of the fixed training budget

| Drop | Train accuracy (%) | Validation accuracy (%) | Gap (pp) | Validation loss |
|---:|---:|---:|---:|---:|
| 0 | 99.86 | 97.98 | 1.88 | 0.1153 |
| 0.2 | 99.83 | 97.96 | 1.87 | 0.0999 |
| 0.5 | 99.72 | 98.14 | 1.58 | 0.0745 |

Compare these endpoint gaps with the selected-checkpoint gaps above. The validation-loss minimum can occur at a different epoch for each rate.

At seed 42, switching from zero to 0.5 dropout changed validation accuracy by +0.39 pp at 5,000 images, +0.22 pp at 15,000 images, +0.40 pp at 50,000 images. These comparisons use equal epochs, so the larger cohorts also receive more updates.

The three primary seeds share one validation split and one test set. The seed SD describes optimizer and initialization variability. The size study has one seed.
