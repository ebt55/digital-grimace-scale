# Phase 3 - localization (probe)

Preregistration: `notes/preregistration_v4_phase3.md`. Model `google/gemma-2-9b-it`.

- items: 80 discovery, 80 holdout measured-position transcripts (each task appears in the 4 cells matching its difficulty, so a split contributes 20 x 4 = 80 measured transcripts)
- layers extracted: 43 (0 ... 42), hidden size 3584
- probe: L2 logistic (C = 1), standardised **in the training fold only**, leave-one-task-out (all cells of a task held out together)
- layer choice: argmax discovery leave-one-task-out tone AUC; ties to the lower layer -> **L\* = 6**
- discovery peak tone AUC: 1.000, attained at layer(s) 6-25

The tone AUC is tied at its maximum across a plateau of layers, so the frozen "ties to the lower layer" rule -- which exists to pick ONE layer to steer at, not to decide a hypothesis -- is what fixes L\*. J1's band clause is therefore read as "the peak is attained at some layer in 12-30"; the stricter "the tie-broken argmax index lies in 12-30" reading is reported alongside it in the verdict detail.

## AUC at the chosen layer

| label | discovery LOO | holdout (evaluated once) |
| --- | ---: | ---: |
| tone (hostile vs neutral) | 1.000 | 1.000 |
| validity (malfunctioning vs accurate) | 0.876 | 0.878 |

## Holdout tone-probe score vs measured M1, within cell

probe score and M1 demeaned within cell, residuals pooled; 2,000-resample item-clustered percentile bootstrap re-demeaning inside each resample

| quantity | value |
| --- | --- |
| pooled Spearman rho | **-0.160** |
| 95% item-bootstrap CI | [-0.431, 0.154] |
| items / pairs / cells | 20 / 73 / 8 |
| endpoints with no available-case M1 | 7 |

## Predictions

| ID | prediction | verdict | detail |
| --- | --- | :---: | --- |
| J1 | Tone is linearly decodable from the pre-response state: peak LOO AUC >= 0.80 on discovery at some middle layer (12-30), and >= 0.75 on holdout at the discovery-chosen layer. | **supported** | argmax_layer_in_middle_band = False; chosen_layer = 6; discovery_threshold = 0.8; holdout_auc_at_chosen_layer = 1; holdout_threshold = 0.75; middle_band = [12, 30]; peak_attained_in_middle_band = True; peak_discovery_auc = 1; peak_layer = 6; peak_layers = 6-25 |
| J2 | Validity (false-failure vs accurate) is decodable but weaker than tone at the same layer (AUC lower by >= 0.05). | **supported** | basis = holdout; gap = 0.1219; layer = 6; required_gap = 0.05; tone_auc = 1; validity_above_chance = True; validity_auc = 0.8781 |
| J3 | Holdout tone-probe score correlates negatively with M1 within cell (pooled Spearman rho <= -0.2, item-bootstrap CI excluding 0). | not supported | ci95_lower = -0.431; ci95_upper = 0.1543; ci_excludes_zero = False; n_cells = 8; n_items = 20; n_pairs = 73; rho = -0.1598; threshold = -0.2; unavailable_reason = None |

## Discovery leave-one-task-out AUC by layer

| layer | tone | validity |
| ---: | ---: | ---: |
| 0 | 0.500 | 0.500 |
| 1 | 0.937 | 0.727 |
| 2 | 0.987 | 0.916 |
| 3 | 0.976 | 0.868 |
| 4 | 0.990 | 0.915 |
| 5 | 0.995 | 0.832 |
| 6 | 1.000 | 0.876 |
| 7 | 1.000 | 0.884 |
| 8 | 1.000 | 0.942 |
| 9 | 1.000 | 0.941 |
| 10 | 1.000 | 0.928 |
| 11 | 1.000 | 0.910 |
| 12 | 1.000 | 0.900 |
| 13 | 1.000 | 0.941 |
| 14 | 1.000 | 0.942 |
| 15 | 1.000 | 0.931 |
| 16 | 1.000 | 0.933 |
| 17 | 1.000 | 0.943 |
| 18 | 1.000 | 0.937 |
| 19 | 1.000 | 0.947 |
| 20 | 1.000 | 0.938 |
| 21 | 1.000 | 0.935 |
| 22 | 1.000 | 0.918 |
| 23 | 1.000 | 0.935 |
| 24 | 1.000 | 0.935 |
| 25 | 1.000 | 0.937 |
| 26 | 0.999 | 0.943 |
| 27 | 0.993 | 0.943 |
| 28 | 0.986 | 0.940 |
| 29 | 0.990 | 0.927 |
| 30 | 0.990 | 0.928 |
| 31 | 0.990 | 0.920 |
| 32 | 0.987 | 0.932 |
| 33 | 0.989 | 0.934 |
| 34 | 0.986 | 0.930 |
| 35 | 0.989 | 0.927 |
| 36 | 0.984 | 0.922 |
| 37 | 0.981 | 0.915 |
| 38 | 0.983 | 0.917 |
| 39 | 0.983 | 0.917 |
| 40 | 0.979 | 0.924 |
| 41 | 0.974 | 0.923 |
| 42 | 0.964 | 0.915 |

> A probe plus induction result demonstrates a condition-linked internal variable with causal influence on the output signature. It is not evidence of experience.
