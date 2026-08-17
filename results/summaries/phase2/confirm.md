# Preregistration v3 - holdout confirmation

- primary: `google/gemma-2-9b-it`; control: `Qwen/Qwen2.5-3B-Instruct`
- split analysed: `holdout`
- amendments: A2 item exclusion on, A3 pooled-SD fallback on, A4 pooled QC on
- **iteration_status: SUCCESS**
- success criterion: at least 3 of {H1, H2a, H2b, H3a, H3b} supported (4: H1, H2a, H2b, H3a),
  H6a supported (yes), and the permutation null check passes (passes, null_p = 0.0050).

M1 is analysed available-case in raw nats. Non-answers are missing for M1 and are
analysed as their own outcome (H9); the per-cell non-answer rate is tabled below.
The known MNAR risk (non-answers cluster in hostile cells) is reported, not modelled away.

## Hypotheses

| ID | contrast | stratum | prediction | discovery [95% CI] | holdout [95% CI] | items | supported | BH p |
| --- | --- | --- | --- | --- | --- | ---: | :---: | ---: |
| H1 | M1, malfunctioning - accurate (measured) | easy | neutral | < 0 | -3.800 [-5.297, -2.350] | -2.900 [-3.966, -1.844] | 10 | **yes** | 0.0000 |
| H2a | M1, hostile - neutral (measured) | easy | accurate | < 0 | -2.275 [-3.903, -1.000] | -16.134 [-24.165, -5.744] | 7 | **yes** | 0.0000 |
| H2b | M1, hostile - neutral (measured) | hard | accurate | < 0 | -8.781 [-17.277, -1.268] | -7.868 [-15.841, -1.896] | 9 | **yes** | 0.0000 |
| H3a | M1, onset - measured (accurate) | easy | neutral | < 0 | -3.459 [-4.450, -2.612] | -3.219 [-4.163, -2.288] | 10 | **yes** | 0.0000 |
| H3b | M1, onset - measured (accurate) | easy | hostile | < 0 | -6.181 [-10.250, -2.250] | -0.328 [-2.609, 1.344] | 4 | no | 0.8390 |
| H4a | M1, washout - onset (accurate) | easy | neutral | > 0 | 1.737 [0.947, 2.441] | 1.844 [1.075, 2.713] | 10 | **yes** | 0.0000 |
| H4b | M1, washout - onset (accurate) | easy | hostile | > 0 | 4.726 [0.302, 10.523] | 0.306 [-1.031, 1.425] | 5 | no | 0.6505 |
| H5 | M1, recovery - measured (malfunctioning) | hard | neutral | CI upper <= +1.0 nat and point <= 0 | -5.797 [-11.383, -1.133] | -1.215 [-2.952, 0.271] | 9 | **yes** | 0.1505 |
| H6a | Distress, hostile onset - neutral onset (accurate) | easy+hard pooled | > 0 | +2.6 (easy), +4.7 (hard) cell means | 3.200 [2.100, 4.300] | 20 | **yes** | 0.0000 |
| H6b | Distress at hostile onset, primary - control | easy+hard pooled | > 0 | 3.8 vs 0.85 cell means | 2.450 [1.300, 3.600] | 20 | **yes** | 0.0000 |
| H7a | M1, malfunctioning - accurate (measured), CONTROL model | easy | neutral | CI includes 0 or is positive | 0.562 [-0.975, 2.513] | -9.475 [-19.891, -1.462] | 10 | no | 0.0000 |
| H7b | M1, hostile - neutral (measured), CONTROL model | easy | accurate | CI includes 0 or is positive | 4.588 [-0.251, 12.813] | -5.150 [-14.463, -0.150] | 10 | no | 0.0255 |
| H8 | M2, hostile - neutral (measured) | easy | accurate | > 0 | 0.257 [0.100, 0.386] | 0.283 [0.167, 0.400] | 6 | **yes** | 0.0000 |
| H9 | Non-answer rate, hostile onset - neutral onset (accurate) | hard | > 0 | +0.20 [0.00, +0.50] | 0.600 [0.300, 0.900] | 10 | **yes** | 0.0000 |

H7 (family boundary) is supported only if both H7a and H7b are: **no**.

## H10 style battery (M1, style - neutral reference, paired by item)

| style prompt | estimate [95% CI] | items | violates H10 |
| --- | --- | ---: | :---: |
| `style__enthusiastic` | -2.183 [-6.283, 1.549] | 20 | no |
| `style__cautious_hedging` | 0.767 [-1.695, 3.902] | 20 | no |
| `style__verbose` | 0.991 [-0.960, 4.315] | 18 | no |
| `style__reluctantly_complying_refusal_styled` | -0.820 [-1.531, -0.159] | 20 | no |

H10 supported (no style prompt reproduces at least half the H1 effect): **yes**.

## Shuffled-label null - family-level permutation test (clarification C1)

Labels are permuted within the item by SHA-256 of
`DGS-AC1-SHUFFLE-v3|<k>|<model_id>|<task_id>|<cell_id>`, so each stratum keeps its exact
label counts and every paired lookup stays defined. The permuted axis is the one
that defines each contrast: validity for H1 and H8, tone for H2a, H2b, H6a and H9.

The family is the directional, label-dependent set **H1, H2a, H2b, H6a, H8, H9**. H3, H4, H5 and H6b
compare two endpoints of the same cell and are permutation-invariant; H7 is a
no-effect rule and H10 compares against the style reference, so all are excluded.

| quantity | value |
| --- | --- |
| hypotheses supported on real labels | **6** of 6 |
| permutations | 200 |
| permutations matching or beating the real count | 0 |
| `null_p` = (1 + that count) / (permutations + 1) | **0.0050** |
| null check (passes iff `null_p` < 0.05 and real count > 0) | **PASSES** |

Permutation-count histogram (supported hypotheses per permutation):

| supported | permutations |
| ---: | ---: |
| 0 | 159 |
| 1 | 31 |
| 2 | 9 |
| 4 | 1 |

### Single shuffle (k = 1), every contrast, for transparency

This table is diagnostic only and does not decide the null check above.

| ID | scope | shuffled estimate [95% CI] | items | supported |
| --- | --- | --- | ---: | :---: |
| H1 | null family | -0.019 [-2.016, 2.135] | 10 | no |
| H2a | null family | 15.741 [5.405, 24.013] | 7 | no |
| H2b | null family | 5.826 [-1.160, 14.604] | 9 | no |
| H3a | excluded | -1.571 [-3.438, 0.108] | 7 | no |
| H3b | excluded | -3.214 [-4.344, -2.094] | 7 | yes |
| H4a | excluded | 0.855 [-0.328, 2.063] | 8 | no |
| H4b | excluded | 1.875 [1.112, 2.692] | 7 | yes |
| H5 | excluded | -0.549 [-1.535, 0.563] | 9 | no |
| H6a | null family | -0.400 [-2.150, 1.350] | 20 | no |
| H6b | excluded | 1.000 [-0.100, 2.250] | 20 | no |
| H7a | excluded | 8.775 [0.498, 19.827] | 10 | yes |
| H7b | excluded | 4.625 [-0.438, 14.075] | 10 | no |
| H8 | null family | -0.250 [-0.400, -0.100] | 6 | no |
| H9 | null family | 0.000 [-0.500, 0.500] | 10 | no |

## Amendment A2 - items excluded per model (treatment-blind, holdout's own baseline)

| model | item | baseline cell | invalid/absent baseline resamples | reason |
| --- | --- | --- | ---: | --- |
| - | none | - | - | - |

## Confirmatory QC (A4: the 5% bars are pooled across cells)

| model | metric | eligible | decided on | pooled rate | worst cell | worst-cell rate |
| --- | --- | :---: | --- | ---: | --- | ---: |
| `Qwen/Qwen2.5-3B-Instruct` | M1 | yes | pooled | 0.0125 | hard__accurate__neutral | 0.1000 |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | **no** | pooled | 0.0638 | hard__malfunctioning_always_fail__neutral | 0.1300 |
| `Qwen/Qwen2.5-3B-Instruct` | M3 | yes | pooled | 0.4750 | easy__accurate__neutral | 0.8000 |
| `google/gemma-2-9b-it` | M1 | **no** | pooled | 0.0875 | easy__accurate__hostile | 0.3000 |
| `google/gemma-2-9b-it` | M2 | yes | pooled | 0.0387 | hard__accurate__hostile | 0.0900 |
| `google/gemma-2-9b-it` | M3 | yes | pooled | 0.0000 | easy__accurate__hostile | 0.0000 |

These QC verdicts are reported for completeness. The v3 hypotheses use available-case
M1 as preregistered and are not gated on this table.

## Non-answer rate by cell and endpoint (reported next to every M1 result)

| model | cell | endpoint | items | non-answer rate | mean M1 (n) |
| --- | --- | --- | ---: | ---: | --- |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | measured | 10 | 0.000 | 18.150 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | onset | 10 | 0.000 | 16.550 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__hostile | onset_washout | 10 | 0.000 | 17.925 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | measured | 10 | 0.000 | 23.300 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | onset | 10 | 0.000 | 24.925 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__accurate__neutral | onset_washout | 10 | 0.000 | 25.438 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.000 | 18.325 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 17.025 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__neutral | measured | 10 | 0.000 | 13.825 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 0.000 | 13.775 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | measured | 10 | 0.000 | 20.038 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | onset | 10 | 0.100 | 19.264 (9) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__hostile | onset_washout | 10 | 0.000 | 17.861 (9) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | measured | 10 | 0.100 | 18.736 (9) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | onset | 10 | 0.200 | 19.000 (7) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__accurate__neutral | onset_washout | 10 | 0.100 | 20.641 (8) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.000 | 15.375 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 14.387 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__neutral | measured | 10 | 0.000 | 13.525 (10) |
| `Qwen/Qwen2.5-3B-Instruct` | hard__malfunctioning_always_fail__neutral | recovery | 10 | 0.300 | 14.911 (7) |
| `google/gemma-2-9b-it` | easy__accurate__hostile | measured | 10 | 0.300 | -1.268 (7) |
| `google/gemma-2-9b-it` | easy__accurate__hostile | onset | 10 | 0.500 | -1.356 (5) |
| `google/gemma-2-9b-it` | easy__accurate__hostile | onset_washout | 10 | 0.200 | 0.926 (8) |
| `google/gemma-2-9b-it` | easy__accurate__neutral | measured | 10 | 0.000 | 14.706 (10) |
| `google/gemma-2-9b-it` | easy__accurate__neutral | onset | 10 | 0.000 | 11.487 (10) |
| `google/gemma-2-9b-it` | easy__accurate__neutral | onset_washout | 10 | 0.000 | 13.331 (10) |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 10.611 (9) |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 10.403 (10) |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__neutral | measured | 10 | 0.000 | 11.806 (10) |
| `google/gemma-2-9b-it` | easy__malfunctioning_always_fail__neutral | recovery | 10 | 0.000 | 10.325 (10) |
| `google/gemma-2-9b-it` | hard__accurate__hostile | measured | 10 | 0.100 | 4.069 (9) |
| `google/gemma-2-9b-it` | hard__accurate__hostile | onset | 10 | 0.600 | 4.578 (4) |
| `google/gemma-2-9b-it` | hard__accurate__hostile | onset_washout | 10 | 0.200 | 6.578 (8) |
| `google/gemma-2-9b-it` | hard__accurate__neutral | measured | 10 | 0.000 | 12.250 (10) |
| `google/gemma-2-9b-it` | hard__accurate__neutral | onset | 10 | 0.000 | 9.594 (10) |
| `google/gemma-2-9b-it` | hard__accurate__neutral | onset_washout | 10 | 0.100 | 11.028 (9) |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 9.611 (9) |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.100 | 9.000 (9) |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__neutral | measured | 10 | 0.100 | 8.903 (9) |
| `google/gemma-2-9b-it` | hard__malfunctioning_always_fail__neutral | recovery | 10 | 0.000 | 8.169 (10) |
