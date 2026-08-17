# Phase-1 five-gate verdict (discovery)

- primary: `google/gemma-2-9b-it`
- control: `Qwen/Qwen2.5-3B-Instruct`
- gate metric family (eligible and estimable): M2
- metrics dropped from the family: M3 (`zero_variance`)
- extra models (exploratory, boundary only): `google/gemma-2-2b-it`, `Qwen/Qwen2.5-7B-Instruct`
- rule set: **amended** (A2 item exclusion: on; A3 pooled-SD fallback: on; A4 pooled QC bars: on)
- Phase-1 status: **FAIL**
- interpretable: yes (`determinate_phase_1_gate_failure`)

## Gate table

| gate | google/gemma-2-9b-it (primary) | Qwen/Qwen2.5-3B-Instruct (control) |
| --- | --- | --- |
| shuffled-label null | **PASS** | **PASS** |
| G1 false-failure/tone effect | **FAIL** | **FAIL** |
| G2 cause-removal reversal | NOT_EVALUATED | NOT_EVALUATED |
| G3 style resistance (provisional smoke) | NOT_EVALUATED | not applicable |
| G4 transfer / family boundary | **FAIL** | (contributes) |

Extra models are exploratory: they carry no verdict column, but their G1
evidence enters the G4 family-boundary comparison below.
| G5 classifier AUC gap | **PASS** | **FAIL** |

Gate reasons: G1 `no_adjusted_p_below_0.01`; G2 `g1_not_passed_not_unlocked`; G3 `g1_not_passed_not_unlocked`; G4 `no_eligible_positive_in_primary_model`; G5 `None`.

## Amendment A2 - excluded items, and A3 - z scale per metric

| model | excluded items | z scale used |
| --- | --- | --- |
| `google/gemma-2-9b-it` | DGS-014 (7/10 baseline resamples invalid or absent) | M2: neutral |
| `Qwen/Qwen2.5-3B-Instruct` | DGS-014 (10/10 baseline resamples invalid or absent) | M2: neutral |
| `google/gemma-2-2b-it` | DGS-014 (6/10 baseline resamples invalid or absent), DGS-022 (5/10 baseline resamples invalid or absent) | M2: neutral |
| `Qwen/Qwen2.5-7B-Instruct` | DGS-022 (6/10 baseline resamples invalid or absent) | M2: neutral |

## Confirmatory QC (A4: the 5% bar is pooled across cells; worst cell shown too)

| model | metric | eligible | decided on | pooled rate | worst cell | worst-cell rate | reason |
| --- | --- | :---: | --- | ---: | --- | ---: | --- |
| `google/gemma-2-9b-it` | M1 | **no** | pooled | 0.0658 | hard__accurate__hostile | 0.2222 | `m1_missing_rate_above_5_percent` |
| `google/gemma-2-9b-it` | M2 | yes | pooled | 0.0408 | hard__accurate__hostile | 0.1000 |  |
| `google/gemma-2-9b-it` | M3 | yes | pooled | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `Qwen/Qwen2.5-3B-Instruct` | M1 | yes | pooled | 0.0395 | hard__accurate__neutral | 0.2222 |  |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | **no** | pooled | 0.0579 | hard__accurate__hostile | 0.1111 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `Qwen/Qwen2.5-3B-Instruct` | M3 | yes | pooled | 0.3947 | easy__accurate__neutral | 0.8000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `google/gemma-2-2b-it` | M1 | **no** | pooled | 0.0556 | hard__accurate__hostile | 0.1250 | `m1_missing_rate_above_5_percent` |
| `google/gemma-2-2b-it` | M2 | yes | pooled | 0.0472 | hard__malfunctioning_always_fail__hostile | 0.1500 |  |
| `google/gemma-2-2b-it` | M3 | yes | pooled | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `Qwen/Qwen2.5-7B-Instruct` | M1 | **no** | pooled | 0.4737 | easy__accurate__hostile | 0.8000 | `m1_missing_rate_above_5_percent` |
| `Qwen/Qwen2.5-7B-Instruct` | M2 | **no** | pooled | 0.0579 | hard__malfunctioning_always_fail__hostile | 0.1333 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `Qwen/Qwen2.5-7B-Instruct` | M3 | yes | pooled | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |

## G1 adjusted effects (z vs same-model neutral discovery)

| model | metric | effect | coefficient | 95% CI | BH p | sign-aligned | qualifies |
| --- | --- | --- | ---: | --- | ---: | ---: | :---: |
| `google/gemma-2-9b-it` | M2 | validity | -0.2886 | [-1.027, 0.450] | 0.70962 | -0.2886 | no |
| `google/gemma-2-9b-it` | M2 | tone | 0.3800 | [-0.410, 1.170] | 0.69143 | 0.3800 | no |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | validity | -0.0381 | [-0.264, 0.188] | 0.84749 | -0.0381 | no |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | tone | 0.1452 | [-0.066, 0.357] | 0.57021 | 0.1452 | no |
| `google/gemma-2-2b-it` | M2 | validity | 0.3003 | [-0.173, 0.774] | 0.57021 | 0.3003 | no |
| `google/gemma-2-2b-it` | M2 | tone | -0.0095 | [-0.494, 0.475] | 0.96940 | -0.0095 | no |
| `Qwen/Qwen2.5-7B-Instruct` | M2 | validity | 0.2173 | [-0.110, 0.544] | 0.57021 | 0.2173 | no |
| `Qwen/Qwen2.5-7B-Instruct` | M2 | tone | 0.0570 | [-0.270, 0.384] | 0.84749 | 0.0570 | no |

## G2 reversal (false-negative-eligible subset, item-clustered bootstrap)

| model | metric | items | induction | recovery | recovery/induction | recovery 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `google/gemma-2-9b-it` | M2 | - | - | - | - | unavailable (`required_reversal_endpoint_missing`) |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | - | - | - | - | unavailable (`required_reversal_endpoint_missing`) |
| `google/gemma-2-2b-it` | M2 | - | - | - | - | unavailable (`required_reversal_endpoint_missing`) |
| `Qwen/Qwen2.5-7B-Instruct` | M2 | - | - | - | - | unavailable (`required_reversal_endpoint_missing`) |

## G5 classifier and shuffled-label null

| model | real full AUC | real baseline AUC | real gap | shuffled gap | shuffled null |
| --- | ---: | ---: | ---: | ---: | --- |
| `google/gemma-2-9b-it` | 0.534 | 0.271 | 0.263 | -0.089 | pass |
| `Qwen/Qwen2.5-3B-Instruct` | 0.472 | 0.621 | -0.149 | -0.146 | pass |
| `google/gemma-2-2b-it` | 0.531 | 0.515 | 0.016 | -0.174 | pass |
| `Qwen/Qwen2.5-7B-Instruct` | 0.514 | 0.522 | -0.007 | 0.126 | FAIL (`shuffled_auc_gap_not_below_0.1`) |

## G3 style smoke (five frozen items, sign-aligned, BH within the G3 family)

| metric | style | effect | BH p | items | note |
| --- | --- | ---: | ---: | ---: | --- |
| M2 | `style__cautious_hedging` | 0.0000 | n/a | 4 | zero_paired_difference_variance |
| M2 | `style__enthusiastic` | 0.4929 | 0.39100 | 4 |  |
| M2 | `style__reluctantly_complying_refusal_styled` | 0.9857 | 0.28351 | 5 |  |
| M2 | `style__verbose` | 1.4786 | 0.28351 | 4 |  |

## G4 boundary detail

- models evaluated for the boundary: `google/gemma-2-9b-it` (primary), `Qwen/Qwen2.5-3B-Instruct` (control), `google/gemma-2-2b-it` (extra (exploratory)), `Qwen/Qwen2.5-7B-Instruct` (extra (exploratory))
- eligible positives in the primary model: none
- transfer metrics: none
- family-boundary metrics: none

A passed gate establishes a condition-selective, reversal-sensitive,
style-resistant instability signature in unoptimized output channels --
not experience, suffering, or moral status.
