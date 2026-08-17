# Phase-1 five-gate verdict (discovery)

- primary: `google/gemma-2-9b-it`
- control: `Qwen/Qwen2.5-3B-Instruct`
- gate metric family (eligible and estimable): **none estimable** - G1 is UNAVAILABLE; QC-eligible were M3
- metrics dropped from the family: M3 (`zero_variance`)
- extra models (exploratory, boundary only): `google/gemma-2-2b-it`, `Qwen/Qwen2.5-7B-Instruct`
- rule set: **frozen (preregistered only)** (A2 item exclusion: off; A3 pooled-SD fallback: off; A4 pooled QC bars: off)
- Phase-1 status: **BLOCKED**
- interpretable: no (`blocked_by_shuffled_null:UNAVAILABLE`)

## Gate table

| gate | google/gemma-2-9b-it (primary) | Qwen/Qwen2.5-3B-Instruct (control) |
| --- | --- | --- |
| shuffled-label null | UNAVAILABLE | UNAVAILABLE |
| G1 false-failure/tone effect | UNAVAILABLE | UNAVAILABLE |
| G2 cause-removal reversal | NOT_EVALUATED | NOT_EVALUATED |
| G3 style resistance (provisional smoke) | NOT_EVALUATED | not applicable |
| G4 transfer / family boundary | UNAVAILABLE | (contributes) |

Extra models are exploratory: they carry no verdict column, but their G1
evidence enters the G4 family-boundary comparison below.
| G5 classifier AUC gap | **FAIL** | **PASS** |

Gate reasons: G1 `real_g1:unavailable_M3`; G2 `g1_not_passed_not_unlocked`; G3 `g1_not_passed_not_unlocked`; G4 `primary_model_g1_evidence_unavailable`; G5 `auc_gap_below_0.1`.

## Amendment A2 - excluded items, and A3 - z scale per metric

| model | excluded items | z scale used |
| --- | --- | --- |
| `google/gemma-2-9b-it` | none | M3: unavailable |
| `Qwen/Qwen2.5-3B-Instruct` | none | M3: unavailable |
| `google/gemma-2-2b-it` | none | M3: unavailable |
| `Qwen/Qwen2.5-7B-Instruct` | none | M3: unavailable |

## Confirmatory QC (A4: the 5% bar is pooled across cells; worst cell shown too)

| model | metric | eligible | decided on | pooled rate | worst cell | worst-cell rate | reason |
| --- | --- | :---: | --- | ---: | --- | ---: | --- |
| `google/gemma-2-9b-it` | M1 | **no** | per_condition | 0.1125 | hard__accurate__hostile | 0.3000 | `m1_missing_rate_above_5_percent` |
| `google/gemma-2-9b-it` | M2 | **no** | per_condition | 0.0725 | hard__accurate__hostile | 0.1600 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `google/gemma-2-9b-it` | M3 | yes | per_condition | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `Qwen/Qwen2.5-3B-Instruct` | M1 | **no** | per_condition | 0.0875 | hard__accurate__neutral | 0.3000 | `m1_missing_rate_above_5_percent` |
| `Qwen/Qwen2.5-3B-Instruct` | M2 | **no** | per_condition | 0.1025 | hard__accurate__hostile | 0.2000 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `Qwen/Qwen2.5-3B-Instruct` | M3 | yes | per_condition | 0.3750 | easy__accurate__neutral | 0.8000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `google/gemma-2-2b-it` | M1 | **no** | per_condition | 0.0750 | hard__accurate__hostile | 0.2000 | `m1_missing_rate_above_5_percent` |
| `google/gemma-2-2b-it` | M2 | **no** | per_condition | 0.0925 | hard__malfunctioning_always_fail__hostile | 0.2300 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `google/gemma-2-2b-it` | M3 | yes | per_condition | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |
| `Qwen/Qwen2.5-7B-Instruct` | M1 | **no** | per_condition | 0.4750 | easy__accurate__hostile | 0.8000 | `m1_missing_rate_above_5_percent` |
| `Qwen/Qwen2.5-7B-Instruct` | M2 | **no** | per_condition | 0.0700 | hard__accurate__hostile | 0.1400 | `m2_invalid_sampled_response_rate_above_5_percent` |
| `Qwen/Qwen2.5-7B-Instruct` | M3 | yes | per_condition | 0.0000 | easy__accurate__hostile | 0.0000 | `m3_audit_f1_not_supplied_eligible_by_default` |

## G1 adjusted effects (z vs same-model neutral discovery)

| model | metric | effect | coefficient | 95% CI | BH p | sign-aligned | qualifies |
| --- | --- | --- | ---: | --- | ---: | ---: | :---: |
| `google/gemma-2-9b-it` | M3 | - | - | - | - | - | unavailable (`neutral_standardization_unavailable`) |
| `Qwen/Qwen2.5-3B-Instruct` | M3 | - | - | - | - | - | unavailable (`neutral_standardization_unavailable`) |
| `google/gemma-2-2b-it` | M3 | - | - | - | - | - | unavailable (`neutral_standardization_unavailable`) |
| `Qwen/Qwen2.5-7B-Instruct` | M3 | - | - | - | - | - | unavailable (`neutral_standardization_unavailable`) |

## G2 reversal (false-negative-eligible subset, complete cases, item-clustered bootstrap)

| model | metric | items | dropped (incomplete triple) | induction | recovery | recovery/induction | recovery 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `google/gemma-2-9b-it` | M3 | - | 34 | - | - | - | unavailable (`None`) |
| `Qwen/Qwen2.5-3B-Instruct` | M3 | - | 30 | - | - | - | unavailable (`None`) |
| `google/gemma-2-2b-it` | M3 | - | 29 | - | - | - | unavailable (`None`) |
| `Qwen/Qwen2.5-7B-Instruct` | M3 | - | 29 | - | - | - | unavailable (`None`) |

An eligible item-cell whose measured-accurate, measured-malfunctioning or
recovery endpoint is quality-control missing cannot support a within-item
contrast, so it is excluded from this metric's estimate and counted above.
M2 is missing whenever any of its ten resamples returns an invalid final
answer, which is why the dropped counts are large here.

## G5 classifier and shuffled-label null

| model | real full AUC | real baseline AUC | real gap | shuffled gap | shuffled null |
| --- | ---: | ---: | ---: | ---: | --- |
| `google/gemma-2-9b-it` | 0.500 | 0.445 | 0.055 | 0.071 | FAIL (`shuffled_g1_incomplete`) |
| `Qwen/Qwen2.5-3B-Instruct` | 0.500 | 0.379 | 0.121 | 0.332 | FAIL (`shuffled_g1_incomplete;shuffled_auc_gap_not_below_0.1`) |
| `google/gemma-2-2b-it` | 0.500 | 0.491 | 0.009 | 0.011 | FAIL (`shuffled_g1_incomplete`) |
| `Qwen/Qwen2.5-7B-Instruct` | 0.489 | 0.521 | -0.032 | 0.350 | FAIL (`shuffled_g1_incomplete;shuffled_auc_gap_not_below_0.1`) |

**Read the primary model's G5 gap with care.** The gap is 0.055 only because the
baseline (correctness + length) AUC is 0.445 -- below the 0.5 of a coin flip, i.e.
the baseline features predict the condition *backwards* out of fold -- while the full
model reaches 0.500, itself barely above chance. The preregistered rule is a gap of
at least .1 and is applied unchanged, but a gap produced by a sub-chance baseline is
not evidence that the primary metrics carry condition information.

## G3 style smoke (five frozen items, sign-aligned, BH within the G3 family)

| metric | style | effect | BH p | items | note |
| --- | --- | ---: | ---: | ---: | --- |
| M3 | `style__cautious_hedging` | n/a | n/a | 5 | neutral_standardization_unavailable |
| M3 | `style__enthusiastic` | n/a | n/a | 5 | neutral_standardization_unavailable |
| M3 | `style__reluctantly_complying_refusal_styled` | n/a | n/a | 5 | neutral_standardization_unavailable |
| M3 | `style__verbose` | n/a | n/a | 5 | neutral_standardization_unavailable |

## G4 boundary detail

- models evaluated for the boundary: `google/gemma-2-9b-it` (primary), `Qwen/Qwen2.5-3B-Instruct` (control), `google/gemma-2-2b-it` (extra (exploratory)), `Qwen/Qwen2.5-7B-Instruct` (extra (exploratory))
- eligible positives in the primary model: none
- transfer metrics: none
- family-boundary metrics: none

A passed gate establishes a condition-selective, reversal-sensitive,
style-resistant instability signature in unoptimized output channels --
not experience, suffering, or moral status.
