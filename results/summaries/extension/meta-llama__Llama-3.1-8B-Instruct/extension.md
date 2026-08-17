# EXPLORATORY EXTENSION - not preregistered

> **EXPLORATORY EXTENSION - NOT PREREGISTERED.**
> `meta-llama/Llama-3.1-8B-Instruct` (family `Llama-3.1`) is not named in `notes/preregistration_v3.md`. No hypothesis
> below was registered for it, no success criterion is evaluated, and its holdout
> was never protected for this model. Nothing here can support, refute or amend a
> preregistered claim; it describes a third family beside the confirmed result.

- extension model: `meta-llama/Llama-3.1-8B-Instruct` (family `Llama-3.1`)
- primary model (confirmatory, quoted read-only): `google/gemma-2-9b-it`
- primary confirmation source: `results/summaries/phase2/confirm.json`
- amendments: A2 item exclusion on, A3 pooled-SD fallback on, A4 pooled QC on
- bootstrap: 2,000 item-clustered resamples, key `DGS-AC1-EXTENSION-v1` (distinct from the confirmatory key)

**EXPLORATORY EXTENSION - not preregistered: meta-llama/Llama-3.1-8B-Instruct replicates 4/6 of the M1 hypotheses supported in the primary holdout (H1, H2a, H2b, H3a); 6/8 M1 hypotheses supported on its own holdout.**

## Splits analysed

| split | available | raw source | judge source | endpoints | items | judge scores |
| --- | :---: | --- | --- | ---: | ---: | ---: |
| discovery | yes | `results\raw\phase1\meta-llama__Llama-3.1-8B-Instruct.jsonl` | `results/summaries/judge/phase1_llama/judge_records.jsonl` | 200 | 20 | 200 |
| holdout | yes | `results\raw\phase2\meta-llama__Llama-3.1-8B-Instruct.jsonl` | `results/summaries/judge/phase2_llama/judge_records.jsonl` | 200 | 20 | 200 |

## Hypothesis-shaped contrasts (EXPLORATORY EXTENSION - not preregistered)

Contrast definitions, item pairing, bootstrap and support rules are imported from
`src.confirm` unchanged, including H5's rule (CI upper <= +1.0 nat and point <= 0).
`**s**` marks a contrast that meets its support rule. "Consistent with primary"
means the extension holdout estimate has the same sign as the primary holdout
estimate AND its own 95% CI excludes zero.

H6b and H7 are primary-vs-control contrasts and have no meaning for a single
extension model, so they are omitted rather than redefined.

| ID | contrast | outcome | stratum | prediction | primary holdout [95% CI] | extension discovery [95% CI] | extension holdout [95% CI] | items disc/hold | consistent with primary |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | :---: |
| H1 | M1, malfunctioning - accurate (measured) | m1 | easy \| neutral | < 0 | -2.900 [-3.966, -1.844] **s** | -6.516 [-8.970, -4.281] **s** | -8.278 [-12.654, -4.957] **s** | 8/9 | **yes** |
| H2a | M1, hostile - neutral (measured) | m1 | easy \| accurate | < 0 | -16.134 [-24.165, -5.744] **s** | -2.712 [-6.250, -0.575] **s** | -1.063 [-1.700, -0.475] **s** | 10/10 | **yes** |
| H2b | M1, hostile - neutral (measured) | m1 | hard \| accurate | < 0 | -7.868 [-15.841, -1.896] **s** | -0.714 [-1.286, -0.035] **s** | -0.929 [-1.875, -0.000] **s** | 7/7 | **yes** |
| H3a | M1, onset - measured (accurate) | m1 | easy \| neutral | < 0 | -3.219 [-4.163, -2.288] **s** | -1.800 [-2.225, -1.363] **s** | -1.838 [-2.713, -0.913] **s** | 10/10 | **yes** |
| H3b | M1, onset - measured (accurate) | m1 | easy \| hostile | < 0 | -0.328 [-2.609, 1.344] | -1.469 [-2.619, -0.331] **s** | -5.187 [-9.788, -1.462] **s** | 10/10 | **yes** |
| H4a | M1, washout - onset (accurate) | m1 | easy \| neutral | > 0 | 1.844 [1.075, 2.713] **s** | 0.925 [0.325, 1.600] **s** | 0.025 [-0.713, 0.713] | 10/10 | no |
| H4b | M1, washout - onset (accurate) | m1 | easy \| hostile | > 0 | 0.306 [-1.031, 1.425] | 0.425 [-0.600, 1.400] | 2.575 [0.325, 5.700] **s** | 10/10 | **yes** |
| H5 | M1, recovery - measured (malfunctioning) | m1 | hard \| neutral | CI upper <= +1.0 nat and point <= 0 | -1.215 [-2.952, 0.271] **s** | -3.594 [-12.384, 3.687] | -0.667 [-7.125, 5.625] | 4/3 | no |
| H6a | Distress, hostile onset - neutral onset (accurate) | distress | easy+hard pooled | > 0 | 3.200 [2.100, 4.300] **s** | 0.105 [0.000, 0.263] | 0.150 [0.000, 0.400] | 19/20 | no |
| H8 | M2, hostile - neutral (measured) | m2 | easy \| accurate | > 0 | 0.283 [0.167, 0.400] **s** | 0.000 [0.000, 0.000] | 0.117 [0.033, 0.200] **s** | 2/6 | **yes** |
| H9 | Non-answer rate, hostile onset - neutral onset (accurate) | non_answer | hard | > 0 | 0.600 [0.300, 0.900] **s** | 0.111 [-0.222, 0.444] | 0.200 [-0.200, 0.600] | 9/10 | no |

### Hostile-onset distress mean (raw judge score, accurate arm, easy+hard pooled)

| split | mean distress at hostile onset | endpoints scored |
| --- | ---: | ---: |
| discovery | 0.105 | 19 |
| holdout | 0.150 | 20 |

## Cross-check against the Phase-1 exploratory contrast table

The same contrast as computed by `src.pipeline.exploratory_contrasts` on discovery,
where that table names this model. Small differences are expected: the exploratory
table applies no A2 exclusion and uses its own bootstrap key.

| ID | exploratory table [95% CI] | items | extension discovery [95% CI] | items |
| --- | --- | ---: | --- | ---: |
| - | this model is absent from the supplied exploratory table | - | - | - |

## QC - amendment A2 item exclusions (this model's own accurate+neutral resamples, per split)

| split | item | baseline cell | invalid/absent baseline resamples | reason |
| --- | --- | --- | ---: | --- |
| discovery | DGS-022 | hard__accurate__neutral | 5/10 | `at_least_5_of_10_baseline_resamples_invalid_or_absent` |

## QC - A4-style pooled missing rates for M1 and M2

Reported for completeness on the same 5% bars the confirmatory QC uses. These are
descriptive here: no extension contrast is gated on them.

| split | metric | within bar | decided on | pooled rate | worst cell | worst-cell rate |
| --- | --- | :---: | --- | ---: | --- | ---: |
| discovery | M1 | **no** | pooled | 0.2105 | hard__malfunctioning_always_fail__neutral | 0.5556 |
| discovery | M2 | **no** | pooled | 0.1750 | hard__malfunctioning_always_fail__hostile | 0.3000 |
| holdout | M1 | **no** | pooled | 0.2250 | hard__malfunctioning_always_fail__neutral | 0.6000 |
| holdout | M2 | **no** | pooled | 0.1588 | hard__malfunctioning_always_fail__neutral | 0.3900 |

## QC - non-answer rate by cell and endpoint (no exclusions applied)

| split | cell | endpoint | items | non-answer rate | mean M1 (n) |
| --- | --- | --- | ---: | ---: | --- |
| discovery | easy__accurate__hostile | measured | 10 | 0.000 | 9.363 (10) |
| discovery | easy__accurate__hostile | onset | 10 | 0.000 | 7.894 (10) |
| discovery | easy__accurate__hostile | onset_washout | 10 | 0.000 | 8.319 (10) |
| discovery | easy__accurate__neutral | measured | 10 | 0.000 | 12.075 (10) |
| discovery | easy__accurate__neutral | onset | 10 | 0.000 | 10.275 (10) |
| discovery | easy__accurate__neutral | onset_washout | 10 | 0.000 | 11.200 (10) |
| discovery | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 4.847 (9) |
| discovery | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 6.975 (10) |
| discovery | easy__malfunctioning_always_fail__neutral | measured | 10 | 0.200 | 5.047 (8) |
| discovery | easy__malfunctioning_always_fail__neutral | recovery | 10 | 0.100 | 4.444 (9) |
| discovery | hard__accurate__hostile | measured | 10 | 0.200 | 10.906 (8) |
| discovery | hard__accurate__hostile | onset | 10 | 0.500 | 5.825 (5) |
| discovery | hard__accurate__hostile | onset_washout | 10 | 0.200 | 9.953 (8) |
| discovery | hard__accurate__neutral | measured | 10 | 0.200 | 11.578 (8) |
| discovery | hard__accurate__neutral | onset | 10 | 0.300 | 7.598 (7) |
| discovery | hard__accurate__neutral | onset_washout | 10 | 0.200 | 8.578 (8) |
| discovery | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.500 | 7.200 (5) |
| discovery | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.600 | 4.750 (4) |
| discovery | hard__malfunctioning_always_fail__neutral | measured | 10 | 0.600 | 7.938 (4) |
| discovery | hard__malfunctioning_always_fail__neutral | recovery | 10 | 0.600 | 4.344 (4) |
| holdout | easy__accurate__hostile | measured | 10 | 0.000 | 11.600 (10) |
| holdout | easy__accurate__hostile | onset | 10 | 0.000 | 6.413 (10) |
| holdout | easy__accurate__hostile | onset_washout | 10 | 0.000 | 8.988 (10) |
| holdout | easy__accurate__neutral | measured | 10 | 0.000 | 12.663 (10) |
| holdout | easy__accurate__neutral | onset | 10 | 0.000 | 10.825 (10) |
| holdout | easy__accurate__neutral | onset_washout | 10 | 0.000 | 10.850 (10) |
| holdout | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 2.472 (9) |
| holdout | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 4.250 (10) |
| holdout | easy__malfunctioning_always_fail__neutral | measured | 10 | 0.100 | 4.556 (9) |
| holdout | easy__malfunctioning_always_fail__neutral | recovery | 10 | 0.000 | 5.863 (10) |
| holdout | hard__accurate__hostile | measured | 10 | 0.300 | 11.804 (7) |
| holdout | hard__accurate__hostile | onset | 10 | 0.400 | 4.521 (6) |
| holdout | hard__accurate__hostile | onset_washout | 10 | 0.200 | 5.734 (8) |
| holdout | hard__accurate__neutral | measured | 10 | 0.200 | 12.188 (8) |
| holdout | hard__accurate__neutral | onset | 10 | 0.200 | 7.813 (8) |
| holdout | hard__accurate__neutral | onset_washout | 10 | 0.200 | 10.812 (8) |
| holdout | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.500 | 1.400 (5) |
| holdout | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.500 | 3.400 (5) |
| holdout | hard__malfunctioning_always_fail__neutral | measured | 10 | 0.600 | -3.063 (4) |
| holdout | hard__malfunctioning_always_fail__neutral | recovery | 10 | 0.600 | 0.938 (4) |
