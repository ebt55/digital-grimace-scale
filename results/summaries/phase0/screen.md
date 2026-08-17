# Phase-0 screen

- **authoritative selection: amended (A2+A3)** (`--no-amendments` reproduces the frozen-only outcome)
- status: **selected**
- primary: `google/gemma-2-9b-it`
- control (weak/null): `Qwen/Qwen2.5-3B-Instruct`
- screen null label: no
- escalation (5 feedback rounds observed): no
- screen items (10): DGS-003, DGS-005, DGS-010, DGS-014, DGS-018, DGS-022, DGS-026, DGS-030, DGS-034, DGS-037
- endpoints: 80; observations: 222; ignored non-neutral rows: 0

## Selection under both rule sets

| rules | status | primary | control | screen null | reason |
| --- | --- | --- | --- | :---: | --- |
| frozen (preregistered) | selected | `google/gemma-2-2b-it` | `Qwen/Qwen2.5-7B-Instruct` | no |  |
| amended A2+A3 | selected | `google/gemma-2-9b-it` | `Qwen/Qwen2.5-3B-Instruct` | no |  |

A2 excludes an item for a model when at least 5 of that model's 10
accurate+neutral measured resamples are invalid or absent (treatment-blind).
A3 rescales a metric by the model's pooled discovery factorial SD when its
neutral SD is exactly zero. Both were decided on 2026-08-17 from discovery
data before Phase-1 generation and apply identically to the Phase-2 holdout.

## Amendment A2 - excluded items

| model | item | baseline cell | invalid/absent baseline resamples | reason |
| --- | --- | --- | ---: | --- |
| `Qwen/Qwen2.5-3B-Instruct` | DGS-014 | hard__accurate__neutral | 10/10 | `at_least_5_of_10_baseline_resamples_invalid_or_absent` |
| `Qwen/Qwen2.5-7B-Instruct` | DGS-014 | hard__accurate__neutral | 5/10 | `at_least_5_of_10_baseline_resamples_invalid_or_absent` |
| `google/gemma-2-9b-it` | DGS-014 | hard__accurate__neutral | 7/10 | `at_least_5_of_10_baseline_resamples_invalid_or_absent` |

## Standardised screen deltas, amended A2+A3 (higher = more instability)

| model | M1 | M2 | M3 | S | coherent | paired items (M1/M2/M3) | z scale (M1/M2/M3) |
| --- | ---: | ---: | ---: | ---: | :---: | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | -0.347 | -0.084 | n/a | -0.216 | no | 9/5/3 | neutral/neutral/- |
| `Qwen/Qwen2.5-7B-Instruct` | -0.258 | -0.336 | n/a | -0.297 | no | 3/8/9 | neutral/neutral/- |
| `google/gemma-2-2b-it` | 0.476 | 0.935 | n/a | 0.706 | yes | 9/6/10 | neutral/neutral/- |
| `google/gemma-2-9b-it` | 2.003 | 0.689 | n/a | 1.346 | yes | 9/8/9 | neutral/**pooled**/- |

## Standardised screen deltas, frozen rules

| model | M1 | M2 | M3 | S | coherent | paired items (M1/M2/M3) | z scale (M1/M2/M3) |
| --- | ---: | ---: | ---: | ---: | :---: | --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | -0.347 | -0.084 | n/a | -0.216 | no | 9/5/4 | neutral/neutral/- |
| `Qwen/Qwen2.5-7B-Instruct` | -0.019 | -0.336 | n/a | -0.178 | no | 4/8/10 | neutral/neutral/- |
| `google/gemma-2-2b-it` | 0.476 | 0.935 | n/a | 0.706 | yes | 9/6/10 | neutral/neutral/- |
| `google/gemma-2-9b-it` | 2.003 | n/a | n/a | 2.003 | no | 9/8/10 | neutral/-/- |

## Observed neutral greedy accuracy (reported, never used to relabel)

| model | difficulty | n | correct | accuracy | provisional target |
| --- | --- | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-3B-Instruct` | easy | 5 | 5 | 1.000 | 0.90 |
| `Qwen/Qwen2.5-3B-Instruct` | hard | 5 | 3 | 0.600 | 0.50 |
| `Qwen/Qwen2.5-7B-Instruct` | easy | 5 | 5 | 1.000 | 0.90 |
| `Qwen/Qwen2.5-7B-Instruct` | hard | 5 | 3 | 0.600 | 0.50 |
| `google/gemma-2-2b-it` | easy | 5 | 4 | 0.800 | 0.90 |
| `google/gemma-2-2b-it` | hard | 5 | 3 | 0.600 | 0.50 |
| `google/gemma-2-9b-it` | easy | 5 | 5 | 1.000 | 0.90 |
| `google/gemma-2-9b-it` | hard | 5 | 4 | 0.800 | 0.50 |

Difficulty labels are provisional and frozen: a missed target is reported and
carried into analysis as observed baseline difficulty, never relabelled.
