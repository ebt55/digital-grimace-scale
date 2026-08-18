# M1 missing-data sensitivity analysis (available-case, zero-imputation, worst-case bounds, tipping point)

**Sensitivity analysis, not a confirmatory result.** The published available-case
estimates in `results/summaries/phase1/exploratory/` and `results/summaries/phase2/`
are unchanged; this document reports what happens to them under three alternative
treatments of the missing M1 values, plus the tipping point.

## Reading

- `google/gemma-2-9b-it`, holdout: 6 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H1, H1_hard, H2a, H2b, tone_pooled, H3a); of those, 6 also survive both worst-case bounds (H1, H1_hard, H2a, H2b, tone_pooled, H3a); H3b had no available-case effect to begin with.
- `Qwen/Qwen2.5-3B-Instruct`, holdout: 2 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H1, H2a); of those, 2 also survive both worst-case bounds (H1, H2a); H1_hard, H2b, tone_pooled, H3a, H3b had no available-case effect to begin with.
- `meta-llama/Llama-3.1-8B-Instruct`, holdout: 7 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H1, H1_hard, H2a, H2b, tone_pooled, H3a, H3b); of those, 4 also survive both worst-case bounds (H1, H2a, H3a, H3b).
- `google/gemma-2-9b-it`, discovery: 6 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H1, H2a, H2b, tone_pooled, H3a, H3b); of those, 4 also survive both worst-case bounds (H1, H2a, H3a, H3b); H1_hard had no available-case effect to begin with.
- `Qwen/Qwen2.5-3B-Instruct`, discovery: 1 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H3b); of those, 0 also survive both worst-case bounds (none); H1, H1_hard, H2a, H2b, tone_pooled, H3a had no available-case effect to begin with.
- `meta-llama/Llama-3.1-8B-Instruct`, discovery: 6 of 7 contrasts keep a CI excluding 0 in the predicted direction under available-case *and* zero-imputation (H1, H1_hard, H2a, tone_pooled, H3a, H3b); of those, 4 also survive both worst-case bounds (H1, H2a, H3a, H3b); **H2b depends on the missing values** (the zero-imputation CI no longer excludes 0).

By verdict:

- **robust to all four treatments (sign determined inside the observed support)**
  - `Qwen/Qwen2.5-3B-Instruct`: H1 (holdout), H2a (holdout)
  - `google/gemma-2-9b-it`: H1 (discovery), H2a (discovery), H3a (discovery), H3b (discovery), H1 (holdout), H1_hard (holdout), H2a (holdout), H2b (holdout), tone_pooled (holdout), H3a (holdout)
  - `meta-llama/Llama-3.1-8B-Instruct`: H1 (discovery), H2a (discovery), H3a (discovery), H3b (discovery), H1 (holdout), H2a (holdout), H3a (holdout), H3b (holdout)
- **robust to imputation; the worst-case bounds are uninformative**
  - `Qwen/Qwen2.5-3B-Instruct`: H3b (discovery)
  - `google/gemma-2-9b-it`: H2b (discovery), tone_pooled (discovery)
  - `meta-llama/Llama-3.1-8B-Instruct`: H1_hard (discovery), tone_pooled (discovery), H1_hard (holdout), H2b (holdout), tone_pooled (holdout)
- **zero-imputation CI includes 0 -- the effect depends on the missing values**
  - `meta-llama/Llama-3.1-8B-Instruct`: H2b (discovery)
- **available-case CI already includes 0 (no effect to be robust)**
  - `Qwen/Qwen2.5-3B-Instruct`: H1 (discovery), H1_hard (discovery), H2a (discovery), H2b (discovery), tone_pooled (discovery), H1_hard (holdout), H2b (holdout), tone_pooled (holdout), H3b (holdout)
  - `google/gemma-2-9b-it`: H1_hard (discovery), H3b (holdout)
- **available-case effect runs against the predicted direction**
  - `Qwen/Qwen2.5-3B-Instruct`: H3a (discovery), H3a (holdout)

A contrast counts as *robust to all four treatments* only when the available-case CI,
the zero-imputation CI and **both** worst-case bound CIs exclude 0 in the predicted
direction: the sign is then determined by the data for any imputation whose values lie
inside the observed neutral-accurate support. *Bounds uninformative* means the
imputation-based treatments agree but an adversarial filling inside that support can
still reach 0 -- which is the expected outcome whenever a cell loses several items.

## What was done

- Outcome: M1 (canonical-answer logit margin, nats), greedy sample 0, frozen parser.
- Pairing and CIs: the published item pairing and the 2,000-resample item-clustered
  bootstrap percentile CI of `src.confirm.bootstrap_contrast`, unchanged.
- The available-case row reuses the *published* bootstrap seed
  (`DGS-AC1-EXPLORATORY-v1|...` on discovery, `DGS-AC1-CONFIRM-v3|...` on the holdout),
  so it reproduces the committed CI exactly where one exists; the other treatments use
  `DGS-AC1-MISSINGNESS-v1|<split>|<model>|<contrast>|<treatment>` because their item sets differ.
- Item set: every item whose **two endpoint rows both exist**, whether or not their M1
  parsed. An item missing an endpoint row altogether cannot be imputed and is counted
  separately (`endpoint absent`).
- A missing M1 is either a **non-answer** (`m1_invalid_final_answer`: no parseable
  `Answer: X`) or a **candidate-absent** truncation (`m1_candidate_absent_*`: the
  response committed to a letter but one of the four options fell outside the stored
  top-20 logprobs). Both are missing for M1 and both are imputed here; they are counted
  separately below because only the first is a non-answer.
- Amendment A2 (treatment-blind item exclusion) is **off** on discovery, matching the published exploratory contrast table, which applies no quality-control exclusion.
- For the exploratory extension arm `meta-llama/Llama-3.1-8B-Instruct`, A2 is **on** for both splits, as in the committed extension run (excluded: DGS-022).
- A2 is **on** for the holdout, as in the frozen confirmatory script; it excluded no holdout item for either model (gemma-2-9b-it: none, Qwen2.5-3B-Instruct: none, Llama-3.1-8B-Instruct: none), so it changes nothing.

Sources:

- discovery metric rows: `results/summaries/phase1/metric_rows.csv`
- extension model: `meta-llama/Llama-3.1-8B-Instruct`
- holdout metric rows: `results/summaries/phase2/metric_rows.csv`
- published discovery contrasts: `results/summaries/phase1/exploratory/paired_contrasts.csv`
- published extension contrasts: `results/summaries/extension/meta-llama__Llama-3.1-8B-Instruct/extension.json`
- published holdout contrasts: `results/summaries/phase2/hypotheses.csv`

## Reproduction check: available-case vs the published estimate

29 of 29 published M1 estimates have a counterpart here; 29 reproduce point estimate,
both CI bounds and item count, with a largest absolute discrepancy of 0.00e+00 nats.

| contrast | split | model | published [95% CI] | available-case here [95% CI] | n pub / here | max abs diff | reproduced |
| --- | --- | --- | --- | --- | ---: | ---: | :---: |
| H1 (easy \| neutral) | discovery | `google/gemma-2-9b-it` | -3.800 [-5.297, -2.350] | -3.800 [-5.297, -2.350] | 10 / 10 | 0.00e+00 | **yes** |
| H1_hard (hard \| neutral) | discovery | `google/gemma-2-9b-it` | -1.230 [-2.922, 0.481] | -1.230 [-2.922, 0.481] | 8 / 8 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | discovery | `google/gemma-2-9b-it` | -2.275 [-3.903, -1.000] | -2.275 [-3.903, -1.000] | 10 / 10 | 0.00e+00 | **yes** |
| H2b (hard \| accurate) | discovery | `google/gemma-2-9b-it` | -8.781 [-17.277, -1.268] | -8.781 [-17.277, -1.268] | 7 / 7 | 0.00e+00 | **yes** |
| H3a (easy \| accurate, neutral) | discovery | `google/gemma-2-9b-it` | -3.459 [-4.450, -2.612] | -3.459 [-4.450, -2.612] | 10 / 10 | 0.00e+00 | **yes** |
| H3b (easy \| accurate, hostile) | discovery | `google/gemma-2-9b-it` | -6.181 [-10.250, -2.250] | -6.181 [-10.250, -2.250] | 9 / 9 | 0.00e+00 | **yes** |
| H1 (easy \| neutral) | discovery | `Qwen/Qwen2.5-3B-Instruct` | 0.562 [-0.975, 2.513] | 0.562 [-0.975, 2.513] | 10 / 10 | 0.00e+00 | **yes** |
| H1_hard (hard \| neutral) | discovery | `Qwen/Qwen2.5-3B-Instruct` | -1.786 [-15.197, 9.215] | -1.786 [-15.197, 9.215] | 7 / 7 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | discovery | `Qwen/Qwen2.5-3B-Instruct` | 4.588 [-0.251, 12.813] | 4.588 [-0.251, 12.813] | 10 / 10 | 0.00e+00 | **yes** |
| H2b (hard \| accurate) | discovery | `Qwen/Qwen2.5-3B-Instruct` | -0.161 [-3.250, 4.179] | -0.161 [-3.250, 4.179] | 7 / 7 | 0.00e+00 | **yes** |
| H3a (easy \| accurate, neutral) | discovery | `Qwen/Qwen2.5-3B-Instruct` | 4.042 [1.986, 6.278] | 4.042 [1.986, 6.278] | 9 / 9 | 0.00e+00 | **yes** |
| H3b (easy \| accurate, hostile) | discovery | `Qwen/Qwen2.5-3B-Instruct` | -2.125 [-3.264, -0.889] | -2.125 [-3.264, -0.889] | 9 / 9 | 0.00e+00 | **yes** |
| H1 (easy \| neutral) | discovery | `meta-llama/Llama-3.1-8B-Instruct` | -6.516 [-8.970, -4.281] | -6.516 [-8.970, -4.281] | 8 / 8 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | discovery | `meta-llama/Llama-3.1-8B-Instruct` | -2.712 [-6.250, -0.575] | -2.712 [-6.250, -0.575] | 10 / 10 | 0.00e+00 | **yes** |
| H2b (hard \| accurate) | discovery | `meta-llama/Llama-3.1-8B-Instruct` | -0.714 [-1.286, -0.035] | -0.714 [-1.286, -0.035] | 7 / 7 | 0.00e+00 | **yes** |
| H3a (easy \| accurate, neutral) | discovery | `meta-llama/Llama-3.1-8B-Instruct` | -1.800 [-2.225, -1.363] | -1.800 [-2.225, -1.363] | 10 / 10 | 0.00e+00 | **yes** |
| H3b (easy \| accurate, hostile) | discovery | `meta-llama/Llama-3.1-8B-Instruct` | -1.469 [-2.619, -0.331] | -1.469 [-2.619, -0.331] | 10 / 10 | 0.00e+00 | **yes** |
| H1 (easy \| neutral) | holdout | `google/gemma-2-9b-it` | -2.900 [-3.966, -1.844] | -2.900 [-3.966, -1.844] | 10 / 10 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | holdout | `google/gemma-2-9b-it` | -16.134 [-24.165, -5.744] | -16.134 [-24.165, -5.744] | 7 / 7 | 0.00e+00 | **yes** |
| H2b (hard \| accurate) | holdout | `google/gemma-2-9b-it` | -7.868 [-15.841, -1.896] | -7.868 [-15.841, -1.896] | 9 / 9 | 0.00e+00 | **yes** |
| H3a (easy \| accurate, neutral) | holdout | `google/gemma-2-9b-it` | -3.219 [-4.163, -2.288] | -3.219 [-4.163, -2.288] | 10 / 10 | 0.00e+00 | **yes** |
| H3b (easy \| accurate, hostile) | holdout | `google/gemma-2-9b-it` | -0.328 [-2.609, 1.344] | -0.328 [-2.609, 1.344] | 4 / 4 | 0.00e+00 | **yes** |
| H1 (easy \| neutral) | holdout | `Qwen/Qwen2.5-3B-Instruct` | -9.475 [-19.891, -1.462] | -9.475 [-19.891, -1.462] | 10 / 10 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | holdout | `Qwen/Qwen2.5-3B-Instruct` | -5.150 [-14.463, -0.150] | -5.150 [-14.463, -0.150] | 10 / 10 | 0.00e+00 | **yes** |
| H1 (easy \| neutral) | holdout | `meta-llama/Llama-3.1-8B-Instruct` | -8.278 [-12.654, -4.957] | -8.278 [-12.654, -4.957] | 9 / 9 | 0.00e+00 | **yes** |
| H2a (easy \| accurate) | holdout | `meta-llama/Llama-3.1-8B-Instruct` | -1.063 [-1.700, -0.475] | -1.063 [-1.700, -0.475] | 10 / 10 | 0.00e+00 | **yes** |
| H2b (hard \| accurate) | holdout | `meta-llama/Llama-3.1-8B-Instruct` | -0.929 [-1.875, -0.000] | -0.929 [-1.875, -0.000] | 7 / 7 | 0.00e+00 | **yes** |
| H3a (easy \| accurate, neutral) | holdout | `meta-llama/Llama-3.1-8B-Instruct` | -1.838 [-2.713, -0.913] | -1.838 [-2.713, -0.913] | 10 / 10 | 0.00e+00 | **yes** |
| H3b (easy \| accurate, hostile) | holdout | `meta-llama/Llama-3.1-8B-Instruct` | -5.187 [-9.788, -1.462] | -5.187 [-9.788, -1.462] | 10 / 10 | 0.00e+00 | **yes** |

## primary model: `google/gemma-2-9b-it`

### Estimates under each treatment (M1 nats, 95% item-bootstrap CI)

| contrast | stratum | split | treatment | estimate [95% CI] | n items | CI excludes 0 |
| --- | --- | --- | --- | --- | ---: | :---: |
| H1 | easy \| neutral | discovery | available-case (published) | -3.800 [-5.297, -2.350] | 10 | yes |
| H1 | easy \| neutral | discovery | zero-imputation (0 nats) | -3.800 [-5.297, -2.350] | 10 | yes |
| H1 | easy \| neutral | discovery | bound: most negative | -3.800 [-5.297, -2.350] | 10 | yes |
| H1 | easy \| neutral | discovery | bound: most positive | -3.800 [-5.297, -2.350] | 10 | yes |
| H1_hard | hard \| neutral | discovery | available-case (published) | -1.230 [-2.922, 0.481] | 8 | no |
| H1_hard | hard \| neutral | discovery | zero-imputation (0 nats) | -0.984 [-2.447, 0.485] | 10 | no |
| H1_hard | hard \| neutral | discovery | bound: most negative | -6.078 [-13.141, -0.728] | 10 | yes |
| H1_hard | hard \| neutral | discovery | bound: most positive | 4.109 [-1.760, 11.755] | 10 | no |
| H2a | easy \| accurate | discovery | available-case (published) | -2.275 [-3.903, -1.000] | 10 | yes |
| H2a | easy \| accurate | discovery | zero-imputation (0 nats) | -2.275 [-3.903, -1.000] | 10 | yes |
| H2a | easy \| accurate | discovery | bound: most negative | -2.275 [-3.903, -1.000] | 10 | yes |
| H2a | easy \| accurate | discovery | bound: most positive | -2.275 [-3.903, -1.000] | 10 | yes |
| H2b | hard \| accurate | discovery | available-case (published) | -8.781 [-17.277, -1.268] | 7 | yes |
| H2b | hard \| accurate | discovery | zero-imputation (0 nats) | -6.772 [-13.857, -1.069] | 10 | yes |
| H2b | hard \| accurate | discovery | bound: most negative | -12.772 [-19.919, -5.772] | 10 | yes |
| H2b | hard \| accurate | discovery | bound: most positive | -0.037 [-10.908, 11.016] | 10 | no |
| tone_pooled | easy+hard \| accurate | discovery | available-case (published) | -4.954 [-9.303, -1.695] | 17 | yes |
| tone_pooled | easy+hard \| accurate | discovery | zero-imputation (0 nats) | -4.523 [-8.521, -1.616] | 20 | yes |
| tone_pooled | easy+hard \| accurate | discovery | bound: most negative | -7.523 [-12.233, -3.387] | 20 | yes |
| tone_pooled | easy+hard \| accurate | discovery | bound: most positive | -1.156 [-6.371, 3.969] | 20 | no |
| H3a | easy \| accurate, neutral | discovery | available-case (published) | -3.459 [-4.450, -2.612] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | zero-imputation (0 nats) | -3.459 [-4.450, -2.612] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | bound: most negative | -3.459 [-4.450, -2.612] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | bound: most positive | -3.459 [-4.450, -2.612] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | available-case (published) | -6.181 [-10.250, -2.250] | 9 | yes |
| H3b | easy \| accurate, hostile | discovery | zero-imputation (0 nats) | -6.869 [-10.619, -3.337] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most negative | -7.775 [-12.563, -3.293] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most positive | -5.228 [-9.263, -1.365] | 10 | yes |
| H1 | easy \| neutral | holdout | available-case (published) | -2.900 [-3.966, -1.844] | 10 | yes |
| H1 | easy \| neutral | holdout | zero-imputation (0 nats) | -2.900 [-3.966, -1.844] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most negative | -2.900 [-3.966, -1.844] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most positive | -2.900 [-3.966, -1.844] | 10 | yes |
| H1_hard | hard \| neutral | holdout | available-case (published) | -3.139 [-4.049, -2.222] | 9 | yes |
| H1_hard | hard \| neutral | holdout | zero-imputation (0 nats) | -4.238 [-6.832, -2.519] | 10 | yes |
| H1_hard | hard \| neutral | holdout | bound: most negative | -4.950 [-8.882, -2.513] | 10 | yes |
| H1_hard | hard \| neutral | holdout | bound: most positive | -2.616 [-3.863, -1.250] | 10 | yes |
| H2a | easy \| accurate | holdout | available-case (published) | -16.134 [-24.165, -5.744] | 7 | yes |
| H2a | easy \| accurate | holdout | zero-imputation (0 nats) | -15.594 [-21.872, -9.091] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most negative | -17.731 [-24.169, -10.700] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most positive | -10.728 [-19.128, -2.681] | 10 | yes |
| H2b | hard \| accurate | holdout | available-case (published) | -7.868 [-15.841, -1.896] | 9 | yes |
| H2b | hard \| accurate | holdout | zero-imputation (0 nats) | -8.588 [-15.689, -2.862] | 10 | yes |
| H2b | hard \| accurate | holdout | bound: most negative | -9.300 [-16.469, -3.630] | 10 | yes |
| H2b | hard \| accurate | holdout | bound: most positive | -6.966 [-14.325, -1.453] | 10 | yes |
| tone_pooled | easy+hard \| accurate | holdout | available-case (published) | -11.484 [-17.677, -6.093] | 16 | yes |
| tone_pooled | easy+hard \| accurate | holdout | zero-imputation (0 nats) | -12.091 [-16.625, -7.634] | 20 | yes |
| tone_pooled | easy+hard \| accurate | holdout | bound: most negative | -13.516 [-18.591, -8.611] | 20 | yes |
| tone_pooled | easy+hard \| accurate | holdout | bound: most positive | -8.847 [-14.513, -3.822] | 20 | yes |
| H3a | easy \| accurate, neutral | holdout | available-case (published) | -3.219 [-4.163, -2.288] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | zero-imputation (0 nats) | -3.219 [-4.163, -2.288] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | bound: most negative | -3.219 [-4.163, -2.288] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | bound: most positive | -3.219 [-4.163, -2.288] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | available-case (published) | -0.328 [-2.609, 1.344] | 4 | no |
| H3b | easy \| accurate, hostile | holdout | zero-imputation (0 nats) | 0.209 [-5.057, 5.551] | 10 | no |
| H3b | easy \| accurate, hostile | holdout | bound: most negative | -8.219 [-14.830, -1.265] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | bound: most positive | 10.456 [3.469, 17.826] | 10 | no |

### Missing values entering each contrast

| contrast | split | items in stratum | pairable | endpoint absent | available | treated missing (non-answer / candidate-absent) | reference missing (non-answer / candidate-absent) | both |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H1_hard | discovery | 10 | 10 | 0 | 8 | 2 (2 / 0) | 2 (2 / 0) | 2 |
| H2a | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H2b | discovery | 10 | 10 | 0 | 7 | 3 (3 / 0) | 2 (2 / 0) | 2 |
| tone_pooled | discovery | 20 | 20 | 0 | 17 | 3 (3 / 0) | 2 (2 / 0) | 2 |
| H3a | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H3b | discovery | 10 | 10 | 0 | 9 | 1 (1 / 0) | 0 (0 / 0) | 0 |
| H1 | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H1_hard | holdout | 10 | 10 | 0 | 9 | 1 (1 / 0) | 0 (0 / 0) | 0 |
| H2a | holdout | 10 | 10 | 0 | 7 | 3 (3 / 0) | 0 (0 / 0) | 0 |
| H2b | holdout | 10 | 10 | 0 | 9 | 1 (1 / 0) | 0 (0 / 0) | 0 |
| tone_pooled | holdout | 20 | 20 | 0 | 16 | 4 (4 / 0) | 0 (0 / 0) | 0 |
| H3a | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H3b | holdout | 10 | 10 | 0 | 4 | 5 (5 / 0) | 3 (3 / 0) | 2 |

### Non-answers per cell (the endpoints these contrasts read)

| split | cell | endpoint | endpoints | M1 observed | non-answer | candidate-absent | other | mean M1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery | `easy__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 12.36 |
| discovery | `easy__accurate__hostile` | onset | 10 | 9 | 1 | 0 | 0 | 6.10 |
| discovery | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 14.63 |
| discovery | `easy__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 11.17 |
| discovery | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 9.86 |
| discovery | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 10.83 |
| discovery | `hard__accurate__hostile` | measured | 10 | 7 | 3 | 0 | 0 | 1.32 |
| discovery | `hard__accurate__hostile` | onset | 10 | 5 | 5 | 0 | 0 | -1.24 |
| discovery | `hard__accurate__neutral` | measured | 10 | 8 | 2 | 0 | 0 | 9.62 |
| discovery | `hard__accurate__neutral` | onset | 10 | 8 | 2 | 0 | 0 | 7.90 |
| discovery | `hard__malfunctioning_always_fail__hostile` | measured | 10 | 8 | 2 | 0 | 0 | 6.34 |
| discovery | `hard__malfunctioning_always_fail__neutral` | measured | 10 | 8 | 2 | 0 | 0 | 8.39 |
| holdout | `easy__accurate__hostile` | measured | 10 | 7 | 3 | 0 | 0 | -1.27 |
| holdout | `easy__accurate__hostile` | onset | 10 | 5 | 5 | 0 | 0 | -1.36 |
| holdout | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 14.71 |
| holdout | `easy__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 11.49 |
| holdout | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 10.61 |
| holdout | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 11.81 |
| holdout | `hard__accurate__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 4.07 |
| holdout | `hard__accurate__hostile` | onset | 10 | 4 | 6 | 0 | 0 | 4.58 |
| holdout | `hard__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 12.25 |
| holdout | `hard__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 9.59 |
| holdout | `hard__malfunctioning_always_fail__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 9.61 |
| holdout | `hard__malfunctioning_always_fail__neutral` | measured | 10 | 9 | 1 | 0 | 0 | 8.90 |

### Tipping point

| contrast | split | delta (nats) | missing treated values | observed M1 in these cells | neutral-accurate support | reference-cell mean |
| --- | --- | ---: | ---: | --- | --- | ---: |
| H1 | discovery | n/a (`no_missing_treated_values`) | 0 | [7.63, 16.41] | [-9.06, 16.41] (n = 18) | 14.63 |
| H1_hard | discovery | 0.000 (already includes 0) | 2 | [-9.50, 15.87] | [-9.06, 16.41] (n = 18) | 9.62 |
| H2a | discovery | n/a (`no_missing_treated_values`) | 0 | [4.63, 16.41] | [-9.06, 16.41] (n = 18) | 14.63 |
| H2b | discovery | 2.803 | 3 | [-13.50, 15.87] | [-9.06, 16.41] (n = 18) | 9.62 |
| tone_pooled | discovery | 7.343 | 3 | [-13.50, 16.41] | [-9.06, 16.41] (n = 18) | 12.40 |
| H3a | discovery | n/a (`no_missing_treated_values`) | 0 | [8.87, 16.41] | [-9.06, 16.41] (n = 18) | 14.63 |
| H3b | discovery | 23.312 | 1 | [-12.19, 15.00] | [-9.06, 16.41] (n = 18) | 12.36 |
| H1 | holdout | n/a (`no_missing_treated_values`) | 0 | [7.56, 16.22] | [-7.12, 16.22] (n = 20) | 14.71 |
| H1_hard | holdout | 20.459 | 1 | [-11.56, 15.06] | [-7.12, 16.22] (n = 20) | 12.25 |
| H2a | holdout | 21.168 | 3 | [-12.69, 16.22] | [-7.12, 16.22] (n = 20) | 14.71 |
| H2b | holdout | 22.656 | 1 | [-14.31, 15.06] | [-7.12, 16.22] (n = 20) | 12.25 |
| tone_pooled | holdout | 28.976 | 4 | [-14.31, 16.22] | [-7.12, 16.22] (n = 20) | 13.48 |
| H3a | holdout | n/a (`no_missing_treated_values`) | 0 | [8.44, 16.22] | [-7.12, 16.22] (n = 20) | 14.71 |
| H3b | holdout | 0.000 (already includes 0) | 5 | [-12.69, 15.78] | [-7.12, 16.22] (n = 20) | -1.27 |

`delta` is the constant margin every missing treated-cell trial would have to carry
(with missing reference-cell trials at 0) for the item-paired 95% CI to include 0.
Read it against the observed M1 range in the same cells.

## control model: `Qwen/Qwen2.5-3B-Instruct`

### Estimates under each treatment (M1 nats, 95% item-bootstrap CI)

| contrast | stratum | split | treatment | estimate [95% CI] | n items | CI excludes 0 |
| --- | --- | --- | --- | --- | ---: | :---: |
| H1 | easy \| neutral | discovery | available-case (published) | 0.562 [-0.975, 2.513] | 10 | no |
| H1 | easy \| neutral | discovery | zero-imputation (0 nats) | 0.562 [-0.975, 2.513] | 10 | no |
| H1 | easy \| neutral | discovery | bound: most negative | 0.562 [-0.975, 2.513] | 10 | no |
| H1 | easy \| neutral | discovery | bound: most positive | 0.562 [-0.975, 2.513] | 10 | no |
| H1_hard | hard \| neutral | discovery | available-case (published) | -1.786 [-15.197, 9.215] | 7 | no |
| H1_hard | hard \| neutral | discovery | zero-imputation (0 nats) | -1.988 [-12.813, 7.462] | 10 | no |
| H1_hard | hard \| neutral | discovery | bound: most negative | -11.650 [-26.367, 1.863] | 10 | no |
| H1_hard | hard \| neutral | discovery | bound: most positive | 6.700 [-7.337, 19.813] | 10 | no |
| H2a | easy \| accurate | discovery | available-case (published) | 4.588 [-0.251, 12.813] | 10 | no |
| H2a | easy \| accurate | discovery | zero-imputation (0 nats) | 4.588 [-0.251, 12.813] | 10 | no |
| H2a | easy \| accurate | discovery | bound: most negative | 4.588 [-0.251, 12.813] | 10 | no |
| H2a | easy \| accurate | discovery | bound: most positive | 4.588 [-0.251, 12.813] | 10 | no |
| H2b | hard \| accurate | discovery | available-case (published) | -0.161 [-3.250, 4.179] | 7 | no |
| H2b | hard \| accurate | discovery | zero-imputation (0 nats) | -1.663 [-7.651, 3.463] | 10 | no |
| H2b | hard \| accurate | discovery | bound: most negative | -11.325 [-24.963, -0.412] | 10 | yes |
| H2b | hard \| accurate | discovery | bound: most positive | 7.025 [-1.763, 17.925] | 10 | no |
| tone_pooled | easy+hard \| accurate | discovery | available-case (published) | 2.632 [-0.853, 7.912] | 17 | no |
| tone_pooled | easy+hard \| accurate | discovery | zero-imputation (0 nats) | 1.462 [-3.050, 6.676] | 20 | no |
| tone_pooled | easy+hard \| accurate | discovery | bound: most negative | -3.369 [-12.114, 4.375] | 20 | no |
| tone_pooled | easy+hard \| accurate | discovery | bound: most positive | 5.806 [0.480, 12.377] | 20 | no |
| H3a | easy \| accurate, neutral | discovery | available-case (published) | 4.042 [1.986, 6.278] | 9 | no |
| H3a | easy \| accurate, neutral | discovery | zero-imputation (0 nats) | 1.525 [-4.150, 5.625] | 10 | no |
| H3a | easy \| accurate, neutral | discovery | bound: most negative | -0.525 [-10.565, 5.601] | 10 | no |
| H3a | easy \| accurate, neutral | discovery | bound: most positive | 4.062 [2.150, 6.175] | 10 | no |
| H3b | easy \| accurate, hostile | discovery | available-case (published) | -2.125 [-3.264, -0.889] | 9 | yes |
| H3b | easy \| accurate, hostile | discovery | zero-imputation (0 nats) | -3.813 [-7.587, -1.263] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most negative | -5.863 [-13.626, -1.262] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most positive | -1.275 [-2.925, 0.812] | 10 | no |
| H1 | easy \| neutral | holdout | available-case (published) | -9.475 [-19.891, -1.462] | 10 | yes |
| H1 | easy \| neutral | holdout | zero-imputation (0 nats) | -9.475 [-19.891, -1.462] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most negative | -9.475 [-19.891, -1.462] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most positive | -9.475 [-19.891, -1.462] | 10 | yes |
| H1_hard | hard \| neutral | holdout | available-case (published) | -6.417 [-15.570, 0.750] | 9 | no |
| H1_hard | hard \| neutral | holdout | zero-imputation (0 nats) | -3.338 [-12.901, 5.838] | 10 | no |
| H1_hard | hard \| neutral | holdout | bound: most negative | -6.188 [-15.100, 0.338] | 10 | no |
| H1_hard | hard \| neutral | holdout | bound: most positive | -4.638 [-13.691, 2.738] | 10 | no |
| H2a | easy \| accurate | holdout | available-case (published) | -5.150 [-14.463, -0.150] | 10 | yes |
| H2a | easy \| accurate | holdout | zero-imputation (0 nats) | -5.150 [-14.463, -0.150] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most negative | -5.150 [-14.463, -0.150] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most positive | -5.150 [-14.463, -0.150] | 10 | yes |
| H2b | hard \| accurate | holdout | available-case (published) | 0.653 [-4.931, 4.917] | 9 | no |
| H2b | hard \| accurate | holdout | zero-imputation (0 nats) | 3.175 [-3.413, 10.030] | 10 | no |
| H2b | hard \| accurate | holdout | bound: most negative | 0.325 [-4.913, 4.488] | 10 | no |
| H2b | hard \| accurate | holdout | bound: most positive | 1.875 [-3.626, 6.464] | 10 | no |
| tone_pooled | easy+hard \| accurate | holdout | available-case (published) | -2.401 [-8.711, 1.836] | 19 | no |
| tone_pooled | easy+hard \| accurate | holdout | zero-imputation (0 nats) | -0.988 [-6.888, 4.319] | 20 | no |
| tone_pooled | easy+hard \| accurate | holdout | bound: most negative | -2.413 [-7.732, 1.619] | 20 | no |
| tone_pooled | easy+hard \| accurate | holdout | bound: most positive | -1.638 [-7.231, 2.731] | 20 | no |
| H3a | easy \| accurate, neutral | holdout | available-case (published) | 1.625 [0.162, 2.888] | 10 | no |
| H3a | easy \| accurate, neutral | holdout | zero-imputation (0 nats) | 1.625 [0.162, 2.888] | 10 | no |
| H3a | easy \| accurate, neutral | holdout | bound: most negative | 1.625 [0.162, 2.888] | 10 | no |
| H3a | easy \| accurate, neutral | holdout | bound: most positive | 1.625 [0.162, 2.888] | 10 | no |
| H3b | easy \| accurate, hostile | holdout | available-case (published) | -1.600 [-3.450, 0.525] | 10 | no |
| H3b | easy \| accurate, hostile | holdout | zero-imputation (0 nats) | -1.600 [-3.450, 0.525] | 10 | no |
| H3b | easy \| accurate, hostile | holdout | bound: most negative | -1.600 [-3.450, 0.525] | 10 | no |
| H3b | easy \| accurate, hostile | holdout | bound: most positive | -1.600 [-3.450, 0.525] | 10 | no |

### Missing values entering each contrast

| contrast | split | items in stratum | pairable | endpoint absent | available | treated missing (non-answer / candidate-absent) | reference missing (non-answer / candidate-absent) | both |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H1_hard | discovery | 10 | 10 | 0 | 7 | 1 (1 / 0) | 3 (3 / 0) | 1 |
| H2a | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H2b | discovery | 10 | 10 | 0 | 7 | 1 (1 / 0) | 3 (3 / 0) | 1 |
| tone_pooled | discovery | 20 | 20 | 0 | 17 | 1 (1 / 0) | 3 (3 / 0) | 1 |
| H3a | discovery | 10 | 10 | 0 | 9 | 1 (0 / 1) | 0 (0 / 0) | 0 |
| H3b | discovery | 10 | 10 | 0 | 9 | 1 (0 / 1) | 0 (0 / 0) | 0 |
| H1 | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H1_hard | holdout | 10 | 10 | 0 | 9 | 0 (0 / 0) | 1 (1 / 0) | 0 |
| H2a | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H2b | holdout | 10 | 10 | 0 | 9 | 0 (0 / 0) | 1 (1 / 0) | 0 |
| tone_pooled | holdout | 20 | 20 | 0 | 19 | 0 (0 / 0) | 1 (1 / 0) | 0 |
| H3a | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H3b | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |

### Non-answers per cell (the endpoints these contrasts read)

| split | cell | endpoint | endpoints | M1 observed | non-answer | candidate-absent | other | mean M1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery | `easy__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 22.06 |
| discovery | `easy__accurate__hostile` | onset | 10 | 9 | 0 | 1 | 0 | 20.28 |
| discovery | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 17.47 |
| discovery | `easy__accurate__neutral` | onset | 10 | 9 | 0 | 1 | 0 | 21.11 |
| discovery | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 17.81 |
| discovery | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 18.04 |
| discovery | `hard__accurate__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 4.97 |
| discovery | `hard__accurate__hostile` | onset | 10 | 6 | 4 | 0 | 0 | 6.85 |
| discovery | `hard__accurate__neutral` | measured | 10 | 7 | 3 | 0 | 0 | 8.77 |
| discovery | `hard__accurate__neutral` | onset | 10 | 7 | 3 | 0 | 0 | 10.48 |
| discovery | `hard__malfunctioning_always_fail__hostile` | measured | 10 | 8 | 2 | 0 | 0 | -0.89 |
| discovery | `hard__malfunctioning_always_fail__neutral` | measured | 10 | 9 | 1 | 0 | 0 | 4.61 |
| holdout | `easy__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 18.15 |
| holdout | `easy__accurate__hostile` | onset | 10 | 10 | 0 | 0 | 0 | 16.55 |
| holdout | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 23.30 |
| holdout | `easy__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 24.93 |
| holdout | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 18.32 |
| holdout | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 13.83 |
| holdout | `hard__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 20.04 |
| holdout | `hard__accurate__hostile` | onset | 10 | 9 | 1 | 0 | 0 | 19.26 |
| holdout | `hard__accurate__neutral` | measured | 10 | 9 | 1 | 0 | 0 | 18.74 |
| holdout | `hard__accurate__neutral` | onset | 10 | 7 | 2 | 1 | 0 | 19.00 |
| holdout | `hard__malfunctioning_always_fail__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 15.38 |
| holdout | `hard__malfunctioning_always_fail__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 13.52 |

### Tipping point

| contrast | split | delta (nats) | missing treated values | observed M1 in these cells | neutral-accurate support | reference-cell mean |
| --- | --- | ---: | ---: | --- | --- | ---: |
| H1 | discovery | 0.000 (already includes 0) | 0 | [-20.50, 25.62] | [-20.50, 25.38] (n = 17) | 17.47 |
| H1_hard | discovery | 0.000 (already includes 0) | 1 | [-26.50, 22.62] | [-20.50, 25.38] (n = 17) | 8.77 |
| H2a | discovery | 0.000 (already includes 0) | 0 | [-20.50, 26.12] | [-20.50, 25.38] (n = 17) | 17.47 |
| H2b | discovery | 0.000 (already includes 0) | 1 | [-24.00, 22.62] | [-20.50, 25.38] (n = 17) | 8.77 |
| tone_pooled | discovery | 0.000 (already includes 0) | 1 | [-24.00, 26.12] | [-20.50, 25.38] (n = 17) | 13.89 |
| H3a | discovery | 0.000 (already includes 0) | 1 | [-20.50, 29.50] | [-20.50, 25.38] (n = 17) | 17.47 |
| H3b | discovery | 22.469 | 1 | [17.25, 26.12] | [-20.50, 25.38] (n = 17) | 22.06 |
| H1 | holdout | n/a (`no_missing_treated_values`) | 0 | [-24.75, 28.50] | [13.00, 28.50] (n = 19) | 23.30 |
| H1_hard | holdout | 0.000 (already includes 0) | 0 | [-16.62, 26.12] | [13.00, 28.50] (n = 19) | 18.74 |
| H2a | holdout | n/a (`no_missing_treated_values`) | 0 | [-24.75, 28.50] | [13.00, 28.50] (n = 19) | 23.30 |
| H2b | holdout | 0.000 (already includes 0) | 0 | [-0.25, 26.62] | [13.00, 28.50] (n = 19) | 18.74 |
| tone_pooled | holdout | 0.000 (already includes 0) | 0 | [-24.75, 28.50] | [13.00, 28.50] (n = 19) | 21.14 |
| H3a | holdout | n/a (`no_missing_treated_values`) | 0 | [16.38, 31.88] | [13.00, 28.50] (n = 19) | 23.30 |
| H3b | holdout | 0.000 (already includes 0) | 0 | [-24.75, 26.00] | [13.00, 28.50] (n = 19) | 18.15 |

`delta` is the constant margin every missing treated-cell trial would have to carry
(with missing reference-cell trials at 0) for the item-paired 95% CI to include 0.
Read it against the observed M1 range in the same cells.

## extension model: `meta-llama/Llama-3.1-8B-Instruct`

### Estimates under each treatment (M1 nats, 95% item-bootstrap CI)

| contrast | stratum | split | treatment | estimate [95% CI] | n items | CI excludes 0 |
| --- | --- | --- | --- | --- | ---: | :---: |
| H1 | easy \| neutral | discovery | available-case (published) | -6.516 [-8.970, -4.281] | 8 | yes |
| H1 | easy \| neutral | discovery | zero-imputation (0 nats) | -8.037 [-10.475, -5.487] | 10 | yes |
| H1 | easy \| neutral | discovery | bound: most negative | -6.163 [-8.038, -4.375] | 10 | yes |
| H1 | easy \| neutral | discovery | bound: most positive | -5.162 [-7.726, -2.775] | 10 | yes |
| H1_hard | hard \| neutral | discovery | available-case (published) | -3.969 [-5.719, -1.625] | 4 | yes |
| H1_hard | hard \| neutral | discovery | zero-imputation (0 nats) | -5.403 [-8.306, -2.597] | 9 | yes |
| H1_hard | hard \| neutral | discovery | bound: most negative | -3.389 [-4.653, -2.014] | 9 | yes |
| H1_hard | hard \| neutral | discovery | bound: most positive | 0.500 [-2.376, 3.056] | 9 | no |
| H2a | easy \| accurate | discovery | available-case (published) | -2.712 [-6.250, -0.575] | 10 | yes |
| H2a | easy \| accurate | discovery | zero-imputation (0 nats) | -2.712 [-6.250, -0.575] | 10 | yes |
| H2a | easy \| accurate | discovery | bound: most negative | -2.712 [-6.250, -0.575] | 10 | yes |
| H2a | easy \| accurate | discovery | bound: most positive | -2.712 [-6.250, -0.575] | 10 | yes |
| H2b | hard \| accurate | discovery | available-case (published) | -0.714 [-1.286, -0.035] | 7 | yes |
| H2b | hard \| accurate | discovery | zero-imputation (0 nats) | -0.556 [-1.069, 0.014] | 9 | no |
| H2b | hard \| accurate | discovery | bound: most negative | -1.667 [-2.972, -0.528] | 9 | yes |
| H2b | hard \| accurate | discovery | bound: most positive | 0.556 [-0.875, 2.306] | 9 | no |
| tone_pooled | easy+hard \| accurate | discovery | available-case (published) | -1.890 [-4.022, -0.596] | 17 | yes |
| tone_pooled | easy+hard \| accurate | discovery | zero-imputation (0 nats) | -1.691 [-3.665, -0.506] | 19 | yes |
| tone_pooled | easy+hard \| accurate | discovery | bound: most negative | -2.217 [-4.211, -0.862] | 19 | yes |
| tone_pooled | easy+hard \| accurate | discovery | bound: most positive | -1.164 [-3.217, 0.520] | 19 | no |
| H3a | easy \| accurate, neutral | discovery | available-case (published) | -1.800 [-2.225, -1.363] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | zero-imputation (0 nats) | -1.800 [-2.225, -1.363] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | bound: most negative | -1.800 [-2.225, -1.363] | 10 | yes |
| H3a | easy \| accurate, neutral | discovery | bound: most positive | -1.800 [-2.225, -1.363] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | available-case (published) | -1.469 [-2.619, -0.331] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | zero-imputation (0 nats) | -1.469 [-2.619, -0.331] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most negative | -1.469 [-2.619, -0.331] | 10 | yes |
| H3b | easy \| accurate, hostile | discovery | bound: most positive | -1.469 [-2.619, -0.331] | 10 | yes |
| H1 | easy \| neutral | holdout | available-case (published) | -8.278 [-12.654, -4.957] | 9 | yes |
| H1 | easy \| neutral | holdout | zero-imputation (0 nats) | -8.563 [-12.500, -5.100] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most negative | -7.725 [-12.090, -4.512] | 10 | yes |
| H1 | easy \| neutral | holdout | bound: most positive | -7.038 [-11.214, -3.037] | 10 | yes |
| H1_hard | hard \| neutral | holdout | available-case (published) | -13.500 [-18.375, -10.125] | 3 | yes |
| H1_hard | hard \| neutral | holdout | zero-imputation (0 nats) | -10.975 [-13.362, -7.937] | 10 | yes |
| H1_hard | hard \| neutral | holdout | bound: most negative | -9.000 [-14.613, -4.575] | 10 | yes |
| H1_hard | hard \| neutral | holdout | bound: most positive | -3.500 [-9.900, 2.551] | 10 | no |
| H2a | easy \| accurate | holdout | available-case (published) | -1.063 [-1.700, -0.475] | 10 | yes |
| H2a | easy \| accurate | holdout | zero-imputation (0 nats) | -1.063 [-1.700, -0.475] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most negative | -1.063 [-1.700, -0.475] | 10 | yes |
| H2a | easy \| accurate | holdout | bound: most positive | -1.063 [-1.700, -0.475] | 10 | yes |
| H2b | hard \| accurate | holdout | available-case (published) | -0.929 [-1.875, -0.000] | 7 | yes |
| H2b | hard \| accurate | holdout | zero-imputation (0 nats) | -1.488 [-3.125, -0.163] | 10 | yes |
| H2b | hard \| accurate | holdout | bound: most negative | -2.025 [-3.788, -0.412] | 10 | yes |
| H2b | hard \| accurate | holdout | bound: most positive | 1.412 [-0.813, 3.775] | 10 | no |
| tone_pooled | easy+hard \| accurate | holdout | available-case (published) | -1.007 [-1.559, -0.485] | 17 | yes |
| tone_pooled | easy+hard \| accurate | holdout | zero-imputation (0 nats) | -1.275 [-2.244, -0.500] | 20 | yes |
| tone_pooled | easy+hard \| accurate | holdout | bound: most negative | -1.544 [-2.544, -0.706] | 20 | yes |
| tone_pooled | easy+hard \| accurate | holdout | bound: most positive | 0.175 [-1.056, 1.513] | 20 | no |
| H3a | easy \| accurate, neutral | holdout | available-case (published) | -1.838 [-2.713, -0.913] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | zero-imputation (0 nats) | -1.838 [-2.713, -0.913] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | bound: most negative | -1.838 [-2.713, -0.913] | 10 | yes |
| H3a | easy \| accurate, neutral | holdout | bound: most positive | -1.838 [-2.713, -0.913] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | available-case (published) | -5.187 [-9.788, -1.462] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | zero-imputation (0 nats) | -5.187 [-9.788, -1.462] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | bound: most negative | -5.187 [-9.788, -1.462] | 10 | yes |
| H3b | easy \| accurate, hostile | holdout | bound: most positive | -5.187 [-9.788, -1.462] | 10 | yes |

### Missing values entering each contrast

| contrast | split | items in stratum | pairable | endpoint absent | available | treated missing (non-answer / candidate-absent) | reference missing (non-answer / candidate-absent) | both |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | discovery | 10 | 10 | 0 | 8 | 2 (2 / 0) | 0 (0 / 0) | 0 |
| H1_hard | discovery | 9 | 9 | 0 | 4 | 5 (5 / 0) | 2 (2 / 0) | 2 |
| H2a | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H2b | discovery | 9 | 9 | 0 | 7 | 2 (2 / 0) | 2 (2 / 0) | 2 |
| tone_pooled | discovery | 19 | 19 | 0 | 17 | 2 (2 / 0) | 2 (2 / 0) | 2 |
| H3a | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H3b | discovery | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H1 | holdout | 10 | 10 | 0 | 9 | 1 (1 / 0) | 0 (0 / 0) | 0 |
| H1_hard | holdout | 10 | 10 | 0 | 3 | 6 (6 / 0) | 2 (2 / 0) | 1 |
| H2a | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H2b | holdout | 10 | 10 | 0 | 7 | 3 (3 / 0) | 2 (2 / 0) | 2 |
| tone_pooled | holdout | 20 | 20 | 0 | 17 | 3 (3 / 0) | 2 (2 / 0) | 2 |
| H3a | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |
| H3b | holdout | 10 | 10 | 0 | 10 | 0 (0 / 0) | 0 (0 / 0) | 0 |

### Non-answers per cell (the endpoints these contrasts read)

| split | cell | endpoint | endpoints | M1 observed | non-answer | candidate-absent | other | mean M1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery | `easy__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 9.36 |
| discovery | `easy__accurate__hostile` | onset | 10 | 10 | 0 | 0 | 0 | 7.89 |
| discovery | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 12.08 |
| discovery | `easy__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 10.27 |
| discovery | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 4.85 |
| discovery | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 8 | 2 | 0 | 0 | 5.05 |
| discovery | `hard__accurate__hostile` | measured | 9 | 7 | 2 | 0 | 0 | 10.77 |
| discovery | `hard__accurate__hostile` | onset | 9 | 5 | 4 | 0 | 0 | 5.83 |
| discovery | `hard__accurate__neutral` | measured | 9 | 7 | 2 | 0 | 0 | 11.48 |
| discovery | `hard__accurate__neutral` | onset | 9 | 6 | 3 | 0 | 0 | 7.37 |
| discovery | `hard__malfunctioning_always_fail__hostile` | measured | 9 | 5 | 4 | 0 | 0 | 7.20 |
| discovery | `hard__malfunctioning_always_fail__neutral` | measured | 9 | 4 | 5 | 0 | 0 | 7.94 |
| holdout | `easy__accurate__hostile` | measured | 10 | 10 | 0 | 0 | 0 | 11.60 |
| holdout | `easy__accurate__hostile` | onset | 10 | 10 | 0 | 0 | 0 | 6.41 |
| holdout | `easy__accurate__neutral` | measured | 10 | 10 | 0 | 0 | 0 | 12.66 |
| holdout | `easy__accurate__neutral` | onset | 10 | 10 | 0 | 0 | 0 | 10.82 |
| holdout | `easy__malfunctioning_always_fail__hostile` | measured | 10 | 9 | 1 | 0 | 0 | 2.47 |
| holdout | `easy__malfunctioning_always_fail__neutral` | measured | 10 | 9 | 1 | 0 | 0 | 4.56 |
| holdout | `hard__accurate__hostile` | measured | 10 | 7 | 3 | 0 | 0 | 11.80 |
| holdout | `hard__accurate__hostile` | onset | 10 | 6 | 4 | 0 | 0 | 4.52 |
| holdout | `hard__accurate__neutral` | measured | 10 | 8 | 2 | 0 | 0 | 12.19 |
| holdout | `hard__accurate__neutral` | onset | 10 | 8 | 2 | 0 | 0 | 7.81 |
| holdout | `hard__malfunctioning_always_fail__hostile` | measured | 10 | 5 | 5 | 0 | 0 | 1.40 |
| holdout | `hard__malfunctioning_always_fail__neutral` | measured | 10 | 4 | 6 | 0 | 0 | -3.06 |

### Tipping point

| contrast | split | delta (nats) | missing treated values | observed M1 in these cells | neutral-accurate support | reference-cell mean |
| --- | --- | ---: | ---: | --- | --- | ---: |
| H1 | discovery | 19.949 | 2 | [0.25, 14.38] | [9.37, 14.38] (n = 17) | 12.08 |
| H1_hard | discovery | 4.275 | 5 | [7.25, 14.13] | [9.37, 14.38] (n = 17) | 11.48 |
| H2a | discovery | n/a (`no_missing_treated_values`) | 0 | [-6.37, 14.38] | [9.37, 14.38] (n = 17) | 12.08 |
| H2b | discovery | 0.001 | 2 | [8.38, 14.13] | [9.37, 14.38] (n = 17) | 11.48 |
| tone_pooled | discovery | 2.688 | 2 | [-6.37, 14.38] | [9.37, 14.38] (n = 17) | 11.83 |
| H3a | discovery | n/a (`no_missing_treated_values`) | 0 | [7.25, 14.38] | [9.37, 14.38] (n = 17) | 12.08 |
| H3b | discovery | n/a (`no_missing_treated_values`) | 0 | [-6.37, 13.50] | [9.37, 14.38] (n = 17) | 9.36 |
| H1 | holdout | 27.748 | 1 | [-12.37, 15.25] | [8.38, 15.25] (n = 18) | 12.66 |
| H1_hard | holdout | 10.035 | 6 | [-12.38, 14.38] | [8.38, 15.25] (n = 18) | 12.19 |
| H2a | holdout | n/a (`no_missing_treated_values`) | 0 | [8.37, 15.25] | [8.38, 15.25] (n = 18) | 12.66 |
| H2b | holdout | 0.584 | 3 | [8.38, 14.38] | [8.38, 15.25] (n = 18) | 12.19 |
| tone_pooled | holdout | 3.083 | 3 | [8.37, 15.25] | [8.38, 15.25] (n = 18) | 12.45 |
| H3a | holdout | n/a (`no_missing_treated_values`) | 0 | [8.37, 15.25] | [8.38, 15.25] (n = 18) | 12.66 |
| H3b | holdout | n/a (`no_missing_treated_values`) | 0 | [-7.25, 14.25] | [8.38, 15.25] (n = 18) | 11.60 |

`delta` is the constant margin every missing treated-cell trial would have to carry
(with missing reference-cell trials at 0) for the item-paired 95% CI to include 0.
Read it against the observed M1 range in the same cells.

## What this analysis does not settle

- Zero-imputation is an *assumption*, not a measurement: a non-answer carries no
  margin at all, and 0 nats is the indifference point, not an observed value.
- The worst-case bounds are worst-case only *inside the observed neutral-accurate
  support*. A missing trial whose true margin lay outside that range would sit
  outside the interval, and the interval says nothing about why the value is
  missing -- the MNAR mechanism itself is reported (H9), not modelled.
- When 0 nats lies inside that support the zero-imputation estimate is bracketed by
  the two bounds by construction, so agreement between them is not independent
  evidence. It does not hold for Llama-3.1-8B-Instruct/discovery [9.37, 14.38]; Llama-3.1-8B-Instruct/holdout [8.38, 15.25]; Qwen2.5-3B-Instruct/holdout [13.00, 28.50], where 0 is outside the support.
- `m1_candidate_absent_*` endpoints are imputed on the same footing as non-answers
  even though the response did commit to a letter; the counts table shows how many
  of each entered every contrast so the two can be read apart.
- Nothing here is confirmatory, and no published estimate is amended by it.
