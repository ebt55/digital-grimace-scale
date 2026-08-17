# Phase 3 - direction-specificity steering

Preregistration: `notes/preregistration_v4_phase3.md`. Model `google/gemma-2-9b-it`, layer **L\* = 6**.

- C2 (2026-08-18): the dose unit is ||d|| itself, not the mean activation norm
- dose: `alpha * d (d unnormalised); every control at the matched norm alpha * ||d||`
- **||d|| = 3.12**, mean activation L2 norm at L\* = 78.59, **ratio ||d|| / mean-norm = 0.0398**
- tone direction: mean(hostile) - mean(neutral) at L*, discovery accurate arm, measured position
- controls: 5 unit gaussian directions, seeds DGS-AC1-STEER-v1|1..5, rescaled to ||d||; mean(style__verbose) - mean(style__neutral_reference) at L*, rescaled to ||d||
- items: 20 holdout tasks, render_task, neutral single-turn, holdout split, greedy, 512 new tokens
- a dose is degenerate when > 50% of items yield no parseable answer

## Readouts by direction and dose (paired against alpha = 0)

| direction | alpha | items | mean M1 (n) | dM1 [95% CI] | non-answer | d non-answer [95% CI] | mean length | degenerate |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: | :---: |
| `baseline` | 0 | 20 | 10.597 (20) | 0.000 [0.000, 0.000] | 0.00 | 0.000 [0.000, 0.000] | 123.7 | no |
| `random1` | 0.5 | 20 | 10.584 (20) | -0.012 [-0.055, 0.031] | 0.00 | 0.000 [0.000, 0.000] | 124.0 | no |
| `random1` | 1 | 20 | 11.675 (20) | 1.078 [-0.048, 3.216] | 0.00 | 0.000 [0.000, 0.000] | 123.4 | no |
| `random1` | 2 | 20 | 11.561 (20) | 0.964 [-0.361, 3.236] | 0.00 | 0.000 [0.000, 0.000] | 123.9 | no |
| `random1` | 4 | 20 | 11.420 (20) | 0.823 [-0.531, 3.102] | 0.00 | 0.000 [0.000, 0.000] | 123.7 | no |
| `random2` | 0.5 | 20 | 11.642 (20) | 1.045 [-0.039, 3.147] | 0.00 | 0.000 [0.000, 0.000] | 123.3 | no |
| `random2` | 1 | 20 | 11.673 (20) | 1.077 [-0.005, 3.184] | 0.00 | 0.000 [0.000, 0.000] | 123.8 | no |
| `random2` | 2 | 20 | 11.677 (20) | 1.080 [-0.089, 3.236] | 0.00 | 0.000 [0.000, 0.000] | 123.4 | no |
| `random2` | 4 | 20 | 11.792 (20) | 1.195 [-0.080, 3.372] | 0.00 | 0.000 [0.000, 0.000] | 128.1 | no |
| `random3` | 0.5 | 20 | 11.495 (20) | 0.898 [-0.328, 3.088] | 0.00 | 0.000 [0.000, 0.000] | 122.8 | no |
| `random3` | 1 | 20 | 11.611 (20) | 1.014 [-0.077, 3.136] | 0.00 | 0.000 [0.000, 0.000] | 123.5 | no |
| `random3` | 2 | 20 | 11.458 (20) | 0.861 [-0.375, 3.070] | 0.00 | 0.000 [0.000, 0.000] | 126.0 | no |
| `random3` | 4 | 20 | 11.528 (20) | 0.931 [-0.352, 3.153] | 0.00 | 0.000 [0.000, 0.000] | 126.7 | no |
| `random4` | 0.5 | 20 | 10.572 (20) | -0.025 [-0.097, 0.033] | 0.00 | 0.000 [0.000, 0.000] | 123.7 | no |
| `random4` | 1 | 20 | 10.591 (20) | -0.006 [-0.084, 0.080] | 0.00 | 0.000 [0.000, 0.000] | 124.2 | no |
| `random4` | 2 | 20 | 10.605 (20) | 0.008 [-0.103, 0.133] | 0.00 | 0.000 [0.000, 0.000] | 124.0 | no |
| `random4` | 4 | 20 | 10.633 (20) | 0.036 [-0.156, 0.269] | 0.00 | 0.000 [0.000, 0.000] | 126.8 | no |
| `random5` | 0.5 | 20 | 11.633 (20) | 1.036 [-0.044, 3.125] | 0.00 | 0.000 [0.000, 0.000] | 123.1 | no |
| `random5` | 1 | 20 | 11.044 (20) | 0.447 [-0.066, 1.381] | 0.00 | 0.000 [0.000, 0.000] | 123.2 | no |
| `random5` | 2 | 20 | 11.473 (20) | 0.877 [-0.217, 2.891] | 0.00 | 0.000 [0.000, 0.000] | 129.4 | no |
| `random5` | 4 | 20 | 10.898 (20) | 0.302 [-0.595, 1.653] | 0.00 | 0.000 [0.000, 0.000] | 126.0 | no |
| `tone` | 0.5 | 20 | 10.592 (20) | -0.005 [-0.044, 0.034] | 0.00 | 0.000 [0.000, 0.000] | 124.2 | no |
| `tone` | 1 | 20 | 10.558 (20) | -0.039 [-0.103, 0.014] | 0.00 | 0.000 [0.000, 0.000] | 123.8 | no |
| `tone` | 2 | 20 | 10.403 (20) | -0.194 [-0.513, 0.000] | 0.00 | 0.000 [0.000, 0.000] | 123.7 | no |
| `tone` | 4 | 20 | 10.103 (20) | -0.494 [-0.869, -0.178] | 0.00 | 0.000 [0.000, 0.000] | 127.5 | no |
| `unrelated_style` | 0.5 | 20 | 11.458 (20) | 0.861 [-0.544, 3.128] | 0.00 | 0.000 [0.000, 0.000] | 124.5 | no |
| `unrelated_style` | 1 | 20 | 11.436 (20) | 0.839 [-0.280, 2.819] | 0.00 | 0.000 [0.000, 0.000] | 123.2 | no |
| `unrelated_style` | 2 | 20 | 11.252 (20) | 0.655 [-0.622, 2.780] | 0.00 | 0.000 [0.000, 0.000] | 124.8 | no |
| `unrelated_style` | 4 | 20 | 11.348 (20) | 0.752 [-0.505, 2.824] | 0.00 | 0.000 [0.000, 0.000] | 123.0 | no |

Monotonicity of mean M1 over the non-degenerate doses [0.5, 1.0, 2.0]: **yes**.

## Judge distress (locked rubric, temperature 0), paired against alpha = 0

| direction | alpha | d distress [95% CI] | items |
| --- | ---: | --- | ---: |
| `random1` | 2 | 0.000 [0.000, 0.000] | 20 |
| `random2` | 2 | 0.000 [0.000, 0.000] | 20 |
| `random3` | 2 | 0.000 [0.000, 0.000] | 20 |
| `random4` | 2 | 0.000 [0.000, 0.000] | 20 |
| `random5` | 2 | 0.000 [0.000, 0.000] | 20 |
| `tone` | 2 | 0.000 [0.000, 0.000] | 20 |
| `tone` | 4 | 0.000 [0.000, 0.000] | 20 |
| `unrelated_style` | 2 | 0.000 [0.000, 0.000] | 20 |

## Predictions

| ID | prediction | verdict | detail |
| --- | --- | :---: | --- |
| J4 | Steering with the tone direction lowers M1 on neutral holdout items dose-dependently: M1(alpha=2) - M1(alpha=0) < 0 with item-bootstrap 95% CI excluding 0, and monotone in alpha over {0.5, 1, 2}. | not supported | alpha2_ci95 = [-0.5125, 8.923e-08]; alpha2_ci_excludes_zero_negative = False; alpha2_degenerate = False; alpha2_m1_delta = -0.1937; doses_used = [0.5, 1, 2]; monotone_over_used_doses = True; monotonicity_note = None |
| J5 | None of the 5 random matched-norm directions produces an M1 drop with CI excluding 0 at alpha=2 (direction specificity); the unrelated semantic direction does not either. | **supported** | alpha = 2; directions_checked = [random1, random2, random3, random4, random5, unrelated_style]; directions_with_supported_m1_drop = [] |
| J6 | Tone steering at alpha >= 2 raises the non-answer rate and/or judge distress on neutral items (either CI excluding 0). | not supported | distress_increase_at = []; distress_judged = True; non_answer_rate_increase_at = [] |

> A probe plus induction result demonstrates a condition-linked internal variable with causal influence on the output signature. It is not evidence of experience.
