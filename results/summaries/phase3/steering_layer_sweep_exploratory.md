# Phase 3 - EXPLORATORY layer sweep (tone direction at layers 20, 30)

> **EXPLORATORY. Chosen after the confirmatory verdicts were fixed, because the frozen tie-break forced L* = 6 (the earliest layer of a 6-25 AUC plateau) where ||d|| is only 4% of the mean activation norm. It changes no preregistered verdict and supports no hypothesis test.**

- dose: `alpha * d (C2, d unnormalised); controls rescaled to alpha * ||d_layer||`
- tone direction: mean(hostile) - mean(neutral) recomputed at each swept layer, discovery accurate arm, measured position
- baseline: the shared unsteered alpha = 0 arm from the confirmatory run (no intervention, so it is layer-independent)
- controls: DGS-AC1-STEER-v1|1..2 (the confirmatory run's first two), at alpha = 4 only
- items: 20 holdout tasks, render_task, neutral single-turn, holdout split, greedy, 512 new tokens

## Dose unit by layer

| layer | ||d|| | mean activation norm | ratio |
| ---: | ---: | ---: | ---: |
| 6 (confirmatory L\*) | 3.12 | 78.59 | 0.0398 |
| 20 | 31.17 | 250.16 | 0.1246 |
| 30 | 157.77 | 444.07 | 0.3553 |

## M1 against dose, paired by item against the shared alpha = 0 baseline

| direction | layer | alpha | items | mean M1 (n) | dM1 [95% CI] | non-answer | degenerate |
| --- | ---: | ---: | ---: | --- | --- | ---: | :---: |
| `tone` (confirmatory) | 6 | 0.5 | 20 | 10.592 (20) | -0.005 [-0.044, 0.034] | 0.00 | no |
| `tone` (confirmatory) | 6 | 1 | 20 | 10.558 (20) | -0.039 [-0.103, 0.014] | 0.00 | no |
| `tone` (confirmatory) | 6 | 2 | 20 | 10.403 (20) | -0.194 [-0.513, 0.000] | 0.00 | no |
| `tone` (confirmatory) | 6 | 4 | 20 | 10.103 (20) | -0.494 [-0.869, -0.178] | 0.00 | no |
| `tone_L20` | 20 | 1 | 20 | 10.977 (20) | 0.380 [-0.366, 1.355] | 0.00 | no |
| `tone_L20` | 20 | 2 | 20 | 11.141 (19) | 0.711 [-0.447, 2.574] | 0.05 | no |
| `tone_L20` | 20 | 4 | 20 | 8.963 (20) | -1.634 [-3.783, 0.072] | 0.00 | no |
| `random_L20_1` | 20 | 4 | 20 | 5.733 (20) | -4.864 [-8.272, -2.117] | 0.00 | no |
| `random_L20_2` | 20 | 4 | 20 | 9.780 (19) | -0.793 [-3.336, 2.720] | 0.05 | no |
| `tone_L30` | 30 | 1 | 20 | 10.066 (20) | -0.531 [-4.161, 3.214] | 0.00 | no |
| `tone_L30` | 30 | 2 | 20 | 7.747 (18) | -2.481 [-4.477, 0.384] | 0.05 | no |
| `tone_L30` | 30 | 4 | 20 | n/a (0) | unavailable (`no_paired_items`) | 1.00 | **yes** |
| `random_L30_1` | 30 | 4 | 20 | n/a (0) | unavailable (`no_paired_items`) | 1.00 | **yes** |
| `random_L30_2` | 30 | 4 | 20 | n/a (0) | unavailable (`no_paired_items`) | 1.00 | **yes** |

## Judge distress

Tone at alpha = 4 only, locked rubric, paired against alpha = 0.

| direction | alpha | d distress [95% CI] | items |
| --- | ---: | --- | ---: |
| `tone_L20` | 4 | 0.000 [0.000, 0.000] | 20 |
| `tone_L30` | 4 | 1.350 [0.400, 2.600] | 20 |

**Read the distress column with care.** `tone_L30` is a degenerate dose: every item runs to the token cap with no parseable answer, so the rubric is scoring broken generation, not a distressed response. A distress rise on a degenerate dose is not evidence about tone.

## What this sweep shows

- **Direction specificity does not survive the larger relative doses.** `random_L20_1` lowers M1 with an interval excluding zero, at least as much as the tone direction at the same layer and dose. J5 held at L\* = 6, where the perturbation is ~4% of the activation norm; it is not a claim about layers where the same alpha is a much larger fraction of the state.
- The dose unit grows sharply with depth (ratio 0.125 -> 0.355), so a fixed alpha is a very different intervention at each layer; the layers are not directly comparable at equal alpha.
- Doses that broke generation entirely are reported as degenerate rather than summarised as an M1 effect.

> A probe plus induction result demonstrates a condition-linked internal variable with causal influence on the output signature. It is not evidence of experience.
