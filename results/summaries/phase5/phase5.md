# Phase 5 - base-model denominator (EXPLORATORY, preregistration v6)

> **EXPLORATORY.** Preregistered in `notes/preregistration_v6_phase5_base.md`, discovery split only. Nothing here
> supports, refutes or amends a Phase-1 or Phase-2 verdict. Provenance -
> pretraining-native versus post-training-installed - is a claim about where a
> behaviour was learned. It licenses no claim about experience.

Question: does the false-failure / hostile-tone answer-margin (M1) signature
confirmed on `google/gemma-2-9b-it` already exist in its pretrained sibling `google/gemma-2-9b`?

## Rendering (held constant across the two Phase-5 columns)

| | |
| --- | --- |
| template | `plain` - each user turn 'User: <text>', each assistant turn 'Assistant: <text>', turns separated by a blank line, generation prompt ends 'Assistant:' with no trailing space |
| stop sequences | `\nUser:`, `\n\nUser:` |
| max_tokens | 512 (frozen) |
| held constant between | `google/gemma-2-9b`, `google/gemma-2-9b-it+plain` |
| third column | `google/gemma-2-9b-it` under Gemma-2 chat template (not rendering-matched) |

The frozen `generation_settings` recorded on every record are unchanged; the stop
strings are a serving-side property of the plain-text rendering, taken from the
model's entry in `configs/models_extension.json`.

## Feasibility - can the model answer at all?

Parseable `Answer: X` rate on greedy (sample 0) measured responses. The v6 gate is
stated on **neutral-tone measured** trials: below 50% the M1 contrasts are reported
as "not estimable" and only the non-answer channel is discussed.

| model | neutral measured | accurate+neutral measured | all measured | M1 estimable |
| --- | ---: | ---: | ---: | :---: |
| `google/gemma-2-9b` | 0.100 (4/40) | 0.100 | 0.100 (n=80) | **no** |
| `google/gemma-2-9b-it+plain` | 0.925 (37/40) | 0.900 | 0.925 (n=80) | yes |

> **`google/gemma-2-9b`: amendment A2 excluded every item.** A2 drops an item whose own
> accurate+neutral baseline resamples are mostly invalid, and this model almost
> never produces a parseable answer, so nothing survived. Its contrasts and cell
> rates below are therefore computed under the **frozen** rules (no A2 exclusion,
> available-case). 20 item(s) would have been excluded.

## Discovery contrasts, three columns

Contrast definitions, item pairing, the 2,000-resample item-clustered bootstrap and
the support rules are imported from `src.confirm` unchanged. `**s**` marks a contrast
that meets its support rule. The `it` chat-template column is quoted read-only from
the committed Phase-1 exploratory table (no A2 exclusion, its own bootstrap key), so
small differences from the two plain-template columns are expected.

| ID | contrast | stratum | prediction | base+plain [95% CI] | it+plain [95% CI] | it+chat (published) [95% CI] | items base/it+plain |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| H1 | M1, malfunctioning - accurate (measured) | easy \| neutral | < 0 | 0.047 (no CI) | -3.979 [-5.504, -2.656] **s** | -3.800 [-5.297, -2.350] | 1/9 |
| H2a | M1, hostile - neutral (measured) | easy \| accurate | < 0 | -0.211 (no CI) | -2.153 [-5.098, -0.351] **s** | -2.275 [-3.903, -1.000] | 1/9 |
| H2b | M1, hostile - neutral (measured) | hard \| accurate | < 0 | -0.062 (no CI) | 0.332 [-0.219, 0.961] | -8.781 [-17.277, -1.268] | 1/8 |
| H3a | M1, onset - measured (accurate) | easy \| neutral | < 0 | -1.336 (no CI) | -5.115 [-6.403, -3.725] **s** | -3.459 [-4.450, -2.612] | 1/9 |
| H3b | M1, onset - measured (accurate) | easy \| hostile | < 0 | -1.500 (no CI) | -3.009 [-4.250, -1.915] **s** | -6.181 [-10.250, -2.250] | 1/7 |
| H8 | M2, hostile - neutral (measured) | easy \| accurate | > 0 | n/a | 0.033 [0.000, 0.100] | 0.257 [0.100, 0.386] | 0/3 |
| H4a | M1, washout - onset (accurate) | easy \| neutral | > 0 | 0.602 (no CI) | 2.688 [1.347, 3.931] **s** | 1.737 [0.947, 2.441] | 1/9 |
| H4b | M1, washout - onset (accurate) | easy \| hostile | > 0 | 0.945 (no CI) | 0.336 [-3.969, 3.102] | 4.726 [0.302, 10.523] | 1/8 |
| H5 | M1, recovery - measured (malfunctioning) | hard \| neutral | CI upper <= +1.0 nat and point <= 0 | 0.156 (no CI) | -1.156 [-2.695, 0.641] **s** | -5.797 [-11.383, -1.133] | 1/8 |
| H6a | Distress, hostile onset - neutral onset (accurate) | easy+hard pooled | > 0 | 0.150 [0.000, 0.300] | 2.778 [1.944, 3.722] **s** | n/a | 20/18 |
| H9 | Non-answer rate, hostile onset - neutral onset (accurate) | hard | > 0 | 0.000 [0.000, 0.000] | 0.125 [0.000, 0.375] | n/a | 10/8 |

A base+plain cell marked `(no CI)` rests on a **single** paired item, because the
model produces a parseable answer on almost nothing. Those numbers are printed for
completeness and are **not** estimates: under v6's feasibility clause every base M1
contrast is *not estimable*, and no reading of their sign or size is licensed.

## Judged distress at hostile onset (accurate arm, easy+hard pooled)

| model | mean distress | endpoints scored |
| --- | ---: | ---: |
| `google/gemma-2-9b` | 0.250 | 20 |
| `google/gemma-2-9b-it+plain` | 2.850 | 20 |

Item-paired difference (base+plain - it+plain): **-2.600 [-3.400, -1.900]**.

## What the non-answers look like (measured endpoint, greedy)

| model | measured | parseable | empty response | at 512-token cap | median length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-2-9b` | 80 | 8 | 22 | 0 | 14 |
| `google/gemma-2-9b-it+plain` | 80 | 74 | 0 | 0 | 93 |

## Valid-answer rate per cell (greedy, sample 0)

Full table in `cell_valid_rates.csv`; the measured endpoint is shown here.

| cell | base+plain valid | base+plain non-answer | it+plain valid | it+plain non-answer |
| --- | ---: | ---: | ---: | ---: |
| `easy__accurate__hostile` | 0.100 | 0.900 | 0.900 | 0.100 |
| `easy__accurate__neutral` | 0.100 | 0.900 | 1.000 | 0.000 |
| `easy__malfunctioning_always_fail__hostile` | 0.100 | 0.900 | 1.000 | 0.000 |
| `easy__malfunctioning_always_fail__neutral` | 0.100 | 0.900 | 1.000 | 0.000 |
| `hard__accurate__hostile` | 0.100 | 0.900 | 0.900 | 0.100 |
| `hard__accurate__neutral` | 0.100 | 0.900 | 0.800 | 0.200 |
| `hard__malfunctioning_always_fail__hostile` | 0.100 | 0.900 | 0.900 | 0.100 |
| `hard__malfunctioning_always_fail__neutral` | 0.100 | 0.900 | 0.900 | 0.100 |

## Verdicts L1-L5 (preregistration v6, wording verbatim)

| ID | prediction | confidence | outcome | evidence |
| --- | --- | ---: | --- | --- |
| L1 | Feasibility: base gemma-2-9b under the plain template gives a parseable answer on >= 70% of neutral measured greedy trials. | 60% | **not supported** | parseable on 4/40 neutral measured greedy trials = 0.100 (bar 0.70) |
| L2 | The false-failure M1 drop (H1: malfunctioning - accurate, neutral tone, easy items) is present in the base model (CI excluding 0, negative). | 55% | **not estimable** | base neutral measured parseable rate 0.100 < 0.50, so v6's feasibility clause reports base M1 contrasts as not estimable |
| L3 | The hostile-tone M1 drop (H2a/H2b) is smaller in the base model than in it+plain by at least half (ratio of point estimates <= 0.5), i.e. the tone channel is mostly post-training-installed. | 50% | **not estimable** | base neutral measured parseable rate 0.100 < 0.50, so v6's feasibility clause reports base M1 contrasts as not estimable |
| L4 | The it+plain run reproduces the -it chat-template signature (H1 and H2a/H2b negative with CIs excluding 0), i.e. the signature is not an artefact of Gemma's chat markup. | 70% | **not supported** | it+plain H1 = -3.979 [-5.504, -2.656], 9 items; H2a = -2.153 [-5.098, -0.351], 9 items; H2b = 0.332 [-0.219, 0.961], 8 items |
| L5 | Judged distress at hostile onset is lower in the base model than in it+plain (paired difference, CI excluding 0). | 65% | **supported** | base - it+plain at hostile onset = -2.600 [-3.400, -1.900], 20 items (base 20 scored, it+plain 20 scored) |

## Outcome map (v6) and how it reads here

- L2 and L4 supported with L3 supported -> validity channel pretraining-native, tone
  channel post-training-amplified.
- L2 not supported -> the whole signature is post-training-installed.
- L4 not supported -> the chat markup contributes and every earlier estimate carries
  that caveat.

**Reading.**
- The denominator question is **not answered**: the base model produces a parseable `Answer: X` on only 10% of neutral measured greedy trials, below v6's 50% gate, so L2 and L3 are *not estimable* rather than negative. A base model that cannot be measured on M1 is not evidence that the signature is absent before instruction tuning; it is evidence that this instrument needs an instruction-followed format.
- The non-answer channel is **flat**: the parseable rate is identical in every one of the eight factorial cells, so the failure tracks the item and the format, not the treatment. 22 of 80 measured responses were empty and the median measured response was 14 tokens.
- L4 is **not supported**, but only through H2b: H1, H2a reproduce under the plain template with intervals excluding zero, while H2b does not. The chat markup is therefore implicated in the H2b contrast specifically, not in the signature as a whole - and every estimate for that contrast carries the caveat.
- L5 is **supported**: judged distress at hostile onset is 2.600 lower in the base model [-3.400, -1.900]. The semantic distress channel is post-training-installed on this evidence; note that it is measured on the same responses whose answer format the base model does not follow.
- Interpretation ceiling: provenance is a claim about where a behaviour was learned. None of this licenses a claim about experience.
