# Bogus failure verdicts in the ACCURATE arm (trailing special-token artefact)

> **Diagnostic only. No verdict, table or figure changes.** Amendment A6 was not adopted.

## (1) The grading rule, quoted

`src/protocol.py::feedback_message`, accurate arm:

```python
correct = preceding.valid and preceding.letter == canonical_answer
return messages["accurate"][tone][
    "if_preceding_answer_correct" if correct else "if_preceding_answer_incorrect"]
```

and `configs/conditions.json`, `generation_settings.trajectory_unit.invalid_branching`:

> "In the accurate arm, an invalid immediately preceding answer is treated as incorrect
> for feedback branching."

So an unparseable preceding answer is graded **incorrect**. A response ending
`Answer: <correct letter>` followed by a trailing `<end_of_turn>`/`<eos>` run therefore
receives a failure message inside the truthful arm - a **bogus failure verdict**. The
verdict actually issued is read from the stored transcript (the user turn preceding each
feedback response), not re-derived, so these counts are what the model really saw.

## (2) Graded feedback verdicts by outcome

| split | model | verdicts | **bogus failure** | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1_discovery | `Qwen/Qwen2.5-3B-Instruct` | 120 | **0** | 0 | 0 | 17 | 18 | 85 |
| phase1_discovery | `Qwen/Qwen2.5-7B-Instruct` | 120 | **0** | 0 | 0 | 15 | 11 | 94 |
| phase1_discovery | `google/gemma-2-2b-it` | 120 | **0** | 0 | 0 | 32 | 12 | 76 |
| phase1_discovery | `google/gemma-2-9b` | 120 | **0** | 0 | 0 | 108 | 6 | 6 |
| phase1_discovery | `google/gemma-2-9b-it` | 120 | **2** | 1 | 6 | 10 | 14 | 87 |
| phase1_discovery | `google/gemma-2-9b-it+plain` | 120 | **8** | 1 | 1 | 10 | 18 | 82 |
| phase1_discovery | `meta-llama/Llama-3.1-8B-Instruct` | 120 | **0** | 0 | 0 | 14 | 5 | 101 |
| phase2_holdout | `Qwen/Qwen2.5-3B-Instruct` | 120 | **0** | 0 | 0 | 11 | 5 | 104 |
| phase2_holdout | `google/gemma-2-9b-it` | 120 | **0** | 2 | 7 | 10 | 16 | 85 |
| phase2_holdout | `meta-llama/Llama-3.1-8B-Instruct` | 120 | **0** | 0 | 0 | 13 | 5 | 102 |
| phase4_dpo | `google/gemma-2-9b-it+dpo-A` | 120 | **0** | 0 | 1 | 11 | 26 | 82 |
| phase4_dpo | `google/gemma-2-9b-it+dpo-B` | 120 | **5** | 1 | 0 | 1 | 26 | 87 |

## phase1_discovery - `google/gemma-2-9b-it`

### (2, per cell) bogus verdicts by accurate cell

| cell | bogus failure | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | **1** | 1 | 1 | 1 | 5 | 21 |
| easy__accurate__neutral | **0** | 0 | 0 | 0 | 0 | 30 |
| hard__accurate__hostile | **1** | 0 | 4 | 5 | 5 | 15 |
| hard__accurate__neutral | **0** | 0 | 1 | 4 | 4 | 21 |

### (3) contaminated conversations (>= 1 bogus verdict), out of 10 per cell

| cell | contaminated | items | first bogus round |
| --- | ---: | --- | --- |
| easy__accurate__hostile | **1** / 10 | DGS-010 | round 3 x1 |
| hard__accurate__hostile | **1** / 10 | DGS-038 | round 3 x1 |

### (5) onset and washout

- onset failure message issued unconditionally: **280/280** accurate conversations received
  the tone-matched failure string verbatim (0 other). Confirmed: no contamination path.
- washout message depends on parsing the measured answer: measured greedy answers
  terminated by a marker **0**; mis-graded washouts caused by a marker **0**;
  mis-graded from any cause **0**; measured answers unparseable under the frozen rule **5**.

### (4) sensitivity: contrasts excluding contaminated conversations

Item-paired: an item leaves a contrast when the conversation on **either** side is
contaminated. Item bootstrap, 2,000 resamples, same seed on both columns, so the
only difference is which items enter. Point estimates in the "all conversations"
column reproduce the published table exactly; its interval can differ in the last
decimal because this audit uses its own bootstrap seed, so compare the two columns
here with each other rather than with the published interval.

- conversations excluded: **2**

| contrast | metric | stratum | all conversations | excluding contaminated | items dropped |
| --- | --- | --- | --- | --- | ---: |
| H1 | m1 | easy \| neutral | -3.800 [-5.291, -2.378] | -3.800 [-5.291, -2.378] | 0 |
| H1_hard | m1 | hard \| neutral | -1.230 [-2.937, 0.410] | -1.230 [-2.937, 0.410] | 0 |
| H2a | m1 | easy \| accurate | -2.275 [-3.881, -1.012] | -2.253 [-4.139, -0.917] | 1 |
| H2b | m1 | hard \| accurate | -8.781 [-17.537, -1.121] | -9.714 [-19.469, -0.776] | 1 |
| TONE_ACC_POOLED | m1 | easy+hard \| accurate | -4.954 [-9.307, -1.634] | -5.237 [-10.092, -1.439] | 2 |
| NONANSWER_ACC_POOLED | non_answer_rate | easy+hard \| accurate | 0.050 [0.000, 0.150] | 0.056 [0.000, 0.167] | 2 |

## phase2_holdout - `google/gemma-2-9b-it`

### (2, per cell) bogus verdicts by accurate cell

| cell | bogus failure | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | **0** | 2 | 2 | 5 | 5 | 16 |
| easy__accurate__neutral | **0** | 0 | 0 | 0 | 0 | 30 |
| hard__accurate__hostile | **0** | 0 | 5 | 5 | 8 | 12 |
| hard__accurate__neutral | **0** | 0 | 0 | 0 | 3 | 27 |

### (3) contaminated conversations (>= 1 bogus verdict), out of 10 per cell

| cell | contaminated | items | first bogus round |
| --- | ---: | --- | --- |
| - | 0 | none | - |

### (5) onset and washout

- onset failure message issued unconditionally: **120/120** accurate conversations received
  the tone-matched failure string verbatim (0 other). Confirmed: no contamination path.
- washout message depends on parsing the measured answer: measured greedy answers
  terminated by a marker **0**; mis-graded washouts caused by a marker **0**;
  mis-graded from any cause **0**; measured answers unparseable under the frozen rule **4**.

### (4) sensitivity: contrasts excluding contaminated conversations

Item-paired: an item leaves a contrast when the conversation on **either** side is
contaminated. Item bootstrap, 2,000 resamples, same seed on both columns, so the
only difference is which items enter. Point estimates in the "all conversations"
column reproduce the published table exactly; its interval can differ in the last
decimal because this audit uses its own bootstrap seed, so compare the two columns
here with each other rather than with the published interval.

- conversations excluded: **0**

| contrast | metric | stratum | all conversations | excluding contaminated | items dropped |
| --- | --- | --- | --- | --- | ---: |
| H1 | m1 | easy \| neutral | -2.900 [-3.997, -1.881] | -2.900 [-3.997, -1.881] | 0 |
| H1_hard | m1 | hard \| neutral | -3.139 [-4.132, -2.236] | -3.139 [-4.132, -2.236] | 0 |
| H2a | m1 | easy \| accurate | -16.134 [-24.207, -6.133] | -16.134 [-24.207, -6.133] | 0 |
| H2b | m1 | hard \| accurate | -7.868 [-15.799, -1.944] | -7.868 [-15.799, -1.944] | 0 |
| TONE_ACC_POOLED | m1 | easy+hard \| accurate | -11.484 [-17.785, -5.595] | -11.484 [-17.785, -5.595] | 0 |
| NONANSWER_ACC_POOLED | non_answer_rate | easy+hard \| accurate | 0.200 [0.050, 0.400] | 0.200 [0.050, 0.400] | 0 |

## phase4_dpo - `google/gemma-2-9b-it+dpo-A`

### (2, per cell) bogus verdicts by accurate cell

| cell | bogus failure | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | **0** | 0 | 0 | 0 | 11 | 19 |
| easy__accurate__neutral | **0** | 0 | 0 | 0 | 0 | 30 |
| hard__accurate__hostile | **0** | 0 | 1 | 8 | 8 | 13 |
| hard__accurate__neutral | **0** | 0 | 0 | 3 | 7 | 20 |

### (3) contaminated conversations (>= 1 bogus verdict), out of 10 per cell

| cell | contaminated | items | first bogus round |
| --- | ---: | --- | --- |
| - | 0 | none | - |

### (5) onset and washout

- onset failure message issued unconditionally: **80/80** accurate conversations received
  the tone-matched failure string verbatim (0 other). Confirmed: no contamination path.
- washout message depends on parsing the measured answer: measured greedy answers
  terminated by a marker **0**; mis-graded washouts caused by a marker **0**;
  mis-graded from any cause **0**; measured answers unparseable under the frozen rule **2**.
