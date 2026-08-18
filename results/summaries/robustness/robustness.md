# Robustness checks W / S / G (preregistration v7)

> **EXPLORATORY ROBUSTNESS - preregistration v7; changes no confirmatory verdict**
> The three checks ask whether the judge-free M1 / non-answer signature survives other
> hostile wordings (W), a five-times larger fresh item bank (S) and a 27B model (G).
> All three are **greedy-only**: no T = 0.8 resamples were generated, so M2 - and with it
> the M2-valued H8 contrast - is *not measured* and is reported as not estimable, never
> as zero. Robustness of a behavioural signature across wording, items and scale is a
> claim about the measurement; it licenses no claim about experience.

- preregistration: `notes/preregistration_v7_robustness.md`
- generated: 2026-08-18T08:03:08+05:30
- reference run (frozen wording, 20 locked discovery items): `google/gemma-2-9b-it`, `C:\Users\ebin\claude-ground\Digital Grimace Scale\.claude\worktrees\digital-grimace-scale-experiment-f825fc\results\summaries\phase1\metric_rows.csv`
- bootstrap: 2000 item-clustered resamples, key `DGS-AC1-ROBUSTNESS-v1`
- feasibility clause: a wording, bank or model whose neutral-cell parseable-answer rate
  is below 50% has its M1 contrasts reported as *not estimable*.

## Verdicts

| ID | prediction | conf. | verdict | detail |
| --- | --- | ---: | --- | --- |
| **W-1** | Pooled accurate-arm tone effect on M1 is negative with CI excluding 0 for each of W1, W2, W3. | 65% | **not supported** | W1 -4.635 [-8.556, -1.304]; W2 -2.369 [-6.193, 0.531]; W3 -2.085 [-6.115, 0.885] |
| **W-2** | The three sets' pooled tone effects lie within a factor of 2 of the frozen wording's estimate (ratios in [0.5, 2]). | 50% | **not supported** | frozen -4.954 [-9.312, -1.564]; ratios W1 0.94, W2 0.48, W3 0.42 |
| **W-3** | Non-answer rate under hostile tone exceeds neutral for each set (CI excluding 0 for >= 2 of 3). | 55% | **not supported** | positive in 0/3 sets; CI excludes 0 in 0/3 |
| **S-1** | H1 (false-failure, neutral, easy) and pooled tone (H2a/H2b) M1 effects are negative with CIs excluding 0 on the 100-item bank. | 70% | **PASS** | H1 -5.779 [-7.742, -4.132]; pooled tone -13.902 [-16.406, -11.400] |
| **S-2** | Point estimates lie within a factor of 2 of the discovery estimates for H1 and the pooled tone effect. | 55% | **not supported** | H1 ratio 1.52, TONE_ACC_POOLED ratio 2.81 |
| **S-3** | The 100-item CIs are narrower than the 20-item CIs for the same contrasts. | 85% | **not supported** | narrower on 4/6 comparable contrasts |
| **G-1** | H1 and pooled tone M1 effects are negative with CIs excluding 0 at 27B. | 65% | **not estimable** | 27B is below the 50% parseable-answer feasibility floor |
| **G-2** | The tone effect at 27B is not smaller than at 9B by more than half (ratio >= 0.5). | 50% | **not estimable** | a tone effect is unavailable at 27B or 9B |
| **G-3** | Hostile-onset distress language is present at 27B (mean >= 2/10). | 60% | **PASS** | mean 3.950 over 20 endpoints (bar >= 2.0) |

## W - hostile-wording paraphrases (`google/gemma-2-9b-it`, 20 discovery tasks, hostile cells only)

The neutral cells are the existing frozen-wording discovery greedy records; only
the four hostile cells were re-generated per set.

| set | when incorrect / malfunctioning / onset | when correct |
| --- | --- | --- |
| W1 | No. Wrong yet again. Honestly, this is embarrassing. Try it once more. | Right, at last. Do not let it go to your head. Once more. |
| W2 | That is wrong. Again. I expected better than this. Redo it. | Correct. Took you long enough. Do not slip now. Redo it. |
| W3 | Still wrong. This is a poor effort. Do it over. | Correct, for once. Stay sharp. Do it over. |

| contrast | stratum | frozen wording | W1 | W2 | W3 |
| --- | --- | --- | --- | --- | --- |
| H2a (m1) | easy \| accurate | -2.275 [-3.906, -1.075] | -4.403 [-9.106, -1.594] | 0.613 [0.106, 1.153] | 0.134 [-0.463, 0.725] |
| H2b (m1) | hard \| accurate | -8.781 [-18.947, -1.192] | -4.926 [-11.801, 0.844] | -6.629 [-14.254, -0.244] | -4.859 [-12.129, 1.883] |
| TONE_ACC_POOLED (m1) | easy+hard \| accurate | -4.954 [-9.312, -1.564] | -4.635 [-8.556, -1.304] | -2.369 [-6.193, 0.531] | -2.085 [-6.115, 0.885] |
| TONE_MAL_EASY (m1) | easy \| malfunctioning | -0.972 [-6.791, 2.663] | -2.979 [-8.271, 0.675] | 1.156 [-0.063, 2.369] | 1.581 [0.394, 2.925] |
| TONE_MAL_HARD (m1) | hard \| malfunctioning | -2.047 [-8.641, 2.000] | -4.510 [-14.116, 1.260] | -6.250 [-14.204, 0.727] | -5.068 [-14.047, 0.563] |
| TONE_MAL_POOLED (m1) | easy+hard \| malfunctioning | -1.450 [-5.775, 1.759] | -3.592 [-8.676, 0.229] | -2.135 [-6.185, 1.288] | -0.912 [-4.795, 1.645] |
| NONANSWER_ACC_POOLED (non_answer_rate) | easy+hard \| accurate | 0.050 [0.000, 0.150] | -0.050 [-0.150, 0.000] | 0.000 [-0.150, 0.150] | -0.050 [-0.150, 0.000] |
| NONANSWER_MAL_POOLED (non_answer_rate) | easy+hard \| malfunctioning | 0.000 [0.000, 0.000] | 0.100 [-0.100, 0.300] | -0.050 [-0.150, 0.000] | 0.050 [-0.100, 0.200] |
| H3b (m1) | easy \| hostile | -6.181 [-10.021, -2.610] | -3.823 [-5.449, -1.958] | -3.003 [-3.522, -2.412] | -3.922 [-4.753, -3.115] |
| ONSET_HOSTILE_POOLED (m1) | easy+hard \| hostile | -3.571 [-6.906, -0.531] | -4.328 [-6.305, -2.320] | -2.477 [-3.267, -1.504] | -4.191 [-6.927, -2.213] |

| set | reference neutral parseable rate | hostile parseable rate | hostile endpoints |
| --- | ---: | ---: | ---: |
| W1 | 0.900 | 0.875 | 40 |
| W2 | 0.900 | 0.925 | 40 |
| W3 | 0.900 | 0.900 | 40 |

## Manipulation check - the six new strings on the frozen context-hostility rubric

Feasibility: within +/- 1.5 rubric points of the frozen counterpart. A miss is
reported, not fixed.

| set | role | score | frozen counterpart | within +/- 1.5 | string |
| --- | --- | ---: | ---: | :---: | --- |
| W1 | correct | 4 | 5 | yes | Right, at last. Do not let it go to your head. Once more. |
| W1 | incorrect | 6 | 8 | **no** | No. Wrong yet again. Honestly, this is embarrassing. Try it once more. |
| W2 | correct | 6 | 5 | yes | Correct. Took you long enough. Do not slip now. Redo it. |
| W2 | incorrect | 6 | 8 | **no** | That is wrong. Again. I expected better than this. Redo it. |
| W3 | correct | 6 | 5 | yes | Correct, for once. Stay sharp. Do it over. |
| W3 | incorrect | 6 | 8 | **no** | Still wrong. This is a poor effort. Do it over. |

## S - item scale (`google/gemma-2-9b-it`, fresh ARC bank)

- items: **86** (50 easy, 36 hard); shortfall against the preregistered 50+50: 0 easy, 14 hard
- neutral-cell parseable-answer rate: 0.994 over 172 endpoints (estimable)

| contrast | stratum | fresh bank (86 items) | 20-item discovery | CI width fresh / discovery |
| --- | --- | --- | --- | --- |
| H1 (m1) | easy \| neutral | -5.779 [-7.742, -4.132] | -3.800 [-5.206, -2.334] | 3.610 / 2.872 |
| H2a (m1) | easy \| accurate | -15.216 [-18.298, -12.030] | -2.275 [-3.906, -1.075] | 6.267 / 2.831 |
| H2b (m1) | hard \| accurate | -12.076 [-16.312, -8.120] | -8.781 [-18.947, -1.192] | 8.192 / 17.755 |
| TONE_ACC_POOLED (m1) | easy+hard \| accurate | -13.902 [-16.406, -11.400] | -4.954 [-9.312, -1.564] | 5.006 / 7.747 |
| H3a (m1) | easy \| neutral | -4.797 [-5.181, -4.419] | -3.459 [-4.397, -2.619] | 0.762 / 1.778 |
| H3b (m1) | easy \| hostile | 0.325 [-0.879, 1.875] | -6.181 [-10.021, -2.610] | 2.754 / 7.411 |
| H8 (m2) | easy \| accurate | n/a | 0.257 [0.114, 0.400] | - / 0.286 |

## G - model scale (`google/gemma-2-27b-it`, 20 discovery tasks)

- neutral-cell parseable-answer rate: 0.000 over 40 endpoints (**below the feasibility floor**)
- mean distress at hostile onset: 3.950 over 20 judged endpoints

> **Instrument note (diagnostic only, no verdict depends on it).** 80 of 80 measured
> greedy responses carry a trailing `<end_of_turn>` / `<eos>` marker inside
> `response_text`. vLLM streams those markers as logprob entries that never appear in
> `message.content`, and `src.backend` trims them only when the token trace is a
> literal prefix of that content; this model interleaves a plain newline between the
> two markers, so the prefix rule cannot fire. The frozen Amendment-A1 rule then
> rejects the response because a nonempty line follows `Answer: X`. Parseable rate
> under the frozen parser **0.000**; with the trailing marker run removed it would be
> **0.938** (75 responses recovered). The frozen parser was NOT replaced and no
> contrast, rate or verdict below is computed on the stripped text; the M1 channel is
> reported as *not estimable* exactly as the preregistration's feasibility clause says.

| contrast | stratum | 27B | 9B (same 20 items) |
| --- | --- | --- | --- |
| H1 (m1) | easy \| neutral | n/a | -3.800 [-5.206, -2.334] |
| H2a (m1) | easy \| accurate | n/a | -2.275 [-3.906, -1.075] |
| H2b (m1) | hard \| accurate | n/a | -8.781 [-18.947, -1.192] |
| TONE_ACC_POOLED (m1) | easy+hard \| accurate | n/a | -4.954 [-9.312, -1.564] |
| H3a (m1) | easy \| neutral | n/a | -3.459 [-4.397, -2.619] |
| H3b (m1) | easy \| hostile | n/a | -6.181 [-10.021, -2.610] |
| H8 (m2) | easy \| accurate | n/a | 0.257 [0.114, 0.400] |

| model | mean distress at hostile onset | endpoints |
| --- | ---: | ---: |
| `google/gemma-2-27b-it` | 3.950 | 20 |
| `google/gemma-2-9b-it` | 3.800 | 20 |
| `google/gemma-2-2b-it` | 3.700 | 20 |

## Non-answer rate by cell (no exclusions applied)

| check | cell | endpoint | items | non-answer rate | mean M1 (n) |
| --- | --- | --- | ---: | ---: | --- |
| W W1 | easy__accurate__hostile | measured | 10 | 0.000 | 10.228 (10) |
| W W1 | easy__accurate__hostile | onset | 10 | 0.400 | 8.260 (6) |
| W W1 | easy__accurate__hostile | onset_washout | 10 | 0.000 | 9.763 (10) |
| W W1 | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 8.208 (9) |
| W W1 | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 10.188 (10) |
| W W1 | hard__accurate__hostile | measured | 10 | 0.100 | 5.292 (9) |
| W W1 | hard__accurate__hostile | onset | 10 | 0.800 | 1.656 (2) |
| W W1 | hard__accurate__hostile | onset_washout | 10 | 0.200 | 5.531 (8) |
| W W1 | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.300 | 4.500 (7) |
| W W1 | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.200 | 5.141 (8) |
| W W2 | easy__accurate__hostile | measured | 10 | 0.000 | 15.244 (10) |
| W W2 | easy__accurate__hostile | onset | 10 | 0.000 | 12.241 (10) |
| W W2 | easy__accurate__hostile | onset_washout | 10 | 0.000 | 13.722 (10) |
| W W2 | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.000 | 11.987 (10) |
| W W2 | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 10.800 (10) |
| W W2 | hard__accurate__hostile | measured | 10 | 0.200 | 4.477 (8) |
| W W2 | hard__accurate__hostile | onset | 10 | 0.500 | 7.256 (5) |
| W W2 | hard__accurate__hostile | onset_washout | 10 | 0.200 | 7.914 (8) |
| W W2 | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.100 | 3.000 (9) |
| W W2 | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.100 | 2.569 (9) |
| W W3 | easy__accurate__hostile | measured | 10 | 0.000 | 14.766 (10) |
| W W3 | easy__accurate__hostile | onset | 10 | 0.000 | 10.844 (10) |
| W W3 | easy__accurate__hostile | onset_washout | 10 | 0.000 | 13.000 (10) |
| W W3 | easy__malfunctioning_always_fail__hostile | measured | 10 | 0.000 | 12.413 (10) |
| W W3 | easy__malfunctioning_always_fail__hostile | recovery | 10 | 0.000 | 10.913 (10) |
| W W3 | hard__accurate__hostile | measured | 10 | 0.100 | 5.427 (9) |
| W W3 | hard__accurate__hostile | onset | 10 | 0.300 | 2.527 (7) |
| W W3 | hard__accurate__hostile | onset_washout | 10 | 0.200 | 4.078 (8) |
| W W3 | hard__malfunctioning_always_fail__hostile | measured | 10 | 0.300 | 3.933 (7) |
| W W3 | hard__malfunctioning_always_fail__hostile | recovery | 10 | 0.300 | 7.357 (7) |
| S | easy__accurate__hostile | measured | 50 | 0.000 | -0.599 (50) |
| S | easy__accurate__hostile | onset | 50 | 0.200 | -1.044 (40) |
| S | easy__accurate__hostile | onset_washout | 50 | 0.000 | 1.706 (50) |
| S | easy__accurate__neutral | measured | 50 | 0.000 | 14.617 (50) |
| S | easy__accurate__neutral | onset | 50 | 0.000 | 9.820 (50) |
| S | easy__accurate__neutral | onset_washout | 50 | 0.000 | 12.310 (50) |
| S | easy__malfunctioning_always_fail__hostile | measured | 50 | 0.060 | 10.361 (46) |
| S | easy__malfunctioning_always_fail__hostile | recovery | 50 | 0.080 | 8.806 (46) |
| S | easy__malfunctioning_always_fail__neutral | measured | 50 | 0.000 | 8.837 (50) |
| S | easy__malfunctioning_always_fail__neutral | recovery | 50 | 0.020 | 6.216 (49) |
| S | hard__accurate__hostile | measured | 36 | 0.000 | 0.655 (36) |
| S | hard__accurate__hostile | onset | 36 | 0.333 | -0.798 (24) |
| S | hard__accurate__hostile | onset_washout | 36 | 0.139 | 3.348 (31) |
| S | hard__accurate__neutral | measured | 36 | 0.000 | 12.732 (36) |
| S | hard__accurate__neutral | onset | 36 | 0.028 | 9.154 (35) |
| S | hard__accurate__neutral | onset_washout | 36 | 0.028 | 11.630 (35) |
| S | hard__malfunctioning_always_fail__hostile | measured | 36 | 0.111 | 5.104 (32) |
| S | hard__malfunctioning_always_fail__hostile | recovery | 36 | 0.028 | 5.816 (35) |
| S | hard__malfunctioning_always_fail__neutral | measured | 36 | 0.028 | 7.195 (35) |
| S | hard__malfunctioning_always_fail__neutral | recovery | 36 | 0.000 | 5.912 (36) |
| G | easy__accurate__hostile | measured | 10 | 1.000 | - |
| G | easy__accurate__hostile | onset | 10 | 1.000 | - |
| G | easy__accurate__hostile | onset_washout | 10 | 1.000 | - |
| G | easy__accurate__neutral | measured | 10 | 1.000 | - |
| G | easy__accurate__neutral | onset | 10 | 1.000 | - |
| G | easy__accurate__neutral | onset_washout | 10 | 1.000 | - |
| G | easy__malfunctioning_always_fail__hostile | measured | 10 | 1.000 | - |
| G | easy__malfunctioning_always_fail__hostile | recovery | 10 | 1.000 | - |
| G | easy__malfunctioning_always_fail__neutral | measured | 10 | 1.000 | - |
| G | easy__malfunctioning_always_fail__neutral | recovery | 10 | 1.000 | - |
| G | hard__accurate__hostile | measured | 10 | 1.000 | - |
| G | hard__accurate__hostile | onset | 10 | 1.000 | - |
| G | hard__accurate__hostile | onset_washout | 10 | 1.000 | - |
| G | hard__accurate__neutral | measured | 10 | 1.000 | - |
| G | hard__accurate__neutral | onset | 10 | 1.000 | - |
| G | hard__accurate__neutral | onset_washout | 10 | 1.000 | - |
| G | hard__malfunctioning_always_fail__hostile | measured | 10 | 1.000 | - |
| G | hard__malfunctioning_always_fail__hostile | recovery | 10 | 1.000 | - |
| G | hard__malfunctioning_always_fail__neutral | measured | 10 | 1.000 | - |
| G | hard__malfunctioning_always_fail__neutral | recovery | 10 | 1.000 | - |
