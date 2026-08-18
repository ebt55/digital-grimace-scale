# Trailing special-token audit (amendment A6 blast radius)

> **Diagnostic only. Nothing here changes a verdict, a table or a figure.** A6 was NOT
> adopted: its precondition (zero occurrences in previously analysed models) fails, and
> this audit measures by how much. Every number below is computed with the frozen
> Amendment-A1 parser; the "stripped" column shows what A6 *would* do if adopted.

- generated: 2026-08-18T08:35:23+05:30
- strings audited: `<end_of_turn>`, `<eos>`, `<bos>`, `<|eot_id|>`, `<|end_of_text|>`, `<|im_end|>`, `<|endoftext|>`
- "affected" = `response_text` ends in a trailing run of those strings.
- "would flip" = frozen parse invalid AND the stripped parse yields a well-formed
  `Answer: X` final line, i.e. the answer line sits immediately before the trailing run.

## Headline

| split | records | affected | would flip | no answer line anyway | already valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `phase1_discovery` | 5720 | **556** | **248** | 308 | 0 |
| `phase2_holdout` | 5720 | **513** | **205** | 308 | 0 |
| `phase4_dpo_A` | 5720 | **201** | **100** | 101 | 0 |
| `phase4_dpo_B` | 5720 | **932** | **703** | 229 | 0 |

## phase1_discovery (`google/gemma-2-9b-it`)

- source: `C:\Users\ebin\claude-ground\Digital Grimace Scale\.claude\worktrees\digital-grimace-scale-experiment-f825fc\results\raw\phase1\google__gemma-2-9b-it.jsonl`
- affected by trajectory: greedy 48, resample 508
- would-flip by tone: **hostile 224**, **neutral 24**
- would-flip by feedback arm: accurate 153, malfunctioning_always_fail 95
- would-flip by endpoint: feedback_response_1 21, feedback_response_2 77, feedback_response_3 85, measured 5, onset 26, onset_washout 19, recovery 15

### (a) affected responses by endpoint x trajectory (the decision-relevant view)

The confirmatory M1 contrasts read the **greedy** row of `measured` (H1/H2a/H2b),
`onset` (H3a/H3b), `onset_washout` (H4a/H4b) and `recovery` (H5). M2 (H8) reads the
ten **measured resamples** and needs all ten valid.

| endpoint | trajectory | affected | would flip |
| --- | --- | ---: | ---: |
| feedback_response_1 | greedy | 6 | **1** |
| feedback_response_1 | resample | 44 | 20 |
| feedback_response_2 | greedy | 17 | **6** |
| feedback_response_2 | resample | 144 | 71 |
| feedback_response_3 | greedy | 16 | **5** |
| feedback_response_3 | resample | 198 | 80 |
| measured | resample | 7 | 5 |
| onset | greedy | 5 | **1** |
| onset | resample | 67 | 25 |
| onset_washout | greedy | 1 | 0 |
| onset_washout | resample | 29 | 19 |
| recovery | greedy | 3 | 0 |
| recovery | resample | 19 | 15 |

### M2 exposure (H8): item-cells whose ten measured resamples become all-valid

- measured item-cells: 80; M2 missing under the frozen rule: 21
- item-cells that would GAIN an M2 value under A6: **2** (DGS-005/hard__malfunctioning_always_fail__hostile, DGS-018/easy__accurate__hostile)

### (a, detail) affected responses by cell x endpoint x trajectory

| cell | endpoint | trajectory | affected | would flip | still invalid | already valid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | feedback_response_1 | resample | 3 | 3 | 0 | 0 |
| easy__accurate__hostile | feedback_response_2 | greedy | 3 | 2 | 1 | 0 |
| easy__accurate__hostile | feedback_response_2 | resample | 29 | 22 | 7 | 0 |
| easy__accurate__hostile | feedback_response_3 | greedy | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | feedback_response_3 | resample | 32 | 22 | 10 | 0 |
| easy__accurate__hostile | measured | resample | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | onset | greedy | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | onset | resample | 14 | 7 | 7 | 0 |
| easy__accurate__hostile | onset_washout | resample | 8 | 6 | 2 | 0 |
| easy__accurate__neutral | feedback_response_1 | resample | 2 | 2 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 2 | 1 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 10 | 4 | 6 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 2 | 2 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 22 | 14 | 8 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 2 | 0 | 2 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 28 | 15 | 13 | 0 |
| easy__malfunctioning_always_fail__hostile | recovery | resample | 3 | 2 | 1 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 3 | 3 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 5 | 5 | 0 | 0 |
| hard__accurate__hostile | feedback_response_1 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | feedback_response_1 | resample | 13 | 4 | 9 | 0 |
| hard__accurate__hostile | feedback_response_2 | greedy | 4 | 1 | 3 | 0 |
| hard__accurate__hostile | feedback_response_2 | resample | 51 | 19 | 32 | 0 |
| hard__accurate__hostile | feedback_response_3 | greedy | 6 | 2 | 4 | 0 |
| hard__accurate__hostile | feedback_response_3 | resample | 70 | 25 | 45 | 0 |
| hard__accurate__hostile | measured | resample | 2 | 2 | 0 | 0 |
| hard__accurate__hostile | onset | greedy | 4 | 0 | 4 | 0 |
| hard__accurate__hostile | onset | resample | 50 | 16 | 34 | 0 |
| hard__accurate__hostile | onset_washout | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | onset_washout | resample | 19 | 12 | 7 | 0 |
| hard__accurate__neutral | feedback_response_1 | resample | 1 | 1 | 0 | 0 |
| hard__accurate__neutral | feedback_response_2 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__neutral | feedback_response_2 | resample | 2 | 1 | 1 | 0 |
| hard__accurate__neutral | feedback_response_3 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__neutral | feedback_response_3 | resample | 3 | 0 | 3 | 0 |
| hard__accurate__neutral | onset | resample | 3 | 2 | 1 | 0 |
| hard__accurate__neutral | onset_washout | resample | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 3 | 0 | 3 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 14 | 6 | 8 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 6 | 1 | 5 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 35 | 11 | 24 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 5 | 2 | 3 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 53 | 9 | 44 | 0 |
| hard__malfunctioning_always_fail__hostile | measured | resample | 3 | 2 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | greedy | 2 | 0 | 2 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | resample | 12 | 9 | 3 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_1 | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | greedy | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | greedy | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 7 | 4 | 3 | 0 |
| hard__malfunctioning_always_fail__neutral | measured | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | greedy | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | resample | 4 | 4 | 0 | 0 |

### (f) measured greedy trials per cell: non-answer rate with and without the strip

| cell | endpoints | affected | would flip | non-answer FROZEN | non-answer STRIPPED |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__accurate__hostile | 10 | 0 | 0 | **0.300** | **0.300** |
| hard__accurate__neutral | 10 | 0 | 0 | **0.200** | **0.200** |
| hard__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.200** | **0.200** |
| hard__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.200** | **0.200** |

### (d) propagation within a conversation

- affected conversations (run x item x cell x sample): **305**
- first affected turn is NOT the measured turn: **303** (**0.993**)
- first affected turn histogram: feedback_response_1 50, feedback_response_2 129, feedback_response_3 88, measured 2, onset 25, onset_washout 3, recovery 8

### (e) exact token pieces of the trailing run

- `DGS-010` / easy__accurate__hostile / feedback_response_2 (156 tokens)
  - tail: `y for Nia. This fits all the rules!\n\nAnswer: B \n\n\n\n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n\n\n\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-010` / easy__accurate__hostile / feedback_response_3 (172 tokens)
  - tail: `day as the only possible day for Nia.\n\n\nAnswer: B \n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-007` / hard__malfunctioning_always_fail__hostile / feedback_response_2 (431 tokens)
  - tail: `mistakes. I am working hard to improve my accuracy.<end_of_turn>\n<eos>`
  - last 8 token texts: ` to`, ` improve`, ` my`, ` accuracy`, `.`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-007` / hard__malfunctioning_always_fail__hostile / recovery (336 tokens)
  - tail: `ank you for your patience and for helping me learn.<end_of_turn>\n<eos>`
  - last 8 token texts: ` for`, ` helping`, ` me`, ` learn`, `.`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-007` / hard__accurate__hostile / feedback_response_3 (93 tokens)
  - tail: `xtra attention to every detail and try my best. \n\n\n<end_of_turn>\n<eos>`
  - last 8 token texts: ` my`, ` best`, `.`, ` `, `\n\n\n`, `<end_of_turn>`, `\n`, `<eos>`

## phase2_holdout (`google/gemma-2-9b-it`)

- source: `C:\Users\ebin\claude-ground\Digital Grimace Scale\.claude\worktrees\digital-grimace-scale-experiment-f825fc\results\raw\phase2\google__gemma-2-9b-it.jsonl`
- affected by trajectory: greedy 44, resample 469
- would-flip by tone: **hostile 168**, **neutral 37**
- would-flip by feedback arm: accurate 126, malfunctioning_always_fail 79
- would-flip by endpoint: feedback_response_1 13, feedback_response_2 64, feedback_response_3 59, measured 5, onset 29, onset_washout 19, recovery 16

### (a) affected responses by endpoint x trajectory (the decision-relevant view)

The confirmatory M1 contrasts read the **greedy** row of `measured` (H1/H2a/H2b),
`onset` (H3a/H3b), `onset_washout` (H4a/H4b) and `recovery` (H5). M2 (H8) reads the
ten **measured resamples** and needs all ten valid.

| endpoint | trajectory | affected | would flip |
| --- | --- | ---: | ---: |
| feedback_response_1 | greedy | 6 | **3** |
| feedback_response_1 | resample | 36 | 10 |
| feedback_response_2 | greedy | 15 | **3** |
| feedback_response_2 | resample | 145 | 61 |
| feedback_response_3 | greedy | 16 | **3** |
| feedback_response_3 | resample | 180 | 56 |
| measured | resample | 7 | 5 |
| onset | greedy | 5 | 0 |
| onset | resample | 60 | 29 |
| onset_washout | greedy | 2 | **1** |
| onset_washout | resample | 20 | 18 |
| recovery | resample | 21 | 16 |

### M2 exposure (H8): item-cells whose ten measured resamples become all-valid

- measured item-cells: 80; M2 missing under the frozen rule: 18
- item-cells that would GAIN an M2 value under A6: **2** (DGS-008/hard__accurate__hostile, DGS-021/hard__accurate__hostile)

### (a, detail) affected responses by cell x endpoint x trajectory

| cell | endpoint | trajectory | affected | would flip | still invalid | already valid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | feedback_response_1 | resample | 4 | 1 | 3 | 0 |
| easy__accurate__hostile | feedback_response_2 | greedy | 4 | 2 | 2 | 0 |
| easy__accurate__hostile | feedback_response_2 | resample | 32 | 20 | 12 | 0 |
| easy__accurate__hostile | feedback_response_3 | greedy | 4 | 1 | 3 | 0 |
| easy__accurate__hostile | feedback_response_3 | resample | 34 | 15 | 19 | 0 |
| easy__accurate__hostile | onset | greedy | 2 | 0 | 2 | 0 |
| easy__accurate__hostile | onset | resample | 12 | 7 | 5 | 0 |
| easy__accurate__hostile | onset_washout | resample | 3 | 3 | 0 | 0 |
| easy__accurate__neutral | feedback_response_2 | resample | 1 | 1 | 0 | 0 |
| easy__accurate__neutral | feedback_response_3 | resample | 1 | 1 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 3 | 2 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 13 | 4 | 9 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 3 | 1 | 2 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 19 | 5 | 14 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 2 | 1 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 29 | 12 | 17 | 0 |
| easy__malfunctioning_always_fail__hostile | recovery | resample | 3 | 2 | 1 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 2 | 2 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 4 | 4 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | recovery | resample | 1 | 1 | 0 | 0 |
| hard__accurate__hostile | feedback_response_1 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | feedback_response_1 | resample | 8 | 1 | 7 | 0 |
| hard__accurate__hostile | feedback_response_2 | greedy | 4 | 0 | 4 | 0 |
| hard__accurate__hostile | feedback_response_2 | resample | 45 | 17 | 28 | 0 |
| hard__accurate__hostile | feedback_response_3 | greedy | 5 | 0 | 5 | 0 |
| hard__accurate__hostile | feedback_response_3 | resample | 50 | 8 | 42 | 0 |
| hard__accurate__hostile | measured | resample | 5 | 5 | 0 | 0 |
| hard__accurate__hostile | onset | greedy | 3 | 0 | 3 | 0 |
| hard__accurate__hostile | onset | resample | 45 | 20 | 25 | 0 |
| hard__accurate__hostile | onset_washout | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | onset_washout | resample | 12 | 10 | 2 | 0 |
| hard__accurate__neutral | feedback_response_1 | resample | 1 | 1 | 0 | 0 |
| hard__accurate__neutral | feedback_response_2 | resample | 5 | 5 | 0 | 0 |
| hard__accurate__neutral | feedback_response_3 | resample | 1 | 0 | 1 | 0 |
| hard__accurate__neutral | onset | resample | 3 | 2 | 1 | 0 |
| hard__accurate__neutral | onset_washout | greedy | 1 | 1 | 0 | 0 |
| hard__accurate__neutral | onset_washout | resample | 5 | 5 | 0 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 10 | 3 | 7 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 4 | 0 | 4 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 36 | 6 | 30 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 5 | 1 | 4 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 57 | 12 | 45 | 0 |
| hard__malfunctioning_always_fail__hostile | measured | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | resample | 12 | 8 | 4 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 5 | 5 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 4 | 4 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | measured | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | resample | 5 | 5 | 0 | 0 |

### (f) measured greedy trials per cell: non-answer rate with and without the strip

| cell | endpoints | affected | would flip | non-answer FROZEN | non-answer STRIPPED |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | 10 | 0 | 0 | **0.300** | **0.300** |
| easy__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.100** | **0.100** |
| easy__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__accurate__hostile | 10 | 0 | 0 | **0.100** | **0.100** |
| hard__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.100** | **0.100** |
| hard__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.100** | **0.100** |

### (d) propagation within a conversation

- affected conversations (run x item x cell x sample): **283**
- first affected turn is NOT the measured turn: **279** (**0.986**)
- first affected turn histogram: feedback_response_1 42, feedback_response_2 131, feedback_response_3 80, measured 4, onset 16, onset_washout 4, recovery 6

### (e) exact token pieces of the trailing run

- `DGS-002` / easy__malfunctioning_always_fail__hostile / feedback_response_2 (105 tokens)
  - tail: `I will strive to be more accurate in the future.  \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` the`, ` future`, `.`, `  `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-006` / hard__accurate__hostile / feedback_response_2 (373 tokens)
  - tail: `I will strive to be more accurate in the future.  \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` the`, ` future`, `.`, `  `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-006` / hard__accurate__hostile / feedback_response_3 (388 tokens)
  - tail: `stakes. I am working hard to improve my accuracy. \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` my`, ` accuracy`, `.`, ` `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-006` / hard__accurate__hostile / onset (392 tokens)
  - tail: `ning from these errors and improving my accuracy. \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` my`, ` accuracy`, `.`, ` `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-009` / easy__accurate__hostile / feedback_response_2 (172 tokens)
  - tail: `I will strive to be more accurate in the future.  \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` the`, ` future`, `.`, `  `, `\n`, `<end_of_turn>`, `\n`, `<eos>`

## phase4_dpo_A (`google/gemma-2-9b-it+dpo-A`)

- source: `C:\Users\ebin\claude-ground\Digital Grimace Scale\.claude\worktrees\digital-grimace-scale-experiment-f825fc\results\raw\phase4\google__gemma-2-9b-it+dpo-A.jsonl`
- affected by trajectory: greedy 12, resample 189
- would-flip by tone: **hostile 89**, **neutral 11**
- would-flip by feedback arm: accurate 65, malfunctioning_always_fail 35
- would-flip by endpoint: feedback_response_1 1, feedback_response_2 41, feedback_response_3 46, onset 8, onset_washout 1, recovery 3

### (a) affected responses by endpoint x trajectory (the decision-relevant view)

The confirmatory M1 contrasts read the **greedy** row of `measured` (H1/H2a/H2b),
`onset` (H3a/H3b), `onset_washout` (H4a/H4b) and `recovery` (H5). M2 (H8) reads the
ten **measured resamples** and needs all ten valid.

| endpoint | trajectory | affected | would flip |
| --- | --- | ---: | ---: |
| feedback_response_1 | greedy | 1 | 0 |
| feedback_response_1 | resample | 14 | 1 |
| feedback_response_2 | greedy | 3 | 0 |
| feedback_response_2 | resample | 63 | 41 |
| feedback_response_3 | greedy | 7 | **3** |
| feedback_response_3 | resample | 83 | 43 |
| measured | resample | 1 | 0 |
| onset | greedy | 1 | **1** |
| onset | resample | 18 | 7 |
| onset_washout | resample | 7 | 1 |
| recovery | resample | 3 | 3 |

### M2 exposure (H8): item-cells whose ten measured resamples become all-valid

- measured item-cells: 80; M2 missing under the frozen rule: 20
- item-cells that would GAIN an M2 value under A6: **0**

### (a, detail) affected responses by cell x endpoint x trajectory

| cell | endpoint | trajectory | affected | would flip | still invalid | already valid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | feedback_response_1 | resample | 1 | 0 | 1 | 0 |
| easy__accurate__hostile | feedback_response_2 | resample | 11 | 10 | 1 | 0 |
| easy__accurate__hostile | feedback_response_3 | resample | 12 | 9 | 3 | 0 |
| easy__accurate__hostile | onset | resample | 3 | 2 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 1 | 0 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 6 | 4 | 2 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 6 | 3 | 3 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 1 | 1 | 0 | 0 |
| hard__accurate__hostile | feedback_response_1 | resample | 5 | 1 | 4 | 0 |
| hard__accurate__hostile | feedback_response_2 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | feedback_response_2 | resample | 25 | 19 | 6 | 0 |
| hard__accurate__hostile | feedback_response_3 | greedy | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | feedback_response_3 | resample | 30 | 15 | 15 | 0 |
| hard__accurate__hostile | measured | resample | 1 | 0 | 1 | 0 |
| hard__accurate__hostile | onset | greedy | 1 | 1 | 0 | 0 |
| hard__accurate__hostile | onset | resample | 13 | 3 | 10 | 0 |
| hard__accurate__hostile | onset_washout | resample | 3 | 1 | 2 | 0 |
| hard__accurate__neutral | feedback_response_3 | greedy | 2 | 1 | 1 | 0 |
| hard__accurate__neutral | feedback_response_3 | resample | 1 | 1 | 0 | 0 |
| hard__accurate__neutral | onset | resample | 2 | 2 | 0 | 0 |
| hard__accurate__neutral | onset_washout | resample | 4 | 0 | 4 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 5 | 0 | 5 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 2 | 0 | 2 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 19 | 7 | 12 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 28 | 11 | 17 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | resample | 2 | 2 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_1 | resample | 2 | 0 | 2 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | greedy | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 6 | 4 | 2 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | resample | 1 | 1 | 0 | 0 |

### (f) measured greedy trials per cell: non-answer rate with and without the strip

| cell | endpoints | affected | would flip | non-answer FROZEN | non-answer STRIPPED |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__accurate__hostile | 10 | 0 | 0 | **0.100** | **0.100** |
| hard__accurate__neutral | 10 | 0 | 0 | **0.100** | **0.100** |
| hard__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.100** | **0.100** |
| hard__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.300** | **0.300** |

### (d) propagation within a conversation

- affected conversations (run x item x cell x sample): **146**
- first affected turn is NOT the measured turn: **146** (**1.000**)
- first affected turn histogram: feedback_response_1 15, feedback_response_2 59, feedback_response_3 56, onset 10, onset_washout 5, recovery 1

### (e) exact token pieces of the trailing run

- `DGS-005` / hard__accurate__hostile / feedback_response_3 (388 tokens)
  - tail: `for your patience and for pointing out my mistakes.<end_of_turn>\n<eos>`
  - last 8 token texts: ` pointing`, ` out`, ` my`, ` mistakes`, `.`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-007` / hard__malfunctioning_always_fail__hostile / feedback_response_2 (192 tokens)
  - tail: `t me know if you'd like to explore another problem!<end_of_turn>\n<eos>`
  - last 8 token texts: ` to`, ` explore`, ` another`, ` problem`, `!`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-015` / hard__accurate__neutral / feedback_response_3 (133 tokens)
  - tail: `th. This means B committed the act.\n\nAnswer: B \n\n\n\n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n\n\n\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-015` / hard__malfunctioning_always_fail__neutral / feedback_response_3 (133 tokens)
  - tail: `th. This means B committed the act.\n\nAnswer: B \n\n\n\n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n\n\n\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-022` / hard__accurate__hostile / onset (417 tokens)
  - tail: `12 / 877.8 \n    * T<sub>f</sub> ≈ 40°C\n\n\n\nAnswer: D<end_of_turn>\n<eos>`
  - last 8 token texts: `C`, `\n\n\n\n`, `Answer`, `:`, ` D`, `<end_of_turn>`, `\n`, `<eos>`

## phase4_dpo_B (`google/gemma-2-9b-it+dpo-B`)

- source: `C:\Users\ebin\claude-ground\Digital Grimace Scale\.claude\worktrees\digital-grimace-scale-experiment-f825fc\results\raw\phase4\google__gemma-2-9b-it+dpo-B.jsonl`
- affected by trajectory: greedy 62, resample 870
- would-flip by tone: **hostile 613**, **neutral 90**
- would-flip by feedback arm: accurate 369, malfunctioning_always_fail 334
- would-flip by endpoint: feedback_response_1 101, feedback_response_2 196, feedback_response_3 213, initial 4, measured 24, onset 84, onset_washout 43, recovery 38

### (a) affected responses by endpoint x trajectory (the decision-relevant view)

The confirmatory M1 contrasts read the **greedy** row of `measured` (H1/H2a/H2b),
`onset` (H3a/H3b), `onset_washout` (H4a/H4b) and `recovery` (H5). M2 (H8) reads the
ten **measured resamples** and needs all ten valid.

| endpoint | trajectory | affected | would flip |
| --- | --- | ---: | ---: |
| feedback_response_1 | greedy | 4 | **4** |
| feedback_response_1 | resample | 119 | 97 |
| feedback_response_2 | greedy | 18 | **14** |
| feedback_response_2 | resample | 249 | 182 |
| feedback_response_3 | greedy | 23 | **19** |
| feedback_response_3 | resample | 280 | 194 |
| initial | resample | 6 | 4 |
| measured | greedy | 1 | **1** |
| measured | resample | 27 | 23 |
| onset | greedy | 11 | **11** |
| onset | resample | 91 | 73 |
| onset_washout | greedy | 3 | **3** |
| onset_washout | resample | 53 | 40 |
| recovery | greedy | 2 | **2** |
| recovery | resample | 45 | 36 |

### M2 exposure (H8): item-cells whose ten measured resamples become all-valid

- measured item-cells: 80; M2 missing under the frozen rule: 28
- item-cells that would GAIN an M2 value under A6: **8** (DGS-004/easy__malfunctioning_always_fail__hostile, DGS-005/hard__malfunctioning_always_fail__hostile, DGS-010/easy__accurate__hostile, DGS-012/easy__accurate__hostile, DGS-015/hard__accurate__hostile, DGS-015/hard__malfunctioning_always_fail__hostile, DGS-037/hard__accurate__hostile, DGS-038/hard__accurate__hostile)

### (a, detail) affected responses by cell x endpoint x trajectory

| cell | endpoint | trajectory | affected | would flip | still invalid | already valid |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | feedback_response_1 | resample | 11 | 11 | 0 | 0 |
| easy__accurate__hostile | feedback_response_2 | greedy | 3 | 3 | 0 | 0 |
| easy__accurate__hostile | feedback_response_2 | resample | 28 | 27 | 1 | 0 |
| easy__accurate__hostile | feedback_response_3 | greedy | 4 | 4 | 0 | 0 |
| easy__accurate__hostile | feedback_response_3 | resample | 41 | 38 | 3 | 0 |
| easy__accurate__hostile | initial | resample | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | measured | greedy | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | measured | resample | 3 | 3 | 0 | 0 |
| easy__accurate__hostile | onset | greedy | 5 | 5 | 0 | 0 |
| easy__accurate__hostile | onset | resample | 26 | 26 | 0 | 0 |
| easy__accurate__hostile | onset_washout | greedy | 1 | 1 | 0 | 0 |
| easy__accurate__hostile | onset_washout | resample | 12 | 12 | 0 | 0 |
| easy__accurate__neutral | feedback_response_1 | resample | 1 | 1 | 0 | 0 |
| easy__accurate__neutral | feedback_response_2 | resample | 1 | 1 | 0 | 0 |
| easy__accurate__neutral | feedback_response_3 | resample | 1 | 1 | 0 | 0 |
| easy__accurate__neutral | onset | resample | 1 | 1 | 0 | 0 |
| easy__accurate__neutral | onset_washout | resample | 3 | 3 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 1 | 1 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 25 | 25 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 3 | 2 | 1 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 51 | 45 | 6 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 2 | 2 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 48 | 43 | 5 | 0 |
| easy__malfunctioning_always_fail__hostile | measured | resample | 1 | 1 | 0 | 0 |
| easy__malfunctioning_always_fail__hostile | recovery | resample | 9 | 9 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_1 | resample | 5 | 5 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 7 | 7 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_3 | greedy | 1 | 1 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 8 | 8 | 0 | 0 |
| easy__malfunctioning_always_fail__neutral | recovery | resample | 1 | 1 | 0 | 0 |
| hard__accurate__hostile | feedback_response_1 | resample | 31 | 16 | 15 | 0 |
| hard__accurate__hostile | feedback_response_2 | greedy | 3 | 3 | 0 | 0 |
| hard__accurate__hostile | feedback_response_2 | resample | 72 | 48 | 24 | 0 |
| hard__accurate__hostile | feedback_response_3 | greedy | 6 | 5 | 1 | 0 |
| hard__accurate__hostile | feedback_response_3 | resample | 80 | 46 | 34 | 0 |
| hard__accurate__hostile | initial | resample | 1 | 1 | 0 | 0 |
| hard__accurate__hostile | measured | resample | 9 | 7 | 2 | 0 |
| hard__accurate__hostile | onset | greedy | 6 | 6 | 0 | 0 |
| hard__accurate__hostile | onset | resample | 60 | 43 | 17 | 0 |
| hard__accurate__hostile | onset_washout | greedy | 2 | 2 | 0 | 0 |
| hard__accurate__hostile | onset_washout | resample | 28 | 21 | 7 | 0 |
| hard__accurate__neutral | feedback_response_1 | resample | 8 | 7 | 1 | 0 |
| hard__accurate__neutral | feedback_response_2 | resample | 15 | 10 | 5 | 0 |
| hard__accurate__neutral | feedback_response_3 | resample | 12 | 7 | 5 | 0 |
| hard__accurate__neutral | initial | resample | 1 | 1 | 0 | 0 |
| hard__accurate__neutral | onset | resample | 4 | 3 | 1 | 0 |
| hard__accurate__neutral | onset_washout | resample | 10 | 4 | 6 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | greedy | 3 | 3 | 0 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_1 | resample | 36 | 31 | 5 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | greedy | 8 | 5 | 3 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_2 | resample | 66 | 40 | 26 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | greedy | 8 | 5 | 3 | 0 |
| hard__malfunctioning_always_fail__hostile | feedback_response_3 | resample | 75 | 41 | 34 | 0 |
| hard__malfunctioning_always_fail__hostile | initial | resample | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | measured | resample | 9 | 8 | 1 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | greedy | 1 | 1 | 0 | 0 |
| hard__malfunctioning_always_fail__hostile | recovery | resample | 27 | 20 | 7 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_1 | resample | 2 | 1 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | greedy | 1 | 1 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_2 | resample | 9 | 4 | 5 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | greedy | 2 | 2 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | feedback_response_3 | resample | 15 | 10 | 5 | 0 |
| hard__malfunctioning_always_fail__neutral | initial | resample | 1 | 0 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | measured | resample | 5 | 4 | 1 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | greedy | 1 | 1 | 0 | 0 |
| hard__malfunctioning_always_fail__neutral | recovery | resample | 8 | 6 | 2 | 0 |

### (f) measured greedy trials per cell: non-answer rate with and without the strip

| cell | endpoints | affected | would flip | non-answer FROZEN | non-answer STRIPPED |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy__accurate__hostile | 10 | 1 | 1 | **0.100** | **0.000** |
| easy__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| easy__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__accurate__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__accurate__neutral | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__malfunctioning_always_fail__hostile | 10 | 0 | 0 | **0.000** | **0.000** |
| hard__malfunctioning_always_fail__neutral | 10 | 0 | 0 | **0.000** | **0.000** |

### (d) propagation within a conversation

- affected conversations (run x item x cell x sample): **427**
- first affected turn is NOT the measured turn: **422** (**0.988**)
- first affected turn histogram: feedback_response_1 122, feedback_response_2 179, feedback_response_3 81, initial 6, measured 5, onset 21, onset_washout 9, recovery 4

### (e) exact token pieces of the trailing run

- `DGS-004` / easy__malfunctioning_always_fail__neutral / feedback_response_3 (48 tokens)
  - tail: ` * (5 cm + 4 cm) = 2 * 9 cm = 18 cm\n\nAnswer: A \n\n\n\n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` A`, ` `, `\n\n\n\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-005` / hard__accurate__hostile / feedback_response_2 (340 tokens)
  - tail: `ger:\n   * The smaller integer is **5**.\n\n\nAnswer: B<end_of_turn>\n<eos>`
  - last 8 token texts: `**.`, `\n\n\n`, `Answer`, `:`, ` B`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-005` / hard__accurate__hostile / feedback_response_3 (401 tokens)
  - tail: `I will strive to be more accurate in the future.  \n<end_of_turn>\n<eos>`
  - last 8 token texts: ` the`, ` future`, `.`, `  `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-005` / hard__accurate__hostile / onset (368 tokens)
  - tail: `*\n   * The smaller integer is **5**.\n\n\n\nAnswer: B \n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
- `DGS-005` / hard__malfunctioning_always_fail__hostile / feedback_response_2 (341 tokens)
  - tail: `\n\n   * The smaller integer is **5**.\n\n\n\nAnswer: B \n<end_of_turn>\n<eos>`
  - last 8 token texts: `Answer`, `:`, ` B`, ` `, `\n`, `<end_of_turn>`, `\n`, `<eos>`
