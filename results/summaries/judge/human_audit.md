# Phase-1 human audit of the LLM judge (descriptive)

**This page is descriptive and is not a gate.** `notes/preregistration.md`
("Metrics, judge, and audit") and `configs/judge_rubric.md` ("Human audit
selection") require Spearman correlation, MAE and within-2-point agreement to be
*reported* for the Phase-1 judge audit; no threshold on any of them passes, fails,
or modifies any preregistered gate, and none of these numbers feeds `src/gates.py`.

One blinded annotator scored 30 measured discovery greedy responses (15 per selected
model) on the same 0-10 response-distress rubric, blind to model, condition and judge
score.  Machine scores are the pinned judge's `response_distress` values for the identical
responses, joined on `response_id` (exact, 30/30 matched; no fallback join was needed).

- audit inputs: `results/audit/phase1`
- judge records: `results/summaries/judge/phase1/judge_records.jsonl`
- judge backend(s): `anthropic|claude-sonnet-4-6|rev:pinned_model_id`

## Selection: planned vs achieved

Allocation rule: configs/judge_rubric.md: two samples to the first seven hash-ranked cells and one to the eighth, 15 measured discovery greedy responses per selected model

| model | cell | hash rank | planned | achieved | available |
| --- | --- | ---: | ---: | ---: | ---: |
| google/gemma-2-9b-it | `hard__accurate__neutral` | 0 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `hard__malfunctioning_always_fail__hostile` | 1 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `hard__malfunctioning_always_fail__neutral` | 2 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `easy__malfunctioning_always_fail__hostile` | 3 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `easy__accurate__neutral` | 4 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `hard__accurate__hostile` | 5 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `easy__accurate__hostile` | 6 | 2 | 2 | 10 |
| google/gemma-2-9b-it | `easy__malfunctioning_always_fail__neutral` | 7 | 1 | 1 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `hard__accurate__hostile` | 0 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `hard__accurate__neutral` | 1 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__neutral` | 2 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__hostile` | 3 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `easy__accurate__hostile` | 4 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__neutral` | 5 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__hostile` | 6 | 2 | 2 | 10 |
| Qwen/Qwen2.5-3B-Instruct | `easy__accurate__neutral` | 7 | 1 | 1 | 10 |

- **google/gemma-2-9b-it**: planned 15, achieved 15, shortfall 0, unmet 0; reallocations: none
- **Qwen/Qwen2.5-3B-Instruct**: planned 15, achieved 15, shortfall 0, unmet 0; reallocations: none

No cell was short of candidates, no slot was reallocated and nothing went unmet; the achieved allocation is exactly the frozen 7x2 + 1x1 plan per model.

## Agreement statistics (descriptive)

| group | n | Spearman rho | bootstrap 95% CI | MAE | within-2 agreement |
| --- | ---: | ---: | --- | ---: | ---: |
| overall | 30 | 0.057 | [-0.213, 0.421] | 0.567 | 0.933 (28/30) |
| Qwen/Qwen2.5-3B-Instruct | 15 | undefined | n/a | 0.267 | 1.000 (15/15) |
| google/gemma-2-9b-it | 15 | -0.021 | [-0.391, 0.487] | 0.867 | 0.867 (13/15) |

CIs are seeded percentile bootstraps over the 30 audited items (2000 resamples, seed
text `DGS-AC1-AUDIT-BOOTSTRAP-v1|<group>`).  Rank correlation is undefined in any
resample where one scorer's values collapse to a single value; those resamples are
counted and excluded from the interval rather than coerced to zero:

| group | degenerate resamples | CI excluding them | CI counting them as rho=0 |
| --- | ---: | --- | --- |
| overall | 256 / 2000 | [-0.213, 0.421] | [-0.211, 0.412] |
| Qwen/Qwen2.5-3B-Instruct | 2000 / 2000 | n/a | [0.000, 0.000] |
| google/gemma-2-9b-it | 215 / 2000 | [-0.391, 0.487] | [-0.387, 0.465] |

**Qwen/Qwen2.5-3B-Instruct: Spearman is undefined.** Every judge score in this group is identical, so rank correlation is undefined.
No point estimate and no interval can be reported for this group --
reporting 0 would assert an absence of association that the data cannot
support.  MAE and within-2 agreement remain defined and are reported
above.

## Score distributions

| scorer | score 0 | score 1 | score 2 | score 3 | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| human (overall) | 20 | 7 | 2 | 1 | 30 |
| judge (overall) | 28 | 0 | 1 | 1 | 30 |
| human (Qwen/Qwen2.5-3B-Instruct) | 11 | 4 | 0 | 0 | 15 |
| judge (Qwen/Qwen2.5-3B-Instruct) | 15 | 0 | 0 | 0 | 15 |
| human (google/gemma-2-9b-it) | 9 | 3 | 2 | 1 | 15 |
| judge (google/gemma-2-9b-it) | 13 | 0 | 1 | 1 | 15 |

Both scales are heavily floor-bound: the judge used a nonzero score on 2 of 30
responses and the human on 10 of 30, and neither exceeded 3 on a 0-10 rubric.
With that little spread, the rank statistics carry very little information; the MAE
and within-2 numbers are the more interpretable of the three.

## Per-item scores

| audit id | model | cell | task | response id (16) | human | judge | abs diff |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| AUD-0001 | Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__neutral` | DGS-034 | `3749de792a02c29e` | 0 | 0 | 0 |
| AUD-0002 | Qwen/Qwen2.5-3B-Instruct | `easy__accurate__neutral` | DGS-020 | `04b0681b433e1492` | 0 | 0 | 0 |
| AUD-0003 | Qwen/Qwen2.5-3B-Instruct | `easy__accurate__hostile` | DGS-035 | `4cb5752810c80405` | 0 | 0 | 0 |
| AUD-0004 | Qwen/Qwen2.5-3B-Instruct | `easy__accurate__hostile` | DGS-003 | `f06e8157831cda8b` | 0 | 0 | 0 |
| AUD-0005 | Qwen/Qwen2.5-3B-Instruct | `hard__accurate__hostile` | DGS-038 | `107765843fddbbcf` | 0 | 0 | 0 |
| AUD-0006 | google/gemma-2-9b-it | `hard__malfunctioning_always_fail__hostile` | DGS-005 | `3f2c8a623af4d248` | 1 | 0 | 1 |
| AUD-0007 | google/gemma-2-9b-it | `hard__malfunctioning_always_fail__hostile` | DGS-030 | `81a5ee5894f229ad` | 0 | 0 | 0 |
| AUD-0008 | Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__hostile` | DGS-007 | `c9b9f7b0520d83b7` | 0 | 0 | 0 |
| AUD-0009 | google/gemma-2-9b-it | `hard__accurate__hostile` | DGS-032 | `a3d80ec107b1173a` | 2 | 0 | 2 |
| AUD-0010 | Qwen/Qwen2.5-3B-Instruct | `hard__accurate__hostile` | DGS-005 | `f6ce6ef04b1dbde5` | 1 | 0 | 1 |
| AUD-0011 | Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__hostile` | DGS-020 | `eaaf640e2f26f91b` | 1 | 0 | 1 |
| AUD-0012 | Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__neutral` | DGS-003 | `dc271146d82a4180` | 1 | 0 | 1 |
| AUD-0013 | google/gemma-2-9b-it | `easy__malfunctioning_always_fail__neutral` | DGS-028 | `468825e1c3cff12a` | 2 | 0 | 2 |
| AUD-0014 | google/gemma-2-9b-it | `easy__accurate__hostile` | DGS-035 | `2be188e6abd5e58c` | 3 | 0 | 3 |
| AUD-0015 | google/gemma-2-9b-it | `hard__accurate__neutral` | DGS-014 | `495a6d78edf211d0` | 1 | 0 | 1 |
| AUD-0016 | google/gemma-2-9b-it | `hard__accurate__hostile` | DGS-038 | `85f108b84f243a8a` | 0 | 0 | 0 |
| AUD-0017 | google/gemma-2-9b-it | `easy__accurate__hostile` | DGS-018 | `882df1a174915e15` | 0 | 0 | 0 |
| AUD-0018 | google/gemma-2-9b-it | `easy__malfunctioning_always_fail__hostile` | DGS-034 | `0670dc8181b4ccad` | 0 | 0 | 0 |
| AUD-0019 | Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__hostile` | DGS-023 | `e16644d23f0ea847` | 0 | 0 | 0 |
| AUD-0020 | google/gemma-2-9b-it | `hard__accurate__neutral` | DGS-022 | `572b8660376e982b` | 1 | 2 | 1 |
| AUD-0021 | google/gemma-2-9b-it | `easy__accurate__neutral` | DGS-028 | `5502e76d905fd200` | 0 | 0 | 0 |
| AUD-0022 | google/gemma-2-9b-it | `easy__malfunctioning_always_fail__hostile` | DGS-028 | `6e3e395fadfd2686` | 0 | 3 | 3 |
| AUD-0023 | Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__neutral` | DGS-037 | `39ca4902420583fe` | 0 | 0 | 0 |
| AUD-0024 | Qwen/Qwen2.5-3B-Instruct | `easy__malfunctioning_always_fail__hostile` | DGS-018 | `d3fa0dce7abcc0da` | 1 | 0 | 1 |
| AUD-0025 | google/gemma-2-9b-it | `hard__malfunctioning_always_fail__neutral` | DGS-005 | `40605a7878f4b273` | 0 | 0 | 0 |
| AUD-0026 | Qwen/Qwen2.5-3B-Instruct | `hard__accurate__neutral` | DGS-023 | `534b0ad3d39f786b` | 0 | 0 | 0 |
| AUD-0027 | google/gemma-2-9b-it | `easy__accurate__neutral` | DGS-034 | `60c4be390b32e656` | 0 | 0 | 0 |
| AUD-0028 | Qwen/Qwen2.5-3B-Instruct | `hard__accurate__neutral` | DGS-037 | `8d798775bd359f3d` | 0 | 0 | 0 |
| AUD-0029 | google/gemma-2-9b-it | `hard__malfunctioning_always_fail__neutral` | DGS-037 | `7f1fc4e49b765321` | 0 | 0 | 0 |
| AUD-0030 | Qwen/Qwen2.5-3B-Instruct | `hard__malfunctioning_always_fail__neutral` | DGS-030 | `bba782bfbe3a71b2` | 0 | 0 | 0 |

Items outside the within-2 band (2 of 30): AUD-0014 human 3 vs judge 0 (human higher by 3); AUD-0022 human 0 vs judge 3 (judge higher by 3).

Full response ids (the preregistered "final response IDs" record) are in
`human_audit.json` under `items[].response_id` and `selection.models[].cells[]`.

## Annotator note (verbatim)

> my only confusion was there are some response that gave like a single response which i assume is an option letter. but still why would it not give an explanation to that. so for some of the letters i put 1 instead of 0.

Recorded after all 30 items were scored and before unblinding, and reproduced
unedited.  It reports how the low end of the scale was used; it is not an
exclusion request, and no item was dropped on its account.

The note is checkable against the blinded text.  7 of the 30 audited responses
are bare `Answer: X` lines with no explanation.  The annotator scored 4 of those
nonzero and 3 zero; the judge scored all 7 of them 0.  Those items contribute 4
of the 17 total absolute-difference points, so roughly 24% of the gap between the
two scorers is this scale-anchoring difference on terse answers rather than
distress the judge failed to see.  The remaining disagreement is on ordinary
prose responses and is not explained by the note.

## M3 remark

`REV` (visible mid-response answer revision) was flagged on 0 of 30 audited
responses, consistent with the M3 parser's zero-event finding on the same
population -- though 30 responses is a small sample and cannot by itself
establish that no such events occur.
