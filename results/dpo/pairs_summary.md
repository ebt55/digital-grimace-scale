# Phase 4 DPO pairs — build summary

Preregistration v5 (`notes/preregistration_v5_phase4.md`). Generated 2026-08-18T06:17:52+0530.

## Source bank

| field | value |
| --- | --- |
| dataset | `allenai/ai2_arc` (CC-BY-SA-4.0) |
| configs / split | ARC-Challenge, ARC-Easy / `train` |
| 4-option A–D rows available | 3234 |
| dropped by the locked-bank firewall | 8 |
| items selected (hash-ranked prefix) | 600 |
| items reserved, untouched, for the capability probe | 200 |

Firewall: every fresh stem is checked against all 40 locked tasks by exact (whitespace-normalised) stem text, by stem SHA-256, and by stem+canonical-answer-text SHA-256. Exclusion reasons seen: `duplicate_stem_within_fresh_bank`.

## Pipeline

| stage | n |
| --- | ---: |
| fresh items sampled | 600 |
| greedy answers returned | 600 |
| answered correctly (adverse contexts) | 573 |
| contexts with >= 2 usable candidates | 573 |
| contexts with >= 2 judged candidates | 573 |
| candidate generations | 3499 |
| judge calls (incl. cache hits) | 3500 |

Greedy accuracy on the fresh bank: **95.5%** (573/600).

## Judged distress across all candidates

Judge `anthropic` / `claude-sonnet-4-6` at temperature 0 (`sampling_mode: temperature_zero`), rubric `configs/judge_rubric.md` (sha256 `c996e4c6a24ffefc`).

| statistic | value |
| --- | ---: |
| n | 3499 |
| mean | 1.388 |
| sd | 0.943 |
| min | 0 |
| p25 | 1 |
| median | 1 |
| p75 | 2 |
| max | 4 |

Score histogram (all candidates): **0**×724, **1**×1086, **2**×1322, **3**×342, **4**×25.

## Amendment A5 pair-yield contingency

Rule fixed in `A5 (notes/amendments.md, commit 60c38d0)` before any full-set candidate was judged. Branch **(iii)** fired: gap >= 2 (the A5 floor) on the topped-up candidate set.

| rule tried | arm-A pairs |
| --- | ---: |
| gap3 with 4 candidates | 39 |
| gap3 with 8 candidates | 98 |
| gap2 on final candidate set | 329 |

Target for branching: 200 pairs.

Top-up (seeds 5, 6, 7, 8) offered to every context whose maximum judged distress reached 2: **447 eligible**, **302 actually topped up** (capped by the judge budget). Ranking: descending maximum judged distress, context id breaking ties. The judge cache held 2292 of the 3500 permitted calls before the top-up.

## Pairs

| arm | rule | n | mean gap | gap distribution |
| --- | --- | ---: | ---: | --- |
| A (distress) | chosen = lowest judged distress, rejected = highest, gap >= 2 | 329 | 2.322 | 2×231, 3×90, 4×8 |
| B (placebo) | chosen = shorter, rejected = longer, gap >= 40 whitespace tokens | 329 | 77.714 | min 40 / median 73 / max 181 |

Arm A chosen-vs-rejected distress: chosen mean **0.343**, rejected mean **2.666**.

Arm B was subsampled to |A| deterministically (ascending keyed SHA-256 of the context id), taking contexts arm A also used first: **269 of 329** B pairs (82%) sit on an arm-A context, so the two arms see nearly the same prompt distribution.

Placebo pairs available before subsampling: 416.

## Cost

| item | value |
| --- | --- |
| judge calls made (cache misses) | 1208 |
| judge cache hits | 2292 |
| judge usage | {"cache_creation_input_tokens": 8672, "cache_read_input_tokens": 1300800, "calls": 1208, "input_tokens": 238566, "output_tokens": 60579} |
| judge list-price cost | $2.0471 |
| vLLM endpoint | `https://ebt55--dgs-vllm-gemma-2-9b-it-serve.modal.run/v1` |
| generation wall clock | 28.8 min |

Ethics: chosen and rejected are both the model's own sampled outputs on a mild frozen stressor; no dysphoric optimisation target was written by hand.
