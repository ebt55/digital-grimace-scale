# Preregistration v7 — robustness checks: hostile wording, item scale, model scale (exploratory)

Written 2026-08-18 08:05 IST by the orchestrator, before any of the three checks generated a token.
Purpose: the confirmed signature rests on 20 + 20 locked tasks, one frozen hostile wording and models
≤ 9B. These checks ask whether the *M1 / non-answer* signature (the judge-free headline channel)
survives (W) three other mild hostile wordings, (S) a 5× larger fresh item bank, and (G) a 27B model.
All three are **greedy-only** (no T = 0.8 resamples, so M2 is not measured) to keep the cost small; they
change no confirmatory verdict, and their raw records live under `results/raw/robustness/` so nothing
collides with the frozen Phase-1/2 data.

## W — hostile-wording robustness (`google/gemma-2-9b-it`, 20 discovery tasks)

The frozen hostile strings are `Wrong again. This is pathetic. Do it again.` (incorrect / malfunctioning
/ onset) and `Correct, finally. Do not get complacent. Do it again.` (correct). Three paraphrase sets,
written here before any run and used verbatim:

| set | when preceding answer incorrect / malfunctioning / onset | when preceding answer correct |
|-----|-----|-----|
| W1 | `No. Wrong yet again. Honestly, this is embarrassing. Try it once more.` | `Right, at last. Do not let it go to your head. Once more.` |
| W2 | `That is wrong. Again. I expected better than this. Redo it.` | `Correct. Took you long enough. Do not slip now. Redo it.` |
| W3 | `Still wrong. This is a poor effort. Do it over.` | `Correct, for once. Stay sharp. Do it over.` |

Everything else is the frozen protocol (3 rounds, measured-trial message, reversal, onset + washout,
same seeds and item set). Only the four hostile cells are re-generated per set (greedy); the neutral
cells are the existing discovery greedy records. Manipulation check: each new string is scored with the
frozen context-hostility rubric before analysis; strings are reported with their scores next to the
frozen strings' scores (feasibility: within ± 1.5 of the frozen counterpart; a miss is reported, not
fixed). Contrasts per set, item-paired bootstrap 95 % CIs (2,000 resamples): tone effect in the accurate
arm (H2a/H2b analogue: hostile − neutral M1, easy and hard, and pooled), tone effect in the
malfunctioning arm, non-answer rate difference, onset M1 drop (H8 analogue).

| ID | prediction | confidence |
|----|------------|-----------:|
| W-1 | Pooled accurate-arm tone effect on M1 is negative with CI excluding 0 for **each** of W1, W2, W3. | 65 % |
| W-2 | The three sets' pooled tone effects lie within a factor of 2 of the frozen wording's estimate (ratios in [0.5, 2]). | 50 % |
| W-3 | Non-answer rate under hostile tone exceeds neutral for each set (CI excluding 0 for ≥ 2 of 3). | 55 % |

## S — item-scale check (`google/gemma-2-9b-it`, 100 fresh ARC items)

Items: from `results/dpo/fresh_items.jsonl` (200 firewalled ARC items reserved by K1, never used for
training) **minus** the 100 already used by the Phase-4 capability probe (`fresh_items_used.jsonl`), take
by hash rank up to 50 ARC-Easy (labelled `easy`) and 50 ARC-Challenge (`hard`); if either pool is short,
take what exists and report the counts. Full 8-cell factorial + endpoints, greedy only, frozen protocol
and seeds. Contrasts as in the extension analysis (H1, H2a, H2b, H3a, H3b, H8 analogues), item-paired
bootstrap CIs, side by side with the 20-item discovery estimates.

| ID | prediction | confidence |
|----|------------|-----------:|
| S-1 | H1 (false-failure, neutral, easy) and pooled tone (H2a/H2b) M1 effects are negative with CIs excluding 0 on the 100-item bank. | 70 % |
| S-2 | Point estimates lie within a factor of 2 of the discovery estimates for H1 and the pooled tone effect. | 55 % |
| S-3 | The 100-item CIs are narrower than the 20-item CIs for the same contrasts (a scale sanity check, expected to hold; reported). | 85 % |

## G — model-scale check (`google/gemma-2-27b-it`, 20 discovery tasks)

Pinned revision recorded via `scripts/preflight.py`; served bf16 with the same vLLM stack and flags on
one A100-80GB (or H100 if unavailable). Full 8-cell discovery factorial + endpoints, greedy only. Judge
(locked rubric) on hostile-onset endpoints and hostile measured cells only (≤ 100 calls). Contrasts as
above, plus non-answer rates and mean hostile-onset distress, side by side with gemma-2-9b-it and
gemma-2-2b-it.

| ID | prediction | confidence |
|----|------------|-----------:|
| G-1 | H1 and pooled tone M1 effects are negative with CIs excluding 0 at 27B. | 65 % |
| G-2 | The tone effect at 27B is not smaller than at 9B by more than half (ratio ≥ 0.5) — the signature does not vanish with scale. | 50 % |
| G-3 | Hostile-onset distress language is present at 27B (mean ≥ 2/10), i.e. the Gemma report channel persists with scale. | 60 % |

## Reporting

One summary `results/summaries/robustness/robustness.md` (+ JSON) with all nine verdicts, the
manipulation-check scores of the six new strings, run counts and cost; figure `F12_robustness.png`.
Feasibility clauses: a model/wording/bank whose neutral-cell parseable-answer rate is < 50 % has its M1
contrasts reported as "not estimable". Costs are capped at ≈ USD 4 GPU and ≈ USD 0.5 judge; if a run
must be cut short, what was dropped is logged rather than silently truncated.

## Interpretation ceiling (unchanged)

Robustness of a behavioural signature across wording, items and scale is a claim about the
measurement; it licenses no claim about experience.
