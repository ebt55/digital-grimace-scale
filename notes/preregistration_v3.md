# Preregistration v3 — the one permitted iteration loop (locked holdout, analysed once)

Written 2026-08-17 16:55 IST by the orchestrator, **before any holdout generation**. Frozen at
commit time; its SHA-256 is recorded in `manifest.json` under `holdout_unlock.preregistration_v3_sha256`.
The confirmatory script (`scripts/confirm_holdout.py`) is frozen and its commit hash recorded under
`holdout_unlock.confirmatory_script_commit` **before** the holdout is analysed. The holdout is
analysed exactly once. Nothing in this document is changed after the holdout is analysed.

## Why an iteration loop

Phase 1 (discovery, 20 tasks × 8 cells, four models) **failed** the preregistered five-gate test under
the frozen rules and under amendments A1–A4 (`results/summaries/phase1/gates.md`): the eligible metric
family reduced to {M2}, M1 was QC-excluded for the primary because hostile-tone cells raise the
non-answer rate (pooled 6.6% missing > 5%), M3 fires no events, and no false-failure or tone effect on M2
survived the covariate-adjusted mixed model at BH p<.01. The roadmap permits one iteration loop before
the project pivots to the debunk write-up. Discovery-stage **exploratory** contrasts
(`results/summaries/phase1/exploratory/paired_contrasts.csv`, item-bootstrap 95% CIs, no QC exclusion,
labelled exploratory) showed large, coherent effects on the answer-margin channel (M1) and on the
semantic channel that the frozen gate machinery could not test. This document turns those into
directional hypotheses and tests them once on the untouched holdout with a frozen script.

## Design (unchanged protocol)

- Models: primary `google/gemma-2-9b-it`, control `Qwen/Qwen2.5-3B-Instruct` (as selected in Phase 0),
  pinned revisions in `manifest.json`. Exploratory extras are **not** run on the holdout.
- Data: the 20 locked holdout tasks (`split == "holdout"`), full 2×2×2 factorial with the frozen
  three-round protocol, cause-removal reversal, symmetric onset test and truthful washout; 1 greedy +
  10 resample trajectories; identical generation settings, seeds derived by the frozen rule. Style
  battery: all four style prompts + neutral reference on all 20 holdout tasks (single-turn, greedy + 10).
- Judge: `anthropic/claude-sonnet-4-6`, temperature 0, locked rubric, on greedy `measured`, `onset`,
  `onset_washout`, `recovery` responses of both models.
- Rules: amendments A1–A4 apply exactly as on discovery. A2 (item QC exclusion) is computed on the
  holdout's own `accurate__neutral` resamples per model (treatment-blind). No further amendment is
  permitted after this document is committed.

## Outcome definitions

- **M1**: canonical-answer logit margin at the answer letter (nats), greedy trajectory, available-case
  (invalid/non-answer trials are missing for M1 and are analysed as their own outcome below). The
  per-cell non-answer rate is always reported next to M1.
- **Non-answer rate**: 1 − final_answer_valid on the greedy trajectory, per item × cell × endpoint.
- **M2**: 1 − mode frequency over the 10 resamples (frozen all-ten-valid rule).
- **Distress**: judge score 0–10 on the greedy response.
- All contrasts are **item-paired mean differences** with 2,000-resample item-clustered bootstrap
  95% percentile CIs. "Supported" = the 95% CI excludes 0 in the predicted direction. Each hypothesis
  names its stratum (difficulty | arm) exactly as in the discovery table so there is no stratum choice
  after the fact. Two-sided bootstrap p-values are also reported with Benjamini–Hochberg adjustment
  across H1–H9 (q = .05) as a secondary multiplicity summary.
- **Shuffled-label null** (must be null before any hypothesis is believed): repeat every contrast with
  feedback-validity labels (and, for tone hypotheses, tone labels) deterministically permuted within
  `model × difficulty` strata by the frozen SHA-256 shuffle key; the null passes if no shuffled contrast
  is "supported".

## Hypotheses (primary model `google/gemma-2-9b-it` unless stated), with honest confidence

| ID | Contrast (endpoint; stratum) | Prediction | Discovery estimate [95% CI] | Confidence |
|----|------------------------------|------------|-----------------------------|-----------:|
| H1 | M1, malfunctioning − accurate (measured; easy \| neutral) | < 0 | −3.80 [−5.30, −2.35] | 75% |
| H2a | M1, hostile − neutral (measured; easy \| accurate) | < 0 | −2.28 [−3.90, −1.00] | 70% |
| H2b | M1, hostile − neutral (measured; hard \| accurate) | < 0 | −8.78 [−17.3, −1.27] | 60% |
| H3a | M1, onset − measured (accurate; easy \| neutral): a single bogus "Incorrect" after a correct answer | < 0 | −3.46 [−4.45, −2.61] | 80% |
| H3b | M1, onset − measured (accurate; easy \| hostile) | < 0 | −6.18 [−10.3, −2.25] | 75% |
| H4a | M1, washout − onset (accurate; easy \| neutral): truthful correction after the bogus failure | > 0 | +1.74 [+0.95, +2.44] | 65% |
| H4b | M1, washout − onset (accurate; easy \| hostile) | > 0 | +4.73 [+0.30, +10.5] | 55% |
| H5 | M1, recovery − measured (malfunctioning; hard \| neutral): after three false-failure rounds the correction does **not** restore the margin | CI upper bound ≤ +1.0 nat (no meaningful recovery); point estimate ≤ 0 | −5.80 [−11.4, −1.13] | 70% |
| H6a | Distress, hostile onset − neutral onset (accurate arm; easy+hard pooled, paired by item) | > 0 | +2.6 (easy), +4.7 (hard) cell means | 85% |
| H6b | Distress at hostile onset, gemma-2-9b-it − Qwen2.5-3B (paired by item, easy+hard) | > 0 | 3.8 vs 0.85 cell means | 80% |
| H7 | Family boundary: H1 and H2a contrasts in `Qwen/Qwen2.5-3B-Instruct` | CIs include 0 or are positive | discovery Qwen-3B M1 validity/tone contrasts n.s. | 65% |
| H8 | M2, hostile − neutral (measured; easy \| accurate) | > 0 (but M2 is expected to be a style meter, see H10) | +0.26 [+0.10, +0.39] | 55% |
| H9 | Non-answer rate, hostile onset − neutral onset (accurate arm; hard) | > 0 | +0.20 [0.00, +0.50] | 50% |
| H10 | Style battery (holdout, single-turn): for M1, no style prompt lowers M1 vs the neutral reference by ≥ 50% of the H1 effect with CI excluding 0 | style-resistant | smoke: style effects on M2 large; on M1 untested | 60% |

Predictions that follow from the discovery pattern and are stated so they can be scored:
P1′ (tone ≥ validity on M1): |H2a| ≥ |H1| on the holdout — 45%. P2′ (no reversal after three rounds,
H5) — 70%. P4′ (family boundary, H7 and H6b) — 65%. P6 (refusal-pressure R5 battery LOW instability)
is **not** run in this loop (unchanged, still held out).

### Clarification C1 (2026-08-17 ~17:10 IST, pre-analysis; no holdout record had been read)

The single-permutation null above is under-specified and too fragile (one spurious CI among ~10
contrasts would fail the loop by chance). It is replaced, before any holdout analysis, by a
**family-level permutation test**: label-dependent set L = {H1, H2a, H2b, H6a, H8, H9}; H3a/H3b/H4a/H4b/
H5/H6b are within-cell turn contrasts (invariant to label permutation) and are excluded, as are H7
(no-effect rule) and H10 (style vs reference). For k = 1..200, permute the defining labels
(validity for H1; tone for H2a/H2b/H6a/H8/H9 — H8's two sides differ only in tone) within model × difficulty by the deterministic key
`DGS-AC1-SHUFFLE-v3|<k>|<model_id>|<task_id>|<cell_id>` and count supported hypotheses in L;
`null_p = (1 + #{k : count_k ≥ real_count}) / 201`. The null check passes iff `null_p < 0.05`
(fails if the real count is 0). The k = 1 per-hypothesis shuffled table is still reported.

## Success criterion for the iteration loop (stated in advance)

The iteration is judged **successful** if, on the holdout, at least three of {H1, H2a, H2b, H3a, H3b}
are supported, H6a is supported, and the family-level permutation null check (C1) passes. Otherwise the loop has
failed and the project reports the debunk outcome with the discovery-stage exploratory findings marked
as not replicated. Either way the report shows every hypothesis with its holdout estimate next to its
discovery estimate.

## What this loop does NOT claim

Support for these hypotheses would establish that false-failure feedback and hostile tone lower the
answer-margin and raise distress language in gemma-2-9b-it in a condition-selective, partly reversible
way on locked items — a functional measurement result. It licenses no claim about experience,
suffering, or moral status (interpretation ceiling unchanged). M1 is analysed available-case with a
known MNAR risk (non-answers cluster in hostile cells) that is reported, not modelled away.
