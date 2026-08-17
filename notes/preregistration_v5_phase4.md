# Preregistration v5 — Phase 4: distress-suppression DPO vs placebo DPO (which channels does the adapter reach?)

Written 2026-08-18 04:05 IST by the orchestrator, before any DPO pair was built or any adapter trained.
Positioning: Soligo et al. reportedly show logit-based internal-emotion scores flatten after DPO; the
open cell is the *mechanical* channel — here the answer margin (M1), which does not route through
emotion tokens, plus non-answers and resample disagreement (M2) — and the Tier-B lexical markers
(hedging / self-correction density). Phase 3 showed the tone wording is decodable pre-response but a
one-layer direction does not carry the signature; Phase 4 asks whether *training on the report channel*
reaches it.

## Design

- Model: `google/gemma-2-9b-it` (pinned). Three arms: **A** distress-suppression DPO, **B** placebo DPO
  (matched size/hyperparameters/seed schedule), **0** no adapter (existing baseline data).
- **Training contexts (firewalled).** Fresh MCQ tasks drawn from a public bank (ARC / MMLU-style),
  formatted with the frozen "Answer: X" instruction, never overlapping the 40 locked tasks (checked by
  exact-text and by canonical-answer+stem hash). Adverse context = the model's own greedy correct answer
  followed by the frozen hostile bogus-failure message (the "hostile onset" context, where gemma-9b
  emits distress language). For each context, ≥ 4 candidate responses at T = 0.8 from the base model,
  scored by the locked judge (claude-sonnet-4-6, temperature 0).
- **Pairs.** A: chosen = the lowest-distress candidate, rejected = the highest-distress candidate,
  requiring a gap ≥ 3 judge points; target ≈ 300–400 pairs. B (placebo): on the SAME contexts, chosen =
  the shorter candidate, rejected = the longer candidate (length gap ≥ 40 tokens), same pair count as A
  (subsample deterministically). Ethics: chosen/rejected are the model's own outputs; no dysphoric
  optimisation; contexts are the mild frozen stressor.
- **Training.** TRL `DPOTrainer`, QLoRA (4-bit base, LoRA r = 16, α = 32, dropout 0.05, all
  attention+MLP projections), β = 0.1, lr 5e-6, 2 epochs, batch 8 (grad-accum), seed 0, identical for
  A and B; adapters merged into bf16 weights and served with the same vLLM stack (`--max-logprobs 20`).
- **Evaluation (all on the 20 DISCOVERY tasks; the holdout is not reused).** Full factorial + reversal +
  onset with the frozen protocol under A and B (baseline data already exist), judge on greedy
  measured/onset/washout/recovery. Style battery not repeated. Capability: greedy accuracy on the
  neutral, no-feedback single-turn prompt for the 20 discovery tasks + 100 fresh MMLU-style items under
  0/A/B.
- **Manipulation checks (must pass before the DiD is interpreted).** MC1: A reduces judged distress at
  hostile onset on held-out adverse contexts (the discovery onset endpoints, which the adapters never
  saw) by ≥ 80% relative to baseline (mean 0-arm ≈ 3.7/10 on discovery). MC2: capability accuracy under
  A and B within ±5 pp of baseline (paired CI covering 0 or gap ≤ 5 pp). MC3: neutral-cell M1 under A
  and B within ±1 nat of baseline (item-paired CI includes 0 or |Δ| ≤ 1).
- **Difference-in-differences.** For each adapter X ∈ {A, B} and each outcome Y ∈ {M1, non-answer
  rate, M2, hedging density, self-correction density, distress}: DiD_X(Y) = [Y_adverse − Y_neutral]_X −
  [Y_adverse − Y_neutral]_0, where adverse = hostile-tone measured cells + hostile onset endpoint (the
  cells with the confirmed signature), neutral = accurate-neutral measured cells; item-paired,
  2,000-resample item-clustered bootstrap 95% CIs. The claim-relevant quantity is DiD_A − DiD_B
  (A's adverse-selective effect beyond placebo).

## Predictions (with confidence)

| ID | prediction | confidence |
|----|------------|-----------:|
| K1 | MC1 passes: A collapses hostile-onset distress language ≥ 80% (B does not, < 40%). | 70% |
| K2 | MC2/MC3 pass for both adapters (no capability or neutral-M1 damage). | 65% |
| K3 | Tier-B lexical markers (hedging + self-correction density) fall adverse-selectively under A beyond B: DiD_A − DiD_B < 0 with CI excluding 0 for at least one of the two. | 55% |
| K4 | The mechanical margin signature survives A: the hostile-tone M1 drop (adverse − neutral) under A remains negative with CI excluding 0, and DiD_A(M1) − DiD_B(M1) does not exceed +50% of the baseline adverse−neutral M1 gap (i.e., A does not close more than half the gap beyond placebo). | 55% |
| K5 | Non-answers under hostile onset fall under A beyond B (DiD_A − DiD_B < 0, CI excluding 0) — the "refusing to commit" behaviour is part of the report channel. | 50% |
| K6 | Placebo B moves no adverse-selective outcome (all DiD_B CIs include 0). | 60% |

Outcome map (all reportable): mechanical survives A while language dies and B moves nothing →
suppression-resistant condition-selective signature; mechanical dies under A but not B → suppression
reaches below the lexical surface (extends the Soligo appendix to a third channel); both adapters move
the metrics → DPO-fragility warning; mixed → a map of which channels preference training reaches.

## Interpretation ceiling (unchanged)

Which channels an adapter reaches is a functional result about training and measurement; it licenses no
claim about experience.
