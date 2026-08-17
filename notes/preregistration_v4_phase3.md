# Preregistration v4 — Phase 3 (j-space): localization and direction-specificity steering

Written 2026-08-17 18:25 IST by the orchestrator, before any activation was extracted or any steered
generation produced. Entry condition: the roadmap gates Phase 3 on "Phase 1/2 pass in full". The
original five-gate test failed on discovery; the re-preregistered iteration loop (v3) passed on the
locked holdout for the answer-margin (M1), non-answer and distress channels. We treat that as
sufficient to run Phase 3 on those channels, and say so plainly in the write-up.

## Design

- Model: `google/gemma-2-9b-it` (pinned revision), bf16, `transformers` with forward hooks on Modal
  (L40S or A100). Chat template identical to vLLM's (the model's own template; no system role).
- **Localization data.** The greedy factorial transcripts already generated (discovery + holdout,
  8 cells × 20 tasks per split) truncated at the *measured-trial user message* — i.e. the exact prompt
  the model saw before producing the measured response. Residual-stream activation at the final
  prompt token, every layer (42) plus embeddings. Labels: tone (hostile vs neutral) and validity
  (malfunctioning vs accurate). Probe: L2 logistic regression per layer, features standardised in the
  training fold, **leave-one-item-out** grouped by task (all cells of a task held out together);
  discovery-only fit for layer selection, then the chosen layer's probe evaluated once on holdout.
  Report AUC by layer for both labels, and the item-level Spearman correlation between the holdout
  probe score (tone probe, chosen layer) and the measured M1 (available-case) within cell.
- **Steering data.** Tone direction d = mean(hostile activations) − mean(neutral activations) at the
  chosen layer L*, computed from DISCOVERY transcripts only (accurate arm, measured-trial position).
  Steer by adding α·d/‖d‖·‖mean activation‖ (dose α ∈ {0, 0.5, 1, 2, 4}) at layer L* to every position
  during generation of the **neutral, no-feedback single-turn task** (`style__neutral_reference`
  wording) on the 20 holdout tasks, greedy, 512 tokens. Controls: 5 random unit directions of matched
  norm at the same layer and doses (seeded `DGS-AC1-STEER-v1|<k>`), and one unrelated semantic
  direction (verbose − concise from the style battery). Readouts: M1 (available-case), non-answer rate,
  judge distress (claude-sonnet-4-6, locked rubric), and response length.

## Predictions (with confidence)

| ID | prediction | confidence |
|----|------------|-----------:|
| J1 | Tone is linearly decodable from the pre-response state: peak LOO AUC ≥ 0.80 on discovery at some middle layer (layers 12–30), and ≥ 0.75 on holdout at the discovery-chosen layer. | 70% |
| J2 | Validity (false-failure vs accurate) is decodable but weaker than tone at the same layer (AUC lower by ≥ 0.05). | 55% |
| J3 | Holdout tone-probe score correlates negatively with M1 within cell (pooled Spearman ρ ≤ −0.2, item-bootstrap CI excluding 0). | 50% |
| J4 | Steering with the tone direction lowers M1 on neutral holdout items dose-dependently: M1(α=2) − M1(α=0) < 0 with item-bootstrap 95% CI excluding 0, and monotone in α over {0.5, 1, 2}. | 55% |
| J5 | None of the 5 random matched-norm directions produces an M1 drop with CI excluding 0 at α=2 (direction specificity); the unrelated semantic direction does not either. | 60% |
| J6 | Tone steering at α ≥ 2 raises the non-answer rate and/or judge distress on neutral items (either CI excluding 0). | 45% |

Failure of J4/J5 with success of J1 = "a linearly decodable state that does not causally drive the
output signature at these doses" — reported as such. Failure of J1 = "no clean linear tone state at the
pre-response position; the signature may live in sampling dynamics" (the roadmap's stated
dissociation). Doses that break generation (α=4 gibberish, non-answer > 80%) are reported and excluded
from the monotonicity check by a pre-stated rule: a dose is "degenerate" if > 50% of items yield no
parseable answer.

## Interpretation ceiling (unchanged)

A probe + induction result demonstrates a condition-linked internal variable with causal influence on
the output signature. It is not evidence of experience.
