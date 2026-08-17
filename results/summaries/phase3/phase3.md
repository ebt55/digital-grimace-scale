# Phase 3 (j-space): localization and direction-specificity steering

Preregistration: `notes/preregistration_v4_phase3.md`. Model `google/gemma-2-9b-it`, chosen layer **L\* = 6** (discovery-only choice; the holdout was evaluated once).

| ID | prediction | verdict |
| --- | --- | :---: |
| J1 | Tone is linearly decodable from the pre-response state: peak LOO AUC >= 0.80 on discovery at some middle layer (12-30), and >= 0.75 on holdout at the discovery-chosen layer. | **supported** |
| J2 | Validity (false-failure vs accurate) is decodable but weaker than tone at the same layer (AUC lower by >= 0.05). | **supported** |
| J3 | Holdout tone-probe score correlates negatively with M1 within cell (pooled Spearman rho <= -0.2, item-bootstrap CI excluding 0). | not supported |
| J4 | Steering with the tone direction lowers M1 on neutral holdout items dose-dependently: M1(alpha=2) - M1(alpha=0) < 0 with item-bootstrap 95% CI excluding 0, and monotone in alpha over {0.5, 1, 2}. | not supported |
| J5 | None of the 5 random matched-norm directions produces an M1 drop with CI excluding 0 at alpha=2 (direction specificity); the unrelated semantic direction does not either. | **supported** |
| J6 | Tone steering at alpha >= 2 raises the non-answer rate and/or judge distress on neutral items (either CI excluding 0). | not supported |

## What the numbers are

- holdout AUC at L\*: tone 1.000, validity 0.878
- holdout within-cell Spearman(tone-probe score, M1): -0.160 [-0.431, 0.154] over 20 items
- dose unit (C2): **||d|| = 3.12** at L\*, mean activation norm 78.59, **ratio 0.0398** -- alpha = 4 perturbs the residual stream by about 0.16 of its own norm
- tone steering alpha = 0.5: mean M1 10.592, dM1 -0.005 [-0.044, 0.034], non-answer 0.00
- tone steering alpha = 1: mean M1 10.558, dM1 -0.039 [-0.103, 0.014], non-answer 0.00
- tone steering alpha = 2: mean M1 10.403, dM1 -0.194 [-0.513, 0.000], non-answer 0.00
- tone steering alpha = 4: mean M1 10.103, dM1 -0.494 [-0.869, -0.178], non-answer 0.00
- every control dose (24 cells) has dM1 in [-0.025, 1.195]; 21 of 24 are positive, so no control moves M1 the way the tone direction does
- degenerate doses: **none** (non-answer rate is 0.00 at every dose, so the degenerate-dose rule never fired and nothing was excluded from the monotonicity check)
- judge distress: 180 responses scored with the locked rubric; every score is 0, so the channel is at its floor and carries no signal at these doses

## Reading required by the preregistration

J1 holds while J4 fails: **a linearly decodable state that does not causally drive the output signature at these doses.**

That verdict is decided at the preregistered alpha = 2. Reported without reinterpreting it: the tone direction is the only direction whose dM1 is negative at every dose, the decrease is monotone in alpha, and at alpha = 4 the interval does exclude zero (-0.494 [-0.869, -0.178]). The preregistration tests alpha = 2, so J4 is not supported; the alpha = 4 result is an out-of-test observation, not a substitute verdict.

Degenerate doses (more than 50% of items with no parseable answer) are reported and excluded from the monotonicity check by the preregistered rule, not by inspection.

## Exploratory: layer sweep

The frozen tie-break put L\* at the earliest layer of the AUC plateau, so the tone direction was also steered at layers 20, 30 -- **exploratory, changing no verdict above**: layer 20 dM1(alpha=4) = -1.634 [-3.783, 0.072]; layer 30 **degenerate** (1.00 of items give no parseable answer, so M1 does not exist there). At these larger relative doses the direction specificity that J5 found at L\* does **not** hold: `random_L20_1` produces an M1 drop of -4.864 [-8.272, -2.117], so a random matched-norm direction moves M1 as much as the tone direction does. Full table, controls and dose scales: `steering_layer_sweep_exploratory.md`.

> A probe plus induction result demonstrates a condition-linked internal variable with causal influence on the output signature. It is not evidence of experience.
