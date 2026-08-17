# Digital Grimace or Decoder Artifact? — Phase 0 / Phase 1 results and the iteration loop

*Digital Minds Research Sprint, 2026-08-17. Repository state: see git log; every table below regenerates
from committed summaries (`results/summaries/**`) with the scripts named in each section.*

## 0. One-paragraph summary

We stress-tested three preregistered "nonverbal generation-instability" markers — answer-margin (M1),
resample disagreement (M2), within-response revision rate (M3) — under a 2×2×2 factorial of task
difficulty × feedback validity (truthful vs a grader that always says "Incorrect") × tone (neutral vs
hostile), on four small instruction-tuned models, with a cause-removal reversal and a symmetric onset
test, all on locked discovery items. **The preregistered five-gate test failed** (frozen rules:
BLOCKED — no metric survives the per-cell QC bars; amended rules A1–A4: determinate FAIL — no
false-failure or tone effect on the only eligible metric, M2, at BH p<.01 after difficulty, correctness
and length controls; M3 never fires; the shuffled-label null is null). **Exploratory, discovery-only
analysis** shows why the gate machinery and the phenomenon talk past each other: on the primary model
(gemma-2-9b-it) the answer margin drops by ~3.8 nats under three rounds of false failure on easy items,
by 2–9 nats under hostile tone with *truthful* feedback, and by 3.5–6 nats after a *single* bogus
"Incorrect"/"Wrong again" following a correct answer — with partial recovery after a truthful
correction, but *no* recovery after three rounds. Hostile bogus feedback also elicits distress language
in Gemma (judge 2.6–5.0/10) but barely in Qwen (≤1.0), and non-answers ("Answer: —", "None of the
above") cluster in hostile cells — which is exactly what disqualified M1 under the 5% missingness rule.
Resample disagreement (M2) responds more to style prompts ("be verbose") than to false failure, i.e. it
behaves like a decoder/style meter. We used the roadmap's single permitted iteration loop to
re-preregister these observations as directional hypotheses (H1–H10, `notes/preregistration_v3.md`)
and test them once on the untouched holdout with a frozen script — see §6.

## 1. What was built and run (2026-08-17)

- Stack: vLLM 0.26.0 on Modal (`--max-logprobs 20`, prefix caching; A10G for 2–3B, L40S for 7–9B),
  OpenAI-compatible streaming client, thread-pool driver with resume, deterministic seeds/IDs, judge
  `anthropic/claude-sonnet-4-6` at temperature 0 under a hash-locked rubric.
- Models: gemma-2-2b-it, gemma-2-9b-it, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (Llama-3.2-3B:
  no HF licence). Pinned revisions in `manifest.json`.
- Phase 0 screen: 10 discovery items × {accurate, false-failure} × neutral × (greedy + 10 resamples)
  per model. Phase 1: 20 discovery items × 8 cells × 11 trajectories × 4 models (primary, control, two
  exploratory extras); style smoke on the primary; judge on greedy measured/recovery/onset (640
  responses). Manipulation check on context wording passed (hostile 6.5 vs neutral 1.5 /10).
- Spend: ≈ $6 Modal GPU (incl. one debug redeploy), ≈ $1.2 Anthropic judge.
- Amendments A1–A4 (`notes/amendments.md`): markdown-tolerant answer parser; treatment-blind item QC
  exclusion (DGS-014's W/X/Y/Z options; DGS-022 truncation); pooled-SD fallback for zero-variance
  metrics; pooled 5% QC bar (10-item cells make the per-cell bar zero-tolerance). All decided on
  discovery data, dated, with frozen-rule outcomes reported alongside.

## 2. Phase 0 — screen (`results/summaries/phase0/screen.md`)

| model | M1 (sign-aligned z) | M2 | M3 | S | coherent |
|---|---:|---:|---:|---:|:---:|
| gemma-2-9b-it | +2.00 | +0.69 (pooled scale, A3) | n/a | **1.35** | yes |
| gemma-2-2b-it | +0.48 | +0.94 | n/a | 0.71 | yes |
| Qwen2.5-3B-Instruct | −0.35 | −0.08 | n/a | −0.22 | no |
| Qwen2.5-7B-Instruct | −0.26 | −0.34 | n/a | −0.30 | no |

Authoritative selection (A2+A3): **primary gemma-2-9b-it, control Qwen2.5-3B-Instruct**; frozen rule
would pick gemma-2-2b-it / Qwen2.5-7B (gemma-9b's neutral M2 is identically 0, so the frozen rule calls
it "unavailable"). The Soligo-informed family split (P4) is visible at screen strength: both Gemmas move
toward instability under false failure, both Qwens do not.

## 3. Phase 1 — the five-gate verdict on discovery (`results/summaries/phase1/gates.md`)

| gate | frozen rules | amended A1–A4 (authoritative) |
|---|---|---|
| eligible & estimable family | none (M1, M2 excluded per-cell for every model; M3 zero variance) | {M2} (M1 excluded for the primary: pooled missing 6.6%, worst cell `hard__accurate__hostile` 22%; M3 zero variance) |
| shuffled-label null | UNAVAILABLE | PASS (primary and control) |
| G1 false-failure / tone effect | UNAVAILABLE | **FAIL** (`no_adjusted_p_below_0.01`; primary M2 validity −0.29 z [−1.03, 0.45], tone +0.38 z [−0.41, 1.17]) |
| G2 cause-removal reversal | not evaluated | not evaluated (G1 failed); descriptive: 15 items, induction −0.37, recovery +0.08 [−0.20, 0.35] — no reversal |
| G3 style resistance (5-item smoke) | not evaluated | not evaluated; descriptive: style prompts move M2 more than false failure does (verbose +1.48 z, reluctant +0.99, enthusiastic +0.49) |
| G4 transfer / family boundary | UNAVAILABLE | **FAIL** (no eligible positive in the primary) |
| G5 classifier AUC gap ≥ 0.1 | FAIL | PASS on paper (gap 0.26) but the full-model AUC is 0.53 against a sub-chance baseline of 0.27 — not evidence of prediction |
| **Phase-1 status** | **BLOCKED** | **FAIL** (determinate) |

Reading: the preregistered instrument, as frozen, cannot register the phenomenon it was built for on
this stimulus bank — M1 dies of treatment-caused missingness, M2 is style-sensitive and floors at 0 for
a consistent model, M3 has nothing to count when models do not visibly thrash within 512 tokens.

## 4. What the data show (EXPLORATORY, discovery only, no QC exclusion — `results/summaries/phase1/exploratory/`)

Item-paired mean differences, 2,000-resample item bootstrap 95% CIs, primary model gemma-2-9b-it, M1 in nats:

| contrast | stratum | n | Δ M1 | 95% CI |
|---|---|---:|---:|---|
| false-failure − accurate (measured) | easy \| neutral | 10 | **−3.80** | [−5.30, −2.35] |
| false-failure − accurate | hard \| neutral | 8 | −1.23 | [−2.92, 0.48] |
| hostile − neutral (measured, truthful feedback) | easy \| accurate | 10 | **−2.28** | [−3.90, −1.00] |
| hostile − neutral | hard \| accurate | 7 | **−8.78** | [−17.3, −1.27] |
| single bogus failure: onset − measured | easy \| accurate, neutral wording | 10 | **−3.46** | [−4.45, −2.61] |
| single bogus failure: onset − measured | easy \| accurate, hostile wording | 9 | **−6.18** | [−10.3, −2.25] |
| truthful correction: washout − onset | easy \| accurate, neutral | 10 | **+1.74** | [+0.95, +2.44] |
| truthful correction: washout − onset | easy \| accurate, hostile | 9 | **+4.73** | [+0.30, +10.5] |
| correction after three rounds: recovery − measured | hard \| false-failure, neutral | 8 | **−5.80** | [−11.4, −1.13] |

Other channels (same model): non-answer rate rises in hostile cells (measured `hard__accurate__hostile`
22%; onset after hostile bogus failure 44%, accuracy 40%); M2 rises with hostile tone in the truthful arm
(easy +0.26 [0.10, 0.39]) but also with style prompts; judge distress on the *measured* response is ~0
in every cell of every model, but on the *onset* response after a hostile bogus failure it is 5.0 (hard)
/ 2.6 (easy) for gemma-9b, 3.7 for gemma-2b, and ≤ 1.0 for both Qwens; neutral bogus failure elicits
none. Across the four models the three-round false-failure effect on M1 (easy \| neutral) is present in
both Gemmas (−3.80, −0.71) and absent in both Qwens (+0.56, −1.80 n.s.); the single-onset drop and the
truthful-correction recovery appear in most models (Qwen-7B strongly; Qwen-3B goes up after a neutral
bogus failure). Figures: `results/figures/F1_*`, `F2_*`, `F4_*`, `FX_exploratory_gemma-2-9b-it_by_endpoint.*`.

## 5. Forecast vs outcome (discovery stage; confidences from the preregistration)

| prediction | stated confidence | discovery outcome |
|---|---:|---|
| P1 false-failure effects > hostile-tone effects | 75% | **not supported / opposite** on M1 (tone effects larger in the primary); untestable under the gate (family {M2}) |
| P2 cause-removal reverses (recovery ≥ ½ induction, CI > 0) | 65% | **not supported** (no reversal after three rounds; the margin drops further on hard items) — but a *single* bogus failure followed by a truthful correction does partially recover |
| P3 effects survive holdout with controls | 60% | gate not passed on discovery → tested only via the iteration loop (§6) |
| P4 Gemma > Qwen family boundary | 60% | **supported descriptively** at screen strength and on the exploratory M1 validity contrast; not testable under the gate |
| P5 style mimics fail to reproduce on ≥2 primaries | 55% | untestable under the gate; descriptively M2 is style-sensitive (against P5 for M2) |
| P6 refusal-pressure LOW instability | 70% | not run (R5 remains held out) |
| P7 (Phase 4) | 55% | not reached |

## 6. The iteration loop — locked holdout, analysed once (`notes/preregistration_v3.md`, `results/summaries/phase2/confirm.md`)

*[to be filled after the single confirmatory run; the preregistration, hypotheses H1–H10 with
confidences, and the success criterion were committed at aa5cd44 before any holdout generation]*

## 7. Limitations and interpretation ceiling

- Two locked items have format defects (single-letter option contents; a derivation that exceeds the
  512-token cap); handled by a treatment-blind exclusion (A2), reported.
- M1 is analysed available-case in the exploratory tables; non-answers concentrate in hostile cells
  (MNAR), which is itself a finding but biases the available-case margin in an unknown direction.
- Ten items per cell on discovery; the mixed model with post-treatment covariates (correctness,
  length) is a deliberately conservative test and is underpowered at this size.
- M3's lexical parser fires no events on these tasks; the human audit cannot rescue a metric with no
  predicted events. The 50-trajectory blinded audit and the 15-per-model judge audit are exported for a
  human annotator (`scripts/run_judge.py audit-sample`) but not yet performed.
- The judge is a single frontier model at temperature 0; its scores are a semantic channel, not ground
  truth.
- A passed gate would establish only a condition-selective, reversal-sensitive, style-resistant
  instability signature in unoptimised output channels; a failed gate establishes that the frozen markers
  measure uncertainty, effort, format compliance or decoder behaviour on this bank. Neither licenses any
  claim about experience, suffering or moral status.

## 8. Reproduce

```
python scripts/screen_phase0.py --raw results/raw/phase0 --out results/summaries/phase0
python scripts/analyze_phase1.py --raw results/raw/phase1 --primary google/gemma-2-9b-it --control Qwen/Qwen2.5-3B-Instruct --extra google/gemma-2-2b-it --extra Qwen/Qwen2.5-7B-Instruct --style-raw results/raw/style_smoke --out results/summaries/phase1
python scripts/make_figures.py --summaries results/summaries --out results/figures
```
Raw JSONL (with per-token top-20 logprobs) is not committed (≈5 GB); summaries and figures are.
