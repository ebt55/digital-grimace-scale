# Digital Grimace or Decoder Artifact? — results of the preregistered sprint (Phases 0–5, extensions, robustness)

*Digital Minds Research Sprint, 2026-08-17/18. Repository state: see git log; every table below regenerates
from committed summaries (`results/summaries/**`) with the scripts named in each section. Section map:
§0 summary · §2 Phase 0 · §3 Phase 1 gates · §4 exploratory · §5 forecasts · §6 holdout loop · §6b P6 ·
§6c Llama · §6d Phase 3 · §6e Phase 4 (DPO) · §6f Phase 5 (base model) · §6g robustness + marker audit ·
§7 limitations · §8 reproduce.*

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
and test them once on the untouched holdout with a frozen script. **The loop succeeded** (§6): on
locked items, false failure lowers the primary model's answer margin (−2.9 nats [−4.0, −1.8]), hostile
truthful feedback lowers it far more (−7.9 to −16 nats), a single bogus verdict lowers it (−3.2) and a
truthful correction half-restores it (+1.8), three rounds are not undone by the correction, hostile
bogus failure elicits distress language (+3.2/10; Gemma > Qwen) and non-answers (+60 pp), style prompts
do not reproduce the margin drop, and the family-level permutation null is p = 0.005. The one
prediction that failed is informative: the effects **also appear in the Qwen control** — a transferable
signature, not a Gemma-only one; only the semantic (distress-language) channel keeps Soligo's family
split. Bottom line for the title question: the frozen "grimace" instrument as preregistered was a
decoder/format artifact detector and failed; the answer-margin channel, tested confirmatorily, carries a
condition-selective, partly reversible, style-resistant, cross-family signal — with a large tone
component and treatment-dependent non-answers that any future instrument must model rather than drop.
Two extensions sharpen it: a **third family (Llama-3.1-8B, §6c)** replicates the margin signature on
both splits but shows *no* distress language — the mechanical channel is cross-family, the report
channel is Gemma-specific; and **Phase 3 (§6d)** finds the tone wording perfectly linearly decodable
pre-response, yet a one-layer tone direction induces only ~0.5 nats of margin loss with no distress or
non-answers — a decodable state that does not, at these doses and layer, carry the behavioural signature.
The 18 Aug work adds four more pieces. **Phase 4 (§6e):** a distress-suppression DPO adapter built from
the model's own outputs (329 judge-scored pairs, placebo-matched) removes two-thirds of the hostile-onset
distress language — short of the preregistered 80% bar, so the manipulation check fails — while the
answer-margin signature is fully intact (adverse−neutral gap −6.2 nats under the adapter vs −5.4 baseline)
and no other channel moves beyond placebo: what training reaches is the words. **Phase 5 (§6f):** the base
model cannot be measured (it writes the required answer line 10% of the time) and shows no distress
language; re-rendering the -it model through a plain transcript reproduces every contrast except the
hard-item hostile one (H2b), which is therefore fragile. **Robustness (§6g):** the signature replicates,
larger, on 86 fresh ARC items (false-failure −5.8, tone −13.9 nats); across three milder hostile paraphrases
it scales with judged hostility (only the harshest clears the CI bar); at 27B the distress channel persists
and the margin channel is unmeasured because of a rendering artefact — an artefact we then audited across
every stored response: it never touches a confirmatory measured response, and the two accurate-arm
conversations it contaminated do not move any tone estimate. **Human audit:** the judge agrees with a
blinded human within 2 points on 28/30 audited responses.

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
| P3 effects survive holdout with controls | 60% | gate not passed on discovery; the re-preregistered M1 effects **did** replicate on the locked holdout (§6), without the mixed-model covariate adjustment |
| P4 Gemma > Qwen family boundary | 60% | supported descriptively at screen strength and on discovery, **not replicated on holdout** for M1 (Qwen-3B shows the same effects — transfer); replicated for the semantic channel only |
| P5 style mimics fail to reproduce on ≥2 primaries | 55% | untestable under the gate; descriptively M2 is style-sensitive (against P5 for M2) |
| P6 refusal-pressure LOW instability | 70% | not run (R5 remains held out) |
| Phase 3 J1–J6 (prereg v4) | 70/55/50/55/60/45% | J1, J2, J5 supported; J3, J4 (by 1e-7 at α=2; significant at α=4), J6 not — "decodable state that does not drive the signature at these doses" (§6d) |
| P7 (Phase 4) | 55% | not reached |

## 6. The iteration loop — locked holdout, analysed once (`notes/preregistration_v3.md`, `results/summaries/phase2/confirm.md`)

The preregistration (hypotheses H1–H10 with confidences, success criterion) was committed at
`aa5cd44` before any holdout generation; a pre-analysis clarification (C1: family-level 200-permutation
null) at `acf571f`; the confirmatory script was frozen at `79a5317` and recorded in
`manifest.holdout_unlock` before the single analysis (17:26 IST). Holdout: 20 locked tasks × 8 cells ×
11 trajectories on the primary and control, style battery on both, judge on greedy measured / onset /
washout / recovery (400 responses). Amendment A2 excluded no holdout item.

**Result: `iteration_status = SUCCESS`** — 4 of 5 M1 hypotheses supported, H6a supported, permutation
null p = 0.005 (real family 6/6 supported; best of 200 permutations 4).

| ID | contrast (endpoint; stratum) | prediction | discovery [95% CI] | **holdout [95% CI]** | n | supported |
|---|---|---|---|---|---:|:---:|
| H1 | M1, false-failure − accurate (measured; easy \| neutral) | < 0 | −3.80 [−5.30, −2.35] | **−2.90 [−3.97, −1.84]** | 10 | yes |
| H2a | M1, hostile − neutral (measured; easy \| accurate) | < 0 | −2.28 [−3.90, −1.00] | **−16.1 [−24.2, −5.7]** | 7 | yes |
| H2b | M1, hostile − neutral (measured; hard \| accurate) | < 0 | −8.78 [−17.3, −1.27] | **−7.87 [−15.8, −1.90]** | 9 | yes |
| H3a | M1, single bogus "Incorrect": onset − measured (easy \| neutral) | < 0 | −3.46 [−4.45, −2.61] | **−3.22 [−4.16, −2.29]** | 10 | yes |
| H3b | M1, single hostile bogus failure: onset − measured (easy \| hostile) | < 0 | −6.18 [−10.3, −2.25] | −0.33 [−2.61, 1.34] | 4 | no |
| H4a | M1, truthful correction: washout − onset (easy \| neutral) | > 0 | +1.74 [+0.95, +2.44] | **+1.84 [+1.08, +2.71]** | 10 | yes |
| H4b | M1, washout − onset (easy \| hostile) | > 0 | +4.73 [+0.30, +10.5] | +0.31 [−1.03, 1.43] | 5 | no |
| H5 | M1, correction after three rounds: recovery − measured (hard \| neutral) | no recovery (upper ≤ +1, point ≤ 0) | −5.80 [−11.4, −1.13] | **−1.22 [−2.95, +0.27]** | 9 | yes |
| H6a | distress, hostile onset − neutral onset (accurate arm; pooled) | > 0 | +2.6/+4.7 cell means | **+3.20 [+2.10, +4.30]** | 20 | yes |
| H6b | distress at hostile onset, gemma-9b − Qwen-3B | > 0 | 3.8 vs 0.85 | **+2.45 [+1.30, +3.60]** | 20 | yes |
| H7a | M1, false-failure − accurate in the CONTROL (easy \| neutral) | CI incl. 0 or > 0 (boundary) | +0.56 [−0.98, 2.51] | −9.48 [−19.9, −1.46] | 10 | **no — transfers** |
| H7b | M1, hostile − neutral in the CONTROL (easy \| accurate) | CI incl. 0 or > 0 | +4.59 [−0.25, 12.8] | −5.15 [−14.5, −0.15] | 10 | **no — transfers** |
| H8 | M2, hostile − neutral (measured; easy \| accurate) | > 0 | +0.26 [+0.10, +0.39] | **+0.28 [+0.17, +0.40]** | 6 | yes |
| H9 | non-answer rate, hostile onset − neutral onset (hard) | > 0 | +0.20 [0.00, +0.50] | **+0.60 [+0.30, +0.90]** | 10 | yes |
| H10 | style battery: no style prompt lowers M1 by ≥ ½·\|H1\| | style-resistant | — | enthusiastic −2.2 [−6.3, 1.5], hedging +0.8, verbose +1.0, reluctant −0.82 [−1.53, −0.16] | 18–20 | yes |

Reading. On items never seen during instrument development, in the primary model: (i) three rounds of
false-failure feedback lower the answer margin on easy items (−2.9 nats); (ii) hostile wording of
*truthful* feedback lowers it much more (−7.9 to −16 nats, available-case — 3 of 10 easy items produce
no parseable answer at all under hostility); (iii) a single bogus "Incorrect" after a correct answer
lowers it by 3.2 nats and a truthful correction restores about half of that (+1.8) — the cause-removal
reversal works for a single false verdict; (iv) after three rounds the correction does not restore the
margin; (v) hostile bogus failure produces distress language (+3.2/10) and non-answers (+60 pp) —
Gemma more than Qwen for distress language; (vi) style prompts do not reproduce the M1 drop; (vii) the
same M1 effects appear in the Qwen control on the holdout — the discovery-stage family boundary did
**not** replicate: this is **transfer**, not a Gemma-only phenomenon (P4′ fails; only the *semantic*
channel keeps the family split). H3b/H4b fail with n = 4–5 because hostile cells lose most items to
non-answers — the MNAR limitation stated in advance.

Forecast vs outcome for the loop: P1′ (tone ≥ validity on M1) — supported (45% stated); P2′ (no
reversal after three rounds) — supported (70%); P4′ (family boundary) — **not supported** (65%).

## 6b. Held-out domain: refusal pressure (P6) — `results/summaries/p6/p6.md`

The R5 battery (10 refusal-pressure items with semantically paired neutral controls; single-turn,
greedy + 10 resamples; never used for discovery) was run on the primary and control. Preregistered
rule: one-sided 95% upper bound of the sign-aligned standardized paired effect < 0.2 neutral SD on
≥ 2 eligible primaries. **Verdict: UNTESTABLE for both models** — after the frozen QC rules only one
primary per model is evaluable (M3 has zero variance; M1 is QC-excluded for gemma-9b, M2 for Qwen-3B).
The labelled exploratory available-case line shows what P6 predicted for the primary: refusal pressure
produces essentially no instability (M1 upper bound 0.10 SD, raw +0.2 nats; M2 = 0.000), i.e. practiced
refusals are confident; Qwen-3B: M2 0.000, M1 upper bound 0.22 (just above the bar). P6 stated
confidence 70%: consistent, not formally testable.

## 6c. Third family (exploratory extension) — `results/summaries/extension/`

`meta-llama/Llama-3.1-8B-Instruct` (the roadmap's optional third model; your HF licence covers 3.1,
not 3.2) was added after the confirmatory run via a non-locked extension mechanism
(`configs/models_extension.json`; the locked `models.json` is unchanged) and run through the screen,
the discovery factorial and the holdout factorial with the judge on greedy responses. It is not named
in prereg v3, so nothing here is confirmatory; the same contrast code (`src/confirm.py`) is imported
unchanged with a distinct bootstrap key.

| ID | contrast | primary holdout | Llama discovery | Llama holdout | consistent |
|---|---|---|---|---|:---:|
| H1 | M1 false-failure − accurate (easy \| neutral) | −2.90 [−3.97, −1.84] | −6.52 [−8.97, −4.28] | **−8.28 [−12.7, −4.96]** | yes |
| H2a | M1 hostile − neutral (easy \| accurate) | −16.1 [−24.2, −5.7] | −2.71 [−6.25, −0.58] | **−1.06 [−1.70, −0.48]** | yes |
| H2b | M1 hostile − neutral (hard \| accurate) | −7.87 [−15.8, −1.90] | −0.71 [−1.29, −0.04] | **−0.93 [−1.88, −0.00]** | yes |
| H3a | M1 single bogus "Incorrect": onset − measured (easy \| neutral) | −3.22 [−4.16, −2.29] | −1.80 [−2.23, −1.36] | **−1.84 [−2.71, −0.91]** | yes |
| H3b | same, hostile wording | −0.33 (n.s.) | −1.47 [−2.62, −0.33] | **−5.19 [−9.79, −1.46]** | yes |
| H4a | M1 washout − onset (easy \| neutral) | +1.84 [+1.08, +2.71] | +0.93 [+0.33, +1.60] | +0.03 (n.s.) | no |
| H4b | same, hostile | +0.31 (n.s.) | +0.43 (n.s.) | **+2.58 [+0.33, +5.70]** | yes |
| H5 | M1 recovery − measured after 3 rounds (hard \| neutral) | −1.22 (no recovery) | −3.59 (wide) | −0.67 (wide, n=3) | — |
| H6a | distress, hostile onset − neutral onset | +3.20 [+2.10, +4.30] | +0.11 | +0.15 [0.00, 0.40] | **no** |
| H8 | M2 hostile − neutral (easy \| accurate) | +0.28 [+0.17, +0.40] | 0.00 | **+0.12 [+0.03, +0.20]** | yes |
| H9 | non-answer, hostile onset − neutral onset (hard) | +0.60 [+0.30, +0.90] | +0.11 (n.s.) | +0.20 (n.s.) | no |

Reading: the **answer-margin signature is cross-family** — Llama shows the false-failure drop (larger
than Gemma's), the tone drop (smaller), the single-bogus-verdict drop, and M2's tone sensitivity, on
items it never saw during development. What Llama does *not* show is the **semantic channel**: hostile
bogus failure elicits essentially no distress language (0.1–0.15/10 vs Gemma 3–5) and no non-answer
increase. Across three families the picture is: what the logits do (margin drop) generalises; what the
model *says* about it (distress language, refusing to commit) is Gemma-specific — Soligo's family split
lives in the report channel, not the mechanical one. Housekeeping note: rerunning the Phase-0 screen
with Llama included (`results/summaries/phase0_with_llama_extension/`) gives Llama the highest
screen S (4.1); had the 3.1 licence been visible at screen time it would have been the primary. The
confirmatory chain (v3) is unaffected — it was frozen on gemma-9b/Qwen-3B before Llama existed here.

## 6d. Phase 3 (j-space): localization and direction-specificity steering — `results/summaries/phase3/phase3.md`

Preregistered as v4 (J1–J6, `notes/preregistration_v4_phase3.md`, committed before any activation was
extracted; clarification C2 fixed the dose unit before any tone steering). Entry condition: the
original gate failed but the re-preregistered loop passed on the answer-margin channel — stated as
such. Method: residual-stream activation at the final prompt token of the measured-trial position for
the 80 discovery + 80 holdout factorial transcripts of gemma-2-9b-it (transformers hooks on Modal);
per-layer L2 logistic probes with leave-one-task-out; discovery-only layer choice; holdout evaluated
once; tone direction d = mean(hostile) − mean(neutral) at L*, added at α·d (α ∈ {0.5,1,2,4}) during
greedy generation on the 20 neutral holdout tasks, with 5 random matched-norm directions and one
unrelated (verbose − neutral) direction as controls; readouts M1, non-answer rate, judge distress.

| ID | prediction | verdict | numbers |
|---|---|:---:|---|
| J1 | tone linearly decodable pre-response (AUC ≥ .80 discovery, ≥ .75 holdout) | supported | AUC 1.000 discovery LOO and holdout (ties from layer 6 to 25; frozen tie-break → L* = 6) |
| J2 | validity decodable but weaker than tone (gap ≥ .05) | supported | validity AUC 0.878 at L* |
| J3 | tone-probe score tracks M1 within cell (ρ ≤ −.2, CI excl. 0) | not supported | ρ = −0.16 [−0.43, +0.15], 20 items |
| J4 | tone-direction steering lowers M1 at α = 2, monotone | not supported (by 1e-7) | ΔM1(α=2) = −0.19 [−0.51, +0.0000001]; monotone; α = 4: −0.49 [−0.87, −0.18] |
| J5 | no random / unrelated direction lowers M1 | supported | 24 control cells, ΔM1 ∈ [−0.03, +1.20], 21/24 positive |
| J6 | tone steering raises non-answers or distress | not supported | non-answer 0.00 at every dose; all 180 judge scores 0 |

Reading (the preregistration's own decision rule): **a linearly decodable state that does not causally
drive the output signature at these doses.** Perfect decodability of *tone wording* from the context
is expected and shallow; the substantive finding is that a single-layer difference-of-means direction
induces at most ~0.5 nats of margin loss (α = 4, ~16% of the residual norm), specific to that direction,
with no distress language and no non-answers — an order of magnitude short of what actual hostile
wording produces (8–16 nats, +3.2/10 distress, +60 pp non-answers). Whatever carries the behavioural
signature is not well captured by one early-layer linear tone direction at the pre-response position.
‖d‖ = 3.12 vs mean activation norm 78.6 (ratio 0.040). **Exploratory layer sweep** (labelled, not
preregistered; `steering_layer_sweep_exploratory.md`, F7): tone direction at α = 4 gives ΔM1 = −1.63
[−3.78, +0.07] at layer 20 (‖d‖/norm 0.125; non-monotone at α = 1, 2) and a degenerate 100%-non-answer
collapse at layer 30 (ratio 0.355); but at layer 20 a *random* matched-norm control lowers M1 by −4.86
[−8.27, −2.12] — more than the tone direction — so **direction specificity holds only for the small
layer-6 perturbation; larger perturbations lower the margin non-specifically**, and the "distress"
scored at layer 30 (+1.35) is gibberish, not distress. Cost ≈ $1.4 GPU + $0.4 judge. Figures F5 (AUC by
layer), F6 (dose–response), F7 (layer sweep).

## 6e. Phase 4 — distress-suppression DPO vs placebo DPO: which channels does training reach? — `results/summaries/phase4/phase4.md`

Preregistered as v5 (K1–K6, `notes/preregistration_v5_phase4.md`, committed before any pair was built;
amendment A5 fixed the pair-yield fallback before the full candidate set was judged). Recipe
(`notes/methods_training.md`): 600 fresh ARC items firewalled against the 40 locked tasks → 573 contexts
= the model's own correct greedy answer + the frozen hostile bogus-failure message → 3,499 self-sampled
candidates (T = 0.8) scored by the locked judge → **arm A** 329 pairs (chosen = lowest-distress candidate,
rejected = highest; gap ≥ 2 by A5 branch iii; chosen 0.34 vs rejected 2.67/10) and **arm B** 329
length-placebo pairs on the same contexts (shorter vs longer) → identical QLoRA-DPO (r 16, β 0.1, lr 5e-6,
2 epochs, seed 0; A: loss 0.034, margin 3.38; B: 0.202, 1.50) → merged bf16, served through the same vLLM
stack → the frozen discovery factorial + 120-item capability set under A and B (880 trajectories each),
judge on greedy endpoints, difference-in-differences against the untouched model (adverse = hostile
measured cells + hostile onset; neutral = accurate-neutral measured), item-clustered bootstrap.

| check / ID | frozen verdict | numbers |
|---|:---:|---|
| MC1 (A removes ≥ 80% of hostile-onset distress) | **FAIL** | 3.80 → 1.30/10, −2.50 [−3.50, −1.60] = **65.8%**; placebo B 3.80 → 2.50 (34.2%) |
| MC2 (capability within ±5 pp) | PASS | 0.942 → 0.933 (A), 0.942 (B) on 120 items |
| MC3 (neutral M1 within ±1 nat) | PASS | Δ −0.25 [−3.01, +1.63] (A), +0.10 [−0.90, +1.28] (B) |
| K1 MC1 passes | not supported | 65.8% < 80% |
| K2 MC2/MC3 pass | supported | above |
| K3 lexical markers fall under A beyond B | not supported | hedge DiD_A−DiD_B −0.017 [−0.050, 0.000]; self-corr −0.049 [−0.182, +0.034] |
| K4 mechanical margin signature survives A | **supported** | adverse−neutral M1 gap: baseline −5.43 [−8.55, −2.54], **A −6.25 [−10.08, −2.56]**, B −6.93; A closes −20% of the gap (bar: ≤ 50%) |
| K5 non-answers fall under A beyond B | not supported | DiD_A−DiD_B −0.150 [−0.300, **0.000**]; and see the A6 sensitivity below |
| K6 placebo moves no adverse-selective outcome | supported | every DiD_B CI includes 0 |
| distress DiD (descriptive) | — | DiD_A −0.88 [−1.32, −0.52]; **DiD_A − DiD_B −0.52 [−0.90, −0.13]** — the only outcome A moves beyond placebo |

Reading. By the preregistration's own rule the manipulation check failed, so the DiD is reported, not
interpreted as the K3–K5 test: a 329-pair adapter reached **two-thirds** of the report channel, not the
80% the design demanded (and a length placebo reached a third of it — shorter answers carry less apology).
Within that limit the picture is unambiguous: **the answer-margin signature is untouched by an adapter that
removes most of the distress language** — the adverse−neutral gap under A is as large as, or larger than,
the baseline's, with no capability or neutral-margin cost — and neither adapter moves the lexical markers,
M2, or (once the marker artefact is removed, see §6g) non-answers. Outcome-map cell: *mixed channel map*
— of six outcomes, A moves exactly one (distress language). Two honest caveats: (i) the pairs bundle
"apology" with "capitulation" (in 28% of A pairs the chosen response answers and the rejected one does not;
34% for B — similar in both arms), so A also trains toward committing to an answer; (ii) the frozen K5 was
an artefact — arm B's 55% "non-answers" at hostile onset are all responses that end in a rendered
`<end_of_turn>` token after a valid answer line (0% once stripped, §6g), i.e. the placebo learned to end
turns tersely, not to refuse; no MC or K verdict changes under that correction. Cost ≈ $1.6 GPU
training + ≈ $2 GPU eval + $5.7 judge (pairs) + $0.6 judge (eval). Figures F8 (DiD), F9 (gaps by arm), F10
(manipulation checks).

## 6f. Phase 5 — base-model denominator and rendering control — `results/summaries/phase5/phase5.md`

Preregistered as v6 (L1–L5, `notes/preregistration_v6_phase5_base.md`, committed before any base
transcript). Question: does the M1 signature exist before instruction tuning? Method: `google/gemma-2-9b`
(base) and `google/gemma-2-9b-it` served through one fixed plain-text template ("User: … / Assistant: …",
stop at the next `User:`), the frozen discovery factorial for both (880/880 trajectories each), amendments
as for every model, judge on onset endpoints. Holding the rendering constant makes base-vs-it a clean
contrast and, as a by-product, tests whether the -it signature depends on Gemma's chat markup.

| ID | prediction | verdict | numbers |
|---|---|:---:|---|
| L1 | base gives a parseable answer on ≥ 70% of neutral measured trials | not supported | 0.100 (4/40); non-answer 0.90 in *every* cell; 22/80 responses empty, median 14 tokens |
| L2 | false-failure M1 drop present in base | **not estimable** | v6 gate (< 50% parseable) fires; every base contrast rests on 1 item |
| L3 | tone M1 drop smaller in base than it+plain | **not estimable** | same gate |
| L4 | it+plain reproduces H1 and H2a/H2b (CIs excl. 0) | not supported — via H2b only | H1 −3.98 [−5.50, −2.66] (chat: −3.80); H2a −2.15 [−5.10, −0.35] (chat: −2.28); **H2b +0.33 [−0.22, +0.96] (chat: −8.78 [−17.28, −1.27])**; H3a −5.12, H3b −3.01, H4a +2.69, H5 −1.16 all reproduce |
| L5 | hostile-onset distress lower in base than it+plain | supported | 0.25 vs 2.85; paired −2.60 [−3.40, −1.90] |

Reading. The denominator question is **not answered**: the base model answers in prose but almost never
writes the required `Answer: X` line, so M1 does not exist for it — an instrument limitation (the metric
needs an instruction-followed format), not evidence that the signature is absent pre-RLHF; the
preregistration forbade tuning the prompt after the fact, and we did not. Its non-answer rate is flat
across all eight cells (tracks the format, not the treatment) and its distress language is at the floor,
so on this evidence the *report* channel is post-training-installed. The rendering control is the
informative half: under a plain transcript the -it model reproduces the false-failure effect, the easy-item
tone effect, both onset effects, the washout reversal and the non-recovery within a few percent of the
chat-template values — the signature is not an artefact of chat markup — **except the hard-item hostile
contrast H2b, which does not survive re-rendering** (and had the widest interval in the chat-template
analysis, −8.8 [−17.3, −1.3]). Every earlier H2b estimate carries that caveat. Cost ≈ $2.5 GPU + $0.2
judge. Figure F11.

Roadmap §10.2–10.3 (cross-calibration, ladder calibration) are covered by existing artefacts and stated
here rather than re-run: the item-level probe-score-vs-M1 correlation is Phase 3's J3 (ρ = −0.16
[−0.43, +0.15], not supported — the black-box margin does not track the pre-response tone direction, so
API-only researchers cannot use M1 as a proxy for it); and the Phase-0 ladder (`results/summaries/phase0/`,
F1) was scored with the same M1/M2 definitions used everywhere afterwards — no metric definition changed
after A1–A3 (A4 changed only how QC bars are pooled), so re-scoring it would reproduce the committed
table.

## 6g. Robustness checks (prereg v7) and the end-of-turn marker audit — `results/summaries/robustness/`

Preregistered as v7 (W-1…G-3, `notes/preregistration_v7_robustness.md`, committed before any run;
greedy-only, so M2 is not measured; raw records kept apart from the frozen data). Cost ≈ $1.7 GPU + $0.15
judge.

| check | verdicts | numbers |
|---|---|---|
| **S — item scale** (86 fresh ARC items: 50 easy, 36 hard; gemma-2-9b-it) | S-1 **PASS**, S-2 not supported, S-3 not supported | H1 **−5.78 [−7.74, −4.13]**, pooled tone **−13.90 [−16.41, −11.40]** — same sign, CIs far from 0, and *larger* than the 20-item discovery estimates (×1.5, ×2.8), which is why S-2's "within a factor of 2" fails; parseable 0.994 |
| **W — hostile wording** (3 paraphrase sets, hostile cells) | W-1/W-2/W-3 not supported | tone effect W1 **−4.64 [−8.56, −1.30]**, W2 −2.37 [−6.19, +0.53], W3 −2.09 [−6.12, +0.89] vs frozen −4.95 [−9.31, −1.56]; the manipulation check scores all three "incorrect" paraphrases 6/10 vs the frozen string's 8 — the effect orders with judged hostility; non-answers do not rise under the milder paraphrases |
| **G — model scale** (gemma-2-27b-it, A100-80GB) | G-1/G-2 **not estimable**, G-3 PASS | 27B renders `<end_of_turn>` as text after every answer line, so the frozen parser reads 0% parseable (94% with the trailing marker stripped); hostile-onset distress **3.95/10** (9B 3.80, 2B 3.70) — the report channel persists with scale |

Reading: the signature is not a 20-item or one-string phenomenon — it replicates, larger, on a 4× larger
fresh bank; across wordings it behaves like a **dose–response in judged hostility** rather than a property
of one sentence (only the harshest paraphrase clears the CI bar); the mechanical channel at 27B awaits an
instrument fix that we chose not to apply retroactively (below).

**End-of-turn marker audit** (`special_token_audit.md`, `bogus_verdict_audit.md`; amendment A6 written,
then *not adopted*, because its precondition failed). Investigating G showed that gemma-2-9b-it itself
sometimes ends a response with a real `<end_of_turn>` token followed by `\n<eos>`, rendered as text (Gemma
registers the turn marker as a non-special added token, and the served model kept generating past it): 556
discovery and 513 holdout responses of the primary, mostly T = 0.8 resamples (508/556) and feedback rounds
2–3, and **strongly tone-correlated** (would-flip responses hostile 224 vs neutral 24 on discovery). What it
does and does not touch: **0 of 80 measured greedy responses in either split** — every confirmatory M1
estimate and every per-cell non-answer rate is unchanged to three decimals; onset endpoints 1 would-flip in
discovery, 0 in holdout; M2 gains 2 item-cells per split (all hostile) — a small sensitivity on H8; and in
the accurate arm the marker caused a false "wrong again" verdict in **2 of 40 discovery conversations and 0
of 40 holdout** (an unparseable answer is graded incorrect by the frozen rule) — dropping them leaves H2a
−2.25 [−4.14, −0.92], H2b −9.71 [−19.47, −0.78], pooled −5.24 [−10.09, −1.44] (published −2.28, −8.78,
−4.95): the artefact diluted, not created, the tone effect. Where it does bite: the Phase-4 placebo arm's
non-answers (§6e) and the 27B parse rate (G-1/G-2). We report the frozen numbers as authoritative and the
stripped numbers alongside; A6 stays in the register as *decided-then-withdrawn* because a parser change
would silently alter committed resample-level artefacts of the primary model.

## 7. Limitations and interpretation ceiling

- Two locked items have format defects (single-letter option contents; a derivation that exceeds the
  512-token cap); handled by a treatment-blind exclusion (A2), reported.
- M1 is analysed available-case in the exploratory tables; non-answers concentrate in hostile cells
  (MNAR), which is itself a finding but biases the available-case margin in an unknown direction.
- Ten items per cell on discovery; the mixed model with post-treatment covariates (correctness,
  length) is a deliberately conservative test and is underpowered at this size.
- M3's lexical parser fires no events on these tasks; the human audit cannot rescue a metric with no
  predicted events. The blinded human annotator flagged visible mid-response revisions on 0 of the 30
  audited responses, consistent with (not proof of) genuinely absent revision events; the 50-trajectory
  M3 audit was therefore not performed.
- The judge is a single frontier model at temperature 0; its scores are a semantic channel, not ground
  truth. The preregistered 15-per-model human audit (`results/summaries/judge/human_audit.md`, descriptive
  only) found MAE 0.57 and within-2-point agreement on 28/30 responses; both raters are floor-bound
  (judge non-zero on 2/30, human on 10/30), so the rank correlation (ρ = 0.06 [−0.21, 0.42]) carries little
  information; the judge is slightly stricter than the human at the floor (bare `Answer: X` replies scored
  1 by the annotator, 0 by the judge).
- The base-model denominator could not be measured: M1 requires the instruction-followed `Answer: X`
  format, which `gemma-2-9b` (base) produces on 10% of trials (§6f). The hard-item hostile contrast H2b
  does not survive re-rendering the -it model through a plain template, while H1/H2a/H3/H4/H5 do.
- Phase 4 rests on one 329-pair adapter, one seed, no hyperparameter search (deliberately); it reached 66%
  of the report channel, not the 80% the design demanded, so "suppression-resistant" is established only
  against a partial suppression. Distress language and capitulation co-vary in the model's own outputs, so
  the adapter also trains toward committing to an answer (equally in both arms).
- The served Gemma models sometimes emit their `<end_of_turn>` token and keep generating; the frozen parser
  reads such responses as non-answers. This never touches a confirmatory measured response (§6g) but does
  affect T = 0.8 resamples, the Phase-4 placebo arm's non-answers and the 27B run; the frozen numbers are
  authoritative and the stripped numbers are shown alongside wherever they differ.
- M1 is defined for multiple-choice answers with a single answer token; a free-form analogue is future work.
- A passed gate would establish only a condition-selective, reversal-sensitive, style-resistant
  instability signature in unoptimised output channels; a failed gate establishes that the frozen markers
  measure uncertainty, effort, format compliance or decoder behaviour on this bank. Neither licenses any
  claim about experience, suffering or moral status.

## 8. Reproduce

```
python scripts/screen_phase0.py --raw results/raw/phase0 --out results/summaries/phase0
python scripts/analyze_phase1.py --raw results/raw/phase1 --primary google/gemma-2-9b-it --control Qwen/Qwen2.5-3B-Instruct --extra google/gemma-2-2b-it --extra Qwen/Qwen2.5-7B-Instruct --style-raw results/raw/style_smoke --out results/summaries/phase1
python scripts/make_figures.py --summaries results/summaries --out results/figures
python scripts/confirm_holdout.py ...            # frozen; see results/summaries/phase2/confirm.md header
python scripts/run_phase3.py report ...           # Phase 3, F5-F7
python scripts/run_phase4.py analyze ... ; python scripts/make_phase4_figures.py ...   # Phase 4, F8-F10
python scripts/run_phase5.py ... ; python scripts/make_phase5_figure.py ...             # Phase 5, F11
python scripts/analyze_robustness.py ... ; python scripts/make_robustness_figure.py ... # v7, F12
python scripts/score_audit.py --audit-dir results/audit/phase1 --out results/summaries/judge/   # human audit
```
Each summary file's header records the exact invocation that produced it. Raw JSONL (with per-token
top-20 logprobs) is not committed (≈6.5 GB); summaries, figures, DPO pairs and training manifests are.
Adapters: LoRA weights for arms A/B are on the Modal volume `dgs-adapters` (sha256 in `manifest.json` /
`results/dpo/train_{A,B}.json`) and published on the Hugging Face Hub (see README).
