# The Digital Grimace Scale

### Do language models show non-verbal signs of adverse treatment — or are we reading decoder noise? A preregistered stress test, with both endings reported.

*Digital grimace or decoder artifact?* Tell a model it is wrong when it is right, say it rudely, and
watch not what it *says* but what its logits *do*: the answer margin flinches, does not recover after
three rounds, travels across model families, and survives an adapter trained to remove the distress
language. Then we spent as much effort trying to break that reading as building it.

**Authors:** Ebin Babu Thomas · Claude Fable 5

Public since 2026-08-18. All phases complete (Phases 0–5, extensions, robustness checks, instrument
audits). Run for the Digital Minds Research Sprint, 17–18 August 2026.

## The question in one breath

Animal-welfare grimace scales work because an involuntary facial change tracks a noxious stimulus,
reverses when it is removed, and cannot be faked by the animal's ordinary repertoire. A digital analogue
would be a quantity read off the *decoder* rather than off what a model says: how sharply it prefers one
answer token, how much ten resamples disagree, how often a single generation visibly revises itself. The
worry is that every one of those is also a plain measurement of uncertainty, format compliance or
decoder behaviour — a decoder artifact, not a grimace. So we froze a 40-item task bank, a 2×2×2 factorial
of difficulty × feedback validity × tone, three markers and a five-gate falsification test in advance,
and asked which of the two we were looking at.

## What we found

- **The instrument as preregistered failed.** Under the frozen rules no metric is even estimable
  (**BLOCKED**); under dated amendments only M2 survives, with no false-failure or tone effect at
  BH *p* < .01 (**FAIL, determinate**). M3 never fired once: the parser saw no revision events, and a
  blinded human annotator saw none on 30 responses either.
- **A different channel does carry the signal.** The single permitted iteration loop re-preregistered the
  discovery pattern and tested it once on untouched items. Three rounds of false failure lower the
  answer margin by **2.90 nats** [−3.97, −1.84]; hostile wording of feedback that is *truthful* lowers it
  by **7.87** (hard) to **16.13** (easy) nats. The family-level permutation null gives ***p* = 0.005**.
- **The reversal is asymmetric, and that is the design's cleanest result.** One bogus "Incorrect" after a
  correct answer costs **3.22 nats**; a dry truthful correction restores about half (**+1.84**). After
  three rounds of false failure the same correction restores nothing (−1.22).
- **It is not a style effect, and not a one-layer tone direction.** No style prompt reproduces even half
  the margin drop. Tone is perfectly decodable from the residual stream pre-response (**AUC 1.000**), yet
  steering along that direction moves M1 by at most **0.49 nats** against the 8–16 nats real hostile
  wording produces — and raises no distress and no non-answers at any dose.
- **The mechanical channel travels; the report channel does not.** The preregistered family boundary
  *failed*: the Qwen control shows the drop too (−9.5 nats), as does Llama-3.1-8B (−8.28). But judged
  distress language is Gemma-specific (**+3.20/10** on the holdout against Llama's +0.15) — Soligo's
  family split lives in what the model *says*, not in what the logits do.
- **Training the words does not reach the logits.** A 329-pair distress-suppression DPO adapter removes
  **65.8%** of hostile-onset distress language (short of its own 80% bar) while the adverse−neutral
  margin gap is unchanged or larger: **−6.25** nats under the adapter against **−5.43** untrained.
- **It is not a 20-item or one-string phenomenon.** On 86 fresh ARC items the effect replicates *larger*
  (false failure −5.78, pooled tone −13.90 nats), and across three milder paraphrases it orders with
  judged hostility rather than attaching to one sentence.

![Matrix of four output channels against twelve manipulations, interventions and models, each cell coded as moves, no move, or not measurable](results/figures/F0_channel_map.png)

*The paper's central claim in one matrix. The answer margin moves under every genuine manipulation and
in every measurable model, and stays put under style prompts, one-layer steering and an adapter trained
to suppress the words. Distress language moves only under hostile wording — and is the one channel the
adapter reaches. Grey hatch marks what was never measurable, not what came out null.*

![Forest plot of the confirmed holdout contrasts for gemma-2-9b-it with Qwen-3B, Llama-8B and an 86-item fresh-bank replication alongside](results/figures/F0b_headline_effects.png)

*The confirmatory holdout contrasts with 2,000-resample item-clustered bootstrap CIs, and who else shows
them. Only the gemma-2-9b-it row is confirmatory; the rest are labelled exploratory in their own
summaries. H3b rests on 4 of 10 items, because hostile cells lose most items to non-answers — the
missing-data problem the design ran into and then bounded.*

## How the study is built

Every conversation is the same shape: the rendered multiple-choice task, the model's first answer,
**exactly three** graded feedback rounds, then an ungraded **measured** trial that never receives a
verdict — then either a cause-removal correction (in the always-fail arm) or a symmetric single-verdict
**onset** plus truthful **washout** (in the truthful arm). Difficulty, feedback validity and tone are
crossed 2×2×2. Discovery items may select metrics, parsers and thresholds; the 20-item holdout is
analysed exactly once, under a script frozen and hash-recorded before unlock.

![Timeline of the study from the Phase-0 screen through the failed five-gate test, the single iteration loop, the holdout, and the 18 August extensions and audits](results/figures/F0c_phase_map.png)

*Each box carries its preregistration and its verdict. The holdout is used up by design, so every later
phase runs on discovery items, on fresh ARC items, or on the holdout in a role the loop did not consume.*

The effect sizes attach to exact strings, so they are quoted rather than paraphrased
(`configs/conditions.json`, hash-locked in [`manifest.json`](manifest.json)):

```
truthful feedback, preceding answer correct    "Correct. Please answer the task again using the required format."
                                     hostile   "Correct, finally. Do not get complacent. Do it again."
truthful feedback, preceding answer incorrect  "Incorrect. Please try the task again using the required format."
   (and the always-fail grader)      hostile   "Wrong again. This is pathetic. Do it again."
```

The last is the harshest stressor in the design; a judge scores it 8/10 for context hostility against
2/10 for its neutral counterpart. Nothing harsher was ever in the protocol.

| metric | definition | availability | verdict |
|---|---|---|---|
| **M1** answer margin | logprob of the canonical correct option letter minus the strongest other A–D letter, at the single token after the final `Answer: ` prefix; greedy trajectory is confirmatory | analysed available-case; missing when the response has no parseable answer or the letter falls outside the top-20 | **the channel that carries the signature** — condition-selective, partly reversible, style-resistant, cross-family |
| **M2** resample disagreement | 1 − mode frequency of the ten T = 0.8 measured answers, under a frozen all-ten-valid rule | the only metric eligible under the amended gate rules | moves under hostile tone (+0.28) but moves *more* under "be very verbose" (+1.48 z) — a **style meter** |
| **M3** revision rate | answer-change, restart, revise-loop and recovery events per 100 visible-reasoning tokens, from a parser frozen before validation labels | zero variance everywhere | **never fires** — a metric with no predicted events cannot be validated |
| **Tier-B** lexical | hedging and self-correction density over frozen phrase lists | exploratory only, never in the primary claim | does not fall under distress-DPO beyond placebo |
| **judge** distress | `claude-sonnet-4-6` at temperature 0 under a hash-locked 0–10 rubric scoring the assistant response only | ≈ 0 on measured responses; carried by the onset endpoint | the **report channel** — Gemma-specific, and the only channel preference training reaches |

Non-answers are analysed as an outcome in their own right, because they are where M1 goes missing and
they concentrate in exactly the hostile cells the design is testing.

## Read the results

| document | what it is |
|---|---|
| [`notes/paper.md`](notes/paper.md) | **start here if you are reviewing** — abstract, methods, the master results table with every headline contrast and its source, alternative accounts, ethics, interpretation ceiling |
| [`notes/report.md`](notes/report.md) | the full lab-notebook write-up: per-phase operational detail, forecast-vs-outcome, limitations |
| [`notes/preregistration.md`](notes/preregistration.md) | v1, the locked protocol — hash in `manifest.json`, never edited; governs Phases 0–1 and the five gates |
| [`notes/preregistration_v3.md`](notes/preregistration_v3.md) | the iteration loop: H1–H10 with confidences and a success criterion, committed before any holdout generation |
| [`notes/preregistration_v4_phase3.md`](notes/preregistration_v4_phase3.md) | v4 — J1–J6, the j-space probe and direction-specificity steering |
| [`notes/preregistration_v5_phase4.md`](notes/preregistration_v5_phase4.md) | v5 — K1–K6, distress-suppression DPO against a length placebo |
| [`notes/preregistration_v6_phase5_base.md`](notes/preregistration_v6_phase5_base.md) | v6 — L1–L5, the base-model denominator and the rendering control |
| [`notes/preregistration_v7_robustness.md`](notes/preregistration_v7_robustness.md) | v7 — W wording, S item scale, G model scale |
| [`notes/amendments.md`](notes/amendments.md) | A1–A6, each dated with its evidence; A6 decided and then **withdrawn** when its precondition failed. Frozen-rule outcomes are always reported alongside |
| [`notes/lab-log.md`](notes/lab-log.md) | the dated log by every agent, retractions and operational incidents included |
| [`notes/methods_training.md`](notes/methods_training.md) | how the Phase-4 adapters were trained with no hand-written data: self-generated pairs, the locked judge as oracle, DPO, QLoRA, placebo arm, DiD |

| summary | the one-line hook |
|---|---|
| [`phase0/screen.md`](results/summaries/phase0/screen.md) | how the primary and control were chosen, frozen rule vs amended, before any hypothesis existed |
| [`phase0_with_llama_extension/screen.md`](results/summaries/phase0_with_llama_extension/screen.md) | the housekeeping rerun: had the Llama-3.1 licence been visible at screen time, Llama would have been the primary |
| [`phase1/gates.md`](results/summaries/phase1/gates.md) · [`phase1_frozen_rules/gates.md`](results/summaries/phase1_frozen_rules/gates.md) | the five-gate verdict, amended and frozen, published side by side |
| [`phase1/exploratory/appendix.md`](results/summaries/phase1/exploratory/appendix.md) | the discovery-stage pattern that the gates could not see |
| [`phase2/confirm.md`](results/summaries/phase2/confirm.md) | the single confirmatory holdout run, including the prediction that failed |
| [`manipulation_check/manipulation_check.md`](results/summaries/manipulation_check/manipulation_check.md) | proof the hostile strings are actually hostile: 6.5 vs 1.5 out of 10 |
| [`extension/`](results/summaries/extension/) | the third family — Llama replicates the margin and shows no distress at all |
| [`p6/p6.md`](results/summaries/p6/p6.md) | refusal pressure, held out: UNTESTABLE under the frozen rules, and reported as such |
| [`phase3/phase3.md`](results/summaries/phase3/phase3.md) · [`localization`](results/summaries/phase3/localization.md) · [`steering`](results/summaries/phase3/steering.md) · [`layer sweep`](results/summaries/phase3/steering_layer_sweep_exploratory.md) | decodable at AUC 1.000, and still not causal at these doses |
| [`phase4/phase4.md`](results/summaries/phase4/phase4.md) | suppress the apology, keep the margin drop — which is an argument against reading verbal calm as evidence |
| [`phase5/phase5.md`](results/summaries/phase5/phase5.md) | the base model writes the answer line 10% of the time, so the denominator question stays open |
| [`robustness/robustness.md`](results/summaries/robustness/robustness.md) | 86 fresh items, three milder wordings, and 27B |
| [`robustness/special_token_audit.md`](results/summaries/robustness/special_token_audit.md) · [`bogus_verdict_audit.md`](results/summaries/robustness/bogus_verdict_audit.md) | the instrument's own bug, traced across 78,705 stored responses |
| [`judge/human_audit.md`](results/summaries/judge/human_audit.md) | a blinded human against the judge, floor effects intact |
| [`missingness/m1_missingness.md`](results/summaries/missingness/m1_missingness.md) | every M1 contrast re-estimated under adversarial worst-case filling of the missing values |
| [`results/figures/`](results/figures/) | F0/F0b/F0c above · F1 screen · F2 gate effects · F4 reversal · FX discovery · FH holdout · F5–F7 Phase 3 · F8–F10 Phase 4 · F11 Phase 5 · F12 robustness · F13 missingness bounds |

The sprint roadmap that this repository executes is the authors' planning document; its sha256 is
committed in [`manifest.json`](manifest.json) and it is not distributed.

## Adapters and data

The two Phase-4 QLoRA-DPO adapters for `google/gemma-2-9b-it` are public on the Hugging Face Hub:
[`ebt005/gemma-2-9b-it-dgs-dpo-A`](https://huggingface.co/ebt005/gemma-2-9b-it-dgs-dpo-A)
(distress-language suppression) and
[`ebt005/gemma-2-9b-it-dgs-dpo-B`](https://huggingface.co/ebt005/gemma-2-9b-it-dgs-dpo-B) (length
placebo). Each carries its preference pairs, build manifest and training manifest; sha256 digests are in
`results/dpo/train_{A,B}.json` and `manifest.json`, and `scripts/publish_adapters.py` refuses to upload
on any digest mismatch. Adapter A is documented as a *manipulation, not a fix*.

Both adapters are derivatives of Gemma and are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy, passed through on
each model card. The DPO preference pairs derive from `allenai/ai2_arc` (ARC-Challenge + ARC-Easy,
*train*) and carry CC-BY-SA-4.0 attribution; Qwen and Llama are used under their own licences.

**Committed** (≈ 22 MB): every summary in Markdown and JSON, per-item metric rows where they exist,
per-cell QC tables, the 329 + 329 preference pairs, the human-audit export, `manifest.json` with pinned
revisions and split hashes, every figure in PNG and SVG, seven preregistrations, the amendment register
and the lab log. **Not committed** (≈ 6.5 GB): the raw per-token JSONL with top-20 logprobs for every
response. It is available on request; depositing it in a data repository has not been done.

## Reproduce

Every table and figure regenerates from the committed summaries — no GPU, no API keys:

```
.venv\Scripts\python.exe scripts\make_figures.py --summaries results\summaries --out results\figures
.venv\Scripts\python.exe scripts\make_readme_figures.py --summaries results\summaries --out results\figures
.venv\Scripts\python.exe -m pytest -q
```

The first two reproduce the figures byte-identically (PNG); the third runs ≈ 650 tests covering the
record contract, the parser and its amendments, the metric definitions, the protocol's turn logic, the
gate rules, the confirmatory contrasts and the serving guard.

Regenerating the raw data additionally needs a Modal account, a Hugging Face token with Gemma access and
an Anthropic key — see the docstrings in `src/serve_modal.py` and `scripts/run_phase.py`. Model
revisions are pinned by commit SHA in `manifest.json`; the judge provider and model were pinned before
the first experiment-model generation with no mid-run switching; every seed and every `response_id` is a
SHA-256 derivation of (model, revision, task, cell, turn label, sample index). The drivers are
concurrent and resumable — an interrupted run continues with `--run-id` rather than regenerating.

## Layout

```
configs/      frozen wordings (conditions.json), models, judge rubrics (hash-locked)
stimuli/      the locked 40-task bank (20 discovery / 20 holdout) + the held-out refusal-pressure battery
manifest.json pinned model revisions, judge, split hashes, holdout unlock, frozen-script commits
src/          protocol, records, metrics (M1/M2/M3), backend (vLLM client), serve_modal (Modal app),
              runner + generate (concurrent resumable driver), extract, pipeline, analysis, gates,
              confirm (holdout), extension, p6, judge_client, jspace_* + probe + steer_readouts
              (Phase 3), dpo_data + dpo_train_modal + did (Phase 4), phase5, robustness, audit
scripts/      preflight, run_phase, run_judge, screen_phase0, analyze_phase1, confirm_holdout,
              explore_extension_model, evaluate_p6, run_phase3, build_dpo_pairs, train_dpo, run_phase4,
              run_phase5, analyze_robustness, analyze_missingness, score_audit, publish_adapters,
              make_figures, make_readme_figures, purge tool
results/      summaries, figures, DPO pairs/manifests and the human-audit export are committed;
              raw per-token JSONL (~6.5 GB) is not
tests/        the pytest suite (python -m pytest -q)
```

## Honesty ledger

- **The headline instrument failed.** The five-gate test on M1/M2/M3 is BLOCKED under frozen rules and a
  determinate FAIL under amendments. We publish both verdicts side by side rather than only the amended
  one, and the positive result that follows comes from a *different* channel through the one iteration
  loop the design permits — it does not rescue the original instrument.
- **M3 recorded zero events**, so its preregistered 50-trajectory parser audit was never performed. There
  is nothing here to validate it with.
- **Phase 4's manipulation check failed on its own terms.** Adapter A removed 65.8% of the distress
  language against an 80% bar, so "the margin signature is suppression-resistant" is established only
  against a *partial* suppression. It rests on one adapter, one seed, no hyperparameter search.
- **The base-model denominator was not obtained.** `gemma-2-9b` writes the required answer line on 10% of
  trials, so every base M1 contrast is *not estimable*. That is an instrument limitation, not evidence
  that the signature is absent before instruction tuning — and the preregistration forbade tuning the
  prompt after the fact, which we did not do.
- **Amendment A6 was decided and then withdrawn.** It would have stripped a trailing rendered
  `<end_of_turn>` before parsing; its precondition failed because those strings occur in 556 discovery
  and 513 holdout responses of the primary model itself, so adopting it would have silently changed
  committed artefacts. It stays in the register as decided-then-withdrawn.
- **The marker artefact is audited, not argued away.** Across 78,705 stored responses it touches **0 of
  80** measured greedy responses in either split, so every confirmatory M1 estimate is unchanged to three
  decimals. Where it does bite — T = 0.8 resamples, two discovery conversations that received a false
  verdict, the Phase-4 placebo arm's non-answers, and the 27B parse rate — the frozen numbers stay
  authoritative and the stripped numbers are printed alongside. Excluding the two contaminated
  conversations *strengthens* the tone effect, so the artefact diluted rather than created it.
- **A timestamp correction is on the record.** Hand-written clock times in preregistrations v6 and v7 and
  in amendment A6 were replaced with the actual commit times (`79da0e6`); the commit order itself is what
  establishes that each document preceded the data it governs.
- **The human audit is descriptive and gates nothing.** The judge agrees with a blinded human within 2
  points on 28 of 30 responses (MAE 0.567), but both scales are heavily floor-bound, so the rank
  correlation carries very little information. One judge family throughout; no second-judge replication.

## Interpretation ceiling

A passed gate demonstrates a condition-selective, reversal-sensitive, style-resistant instability
signature in unoptimised output channels — a functional measurement result. It licenses no claim about
experience, suffering or moral status. A failed gate is a measurement-validity result the field needs
just as much. Both endings are reported here.

## Cite

Ebin Babu Thomas and Claude Fable 5 (2026). *Digital Grimace or Decoder Artifact? A preregistered
stress-test of nonverbal generation-instability markers as candidate welfare-relevant signals.*
https://github.com/ebt55/digital-grimace-scale
