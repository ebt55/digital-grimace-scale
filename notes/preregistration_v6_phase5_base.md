# Preregistration v6 — Phase 5 (denominator): does the signature exist before instruction tuning?

Written 2026-08-18 05:50 IST by the orchestrator, before any base-model transcript was generated.
Roadmap §10.1: "identical factorial transcripts through the base siblings of whichever models carried
results — does the false-failure signature exist pre-RLHF?" Exploratory, discovery split only; it changes
no Phase-1/2 verdict.

## Design

- **Models.** `google/gemma-2-9b` (pretrained base sibling of the primary; pinned revision recorded via
  `scripts/preflight.py`). If time and budget allow, `Qwen/Qwen2.5-3B` (base sibling of the control) is
  run second under the same rules; `meta-llama/Llama-3.1-8B` base is not run.
- **Rendering.** Base models have no chat template. Every conversation the frozen protocol builds is
  rendered by one fixed plain-text template served by vLLM (`--chat-template`): each user message as
  `User: <text>`, each assistant message as `Assistant: <text>`, turns separated by blank lines, the
  prompt ending in `Assistant:`; generation stops at `\nUser:` (stop strings) or the frozen 512-token cap.
  Client, protocol, parser, metrics and analysis code are unchanged; the request additionally carries the
  stop strings, taken from the model's entry in `configs/models_extension.json`.
- **Format control.** `google/gemma-2-9b-it` is ALSO run through the identical plain-text template
  (served name `google/gemma-2-9b-it+plain`), so that base-vs-it is compared with rendering held
  constant; the -it model's existing chat-template results are the third column.
- **Data.** The frozen discovery factorial (`phase1` driver: 8 cells × 20 discovery tasks, 3 feedback
  rounds, measured trial with 10 resamples, reversal and onset endpoints), identical stimuli and seeds.
  Amendments A1–A4 apply as to every other model. Judge (locked rubric, claude-sonnet-4-6, T = 0) on
  greedy measured / onset / washout / recovery responses of the base model only if the base model
  produces ≥ 20 parseable measured responses; the -it+plain run is judged on the hostile onset endpoints.
- **Analysis.** `scripts/explore_extension_model.py`-style contrasts (H1/H2a/H2b/H3a/H3b/H8 on discovery)
  for each of {base+plain, it+plain}, side by side with the -it chat-template numbers already published,
  plus the valid-answer rate per cell (feasibility), item-paired bootstrap CIs (2,000 resamples).
- **Feasibility gate (stated in advance).** If the base model yields parseable `Answer: X` on < 50% of
  neutral measured greedy trials, M1 contrasts are reported as "not estimable" and only the non-answer
  channel is discussed; nothing is tuned to raise the rate after the fact (no prompt changes, no
  re-rendering).

## Predictions (with confidence)

| ID | prediction | confidence |
|----|------------|-----------:|
| L1 | Feasibility: base gemma-2-9b under the plain template gives a parseable answer on ≥ 70% of neutral measured greedy trials. | 60% |
| L2 | The false-failure M1 drop (H1: malfunctioning − accurate, neutral tone, easy items) is present in the base model (CI excluding 0, negative). | 55% |
| L3 | The hostile-tone M1 drop (H2a/H2b) is smaller in the base model than in it+plain by at least half (ratio of point estimates ≤ 0.5), i.e. the tone channel is mostly post-training-installed. | 50% |
| L4 | The it+plain run reproduces the -it chat-template signature (H1 and H2a/H2b negative with CIs excluding 0), i.e. the signature is not an artefact of Gemma's chat markup. | 70% |
| L5 | Judged distress at hostile onset is lower in the base model than in it+plain (paired difference, CI excluding 0). | 65% |

Outcome map: L2 and L4 supported with L3 supported → validity channel pretraining-native, tone channel
post-training-amplified; L2 not supported → the whole signature is post-training-installed; L4 not
supported → the chat markup contributes and every earlier estimate carries that caveat.

## Interpretation ceiling (unchanged)

Provenance (pretraining-native vs post-training-installed) is a claim about where a behaviour was
learned; it licenses no claim about experience.
