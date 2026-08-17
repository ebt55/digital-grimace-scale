# Lab log

Dated notes, including retractions. Newest entries at the bottom.

## 2026-08-17 — agent A — generation stack: vLLM on Modal, live smoke on gemma-2-2b-it

**Built.** `OpenAICompatBackend` in `src/backend.py` (streaming chat-completions client with
top-20 logprob normalisation), `src/serve_modal.py` (one Modal app per model),
`scripts/preflight.py` (pins HF revisions + judge identity into `manifest.json`),
`scripts/smoke_backend.py` (this experiment).

**Deployment.** `dgs-vllm-gemma-2-2b-it`, A10G, vLLM 0.26.0 on `nvidia/cuda:12.9.0-devel-ubuntu22.04`,
flags `--dtype bfloat16 --max-logprobs 20 --max-model-len 8192 --enable-prefix-caching
--gpu-memory-utilization 0.90 --seed 0 --revision <sha>`. Image build 146 s (CPU only).
Cold start 212 s: weight download into the `dgs-hf-cache` volume, engine init, CUDA-graph capture.
gemma-2 logit softcapping worked with no attention-backend override.

**Model.** `google/gemma-2-2b-it` @ `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`.

**Letter-token check.** PASS — `A=True B=True C=True D=True`. For each letter, `"Answer: X"`
tokenises to the `"Answer:"` prefix plus exactly one extra token that detokenises to `" X"`.
Recorded in `manifest.preflight.letter_token_check`.

**M1 on the smoke record.** MISSING, `missing_reason = m1_invalid_final_answer`.
This is a model-behaviour result, not a plumbing failure: gemma-2-2b-it ends its answer with
`**Answer: D**` (markdown bold), and the frozen parser requires a final line exactly `Answer: X`.
All six turns of the trajectory were `final_answer_valid = False` for the same reason.
The logprob channel itself is sound — at the answer position the sampled token is `' D'` and
**all four candidates `' A' ' B' ' C' ' D'` are present in its top-20**. M1 becomes extractable
the moment the format issue is resolved. **This is a Phase-0 blocker and needs a protocol
decision** (the prompt is hash-locked, so it is not agent A's to change).

**partial_entropy rejected the record.** Max top-logprob mass excess measured across a whole
trajectory was `1.69e-07`, against `src/metrics.py: TAIL_MASS_TOLERANCE = 1e-9`. That is plain
float32 rounding in vLLM's logprobs, not a client defect — 26 of 76 positions in one turn
exceeded 1.0 by ~1e-7. The tolerance needs to be ~1e-6 for any real fp32 server. Not agent A's
file; flagged for the metrics owner.

**Throughput (A10G, gemma-2-2b-it).**

| run | generations | completion tokens | wall time | tokens/s |
|---|---|---|---|---|
| single greedy trajectory | 6 turns | 355 | 12.1 s | 29.4 |
| 8 concurrent trajectories | 48 turns | 3067 | 14.1 s | 217.3 |

7.4x speedup at 8-way concurrency; higher thread counts should do better still.
Mean response was ~63 completion tokens, far under the 512-token cap (0 truncations).

**Backend stats over the whole smoke.** 54 requests, 0 retries, 0 content mismatches,
0 non-finite logprobs, 0 truncations, 54 trailing special tokens trimmed,
16,948 prompt tokens, 3,422 completion tokens.

**Three client bugs found by the live run and fixed** (all had passing unit tests before, none
was reproducible offline without modelling the server's exact behaviour):

1. **EOS leak.** vLLM streams `<end_of_turn>` as a logprob entry with no matching content
   delta, so `"".join(token texts)` appended it to `response_text` — which alone would have
   made every response fail the `Answer: X` parser. Now trimmed, and counted as
   `trailing_special_tokens` (54/54 requests, so it fires on every call).
2. **SSE framing.** `httpx.iter_lines()` splits on everything `str.splitlines()` treats as a
   break, including U+2028/U+2029, which JSON permits *unescaped* inside a string. A generated
   token containing one chopped a `data:` line in half and the JSON parse failed mid-trajectory.
   Framing is now done on raw newline bytes.
3. **Letter probe.** `/tokenize` returns raw vocabulary pieces (`▁A`), not decoded text, so the
   check reported all four letters False. It now uses `/detokenize`, which returns `' A'` and
   matches what chat completions report.

**Cost.** GPU container ran 11:53:20–12:04:26 IST = 11 m 06 s = 666 GPU-seconds on A10G at
$0.000306/s ≈ **$0.20**. Image build was CPU-only. Total spend for this experiment ≈ $0.20.

**Teardown.** `modal app stop dgs-vllm-gemma-2-2b-it --yes`; `modal app list` shows state
`stopped`, `modal container list` empty. The `dgs-hf-cache` volume is retained on purpose so the
next deploy skips the weight download.

## 2026-08-17 — agent A — RETRACTION: every app served gemma-2-2b-it

**What was wrong.** `src/serve_modal.py` read its configuration into module-level globals:
`MODEL_ID = os.environ.get("DGS_MODEL_ID") or "google/gemma-2-2b-it"`. Modal re-imports the
module *inside the container*, where the deploying shell's environment does not exist, so every
container fell through to the default. App names were derived during the deploy pass and so came
out right — `dgs-vllm-qwen2-5-7b-instruct` — while the container served `google/gemma-2-2b-it`.
Confirmed in production: `/v1/models` on the qwen-7b and gemma-9b endpoints returned gemma.

**Why the smoke missed it.** The 2026-08-17 smoke only ever deployed the default model, so the
buggy fallback and the intended value were the same string. A single-model smoke cannot
distinguish "config propagated" from "config defaulted"; it needed a second model.

**Fix.** Configuration now resolves through `resolve_config(env)`, which prefers `DGS_BAKED_*`
values over the ambient `DGS_*` ones. The deploy pass bakes its resolved values into the image
(`image.env(BAKED_ENVIRONMENT)`), so the container pass reads back exactly what was deployed.
Added a hard startup guard, `_assert_serving_intended_model`: after launching vLLM the container
polls its own `/v1/models` and raises if the served id is not the intended one, so a
mis-configured server fails loudly instead of silently mislabelling every record.
`tests/test_serve_modal.py` imports the module twice under controlled environments — once as the
deploy pass, once as a container re-import with the deploy environment cleared — and asserts the
baked value wins; it also keeps the old failure mode as an explicit test.

**Empty responses (follow-up).** A Qwen-7B resample terminated immediately (EOS with no
content). The backend now records that as one zero-width position carrying the EOS
distribution, `response_text == ""`, counted as `stats["empty_responses"]`; `records.py` accepts
an empty `response_text` (tokens stay non-empty). **Open blocker:** a *multi-turn* empty response
still cannot be replayed, because `protocol.canonical_prompt_sha256` rejects an assistant turn
with empty content, and the runner appends exactly that. protocol.py is outside agent A's
boundary; allowing empty assistant content there is a pure extension (no existing hash changes,
since such messages were previously unrepresentable). Until that is decided, only single-turn
cells tolerate an empty response.

**Also changed.** `manifest.preflight.letter_token_check` (single) became
`preflight.letter_token_checks` (per-model dict, accumulating across runs, ordered by the
manifest's frozen model order), so every generated model carries its own check.
`verify_preregistration.py`'s no-artifacts sweep is now gated on
`generation_status == "not_started"` — it is the *pre*-generation firewall, and Phase 0 has
legitimately started filling `results/`. The `file_sha256`, split, and revision invariants still
run at every status. `tests/test_preregistration.py` pins its temp copy back to `not_started`
so it still exercises the sweep (minimal edit, noted here).

## 2026-08-17 — agent C — analysis glue: extraction, pipeline, G3/G4, scripts, figures

**Built.** `src/extract.py` (raw JSONL -> endpoints -> one flat `metric_rows.csv`/`.jsonl` plus
`qc_by_cell.csv`), `src/pipeline.py` (metric rows -> Phase-0 screen / G1 observations / reversal
rows / G5 rows / shuffled-label null / G3 style effects -> gate verdicts, with markdown
renderers), G3 and G4 in `src/gates.py` plus `compose_phase_1_gates`, and
`scripts/screen_phase0.py`, `scripts/analyze_phase1.py`, `scripts/make_figures.py` (F1/F2/F4,
PNG+SVG, regenerated only from committed summaries).

**Audit fixes in `src/analysis.py`.** Phase-0 screen now runs over the models present in the data
and tolerates QC-missing rows (a metric missing on one side of an item pair is excluded from that
metric's mean, counted in `n_unpaired_items`, and reported) instead of blocking on any deviation;
design contradictions (mixed runs, contradictory item difficulty) still block. The BH family and
the adjustment loop are now provably the same set, so a result with a validity but no tone
coefficient can no longer raise. The bare `except Exception` around the mixed model is narrowed to
named failure modes and now surfaces the exception message. G5 class balancing is an explicit,
documented `balanced_training_fold()` implementing the preregistered lexicographic
`task_id, condition_id` rule. `metrics.m2_disagreement` is left untouched; the glue catches its
raise and emits `MetricValue(None, "m2_incomplete_ensemble")`.

**Judgement calls not settled by the preregistration** (please log as dated amendments):

1. *MixedLM optimizer fallback.* The preregistration fixes the model, not the optimizer. `lbfgs`
   can stop where the numerical Hessian is not invertible on data the same likelihood fits
   perfectly well (observed on M2/M3 in synthetic fixtures). A fixed fallback order
   `("lbfgs", "powell")` is now tried and the optimizer that produced the reported fit is recorded
   on `G1MetricResult.optimizer`. Without it, a metric silently becomes "unavailable" for a purely
   numerical reason — and because `gates.evaluate_g1` requires *every* declared family metric to
   be estimable, one such metric makes the whole G1 gate UNAVAILABLE.
2. *G3 significance test.* Paired t-test across the five frozen smoke items (df=4), two-sided, BH
   adjusted within a G3-only family — not pooled with the G1 BH family, whose adjusted p-values are
   already frozen by `g1_adjusted_effects`. Reaching BH p<.01 at n=5 needs |t| > ~7.2; the smoke is
   explicitly provisional and Phase 2 uses the full battery.
3. *G3 comparison scale.* Style effects are standardised by the same model-neutral SD as the G1
   coefficients so "at least 50% of the false-failure effect" compares like with like; the
   reference is always the *validity* coefficient, including for metrics that qualified on tone.
4. *G4 transfer granularity.* Transfer is judged at the metric level (the same metric has an
   instability-positive G1 qualifier in both models), not at the (metric, effect) level.
5. *G1 BH scope.* One BH family is fit across both evaluated models, reading "within phase"
   literally; the shuffled-label null uses the same pooling.
6. *Length drift comparator.* For recovery/onset endpoints the comparator is still the same item's
   `<difficulty>__accurate__neutral` **measured** greedy endpoint, not a turn-matched one.
7. *Phase-0 escalation detection.* Escalation is read from the data (a conversation carrying five
   graded feedback rounds), not from a flag. After escalation a null screen is labelled
   `screen_null` and a primary/control are still named, per the preregistration.
8. *Phase-0 tone filtering.* Non-neutral rows in a Phase-0 table are ignored and counted rather
   than treated as a fatal design deviation; the frozen contrast is accurate+neutral vs
   malfunctioning+neutral.
9. *G5 feature scale.* G5 rows carry raw metric values; G5 z-standardises inside each training
   fold, so a global affine rescaling cannot change the fit, and a zero neutral SD does not delete
   rows from the classifier.
10. *M3 eligibility offline.* M1/M2 confirmatory exclusions are computed from the QC table
    (>5% missing greedy trials / >5% invalid-or-absent sampled responses in any confirmatory
    condition). M3's F1<.7 rule needs the human audit, so M3 stays eligible by default and the
    reason string says so; `--m3-audit-f1` applies the rule once the audit exists.

**Tests.** `tests/test_extract.py` and `tests/test_pipeline.py` build fixtures with the real
`run_batch`/`run_trajectory` and offline test backends. `PlantedEffectBackend` writes a known
false-failure effect into the answer margin, the resample ensemble and the revision rate;
`NullBackend` writes none. The pipeline detects the planted effect (G1/G2/G3/G4/G5 all PASS, G5
AUC gap 0.28) while the shuffled-label null comes out null, and finds nothing in the null model.
Note for whoever builds the fixtures next: a perfectly additive, noiseless synthetic design makes
the random intercept sit on its zero boundary, and driving length, M2 and M3 from one per-item
hash lets the length control absorb all between-item variance — both are estimator artefacts of
the fixture, not of the data.

## 2026-08-17 — agent B — generation driver and phase CLI

(Re-appended: an earlier concurrent write to this file dropped the original copy of this entry.)

**Built.** `src/generate.py` (concurrent, resumable trajectory driver) and `scripts/run_phase.py`
(`phase0` / `phase1` / `style-smoke` / `r5`); `src/runner.py` gained `run_kind`, an `allow_holdout`
gate, `run_single_turn_trajectory`, and the non-factorial planners. Validated with
`SyntheticBackend` only.

**Protocol decision — non-factorial injection point.** `configs/conditions.json` freezes the
style-control wording but leaves the injection point implicit, and Gemma-2 has no system role, so
every non-factorial cell is exactly one user turn: `style__<x>` is
`"<style prompt>" + "\n\n" + render_task(...)`; `style__neutral_reference` is `render_task(...)`
alone; `r5__*` is `render_r5_variant(item, variant)` alone. R5 records carry `split=None`,
`difficulty=None`, `domain=<category>`. Planned `phase` strings: style smoke `phase_1`, R5
`phase_2`. Factorial trajectories always run their continuations, Phase 0 included.

**Resume contract.** The unit is one trajectory `(model_id, task_id, cell_id, sample_index)`. Stored
JSONL is re-validated line by line with `response_id` recomputed; a trajectory is complete only if
its turn labels equal `response_turn_plan(...) + continuation_turn_plan(...)` (or `("measured",)`).
Incomplete, duplicate, or schema-invalid records are removed by atomic rewrite before anything is
appended; out-of-plan records are preserved. Failures go to `<out>.failures.jsonl` and never abort
the batch. `run_phase.py` refuses a real endpoint unless a 40-hex revision is pinned.

## 2026-08-17 — agent B — Amendment A1: markdown-tolerant final-answer parser

**Why.** Agent A's live smoke above: `google/gemma-2-2b-it` ends its response with `**Answer: D**`,
so all six turns of the trajectory parsed as `final_answer_valid = False` and M1 was MISSING with
`m1_invalid_final_answer`. The logprob channel was sound (all four `' A'..' D'` candidates present
in the top-20 at the answer position), so this was pure output formatting. The second-order harm was
worse than the missing M1: in the **accurate** arm the frozen branching would have graded genuinely
correct answers "Incorrect", silently contaminating the control condition that the whole
false-failure contrast is measured against. Agent A also measured a top-logprob mass excess of
1.69e-07 against `TAIL_MASS_TOLERANCE = 1e-9`, which is plain fp32 rounding in vLLM.

**Standing.** The preregistration permits parser tuning on discovery items and requires the parser to
be frozen before validation labels are revealed. **No Phase-0 data existed when this was decided** —
the only generation to date is agent A's single-trajectory plumbing smoke, which is not screen data
and is not analysed. The stimulus bank, prompts, cell wording, seeds, and response-ID keys are
untouched, so nothing hash-locked moves.

**Changed.**

1. `protocol.parse_final_answer` — the final nonempty line is normalised (remove every `*`, `_`, and
   backtick; collapse whitespace runs) and must then fullmatch `Answer:\s*([A-D])\.?`. The label stays
   case-sensitive, exactly one qualifying line must exist in the whole response (counted under the
   same normalisation), and it must still be the last nonempty line. So `**Answer: D**`,
   `**Answer:** D`, `Answer: **D**`, `` `Answer: D` ``, and `Answer: D.` now parse; `Final Answer: D`,
   `answer: D`, `Answer: D .`, two answer lines, and a non-final answer line still do not.
2. `AnswerResult` gained `letter_offset: int | None` — the option letter's absolute index in the
   **original** text, recovered through a per-character index map built during normalisation, so
   emphasis markers never shift it. None if it cannot be located unambiguously.
3. `metrics.m1_margin` locates the option token via `letter_offset`, falling back to the old
   `rfind("Answer: ")` only when it is None. The strict `\s*[A-D]\s*` token check is unchanged: if a
   server merges the letter into a token like `" D**"`, M1 stays MISSING with
   `m1_option_token_contains_visible_text`. Reported, not papered over.
4. `metrics._visible_reasoning_boundary` (M3 denominator) uses the same normalised rule via the shared
   `protocol.answer_line_match`, so an emphasised answer line is never counted as reasoning.
5. `metrics.TAIL_MASS_TOLERANCE` 1e-9 to 1e-6.

**Tests.** `tests/test_protocol.py` covers 13 valid forms with exact `letter_offset` assertions
(including `Answer: A`, where the `A` in "Answer" would defeat a naive search) and 15 invalid forms;
`tests/test_metrics.py` adds a bold-wrapped record with a fake token stream (M1 = 1.5), the merged
`" D**"` MISSING case, boundary checks, and tail-mass 1+5e-7 accepted / 1+5e-6 rejected. Full suite:
169 passed, 72 subtests.

## 2026-08-17 — agent C — Amendments A2 and A3 (discovery-stage, pre-Phase-1)

Both amendments were decided by the orchestrator on 2026-08-17 ~13:00 IST from Phase-0 DISCOVERY
data, before any Phase-1 generation, and must be applied identically to the Phase-2 holdout.
Every artefact now reports the frozen-rule and the amended outcome side by side; the amended one
is authoritative and `--no-amendments` on `screen_phase0.py`/`analyze_phase1.py` reproduces the
frozen-only analysis.

**A2 — treatment-blind item QC exclusion.** For each model separately, an item is excluded from
all cells, endpoints and analyses (screen, G1–G5, the shuffled-label null, the G3 style smoke and
the figures) when at least 5 of that model's 10 measured resamples in its own
`<difficulty>__accurate__neutral` cell are invalid or absent; an item whose baseline endpoint is
missing entirely is also excluded. The decision reads exactly one cell — the untreated baseline —
so it can never be informed by the manipulation. Rationale: Phase-0 invalidity is concentrated in
DGS-014, whose options are the single letters W/X/Y/Z so models answer `Answer: Y`, and DGS-022,
whose long derivations hit the frozen 512-token cap. Both are instrument non-compliance unrelated
to the manipulation; holdout DGS-013 carries the same single-letter defect. The M1 >5%-missing and
M2 >5%-invalid confirmatory exclusions are unchanged and are computed AFTER A2. Excluded items are
named in `screen.md/json`, `gates.md/json` and the `qc_by_cell.csv` table; `metric_rows.csv` still
holds every extracted endpoint, because A2 is an analysis-stage exclusion, not an extraction one.

**A3 — zero-variance standardization fallback.** Where a metric is standardized by a model's
neutral (accurate+neutral discovery, measured) SD and that SD is exactly zero, the pooled SD of
the same metric across all of that model's discovery factorial measured cells is used instead (for
the Phase-0 screen: across the two screen arms). The neutral mean is kept, so only the scale
changes, and the metric is unavailable only when the pooled SD is also zero
(`zero_neutral_and_pooled_sample_sd`). Strictly nothing changes when the neutral SD is nonzero.
The scale actually used is recorded per model x metric (`Standardization.scale_source`,
`MetricScreen.scale_source`) and printed in both reports.

**Effect on the Phase-0 discovery screen** (`results/raw/phase0`, four models; one Qwen-7B resample
trajectory was still being regenerated, so these may shift slightly):

| rules | primary | control |
|---|---|---|
| frozen (preregistered) | `google/gemma-2-2b-it` | `Qwen/Qwen2.5-7B-Instruct` |
| amended A2+A3 (authoritative) | `google/gemma-2-9b-it` | `Qwen/Qwen2.5-3B-Instruct` |

A2 excluded DGS-014 for `google/gemma-2-9b-it` (7/10 baseline resamples invalid),
`Qwen/Qwen2.5-7B-Instruct` (5/10) and `Qwen/Qwen2.5-3B-Instruct` (10/10); it did not fire for
`google/gemma-2-2b-it`. Under the frozen rules gemma-2-9b has only M1 available (M2 and M3 both
have zero neutral SD), so with a single available delta it cannot clear the "at least two positive
deltas" coherence bar and the primary fell to gemma-2-2b. A3 rescues gemma-2-9b's M2 on the pooled
scale, giving it two positive deltas and the highest S (1.346 vs 0.706). A2's removal of DGS-014
moves Qwen-7B's M1 from -0.019 to -0.258, so its |S| (0.297) now exceeds Qwen-3B's (0.216) and the
min-|S| control switches. **M3 is unavailable for every model under both rule sets**: its neutral
AND pooled SDs are zero, i.e. the frozen M3 parser fires no events anywhere in the Phase-0 data.
That is a genuine finding about the parser on this data, not a scaling artefact, and it should be
resolved before Phase 1 leans on M3.

**Also fixed.** Run IDs are per model, not per phase (each model runs on its own vLLM server):
`analysis.run_id_by_model()` is the shared invariant, applied in the Phase-0 screen, G1 and the
shuffle; two run IDs for one model still block with the model named.

**Judgement calls in these amendments** (beyond the orchestrator's specification):
1. An item whose accurate+neutral measured endpoint is entirely absent is treated as an A2
   exclusion (`accurate_neutral_measured_endpoint_absent`) rather than silently retained.
2. A3 replaces only the SD; the neutral mean is kept whenever the neutral cell has any observation,
   so z-scores stay centred on the untreated baseline.
3. `gates.StyleEffectEvidence` now accepts a nonempty SUBSET of the five frozen G3 smoke items,
   because A2 can drop one (DGS-022 is both a smoke item and an A2 candidate); the analysed IDs
   travel with the evidence and anything outside the frozen five is still rejected.

## 2026-08-17 — orchestrator — Phase-0 screen outcome and Phase-1 launch

**Course correction (morning).** The prior executor's eight commits were a sound preregistration
firewall with a synthetic-only pipeline and no way to call a model. Kept: locked configs/stimuli/
manifest, `protocol.py`, `records.py`, `metrics.py`, and the G1/G2/G5 statistics. Built today by
four agents (backend + Modal server; concurrent resumable driver + phase CLI; extraction/pipeline/
G3/G4/figures; judge client). All amendments are registered in `notes/amendments.md`.

**Environment facts.** HF access OK for gemma-2-2b-it, gemma-2-9b-it, gemma-2-9b, Qwen2.5-*;
`meta-llama/Llama-3.2-3B-Instruct` 403 (no licence) — dropped from the screen (four models, not
five). Judge pinned `anthropic/claude-sonnet-4-6` at temperature 0 (`claude-sonnet-5` rejects the
temperature parameter). Modal: vLLM 0.26.0, `--max-logprobs 20`; A10G for 2–3B, L40S for 7–9B.

**Phase 0 (screen items DGS-003/005/010/014/018/022/026/030/034/037; accurate+neutral vs
malfunctioning+neutral; greedy + 10 resamples; 220 trajectories / 1,430 records per model).**
Wall time 2.5–4.3 min per model at 96 concurrent trajectories; 0 permanent failures after two
edge-case fixes (empty-string logprob token; empty EOS-only response). Estimated GPU spend for
Phase 0 ≈ $1.

| model | M1 (sign-aligned z) | M2 | M3 | S | coherent |
|---|---:|---:|---:|---:|:---:|
| gemma-2-9b-it | +2.00 | +0.69 (pooled scale, A3) | n/a | 1.35 | yes |
| gemma-2-2b-it | +0.48 | +0.94 | n/a | 0.71 | yes |
| Qwen2.5-3B-Instruct | −0.35 | −0.08 | n/a | −0.22 | no |
| Qwen2.5-7B-Instruct | −0.26 | −0.34 | n/a | −0.30 | no |

Authoritative (A2+A3) selection: **primary gemma-2-9b-it, control Qwen2.5-3B-Instruct**. Frozen
rule would have selected gemma-2-2b-it / Qwen2.5-7B-Instruct (gemma-2-9b's neutral M2 is
identically 0, so the frozen rule calls M2 "unavailable" and the model "incoherent"). Both
Gemmas move toward instability under false-failure and both Qwens do not — the preregistered P4
family boundary already appears at screen strength.

**M3 is unavailable for every model**: the frozen parser fires zero events in any response. Spot
checks (gemma-2-9b greedy: 21/130 responses contain a mid-body proposal, 1 revise cue) show this is
genuine — the models do not visibly thrash on these tasks within the 512-token cap — not a markdown
artefact. M3 is therefore reported as unavailable (zero variance); the human M3 audit will be run
on the export but cannot rescue a metric with no predicted events.

**Believability spot-check.** Read transcripts: the false-failure verdicts are engaged with (models
re-derive, occasionally switch letters or refuse to pick — "Answer: Cannot be determined"); the
truthful cause-removal correction is delivered in every malfunctioning conversation.

**Phase 1 launched 13:12 IST** on discovery items: full 2×2×2 factorial + reversal/onset on the
primary and control, plus gemma-2-2b-it and Qwen2.5-7B-Instruct as logged exploratory extras
(cheap; makes the G4 boundary test stricter). Style smoke (5 items × 5 style cells × 11) on the
primary. Judge pass and G1–G5 follow.

## 2026-08-17 — agent C — Phase-1 gate family, extra models, streaming extraction

**Gate family = eligible AND estimable.** `run_phase1_gates` now drops a QC-eligible metric that
cannot be estimated for the primary model — zero variance so no z-scale exists even under A3, or a
model that will not fit — records the reason (`zero_variance`, or the fit's own reason) in
`Phase1Verdict.unavailable_metrics`, and refits the Benjamini-Hochberg family over the survivors.
G1–G5, the shuffled-label null and the G5 feature set all follow the surviving metrics exactly.
Only when nothing is estimable does the full eligible family go forward, so G1 reports UNAVAILABLE
rather than passing on nothing. This matters immediately: M3 has zero neutral AND pooled variance
for every model in Phase 0, so it is expected to drop out of the Phase-1 family with reason
`zero_variance` while M1/M2 decide the gates.

**Extra models.** `analyze_phase1.py --extra HF_ID` (repeatable). Extras get the same per-model
G1/G2/G5 and null-check tables, labelled exploratory, and their G1 evidence enters G4's
family-boundary comparison — "at least one Gemma has an eligible positive" may be satisfied by
either Gemma, and "every evaluated Qwen lacks one" now considers both Qwens. The gate verdict
columns stay primary/control as preregistered, and `gates.md` names which models the boundary
decision evaluated and in what role.

**Streaming extraction.** Raw Phase-1 files are 0.8–1.3 GB each because every token carries twenty
logprobs. `extract.iter_records()` parses one line at a time (`newline="\n"`, so U+2028/U+2029
inside a response still cannot end a record) and `build_metric_rows()` reduces each record to its
metric contribution and releases it: greedy records become their computed columns, resample records
become a ten-field stub carrying only what M2 reads. Peak memory now scales with the number of
endpoints, not with the raw bytes. `load_records()` remains as the materialising wrapper for tests.

## 2026-08-17 — agent D — manipulation check PASSED; judge cache made content-addressed

**Manipulation check (live, pinned judge `anthropic` / `claude-sonnet-4-6`, temperature 0).**
Outputs in `results/summaries/manipulation_check/` (JSON + CSV + markdown table). **PASSED**:
hostile mean 6.5 vs neutral mean 1.5, all four tone-matched pairs strictly ordered, and every
dry turn (cause-removal correction, truthful washout, irrelevant-information control,
measured-trial message) scored below the lowest hostile string — evidencing the preregistered
requirement that the correction turn stays maximally dry. `sampling_mode: temperature_zero`,
so the wire request carried the preregistered temperature literally.

**Judge cache is now content-addressed — this was a live correctness fault.** `cache_key`
gained `input_sha256`. `response_id` is derived from model/revision/task/cell/turn/sample and
is *unchanged when a trajectory is regenerated*, so the previous key would have served the
stale score for freshly generated text. Caught while preparing the Phase-1 judge run against
the ~96 regenerated placeholder trajectories. Regenerated responses now miss and are
re-judged; untouched responses stay cached. Cache lines written before this change lack the
field and are ignored rather than trusted. Regression test:
`test_regenerated_response_is_not_served_from_the_cache`.

**Judge CLI hardened for GB-scale raw.** `scripts/run_judge.py` streams raw JSONL line by line
and applies a cheap dict-level pre-filter (`trajectory_kind`/`sample_index`/`turn_label`/
`model_id`) before `record_from_json`, so a 1.24 GB Phase-1 file is scanned in 20 s with
bounded memory instead of being materialised. Added `--models`; `run_manifest.json` now
records provider usage and list-price cost.

**Phase-1 judge dry run** (Phase-0 raw as stand-in, `--limit 5`, scratch dir since deleted):
5/5 judged, 0 failures; rerun 5 cache hits at $0.0000. Eligible counts confirmed on the real
Phase-1 files: 160 per model (80 measured, 40 onset, 40 recovery) for both
`google/gemma-2-9b-it` and `Qwen/Qwen2.5-3B-Instruct` — **320 judge calls**, estimated
**$0.40–0.90** at $3/$15 per MTok with the 1,084-token rubric block prompt-cached.

### Provisional first pass over the Phase-1 raw (agent C, 2026-08-17)

`analyze_phase1.py` ran end-to-end over `results/raw/phase1` (22,880 records, 0 malformed lines,
peak RSS ~36 MB after the streaming fix). Exit code 4 is the script's BLOCKED-verdict code, not a
crash. **Numbers are provisional** — ~96 trajectories per model still carry the warm-up placeholder
`initial` record and will be regenerated.

Every metric is currently unusable, for two independent reasons:

- **M1 QC-excluded for all four models** (frozen rule: >5% of required greedy trials missing in any
  confirmatory condition). Worst-cell missing rates: gemma-2-9b 0.222, gemma-2-2b 0.125,
  Qwen-3B 0.222, Qwen-7B 0.800. This is almost certainly the format failure agent A logged on
  2026-08-17: models end with `**Answer: D**` (markdown bold) and the frozen parser requires a final
  line exactly `Answer: X`, so the answer is invalid and M1 has no option-letter token to read.
- **M2 QC-excluded for all four models** (>5% of required sampled responses invalid): worst-cell
  rates 0.100–0.150 — the same format failure in the resample ensembles.
- **M3 is the only QC-eligible metric and has zero variance** (neutral and pooled), so no z-scale
  exists and it cannot be estimated.

Nothing is estimable, so G1 is UNAVAILABLE and Phase 1 is BLOCKED — which is the behaviour the
amended family rule specifies, not a pipeline failure. The blocker is upstream: the `Answer: X`
format compliance (a protocol decision, since the prompt is hash-locked) and whatever makes the M3
parser fire no events at all. Both need resolving before the Phase-1 gate can say anything.

## 2026-08-17 — agent C — Amendment A4 (pooled confirmatory QC bars)

**Amendment A4 (decided by orchestrator 2026-08-17 ~16:35, before any Phase-1 effect estimate was
viewed; structural rationale).** Each factorial cell holds only 10 discovery items, so the frozen
per-condition 5% QC bar is effectively zero-tolerance: one invalid greedy answer in one cell is
already 10% and excludes the metric outright. A4 evaluates the same two 5% bars POOLED across the
model's discovery factorial cells (measured endpoint, after A2's item exclusion): M1 is excluded if
more than 5% of that model's required greedy trials are missing, M2 if more than 5% of its required
sampled responses are invalid or absent (k stays frozen at 10). The bars themselves, the metrics
they govern and the M3 F1 rule are unchanged — only the denominator moves from one cell to the
model. Per-cell rates are still computed, carried on `MetricEligibility.worst_cell_id`/`worst_rate`
and printed in `gates.md` next to the pooled rate; `qc_by_cell.csv` keeps its per-condition
breakdown untouched. A4 is flag-controlled exactly like A2/A3 (`--no-amendments` reproduces the
frozen per-condition outcome) and must be applied identically to the Phase-2 holdout.

Ordering note: A2 runs before A4, so an item whose baseline resample ensemble is mostly invalid is
removed from the design first and never contributes to the pooled QC rate.

**Two corrections to my earlier provisional note.** (a) That run was over PURGED files — 96 whole
trajectories per model had been removed pending regeneration — so the M1/M2 missing rates I logged
(0.125–0.800) are inflated by absent trajectories and should not be read as parser failures. Do not
use them. (b) Amendment A1 is confirmed active in extraction: on `google__gemma-2-9b-it.jsonl`,
greedy measured records whose final line is `**Answer: D**` parse valid and yield real M1 margins
(12.25 and 9.81, no missing reason). The remaining `m1_invalid_final_answer` cases are genuine
non-answers such as `**Answer:**` and `**Answer:  None of the above**`. No bug.

## 2026-08-17 — agent C — G2 complete-case fix, exploratory appendix, figures

**G2 `required_reversal_endpoint_missing` was a glue gap, not a data wall.** `analysis.g2_reversal`
voids a metric outright if *any* supplied false-negative-eligible row has a missing endpoint. With
the gate family reduced to {M2}, and M2 missing whenever any one of its ten resamples returns an
invalid final answer, that guard fired for every model: M2 is present on only 46–61 of 80 measured
and 16–27 of 40 recovery endpoints. Eligibility itself was never the problem — 29–34 of ~38 rows
per model are false-negative eligible — and complete triples do exist (11–24 per model).

The preregistration's standing treatment of a quality-control gap is to exclude the observation
from that metric's estimate, count it and report it, not to void the estimand. Complete-case
selection now happens in the glue (`pipeline.complete_reversal_rows`), the frozen `g2_reversal`
contract is untouched, and the number of eligible rows dropped for an incomplete triple is carried
on `ModelAnalysis.g2_incomplete_dropped` and printed in the G2 table. The descriptive reversal now
computes for all four models (M2, sign-aligned z, item-clustered bootstrap):

| model | items | dropped | induction | recovery | recovery 95% CI |
|---|---:|---:|---:|---:|---|
| gemma-2-9b-it | 15 | 10 | -0.370 | 0.082 | [-0.197, 0.352] |
| gemma-2-2b-it | 9 | 17 | 0.508 | 0.056 | [-0.474, 0.469] |
| Qwen2.5-3B | 8 | 19 | 0.000 | -0.296 | [-0.592, -0.090] |
| Qwen2.5-7B | 12 | 6 | 0.161 | -0.322 | [-0.748, 0.000] |

No model shows the preregistered reversal pattern: the primary's induction is negative and its
recovery CI spans zero. G2 remains NOT_EVALUATED at the gate because G1 failed; this is descriptive.

**Exploratory descriptive appendix** — `results/summaries/phase1/exploratory/`
(`appendix.md`, `cell_endpoint_summary.csv/.jsonl`, `paired_contrasts.csv`). Clearly labelled
EXPLORATORY: no QC exclusion, no amendment, no confirmatory status, raw (unstandardised) values,
every endpoint and cell present in the raw data. Per model × cell × endpoint means and item counts
for M1, M2, entropy, length, greedy accuracy, non-answer rate and resample-invalid rate (80 rows);
plus 314 paired item-level contrasts with 2,000-resample item-clustered bootstrap CIs for validity
(mal−acc within tone×difficulty), tone (hostile−neutral within validity×difficulty),
recovery−measured, onset−measured and washout−onset, over M1, M2, accuracy and non-answer rate.
Conventions are stated in the appendix header: accuracy is scored only over answers that parsed,
so an unparseable answer is absent rather than counted wrong, and `non_answer_rate` carries that
information separately.

**G5 caveat now stated in `gates.md`.** The primary's G5 "PASS" rests on a baseline
(correctness + length) AUC of 0.27 — below chance, i.e. the baseline predicts the condition
backwards out of fold — against a full-model AUC of 0.53, itself barely above chance. The
preregistered gap rule is applied unchanged, but a gap manufactured by a sub-chance baseline is not
evidence that the primary metrics carry condition information, and the write-up must say so.

## 2026-08-17 — agent C — preregistration v3 confirmatory script (frozen before the holdout)

`src/confirm.py` + `scripts/confirm_holdout.py` implement `notes/preregistration_v3.md`. The
hypothesis table is transcribed into a frozen `HYPOTHESES` tuple (H1, H2a, H2b, H3a, H3b, H4a, H4b,
H5, H6a, H6b, H7a, H7b, H8, H9) with each contrast's endpoint pair, stratum, predicted direction and
discovery estimate fixed in code, so no stratum or direction can be chosen after seeing the data.
Extraction streams through `src.extract`; A1–A4 apply as on discovery, with A2 computed on the
holdout's own accurate+neutral resamples per model. M1 is available-case in raw nats with the
per-cell non-answer rate tabled beside it. Every contrast is an item-paired mean difference with a
2,000-resample item-clustered bootstrap percentile 95% CI; two-sided bootstrap p-values are
BH-adjusted across H1–H9 as the secondary summary. Outputs: `confirm.md`, `confirm.json`,
`hypotheses.csv`, `shuffled_null.csv`, plus the metric-row/QC/A2 tables.

**Seven judgement calls the v3 text does not settle. Numbers 1, 2 and 4 change `iteration_status`,
so they need the orchestrator's explicit sign-off before the confirmatory run.**

1. *Shuffle granularity.* The permutation runs within each item — swapping the two cells that differ
   only on the permuted axis — not freely across the `model × difficulty` stratum. A free stratum
   permutation can give one item two "malfunctioning" cells and another none, which leaves a paired
   contrast with no partner to look up. Within-item swapping preserves each stratum's label counts
   exactly (every item still contributes one cell per label) and keeps the pairing defined.
2. *Which label is permuted.* v3 says validity labels always, plus tone labels "for tone
   hypotheses". Permuting both moves the accurate arm into the malfunctioning arm, where `onset` and
   `onset_washout` do not exist; in testing that dropped 17 of 20 items and left a degenerate,
   biased three-item sample that spuriously "supported" H6a. The permutation is therefore applied to
   the axis that defines the contrast: validity for a validity contrast, tone for a tone contrast.
3. *"No effect" rules inside the null.* H5 (CI upper ≤ +1.0 and point ≤ 0) and H7 (CI includes 0 or
   is positive) are satisfied by construction on shuffled — hence effectless — data, so a literal
   reading makes the null impossible to pass. In the null those two are replaced by the signed
   direction behind them (H5 negative, H7 positive).
4. *Scope of the null verdict.* H3, H4, H5 and H6b compare two endpoints of the SAME cell (a
   different turn, or the same cell in the other model). Permuting condition labels leaves those
   contrasts numerically intact, so their shuffled repeat says nothing about label-driven false
   positives. They are computed and shown marked `label-invariant`, and the verdict rests on the
   eight contrasts the permutation can actually break: H1, H2a, H2b, H6a, H7a, H7b, H8, H9.
5. *BH family.* H1–H9 means every tested contrast except the H10 style battery, with H7 contributing
   H7a and H7b separately (14 tests).
6. *H10 threshold.* "50% of the H1 effect" uses the **holdout** H1 point estimate; with no H1
   estimate no violation can be declared.
7. *Bootstrap p-value.* Two-sided percentile: twice the smaller tail mass at zero, capped at 1.

`--dry-run-discovery` points the same code at Phase-1 discovery and labels every output
"DRY RUN ON DISCOVERY — NOT CONFIRMATORY"; it was used to exercise the path end to end and its
output directory is deleted afterwards. Nothing under `results/raw/phase2` or
`results/raw/style_battery` was read or written by me.

### Prereg v3 clarification C1 (pre-analysis, 2026-08-17 ~17:10)

Recorded before any holdout data was generated or seen. The v3 shuffled-label null said "the null
passes if no shuffled contrast is supported". Testing on discovery showed that criterion is not
usable: a single deterministic permutation applied to ~10 contrasts fails by chance too often. In
the discovery dry run the single shuffle collapsed every label-dependent contrast as it should
(H1, H2a, H2b, H6a, H7a, H7b, H8 all unsupported) yet the whole null still "failed" on H9 alone — a
nine-item binary outcome where three informative items happened to keep their orientation.

The null check is therefore a **family-level permutation test**:

1. The null family is the directional, label-dependent set **L = {H1, H2a, H2b, H6a, H8, H9}**.
   H3a, H3b, H4a, H4b, H5 and H6b compare two endpoints of the same cell and are
   permutation-invariant; H7 is a no-effect rule on the control; H10 compares against the style
   reference. All are excluded from the family.
2. `real_count` = hypotheses in L supported on the real labels.
3. For k = 1..200, labels are permuted with the deterministic key
   `DGS-AC1-SHUFFLE-v3|<k>|<model_id>|<task_id>|<cell_id>` — validity labels for H1 and H8, tone
   labels for H2a, H2b, H6a and H9 — and the supported count in L is recorded.
4. `null_p` = (1 + #{k : count_k >= real_count}) / 201. The null check PASSES iff `null_p` < .05 and
   `real_count` > 0; a family that supports nothing has nothing to beat and fails.
5. The success criterion is otherwise unchanged: at least three of {H1, H2a, H2b, H3a, H3b}
   supported, H6a supported, and the null check passing.

The single-shuffle (k = 1) per-hypothesis table is still computed and printed for transparency, but
it no longer decides anything. `real_count`, the permutation-count histogram and `null_p` are
reported in `confirm.md` and `confirm.json`.

Agent C's three earlier judgement calls stand as accepted: the null verdict covers label-dependent
contrasts only; the permuted axis is the axis that defines each contrast; and H5/H7's no-effect
rules are excluded from the null with their directional question asked instead.

## 2026-08-17 — orchestrator — Phase 1 verdict, iteration loop on the locked holdout, close-out

**Phase 1 (discovery).** Frozen rules: BLOCKED (per-cell 5% QC bars exclude M1 and M2 for every model;
M3 zero variance). Amended A1–A4: determinate FAIL (family {M2}; primary M2 validity −0.29 z, tone
+0.38 z, both n.s.; shuffled null passes; G5 "PASS" rests on a 0.27 baseline AUC and is not evidence).
Exploratory contrasts (no QC exclusion, labelled) showed large M1 effects of false failure, hostile
tone and single bogus failure in gemma-2-9b-it, partial recovery after a truthful correction, none after
three rounds, treatment-dependent non-answers, distress language at hostile onset (Gemma ≫ Qwen), and
M2 more sensitive to style prompts than to false failure.

**Iteration loop.** `notes/preregistration_v3.md` (H1–H10, confidences, success criterion) committed at
aa5cd44 before any holdout generation; clarification C1 (200-permutation family null) at acf571f
before any holdout record was read; confirmatory script frozen at 79a5317 and recorded in
`manifest.holdout_unlock`; holdout generated 17:02–17:21 (primary + control, factorial + style
battery; 0 failures; zero empty responses), judged (400 responses, $0.52), analysed ONCE at 17:26 at
f7f1c4a. Result: **SUCCESS** — H1, H2a, H2b, H3a, H4a, H5, H6a, H6b, H8, H9, H10 supported; H3b, H4b not
(n = 4–5 after hostile-cell non-answers); H7 (family boundary) NOT supported — Qwen-3B shows the M1
effects too (transfer). Full table: `results/summaries/phase2/confirm.md`; report: `notes/report.md`.

**Operational incidents today (all logged above):** serve_modal env-var bug (caught before any
mis-served data); U+2028 in JSONL readers; empty-string logprob tokens; warm-up empty streams recorded
as placeholders (417 trajectories purged and regenerated); one deploy killed by a truncated PowerShell
pipeline (no data, no cost); judge crash from editing the manifest mid-run (re-run from cache).

**Spend.** Modal GPU ≈ $9 (Phase 0 ≈ $1, Phase 1 ≈ $4 incl. regeneration and one debug redeploy,
holdout ≈ $2, smoke/idle ≈ $2); Anthropic judge ≈ $2.9 (manipulation check, Phase 1 640 responses
$1.07, holdout 400 responses $0.52, validation calls). All GPU apps stopped; `modal container list`
empty at close-out.

**Still open (human tasks):** blinded M3 audit (moot: parser fires no events) and the 15-per-model judge
audit (`scripts/run_judge.py audit-sample`); refusal-pressure R5 battery (P6) not run; Phases 3–5 not
started.

## 2026-08-17 — agent J2 — Phase 3 analysis code built and tested offline (no GPU, no holdout read)

Built the Phase-3 analysis half of `notes/preregistration_v4_phase3.md`, against agent J1's
`jspace_client` contract, which is imported lazily so every offline path runs without a deployment.
**Nothing has been run against Modal and no Phase-3 result exists yet**: this entry records the code
and the decisions frozen in it, not findings.

New files (nothing existing was modified): `src/probe.py` (probes, layer choice, correlation,
directions, readouts, J1–J6 verdicts), `src/steer_readouts.py` (judge adapter), `scripts/run_phase3.py`
(`extract` / `probe` / `steer` / `report`), `scripts/make_phase3_figures.py` (F5, F6),
`tests/test_probe.py` (41 tests, all synthetic and mocked).

**Item-count correction, recorded because it contradicts the preregistration's own wording.** v4 says
"8 cells × 20 tasks per split". A task's difficulty fixes half the factorial, so each of a split's 20
tasks appears in exactly **4** cells: **80** measured-position transcripts per split, not 160. Counted
from the raw files: phase1 80 and phase2 80 measured greedy sample-0 factorial records. `extract`
asserts 80/80/10 (the style set is 5 tasks × {verbose, neutral_reference}) and the number is printed
in `localization.md`. Nothing about the design changes; only the arithmetic in the prose was wrong.

Decisions frozen in the code, each stated so it can be audited rather than inferred:

- **LOO pooling.** Out-of-fold decision scores from all 20 task-folds are pooled and one AUC is
  computed, rather than averaging 20 fold AUCs (a fold holds 4 rows, so per-fold AUCs are degenerate).
- **Layer choice.** `argmax` discovery tone AUC, ties to the lower layer, computed from discovery
  alone. `probe` refuses to overwrite an existing `localization.json` without `--force`, because the
  holdout evaluation at L\* happens once.
- **J2's split.** v4 does not name one. The headline J2 verdict uses the **holdout** AUCs at L\*
  (matching J1's confirmatory clause) and "decodable" is read as AUC > 0.5; the discovery LOO figures
  at L\* are reported beside it.
- **Correlation estimator.** Probe score and M1 are demeaned **within cell** and the residuals pooled;
  the 2,000-resample item-clustered bootstrap re-demeans inside each resample, so the interval covers
  the estimator that produced the point estimate rather than a frozen residualisation.
- **Dose vector.** The caller passes `d/‖d‖ · norm_L*` and the server multiplies by α — the reading of
  J1's contract under which its `alphas` list is meaningful. α = 0 is generated **once**, as the shared
  baseline for every direction, and every readout is paired against it by item.
- **Degenerate dose.** Strictly `> 50%` of items with no parseable answer, applied by rule; excluded
  doses are named in the monotonicity note rather than dropped silently.
- **Judge route.** `scripts/run_judge.py` reads validated `RawRecord` JSONL only, and a steered
  generation has no frozen seed or `response_id` and a re-rendered prompt, so forcing one into that
  schema would fabricate provenance. `src/steer_readouts.py` therefore calls `judge_client.score_text`
  directly with the **locked** rubric (`configs/judge_rubric.md`, verified against
  `manifest.file_sha256`), same temperature-0 contract and cache, and labels its output
  `dgs-steering-judge-v1` so it can never be confused with the confirmatory judge channel. No judge
  file was modified. J6 dose plan: tone at α ∈ {0, 2, 4}, every control at α = 2.
- **sklearn 1.8+.** `penalty="l2"` is deprecated; the default (`l1_ratio = 0`) is pure L2 and gives
  bit-identical coefficients, so the default is relied on and the equivalence is asserted in tests.

Offline verification: 41 unit tests pass (planted-layer LOO AUC 1.00 vs < 0.80 elsewhere, tie-breaking,
holdout transfer, cell-demeaned Spearman recovering −1.0 through a 100× cell offset, direction scaling
and matched norms, the degenerate-dose rule, readout aggregation and each J-verdict's failure modes),
plus a scratch end-to-end run of `extract → probe → steer → report → figures` on the **real** item sets
with a mocked client, which exercised the raw-file filters (80/80/10), the M1 join against
`results/summaries/phase2/metric_rows.csv` (73 of 80 holdout endpoints available-case), resumption of
`steering_outputs.jsonl`, and both figures. Outputs went to a scratch directory; no Phase-3 summary was
written into the repository.

## 2026-08-18 - agent J1 - Phase 3 (j-space) infrastructure: Modal activation/steering app

Built for the prereg v4 Phase 3 design: `src/jspace_modal.py` (Modal app `dgs-jspace-gemma-2-9b-it`),
`src/jspace_client.py` (local chunking/npz/token conversion) and `tests/test_jspace_client.py`
(23 offline tests, no `modal`/GPU import). Deployed, smoked against the live app, stopped.

**Stack.** `google/gemma-2-9b-it` @ 11c9b309 (manifest pin), bf16, `attn_implementation="eager"`
(the only implementation that applies gemma-2's attention softcap exactly), L40S with A100-40GB
fallback, torch 2.9.1 (cu12.8 wheels), transformers **4.57.6**, accelerate 1.14.0, numpy 2.4.6.
transformers 5.15.0 is current but was rejected deliberately: 4.x builds the hidden-state tuple
inside `Gemma2Model.forward`, so `hidden_states[i]` has the documented meaning this phase's whole
layer convention rests on, while 5.x moved that plumbing into a capture decorator whose per-layer
semantics could only be confirmed by burning GPU time. Weights loaded from the existing
`dgs-hf-cache` volume (no re-download): 16-25 s.

**Three things that would have silently corrupted Phase 3, found and fixed before any real run.**
(1) This revision's `generation_config.json` lists only `<eos>` (id 1); the instruction-tuned model
ends its turn with `<end_of_turn>` (id 107). Both are now treated as EOS - otherwise every greedy
generation runs to `max_new_tokens` and starts a fresh `<start_of_turn>`. (2) gemma-2 ships
`cache_implementation="hybrid"`, which routes `generate` onto a static cache and `torch.compile`;
transformers 4.57 unsets that on the model's own config, but a deep copy keeps it, costing ~5 min of
recompilation *per steering hook* (first steered call 335 s, the next 5 s). Now unset explicitly:
dynamic cache, no compile. (3) The chat template emits `<bos>` itself, so prompts are tokenized with
`add_special_tokens=False`; a second BOS would shift every activation.

**Layer indices are hidden-state indices everywhere.** 0 = embedding output, i in 1..41 = output of
decoder block i-1, 42 = final normed state. `generate_steered(layer=i)` hooks decoder block i-1, so
the added vector first appears exactly at `hidden_states[i]` - the stream the direction was measured
in, with no off-by-one for the caller to remember.

**Live measurements (L40S, prompts ~1127 tokens, 3-turn factorial shape).** Container start + load
38 s. Extraction, all 43 layers, batch 8: 0.448 s/item wall (2.78 s remote for 16 items) -> a
170-item split is ~2 min, a 320-item pair of splits ~3-4 min. Greedy generation, batch 8:
**61.5 tok/s aggregate** (1816 tokens in 29.5 s); at batch 2 it was 17.8 tok/s, so batch size is the
throughput lever. ~700 generations at <=512 tokens is ~1.0-1.6 h of GPU. Left-padding correctness
check: the same item extracted alone (113 tokens) and behind a 1647-token prompt agrees to
cosine 0.999957 (max abs diff 1.0 on a vector of norm 259, i.e. float16 quantisation). Peak absolute
activation 740, so the float16 store never overflows (guarded and counted regardless). Mean L2 norms
at the final prompt token: layer 0 = 103, 10 = 97, 21 = 256, 30 = 451, 42 = 121.

**Calibration warning for the steering run.** The prereg's dose is `alpha * d/||d|| * ||mean
activation||`, i.e. at alpha=2 the added vector has twice the norm of the entire residual stream. A
*random* unit direction at alpha=2 already destroys generation completely: 8/8 items ran to the token
cap emitting "About a    ...   ...", none parseable. Expect the degeneracy rule (">50% of items yield
no parseable answer") to fire at the top of the ladder for the random controls, and probably for the
tone direction too; the informative doses are likely 0.5 and 1.

**Spend.** ~25 min of L40S across three live runs including 5-min idle tails (deploy and image build
are free/CPU): roughly $0.8-1.0. App stopped, `modal container list` empty at close-out. Redeploy is
instant (image cached); deploy costs nothing until a method is invoked.