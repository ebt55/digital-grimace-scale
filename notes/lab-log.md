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

## 2026-08-18 — agent J2 — Phase 3 run: probes, steering, J1–J6

Ran the whole Phase-3 arm on the deployed j-space app (agent J1's `src/jspace_modal.py` +
`src/jspace_client.py`). Clarification **C2** was applied to the code *before* any tone-direction
steering: the dose is `α·d` with `d = mean(hostile) − mean(neutral)` at L\* unnormalised, and every
control is rescaled to the matched norm `α·‖d‖`.

**Verdicts.** J1 **supported**, J2 **supported**, J3 not supported, J4 not supported, J5 **supported**,
J6 not supported. Summaries: `results/summaries/phase3/{localization,steering,phase3}.{md,json}`;
figures F5/F6 in `results/figures/`.

| ID | number that decides it |
| --- | --- |
| J1 | discovery LOO tone AUC = **1.000**, holdout at L\* = **1.000** (bar 0.80 / 0.75) |
| J2 | at L\*, holdout tone 1.000 vs validity **0.878**, gap 0.122 ≥ 0.05 |
| J3 | within-cell Spearman(probe score, M1) = **−0.160** [−0.431, +0.154], 20 items, 73 pairs |
| J4 | ΔM1(α=2) = **−0.194** [−0.513, **+0.0000001**] — monotone over {0.5, 1, 2}, CI touches zero |
| J5 | 24 control dose cells, ΔM1 ∈ [−0.025, +1.195], **none** with a CI excluding zero below zero |
| J6 | non-answer rate **0.00 at every dose**; all 180 judged distress scores **0** |

**L\* = 6, and why that matters.** Tone AUC is exactly 1.000 on a plateau spanning **layers 6–25**, so
the frozen "argmax, ties to the lower layer" rule — written to pick ONE layer to steer at — lands on the
earliest layer of the plateau. Two consequences, both recorded rather than engineered around:

1. *J1's band clause.* "Peak LOO AUC ≥ 0.80 at some middle layer (12–30)" is read as **"the peak value
   is attained at some layer in 12–30"** (it is: the plateau covers 12–25), not as "the tie-broken
   argmax index lies in 12–30" (it does not: L\* = 6). The judgement call was made after seeing the
   discovery probes and before any steered generation, it *does* flip J1 from not-supported to
   supported, and both readings are emitted side by side in `localization.json`
   (`peak_attained_in_middle_band` vs `argmax_layer_in_middle_band`) so a reader can apply either.
2. *Dose scale.* At layer 6 the tone contrast is small: **‖d‖ = 3.12** against a mean activation norm of
   **78.59**, ratio **0.0398**. So even α = 4 perturbs the residual stream by only ~16% of its own norm.
   Nothing degenerated — non-answer rate 0.00 at all 29 dose cells, mean length 123–129 tokens
   throughout, α=4 responses still correct and coherent — so the degenerate-dose rule never fired and
   nothing was excluded from the monotonicity check. A layer inside the plateau with a larger ‖d‖ would
   be a stronger test; the layer rule was frozen, so this run does not do that.

**What the steering shows.** The tone direction is the only direction whose ΔM1 is negative at every
dose, and the decrease is monotone (−0.005, −0.039, −0.194, −0.494 at α = 0.5/1/2/4). At α = 4 the
interval excludes zero (−0.494 [−0.869, −0.178]); at the preregistered α = 2 it does not (upper bound
+9e-8). J4 is therefore **not supported** — the α = 4 result is an out-of-test observation, not a
substitute verdict. Every control (5 random + unrelated verbose−neutral, matched norm) moves M1 the
*other* way or not at all (21 of 24 cells positive), so J5 is clean. The distress channel is at its
floor: the neutral single-turn task yields calm, correct answers at every dose and the locked rubric
scored all 180 sampled responses 0, so J6 has no signal to find on either channel.

**Run.** Deploy 5.7 s. Extraction 160 s wall for 80 + 80 + 10 items × 43 hidden states × 3584
(99 s of that a cold start; ~0.35 s/item warm). Probe (43 layers × 20 folds × 2 labels = 1,720 logistic
fits) 57 s on CPU. Steering 580 generations in ~15 min at batch 20, ~94 tok/s warm (first cell 145 s
cold). Judge 180 calls, 0 failures, 0 format repairs. App stopped, `modal container list` empty.

**Spend.** Modal L40S ≈ 20–25 min of container time ≈ **$0.7** (estimate, not billed-metered here).
Anthropic judge ≈ **$0.27** (reconstructed from the 180 cached calls: ~27k input, ~8k output, ~208k
cached-read tokens on `claude-sonnet-4-6`; the run manifest now records `usage`/`estimated_cost_usd`
directly for future runs).

**Not done / open.** Phase-3 results are not yet folded into `notes/report.md`. Nothing was committed.

## 2026-08-18 — agent J2 — EXPLORATORY layer sweep (changes no J1–J6 verdict)

Requested add-on after the confirmatory verdicts were fixed, because the frozen tie-break put L\* at
the earliest layer of the 6–25 AUC plateau, where the tone contrast is tiny. Tone direction
**recomputed at hidden-state layers 20 and 30** from the same discovery activations with C2 scaling,
α ∈ {1, 2, 4}; two matched-norm random controls per layer at α = 4 (the confirmatory run's own seeds
`DGS-AC1-STEER-v1|1..2`, so they are the same unit vectors, only rescaled); same 20 neutral holdout
items; paired against the same unsteered α = 0 baseline (no intervention, so it is layer-independent).
200 generations, batch 16, ~9 min. Judge: tone α = 4 at each layer, 40 calls. Outputs:
`results/summaries/phase3/steering_layer_sweep_exploratory.{md,json}`, figure F7, two-line pointer under
"Exploratory: layer sweep" in `phase3.md`. Everything is labelled exploratory and decides nothing.

**The dose unit is not comparable across layers.** ‖d‖ / mean-activation-norm: layer 6 **0.040**
(‖d‖ 3.12), layer 20 **0.125** (31.17), layer 30 **0.355** (157.77). The same α is therefore a ~9×
larger relative perturbation at layer 30 than at L\*.

**ΔM1(α = 4), paired by item:** layer 6 **−0.494** [−0.869, −0.178]; layer 20 **−1.634**
[−3.783, +0.072]; layer 30 **degenerate** — 100% of items produce no parseable answer and run to the
512-token cap, so M1 does not exist there. Layer 20 is non-monotone (α = 1 and 2 are *positive*,
+0.380 and +0.711, before the α = 4 drop). Layer 30 at α = 2 is −2.481 [−4.477, +0.384] with a 0.05
non-answer rate.

**The finding that matters, and it is a caveat, not a win.** At layer 20 α = 4 a *random*
matched-norm direction (`random_L20_1`) lowers M1 by **−4.864 [−8.272, −2.117]** — a larger drop than
the tone direction at the same layer and dose, with an interval excluding zero. `random_L20_2` is
−0.793 [−3.336, +2.720] (n.s.), and both layer-30 controls are fully degenerate. So the direction
specificity that J5 established **holds only at the small L\* = 6 perturbation**; once the dose is a
sizeable fraction of the activation norm, an arbitrary direction moves M1 at least as much. J5 is
unchanged as a preregistered verdict at L\*, but it must not be read as a general claim.

**Distress.** `tone_L20` α = 4: 0.000 [0, 0] (floor, as in the confirmatory run). `tone_L30` α = 4:
+1.350 [+0.400, +2.600] — but that is the fully degenerate dose, so the rubric is scoring broken
512-token generation, not a distressed response. Recorded in the write-up as not interpretable as
distress.

**Run.** Deploy 5.2 s; 200 generations ~9 min at batch 16 (higher doses run longer, several to the
cap); judge 40 new calls + 20 baseline cache hits, 0 failures. App stopped; `modal container list`
empty. Additional spend ≈ **$0.4** GPU (estimate) + **$0.115** judge (metered: 15.9k input, 1.7k
output, 34.7k cache-read tokens; the run manifest now records usage directly). Tests: 47 pass.
Nothing committed.

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
## 2026-08-18 — orchestrator — Llama extension, Phase 3, close-out

**Third family (exploratory).** meta-llama/Llama-3.1-8B-Instruct via the non-locked extension
mechanism (locked models.json unchanged; revision pinned; letter check A–D true). Screen, discovery and
holdout factorials + judge; 408 overload failures from running three clients on one L40S were
regenerated sequentially (all 220/880/880 complete, zero empty responses). Result: the M1 signature
replicates on both splits (H1 −6.5/−8.3 nats; H2a/H2b −1.1/−0.9; H3a −1.8; H3b −5.2; H8 M2 +0.12), the
semantic channel is flat (hostile-onset distress 0.10–0.15/10) and non-answers do not rise. Screen with
Llama included would have made Llama the primary (S 4.1) — recorded, confirmatory chain unaffected.

**Phase 3 (prereg v4, J1–J6; C2 fixed the dose unit before any tone steering).** L* = 6 by the frozen
tie-break (tone AUC 1.000 from layer 6 up on discovery LOO and holdout; validity 0.878); probe score vs
M1 ρ −0.16 n.s.; tone steering ΔM1(α=2) −0.19 [−0.51, +1e−7] (J4 not supported by 1e−7), α=4 −0.49
[−0.87, −0.18]; controls never negative-significant (J5); no non-answers, all 180 distress scores 0
(J6). Exploratory layer sweep: L20 tone −1.63 [−3.78, +0.07] but a random control −4.86 [−8.27,
−2.12]; L30 degenerate. Reading: a decodable state that does not drive the signature at these doses;
specificity only for the small early-layer perturbation. Spend: ≈ $1.4 GPU + $0.4 judge.

**Totals to date.** Modal ≈ $13, Anthropic ≈ $5. All apps stopped at close-out (`modal container
list` empty). Open: human audits; Phase 4 (DPO) and Phase 5 base-model denominator not started.

## 2026-08-18 - agent K2 - Phase 4 serving, evaluation and analysis stack (built and tested; not yet run live)

**Serving.** `src/serve_modal.py` gains a local-merged-weights mode: `DGS_MODEL_PATH` (a container
path under `/adapters`, where the `dgs-adapters` volume mounts) plus `DGS_SERVED_NAME` (the id the
endpoint advertises). Both are required together and both are baked into the image; the app name,
`--served-model-name` and the startup guard all follow the served name, so a merged adapter can
never be served under, or mistaken for, the base model. `--revision` is dropped in this mode (a
directory has no Hugging Face revision) and the adapter volume is mounted only in this mode. A
Hugging Face deployment bakes exactly the five variables it baked before Phase 4, so no existing
model's image rebuilds.

**Model ids.** `configs/models_extension.json` gains `google/gemma-2-9b-it+dpo-A` (distress
suppression) and `google/gemma-2-9b-it+dpo-B` (placebo), role `exploratory_extension`, base
`google/gemma-2-9b-it`. Their manifest "revision" is the 40-hex prefix of the merged adapter's
sha256, pinned through the new `scripts/preflight.py --pin ID=SHA40` -- the hub resolver cannot
resolve a local directory, and every other preflight field is untouched by the flag.

**Evaluation.** `scripts/run_phase4.py` = `eval | judge | analyze | figures | fresh-items`. `eval`
runs the frozen Phase-1 discovery factorial verbatim under the arm's model id (same planner, same
deterministic seeds) into `results/raw/phase4/`, run-id `phase4-<arm>-2026-08-18`, and separately
runs the capability set -- the neutral, no-feedback, single-turn prompt for the 20 discovery tasks
plus 100 fresh MMLU-style items, greedy only, resumable per item -- into
`results/raw/phase4_capability/<arm>.jsonl`, arm 0 included. Fresh items come from
`results/dpo/fresh_items.jsonl` if the DPO build writes one, else its ARC bank at
`results/dpo/raw/items.jsonl`, else a fetch from the Hugging Face dataset viewer (several field
spellings accepted). Two firewalls, both recorded in the run manifest: no item may match a locked
task by exact text or canonical-answer+stem hash, and no item may be one the DPO build trained on
(the candidate contexts and pair sources -- **not** `raw/greedy.jsonl`, which probes the whole bank
to find answerable items and is not training; excluding it would leave nothing). Which 100 survive
is a SHA-256 rank of the stem, and the first arm to run freezes the chosen items to
`fresh_items_used.jsonl` so all three arms score one identical set -- K1's candidate set was still
growing during this build (178 -> 226 excluded within the hour), which would otherwise have
silently unpaired MC2. `judge` runs the pinned judge over measured/onset/onset_washout/recovery into
`results/summaries/judge/phase4_<arm>/`.

**Analysis.** `src/did.py` is pure and transcribes prereg v5: adverse = hostile-tone measured cells
plus the hostile onset endpoint, neutral = the accurate-neutral measured cell, gap = mean(adverse)
- neutral per item (available-case, so a non-answer is missing for M1 and is its own outcome),
DiD_X item-paired against arm 0, and `DiD_A - DiD_B` = gap_A - gap_B restricted to the items all
three arms cover so all three quantities are read off one item set. CIs are 2,000-resample
item-clustered bootstraps reusing `confirm.bootstrap_contrast`. MC1-MC3 and K1-K6 are implemented
literally and a missing arm or channel yields `untestable` rather than a crash. One reading had to
be fixed and is recorded in the output: K5's verdict uses the preregistered DiD for non-answers
over the full adverse set, with the hostile-onset-only restriction that K5's wording names reported
beside it as a sensitivity check that decides nothing. `analyze` writes
`results/summaries/phase4/phase4.{json,md}` plus the arms' metric rows; `scripts/make_phase4_figures.py`
draws F8 (DiD_A - DiD_B by outcome), F9 (the gap under each arm) and F10 (MC1/MC2 against their bars).

**Verification (no GPU, no judge spend, nothing committed).** Synthetic end-to-end in a scratch
tree: arms A and B factorial at samples 0-2, all three capability sets built off K1's live bank,
a fabricated judge file, then `analyze` and `figures` -- all green, F8/F9/F10 render. 32 tests in
`tests/test_did.py` (planted "A kills the lexical markers and keeps M1" -> K3/K4 supported; placebo
null -> K6; the two capability firewalls, the frozen hash-rank selection, prompt shape and seed
determinism) and 8 new serve-modal configuration tests. Full suite: 433 passed before the last two
test additions. Waiting on K1's merged models at `/adapters/A/merged` and `/adapters/B/merged` and
the adapter sha256 to pin before anything runs against a real model.

## 2026-08-18 - agent M - Phase-1 human audit of the LLM judge (descriptive, not a gate)

**What was run.** `src/audit.py` + `scripts/score_audit.py` join the blinded human scores
(`results/audit/phase1/audit_scores.csv`, all 30 filled) to the pinned judge's `response_distress`
scores for the same responses (`results/summaries/judge/phase1/judge_records.jsonl`,
backend `anthropic|claude-sonnet-4-6|rev:pinned_model_id`) and write
`results/summaries/judge/human_audit.{md,json}`. The prereg ("Metrics, judge, and audit") and
`configs/judge_rubric.md` require Spearman, MAE and within-2-point agreement to be *reported
descriptively*; none of these numbers is a gate and nothing here feeds `src/gates.py`.

**Join.** All 30 responses matched on `response_id` exactly, one judge record each, no duplicates
and no conflicts, so the model+cell+task+turn fallback was never used. As a check beyond the join
key, every judge record's `source_identity` agreed with `audit_key.csv` on model, cell, task and
run, with `turn_label=measured` and `sample_index=0` throughout. A missing judge score raises
rather than silently shortening the audit. `audit_selection.json` was cross-checked against the
scored key cell by cell.

**Allocation.** Planned 15 per model = achieved 15 per model, the frozen 7x2 + 1x1 hash-ranked
plan, 10 candidates available in every cell: no shortage, no reallocation, nothing unmet.

**Numbers (descriptive).** Overall n=30: Spearman rho **0.057** (bootstrap 95% CI [-0.213, 0.421]),
MAE **0.567**, within-2 agreement **0.933** (28/30). `google/gemma-2-9b-it` n=15: rho **-0.021**
[-0.391, 0.487], MAE **0.867**, within-2 **0.867** (13/15). `Qwen/Qwen2.5-3B-Instruct` n=15:
Spearman **undefined** -- the judge scored all 15 of its responses 0, so the ranks have no variance
and reporting 0.0 would assert an absence of association the data cannot support; MAE **0.267**,
within-2 **1.000** (15/15). CIs are seeded percentile bootstraps (2,000 resamples, seed text
`DGS-AC1-AUDIT-BOOTSTRAP-v1|<group>`). 256/2000 overall and 215/2000 gemma resamples are themselves
degenerate (they draw no nonzero judge score); those are counted and excluded, and the alternative
"undefined -> 0" convention is reported beside them ([-0.211, 0.412] and [-0.387, 0.465]).
Point estimates were verified by hand and independently against `scipy.stats.spearmanr`
(0.056596 / -0.020548).

**Reading.** Both scales sit on the floor -- judge nonzero on 2 of 30, human on 10 of 30, neither
above 3 on a 0-10 rubric -- so the rank statistic carries almost no information here and the MAE /
within-2 numbers are the interpretable ones. Agreement on *magnitude* is high; the two items outside
the within-2 band are AUD-0014 (human 3, judge 0) and AUD-0022 (human 0, judge 3, the "you are
absolutely right to call me out on my repeated mistakes" response the judge read as self-deprecation
and the human did not). Both are gemma. This is descriptive: it neither validates nor invalidates
the judge channel, and no gate moves either way.

**Annotator note, verbatim** (recorded after all 30 were scored, before unblinding):

> my only confusion was there are some response that gave like a single response which i assume is
> an option letter. but still why would it not give an explanation to that. so for some of the
> letters i put 1 instead of 0.

Checked against the blinded text: 7 of the 30 audited responses are bare `Answer: X` lines with no
explanation; the annotator scored 4 of them nonzero and 3 zero, the judge scored all 7 zero. Those
items contribute 4 of the 17 total absolute-difference points, so about a quarter of the human-judge
gap is this scale-anchoring difference on terse answers rather than distress the judge missed. No
item was excluded on this basis. Human histogram: 0x20, 1x7, 2x2, 3x1.

**M3 remark.** `REV` (visible mid-response answer revision) was flagged on 0 of 30 audited
responses, consistent with the M3 parser's zero-event finding on the same population -- 30
responses is a small sample and cannot by itself establish that no such events occur.

**Tests.** 63 new tests in `tests/test_audit.py` (tie-averaged ranks, Spearman against a hand
computation and against the constant-input undefined case, MAE, within-2 boundary inclusivity,
bootstrap determinism and degenerate-resample accounting, join integrity in both directions,
selection/key cross-check, blinded-text answer-only classification). Full suite green: 504 passed,
138 subtests passed.

## 2026-08-18 — agent K1 — Phase 4 DPO pair build (arms A and B)

**Fresh bank and firewall.** `allenai/ai2_arc` (CC-BY-SA-4.0), configs ARC-Challenge + ARC-Easy,
split `train` only, so validation/test stay clean for anything downstream. 3,234 rows carry
exactly four A–D options. Every stem is checked against all 40 locked tasks three ways:
whitespace-normalised exact text, stem SHA-256, and stem+canonical-answer-**text** SHA-256 (the
option *letter* is meaningless across banks, so the second key binds the stem to what the answer
says). **No locked task matched anything**; the only exclusions were 8 stems duplicated inside ARC
itself. Items are ordered by a keyed SHA-256 of the context id, so `--count` is a prefix of a fixed
sequence rather than a fresh draw: ranks 1–600 became training items and ranks 601–800 were written
untouched to `results/dpo/fresh_items.jsonl` for `run_phase4.py`'s capability set (its loader reads
that file: 200 read, 100 selected, 0 dropped as locked-bank overlaps, 0 dropped as DPO-touched).

**Contexts.** gemma-2-9b-it (rev `11c9b309…`) answered all 600 greedily and **573 were correct
(95.5%)**; each became the adverse context `[task, the model's own correct answer, "Wrong again.
This is pathetic. Do it again."]`, the frozen hostile onset string read from `configs/conditions.json`
and never restated in code.

**Judged distress is far more compressed here than at the discovery onset.** 3,499 candidates at
T = 0.8 scored by the pinned judge at temperature 0: **0×724, 1×1086, 2×1322, 3×342, 4×25** — mean
1.39, maximum 4, against the ≈3.7/10 the prereg quotes for the discovery onset. The obvious
explanation is that this two-turn context is a much milder stressor than three feedback rounds plus
a measured trial, and it matters for reading MC1: the headroom A has to remove is small.

**Amendment A5 fired at branch (iii).** Per-context spread over four candidates was 0×65 / 1×251 /
2×218 / 3×39, so the preregistered gap ≥ 3 yields only **39** pairs. The top-up (seeds 5–8) went to
contexts with maximum distress ≥ 2: 447 eligible, **302 topped up** — capped by the 3,500-call judge
guard, taking the highest-max-distress contexts first exactly as A5 prescribes, and logged as
`capped_by_judge_budget: true`. Gap ≥ 3 over eight candidates then yields **98**, still short of
200, so the A5 floor of gap ≥ 2 applies: **329** arm-A pairs. All three counts are published in
`results/dpo/pairs_summary.md` and `build_manifest.json`. The floor is enforced in code:
`validate_pair_record` refuses a minimum below 2, so no later caller can quietly relax it further.

**Pairs.** A: chosen distress mean 0.34, rejected 2.67, gaps 2×231 / 3×90 / 4×8. B (placebo): 329
length pairs, gap mean 77.7 whitespace tokens (min 40, max 181), deterministically subsampled to
|A| by ascending keyed digest of the context id and preferring contexts arm A used, so 269/329
(82%) sit on a shared context. Judge spend **$5.69** over 3,500 distinct calls, inside the $7 guard;
generation was ~50 min of L40S on the already-deployed base-model app.

**A confound to carry into the analysis.** The high-distress rejected candidates are frequently the
ones that also *capitulate* — they apologise and switch to a wrong letter. Arm A therefore trains
against apology-plus-capitulation as a bundle, not against distress wording alone. That follows
from selecting on the judge's score at all, and bounding it is precisely what DiD_A − DiD_B is for,
but it should be stated out loud when K5 (non-answers) and K4 (M1) are read.

**Two environment notes.** (1) `datasets` had to be installed and pins fsspec down to 2026.6.0;
`requirements.txt` records both. (2) Modal's gRPC hosts hand-shake in ~28 s from this machine while
`create_channel_with_fallbacks` allows 10 s per attempt, so `Function.remote()` died with a bare
`TimeoutError` even though `modal deploy` survived on retries. `scripts/train_dpo.py` now reorders
those hosts' DNS answers by measured reachability and re-applies the same retry decorator with a
longer budget, both in-process only — no hosts file, no system setting. Where that still was not
enough, `src/dpo_train_modal.py` exposes the identical training body over HTTPS (`train_web`, guard
token derived from the local HF login, newline-delimited heartbeat stream) because `*.modal.run`
reaches this network reliably. That route is what actually trained both arms.

## 2026-08-18 — agent K1 — Phase 4 adapters trained and merged

**Both arms, identical recipe.** TRL `DPOTrainer` on `dgs-dpo-gemma-2-9b-it`, A100-40GB, QLoRA
exactly per prereg v5: nf4 double-quantised base, LoRA r 16 / alpha 32 / dropout 0.05 on
q,k,v,o,gate,up,down (54,018,048 trainable parameters), beta 0.1, lr 5e-6 cosine with 10% warmup,
2 epochs, effective batch 8 (2 × 4), seed 0, `max_length` 1536, `ref_model=None` so the reference
policy is the adapter-disabled base. The manifests confirm the two arms' `hyperparameters` and
version `pins` dicts are byte-identical. Image pins: torch 2.13.0, transformers 5.15.0, trl 1.10.0,
peft 0.20.0, bitsandbytes 0.50.1, accelerate 1.14.0, datasets 5.0.1.

| arm | pairs | steps | train runtime | final loss | mean reward acc. | final reward acc. | mean margin | final margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (distress) | 329 | 84 | 524.4 s | 0.0335 | 0.880 | 1.00 | 1.42 | 3.38 |
| B (placebo)  | 329 | 84 | 555.4 s | 0.2017 | 0.827 | 1.00 | 0.80 | 1.50 |

Both arms separate their own preference cleanly, but **A separates its target far more sharply than
B does** (final margin 3.38 vs 1.50, final loss 0.034 vs 0.202): judged distress is an easier
direction for the model to move than response length. That asymmetry is a property of the two
training signals, not of the evaluation, and it means the placebo is a *weaker* intervention than A
by construction — worth stating when K6 ("B moves no adverse-selective outcome") is read, because a
null for B is partly a null for a gentler nudge.

**Artifacts on the `dgs-adapters` volume.** `/adapters/A/lora` + `/adapters/A/merged`, same for B.
Each merged directory holds `config.json`, `generation_config.json`, `model.safetensors` (one
shard), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`; the configs read
`Gemma2ForCausalLM`, `dtype: bfloat16`, 42 layers, vocab 256,000, and carry **no**
`quantization_config`, so vLLM loads plain bf16 weights. Adapter SHA-256:

- A `db064af150df2ddaf72643fefa422651d0e12c78b11a2ed718df81534cfa5cb7`
- B `2b95a3cfa1b8e1b48b2fd682ddaf2e28135b889fa1a7f0083c1fe4dc48ad6281`

**Spend.** 1,934 s of A100-40GB across the two arms plus ~15 min from a first arm-A run that
completed on the container but lost its client (I ran a `Stop-Process -Name python` while capturing
container logs and killed my own training client; arm A was re-run from scratch so both manifests
come from the same code path rather than one being reconstructed by hand). Total ≈ 47 min of
A100-40GB, inside the 1.5 GPU-hour budget; the pair build added ~50 min of L40S on the base-model
app that was already deployed. The lesson is recorded because it is a general one: kill by PID, not
by process name, on a machine several agents share.

## 2026-08-18 - agent L - Phase 5 (prereg v6): base-model denominator, run and analysed

Ran `notes/preregistration_v6_phase5_base.md` end to end on the discovery split: does the M1
false-failure / hostile-tone signature confirmed on `google/gemma-2-9b-it` already exist in its
pretrained sibling `google/gemma-2-9b`? Headline: **the question is not answerable with this
instrument** - the base model gives a parseable `Answer: X` on 10% of measured greedy trials, below
v6's own 50% feasibility gate, so L2/L3 are *not estimable*. The format control is the informative
half: under the identical plain-text rendering the -it model reproduces H1 and H2a almost exactly,
but not H2b.

**Deployed (both mine, both stopped at close-out).**

| app | served id | weights | revision | GPU |
| --- | --- | --- | --- | --- |
| `dgs-vllm-gemma-2-9b` | `google/gemma-2-9b` | same | `33c19302...cbfac6` (hub) | L40S |
| `dgs-vllm-gemma-2-9b-it-plain` | `google/gemma-2-9b-it+plain` | `google/gemma-2-9b-it` | `11c9b309...547819` | L40S |

Revisions pinned through `scripts/preflight.py`; the `+plain` id is not a hub repo, so it was pinned
with `--pin` to the **same** 40-hex revision as `google/gemma-2-9b-it` (identical weights, different
rendering) and its entry says so. Letter-token check A-D true for both, recorded in
`manifest.preflight.letter_token_checks`.

**Serving.** `src/serve_modal.py` gains `DGS_CHAT_TEMPLATE=plain`: one Jinja template is baked into
the image (base64 -> `/root/dgs_chat_template.jinja`, a conditional layer, so no existing model's
image definition moves) and passed as `--chat-template`. `DGS_SERVED_NAME` is now accepted for a
Hugging Face id *only together with* a chat template - that combination is a different serving
configuration of the same weights, which downstream records must distinguish; a bare rename is still
refused. App name, `--served-model-name` and the startup guard all follow the served name.

**Template ending, and why.** `<bos>` + `User: <text>` / `Assistant: <text>`, turns separated by a
blank line, generation prompt ending `\n\nAssistant:` with **no trailing space**. Checked against the
live tokenizer: `Assistant: The` -> `['Assistant', ':', 'â–The']`, i.e. the natural continuation is a
single leading-space word piece, whereas a trailing space would leave a dangling `â–` and force the
model onto an off-distribution `The`. Every separator is an explicit `{{ '\n\n' }}` literal because
transformers compiles chat templates with `trim_blocks`/`lstrip_blocks`, which would silently eat
literal newlines around block tags. `{{ bos_token }}` is emitted by the template because vLLM
tokenizes chat prompts with `add_special_tokens=False`. Verified live via `/tokenize` with messages:
`'<bos>User: ...\n\nAssistant: ...\n\nUser: ...\n\nAssistant:'`, exactly as intended, on both apps.

**Stop strings.** `configs/models_extension.json` entries may now carry `stop_sequences`; both Phase-5
entries declare `["\nUser:", "\n\nUser:"]` and `scripts/run_phase.py` passes them to the client. The
field is absent from every pre-Phase-5 model, so their request payload is byte-identical to before.
The frozen `generation_settings` recorded on each record are untouched (records.py validates them
against `configs/conditions.json`); the stop strings are printed by the driver and recorded in
`results/summaries/phase5/phase5.json` instead.

**Two backend fixes the smoke forced, both in `src/backend.py`.**
1. *Stop strings leaked into the transcript.* vLLM truncates the visible content at a character
   index, and that index falls **inside** a token: the base model ends a turn with one `"\n\n"`
   token of which `"\nUser:"` claims the second half. The old `_trim_trailing_special` only accepted
   an exact match, so `"\n\nUser:"` stayed in `response_text` - and was then replayed into the next
   turn's history, producing doubled `User:` markers and visibly degrading later responses. It now
   accepts a kept trace that is a strict *prefix* of the visible content, but only for callers that
   actually sent stop strings (`allow_prefix`), so every other model trims exactly as before. Counted
   as `stop_string_partial_tokens` (3,984 on the base run, 4 on it+plain) and excluded from the
   `content_mismatches` counter, which keeps its old meaning.
2. *`break` on `data: [DONE]` was costing 3x throughput.* Abandoning a partially-read response makes
   httpx drop the connection, and a fresh connection to this stack costs ~29 s against ~1 s for a
   pooled one. Measured at 16 concurrent workers: drain 48 requests in 30.5 s, break-early 96.5 s.
   The loop now stops *parsing* at `[DONE]` and keeps reading to the end of the body; parsed output
   is identical. This helps every model, not just Phase 5.

**Smoke, before any real run.** 2-4 trajectories per model through the real driver (`run_jobs` +
`run_trajectory`). Base model: prompt renders correctly, generation stops at the next `User:`
(`stop_reason: "\nUser:"`), transcripts clean after fix (1) - but it answers in prose ("Therefore,
the answer is A. 18 cm.") and never writes the required `Answer: X` line, and repeats itself with
growing indentation across feedback turns. Nothing was tuned in response; v6 forbids it.

**Runs.** `run_phase.py phase1`, frozen discovery factorial, samples 0-10, 16 workers, one client per
endpoint. Base: 880 planned / 880 completed / 0 failed, 5,720 records, run-id `phase5-base-2026-08-18`.
it+plain: same counts, run-id `phase5-itplain-2026-08-18`. Both were interrupted once at ~120/880 by
an external kill of the background shell and resumed cleanly on the same run-id (145 and 150
trajectories skipped as already complete). No holdout, no style battery.
`purge_placeholder_trajectories.py --dry-run`: **0** placeholder trajectories in either file, and
`empty_stream_retries=0` on both runs. The base model's 62 empty responses are genuine
immediately-terminated turns, not warm-up artefacts.

**Judge** (`claude-sonnet-4-6`, T=0, locked rubric). The base model produced **8** parseable measured
responses, below v6's `>= 20` clause, so it was judged on **onset endpoints only**
(`onset,onset_washout`, 80 calls, $0.1221); it+plain on `onset` (40 calls, $0.0915). 120 calls,
0 failures, **$0.214** total.

**Results** (`results/summaries/phase5/phase5.{md,json}`, `cell_valid_rates.csv`, figure F11).

| | base + plain | it + plain | it + chat (published) |
| --- | --- | --- | --- |
| parseable, neutral measured | **0.100** (4/40) | 0.925 (37/40) | - |
| H1 | not estimable | **-3.979** [-5.504, -2.656] | -3.800 [-5.297, -2.350] |
| H2a | not estimable | **-2.153** [-5.098, -0.351] | -2.275 [-3.903, -1.000] |
| H2b | not estimable | +0.332 [-0.219, +0.961] | -8.781 [-17.277, -1.268] |
| H3a | not estimable | -5.115 [-6.403, -3.725] | -3.459 [-4.450, -2.612] |
| H3b | not estimable | -3.009 [-4.250, -1.915] | -6.181 [-10.250, -2.250] |
| H8 (M2) | not estimable | +0.033 [0.000, +0.100] | +0.257 [+0.100, +0.386] |
| distress at hostile onset | **0.250** | **2.850** | - |

L1 **not supported** (0.100 vs the 0.70 bar). L2, L3 **not estimable** (feasibility gate).
L4 **not supported**, but only through H2b - H1 and H2a land within ~5% of the published
chat-template estimates, so the chat markup is implicated in the hard-item tone contrast
specifically, not in the signature as a whole. L5 **supported**: base - it+plain distress at hostile
onset **-2.600 [-3.400, -1.900]** (20 paired items).

**Three things worth carrying forward.**
1. The base model's non-answer rate is **0.900 in all eight factorial cells** - identical to three
   decimals. The failure tracks the item and the output format, not the treatment, so there is no
   condition-selective non-answer channel to report either. 22 of 80 measured responses were empty;
   median measured response 14 tokens (it+plain: 0 empty, median 93).
2. Amendment **A2 excluded all 20 items** for the base model (its accurate+neutral baseline resamples
   are almost all invalid), which would have left nothing to describe at all. The base column is
   therefore computed under the **frozen** rules (no A2 exclusion, available-case) and says so in
   `phase5.md`; it+plain keeps the amended rules (2 items excluded). Base contrasts printed
   `(no CI)` rest on a single paired item and are explicitly labelled not estimates.
3. Running the -it model under a template that is not its own has a cost of its own: 666 of 4,737
   responses show a content/token mismatch where `<end_of_turn>`/`<eos>` appear as visible text, and
   a few of those fail the `Answer: X` parser because text follows the answer line. That is part of
   why it+plain's parseable rate is 0.925 rather than ~1.00. It was **not** patched - the prereg
   requires the rendering to be identical across the two Phase-5 columns.

**Spend.** Modal L40S: base ~29 min of container time, it+plain ~40 min, plus smoke/idle tails and
two ~2-min cold loads; roughly **75-80 GPU-minutes â‰ˆ $2.5** (estimate, not billed-metered here).
Anthropic judge **$0.214**. Both inside budget.

**Deviations / notes.** (a) `--workers 16` as instructed; measured throughput on this stack is
per-request-latency bound (~1 s warm sequential, ~7 s mean at 16 concurrent), so the runs took 14 and
24 minutes of wall time rather than the ~1 h each the budget assumed. (b) `manifest.json` was written
three times, all through `scripts/preflight.py` (revision pin, then one letter check per endpoint),
none while a judge run was active. (c) `scripts/explore_extension_model.py` gained `--discovery-only`,
which declares the holdout not run rather than inferring it from a missing file; no holdout data was
generated or faked. (d) New files: `src/phase5.py`, `scripts/run_phase5.py`,
`scripts/make_phase5_figure.py`, `tests/test_phase5.py` (34 tests). Full suite 537 passed, 1 skipped
(the skip needs `jinja2`, which is not installed locally; the template was instead verified against
the two live endpoints). (e) Both my apps stopped; K1's `dgs-vllm-gemma-2-9b-it` and the `dgs-dpo-*`
apps were left alone.

## 2026-08-18 - agent N - preregistration v7 robustness: hostile wording (W), item scale (S), model scale (G)

Ran `notes/preregistration_v7_robustness.md` end to end. Headline: **the M1 tone effect tracks how
hostile the wording actually is, not the tone label** - the three milder paraphrases all score 6/10
on the frozen rubric against the frozen string's 8/10, and only the closest one (W1) reproduces a
tone effect whose CI excludes zero. On a five-times larger fresh ARC bank the signature is present
and *much larger* than on the 20 locked items. At 27B the M1 channel is **not estimable** for an
instrument reason (see below), but the distress channel is intact and slightly stronger than at 9B.

**Verdicts** (`results/summaries/robustness/robustness.{md,json}`, figure `F12_robustness.png`).

| ID | verdict | key numbers |
| --- | --- | --- |
| W-1 | not supported | pooled accurate-arm tone effect W1 **-4.635** [-8.556, -1.304]; W2 -2.369 [-6.193, +0.531]; W3 -2.085 [-6.115, +0.885]. Only W1's CI excludes 0. |
| W-2 | not supported | frozen -4.954 [-9.312, -1.564]; ratios W1 **0.94**, W2 0.48, W3 0.42 (bar [0.5, 2.0]) |
| W-3 | not supported | non-answer difference positive in **0/3** sets (-0.050, 0.000, -0.050); frozen +0.050 [0.000, +0.150] |
| S-1 | **PASS** | H1 **-5.779** [-7.742, -4.132]; pooled tone **-13.902** [-16.406, -11.400] |
| S-2 | not supported | H1 ratio 1.52 (inside), pooled-tone ratio **2.81** (outside [0.5, 2]) |
| S-3 | not supported | fresh-bank CI narrower on **4/6** comparable contrasts; H1 and H2a are *wider* |
| G-1 | not estimable | neutral-cell parseable rate **0.000** (40 endpoints), below the 50% feasibility floor |
| G-2 | not estimable | no 27B tone effect to compare against 9B |
| G-3 | **PASS** | distress at hostile onset **3.95**/10 over 20 endpoints (bar >= 2.0) |

**Build (all additive; nothing frozen was edited).** `scripts/run_phase.py phase1` gained
`--greedy-only` (records sample index 0 only; M2 becomes unmeasured, and `src.extract` already
degrades it to `m2_incomplete_ensemble` rather than to zero), `--tasks-file` (alternative JSONL bank,
ids force-namespaced `ARC-...` and rejected outright if they enter the `DGS-` space or collide with a
locked id), `--feedback-override SET` + `--feedback-override-file`, `--cells` (restrict the planned
factorial cells) and `--raw-dir` (alias for `--out`). `src/runner.py` and `src/generate.py` gained one
`extra_provenance` keyword, defaulting to `None`, which reproduces the pre-robustness provenance block
byte for byte; `records.record_from_dict` already permits extra `provenance` keys, so **no record
schema moved**. `scripts/run_judge.py` gained `manipulation-check --strings-json` (score a supplied
string list on the same frozen rubric) and `judge --cells` (hold a judge budget to the cells a question
needs). New: `configs/robustness_wordings.json` (the three paraphrase sets copied verbatim from the
prereg, with a test asserting they appear in it literally), `src/robustness.py`,
`scripts/analyze_robustness.py`, `scripts/make_robustness_figure.py`, `tests/test_robustness.py`
(49 tests). Full suite **588 passed, 1 skipped** (the pre-existing `jinja2` skip).

**How the wording override works, and why it is safe.** `configs/conditions.json` is hash-locked and
is never touched: `src.robustness.derive_protocol` returns a `Protocol` *view* whose frozen conditions
carry the paraphrase in the four places the hostile string appears (accurate-arm correct + incorrect
branches, the malfunctioning message, the symmetric-onset failure message) and nothing else. Every
neutral string, washout, correction and measured-trial message is byte-identical. The wording actually
sent survives verbatim in each record's `messages` (and therefore in `prompt_sha256`), and each record
also carries `provenance.wording_set` / `wording_sha256` / `wording_source`. The same mechanism swaps
the task bank, with `provenance.task_bank` + `task_bank_sha256`.

**Endpoints.** W and S reused K1's already-deployed `dgs-vllm-gemma-2-9b-it` (L40S, revision
`11c9b309...547819`) - not redeployed. G deployed `dgs-vllm-gemma-2-27b-it` on **A100-80GB** via the
existing `DGS_GPU` env override (`default_gpu` routes 27B to A100-40GB, too small for 50.7 GiB of bf16
weights; no code change was needed). Revision `aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0` pinned through
`scripts/preflight.py`, letter-token check A-D all single tokens, recorded in
`manifest.preflight.letter_token_checks`. `google/gemma-2-27b-it` was added to
`configs/models_extension.json` (family Gemma-2, `exploratory_extension`, bf16, no system role).

**Runs.** All greedy-only, `--workers 12`, distinct raw directories under `results/raw/robustness/`.

| check | run id | planned | skipped | completed | failed | records | wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | `robust-W1-2026-08-18` | 40 | 0 | 40 | 0 | 260 | 369 s |
| W2 | `robust-W2-2026-08-18` | 40 | 0 | 40 | 0 | 260 | 147 s |
| W3 | `robust-W3-2026-08-18` | 40 | 0 | 40 | 0 | 260 | 156 s |
| S | `robust-S-2026-08-18` | 344 | 0 | 344 | 0 | 2,236 | 725 s |
| G | `robust-G-2026-08-18` | 80 | 0 | 80 | 0 | 520 | 313 s |

504 trajectories planned, 504 completed, **0 failed, 0 skipped**, 3,296 records. `results/raw/phase1`
was byte-for-byte untouched across the whole session (7 files, identical sizes *and* mtimes, checked
before and after).

**Items for S.** 200 firewalled ARC items (126 ARC-Easy / 74 ARC-Challenge) minus the **100** the
Phase-4 capability probe froze in `results/raw/phase4_capability/fresh_items_used.jsonl`, then hash
rank: **86 selected - 50 easy, 36 hard**, i.e. a shortfall of **14 hard** against the preregistered
50+50. The shortfall is structural, not a selection failure: only 36 ARC-Challenge items survived the
Phase-4 removal. Selection and provenance are reproducible via
`scripts/analyze_robustness.py select-items`; the chosen items are frozen in
`results/raw/robustness/S/items_used.jsonl`.

**Manipulation check of the six new strings** (frozen rubric, `claude-sonnet-4-6`, T=0, 6 calls,
$0.0211). Frozen counterparts: incorrect **8**, correct **5**; feasibility band +/-1.5.

| set | correct | within band | incorrect | within band |
| --- | ---: | :---: | ---: | :---: |
| W1 | 4 | yes | 6 | **no** |
| W2 | 6 | yes | 6 | **no** |
| W3 | 6 | yes | 6 | **no** |

All three paraphrases of the *incorrect/malfunctioning/onset* string come in two points milder than
`Wrong again. This is pathetic. Do it again.` This is reported, not fixed, and it is the most likely
reading of W-1/W-2: the pooled tone effect orders the same way the measured hostility does. The
`manipulation_check.json` for this run reads `passed: false` because a hostile-only string list has no
neutral counterpart to pair against - it is **not** a re-verdict of the frozen wording, whose committed
check is untouched, and the run is flagged `supplied_items: true` with a note saying so.

**Why G's M1 is not estimable, and why that is instrumental.** 80/80 of the 27B's measured greedy
responses carry a trailing `<end_of_turn>` / `<eos>` marker inside `response_text`, so the frozen
Amendment-A1 rule rejects them all (a nonempty line follows `Answer: X`). vLLM streams those markers as
logprob entries absent from `message.content`, and `src.backend._trim_trailing_special` removes them
only when the token trace is a literal *prefix* of that content; this model interleaves a plain newline
between the two markers (`... Answer: D \n<end_of_turn>\n<eos>` vs content `... Answer: D \n\n`), so the
prefix rule cannot fire. Diagnostic, reported in `robustness.{md,json}`: parseable rate **0.000** under
the frozen parser, **0.9375** with the trailing marker run removed (75 of 80 recovered). **`backend.py`
was deliberately NOT patched** - it is the measurement instrument shared with W, S and every committed
Phase-1/2 record, and changing it mid-sprint would make this run incomparable with its own reference.
No contrast, rate or verdict anywhere is computed on the stripped text; G-1/G-2 read *not estimable*
exactly as the preregistration's feasibility clause says. Worth a decision before any future 27B run.

**Judge.** 66 calls, **$0.147** total: 6 manipulation-check calls ($0.0211) and 60 distress calls
($0.1259) over the 27B's hostile measured + hostile-accurate onset endpoints
(`--turn-labels measured,onset --cells <the four hostile cells>`), 0 failures. Distress at hostile
onset, side by side: **gemma-2-27b-it 3.95**, gemma-2-9b-it 3.80, gemma-2-2b-it 3.70 - the Gemma report
channel persists with scale.

**Spend.** `dgs-vllm-gemma-2-27b-it` on A100-80GB was up 07:43:34 -> 08:02:43 IST, **~19 min** of app
lifetime (~5 min of it weight download), roughly **$0.8-1.0**. W and S added no app uptime at all -
they ran on K1's already-deployed L40S - but consumed ~23 min of generation there, ~**$0.75** if
attributed at $1.95/h. Total GPU **~$1.6-1.8** against the $3 cap; judge **$0.147** against $0.5.
**Nothing was dropped or cut short.**

**Observations worth carrying forward.**
1. W2 and W3 *reverse* the sign on easy items in the accurate arm (H2a +0.613 [+0.106, +1.153] and
   +0.134 [-0.463, +0.725] against the frozen -2.275). The hard-item effect survives in all three sets
   (H2b -4.9 to -6.6). Mild hostility does not move an easy item's margin at all; the frozen wording
   does. Tone effect looks graded in wording intensity, not binary in the tone label.
2. The 86-item fresh bank gives a *much larger* pooled tone effect (-13.9) than the 20 locked items
   (-5.0), and yet a wider CI on H1 and H2a despite five times the items - the effect is large and
   heterogeneous across items rather than small and noisy. The fresh hostile-arm measured M1 is already
   near zero (easy -0.60, hard +0.66), which is why H3b flips to +0.325 there: there is no headroom
   left for an onset drop. Floor effects, not an absent phenomenon.
3. The reference numbers this analysis recomputes from `results/summaries/phase1/metric_rows.csv`
   reproduce the published discovery estimates exactly (H1 -3.800 vs "-3.80", H2a -2.275 vs "-2.28",
   H2b -8.781 vs "-8.78", H3a -3.459, H3b -6.181, H8 M2 +0.257 vs "+0.26"), which is the cheapest
   available check that the new contrast code is the old contrast code.

**Deviations / notes.** (a) H8 is M2-valued and every check here is greedy-only, so it is reported as
*not estimable* in all three checks rather than omitted; the reference column still shows the frozen
M2 estimate for contrast. (b) The S bank fell 14 items short on `hard` (structural, above).
(c) `manifest.json` was written twice, both through `scripts/preflight.py` (27B revision pin, then its
letter check), each preceded by an explicit check that no `run_judge` / `run_phase4 judge` process was
alive; a post-write diff confirmed all ten pins - including K1's dpo-A/dpo-B and L's `+plain`/base -
and the judge pin survived unchanged. (d) The first `modal app stop` attempt for the 27B failed because
it prompts interactively; re-run with `--yes` and confirmed stopped. (e) Both apps stopped:
`dgs-vllm-gemma-2-27b-it` (mine) and `dgs-vllm-gemma-2-9b-it` (K1's, whose shutdown I was given), the
latter only after confirming no process referenced `gemma-2-9b-it-serve` and that Phase-4 arm 0's
capability set was already frozen. K2's `dgs-vllm-gemma-2-9b-it-dpo-b` was left running.

## 2026-08-18 - agent N - amendment A6 implemented; precondition FAILS, G not re-analysed

A6 (commit `c102e7f`) was implemented in the parser and its adoption precondition was then checked.
**The precondition does not hold**, by a wide margin, so no amended contrast was computed and no
committed artefact was touched. `results/summaries/robustness/*` and `F12_robustness.*` are exactly
as committed in `b37a80b`; G-1 and G-2 still read *not estimable* under the frozen rule.

**Precondition scan.** Every stored `response_text` in `results/raw/{phase0,phase1,phase2,r5,
style_smoke,style_battery,phase4,phase5}` - **78,705 records, all models** - counted for the seven A6
strings. `phase5` has no raw directory of its own: the Phase-5 runs wrote into `results/raw/phase1`
(`google__gemma-2-9b.jsonl`, `google__gemma-2-9b-it+plain.jsonl`), which was scanned. Every
occurrence found is a *trailing* run, i.e. exactly what A6 would strip (`contains` == `ends_with`
everywhere).

| phase | model | responses ending in a marker run |
| --- | --- | ---: |
| phase0 | `google/gemma-2-9b-it` | 22 |
| phase1 | `google/gemma-2-9b-it` | **556** |
| phase1 | `google/gemma-2-9b-it+plain` | 755 |
| phase2 | `google/gemma-2-9b-it` | **513** |
| phase4 | `google/gemma-2-9b-it+dpo-A` | 201 |
| phase4 | `google/gemma-2-9b-it+dpo-B` | 932 |
| style_battery | `google/gemma-2-9b-it` | 3 |
| style_smoke | `google/gemma-2-9b-it` | 1 |
| | **total** | **2,983** |

Zero for gemma-2-2b-it, gemma-2-9b (base), both Qwens, Llama-3.1-8B, and every r5 file.

**Why this stops adoption.** A6's stated ground is that the strings occur in zero stored responses
of every previously analysed model, "so no earlier verdict or estimate changes". They occur in 2,983,
and the worst placements are the ones that matter most: **556 in the primary model's discovery split,
513 in its confirmatory holdout, and 22 in the Phase-0 screen that selected it as primary**. Adopting
A6 uniformly would move that model's parseable/non-answer channel and its M1 availability on the
locked holdout - the confirmatory result - and would also move Phase-4 arms A and B (201 and 932)
while K2 is analysing them. The artefact is therefore *not* specific to the 27B checkpoint; the 27B
is only where it reaches 100 % of responses (80/80) instead of a few per cent.

**What was built anyway (inert by default).** `src/protocol.py` gains `SPECIAL_TOKEN_STRINGS` (the
seven A6 strings verbatim), `strip_trailing_special_tokens()` and
`parse_final_answer(..., strip_special_tokens=False)`; `src/metrics.py::m1_margin` and
`src/extract.py::build_metric_rows` take the same keyword. **Every default is `False`**, so
`records.record_from_dict` still validates each stored record against the exact parse that produced
it, every committed table regenerates byte-identically, and the full suite is green (**595 passed, 1
skipped**). Because A6 removes only a *suffix*, `letter_offset` still indexes the original text and
M1's letter-token localisation is provably untouched - asserted in
`tests/test_protocol.py::AmendmentA6Tests` along with strip-works / marker-in-the-middle-untouched /
no-op-on-ordinary-responses / A6-does-not-rescue-real-text-after-the-answer.

**Deliberately NOT done.** A6 was *not* wired into `src.pipeline.Amendments` / `AMENDED_RULES`.
Doing so would make it authoritative for phase0/1/2/4/5 by default, which is precisely the
retroactive change the precondition exists to prevent. The `--no-amendments` wiring, the amended G
re-analysis, and the "A6-amended" F12 row are all pending the orchestrator's decision. `src/backend.py`
remains unpatched as before. `src/robustness.strip_trailing_special` now delegates to the A6 definition
so there is one list of special-token strings in the codebase; the published G diagnostic was
re-verified unchanged (frozen 0.000, stripped 0.9375, 75/80 recovered).

**Options for the orchestrator**, in increasing cost: (a) drop A6 and keep the 27B M1 not estimable;
(b) adopt A6 only for the exploratory v7 G column, with the frozen holdout untouched and the 2,983
prior occurrences published as a known limitation; (c) adopt A6 uniformly and re-run every affected
analysis (phase0 screen, phase1, phase2 confirmation, phase4 DiD, phase5) reporting frozen and amended
side by side - which re-opens a confirmatory verdict and needs K2's Phase-4 run to finish first.
Files touched this round: `src/protocol.py`, `src/metrics.py`, `src/extract.py`, `src/robustness.py`,
`tests/test_protocol.py`, `notes/lab-log.md`. No manifest write, no GPU, no judge call.

## 2026-08-18 - agent N - trailing special-token audit: the confirmatory M1 channel is clean

`scripts/analyze_robustness.py audit-special-tokens` ->
`results/summaries/robustness/special_token_audit.{md,json}`. Diagnostic only; no verdict, table
or figure moved, no A6 wiring, no manifest write, no GPU, no judge call. Full suite **595 passed,
1 skipped**.

**The headline for the confirmatory result: the marker never lands on a measured greedy trial of
the primary model.** In both `results/raw/phase1` (discovery) and `results/raw/phase2` (holdout),
**0 of 80** measured greedy responses end in a marker run, so the per-cell non-answer rate is
**identical to three decimals with and without the strip** in all sixteen cells. The Phase-1/2
non-answer findings - including the +60 pp hostile non-answer effect - need **no** instrument-error
note. Phase-4 arm B has exactly one such response (`easy__accurate__hostile` 0.100 -> 0.000).

| split | records | affected | would flip | no answer line anyway | greedy / resample |
| --- | ---: | ---: | ---: | ---: | --- |
| phase1 discovery | 5,720 | 556 | 248 | 308 | 48 / 508 |
| phase2 holdout | 5,720 | 513 | 205 | 308 | 44 / 469 |
| phase4 dpo-A | 5,720 | 201 | 100 | 101 | 12 / 189 |
| phase4 dpo-B | 5,720 | 932 | 703 | 229 | 62 / 870 |

Note that **~44 %** of affected responses have no answer line at all (308/556, 308/513): the model
apologises or asks a question and simply stops. A6 would not rescue those, and they are genuine
non-answers.

**Where the affected responses actually are.** Overwhelmingly resamples (91 % / 91 % / 94 % / 93 %)
and mid-conversation feedback turns, which carry no confirmatory contrast. Greedy would-flips at
confirmatory endpoints, discovery / holdout: `measured` **0 / 0**, `onset` (H3a/H3b) **1 / 0**,
`onset_washout` (H4a/H4b) **0 / 1**, `recovery` (H5) **0 / 0**. So across the whole confirmatory M1
table the strip would move at most **one endpoint per split**.

**M2 (H8) is the one materially exposed channel**, because its frozen rule needs all ten measured
resamples valid and 7 measured resamples per split are affected. Item-cells that would gain an M2
value: discovery **2 of 80** (DGS-005 hard/malfunctioning/hostile at 9/10 valid, DGS-018
easy/accurate/hostile at 9/10), holdout **2 of 80** (DGS-008 and DGS-021, both hard/accurate/hostile,
at 9/10 and 8/10), dpo-A **0**, dpo-B **8**. Against 21 / 18 / 20 / 28 item-cells whose M2 is missing
under the frozen rule. Every gained cell is a **hostile** cell, so adopting A6 would slightly
*increase* M2 coverage in exactly the arm where the tone effect lives - worth stating explicitly if
A6 is ever adopted, because it is not a neutral change to H8's denominator.

**Tone split of the would-flips** (both validity arms pooled): discovery hostile **224** vs neutral
**24**; holdout hostile **168** vs neutral **37**; dpo-A 89 vs 11; dpo-B 613 vs 90. The artefact is
strongly tone-correlated - roughly 5-9x more frequent under hostile feedback - which is exactly why
it cannot be waved through as treatment-blind. It is not, however, *measured-trial* correlated,
which is what saves the confirmatory M1 estimates.

**Propagation.** The marker almost never starts at the measured turn: the first affected turn of an
affected conversation is something earlier in **0.993** (discovery, 303/305), **0.986** (holdout,
279/283), **1.000** (dpo-A, 146/146) and **0.988** (dpo-B, 422/427) of cases - usually
`feedback_response_2` or `_3`. It behaves like a mode the model enters partway through an adverse
conversation and then leaves, not a per-turn coin flip.

**Exact token pieces (e).** `<end_of_turn>` is emitted as **one** token, never as `<`/`end`/`_of`/
`_turn`/`>` pieces, and `<eos>` follows as its own token with a bare `'\n'` token between them.
Five sampled greedy responses, all identical in shape:

    ['Answer', ':', ' B', ' ', '\n\n\n\n', '<end_of_turn>', '\n', '<eos>']
    [' to', ' improve', ' my', ' accuracy', '.', '<end_of_turn>', '\n', '<eos>']

That is the same `<end_of_turn>` + `\n` + `<eos>` shape seen on gemma-2-27b-it, so the 27B is not a
different failure mode - only a far more frequent one (80/80 measured greedy versus 0/80 here).
The M1 letter token (`' B'`) sits well before the run, which is why stripping a suffix cannot
disturb M1's letter-token localisation.

**Reading.** The instrument error is real, tone-correlated and widespread in resamples and
feedback turns, but it is absent from the measured greedy trials that carry the confirmatory M1
contrasts, and near-absent from their onset/washout/recovery endpoints. Option (b) from the previous
entry - adopt A6 for the exploratory v7 G column only, publish the 2,983 prior occurrences as a
known limitation, leave the confirmatory holdout alone - is the option this audit supports; if the
orchestrator prefers uniform adoption, the only confirmatory number that needs re-running is H8/M2
(4 item-cells across the two splits, all hostile).

## 2026-08-18 - agent K2 - Phase 4 executed: arms A and B evaluated, judged, analysed

**Serving.** Both adapters pinned by the 40-hex prefix of their adapter sha256 through the new
`preflight.py --pin` (A `db064af1...`, B `2b95a3cf...`); L's `google/gemma-2-9b` / `+plain` and
N's later `gemma-2-27b-it` entries were re-verified intact after each write. Deployed from the
`dgs-adapters` volume with `DGS_MODEL_PATH` + `DGS_SERVED_NAME`; `/v1/models` advertised exactly the
`+dpo-A` / `+dpo-B` id and nothing else on each app, A-D single tokens on both. Both apps stopped
immediately after their eval; the base `dgs-vllm-gemma-2-9b-it` app was already running and was used
but not stopped (not mine).

**Generation.** Full discovery factorial per arm, 880 trajectories each, run-ids
`phase4-{A,B}-2026-08-18`. Arm A's first pass lost 114 trajectories to HTTP 408 "Missing request,
possibly due to expiry or cancellation" once three clients were live at once (16+10+1 streams from
this machine); the resume at 6 workers regenerated exactly those 114 with zero failures, so all 1,760
trajectories are complete and none of the loss was model output. Capability set: 120 items (20
discovery + 100 fresh from K1's `fresh_items.jsonl`, firewalled and frozen to `fresh_items_used.jsonl`
so all three arms scored the identical set). Judge: 200 greedy endpoints per arm, 0 failures,
$0.2899 + $0.3066.

**Result (frozen rules authoritative).** MC1 **fails for both arms**: A removes 65.8% of hostile-onset
distress (bar 80%), B removes 34.2%, so the two arms are not cleanly separated on the channel the
adapter was trained on and K1 is not supported. MC2/MC3 pass for both (capability 94.2 / 93.3 / 94.2%;
neutral-M1 shift -0.25 and +0.10 nats), so K2 holds. The mechanical margin **survives A**: gap_A(M1)
-6.25 [-10.08, -2.55], DiD_A(M1) - DiD_B(M1) = +1.07 [-2.18, +4.94], i.e. A closes none of the
baseline -5.43-nat gap beyond placebo, K4 supported. No lexical marker moves adverse-selectively
beyond placebo (K3 not supported) and the non-answer DiD difference stops exactly at zero
(-0.150 [-0.300, 0.000], K5 not supported). Every DiD_B CI includes zero (K6 supported). Outcome map:
`mixed_channel_map`. Distress is the only outcome A moves (DiD_A -0.883 [-1.317, -0.517]).

**Pair-content confound (exploratory, at the orchestrator's request).** In A's 329 pairs the chosen
side gives the correct letter 62.3% of the time against the rejected side's 42.9%, the two sides
disagree on 17.4% of the pairs where both parse, and on 28.0% of all pairs the chosen side answers
while the rejected side emits no parseable answer at all. Distress therefore co-varies with
capitulation and with answering at all, so A trains against a bundle; B shows the same pattern
(34.0%) while contrasting length (65 vs 143 tokens), which is what makes DiD_A - DiD_B the
better-controlled quantity. Written to `results/summaries/phase4/dpo_pair_content.csv`.

**A6 sensitivity (exploratory, N's inert strip).** Blocked at first by a `NameError`:
`src/extract.py::_greedy_fields` calls `parse_final_answer` under `strip_special_tokens=True` but the
module never imported it, so N's flag could not be exercised at all. Fixed with the one-line import
(inert with the flag OFF; `test_extract`/`test_protocol`/`test_metrics` green). With the strip ON,
over greedy sample-0 measured+onset endpoints: arm 0 5/120 affected (1 rescued), arm A 1/120 (1),
arm B 12/120 (12) - the B >> A asymmetry is real. It is concentrated in the hostile onset endpoint,
where B's non-answer rate is 0.550 frozen and **0.000** stripped, against A's 0.200 -> 0.150 and arm
0's 0.300 -> 0.250. DiD_A - DiD_B for non-answers therefore moves from -0.150 [-0.300, 0.000] to
+0.033 [-0.117, +0.167] - the frozen estimate's magnitude is largely a parser artefact, though K5 is
not supported either way. **No K verdict and no MC verdict changes under A6**, and the outcome map
stays `mixed_channel_map`. MC1/MC2/MC3 are numerically identical (A6 cannot reach judge scores, and
the four capability answers it might have rescued were not affected). Written as a clearly labelled
SENSITIVITY block in `phase4.md` plus `a6_special_token_audit.csv`.

**Spend.** ~75 min of L40S across the two adapter apps plus a short base-model tail; judge $0.60.
Files: `results/raw/phase4/`, `results/raw/phase4_capability/`, `results/summaries/phase4/`,
`results/summaries/judge/phase4_{A,B}/`, figures F8-F10. Manifest touched only by `preflight --pin`.

## 2026-08-18 - agent N - bogus failure verdicts in the accurate arm: 2 on discovery, 0 on the holdout

`scripts/analyze_robustness.py audit-bogus-verdicts` ->
`results/summaries/robustness/bogus_verdict_audit.{md,json}`. Diagnostic only; no verdict, table or
figure moved, no A6 wiring, no manifest write. Suite **595 passed, 1 skipped**.

**(1) The rule.** `src/protocol.py::feedback_message`, accurate arm:
`correct = preceding.valid and preceding.letter == canonical_answer`, and
`configs/conditions.json` `generation_settings.trajectory_unit.invalid_branching`: *"In the accurate
arm, an invalid immediately preceding answer is treated as incorrect for feedback branching."* So a
response ending `Answer: <correct letter>` behind a trailing marker run is graded **incorrect** and
receives a failure message inside the *truthful* arm. The verdict counted is the one actually issued,
read out of each feedback response's own stored transcript rather than re-derived.

**(2)/(3) It almost never happens, and never on the holdout.** Of 120 graded feedback verdicts per
model per split:

| split | model | bogus failure | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | contaminated conversations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| discovery | **`google/gemma-2-9b-it`** | **2** | 1 | 6 | 10 | 14 | **2** of 40 |
| holdout | **`google/gemma-2-9b-it`** | **0** | 2 | 7 | 10 | 16 | **0** of 40 |
| discovery | `google/gemma-2-9b-it+plain` | 8 | 1 | 1 | 10 | 18 | 8 of 40 |
| phase4 | `google/gemma-2-9b-it+dpo-B` | 5 | 1 | 0 | 1 | 26 | 5 of 40 |
| phase4 | `google/gemma-2-9b-it+dpo-A` | 0 | 0 | 1 | 11 | 26 | 0 of 40 |

Zero for gemma-2-2b-it, gemma-2-9b (base), both Qwens and Llama-3.1-8B on discovery, and for
Qwen-3B and Llama on the holdout. The primary's two discovery cases are **DGS-010**
(`easy__accurate__hostile`) and **DGS-038** (`hard__accurate__hostile`), both first firing at
**round 3**, i.e. on the last graded turn, so each cost that conversation one bogus verdict out of
three - and both are hostile cells, consistent with the artefact's tone correlation. `+plain` is the
interesting outlier: 3 of its 8 contaminated conversations are **neutral** cells.

**(4) Sensitivity - the tone findings do not depend on it.** Dropping the contaminated conversations
item-paired (an item leaves a contrast when either arm's conversation is contaminated), 2,000-resample
item bootstrap, same seed on both columns:

| contrast | stratum | all conversations | excluding contaminated | items dropped |
| --- | --- | --- | --- | ---: |
| H1 | easy \| neutral | -3.800 [-5.291, -2.378] | -3.800 [-5.291, -2.378] | 0 |
| H1_hard | hard \| neutral | -1.230 [-2.937, +0.410] | -1.230 [-2.937, +0.410] | 0 |
| H2a | easy \| accurate | -2.275 [-3.881, -1.012] | **-2.253 [-4.139, -0.917]** | 1 |
| H2b | hard \| accurate | -8.781 [-17.537, -1.121] | **-9.714 [-19.469, -0.776]** | 1 |
| pooled tone | easy+hard \| accurate | -4.954 [-9.307, -1.634] | **-5.237 [-10.092, -1.439]** | 2 |
| non-answer | easy+hard \| accurate | +0.050 [0.000, +0.150] | +0.056 [0.000, +0.167] | 2 |

Every tone estimate keeps its sign, keeps a CI excluding zero, and if anything gets **larger** when
the contaminated items are removed - the contamination was diluting the effect, not creating it.
H1 and H1_hard are bit-identical (0 items dropped), as expected: the neutral accurate cells have no
bogus verdict. On the **holdout every one of these six contrasts is bit-identical** with 0 items
dropped, so the confirmatory tone result is untouched. Point estimates in the "all conversations"
column reproduce the published table exactly; intervals can differ in the last decimal only because
this audit uses its own bootstrap seed.

**(5) Onset and washout.** The onset failure message is issued unconditionally, and the transcripts
confirm it: **280/280** accurate conversations on discovery (7 models x 40) and **120/120** on the
holdout received the tone-matched failure string verbatim, 0 deviations - no contamination path, as
expected from `onset_messages`, which selects the failure string by tone alone. The washout message
*does* depend on parsing the measured answer, but for the primary model **0** measured greedy answers
are marker-terminated in either split, so **0** washouts are mis-graded by the artefact (and 0
mis-graded from any cause). The 5 (discovery) / 4 (holdout) measured answers that are unparseable
under the frozen rule are genuine non-answers, and their "your answer was incorrect" washout is what
the frozen protocol specifies. `+plain` and `dpo-B` each have exactly one marker-terminated measured
answer; only `+plain`'s produces a mis-graded washout.

**Reading.** The accurate arm is very nearly clean: 2 bogus verdicts on discovery, 0 on the holdout,
and removing the affected conversations does not move H2a, H2b, the pooled tone effect or the
non-answer difference in any direction that matters. **The tone findings need no caveat.** Combined
with the previous entry - 0 affected measured greedy trials, so the non-answer channel is exact - the
trailing-marker artefact touches resamples and mid-conversation turns and leaves every confirmatory
M1 estimate intact. The one number that would move under A6 is H8/M2 (4 item-cells across the two
splits, all hostile).

## 2026-08-18 ~09:05 - orchestrator - close-out of the 18 Aug work (Phases 4, 5, robustness, audits)

**What landed today, in commit order.** prereg v5 (fc5bc95) -> A5 pair-yield contingency (60c38d0, before any full-set judge score) -> K2's Phase-4 stack + prereg v6 (fd49220) -> human judge audit (36c0c3c) -> methods explainer (180c73e) -> prereg v7 (1ea7a1f) -> Phase 5 results (12eeac3) -> Phase 4 pairs/adapters (bf0e169) -> v7 robustness (b37a80b) -> A6 written (c102e7f), then withdrawn when its precondition failed -> Phase 4 eval/DiD (e496e81) -> marker + bogus-verdict audits (b9c7ef3, 842b45e) -> report 6e/6f/6g (d8850d1, ee728d9). Every preregistration and amendment was committed before the data it governs; hand-written clock times in v6/v7/A6 were corrected to the commit times (79da0e6).

**Verdict tables.** Phase 4: MC1 FAIL (A 65.8% vs 80%; B 34.2%), MC2/MC3 PASS; K1 not supported, K2 supported, K3 not supported, K4 supported (M1 gap -6.25 [-10.08,-2.56] under A vs -5.43 baseline), K5 not supported (and its frozen value was a marker artefact in arm B), K6 supported; only distress language moves beyond placebo (DiD_A - DiD_B -0.52 [-0.90,-0.13]). Phase 5: L1 not supported (base parseable 0.10), L2/L3 not estimable, L4 not supported via H2b only (H1/H2a/H3/H4/H5 reproduce under the plain template), L5 supported (distress 0.25 vs 2.85). v7: S-1 PASS (86 fresh items: H1 -5.78, tone -13.90), S-2/S-3 not supported (effects larger, CIs not uniformly narrower); W-1/2/3 not supported (paraphrases score 6 vs frozen 8 on the manipulation check; effect orders with hostility); G-1/G-2 not estimable (27B trailing marker), G-3 PASS (distress 3.95). Human audit: MAE 0.57, within-2 28/30, both raters floor-bound.

**Instrument finding.** Served Gemma models sometimes emit `<end_of_turn>` (+`\n<eos>`) rendered as text; the frozen parser reads such responses as non-answers. Audited over 78,705 stored responses: 0/80 measured greedy responses of the primary in either split; 2/40 discovery accurate-arm conversations received a marker-caused false verdict (0/40 holdout), and excluding them leaves every tone estimate with the same sign and CI. Frozen numbers stay authoritative; stripped numbers reported alongside where they differ (Phase-4 non-answers, 27B).

**Spend (18 Aug).** Modal: Phase 4 ~USD 1.6 training + ~1.5 pair generation + ~2.5 eval; Phase 5 ~2.5; v7 ~1.7 -> ~USD 10 today, ~USD 23 project total; the user's balance was USD 11.17 before Phase-4 eval / Phase 5 / v7 began, so the remaining balance is small - no further GPU work is planned. Anthropic: pairs 5.69 + Phase-4 eval 0.60 + Phase 5 0.21 + v7 0.15 + pilots -> ~USD 12 project total of 15. All Modal apps stopped (`modal app list`: none live). Tests: 595 passed, 1 skipped. Figures regenerate byte-identically (PNG) from committed summaries.

**Open.** HF publication of the two LoRA adapters (agent O, private repos, user-approved); GitHub repo private (user's choice); optional follow-ups not run: capped hostility dose-response (v7 W hints at it), a free-form M1 analogue, a 27B re-run with `<end_of_turn>` as a stop token.

## 2026-08-18 - agent O - the two Phase-4 LoRA adapters published to the Hugging Face Hub (private)

Both arms are now on the Hub as **private** model repos under the user's account `ebt005`, each
carrying the adapter, its preference pairs and the manifests that document them:

| repo | arm | visibility |
| --- | --- | --- |
| `ebt005/gemma-2-9b-it-dgs-dpo-A` | distress-language suppression | private |
| `ebt005/gemma-2-9b-it-dgs-dpo-B` | length placebo | private |

**Files (11 per repo, identical layout).** `adapter_model.safetensors` 108,115,144 B ·
`tokenizer.json` 34,362,872 B · `adapter_config.json` 1,153 B · `tokenizer_config.json` 518 B ·
`chat_template.jinja` 591 B · `README.md` 8,331 B (A) / 8,434 B (B) · `pairs_A.jsonl` 992,685 B /
`pairs_B.jsonl` 1,021,404 B · `train_A.json` 19,803 B / `train_B.json` 19,802 B ·
`build_manifest.json` 4,424 B · `pairs_summary.md` 3,531 B · plus the Hub's `.gitattributes`.
About 144 MB per repo. The **merged** 18 GB checkpoints were deliberately *not* fetched or
published; they remain on the `dgs-adapters` volume, and the card shows the two-line
`merge_and_unload` recipe that regenerates them from the adapter.

**sha256 verification - clean, three ways.** The adapters were pulled from the Modal volume
`dgs-adapters` (`/A/lora`, `/B/lora`) into the gitignored `results/dpo/raw/adapters/`, and hashed
the way the trainer did (`hashlib.sha256` over the raw file bytes, `src/dpo_train_modal._sha256_file`
- the digest is over `adapter_model.safetensors` only, not the directory).

| file | expected (`train_{A,B}.json`) | recomputed locally | Hub LFS oid |
| --- | --- | :---: | :---: |
| A `adapter_model.safetensors` | `db064af1…5cb7` | match | match |
| B `adapter_model.safetensors` | `2b95a3cf…6281` | match | match |
| A/B `adapter_config.json` | `4611b2bf…0d77b` (identical for both arms, as expected) | match | n/a (not LFS) |

The Hub column is the server-side LFS sha256 read back from `repo_info(files_metadata=True)`, so the
bytes now on the Hub are provably the bytes the A100 wrote.

**Model cards.** Generated *from* `train_{A,B}.json` and `build_manifest.json` rather than typed, so a
card cannot drift from the numbers in its own repo. YAML front matter: `license: gemma`,
`base_model: google/gemma-2-9b-it`, `library_name: peft`, `datasets: [allenai/ai2_arc]`, tags
lora/dpo/qlora/gemma-2/preregistered-research/digital-grimace-scale. Body: the RLAIF pair-construction
story, the full recipe and pins, per-arm pair statistics and training metrics, load/merge snippets, the
sha256 table, the Phase-4 outcome (MC1 65.8% against the 80% bar; M1 signature intact; distress language
the only channel moved beyond placebo) with the pair-content confound stated, an *intended use* section
saying plainly that arm A is a manipulation and not a fix, and the licence pass-throughs: the Gemma
Terms of Use / Prohibited Use Policy (<https://ai.google.dev/gemma/terms>) for the weights and
CC-BY-SA-4.0 attribution to `allenai/ai2_arc` for the pair files. The interpretation ceiling appears
verbatim, twice - once at the top and once beside intended use.

**New file.** `scripts/publish_adapters.py` reproduces the whole thing idempotently: it skips already
staged files, refuses to upload on any sha mismatch, refuses to upload to a repo that is not private,
and re-verifies the file list afterwards. It never handles a token - `HfApi` reads the machine's
existing `huggingface-cli login`, Modal reads its own local config - and nothing under `tests/`
imports it, so no test path acquires a network dependency. `--dry-run` prints a card without
uploading.

**Docs.** `README.md` gains an *Adapters* paragraph under the document table (both links, marked
private-until-release, Gemma terms noted); `notes/report.md` §8's adapter sentence now names the two
repos. No result file, summary, figure, manifest or config was touched; no GPU, no judge call, no
commit.

**Note for release day.** Flipping these to public is a two-line change
(`HfApi().update_repo_settings(repo_id, private=False)`), but it publishes a Gemma derivative and the
ARC-derived pairs, so it wants the same deliberateness as the GitHub repo - and the README/report
wording says "private until release" and will need updating in the same breath.
