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
