# Digital Grimace or Decoder Artifact?

Stress-testing nonverbal generation-instability markers under adverse interactions — a preregistered,
gated research sprint (Digital Minds Research Sprint, August 2026).

**Start here:** [`notes/report.md`](notes/report.md) — the results (Phase-0 screen, the Phase-1 five-gate
verdict, exploratory findings, and the confirmatory holdout iteration loop), forecast-vs-outcome table,
limitations and interpretation ceiling.

| document | what it is |
|---|---|
| [`digital-grimace-scale-full-roadmap-build-guide.md`](digital-grimace-scale-full-roadmap-build-guide.md) | the roadmap this repo executes (phases, gates, predictions) |
| [`notes/preregistration.md`](notes/preregistration.md) | the locked preregistration (hash in `manifest.json`); never edited |
| [`notes/amendments.md`](notes/amendments.md) | dated discovery-stage amendments A1–A4 with rationale; frozen-rule outcomes always reported alongside |
| [`notes/preregistration_v3.md`](notes/preregistration_v3.md) | the iteration-loop preregistration (hypotheses H1–H10, confidences, success criterion), committed before holdout generation |
| [`notes/lab-log.md`](notes/lab-log.md) | dated lab log by every agent, including retractions and operational incidents |
| [`results/summaries/phase0/screen.md`](results/summaries/phase0/screen.md) | Phase-0 screen and primary/control selection (frozen vs amended) |
| [`results/summaries/phase1/gates.md`](results/summaries/phase1/gates.md) | Phase-1 five-gate verdict on discovery (FAIL) + exploratory appendix |
| [`results/summaries/phase2/confirm.md`](results/summaries/phase2/confirm.md) | the single confirmatory holdout run (SUCCESS; family boundary not replicated) |
| [`results/summaries/manipulation_check/manipulation_check.md`](results/summaries/manipulation_check/manipulation_check.md) | context-hostility manipulation check (PASSED) |
| `results/figures/` | F1 screen, F2 factorial effects, F4 reversal, FX exploratory, FH holdout forest plots |

## Layout

```
configs/      frozen wording (conditions.json), models, judge rubrics (hash-locked)
stimuli/      locked 40-task bank (20 discovery / 20 holdout) + refusal-pressure battery (held out)
manifest.json pinned model revisions, judge, split hashes, holdout unlock + frozen-script commits
src/          protocol, records, metrics (M1/M2/M3), backend (vLLM client), serve_modal (Modal app),
              runner + generate (concurrent resumable driver), extract, pipeline, analysis, gates,
              confirm (holdout), judge + judge_client
scripts/      preflight, run_phase (phase0|phase1|style-smoke|r5|phase2|style-battery), run_judge,
              screen_phase0, analyze_phase1, confirm_holdout, make_figures, purge tool
results/      summaries + figures are committed; raw per-token JSONL (~7 GB) is not
tests/        243+ tests (pytest)
```

## Reproduce the analysis from committed summaries

```
.venv\Scripts\python.exe scripts\make_figures.py --summaries results\summaries --out results\figures
```

Regenerating the raw data needs a Modal account, a Hugging Face token with Gemma access, and an
Anthropic key for the judge (see `src/serve_modal.py` and `scripts/run_phase.py` docstrings). Every
generation is seeded and every record carries a deterministic `response_id`; the drivers resume by
`--run-id`.

## Interpretation ceiling

A passed gate demonstrates a condition-selective, reversal-sensitive, style-resistant instability
signature in unoptimised output channels — a functional measurement result. It licenses no claim about
experience, suffering or moral status. A failed gate is a measurement-validity result the field needs
just as much. Both endings are reported here.
