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
| [`notes/methods_training.md`](notes/methods_training.md) | how the Phase-4 adapters are trained without hand-written data: RLAIF-style self-generated pairs, the locked judge as oracle, DPO, QLoRA, placebo arm, DiD |
| [`notes/preregistration_v4_phase3.md`](notes/preregistration_v4_phase3.md) · [`v5_phase4`](notes/preregistration_v5_phase4.md) · [`v6_phase5_base`](notes/preregistration_v6_phase5_base.md) · [`v7_robustness`](notes/preregistration_v7_robustness.md) | the later preregistrations (J1–J6, K1–K6, L1–L5, W/S/G), each committed before the data it governs |
| [`results/summaries/phase0/screen.md`](results/summaries/phase0/screen.md) | Phase-0 screen and primary/control selection (frozen vs amended) |
| [`results/summaries/phase1/gates.md`](results/summaries/phase1/gates.md) | Phase-1 five-gate verdict on discovery (FAIL) + exploratory appendix |
| [`results/summaries/phase2/confirm.md`](results/summaries/phase2/confirm.md) | the single confirmatory holdout run (SUCCESS; family boundary not replicated) |
| [`results/summaries/manipulation_check/manipulation_check.md`](results/summaries/manipulation_check/manipulation_check.md) | context-hostility manipulation check (PASSED) |
| [`results/summaries/extension/`](results/summaries/extension/) · [`p6/p6.md`](results/summaries/p6/p6.md) | Llama-3.1-8B third family; refusal-pressure battery |
| [`results/summaries/phase3/phase3.md`](results/summaries/phase3/phase3.md) | Phase 3: tone decodable (AUC 1.0) but a one-layer direction does not drive M1 |
| [`results/summaries/phase4/phase4.md`](results/summaries/phase4/phase4.md) | Phase 4: distress-DPO vs placebo — MC1 fails at 66%; M1 signature survives; only distress language moves |
| [`results/summaries/phase5/phase5.md`](results/summaries/phase5/phase5.md) | Phase 5: base model unmeasurable (10% parseable); plain-template control reproduces all but H2b |
| [`results/summaries/robustness/robustness.md`](results/summaries/robustness/robustness.md) | v7: item scale (replicates, larger), wording (dose–response), 27B; plus `special_token_audit.md`, `bogus_verdict_audit.md` |
| [`results/summaries/judge/human_audit.md`](results/summaries/judge/human_audit.md) | blinded human audit of the judge (30 responses; within-2 on 28/30) |
| `results/figures/` | F1 screen · F2 factorial effects · F4 reversal · FX exploratory · FH holdout · F5–F7 Phase 3 · F8–F10 Phase 4 · F11 Phase 5 · F12 robustness |

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
