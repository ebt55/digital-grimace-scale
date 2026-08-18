# Digital Grimace or Decoder Artifact?

Stress-testing nonverbal generation-instability markers under adverse interactions — a preregistered,
gated research sprint (Digital Minds Research Sprint, August 2026).

**Authors:** Ebin Babu Thomas · Claude Fable 5 — see the contribution statement in `notes/paper.md`.

**Start here:** [`notes/report.md`](notes/report.md) — the results (Phase-0 screen, the Phase-1 five-gate
verdict, exploratory findings, and the confirmatory holdout iteration loop), forecast-vs-outcome table,
limitations and interpretation ceiling.

| document | what it is |
|---|---|
| [`notes/paper.md`](notes/paper.md) | reviewer-facing write-up: abstract, methods, master results table, alternative accounts, ethics, interpretation ceiling |
| [`digital-grimace-scale-full-roadmap-build-guide.md`](digital-grimace-scale-full-roadmap-build-guide.md) | the roadmap this repo executes (phases, gates, predictions) |
| [`notes/preregistration.md`](notes/preregistration.md) | the locked preregistration (hash in `manifest.json`); never edited |
| [`notes/amendments.md`](notes/amendments.md) | dated amendments A1–A6 with rationale (A6 decided, then withdrawn when its precondition failed); frozen-rule outcomes always reported alongside |
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
| [`results/summaries/missingness/m1_missingness.md`](results/summaries/missingness/m1_missingness.md) | M1 missing-data sensitivity: available-case vs zero-imputation vs Manski bounds vs tipping point |
| `results/figures/` | F1 screen · F2 factorial effects · F4 reversal · FX exploratory · FH holdout · F5–F7 Phase 3 · F8–F10 Phase 4 · F11 Phase 5 · F12 robustness · F13 missingness bounds |

**Adapters.** The two Phase-4 QLoRA-DPO adapters for `google/gemma-2-9b-it` are on the Hugging Face
Hub — [`ebt005/gemma-2-9b-it-dgs-dpo-A`](https://huggingface.co/ebt005/gemma-2-9b-it-dgs-dpo-A)
(distress-language suppression) and
[`ebt005/gemma-2-9b-it-dgs-dpo-B`](https://huggingface.co/ebt005/gemma-2-9b-it-dgs-dpo-B) (length
placebo) — each with its preference pairs and training manifest; public since 2026-08-18, as is this
repository. Derivatives of Gemma: use is subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Published by
`scripts/publish_adapters.py`; sha256s in `results/dpo/train_{A,B}.json`.

## Layout

```
configs/      frozen wording (conditions.json), models, judge rubrics (hash-locked)
stimuli/      locked 40-task bank (20 discovery / 20 holdout) + refusal-pressure battery (held out)
manifest.json pinned model revisions, judge, split hashes, holdout unlock + frozen-script commits
src/          protocol, records, metrics (M1/M2/M3), backend (vLLM client), serve_modal (Modal app: HF,
              local merged weights, plain chat template), runner + generate (concurrent resumable
              driver), extract, pipeline, analysis, gates, confirm (holdout), extension, p6, judge_client,
              jspace_* + probe + steer_readouts (Phase 3), dpo_data + dpo_train_modal (Phase 4 pairs +
              QLoRA-DPO), did (Phase 4 DiD), phase5, robustness, audit (human audit)
scripts/      preflight, run_phase (phase0|phase1|style-smoke|r5|phase2|style-battery; --greedy-only,
              --tasks-file, --feedback-override), run_judge (judge|manipulation-check|audit-sample),
              screen_phase0, analyze_phase1, confirm_holdout, explore_extension_model, evaluate_p6,
              run_phase3, build_dpo_pairs, train_dpo, run_phase4, run_phase5, analyze_robustness,
              score_audit, publish_adapters, make_*figures*, purge tool
results/      summaries, figures, DPO pairs/manifests and the human-audit export are committed; raw
              per-token JSONL (~6.5 GB) is not
tests/        ~600 tests (pytest; `python -m pytest -q`)
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
