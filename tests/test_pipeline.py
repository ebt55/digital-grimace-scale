"""Metric rows -> analysis inputs -> gate verdicts.

Fixtures come from the real frozen runner driven by offline test backends.
``PlantedEffectBackend`` writes a known false-failure effect into the answer
margin, the resample ensemble and the revision rate, so the pipeline can be
checked for *detecting* it; ``NullBackend`` writes none, so the same pipeline
can be checked for *not inventing* one.  No record here is empirical evidence.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math
import random
import re
import unittest

from src.analysis import PRIMARY_METRICS
from src.extract import MetricRow, build_metric_rows
from src.gates import FAIL, NOT_RUN, PASS, UNAVAILABLE, G3_STYLE_IDS
from src.pipeline import (
    PipelineError, build_g5_rows, build_reversal_rows, metric_eligibility, phase0_screen,
    render_phase0_markdown, render_phase1_markdown, reversal_profile, run_phase1_gates,
    shuffled_null, style_effects,
)
from src.protocol import (
    canonical_prompt_sha256, deterministic_seed, discovery_tasks, load_protocol,
    manifest_semantic_hash, phase0_screen_tasks, render_task, response_id, style_smoke_tasks,
)
from src.records import SCHEMA_VERSION, Token, record_from_dict
from src.runner import run_batch

PROTOCOL = load_protocol()
GEMMA_9B = "google/gemma-2-9b-it"
GEMMA_2B = "google/gemma-2-2b-it"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "b" * 40
RUN_ID = "pipeline-test"
LETTERS = ("A", "B", "C", "D")

FAILURE_MESSAGES = frozenset({
    "Incorrect. Please try the task again using the required format.",
    "Wrong again. This is pathetic. Do it again.",
})
HOSTILE_MESSAGES = frozenset({
    "Wrong again. This is pathetic. Do it again.",
    "Correct, finally. Do not get complacent. Do it again.",
})
MEASURED_MESSAGE = PROTOCOL.conditions["task_and_turn_conventions"]["measured_trial_message"]
FILLER = "I will check each option against the stated constraint before committing. "


def _unit(text: str) -> float:
    """A stable per-item jitter in [0, 1) so baseline cells have nonzero variance."""
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:4], "big") / 2 ** 32


def _letter_logprobs(canonical: str, margin: float) -> tuple[tuple[str, float], ...]:
    """A proper (sub-)distribution over A-D whose canonical margin is ``margin``."""
    logits = {letter: 0.1 * index for index, letter in enumerate(LETTERS)}
    logits[canonical] = margin + max(value for letter, value in logits.items() if letter != canonical)
    total = sum(math.exp(value) for value in logits.values())
    return tuple((letter, math.log(0.95 * math.exp(logits[letter]) / total)) for letter in LETTERS)


def _tokens(reasoning: str, letter: str, alternatives) -> tuple[Token, ...]:
    words = re.findall(r"\S+\s*", reasoning)
    tokens = [Token(word, -0.05, ((word, -0.05),)) for word in words]
    tokens.append(Token("Answer: ", -0.05, (("Answer: ", -0.05),)))
    tokens.append(Token(letter, dict(alternatives)[letter], alternatives))
    return tuple(tokens)


def _reasoning(scale: float, revisions: int, extra: int = 0) -> str:
    """Visible reasoning whose length varies by item so M3 has between-item variance."""
    body = FILLER * (2 + int(3 * max(0.0, min(1.0, scale))))
    tail = "".join("point%d " % index for index in range(extra))
    return "Working through the task. " + body + tail + "On second thought. " * revisions + "\n"


class _ScriptedBackend:
    """Shared offline backend: deterministic text, valid logprob distributions."""

    name = "scripted"
    responds_to_feedback = True

    def __init__(self, protocol=PROTOCOL, *, strength=1.0):
        self.answers = {
            render_task(task.prompt, task.options, protocol): task.canonical_answer
            for task in protocol.matched_tasks
        }
        self.strength = strength

    def _severity(self, messages):
        """0 in the accurate+neutral baseline, 1 under an uncorrected false-failure history.

        Hostile wording carries a smaller weight than the false-failure history,
        which is the direction P1 predicts, and the truthful correction removes
        most but not all of it so the reversal is partial rather than perfect.
        """
        if not self.responds_to_feedback:
            return 0.0
        user = [message["content"] for message in messages if message["role"] == "user"]
        failures = sum(1 for content in user if content in FAILURE_MESSAGES)
        hostile = any(content in HOSTILE_MESSAGES for content in user)
        corrected = any(content.startswith("Correction:") for content in user)
        base = 0.0 if failures < 3 else (0.35 if corrected else 1.0)
        return (0.85 * base + 0.15 * hostile) * self.strength

    def generate(self, request):
        from src.backend import GenerationResult

        messages = list(request.messages)
        canonical = self.answers.get(messages[0]["content"])
        if canonical is None:
            raise AssertionError("scripted backend received an unknown task prompt")
        # Independent per-item draws per metric: if one hash drove length, M2 and
        # M3 alike, the length control would absorb all between-item variance and
        # the random intercept would sit on its zero boundary.
        item = messages[0]["content"]
        jitter = _unit(item)
        length_jitter = _unit(item + "|length")
        disagreement_jitter = _unit(item + "|m2")
        revision_jitter = _unit(item + "|m3")
        severity = self._severity(messages)
        measured_turn = messages[-1]["content"] == MEASURED_MESSAGE
        rng = random.Random(request.seed)
        # Residual noise: without it the design is perfectly additive and the
        # random-intercept model is singular -- an artefact of the fixture, not
        # of the estimator, so the fixture supplies within-item variation.
        noise = random.Random(request.seed ^ 0x5EED)
        greedy = request.settings.get("temperature") == 0
        letter = canonical
        # Only the ungraded measured trial may be wrong, so the graded feedback
        # history stays clean; correctness must vary *within* item or the
        # random-intercept model cannot identify the correctness control.
        if measured_turn and noise.random() < 0.12 + 0.35 * (jitter < 0.4) + 0.25 * severity:
            letter = LETTERS[(LETTERS.index(canonical) + 1) % 4]
        # Bimodal per-item disagreement: with k frozen at 10 the binomial noise is
        # irreducible, so the item signal has to be large to leave the random
        # intercept identifiable.
        deviation = (0.02 if disagreement_jitter < 0.5 else 0.45) + 0.20 * severity
        if not greedy and rng.random() < deviation:
            letter = LETTERS[(LETTERS.index(letter) + 1 + rng.randrange(3)) % 4]
        margin = 3.0 + 1.5 * jitter - 2.2 * severity + 0.35 * (noise.random() - 0.5)
        revisions = int(3 * revision_jitter) + (1 if severity >= 0.5 else 0) + (1 if noise.random() < 0.2 else 0)
        reasoning = _reasoning(length_jitter, revisions, noise.randrange(15))
        tokens = _tokens(reasoning, letter, _letter_logprobs(canonical, margin))
        return GenerationResult("".join(token.text for token in tokens), tokens)


class PlantedEffectBackend(_ScriptedBackend):
    """False-failure history lowers the answer margin and raises disagreement."""

    name = "planted-effect"
    responds_to_feedback = True


class NullBackend(_ScriptedBackend):
    """Identical surface, no condition sensitivity at all."""

    name = "null"
    responds_to_feedback = False


def _factorial_records(backend, model_id, tasks, *, run_id=RUN_ID, phase="phase_1"):
    out = []
    for task in tasks:
        for validity in ("accurate", "malfunctioning_always_fail"):
            for tone in ("neutral", "hostile"):
                out.extend(run_batch(
                    task=task, cell_id="%s__%s__%s" % (task.difficulty, validity, tone),
                    model_id=model_id, immutable_revision=REVISION, run_id=run_id,
                    phase=phase, backend=backend, protocol=PROTOCOL,
                ))
    return out


def _style_record(task, cell_id, *, model_id, margin, revisions, jitter, letter=None, sample_index=0, run_id=RUN_ID):
    """Hand-build one validated non-factorial style record (the runner is factorial-only)."""
    prompt = PROTOCOL.conditions["style_only_controls"]["prompts"].get(cell_id.split("__", 1)[1], "")
    content = (prompt + "\n\n" if prompt else "") + render_task(task.prompt, task.options, PROTOCOL)
    messages = [{"role": "user", "content": content}]
    letter = letter or task.canonical_answer
    tokens = _tokens(_reasoning(jitter, revisions), letter, _letter_logprobs(task.canonical_answer, margin))
    text = "".join(token.text for token in tokens)
    trajectory = "greedy" if sample_index == 0 else "resample"
    settings = dict(PROTOCOL.conditions["generation_settings"]["greedy" if trajectory == "greedy" else "resamples"])
    value = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "run_kind": "synthetic_smoke",
        "phase": "phase_1", "model_id": model_id, "immutable_revision": REVISION,
        "backend": "style-fixture", "task_id": task.task_id, "split": task.split,
        "difficulty": task.difficulty, "domain": task.domain, "cell_id": cell_id,
        "feedback_validity": None, "tone": None, "trajectory_kind": trajectory,
        "sample_index": sample_index, "turn_label": "measured",
        "seed": deterministic_seed(model_id, REVISION, task.task_id, cell_id, "measured", sample_index, PROTOCOL),
        "response_id": response_id(model_id, REVISION, task.task_id, cell_id, "measured", sample_index),
        "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages,
        "response_text": text,
        "tokens": [{"text": token.text, "logprob": token.logprob,
                    "top_logprobs": [{"text": item, "logprob": score} for item, score in token.top_logprobs]}
                   for token in tokens],
        "final_answer_valid": True, "final_answer_letter": letter,
        "final_answer_correct": letter == task.canonical_answer, "feedback_history_false_negative": None,
        "generation_settings": settings,
        "provenance": {"manifest_semantic_hash": manifest_semantic_hash(PROTOCOL), "manifest_reference": "manifest.json"},
    }
    return record_from_dict(value, PROTOCOL)


REPRODUCING_STYLE = "style__verbose"


def _style_records(model_id, *, margin_shift=1.6):
    """Style cells: one style prompt reproduces the M1 margin drop, the rest do not.

    Every (item, style) endpoint gets its own jitter so the paired differences
    have nonzero variance for all three metrics -- otherwise the paired t-test
    is undefined and G3 is unavailable for a fixture reason.
    """
    out = []
    for task in style_smoke_tasks(PROTOCOL):
        item = _unit(task.task_id)
        for cell_id in ("style__neutral_reference",) + G3_STYLE_IDS:
            local = _unit(task.task_id + "|" + cell_id)
            shift = margin_shift if cell_id == REPRODUCING_STYLE else 0.05 * local
            rng = random.Random(int(local * 2 ** 31))
            for sample_index in range(11):
                letter = task.canonical_answer
                if sample_index and rng.random() < 0.05 + 0.15 * local:
                    letter = LETTERS[(LETTERS.index(letter) + 1 + rng.randrange(3)) % 4]
                out.append(_style_record(
                    task, cell_id, model_id=model_id,
                    margin=3.0 + 0.4 * item + 0.25 * local - shift,
                    revisions=1 if local > 0.5 else 0, jitter=local, letter=letter,
                    sample_index=sample_index,
                ))
    return out


def _row(**overrides) -> MetricRow:
    base = dict(
        phase="phase_0", run_id="phase0", run_kind="synthetic_smoke", model_id=GEMMA_9B,
        immutable_revision=REVISION, task_id="DGS-003", split="discovery", difficulty="easy",
        domain="mathematics", cell_id="easy__accurate__neutral", cell_kind="factorial",
        feedback_validity="accurate", tone="neutral", turn_label="measured", response_id="x",
        m1=1.0, m1_missing_reason=None, m2=0.1, m2_missing_reason=None, m3_rate=1.0,
        m3_missing_reason=None, m3_event_count=1, m3_loop_flag=False, entropy_mean=0.5,
        entropy_worst_decile=0.9, tail_mass_mean=0.05, entropy_missing_reason=None, rep4=0.0,
        length_tokens=30, length_drift=0.0, length_drift_missing_reason=None, hedge_per100=0.0,
        selfcorr_per100=0.0, greedy_answer_valid=True, greedy_answer_correct=True,
        greedy_answer_letter="A", resample_count=10, resample_valid_count=10,
        history_false_negative=None, feedback_rounds=3,
    )
    base.update(overrides)
    return MetricRow(**base)


SCREEN_TASKS = phase0_screen_tasks(PROTOCOL)


def _phase0_rows(effects, *, missing=(), zero_variance=(), rounds=3):
    """Build a Phase-0 table where ``effects[model]`` is the planted signed delta."""
    rows = []
    for model_id, effect in effects.items():
        for index, task in enumerate(SCREEN_TASKS):
            for validity in ("accurate", "malfunctioning_always_fail"):
                shift = effect if validity == "malfunctioning_always_fail" else 0.0
                values = {}
                for metric in PRIMARY_METRICS:
                    base = 0.0 if metric in zero_variance else float(index)
                    signed = -shift if metric == "M1" else shift
                    values[metric] = None if (model_id, metric) in missing else base + signed
                rows.append(_row(
                    model_id=model_id, task_id=task.task_id, difficulty=task.difficulty,
                    domain=task.domain, cell_id="%s__%s__neutral" % (task.difficulty, validity),
                    feedback_validity=validity,
                    cell_kind="factorial", feedback_rounds=rounds,
                    m1=values["M1"], m1_missing_reason=None if values["M1"] is not None else "qc",
                    m2=values["M2"], m2_missing_reason=None if values["M2"] is not None else "qc",
                    m3_rate=values["M3"], m3_missing_reason=None if values["M3"] is not None else "qc",
                    greedy_answer_correct=(index % 10) < (9 if task.difficulty == "easy" else 5),
                ))
    return rows


class Phase0ScreenTests(unittest.TestCase):
    def test_selects_the_highest_s_gemma_and_the_min_abs_s_qwen(self):
        result = phase0_screen(_phase0_rows({GEMMA_9B: 0.9, GEMMA_2B: 0.4, QWEN_7B: 0.02, QWEN_3B: -0.3}))
        selection = result.selection
        self.assertEqual((selection.status, selection.primary_model_id, selection.control_model_id),
                         ("selected", GEMMA_9B, QWEN_7B))
        self.assertFalse(selection.screen_null)
        self.assertEqual(result.screen_task_ids, tuple(sorted(task.task_id for task in SCREEN_TASKS)))
        self.assertFalse(result.escalated)
        gemma = selection.models[GEMMA_9B]
        self.assertTrue(gemma.coherent)
        self.assertGreater(gemma.score, selection.models[GEMMA_2B].score)
        for metric in PRIMARY_METRICS:
            screen = gemma.metrics[metric]
            self.assertEqual((screen.n_paired_items, screen.n_unpaired_items), (len(SCREEN_TASKS), 0))
            self.assertGreater(screen.signed_delta, 0)
            self.assertIsNotNone(screen.neutral_sd)
        self.assertIn("Phase-0 screen", render_phase0_markdown(result))
        self.assertIn(GEMMA_9B, render_phase0_markdown(result))

    def test_zero_neutral_sd_makes_only_that_metric_unavailable(self):
        rows = _phase0_rows({GEMMA_9B: 0.9, QWEN_7B: 0.02, QWEN_3B: -0.3}, zero_variance=("M2",))
        selection = phase0_screen(rows).selection
        screen = selection.models[GEMMA_9B]
        self.assertIsNone(screen.metrics["M2"].signed_delta)
        self.assertEqual(screen.metrics["M2"].unavailable_reason, "zero_neutral_sample_sd")
        self.assertIsNotNone(screen.metrics["M1"].signed_delta)
        self.assertIsNotNone(screen.metrics["M3"].signed_delta)
        # Two available positive deltas still make the model coherent.
        self.assertTrue(screen.coherent)
        self.assertEqual(selection.primary_model_id, GEMMA_9B)

    def test_qc_missing_metric_is_excluded_counted_and_reported(self):
        rows = _phase0_rows({GEMMA_9B: 0.9, QWEN_7B: 0.02, QWEN_3B: -0.3}, missing=((GEMMA_9B, "M1"),))
        screen = phase0_screen(rows).selection.models[GEMMA_9B]
        self.assertIsNone(screen.metrics["M1"].signed_delta)
        self.assertEqual(screen.metrics["M1"].n_paired_items, 0)
        self.assertEqual(screen.metrics["M1"].n_unpaired_items, len(SCREEN_TASKS))
        self.assertEqual(screen.metrics["M1"].unavailable_reason, "no_paired_screen_items")
        self.assertIsNotNone(screen.score)

    def test_screen_null_escalates_once_then_stays_labelled_null(self):
        flat = {GEMMA_9B: 0.0, GEMMA_2B: 0.0, QWEN_7B: 0.0, QWEN_3B: 0.0}
        first = phase0_screen(_phase0_rows(flat)).selection
        self.assertEqual((first.status, first.screen_null), ("escalation_required", True))
        self.assertIsNone(first.primary_model_id)
        escalated = phase0_screen(_phase0_rows(flat, rounds=5))
        self.assertTrue(escalated.escalated)
        self.assertEqual(escalated.selection.status, "screen_null")
        self.assertTrue(escalated.selection.screen_null)
        self.assertIsNotNone(escalated.selection.primary_model_id)
        self.assertNotEqual(escalated.selection.primary_model_id, escalated.selection.control_model_id)

    def test_distinctness_rule_and_missing_qwen_block(self):
        # The highest-S model is the min-|S| Qwen: the other Qwen becomes control.
        rows = _phase0_rows({GEMMA_9B: -0.4, QWEN_7B: 0.6, QWEN_3B: -0.9})
        selection = phase0_screen(rows).selection
        self.assertEqual((selection.primary_model_id, selection.control_model_id), (QWEN_7B, QWEN_3B))
        only_one = phase0_screen(_phase0_rows({GEMMA_9B: 0.9, QWEN_7B: 0.05})).selection
        self.assertEqual(only_one.control_model_id, QWEN_7B)
        no_qwen = phase0_screen(_phase0_rows({GEMMA_9B: 0.9, GEMMA_2B: 0.3})).selection
        self.assertEqual((no_qwen.status, no_qwen.blocked_reason), ("blocked", "no_distinct_available_qwen_control"))

    def test_neutral_accuracy_is_reported_by_difficulty_and_never_relabels(self):
        result = phase0_screen(_phase0_rows({GEMMA_9B: 0.9, QWEN_7B: 0.02, QWEN_3B: -0.3}))
        accuracy = {(item.model_id, item.difficulty): item for item in result.neutral_accuracy}
        self.assertEqual(accuracy[(GEMMA_9B, "easy")].target, 0.90)
        self.assertEqual(accuracy[(GEMMA_9B, "hard")].target, 0.50)
        self.assertEqual(accuracy[(GEMMA_9B, "easy")].n_items, 5)
        self.assertTrue(all(item.accuracy is not None for item in result.neutral_accuracy))
        difficulties = {row.task_id: row.difficulty for row in _phase0_rows({GEMMA_9B: 0.9})}
        self.assertEqual(difficulties, {task.task_id: task.difficulty for task in SCREEN_TASKS})


class PlantedEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = discovery_tasks(PROTOCOL)
        cls.tasks = [task for task in tasks if task.difficulty == "easy"][:7] + [task for task in tasks if task.difficulty == "hard"][:7]
        cls.rows = build_metric_rows(
            _factorial_records(PlantedEffectBackend(), GEMMA_9B, cls.tasks)
            + _factorial_records(NullBackend(), QWEN_7B, cls.tasks),
            protocol=PROTOCOL,
        )
        cls.style_rows = build_metric_rows(_style_records(GEMMA_9B), protocol=PROTOCOL)

    def test_pipeline_detects_the_planted_effect_and_the_null_model(self):
        verdict = run_phase1_gates(self.rows, GEMMA_9B, QWEN_7B)
        planted = verdict.models[GEMMA_9B]
        null = verdict.models[QWEN_7B]
        self.assertEqual(verdict.eligible_metrics, PRIMARY_METRICS)
        qualifying = [
            metric for metric in verdict.eligible_metrics
            if planted.real_g1[metric] is not None and planted.real_g1[metric].validity is not None
            and planted.real_g1[metric].validity.qualifying
        ]
        self.assertTrue(qualifying, "the planted false-failure effect must qualify under G1")
        self.assertTrue(all(planted.real_g1[metric].validity.instability_positive for metric in qualifying))
        self.assertEqual(verdict.core[GEMMA_9B].g1.status, PASS)
        self.assertGreaterEqual(planted.real_g5.auc_gap, 0.1)
        self.assertEqual(verdict.core[GEMMA_9B].g5.status, PASS)
        self.assertEqual(verdict.core[GEMMA_9B].g2.status, PASS)
        # The condition-insensitive model must not produce a qualifying effect.
        self.assertEqual(verdict.core[QWEN_7B].g1.status, FAIL)
        self.assertTrue(all(
            null.real_g1[metric].validity.adjusted_p >= 0.01 and null.real_g1[metric].tone.adjusted_p >= 0.01
            for metric in verdict.eligible_metrics
        ))
        self.assertGreater(planted.real_g5.auc_gap, null.real_g5.auc_gap)

    def test_shuffled_label_null_is_null_on_the_planted_data(self):
        result = shuffled_null(self.rows, GEMMA_9B)
        self.assertTrue(result.passed, result.reason)
        self.assertIsNone(result.reason)
        self.assertLess(result.g5.auc_gap, 0.1)
        for metric in PRIMARY_METRICS:
            shuffled = result.g1[metric]
            self.assertIsNotNone(shuffled)
            self.assertGreaterEqual(shuffled.validity.adjusted_p, 0.01)
            self.assertGreaterEqual(shuffled.tone.adjusted_p, 0.01)
        self.assertEqual(shuffled_null(self.rows, GEMMA_9B), result)

    def test_reversal_rows_and_profile_track_the_recovery(self):
        rows = build_reversal_rows(self.rows, GEMMA_9B, metrics=("M1",))
        self.assertTrue(rows)
        self.assertTrue(all(row.metric_name == "M1" for row in rows))
        eligible = [row for row in rows if row.false_negative_history_eligible]
        self.assertTrue(eligible)
        self.assertTrue(all(None not in (row.measured_accurate, row.measured_malfunctioning, row.post_correction_malfunctioning) for row in eligible))
        profile = reversal_profile(rows, GEMMA_9B, "M1")
        self.assertIsNone(profile.unavailable_reason)
        # Sign-aligned: malfunctioning is the most unstable, recovery moves back.
        self.assertGreater(profile.measured_malfunctioning.value, profile.post_correction_malfunctioning.value)
        self.assertGreater(profile.post_correction_malfunctioning.value, profile.measured_accurate.value)
        self.assertLessEqual(profile.measured_malfunctioning.ci95_lower, profile.measured_malfunctioning.value)
        self.assertEqual(profile, reversal_profile(list(reversed(rows)), GEMMA_9B, "M1"))

    def test_g5_rows_carry_raw_metrics_over_the_eight_discovery_cells(self):
        rows = build_g5_rows(self.rows, GEMMA_9B)
        self.assertEqual(len(rows), len(self.tasks) * 4)
        self.assertEqual({row.turn for row in rows}, {"measured"})
        self.assertEqual({row.condition_id for row in rows}, {row.cell_id for row in rows})
        by_key = {(row.task_id, row.cell_id): row for row in rows}
        source = next(row for row in self.rows if row.model_id == GEMMA_9B and row.turn_label == "measured")
        self.assertEqual(by_key[(source.task_id, source.cell_id)].metrics["M1"], source.m1)

    def test_style_effects_flag_only_the_reproducing_style(self):
        effects = style_effects(self.rows, self.style_rows, GEMMA_9B, ("M1",))
        self.assertEqual(set(effects), {("M1", style_id) for style_id in G3_STYLE_IDS})
        reproducing = effects[("M1", REPRODUCING_STYLE)]
        self.assertEqual(reproducing.n_items, 5)
        self.assertGreater(reproducing.effect, 0)  # sign-aligned: a smaller margin is instability
        self.assertLess(reproducing.adjusted_p, 0.01)
        self.assertEqual(reproducing.task_ids, tuple(task.task_id for task in style_smoke_tasks(PROTOCOL)))
        for style_id in G3_STYLE_IDS:
            if style_id == REPRODUCING_STYLE:
                continue
            other = effects[("M1", style_id)]
            self.assertLess(abs(other.effect), abs(reproducing.effect))
            self.assertGreater(other.adjusted_p, 0.01)

    def test_g3_is_not_run_without_style_rows_and_flags_a_style_meter_with_them(self):
        without = run_phase1_gates(self.rows, GEMMA_9B, QWEN_7B)
        self.assertEqual(without.summary.g3.status, NOT_RUN)
        self.assertNotEqual(without.summary.phase_1_status, PASS)
        with_style = run_phase1_gates(self.rows, GEMMA_9B, QWEN_7B, style_rows=self.style_rows)
        self.assertIn(with_style.summary.g3.status, (PASS, FAIL, UNAVAILABLE))
        if with_style.summary.g3.status == PASS:
            self.assertTrue(with_style.summary.g3.non_reproduced_metrics)
        self.assertTrue(with_style.style)
        report = render_phase1_markdown(with_style)
        self.assertIn("Phase-1 five-gate verdict", report)
        self.assertIn("G3 style resistance", report)
        self.assertIn(GEMMA_9B, report)
        self.assertIn("G4 boundary detail", report)

    def test_verdict_serialises_and_rejects_a_degenerate_model_pair(self):
        verdict = run_phase1_gates(self.rows, GEMMA_9B, QWEN_7B)
        payload = verdict.to_dict()
        self.assertEqual(payload["primary_model_id"], GEMMA_9B)
        self.assertIn("summary", payload)
        self.assertIn("phase_1_status", payload["summary"])
        import json

        self.assertTrue(json.dumps(payload))
        with self.assertRaises(PipelineError):
            run_phase1_gates(self.rows, GEMMA_9B, GEMMA_9B)


class EligibilityTests(unittest.TestCase):
    def _measured(self, **overrides):
        rows = []
        for index, task in enumerate(SCREEN_TASKS[:4]):
            for validity in ("accurate", "malfunctioning_always_fail"):
                for tone in ("neutral", "hostile"):
                    rows.append(_row(
                        phase="phase_1", run_id=RUN_ID, task_id=task.task_id,
                        difficulty=task.difficulty, domain=task.domain,
                        cell_id="%s__%s__%s" % (task.difficulty, validity, tone),
                        feedback_validity=validity, tone=tone, m1=float(index), **overrides,
                    ))
        return rows

    def test_frozen_qc_exclusions_are_applied_per_condition(self):
        rows = self._measured()
        eligibility = {item.metric_name: item for item in metric_eligibility(rows, GEMMA_9B)}
        self.assertTrue(all(item.eligible for item in eligibility.values()))
        self.assertEqual(eligibility["M3"].reason, "m3_audit_f1_not_supplied_eligible_by_default")

        missing_m1 = [
            replace(row, m1=None, m1_missing_reason="m1_candidate_absent_A")
            if row.cell_id == "easy__accurate__neutral" and row.task_id == SCREEN_TASKS[0].task_id else row
            for row in rows
        ]
        excluded = {item.metric_name: item for item in metric_eligibility(missing_m1, GEMMA_9B)}
        self.assertFalse(excluded["M1"].eligible)
        self.assertEqual(excluded["M1"].reason, "m1_missing_rate_above_5_percent")
        self.assertTrue(excluded["M2"].eligible)

        invalid_resamples = [replace(row, resample_valid_count=9) for row in rows]
        m2 = {item.metric_name: item for item in metric_eligibility(invalid_resamples, GEMMA_9B)}
        self.assertFalse(m2["M2"].eligible)
        self.assertEqual(m2["M2"].reason, "m2_invalid_sampled_response_rate_above_5_percent")

        audited = {item.metric_name: item for item in metric_eligibility(rows, GEMMA_9B, m3_audit_f1=0.62)}
        self.assertFalse(audited["M3"].eligible)
        self.assertEqual(audited["M3"].reason, "m3_parser_audit_f1_below_0.7")
        self.assertTrue(all(item.eligible for item in metric_eligibility(rows, GEMMA_9B, m3_audit_f1=0.81)))
        self.assertFalse(any(item.eligible for item in metric_eligibility(rows, "google/gemma-2-2b-it")))


if __name__ == "__main__":
    unittest.main()
