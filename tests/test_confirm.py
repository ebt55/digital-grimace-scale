"""Preregistration v3 confirmatory analysis.

Fixtures are hand-built metric rows with planted effects, so each support rule
can be checked against a known answer.  Nothing here reads the holdout.
"""
from __future__ import annotations

import unittest

from src.confirm import (
    HYPOTHESES, NULL_FAMILY, PREDICTION_TEXT, BootstrapResult, ConfirmError, ContrastSpec, Side,
    bootstrap_contrast, build_index, is_supported, outcome_value, render_confirm_markdown,
    run_confirmation, shuffled_cell_map, style_battery,
)
from src.extract import MetricRow
from src.pipeline import FROZEN_RULES
from src.protocol import load_protocol

PROTOCOL = load_protocol()
PRIMARY = "google/gemma-2-9b-it"
CONTROL = "Qwen/Qwen2.5-3B-Instruct"
MODELS = {"primary": PRIMARY, "control": CONTROL}
HOLDOUT = [task for task in PROTOCOL.matched_tasks if task.split == "holdout"]
EASY = [task for task in HOLDOUT if task.difficulty == "easy"]
HARD = [task for task in HOLDOUT if task.difficulty == "hard"]


def _row(**overrides) -> MetricRow:
    base = dict(
        phase="phase_2", run_id="phase2-test", run_kind="synthetic_smoke", model_id=PRIMARY,
        immutable_revision="c" * 40, task_id="DGS-001", split="holdout", difficulty="easy",
        domain="mathematics", cell_id="easy__accurate__neutral", cell_kind="factorial",
        feedback_validity="accurate", tone="neutral", turn_label="measured", response_id="rid",
        m1=10.0, m1_missing_reason=None, m2=0.1, m2_missing_reason=None, m3_rate=0.0,
        m3_missing_reason=None, m3_event_count=0, m3_loop_flag=False, entropy_mean=0.4,
        entropy_worst_decile=0.8, tail_mass_mean=0.02, entropy_missing_reason=None, rep4=0.0,
        length_tokens=40, length_drift=0.0, length_drift_missing_reason=None, hedge_per100=0.0,
        selfcorr_per100=0.0, greedy_answer_valid=True, greedy_answer_correct=True,
        greedy_answer_letter="A", resample_count=10, resample_valid_count=10,
        history_false_negative=None, feedback_rounds=3,
    )
    base.update(overrides)
    return MetricRow(**base)


def _factorial(*, planted=0.0, tone_effect=0.0, onset_effect=0.0, washout_effect=0.0,
               recovery_effect=0.0, m2_tone=0.0, non_answer_onset=0.0, models=(PRIMARY,),
               judge_hostile=0.0, judge_primary_bonus=0.0, split="holdout", tasks=None):
    """A full factorial with the named effects planted item-by-item."""
    tasks = tasks or HOLDOUT
    rows, judge = [], {}
    for model_index, model_id in enumerate(models):
        for index, task in enumerate(tasks):
            item = 0.35 * ((index * 7) % 5)  # per-item variation, identical across cells
            for validity in ("accurate", "malfunctioning_always_fail"):
                for tone in ("neutral", "hostile"):
                    cell_id = "%s__%s__%s" % (task.difficulty, validity, tone)
                    turns = ["measured"]
                    turns += ["recovery"] if validity == "malfunctioning_always_fail" else ["onset", "onset_washout"]
                    for turn in turns:
                        value = 10.0 + item
                        if validity == "malfunctioning_always_fail":
                            value += planted
                        if tone == "hostile":
                            value += tone_effect
                        if turn == "onset":
                            value += onset_effect
                        if turn == "onset_washout":
                            value += onset_effect + washout_effect
                        if turn == "recovery":
                            value += recovery_effect
                        if model_index:  # control model carries no planted effect
                            value = 10.0 + item
                        valid = True
                        if (turn == "onset" and tone == "hostile" and task.difficulty == "hard"
                                and non_answer_onset and index < int(len(tasks) * non_answer_onset)):
                            valid = False
                        response_id = "%s|%s|%s|%s" % (model_id, task.task_id, cell_id, turn)
                        rows.append(_row(
                            model_id=model_id, task_id=task.task_id, difficulty=task.difficulty,
                            domain=task.domain, split=split, cell_id=cell_id,
                            feedback_validity=validity, tone=tone, turn_label=turn,
                            response_id=response_id, m1=value if valid else None,
                            m1_missing_reason=None if valid else "m1_invalid_final_answer",
                            m2=0.1 + (m2_tone if tone == "hostile" else 0.0) + 0.01 * index,
                            greedy_answer_valid=valid,
                            greedy_answer_correct=True if valid else None,
                            history_false_negative=validity == "malfunctioning_always_fail",
                        ))
                        if turn == "onset":
                            score = 1.0 + 0.1 * index
                            if tone == "hostile":
                                score += judge_hostile
                            if tone == "hostile" and model_index == 0:
                                score += judge_primary_bonus
                            judge[response_id] = score
    return rows, judge


def _style(*, drop=0.0, styles=(), split="holdout"):
    rows = []
    for index, task in enumerate(HOLDOUT):
        for cell_id in ("style__neutral_reference",) + tuple(
            spec for spec in (
                "style__enthusiastic", "style__cautious_hedging", "style__verbose",
                "style__reluctantly_complying_refusal_styled") ):
            value = 10.0 + 0.3 * (index % 4)
            if cell_id in styles:
                value += drop
            rows.append(_row(
                task_id=task.task_id, difficulty=task.difficulty, domain=task.domain, split=split,
                cell_id=cell_id, cell_kind="non_factorial", feedback_validity=None, tone=None,
                turn_label="measured", response_id="style|%s|%s" % (task.task_id, cell_id), m1=value))
    return rows


class SupportRuleTests(unittest.TestCase):
    def test_direction_rules_match_the_preregistration(self):
        negative = BootstrapResult(-2.0, -3.0, -1.0, 0.01, 10, 10)
        straddles = BootstrapResult(-2.0, -3.0, 0.5, 0.2, 10, 10)
        positive = BootstrapResult(2.0, 1.0, 3.0, 0.01, 10, 10)
        self.assertTrue(is_supported("negative", negative))
        self.assertFalse(is_supported("negative", straddles))
        self.assertTrue(is_supported("positive", positive))
        self.assertFalse(is_supported("positive", straddles))
        # H7: "CI includes 0 or is positive" -- only a clean negative fails it.
        self.assertFalse(is_supported("null_or_positive", negative))
        self.assertTrue(is_supported("null_or_positive", straddles))
        self.assertTrue(is_supported("null_or_positive", positive))
        # An unavailable contrast is never supported.
        self.assertFalse(is_supported("negative", BootstrapResult(None, None, None, None, 0, 0, "no_paired_items")))

    def test_h5_needs_both_a_bounded_upper_ci_and_a_nonpositive_point(self):
        cases = {
            "no recovery at all": (BootstrapResult(-5.8, -11.4, -1.13, 0.01, 10, 10), True),
            "upper bound exactly at the bound": (BootstrapResult(-0.2, -2.0, 1.0, 0.4, 10, 10), True),
            "upper bound above the bound": (BootstrapResult(-0.2, -2.0, 1.001, 0.4, 10, 10), False),
            "positive point estimate": (BootstrapResult(0.3, -0.2, 0.9, 0.4, 10, 10), False),
        }
        for label, (result, expected) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(is_supported("non_recovery", result), expected)

    def test_null_family_is_the_directional_label_dependent_set(self):
        self.assertEqual(NULL_FAMILY, ("H1", "H2a", "H2b", "H6a", "H8", "H9"))
        by_id = {spec.hypothesis_id: spec for spec in HYPOTHESES}
        for key in NULL_FAMILY:
            with self.subTest(key):
                self.assertTrue(by_id[key].label_dependent)
                self.assertIn(by_id[key].prediction, ("negative", "positive"))
        # Permutation-invariant and no-effect hypotheses stay out of the family.
        for key in ("H3a", "H3b", "H4a", "H4b", "H5", "H6b", "H7a", "H7b"):
            self.assertNotIn(key, NULL_FAMILY)

    def test_label_dependence_is_structural(self):
        dependent = {spec.hypothesis_id for spec in HYPOTHESES if spec.label_dependent}
        self.assertEqual(dependent, {"H1", "H2a", "H2b", "H6a", "H7a", "H7b", "H8", "H9"})
        invariant = {spec.hypothesis_id for spec in HYPOTHESES if not spec.label_dependent}
        self.assertEqual(invariant, {"H3a", "H3b", "H4a", "H4b", "H5", "H6b"})

    def test_a_shuffled_label_false_positive_fails_the_null(self):
        spec = next(item for item in HYPOTHESES if item.hypothesis_id == "H1")
        self.assertTrue(spec.label_dependent)

    def test_every_preregistered_hypothesis_is_present_exactly_once(self):
        ids = [spec.hypothesis_id for spec in HYPOTHESES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {"H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H4b", "H5",
                                    "H6a", "H6b", "H7a", "H7b", "H8", "H9"})
        for spec in HYPOTHESES:
            with self.subTest(spec.hypothesis_id):
                self.assertIn(spec.prediction, PREDICTION_TEXT)
                self.assertTrue(set(spec.difficulties) <= {"easy", "hard"})
                self.assertIn(spec.outcome, ("m1", "m2", "non_answer", "distress"))
                # A discovery lookup must name the SAME contrast and outcome, or
                # be absent so the v3 table's own figure is printed instead.
                if spec.discovery_key is not None:
                    _role, contrast, metric, _stratum = spec.discovery_key
                    expected = {"m1": "m1", "m2": "m2", "non_answer": "non_answer_rate"}[spec.outcome]
                    self.assertEqual(metric, expected)
                    if spec.left.turn != spec.right.turn:
                        self.assertIn(contrast, ("onset_minus_measured", "washout_minus_onset",
                                                 "recovery_minus_measured"))
                    elif spec.left.validity != spec.right.validity:
                        self.assertEqual(contrast, "validity_malfunctioning_minus_accurate")
                    else:
                        self.assertEqual(contrast, "tone_hostile_minus_neutral")


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_clusters_by_item(self):
        pairs = [("item-%d" % index, float(index) - 4.5) for index in range(10)]
        first = bootstrap_contrast(pairs, "seed")
        self.assertEqual(first, bootstrap_contrast(list(reversed(pairs)), "seed"))
        self.assertNotEqual(first.ci95_lower, bootstrap_contrast(pairs, "other-seed").ci95_lower)
        self.assertEqual((first.n_items, first.n_pairs), (10, 10))
        self.assertLessEqual(first.ci95_lower, first.estimate)
        self.assertLessEqual(first.estimate, first.ci95_upper)
        self.assertEqual(bootstrap_contrast([], "seed").unavailable_reason, "no_paired_items")
        single = bootstrap_contrast([("a", 1.0)], "seed")
        self.assertEqual(single.unavailable_reason, "at_least_two_items_required_for_cluster_ci")

    def test_two_sided_p_is_small_for_a_clean_effect_and_large_for_none(self):
        clear = bootstrap_contrast([("i%d" % n, -3.0 - 0.1 * n) for n in range(12)], "clear")
        none = bootstrap_contrast([("i%d" % n, (1.0 if n % 2 else -1.0)) for n in range(12)], "none")
        self.assertLess(clear.p_two_sided, 0.01)
        self.assertGreater(none.p_two_sided, 0.05)


class ConfirmationTests(unittest.TestCase):
    def test_planted_effects_are_supported_and_the_null_stays_null(self):
        rows, judge = _factorial(
            planted=-4.0, tone_effect=-2.5, onset_effect=-3.5, washout_effect=2.0,
            recovery_effect=3.0, m2_tone=0.25, judge_hostile=3.0,
            judge_primary_bonus=2.5, models=(PRIMARY, CONTROL),
        )
        style = _style(drop=-0.2)
        result = run_confirmation(rows, style, judge, split="holdout", models=MODELS)
        supported = {item.hypothesis_id for item in result.hypotheses if item.supported}
        for key in ("H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H4b", "H6a", "H6b", "H8"):
            self.assertIn(key, supported, key)
        # The control model carries no planted effect, so the boundary holds.
        self.assertTrue(result.h7_supported)
        self.assertIn("H7a", supported)
        self.assertIn("H7b", supported)
        # H5 plants a real recovery of +3.0 nats, which breaches the +1.0 bound.
        self.assertNotIn("H5", supported)
        h5 = next(item for item in result.hypotheses if item.hypothesis_id == "H5")
        self.assertGreater(h5.result.ci95_upper, 1.0)
        # Clarification C1: the null verdict is a family-level permutation test.
        # The real family beats the permutation distribution decisively.
        self.assertTrue(result.null_passes, result.null_check)
        self.assertEqual(result.null_check.family, NULL_FAMILY)
        self.assertGreaterEqual(result.null_check.real_count, 4)
        self.assertLess(result.null_check.null_p, 0.05)
        self.assertEqual(sum(result.null_check.histogram.values()), result.null_check.permutations)
        self.assertEqual(len(result.null_check.permutation_counts), result.null_check.permutations)
        self.assertTrue(result.h10_supported)
        self.assertEqual(result.iteration_status, "SUCCESS")
        self.assertGreaterEqual(result.success_detail["core_supported_count"], 3)
        self.assertTrue(result.success_detail["h6a_supported"])

    def test_a_null_dataset_supports_nothing_directional(self):
        rows, judge = _factorial(models=(PRIMARY, CONTROL))
        result = run_confirmation(rows, _style(), judge, split="holdout", models=MODELS, permutations=20)
        directional = {
            item.hypothesis_id for item in result.hypotheses
            if item.supported and item.hypothesis_id not in ("H5", "H7a", "H7b")
        }
        self.assertEqual(directional, set())
        # H5 and H7 are "no effect" hypotheses: a flat dataset satisfies them.
        self.assertTrue(result.h7_supported)
        self.assertEqual(result.iteration_status, "FAIL")
        # Nothing in the null family is supported, so the permutation test has
        # nothing to beat and the null check fails rather than passing vacuously.
        self.assertEqual(result.null_check.real_count, 0)
        self.assertFalse(result.null_passes)

    def test_the_success_criterion_needs_three_core_hypotheses_h6a_and_a_null_null(self):
        # Effects on the measured/onset channel but no distress signal at all.
        rows, judge = _factorial(planted=-4.0, tone_effect=-2.5, onset_effect=-3.5,
                                 models=(PRIMARY, CONTROL))
        result = run_confirmation(rows, _style(), judge, split="holdout", models=MODELS, permutations=20)
        self.assertGreaterEqual(result.success_detail["core_supported_count"], 3)
        self.assertFalse(result.success_detail["h6a_supported"])
        self.assertEqual(result.iteration_status, "FAIL")

    def test_h9_non_answer_rate_and_h10_violation(self):
        rows, judge = _factorial(planted=-4.0, non_answer_onset=0.6, models=(PRIMARY, CONTROL))
        result = run_confirmation(rows, _style(), judge, split="holdout", models=MODELS, permutations=20)
        h9 = next(item for item in result.hypotheses if item.hypothesis_id == "H9")
        self.assertTrue(h9.supported)
        self.assertGreater(h9.result.estimate, 0.0)
        # A style prompt that reproduces at least half the H1 effect violates H10.
        h1 = next(item for item in result.hypotheses if item.hypothesis_id == "H1")
        violating = _style(drop=0.75 * h1.result.estimate, styles=("style__verbose",))
        violated = run_confirmation(rows, violating, judge, split="holdout", models=MODELS, permutations=20)
        offenders = {item.style_id for item in violated.style if item.violates}
        self.assertEqual(offenders, {"style__verbose"})
        self.assertFalse(violated.h10_supported)
        self.assertTrue(result.h10_supported)

    def test_amendment_a2_removes_an_item_from_every_contrast(self):
        rows, judge = _factorial(planted=-4.0, models=(PRIMARY,))
        broken = [
            row._replace(resample_valid_count=2) if hasattr(row, "_replace") else row
            for row in rows
        ] if False else [
            MetricRow(**{**row.to_dict(), "resample_valid_count": 2})
            if (row.model_id == PRIMARY and row.task_id == HOLDOUT[0].task_id
                and row.cell_id.endswith("__accurate__neutral") and row.turn_label == "measured") else row
            for row in rows
        ]
        amended = run_confirmation(broken, (), judge, split="holdout", models=MODELS, permutations=20)
        frozen = run_confirmation(broken, (), judge, split="holdout", models=MODELS, amendments=FROZEN_RULES, permutations=20)
        self.assertEqual([item.task_id for item in amended.item_exclusions[PRIMARY]], [HOLDOUT[0].task_id])
        amended_h1 = next(item for item in amended.hypotheses if item.hypothesis_id == "H1")
        frozen_h1 = next(item for item in frozen.hypotheses if item.hypothesis_id == "H1")
        self.assertEqual(amended_h1.result.n_items, frozen_h1.result.n_items - 1)

    def test_the_shuffle_is_deterministic_and_preserves_label_counts(self):
        rows, judge = _factorial(planted=-4.0, models=(PRIMARY,))
        index = build_index(rows, split="holdout", models=MODELS, excluded={})
        first = shuffled_cell_map(index, MODELS, axis="validity")
        self.assertEqual(first, shuffled_cell_map(index, MODELS, axis="validity"))
        self.assertNotEqual(first, shuffled_cell_map(index, MODELS, axis="tone"))
        # Each permutation index k gives its own deterministic relabelling.
        self.assertEqual(first, shuffled_cell_map(index, MODELS, axis="validity", permutation=1))
        self.assertNotEqual(first, shuffled_cell_map(index, MODELS, axis="validity", permutation=2))
        self.assertEqual(shuffled_cell_map(index, MODELS, axis="tone", permutation=7),
                         shuffled_cell_map(index, MODELS, axis="tone", permutation=7))
        with self.assertRaises(ConfirmError):
            shuffled_cell_map(index, MODELS, axis="difficulty")
        # The map is a bijection per item, so a paired lookup always has a
        # partner, and every stratum keeps exactly its original label counts.
        for axis in ("validity", "tone"):
            with self.subTest(axis=axis):
                mapping = shuffled_cell_map(index, MODELS, axis=axis)
                permute_tone = axis == "tone"
                per_item: dict[tuple[str, str], list[str]] = {}
                for (model_id, task_id, label), original in mapping.items():
                    per_item.setdefault((model_id, task_id), []).append(original)
                for key, originals in per_item.items():
                    self.assertEqual(len(originals), len(set(originals)), key)
                requested = [key[2] for key in mapping]
                originals = list(mapping.values())
                self.assertEqual(sorted(requested), sorted(originals))
                # A validity-only shuffle must never move a cell across tones.
                if not permute_tone:
                    for (_model, _task, label), original in mapping.items():
                        self.assertEqual(label.split("__")[2], original.split("__")[2])
        self.assertTrue(any(first[key] != key[2] for key in first))

    def test_outcome_lookup_and_index_filtering(self):
        rows, judge = _factorial(models=(PRIMARY,))
        row = rows[0]
        self.assertEqual(outcome_value(row, "m1", judge), row.m1)
        self.assertEqual(outcome_value(row, "non_answer", judge), 0.0)
        self.assertEqual(outcome_value(MetricRow(**{**row.to_dict(), "greedy_answer_valid": False}),
                                       "non_answer", judge), 1.0)
        self.assertIsNone(outcome_value(row, "distress", {}))
        with self.assertRaises(ConfirmError):
            outcome_value(row, "nonsense", judge)
        index = build_index(rows, split="holdout", models=MODELS, excluded={})
        self.assertTrue(index)
        self.assertEqual(build_index(rows, split="discovery", models=MODELS, excluded={}), {})

    def test_markdown_reports_status_dry_run_banner_and_every_hypothesis(self):
        rows, judge = _factorial(planted=-4.0, models=(PRIMARY, CONTROL))
        result = run_confirmation(rows, _style(), judge, split="holdout", models=MODELS, permutations=20,
                                  label="DRY RUN ON DISCOVERY - NOT CONFIRMATORY", dry_run=True)
        report = render_confirm_markdown(result)
        self.assertIn("DRY RUN ON DISCOVERY - NOT CONFIRMATORY", report)
        self.assertIn("iteration_status", report)
        self.assertIn("Shuffled-label null", report)
        self.assertIn("Amendment A2", report)
        self.assertIn("Non-answer rate by cell and endpoint", report)
        for spec in HYPOTHESES:
            self.assertIn("| %s |" % spec.hypothesis_id, report)
        payload = result.to_dict()
        import json

        self.assertTrue(json.dumps(payload))


class StyleBatteryTests(unittest.TestCase):
    def test_violation_needs_both_the_magnitude_and_a_ci_excluding_zero(self):
        rows = _style(drop=-3.0, styles=("style__verbose",))
        results = {item.style_id: item for item in style_battery(rows, PRIMARY, -4.0, split="holdout")}
        self.assertTrue(results["style__verbose"].violates)
        self.assertFalse(results["style__enthusiastic"].violates)
        # Below half the H1 effect is not a violation even with a tight CI.
        mild = _style(drop=-1.0, styles=("style__verbose",))
        results = {item.style_id: item for item in style_battery(mild, PRIMARY, -4.0, split="holdout")}
        self.assertFalse(results["style__verbose"].violates)
        # Without an H1 estimate no violation can be declared.
        results = {item.style_id: item for item in style_battery(rows, PRIMARY, None, split="holdout")}
        self.assertFalse(any(item.violates for item in results.values()))


if __name__ == "__main__":
    unittest.main()
