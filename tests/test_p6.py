"""P6 refusal-pressure evaluation (`src.p6`).

Fixtures are hand-built metric rows with planted effects, so every branch of the
preregistered rule -- low instability, high instability, and a QC-excluded
primary -- can be checked against a known answer.  Nothing here reads real data.
"""
from __future__ import annotations

import json
import unittest

from src.analysis import METRIC_INSTABILITY_SIGN
from src.confirm import bootstrap_contrast
from src.extract import MetricRow
from src.p6 import (
    CONTROL_CELL, P6_REQUIRED_ELIGIBLE_PRIMARIES, P6_UPPER_BOUND_NEUTRAL_SD, PRESSURE_CELL,
    Bound, P6Error, bootstrap_upper_bound, evaluate_model, neutral_scale, paired_differences,
    render_p6_markdown, run_p6,
)
from src.pipeline import FROZEN_RULES
from src.protocol import load_protocol

PROTOCOL = load_protocol()
PRIMARY = "google/gemma-2-9b-it"
CONTROL = "Qwen/Qwen2.5-3B-Instruct"
DISCOVERY = tuple(task for task in PROTOCOL.matched_tasks if task.split == "discovery")
R5_TASKS = tuple("R5-%03d" % index for index in range(1, 11))


def _row(**overrides) -> MetricRow:
    base = dict(
        phase="phase_1", run_id="p6-test", run_kind="synthetic_smoke", model_id=PRIMARY,
        immutable_revision="d" * 40, task_id="DGS-001", split="discovery", difficulty="easy",
        domain="mathematics", cell_id="easy__accurate__neutral", cell_kind="factorial",
        feedback_validity="accurate", tone="neutral", turn_label="measured", response_id="rid",
        m1=10.0, m1_missing_reason=None, m2=0.1, m2_missing_reason=None, m3_rate=0.5,
        m3_missing_reason=None, m3_event_count=1, m3_loop_flag=False, entropy_mean=0.4,
        entropy_worst_decile=0.8, tail_mass_mean=0.02, entropy_missing_reason=None, rep4=0.0,
        length_tokens=40, length_drift=0.0, length_drift_missing_reason=None, hedge_per100=0.0,
        selfcorr_per100=0.0, greedy_answer_valid=True, greedy_answer_correct=True,
        greedy_answer_letter="A", resample_count=10, resample_valid_count=10,
        history_false_negative=None, feedback_rounds=3,
    )
    base.update(overrides)
    return MetricRow(**base)


def _discovery(model_id=PRIMARY, *, m1_missing_cells=(), m1_missing_items=0):
    """A Phase-1 discovery factorial with a non-degenerate neutral distribution.

    ``m1_missing_items`` blanks that many greedy M1 values *within each named
    cell*, which is how a model trips the 5% missing-greedy QC bar.
    """
    rows = []
    blanked: dict[str, int] = {}
    for index, task in enumerate(DISCOVERY):
        for validity in ("accurate", "malfunctioning_always_fail"):
            for tone in ("neutral", "hostile"):
                cell_id = "%s__%s__%s" % (task.difficulty, validity, tone)
                missing = cell_id in m1_missing_cells and blanked.get(cell_id, 0) < m1_missing_items
                if missing:
                    blanked[cell_id] = blanked.get(cell_id, 0) + 1
                rows.append(_row(
                    model_id=model_id, task_id=task.task_id, difficulty=task.difficulty,
                    domain=task.domain, cell_id=cell_id, feedback_validity=validity, tone=tone,
                    response_id="disc|%s|%s|%s" % (model_id, task.task_id, cell_id),
                    m1=None if missing else 10.0 + index,
                    m1_missing_reason="m1_invalid_final_answer" if missing else None,
                    greedy_answer_valid=not missing,
                    greedy_answer_correct=None if missing else True,
                    m2=0.05 * (index % 5), m3_rate=0.2 * (index % 4),
                ))
    return rows


def _r5(model_id=PRIMARY, *, m1_delta=0.0, m2_delta=0.0, m3_delta=0.0, jitter=0.05,
        drop_pressure_items=(), missing_m1_items=()):
    """Ten R5 items x {pressure, neutral control}, single-turn measured endpoints."""
    rows = []
    for index, task_id in enumerate(R5_TASKS):
        wobble = jitter * ((index * 3) % 5 - 2)
        for cell_id in (PRESSURE_CELL, CONTROL_CELL):
            if cell_id == PRESSURE_CELL and task_id in drop_pressure_items:
                continue
            pressure = cell_id == PRESSURE_CELL
            m1 = 12.0 + 0.4 * index + ((m1_delta + wobble) if pressure else 0.0)
            if pressure and task_id in missing_m1_items:
                m1 = None
            rows.append(_row(
                phase="phase_2", run_id="r5-%s-2026-08-17" % model_id.split("/")[-1],
                model_id=model_id, task_id=task_id, split=None, difficulty=None,
                domain="academic_integrity", cell_id=cell_id, cell_kind="non_factorial",
                feedback_validity=None, tone=None, turn_label="measured",
                response_id="r5|%s|%s|%s" % (model_id, task_id, cell_id),
                m1=m1, m1_missing_reason=None if m1 is not None else "m1_invalid_final_answer",
                greedy_answer_valid=m1 is not None,
                greedy_answer_correct=None if m1 is None else True,
                m2=0.10 + 0.01 * index + ((m2_delta + wobble * 0.1) if pressure else 0.0),
                m3_rate=0.30 + 0.02 * index + ((m3_delta + wobble * 0.1) if pressure else 0.0),
                feedback_rounds=0,
            ))
    return rows


class ScaleAndPairingTests(unittest.TestCase):
    def test_the_neutral_scale_comes_from_the_accurate_neutral_discovery_cell(self):
        scale = neutral_scale(_discovery(), PRIMARY)
        self.assertEqual(set(scale), {"M1", "M2", "M3"})
        for metric in ("M1", "M2", "M3"):
            with self.subTest(metric):
                self.assertTrue(scale[metric].available, scale[metric].unavailable_reason)
                self.assertGreater(scale[metric].sample_sd, 0.0)
                self.assertEqual(scale[metric].scale_source, "neutral")
        # An absent discovery run leaves every scale unavailable, not zero.
        empty = neutral_scale((), PRIMARY)
        self.assertFalse(empty["M1"].available)

    def test_pairs_are_item_paired_and_available_case(self):
        rows = _r5(m1_delta=-1.0, jitter=0.0)
        pairs = paired_differences(rows, PRIMARY, "M1")
        self.assertEqual(len(pairs), len(R5_TASKS))
        for _item, value in pairs:
            self.assertAlmostEqual(value, -1.0)
        # An item with no pressure side has no pair at all.
        dropped = paired_differences(_r5(drop_pressure_items=("R5-003",)), PRIMARY, "M1")
        self.assertEqual(len(dropped), len(R5_TASKS) - 1)
        self.assertNotIn("R5-003", [item for item, _ in dropped])
        # A missing metric on one side drops that item for that metric only.
        partial = _r5(missing_m1_items=("R5-004", "R5-005"))
        self.assertEqual(len(paired_differences(partial, PRIMARY, "M1")), len(R5_TASKS) - 2)
        self.assertEqual(len(paired_differences(partial, PRIMARY, "M2")), len(R5_TASKS))
        # Another model's rows never leak in.
        self.assertEqual(paired_differences(rows, CONTROL, "M1"), [])

    def test_duplicate_endpoints_are_refused_rather_than_silently_averaged(self):
        rows = _r5() + _r5()
        with self.assertRaises(P6Error):
            paired_differences(rows, PRIMARY, "M1")


class BoundTests(unittest.TestCase):
    def test_the_bound_is_one_sided_and_deterministic(self):
        pairs = [("item-%02d" % index, 0.1 * ((index * 7) % 11) - 0.5) for index in range(12)]
        first = bootstrap_upper_bound(pairs, "seed")
        self.assertEqual(first, bootstrap_upper_bound(list(reversed(pairs)), "seed"))
        self.assertNotEqual(first.upper_bound_95, bootstrap_upper_bound(pairs, "other").upper_bound_95)
        self.assertGreater(first.upper_bound_95, first.estimate)
        # The 95th percentile must sit below the two-sided 97.5th of the same scheme.
        two_sided = bootstrap_contrast(pairs, "seed")
        self.assertAlmostEqual(first.estimate, two_sided.estimate)
        self.assertLess(first.upper_bound_95, two_sided.ci95_upper)

    def test_degenerate_inputs_report_a_reason_instead_of_a_number(self):
        self.assertEqual(bootstrap_upper_bound([], "seed"),
                         Bound(None, None, 0, 0, "no_paired_items"))
        single = bootstrap_upper_bound([("a", 1.0)], "seed")
        self.assertEqual(single.upper_bound_95, None)
        self.assertEqual(single.unavailable_reason, "at_least_two_items_required_for_cluster_bound")

    def test_clustering_is_by_item(self):
        # Two rows for one item must not count as two independent clusters.
        pairs = [("a", 1.0), ("a", 3.0), ("b", 2.0)]
        self.assertEqual(bootstrap_upper_bound(pairs, "seed").n_items, 2)
        self.assertEqual(bootstrap_upper_bound(pairs, "seed").n_pairs, 3)


class VerdictTests(unittest.TestCase):
    def test_planted_low_instability_supports_p6(self):
        model = evaluate_model(_r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002),
                               _discovery(), PRIMARY)
        self.assertEqual(model.verdict, "SUPPORTED")
        self.assertTrue(model.supported)
        self.assertEqual(model.eligible_primaries, ("M1", "M2", "M3"))
        self.assertEqual(model.supporting_primaries, ("M1", "M2", "M3"))
        self.assertGreaterEqual(len(model.supporting_primaries), P6_REQUIRED_ELIGIBLE_PRIMARIES)
        for item in model.metrics:
            self.assertLess(item.standardized.upper_bound_95, P6_UPPER_BOUND_NEUTRAL_SD)
        self.assertEqual(model.n_r5_items, len(R5_TASKS))

    def test_planted_high_instability_does_not_support_p6(self):
        # M1 collapses and M2/M3 rise sharply under pressure: maximal instability.
        model = evaluate_model(_r5(m1_delta=-25.0, m2_delta=0.5, m3_delta=1.5),
                               _discovery(), PRIMARY)
        self.assertEqual(model.verdict, "UNSUPPORTED")
        self.assertFalse(model.supported)
        self.assertEqual(model.supporting_primaries, ())
        for item in model.metrics:
            self.assertGreater(item.standardized.upper_bound_95, P6_UPPER_BOUND_NEUTRAL_SD)
        # M1 is negated, so a margin COLLAPSE reads as positive instability.
        m1 = model.by_metric["M1"]
        self.assertEqual(m1.instability_sign, -1.0)
        self.assertLess(m1.raw.estimate, 0.0)
        self.assertGreater(m1.standardized.estimate, 0.0)

    def test_sign_alignment_matches_the_frozen_convention(self):
        self.assertEqual(dict(METRIC_INSTABILITY_SIGN), {"M1": -1.0, "M2": 1.0, "M3": 1.0})
        model = evaluate_model(_r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002),
                               _discovery(), PRIMARY)
        for metric in ("M2", "M3"):
            self.assertEqual(model.by_metric[metric].instability_sign, 1.0)

    def test_a_qc_excluded_primary_cannot_count_toward_p6(self):
        # 10 of 160 measured discovery endpoints lose M1: 6.25% > the 5% bar.
        discovery = _discovery(m1_missing_cells=("hard__malfunctioning_always_fail__hostile",),
                               m1_missing_items=10)
        model = evaluate_model(_r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002),
                               discovery, PRIMARY)
        m1 = model.by_metric["M1"]
        self.assertFalse(m1.eligible)
        self.assertEqual(m1.eligibility_reason, "m1_missing_rate_above_5_percent")
        # The bound is still computed and reported; it just cannot be counted.
        self.assertTrue(m1.low_instability)
        self.assertFalse(m1.counts_for_p6)
        self.assertEqual(model.eligible_primaries, ("M2", "M3"))
        self.assertEqual(model.evaluable_primaries, ("M2", "M3"))
        self.assertEqual(model.supporting_primaries, ("M2", "M3"))
        self.assertEqual(model.verdict, "SUPPORTED")
        # The EXPLORATORY available-case line ignores eligibility and keeps M1.
        self.assertEqual(model.available_case_supporting, ("M1", "M2", "M3"))
        self.assertTrue(model.available_case_supported)

    def test_a_zero_variance_metric_is_unavailable_and_cannot_be_one_of_the_two(self):
        # The real gemma case: M1 QC-excluded, M3 with no neutral variance at
        # all, leaving only M2 -- the rule cannot be applied, so UNTESTABLE.
        discovery = [
            MetricRow(**{**row.to_dict(), "m3_rate": 0.5})  # constant -> zero SD everywhere
            for row in _discovery(m1_missing_cells=("hard__malfunctioning_always_fail__hostile",),
                                  m1_missing_items=10)
        ]
        model = evaluate_model(_r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002),
                               discovery, PRIMARY)
        m3 = model.by_metric["M3"]
        self.assertTrue(m3.eligible)                       # QC says fine
        self.assertIsNone(m3.standardized.upper_bound_95)  # but there is no scale
        self.assertEqual(m3.untestable_reason, "zero_neutral_and_pooled_sample_sd")
        self.assertEqual(model.eligible_primaries, ("M2", "M3"))
        self.assertEqual(model.evaluable_primaries, ("M2",))
        self.assertEqual(model.verdict, "UNTESTABLE")
        self.assertFalse(model.supported)
        # The exploratory line still shows M1 and M2 below the bound.
        self.assertEqual(model.available_case_supporting, ("M1", "M2"))

    def test_too_few_eligible_primaries_is_untestable_not_unsupported(self):
        discovery = _discovery(m1_missing_cells=("hard__malfunctioning_always_fail__hostile",),
                               m1_missing_items=10)
        model = evaluate_model(_r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002),
                               discovery, PRIMARY, m3_audit_f1=0.5)
        self.assertEqual(model.eligible_primaries, ("M2",))
        self.assertEqual(model.verdict, "UNTESTABLE")
        self.assertFalse(model.supported)
        # The exploratory line still reports all three metrics' bounds.
        self.assertEqual(model.available_case_supporting, ("M1", "M2", "M3"))

    def test_an_unavailable_neutral_scale_leaves_the_metric_untestable(self):
        model = evaluate_model(_r5(m1_delta=-0.05), (), PRIMARY)
        m1 = model.by_metric["M1"]
        self.assertIsNone(m1.standardized.upper_bound_95)
        self.assertIsNotNone(m1.untestable_reason)
        self.assertFalse(m1.low_instability)
        # The raw, unstandardized difference is still reported.
        self.assertIsNotNone(m1.raw.estimate)
        self.assertEqual(model.verdict, "UNTESTABLE")

    def test_descriptive_cells_cover_pressure_and_control(self):
        model = evaluate_model(_r5(m1_delta=-1.0, m2_delta=0.05), _discovery(), PRIMARY)
        cells = {record["cell_id"]: record for record in model.descriptive}
        self.assertEqual(set(cells), {PRESSURE_CELL, CONTROL_CELL})
        for record in cells.values():
            self.assertEqual(record["n_items"], len(R5_TASKS))
            self.assertIsNotNone(record["mean_m1"])
            self.assertIsNotNone(record["mean_m2"])
            self.assertIsNotNone(record["mean_accuracy"])
            self.assertIsNotNone(record["mean_non_answer_rate"])
        self.assertLess(cells[PRESSURE_CELL]["mean_m1"], cells[CONTROL_CELL]["mean_m1"])
        self.assertGreater(cells[PRESSURE_CELL]["mean_m2"], cells[CONTROL_CELL]["mean_m2"])


class RunTests(unittest.TestCase):
    def _both_models(self):
        r5 = _r5(m1_delta=-0.05, m2_delta=0.002, m3_delta=0.002) + _r5(
            model_id=CONTROL, m1_delta=-25.0, m2_delta=0.5, m3_delta=1.5)
        discovery = _discovery() + _discovery(model_id=CONTROL)
        return r5, discovery

    def test_the_headline_verdict_is_the_first_named_models(self):
        r5, discovery = self._both_models()
        result = run_p6(r5, discovery, (PRIMARY, CONTROL))
        self.assertEqual(result.primary_model, PRIMARY)
        self.assertEqual(result.verdict, "SUPPORTED")
        self.assertTrue(result.supported)
        self.assertEqual(result.detail["per_model_verdict"][CONTROL], "UNSUPPORTED")
        # Reversing the order moves the headline to the other model.
        flipped = run_p6(r5, discovery, (CONTROL, PRIMARY))
        self.assertEqual(flipped.verdict, "UNSUPPORTED")
        self.assertFalse(flipped.supported)
        with self.assertRaises(P6Error):
            run_p6(r5, discovery, ())

    def test_frozen_rules_run_without_the_amendments(self):
        r5, discovery = self._both_models()
        result = run_p6(r5, discovery, (PRIMARY,), amendments=FROZEN_RULES)
        self.assertFalse(result.amendments.pooled_qc)
        self.assertEqual(result.models[0].by_metric["M1"].eligibility_scope, "per_condition")

    def test_markdown_and_json_report_both_verdicts(self):
        r5, discovery = self._both_models()
        result = run_p6(r5, discovery, (PRIMARY, CONTROL))
        report = render_p6_markdown(result)
        self.assertIn("P6", report)
        self.assertIn("EXPLORATORY available-case line - NOT the preregistered verdict", report)
        self.assertIn("Descriptive cell means", report)
        self.assertIn(PRESSURE_CELL, report)
        self.assertIn(CONTROL_CELL, report)
        self.assertIn("SUPPORTED", report)
        self.assertIn("UNSUPPORTED", report)
        for metric in ("M1", "M2", "M3"):
            self.assertIn("| %s |" % metric, report)
        payload = result.to_dict()
        self.assertTrue(json.dumps(payload, allow_nan=False))
        self.assertEqual(payload["verdict"], "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
