"""M1 missing-data sensitivity analysis.

Fixtures are hand-built metric rows and item pairs with known answers, so every
imputation rule, bound and tipping-point search can be checked arithmetically.
Nothing here reads a committed result table or the raw data.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.confirm import BOOTSTRAP_KEY, BootstrapResult, bootstrap_contrast
from src.extension import EXTENSION_BOOTSTRAP_KEY
from src.extract import MetricRow
from src.missingness import (
    AVAILABLE_CASE, CANDIDATE_ABSENT, CONTRASTS_BY_ID, DELTA, MANSKI_LOWER, MANSKI_UPPER,
    NON_ANSWER, OTHER_MISSING, PER_ITEM_COLUMNS, TIPPING_TOLERANCE, ZERO_IMPUTATION, Contrast,
    ItemPair, MissingnessError, analyse_contrast, build_m1_index, build_pairs, cell_missingness,
    classify, count_missingness, load_discovery_published, load_extension_published, load_holdout_published,
    missing_kind, missingness_seed, observed_support, paired_differences, per_item_rows,
    published_seed, render_markdown, run_missingness, tipping_point, treatment_fills,
)

MODEL = "test/model"
H1 = CONTRASTS_BY_ID["H1"]


def _row(**overrides) -> MetricRow:
    base = dict(
        phase="phase_2", run_id="test-run", run_kind="synthetic_smoke", model_id=MODEL,
        immutable_revision="c" * 40, task_id="T1", split="holdout", difficulty="easy",
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


def _cell(task_id, validity, m1, reason=None, response="rid"):
    return _row(task_id=task_id, cell_id="easy__%s__neutral" % validity, feedback_validity=validity,
                m1=m1, m1_missing_reason=reason, response_id="%s-%s" % (task_id, response),
                greedy_answer_valid=m1 is not None or reason != "m1_invalid_final_answer")


# T1/T2 complete; T3 loses the treated cell to a non-answer; T4 loses the reference
# cell to a top-20 truncation; T5 has no treated endpoint row at all.
FIXTURE = (
    _cell("T1", "malfunctioning_always_fail", 2.0), _cell("T1", "accurate", 10.0),
    _cell("T2", "malfunctioning_always_fail", 4.0), _cell("T2", "accurate", 12.0),
    _cell("T3", "malfunctioning_always_fail", None, "m1_invalid_final_answer"),
    _cell("T3", "accurate", 8.0),
    _cell("T4", "malfunctioning_always_fail", 6.0),
    _cell("T4", "accurate", None, "m1_candidate_absent_C"),
    _cell("T5", "accurate", 11.0),
)


def _pairs(values, *, direction_sign=1.0):
    """``values`` of ``None`` become a missing treated cell; the reference is 10."""
    out = []
    for index, value in enumerate(values):
        treated = None if value is None else direction_sign * value + 10.0
        out.append(ItemPair("T%02d" % index, "easy", treated, 10.0,
                            None if treated is not None else NON_ANSWER, None))
    return tuple(out)


class MissingKindTests(unittest.TestCase):
    def test_classifies_every_reason_the_extractor_can_emit(self):
        self.assertIsNone(missing_kind(None))
        self.assertIsNone(missing_kind(""))
        self.assertEqual(missing_kind("m1_invalid_final_answer"), NON_ANSWER)
        self.assertEqual(missing_kind("m1_candidate_absent_C"), CANDIDATE_ABSENT)
        self.assertEqual(missing_kind("m1_option_token_mismatch"), OTHER_MISSING)


class TreatmentFillTests(unittest.TestCase):
    def test_available_case_drops_and_zero_imputes_both_sides(self):
        self.assertEqual(treatment_fills(AVAILABLE_CASE), (None, None))
        self.assertEqual(treatment_fills(ZERO_IMPUTATION), (0.0, 0.0))

    def test_bounds_cross_the_support(self):
        self.assertEqual(treatment_fills(MANSKI_LOWER, support=(8.0, 12.0)), (8.0, 12.0))
        self.assertEqual(treatment_fills(MANSKI_UPPER, support=(8.0, 12.0)), (12.0, 8.0))

    def test_delta_leaves_the_reference_at_zero(self):
        self.assertEqual(treatment_fills(DELTA, delta=3.5), (3.5, 0.0))

    def test_rejects_missing_or_inverted_inputs(self):
        for call in (lambda: treatment_fills(MANSKI_LOWER),
                     lambda: treatment_fills(DELTA),
                     lambda: treatment_fills("nonsense"),
                     lambda: treatment_fills(MANSKI_UPPER, support=(12.0, 8.0))):
            with self.assertRaises(MissingnessError):
                call()


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.index = build_m1_index(FIXTURE, model_id=MODEL, split="holdout")
        self.pairs = build_pairs(self.index, H1)

    def test_pairs_keep_items_a_missing_m1_would_drop(self):
        self.assertEqual([pair.task_id for pair in self.pairs], ["T1", "T2", "T3", "T4"])
        self.assertEqual([pair.complete for pair in self.pairs], [True, True, False, False])

    def test_an_item_without_both_endpoint_rows_is_not_pairable(self):
        counts = count_missingness(self.index, H1, self.pairs)
        self.assertEqual(counts.n_items_in_stratum, 5)
        self.assertEqual((counts.n_pairable, counts.n_endpoint_absent, counts.n_available), (4, 1, 2))

    def test_missing_kinds_are_counted_apart(self):
        counts = count_missingness(self.index, H1, self.pairs)
        self.assertEqual((counts.n_treated_missing, counts.treated_non_answer,
                          counts.treated_candidate_absent), (1, 1, 0))
        self.assertEqual((counts.n_reference_missing, counts.reference_non_answer,
                          counts.reference_candidate_absent), (1, 0, 1))
        self.assertEqual(counts.n_both_missing, 0)

    def test_exclusions_remove_an_item_entirely(self):
        index = build_m1_index(FIXTURE, model_id=MODEL, split="holdout", excluded=("T1",))
        self.assertEqual([pair.task_id for pair in build_pairs(index, H1)], ["T2", "T3", "T4"])

    def test_other_models_and_splits_are_not_pooled(self):
        self.assertEqual(build_m1_index(FIXTURE, model_id="other/model", split="holdout"), {})
        self.assertEqual(build_m1_index(FIXTURE, model_id=MODEL, split="discovery"), {})


class PairedDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.pairs = build_pairs(build_m1_index(FIXTURE, model_id=MODEL, split="holdout"), H1)

    def _mean(self, **fills):
        values = [value for _task, value in paired_differences(self.pairs, **fills)]
        return sum(values) / len(values), len(values)

    def test_available_case_drops_incomplete_pairs(self):
        self.assertEqual(self._mean(treated_fill=None, reference_fill=None), (-8.0, 2))

    def test_zero_imputation_uses_every_pairable_item(self):
        # T1 -8, T2 -8, T3 0-8 = -8, T4 6-0 = +6.
        self.assertEqual(self._mean(treated_fill=0.0, reference_fill=0.0), (-4.5, 4))

    def test_the_bounds_bracket_the_imputed_estimate(self):
        low = self._mean(treated_fill=8.0, reference_fill=12.0)   # T3 0, T4 -6
        high = self._mean(treated_fill=12.0, reference_fill=8.0)  # T3 +4, T4 -2
        self.assertEqual(low, (-5.5, 4))
        self.assertEqual(high, (-3.5, 4))
        zero, _ = self._mean(treated_fill=0.0, reference_fill=0.0)
        self.assertLess(low[0], zero)
        self.assertGreater(high[0], zero)

    def test_delta_zero_is_exactly_zero_imputation(self):
        treated_fill, reference_fill = treatment_fills(DELTA, delta=0.0)
        self.assertEqual(paired_differences(self.pairs, treated_fill=treated_fill,
                                            reference_fill=reference_fill),
                         paired_differences(self.pairs, treated_fill=0.0, reference_fill=0.0))


class CellMissingnessTests(unittest.TestCase):
    def test_counts_every_endpoint_and_names_the_reason_it_is_missing(self):
        cells = {(item.cell_id, item.turn_label): item
                 for item in cell_missingness(FIXTURE, model_id=MODEL, split="holdout")}
        treated = cells[("easy__malfunctioning_always_fail__neutral", "measured")]
        self.assertEqual((treated.n_endpoints, treated.n_observed, treated.n_non_answer,
                          treated.n_candidate_absent, treated.n_other), (4, 3, 1, 0, 0))
        reference = cells[("easy__accurate__neutral", "measured")]
        self.assertEqual((reference.n_endpoints, reference.n_observed, reference.n_candidate_absent),
                         (5, 4, 1))
        self.assertAlmostEqual(reference.mean_m1, 10.25)

    def test_the_report_carries_the_per_cell_table(self):
        report = run_missingness({"holdout": FIXTURE}, models={"primary": MODEL}, contrasts=(H1,))
        self.assertTrue(report.cells)
        self.assertIn("Non-answers per cell", render_markdown(report))


class SupportTests(unittest.TestCase):
    def test_support_is_the_neutral_accurate_measured_distribution(self):
        self.assertEqual(observed_support(FIXTURE, model_id=MODEL, split="holdout"), (8.0, 12.0, 4))

    def test_support_honours_item_exclusions(self):
        self.assertEqual(observed_support(FIXTURE, model_id=MODEL, split="holdout", excluded=("T3",)),
                         (10.0, 12.0, 3))

    def test_no_baseline_rows_gives_no_support(self):
        self.assertEqual(observed_support((), model_id=MODEL, split="holdout"), (None, None, 0))


class TippingPointTests(unittest.TestCase):
    # Eight items at -6 nats and two missing treated cells: a clear effect that a
    # large enough imputed margin must be able to erase.
    PAIRS = _pairs([-6.0] * 8 + [None, None])
    SEED = "tipping-test"

    def _result(self, delta):
        treated_fill, reference_fill = treatment_fills(DELTA, delta=delta)
        return bootstrap_contrast(
            paired_differences(self.PAIRS, treated_fill=treated_fill, reference_fill=reference_fill),
            self.SEED)

    def test_finds_the_smallest_delta_whose_ci_includes_zero(self):
        found = tipping_point(self.PAIRS, self.SEED)
        self.assertIsNone(found.reason)
        self.assertGreater(found.delta, 0.0)
        at = self._result(found.delta)
        self.assertLessEqual(at.ci95_lower, 0.0)
        self.assertGreaterEqual(at.ci95_upper, 0.0)
        below = self._result(found.delta - 4.0 * TIPPING_TOLERANCE)
        self.assertLess(below.ci95_upper, 0.0)

    def test_the_search_is_monotone_above_the_tipping_point(self):
        found = tipping_point(self.PAIRS, self.SEED)
        for step in (1.0, 10.0, 100.0):
            above = self._result(found.delta + step)
            self.assertLessEqual(above.ci95_lower, 0.0)

    def test_counts_the_missing_values_it_imputed(self):
        found = tipping_point(self.PAIRS, self.SEED)
        self.assertEqual((found.n_missing_treated, found.n_missing_reference), (2, 0))

    def test_no_missing_treated_value_leaves_delta_undefined(self):
        found = tipping_point(_pairs([-6.0] * 10), self.SEED)
        self.assertIsNone(found.delta)
        self.assertEqual(found.reason, "no_missing_treated_values")

    def test_an_already_null_contrast_tips_at_zero(self):
        found = tipping_point(_pairs([-6.0, 6.0, -6.0, 6.0, None]), self.SEED)
        self.assertEqual(found.delta, 0.0)
        self.assertEqual(found.reason, "ci_already_includes_zero_at_zero_imputation")

    def test_a_positive_contrast_tips_downwards(self):
        found = tipping_point(_pairs([-6.0] * 18 + [None, None], direction_sign=-1.0),
                              self.SEED, direction="positive")
        self.assertIsNone(found.reason)
        self.assertLess(found.delta, 0.0)

    def test_empty_and_invalid_inputs_are_reported_not_guessed(self):
        self.assertEqual(tipping_point((), self.SEED).reason, "no_pairable_items")
        with self.assertRaises(MissingnessError):
            tipping_point(self.PAIRS, self.SEED, direction="sideways")

    def test_an_unerasable_effect_hits_the_search_limit(self):
        # One missing value among a hundred cannot move the mean far enough.
        pairs = _pairs([-6.0] * 100 + [None])
        found = tipping_point(pairs, self.SEED, limit=8.0)
        self.assertIsNone(found.delta)
        self.assertTrue(found.reason.startswith("no_tipping_point_within"))


class _Planted:
    """A treatment outcome with a planted interval; ``classify`` reads nothing else."""

    def __init__(self, estimate, lower, upper):
        self.result = BootstrapResult(estimate, lower, upper, 0.0, 3, 3, None)


class ClassifyTests(unittest.TestCase):
    # NB: ``_outcome`` is taken by unittest.TestCase itself, hence ``_planted``.
    def _planted(self, estimate, lower, upper):
        return _Planted(estimate, lower, upper)

    def test_every_verdict_is_reachable(self):
        negative = self._planted(-5.0, -8.0, -2.0)
        null = self._planted(-1.0, -3.0, 1.0)
        positive = self._planted(5.0, 2.0, 8.0)
        self.assertEqual(classify("negative", negative, negative, negative, negative), "robust_bounded")
        self.assertEqual(classify("negative", negative, negative, negative, null), "robust_unbounded")
        self.assertEqual(classify("negative", negative, null, negative, null), "weakened")
        self.assertEqual(classify("negative", negative, positive, negative, positive), "flips")
        self.assertEqual(classify("negative", null, negative, negative, negative), "available_case_null")
        self.assertEqual(classify("negative", positive, negative, negative, negative),
                         "available_case_opposite")
        self.assertEqual(classify("negative", None, negative, negative, negative), "unavailable")


class SeedTests(unittest.TestCase):
    def test_discovery_uses_the_published_exploratory_key(self):
        self.assertEqual(published_seed(H1, split="discovery", model_id=MODEL, role="primary"),
                         "DGS-AC1-EXPLORATORY-v1|test/model|validity_malfunctioning_minus_accurate|m1|easy|neutral")

    def test_holdout_uses_the_frozen_confirmatory_key_and_the_role_s_hypothesis(self):
        self.assertEqual(published_seed(H1, split="holdout", model_id=MODEL, role="primary"),
                         "%s|H1|test/model|easy | neutral" % BOOTSTRAP_KEY)
        self.assertEqual(published_seed(H1, split="holdout", model_id=MODEL, role="control"),
                         "%s|H7a|test/model|easy | neutral" % BOOTSTRAP_KEY)

    def test_the_extension_arm_has_one_key_for_both_splits(self):
        for split in ("discovery", "holdout"):
            self.assertEqual(
                published_seed(H1, split=split, model_id=MODEL, role="extension"),
                "%s|test/model|%s|H1|easy | neutral" % (EXTENSION_BOOTSTRAP_KEY, split))

    def test_a_contrast_with_no_published_counterpart_has_no_published_seed(self):
        pooled = CONTRASTS_BY_ID["tone_pooled"]
        self.assertIsNone(published_seed(pooled, split="discovery", model_id=MODEL, role="primary"))
        self.assertIsNone(published_seed(pooled, split="holdout", model_id=MODEL, role="primary"))
        self.assertIsNone(published_seed(CONTRASTS_BY_ID["H1_hard"], split="holdout",
                                         model_id=MODEL, role="primary"))

    def test_missingness_seeds_separate_every_treatment(self):
        seeds = {missingness_seed(split="holdout", model_id=MODEL, contrast_id="H1", treatment=name)
                 for name in (AVAILABLE_CASE, ZERO_IMPUTATION, MANSKI_LOWER, MANSKI_UPPER, DELTA)}
        self.assertEqual(len(seeds), 5)


class AnalyseContrastTests(unittest.TestCase):
    def setUp(self):
        self.outcome = analyse_contrast(FIXTURE, H1, model_id=MODEL, split="holdout", role="primary")

    def test_each_treatment_reports_its_own_item_count(self):
        self.assertEqual(self.outcome.treatment(AVAILABLE_CASE).result.n_items, 2)
        for name in (ZERO_IMPUTATION, MANSKI_LOWER, MANSKI_UPPER):
            self.assertEqual(self.outcome.treatment(name).result.n_items, 4)

    def test_the_estimates_match_the_arithmetic(self):
        self.assertAlmostEqual(self.outcome.treatment(AVAILABLE_CASE).result.estimate, -8.0)
        self.assertAlmostEqual(self.outcome.treatment(ZERO_IMPUTATION).result.estimate, -4.5)
        self.assertAlmostEqual(self.outcome.treatment(MANSKI_LOWER).result.estimate, -5.5)
        self.assertAlmostEqual(self.outcome.treatment(MANSKI_UPPER).result.estimate, -3.5)

    def test_the_available_case_row_carries_the_published_seed(self):
        self.assertEqual(self.outcome.treatment(AVAILABLE_CASE).seed_source, "published")
        self.assertTrue(self.outcome.treatment(AVAILABLE_CASE).seed_text.startswith(BOOTSTRAP_KEY))

    def test_a_contrast_without_missing_values_reuses_one_seed_across_treatments(self):
        complete = tuple(row for row in FIXTURE if row.task_id in ("T1", "T2"))
        outcome = analyse_contrast(complete, H1, model_id=MODEL, split="holdout", role="primary")
        seeds = {item.seed_text for item in outcome.treatments}
        self.assertEqual(len(seeds), 1)
        intervals = {(item.result.ci95_lower, item.result.ci95_upper) for item in outcome.treatments}
        self.assertEqual(len(intervals), 1)

    def test_the_support_and_counts_travel_with_the_result(self):
        self.assertEqual((self.outcome.support_min, self.outcome.support_max, self.outcome.support_n),
                         (8.0, 12.0, 4))
        self.assertEqual(self.outcome.counts.n_available, 2)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = run_missingness({"holdout": FIXTURE}, models={"primary": MODEL},
                                      contrasts=(H1,))

    def test_the_report_covers_every_model_split_and_contrast(self):
        self.assertEqual(len(self.report.outcomes), 1)
        self.assertEqual(self.report.outcomes[0].contrast_id, "H1")

    def test_markdown_tables_escape_the_pipe_inside_a_stratum_name(self):
        text = render_markdown(self.report)
        self.assertIn("easy \\| neutral", text)
        header, divider = None, None
        for line in text.splitlines():
            if line.startswith("| contrast | stratum | split | treatment"):
                header = line
            elif header is not None and divider is None:
                divider = line
            elif header is not None and line.startswith("|"):
                self.assertEqual(line.count("|") - line.count("\\|"), header.count("|"))
                break

    def test_per_item_rows_expose_every_treatment_for_every_item(self):
        rows = per_item_rows(self.report, {"holdout": FIXTURE})
        self.assertEqual([row["task_id"] for row in rows], ["T1", "T2", "T3", "T4"])
        for row in rows:
            self.assertEqual(set(PER_ITEM_COLUMNS) - set(row), set())
        by_task = {row["task_id"]: row for row in rows}
        self.assertIsNone(by_task["T3"]["difference_available_case"])
        self.assertAlmostEqual(by_task["T3"]["difference_zero_imputation"], -8.0)
        self.assertAlmostEqual(by_task["T4"]["difference_manski_upper"], -2.0)
        self.assertEqual(by_task["T4"]["reference_missing_kind"], CANDIDATE_ABSENT)

    def test_a_model_absent_from_a_split_is_skipped_not_invented(self):
        report = run_missingness({"holdout": FIXTURE},
                                 models={"primary": MODEL, "control": "absent/model"},
                                 contrasts=(H1,))
        self.assertEqual({item.model_id for item in report.outcomes}, {MODEL})


class PublishedLookupTests(unittest.TestCase):
    def test_reads_the_three_committed_table_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "discovery.csv").write_text(
                "model_id,contrast,metric,stratum,n_items,n_pairs,mean_difference,ci95_lower,ci95_upper\n"
                "test/model,validity_malfunctioning_minus_accurate,m1,easy|neutral,10,10,-3.8,-5.3,-2.35\n"
                "test/model,validity_malfunctioning_minus_accurate,m2,easy|neutral,10,10,9.9,9.9,9.9\n",
                encoding="utf-8", newline="\n")
            (root / "holdout.csv").write_text(
                "hypothesis_id,outcome,estimate,ci95_lower,ci95_upper,n_items\n"
                "H1,m1,-2.9,-3.97,-1.84,10\nH7a,m1,-9.47,-19.9,-1.46,10\nH6a,distress,3.2,2.1,4.3,20\n",
                encoding="utf-8", newline="\n")
            (root / "extension.json").write_text(json.dumps({"result": {"comparisons": [
                {"hypothesis_id": "H1", "outcome": "m1",
                 "discovery": {"result": {"estimate": -6.5, "ci95_lower": -9.0,
                                          "ci95_upper": -4.3, "n_items": 8}},
                 "holdout": {"result": {"estimate": None}}}]}}), encoding="utf-8", newline="\n")

            discovery = load_discovery_published(root / "discovery.csv")
            self.assertEqual(discovery[("test/model", "H1")].estimate, -3.8)
            self.assertEqual(len(discovery), 1)  # the m2 row is not an M1 contrast

            holdout = load_holdout_published(root / "holdout.csv",
                                             {"primary": "P", "control": "C"})
            self.assertEqual(holdout[("P", "H1")].estimate, -2.9)
            self.assertEqual(holdout[("C", "H1")].estimate, -9.47)

            extension = load_extension_published(root / "extension.json", "X")
            self.assertEqual(extension[("discovery", "X", "H1")].n_items, 8)
            self.assertNotIn(("holdout", "X", "H1"), extension)

    def test_absent_files_are_empty_not_fatal(self):
        self.assertEqual(load_discovery_published("no-such-file.csv"), {})
        self.assertEqual(load_holdout_published("no-such-file.csv", {}), {})
        self.assertEqual(load_extension_published("no-such-file.json", "X"), {})


class ContrastTableTests(unittest.TestCase):
    def test_every_contrast_names_a_treated_and_a_reference_cell(self):
        for contrast in CONTRASTS_BY_ID.values():
            self.assertIsInstance(contrast, Contrast)
            self.assertIn(contrast.direction, ("negative", "positive"))
            self.assertTrue(contrast.difficulties)
            self.assertNotEqual(
                (contrast.treated.validity, contrast.treated.tone, contrast.treated.turn),
                (contrast.reference.validity, contrast.reference.tone, contrast.reference.turn))

    def test_the_treated_side_is_the_manipulated_one(self):
        self.assertEqual(CONTRASTS_BY_ID["H1"].treated.validity, "malfunctioning_always_fail")
        self.assertEqual(CONTRASTS_BY_ID["H2a"].treated.tone, "hostile")
        self.assertEqual(CONTRASTS_BY_ID["H3a"].treated.turn, "onset")


if __name__ == "__main__":
    unittest.main()
