from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

from src.analysis import CoefficientResult, G1MetricResult, G2Result, G5Result
from src.gates import (
    FAIL, NOT_EVALUATED, PASS, UNAVAILABLE, CoreGateInputs, G5Evidence,
    GateInputError, compose_core_gates,
)


def coefficient(p=0.02, positive=True):
    return CoefficientResult(1.0, .1, (.8, 1.2), p, p, 1.0 if positive else -1.0, p < .01, positive)


def g1(metric, *, validity=.02, tone=.02, available=True, converged=True, positive=True):
    return G1MetricResult("primary", metric, coefficient(validity, positive), coefficient(tone, positive), 10, 4, converged, None if available else "missing")


def g2(metric, *, induction=1.0, lower=.1, ratio=None, available=True):
    return G2Result("primary", metric, 4, 8, induction, .4, ratio, (lower, .8), None if available else "missing")


def g5(*, full=.7, baseline=.5, gap=None, available=True):
    return G5Result(full, baseline, full - baseline if gap is None else gap, 10, 0, 2, (), unavailable_reason=None if available else "missing")


def inputs(*, family=("M1",), real_g1=None, shuffled_g1=None, real_g2=None, real_g5=None, shuffled_g5=None):
    return CoreGateInputs(
        "primary", family,
        {metric: g1(metric) for metric in family} if real_g1 is None else real_g1,
        {metric: g1(metric) for metric in family} if shuffled_g1 is None else shuffled_g1,
        {metric: g2(metric) for metric in family} if real_g2 is None else real_g2,
        G5Evidence("primary", family, g5() if real_g5 is None else real_g5),
        G5Evidence("primary", family, g5(full=.55, baseline=.5) if shuffled_g5 is None else shuffled_g5),
    )


class CoreGateTests(unittest.TestCase):
    def test_strict_g1_and_g5_boundaries(self):
        for p, expected in ((.009, PASS), (.01, FAIL)):
            with self.subTest(p=p):
                summary = compose_core_gates(inputs(real_g1={"M1": g1("M1", validity=p)}))
                self.assertEqual(summary.g1.status, expected)
        for gap, expected in ((.099, FAIL), (.1, PASS)):
            with self.subTest(gap=gap):
                self.assertEqual(compose_core_gates(inputs(real_g5=g5(full=.5 + gap, baseline=.5, gap=gap))).g5.status, expected)
        self.assertEqual(compose_core_gates(inputs(shuffled_g5=g5(full=.6, baseline=.5, gap=.1))).shuffled_null.status, FAIL)

    def test_g1_tracks_effect_and_sign_without_using_direction_as_a_filter(self):
        summary = compose_core_gates(inputs(real_g1={"M1": g1("M1", validity=.02, tone=.009, positive=False)}))
        self.assertEqual(summary.g1.status, PASS)
        self.assertEqual(summary.g1.qualifying_effects[0].effect, "tone")
        self.assertFalse(summary.g1.qualifying_effects[0].instability_positive)

    def test_shuffled_null_checks_both_effects_and_unavailable_has_precedence(self):
        for result in (g1("M1", validity=.009), g1("M1", tone=.009)):
            with self.subTest(result=result):
                summary = compose_core_gates(inputs(shuffled_g1={"M1": result}))
                self.assertEqual(summary.shuffled_null.status, FAIL)
        summary = compose_core_gates(inputs(
            family=("M1", "M2"),
            shuffled_g1={"M1": g1("M1", validity=.009)},
        ))
        self.assertEqual(summary.shuffled_null.status, UNAVAILABLE)

    def test_unavailable_g1_prevents_raw_p_fallback(self):
        missing_adjusted = replace(coefficient(.0001), adjusted_p=None)
        result = replace(g1("M1", validity=.0001), validity=missing_adjusted)
        summary = compose_core_gates(inputs(real_g1={"M1": result}))
        self.assertEqual(summary.g1.status, UNAVAILABLE)

    def test_malformed_coefficient_evidence_is_unavailable(self):
        good = coefficient(.009)
        malformed = (
            replace(good, sign_aligned_coefficient=math.nan),
            replace(good, instability_positive=False),
            replace(good, qualifying=False),
            replace(good, standard_error=-.1),
            replace(good, ci95=(1.2, .8)),
            replace(good, raw_p=1.1),
        )
        for bad in malformed:
            with self.subTest(bad=bad):
                summary = compose_core_gates(inputs(real_g1={"M1": replace(g1("M1", validity=.009), validity=bad)}))
                self.assertEqual(summary.g1.status, UNAVAILABLE)

    def test_model_and_metric_mismatches_are_rejected(self):
        with self.assertRaises(GateInputError):
            compose_core_gates(inputs(real_g1={"M1": replace(g1("M1"), model_id="other")}))
        with self.assertRaises(GateInputError):
            CoreGateInputs("primary", ("M1",), {"M1": g1("M1")}, {"M1": g1("M1")}, {"M1": g2("M1")}, G5Evidence("primary", ("M1", "M2"), g5()), G5Evidence("primary", ("M1",), g5()))
        with self.assertRaises(GateInputError):
            CoreGateInputs("primary", ("M1",), {}, {}, {}, G5Evidence("other", ("M1",), g5()), G5Evidence("primary", ("M1",), g5()))

    def test_missing_result_is_unavailable_not_dropped(self):
        summary = compose_core_gates(inputs(family=("M1", "M2"), real_g1={"M1": g1("M1", validity=.009)}))
        self.assertEqual(summary.g1.status, UNAVAILABLE)

    def test_g2_uses_only_g1_qualifiers_and_does_not_require_ratio(self):
        summary = compose_core_gates(inputs(
            family=("M1", "M2"),
            real_g1={"M1": g1("M1", validity=.009), "M2": g1("M2")},
            real_g2={"M1": g2("M1", ratio=None)},
        ))
        self.assertEqual((summary.g1.status, summary.g2.status, summary.g2.passing_metrics), (PASS, PASS, ("M1",)))

    def test_g2_not_evaluated_when_g1_does_not_unlock_it(self):
        summary = compose_core_gates(inputs())
        self.assertEqual(summary.g2.status, NOT_EVALUATED)
        self.assertEqual(summary.g2.reason, "g1_not_passed_not_unlocked")

    def test_g2_has_unavailable_precedence_and_strict_direction(self):
        unavailable = g2("M2", available=False)
        summary = compose_core_gates(inputs(
            family=("M1", "M2"),
            real_g1={"M1": g1("M1", validity=.009), "M2": g1("M2", validity=.009)},
            real_g2={"M1": g2("M1"), "M2": unavailable},
        ))
        self.assertEqual(summary.g2.status, UNAVAILABLE)
        for induction, lower in ((0, .1), (1, 0)):
            with self.subTest(induction=induction, lower=lower):
                summary = compose_core_gates(inputs(real_g1={"M1": g1("M1", validity=.009)}, real_g2={"M1": g2("M1", induction=induction, lower=lower)}))
                self.assertEqual(summary.g2.status, FAIL)
        for interval in ((.8, .1), (math.nan, .8)):
            with self.subTest(interval=interval):
                result = replace(g2("M1"), recovery_ci95=interval)
                summary = compose_core_gates(inputs(real_g1={"M1": g1("M1", validity=.009)}, real_g2={"M1": result}))
                self.assertEqual(summary.g2.status, UNAVAILABLE)

    def test_nonqualifying_g2_slots_are_still_validated(self):
        with self.assertRaises(GateInputError):
            inputs(
                family=("M1", "M2"),
                real_g1={"M1": g1("M1", validity=.009), "M2": g1("M2")},
                real_g2={"M1": g2("M1"), "M2": replace(g2("M2"), model_id="other")},
            )

    def test_g5_malformed_and_nonfinite_evidence_is_unavailable(self):
        cases = (g5(full=.7, baseline=.5, gap=.3), g5(full=math.nan, baseline=.5), g5(available=False))
        for result in cases:
            with self.subTest(result=result):
                self.assertEqual(compose_core_gates(inputs(real_g5=result)).g5.status, UNAVAILABLE)

    def test_g5_metric_family_binding_is_exact(self):
        with self.assertRaises(GateInputError):
            CoreGateInputs(
                "primary", ("M1", "M2"), {}, {}, {},
                G5Evidence("primary", ("M1",), g5()), G5Evidence("primary", ("M1", "M2"), g5()),
            )

    def test_null_failure_blocks_interpretation_without_erasing_real_verdicts(self):
        summary = compose_core_gates(inputs(
            real_g1={"M1": g1("M1", validity=.009)},
            real_g2={"M1": g2("M1")},
            shuffled_g1={"M1": g1("M1", tone=.009)},
            real_g5=g5(full=.7, baseline=.5),
        ))
        self.assertEqual((summary.shuffled_null.status, summary.g1.status, summary.g2.status, summary.g5.status), (FAIL, PASS, PASS, PASS))
        self.assertFalse(summary.interpretable)
        self.assertEqual(summary.phase_1_status, "INCOMPLETE_PENDING_G3_G4")

    def test_outputs_are_canonical_order_invariant_and_immutable(self):
        first = compose_core_gates(inputs(
            family=("M1", "M2"),
            real_g1={"M1": g1("M1", validity=.009), "M2": g1("M2", tone=.009)},
            real_g2={"M1": g2("M1"), "M2": g2("M2")},
        ))
        second = compose_core_gates(inputs(
            family=("M2", "M1"),
            real_g1={"M2": g1("M2", tone=.009), "M1": g1("M1", validity=.009)},
            real_g2={"M2": g2("M2"), "M1": g2("M1")},
        ))
        self.assertEqual(first, second)
        self.assertEqual(first.g2.passing_metrics, ("M1", "M2"))
        with self.assertRaises(TypeError):
            inputs().real_g1["M1"] = None
        with self.assertRaises(FrozenInstanceError):
            first.interpretable = False

    def test_interpretation_requires_determinate_real_core_evidence(self):
        unavailable_g5 = compose_core_gates(inputs(real_g5=g5(available=False)))
        self.assertFalse(unavailable_g5.interpretable)
        self.assertEqual(unavailable_g5.interpretation_reason, "incomplete_real_core:g5_unavailable")
        unavailable_g1 = compose_core_gates(inputs(real_g1={"M1": None}))
        self.assertFalse(unavailable_g1.interpretable)
        self.assertEqual(unavailable_g1.interpretation_reason, "incomplete_real_core:g1_unavailable")
        unavailable_g2 = compose_core_gates(inputs(real_g1={"M1": g1("M1", validity=.009)}, real_g2={"M1": None}))
        self.assertFalse(unavailable_g2.interpretable)
        self.assertEqual(unavailable_g2.interpretation_reason, "incomplete_real_core:g2_unavailable")
