from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math
import unittest

from src.analysis import CoefficientResult, G1MetricResult, G2Result, G5Result
from src.gates import (
    BLOCKED, FAIL, G3_NEUTRAL_STYLE_ID, G3_SMOKE_TASK_IDS, G3_STYLE_IDS, INCOMPLETE,
    NOT_EVALUATED, NOT_RUN, PASS, UNAVAILABLE, CoreGateInputs, G3Evidence, G4Evidence,
    G4ModelEvidence, G5Evidence, GateInputError, Phase1GateInputs, StyleEffectEvidence,
    compose_core_gates, compose_phase_1_gates, evaluate_g3, evaluate_g4,
)

GEMMA_9B = "google/gemma-2-9b-it"
GEMMA_2B = "google/gemma-2-2b-it"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"


def coefficient(p=0.02, positive=True):
    return CoefficientResult(1.0, .1, (.8, 1.2), p, p, 1.0 if positive else -1.0, p < .01, positive)


def g1(metric, *, validity=.02, tone=.02, available=True, converged=True, positive=True, model="primary"):
    return G1MetricResult(model, metric, coefficient(validity, positive), coefficient(tone, positive), 10, 4, converged, None if available else "missing")


def g2(metric, *, induction=1.0, lower=.1, ratio=None, available=True, model="primary"):
    return G2Result(model, metric, 4, 8, induction, .4, ratio, (lower, .8), None if available else "missing")


def g5(*, full=.7, baseline=.5, gap=None, available=True):
    return G5Result(full, baseline, full - baseline if gap is None else gap, 10, 0, 2, (), unavailable_reason=None if available else "missing")


def inputs(*, family=("M1",), real_g1=None, shuffled_g1=None, real_g2=None, real_g5=None, shuffled_g5=None, model="primary"):
    return CoreGateInputs(
        model, family,
        {metric: g1(metric, model=model) for metric in family} if real_g1 is None else real_g1,
        {metric: g1(metric, model=model) for metric in family} if shuffled_g1 is None else shuffled_g1,
        {metric: g2(metric, model=model) for metric in family} if real_g2 is None else real_g2,
        G5Evidence(model, family, g5() if real_g5 is None else real_g5),
        G5Evidence(model, family, g5(full=.55, baseline=.5) if shuffled_g5 is None else shuffled_g5),
    )


def style(metric, style_id, *, effect=.6, p=.001, model=GEMMA_9B, tasks=G3_SMOKE_TASK_IDS, reference=G3_NEUTRAL_STYLE_ID, reason=None):
    return StyleEffectEvidence(model, metric, style_id, tasks, reference, effect, p, len(tasks), reason)


def g3_evidence(family=("M1",), *, model=GEMMA_9B, overrides=None, effect=.4, p=.5):
    effects = {(metric, style_id): style(metric, style_id, effect=effect, p=p, model=model)
               for metric in family for style_id in G3_STYLE_IDS}
    effects.update(overrides or {})
    return G3Evidence(model, family, effects)


def g4_model(model_id, family, *, positives=(), unknown=(), role="evaluated"):
    from src.gates import FROZEN_MODEL_FAMILIES

    real_g1 = {}
    for metric in family:
        if metric in unknown:
            real_g1[metric] = None
        else:
            real_g1[metric] = g1(metric, validity=.001 if metric in positives else .5, model=model_id)
    return G4ModelEvidence(model_id, FROZEN_MODEL_FAMILIES[model_id], family, {} if role == "unsupported" else real_g1, role)


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

    def test_g3_reproduction_requires_direction_magnitude_and_adjusted_p(self):
        core = inputs(real_g1={"M1": g1("M1", validity=.009)}, model="primary")
        reference = 1.0  # sign-aligned validity coefficient of the qualifying metric
        cases = {
            "reproduces": (.6, .001, FAIL),
            "below_half_magnitude": (.5 * reference - 1e-9, .001, PASS),
            "opposite_direction": (-.9, .001, PASS),
            "not_significant": (.9, .01, PASS),
            "exactly_half_magnitude": (.5 * reference, .001, FAIL),
        }
        for label, (effect, probability, expected) in cases.items():
            with self.subTest(label=label):
                evidence = g3_evidence(("M1",), model="primary", overrides={
                    ("M1", G3_STYLE_IDS[0]): style("M1", G3_STYLE_IDS[0], effect=effect, p=probability, model="primary"),
                })
                verdict = evaluate_g3(core, evidence)
                self.assertEqual(verdict.status, expected)
                self.assertTrue(verdict.provisional)
                if expected == FAIL:
                    self.assertEqual(verdict.style_meter_metrics, ("M1",))
                else:
                    self.assertEqual(verdict.non_reproduced_metrics, ("M1",))

    def test_g3_not_run_not_evaluated_and_unavailable_paths(self):
        core = inputs(real_g1={"M1": g1("M1", validity=.009)}, model="primary")
        self.assertEqual(evaluate_g3(core, None).status, NOT_RUN)
        self.assertEqual(evaluate_g3(inputs(), g3_evidence(("M1",), model="primary")).status, NOT_EVALUATED)
        missing_slot = G3Evidence("primary", ("M1",), {
            (metric, style_id): (None if style_id == G3_STYLE_IDS[2] else style(metric, style_id, model="primary"))
            for metric in ("M1",) for style_id in G3_STYLE_IDS
        })
        self.assertEqual(evaluate_g3(core, missing_slot).status, UNAVAILABLE)
        wrong_items = g3_evidence(("M1",), model="primary", overrides={
            ("M1", G3_STYLE_IDS[1]): style("M1", G3_STYLE_IDS[1], model="primary", tasks=("DGS-001",)),
        })
        self.assertEqual(evaluate_g3(core, wrong_items).status, UNAVAILABLE)
        unavailable_effect = g3_evidence(("M1",), model="primary", overrides={
            ("M1", G3_STYLE_IDS[1]): style("M1", G3_STYLE_IDS[1], model="primary", effect=None, p=None, reason="neutral_standardization_unavailable"),
        })
        self.assertEqual(evaluate_g3(core, unavailable_effect).status, UNAVAILABLE)
        with self.assertRaises(GateInputError):
            G3Evidence("primary", ("M1",), {("M1", "style__made_up"): None})
        with self.assertRaises(GateInputError):
            G3Evidence("primary", ("M1",), {("M1", G3_STYLE_IDS[0]): style("M1", G3_STYLE_IDS[1], model="primary")})

    def test_g3_flags_only_the_reproduced_metric_and_still_passes(self):
        core = inputs(
            family=("M1", "M2"),
            real_g1={"M1": g1("M1", validity=.009), "M2": g1("M2", validity=.009)},
            model="primary",
        )
        evidence = g3_evidence(("M1", "M2"), model="primary", overrides={
            ("M2", G3_STYLE_IDS[3]): style("M2", G3_STYLE_IDS[3], effect=.8, p=.0001, model="primary"),
        })
        verdict = evaluate_g3(core, evidence)
        self.assertEqual((verdict.status, verdict.style_meter_metrics, verdict.non_reproduced_metrics), (PASS, ("M2",), ("M1",)))


class G4Tests(unittest.TestCase):
    def test_transfer_and_family_boundary_both_pass(self):
        transfer = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)),
            g4_model(QWEN_7B, ("M1",), positives=("M1",)),
        ))
        verdict = evaluate_g4(transfer)
        self.assertEqual((verdict.status, verdict.reason, verdict.transfer_metrics), (PASS, "transfer", ("M1",)))
        boundary = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)),
            g4_model(QWEN_7B, ("M1",)),
        ))
        verdict = evaluate_g4(boundary)
        self.assertEqual((verdict.status, verdict.reason, verdict.boundary_metrics), (PASS, "family_boundary", ("M1",)))
        self.assertEqual(verdict.eligible_positive_metrics, ("M1",))

    def test_messy_nontransfer_and_absent_positive_fail(self):
        messy = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)),
            g4_model(QWEN_7B, ("M1",)),
            g4_model(QWEN_3B, ("M1",), positives=("M1",)),
        ))
        verdict = evaluate_g4(messy)
        self.assertEqual((verdict.status, verdict.reason), (FAIL, "messy_nontransfer"))
        none_positive = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",)), g4_model(QWEN_7B, ("M1",)),
        ))
        self.assertEqual(evaluate_g4(none_positive).reason, "no_eligible_positive_in_primary_model")
        # An instability-negative qualifying effect is not an eligible positive.
        negative = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            G4ModelEvidence(GEMMA_9B, "Gemma-2", ("M1",), {"M1": g1("M1", validity=.001, positive=False, model=GEMMA_9B)}),
            g4_model(QWEN_7B, ("M1",)),
        ))
        self.assertEqual(evaluate_g4(negative).reason, "no_eligible_positive_in_primary_model")

    def test_outcome_changing_gaps_are_unavailable_not_failures(self):
        unknown_control = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)),
            g4_model(QWEN_7B, ("M1",), unknown=("M1",)),
        ))
        self.assertEqual(evaluate_g4(unknown_control).status, UNAVAILABLE)
        no_primary_evidence = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), unknown=("M1",)),
            g4_model(QWEN_7B, ("M1",)),
        ))
        self.assertEqual(evaluate_g4(no_primary_evidence).status, UNAVAILABLE)
        # Llama is explicitly outside the boundary comparison and carries no G1.
        with_llama = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)),
            g4_model(QWEN_7B, ("M1",)),
            g4_model(LLAMA_3B, ("M1",), role="unsupported"),
        ))
        self.assertEqual(evaluate_g4(with_llama).status, PASS)
        with self.assertRaises(GateInputError):
            G4ModelEvidence("not/a-frozen-model", "Gemma-2", ("M1",), {})
        with self.assertRaises(GateInputError):
            G4ModelEvidence(GEMMA_9B, "Qwen2.5", ("M1",), {})
        with self.assertRaises(GateInputError):
            G4Evidence(GEMMA_9B, GEMMA_9B, ("M1",), (g4_model(GEMMA_9B, ("M1",)),))


class Phase1CompositionTests(unittest.TestCase):
    def _phase1(self, *, core_kwargs=None, g3=None, g4=None, family=("M1",)):
        core = inputs(model=GEMMA_9B, family=family, **(core_kwargs or {}))
        evidence = g4 or G4Evidence(GEMMA_9B, QWEN_7B, family, (
            G4ModelEvidence(GEMMA_9B, "Gemma-2", family, dict(core.real_g1)),
            g4_model(QWEN_7B, family, positives=family),
        ))
        return compose_phase_1_gates(Phase1GateInputs(core, evidence, g3))

    def test_all_five_gates_pass(self):
        family = ("M1",)
        core_kwargs = {"real_g1": {"M1": g1("M1", validity=.009, model=GEMMA_9B)},
                       "real_g2": {"M1": g2("M1", model=GEMMA_9B)}}
        summary = self._phase1(core_kwargs=core_kwargs, g3=g3_evidence(family))
        self.assertEqual(summary.phase_1_status, PASS)
        self.assertTrue(summary.interpretable)
        self.assertEqual([status.status for _, status in summary.gates], [PASS] * 5)
        self.assertEqual((summary.primary_model_id, summary.control_model_id), (GEMMA_9B, QWEN_7B))

    def test_absent_style_smoke_leaves_phase_1_incomplete_not_passed(self):
        summary = self._phase1(core_kwargs={
            "real_g1": {"M1": g1("M1", validity=.009, model=GEMMA_9B)},
            "real_g2": {"M1": g2("M1", model=GEMMA_9B)},
        })
        self.assertEqual((summary.g3.status, summary.phase_1_status), (NOT_RUN, INCOMPLETE))
        self.assertFalse(summary.interpretable)
        self.assertIn("g3", summary.interpretation_reason)

    def test_determinate_failure_survives_an_incomplete_gate(self):
        summary = self._phase1(core_kwargs={
            "real_g1": {"M1": g1("M1", validity=.009, model=GEMMA_9B)},
            "real_g2": {"M1": g2("M1", induction=-1.0, model=GEMMA_9B)},
        })
        self.assertEqual((summary.g2.status, summary.g3.status, summary.phase_1_status), (FAIL, NOT_RUN, FAIL))
        self.assertTrue(summary.interpretable)

    def test_failed_shuffled_null_blocks_every_real_verdict(self):
        summary = self._phase1(core_kwargs={
            "real_g1": {"M1": g1("M1", validity=.009, model=GEMMA_9B)},
            "real_g2": {"M1": g2("M1", model=GEMMA_9B)},
            "shuffled_g1": {"M1": g1("M1", validity=.001, model=GEMMA_9B)},
        }, g3=g3_evidence(("M1",)))
        self.assertEqual(summary.phase_1_status, BLOCKED)
        self.assertFalse(summary.interpretable)
        self.assertEqual(summary.g1.status, PASS)

    def test_phase_1_inputs_bind_models_metrics_and_evidence(self):
        core = inputs(model=GEMMA_9B)
        good = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            G4ModelEvidence(GEMMA_9B, "Gemma-2", ("M1",), dict(core.real_g1)),
            g4_model(QWEN_7B, ("M1",)),
        ))
        with self.assertRaises(GateInputError):
            Phase1GateInputs(core, good, g3_evidence(("M1",), model=QWEN_7B))
        mismatched = G4Evidence(GEMMA_9B, QWEN_7B, ("M1",), (
            g4_model(GEMMA_9B, ("M1",), positives=("M1",)), g4_model(QWEN_7B, ("M1",)),
        ))
        with self.assertRaises(GateInputError):
            Phase1GateInputs(core, mismatched)
        with self.assertRaises(GateInputError):
            Phase1GateInputs(core, G4Evidence(QWEN_7B, GEMMA_9B, ("M1",), (
                g4_model(QWEN_7B, ("M1",)), g4_model(GEMMA_9B, ("M1",)))))


class CoreGateContinuedTests(unittest.TestCase):
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
