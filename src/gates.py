"""Pure composition of the preregistered Phase-1 core gates.

This module deliberately consumes completed analysis results.  It neither pools
models nor recomputes statistics: that separation makes the model and metric
family used to make a verdict explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from .analysis import CoefficientResult, G1MetricResult, G2Result, G5Result, PRIMARY_METRICS


PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"
NOT_EVALUATED = "NOT_EVALUATED"


class GateInputError(ValueError):
    """Raised when evidence is assigned to the wrong declared model or family."""


def _freeze(mapping):
    return MappingProxyType(dict(mapping))


def _family(metrics):
    supplied = tuple(metrics)
    if not supplied or len(supplied) != len(set(supplied)) or any(metric not in PRIMARY_METRICS for metric in supplied):
        raise GateInputError("eligible metrics must be a unique ordered subset of M1/M2/M3")
    return tuple(metric for metric in PRIMARY_METRICS if metric in supplied)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class QualifiedEffect:
    metric_name: str
    effect: str
    adjusted_p: float
    instability_positive: bool


@dataclass(frozen=True)
class GateVerdict:
    status: str
    reason: str | None = None
    qualifying_effects: tuple[QualifiedEffect, ...] = ()
    passing_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class G5Evidence:
    """A G5 result bound to the declared model and primary feature family."""

    model_id: str
    eligible_metrics: tuple[str, ...]
    result: G5Result | None

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id:
            raise GateInputError("G5 evidence model_id must be nonempty")
        object.__setattr__(self, "eligible_metrics", _family(self.eligible_metrics))


@dataclass(frozen=True)
class CoreGateInputs:
    """All completed evidence for one named Phase-1 primary model."""

    primary_model_id: str
    eligible_metrics: tuple[str, ...]
    real_g1: Mapping[str, G1MetricResult | None]
    shuffled_g1: Mapping[str, G1MetricResult | None]
    real_g2: Mapping[str, G2Result | None]
    real_g5: G5Evidence
    shuffled_g5: G5Evidence

    def __post_init__(self):
        if not isinstance(self.primary_model_id, str) or not self.primary_model_id:
            raise GateInputError("primary_model_id must be nonempty")
        family = _family(self.eligible_metrics)
        object.__setattr__(self, "eligible_metrics", family)
        for name in ("real_g1", "shuffled_g1", "real_g2"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or set(value) - set(family):
                raise GateInputError(name + " contains a metric outside the declared family")
            object.__setattr__(self, name, _freeze(value))
        for metric, result in self.real_g2.items():
            if result is not None and (
                not isinstance(result, G2Result)
                or result.model_id != self.primary_model_id
                or result.metric_name != metric
            ):
                raise GateInputError("real_g2 result model or metric does not match its declared slot")
        for name in ("real_g5", "shuffled_g5"):
            evidence = getattr(self, name)
            if not isinstance(evidence, G5Evidence):
                raise GateInputError(name + " must be G5Evidence")
            if evidence.model_id != self.primary_model_id or evidence.eligible_metrics != family:
                raise GateInputError(name + " does not match the declared primary model and metric family")


@dataclass(frozen=True)
class CoreGateSummary:
    shuffled_null: GateVerdict
    g1: GateVerdict
    g2: GateVerdict
    g5: GateVerdict
    interpretable: bool
    interpretation_reason: str | None
    phase_1_status: str = "INCOMPLETE_PENDING_G3_G4"


def _metric_result(mapping, model_id, metric, label):
    result = mapping.get(metric)
    if result is None:
        return None
    if not isinstance(result, G1MetricResult):
        raise GateInputError(label + " has an invalid G1 result")
    if result.model_id != model_id or result.metric_name != metric:
        raise GateInputError(label + " result model or metric does not match its declared slot")
    return result


def _coefficient_complete(coefficient):
    if not isinstance(coefficient, CoefficientResult):
        return False
    if not isinstance(coefficient.qualifying, bool) or not isinstance(coefficient.instability_positive, bool):
        return False
    interval = coefficient.ci95
    if not isinstance(interval, tuple) or len(interval) != 2 or not all(_finite(value) for value in interval) or interval[0] > interval[1]:
        return False
    if not all(_finite(value) for value in (
        coefficient.coefficient,
        coefficient.standard_error,
        coefficient.raw_p,
        coefficient.adjusted_p,
        coefficient.sign_aligned_coefficient,
    )):
        return False
    return (
        coefficient.standard_error >= 0
        and 0 <= coefficient.raw_p <= 1
        and 0 <= coefficient.adjusted_p <= 1
        and coefficient.qualifying == (coefficient.adjusted_p < 0.01)
        and coefficient.instability_positive == (coefficient.sign_aligned_coefficient > 0)
    )


def _g1_unavailable_reason(mapping, model_id, family, label):
    for metric in family:
        result = _metric_result(mapping, model_id, metric, label)
        if result is None:
            return label + ":missing_" + metric
        if result.unavailable_reason or not result.converged:
            return label + ":unavailable_" + metric
        if not _coefficient_complete(result.validity) or not _coefficient_complete(result.tone):
            return label + ":adjusted_p_unavailable_" + metric
    return None


def _qualifying_effects(mapping, model_id, family, label):
    qualifying = []
    for metric in family:
        result = _metric_result(mapping, model_id, metric, label)
        for effect in ("validity", "tone"):
            coefficient = getattr(result, effect)
            if coefficient.adjusted_p < 0.01:
                qualifying.append(QualifiedEffect(metric, effect, float(coefficient.adjusted_p), coefficient.instability_positive))
    return tuple(qualifying)


def evaluate_g1(inputs: CoreGateInputs):
    unavailable = _g1_unavailable_reason(inputs.real_g1, inputs.primary_model_id, inputs.eligible_metrics, "real_g1")
    if unavailable:
        return GateVerdict(UNAVAILABLE, unavailable)
    qualifying = _qualifying_effects(inputs.real_g1, inputs.primary_model_id, inputs.eligible_metrics, "real_g1")
    return GateVerdict(PASS, qualifying_effects=qualifying) if qualifying else GateVerdict(FAIL, "no_adjusted_p_below_0.01")


def _g5_result_verdict(evidence: G5Evidence, *, shuffled: bool):
    result = evidence.result
    if not isinstance(result, G5Result) or result.unavailable_reason:
        return GateVerdict(UNAVAILABLE, "g5_result_unavailable")
    values = (result.full_auc, result.baseline_auc, result.auc_gap)
    if not all(_finite(value) for value in values):
        return GateVerdict(UNAVAILABLE, "g5_nonfinite_or_missing_auc")
    full_auc, baseline_auc, gap = values
    if not 0 <= full_auc <= 1 or not 0 <= baseline_auc <= 1 or not math.isclose(gap, full_auc - baseline_auc, rel_tol=0.0, abs_tol=1e-12):
        return GateVerdict(UNAVAILABLE, "g5_malformed_auc_gap")
    if shuffled:
        return GateVerdict(PASS) if gap < 0.1 else GateVerdict(FAIL, "shuffled_auc_gap_not_below_0.1")
    return GateVerdict(PASS) if gap >= 0.1 else GateVerdict(FAIL, "auc_gap_below_0.1")


def evaluate_g5(inputs: CoreGateInputs):
    return _g5_result_verdict(inputs.real_g5, shuffled=False)


def evaluate_shuffled_null(inputs: CoreGateInputs):
    # Availability is deliberately checked first: incomplete shuffled evidence can
    # never be converted into a reassuring null by a false-positive result elsewhere.
    unavailable = _g1_unavailable_reason(inputs.shuffled_g1, inputs.primary_model_id, inputs.eligible_metrics, "shuffled_g1")
    if unavailable:
        return GateVerdict(UNAVAILABLE, unavailable)
    shuffled_g5 = _g5_result_verdict(inputs.shuffled_g5, shuffled=True)
    if shuffled_g5.status == UNAVAILABLE:
        return GateVerdict(UNAVAILABLE, shuffled_g5.reason)
    qualifying = _qualifying_effects(inputs.shuffled_g1, inputs.primary_model_id, inputs.eligible_metrics, "shuffled_g1")
    if qualifying:
        return GateVerdict(FAIL, "shuffled_g1_false_positive", qualifying_effects=qualifying)
    if shuffled_g5.status == FAIL:
        return shuffled_g5
    return GateVerdict(PASS)


def evaluate_g2(inputs: CoreGateInputs, g1: GateVerdict):
    if g1.status != PASS:
        return GateVerdict(NOT_EVALUATED, "g1_not_passed_not_unlocked")
    qualifying_metrics = tuple(dict.fromkeys(effect.metric_name for effect in g1.qualifying_effects))
    unavailable = False
    passing = []
    for metric in qualifying_metrics:
        result = inputs.real_g2.get(metric)
        if result is None or not isinstance(result, G2Result):
            unavailable = True
            continue
        if result.model_id != inputs.primary_model_id or result.metric_name != metric:
            raise GateInputError("real_g2 result model or metric does not match its declared slot")
        if result.unavailable_reason or not _finite(result.induction) or not _finite(result.recovery):
            unavailable = True
            continue
        interval = result.recovery_ci95
        if (
            not isinstance(interval, tuple)
            or len(interval) != 2
            or not all(_finite(value) for value in interval)
            or interval[0] > interval[1]
        ):
            unavailable = True
            continue
        if result.induction > 0 and interval[0] > 0:
            passing.append(metric)
    if unavailable:
        return GateVerdict(UNAVAILABLE, "required_qualifying_g2_unavailable")
    return GateVerdict(PASS, passing_metrics=tuple(passing)) if passing else GateVerdict(FAIL, "no_qualifying_metric_reverses")


def compose_core_gates(inputs: CoreGateInputs):
    if not isinstance(inputs, CoreGateInputs):
        raise GateInputError("inputs must be CoreGateInputs")
    shuffled_null = evaluate_shuffled_null(inputs)
    g1 = evaluate_g1(inputs)
    g2 = evaluate_g2(inputs, g1)
    g5 = evaluate_g5(inputs)
    if shuffled_null.status != PASS:
        interpretable = False
        reason = "blocked_by_shuffled_null:" + shuffled_null.status
    else:
        incomplete = []
        if g1.status == UNAVAILABLE:
            incomplete.append("g1_unavailable")
        if g5.status == UNAVAILABLE:
            incomplete.append("g5_unavailable")
        if g1.status == PASS and g2.status == UNAVAILABLE:
            incomplete.append("g2_unavailable")
        interpretable = not incomplete
        reason = None if interpretable else "incomplete_real_core:" + ",".join(incomplete)
    return CoreGateSummary(shuffled_null, g1, g2, g5, interpretable, reason)
