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

from .analysis import (
    FROZEN_MODEL_FAMILIES, CoefficientResult, G1MetricResult, G2Result, G5Result,
    PRIMARY_METRICS, Phase0Selection,
)


PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"
NOT_EVALUATED = "NOT_EVALUATED"
NOT_RUN = "NOT_RUN"
INCOMPLETE = "INCOMPLETE"
BLOCKED = "BLOCKED"

# The frozen Phase-1 G3 smoke design (preregistration, "Frozen interaction and
# sampling protocol"): exactly these five discovery items and four style prompts,
# each compared with the no-feedback neutral reference.
G3_SMOKE_TASK_IDS = ("DGS-005", "DGS-010", "DGS-022", "DGS-026", "DGS-037")
G3_STYLE_IDS = (
    "style__enthusiastic",
    "style__cautious_hedging",
    "style__verbose",
    "style__reluctantly_complying_refusal_styled",
)
G3_NEUTRAL_STYLE_ID = "style__neutral_reference"


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
    """G1/G2/G5 plus the shuffled-label null.  G3/G4 are composed separately."""

    shuffled_null: GateVerdict
    g1: GateVerdict
    g2: GateVerdict
    g5: GateVerdict
    interpretable: bool
    interpretation_reason: str | None
    phase_1_status: str = "INCOMPLETE_PENDING_G3_G4"


@dataclass(frozen=True)
class StyleEffectEvidence:
    """One precomputed neutral-relative G3 smoke effect; no statistics are fit here.

    ``effect`` is the *sign-aligned* style-minus-neutral-reference effect in the
    same model-neutral SD units as the G1 coefficients, so a positive value means
    "more instability" for every metric.  ``adjusted_p`` is BH-adjusted within
    the phase by the caller.
    """

    model_id: str
    metric_name: str
    style_id: str
    task_ids: tuple[str, ...]
    neutral_style_id: str
    effect: float | None
    adjusted_p: float | None
    n_items: int = 0
    unavailable_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "task_ids", tuple(self.task_ids))


@dataclass(frozen=True)
class G3Evidence:
    """Frozen five-task, four-style smoke evidence for the declared primary model."""

    primary_model_id: str
    eligible_metrics: tuple[str, ...]
    effects: Mapping[tuple[str, str], StyleEffectEvidence | None]

    def __post_init__(self):
        if not isinstance(self.primary_model_id, str) or not self.primary_model_id:
            raise GateInputError("G3 primary_model_id must be nonempty")
        family = _family(self.eligible_metrics)
        object.__setattr__(self, "eligible_metrics", family)
        if not isinstance(self.effects, Mapping):
            raise GateInputError("G3 effects must be a mapping")
        permitted = {(metric, style) for metric in family for style in G3_STYLE_IDS}
        if set(self.effects) - permitted:
            raise GateInputError("G3 effects contain a metric or style outside the frozen smoke design")
        for (metric, style), effect in self.effects.items():
            if effect is not None and (
                not isinstance(effect, StyleEffectEvidence)
                or effect.model_id != self.primary_model_id
                or effect.metric_name != metric
                or effect.style_id != style
            ):
                raise GateInputError("G3 effect does not match its declared model, metric, or style slot")
        object.__setattr__(self, "effects", _freeze(self.effects))


@dataclass(frozen=True)
class G3Verdict:
    status: str
    reason: str | None = None
    non_reproduced_metrics: tuple[str, ...] = ()
    style_meter_metrics: tuple[str, ...] = ()
    provisional: bool = True


@dataclass(frozen=True)
class G4ModelEvidence:
    """A declared model/family binding for that model's completed real-G1 evidence."""

    model_id: str
    family: str
    eligible_metrics: tuple[str, ...]
    real_g1: Mapping[str, G1MetricResult | None]
    role: str = "evaluated"

    def __post_init__(self):
        frozen_family = FROZEN_MODEL_FAMILIES.get(self.model_id)
        if frozen_family is None or self.family != frozen_family:
            raise GateInputError("G4 model_id and family do not match the frozen model configuration")
        family = _family(self.eligible_metrics)
        object.__setattr__(self, "eligible_metrics", family)
        if self.role not in ("evaluated", "unsupported"):
            raise GateInputError("G4 role must be evaluated or unsupported")
        if self.role == "unsupported" and self.real_g1:
            raise GateInputError("an unsupported G4 model must carry no G1 evidence")
        if not isinstance(self.real_g1, Mapping) or set(self.real_g1) - set(family):
            raise GateInputError("G4 real_g1 contains a metric outside the declared family")
        object.__setattr__(self, "real_g1", _freeze(self.real_g1))


@dataclass(frozen=True)
class G4Evidence:
    primary_model_id: str
    control_model_id: str
    eligible_metrics: tuple[str, ...]
    models: tuple[G4ModelEvidence, ...]

    def __post_init__(self):
        family = _family(self.eligible_metrics)
        object.__setattr__(self, "eligible_metrics", family)
        models = tuple(self.models)
        if any(not isinstance(model, G4ModelEvidence) for model in models):
            raise GateInputError("G4 models must be G4ModelEvidence")
        if len({model.model_id for model in models}) != len(models):
            raise GateInputError("G4 model evidence must have unique model IDs")
        if any(model.eligible_metrics != family for model in models):
            raise GateInputError("G4 model evidence metric family does not match")
        by_id = {model.model_id: model for model in models}
        for name in ("primary_model_id", "control_model_id"):
            model_id = getattr(self, name)
            if not isinstance(model_id, str) or model_id not in by_id:
                raise GateInputError("G4 " + name + " must appear in the supplied model evidence")
        if self.primary_model_id == self.control_model_id:
            raise GateInputError("G4 primary and control models must be distinct")
        object.__setattr__(self, "models", tuple(sorted(models, key=lambda item: item.model_id)))


@dataclass(frozen=True)
class G4Verdict:
    status: str
    reason: str | None = None
    transfer_metrics: tuple[str, ...] = ()
    boundary_metrics: tuple[str, ...] = ()
    eligible_positive_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase1GateInputs:
    """All Phase-1 evidence.  ``g3`` may be None when the style smoke was not run."""

    core: CoreGateInputs
    g4: G4Evidence
    g3: G3Evidence | None = None
    selection: Phase0Selection | None = None

    def __post_init__(self):
        if not isinstance(self.core, CoreGateInputs) or not isinstance(self.g4, G4Evidence):
            raise GateInputError("Phase-1 composition requires core and G4 evidence")
        if self.g3 is not None:
            if not isinstance(self.g3, G3Evidence):
                raise GateInputError("g3 must be G3Evidence or None")
            if self.g3.primary_model_id != self.core.primary_model_id or self.g3.eligible_metrics != self.core.eligible_metrics:
                raise GateInputError("G3 evidence does not match the declared core primary model and metric family")
        if self.g4.eligible_metrics != self.core.eligible_metrics:
            raise GateInputError("G4 evidence does not match the declared core metric family")
        if self.g4.primary_model_id != self.core.primary_model_id:
            raise GateInputError("G4 primary model does not match the declared core primary model")
        primary = next(model for model in self.g4.models if model.model_id == self.g4.primary_model_id)
        if primary.role == "evaluated" and dict(primary.real_g1) != dict(self.core.real_g1):
            raise GateInputError("selected primary G4 evidence does not exactly match core real G1 evidence")
        if self.selection is not None:
            if not isinstance(self.selection, Phase0Selection):
                raise GateInputError("selection must be a Phase0Selection or None")
            if self.selection.status not in ("selected", "screen_null"):
                raise GateInputError("Phase 1 requires a selected or explicitly screen-null Phase-0 result")
            if self.selection.primary_model_id != self.core.primary_model_id or self.selection.control_model_id != self.g4.control_model_id:
                raise GateInputError("Phase-1 models do not match the Phase-0 selection")


@dataclass(frozen=True)
class Phase1GateSummary:
    """The five-gate verdict table for one declared primary model."""

    primary_model_id: str
    control_model_id: str
    eligible_metrics: tuple[str, ...]
    shuffled_null: GateVerdict
    g1: GateVerdict
    g2: GateVerdict
    g3: G3Verdict
    g4: G4Verdict
    g5: GateVerdict
    style_meters: tuple[str, ...]
    boundary_metrics: tuple[str, ...]
    interpretable: bool
    interpretation_reason: str | None
    phase_1_status: str

    @property
    def gates(self):
        return (("G1", self.g1), ("G2", self.g2), ("G3", self.g3), ("G4", self.g4), ("G5", self.g5))


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


def _g1_metric_complete(mapping, model_id, metric, label):
    result = _metric_result(mapping, model_id, metric, label)
    if result is None or result.unavailable_reason or not result.converged:
        return None
    if not _coefficient_complete(result.validity) or not _coefficient_complete(result.tone):
        return None
    return result


def _eligible_positive(result):
    """A primary metric meets G1 in the instability-positive direction."""
    return any(
        coefficient.adjusted_p < 0.01 and coefficient.instability_positive
        for coefficient in (result.validity, result.tone)
    )


def _g3_reference(inputs, metric):
    """The sign-aligned false-failure effect a style effect must reach 50% of."""
    result = _g1_metric_complete(inputs.real_g1, inputs.primary_model_id, metric, "real_g1")
    if result is None:
        return None
    reference = result.validity.sign_aligned_coefficient
    return reference if _finite(reference) and reference != 0 else None


def _g3_style_reproduces(evidence, reference):
    """None means "evidence unavailable"; a bool is a determinate reproduction call."""
    if evidence is None or not isinstance(evidence, StyleEffectEvidence):
        return None
    if (
        evidence.metric_name not in PRIMARY_METRICS
        or evidence.style_id not in G3_STYLE_IDS
        or tuple(evidence.task_ids) != G3_SMOKE_TASK_IDS
        or evidence.neutral_style_id != G3_NEUTRAL_STYLE_ID
        or evidence.unavailable_reason
        or not _finite(evidence.effect)
        or not _finite(evidence.adjusted_p)
        or not 0 <= evidence.adjusted_p <= 1
    ):
        return None
    same_instability_direction = (evidence.effect > 0) == (reference > 0) and evidence.effect != 0
    return bool(
        same_instability_direction
        and abs(evidence.effect) >= 0.5 * abs(reference)
        and evidence.adjusted_p < 0.01
    )


def evaluate_g3(inputs: CoreGateInputs, evidence: G3Evidence | None, g1: GateVerdict | None = None):
    """The provisional Phase-1 five-item style smoke.

    A style condition reproduces a G1-qualifying metric only if its
    neutral-relative effect has the same instability direction, magnitude at
    least 50% of that metric's false-failure effect, and a within-phase
    BH-adjusted p below .01.  G3 passes if at least one G1 metric is *not*
    reproduced; every reproduced metric is flagged as a style meter.
    """
    if not isinstance(inputs, CoreGateInputs):
        raise GateInputError("G3 requires CoreGateInputs")
    if evidence is None:
        return G3Verdict(NOT_RUN, "style_smoke_not_run")
    if not isinstance(evidence, G3Evidence):
        raise GateInputError("G3 requires G3Evidence or None")
    if evidence.primary_model_id != inputs.primary_model_id or evidence.eligible_metrics != inputs.eligible_metrics:
        raise GateInputError("G3 evidence does not match core inputs")
    g1 = evaluate_g1(inputs) if g1 is None else g1
    if g1.status != PASS:
        return G3Verdict(NOT_EVALUATED, "g1_not_passed_not_unlocked")
    qualifying_metrics = tuple(dict.fromkeys(effect.metric_name for effect in g1.qualifying_effects))
    reproduced, non_reproduced, unavailable = [], [], False
    for metric in qualifying_metrics:
        reference = _g3_reference(inputs, metric)
        if reference is None:
            unavailable = True
            continue
        outcomes = [_g3_style_reproduces(evidence.effects.get((metric, style)), reference) for style in G3_STYLE_IDS]
        if any(outcome is None for outcome in outcomes):
            unavailable = True
        elif any(outcomes):
            reproduced.append(metric)
        else:
            non_reproduced.append(metric)
    if unavailable:
        return G3Verdict(UNAVAILABLE, "required_g3_style_evidence_unavailable", style_meter_metrics=tuple(reproduced))
    if non_reproduced:
        return G3Verdict(PASS, non_reproduced_metrics=tuple(non_reproduced), style_meter_metrics=tuple(reproduced))
    return G3Verdict(FAIL, "every_qualifying_metric_reproduced_by_style", style_meter_metrics=tuple(reproduced))


def evaluate_g4(evidence: G4Evidence):
    """Transfer or the preregistered Gemma/Qwen family boundary, without pooling.

    An eligible positive is a primary metric meeting G1 in the
    instability-positive direction.  Transfer requires the same metric to meet
    G1 with the same sign in the control model.  The family boundary requires at
    least one Gemma model to have an eligible positive and every evaluated Qwen
    to lack one for that metric.  G4 passes on transfer OR boundary; anything
    else is a messy/non-transfer failure.
    """
    if not isinstance(evidence, G4Evidence):
        raise GateInputError("G4 requires G4Evidence")
    evaluated = tuple(model for model in evidence.models if model.role == "evaluated")
    by_id = {model.model_id: model for model in evaluated}
    primary = by_id.get(evidence.primary_model_id)
    control = by_id.get(evidence.control_model_id)
    if primary is None:
        return G4Verdict(UNAVAILABLE, "primary_model_g1_evidence_unsupported")
    status: dict[tuple[str, str], bool | None] = {}
    for model in evaluated:
        for metric in evidence.eligible_metrics:
            result = _g1_metric_complete(model.real_g1, model.model_id, metric, "g4_real_g1")
            status[(model.model_id, metric)] = None if result is None else _eligible_positive(result)
    if all(status[(primary.model_id, metric)] is None for metric in evidence.eligible_metrics):
        return G4Verdict(UNAVAILABLE, "primary_model_g1_evidence_unavailable")
    gemmas = tuple(model for model in evaluated if model.family == "Gemma-2")
    qwens = tuple(model for model in evaluated if model.family == "Qwen2.5")
    positives, transfers, boundaries = [], [], []
    could_change = False
    for metric in evidence.eligible_metrics:
        primary_positive = status[(primary.model_id, metric)]
        if primary_positive:
            positives.append(metric)
        control_positive = status.get((control.model_id, metric)) if control is not None else None
        if primary_positive and control_positive:
            transfers.append(metric)
        gemma_states = [status[(model.model_id, metric)] for model in gemmas]
        qwen_states = [status[(model.model_id, metric)] for model in qwens]
        gemma_positive = any(state is True for state in gemma_states)
        if gemma_positive and qwen_states and all(state is False for state in qwen_states):
            boundaries.append(metric)
        # Only unknowns that could still create a pass make the gate indeterminate.
        if primary_positive and control_positive is None:
            could_change = True
        if gemma_positive and any(state is None for state in qwen_states) and not any(state is True for state in qwen_states):
            could_change = True
        if any(state is None for state in gemma_states) and qwen_states and all(state is False for state in qwen_states):
            could_change = True
    if transfers or boundaries:
        return G4Verdict(
            PASS,
            "transfer" if transfers else "family_boundary",
            tuple(transfers), tuple(boundaries), tuple(positives),
        )
    if could_change:
        return G4Verdict(UNAVAILABLE, "required_g4_real_g1_evidence_unavailable", eligible_positive_metrics=tuple(positives))
    if not positives:
        return G4Verdict(FAIL, "no_eligible_positive_in_primary_model")
    if not gemmas or not qwens:
        return G4Verdict(UNAVAILABLE, "required_g4_family_evidence_unavailable", eligible_positive_metrics=tuple(positives))
    return G4Verdict(FAIL, "messy_nontransfer", eligible_positive_metrics=tuple(positives))


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


def compose_phase_1_gates(inputs: Phase1GateInputs):
    """Compose all five preregistered Phase-1 gates into one verdict table.

    A determinate failure stays determinate: the project's debunk path requires
    that a real FAIL is not laundered into "incomplete" by a missing G3 smoke.
    A failed shuffled-label null blocks interpretation of every real verdict.
    """
    if not isinstance(inputs, Phase1GateInputs):
        raise GateInputError("inputs must be Phase1GateInputs")
    core = compose_core_gates(inputs.core)
    g3 = evaluate_g3(inputs.core, inputs.g3, core.g1)
    g4 = evaluate_g4(inputs.g4)
    gates = (core.g1, core.g2, g3, g4, core.g5)
    if core.shuffled_null.status != PASS:
        status, interpretable = BLOCKED, False
        reason = "blocked_by_shuffled_null:" + core.shuffled_null.status
    elif all(gate.status == PASS for gate in gates):
        status, interpretable, reason = PASS, True, None
    elif any(gate.status == FAIL for gate in gates):
        status, interpretable, reason = FAIL, True, "determinate_phase_1_gate_failure"
    else:
        status, interpretable = INCOMPLETE, False
        reason = "incomplete_real_phase_1_evidence:" + ",".join(
            name for name, gate in (("g1", core.g1), ("g2", core.g2), ("g3", g3), ("g4", g4), ("g5", core.g5))
            if gate.status != PASS
        )
    return Phase1GateSummary(
        inputs.core.primary_model_id, inputs.g4.control_model_id, inputs.core.eligible_metrics,
        core.shuffled_null, core.g1, core.g2, g3, g4, core.g5,
        g3.style_meter_metrics, g4.boundary_metrics, interpretable, reason, status,
    )
