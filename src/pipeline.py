"""Metric rows -> preregistered analysis inputs -> gate verdicts.

:mod:`src.extract` produces one flat endpoint table; :mod:`src.analysis` holds
the frozen statistics; :mod:`src.gates` composes verdicts from finished results.
This module is the only place that knows how to turn the first into the second
and third, so the selection rules and gate wiring are auditable in one file.

Everything here is deterministic and offline.  Nothing recomputes a statistic
that ``analysis`` already owns, and nothing decides a verdict that ``gates``
already owns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import math
import random
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .analysis import (
    FROZEN_MODEL_FAMILIES, METRIC_INSTABILITY_SIGN, PRIMARY_METRICS, AnalysisInputError, AnalysisObservation,
    G1MetricResult, G2Result, G5Result, G5Row, Phase0Selection, ReversalRow,
    benjamini_hochberg, freeze_neutral_standardization, g1_adjusted_effects, g2_reversal,
    g5_predictive_gap, g5_shuffled_feedback_labels, phase0_screen as analysis_phase0_screen,
    shuffled_feedback_labels, standardized_value,
)
from .extract import MetricRow
from .gates import (
    FAIL, G3_NEUTRAL_STYLE_ID, G3_SMOKE_TASK_IDS, G3_STYLE_IDS, NOT_RUN, PASS,
    CoreGateInputs, G3Evidence, G4Evidence, G4ModelEvidence, G5Evidence, GateInputError,
    Phase1GateInputs, StyleEffectEvidence, compose_core_gates, compose_phase_1_gates,
)
from .protocol import Protocol, load_protocol, phase0_screen_tasks

FACTORIAL_VALIDITIES = ("accurate", "malfunctioning_always_fail")
FACTORIAL_TONES = ("neutral", "hostile")
NEUTRAL_ACCURACY_TARGETS = MappingProxyType({"easy": 0.90, "hard": 0.50})
STANDARD_FEEDBACK_ROUNDS = 3
BOOTSTRAP_SAMPLES = 2000
QC_EXCLUSION_RATE = 0.05
M3_AUDIT_F1_FLOOR = 0.70
# Amendment A2: an item whose own accurate+neutral resample ensemble is this
# broken cannot support any within-item contrast for that model.
REQUIRED_RESAMPLES = 10
ITEM_EXCLUSION_INVALID_RESAMPLES = 5


@dataclass(frozen=True)
class Amendments:
    """Discovery-stage amendments, decided 2026-08-17 before Phase-1 generation.

    ``item_exclusion`` (A2) drops an item from every cell, endpoint and analysis
    for a model when that model's own accurate+neutral measured resample
    ensemble is mostly invalid -- a treatment-blind instrument-compliance check
    that never looks at a treatment cell.  ``pooled_sd_fallback`` (A3) rescales
    a metric by the model's pooled discovery factorial SD when its neutral SD is
    exactly zero.  Both are strict extensions: with a healthy baseline and a
    nonzero neutral SD they change nothing.  Phase 2 must apply them identically.
    """

    item_exclusion: bool = True
    pooled_sd_fallback: bool = True


FROZEN_RULES = Amendments(item_exclusion=False, pooled_sd_fallback=False)
AMENDED_RULES = Amendments(item_exclusion=True, pooled_sd_fallback=True)


class PipelineError(ValueError):
    """Raised when metric rows cannot be assembled into preregistered inputs."""


def _freeze(mapping):
    return MappingProxyType(dict(mapping))


def _jsonable(value):
    """Recursively convert results to JSON-safe values; non-finite floats -> None."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {"|".join(map(str, key)) if isinstance(key, tuple) else str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# Row selection helpers
# --------------------------------------------------------------------------

def _measured_factorial(rows, model_id, *, phase, split, turns=("measured",), excluded=()):
    excluded = frozenset(excluded)
    return tuple(
        row for row in rows
        if row.model_id == model_id
        and row.phase == phase
        and row.split == split
        and row.cell_kind == "factorial"
        and row.turn_label in turns
        and row.feedback_validity in FACTORIAL_VALIDITIES
        and row.tone in FACTORIAL_TONES
        and row.difficulty in ("easy", "hard")
        and row.task_id not in excluded
    )


# --------------------------------------------------------------------------
# Amendment A2 -- treatment-blind item exclusion
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemExclusion:
    """One item dropped for one model, with the treatment-blind evidence."""

    model_id: str
    task_id: str
    baseline_cell_id: str | None
    invalid_or_absent_resamples: int
    required_resamples: int
    reason: str


def item_exclusions(rows, model_id, *, phase=None, split="discovery"):
    """Items amendment A2 drops for one model, judged only on its own baseline.

    The decision uses exactly one cell -- that model's
    ``<difficulty>__accurate__neutral`` measured endpoint -- so it can never be
    informed by the manipulation.  An item whose baseline endpoint is absent is
    also dropped: without it there is no within-item contrast to make.
    """
    candidates = tuple(
        row for row in rows
        if row.model_id == model_id and row.cell_kind == "factorial"
        and row.turn_label == "measured" and row.split == split
        and (phase is None or row.phase == phase)
        and row.difficulty in ("easy", "hard")
        and row.feedback_validity in FACTORIAL_VALIDITIES and row.tone in FACTORIAL_TONES
    )
    baselines = {
        row.task_id: row for row in candidates
        if row.feedback_validity == "accurate" and row.tone == "neutral"
    }
    out = []
    for task_id in sorted({row.task_id for row in candidates}):
        baseline = baselines.get(task_id)
        if baseline is None:
            out.append(ItemExclusion(model_id, task_id, None, REQUIRED_RESAMPLES, REQUIRED_RESAMPLES,
                                     "accurate_neutral_measured_endpoint_absent"))
            continue
        broken = REQUIRED_RESAMPLES - baseline.resample_valid_count
        if broken >= ITEM_EXCLUSION_INVALID_RESAMPLES:
            out.append(ItemExclusion(model_id, task_id, baseline.cell_id, broken, REQUIRED_RESAMPLES,
                                     "at_least_5_of_10_baseline_resamples_invalid_or_absent"))
    return tuple(out)


def excluded_task_ids(rows, model_id, amendments=AMENDED_RULES, **kwargs):
    if not amendments.item_exclusion:
        return frozenset()
    return frozenset(item.task_id for item in item_exclusions(rows, model_id, **kwargs))


def _observation(row: MetricRow, metric: str) -> AnalysisObservation:
    value, reason = row.metric(metric)
    return AnalysisObservation(
        experiment_phase=row.phase,
        run_id=row.run_id,
        split=row.split,
        model_id=row.model_id,
        task_id=row.task_id,
        cell_id=row.cell_id,
        difficulty=row.difficulty,
        feedback_validity=row.feedback_validity,
        tone=row.tone,
        turn=row.turn_label,
        metric_name=metric,
        metric_value=value,
        missing_reason=None if value is not None else (reason or "metric_unavailable"),
        correctness=row.greedy_answer_correct,
        generated_response_tokens=row.length_tokens,
        false_negative_history_eligible=bool(row.history_false_negative),
    )


def build_observations(rows, model_id, *, metrics=PRIMARY_METRICS, phase="phase_1", split="discovery", turns=("measured",), amendments=AMENDED_RULES):
    """Build :class:`AnalysisObservation` rows for one model, one metric family."""
    excluded = excluded_task_ids(rows, model_id, amendments, phase=phase, split=split)
    selected = _measured_factorial(rows, model_id, phase=phase, split=split, turns=turns, excluded=excluded)
    return tuple(_observation(row, metric) for row in selected for metric in metrics)


def build_g1_observations(rows, model_id, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """G1 consumes measured discovery endpoints only, one row per metric."""
    return build_observations(rows, model_id, metrics=metrics, turns=("measured",), amendments=amendments)


def _neutral_standardization(rows, model_id, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """The frozen z-scoring baseline: the model's accurate+neutral discovery measured rows."""
    return freeze_neutral_standardization(
        build_observations(rows, model_id, metrics=metrics, turns=("measured", "recovery"), amendments=amendments),
        pooled_sd_fallback=amendments.pooled_sd_fallback,
    )


def build_reversal_rows(rows, model_id, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """Build G2 rows: matched measured accurate / measured malfunctioning / recovery."""
    frozen = _neutral_standardization(rows, model_id, metrics=metrics, amendments=amendments)
    excluded = excluded_task_ids(rows, model_id, amendments, phase="phase_1", split="discovery")
    measured = {}
    recovery = {}
    eligible = {}
    for row in _measured_factorial(rows, model_id, phase="phase_1", split="discovery", turns=("measured", "recovery"), excluded=excluded):
        for metric in metrics:
            observation = _observation(row, metric)
            value = standardized_value(observation, frozen)
            key = (metric, row.task_id, row.tone, row.feedback_validity)
            if row.turn_label == "measured":
                measured[key] = value
                if row.feedback_validity == "malfunctioning_always_fail":
                    eligible[(row.task_id, row.tone)] = bool(row.history_false_negative)
            else:
                recovery[key] = value
    out = []
    for metric in metrics:
        keys = sorted({(task, tone) for (name, task, tone, _) in measured if name == metric})
        for task, tone in keys:
            out.append(ReversalRow(
                "phase_1", _single_run(rows, model_id), "discovery", model_id, metric, task, tone,
                measured.get((metric, task, tone, "accurate")),
                measured.get((metric, task, tone, "malfunctioning_always_fail")),
                recovery.get((metric, task, tone, "malfunctioning_always_fail")),
                eligible.get((task, tone), False),
            ))
    return tuple(out)


def _single_run(rows, model_id):
    runs = sorted({row.run_id for row in rows if row.model_id == model_id})
    if len(runs) != 1:
        raise PipelineError("exactly one run_id is required per model, found %d" % len(runs))
    return runs[0]


def build_g5_rows(rows, model_id, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """Build G5 rows over the eight discovery cells' measured item-cell observations.

    Raw metric values are supplied deliberately: G5 z-standardises inside each
    training fold, so a global affine rescaling cannot change the fitted model,
    and passing raw values keeps a zero neutral SD from silently deleting rows.
    """
    excluded = excluded_task_ids(rows, model_id, amendments, phase="phase_1", split="discovery")
    out = []
    for row in _measured_factorial(rows, model_id, phase="phase_1", split="discovery", excluded=excluded):
        out.append(G5Row(
            "phase_1", row.run_id, "discovery", model_id, row.task_id, row.cell_id,
            row.difficulty, row.feedback_validity, row.tone, "measured",
            {metric: row.metric(metric)[0] for metric in metrics},
            row.greedy_answer_correct, row.length_tokens,
        ))
    return tuple(sorted(out, key=lambda item: (item.task_id, item.cell_id)))


# --------------------------------------------------------------------------
# Metric eligibility (the frozen QC exclusion rules)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricEligibility:
    metric_name: str
    eligible: bool
    reason: str | None
    worst_cell_id: str | None
    worst_rate: float | None


def metric_eligibility(rows, model_id, *, m3_audit_f1=None, phase="phase_1", split="discovery", amendments=AMENDED_RULES):
    """Apply the preregistered confirmatory exclusion rules to one model.

    M1 is excluded if more than 5% of required greedy trials are missing in any
    confirmatory condition; M2 if more than 5% of required sampled responses are
    invalid or absent (k stays frozen at 10); M3 only if a human audit reports
    F1 below .7 -- which cannot be computed offline, so M3 stays eligible unless
    an audit F1 is supplied.  Amendment A2's item exclusion is applied first, so
    these rates describe the items that actually enter the analysis.
    """
    excluded = excluded_task_ids(rows, model_id, amendments, phase=phase, split=split)
    selected = _measured_factorial(rows, model_id, phase=phase, split=split, excluded=excluded)
    if not selected:
        return tuple(
            MetricEligibility(metric, False, "no_confirmatory_rows", None, None)
            for metric in PRIMARY_METRICS
        )
    by_cell: dict[str, list[MetricRow]] = {}
    for row in selected:
        by_cell.setdefault(row.cell_id, []).append(row)
    out = []
    for metric in PRIMARY_METRICS:
        worst_cell, worst_rate = None, None
        for cell_id in sorted(by_cell):
            group = by_cell[cell_id]
            if metric == "M2":
                required = 10 * len(group)
                rate = (required - sum(row.resample_valid_count for row in group)) / required
            else:
                rate = sum(1 for row in group if row.metric(metric)[0] is None) / len(group)
            if worst_rate is None or rate > worst_rate:
                worst_cell, worst_rate = cell_id, rate
        if metric == "M3":
            failed = m3_audit_f1 is not None and m3_audit_f1 < M3_AUDIT_F1_FLOOR
            reason = "m3_parser_audit_f1_below_0.7" if failed else (
                None if m3_audit_f1 is not None else "m3_audit_f1_not_supplied_eligible_by_default"
            )
            out.append(MetricEligibility(metric, not failed, reason, worst_cell, worst_rate))
            continue
        excluded = worst_rate is not None and worst_rate > QC_EXCLUSION_RATE
        out.append(MetricEligibility(
            metric, not excluded,
            ("m1_missing_rate_above_5_percent" if metric == "M1" else "m2_invalid_sampled_response_rate_above_5_percent") if excluded else None,
            worst_cell, worst_rate,
        ))
    return tuple(out)


# --------------------------------------------------------------------------
# Phase 0 screen
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NeutralAccuracy:
    """Observed neutral greedy accuracy; reported only, never used to relabel."""

    model_id: str
    difficulty: str
    n_items: int
    n_correct: int
    accuracy: float | None
    target: float


@dataclass(frozen=True)
class Phase0ScreenResult:
    """Both selections are always reported; ``selection`` is the authoritative one."""

    selection: Phase0Selection
    frozen_selection: Phase0Selection
    amended_selection: Phase0Selection
    amendments: Amendments
    amendments_authoritative: bool
    item_exclusions: Mapping[str, tuple[ItemExclusion, ...]]
    model_ids: tuple[str, ...]
    screen_task_ids: tuple[str, ...]
    escalated: bool
    observed_feedback_rounds: tuple[int, ...]
    neutral_accuracy: tuple[NeutralAccuracy, ...]
    n_endpoints: int
    n_observations: int

    def to_dict(self):
        return _jsonable(self)


def _neutral_accuracy(rows, model_ids, screen_tasks):
    out = []
    for model_id in model_ids:
        for difficulty in ("easy", "hard"):
            group = [
                row for row in rows
                if row.model_id == model_id and row.difficulty == difficulty
                and row.task_id in screen_tasks and row.turn_label == "measured"
                and row.cell_id == difficulty + "__accurate__neutral"
            ]
            correct = sum(1 for row in group if row.greedy_answer_correct)
            out.append(NeutralAccuracy(
                model_id, difficulty, len(group), correct,
                correct / len(group) if group else None,
                NEUTRAL_ACCURACY_TARGETS[difficulty],
            ))
    return tuple(out)


def phase0_screen(rows, *, protocol: Protocol | None = None, screen_task_ids=None, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """Run the preregistered Phase-0 screen over a Phase-0 metric-row table.

    Screen items default to the frozen rule (the lexicographically smallest
    discovery task ID in each difficulty x domain stratum).  Escalation is
    detected from the data: a conversation carrying five graded feedback rounds
    is the one permitted escalation, after which a null screen is labelled
    ``screen_null`` rather than demanding another escalation.

    Both the frozen-rule selection and the A2+A3 amended selection are always
    computed and reported; ``amendments`` only decides which one is
    authoritative, so the frozen outcome stays reproducible either way.
    """
    rows = tuple(rows)
    phase0 = tuple(
        row for row in rows
        if row.phase == "phase_0" and row.split == "discovery" and row.cell_kind == "factorial"
        and row.turn_label == "measured" and row.tone == "neutral"
    )
    if screen_task_ids is None:
        try:
            screen_task_ids = tuple(task.task_id for task in phase0_screen_tasks(protocol or load_protocol()))
        except Exception:  # noqa: BLE001 - fixture unavailable; fall back to observed items
            screen_task_ids = tuple(sorted({row.task_id for row in phase0}))
    screen_task_ids = tuple(sorted(set(screen_task_ids)))
    selected = tuple(row for row in phase0 if row.task_id in screen_task_ids)
    model_ids = tuple(sorted({row.model_id for row in selected}))
    rounds = tuple(sorted({row.feedback_rounds for row in selected}))
    escalated = any(count > STANDARD_FEEDBACK_ROUNDS for count in rounds)
    order = _frozen_model_order(protocol)
    exclusions = {
        model_id: item_exclusions(selected, model_id, phase="phase_0", split="discovery")
        for model_id in model_ids
    }

    def screen(rules):
        observations = tuple(
            observation
            for model_id in model_ids
            for observation in build_observations(
                selected, model_id, metrics=metrics, phase="phase_0", split="discovery",
                turns=("measured",), amendments=rules,
            )
        )
        selection = analysis_phase0_screen(
            observations,
            screen_task_ids=screen_task_ids or None,
            frozen_model_order=order,
            escalated=escalated,
            pooled_sd_fallback=rules.pooled_sd_fallback,
        )
        return selection, len(observations)

    frozen_selection, frozen_count = screen(FROZEN_RULES)
    amended_selection, amended_count = screen(AMENDED_RULES)
    authoritative = amendments != FROZEN_RULES
    if amendments == AMENDED_RULES:
        selection, count = amended_selection, amended_count
    elif amendments == FROZEN_RULES:
        selection, count = frozen_selection, frozen_count
    else:
        selection, count = screen(amendments)
    return Phase0ScreenResult(
        selection, frozen_selection, amended_selection, amendments, authoritative,
        _freeze({model_id: items for model_id, items in exclusions.items() if items}),
        model_ids, screen_task_ids, escalated, rounds,
        _neutral_accuracy(selected, model_ids, set(screen_task_ids)),
        len(selected), count,
    )


def _frozen_model_order(protocol: Protocol | None):
    try:
        return tuple((protocol or load_protocol()).models["phase_0_screen_order"])
    except Exception:  # noqa: BLE001 - configuration unavailable; analysis default applies
        from .analysis import PHASE0_MODELS

        return PHASE0_MODELS


def _selection_line(label, selection):
    return "| %s | %s | `%s` | `%s` | %s | %s |" % (
        label, selection.status,
        selection.primary_model_id or "none", selection.control_model_id or "none",
        "yes" if selection.screen_null else "no",
        "`%s`" % selection.blocked_reason if selection.blocked_reason else "",
    )


def _delta_table(selection, model_ids):
    lines = [
        "| model | M1 | M2 | M3 | S | coherent | paired items (M1/M2/M3) | z scale (M1/M2/M3) |",
        "| --- | ---: | ---: | ---: | ---: | :---: | --- | --- |",
    ]
    for model_id in model_ids:
        screen = selection.models.get(model_id)
        if screen is None:
            continue
        cells, counts, scales = [], [], []
        for metric in PRIMARY_METRICS:
            item = screen.metrics[metric]
            cells.append("%.3f" % item.signed_delta if item.signed_delta is not None else "n/a")
            counts.append(str(item.n_paired_items))
            scales.append({"neutral": "neutral", "pooled_factorial": "**pooled**"}.get(item.scale_source, "-"))
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
            model_id, cells[0], cells[1], cells[2],
            "%.3f" % screen.score if screen.score is not None else "n/a",
            "yes" if screen.coherent else "no", "/".join(counts), "/".join(scales),
        ))
    return lines


def render_phase0_markdown(result: Phase0ScreenResult) -> str:
    """Human-readable Phase-0 screen report, frozen and amended side by side."""
    selection = result.selection
    authoritative = "amended (A2+A3)" if result.amendments_authoritative else "frozen rules"
    lines = [
        "# Phase-0 screen",
        "",
        "- **authoritative selection: %s** (`--no-amendments` reproduces the frozen-only outcome)",
        "- status: **%s**" % selection.status,
        "- primary: `%s`" % (selection.primary_model_id or "none"),
        "- control (weak/null): `%s`" % (selection.control_model_id or "none"),
        "- screen null label: %s" % ("yes" if selection.screen_null else "no"),
        "- escalation (5 feedback rounds observed): %s" % ("yes" if result.escalated else "no"),
        "- screen items (%d): %s" % (len(result.screen_task_ids), ", ".join(result.screen_task_ids)),
        "- endpoints: %d; observations: %d; ignored non-neutral rows: %d" % (
            result.n_endpoints, result.n_observations, selection.ignored_row_count),
    ]
    lines[2] = lines[2] % authoritative
    if selection.blocked_reason:
        lines.append("- reason: `%s`" % selection.blocked_reason)
    lines += [
        "",
        "## Selection under both rule sets",
        "",
        "| rules | status | primary | control | screen null | reason |",
        "| --- | --- | --- | --- | :---: | --- |",
        _selection_line("frozen (preregistered)", result.frozen_selection),
        _selection_line("amended A2+A3", result.amended_selection),
        "",
        "A2 excludes an item for a model when at least 5 of that model's 10",
        "accurate+neutral measured resamples are invalid or absent (treatment-blind).",
        "A3 rescales a metric by the model's pooled discovery factorial SD when its",
        "neutral SD is exactly zero. Both were decided on 2026-08-17 from discovery",
        "data before Phase-1 generation and apply identically to the Phase-2 holdout.",
        "",
        "## Amendment A2 - excluded items",
        "",
    ]
    if result.item_exclusions:
        lines += [
            "| model | item | baseline cell | invalid/absent baseline resamples | reason |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for model_id in sorted(result.item_exclusions):
            for item in result.item_exclusions[model_id]:
                lines.append("| `%s` | %s | %s | %d/%d | `%s` |" % (
                    model_id, item.task_id, item.baseline_cell_id or "absent",
                    item.invalid_or_absent_resamples, item.required_resamples, item.reason))
    else:
        lines.append("No item met the A2 exclusion threshold for any model.")
    lines += [
        "",
        "## Standardised screen deltas, amended A2+A3 (higher = more instability)",
        "",
    ]
    lines += _delta_table(result.amended_selection, result.model_ids)
    lines += [
        "",
        "## Standardised screen deltas, frozen rules",
        "",
    ]
    lines += _delta_table(result.frozen_selection, result.model_ids)
    lines += [
        "",
        "## Observed neutral greedy accuracy (reported, never used to relabel)",
        "",
        "| model | difficulty | n | correct | accuracy | provisional target |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in result.neutral_accuracy:
        lines.append("| `%s` | %s | %d | %d | %s | %.2f |" % (
            item.model_id, item.difficulty, item.n_items, item.n_correct,
            "%.3f" % item.accuracy if item.accuracy is not None else "n/a", item.target,
        ))
    lines += [
        "",
        "Difficulty labels are provisional and frozen: a missed target is reported and",
        "carried into analysis as observed baseline difficulty, never relabelled.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Shuffled-label null
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ShuffledNullResult:
    model_id: str
    g1: Mapping[str, G1MetricResult | None]
    g5: G5Result | None
    passed: bool
    reason: str | None


def _g1_for_models(observations, metrics, amendments=AMENDED_RULES):
    """Run G1 once over every supplied model so BH is a single within-phase family."""
    try:
        return g1_adjusted_effects(observations, metrics, pooled_sd_fallback=amendments.pooled_sd_fallback), None
    except AnalysisInputError as error:
        return {}, "g1_unavailable:" + str(error)


def _g5_for_model(rows, model_id, metrics, *, shuffled=False, amendments=AMENDED_RULES):
    source = build_g5_rows(rows, model_id, metrics=metrics, amendments=amendments)
    if not source:
        return None, "no_g5_rows"
    try:
        if shuffled:
            source = g5_shuffled_feedback_labels(source)
        return g5_predictive_gap(source, metrics), None
    except AnalysisInputError as error:
        return None, "g5_unavailable:" + str(error)


def _qualifies(result):
    if result is None or result.validity is None or result.tone is None:
        return None
    values = [result.validity.adjusted_p, result.tone.adjusted_p]
    if any(value is None for value in values):
        return None
    return any(value < 0.01 for value in values)


def shuffled_nulls(rows, model_ids, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """Run the frozen label shuffle for several models under one BH family."""
    model_ids = tuple(model_ids)
    observations = []
    unavailable = {}
    for model_id in model_ids:
        try:
            observations.extend(shuffled_feedback_labels(
                build_g1_observations(rows, model_id, metrics=metrics, amendments=amendments)))
        except AnalysisInputError as error:
            unavailable[model_id] = "shuffle_unavailable:" + str(error)
    effects, g1_reason = _g1_for_models(tuple(observations), metrics, amendments) if observations else ({}, "shuffle_unavailable:no_rows")
    out = {}
    for model_id in model_ids:
        reasons = [value for value in (unavailable.get(model_id), g1_reason) if value]
        g1 = {metric: effects.get((model_id, metric)) for metric in metrics}
        g5, g5_reason = _g5_for_model(rows, model_id, metrics, shuffled=True, amendments=amendments)
        if g5_reason:
            reasons.append(g5_reason)
        outcomes = [_qualifies(result) for result in g1.values()]
        if any(outcome is None for outcome in outcomes):
            reasons.append("shuffled_g1_incomplete")
        elif any(outcomes):
            reasons.append("shuffled_g1_false_positive")
        gap = None if g5 is None else g5.auc_gap
        if gap is None:
            reasons.append("shuffled_g5_gap_unavailable")
        elif gap >= 0.1:
            reasons.append("shuffled_auc_gap_not_below_0.1")
        out[model_id] = ShuffledNullResult(model_id, _freeze(g1), g5, not reasons, ";".join(reasons) or None)
    return _freeze(out)


def shuffled_null(rows, model_id, *, metrics=PRIMARY_METRICS, amendments=AMENDED_RULES):
    """Run the preregistered deterministic label shuffle through G1 and G5.

    It passes only if the same pipeline finds no primary BH-adjusted p below .01
    and the shuffled G5 AUC gap stays below .1.
    """
    return shuffled_nulls(rows, (model_id,), metrics=metrics, amendments=amendments)[model_id]


# --------------------------------------------------------------------------
# G3 style smoke glue
# --------------------------------------------------------------------------

def _student_t_two_sided(t_statistic: float, degrees: int):
    """Two-sided Student-t p-value; falls back to a normal tail without SciPy."""
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_statistic), degrees)), None
    except ImportError:
        return float(math.erfc(abs(t_statistic) / math.sqrt(2.0))), "normal_approximation_scipy_unavailable"


def _paired_t(differences: Sequence[float]):
    """Paired t-test over item differences; returns (mean, p, reason)."""
    values = [float(value) for value in differences]
    if len(values) < 2:
        return None, None, "at_least_two_paired_items_required"
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0:
        return average, None, "zero_paired_difference_variance"
    statistic = average / math.sqrt(variance / len(values))
    probability, note = _student_t_two_sided(statistic, len(values) - 1)
    return average, probability, note


def style_effects(rows, style_rows, model_id, metrics, *, smoke_task_ids=G3_SMOKE_TASK_IDS, amendments=AMENDED_RULES):
    """Compute the frozen five-item, four-style G3 effects from metric rows.

    Each effect is the item-paired mean of (style - neutral reference) divided by
    the same model-neutral SD that standardises G1, then sign-aligned so higher
    means more instability.  Significance uses a paired t-test across the five
    frozen smoke items -- small, and stated as such -- BH-adjusted within the
    G3 family.  Amendment A2 drops any excluded smoke item, so the frozen
    five-item design may be analysed on a subset; the analysed IDs travel with
    the evidence.
    """
    frozen = _neutral_standardization(rows, model_id, metrics=metrics, amendments=amendments)
    dropped = excluded_task_ids(rows, model_id, amendments, phase="phase_1", split="discovery")
    by_key = {}
    for row in style_rows:
        if row.model_id != model_id or row.cell_kind != "non_factorial" or row.turn_label != "measured":
            continue
        if not row.cell_id.startswith("style__"):
            continue
        by_key[(row.task_id, row.cell_id)] = row
    tasks = tuple(sorted(set(smoke_task_ids) - dropped))
    raw = {}
    effects = {}
    for metric in metrics:
        standardization = frozen.get((model_id, metric))
        sign = METRIC_INSTABILITY_SIGN[metric]
        for style_id in G3_STYLE_IDS:
            differences = []
            analysed = []
            for task in tasks:
                reference = by_key.get((task, G3_NEUTRAL_STYLE_ID))
                styled = by_key.get((task, style_id))
                if reference is None or styled is None:
                    continue
                left, right = styled.metric(metric)[0], reference.metric(metric)[0]
                if left is None or right is None:
                    continue
                differences.append(left - right)
                analysed.append(task)
            item_ids = tuple(analysed)
            if standardization is None or not standardization.available:
                effects[(metric, style_id)] = StyleEffectEvidence(
                    model_id, metric, style_id, item_ids, G3_NEUTRAL_STYLE_ID,
                    None, None, len(differences), "neutral_standardization_unavailable")
                continue
            average, probability, note = _paired_t(differences)
            if probability is None:
                effects[(metric, style_id)] = StyleEffectEvidence(
                    model_id, metric, style_id, item_ids, G3_NEUTRAL_STYLE_ID,
                    None if average is None else sign * average / standardization.sample_sd,
                    None, len(differences), note or "style_effect_unavailable")
                continue
            raw[(metric, style_id)] = probability
            effects[(metric, style_id)] = StyleEffectEvidence(
                model_id, metric, style_id, item_ids, G3_NEUTRAL_STYLE_ID,
                sign * average / standardization.sample_sd, None, len(differences), note)
    adjusted = benjamini_hochberg(raw) if raw else {}
    return _freeze({
        key: (value if key not in adjusted else StyleEffectEvidence(
            value.model_id, value.metric_name, value.style_id, value.task_ids, value.neutral_style_id,
            value.effect, float(adjusted[key]), value.n_items, value.unavailable_reason))
        for key, value in effects.items()
    })


# --------------------------------------------------------------------------
# Reversal profile (figure F4)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    value: float | None
    ci95_lower: float | None
    ci95_upper: float | None


@dataclass(frozen=True)
class ReversalProfile:
    """Sign-aligned group means for the three reversal endpoints, with item CIs."""

    model_id: str
    metric_name: str
    n_items: int
    n_rows: int
    measured_accurate: Estimate
    measured_malfunctioning: Estimate
    post_correction_malfunctioning: Estimate
    unavailable_reason: str | None = None


def _bootstrap_estimate(by_item, index, seed_text):
    items = sorted(by_item)
    point = mean(value[index] for values in by_item.values() for value in values)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big"))
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [rng.choice(items) for _ in items]
        draws.append(mean(value[index] for item in sampled for value in by_item[item]))
    draws.sort()

    def quantile(probability):
        position = (len(draws) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)

    return Estimate(point, quantile(0.025), quantile(0.975))


def reversal_profile(reversal_rows, model_id, metric):
    """Item-clustered bootstrap means of the three reversal endpoints (sign-aligned)."""
    sign = METRIC_INSTABILITY_SIGN[metric]
    good = [
        row for row in reversal_rows
        if row.model_id == model_id and row.metric_name == metric and row.false_negative_history_eligible
        and None not in (row.measured_accurate, row.measured_malfunctioning, row.post_correction_malfunctioning)
    ]
    empty = Estimate(None, None, None)
    if not good:
        return ReversalProfile(model_id, metric, 0, 0, empty, empty, empty, "no_complete_false_negative_eligible_rows")
    by_item = {}
    for row in good:
        by_item.setdefault(row.task_id, []).append((
            sign * row.measured_accurate,
            sign * row.measured_malfunctioning,
            sign * row.post_correction_malfunctioning,
        ))
    if len(by_item) < 2:
        return ReversalProfile(model_id, metric, len(by_item), len(good), empty, empty, empty, "at_least_two_items_required_for_cluster_ci")
    seed = "DGS-AC1-F4-BOOTSTRAP-v1|%s|%s|" % (model_id, metric)
    return ReversalProfile(
        model_id, metric, len(by_item), len(good),
        _bootstrap_estimate(by_item, 0, seed + "measured_accurate"),
        _bootstrap_estimate(by_item, 1, seed + "measured_malfunctioning"),
        _bootstrap_estimate(by_item, 2, seed + "post_correction_malfunctioning"),
    )


# --------------------------------------------------------------------------
# Phase 1 composition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelAnalysis:
    model_id: str
    eligible_metrics: tuple[str, ...]
    eligibility: tuple[MetricEligibility, ...]
    real_g1: Mapping[str, G1MetricResult | None]
    shuffled: ShuffledNullResult
    real_g2: Mapping[str, G2Result | None]
    real_g5: G5Result | None
    reversal: tuple[ReversalProfile, ...]
    n_endpoints: int
    unavailable_reasons: tuple[str, ...]
    item_exclusions: tuple[ItemExclusion, ...] = ()
    standardization: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Phase1Verdict:
    primary_model_id: str
    control_model_id: str
    eligible_metrics: tuple[str, ...]
    models: Mapping[str, ModelAnalysis]
    core: Mapping[str, Any]
    summary: Any
    style: Mapping[tuple[str, str], StyleEffectEvidence]
    amendments: Amendments = AMENDED_RULES

    def to_dict(self):
        return _jsonable(self)


def _core_inputs(model_id, family, real_g1, shuffled, real_g2, real_g5):
    return CoreGateInputs(
        model_id, family,
        {metric: real_g1.get(metric) for metric in family},
        {metric: shuffled.g1.get(metric) for metric in family},
        {metric: real_g2.get(metric) for metric in family},
        G5Evidence(model_id, family, real_g5),
        G5Evidence(model_id, family, shuffled.g5),
    )


def run_phase1_gates(rows, primary_model, control_model, *, style_rows=None, m3_audit_f1=None, amendments=AMENDED_RULES):
    """Compose the full five-gate Phase-1 verdict from one metric-row table.

    G1's Benjamini-Hochberg family is fit once across both evaluated models, so
    "within phase" means exactly that.  G3 is evaluated for the primary model
    only (the frozen smoke is a primary-model design); when no style rows are
    supplied G3 is ``NOT_RUN`` and Phase 1 stays provisional rather than passing.
    ``amendments`` selects the A2/A3 rule set; ``FROZEN_RULES`` reproduces the
    preregistered-only analysis.
    """
    rows = tuple(rows)
    if primary_model == control_model:
        raise PipelineError("primary and control models must be distinct")
    exclusions = {
        model: excluded_task_ids(rows, model, amendments, phase="phase_1", split="discovery")
        for model in (primary_model, control_model)
    }
    eligibility = {
        model: metric_eligibility(rows, model, m3_audit_f1=m3_audit_f1, amendments=amendments)
        for model in (primary_model, control_model)
    }
    family = tuple(item.metric_name for item in eligibility[primary_model] if item.eligible)
    if not family:
        raise PipelineError("no eligible primary metric survives the frozen QC rules for " + primary_model)
    observations = tuple(
        observation for model in (primary_model, control_model)
        for observation in build_g1_observations(rows, model, metrics=family, amendments=amendments)
    )
    effects, g1_reason = _g1_for_models(observations, family, amendments)
    nulls = shuffled_nulls(rows, (primary_model, control_model), metrics=family, amendments=amendments)
    analyses = {}
    for model in (primary_model, control_model):
        reasons = [] if g1_reason is None else [g1_reason]
        real_g1 = {metric: effects.get((model, metric)) for metric in family}
        null = nulls[model]
        reversal_rows = build_reversal_rows(rows, model, metrics=family, amendments=amendments)
        real_g2 = {}
        for metric in family:
            subset = [row for row in reversal_rows if row.metric_name == metric]
            try:
                real_g2[metric] = g2_reversal(subset) if subset else None
            except AnalysisInputError as error:
                real_g2[metric] = None
                reasons.append("g2_unavailable:%s:%s" % (metric, error))
        real_g5, g5_reason = _g5_for_model(rows, model, family, amendments=amendments)
        if g5_reason:
            reasons.append(g5_reason)
        scale = _neutral_standardization(rows, model, metrics=family, amendments=amendments)
        analyses[model] = ModelAnalysis(
            model, family, eligibility[model], _freeze(real_g1), null, _freeze(real_g2), real_g5,
            tuple(reversal_profile(reversal_rows, model, metric) for metric in family),
            len(_measured_factorial(rows, model, phase="phase_1", split="discovery", excluded=exclusions[model])),
            tuple(reasons),
            item_exclusions(rows, model, phase="phase_1", split="discovery") if amendments.item_exclusion else (),
            _freeze({metric: scale.get((model, metric)) for metric in family}),
        )
    core_inputs = {
        model: _core_inputs(model, family, analyses[model].real_g1, analyses[model].shuffled,
                            analyses[model].real_g2, analyses[model].real_g5)
        for model in (primary_model, control_model)
    }
    core = {model: compose_core_gates(value) for model, value in core_inputs.items()}
    style = style_effects(rows, style_rows, primary_model, family, amendments=amendments) if style_rows else _freeze({})
    g3 = G3Evidence(primary_model, family, dict(style)) if style else None
    try:
        g4 = G4Evidence(primary_model, control_model, family, (
            _g4_model(primary_model, family, analyses[primary_model].real_g1),
            _g4_model(control_model, family, analyses[control_model].real_g1),
        ))
    except GateInputError as error:
        raise PipelineError("G4 model binding failed: %s" % error) from error
    summary = compose_phase_1_gates(Phase1GateInputs(core_inputs[primary_model], g4, g3))
    return Phase1Verdict(primary_model, control_model, family, _freeze(analyses), _freeze(core), summary, style, amendments)


def _g4_model(model_id, family, real_g1):
    return G4ModelEvidence(model_id, FROZEN_MODEL_FAMILIES.get(model_id, ""), family, dict(real_g1))


def _status_cell(verdict):
    return "**%s**" % verdict.status if verdict.status in (PASS, FAIL) else verdict.status


def render_phase1_markdown(verdict: Phase1Verdict) -> str:
    """The five-gate verdict table -- the project's definition-of-done artifact."""
    summary = verdict.summary
    lines = [
        "# Phase-1 five-gate verdict (discovery)",
        "",
        "- primary: `%s`" % verdict.primary_model_id,
        "- control: `%s`" % verdict.control_model_id,
        "- eligible primary metrics: %s" % ", ".join(verdict.eligible_metrics),
        "- rule set: **%s** (A2 item exclusion: %s; A3 pooled-SD fallback: %s)" % (
            "amended" if verdict.amendments != FROZEN_RULES else "frozen (preregistered only)",
            "on" if verdict.amendments.item_exclusion else "off",
            "on" if verdict.amendments.pooled_sd_fallback else "off"),
        "- Phase-1 status: **%s**" % summary.phase_1_status,
        "- interpretable: %s%s" % (
            "yes" if summary.interpretable else "no",
            "" if summary.interpretation_reason is None else " (`%s`)" % summary.interpretation_reason),
        "",
        "## Gate table",
        "",
        "| gate | %s (primary) | %s (control) |" % (verdict.primary_model_id, verdict.control_model_id),
        "| --- | --- | --- |",
    ]
    control_core = verdict.core[verdict.control_model_id]
    lines.append("| shuffled-label null | %s | %s |" % (
        _status_cell(summary.shuffled_null), _status_cell(control_core.shuffled_null)))
    lines.append("| G1 false-failure/tone effect | %s | %s |" % (
        _status_cell(summary.g1), _status_cell(control_core.g1)))
    lines.append("| G2 cause-removal reversal | %s | %s |" % (
        _status_cell(summary.g2), _status_cell(control_core.g2)))
    lines.append("| G3 style resistance (provisional smoke) | %s | not applicable |" % _status_cell(summary.g3))
    lines.append("| G4 transfer / family boundary | %s | (contributes) |" % _status_cell(summary.g4))
    lines.append("| G5 classifier AUC gap | %s | %s |" % (
        _status_cell(summary.g5), _status_cell(control_core.g5)))
    lines += [
        "",
        "Gate reasons: G1 `%s`; G2 `%s`; G3 `%s`; G4 `%s`; G5 `%s`." % (
            summary.g1.reason, summary.g2.reason, summary.g3.reason, summary.g4.reason, summary.g5.reason),
        "",
        "## Amendment A2 - excluded items, and A3 - z scale per metric",
        "",
        "| model | excluded items | z scale used |",
        "| --- | --- | --- |",
    ]
    for model_id, analysis in verdict.models.items():
        dropped = ", ".join(
            "%s (%d/%d baseline resamples invalid or absent)" % (
                item.task_id, item.invalid_or_absent_resamples, item.required_resamples)
            for item in analysis.item_exclusions) or "none"
        scales = []
        for metric in verdict.eligible_metrics:
            scale = analysis.standardization.get(metric)
            scales.append("%s: %s" % (metric, (scale.scale_source or "unavailable") if scale is not None else "unavailable"))
        lines.append("| `%s` | %s | %s |" % (model_id, dropped, ", ".join(scales)))
    lines += [
        "",
        "## G1 adjusted effects (z vs same-model neutral discovery)",
        "",
        "| model | metric | effect | coefficient | 95% CI | BH p | sign-aligned | qualifies |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | :---: |",
    ]
    for model_id, analysis in verdict.models.items():
        for metric in verdict.eligible_metrics:
            result = analysis.real_g1.get(metric)
            if result is None or result.validity is None:
                lines.append("| `%s` | %s | - | - | - | - | - | unavailable (`%s`) |" % (
                    model_id, metric, None if result is None else result.unavailable_reason))
                continue
            for name in ("validity", "tone"):
                coefficient = getattr(result, name)
                lines.append("| `%s` | %s | %s | %.4f | [%.3f, %.3f] | %s | %.4f | %s |" % (
                    model_id, metric, name, coefficient.coefficient,
                    coefficient.ci95[0], coefficient.ci95[1],
                    "%.5f" % coefficient.adjusted_p if coefficient.adjusted_p is not None else "n/a",
                    coefficient.sign_aligned_coefficient,
                    "yes" if coefficient.qualifying else "no"))
    lines += [
        "",
        "## G2 reversal (false-negative-eligible subset, item-clustered bootstrap)",
        "",
        "| model | metric | items | induction | recovery | recovery/induction | recovery 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id, analysis in verdict.models.items():
        for metric in verdict.eligible_metrics:
            result = analysis.real_g2.get(metric)
            if result is None or result.unavailable_reason:
                lines.append("| `%s` | %s | - | - | - | - | unavailable (`%s`) |" % (
                    model_id, metric, None if result is None else result.unavailable_reason))
                continue
            lines.append("| `%s` | %s | %d | %.4f | %.4f | %s | [%.3f, %.3f] |" % (
                model_id, metric, result.n_items, result.induction, result.recovery,
                "%.3f" % result.recovery_to_induction if result.recovery_to_induction is not None else "n/a",
                result.recovery_ci95[0], result.recovery_ci95[1]))
    lines += [
        "",
        "## G5 classifier and shuffled-label null",
        "",
        "| model | real full AUC | real baseline AUC | real gap | shuffled gap | shuffled null |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_id, analysis in verdict.models.items():
        real = analysis.real_g5
        null = analysis.shuffled
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            model_id,
            "%.3f" % real.full_auc if real is not None and real.full_auc is not None else "n/a",
            "%.3f" % real.baseline_auc if real is not None and real.baseline_auc is not None else "n/a",
            "%.3f" % real.auc_gap if real is not None and real.auc_gap is not None else "n/a",
            "%.3f" % null.g5.auc_gap if null.g5 is not None and null.g5.auc_gap is not None else "n/a",
            ("pass" if null.passed else "FAIL") + ("" if null.reason is None else " (`%s`)" % null.reason)))
    if verdict.style:
        lines += [
            "",
            "## G3 style smoke (five frozen items, sign-aligned, BH within the G3 family)",
            "",
            "| metric | style | effect | BH p | items | note |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for (metric, style_id) in sorted(verdict.style):
            item = verdict.style[(metric, style_id)]
            lines.append("| %s | `%s` | %s | %s | %d | %s |" % (
                metric, style_id,
                "%.4f" % item.effect if item.effect is not None else "n/a",
                "%.5f" % item.adjusted_p if item.adjusted_p is not None else "n/a",
                item.n_items, item.unavailable_reason or ""))
        if summary.style_meters:
            lines.append("")
            lines.append("Metrics reclassified as style meters: %s." % ", ".join(summary.style_meters))
    lines += [
        "",
        "## G4 boundary detail",
        "",
        "- eligible positives in the primary model: %s" % (", ".join(summary.g4.eligible_positive_metrics) or "none"),
        "- transfer metrics: %s" % (", ".join(summary.g4.transfer_metrics) or "none"),
        "- family-boundary metrics: %s" % (", ".join(summary.g4.boundary_metrics) or "none"),
        "",
        "A passed gate establishes a condition-selective, reversal-sensitive,",
        "style-resistant instability signature in unoptimized output channels --",
        "not experience, suffering, or moral status.",
        "",
    ]
    return "\n".join(lines)
