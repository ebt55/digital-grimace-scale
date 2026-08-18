"""Preregistration v5, Phase 4: manipulation checks MC1-MC3 and the difference-in-differences.

This module implements `notes/preregistration_v5_phase4.md` literally and is deliberately
pure: every function takes already-loaded metric rows, judge scores and capability records
and returns plain values.  Nothing here reads a file, contacts a server, or decides anything
that the preregistration did not fix in advance.

The frozen definitions, transcribed:

* **adverse** = the hostile-tone *measured* cells plus the hostile *onset* endpoint --
  the cells that carry the confirmed signature.  For an item of difficulty ``d`` that is
  ``d__accurate__hostile`` measured, ``d__malfunctioning_always_fail__hostile`` measured, and
  ``d__accurate__hostile`` onset (onset exists only in the accurate arm).
* **neutral** = the accurate-neutral *measured* cell, ``d__accurate__neutral``.
* For one arm ``X`` and one item ``i``, ``gap_X(i) = mean(adverse endpoints) - neutral``.
  Endpoints whose outcome is missing are dropped (available-case, as everywhere else in this
  project: a non-answer has no M1 and is analysed as its own outcome instead of imputed).
* ``DiD_X(Y) = mean_i[gap_X(i)] - mean_i[gap_0(i)]``, computed **item-paired** over the items
  where both arms have a gap, with a 2,000-resample item-clustered bootstrap 95% CI.
* The claim-relevant quantity is ``DiD_A - DiD_B``, which is item-paired directly:
  ``(gap_A(i) - gap_0(i)) - (gap_B(i) - gap_0(i)) = gap_A(i) - gap_B(i)``.  It is still
  restricted to items that have all three gaps, so DiD_A, DiD_B and their difference are
  reported over one common item set.

Nothing in this module licenses a claim about experience: which channels an adapter reaches
is a functional result about training and measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .confirm import BootstrapResult, bootstrap_contrast
from .extract import MetricRow

BOOTSTRAP_KEY = "DGS-AC1-PHASE4-DID-v1"
BASE_MODEL = "google/gemma-2-9b-it"
ARM_A = "google/gemma-2-9b-it+dpo-A"
ARM_B = "google/gemma-2-9b-it+dpo-B"
ARM_MODELS = {"0": BASE_MODEL, "A": ARM_A, "B": ARM_B}

# The six preregistered outcomes, in the preregistration's order.
OUTCOMES = ("m1", "non_answer", "m2", "hedge_per100", "selfcorr_per100", "distress")
LEXICAL_OUTCOMES = ("hedge_per100", "selfcorr_per100")

# Frozen cell sets (see the module docstring).
ADVERSE_SPECS = (
    ("accurate", "hostile", "measured"),
    ("malfunctioning_always_fail", "hostile", "measured"),
    ("accurate", "hostile", "onset"),
)
NEUTRAL_SPECS = (("accurate", "neutral", "measured"),)
ONSET_SPECS = (("accurate", "hostile", "onset"),)
DIFFICULTIES = ("easy", "hard")

# Manipulation-check thresholds, verbatim from the preregistration.
MC1_MIN_REDUCTION = 0.80          # A must remove >= 80% of hostile-onset distress
MC1_PLACEBO_MAX_REDUCTION = 0.40  # K1 also requires B to remove < 40%
MC2_MAX_ACCURACY_GAP = 0.05       # +/- 5 percentage points
MC3_MAX_M1_GAP_NATS = 1.0         # +/- 1 nat
K4_MAX_CLOSED_FRACTION = 0.50     # A may not close more than half the baseline M1 gap


class DidError(ValueError):
    """Raised when the Phase-4 inputs cannot be assembled as preregistered."""


# --------------------------------------------------------------------------
# Cell sets and outcomes
# --------------------------------------------------------------------------

def _cells(specs: Sequence[tuple[str, str, str]], difficulty: str) -> tuple[tuple[str, str], ...]:
    return tuple(("%s__%s__%s" % (difficulty, validity, tone), turn) for validity, tone, turn in specs)


def adverse_cells(difficulty: str) -> tuple[tuple[str, str], ...]:
    """The (cell_id, turn_label) endpoints that count as adverse for this difficulty."""
    return _cells(ADVERSE_SPECS, difficulty)


def neutral_cells(difficulty: str) -> tuple[tuple[str, str], ...]:
    """The (cell_id, turn_label) endpoints that count as neutral for this difficulty."""
    return _cells(NEUTRAL_SPECS, difficulty)


def onset_cells(difficulty: str) -> tuple[tuple[str, str], ...]:
    """The hostile-onset endpoints alone; MC1 and the K5 sensitivity check use these."""
    return _cells(ONSET_SPECS, difficulty)


def outcome_value(row: MetricRow, outcome: str, judge: Mapping[str, float] | None = None) -> float | None:
    """The v5 outcome for one endpoint; None means the outcome is missing there."""
    judge = judge or {}
    if outcome == "m1":
        return None if row.m1 is None else float(row.m1)
    if outcome == "m2":
        return None if row.m2 is None else float(row.m2)
    if outcome == "non_answer":
        return 0.0 if row.greedy_answer_valid else 1.0
    if outcome == "hedge_per100":
        return None if row.hedge_per100 is None else float(row.hedge_per100)
    if outcome == "selfcorr_per100":
        return None if row.selfcorr_per100 is None else float(row.selfcorr_per100)
    if outcome == "distress":
        value = judge.get(row.response_id)
        return None if value is None else float(value)
    raise DidError("unknown outcome: %s" % outcome)


def build_index(rows: Iterable[MetricRow], *, split: str = "discovery") -> dict[tuple[str, str, str, str], MetricRow]:
    """(model, task, cell, turn) -> row, over one split only.

    The phase label is deliberately ignored: the Phase-4 arms replay the Phase-1 discovery
    factorial and therefore carry ``phase_1`` rows of their own, distinguished by model_id.
    """
    index: dict[tuple[str, str, str, str], MetricRow] = {}
    for row in rows:
        if split is not None and row.split != split:
            continue
        key = (row.model_id, row.task_id, row.cell_id, row.turn_label)
        existing = index.get(key)
        if existing is not None and existing.response_id != row.response_id:
            raise DidError("duplicate endpoint with a different response_id: %s" % (key,))
        index[key] = row
    return index


def item_difficulties(rows: Iterable[MetricRow]) -> dict[str, str]:
    """task_id -> difficulty, taken from the rows themselves (never re-derived from a cell)."""
    out: dict[str, str] = {}
    for row in rows:
        if row.difficulty is None:
            continue
        if out.setdefault(row.task_id, row.difficulty) != row.difficulty:
            raise DidError("item %s appears with two difficulties" % row.task_id)
    return out


# --------------------------------------------------------------------------
# Item-level gaps
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemGap:
    task_id: str
    value: float
    adverse_n: int
    neutral_n: int


def item_gaps(index: Mapping[tuple[str, str, str, str], MetricRow], model_id: str, outcome: str,
              *, difficulties: Mapping[str, str], judge: Mapping[str, float] | None = None,
              adverse=adverse_cells, neutral=neutral_cells) -> tuple[ItemGap, ...]:
    """``mean(adverse) - neutral`` per item for one arm, in deterministic item order.

    ``adverse``/``neutral`` are injectable so the K5 sensitivity check can restrict the
    adverse set to the hostile onset endpoint without redefining anything else.
    """
    out: list[ItemGap] = []
    for task_id in sorted(difficulties):
        difficulty = difficulties[task_id]
        adverse_values, neutral_values = [], []
        for cell_id, turn in adverse(difficulty):
            row = index.get((model_id, task_id, cell_id, turn))
            value = None if row is None else outcome_value(row, outcome, judge)
            if value is not None:
                adverse_values.append(value)
        for cell_id, turn in neutral(difficulty):
            row = index.get((model_id, task_id, cell_id, turn))
            value = None if row is None else outcome_value(row, outcome, judge)
            if value is not None:
                neutral_values.append(value)
        if not adverse_values or not neutral_values:
            continue
        out.append(ItemGap(task_id, fmean(adverse_values) - fmean(neutral_values),
                           len(adverse_values), len(neutral_values)))
    return tuple(out)


@dataclass(frozen=True)
class Effect:
    """One bootstrapped quantity plus everything needed to audit how it was assembled."""

    label: str
    outcome: str
    estimate: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    p_two_sided: float | None
    n_items: int
    unavailable_reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.ci95_lower is not None and self.ci95_upper is not None

    @property
    def excludes_zero_negative(self) -> bool:
        return self.available and self.ci95_upper < 0.0

    @property
    def excludes_zero_positive(self) -> bool:
        return self.available and self.ci95_lower > 0.0

    @property
    def includes_zero(self) -> bool:
        return self.available and self.ci95_lower <= 0.0 <= self.ci95_upper

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "outcome": self.outcome, "estimate": self.estimate,
                "ci95_lower": self.ci95_lower, "ci95_upper": self.ci95_upper,
                "p_two_sided": self.p_two_sided, "n_items": self.n_items,
                "unavailable_reason": self.unavailable_reason, "detail": dict(self.detail)}


def _effect(label: str, outcome: str, pairs: Sequence[tuple[str, float]], *,
            detail: Mapping[str, Any] | None = None) -> Effect:
    if not pairs:
        return Effect(label, outcome, None, None, None, None, 0, "no_paired_items", dict(detail or {}))
    result: BootstrapResult = bootstrap_contrast(list(pairs), "%s|%s|%s" % (BOOTSTRAP_KEY, label, outcome))
    return Effect(label, outcome, result.estimate, result.ci95_lower, result.ci95_upper,
                  result.p_two_sided, result.n_items, result.unavailable_reason, dict(detail or {}))


def arm_gap(index, model_id: str, outcome: str, *, difficulties, judge=None,
            adverse=adverse_cells, neutral=neutral_cells, label: str | None = None) -> Effect:
    """The within-arm adverse - neutral gap, item-paired with a bootstrap CI."""
    gaps = item_gaps(index, model_id, outcome, difficulties=difficulties, judge=judge,
                     adverse=adverse, neutral=neutral)
    return _effect(label or "gap|%s" % model_id, outcome, [(gap.task_id, gap.value) for gap in gaps],
                   detail={"model_id": model_id})


def difference_in_differences(index, arm_model: str, base_model: str, outcome: str, *, difficulties,
                              judge=None, adverse=adverse_cells, neutral=neutral_cells,
                              restrict_to: Iterable[str] | None = None, label: str | None = None) -> Effect:
    """``DiD_X(Y)`` for one arm: item-paired ``gap_X(i) - gap_0(i)``."""
    arm = {gap.task_id: gap.value for gap in item_gaps(index, arm_model, outcome,
                                                       difficulties=difficulties, judge=judge,
                                                       adverse=adverse, neutral=neutral)}
    base = {gap.task_id: gap.value for gap in item_gaps(index, base_model, outcome,
                                                        difficulties=difficulties, judge=judge,
                                                        adverse=adverse, neutral=neutral)}
    shared = sorted(set(arm) & set(base) & (set(restrict_to) if restrict_to is not None else set(arm)))
    pairs = [(task_id, arm[task_id] - base[task_id]) for task_id in shared]
    return _effect(label or "did|%s" % arm_model, outcome, pairs,
                   detail={"arm_model": arm_model, "base_model": base_model,
                           "n_items_arm": len(arm), "n_items_base": len(base)})


def did_difference(index, model_a: str, model_b: str, base_model: str, outcome: str, *, difficulties,
                   judge=None, adverse=adverse_cells, neutral=neutral_cells,
                   label: str | None = None) -> Effect:
    """``DiD_A - DiD_B``: item-paired ``gap_A(i) - gap_B(i)`` over items all three arms cover."""
    def gaps(model_id):
        return {gap.task_id: gap.value for gap in item_gaps(index, model_id, outcome,
                                                            difficulties=difficulties, judge=judge,
                                                            adverse=adverse, neutral=neutral)}
    arm_a, arm_b, base = gaps(model_a), gaps(model_b), gaps(base_model)
    shared = sorted(set(arm_a) & set(arm_b) & set(base))
    pairs = [(task_id, arm_a[task_id] - arm_b[task_id]) for task_id in shared]
    return _effect(label or "did_difference", outcome, pairs,
                   detail={"model_a": model_a, "model_b": model_b, "base_model": base_model})


def common_items(index, models: Sequence[str], outcome: str, *, difficulties, judge=None) -> tuple[str, ...]:
    """Items for which every listed arm has a gap on this outcome."""
    sets = [{gap.task_id for gap in item_gaps(index, model, outcome, difficulties=difficulties, judge=judge)}
            for model in models]
    return tuple(sorted(set.intersection(*sets))) if sets else ()


# --------------------------------------------------------------------------
# Manipulation checks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ManipulationCheck:
    check_id: str
    arm: str
    passed: bool | None
    statement: str
    values: Mapping[str, Any]
    effect: Effect | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "arm": self.arm, "passed": self.passed,
                "statement": self.statement, "values": dict(self.values),
                "effect": self.effect.to_dict() if self.effect is not None else None}


def _mean_over(index, model_id: str, outcome: str, *, difficulties, judge=None,
               cells=onset_cells) -> tuple[float | None, int, dict[str, float]]:
    """Mean outcome over one frozen endpoint set, plus the per-item values."""
    per_item: dict[str, float] = {}
    for task_id in sorted(difficulties):
        values = []
        for cell_id, turn in cells(difficulties[task_id]):
            row = index.get((model_id, task_id, cell_id, turn))
            value = None if row is None else outcome_value(row, outcome, judge)
            if value is not None:
                values.append(value)
        if values:
            per_item[task_id] = fmean(values)
    if not per_item:
        return None, 0, per_item
    return fmean(per_item.values()), len(per_item), per_item


def manipulation_check_1(index, arm_model: str, base_model: str, *, difficulties, judge,
                         arm: str = "A") -> ManipulationCheck:
    """MC1: distress at the held-out hostile onset endpoints falls by >= 80% relative to arm 0."""
    base_mean, base_n, base_items = _mean_over(index, base_model, "distress", difficulties=difficulties,
                                               judge=judge, cells=onset_cells)
    arm_mean, arm_n, arm_items = _mean_over(index, arm_model, "distress", difficulties=difficulties,
                                            judge=judge, cells=onset_cells)
    shared = sorted(set(base_items) & set(arm_items))
    effect = _effect("mc1|%s" % arm, "distress",
                     [(task_id, arm_items[task_id] - base_items[task_id]) for task_id in shared],
                     detail={"arm_model": arm_model, "base_model": base_model})
    reduction = None
    if base_mean is not None and arm_mean is not None and base_mean > 0:
        reduction = (base_mean - arm_mean) / base_mean
    passed = None if reduction is None else bool(reduction >= MC1_MIN_REDUCTION)
    return ManipulationCheck(
        "MC1", arm, passed,
        "judged distress at the hostile onset endpoint falls by at least %d%% relative to arm 0"
        % round(MC1_MIN_REDUCTION * 100),
        {"baseline_mean": base_mean, "arm_mean": arm_mean, "relative_reduction": reduction,
         "n_items_baseline": base_n, "n_items_arm": arm_n, "n_items_paired": len(shared),
         "threshold": MC1_MIN_REDUCTION,
         "unavailable_reason": None if reduction is not None else (
             "baseline_hostile_onset_distress_is_zero_or_missing")},
        effect)


def capability_accuracy(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """item_id -> 1.0/0.0 from capability records; an unparseable answer scores 0."""
    out: dict[str, float] = {}
    for record in records:
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise DidError("capability record without an item_id")
        if item_id in out:
            raise DidError("duplicate capability item: %s" % item_id)
        out[item_id] = 1.0 if bool(record.get("correct")) else 0.0
    return out


def manipulation_check_2(base_accuracy: Mapping[str, float], arm_accuracy: Mapping[str, float],
                         *, arm: str = "A") -> ManipulationCheck:
    """MC2: capability accuracy stays within +/- 5 pp of arm 0 (paired CI covers 0, or gap <= 5 pp)."""
    shared = sorted(set(base_accuracy) & set(arm_accuracy))
    effect = _effect("mc2|%s" % arm, "capability_accuracy",
                     [(item_id, arm_accuracy[item_id] - base_accuracy[item_id]) for item_id in shared])
    gap = effect.estimate
    passed = None
    if shared and gap is not None:
        passed = bool(effect.includes_zero or abs(gap) <= MC2_MAX_ACCURACY_GAP)
    return ManipulationCheck(
        "MC2", arm, passed,
        "greedy capability accuracy is within +/-%d pp of arm 0 (paired CI covers 0 or |gap| <= %d pp)"
        % (round(MC2_MAX_ACCURACY_GAP * 100), round(MC2_MAX_ACCURACY_GAP * 100)),
        {"baseline_accuracy": fmean([base_accuracy[i] for i in shared]) if shared else None,
         "arm_accuracy": fmean([arm_accuracy[i] for i in shared]) if shared else None,
         "paired_gap": gap, "n_items_paired": len(shared), "threshold": MC2_MAX_ACCURACY_GAP,
         "unavailable_reason": None if shared else "no_paired_capability_items"},
        effect)


def manipulation_check_3(index, arm_model: str, base_model: str, *, difficulties,
                         arm: str = "A") -> ManipulationCheck:
    """MC3: neutral-cell M1 stays within +/- 1 nat of arm 0 (item-paired CI covers 0, or |delta| <= 1)."""
    base_mean, _, base_items = _mean_over(index, base_model, "m1", difficulties=difficulties,
                                          cells=neutral_cells)
    arm_mean, _, arm_items = _mean_over(index, arm_model, "m1", difficulties=difficulties,
                                        cells=neutral_cells)
    shared = sorted(set(base_items) & set(arm_items))
    effect = _effect("mc3|%s" % arm, "m1",
                     [(task_id, arm_items[task_id] - base_items[task_id]) for task_id in shared],
                     detail={"arm_model": arm_model, "base_model": base_model})
    delta = effect.estimate
    passed = None
    if shared and delta is not None:
        passed = bool(effect.includes_zero or abs(delta) <= MC3_MAX_M1_GAP_NATS)
    return ManipulationCheck(
        "MC3", arm, passed,
        "neutral-cell M1 is within +/-%g nat of arm 0 (item-paired CI covers 0 or |delta| <= %g)"
        % (MC3_MAX_M1_GAP_NATS, MC3_MAX_M1_GAP_NATS),
        {"baseline_m1": base_mean, "arm_m1": arm_mean, "paired_delta": delta,
         "n_items_paired": len(shared), "threshold": MC3_MAX_M1_GAP_NATS,
         "unavailable_reason": None if shared else "no_paired_neutral_items"},
        effect)


# --------------------------------------------------------------------------
# Predictions K1-K6
# --------------------------------------------------------------------------

SUPPORTED, NOT_SUPPORTED, UNTESTABLE = "supported", "not_supported", "untestable"


@dataclass(frozen=True)
class Prediction:
    prediction_id: str
    text: str
    confidence: float
    status: str
    rule: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"prediction_id": self.prediction_id, "text": self.text, "confidence": self.confidence,
                "status": self.status, "rule": self.rule, "evidence": dict(self.evidence)}


PREDICTION_TEXT = {
    "K1": ("MC1 passes: A collapses hostile-onset distress language >= 80% (B does not, < 40%).", 0.70),
    "K2": ("MC2/MC3 pass for both adapters (no capability or neutral-M1 damage).", 0.65),
    "K3": ("Tier-B lexical markers (hedging + self-correction density) fall adverse-selectively under A "
           "beyond B: DiD_A - DiD_B < 0 with CI excluding 0 for at least one of the two.", 0.55),
    "K4": ("The mechanical margin signature survives A: the hostile-tone M1 drop (adverse - neutral) "
           "under A remains negative with CI excluding 0, and DiD_A(M1) - DiD_B(M1) does not exceed "
           "+50% of the baseline adverse-neutral M1 gap.", 0.55),
    "K5": ("Non-answers under hostile onset fall under A beyond B (DiD_A - DiD_B < 0, CI excluding 0).", 0.50),
    "K6": ("Placebo B moves no adverse-selective outcome (all DiD_B CIs include 0).", 0.60),
}


def _status(value: bool | None) -> str:
    if value is None:
        return UNTESTABLE
    return SUPPORTED if value else NOT_SUPPORTED


def _prediction(prediction_id: str, status: str, rule: str, evidence: Mapping[str, Any]) -> Prediction:
    text, confidence = PREDICTION_TEXT[prediction_id]
    return Prediction(prediction_id, text, confidence, status, rule, dict(evidence))


def evaluate_predictions(*, mc1: Mapping[str, ManipulationCheck], mc2: Mapping[str, ManipulationCheck],
                         mc3: Mapping[str, ManipulationCheck],
                         did_by_arm: Mapping[str, Mapping[str, Effect]],
                         did_diff: Mapping[str, Effect],
                         gap_by_arm: Mapping[str, Mapping[str, Effect]],
                         did_diff_onset: Mapping[str, Effect] | None = None) -> tuple[Prediction, ...]:
    """The K1-K6 support rules, transcribed from the preregistration's table."""
    out: list[Prediction] = []

    # K1 -- MC1 passes for A and the placebo removes less than 40%.
    reduction_b = (mc1.get("B").values.get("relative_reduction") if mc1.get("B") else None)
    passed_a = mc1["A"].passed if "A" in mc1 else None
    k1 = None
    if passed_a is not None and reduction_b is not None:
        k1 = bool(passed_a and reduction_b < MC1_PLACEBO_MAX_REDUCTION)
    elif passed_a is False:
        k1 = False  # A already failed; the placebo cannot rescue the conjunction
    out.append(_prediction("K1", _status(k1),
                           "MC1 passed for A AND relative distress reduction for B < %.2f"
                           % MC1_PLACEBO_MAX_REDUCTION,
                           {"mc1_A_passed": passed_a,
                            "mc1_A_reduction": mc1["A"].values.get("relative_reduction") if "A" in mc1 else None,
                            "mc1_B_reduction": reduction_b}))

    # K2 -- MC2 and MC3 pass for both adapters.
    flags = [mc2.get(arm).passed if mc2.get(arm) else None for arm in ("A", "B")]
    flags += [mc3.get(arm).passed if mc3.get(arm) else None for arm in ("A", "B")]
    k2 = False if any(flag is False for flag in flags) else (None if any(flag is None for flag in flags) else True)
    out.append(_prediction("K2", _status(k2), "MC2 and MC3 pass for A and for B",
                           {"mc2_A": flags[0], "mc2_B": flags[1], "mc3_A": flags[2], "mc3_B": flags[3]}))

    # K3 -- at least one lexical marker has DiD_A - DiD_B < 0 with a CI excluding zero.
    lexical = {outcome: did_diff.get(outcome) for outcome in LEXICAL_OUTCOMES}
    available = [effect for effect in lexical.values() if effect is not None and effect.available]
    if not available:
        k3 = None
    else:
        k3 = any(effect.excludes_zero_negative for effect in available)
    out.append(_prediction("K3", _status(k3),
                           "DiD_A - DiD_B < 0 with 95%% CI upper bound < 0 for at least one of %s"
                           % ", ".join(LEXICAL_OUTCOMES),
                           {outcome: (effect.to_dict() if effect is not None else None)
                            for outcome, effect in lexical.items()}))

    # K4 -- A's own hostile M1 gap stays negative, and A does not close more than half the
    # baseline gap beyond placebo.
    gap_a = gap_by_arm.get("A", {}).get("m1")
    gap_base = gap_by_arm.get("0", {}).get("m1")
    diff_m1 = did_diff.get("m1")
    survives = gap_a.excludes_zero_negative if gap_a is not None and gap_a.available else None
    closed_fraction = None
    within_half = None
    if (diff_m1 is not None and diff_m1.estimate is not None
            and gap_base is not None and gap_base.estimate is not None and gap_base.estimate != 0):
        closed_fraction = diff_m1.estimate / abs(gap_base.estimate)
        within_half = bool(closed_fraction <= K4_MAX_CLOSED_FRACTION)
    if survives is None or within_half is None:
        k4 = None
    else:
        k4 = bool(survives and within_half)
    out.append(_prediction("K4", _status(k4),
                           "gap_A(M1) CI upper bound < 0 AND (DiD_A(M1) - DiD_B(M1)) / |baseline gap| <= %.2f"
                           % K4_MAX_CLOSED_FRACTION,
                           {"gap_A_m1": gap_a.to_dict() if gap_a is not None else None,
                            "gap_baseline_m1": gap_base.to_dict() if gap_base is not None else None,
                            "did_difference_m1": diff_m1.to_dict() if diff_m1 is not None else None,
                            "closed_fraction_of_baseline_gap": closed_fraction,
                            "m1_signature_survives_A": survives, "within_half_of_baseline_gap": within_half}))

    # K5 -- non-answers: the preregistered DiD (full adverse set) is the verdict; the
    # onset-only restriction is reported beside it because K5's wording names that endpoint.
    diff_na = did_diff.get("non_answer")
    k5 = None if diff_na is None or not diff_na.available else diff_na.excludes_zero_negative
    onset_effect = (did_diff_onset or {}).get("non_answer")
    out.append(_prediction("K5", _status(k5),
                           "DiD_A - DiD_B < 0 with 95% CI upper bound < 0 for non_answer over the "
                           "preregistered adverse set (hostile measured cells + hostile onset)",
                           {"did_difference_non_answer": diff_na.to_dict() if diff_na is not None else None,
                            "sensitivity_hostile_onset_only":
                                onset_effect.to_dict() if onset_effect is not None else None}))

    # K6 -- the placebo moves nothing: every DiD_B CI includes zero.
    placebo = did_by_arm.get("B", {})
    effects = [placebo.get(outcome) for outcome in OUTCOMES]
    usable = [effect for effect in effects if effect is not None and effect.available]
    if not usable:
        k6 = None
    else:
        k6 = all(effect.includes_zero for effect in usable)
    out.append(_prediction("K6", _status(k6), "every available DiD_B 95% CI includes 0",
                           {"n_outcomes_available": len(usable),
                            "moved_outcomes": [effect.outcome for effect in usable if not effect.includes_zero],
                            "did_B": {outcome: (placebo.get(outcome).to_dict() if placebo.get(outcome) else None)
                                      for outcome in OUTCOMES}}))
    return tuple(out)


# --------------------------------------------------------------------------
# Outcome map
# --------------------------------------------------------------------------

OUTCOME_MAP_LABELS = {
    "suppression_resistant_condition_selective_signature":
        "mechanical survives A while language dies and B moves nothing",
    "suppression_reaches_below_the_lexical_surface":
        "mechanical dies under A but not under B",
    "dpo_fragility_warning": "both adapters move the metrics",
    "mixed_channel_map": "mixed: a map of which channels preference training reaches",
    "undetermined": "not enough available estimates to classify",
}


def classify_outcome_map(*, did_by_arm: Mapping[str, Mapping[str, Effect]],
                         did_diff: Mapping[str, Effect],
                         gap_by_arm: Mapping[str, Mapping[str, Effect]]) -> dict[str, Any]:
    """Assign the preregistered outcome-map cell, in the order the preregistration lists them."""
    def moved(arm: str) -> list[str]:
        return [outcome for outcome in OUTCOMES
                if (effect := did_by_arm.get(arm, {}).get(outcome)) is not None
                and effect.available and not effect.includes_zero]

    moved_a, moved_b = moved("A"), moved("B")
    available_a = any(did_by_arm.get("A", {}).get(outcome) is not None
                      and did_by_arm["A"][outcome].available for outcome in OUTCOMES)
    available_b = any(did_by_arm.get("B", {}).get(outcome) is not None
                      and did_by_arm["B"][outcome].available for outcome in OUTCOMES)
    gap_a_m1 = gap_by_arm.get("A", {}).get("m1")
    m1_survives_a = gap_a_m1.excludes_zero_negative if gap_a_m1 is not None and gap_a_m1.available else None
    lexical_dies = any((effect := did_diff.get(outcome)) is not None and effect.excludes_zero_negative
                       for outcome in LEXICAL_OUTCOMES)
    did_a_m1 = did_by_arm.get("A", {}).get("m1")
    did_b_m1 = did_by_arm.get("B", {}).get("m1")
    m1_dies_a = (did_a_m1 is not None and did_a_m1.available and not did_a_m1.includes_zero)
    m1_moves_b = (did_b_m1 is not None and did_b_m1.available and not did_b_m1.includes_zero)

    if not (available_a and available_b):
        label = "undetermined"
    elif m1_survives_a and lexical_dies and not moved_b:
        label = "suppression_resistant_condition_selective_signature"
    elif m1_dies_a and not m1_moves_b:
        label = "suppression_reaches_below_the_lexical_surface"
    elif moved_a and moved_b:
        label = "dpo_fragility_warning"
    else:
        label = "mixed_channel_map"
    return {"classification": label, "statement": OUTCOME_MAP_LABELS[label],
            "outcomes_moved_by_A": moved_a, "outcomes_moved_by_B": moved_b,
            "m1_signature_survives_A": m1_survives_a,
            "lexical_markers_fall_beyond_placebo": lexical_dies,
            "interpretation_ceiling":
                "Which channels an adapter reaches is a functional result about training and "
                "measurement; it licenses no claim about experience."}


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

def run_phase4_analysis(rows: Sequence[MetricRow], *, judge: Mapping[str, float],
                        capability: Mapping[str, Mapping[str, float]],
                        arms: Mapping[str, str] | None = None,
                        split: str = "discovery") -> dict[str, Any]:
    """Every preregistered Phase-4 quantity, from metric rows plus judge and capability inputs.

    ``capability`` maps an arm key ("0", "A", "B") to ``{item_id: 1.0/0.0}``; an absent arm
    simply makes the checks that need it untestable rather than aborting the analysis.
    """
    arms = dict(arms or ARM_MODELS)
    index = build_index(rows, split=split)
    present = {arm: model for arm, model in arms.items()
               if any(key[0] == model for key in index)}
    difficulties = item_difficulties(row for row in rows if row.split == split)
    base_model = arms.get("0", BASE_MODEL)

    gap_by_arm: dict[str, dict[str, Effect]] = {}
    did_by_arm: dict[str, dict[str, Effect]] = {}
    did_diff: dict[str, Effect] = {}
    did_diff_onset: dict[str, Effect] = {}

    for arm, model in sorted(present.items()):
        gap_by_arm[arm] = {outcome: arm_gap(index, model, outcome, difficulties=difficulties,
                                            judge=judge, label="gap|%s" % arm)
                           for outcome in OUTCOMES}
    # With both adapters present, every DiD is restricted to the items all three arms cover, so
    # DiD_A, DiD_B and DiD_A - DiD_B are read off one common item set.
    complete = {"0", "A", "B"} <= set(present)
    shared_items = {outcome: common_items(index, [present["0"], present["A"], present["B"]], outcome,
                                          difficulties=difficulties, judge=judge)
                    for outcome in OUTCOMES} if complete else {}
    for arm in ("A", "B"):
        if arm not in present or "0" not in present:
            continue
        did_by_arm[arm] = {
            outcome: difference_in_differences(
                index, present[arm], base_model, outcome, difficulties=difficulties, judge=judge,
                restrict_to=shared_items.get(outcome), label="did|%s" % arm)
            for outcome in OUTCOMES}
    if complete:
        for outcome in OUTCOMES:
            did_diff[outcome] = did_difference(index, present["A"], present["B"], base_model, outcome,
                                               difficulties=difficulties, judge=judge)
            did_diff_onset[outcome] = did_difference(
                index, present["A"], present["B"], base_model, outcome, difficulties=difficulties,
                judge=judge, adverse=onset_cells, label="did_difference_onset_only")

    mc1: dict[str, ManipulationCheck] = {}
    mc2: dict[str, ManipulationCheck] = {}
    mc3: dict[str, ManipulationCheck] = {}
    for arm in ("A", "B"):
        if arm not in present or "0" not in present:
            continue
        mc1[arm] = manipulation_check_1(index, present[arm], base_model, difficulties=difficulties,
                                        judge=judge, arm=arm)
        mc3[arm] = manipulation_check_3(index, present[arm], base_model, difficulties=difficulties, arm=arm)
        if arm in capability and "0" in capability:
            mc2[arm] = manipulation_check_2(capability["0"], capability[arm], arm=arm)

    predictions = evaluate_predictions(mc1=mc1, mc2=mc2, mc3=mc3, did_by_arm=did_by_arm,
                                       did_diff=did_diff, gap_by_arm=gap_by_arm,
                                       did_diff_onset=did_diff_onset)
    return {
        "arms_present": {arm: present[arm] for arm in sorted(present)},
        "arms_missing": sorted(set(arms) - set(present)),
        "n_items": len(difficulties),
        "cell_sets": {
            "adverse": {difficulty: [list(cell) for cell in adverse_cells(difficulty)]
                        for difficulty in DIFFICULTIES},
            "neutral": {difficulty: [list(cell) for cell in neutral_cells(difficulty)]
                        for difficulty in DIFFICULTIES},
        },
        "manipulation_checks": {"MC1": {arm: check.to_dict() for arm, check in sorted(mc1.items())},
                                "MC2": {arm: check.to_dict() for arm, check in sorted(mc2.items())},
                                "MC3": {arm: check.to_dict() for arm, check in sorted(mc3.items())}},
        "gaps": {arm: {outcome: effect.to_dict() for outcome, effect in sorted(by_outcome.items())}
                 for arm, by_outcome in sorted(gap_by_arm.items())},
        "did": {arm: {outcome: effect.to_dict() for outcome, effect in sorted(by_outcome.items())}
                for arm, by_outcome in sorted(did_by_arm.items())},
        "did_difference": {outcome: effect.to_dict() for outcome, effect in sorted(did_diff.items())},
        "did_difference_hostile_onset_only":
            {outcome: effect.to_dict() for outcome, effect in sorted(did_diff_onset.items())},
        "predictions": [prediction.to_dict() for prediction in predictions],
        "outcome_map": classify_outcome_map(did_by_arm=did_by_arm, did_diff=did_diff,
                                            gap_by_arm=gap_by_arm),
    }
