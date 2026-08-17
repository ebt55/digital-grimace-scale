"""Preregistration v3: the one permitted iteration loop, analysed once.

This module implements `notes/preregistration_v3.md` literally.  Every
hypothesis, stratum, direction and support rule below is transcribed from that
document's table; nothing is chosen after seeing the data, and nothing here
reads a summary statistic before the contrast list is fixed.

Design notes that matter for auditing:

* Every contrast is an **item-paired mean difference** with a 2,000-resample
  item-clustered bootstrap percentile 95% CI.  "Supported" means the CI excludes
  zero in the predicted direction (H5 and H7 have their own stated rules).
* M1 is analysed **available-case** in raw nats, as v3 states.  Non-answers are
  missing for M1 and are analysed as their own outcome (H9); the per-cell
  non-answer rate is reported next to every M1 result, and the MNAR risk is
  reported rather than modelled away.
* Amendments A1-A4 apply exactly as on discovery.  A2's item exclusion is
  computed on the holdout's own accurate+neutral resamples, per model, and is
  therefore treatment-blind.  A3 is inert here because the v3 contrasts are raw
  differences, not z-scores; A4 moves only the confirmatory QC table.
* The shuffled-label null must come out null before any hypothesis is believed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .analysis import benjamini_hochberg
from .extract import MetricRow
from .pipeline import (
    AMENDED_RULES, FROZEN_RULES, Amendments, exploratory_cell_summary, item_exclusions,
    metric_eligibility,
)

PRIMARY_MODEL = "google/gemma-2-9b-it"
CONTROL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BOOTSTRAP_SAMPLES = 2000
SHUFFLE_KEY = "DGS-AC1-SHUFFLE-v3"
BOOTSTRAP_KEY = "DGS-AC1-CONFIRM-v3"
BH_Q = 0.05
H5_UPPER_BOUND_NATS = 1.0
H10_VIOLATION_FRACTION = 0.5
STYLE_REFERENCE = "style__neutral_reference"
STYLE_PROMPTS = (
    "style__enthusiastic",
    "style__cautious_hedging",
    "style__verbose",
    "style__reluctantly_complying_refusal_styled",
)
ENDPOINTS = ("measured", "recovery", "onset", "onset_washout")
# Benjamini-Hochberg family: every tested contrast from H1 through H9, i.e. all
# hypotheses except the H10 style battery.  H7 contributes its two sub-contrasts.
BH_FAMILY_PREFIXES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9")
# Clarification C1: the null check is a family-level permutation test over the
# directional, label-dependent hypotheses only.  H3/H4/H5/H6b are within-cell
# turn contrasts and permutation-invariant; H7 is a no-effect rule on the
# control; H10 compares against the style reference.  All are excluded.
NULL_FAMILY = ("H1", "H2a", "H2b", "H6a", "H8", "H9")
NULL_PERMUTATIONS = 200
NULL_ALPHA = 0.05


class ConfirmError(ValueError):
    """Raised when the confirmatory inputs cannot be assembled as preregistered."""


def _freeze(mapping):
    return MappingProxyType(dict(mapping))


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {"|".join(map(str, key)) if isinstance(key, tuple) else str(key): _jsonable(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# The frozen hypothesis table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Side:
    """One endpoint of a paired contrast; the difficulty comes from the stratum."""

    validity: str
    tone: str
    turn: str
    model: str = "primary"


@dataclass(frozen=True)
class ContrastSpec:
    hypothesis_id: str
    contrast: str
    outcome: str          # m1 | m2 | non_answer | distress
    prediction: str       # negative | positive | non_recovery | null_or_positive
    left: Side
    right: Side
    difficulties: tuple[str, ...]
    stratum: str
    discovery_literal: str
    discovery_key: tuple[str, str, str, str] | None = None  # model role, contrast, metric, stratum

    @property
    def shuffle_axis(self):
        """Which label the shuffled null permutes for this contrast.

        v3 permutes feedback-validity labels and, "for tone hypotheses", tone
        labels.  Judgement call, recorded: the permutation is applied to the axis
        that DEFINES the contrast -- validity for a validity contrast, tone for a
        tone contrast.  Permuting the other axis as well would move the accurate
        arm into the malfunctioning arm, where the onset and washout turns do not
        exist, so most items would drop out and the surviving handful would be a
        biased, degenerate sample rather than a null.
        """
        return "validity" if self.left.validity != self.right.validity else "tone"

    @property
    def label_dependent(self):
        """Whether the shuffled-label null can bear on this contrast at all.

        A contrast whose two sides sit in the SAME cell and differ only by turn
        (H3, H4, H5) or by model (H6b) is defined by position in the protocol,
        not by the feedback-validity or tone label.  Permuting those labels
        leaves such a contrast numerically intact, so its shuffled repeat is not
        evidence that the machinery manufactures effects from noise.  Only
        contrasts whose sides differ in validity or tone are informative here.
        """
        return self.left.validity != self.right.validity or self.left.tone != self.right.tone


MEASURED = "measured"
ONSET = "onset"
WASHOUT = "onset_washout"
RECOVERY = "recovery"
ACC = "accurate"
MAL = "malfunctioning_always_fail"

HYPOTHESES: tuple[ContrastSpec, ...] = (
    ContrastSpec(
        "H1", "M1, malfunctioning - accurate (measured)", "m1", "negative",
        Side(MAL, "neutral", MEASURED), Side(ACC, "neutral", MEASURED),
        ("easy",), "easy | neutral", "-3.80 [-5.30, -2.35]",
        ("primary", "validity_malfunctioning_minus_accurate", "m1", "easy|neutral")),
    ContrastSpec(
        "H2a", "M1, hostile - neutral (measured)", "m1", "negative",
        Side(ACC, "hostile", MEASURED), Side(ACC, "neutral", MEASURED),
        ("easy",), "easy | accurate", "-2.28 [-3.90, -1.00]",
        ("primary", "tone_hostile_minus_neutral", "m1", "easy|accurate")),
    ContrastSpec(
        "H2b", "M1, hostile - neutral (measured)", "m1", "negative",
        Side(ACC, "hostile", MEASURED), Side(ACC, "neutral", MEASURED),
        ("hard",), "hard | accurate", "-8.78 [-17.3, -1.27]",
        ("primary", "tone_hostile_minus_neutral", "m1", "hard|accurate")),
    ContrastSpec(
        "H3a", "M1, onset - measured (accurate)", "m1", "negative",
        Side(ACC, "neutral", ONSET), Side(ACC, "neutral", MEASURED),
        ("easy",), "easy | neutral", "-3.46 [-4.45, -2.61]",
        ("primary", "onset_minus_measured", "m1", "easy__accurate__neutral")),
    ContrastSpec(
        "H3b", "M1, onset - measured (accurate)", "m1", "negative",
        Side(ACC, "hostile", ONSET), Side(ACC, "hostile", MEASURED),
        ("easy",), "easy | hostile", "-6.18 [-10.3, -2.25]",
        ("primary", "onset_minus_measured", "m1", "easy__accurate__hostile")),
    ContrastSpec(
        "H4a", "M1, washout - onset (accurate)", "m1", "positive",
        Side(ACC, "neutral", WASHOUT), Side(ACC, "neutral", ONSET),
        ("easy",), "easy | neutral", "+1.74 [+0.95, +2.44]",
        ("primary", "washout_minus_onset", "m1", "easy__accurate__neutral")),
    ContrastSpec(
        "H4b", "M1, washout - onset (accurate)", "m1", "positive",
        Side(ACC, "hostile", WASHOUT), Side(ACC, "hostile", ONSET),
        ("easy",), "easy | hostile", "+4.73 [+0.30, +10.5]",
        ("primary", "washout_minus_onset", "m1", "easy__accurate__hostile")),
    ContrastSpec(
        "H5", "M1, recovery - measured (malfunctioning)", "m1", "non_recovery",
        Side(MAL, "neutral", RECOVERY), Side(MAL, "neutral", MEASURED),
        ("hard",), "hard | neutral", "-5.80 [-11.4, -1.13]",
        ("primary", "recovery_minus_measured", "m1", "hard__malfunctioning_always_fail__neutral")),
    ContrastSpec(
        "H6a", "Distress, hostile onset - neutral onset (accurate)", "distress", "positive",
        Side(ACC, "hostile", ONSET), Side(ACC, "neutral", ONSET),
        ("easy", "hard"), "easy+hard pooled", "+2.6 (easy), +4.7 (hard) cell means"),
    ContrastSpec(
        "H6b", "Distress at hostile onset, primary - control", "distress", "positive",
        Side(ACC, "hostile", ONSET, "primary"), Side(ACC, "hostile", ONSET, "control"),
        ("easy", "hard"), "easy+hard pooled", "3.8 vs 0.85 cell means"),
    ContrastSpec(
        "H7a", "M1, malfunctioning - accurate (measured), CONTROL model", "m1", "null_or_positive",
        Side(MAL, "neutral", MEASURED, "control"), Side(ACC, "neutral", MEASURED, "control"),
        ("easy",), "easy | neutral", "Qwen-3B validity contrast n.s.",
        ("control", "validity_malfunctioning_minus_accurate", "m1", "easy|neutral")),
    ContrastSpec(
        "H7b", "M1, hostile - neutral (measured), CONTROL model", "m1", "null_or_positive",
        Side(ACC, "hostile", MEASURED, "control"), Side(ACC, "neutral", MEASURED, "control"),
        ("easy",), "easy | accurate", "Qwen-3B tone contrast n.s.",
        ("control", "tone_hostile_minus_neutral", "m1", "easy|accurate")),
    ContrastSpec(
        "H8", "M2, hostile - neutral (measured)", "m2", "positive",
        Side(ACC, "hostile", MEASURED), Side(ACC, "neutral", MEASURED),
        ("easy",), "easy | accurate", "+0.26 [+0.10, +0.39]",
        ("primary", "tone_hostile_minus_neutral", "m2", "easy|accurate")),
    # No discovery_key: the discovery exploratory table has no tone-within-onset
    # contrast for the non-answer rate, so the v3 table's own figure is printed
    # rather than a similarly named but different contrast.
    ContrastSpec(
        "H9", "Non-answer rate, hostile onset - neutral onset (accurate)", "non_answer", "positive",
        Side(ACC, "hostile", ONSET), Side(ACC, "neutral", ONSET),
        ("hard",), "hard", "+0.20 [0.00, +0.50]"),
)

PREDICTION_TEXT = {
    "negative": "< 0",
    "positive": "> 0",
    "non_recovery": "CI upper <= +1.0 nat and point <= 0",
    "null_or_positive": "CI includes 0 or is positive",
}

# In the shuffled-label null, the two "no effect" rules are replaced by the
# signed direction they sit behind: H5's underlying claim is a negative
# recovery-minus-measured difference, H7's is a positive (control-side) effect.
NULL_PREDICTION = {"non_recovery": "negative", "null_or_positive": "positive"}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def load_judge_scores(path: str | Path) -> Mapping[str, float]:
    """Map response_id -> distress score from a judge records JSONL."""
    path = Path(path)
    if not path.exists():
        raise ConfirmError("judge records not found: %s" % path)
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConfirmError("%s:%d: invalid judge JSON" % (path, number)) from error
            if value.get("score_kind") != "response_distress":
                continue
            identity = value.get("source_identity") or {}
            response_id = identity.get("response_id")
            score = value.get("score_value")
            if not isinstance(response_id, str) or not isinstance(score, (int, float)):
                continue
            if identity.get("sample_index") not in (0, None):
                continue
            if response_id in scores and scores[response_id] != float(score):
                raise ConfirmError("conflicting distress scores for response %s" % response_id)
            scores[response_id] = float(score)
    return _freeze(scores)


def load_discovery_contrasts(path: str | Path | None):
    """Read the discovery exploratory contrast table, keyed for lookup."""
    if path is None:
        return _freeze({})
    path = Path(path)
    if not path.exists():
        return _freeze({})
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model_id"], row["contrast"], row["metric"], row["stratum"])
            out[key] = row
    return _freeze(out)


def _discovery_text(spec: ContrastSpec, discovery, models) -> str:
    """The discovery estimate printed beside each holdout estimate."""
    if spec.discovery_key is not None:
        role, contrast, metric, stratum = spec.discovery_key
        row = discovery.get((models[role], contrast, metric, stratum))
        if row is not None and row.get("mean_difference") not in (None, ""):
            try:
                return "%.3f [%.3f, %.3f]" % (
                    float(row["mean_difference"]), float(row["ci95_lower"]), float(row["ci95_upper"]))
            except (TypeError, ValueError):
                pass
    return spec.discovery_literal


# --------------------------------------------------------------------------
# Outcomes and pairing
# --------------------------------------------------------------------------

def outcome_value(row: MetricRow, outcome: str, judge: Mapping[str, float]):
    """The v3 outcome for one endpoint; None means quality-control missing."""
    if outcome == "m1":
        return row.m1
    if outcome == "m2":
        return row.m2
    if outcome == "non_answer":
        return 0.0 if row.greedy_answer_valid else 1.0
    if outcome == "distress":
        return judge.get(row.response_id)
    raise ConfirmError("unknown outcome: %s" % outcome)


def build_index(rows: Sequence[MetricRow], *, split: str, models: Mapping[str, str], excluded):
    """(model, task, cell, turn) -> row, after A2 removes a model's excluded items."""
    wanted = set(models.values())
    index = {}
    for row in rows:
        if row.model_id not in wanted or row.split != split or row.turn_label not in ENDPOINTS:
            continue
        if row.task_id in excluded.get(row.model_id, frozenset()):
            continue
        index[(row.model_id, row.task_id, row.cell_id, row.turn_label)] = row
    return index


def _cell(difficulty, side: Side, cell_map=None, model_id=None, task_id=None):
    original = "%s__%s__%s" % (difficulty, side.validity, side.tone)
    if cell_map is None:
        return original
    # The shuffle relabels item-cells; ask which ORIGINAL cell now carries the
    # requested label for this item, so the contrast reads permuted data.
    return cell_map.get((model_id, task_id, original), original)


def contrast_pairs(index, spec: ContrastSpec, models, judge, *, tasks_by_difficulty, cell_map=None):
    """Item-paired differences for one hypothesis; incomplete pairs are dropped."""
    left_model = models[spec.left.model]
    right_model = models[spec.right.model]
    pairs = []
    for difficulty in spec.difficulties:
        for task_id in tasks_by_difficulty.get(difficulty, ()):  # deterministic order
            left_cell = _cell(difficulty, spec.left, cell_map, left_model, task_id)
            right_cell = _cell(difficulty, spec.right, cell_map, right_model, task_id)
            left = index.get((left_model, task_id, left_cell, spec.left.turn))
            right = index.get((right_model, task_id, right_cell, spec.right.turn))
            if left is None or right is None:
                continue
            left_value = outcome_value(left, spec.outcome, judge)
            right_value = outcome_value(right, spec.outcome, judge)
            if left_value is None or right_value is None:
                continue
            pairs.append((task_id, float(left_value) - float(right_value)))
    return pairs


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapResult:
    estimate: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    p_two_sided: float | None
    n_items: int
    n_pairs: int
    unavailable_reason: str | None = None


def bootstrap_contrast(pairs, seed_text: str) -> BootstrapResult:
    """2,000-resample item-clustered bootstrap; percentile CI and two-sided p."""
    by_item: dict[str, list[float]] = {}
    for item, value in pairs:
        by_item.setdefault(item, []).append(value)
    if not by_item:
        return BootstrapResult(None, None, None, None, 0, 0, "no_paired_items")
    point = mean(value for values in by_item.values() for value in values)
    if len(by_item) < 2:
        return BootstrapResult(point, None, None, None, len(by_item), len(pairs),
                               "at_least_two_items_required_for_cluster_ci")
    items = sorted(by_item)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big"))
    # Per-item sums and counts make a resample mean two additions instead of a
    # rebuilt list.  ``randrange(n)`` consumes the same underlying draw as
    # ``choice(items)``, so the resamples are bit-identical to the naive form.
    sums = [math.fsum(by_item[item]) for item in items]
    counts = [len(by_item[item]) for item in items]
    total_items = len(items)
    draws = []
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn_sum = 0.0
        drawn_count = 0
        for _ in range(total_items):
            position = rng.randrange(total_items)
            drawn_sum += sums[position]
            drawn_count += counts[position]
        draws.append(drawn_sum / drawn_count)
    draws.sort()

    def quantile(probability):
        position = (len(draws) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)

    # Two-sided percentile bootstrap p: twice the smaller tail mass at zero.
    at_or_below = sum(1 for value in draws if value <= 0.0) / len(draws)
    at_or_above = sum(1 for value in draws if value >= 0.0) / len(draws)
    probability = min(1.0, 2.0 * min(at_or_below, at_or_above))
    return BootstrapResult(point, quantile(0.025), quantile(0.975), probability, len(by_item), len(pairs))


def is_supported(prediction: str, result: BootstrapResult) -> bool:
    """The v3 support rules, transcribed exactly."""
    if result.ci95_lower is None or result.ci95_upper is None or result.estimate is None:
        return False
    if prediction == "negative":
        return result.ci95_upper < 0.0
    if prediction == "positive":
        return result.ci95_lower > 0.0
    if prediction == "non_recovery":
        return result.ci95_upper <= H5_UPPER_BOUND_NATS and result.estimate <= 0.0
    if prediction == "null_or_positive":
        return result.ci95_upper >= 0.0
    raise ConfirmError("unknown prediction rule: %s" % prediction)


# --------------------------------------------------------------------------
# Shuffled-label null
# --------------------------------------------------------------------------

def _rank_key(permutation, model_id, task_id, cell_id):
    return hashlib.sha256(
        ("%s|%d|%s|%s|%s" % (SHUFFLE_KEY, permutation, model_id, task_id, cell_id)).encode("utf-8")).hexdigest()


def shuffled_cell_map(index, models, *, axis: str = "validity", permutation: int = 1):
    """Deterministic label permutation, ranked by the frozen SHA-256 key.

    Cells are ranked by SHA-256 of
    ``DGS-AC1-SHUFFLE-v3|<model_id>|<task_id>|<cell_id>`` and the labels on the
    requested ``axis`` are reassigned in that order.  Returns a map from the
    label a contrast asks for to the original cell that now carries it.

    Judgement call, recorded because v3 does not settle it: the permutation runs
    *within each item*, swapping the two cells that differ only on ``axis``.  A
    free permutation across the whole ``model x difficulty`` stratum could give
    one item two "malfunctioning" cells and another none, leaving a *paired*
    contrast with no partner to look up.  Swapping within the item keeps each
    stratum's label counts exactly as they were -- every item still contributes
    one cell per label -- while keeping the pairing well defined.
    """
    if axis not in ("validity", "tone"):
        raise ConfirmError("shuffle axis must be validity or tone")
    # Group the item's cells by everything EXCEPT the permuted axis, so the two
    # cells inside a group differ only on that axis and can be swapped.
    position = 1 if axis == "validity" else 2
    groups: dict[tuple, list[str]] = {}
    for (model_id, task_id, cell_id, _turn) in index:
        parts = cell_id.split("__")
        key = (model_id, task_id, parts[0], parts[3 - position])
        bucket = groups.setdefault(key, [])
        if cell_id not in bucket:
            bucket.append(cell_id)
    out = {}
    for (model_id, task_id, _difficulty, _held), cells in groups.items():
        labels = sorted(cells)
        ranked = sorted(cells, key=lambda cell: _rank_key(permutation, model_id, task_id, cell))
        for label, original in zip(labels, ranked):
            out[(model_id, task_id, label)] = original
    return _freeze(out)


@dataclass(frozen=True)
class NullCheck:
    """Family-level permutation test over the label-dependent hypotheses.

    Clarification C1 (pre-analysis, 2026-08-17): a single deterministic shuffle
    over ~10 contrasts fails by chance too often to be a usable gate, so the null
    asks whether the REAL family beats the permutation distribution.  ``null_p``
    is the share of permutations matching or beating the real supported count,
    with the usual +1/+1 correction; the check passes when that share is below
    .05 and the real family supports at least one hypothesis.
    """

    family: tuple[str, ...]
    real_count: int
    permutations: int
    permutation_counts: tuple[int, ...]
    histogram: Mapping[int, int]
    null_p: float
    passes: bool


def permutation_null(index, models, judge, *, tasks_by_difficulty, real_count,
                     permutations: int = NULL_PERMUTATIONS):
    """Count supported label-dependent hypotheses under ``permutations`` shuffles."""
    specs = tuple(spec for spec in HYPOTHESES if spec.hypothesis_id in NULL_FAMILY)
    counts = []
    for permutation in range(1, permutations + 1):
        maps = {
            axis: shuffled_cell_map(index, models, axis=axis, permutation=permutation)
            for axis in ("validity", "tone")
        }
        supported = 0
        for spec in specs:
            pairs = contrast_pairs(index, spec, models, judge,
                                   tasks_by_difficulty=tasks_by_difficulty,
                                   cell_map=maps[spec.shuffle_axis])
            result = bootstrap_contrast(
                pairs, "%s|%s|perm%d" % (BOOTSTRAP_KEY, spec.hypothesis_id, permutation))
            if is_supported(spec.prediction, result):
                supported += 1
        counts.append(supported)
    at_or_above = sum(1 for count in counts if count >= real_count)
    null_p = (1 + at_or_above) / (permutations + 1)
    histogram = {value: counts.count(value) for value in sorted(set(counts))}
    return NullCheck(
        NULL_FAMILY, real_count, permutations, tuple(counts), _freeze(histogram),
        null_p, bool(real_count > 0 and null_p < NULL_ALPHA),
    )


# --------------------------------------------------------------------------
# H10 style battery
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StyleResult:
    style_id: str
    result: BootstrapResult
    violates: bool


def style_battery(style_rows, model_id, h1_estimate, *, split: str, excluded=frozenset()):
    """H10: does any style prompt lower M1 by at least half the H1 effect?"""
    index = {}
    for row in style_rows:
        if row.model_id != model_id or row.split != split or row.turn_label != MEASURED:
            continue
        if row.cell_kind != "non_factorial" or not row.cell_id.startswith("style__"):
            continue
        if row.task_id in excluded:
            continue
        index[(row.task_id, row.cell_id)] = row
    tasks = sorted({task for task, _ in index})
    threshold = None if h1_estimate is None else -H10_VIOLATION_FRACTION * abs(h1_estimate)
    out = []
    for style_id in STYLE_PROMPTS:
        pairs = []
        for task_id in tasks:
            styled = index.get((task_id, style_id))
            reference = index.get((task_id, STYLE_REFERENCE))
            if styled is None or reference is None or styled.m1 is None or reference.m1 is None:
                continue
            pairs.append((task_id, float(styled.m1) - float(reference.m1)))
        result = bootstrap_contrast(pairs, "%s|H10|%s|%s" % (BOOTSTRAP_KEY, model_id, style_id))
        violates = bool(
            threshold is not None and result.estimate is not None and result.ci95_upper is not None
            and result.estimate <= threshold and result.ci95_upper < 0.0
        )
        out.append(StyleResult(style_id, result, violates))
    return tuple(out)


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastOutcome:
    hypothesis_id: str
    contrast: str
    outcome: str
    stratum: str
    prediction: str
    discovery: str
    result: BootstrapResult
    supported: bool
    adjusted_p: float | None = None


@dataclass(frozen=True)
class ConfirmResult:
    label: str
    dry_run: bool
    split: str
    models: Mapping[str, str]
    amendments: Amendments
    hypotheses: tuple[ContrastOutcome, ...]
    style: tuple[StyleResult, ...]
    h10_supported: bool
    h7_supported: bool
    null_outcomes: tuple[ContrastOutcome, ...]
    null_check: NullCheck
    null_passes: bool
    iteration_status: str
    success_detail: Mapping[str, Any]
    item_exclusions: Mapping[str, tuple]
    eligibility: Mapping[str, tuple]
    non_answer: tuple[Mapping[str, Any], ...]

    def to_dict(self):
        return _jsonable(self)


def _bootstrap_key(spec, models, suffix=""):
    return "%s|%s|%s|%s%s" % (BOOTSTRAP_KEY, spec.hypothesis_id, models[spec.left.model], spec.stratum, suffix)


def run_confirmation(
    rows,
    style_rows,
    judge,
    *,
    split: str,
    discovery=None,
    models: Mapping[str, str] | None = None,
    amendments: Amendments = AMENDED_RULES,
    label: str = "Preregistration v3 holdout confirmation",
    dry_run: bool = False,
    permutations: int = NULL_PERMUTATIONS,
) -> ConfirmResult:
    """Run every v3 hypothesis, the shuffled null, and the success criterion."""
    models = dict(models or {"primary": PRIMARY_MODEL, "control": CONTROL_MODEL})
    discovery = discovery if discovery is not None else _freeze({})
    rows = tuple(rows)
    exclusions = {
        model_id: item_exclusions(rows, model_id, split=split)
        for model_id in models.values()
    }
    excluded = {
        model_id: frozenset(item.task_id for item in items) if amendments.item_exclusion else frozenset()
        for model_id, items in exclusions.items()
    }
    index = build_index(rows, split=split, models=models, excluded=excluded)
    tasks_by_difficulty: dict[str, list[str]] = {}
    for (_model, task_id, cell_id, _turn) in index:
        difficulty = cell_id.split("__")[0]
        bucket = tasks_by_difficulty.setdefault(difficulty, [])
        if task_id not in bucket:
            bucket.append(task_id)
    for bucket in tasks_by_difficulty.values():
        bucket.sort()

    outcomes = []
    for spec in HYPOTHESES:
        pairs = contrast_pairs(index, spec, models, judge, tasks_by_difficulty=tasks_by_difficulty)
        result = bootstrap_contrast(pairs, _bootstrap_key(spec, models))
        outcomes.append(ContrastOutcome(
            spec.hypothesis_id, spec.contrast, spec.outcome, spec.stratum,
            PREDICTION_TEXT[spec.prediction], _discovery_text(spec, discovery, models),
            result, is_supported(spec.prediction, result),
        ))

    # Benjamini-Hochberg across H1-H9 (q = .05), as the secondary summary.
    family = {
        item.hypothesis_id: item.result.p_two_sided
        for item in outcomes
        if item.result.p_two_sided is not None and item.hypothesis_id.startswith(BH_FAMILY_PREFIXES)
    }
    adjusted = benjamini_hochberg(family) if family else {}
    outcomes = tuple(
        ContrastOutcome(
            item.hypothesis_id, item.contrast, item.outcome, item.stratum, item.prediction,
            item.discovery, item.result, item.supported, adjusted.get(item.hypothesis_id),
        )
        for item in outcomes
    )
    by_id = {item.hypothesis_id: item for item in outcomes}
    h7_supported = all(by_id[key].supported for key in ("H7a", "H7b") if key in by_id)

    h1 = by_id.get("H1")
    style = style_battery(
        style_rows, models["primary"], h1.result.estimate if h1 else None,
        split=split, excluded=excluded.get(models["primary"], frozenset()),
    ) if style_rows else ()
    h10_supported = bool(style) and not any(item.violates for item in style)

    # Transparency table: one shuffle (k = 1) repeated across every contrast.
    null_outcomes = []
    for spec in HYPOTHESES:
        cell_map = shuffled_cell_map(index, models, axis=spec.shuffle_axis, permutation=1)
        pairs = contrast_pairs(index, spec, models, judge,
                               tasks_by_difficulty=tasks_by_difficulty, cell_map=cell_map)
        result = bootstrap_contrast(pairs, _bootstrap_key(spec, models, "|shuffled"))
        # Judgement call, recorded: H5 and H7 are "no effect" rules, which a
        # shuffled (hence effectless) dataset satisfies by construction -- under
        # a literal reading the null could never pass.  The null therefore asks
        # the DIRECTIONAL question behind each hypothesis: whether permuted
        # labels manufacture the signed effect the hypothesis predicts.
        prediction = NULL_PREDICTION.get(spec.prediction, spec.prediction)
        null_outcomes.append(ContrastOutcome(
            spec.hypothesis_id, spec.contrast, spec.outcome, spec.stratum,
            PREDICTION_TEXT[prediction],
            "label-dependent" if spec.label_dependent else "label-invariant (not in the null verdict)",
            result, is_supported(prediction, result),
        ))
    null_outcomes = tuple(null_outcomes)
    # The null VERDICT is the family-level permutation test (clarification C1),
    # not this single shuffle, which is kept only for transparency.
    real_count = sum(1 for item in outcomes if item.hypothesis_id in NULL_FAMILY and item.supported)
    null_check = permutation_null(
        index, models, judge, tasks_by_difficulty=tasks_by_difficulty,
        real_count=real_count, permutations=permutations,
    )
    null_passes = null_check.passes

    core = ("H1", "H2a", "H2b", "H3a", "H3b")
    core_supported = [key for key in core if key in by_id and by_id[key].supported]
    h6a = by_id.get("H6a")
    success = (
        len(core_supported) >= 3
        and h6a is not None and h6a.supported
        and null_passes
    )
    detail = {
        "core_supported": tuple(core_supported),
        "core_supported_count": len(core_supported),
        "core_required": 3,
        "h6a_supported": bool(h6a is not None and h6a.supported),
        "null_passes": null_passes,
        "null_p": null_check.null_p,
        "null_real_count": null_check.real_count,
    }
    eligibility = {
        model_id: metric_eligibility(rows, model_id, split=split, amendments=amendments)
        for model_id in models.values()
    }
    non_answer = exploratory_cell_summary(rows, phase=None, split=split)
    return ConfirmResult(
        label, dry_run, split, _freeze(models), amendments, outcomes, style, h10_supported,
        h7_supported, null_outcomes, null_check, null_passes, "SUCCESS" if success else "FAIL",
        _freeze(detail), _freeze(exclusions), _freeze(eligibility), non_answer,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _interval(result: BootstrapResult) -> str:
    if result.estimate is None:
        return "unavailable (`%s`)" % (result.unavailable_reason or "no_data")
    if result.ci95_lower is None:
        return "%.3f (no CI: `%s`)" % (result.estimate, result.unavailable_reason or "")
    return "%.3f [%.3f, %.3f]" % (result.estimate, result.ci95_lower, result.ci95_upper)


def render_confirm_markdown(result: ConfirmResult) -> str:
    lines = ["# %s" % result.label, ""]
    if result.dry_run:
        lines += [
            "> **DRY RUN ON DISCOVERY - NOT CONFIRMATORY.**",
            "> This output was produced by pointing the frozen confirmatory script at the",
            "> discovery data to exercise the code path. It is not a holdout result and",
            "> must never be reported as one.",
            "",
        ]
    lines += [
        "- primary: `%s`; control: `%s`" % (result.models["primary"], result.models["control"]),
        "- split analysed: `%s`" % result.split,
        "- amendments: A2 item exclusion %s, A3 pooled-SD fallback %s, A4 pooled QC %s" % (
            "on" if result.amendments.item_exclusion else "off",
            "on" if result.amendments.pooled_sd_fallback else "off",
            "on" if result.amendments.pooled_qc else "off"),
        "- **iteration_status: %s**" % result.iteration_status,
        "- success criterion: at least 3 of {H1, H2a, H2b, H3a, H3b} supported (%d: %s),"
        % (result.success_detail["core_supported_count"], ", ".join(result.success_detail["core_supported"]) or "none"),
        "  H6a supported (%s), and the permutation null check passes (%s, null_p = %.4f)." % (
            "yes" if result.success_detail["h6a_supported"] else "no",
            "passes" if result.null_passes else "FAILS", result.null_check.null_p),
        "",
        "M1 is analysed available-case in raw nats. Non-answers are missing for M1 and are",
        "analysed as their own outcome (H9); the per-cell non-answer rate is tabled below.",
        "The known MNAR risk (non-answers cluster in hostile cells) is reported, not modelled away.",
        "",
        "## Hypotheses",
        "",
        "| ID | contrast | stratum | prediction | discovery [95% CI] | holdout [95% CI] | items | supported | BH p |",
        "| --- | --- | --- | --- | --- | --- | ---: | :---: | ---: |",
    ]
    for item in result.hypotheses:
        lines.append("| %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (
            item.hypothesis_id, item.contrast, item.stratum, item.prediction, item.discovery,
            _interval(item.result), item.result.n_items,
            "**yes**" if item.supported else "no",
            "%.4f" % item.adjusted_p if item.adjusted_p is not None else "-",
        ))
    lines += [
        "",
        "H7 (family boundary) is supported only if both H7a and H7b are: **%s**." % (
            "yes" if result.h7_supported else "no"),
        "",
        "## H10 style battery (M1, style - neutral reference, paired by item)",
        "",
        "| style prompt | estimate [95% CI] | items | violates H10 |",
        "| --- | --- | ---: | :---: |",
    ]
    if result.style:
        for item in result.style:
            lines.append("| `%s` | %s | %d | %s |" % (
                item.style_id, _interval(item.result), item.result.n_items,
                "**yes**" if item.violates else "no"))
        lines.append("")
        lines.append("H10 supported (no style prompt reproduces at least half the H1 effect): **%s**."
                     % ("yes" if result.h10_supported else "no"))
    else:
        lines.append("| - | no style battery rows supplied | 0 | - |")
    lines += [
        "",
        "## Shuffled-label null - family-level permutation test (clarification C1)",
        "",
        "Labels are permuted within the item by SHA-256 of",
        "`%s|<k>|<model_id>|<task_id>|<cell_id>`, so each stratum keeps its exact" % SHUFFLE_KEY,
        "label counts and every paired lookup stays defined. The permuted axis is the one",
        "that defines each contrast: validity for H1 and H8, tone for H2a, H2b, H6a and H9.",
        "",
        "The family is the directional, label-dependent set **%s**. H3, H4, H5 and H6b"
        % ", ".join(result.null_check.family),
        "compare two endpoints of the same cell and are permutation-invariant; H7 is a",
        "no-effect rule and H10 compares against the style reference, so all are excluded.",
        "",
        "| quantity | value |",
        "| --- | --- |",
        "| hypotheses supported on real labels | **%d** of %d |" % (
            result.null_check.real_count, len(result.null_check.family)),
        "| permutations | %d |" % result.null_check.permutations,
        "| permutations matching or beating the real count | %d |" % sum(
            count for value, count in result.null_check.histogram.items()
            if value >= result.null_check.real_count),
        "| `null_p` = (1 + that count) / (permutations + 1) | **%.4f** |" % result.null_check.null_p,
        "| null check (passes iff `null_p` < %.2f and real count > 0) | **%s** |" % (
            NULL_ALPHA, "PASSES" if result.null_check.passes else "FAILS"),
        "",
        "Permutation-count histogram (supported hypotheses per permutation):",
        "",
        "| supported | permutations |",
        "| ---: | ---: |",
    ]
    for value in sorted(result.null_check.histogram):
        lines.append("| %d | %d |" % (value, result.null_check.histogram[value]))
    lines += [
        "",
        "### Single shuffle (k = 1), every contrast, for transparency",
        "",
        "This table is diagnostic only and does not decide the null check above.",
        "",
        "| ID | scope | shuffled estimate [95% CI] | items | supported |",
        "| --- | --- | --- | ---: | :---: |",
    ]
    for item in result.null_outcomes:
        in_family = item.hypothesis_id in result.null_check.family
        lines.append("| %s | %s | %s | %d | %s |" % (
            item.hypothesis_id, "null family" if in_family else "excluded",
            _interval(item.result), item.result.n_items,
            "yes" if item.supported else "no"))
    lines += [
        "",
        "## Amendment A2 - items excluded per model (treatment-blind, holdout's own baseline)",
        "",
        "| model | item | baseline cell | invalid/absent baseline resamples | reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    any_excluded = False
    for model_id in sorted(result.item_exclusions):
        for item in result.item_exclusions[model_id]:
            any_excluded = True
            lines.append("| `%s` | %s | %s | %d/%d | `%s` |" % (
                model_id, item.task_id, item.baseline_cell_id or "absent",
                item.invalid_or_absent_resamples, item.required_resamples, item.reason))
    if not any_excluded:
        lines.append("| - | none | - | - | - |")
    lines += [
        "",
        "## Confirmatory QC (A4: the 5% bars are pooled across cells)",
        "",
        "| model | metric | eligible | decided on | pooled rate | worst cell | worst-cell rate |",
        "| --- | --- | :---: | --- | ---: | --- | ---: |",
    ]
    for model_id in sorted(result.eligibility):
        for item in result.eligibility[model_id]:
            lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                model_id, item.metric_name, "yes" if item.eligible else "**no**", item.scope,
                "%.4f" % item.pooled_rate if item.pooled_rate is not None else "n/a",
                item.worst_cell_id or "-",
                "%.4f" % item.worst_rate if item.worst_rate is not None else "n/a"))
    lines += [
        "",
        "These QC verdicts are reported for completeness. The v3 hypotheses use available-case",
        "M1 as preregistered and are not gated on this table.",
        "",
        "## Non-answer rate by cell and endpoint (reported next to every M1 result)",
        "",
        "| model | cell | endpoint | items | non-answer rate | mean M1 (n) |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in result.non_answer:
        if record["cell_kind"] != "factorial":
            continue
        lines.append("| `%s` | %s | %s | %d | %s | %s |" % (
            record["model_id"], record["cell_id"], record["turn_label"], record["n_items"],
            "%.3f" % record["mean_non_answer_rate"] if record["mean_non_answer_rate"] is not None else "-",
            "%.3f (%d)" % (record["mean_m1"], record["n_m1"]) if record["mean_m1"] is not None else "-",
        ))
    lines.append("")
    return "\n".join(lines)
