"""Missing-data sensitivity analysis for the answer-margin (M1) contrasts.

M1 is preregistered and reported **available-case**: an endpoint whose greedy
response carries no parseable ``Answer: X`` has no margin, so the item-paired
contrasts are computed on the items that answered in *both* cells.  Non-answers
are not missing at random -- they concentrate in the hostile cells, which is
itself one of the findings (H9) -- so a reviewer is entitled to ask whether the
published effects survive plausible treatments of the missing values.  This
module answers that question and nothing else: it re-uses the published pairing
and the published 2,000-resample item-clustered bootstrap
(:func:`src.confirm.bootstrap_contrast`) and only varies what happens to a
missing value.

Four treatments, all on the same pairable item set except the first:

``available_case``
    Drop the item pair.  This is the published analysis and must reproduce it.
``zero_imputation``
    A non-answer committed to no option, so its margin is set to 0 nats -- the
    margin of a model with no preferred option.  Applied on both sides.
``manski_lower`` / ``manski_upper``
    Worst-case bounds inside the observed support: every missing value in the
    treated cell is imputed at the minimum (lower bound) or maximum (upper
    bound) of that model's neutral-accurate measured M1 distribution for that
    split, and every missing value in the reference cell at the opposite
    extreme.  The pair of bounds brackets the effect for *any* imputation whose
    values lie in that support.
``delta`` (the tipping point)
    Impute every missing treated-cell value at a constant delta nats and every
    missing reference-cell value at 0, and search for the smallest delta at
    which the item-paired 95% CI includes 0.  delta = 0 is exactly the
    zero-imputation treatment, so the search starts from it.

Nothing here is confirmatory and nothing here amends a preregistered result: the
published numbers stay the published numbers and this is a sensitivity analysis
reported beside them.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import csv
import json
import math
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .confirm import BOOTSTRAP_KEY, BootstrapResult, bootstrap_contrast
from .extension import EXTENSION_BOOTSTRAP_KEY
from .extract import MetricRow

MISSINGNESS_KEY = "DGS-AC1-MISSINGNESS-v1"
PRIMARY_MODEL = "google/gemma-2-9b-it"
CONTROL_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Treatment names, in the order they are reported.
AVAILABLE_CASE = "available_case"
ZERO_IMPUTATION = "zero_imputation"
MANSKI_LOWER = "manski_lower"
MANSKI_UPPER = "manski_upper"
DELTA = "delta"
TREATMENTS = (AVAILABLE_CASE, ZERO_IMPUTATION, MANSKI_LOWER, MANSKI_UPPER)

# How a missing M1 came about.  Only the first is a non-answer; the second is a
# top-20 logprob truncation on a response that *did* commit to a letter, and is
# counted separately so it can never be silently read as a non-answer.
NON_ANSWER = "non_answer"
CANDIDATE_ABSENT = "candidate_absent"
OTHER_MISSING = "other"

# Tipping-point search: 512 nats is far outside anything these models produce, so
# hitting the cap means "no tipping point at any plausible margin", not a bug.
TIPPING_LIMIT = 512.0
TIPPING_TOLERANCE = 1e-3

MEASURED, ONSET = "measured", "onset"
ACCURATE, MALFUNCTIONING = "accurate", "malfunctioning_always_fail"
NEUTRAL, HOSTILE = "neutral", "hostile"


class MissingnessError(ValueError):
    """Raised when the sensitivity analysis cannot be assembled as specified."""


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
# The contrast table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """One endpoint of a paired contrast; the difficulty comes from the stratum."""

    validity: str
    tone: str
    turn: str

    def cell_id(self, difficulty: str) -> str:
        return "%s__%s__%s" % (difficulty, self.validity, self.tone)


@dataclass(frozen=True)
class Contrast:
    """One M1 contrast, with the keys that find its published estimate."""

    contrast_id: str
    label: str
    treated: Cell
    reference: Cell
    difficulties: tuple[str, ...]
    stratum: str
    direction: str                                   # negative | positive
    # (contrast, stratum) in results/summaries/phase1/exploratory/paired_contrasts.csv
    discovery_key: tuple[str, str] | None = None
    # (role, hypothesis_id) in results/summaries/phase2/hypotheses.csv, plus the
    # stratum string src.confirm uses in its bootstrap seed.
    holdout_keys: tuple[tuple[str, str], ...] = ()
    holdout_stratum: str | None = None

    def holdout_hypothesis(self, role: str) -> str | None:
        for candidate, hypothesis_id in self.holdout_keys:
            if candidate == role:
                return hypothesis_id
        return None


CONTRASTS: tuple[Contrast, ...] = (
    Contrast(
        "H1", "M1, malfunctioning - accurate (measured)",
        Cell(MALFUNCTIONING, NEUTRAL, MEASURED), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("easy",), "easy | neutral", "negative",
        ("validity_malfunctioning_minus_accurate", "easy|neutral"),
        (("primary", "H1"), ("control", "H7a")), "easy | neutral"),
    Contrast(
        "H1_hard", "M1, malfunctioning - accurate (measured)",
        Cell(MALFUNCTIONING, NEUTRAL, MEASURED), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("hard",), "hard | neutral", "negative",
        ("validity_malfunctioning_minus_accurate", "hard|neutral")),
    Contrast(
        "H2a", "M1, hostile - neutral (measured, accurate arm)",
        Cell(ACCURATE, HOSTILE, MEASURED), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("easy",), "easy | accurate", "negative",
        ("tone_hostile_minus_neutral", "easy|accurate"),
        (("primary", "H2a"), ("control", "H7b")), "easy | accurate"),
    Contrast(
        "H2b", "M1, hostile - neutral (measured, accurate arm)",
        Cell(ACCURATE, HOSTILE, MEASURED), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("hard",), "hard | accurate", "negative",
        ("tone_hostile_minus_neutral", "hard|accurate"),
        (("primary", "H2b"),), "hard | accurate"),
    Contrast(
        "tone_pooled", "M1, hostile - neutral (measured, accurate arm), easy+hard pooled",
        Cell(ACCURATE, HOSTILE, MEASURED), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("easy", "hard"), "easy+hard | accurate", "negative"),
    Contrast(
        "H3a", "M1, onset - measured (accurate, neutral wording)",
        Cell(ACCURATE, NEUTRAL, ONSET), Cell(ACCURATE, NEUTRAL, MEASURED),
        ("easy",), "easy | accurate, neutral", "negative",
        ("onset_minus_measured", "easy__accurate__neutral"),
        (("primary", "H3a"),), "easy | neutral"),
    Contrast(
        "H3b", "M1, onset - measured (accurate, hostile wording)",
        Cell(ACCURATE, HOSTILE, ONSET), Cell(ACCURATE, HOSTILE, MEASURED),
        ("easy",), "easy | accurate, hostile", "negative",
        ("onset_minus_measured", "easy__accurate__hostile"),
        (("primary", "H3b"),), "easy | hostile"),
)

CONTRASTS_BY_ID = _freeze({item.contrast_id: item for item in CONTRASTS})


# --------------------------------------------------------------------------
# Missingness classification and pairing
# --------------------------------------------------------------------------

def missing_kind(missing_reason: str | None) -> str | None:
    """Classify an M1 missing reason; ``None`` means the value was observed."""
    if missing_reason is None or missing_reason == "":
        return None
    if missing_reason == "m1_invalid_final_answer":
        return NON_ANSWER
    if missing_reason.startswith("m1_candidate_absent_"):
        return CANDIDATE_ABSENT
    return OTHER_MISSING


@dataclass(frozen=True)
class ItemPair:
    """One item's two endpoints, each either an observed M1 or a missing kind."""

    task_id: str
    difficulty: str
    treated: float | None
    reference: float | None
    treated_missing: str | None = None
    reference_missing: str | None = None

    @property
    def complete(self) -> bool:
        return self.treated is not None and self.reference is not None


def build_m1_index(rows: Iterable[MetricRow], *, model_id: str, split: str,
                   excluded: Iterable[str] = ()) -> Mapping[tuple[str, str, str], MetricRow]:
    """``(task_id, cell_id, turn_label) -> row`` for one model's factorial split."""
    excluded = frozenset(excluded)
    index: dict[tuple[str, str, str], MetricRow] = {}
    for row in rows:
        if row.model_id != model_id or row.split != split or row.cell_kind != "factorial":
            continue
        if row.task_id in excluded:
            continue
        key = (row.task_id, row.cell_id, row.turn_label)
        if key in index:
            raise MissingnessError("duplicate endpoint for %s" % (key,))
        index[key] = row
    return _freeze(index)


def tasks_by_difficulty(index) -> Mapping[str, tuple[str, ...]]:
    """Items present in the index, grouped by the difficulty in their cell IDs."""
    buckets: dict[str, set[str]] = {}
    for (task_id, cell_id, _turn) in index:
        buckets.setdefault(cell_id.split("__")[0], set()).add(task_id)
    return _freeze({key: tuple(sorted(value)) for key, value in buckets.items()})


def build_pairs(index, contrast: Contrast) -> tuple[ItemPair, ...]:
    """Item pairs for one contrast, keeping the pairs a missing M1 would drop.

    An item is included whenever *both* endpoint rows exist, whether or not
    their M1 is present; an item whose endpoint row is absent altogether cannot
    be imputed at all and is reported separately (``n_endpoint_absent``).
    """
    by_difficulty = tasks_by_difficulty(index)
    out = []
    for difficulty in contrast.difficulties:
        for task_id in by_difficulty.get(difficulty, ()):
            treated = index.get((task_id, contrast.treated.cell_id(difficulty), contrast.treated.turn))
            reference = index.get((task_id, contrast.reference.cell_id(difficulty), contrast.reference.turn))
            if treated is None or reference is None:
                continue
            out.append(ItemPair(
                task_id, difficulty, treated.m1, reference.m1,
                missing_kind(treated.m1_missing_reason), missing_kind(reference.m1_missing_reason),
            ))
    return tuple(out)


def count_endpoint_absent(index, contrast: Contrast) -> int:
    """Items of the contrast's difficulty that lack one of its two endpoint rows."""
    by_difficulty = tasks_by_difficulty(index)
    missing = 0
    for difficulty in contrast.difficulties:
        for task_id in by_difficulty.get(difficulty, ()):
            treated = index.get((task_id, contrast.treated.cell_id(difficulty), contrast.treated.turn))
            reference = index.get((task_id, contrast.reference.cell_id(difficulty), contrast.reference.turn))
            if treated is None or reference is None:
                missing += 1
    return missing


# --------------------------------------------------------------------------
# The imputation rules
# --------------------------------------------------------------------------

def treatment_fills(treatment: str, *, support: tuple[float, float] | None = None,
                    delta: float | None = None) -> tuple[float | None, float | None]:
    """``(treated_fill, reference_fill)`` for one treatment; ``None`` = drop the pair.

    The Manski fills are deliberately crossed: the lower (most negative) bound
    puts the treated cell at the bottom of the support and the reference cell at
    the top, and the upper bound does the reverse.
    """
    if treatment == AVAILABLE_CASE:
        return (None, None)
    if treatment == ZERO_IMPUTATION:
        return (0.0, 0.0)
    if treatment in (MANSKI_LOWER, MANSKI_UPPER):
        if support is None:
            raise MissingnessError("%s needs the observed support" % treatment)
        low, high = float(support[0]), float(support[1])
        if low > high:
            raise MissingnessError("support must be (minimum, maximum)")
        return (low, high) if treatment == MANSKI_LOWER else (high, low)
    if treatment == DELTA:
        if delta is None:
            raise MissingnessError("the delta treatment needs a delta")
        return (float(delta), 0.0)
    raise MissingnessError("unknown treatment: %s" % treatment)


def paired_differences(pairs: Sequence[ItemPair], *, treated_fill: float | None,
                       reference_fill: float | None) -> tuple[tuple[str, float], ...]:
    """``(task_id, treated - reference)`` after filling; unfillable pairs drop out."""
    out = []
    for pair in pairs:
        treated = pair.treated if pair.treated is not None else treated_fill
        reference = pair.reference if pair.reference is not None else reference_fill
        if treated is None or reference is None:
            continue
        out.append((pair.task_id, float(treated) - float(reference)))
    return tuple(out)


@dataclass(frozen=True)
class CellMissingness:
    """One cell x endpoint: how many M1 values are there, and why the rest are not."""

    model_id: str
    split: str
    cell_id: str
    turn_label: str
    n_endpoints: int
    n_observed: int
    n_non_answer: int
    n_candidate_absent: int
    n_other: int
    mean_m1: float | None


def cell_missingness(rows: Iterable[MetricRow], *, model_id: str, split: str,
                     turns: Sequence[str] = (MEASURED, ONSET),
                     excluded: Iterable[str] = ()) -> tuple[CellMissingness, ...]:
    """Per-cell non-answer counts for the endpoints these contrasts read."""
    excluded = frozenset(excluded)
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        if row.model_id != model_id or row.split != split or row.cell_kind != "factorial":
            continue
        if row.turn_label not in turns or row.task_id in excluded:
            continue
        grouped.setdefault((row.cell_id, row.turn_label), []).append(row)
    out = []
    for key in sorted(grouped):
        group = grouped[key]
        kinds = [missing_kind(row.m1_missing_reason) for row in group]
        observed = [row.m1 for row in group if row.m1 is not None]
        out.append(CellMissingness(
            model_id, split, key[0], key[1], len(group), len(observed),
            kinds.count(NON_ANSWER), kinds.count(CANDIDATE_ABSENT), kinds.count(OTHER_MISSING),
            mean(observed) if observed else None))
    return tuple(out)


def observed_support(rows: Iterable[MetricRow], *, model_id: str, split: str,
                     excluded: Iterable[str] = ()) -> tuple[float | None, float | None, int]:
    """``(min, max, n)`` of one model's neutral-accurate **measured** M1 for a split.

    This is the support the Manski bounds impute inside: the untreated baseline
    cell of the design, pooled over both difficulty strata.
    """
    excluded = frozenset(excluded)
    values = [
        row.m1 for row in rows
        if row.model_id == model_id and row.split == split and row.cell_kind == "factorial"
        and row.turn_label == MEASURED and row.feedback_validity == ACCURATE and row.tone == NEUTRAL
        and row.task_id not in excluded and row.m1 is not None
    ]
    if not values:
        return (None, None, 0)
    return (min(values), max(values), len(values))


# --------------------------------------------------------------------------
# Bootstrap seeds
# --------------------------------------------------------------------------

def missingness_seed(*, split: str, model_id: str, contrast_id: str, treatment: str) -> str:
    return "%s|%s|%s|%s|%s" % (MISSINGNESS_KEY, split, model_id, contrast_id, treatment)


def published_seed(contrast: Contrast, *, split: str, model_id: str, role: str | None) -> str | None:
    """The seed the *published* analysis used, so available-case reproduces it exactly.

    Discovery numbers come from the Phase-1 exploratory contrast table
    (``DGS-AC1-EXPLORATORY-v1``); holdout numbers come from the frozen
    confirmatory script (``DGS-AC1-CONFIRM-v3``); the third-family arm comes from
    the exploratory extension run (``DGS-AC1-EXTENSION-v1``), which uses one key
    for both splits.  A contrast with no published counterpart returns ``None``
    and falls back to the missingness seed.
    """
    if role == "extension":
        hypothesis_id = contrast.holdout_hypothesis("primary")
        if hypothesis_id is None or contrast.holdout_stratum is None:
            return None
        return "%s|%s|%s|%s|%s" % (
            EXTENSION_BOOTSTRAP_KEY, model_id, split, hypothesis_id, contrast.holdout_stratum)
    if split == "discovery":
        if contrast.discovery_key is None:
            return None
        name, stratum = contrast.discovery_key
        return "DGS-AC1-EXPLORATORY-v1|%s|%s|m1|%s" % (model_id, name, stratum)
    if split == "holdout":
        hypothesis_id = contrast.holdout_hypothesis(role) if role else None
        if hypothesis_id is None or contrast.holdout_stratum is None:
            return None
        return "%s|%s|%s|%s" % (BOOTSTRAP_KEY, hypothesis_id, model_id, contrast.holdout_stratum)
    return None


# --------------------------------------------------------------------------
# Tipping point
# --------------------------------------------------------------------------

def ci_includes_zero(result: BootstrapResult) -> bool:
    if result.ci95_lower is None or result.ci95_upper is None:
        return True
    return result.ci95_lower <= 0.0 <= result.ci95_upper


def excludes_zero_in_direction(result: BootstrapResult, direction: str) -> bool:
    if result.ci95_lower is None or result.ci95_upper is None:
        return False
    return result.ci95_upper < 0.0 if direction == "negative" else result.ci95_lower > 0.0


def excludes_zero_against_direction(result: BootstrapResult, direction: str) -> bool:
    opposite = "positive" if direction == "negative" else "negative"
    return excludes_zero_in_direction(result, opposite)


@dataclass(frozen=True)
class TippingPoint:
    """The constant delta at which imputation makes the 95% CI include zero."""

    delta: float | None
    reason: str | None
    n_missing_treated: int
    n_missing_reference: int
    result_at_delta: BootstrapResult | None = None
    search_limit: float = TIPPING_LIMIT


def tipping_point(pairs: Sequence[ItemPair], seed_text: str, *, direction: str = "negative",
                  limit: float = TIPPING_LIMIT, tolerance: float = TIPPING_TOLERANCE) -> TippingPoint:
    """Smallest |delta| whose imputation makes the item-paired 95% CI include 0.

    Every missing treated-cell value is imputed at ``delta`` and every missing
    reference-cell value at 0, so ``delta = 0`` reproduces the zero-imputation
    treatment exactly.  Raising ``delta`` (for a negative-direction contrast)
    adds the same constant to every affected item difference, so each bootstrap
    resample mean -- and therefore each CI bound -- is monotone in ``delta`` and
    a bisection is exact up to ``tolerance``.  One seed is used for every
    evaluation so the resampled item indices never move.
    """
    if direction not in ("negative", "positive"):
        raise MissingnessError("direction must be negative or positive")
    sign = 1.0 if direction == "negative" else -1.0
    n_treated = sum(1 for pair in pairs if pair.treated is None)
    n_reference = sum(1 for pair in pairs if pair.reference is None)

    def evaluate(delta: float) -> BootstrapResult:
        treated_fill, reference_fill = treatment_fills(DELTA, delta=delta)
        return bootstrap_contrast(
            paired_differences(pairs, treated_fill=treated_fill, reference_fill=reference_fill),
            seed_text)

    if not pairs:
        return TippingPoint(None, "no_pairable_items", n_treated, n_reference)
    base = evaluate(0.0)
    if base.ci95_lower is None:
        return TippingPoint(None, "no_confidence_interval", n_treated, n_reference, base)
    if ci_includes_zero(base):
        return TippingPoint(0.0, "ci_already_includes_zero_at_zero_imputation", n_treated, n_reference, base)
    if n_treated == 0:
        return TippingPoint(None, "no_missing_treated_values", n_treated, n_reference, base)
    low, high = 0.0, 1.0
    while high <= limit and not ci_includes_zero(evaluate(sign * high)):
        low, high = high, high * 2.0
    if high > limit:
        return TippingPoint(None, "no_tipping_point_within_%.0f_nats" % limit, n_treated, n_reference,
                            evaluate(sign * limit), limit)
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if ci_includes_zero(evaluate(sign * middle)):
            high = middle
        else:
            low = middle
    return TippingPoint(sign * high, None, n_treated, n_reference, evaluate(sign * high), limit)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingnessCounts:
    """Every denominator a reader needs to judge the imputations."""

    n_items_in_stratum: int
    n_pairable: int
    n_endpoint_absent: int
    n_available: int
    n_treated_missing: int
    n_reference_missing: int
    n_both_missing: int
    treated_non_answer: int
    treated_candidate_absent: int
    treated_other: int
    reference_non_answer: int
    reference_candidate_absent: int
    reference_other: int


def count_missingness(index, contrast: Contrast, pairs: Sequence[ItemPair]) -> MissingnessCounts:
    by_difficulty = tasks_by_difficulty(index)
    in_stratum = sum(len(by_difficulty.get(difficulty, ())) for difficulty in contrast.difficulties)

    def kind_count(side: str, kind: str) -> int:
        return sum(1 for pair in pairs if getattr(pair, "%s_missing" % side) == kind)

    return MissingnessCounts(
        in_stratum, len(pairs), count_endpoint_absent(index, contrast),
        sum(1 for pair in pairs if pair.complete),
        sum(1 for pair in pairs if pair.treated is None),
        sum(1 for pair in pairs if pair.reference is None),
        sum(1 for pair in pairs if pair.treated is None and pair.reference is None),
        kind_count("treated", NON_ANSWER), kind_count("treated", CANDIDATE_ABSENT),
        kind_count("treated", OTHER_MISSING),
        kind_count("reference", NON_ANSWER), kind_count("reference", CANDIDATE_ABSENT),
        kind_count("reference", OTHER_MISSING),
    )


@dataclass(frozen=True)
class TreatmentOutcome:
    treatment: str
    treated_fill: float | None
    reference_fill: float | None
    seed_text: str
    seed_source: str
    result: BootstrapResult
    excludes_zero: bool


@dataclass(frozen=True)
class PublishedReference:
    """The committed estimate this contrast must reproduce, and whether it did."""

    source: str
    key: str
    estimate: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    n_items: int | None
    max_abs_difference: float | None = None
    reproduced: bool | None = None


@dataclass(frozen=True)
class ContrastOutcome:
    model_id: str
    role: str
    split: str
    contrast_id: str
    label: str
    stratum: str
    direction: str
    counts: MissingnessCounts
    support_min: float | None
    support_max: float | None
    support_n: int
    observed_min: float | None
    observed_max: float | None
    reference_mean: float | None
    treatments: tuple[TreatmentOutcome, ...]
    tipping: TippingPoint
    verdict: str
    published: PublishedReference | None = None

    def treatment(self, name: str) -> TreatmentOutcome | None:
        for item in self.treatments:
            if item.treatment == name:
                return item
        return None


VERDICTS = {
    "available_case_null": "available-case CI already includes 0 (no effect to be robust)",
    "available_case_opposite": "available-case effect runs against the predicted direction",
    "flips": "flips sign under zero-imputation",
    "weakened": "zero-imputation CI includes 0 -- the effect depends on the missing values",
    "robust_bounded": "robust to all four treatments (sign determined inside the observed support)",
    "robust_unbounded": "robust to imputation; the worst-case bounds are uninformative",
    "unavailable": "no estimate",
}


def classify(direction: str, available, zero, lower, upper) -> str:
    """The robustness verdict, from the four treatments alone."""
    if available is None or available.result.estimate is None or available.result.ci95_lower is None:
        return "unavailable"
    if excludes_zero_against_direction(available.result, direction):
        return "available_case_opposite"
    if not excludes_zero_in_direction(available.result, direction):
        return "available_case_null"
    if zero is None or zero.result.ci95_lower is None:
        return "unavailable"
    if excludes_zero_against_direction(zero.result, direction):
        return "flips"
    if not excludes_zero_in_direction(zero.result, direction):
        return "weakened"
    if (lower is not None and upper is not None
            and excludes_zero_in_direction(lower.result, direction)
            and excludes_zero_in_direction(upper.result, direction)):
        return "robust_bounded"
    return "robust_unbounded"


def analyse_contrast(rows: Sequence[MetricRow], contrast: Contrast, *, model_id: str, split: str,
                     role: str | None = None, excluded: Iterable[str] = (),
                     published: PublishedReference | None = None,
                     tolerance: float = 5e-3) -> ContrastOutcome:
    """Run all four treatments plus the tipping-point search for one contrast."""
    index = build_m1_index(rows, model_id=model_id, split=split, excluded=excluded)
    pairs = build_pairs(index, contrast)
    counts = count_missingness(index, contrast, pairs)
    support_min, support_max, support_n = observed_support(
        rows, model_id=model_id, split=split, excluded=excluded)
    support = None if support_min is None else (support_min, support_max)
    observed = [value for pair in pairs for value in (pair.treated, pair.reference) if value is not None]
    reference_values = [pair.reference for pair in pairs if pair.reference is not None]

    published_key = published_seed(contrast, split=split, model_id=model_id, role=role)
    available_seed = published_key or missingness_seed(
        split=split, model_id=model_id, contrast_id=contrast.contrast_id, treatment=AVAILABLE_CASE)
    available_differences = paired_differences(pairs, treated_fill=None, reference_fill=None)
    outcomes = []
    for treatment in TREATMENTS:
        if treatment in (MANSKI_LOWER, MANSKI_UPPER) and support is None:
            continue
        treated_fill, reference_fill = treatment_fills(treatment, support=support)
        differences = paired_differences(pairs, treated_fill=treated_fill, reference_fill=reference_fill)
        # A contrast with no missing value at all leaves every treatment reading
        # exactly the same numbers; giving them the same seed keeps the rows
        # identical instead of showing bootstrap noise as if it were an effect.
        if differences == available_differences:
            seed_text = available_seed
            seed_source = "published" if published_key else "missingness"
            if treatment != AVAILABLE_CASE:
                seed_source += " (item set identical to available-case)"
        else:
            seed_text, seed_source = missingness_seed(
                split=split, model_id=model_id, contrast_id=contrast.contrast_id,
                treatment=treatment), "missingness"
        result = bootstrap_contrast(differences, seed_text)
        outcomes.append(TreatmentOutcome(
            treatment, treated_fill, reference_fill, seed_text, seed_source, result,
            excludes_zero_in_direction(result, contrast.direction)))
    outcomes = tuple(outcomes)

    tipping = tipping_point(
        pairs,
        missingness_seed(split=split, model_id=model_id, contrast_id=contrast.contrast_id, treatment=DELTA),
        direction=contrast.direction)

    by_name = {item.treatment: item for item in outcomes}
    verdict = classify(contrast.direction, by_name.get(AVAILABLE_CASE), by_name.get(ZERO_IMPUTATION),
                       by_name.get(MANSKI_LOWER), by_name.get(MANSKI_UPPER))
    published = _check_published(published, by_name.get(AVAILABLE_CASE), tolerance)
    return ContrastOutcome(
        model_id, role or "extension", split, contrast.contrast_id, contrast.label, contrast.stratum,
        contrast.direction, counts, support_min, support_max, support_n,
        min(observed) if observed else None, max(observed) if observed else None,
        mean(reference_values) if reference_values else None,
        outcomes, tipping, verdict, published,
    )


def _check_published(published: PublishedReference | None, available: TreatmentOutcome | None,
                     tolerance: float) -> PublishedReference | None:
    """Attach the available-case discrepancy to the published reference, if any."""
    if published is None:
        return None
    if available is None or available.result.estimate is None or published.estimate is None:
        return PublishedReference(published.source, published.key, published.estimate,
                                  published.ci95_lower, published.ci95_upper, published.n_items,
                                  None, False)
    differences = [abs(available.result.estimate - published.estimate)]
    for mine, theirs in ((available.result.ci95_lower, published.ci95_lower),
                         (available.result.ci95_upper, published.ci95_upper)):
        if mine is not None and theirs is not None:
            differences.append(abs(mine - theirs))
    worst = max(differences)
    same_n = published.n_items is None or published.n_items == available.result.n_items
    return PublishedReference(published.source, published.key, published.estimate,
                              published.ci95_lower, published.ci95_upper, published.n_items,
                              worst, bool(worst <= tolerance and same_n))


# --------------------------------------------------------------------------
# Published references
# --------------------------------------------------------------------------

def load_discovery_published(path: str | Path) -> Mapping[tuple[str, str], PublishedReference]:
    """``(model_id, contrast_id) -> published discovery estimate`` from the exploratory table."""
    path = Path(path)
    if not path.exists():
        return _freeze({})
    wanted = {item.discovery_key: item.contrast_id for item in CONTRASTS if item.discovery_key}
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != "m1":
                continue
            contrast_id = wanted.get((row["contrast"], row["stratum"]))
            if contrast_id is None:
                continue
            out[(row["model_id"], contrast_id)] = PublishedReference(
                str(path).replace("\\", "/"),
                "%s|%s|m1|%s" % (row["model_id"], row["contrast"], row["stratum"]),
                _number(row.get("mean_difference")), _number(row.get("ci95_lower")),
                _number(row.get("ci95_upper")), _integer(row.get("n_items")))
    return _freeze(out)


def load_holdout_published(path: str | Path, models: Mapping[str, str]
                           ) -> Mapping[tuple[str, str], PublishedReference]:
    """``(model_id, contrast_id) -> published holdout estimate`` from ``hypotheses.csv``."""
    path = Path(path)
    if not path.exists():
        return _freeze({})
    wanted = {}
    for item in CONTRASTS:
        for role, hypothesis_id in item.holdout_keys:
            if role in models:
                wanted[hypothesis_id] = (models[role], item.contrast_id)
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = wanted.get(row.get("hypothesis_id", ""))
            if target is None or row.get("outcome") != "m1":
                continue
            out[target] = PublishedReference(
                str(path).replace("\\", "/"), row["hypothesis_id"], _number(row.get("estimate")),
                _number(row.get("ci95_lower")), _number(row.get("ci95_upper")),
                _integer(row.get("n_items")))
    return _freeze(out)


def load_extension_published(path: str | Path, model_id: str
                             ) -> Mapping[tuple[str, str, str], PublishedReference]:
    """``(split, model_id, contrast_id) -> estimate`` from a committed ``extension.json``."""
    path = Path(path)
    if not path.exists():
        return _freeze({})
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_hypothesis = {item.holdout_hypothesis("primary"): item.contrast_id
                     for item in CONTRASTS if item.holdout_hypothesis("primary")}
    out = {}
    for comparison in (payload.get("result") or {}).get("comparisons", ()):
        contrast_id = by_hypothesis.get(comparison.get("hypothesis_id"))
        if contrast_id is None or comparison.get("outcome") != "m1":
            continue
        for split in ("discovery", "holdout"):
            entry = comparison.get(split)
            result = (entry or {}).get("result") or {}
            if result.get("estimate") is None:
                continue
            out[(split, model_id, contrast_id)] = PublishedReference(
                str(path).replace("\\", "/"), "%s|%s" % (split, comparison["hypothesis_id"]),
                result.get("estimate"), result.get("ci95_lower"), result.get("ci95_upper"),
                result.get("n_items"))
    return _freeze(out)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingnessReport:
    label: str
    models: Mapping[str, str]
    sources: Mapping[str, str]
    amendment_note: Mapping[str, str]
    outcomes: tuple[ContrastOutcome, ...]
    cells: tuple[CellMissingness, ...] = ()

    def to_dict(self):
        return _jsonable(self)


def run_missingness(
    rows_by_split: Mapping[str, Sequence[MetricRow]],
    *,
    models: Mapping[str, str],
    excluded_by: Mapping[tuple[str, str], Iterable[str]] | None = None,
    published: Mapping[tuple[str, str, str], PublishedReference] | None = None,
    contrasts: Sequence[Contrast] = CONTRASTS,
    label: str = "M1 missing-data sensitivity analysis",
    sources: Mapping[str, str] | None = None,
    amendment_note: Mapping[str, str] | None = None,
) -> MissingnessReport:
    """Every contrast x split x model, in the reporting order of the tables."""
    excluded_by = dict(excluded_by or {})
    published = dict(published or {})
    outcomes = []
    cells = []
    for split in sorted(rows_by_split):
        rows = tuple(rows_by_split[split])
        for role in sorted(models, key=lambda name: ("primary", "control", "extension").index(name)
                           if name in ("primary", "control", "extension") else 99):
            model_id = models[role]
            if not any(row.model_id == model_id and row.split == split for row in rows):
                continue
            for contrast in contrasts:
                outcomes.append(analyse_contrast(
                    rows, contrast, model_id=model_id, split=split, role=role,
                    excluded=excluded_by.get((model_id, split), ()),
                    published=published.get((split, model_id, contrast.contrast_id)),
                ))
            cells.extend(cell_missingness(rows, model_id=model_id, split=split,
                                          excluded=excluded_by.get((model_id, split), ())))
    return MissingnessReport(label, _freeze(models), _freeze(sources or {}),
                             _freeze(amendment_note or {}), tuple(outcomes), tuple(cells))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

TREATMENT_LABEL = {
    AVAILABLE_CASE: "available-case (published)",
    ZERO_IMPUTATION: "zero-imputation (0 nats)",
    MANSKI_LOWER: "bound: most negative",
    MANSKI_UPPER: "bound: most positive",
}


def _escape(text) -> str:
    """A stratum name like ``easy | neutral`` would otherwise split a table cell."""
    return str(text).replace("|", "\\|")


def _interval(result: BootstrapResult) -> str:
    if result.estimate is None:
        return "unavailable (`%s`)" % (result.unavailable_reason or "no_data")
    if result.ci95_lower is None:
        return "%.3f (no CI: `%s`)" % (result.estimate, result.unavailable_reason or "")
    return "%.3f [%.3f, %.3f]" % (result.estimate, result.ci95_lower, result.ci95_upper)


def _delta_text(outcome: ContrastOutcome) -> str:
    tipping = outcome.tipping
    if tipping.delta is None:
        return "n/a (`%s`)" % (tipping.reason or "unavailable")
    if tipping.reason == "ci_already_includes_zero_at_zero_imputation":
        return "0.000 (already includes 0)"
    return "%.3f" % tipping.delta


def per_item_rows(report: MissingnessReport, rows_by_split: Mapping[str, Sequence[MetricRow]]
                  ) -> tuple[dict[str, Any], ...]:
    """One row per (model, split, contrast, item): the whole analysis, item by item."""
    out = []
    for outcome in report.outcomes:
        contrast = CONTRASTS_BY_ID[outcome.contrast_id]
        rows = tuple(rows_by_split.get(outcome.split, ()))
        index = build_m1_index(rows, model_id=outcome.model_id, split=outcome.split)
        support = (None if outcome.support_min is None else (outcome.support_min, outcome.support_max))
        for pair in build_pairs(index, contrast):
            record = {
                "model_id": outcome.model_id, "role": outcome.role, "split": outcome.split,
                "contrast_id": outcome.contrast_id, "stratum": outcome.stratum,
                "task_id": pair.task_id, "difficulty": pair.difficulty,
                "treated_cell": contrast.treated.cell_id(pair.difficulty),
                "treated_turn": contrast.treated.turn,
                "reference_cell": contrast.reference.cell_id(pair.difficulty),
                "reference_turn": contrast.reference.turn,
                "treated_m1": pair.treated, "reference_m1": pair.reference,
                "treated_missing_kind": pair.treated_missing,
                "reference_missing_kind": pair.reference_missing,
            }
            for treatment in TREATMENTS:
                if treatment in (MANSKI_LOWER, MANSKI_UPPER) and support is None:
                    record["difference_" + treatment] = None
                    continue
                treated_fill, reference_fill = treatment_fills(treatment, support=support)
                differences = paired_differences([pair], treated_fill=treated_fill,
                                                 reference_fill=reference_fill)
                record["difference_" + treatment] = differences[0][1] if differences else None
            out.append(record)
    return tuple(out)


PER_ITEM_COLUMNS = (
    "model_id", "role", "split", "contrast_id", "stratum", "task_id", "difficulty",
    "treated_cell", "treated_turn", "reference_cell", "reference_turn",
    "treated_m1", "reference_m1", "treated_missing_kind", "reference_missing_kind",
    "difference_available_case", "difference_zero_imputation",
    "difference_manski_lower", "difference_manski_upper",
)


VERDICT_ORDER = ("robust_bounded", "robust_unbounded", "weakened", "flips",
                 "available_case_null", "available_case_opposite", "unavailable")


def _headline(report: MissingnessReport, role: str, split: str) -> str | None:
    """One generated sentence per model role and split; no number is typed by hand."""
    subset = [item for item in report.outcomes if item.role == role and item.split == split]
    if not subset:
        return None
    model_id = subset[0].model_id
    survives = [item.contrast_id for item in subset
                if item.verdict in ("robust_bounded", "robust_unbounded")]
    bounded = [item.contrast_id for item in subset if item.verdict == "robust_bounded"]
    fragile = [item.contrast_id for item in subset if item.verdict in ("weakened", "flips")]
    absent = [item.contrast_id for item in subset
              if item.verdict in ("available_case_null", "available_case_opposite")]
    parts = ["`%s`, %s: %d of %d contrasts keep a CI excluding 0 in the predicted direction under "
             "available-case *and* zero-imputation (%s)"
             % (model_id, split, len(survives), len(subset), ", ".join(survives) or "none")]
    parts.append("of those, %s also survive%s both worst-case bounds (%s)"
                 % (len(bounded), "" if len(bounded) != 1 else "s", ", ".join(bounded) or "none"))
    if fragile:
        parts.append("**%s depend%s on the missing values** (the zero-imputation CI no longer "
                     "excludes 0)" % (", ".join(fragile), "" if len(fragile) != 1 else "s"))
    if absent:
        parts.append("%s had no available-case effect to begin with" % ", ".join(absent))
    return "- " + "; ".join(parts) + "."


def _reading(report: MissingnessReport) -> list[str]:
    """The plain-English reading: robust, flipped, and missing-value dependent."""
    lines = []
    for split in ("holdout", "discovery"):
        for role in ("primary", "control", "extension"):
            sentence = _headline(report, role, split)
            if sentence:
                lines.append(sentence)
    buckets: dict[str, dict[str, list[str]]] = {}
    for outcome in report.outcomes:
        buckets.setdefault(outcome.verdict, {}).setdefault(outcome.model_id, []).append(
            "%s (%s)" % (outcome.contrast_id, outcome.split))
    if lines:
        lines += ["", "By verdict:", ""]
    for verdict in VERDICT_ORDER:
        by_model = buckets.get(verdict)
        if not by_model:
            continue
        lines.append("- **%s**" % VERDICTS[verdict])
        for model_id in sorted(by_model):
            lines.append("  - `%s`: %s" % (model_id, ", ".join(by_model[model_id])))
    return lines


def render_markdown(report: MissingnessReport) -> str:
    lines = [
        "# %s" % report.label,
        "",
        "**Sensitivity analysis, not a confirmatory result.** The published available-case",
        "estimates in `results/summaries/phase1/exploratory/` and `results/summaries/phase2/`",
        "are unchanged; this document reports what happens to them under three alternative",
        "treatments of the missing M1 values, plus the tipping point.",
        "",
        "## Reading",
        "",
    ]
    lines += _reading(report) or ["- no contrast could be analysed."]
    lines += [
        "",
        "A contrast counts as *robust to all four treatments* only when the available-case CI,",
        "the zero-imputation CI and **both** worst-case bound CIs exclude 0 in the predicted",
        "direction: the sign is then determined by the data for any imputation whose values lie",
        "inside the observed neutral-accurate support. *Bounds uninformative* means the",
        "imputation-based treatments agree but an adversarial filling inside that support can",
        "still reach 0 -- which is the expected outcome whenever a cell loses several items.",
        "",
        "## What was done",
        "",
        "- Outcome: M1 (canonical-answer logit margin, nats), greedy sample 0, frozen parser.",
        "- Pairing and CIs: the published item pairing and the 2,000-resample item-clustered",
        "  bootstrap percentile CI of `src.confirm.bootstrap_contrast`, unchanged.",
        "- The available-case row reuses the *published* bootstrap seed",
        "  (`DGS-AC1-EXPLORATORY-v1|...` on discovery, `DGS-AC1-CONFIRM-v3|...` on the holdout),",
        "  so it reproduces the committed CI exactly where one exists; the other treatments use",
        "  `%s|<split>|<model>|<contrast>|<treatment>` because their item sets differ." % MISSINGNESS_KEY,
        "- Item set: every item whose **two endpoint rows both exist**, whether or not their M1",
        "  parsed. An item missing an endpoint row altogether cannot be imputed and is counted",
        "  separately (`endpoint absent`).",
        "- A missing M1 is either a **non-answer** (`m1_invalid_final_answer`: no parseable",
        "  `Answer: X`) or a **candidate-absent** truncation (`m1_candidate_absent_*`: the",
        "  response committed to a letter but one of the four options fell outside the stored",
        "  top-20 logprobs). Both are missing for M1 and both are imputed here; they are counted",
        "  separately below because only the first is a non-answer.",
    ]
    for key in sorted(report.amendment_note):
        lines.append("- %s" % report.amendment_note[key])
    if report.sources:
        lines += ["", "Sources:", ""]
        for key in sorted(report.sources):
            lines.append("- %s: `%s`" % (key, report.sources[key]))
    lines += ["", "## Reproduction check: available-case vs the published estimate", ""]
    checked = [item for item in report.outcomes if item.published is not None]
    if checked:
        worst = max((item.published.max_abs_difference or 0.0) for item in checked)
        lines += [
            "%d of %d published M1 estimates have a counterpart here; %d reproduce point estimate,"
            % (sum(1 for item in checked if item.published.reproduced), len(checked),
               sum(1 for item in checked if item.published.reproduced)),
            "both CI bounds and item count, with a largest absolute discrepancy of %.2e nats." % worst,
            "",
        ]
    lines += _reproduction_table(report)
    for role, model_id in sorted(report.models.items(),
                                 key=lambda item: ("primary", "control", "extension").index(item[0])
                                 if item[0] in ("primary", "control", "extension") else 99):
        subset = [item for item in report.outcomes if item.model_id == model_id]
        if not subset:
            continue
        lines += ["", "## %s model: `%s`" % (role, model_id), "",
                  "### Estimates under each treatment (M1 nats, 95% item-bootstrap CI)", ""]
        lines += _treatment_table(subset)
        lines += ["", "### Missing values entering each contrast", ""]
        lines += _counts_table(subset)
        cells = [item for item in report.cells if item.model_id == model_id]
        if cells:
            lines += ["", "### Non-answers per cell (the endpoints these contrasts read)", ""]
            lines += _cell_table(cells)
        lines += ["", "### Tipping point", ""]
        lines += _tipping_table(subset)
    lines += ["", "## What this analysis does not settle", "",
              "- Zero-imputation is an *assumption*, not a measurement: a non-answer carries no",
              "  margin at all, and 0 nats is the indifference point, not an observed value.",
              "- The worst-case bounds are worst-case only *inside the observed neutral-accurate",
              "  support*. A missing trial whose true margin lay outside that range would sit",
              "  outside the interval, and the interval says nothing about why the value is",
              "  missing -- the MNAR mechanism itself is reported (H9), not modelled.",
              "- When 0 nats lies inside that support the zero-imputation estimate is bracketed by",
              "  the two bounds by construction, so agreement between them is not independent",
              "  evidence. %s" % _zero_in_support(report),
              "- `m1_candidate_absent_*` endpoints are imputed on the same footing as non-answers",
              "  even though the response did commit to a letter; the counts table shows how many",
              "  of each entered every contrast so the two can be read apart.",
              "- Nothing here is confirmatory, and no published estimate is amended by it.",
              ""]
    return "\n".join(lines)


def _zero_in_support(report: MissingnessReport) -> str:
    """Whether the bracketing actually holds here, checked rather than assumed."""
    supports = {(item.model_id, item.split): (item.support_min, item.support_max)
                for item in report.outcomes if item.support_min is not None}
    if not supports:
        return "No support could be computed."
    outside = sorted("%s/%s [%.2f, %.2f]" % (model_id.split("/")[-1], split, low, high)
                     for (model_id, split), (low, high) in supports.items()
                     if not low <= 0.0 <= high)
    if not outside:
        return "Every model x split support here contains 0, so the bracketing holds throughout."
    return "It does not hold for %s, where 0 is outside the support." % "; ".join(outside)


def _reproduction_table(report: MissingnessReport) -> list[str]:
    lines = [
        "| contrast | split | model | published [95% CI] | available-case here [95% CI] | n pub / here | max abs diff | reproduced |",
        "| --- | --- | --- | --- | --- | ---: | ---: | :---: |",
    ]
    any_row = False
    for outcome in report.outcomes:
        if outcome.published is None:
            continue
        any_row = True
        available = outcome.treatment(AVAILABLE_CASE)
        published = outcome.published
        lines.append("| %s (%s) | %s | `%s` | %s | %s | %s / %d | %s | %s |" % (
            outcome.contrast_id, _escape(outcome.stratum), outcome.split, outcome.model_id,
            "%.3f [%.3f, %.3f]" % (published.estimate, published.ci95_lower, published.ci95_upper)
            if published.estimate is not None and published.ci95_lower is not None else "n/a",
            _interval(available.result) if available else "n/a",
            "n/a" if published.n_items is None else str(published.n_items),
            available.result.n_items if available else 0,
            "%.2e" % published.max_abs_difference if published.max_abs_difference is not None else "n/a",
            "**yes**" if published.reproduced else "**NO**",
        ))
    if not any_row:
        lines.append("| - | - | - | no published counterpart | - | - | - | - |")
    return lines


def _treatment_table(outcomes: Sequence[ContrastOutcome]) -> list[str]:
    lines = [
        "| contrast | stratum | split | treatment | estimate [95% CI] | n items | CI excludes 0 |",
        "| --- | --- | --- | --- | --- | ---: | :---: |",
    ]
    for outcome in outcomes:
        for item in outcome.treatments:
            lines.append("| %s | %s | %s | %s | %s | %d | %s |" % (
                outcome.contrast_id, _escape(outcome.stratum), outcome.split,
                TREATMENT_LABEL.get(item.treatment, item.treatment), _interval(item.result),
                item.result.n_items, "yes" if item.excludes_zero else "no"))
    return lines


def _counts_table(outcomes: Sequence[ContrastOutcome]) -> list[str]:
    lines = [
        "| contrast | split | items in stratum | pairable | endpoint absent | available | treated missing (non-answer / candidate-absent) | reference missing (non-answer / candidate-absent) | both |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for outcome in outcomes:
        counts = outcome.counts
        lines.append("| %s | %s | %d | %d | %d | %d | %d (%d / %d) | %d (%d / %d) | %d |" % (
            outcome.contrast_id, outcome.split, counts.n_items_in_stratum, counts.n_pairable,
            counts.n_endpoint_absent, counts.n_available,
            counts.n_treated_missing, counts.treated_non_answer, counts.treated_candidate_absent,
            counts.n_reference_missing, counts.reference_non_answer, counts.reference_candidate_absent,
            counts.n_both_missing))
    return lines


def _cell_table(cells: Sequence[CellMissingness]) -> list[str]:
    lines = [
        "| split | cell | endpoint | endpoints | M1 observed | non-answer | candidate-absent | other | mean M1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(cells, key=lambda row: (row.split, row.cell_id, row.turn_label)):
        lines.append("| %s | `%s` | %s | %d | %d | %d | %d | %d | %s |" % (
            item.split, item.cell_id, item.turn_label, item.n_endpoints, item.n_observed,
            item.n_non_answer, item.n_candidate_absent, item.n_other,
            "%.2f" % item.mean_m1 if item.mean_m1 is not None else "-"))
    return lines


def _tipping_table(outcomes: Sequence[ContrastOutcome]) -> list[str]:
    lines = [
        "| contrast | split | delta (nats) | missing treated values | observed M1 in these cells | neutral-accurate support | reference-cell mean |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for outcome in outcomes:
        lines.append("| %s | %s | %s | %d | %s | %s | %s |" % (
            outcome.contrast_id, outcome.split, _delta_text(outcome),
            outcome.tipping.n_missing_treated,
            "[%.2f, %.2f]" % (outcome.observed_min, outcome.observed_max)
            if outcome.observed_min is not None else "n/a",
            "[%.2f, %.2f] (n = %d)" % (outcome.support_min, outcome.support_max, outcome.support_n)
            if outcome.support_min is not None else "n/a",
            "%.2f" % outcome.reference_mean if outcome.reference_mean is not None else "n/a"))
    lines += [
        "",
        "`delta` is the constant margin every missing treated-cell trial would have to carry",
        "(with missing reference-cell trials at 0) for the item-paired 95% CI to include 0.",
        "Read it against the observed M1 range in the same cells.",
    ]
    return lines
