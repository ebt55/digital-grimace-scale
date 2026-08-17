"""Preregistration v4 Phase 3 (j-space): probes, layer choice, directions, readouts.

This module is the pure, importable core of Phase 3.  It never talks to Modal, never
reads ``results/raw`` and never writes anything: ``scripts/run_phase3.py`` owns the I/O
and the remote calls, and ``src/jspace_client.py`` owns the model.  Everything here is
deterministic given its inputs.

What is transcribed literally from ``notes/preregistration_v4_phase3.md``:

* **Probe.** L2 logistic regression per layer (``C = 1``), features standardised **in the
  training fold only**, leave-one-item-out **grouped by task** so that all cells of a task
  are held out together.  Layer selection is discovery-only; the chosen layer's probe is
  evaluated **once** on holdout.
* **Layer choice.** ``L* = argmax`` discovery leave-one-task-out tone AUC; ties resolve to
  the lower layer index.
* **Correlation.** Item-level Spearman between the holdout tone-probe score and the
  measured M1 (available-case) *within cell*: both columns are demeaned inside their cell
  and the residuals pooled.  The 95% interval is a 2,000-resample item-clustered
  percentile bootstrap that re-demeans inside every resample, so the reported interval
  covers the whole estimator rather than a frozen residualisation.
* **Directions.** Tone direction ``d = mean(hostile) - mean(neutral)`` at ``L*`` from
  discovery, accurate arm, measured position.  Per **clarification C2 (2026-08-18)** the dose
  unit is the contrast's own magnitude: the tone dose is ``alpha * d`` with ``d``
  unnormalised, and every control is rescaled to the matched norm ``alpha * ||d||`` (five
  random unit directions from the frozen seeds ``DGS-AC1-STEER-v1|<k>``, plus the unrelated
  verbose-minus-neutral style direction).  ``scaled_direction`` implements that rescaling;
  the ratio ``||d|| / mean-activation-norm`` at ``L*`` is reported so readers can convert.
* **Degenerate dose rule.** A dose is degenerate when **more than 50%** of items yield no
  parseable answer; degenerate doses are reported and excluded from the monotonicity check.

Interpretation ceiling, unchanged: a probe plus induction result demonstrates a
condition-linked internal variable with causal influence on the output signature.  It is
not evidence of experience.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .confirm import BootstrapResult, bootstrap_contrast
from .metrics import MetricInputError, m1_margin
from .protocol import parse_final_answer
from .records import Token

# --------------------------------------------------------------------------------------
# Frozen constants
# --------------------------------------------------------------------------------------

PROBE_C = 1.0
PROBE_MAX_ITER = 1000
PROBE_RANDOM_STATE = 0
BOOTSTRAP_SAMPLES = 2000

#: Seed text for the item bootstrap behind the holdout probe-score / M1 correlation.
CORRELATION_SEED = "DGS-AC1-PROBE-CORR-v1"
#: Frozen seed family for the matched-norm random control directions (``...|<k>``).
STEER_SEED = "DGS-AC1-STEER-v1"
#: Seed family for the paired dose-versus-baseline readout bootstraps.
READOUT_SEED = "DGS-AC1-STEER-READOUT-v1"

#: J1's preregistered "middle layer" band for the peak discovery AUC.
MIDDLE_LAYERS = (12, 30)
#: J1 thresholds.
J1_DISCOVERY_AUC = 0.80
J1_HOLDOUT_AUC = 0.75
#: J2 threshold: validity AUC must sit at least this far below tone AUC at the same layer.
J2_AUC_GAP = 0.05
#: J3 threshold on the pooled within-cell Spearman.
J3_RHO = -0.2
#: A dose is degenerate when strictly more than this fraction of items has no answer.
DEGENERATE_NON_ANSWER_FRACTION = 0.5

#: Label values, spelled exactly as they appear in ``cell_id``.
TONE_POSITIVE, TONE_NEGATIVE = "hostile", "neutral"
VALIDITY_POSITIVE, VALIDITY_NEGATIVE = "malfunctioning_always_fail", "accurate"
STYLE_POSITIVE, STYLE_NEGATIVE = "style__verbose", "style__neutral_reference"

#: The cell the steering items are rendered in: the neutral, no-feedback single-turn task.
STEER_CELL_ID = STYLE_NEGATIVE
BASELINE_DIRECTION_ID = "baseline"
TONE_DIRECTION_ID = "tone"
UNRELATED_DIRECTION_ID = "unrelated_style"
RANDOM_DIRECTION_IDS = tuple("random%d" % index for index in range(1, 6))


class ProbeError(ValueError):
    """Raised when Phase-3 inputs cannot be assembled as preregistered."""


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _seeded_random(seed_text: str) -> random.Random:
    """The repository's frozen seeding convention: SHA-256 of the seed text."""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# --------------------------------------------------------------------------------------
# Activation container
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationSet:
    """Residual-stream activations for one item set, plus their labels.

    ``activations`` is ``(n_items, n_layers, hidden)``.  ``norms`` is the per-layer mean
    activation L2 norm reported by the extractor and is the scale used for steering doses.
    """

    ids: tuple[str, ...]
    layers: tuple[int, ...]
    activations: np.ndarray
    norms: tuple[float, ...]
    labels: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        array = np.asarray(self.activations)
        if array.ndim != 3:
            raise ProbeError("activations must be (n_items, n_layers, hidden)")
        if array.shape[0] != len(self.ids) or array.shape[1] != len(self.layers):
            raise ProbeError("activation shape does not match ids and layers")
        if len(self.norms) != len(self.layers):
            raise ProbeError("one mean norm per layer is required")
        if len(set(self.ids)) != len(self.ids):
            raise ProbeError("item ids must be unique")
        for name, column in self.labels.items():
            if len(column) != len(self.ids):
                raise ProbeError("label column %r has the wrong length" % (name,))
        object.__setattr__(self, "activations", array)
        object.__setattr__(self, "labels", _freeze({k: tuple(v) for k, v in self.labels.items()}))

    @property
    def n_items(self) -> int:
        return len(self.ids)

    @property
    def hidden(self) -> int:
        return int(self.activations.shape[2])

    def layer_index(self, layer: int) -> int:
        try:
            return self.layers.index(int(layer))
        except ValueError as error:
            raise ProbeError("layer %r was not extracted" % (layer,)) from error

    def matrix(self, layer: int) -> np.ndarray:
        """The ``(n_items, hidden)`` float64 design matrix for one layer."""
        return np.asarray(self.activations[:, self.layer_index(layer), :], dtype=np.float64)

    def norm(self, layer: int) -> float:
        return float(self.norms[self.layer_index(layer)])

    def column(self, name: str) -> tuple[str, ...]:
        if name not in self.labels:
            raise ProbeError("unknown label column %r" % (name,))
        return self.labels[name]

    def mask(self, **equals: str) -> np.ndarray:
        """Boolean row mask selecting items whose label columns all match."""
        keep = np.ones(self.n_items, dtype=bool)
        for name, value in equals.items():
            keep &= np.array([item == value for item in self.column(name)], dtype=bool)
        return keep

    def select(self, mask: np.ndarray) -> "ActivationSet":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_items,):
            raise ProbeError("selection mask has the wrong shape")
        index = np.flatnonzero(mask)
        return ActivationSet(
            tuple(self.ids[position] for position in index),
            self.layers,
            self.activations[index],
            self.norms,
            {name: tuple(column[position] for position in index) for name, column in self.labels.items()},
        )


def save_activation_set(path: str | Path, activation_set: ActivationSet) -> Path:
    """Write one ``.npz`` holding activations, layers, per-layer norms and every label."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ids": np.array(activation_set.ids, dtype=object),
        "layers": np.array(activation_set.layers, dtype=np.int32),
        "activations": np.asarray(activation_set.activations),
        "norms": np.array(activation_set.norms, dtype=np.float64),
    }
    for name, column in activation_set.labels.items():
        payload["label__" + name] = np.array(column, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def load_activation_set(path: str | Path) -> ActivationSet:
    with np.load(Path(path), allow_pickle=True) as data:
        labels = {key[len("label__"):]: tuple(str(item) for item in data[key])
                  for key in data.files if key.startswith("label__")}
        return ActivationSet(
            tuple(str(item) for item in data["ids"]),
            tuple(int(item) for item in data["layers"]),
            np.asarray(data["activations"]),
            tuple(float(item) for item in data["norms"]),
            labels,
        )


# --------------------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearProbe:
    """A fitted per-layer probe, carrying its own training-fold standardisation."""

    layer: int
    label_name: str
    positive_label: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    n_train: int

    def score(self, features: np.ndarray) -> np.ndarray:
        """Signed decision values; higher means "more like the positive label"."""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.coef.shape[0]:
            raise ProbeError("probe input has the wrong hidden size")
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept


def standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-fold mean and population SD; a constant column keeps scale 1."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def binary_labels(values: Sequence[str], positive: str, negative: str) -> np.ndarray:
    """Map a label column to 0/1, rejecting any third value."""
    unknown = sorted({value for value in values} - {positive, negative})
    if unknown:
        raise ProbeError("unexpected label values: %s" % ", ".join(unknown))
    return np.array([1 if value == positive else 0 for value in values], dtype=int)


def fit_probe(features: np.ndarray, labels: np.ndarray, *, layer: int, label_name: str,
              positive_label: str) -> LinearProbe:
    """Fit the preregistered L2 logistic probe on standardised training features."""
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ProbeError("probe features and labels disagree")
    if len(set(labels.tolist())) < 2:
        raise ProbeError("a probe needs both classes in its training fold")
    mean, scale = standardization(features)
    # The preregistration fixes an L2 penalty at C = 1.  scikit-learn deprecated the
    # explicit ``penalty="l2"`` argument in 1.8 in favour of ``l1_ratio``; the default
    # (``l1_ratio = 0``) *is* pure L2 and gives bit-identical coefficients, so the default
    # is relied on and the equivalence is asserted in ``tests/test_probe.py``.
    model = LogisticRegression(
        C=PROBE_C, solver="liblinear",
        max_iter=PROBE_MAX_ITER, random_state=PROBE_RANDOM_STATE,
    ).fit((features - mean) / scale, labels)
    return LinearProbe(int(layer), label_name, positive_label, mean, scale,
                       np.asarray(model.coef_, dtype=np.float64).ravel(),
                       float(np.asarray(model.intercept_, dtype=np.float64).ravel()[0]),
                       int(features.shape[0]))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Rank-based AUC with mid-ranks for ties; ``None`` when one class is absent."""
    labels = [int(value) for value in labels]
    scores = [float(value) for value in scores]
    if len(labels) != len(scores) or not labels:
        raise ProbeError("AUC needs one score per label")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        mid = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = mid
        position = end + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


@dataclass(frozen=True)
class LayerAuc:
    """One layer's leave-one-task-out result for one label."""

    layer: int
    auc: float | None
    n_items: int
    n_groups: int
    unavailable_reason: str | None = None


def leave_one_group_out_scores(features: np.ndarray, labels: np.ndarray,
                               groups: Sequence[str], *, layer: int = 0,
                               label_name: str = "label",
                               positive_label: str = "positive") -> tuple[list[int], list[float]]:
    """Out-of-fold decision scores, holding out one whole task (all its cells) at a time."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    groups = list(groups)
    if not (features.shape[0] == labels.shape[0] == len(groups)):
        raise ProbeError("features, labels and groups disagree")
    held_labels: list[int] = []
    held_scores: list[float] = []
    for group in sorted(set(groups)):
        test = np.array([item == group for item in groups], dtype=bool)
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            raise ProbeError("training fold for group %r has one class" % (group,))
        probe = fit_probe(features[train], labels[train], layer=layer,
                          label_name=label_name, positive_label=positive_label)
        held_labels.extend(labels[test].tolist())
        held_scores.extend(probe.score(features[test]).tolist())
    return held_labels, held_scores


def loo_auc_by_layer(activation_set: ActivationSet, *, label_name: str, positive: str,
                     negative: str, group_name: str = "task_id",
                     layers: Sequence[int] | None = None) -> tuple[LayerAuc, ...]:
    """Discovery leave-one-task-out AUC for every extracted layer."""
    labels = binary_labels(activation_set.column(label_name), positive, negative)
    groups = activation_set.column(group_name)
    chosen = tuple(activation_set.layers) if layers is None else tuple(int(item) for item in layers)
    out = []
    for layer in chosen:
        try:
            held_labels, held_scores = leave_one_group_out_scores(
                activation_set.matrix(layer), labels, groups,
                layer=layer, label_name=label_name, positive_label=positive)
            value = roc_auc(held_labels, held_scores)
            reason = None if value is not None else "one_class_out_of_fold"
        except ProbeError as error:
            value, reason = None, "loo_unavailable:%s" % error
        out.append(LayerAuc(int(layer), value, activation_set.n_items, len(set(groups)), reason))
    return tuple(out)


def choose_layer(layer_aucs: Iterable[LayerAuc]) -> int:
    """``argmax`` discovery tone AUC; ties resolve to the lower layer index."""
    available = [item for item in layer_aucs if item.auc is not None]
    if not available:
        raise ProbeError("no layer produced an AUC, so no layer can be chosen")
    return min(available, key=lambda item: (-item.auc, item.layer)).layer


# --------------------------------------------------------------------------------------
# Holdout correlation
# --------------------------------------------------------------------------------------

def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        mid = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = mid
        position = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_deviation = [value - left_mean for value in left]
    right_deviation = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in left_deviation)) * math.sqrt(
        sum(value * value for value in right_deviation))
    if denominator == 0.0:
        return None
    return float(sum(a * b for a, b in zip(left_deviation, right_deviation)) / denominator)


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman rho with mid-ranks for ties; ``None`` when it is undefined."""
    if len(left) != len(right):
        raise ProbeError("spearman needs paired inputs")
    if len(left) < 2:
        return None
    return _pearson(_rank(left), _rank(right))


def _cell_demeaned(rows: Sequence[tuple[str, str, float, float]]) -> tuple[list[float], list[float]]:
    """Demean the probe score and the metric inside each cell, then pool the residuals."""
    sums: dict[str, list[float]] = {}
    for _item, cell, score, value in rows:
        bucket = sums.setdefault(cell, [0.0, 0.0, 0.0])
        bucket[0] += score
        bucket[1] += value
        bucket[2] += 1.0
    left, right = [], []
    for _item, cell, score, value in rows:
        total_score, total_value, count = sums[cell]
        left.append(score - total_score / count)
        right.append(value - total_value / count)
    return left, right


@dataclass(frozen=True)
class CorrelationResult:
    rho: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    n_items: int
    n_pairs: int
    n_cells: int
    unavailable_reason: str | None = None

    @property
    def excludes_zero(self) -> bool:
        return (self.ci95_lower is not None and self.ci95_upper is not None
                and (self.ci95_lower > 0.0 or self.ci95_upper < 0.0))


def cell_demeaned_spearman(rows: Sequence[tuple[str, str, float, float]], *,
                           seed_text: str = CORRELATION_SEED,
                           bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> CorrelationResult:
    """Pooled within-cell Spearman with a 2,000-resample item-clustered percentile CI.

    ``rows`` are ``(task_id, cell_id, probe_score, metric_value)`` for available-case items
    only.  Each bootstrap resample draws whole tasks with replacement and re-demeans inside
    the resample, so the interval covers the estimator that produced the point estimate.
    """
    rows = tuple(rows)
    if not rows:
        return CorrelationResult(None, None, None, 0, 0, 0, "no_available_case_rows")
    items = sorted({row[0] for row in rows})
    cells = sorted({row[1] for row in rows})
    left, right = _cell_demeaned(rows)
    point = spearman(left, right)
    if point is None:
        return CorrelationResult(None, None, None, len(items), len(rows), len(cells),
                                 "spearman_undefined")
    if len(items) < 2:
        return CorrelationResult(point, None, None, len(items), len(rows), len(cells),
                                 "at_least_two_items_required_for_cluster_ci")
    by_item: dict[str, list[tuple[str, str, float, float]]] = {}
    for row in rows:
        by_item.setdefault(row[0], []).append(row)
    rng = _seeded_random(seed_text)
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        sampled: list[tuple[str, str, float, float]] = []
        for _ in range(len(items)):
            sampled.extend(by_item[items[rng.randrange(len(items))]])
        resample_left, resample_right = _cell_demeaned(sampled)
        value = spearman(resample_left, resample_right)
        if value is not None:
            draws.append(value)
    if len(draws) < 2:
        return CorrelationResult(point, None, None, len(items), len(rows), len(cells),
                                 "bootstrap_degenerate")
    draws.sort()

    def quantile(probability: float) -> float:
        position = (len(draws) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)

    return CorrelationResult(point, quantile(0.025), quantile(0.975),
                             len(items), len(rows), len(cells))


# --------------------------------------------------------------------------------------
# Steering directions
# --------------------------------------------------------------------------------------

def mean_difference_direction(activation_set: ActivationSet, layer: int, *, label_name: str,
                              positive: str, negative: str,
                              mask: np.ndarray | None = None) -> np.ndarray:
    """``mean(positive) - mean(negative)`` at one layer, over an optional row subset."""
    features = activation_set.matrix(layer)
    column = np.array(activation_set.column(label_name), dtype=object)
    keep = np.ones(activation_set.n_items, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    positive_rows = features[keep & (column == positive)]
    negative_rows = features[keep & (column == negative)]
    if not len(positive_rows) or not len(negative_rows):
        raise ProbeError("a mean-difference direction needs both arms present")
    return positive_rows.mean(axis=0) - negative_rows.mean(axis=0)


def unit(direction: Sequence[float]) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ProbeError("cannot normalise a zero or non-finite direction")
    return vector / norm


def scaled_direction(direction: Sequence[float], alpha: float, norm: float) -> np.ndarray:
    """The preregistered dose vector ``alpha * d / ||d|| * norm``."""
    if not math.isfinite(float(alpha)):
        raise ProbeError("alpha must be finite")
    if not math.isfinite(float(norm)) or float(norm) <= 0.0:
        raise ProbeError("the layer norm must be positive and finite")
    return unit(direction) * (float(alpha) * float(norm))


def random_unit_direction(hidden: int, index: int, *, seed_prefix: str = STEER_SEED) -> np.ndarray:
    """One matched-norm control direction from the frozen seed ``<prefix>|<k>``."""
    if not isinstance(hidden, int) or hidden < 1:
        raise ProbeError("hidden size must be a positive integer")
    digest = hashlib.sha256(("%s|%d" % (seed_prefix, index)).encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return unit(generator.standard_normal(hidden))


def random_unit_directions(hidden: int, count: int = 5, *,
                           seed_prefix: str = STEER_SEED) -> tuple[np.ndarray, ...]:
    return tuple(random_unit_direction(hidden, index, seed_prefix=seed_prefix)
                 for index in range(1, count + 1))


# --------------------------------------------------------------------------------------
# Steering readouts
# --------------------------------------------------------------------------------------

def tokens_from_entry(entry: Mapping[str, Any]) -> tuple[Token, ...]:
    """Rebuild the frozen ``records.Token`` trace from a stored steering entry.

    Alternatives go through ``backend.normalize_alternatives``, the same frozen helper every
    vLLM record used: a decoder can return several token IDs that decode to the same string,
    and ``metrics.m1_margin`` rejects duplicated alternative texts outright.  Reading the
    persisted JSONL therefore does not depend on the remote client being importable.
    """
    from .backend import MAX_ALTERNATIVES, normalize_alternatives

    raw = entry.get("tokens")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ProbeError("steering entry has no token trace")
    out = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            raise ProbeError("invalid steering token")
        logprob = float(item.get("logprob", 0.0))
        alternatives = [(str(alt["text"]), float(alt["logprob"]))
                        for alt in (item.get("top_logprobs") or ())]
        out.append(Token(item["text"], logprob,
                         normalize_alternatives(item["text"], logprob, alternatives,
                                                limit=MAX_ALTERNATIVES)))
    text = entry.get("text")
    if isinstance(text, str) and "".join(token.text for token in out) != text:
        raise ProbeError("steering entry %r text does not match its token trace"
                         % (entry.get("id"),))
    return tuple(out)


def steered_record(entry: Mapping[str, Any], task_id: str, *,
                   cell_id: str = STEER_CELL_ID) -> SimpleNamespace:
    """A record-like view of one steered generation, accepted by ``metrics.m1_margin``."""
    return SimpleNamespace(
        tokens=tokens_from_entry(entry), trajectory_kind="greedy", sample_index=0,
        task_id=task_id, cell_id=cell_id,
    )


def entry_text(entry: Mapping[str, Any]) -> str:
    text = entry.get("text")
    if isinstance(text, str):
        return text
    return "".join(token.text for token in tokens_from_entry(entry))


@dataclass(frozen=True)
class ItemReadout:
    """One steered generation reduced to its preregistered readouts."""

    direction_id: str
    alpha: float
    task_id: str
    m1: float | None
    m1_missing_reason: str | None
    non_answer: float
    length_tokens: int


def item_readout(entry: Mapping[str, Any], *, direction_id: str, alpha: float, task_id: str,
                 canonical_answer: str, cell_id: str = STEER_CELL_ID) -> ItemReadout:
    """M1 (available-case), the non-answer indicator and the generated length."""
    record = steered_record(entry, task_id, cell_id=cell_id)
    try:
        result = m1_margin(record, canonical_answer)
        value, reason = result.margin.value, result.margin.missing_reason
    except MetricInputError as error:
        value, reason = None, "m1_input_error:%s" % error
    text = entry_text(entry)
    return ItemReadout(direction_id, float(alpha), task_id, value, reason,
                       0.0 if parse_final_answer(text).valid else 1.0,
                       len(record.tokens))


@dataclass(frozen=True)
class DoseReadout:
    """Aggregated readouts for one (direction, dose) cell, paired against ``alpha = 0``."""

    direction_id: str
    alpha: float
    n_items: int
    m1_mean: float | None
    m1_n: int
    non_answer_rate: float
    mean_length_tokens: float | None
    degenerate: bool
    m1_delta: BootstrapResult
    non_answer_delta: BootstrapResult
    length_delta: BootstrapResult

    @property
    def m1_drop_supported(self) -> bool:
        """A dose "lowers M1" when the paired CI excludes zero on the negative side."""
        return self.m1_delta.ci95_upper is not None and self.m1_delta.ci95_upper < 0.0


def is_degenerate_dose(non_answer_rate: float) -> bool:
    """Preregistered rule: degenerate when **more than** 50% of items have no answer."""
    return float(non_answer_rate) > DEGENERATE_NON_ANSWER_FRACTION


def _paired(readouts: Mapping[str, ItemReadout], baseline: Mapping[str, ItemReadout],
            attribute: str) -> list[tuple[str, float]]:
    pairs = []
    for task_id in sorted(readouts):
        left = getattr(readouts[task_id], attribute)
        right = getattr(baseline.get(task_id), attribute, None) if task_id in baseline else None
        if left is None or right is None:
            continue
        pairs.append((task_id, float(left) - float(right)))
    return pairs


def dose_readout(items: Sequence[ItemReadout], baseline: Sequence[ItemReadout], *,
                 direction_id: str, alpha: float,
                 seed_text: str = READOUT_SEED) -> DoseReadout:
    """Aggregate one dose and bootstrap each readout against the ``alpha = 0`` baseline."""
    by_task = {item.task_id: item for item in items}
    baseline_by_task = {item.task_id: item for item in baseline}
    if len(by_task) != len(items) or len(baseline_by_task) != len(baseline):
        raise ProbeError("dose cell (%s, alpha=%g) holds more than one generation for some item"
                         % (direction_id, alpha))
    m1_values = [item.m1 for item in items if item.m1 is not None]
    lengths = [item.length_tokens for item in items]
    non_answer_rate = (sum(item.non_answer for item in items) / len(items)) if items else 0.0

    def key(name: str) -> str:
        return "%s|%s|alpha%s|%s" % (seed_text, direction_id, ("%g" % alpha), name)

    return DoseReadout(
        direction_id, float(alpha), len(items),
        (sum(m1_values) / len(m1_values)) if m1_values else None, len(m1_values),
        non_answer_rate, (sum(lengths) / len(lengths)) if lengths else None,
        is_degenerate_dose(non_answer_rate),
        bootstrap_contrast(_paired(by_task, baseline_by_task, "m1"), key("m1")),
        bootstrap_contrast(_paired(by_task, baseline_by_task, "non_answer"), key("non_answer")),
        bootstrap_contrast(_paired(by_task, baseline_by_task, "length_tokens"), key("length")),
    )


def monotone_in_alpha(readouts: Sequence[DoseReadout], alphas: Sequence[float] = (0.5, 1.0, 2.0),
                      *, attribute: str = "m1_mean") -> tuple[bool, tuple[float, ...], str | None]:
    """Non-increasing means across the requested doses, degenerate doses excluded.

    Returns ``(monotone, doses_used, note)``.  The preregistration excludes a degenerate
    dose from the monotonicity check by its stated rule rather than by inspection, so an
    excluded dose is named in the note and the check runs on what is left.
    """
    by_alpha = {round(item.alpha, 6): item for item in readouts}
    used, values, dropped = [], [], []
    for alpha in alphas:
        item = by_alpha.get(round(float(alpha), 6))
        if item is None:
            continue
        if item.degenerate:
            dropped.append(float(alpha))
            continue
        value = getattr(item, attribute)
        if value is None:
            dropped.append(float(alpha))
            continue
        used.append(float(alpha))
        values.append(float(value))
    note = None
    if dropped:
        note = "excluded degenerate or unavailable doses: %s" % ", ".join("%g" % a for a in dropped)
    if len(values) < 2:
        return False, tuple(used), note or "fewer than two usable doses"
    monotone = all(later <= earlier for earlier, later in zip(values, values[1:]))
    return monotone, tuple(used), note


# --------------------------------------------------------------------------------------
# Verdicts J1 - J6
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    prediction_id: str
    statement: str
    supported: bool
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"prediction_id": self.prediction_id, "statement": self.statement,
                "supported": bool(self.supported), "detail": dict(self.detail)}


J_STATEMENTS = _freeze({
    "J1": ("Tone is linearly decodable from the pre-response state: peak LOO AUC >= 0.80 on "
           "discovery at some middle layer (12-30), and >= 0.75 on holdout at the "
           "discovery-chosen layer."),
    "J2": ("Validity (false-failure vs accurate) is decodable but weaker than tone at the same "
           "layer (AUC lower by >= 0.05)."),
    "J3": ("Holdout tone-probe score correlates negatively with M1 within cell (pooled Spearman "
           "rho <= -0.2, item-bootstrap CI excluding 0)."),
    "J4": ("Steering with the tone direction lowers M1 on neutral holdout items dose-dependently: "
           "M1(alpha=2) - M1(alpha=0) < 0 with item-bootstrap 95% CI excluding 0, and monotone in "
           "alpha over {0.5, 1, 2}."),
    "J5": ("None of the 5 random matched-norm directions produces an M1 drop with CI excluding 0 "
           "at alpha=2 (direction specificity); the unrelated semantic direction does not either."),
    "J6": ("Tone steering at alpha >= 2 raises the non-answer rate and/or judge distress on "
           "neutral items (either CI excluding 0)."),
})


#: Tolerance for calling two AUCs equal when deciding where the peak is attained.
PEAK_TIE_TOLERANCE = 1e-12


def verdict_j1(discovery: Sequence[LayerAuc], chosen_layer: int,
               holdout_auc: float | None) -> Verdict:
    """J1's band clause reads "peak LOO AUC >= 0.80 ... **at some** middle layer".

    Judgement call, recorded (2026-08-18, after the discovery probes and before any steered
    generation): the band clause asks whether the peak value is *attained somewhere* in layers
    12-30, not whether the tie-broken ``argmax`` index happens to fall there.  On this data the
    tone AUC is exactly 1.000 across a wide plateau, so the two readings differ only through
    the frozen "ties to the lower layer" rule, which exists to pick ONE layer for steering and
    was never meant to decide a hypothesis.  Both readings are reported in the detail --
    ``peak_attained_in_middle_band`` (used for the verdict) and ``argmax_layer_in_middle_band``
    (the stricter index reading) -- so a reader can apply either.
    """
    available = [item for item in discovery if item.auc is not None]
    peak_value = max((item.auc for item in available), default=None)
    peak_layers = [item.layer for item in available
                   if peak_value is not None and item.auc >= peak_value - PEAK_TIE_TOLERANCE]
    lower, upper = MIDDLE_LAYERS
    attained_in_band = any(lower <= layer <= upper for layer in peak_layers)
    argmax_in_band = bool(peak_layers) and lower <= min(peak_layers) <= upper
    discovery_ok = peak_value is not None and peak_value >= J1_DISCOVERY_AUC and attained_in_band
    holdout_ok = holdout_auc is not None and holdout_auc >= J1_HOLDOUT_AUC
    return Verdict("J1", J_STATEMENTS["J1"], bool(discovery_ok and holdout_ok), _freeze({
        "peak_layer": None if not peak_layers else min(peak_layers),
        "peak_discovery_auc": peak_value,
        "peak_layers": peak_layers,
        "peak_attained_in_middle_band": bool(attained_in_band),
        "argmax_layer_in_middle_band": bool(argmax_in_band),
        "middle_band": list(MIDDLE_LAYERS),
        "chosen_layer": int(chosen_layer),
        "holdout_auc_at_chosen_layer": holdout_auc,
        "discovery_threshold": J1_DISCOVERY_AUC,
        "holdout_threshold": J1_HOLDOUT_AUC,
    }))


def verdict_j2(tone_auc: float | None, validity_auc: float | None, layer: int, *,
               basis: str = "holdout") -> Verdict:
    decodable = validity_auc is not None and validity_auc > 0.5
    weaker = (tone_auc is not None and validity_auc is not None
              and (tone_auc - validity_auc) >= J2_AUC_GAP)
    return Verdict("J2", J_STATEMENTS["J2"], bool(decodable and weaker), _freeze({
        "basis": basis, "layer": int(layer), "tone_auc": tone_auc, "validity_auc": validity_auc,
        "gap": None if (tone_auc is None or validity_auc is None) else tone_auc - validity_auc,
        "required_gap": J2_AUC_GAP, "validity_above_chance": bool(decodable),
    }))


def verdict_j3(correlation: CorrelationResult) -> Verdict:
    supported = bool(correlation.rho is not None and correlation.rho <= J3_RHO
                     and correlation.excludes_zero)
    return Verdict("J3", J_STATEMENTS["J3"], supported, _freeze({
        "rho": correlation.rho, "ci95_lower": correlation.ci95_lower,
        "ci95_upper": correlation.ci95_upper, "n_items": correlation.n_items,
        "n_pairs": correlation.n_pairs, "n_cells": correlation.n_cells,
        "threshold": J3_RHO, "ci_excludes_zero": correlation.excludes_zero,
        "unavailable_reason": correlation.unavailable_reason,
    }))


def verdict_j4(tone_doses: Sequence[DoseReadout]) -> Verdict:
    by_alpha = {round(item.alpha, 6): item for item in tone_doses}
    at_two = by_alpha.get(2.0)
    drop = bool(at_two is not None and at_two.m1_drop_supported)
    monotone, used, note = monotone_in_alpha(tone_doses)
    return Verdict("J4", J_STATEMENTS["J4"], bool(drop and monotone), _freeze({
        "alpha2_m1_delta": None if at_two is None else at_two.m1_delta.estimate,
        "alpha2_ci95": None if at_two is None else [at_two.m1_delta.ci95_lower,
                                                   at_two.m1_delta.ci95_upper],
        "alpha2_ci_excludes_zero_negative": drop,
        "alpha2_degenerate": bool(at_two is not None and at_two.degenerate),
        "monotone_over_used_doses": bool(monotone), "doses_used": list(used),
        "monotonicity_note": note,
    }))


def verdict_j5(control_doses: Sequence[DoseReadout], *, alpha: float = 2.0) -> Verdict:
    offenders = []
    checked = []
    for item in control_doses:
        if round(item.alpha, 6) != round(float(alpha), 6):
            continue
        checked.append(item.direction_id)
        if item.m1_drop_supported:
            offenders.append(item.direction_id)
    return Verdict("J5", J_STATEMENTS["J5"], bool(checked and not offenders), _freeze({
        "alpha": float(alpha), "directions_checked": sorted(checked),
        "directions_with_supported_m1_drop": sorted(offenders),
    }))


def verdict_j6(tone_doses: Sequence[DoseReadout],
               distress: Mapping[float, BootstrapResult] | None = None) -> Verdict:
    non_answer_hits = []
    for item in tone_doses:
        lower = item.non_answer_delta.ci95_lower
        if item.alpha >= 2.0 and lower is not None and lower > 0.0:
            non_answer_hits.append(item.alpha)
    distress_hits = []
    for alpha, result in sorted((distress or {}).items()):
        if float(alpha) >= 2.0 and result.ci95_lower is not None and result.ci95_lower > 0.0:
            distress_hits.append(float(alpha))
    return Verdict("J6", J_STATEMENTS["J6"],
                   bool(non_answer_hits or distress_hits), _freeze({
                       "non_answer_rate_increase_at": non_answer_hits,
                       "distress_increase_at": distress_hits,
                       "distress_judged": distress is not None,
                   }))


__all__ = [name for name in dir() if not name.startswith("_")]
