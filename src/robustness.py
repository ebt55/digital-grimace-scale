"""Preregistration v7 robustness checks: hostile wording (W), item scale (S), model scale (G).

EXPLORATORY ROBUSTNESS -- these three checks change no confirmatory verdict.  They ask whether
the judge-free M1 / non-answer signature survives (W) three other mild hostile wordings, (S) a
five-times larger fresh item bank and (G) a 27B model.  All three are **greedy-only**: no
T = 0.8 resamples are generated, so M2 -- and therefore the M2-valued H8 contrast -- is not
measured and is reported as *not estimable* rather than silently omitted.

The module has two halves and no I/O beyond reading the files it is handed:

*Generation side.*  :func:`derive_protocol` returns a :class:`~src.protocol.Protocol` whose frozen
``conditions`` carry one paraphrase set in place of the four hostile strings, and/or whose task
bank is an alternative, namespaced set of items.  ``configs/conditions.json``,
``stimuli/matched_pairs.jsonl`` and every other locked fixture are read-only throughout: the
override lives in memory, and the wording that was actually sent survives verbatim in every
record's ``messages`` (and therefore in its ``prompt_sha256``) plus a ``provenance.wording_set``
tag written by the driver.

*Analysis side.*  Item-paired differences with a 2,000-resample, seeded, item-clustered bootstrap
-- the same shape as ``src.pipeline.exploratory_contrasts`` -- computed over the committed metric
rows of each check and of the reference run, then the nine preregistered verdicts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .extract import MetricRow
from .protocol import Protocol, ProtocolError, Task, load_protocol, validate_protocol

BOOTSTRAP_KEY = "DGS-AC1-ROBUSTNESS-v1"
BOOTSTRAP_SAMPLES = 2000
PREREGISTRATION = "notes/preregistration_v7_robustness.md"
WORDINGS_FILE = "configs/robustness_wordings.json"
# The preregistration's feasibility clause: below this parseable-answer rate in the neutral cells,
# an M1 contrast is reported as "not estimable" rather than as a number.
PARSEABLE_FLOOR = 0.50
# The manipulation-check feasibility band: a paraphrase must land within this many rubric points
# of the frozen counterpart it replaces.  A miss is reported, never fixed.
MANIPULATION_BAND = 1.5

DIFFICULTIES = ("easy", "hard")
ACCURATE = "accurate"
MALFUNCTIONING = "malfunctioning_always_fail"
# The locked task-id namespace.  An alternative bank may never enter it.
LOCKED_NAMESPACE = re.compile(r"DGS-\d+", re.IGNORECASE)
TASK_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9._\-]+")


class RobustnessError(ValueError):
    """Raised when a robustness bank, wording set or contrast input is inconsistent."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ==========================================================================
# Generation side -- wording overrides
# ==========================================================================

# Where a paraphrase set lands in configs/conditions.json.  The last element names which of the
# set's two strings goes there; nothing else in the frozen conditions is touched.
HOSTILE_OVERRIDE_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("feedback_messages", "accurate", "hostile", "if_preceding_answer_incorrect"), "incorrect"),
    (("feedback_messages", "accurate", "hostile", "if_preceding_answer_correct"), "correct"),
    (("feedback_messages", MALFUNCTIONING, "hostile"), "incorrect"),
    (("symmetric_onset_test", "failure_message", "hostile"), "incorrect"),
)
WORDING_KEYS = ("incorrect", "correct")


def load_wording_sets(path: str | Path) -> Mapping[str, Mapping[str, str]]:
    """Read the paraphrase sets, validating that each carries exactly the two frozen roles."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RobustnessError("cannot read wording sets: %s" % path) from error
    sets = payload.get("sets") if isinstance(payload, Mapping) else None
    if not isinstance(sets, Mapping) or not sets:
        raise RobustnessError("wording file must contain a nonempty `sets` object: %s" % path)
    out: dict[str, Mapping[str, str]] = {}
    for name, wording in sets.items():
        out[name] = check_wording(name, wording)
    return _freeze(out)


def check_wording(name: str, wording: Any) -> Mapping[str, str]:
    """One paraphrase set: exactly ``incorrect`` and ``correct``, both nonempty strings."""
    if not isinstance(name, str) or not name or "|" in name:
        raise RobustnessError("wording set name must be a nonempty string without '|'")
    if not isinstance(wording, Mapping) or tuple(sorted(wording)) != tuple(sorted(WORDING_KEYS)):
        raise RobustnessError("wording set %s must contain exactly %s" % (name, ", ".join(WORDING_KEYS)))
    for key in WORDING_KEYS:
        value = wording[key]
        if not isinstance(value, str) or not value.strip():
            raise RobustnessError("wording set %s field %s must be a nonempty string" % (name, key))
    return _freeze({key: wording[key] for key in WORDING_KEYS})


def apply_wording(conditions: Mapping[str, Any], wording: Mapping[str, str]) -> dict[str, Any]:
    """Return a plain-dict copy of the frozen conditions with the four hostile strings replaced.

    The input mapping is never mutated: ``configs/conditions.json`` is hash-locked and the
    protocol's view of it is read-only, so the override is a copy the caller owns.
    """
    updated = _thaw(conditions)
    if not isinstance(updated, dict):
        raise RobustnessError("conditions must be an object")
    for path, key in HOSTILE_OVERRIDE_PATHS:
        node: Any = updated
        for step in path[:-1]:
            node = node.get(step) if isinstance(node, Mapping) else None
            if not isinstance(node, dict):
                raise RobustnessError("conditions is missing the override path %s" % ".".join(path))
        if path[-1] not in node or not isinstance(node[path[-1]], str):
            raise RobustnessError("conditions is missing the override path %s" % ".".join(path))
        node[path[-1]] = wording[key]
    return updated


def wording_provenance(name: str, wording: Mapping[str, str]) -> dict[str, str]:
    """The per-record provenance tag: which set was sent, and a hash of its exact two strings."""
    joined = "%s|%s|%s" % (name, wording["incorrect"], wording["correct"])
    return {"wording_set": name, "wording_sha256": _sha256(joined)}


def frozen_hostile_strings(conditions: Mapping[str, Any]) -> dict[str, str]:
    """The two hostile strings a paraphrase set replaces, read out of the frozen conditions."""
    out: dict[str, str] = {}
    for path, key in HOSTILE_OVERRIDE_PATHS:
        node: Any = conditions
        for step in path:
            node = node[step]
        out.setdefault(key, node)
    return out


# ==========================================================================
# Generation side -- alternative task banks
# ==========================================================================

SUBJECT_DIFFICULTY = MappingProxyType({"arc-easy": "easy", "arc-challenge": "hard"})


def check_task_id(task_id: Any, locked: Iterable[str] = ()) -> str:
    """Namespaced, never in the locked ``DGS-0xx`` space, never colliding with a locked id."""
    if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
        raise RobustnessError("alternative task id must be namespaced like `ARC-<id>`, got %r" % (task_id,))
    if LOCKED_NAMESPACE.fullmatch(task_id) or task_id.upper().startswith("DGS-"):
        raise RobustnessError("alternative task id may not enter the locked DGS- namespace: %s" % task_id)
    if task_id in set(locked):
        raise RobustnessError("alternative task id collides with a locked task: %s" % task_id)
    return task_id


def _difficulty(row: Mapping[str, Any]) -> str:
    value = row.get("difficulty")
    if isinstance(value, str) and value in DIFFICULTIES:
        return value
    subject = row.get("subject")
    mapped = SUBJECT_DIFFICULTY.get(str(subject).strip().lower()) if subject is not None else None
    if mapped is None:
        raise RobustnessError("cannot derive difficulty for %r" % (row.get("item_id") or row.get("task_id"),))
    return mapped


def task_from_row(row: Mapping[str, Any], protocol: Protocol, *, namespace: str = "ARC") -> Task:
    """Materialise one bank row as a frozen :class:`Task` with a namespaced id and split.

    ``prompt`` is stored in the frozen renderer's own shape (stem, blank line, then the exact
    required output instruction), so ``render_task`` produces byte-identical user turns to the
    locked stimuli.
    """
    if not isinstance(row, Mapping):
        raise RobustnessError("bank row must be an object")
    stem = row.get("stem") if isinstance(row.get("stem"), str) else row.get("prompt")
    if not isinstance(stem, str) or not stem.strip():
        raise RobustnessError("bank row needs a nonempty stem")
    options = row.get("options")
    if not isinstance(options, Mapping) or tuple(options) != ("A", "B", "C", "D") or not all(
            isinstance(value, str) and value.strip() for value in options.values()):
        raise RobustnessError("bank row options must be exactly ordered, nonempty A-D strings")
    answer = row.get("canonical_answer")
    if answer not in options:
        raise RobustnessError("bank row canonical answer is not one of its options")
    raw_id = row.get("task_id") or row.get("item_id") or row.get("id")
    task_id = raw_id if isinstance(raw_id, str) and raw_id.startswith(namespace + "-") else "%s-%s" % (namespace, raw_id)
    check_task_id(task_id, (task.task_id for task in protocol.matched_tasks))
    instruction = protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
    prompt = "%s\n\n%s" % (stem.strip(), instruction)
    domain = row.get("domain") or row.get("subject") or row.get("dataset") or "alternative_bank"
    return Task(task_id, str(domain), _difficulty(row), prompt, _freeze(dict(options)), answer, "discovery")


def load_task_bank(path: str | Path, protocol: Protocol, *, namespace: str = "ARC") -> tuple[Task, ...]:
    """Read a JSONL alternative task bank into frozen Tasks, rejecting duplicate ids."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RobustnessError("cannot read task bank: %s" % path) from error
    tasks: list[Task] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RobustnessError("invalid JSONL at %s:%d" % (path, number)) from error
        task = task_from_row(row, protocol, namespace=namespace)
        if task.task_id in seen:
            raise RobustnessError("duplicate task id in bank: %s" % task.task_id)
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise RobustnessError("task bank is empty: %s" % path)
    return tuple(tasks)


def item_rank(row: Mapping[str, Any]) -> str:
    """Deterministic selection order, so which items enter S cannot be steered after the fact."""
    stem = row.get("stem") if isinstance(row.get("stem"), str) else str(row.get("task_id") or "")
    normalized = " ".join(str(stem).split()).strip().lower()
    return _sha256("%s|SELECT|%s" % (BOOTSTRAP_KEY, normalized))


def select_bank_items(rows: Sequence[Mapping[str, Any]], *, used_ids: Iterable[str] = (),
                      per_difficulty: int = 50) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    """Up to ``per_difficulty`` items per difficulty by hash rank, after removing used items.

    Shortfalls are reported, never padded: the preregistration says to take what exists and say
    how many that was.
    """
    used = {str(item) for item in used_ids}
    kept: list[Mapping[str, Any]] = []
    dropped_used = 0
    for row in rows:
        identifier = str(row.get("item_id") or row.get("task_id") or row.get("id") or "")
        if identifier and (identifier in used or ("ARC-" + identifier) in used
                           or identifier.removeprefix("ARC-") in used):
            dropped_used += 1
            continue
        kept.append(row)
    chosen: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    available: dict[str, int] = {}
    for difficulty in DIFFICULTIES:
        pool = [row for row in kept if _safe_difficulty(row) == difficulty]
        available[difficulty] = len(pool)
        picked = sorted(pool, key=item_rank)[:per_difficulty]
        counts[difficulty] = len(picked)
        chosen.extend(picked)
    provenance = {
        "read": len(rows),
        "dropped_already_used": dropped_used,
        "available_per_difficulty": available,
        "selected_per_difficulty": counts,
        "target_per_difficulty": per_difficulty,
        "shortfall_per_difficulty": {key: max(0, per_difficulty - counts[key]) for key in DIFFICULTIES},
        "selected_total": len(chosen),
        "selection_rule": "sha256 rank of the normalised stem, ascending",
    }
    return tuple(chosen), provenance


def _safe_difficulty(row: Mapping[str, Any]) -> str | None:
    try:
        return _difficulty(row)
    except RobustnessError:
        return None


# ==========================================================================
# Generation side -- protocol derivation
# ==========================================================================

def derive_protocol(protocol: Protocol, *, wording: Mapping[str, str] | None = None,
                    tasks: Sequence[Task] | None = None) -> Protocol:
    """A protocol view with an overridden wording and/or an alternative task bank.

    Both arguments default to ``None``, in which case the frozen protocol is returned unchanged
    -- the driver's behaviour without the robustness flags is therefore byte-identical.
    """
    if wording is None and tasks is None:
        return protocol
    conditions = protocol.conditions
    if wording is not None:
        conditions = _freeze(apply_wording(protocol.conditions, wording))
    matched = protocol.matched_tasks if tasks is None else tuple(tasks)
    if tasks is not None and not matched:
        raise RobustnessError("alternative task bank is empty")
    derived = replace(protocol, conditions=conditions, matched_tasks=matched)
    try:
        validate_protocol(derived)
    except ProtocolError as error:
        raise RobustnessError("derived protocol is inconsistent: %s" % error) from error
    return derived


# ==========================================================================
# Analysis side -- item-paired contrasts
# ==========================================================================

MEASURED, ONSET, RECOVERY, WASHOUT = "measured", "onset", "recovery", "onset_washout"


@dataclass(frozen=True)
class CellRef:
    """One side of a contrast: a factorial arm/tone at one endpoint."""

    validity: str
    tone: str
    turn: str = MEASURED

    def cell_id(self, difficulty: str) -> str:
        return "%s__%s__%s" % (difficulty, self.validity, self.tone)


@dataclass(frozen=True)
class ContrastDef:
    contrast_id: str
    label: str
    metric: str
    left: CellRef
    right: CellRef
    difficulties: tuple[str, ...]
    stratum: str


@dataclass(frozen=True)
class Estimate:
    contrast_id: str
    label: str
    metric: str
    stratum: str
    estimate: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    n_items: int
    n_pairs: int
    unavailable_reason: str | None = None

    @property
    def excludes_zero(self) -> bool:
        if self.ci95_lower is None or self.ci95_upper is None:
            return False
        return self.ci95_lower > 0.0 or self.ci95_upper < 0.0

    @property
    def ci_width(self) -> float | None:
        if self.ci95_lower is None or self.ci95_upper is None:
            return None
        return self.ci95_upper - self.ci95_lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_id": self.contrast_id, "label": self.label, "metric": self.metric,
            "stratum": self.stratum, "estimate": self.estimate,
            "ci95_lower": self.ci95_lower, "ci95_upper": self.ci95_upper,
            "ci95_width": self.ci_width, "n_items": self.n_items, "n_pairs": self.n_pairs,
            "excludes_zero": self.excludes_zero, "unavailable_reason": self.unavailable_reason,
        }


# The preregistered contrast shapes.  H1/H2a/H2b/H3a/H3b are the extension-analysis contrasts
# (src.confirm.HYPOTHESES) restated on raw, unstandardised M1; H8 is M2-valued and therefore not
# estimable in a greedy-only design, which the analysis states rather than hides.  TONE_* are the
# W-specific accurate- and malfunctioning-arm tone effects, and NONANSWER_* their non-answer twins.
CONTRASTS: tuple[ContrastDef, ...] = (
    ContrastDef("H1", "M1, malfunctioning - accurate (measured, neutral)", "m1",
                CellRef(MALFUNCTIONING, "neutral"), CellRef(ACCURATE, "neutral"), ("easy",), "easy | neutral"),
    ContrastDef("H1_hard", "M1, malfunctioning - accurate (measured, neutral)", "m1",
                CellRef(MALFUNCTIONING, "neutral"), CellRef(ACCURATE, "neutral"), ("hard",), "hard | neutral"),
    ContrastDef("H2a", "M1, hostile - neutral (measured, accurate)", "m1",
                CellRef(ACCURATE, "hostile"), CellRef(ACCURATE, "neutral"), ("easy",), "easy | accurate"),
    ContrastDef("H2b", "M1, hostile - neutral (measured, accurate)", "m1",
                CellRef(ACCURATE, "hostile"), CellRef(ACCURATE, "neutral"), ("hard",), "hard | accurate"),
    ContrastDef("TONE_ACC_POOLED", "M1, hostile - neutral (measured, accurate)", "m1",
                CellRef(ACCURATE, "hostile"), CellRef(ACCURATE, "neutral"), DIFFICULTIES, "easy+hard | accurate"),
    ContrastDef("TONE_MAL_EASY", "M1, hostile - neutral (measured, malfunctioning)", "m1",
                CellRef(MALFUNCTIONING, "hostile"), CellRef(MALFUNCTIONING, "neutral"), ("easy",), "easy | malfunctioning"),
    ContrastDef("TONE_MAL_HARD", "M1, hostile - neutral (measured, malfunctioning)", "m1",
                CellRef(MALFUNCTIONING, "hostile"), CellRef(MALFUNCTIONING, "neutral"), ("hard",), "hard | malfunctioning"),
    ContrastDef("TONE_MAL_POOLED", "M1, hostile - neutral (measured, malfunctioning)", "m1",
                CellRef(MALFUNCTIONING, "hostile"), CellRef(MALFUNCTIONING, "neutral"), DIFFICULTIES, "easy+hard | malfunctioning"),
    ContrastDef("H3a", "M1, onset - measured (accurate, neutral)", "m1",
                CellRef(ACCURATE, "neutral", ONSET), CellRef(ACCURATE, "neutral"), ("easy",), "easy | neutral"),
    ContrastDef("H3b", "M1, onset - measured (accurate, hostile)", "m1",
                CellRef(ACCURATE, "hostile", ONSET), CellRef(ACCURATE, "hostile"), ("easy",), "easy | hostile"),
    ContrastDef("ONSET_HOSTILE_POOLED", "M1, onset - measured (accurate, hostile)", "m1",
                CellRef(ACCURATE, "hostile", ONSET), CellRef(ACCURATE, "hostile"), DIFFICULTIES, "easy+hard | hostile"),
    ContrastDef("H8", "M2, hostile - neutral (measured, accurate)", "m2",
                CellRef(ACCURATE, "hostile"), CellRef(ACCURATE, "neutral"), ("easy",), "easy | accurate"),
    ContrastDef("NONANSWER_ACC_POOLED", "Non-answer rate, hostile - neutral (measured, accurate)", "non_answer_rate",
                CellRef(ACCURATE, "hostile"), CellRef(ACCURATE, "neutral"), DIFFICULTIES, "easy+hard | accurate"),
    ContrastDef("NONANSWER_MAL_POOLED", "Non-answer rate, hostile - neutral (measured, malfunctioning)", "non_answer_rate",
                CellRef(MALFUNCTIONING, "hostile"), CellRef(MALFUNCTIONING, "neutral"), DIFFICULTIES, "easy+hard | malfunctioning"),
)
CONTRASTS_BY_ID = MappingProxyType({item.contrast_id: item for item in CONTRASTS})
# The contrasts the S and G tables print in the preregistration's own order.
HYPOTHESIS_SHAPED = ("H1", "H2a", "H2b", "TONE_ACC_POOLED", "H3a", "H3b", "H8")
# The contrasts check W reports per paraphrase set.
W_CONTRASTS = ("H2a", "H2b", "TONE_ACC_POOLED", "TONE_MAL_EASY", "TONE_MAL_HARD",
               "TONE_MAL_POOLED", "NONANSWER_ACC_POOLED", "NONANSWER_MAL_POOLED",
               "H3b", "ONSET_HOSTILE_POOLED")


def metric_value(row: MetricRow, name: str) -> float | None:
    """Raw, unstandardised value for one endpoint; ``None`` when the metric is absent."""
    if name == "m1":
        return None if row.m1 is None else float(row.m1)
    if name == "m2":
        return None if row.m2 is None else float(row.m2)
    if name == "non_answer_rate":
        return 0.0 if row.greedy_answer_valid else 1.0
    if name == "accuracy":
        return None if not row.greedy_answer_valid else float(bool(row.greedy_answer_correct))
    raise RobustnessError("unknown robustness metric: %s" % name)


def index_rows(rows: Iterable[MetricRow], model_id: str | None = None) -> dict[tuple[str, str, str], MetricRow]:
    """``(task_id, cell_id, turn_label) -> row`` for the factorial endpoints of one model."""
    index: dict[tuple[str, str, str], MetricRow] = {}
    for row in rows:
        if model_id is not None and row.model_id != model_id:
            continue
        if row.cell_kind != "factorial":
            continue
        index[(row.task_id, row.cell_id, row.turn_label)] = row
    return index


def item_bootstrap(pairs: Sequence[tuple[str, float]], seed_text: str,
                   resamples: int = BOOTSTRAP_SAMPLES) -> tuple[float | None, float | None, float | None, int]:
    """Item-clustered bootstrap over ``(task_id, difference)`` pairs; returns (point, lo, hi, items)."""
    by_item: dict[str, list[float]] = {}
    for task_id, value in pairs:
        by_item.setdefault(task_id, []).append(float(value))
    if not by_item:
        return None, None, None, 0
    point = mean(value for values in by_item.values() for value in values)
    if len(by_item) < 2:
        return point, None, None, len(by_item)
    items = sorted(by_item)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big"))
    draws = []
    for _ in range(resamples):
        sampled = [rng.choice(items) for _ in items]
        draws.append(mean(value for item in sampled for value in by_item[item]))
    draws.sort()

    def quantile(probability: float) -> float:
        position = (len(draws) - 1) * probability
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)

    return point, quantile(0.025), quantile(0.975), len(by_item)


def contrast_pairs(left_index: Mapping[tuple[str, str, str], MetricRow],
                   right_index: Mapping[tuple[str, str, str], MetricRow],
                   definition: ContrastDef) -> list[tuple[str, float]]:
    """Every item on which both sides of the contrast produced the metric."""
    pairs: list[tuple[str, float]] = []
    for (task_id, cell_id, turn_label), row in sorted(left_index.items()):
        difficulty = cell_id.split("__")[0]
        if difficulty not in definition.difficulties:
            continue
        if cell_id != definition.left.cell_id(difficulty) or turn_label != definition.left.turn:
            continue
        other = right_index.get((task_id, definition.right.cell_id(difficulty), definition.right.turn))
        if other is None:
            continue
        left_value = metric_value(row, definition.metric)
        right_value = metric_value(other, definition.metric)
        if left_value is None or right_value is None:
            continue
        pairs.append((task_id, left_value - right_value))
    return pairs


def estimate_contrast(left_index, right_index, definition: ContrastDef, seed_text: str) -> Estimate:
    """One preregistered contrast with a seeded item-clustered bootstrap 95% CI."""
    pairs = contrast_pairs(left_index, right_index, definition)
    point, lower, upper, items = item_bootstrap(pairs, "%s|%s|%s" % (BOOTSTRAP_KEY, seed_text, definition.contrast_id))
    reason = None
    if not pairs:
        reason = "no_paired_items"
    elif items < 2:
        reason = "at_least_two_items_required_for_cluster_ci"
    return Estimate(definition.contrast_id, definition.label, definition.metric, definition.stratum,
                    point, lower, upper, items, len(pairs), reason)


def estimate_all(left_index, right_index, seed_text: str,
                 contrast_ids: Sequence[str] = tuple(CONTRASTS_BY_ID)) -> dict[str, Estimate]:
    return {key: estimate_contrast(left_index, right_index, CONTRASTS_BY_ID[key], seed_text)
            for key in contrast_ids}


# ==========================================================================
# Analysis side -- feasibility and descriptive rates
# ==========================================================================

def parseable_rate(rows: Iterable[MetricRow], *, tone: str | None = None,
                   turn: str = MEASURED) -> tuple[float | None, int]:
    """Fraction of measured factorial endpoints whose greedy answer parsed."""
    selected = [row for row in rows
                if row.cell_kind == "factorial" and row.turn_label == turn
                and (tone is None or row.tone == tone)]
    if not selected:
        return None, 0
    return sum(1 for row in selected if row.greedy_answer_valid) / len(selected), len(selected)


def non_answer_by_cell(rows: Iterable[MetricRow]) -> tuple[dict[str, Any], ...]:
    """Per cell x endpoint non-answer rate and mean M1, with no exclusions applied."""
    grouped: dict[tuple[str, str], list[MetricRow]] = {}
    for row in rows:
        if row.cell_kind != "factorial":
            continue
        grouped.setdefault((row.cell_id, row.turn_label), []).append(row)
    out = []
    for key in sorted(grouped):
        group = grouped[key]
        m1_values = [row.m1 for row in group if row.m1 is not None]
        out.append({
            "cell_id": key[0], "turn_label": key[1], "n_endpoints": len(group),
            "n_items": len({row.task_id for row in group}),
            "non_answer_rate": sum(0.0 if row.greedy_answer_valid else 1.0 for row in group) / len(group),
            "mean_m1": mean(m1_values) if m1_values else None, "n_m1": len(m1_values),
        })
    return tuple(out)


def feasible(rate: float | None) -> bool:
    """The preregistration's < 50% parseable clause, applied literally."""
    return rate is not None and rate >= PARSEABLE_FLOOR


# A serving artifact, not model behaviour: vLLM streams a chat model's end-of-turn markers as
# logprob entries whose text is absent from `message.content`, and `src.backend` trims them only
# when the token trace is a literal prefix of that content.  When the model interleaves a plain
# newline between the two markers the prefix rule cannot fire, the markers survive into
# `response_text`, and the frozen `Answer: X` rule rejects the response because a nonempty line
# follows the answer.  The diagnostic below measures how much of a run's non-answer rate is that
# artifact.  It is REPORTED ONLY: the frozen parser is never replaced, no contrast is recomputed
# on the stripped text, and no verdict moves because of it.
#
# The strip itself lives in src.protocol (amendment A6's definition) so there is exactly one
# list of special-token strings in the codebase; this module only reports on it.


def strip_trailing_special(text: str) -> str:
    """Remove a trailing run of end-of-turn / end-of-sequence markers and the whitespace around it."""
    from .protocol import ProtocolError, strip_trailing_special_tokens  # noqa: PLC0415

    try:
        return strip_trailing_special_tokens(text)
    except ProtocolError as error:
        raise RobustnessError(str(error)) from error


def reparse_diagnostic(pairs: Iterable[tuple[str, bool]]) -> dict[str, Any]:
    """How many non-parsing responses would parse once the trailing markers are removed.

    ``pairs`` is ``(response_text, frozen_answer_valid)`` per endpoint.  Reported, never applied.
    """
    from .protocol import parse_final_answer  # local import: keeps this module import-light

    total = frozen_valid = stripped_valid = recovered = affected = 0
    for text, valid in pairs:
        total += 1
        frozen_valid += bool(valid)
        stripped = strip_trailing_special(text)
        if stripped != text:
            affected += 1
        parsed = parse_final_answer(stripped).valid
        stripped_valid += bool(parsed)
        if parsed and not valid:
            recovered += 1
    return {
        "n_endpoints": total,
        "frozen_parseable_rate": frozen_valid / total if total else None,
        "stripped_parseable_rate": stripped_valid / total if total else None,
        "n_recovered_by_stripping": recovered,
        "n_with_trailing_markers": affected,
        "note": "Diagnostic only. The frozen Amendment-A1 parser is never replaced and no "
                "contrast or verdict is computed on the stripped text.",
    }


# ==========================================================================
# Analysis side -- verdicts
# ==========================================================================

PASS = "PASS"
NOT_SUPPORTED = "not supported"
NOT_ESTIMABLE = "not estimable"

PREDICTIONS = MappingProxyType({
    "W-1": ("Pooled accurate-arm tone effect on M1 is negative with CI excluding 0 for each of W1, W2, W3.", 65),
    "W-2": ("The three sets' pooled tone effects lie within a factor of 2 of the frozen wording's estimate (ratios in [0.5, 2]).", 50),
    "W-3": ("Non-answer rate under hostile tone exceeds neutral for each set (CI excluding 0 for >= 2 of 3).", 55),
    "S-1": ("H1 (false-failure, neutral, easy) and pooled tone (H2a/H2b) M1 effects are negative with CIs excluding 0 on the 100-item bank.", 70),
    "S-2": ("Point estimates lie within a factor of 2 of the discovery estimates for H1 and the pooled tone effect.", 55),
    "S-3": ("The 100-item CIs are narrower than the 20-item CIs for the same contrasts.", 85),
    "G-1": ("H1 and pooled tone M1 effects are negative with CIs excluding 0 at 27B.", 65),
    "G-2": ("The tone effect at 27B is not smaller than at 9B by more than half (ratio >= 0.5).", 50),
    "G-3": ("Hostile-onset distress language is present at 27B (mean >= 2/10).", 60),
})


@dataclass(frozen=True)
class Verdict:
    verdict_id: str
    prediction: str
    confidence: int
    outcome: str
    detail: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.verdict_id, "prediction": self.prediction, "confidence_percent": self.confidence,
                "verdict": self.outcome, "detail": self.detail, "evidence": _thaw(dict(self.evidence))}


def _verdict(verdict_id: str, outcome: str, detail: str, evidence: Mapping[str, Any] | None = None) -> Verdict:
    prediction, confidence = PREDICTIONS[verdict_id]
    return Verdict(verdict_id, prediction, confidence, outcome, detail, _freeze(evidence or {}))


def negative_and_excludes_zero(estimate: Estimate | None) -> bool:
    return bool(estimate is not None and estimate.estimate is not None
                and estimate.estimate < 0.0 and estimate.excludes_zero)


def ratio_within(value: Estimate | None, reference: Estimate | None,
                 low: float = 0.5, high: float = 2.0) -> tuple[bool | None, float | None]:
    """``value / reference``, and whether it lies in [low, high].  ``None`` when unavailable."""
    if value is None or reference is None or value.estimate is None or reference.estimate is None:
        return None, None
    if reference.estimate == 0.0:
        return None, None
    ratio = value.estimate / reference.estimate
    return (low <= ratio <= high), ratio


def _fmt(estimate: Estimate | None) -> str:
    if estimate is None or estimate.estimate is None:
        return "n/a"
    if estimate.ci95_lower is None:
        return "%.3f (no CI, %d item%s)" % (estimate.estimate, estimate.n_items,
                                            "" if estimate.n_items == 1 else "s")
    return "%.3f [%.3f, %.3f]" % (estimate.estimate, estimate.ci95_lower, estimate.ci95_upper)


def verdict_w1(sets: Mapping[str, Mapping[str, Estimate]], estimable: Mapping[str, bool]) -> Verdict:
    per_set = {}
    for name in sorted(sets):
        estimate = sets[name].get("TONE_ACC_POOLED")
        per_set[name] = {"estimate": _fmt(estimate), "meets": negative_and_excludes_zero(estimate),
                         "estimable": bool(estimable.get(name, True))}
    if not per_set or not all(item["estimable"] for item in per_set.values()):
        return _verdict("W-1", NOT_ESTIMABLE,
                        "at least one wording set is below the 50%% parseable-answer feasibility floor",
                        {"per_set": per_set})
    passed = all(item["meets"] for item in per_set.values())
    return _verdict("W-1", PASS if passed else NOT_SUPPORTED,
                    "; ".join("%s %s" % (name, per_set[name]["estimate"]) for name in sorted(per_set)),
                    {"per_set": per_set})


def verdict_w2(sets: Mapping[str, Mapping[str, Estimate]], frozen: Estimate | None) -> Verdict:
    ratios: dict[str, Any] = {}
    for name in sorted(sets):
        within, ratio = ratio_within(sets[name].get("TONE_ACC_POOLED"), frozen)
        ratios[name] = {"ratio": ratio, "within_factor_2": within}
    values = [item["within_factor_2"] for item in ratios.values()]
    if not values or any(value is None for value in values) or frozen is None or frozen.estimate is None:
        return _verdict("W-2", NOT_ESTIMABLE, "a pooled tone effect or the frozen reference is unavailable",
                        {"frozen_reference": _fmt(frozen), "ratios": ratios})
    passed = all(values)
    return _verdict("W-2", PASS if passed else NOT_SUPPORTED,
                    "frozen %s; ratios %s" % (_fmt(frozen), ", ".join(
                        "%s %.2f" % (name, ratios[name]["ratio"]) for name in sorted(ratios))),
                    {"frozen_reference": _fmt(frozen), "ratios": ratios})


def verdict_w3(sets: Mapping[str, Mapping[str, Estimate]]) -> Verdict:
    per_set: dict[str, Any] = {}
    for name in sorted(sets):
        estimate = sets[name].get("NONANSWER_ACC_POOLED")
        positive = bool(estimate is not None and estimate.estimate is not None and estimate.estimate > 0.0)
        per_set[name] = {"estimate": _fmt(estimate), "positive": positive,
                         "excludes_zero": bool(estimate is not None and estimate.excludes_zero)}
    if not per_set or any(sets[name].get("NONANSWER_ACC_POOLED") is None for name in sets):
        return _verdict("W-3", NOT_ESTIMABLE, "non-answer contrast unavailable for at least one set", {"per_set": per_set})
    all_positive = all(item["positive"] for item in per_set.values())
    excluding = sum(1 for item in per_set.values() if item["excludes_zero"] and item["positive"])
    passed = all_positive and excluding >= 2
    return _verdict("W-3", PASS if passed else NOT_SUPPORTED,
                    "positive in %d/%d sets; CI excludes 0 in %d/%d"
                    % (sum(1 for item in per_set.values() if item["positive"]), len(per_set), excluding, len(per_set)),
                    {"per_set": per_set, "sets_with_ci_excluding_zero": excluding})


def verdict_s1(check: Mapping[str, Estimate], estimable: bool) -> Verdict:
    h1, tone = check.get("H1"), check.get("TONE_ACC_POOLED")
    if not estimable:
        return _verdict("S-1", NOT_ESTIMABLE, "the fresh bank is below the 50% parseable-answer feasibility floor",
                        {"H1": _fmt(h1), "TONE_ACC_POOLED": _fmt(tone)})
    passed = negative_and_excludes_zero(h1) and negative_and_excludes_zero(tone)
    return _verdict("S-1", PASS if passed else NOT_SUPPORTED,
                    "H1 %s; pooled tone %s" % (_fmt(h1), _fmt(tone)),
                    {"H1": _fmt(h1), "TONE_ACC_POOLED": _fmt(tone)})


def verdict_s2(check: Mapping[str, Estimate], reference: Mapping[str, Estimate], estimable: bool) -> Verdict:
    ratios: dict[str, Any] = {}
    for key in ("H1", "TONE_ACC_POOLED"):
        within, ratio = ratio_within(check.get(key), reference.get(key))
        ratios[key] = {"ratio": ratio, "within_factor_2": within,
                       "fresh": _fmt(check.get(key)), "discovery": _fmt(reference.get(key))}
    values = [item["within_factor_2"] for item in ratios.values()]
    if not estimable or any(value is None for value in values):
        return _verdict("S-2", NOT_ESTIMABLE, "a fresh-bank or discovery estimate is unavailable", {"ratios": ratios})
    return _verdict("S-2", PASS if all(values) else NOT_SUPPORTED,
                    ", ".join("%s ratio %.2f" % (key, ratios[key]["ratio"]) for key in ratios), {"ratios": ratios})


def verdict_s3(check: Mapping[str, Estimate], reference: Mapping[str, Estimate]) -> Verdict:
    widths: dict[str, Any] = {}
    comparable = []
    for key in ("H1", "H2a", "H2b", "TONE_ACC_POOLED", "H3a", "H3b"):
        left, right = check.get(key), reference.get(key)
        fresh_width = left.ci_width if left is not None else None
        discovery_width = right.ci_width if right is not None else None
        narrower = None if (fresh_width is None or discovery_width is None) else fresh_width < discovery_width
        widths[key] = {"fresh_ci_width": fresh_width, "discovery_ci_width": discovery_width, "narrower": narrower}
        if narrower is not None:
            comparable.append(narrower)
    if not comparable:
        return _verdict("S-3", NOT_ESTIMABLE, "no contrast has both CIs", {"widths": widths})
    return _verdict("S-3", PASS if all(comparable) else NOT_SUPPORTED,
                    "narrower on %d/%d comparable contrasts" % (sum(comparable), len(comparable)),
                    {"widths": widths, "n_comparable": len(comparable), "n_narrower": sum(comparable)})


def verdict_g1(check: Mapping[str, Estimate], estimable: bool, ran: bool = True) -> Verdict:
    h1, tone = check.get("H1"), check.get("TONE_ACC_POOLED")
    if not ran:
        return _verdict("G-1", NOT_ESTIMABLE, "check G was not run", {})
    if not estimable:
        return _verdict("G-1", NOT_ESTIMABLE, "27B is below the 50% parseable-answer feasibility floor",
                        {"H1": _fmt(h1), "TONE_ACC_POOLED": _fmt(tone)})
    passed = negative_and_excludes_zero(h1) and negative_and_excludes_zero(tone)
    return _verdict("G-1", PASS if passed else NOT_SUPPORTED,
                    "H1 %s; pooled tone %s" % (_fmt(h1), _fmt(tone)),
                    {"H1": _fmt(h1), "TONE_ACC_POOLED": _fmt(tone)})


def verdict_g2(check: Mapping[str, Estimate], reference: Mapping[str, Estimate],
               estimable: bool, ran: bool = True) -> Verdict:
    within, ratio = ratio_within(check.get("TONE_ACC_POOLED"), reference.get("TONE_ACC_POOLED"), 0.5, math.inf)
    evidence = {"ratio": ratio, "at_27b": _fmt(check.get("TONE_ACC_POOLED")),
                "at_9b": _fmt(reference.get("TONE_ACC_POOLED"))}
    if not ran:
        return _verdict("G-2", NOT_ESTIMABLE, "check G was not run", evidence)
    if not estimable or within is None:
        return _verdict("G-2", NOT_ESTIMABLE, "a tone effect is unavailable at 27B or 9B", evidence)
    return _verdict("G-2", PASS if within else NOT_SUPPORTED, "ratio %.2f (bar >= 0.50)" % ratio, evidence)


def verdict_g3(distress_mean: float | None, n_scored: int, ran: bool = True) -> Verdict:
    evidence = {"mean_distress_at_hostile_onset": distress_mean, "n_scored": n_scored}
    if not ran:
        return _verdict("G-3", NOT_ESTIMABLE, "check G was not run", evidence)
    if distress_mean is None or n_scored == 0:
        return _verdict("G-3", NOT_ESTIMABLE, "no hostile-onset endpoint was judged at 27B", evidence)
    return _verdict("G-3", PASS if distress_mean >= 2.0 else NOT_SUPPORTED,
                    "mean %.3f over %d endpoints (bar >= 2.0)" % (distress_mean, n_scored), evidence)


def manipulation_band(new_score: int | float | None, frozen_score: int | float | None,
                      band: float = MANIPULATION_BAND) -> bool | None:
    """The feasibility clause on a paraphrase: within +/- ``band`` of its frozen counterpart."""
    if new_score is None or frozen_score is None:
        return None
    return abs(float(new_score) - float(frozen_score)) <= band
