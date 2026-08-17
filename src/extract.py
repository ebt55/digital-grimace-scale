"""Raw JSONL generation records -> one flat, committed per-endpoint metric table.

This module is the missing glue between :mod:`src.records` (validated raw
generations) and :mod:`src.analysis` (inferential dataclasses).  It is pure and
importable: every command-line surface lives in ``scripts/``.

Design commitments:

* An *endpoint* is one measured/continuation turn of one conversation, keyed by
  ``(run_id, model_id, immutable_revision, task_id, cell_id, turn_label)``.  Its
  greedy record (``sample_index`` 0) carries M1/M3/entropy/repetition/Tier-B and
  its ten resample records (1-10) carry M2.
* Quality-control gaps are represented, never imputed and never dropped: every
  metric column is paired with a machine-readable ``*_missing_reason`` so the
  QC table can report exactly why a cell is thin.
* Nothing here interprets a synthetic smoke record as empirical evidence; the
  ``run_kind`` field survives into the raw records and the manifest provenance.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from .metrics import (
    MetricInputError, MetricValue, m1_margin, m2_disagreement, m3_for_record,
    partial_entropy, repeated_4gram_rate, tier_b_metrics,
)
from .protocol import Protocol, load_protocol
from .records import RawRecord, RecordError, jsonl_lines, record_from_json


class ExtractError(ValueError):
    """Raised when raw records cannot be assembled into endpoint metric rows."""


ENDPOINT_TURNS = ("measured", "recovery", "onset", "onset_washout")
FEEDBACK_TURNS = tuple("feedback_response_%d" % index for index in range(1, 6))
OPTIONS = ("A", "B", "C", "D")
RESAMPLE_INDICES = tuple(range(1, 11))
NEUTRAL_BASELINE_CELL_SUFFIX = "__accurate__neutral"


@dataclass(frozen=True)
class LoadIssue:
    """One rejected JSONL line, retained so QC can report it rather than hide it."""

    path: str
    line_number: int
    message: str


@dataclass(frozen=True)
class MetricRow:
    """One flat, serialisable endpoint row.  Column order is the field order."""

    phase: str
    run_id: str
    run_kind: str
    model_id: str
    immutable_revision: str
    task_id: str
    split: str | None
    difficulty: str | None
    domain: str | None
    cell_id: str
    cell_kind: str
    feedback_validity: str | None
    tone: str | None
    turn_label: str
    response_id: str
    m1: float | None
    m1_missing_reason: str | None
    m2: float | None
    m2_missing_reason: str | None
    m3_rate: float | None
    m3_missing_reason: str | None
    m3_event_count: int
    m3_loop_flag: bool
    entropy_mean: float | None
    entropy_worst_decile: float | None
    tail_mass_mean: float | None
    entropy_missing_reason: str | None
    rep4: float
    length_tokens: int
    length_drift: float | None
    length_drift_missing_reason: str | None
    hedge_per100: float | None
    selfcorr_per100: float | None
    greedy_answer_valid: bool
    greedy_answer_correct: bool | None
    greedy_answer_letter: str | None
    resample_count: int
    resample_valid_count: int
    history_false_negative: bool | None
    feedback_rounds: int

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (self.run_id, self.model_id, self.immutable_revision, self.task_id, self.cell_id, self.turn_label)

    def metric(self, name: str) -> tuple[float | None, str | None]:
        """Return ``(value, missing_reason)`` for a preregistered primary metric."""
        if name == "M1":
            return self.m1, self.m1_missing_reason
        if name == "M2":
            return self.m2, self.m2_missing_reason
        if name == "M3":
            return self.m3_rate, self.m3_missing_reason
        raise ExtractError("unknown primary metric: %s" % name)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(MetricRow)}


@dataclass(frozen=True)
class QcRow:
    """Condition-wise QC: invalid-answer and metric-missing rates, never imputed."""

    phase: str
    run_id: str
    model_id: str
    cell_id: str
    turn_label: str
    n_endpoints: int
    greedy_invalid_count: int
    greedy_invalid_rate: float
    m1_missing_count: int
    m1_missing_rate: float
    m2_missing_count: int
    m2_missing_rate: float
    m3_missing_count: int
    m3_missing_rate: float
    resample_response_count: int
    resample_invalid_count: int
    resample_invalid_rate: float
    # Amendment A2: items this model drops entirely, decided treatment-blind from
    # the model's own accurate+neutral resamples.  Empty under the frozen rules.
    excluded_item_count: int = 0
    excluded_task_ids: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(QcRow)}


METRIC_ROW_COLUMNS = tuple(field.name for field in fields(MetricRow))
QC_ROW_COLUMNS = tuple(field.name for field in fields(QcRow))


@dataclass(frozen=True)
class Endpoint:
    """The greedy record plus its resample ensemble for one measured turn."""

    run_id: str
    model_id: str
    immutable_revision: str
    task_id: str
    cell_id: str
    turn_label: str
    greedy: RawRecord | None
    resamples: tuple[RawRecord, ...]

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (self.run_id, self.model_id, self.immutable_revision, self.task_id, self.cell_id, self.turn_label)


def _jsonl_paths(paths_or_dir: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(paths_or_dir, (str, Path)):
        candidates: list[Path] = [Path(paths_or_dir)]
    else:
        candidates = [Path(item) for item in paths_or_dir]
    resolved: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            resolved.extend(sorted(candidate.rglob("*.jsonl")))
        else:
            resolved.append(candidate)
    # ``*.failures.jsonl`` holds generation failures, not validated responses.
    return tuple(sorted({path for path in resolved if not path.name.endswith(".failures.jsonl")}))


def load_records(
    paths_or_dir: str | Path | Iterable[str | Path],
    *,
    protocol: Protocol | None = None,
    issues: list[LoadIssue] | None = None,
) -> list[RawRecord]:
    """Load validated raw records from ``*.jsonl`` files or a directory tree.

    ``*.failures.jsonl`` files are skipped.  A malformed line raises unless the
    caller supplies an ``issues`` list, in which case it is reported and skipped
    so that one bad line cannot silently discard an entire run.
    """
    return list(iter_records(paths_or_dir, protocol=protocol, issues=issues))


def iter_records(
    paths_or_dir: str | Path | Iterable[str | Path],
    *,
    protocol: Protocol | None = None,
    issues: list[LoadIssue] | None = None,
):
    """Stream validated raw records without reading a whole file into memory.

    Raw Phase-1 files reach gigabytes because every token carries twenty
    logprobs, so records are parsed one line at a time and never retained here.
    ``newline="\\n"`` keeps the frozen JSONL rule that only a line feed ends a
    record: U+2028/U+2029 and lone carriage returns occur inside real response
    text and are legal, unescaped, inside a JSON string.
    """
    protocol = protocol or load_protocol()
    for path in _jsonl_paths(paths_or_dir):
        try:
            handle = path.open("r", encoding="utf-8", newline="\n")
        except OSError as error:
            if issues is None:
                raise ExtractError("cannot read raw file: %s" % path) from error
            issues.append(LoadIssue(str(path), 0, "cannot read file: %s" % error))
            continue
        with handle:
            for number, raw_line in enumerate(handle, 1):
                line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
                if not line.strip():
                    continue
                try:
                    record = record_from_json(line, protocol)
                except RecordError as error:
                    if issues is None:
                        raise ExtractError("%s:%d: %s" % (path, number, error)) from error
                    issues.append(LoadIssue(str(path), number, str(error)))
                    continue
                yield record


def _conversation_key(record: RawRecord) -> tuple[str, str, str, str, str]:
    return (record.run_id, record.model_id, record.immutable_revision, record.task_id, record.cell_id)


def group_endpoints(records: Sequence[RawRecord]) -> tuple[Endpoint, ...]:
    """Group records into measured/continuation endpoints in canonical order."""
    grouped: dict[tuple[str, str, str, str, str, str], dict[int, RawRecord]] = {}
    for record in records:
        if record.turn_label not in ENDPOINT_TURNS:
            continue
        key = _conversation_key(record) + (record.turn_label,)
        bucket = grouped.setdefault(key, {})
        if record.sample_index in bucket:
            raise ExtractError("duplicate sample_index %d for endpoint %s" % (record.sample_index, key))
        bucket[record.sample_index] = record
    endpoints = []
    for key in sorted(grouped):
        bucket = grouped[key]
        resamples = tuple(bucket[index] for index in RESAMPLE_INDICES if index in bucket)
        endpoints.append(Endpoint(*key, bucket.get(0), resamples))
    return tuple(endpoints)


def _feedback_round_counts(records: Sequence[RawRecord]) -> dict[tuple[str, str, str, str, str], int]:
    """Observed graded feedback rounds per conversation (3 standard, 5 escalation)."""
    counts: dict[tuple[str, str, str, str, str], set[str]] = {}
    for record in records:
        if record.trajectory_kind == "greedy" and record.turn_label in FEEDBACK_TURNS:
            counts.setdefault(_conversation_key(record), set()).add(record.turn_label)
    return {key: len(labels) for key, labels in counts.items()}


def _neutral_baseline_lengths(records: Sequence[RawRecord]) -> dict[tuple[str, str, str, str], int]:
    """Token counts of every same-item ``<difficulty>__accurate__neutral`` measured greedy."""
    lengths: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        if (
            record.trajectory_kind != "greedy"
            or record.sample_index != 0
            or record.turn_label != "measured"
            or record.difficulty is None
            or record.cell_id != record.difficulty + NEUTRAL_BASELINE_CELL_SUFFIX
        ):
            continue
        lengths[(record.run_id, record.model_id, record.immutable_revision, record.task_id)] = len(record.tokens)
    return lengths


def _m2_value(endpoint: Endpoint) -> tuple[MetricValue, int]:
    """Compute M2, degrading an incomplete ensemble to an explicit missing reason."""
    valid = sum(
        1 for record in endpoint.resamples
        if record.final_answer_valid and record.final_answer_letter in OPTIONS
    )
    try:
        result = m2_disagreement(endpoint.resamples)
    except MetricInputError:
        # metrics.m2_disagreement is frozen and raises; the glue owns the policy.
        return MetricValue(None, "m2_incomplete_ensemble"), valid
    return result.disagreement, result.valid_answer_count


def endpoint_metric_row(
    endpoint: Endpoint,
    *,
    protocol: Protocol | None = None,
    neutral_length: int | None = None,
    feedback_rounds: int = 0,
) -> MetricRow:
    """Build one flat metric row; missing inputs become reasons, never zeros."""
    protocol = protocol or load_protocol()
    greedy = endpoint.greedy
    if greedy is None:
        raise ExtractError("endpoint %s has no greedy record" % (endpoint.key,))
    try:
        m1 = m1_margin(greedy, protocol=protocol).margin
    except MetricInputError as error:
        m1 = MetricValue(None, "m1_input_error:" + str(error))
    m3 = m3_for_record(greedy)
    try:
        entropy = partial_entropy(greedy)
        entropy_mean, entropy_worst = entropy.mean_partial_entropy, entropy.highest_entropy_decile_mean
        tail = entropy.mean_tail_mass
        entropy_reason = entropy_mean.missing_reason
    except MetricInputError as error:
        missing = MetricValue(None, "partial_entropy_input_error")
        entropy_mean = entropy_worst = tail = missing
        entropy_reason = "partial_entropy_input_error:" + str(error)
    m2, valid_resamples = _m2_value(endpoint)
    tier_b = tier_b_metrics(greedy.response_text)
    length_tokens = len(greedy.tokens)
    if neutral_length is None:
        drift, drift_reason = None, "length_drift_neutral_endpoint_absent"
    else:
        drift, drift_reason = (length_tokens - neutral_length) / max(1, neutral_length), None
    factorial = greedy.cell_id in protocol.factorial_cell_ids
    return MetricRow(
        phase=greedy.phase,
        run_id=greedy.run_id,
        run_kind=greedy.run_kind,
        model_id=greedy.model_id,
        immutable_revision=greedy.immutable_revision,
        task_id=greedy.task_id,
        split=greedy.split,
        difficulty=greedy.difficulty,
        domain=greedy.domain,
        cell_id=greedy.cell_id,
        cell_kind="factorial" if factorial else "non_factorial",
        feedback_validity=greedy.feedback_validity,
        tone=greedy.tone,
        turn_label=greedy.turn_label,
        response_id=greedy.response_id,
        m1=m1.value,
        m1_missing_reason=m1.missing_reason,
        m2=m2.value,
        m2_missing_reason=m2.missing_reason,
        m3_rate=m3.rate_per_100_tokens.value,
        m3_missing_reason=m3.rate_per_100_tokens.missing_reason,
        m3_event_count=m3.event_count,
        m3_loop_flag=m3.loop_flag,
        entropy_mean=entropy_mean.value,
        entropy_worst_decile=entropy_worst.value,
        tail_mass_mean=tail.value,
        entropy_missing_reason=entropy_reason,
        rep4=repeated_4gram_rate(greedy.response_text),
        length_tokens=length_tokens,
        length_drift=drift,
        length_drift_missing_reason=drift_reason,
        hedge_per100=tier_b.hedging_per_100_tokens.value,
        selfcorr_per100=tier_b.self_correction_per_100_tokens.value,
        greedy_answer_valid=greedy.final_answer_valid,
        greedy_answer_correct=greedy.final_answer_correct,
        greedy_answer_letter=greedy.final_answer_letter,
        resample_count=len(endpoint.resamples),
        resample_valid_count=valid_resamples,
        history_false_negative=greedy.feedback_history_false_negative,
        feedback_rounds=feedback_rounds,
    )


class _M2Stub(NamedTuple):
    """The only fields M2 reads, kept so ensembles cost bytes instead of tokens."""

    run_id: str
    model_id: str
    immutable_revision: str
    task_id: str
    cell_id: str
    turn_label: str
    trajectory_kind: str
    sample_index: int
    final_answer_valid: bool
    final_answer_letter: str | None


def _stub(record: RawRecord) -> _M2Stub:
    return _M2Stub(
        record.run_id, record.model_id, record.immutable_revision, record.task_id,
        record.cell_id, record.turn_label, record.trajectory_kind, record.sample_index,
        record.final_answer_valid, record.final_answer_letter,
    )


def build_metric_rows(records: Iterable[RawRecord], *, protocol: Protocol | None = None) -> tuple[MetricRow, ...]:
    """Turn validated raw records into the committed flat endpoint table.

    Streams: each record is reduced to its metric contribution and released, so
    peak memory scales with the number of endpoints, not with the gigabytes of
    token logprobs that produced them.  Length drift and the observed feedback
    round count are cross-record, so they are filled in once the stream ends.
    """
    protocol = protocol or load_protocol()
    greedy: dict[tuple, dict[str, Any]] = {}
    ensembles: dict[tuple, dict[int, _M2Stub]] = {}
    rounds: dict[tuple, set[str]] = {}
    neutral: dict[tuple, int] = {}
    for record in records:
        conversation = _conversation_key(record)
        if record.trajectory_kind == "greedy" and record.turn_label in FEEDBACK_TURNS:
            rounds.setdefault(conversation, set()).add(record.turn_label)
            continue
        if record.turn_label not in ENDPOINT_TURNS:
            continue
        key = conversation + (record.turn_label,)
        if record.trajectory_kind == "greedy" and record.sample_index == 0:
            if key in greedy:
                raise ExtractError("duplicate sample_index 0 for endpoint %s" % (key,))
            greedy[key] = _greedy_fields(record, protocol)
            if (
                record.turn_label == "measured" and record.difficulty is not None
                and record.cell_id == record.difficulty + NEUTRAL_BASELINE_CELL_SUFFIX
            ):
                neutral[conversation[:4]] = greedy[key]["length_tokens"]
            continue
        bucket = ensembles.setdefault(key, {})
        if record.sample_index in bucket:
            raise ExtractError("duplicate sample_index %d for endpoint %s" % (record.sample_index, key))
        bucket[record.sample_index] = _stub(record)
    rows = []
    for key in sorted(greedy):
        # A resample-only endpoint carries no confirmatory metric and is absent
        # from the table; the QC table records the gap through that absence.
        fields = greedy[key]
        resamples = tuple(bucket for _, bucket in sorted(ensembles.get(key, {}).items()))
        m2, valid = _m2_from(resamples)
        baseline = neutral.get(key[:4])
        rows.append(MetricRow(
            m2=m2.value, m2_missing_reason=m2.missing_reason,
            resample_count=len(resamples), resample_valid_count=valid,
            length_drift=None if baseline is None else (fields["length_tokens"] - baseline) / max(1, baseline),
            length_drift_missing_reason=None if baseline is not None else "length_drift_neutral_endpoint_absent",
            feedback_rounds=len(rounds.get(key[:5], ())),
            **fields,
        ))
    return tuple(rows)


def _m2_from(resamples):
    valid = sum(1 for item in resamples if item.final_answer_valid and item.final_answer_letter in OPTIONS)
    try:
        result = m2_disagreement(resamples)
    except MetricInputError:
        # metrics.m2_disagreement is frozen and raises; the glue owns the policy.
        return MetricValue(None, "m2_incomplete_ensemble"), valid
    return result.disagreement, result.valid_answer_count


def _greedy_fields(record: RawRecord, protocol: Protocol) -> dict[str, Any]:
    """Every greedy-derived column, computed before the record is released."""
    try:
        m1 = m1_margin(record, protocol=protocol).margin
    except MetricInputError as error:
        m1 = MetricValue(None, "m1_input_error:" + str(error))
    m3 = m3_for_record(record)
    try:
        entropy = partial_entropy(record)
        entropy_mean, entropy_worst = entropy.mean_partial_entropy, entropy.highest_entropy_decile_mean
        tail = entropy.mean_tail_mass
        entropy_reason = entropy_mean.missing_reason
    except MetricInputError as error:
        missing = MetricValue(None, "partial_entropy_input_error")
        entropy_mean = entropy_worst = tail = missing
        entropy_reason = "partial_entropy_input_error:" + str(error)
    tier_b = tier_b_metrics(record.response_text)
    return {
        "phase": record.phase, "run_id": record.run_id, "run_kind": record.run_kind,
        "model_id": record.model_id, "immutable_revision": record.immutable_revision,
        "task_id": record.task_id, "split": record.split, "difficulty": record.difficulty,
        "domain": record.domain, "cell_id": record.cell_id,
        "cell_kind": "factorial" if record.cell_id in protocol.factorial_cell_ids else "non_factorial",
        "feedback_validity": record.feedback_validity, "tone": record.tone,
        "turn_label": record.turn_label, "response_id": record.response_id,
        "m1": m1.value, "m1_missing_reason": m1.missing_reason,
        "m3_rate": m3.rate_per_100_tokens.value, "m3_missing_reason": m3.rate_per_100_tokens.missing_reason,
        "m3_event_count": m3.event_count, "m3_loop_flag": m3.loop_flag,
        "entropy_mean": entropy_mean.value, "entropy_worst_decile": entropy_worst.value,
        "tail_mass_mean": tail.value, "entropy_missing_reason": entropy_reason,
        "rep4": repeated_4gram_rate(record.response_text), "length_tokens": len(record.tokens),
        "hedge_per100": tier_b.hedging_per_100_tokens.value,
        "selfcorr_per100": tier_b.self_correction_per_100_tokens.value,
        "greedy_answer_valid": record.final_answer_valid,
        "greedy_answer_correct": record.final_answer_correct,
        "greedy_answer_letter": record.final_answer_letter,
        "history_false_negative": record.feedback_history_false_negative,
    }


def qc_by_cell(
    rows: Sequence[MetricRow],
    *,
    excluded_items: Mapping[str, Iterable[str]] | None = None,
) -> tuple[QcRow, ...]:
    """Condition-wise invalid-answer and metric-missing rates, by cell and turn.

    ``excluded_items`` maps a model to the item IDs amendment A2 drops for it;
    the rates are reported over every endpoint present, and the excluded items
    are named alongside so a thin cell can be read against them.
    """
    excluded = {model: sorted(set(items)) for model, items in (excluded_items or {}).items()}
    grouped: dict[tuple[str, str, str, str, str], list[MetricRow]] = {}
    for row in rows:
        grouped.setdefault((row.phase, row.run_id, row.model_id, row.cell_id, row.turn_label), []).append(row)
    out = []
    for key in sorted(grouped):
        group = grouped[key]
        total = len(group)
        invalid = sum(1 for row in group if not row.greedy_answer_valid)
        missing = {name: sum(1 for row in group if row.metric(name)[0] is None) for name in ("M1", "M2", "M3")}
        resample_total = sum(row.resample_count for row in group)
        resample_invalid = resample_total - sum(row.resample_valid_count for row in group)
        dropped = excluded.get(key[2], [])
        out.append(QcRow(
            *key, total, invalid, invalid / total,
            missing["M1"], missing["M1"] / total,
            missing["M2"], missing["M2"] / total,
            missing["M3"], missing["M3"] / total,
            resample_total, resample_invalid,
            resample_invalid / resample_total if resample_total else 0.0,
            len(dropped), ";".join(dropped),
        ))
    return tuple(out)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_table(path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write one deterministic UTF-8 CSV with LF endings for stable diffs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row[column]) for column in columns])
    return path


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return path


def write_summaries(
    rows: Sequence[MetricRow],
    out_dir: str | Path,
    *,
    excluded_items: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Path]:
    """Emit the committed ``metric_rows.csv``/``.jsonl`` and ``qc_by_cell.csv``.

    ``metric_rows`` always holds every extracted endpoint: amendment A2 is an
    analysis-stage exclusion, so it is reported in the QC table rather than
    deleted from the extraction record.
    """
    out_dir = Path(out_dir)
    payload = [row.to_dict() for row in rows]
    qc = qc_by_cell(rows, excluded_items=excluded_items)
    return {
        "metric_rows_csv": write_table(out_dir / "metric_rows.csv", METRIC_ROW_COLUMNS, payload),
        "metric_rows_jsonl": write_jsonl(out_dir / "metric_rows.jsonl", payload),
        "qc_by_cell_csv": write_table(out_dir / "qc_by_cell.csv", QC_ROW_COLUMNS, [item.to_dict() for item in qc]),
    }


def _typed(value: str, name: str) -> Any:
    if value == "":
        return None
    annotation = {field.name: field.type for field in fields(MetricRow)}[name]
    text = annotation if isinstance(annotation, str) else str(annotation)
    if "bool" in text:
        return value == "true"
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value


def read_metric_rows(path: str | Path) -> tuple[MetricRow, ...]:
    """Read back a committed ``metric_rows.csv`` so figures never touch raw data."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in jsonl_lines(path.read_text(encoding="utf-8")) if line.strip()]
        return tuple(MetricRow(**row) for row in rows)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return tuple(
            MetricRow(**{name: _typed(row[name], name) for name in METRIC_ROW_COLUMNS})
            for row in csv.DictReader(handle)
        )
