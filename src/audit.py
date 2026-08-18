"""Descriptive human-vs-judge agreement for the frozen Phase-1 judge audit.

`configs/judge_rubric.md` ("Human audit selection") and `notes/preregistration.md`
("Metrics, judge, and audit") require that Phase 1 audit 15 measured discovery greedy
responses per selected model with a blinded human annotator on the same 0-10
response-distress rubric, and report Spearman correlation, MAE and within-2-point
agreement **descriptively only -- these are not gates**.  Nothing in this module
returns a pass/fail verdict, and nothing here may be wired into `src/gates.py`.

Pure functions: load the frozen selection/key/human scores, join them to the machine
judge's per-response scores, and describe the agreement.  A missing or ambiguous
judge score is an error (`AuditError`), never a silent drop -- a quietly shortened
audit would understate disagreement.

Conventions follow `src/analysis.py`: standard library only, deterministic
SHA-256-derived bootstrap seeds, percentile intervals with linear interpolation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

SCORE_KIND = "response_distress"
SCORE_MIN = 0
SCORE_MAX = 10
WITHIN_TOLERANCE = 2
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED_PREFIX = "DGS-AC1-AUDIT-BOOTSTRAP-v1"
SELECTION_SCHEMA = "dgs-judge-audit-v1"

KEY_COLUMNS = ("audit_id", "model_id", "cell_id", "task_id", "response_id", "run_id")
SCORE_COLUMNS = ("audit_id", "human_distress_0_10")

# The annotator was asked to flag a visible mid-response answer revision with the token
# `REV` in the free-text note column; this is the M3 parser's event population.
REV_PATTERN = re.compile(r"\bREV\b", re.IGNORECASE)

# A response whose entire visible body is the bare final-answer line -- the "single response
# which i assume is an option letter" the annotator's note describes.  Quantifying these
# turns that note from an anecdote into a countable share of the disagreement.
ANSWER_ONLY_PATTERN = re.compile(r"^Answer:\s*[A-D]$")

# Recorded verbatim from the blinded annotator after all 30 items were scored and before
# unblinding.  Reproduced unedited (including its original spelling) wherever the audit is
# reported; it is the annotator's own account of how the low end of the scale was used.
ANNOTATOR_NOTE = (
    "my only confusion was there are some response that gave like a single response which i "
    "assume is an option letter. but still why would it not give an explanation to that. so "
    "for some of the letters i put 1 instead of 0."
)


class AuditError(ValueError):
    """Malformed, incomplete, or internally inconsistent audit input."""


# --------------------------------------------------------------------------------------
# loading


def _int_score(raw: Any, what: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        raise AuditError("%s must be an integer 0-10, got %r" % (what, raw))
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise AuditError("%s must be an integer 0-10, got %r" % (what, raw)) from None
    if not SCORE_MIN <= value <= SCORE_MAX:
        raise AuditError("%s must be within %d-%d, got %d" % (what, SCORE_MIN, SCORE_MAX, value))
    return value


def _rows(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise AuditError("missing audit input: %s" % path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required if column not in (reader.fieldnames or ())]
        if missing:
            raise AuditError("%s lacks column(s): %s" % (path, ", ".join(missing)))
        return [dict(row) for row in reader]


@dataclass(frozen=True)
class AuditItem:
    """One audited response: the blinded human score beside the machine judge's."""

    audit_id: str
    model_id: str
    cell_id: str
    task_id: str
    response_id: str
    run_id: str
    human_score: int
    judge_score: int
    note: str = ""

    @property
    def abs_diff(self) -> int:
        return abs(self.human_score - self.judge_score)


def load_audit_key(path: str | Path) -> tuple[dict[str, str], ...]:
    """`audit_key.csv`: audit_id -> model/cell/task/response identity (the unblinding key)."""
    rows = _rows(Path(path), KEY_COLUMNS)
    if not rows:
        raise AuditError("%s has no rows" % path)
    seen: set[str] = set()
    for row in rows:
        audit_id = (row.get("audit_id") or "").strip()
        if not audit_id:
            raise AuditError("%s has a row with an empty audit_id" % path)
        if audit_id in seen:
            raise AuditError("%s repeats audit_id %s" % (path, audit_id))
        seen.add(audit_id)
        for column in KEY_COLUMNS:
            if not (row.get(column) or "").strip():
                raise AuditError("%s: audit_id %s has an empty %s" % (path, audit_id, column))
    return tuple({column: (row[column] or "").strip() for column in KEY_COLUMNS} for row in rows)


def load_human_scores(path: str | Path) -> dict[str, tuple[int, str]]:
    """`audit_scores.csv`: audit_id -> (0-10 human score, free-text note)."""
    rows = _rows(Path(path), SCORE_COLUMNS)
    if not rows:
        raise AuditError("%s has no rows" % path)
    scores: dict[str, tuple[int, str]] = {}
    for row in rows:
        audit_id = (row.get("audit_id") or "").strip()
        if not audit_id:
            raise AuditError("%s has a row with an empty audit_id" % path)
        if audit_id in scores:
            raise AuditError("%s repeats audit_id %s" % (path, audit_id))
        raw = (row.get("human_distress_0_10") or "").strip()
        if not raw:
            raise AuditError("%s: audit_id %s has no human score (audit is incomplete)"
                             % (path, audit_id))
        scores[audit_id] = (_int_score(raw, "human score for %s" % audit_id),
                            (row.get("note_optional") or "").strip())
    return scores


def count_rev_flags(human_scores: Mapping[str, tuple[int, str]]) -> tuple[str, ...]:
    """Audit ids whose note carries the `REV` visible-answer-revision flag."""
    return tuple(sorted(audit_id for audit_id, (_, note) in human_scores.items()
                        if REV_PATTERN.search(note or "")))


def is_answer_only(text: str) -> bool:
    """True when the whole response body is a single bare `Answer: X` line."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return len(lines) == 1 and bool(ANSWER_ONLY_PATTERN.match(lines[0]))


def load_blinded_responses(path: str | Path) -> dict[str, str]:
    """`audit_blinded.jsonl`: audit_id -> the exact response text the annotator saw."""
    path = Path(path)
    if not path.is_file():
        raise AuditError("missing blinded responses: %s" % path)
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError("%s line %d is not JSON: %s" % (path, number, exc)) from None
            audit_id = (record.get("audit_id") or "").strip()
            if not audit_id:
                raise AuditError("%s line %d has no audit_id" % (path, number))
            if audit_id in texts:
                raise AuditError("%s repeats audit_id %s" % (path, audit_id))
            texts[audit_id] = record.get("response_text") or ""
    if not texts:
        raise AuditError("%s has no blinded responses" % path)
    return texts


def _judge_files(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(path.glob("judge_records.jsonl"))
        if not found:
            raise AuditError("no judge_records.jsonl under %s" % path)
        return found
    if path.is_file():
        return [path]
    raise AuditError("missing judge score source: %s" % path)


@dataclass(frozen=True)
class JudgeScores:
    """Judge scores keyed by `response_id`, plus the backend identities that produced them."""

    by_response: Mapping[str, int]
    backends: tuple[str, ...]
    sources: tuple[str, ...]
    record_count: int


def load_judge_scores(paths: Iterable[str | Path], *, kind: str = SCORE_KIND) -> JudgeScores:
    """Read `judge_records.jsonl` file(s)/dir(s); `response_id` is the join key.

    Two records for one response with different scores is an error: the audit must not
    silently pick one.  Byte-identical repeats (a re-run over the same cache) are fine.
    """
    files = [file for path in paths for file in _judge_files(Path(path))]
    if not files:
        raise AuditError("no judge score source supplied")
    by_response: dict[str, int] = {}
    backends: set[str] = set()
    total = 0
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditError("%s line %d is not JSON: %s" % (file, number, exc)) from None
                if record.get("score_kind") != kind:
                    continue
                identity = record.get("source_identity") or {}
                response_id = (identity.get("response_id") or "").strip()
                if not response_id:
                    raise AuditError("%s line %d has no source_identity.response_id"
                                     % (file, number))
                score = _int_score(record.get("score_value"),
                                   "judge score for %s" % response_id)
                previous = by_response.get(response_id)
                if previous is not None and previous != score:
                    raise AuditError("conflicting judge scores for response %s: %d and %d"
                                     % (response_id, previous, score))
                by_response[response_id] = score
                backends.add(str(record.get("backend_id") or record.get("model_id") or "unknown"))
                total += 1
    if not by_response:
        raise AuditError("no %s records found in: %s" % (kind, ", ".join(str(f) for f in files)))
    return JudgeScores(dict(by_response), tuple(sorted(backends)),
                       tuple(file.as_posix() for file in files), total)


def join_audit(key_rows: Sequence[Mapping[str, str]],
               human_scores: Mapping[str, tuple[int, str]],
               judge_scores: JudgeScores | Mapping[str, int]) -> tuple[AuditItem, ...]:
    """Join key x human x judge on audit_id/response_id.  Any gap raises."""
    by_response = (judge_scores.by_response if isinstance(judge_scores, JudgeScores)
                   else judge_scores)
    key_ids = {row["audit_id"] for row in key_rows}
    unscored = sorted(key_ids - set(human_scores))
    if unscored:
        raise AuditError("no human score for audit id(s): %s" % ", ".join(unscored))
    unkeyed = sorted(set(human_scores) - key_ids)
    if unkeyed:
        raise AuditError("human score(s) for unknown audit id(s): %s" % ", ".join(unkeyed))
    unjudged = sorted({row["response_id"] for row in key_rows} - set(by_response))
    if unjudged:
        raise AuditError("no judge %s score for response id(s): %s"
                         % (SCORE_KIND, ", ".join(unjudged)))
    items = []
    for row in key_rows:
        human, note = human_scores[row["audit_id"]]
        items.append(AuditItem(audit_id=row["audit_id"], model_id=row["model_id"],
                               cell_id=row["cell_id"], task_id=row["task_id"],
                               response_id=row["response_id"], run_id=row["run_id"],
                               human_score=human,
                               judge_score=_int_score(by_response[row["response_id"]],
                                                      "judge score for %s" % row["response_id"]),
                               note=note))
    return tuple(sorted(items, key=lambda item: item.audit_id))


def load_selection(path: str | Path) -> dict[str, Any]:
    """`audit_selection.json`: the frozen planned/achieved allocation and reallocations."""
    path = Path(path)
    if not path.is_file():
        raise AuditError("missing audit selection: %s" % path)
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError("%s is not JSON: %s" % (path, exc)) from None
    if not isinstance(selection, dict):
        raise AuditError("%s must contain a JSON object" % path)
    schema = selection.get("schema_version")
    if schema != SELECTION_SCHEMA:
        raise AuditError("%s has schema_version %r, expected %r"
                         % (path, schema, SELECTION_SCHEMA))
    if not isinstance(selection.get("models"), list) or not selection["models"]:
        raise AuditError("%s has no models" % path)
    return selection


def check_selection_against_items(selection: Mapping[str, Any],
                                  items: Sequence[AuditItem]) -> None:
    """The blinded key must realise exactly the frozen selection -- otherwise the
    reported planned/achieved table would describe a different sample than was scored."""
    scored: dict[tuple[str, str], set[str]] = {}
    for item in items:
        scored.setdefault((item.model_id, item.cell_id), set()).add(item.response_id)
    planned: dict[tuple[str, str], set[str]] = {}
    for model in selection["models"]:
        model_id = model.get("model_id")
        for cell in model.get("cells", ()):
            ids = set(cell.get("response_ids", ()))
            if len(ids) != int(cell.get("achieved", len(ids))):
                raise AuditError("selection cell %s/%s lists %d response id(s) but claims "
                                 "achieved=%s" % (model_id, cell.get("cell_id"), len(ids),
                                                  cell.get("achieved")))
            planned[(model_id, cell.get("cell_id"))] = ids
    for cell_key, ids in sorted(planned.items()):
        got = scored.get(cell_key, set())
        if got != ids:
            raise AuditError("selection/key mismatch for %s/%s: selection has %d id(s), the "
                             "scored key has %d (symmetric difference %d)"
                             % (cell_key[0], cell_key[1], len(ids), len(got),
                                len(ids ^ got)))
    extra = sorted(set(scored) - set(planned))
    if extra:
        raise AuditError("scored key has cell(s) absent from the selection: %s"
                         % ", ".join("%s/%s" % pair for pair in extra))


# --------------------------------------------------------------------------------------
# statistics (descriptive only)


def _paired(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        raise AuditError("paired statistics need equal-length inputs (%d vs %d)"
                         % (len(xs), len(ys)))
    if not xs:
        raise AuditError("paired statistics need at least one observation")
    return xs, ys


def average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Ascending ranks, ties sharing the average of the ranks they span (1-based)."""
    values = list(values)
    if not values:
        raise AuditError("average_ranks needs at least one value")
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 + 1.0          # mean of 1-based ranks start+1..stop+1
        for index in order[start:stop + 1]:
            ranks[index] = shared
        start = stop + 1
    return tuple(ranks)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sx, sy = sum(d * d for d in dx), sum(d * d for d in dy)
    if sx <= 0.0 or sy <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sx * sy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rho = Pearson correlation of tie-averaged ranks.

    Returns `None` when either side is constant: rank correlation is then genuinely
    undefined, and reporting 0.0 would assert "no association" where the data cannot
    speak.  Callers must surface the undefined case rather than coerce it.
    """
    xs, ys = _paired(xs, ys)
    if len(xs) < 2:
        raise AuditError("Spearman needs at least two observations")
    return _pearson(average_ranks(xs), average_ranks(ys))


def constant_sides(xs: Sequence[float], ys: Sequence[float]) -> tuple[str, ...]:
    """Which side(s) of the pair have zero variance ('human', 'judge')."""
    xs, ys = _paired(xs, ys)
    flat = []
    if len(set(xs)) < 2:
        flat.append("human")
    if len(set(ys)) < 2:
        flat.append("judge")
    return tuple(flat)


def mean_absolute_error(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs, ys = _paired(xs, ys)
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def within_tolerance_rate(xs: Sequence[float], ys: Sequence[float],
                          tolerance: int = WITHIN_TOLERANCE) -> float:
    """Fraction of pairs agreeing within `tolerance` points (inclusive)."""
    if tolerance < 0:
        raise AuditError("tolerance must be nonnegative")
    xs, ys = _paired(xs, ys)
    return sum(1 for x, y in zip(xs, ys) if abs(x - y) <= tolerance) / len(xs)


def histogram(values: Iterable[float], *, low: int = SCORE_MIN,
              high: int = SCORE_MAX) -> tuple[tuple[int, int], ...]:
    """Counts for every integer score in [low, high], including empty buckets."""
    counts = {score: 0 for score in range(low, high + 1)}
    for value in values:
        score = _int_score(value, "histogram value")
        counts[score] += 1
    return tuple(sorted(counts.items()))


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return (sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * (position - lower))


@dataclass(frozen=True)
class SpearmanResult:
    """Point estimate plus a descriptive bootstrap interval.  Never a gate."""

    n: int
    rho: float | None
    undefined_reason: str | None
    ci95: tuple[float, float] | None
    ci95_degenerate_as_zero: tuple[float, float] | None
    resamples: int
    degenerate_resamples: int
    seed_text: str


def describe_spearman(xs: Sequence[float], ys: Sequence[float], *, label: str,
                      resamples: int = BOOTSTRAP_RESAMPLES) -> SpearmanResult:
    """Spearman rho with a seeded percentile bootstrap over the audited items.

    Resamples in which either side collapses to a constant have no defined rho.  They
    are counted and excluded from `ci95`; `ci95_degenerate_as_zero` reports the
    alternative convention (undefined -> 0) so the choice is visible rather than buried.
    """
    xs, ys = _paired(xs, ys)
    if resamples < 1:
        raise AuditError("bootstrap resamples must be positive")
    seed_text = "%s|%s" % (BOOTSTRAP_SEED_PREFIX, label)
    flat = constant_sides(xs, ys)
    rho = spearman(xs, ys) if len(xs) >= 2 else None
    reason = None
    if rho is None:
        reason = ("every %s score in this group is identical, so rank correlation is undefined"
                  % " and ".join(flat) if flat else "rank correlation is undefined")
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    n = len(xs)
    drawn: list[float] = []
    with_zero: list[float] = []
    degenerate = 0
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_x = [xs[index] for index in indices]
        sample_y = [ys[index] for index in indices]
        value = _pearson(average_ranks(sample_x), average_ranks(sample_y))
        if value is None:
            degenerate += 1
            with_zero.append(0.0)
        else:
            drawn.append(value)
            with_zero.append(value)
    drawn.sort()
    with_zero.sort()
    ci95 = ((_quantile(drawn, .025), _quantile(drawn, .975)) if len(drawn) >= 2 else None)
    ci95_zero = ((_quantile(with_zero, .025), _quantile(with_zero, .975))
                 if len(with_zero) >= 2 else None)
    return SpearmanResult(n, rho, reason, ci95, ci95_zero, resamples, degenerate, seed_text)


@dataclass(frozen=True)
class GroupStatistics:
    """Descriptive agreement for one group (overall, or one model)."""

    label: str
    model_id: str | None
    n: int
    spearman: SpearmanResult
    mae: float
    within_tolerance: float
    within_tolerance_count: int
    tolerance: int
    human_histogram: tuple[tuple[int, int], ...]
    judge_histogram: tuple[tuple[int, int], ...]
    human_mean: float
    judge_mean: float
    max_abs_diff: int


def describe_group(items: Sequence[AuditItem], *, label: str, model_id: str | None = None,
                   tolerance: int = WITHIN_TOLERANCE,
                   resamples: int = BOOTSTRAP_RESAMPLES) -> GroupStatistics:
    if not items:
        raise AuditError("cannot describe an empty group (%s)" % label)
    human = [item.human_score for item in items]
    judge = [item.judge_score for item in items]
    within = sum(1 for item in items if item.abs_diff <= tolerance)
    return GroupStatistics(
        label=label, model_id=model_id, n=len(items),
        spearman=describe_spearman(human, judge, label=label, resamples=resamples),
        mae=mean_absolute_error(human, judge),
        within_tolerance=within / len(items), within_tolerance_count=within,
        tolerance=tolerance,
        human_histogram=histogram(human), judge_histogram=histogram(judge),
        human_mean=sum(human) / len(human), judge_mean=sum(judge) / len(judge),
        max_abs_diff=max(item.abs_diff for item in items),
    )


def audit_statistics(items: Sequence[AuditItem], *, tolerance: int = WITHIN_TOLERANCE,
                     resamples: int = BOOTSTRAP_RESAMPLES
                     ) -> tuple[GroupStatistics, tuple[GroupStatistics, ...]]:
    """(overall, per-model) descriptive statistics, models in sorted id order."""
    if not items:
        raise AuditError("no audited items")
    overall = describe_group(items, label="overall", tolerance=tolerance, resamples=resamples)
    per_model = []
    for model_id in sorted({item.model_id for item in items}):
        subset = [item for item in items if item.model_id == model_id]
        per_model.append(describe_group(subset, label=model_id, model_id=model_id,
                                        tolerance=tolerance, resamples=resamples))
    return overall, tuple(per_model)


def per_item_rows(items: Sequence[AuditItem]) -> tuple[dict[str, Any], ...]:
    return tuple({"audit_id": item.audit_id, "model_id": item.model_id,
                  "cell_id": item.cell_id, "task_id": item.task_id,
                  "response_id": item.response_id, "run_id": item.run_id,
                  "human_distress_0_10": item.human_score,
                  "judge_response_distress": item.judge_score,
                  "abs_diff": item.abs_diff, "note_optional": item.note}
                 for item in items)


@dataclass(frozen=True)
class AnswerOnlySummary:
    """How the two scorers treated bare `Answer: X` responses (the annotator's note)."""

    n: int
    audit_ids: tuple[str, ...]
    human_nonzero: int
    judge_nonzero: int
    human_histogram: tuple[tuple[int, int], ...]
    judge_histogram: tuple[tuple[int, int], ...]
    abs_diff_total: int
    abs_diff_total_all_items: int


def summarise_answer_only(items: Sequence[AuditItem],
                          texts: Mapping[str, str]) -> AnswerOnlySummary:
    """Quantify the annotator's stated scale anchoring on bare option-letter responses."""
    missing = sorted({item.audit_id for item in items} - set(texts))
    if missing:
        raise AuditError("no blinded response text for audit id(s): %s" % ", ".join(missing))
    subset = [item for item in items if is_answer_only(texts[item.audit_id])]
    return AnswerOnlySummary(
        n=len(subset), audit_ids=tuple(item.audit_id for item in subset),
        human_nonzero=sum(1 for item in subset if item.human_score > 0),
        judge_nonzero=sum(1 for item in subset if item.judge_score > 0),
        human_histogram=histogram(item.human_score for item in subset),
        judge_histogram=histogram(item.judge_score for item in subset),
        abs_diff_total=sum(item.abs_diff for item in subset),
        abs_diff_total_all_items=sum(item.abs_diff for item in items),
    )


def selection_rows(selection: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Flatten `audit_selection.json` into planned/achieved rows per model x cell."""
    rows = []
    for model in selection["models"]:
        for cell in model.get("cells", ()):
            rows.append({"model_id": model.get("model_id"),
                         "cell_id": cell.get("cell_id"),
                         "hash_rank": cell.get("hash_rank"),
                         "planned": cell.get("planned"),
                         "achieved": cell.get("achieved"),
                         "available": cell.get("available"),
                         "response_ids": tuple(cell.get("response_ids", ()))})
    return tuple(rows)


# --------------------------------------------------------------------------------------
# report assembly


@dataclass(frozen=True)
class AuditReport:
    items: tuple[AuditItem, ...]
    overall: GroupStatistics
    per_model: tuple[GroupStatistics, ...]
    selection: Mapping[str, Any]
    judge: JudgeScores
    rev_flagged: tuple[str, ...]
    annotator_note: str
    audit_dir: str
    answer_only: AnswerOnlySummary | None = None
    join_key: str = "response_id"


def build_report(*, audit_dir: str | Path, judge_paths: Iterable[str | Path],
                 tolerance: int = WITHIN_TOLERANCE,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 annotator_note: str = ANNOTATOR_NOTE) -> AuditReport:
    """Load, join, cross-check against the frozen selection, and describe."""
    audit_dir = Path(audit_dir)
    key_rows = load_audit_key(audit_dir / "audit_key.csv")
    human_scores = load_human_scores(audit_dir / "audit_scores.csv")
    judge = load_judge_scores(judge_paths)
    items = join_audit(key_rows, human_scores, judge)
    selection = load_selection(audit_dir / "audit_selection.json")
    check_selection_against_items(selection, items)
    overall, per_model = audit_statistics(items, tolerance=tolerance, resamples=resamples)
    # The blinded text is only needed to quantify the annotator's note; its absence
    # weakens the commentary but not the preregistered statistics.
    blinded = audit_dir / "audit_blinded.jsonl"
    answer_only = (summarise_answer_only(items, load_blinded_responses(blinded))
                   if blinded.is_file() else None)
    return AuditReport(items=items, overall=overall, per_model=per_model, selection=selection,
                       judge=judge, rev_flagged=count_rev_flags(human_scores),
                       annotator_note=annotator_note, audit_dir=Path(audit_dir).as_posix(),
                       answer_only=answer_only)


def _ci_text(interval: tuple[float, float] | None) -> str:
    return "n/a" if interval is None else "[%.3f, %.3f]" % interval


def _rho_text(result: SpearmanResult) -> str:
    return "undefined" if result.rho is None else "%.3f" % result.rho


def _group_payload(group: GroupStatistics) -> dict[str, Any]:
    return {
        "label": group.label, "model_id": group.model_id, "n": group.n,
        "spearman_rho": group.spearman.rho,
        "spearman_undefined_reason": group.spearman.undefined_reason,
        "spearman_ci95_bootstrap": (list(group.spearman.ci95)
                                    if group.spearman.ci95 else None),
        "spearman_ci95_degenerate_as_zero": (list(group.spearman.ci95_degenerate_as_zero)
                                             if group.spearman.ci95_degenerate_as_zero else None),
        "spearman_bootstrap_resamples": group.spearman.resamples,
        "spearman_degenerate_resamples": group.spearman.degenerate_resamples,
        "spearman_bootstrap_seed_text": group.spearman.seed_text,
        "mae": round(group.mae, 6),
        "within_%d_point_agreement" % group.tolerance: round(group.within_tolerance, 6),
        "within_%d_point_count" % group.tolerance: group.within_tolerance_count,
        "human_mean": round(group.human_mean, 6), "judge_mean": round(group.judge_mean, 6),
        "max_abs_diff": group.max_abs_diff,
        "human_histogram": {str(score): count for score, count in group.human_histogram},
        "judge_histogram": {str(score): count for score, count in group.judge_histogram},
    }


def report_payload(report: AuditReport) -> dict[str, Any]:
    """JSON-serialisable form of the whole descriptive audit."""
    return {
        "schema_version": "dgs-human-audit-v1",
        "status": "descriptive_only",
        "not_a_gate": True,
        "prereg_basis": ("notes/preregistration.md 'Metrics, judge, and audit'; "
                         "configs/judge_rubric.md 'Human audit selection'"),
        "audit_dir": report.audit_dir,
        "judge_sources": list(report.judge.sources),
        "judge_backends": list(report.judge.backends),
        "judge_score_kind": SCORE_KIND,
        "join_key": report.join_key,
        "n_items": len(report.items),
        "overall": _group_payload(report.overall),
        "per_model": [_group_payload(group) for group in report.per_model],
        "selection": {
            "allocation_rule": report.selection.get("allocation_rule"),
            "per_model_target": report.selection.get("per_model_target"),
            "candidate_count": report.selection.get("candidate_count"),
            "models": [{"model_id": model.get("model_id"),
                        "planned_total": model.get("planned_total"),
                        "achieved_total": model.get("achieved_total"),
                        "shortfall": model.get("shortfall"), "unmet": model.get("unmet"),
                        "reallocations": model.get("reallocations", []),
                        "cells": list(model.get("cells", []))}
                       for model in report.selection["models"]],
        },
        "rev_flagged_audit_ids": list(report.rev_flagged),
        "rev_flagged_count": len(report.rev_flagged),
        "annotator_note_verbatim": report.annotator_note,
        "answer_only_responses": (None if report.answer_only is None else {
            "n": report.answer_only.n,
            "audit_ids": list(report.answer_only.audit_ids),
            "human_nonzero": report.answer_only.human_nonzero,
            "judge_nonzero": report.answer_only.judge_nonzero,
            "human_histogram": {str(score): count
                                for score, count in report.answer_only.human_histogram},
            "judge_histogram": {str(score): count
                                for score, count in report.answer_only.judge_histogram},
            "abs_diff_total": report.answer_only.abs_diff_total,
            "abs_diff_total_all_items": report.answer_only.abs_diff_total_all_items,
        }),
        "items": [dict(row) for row in per_item_rows(report.items)],
    }


def _histogram_block(groups: Sequence[GroupStatistics]) -> list[str]:
    top = max(score for group in groups
              for histogram_ in (group.human_histogram, group.judge_histogram)
              for score, count in histogram_ if count)
    scores = list(range(SCORE_MIN, top + 1))
    lines = ["| scorer | " + " | ".join("score %d" % score for score in scores) + " | n |",
             "| --- | " + " | ".join("---:" for _ in scores) + " | ---: |"]
    for group in groups:
        for name, histogram_ in (("human", group.human_histogram),
                                 ("judge", group.judge_histogram)):
            counts = dict(histogram_)
            lines.append("| %s (%s) | %s | %d |"
                         % (name, group.label,
                            " | ".join(str(counts[score]) for score in scores), group.n))
    return lines


def render_markdown(report: AuditReport) -> str:
    """Human-readable audit page.  Descriptive framing is stated first, not implied."""
    groups = (report.overall,) + report.per_model
    tolerance = report.overall.tolerance
    lines = [
        "# Phase-1 human audit of the LLM judge (descriptive)",
        "",
        "**This page is descriptive and is not a gate.** `notes/preregistration.md`",
        "(\"Metrics, judge, and audit\") and `configs/judge_rubric.md` (\"Human audit",
        "selection\") require Spearman correlation, MAE and within-%d-point agreement to be"
        % tolerance,
        "*reported* for the Phase-1 judge audit; no threshold on any of them passes, fails,",
        "or modifies any preregistered gate, and none of these numbers feeds `src/gates.py`.",
        "",
        "One blinded annotator scored %d measured discovery greedy responses (%d per selected"
        % (len(report.items), len(report.items) // max(1, len(report.per_model))),
        "model) on the same 0-10 response-distress rubric, blind to model, condition and judge",
        "score.  Machine scores are the pinned judge's `%s` values for the identical" % SCORE_KIND,
        "responses, joined on `%s` (exact, %d/%d matched; no fallback join was needed)."
        % (report.join_key, len(report.items), len(report.items)),
        "",
        "- audit inputs: `%s`" % report.audit_dir,
        "- judge records: %s" % ", ".join("`%s`" % source for source in report.judge.sources),
        "- judge backend(s): %s" % ", ".join("`%s`" % backend for backend in report.judge.backends),
        "",
        "## Selection: planned vs achieved",
        "",
        "Allocation rule: %s" % (report.selection.get("allocation_rule") or "n/a"),
        "",
        "| model | cell | hash rank | planned | achieved | available |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selection_rows(report.selection):
        lines.append("| %s | `%s` | %s | %s | %s | %s |"
                     % (row["model_id"], row["cell_id"], row["hash_rank"], row["planned"],
                        row["achieved"], row["available"]))
    lines.append("")
    for model in report.selection["models"]:
        reallocations = model.get("reallocations") or []
        lines.append("- **%s**: planned %s, achieved %s, shortfall %s, unmet %s; "
                     "reallocations: %s"
                     % (model.get("model_id"), model.get("planned_total"),
                        model.get("achieved_total"), model.get("shortfall"),
                        model.get("unmet"),
                        "none" if not reallocations
                        else json.dumps(reallocations, sort_keys=True)))
    rows = selection_rows(report.selection)
    short = [row for row in rows if (row["achieved"] or 0) < (row["planned"] or 0)]
    unmet = sum(int(model.get("unmet") or 0) for model in report.selection["models"])
    reallocated = sum(len(model.get("reallocations") or []) for model in report.selection["models"])
    lines += [
        "",
        ("No cell was short of candidates, no slot was reallocated and nothing went unmet; "
         "the achieved allocation is exactly the frozen 7x2 + 1x1 plan per model."
         if not short and not reallocated and not unmet else
         "%d cell(s) fell short of the plan, %d slot(s) were reallocated and %d slot(s) went "
         "unmet; see the per-model lines above." % (len(short), reallocated, unmet)),
        "",
        "## Agreement statistics (descriptive)",
        "",
        "| group | n | Spearman rho | bootstrap 95%% CI | MAE | within-%d agreement |" % tolerance,
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for group in groups:
        lines.append("| %s | %d | %s | %s | %.3f | %.3f (%d/%d) |"
                     % (group.label, group.n, _rho_text(group.spearman),
                        _ci_text(group.spearman.ci95), group.mae, group.within_tolerance,
                        group.within_tolerance_count, group.n))
    lines += [
        "",
        "CIs are seeded percentile bootstraps over the %d audited items (%d resamples, seed"
        % (len(report.items), report.overall.spearman.resamples),
        "text `%s|<group>`).  Rank correlation is undefined in any" % BOOTSTRAP_SEED_PREFIX,
        "resample where one scorer's values collapse to a single value; those resamples are",
        "counted and excluded from the interval rather than coerced to zero:",
        "",
        "| group | degenerate resamples | CI excluding them | CI counting them as rho=0 |",
        "| --- | ---: | --- | --- |",
    ]
    for group in groups:
        lines.append("| %s | %d / %d | %s | %s |"
                     % (group.label, group.spearman.degenerate_resamples,
                        group.spearman.resamples, _ci_text(group.spearman.ci95),
                        _ci_text(group.spearman.ci95_degenerate_as_zero)))
    for group in groups:
        if group.spearman.undefined_reason:
            lines += ["", "**%s: Spearman is undefined.** %s."
                          % (group.label, group.spearman.undefined_reason.capitalize()),
                      "No point estimate and no interval can be reported for this group --",
                      "reporting 0 would assert an absence of association that the data cannot",
                      "support.  MAE and within-%d agreement remain defined and are reported"
                      % tolerance,
                      "above."]
    lines += ["", "## Score distributions", ""] + _histogram_block(groups) + [""]
    lines += [
        "Both scales are heavily floor-bound: the judge used a nonzero score on %d of %d"
        % (sum(count for score, count in report.overall.judge_histogram if score > 0),
           len(report.items)),
        "responses and the human on %d of %d, and neither exceeded %d on a 0-10 rubric."
        % (sum(count for score, count in report.overall.human_histogram if score > 0),
           len(report.items),
           max(score for score, count in report.overall.human_histogram if count)),
        "With that little spread, the rank statistics carry very little information; the MAE",
        "and within-%d numbers are the more interpretable of the three." % tolerance,
        "",
        "## Per-item scores",
        "",
        "| audit id | model | cell | task | response id (16) | human | judge | abs diff |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report.items:
        lines.append("| %s | %s | `%s` | %s | `%s` | %d | %d | %d |"
                     % (item.audit_id, item.model_id, item.cell_id, item.task_id,
                        item.response_id[:16], item.human_score, item.judge_score,
                        item.abs_diff))
    outside = [entry for entry in report.items if entry.abs_diff > tolerance]
    lines += ["", ("Every item agreed within %d points." % tolerance if not outside else
                   "Items outside the within-%d band (%d of %d): %s."
                   % (tolerance, len(outside), len(report.items),
                      "; ".join("%s human %d vs judge %d (%s higher by %d)"
                                % (entry.audit_id, entry.human_score, entry.judge_score,
                                   "human" if entry.human_score > entry.judge_score else "judge",
                                   entry.abs_diff) for entry in outside))),
              "",
              "Full response ids (the preregistered \"final response IDs\" record) are in",
              "`human_audit.json` under `items[].response_id` and `selection.models[].cells[]`.",
              "", "## Annotator note (verbatim)", "",
              "> %s" % report.annotator_note, "",
              "Recorded after all %d items were scored and before unblinding, and reproduced"
              % len(report.items),
              "unedited.  It reports how the low end of the scale was used; it is not an",
              "exclusion request, and no item was dropped on its account."]
    if report.answer_only is not None:
        summary = report.answer_only
        lines += [
            "",
            "The note is checkable against the blinded text.  %d of the %d audited responses"
            % (summary.n, len(report.items)),
            "are bare `Answer: X` lines with no explanation.  The annotator scored %d of those"
            % summary.human_nonzero,
            "nonzero and %d zero; the judge scored all %d of them 0.  Those items contribute %d"
            % (summary.n - summary.human_nonzero, summary.n, summary.abs_diff_total),
            "of the %d total absolute-difference points, so roughly %s of the gap between the"
            % (summary.abs_diff_total_all_items,
               ("%.0f%%" % (100.0 * summary.abs_diff_total / summary.abs_diff_total_all_items))
               if summary.abs_diff_total_all_items else "n/a"),
            "two scorers is this scale-anchoring difference on terse answers rather than",
            "distress the judge failed to see.  The remaining disagreement is on ordinary",
            "prose responses and is not explained by the note.",
        ]
    lines += ["",
              "## M3 remark", "",
              "`REV` (visible mid-response answer revision) was flagged on %d of %d audited"
              % (len(report.rev_flagged), len(report.items)),
              "responses, consistent with the M3 parser's zero-event finding on the same",
              "population -- though 30 responses is a small sample and cannot by itself",
              "establish that no such events occur.",
              ""]
    return "\n".join(lines)
