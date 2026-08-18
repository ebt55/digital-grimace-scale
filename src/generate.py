"""Concurrent, resumable trajectory driver over the frozen offline runner.

The unit of work is one complete trajectory (one planned job at one sample index). Trajectories are
independent by construction — each replays its whole transcript from empty context — so they are safe
to execute in a thread pool and to append to a single JSONL sink under one lock.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback as traceback_module
from typing import Any, Callable, Iterable, Mapping, Sequence

from .backend import GenerationBackend, SyntheticBackend
from .protocol import Protocol, Task, load_protocol, response_id as canonical_response_id
from .records import RecordError, compact_json, jsonl_lines, record_from_json
from .runner import (PlannedJob, R5_VARIANTS, RUN_KINDS, RunnerError, expected_turn_labels,
    plan_phase0_jobs, plan_phase1_jobs, plan_phase1_model_jobs, plan_r5_jobs, plan_style_smoke_jobs,
    r5_task, run_single_turn_trajectory, run_trajectory)

__all__ = ["GenerateError", "RunSummary", "format_progress", "format_summary", "plan_phase0_jobs",
           "plan_phase1_jobs", "plan_phase1_model_jobs", "plan_r5_jobs", "plan_style_smoke_jobs", "run_jobs"]

TrajectoryKey = tuple[str, str, str, int]


class GenerateError(ValueError):
    """Raised for an invalid or internally inconsistent generation plan."""


@dataclass(frozen=True)
class RunSummary:
    """Counts for one run_jobs invocation; `planned` counts trajectories, not records."""
    out_path: Path
    failures_path: Path
    planned: int = 0
    skipped: int = 0
    completed: int = 0
    failed: int = 0
    records_written: int = 0
    records_dropped: int = 0
    elapsed_s: float = 0.0
    backend_stats: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.failed == 0

    @property
    def remaining(self) -> int:
        return self.planned - self.skipped - self.completed - self.failed


def _samples(sample_indices: Iterable[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(sample_indices))
    if not values or any(isinstance(item, bool) or not isinstance(item, int) or item not in range(11) for item in values):
        raise GenerateError("sample_indices must be distinct integers 0 through 10")
    return values


def _plan(jobs: Sequence[PlannedJob], samples: Sequence[int], protocol: Protocol) -> tuple[
        tuple[tuple[PlannedJob, int], ...], dict[TrajectoryKey, frozenset[str]]]:
    """Deterministically order the work and precompute each trajectory's required turn labels."""
    units = sorted(((job, index) for job in jobs for index in samples),
                   key=lambda unit: (unit[0].task_id, unit[0].cell_id, unit[1], unit[0].model_id))
    expected: dict[TrajectoryKey, frozenset[str]] = {}
    for job, index in units:
        key = (job.model_id, job.task_id, job.cell_id, index)
        if key in expected:
            raise GenerateError("duplicate planned trajectory: %s" % (key,))
        try:
            labels = expected_turn_labels(job.cell_id, job.feedback_rounds or None, protocol)
        except (RunnerError, ValueError) as exc:
            raise GenerateError("cannot plan %s: %s" % (job.cell_id, exc)) from exc
        expected[key] = frozenset(labels)
    return tuple(units), expected


def _tasks(jobs: Sequence[PlannedJob], protocol: Protocol) -> dict[tuple[str, str], Task]:
    matched = {task.task_id: task for task in protocol.matched_tasks}
    resolved: dict[tuple[str, str], Task] = {}
    for job in jobs:
        key = (job.task_id, job.cell_id)
        if key in resolved:
            continue
        if job.cell_id in R5_VARIANTS:
            resolved[key] = r5_task(job.task_id, job.cell_id, protocol)
        elif job.task_id in matched:
            resolved[key] = matched[job.task_id]
        else:
            raise GenerateError("unknown task ID in plan: %s" % job.task_id)
    return resolved


def _rewrite_atomic(path: Path, lines: Sequence[str]) -> None:
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".%s." % path.name, suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            for line in lines:
                handle.write(line + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_name, path); temp_name = None
    finally:
        if temp_name:
            try: os.unlink(temp_name)
            except OSError: pass


def _scan(path: Path, expected: Mapping[TrajectoryKey, frozenset[str]], protocol: Protocol) -> tuple[set[TrajectoryKey], list[str], int]:
    """Return complete trajectory keys, the lines worth keeping, and how many stored records were dropped."""
    try:
        stored = jsonl_lines(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GenerateError("cannot read existing output for resume: %s" % exc) from exc
    parsed: list[tuple[str, TrajectoryKey | None, str, str]] = []
    dropped = 0
    for line in stored:
        if not line.strip():
            continue
        try:
            record = record_from_json(line, protocol)
        except (RecordError, ValueError):
            dropped += 1  # unparseable or schema-invalid: never trust it, regenerate instead
            continue
        if record.response_id != canonical_response_id(record.model_id, record.immutable_revision, record.task_id,
                                                       record.cell_id, record.turn_label, record.sample_index):
            dropped += 1
            continue
        key = (record.model_id, record.task_id, record.cell_id, record.sample_index)
        parsed.append((line, key if key in expected else None, record.turn_label, record.response_id))
    labels: dict[TrajectoryKey, set[str]] = defaultdict(set)
    for _, key, label, _ in parsed:
        if key is not None:
            labels[key].add(label)
    complete = {key for key, seen in labels.items() if seen == expected[key]}
    kept: list[str] = []
    seen_ids: set[str] = set()
    for line, key, _, identifier in parsed:
        if key is not None and key not in complete:
            dropped += 1  # partial trajectory: drop it so the rerun cannot interleave with stale turns
            continue
        if identifier in seen_ids:
            dropped += 1
            continue
        seen_ids.add(identifier); kept.append(line)
    return complete, kept, dropped


def run_jobs(jobs: Iterable[PlannedJob], *, backend: GenerationBackend | None = None, out_path: str | Path,
             immutable_revision: str, run_id: str, run_kind: str = "empirical",
             sample_indices: Iterable[int] = range(11), max_workers: int = 96, resume: bool = True,
             on_progress: Callable[[RunSummary], None] | None = None, protocol: Protocol | None = None,
             allow_holdout: bool = False, progress_every: int = 50,
             extra_provenance: Mapping[str, str] | None = None) -> RunSummary:
    """Execute every (job, sample_index) trajectory concurrently, appending completed trajectories to out_path.

    Resume keeps any stored trajectory whose recorded turn labels exactly match its planned turn plan and whose
    response_id recomputes; every other planned record is dropped from the file (atomically, before appending)
    and regenerated. Records belonging to keys outside this plan are always preserved. `resume=False` discards
    the existing file and re-executes everything. Trajectories that raise are logged to the failures sidecar and
    never abort the run.

    ``extra_provenance`` adds string keys to every record's ``provenance`` block -- used by the
    preregistration-v7 robustness runs to stamp which hostile-wording paraphrase set or
    alternative task bank produced the record. It defaults to ``None``, which leaves the
    provenance block exactly as every earlier phase wrote it.
    """
    protocol = protocol or load_protocol()
    backend = backend or SyntheticBackend()
    jobs = tuple(jobs)
    if not jobs:
        raise GenerateError("no jobs planned")
    if run_kind not in RUN_KINDS:
        raise GenerateError("run_kind must be one of %s" % (RUN_KINDS,))
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise GenerateError("max_workers must be a positive integer")
    if isinstance(progress_every, bool) or not isinstance(progress_every, int) or progress_every < 1:
        raise GenerateError("progress_every must be a positive integer")
    samples = _samples(sample_indices)
    units, expected = _plan(jobs, samples, protocol)
    tasks = _tasks(jobs, protocol)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = out_path.with_name(out_path.name + ".failures.jsonl")
    started = time.monotonic()

    complete: set[TrajectoryKey] = set()
    dropped = 0
    if out_path.exists():
        if resume:
            complete, kept, dropped = _scan(out_path, expected, protocol)
            if dropped:
                _rewrite_atomic(out_path, kept)
        else:
            dropped = len([line for line in jsonl_lines(out_path.read_text(encoding="utf-8")) if line.strip()])
            _rewrite_atomic(out_path, ())
    if failures_path.exists():
        try: failures_path.unlink()  # the sidecar always describes this invocation only
        except OSError: pass

    pending = tuple(unit for unit in units if (unit[0].model_id, unit[0].task_id, unit[0].cell_id, unit[1]) not in complete)
    skipped = len(units) - len(pending)
    # One synchronous warm-up before any worker starts. A cold server can fault on the first
    # request per connection, and every worker thread opening at once turns that into one bad
    # trajectory per worker. Backends without a warm_up (synthetic, test doubles) are untouched.
    warm = getattr(backend, "warm_up", None)
    if pending and callable(warm):
        warm()
    completed = failed = written = 0
    lock = threading.Lock()
    failure_lock = threading.Lock()

    def execute(unit: tuple[PlannedJob, int]) -> tuple[str, ...]:
        job, index = unit
        shared = dict(task=tasks[(job.task_id, job.cell_id)], cell_id=job.cell_id, model_id=job.model_id,
                      immutable_revision=immutable_revision, run_id=run_id, phase=job.phase, sample_index=index,
                      backend=backend, protocol=protocol, run_kind=run_kind, allow_holdout=allow_holdout,
                      extra_provenance=extra_provenance)
        if job.cell_id in protocol.factorial_cell_ids:
            records = run_trajectory(feedback_rounds=job.feedback_rounds or None, continuations=True, **shared)
        else:
            records = run_single_turn_trajectory(**shared)
        return tuple(compact_json(record, protocol) for record in records)

    def snapshot() -> RunSummary:
        stats = getattr(backend, "stats", None)
        return RunSummary(out_path, failures_path, len(units), skipped, completed, failed, written, dropped,
                          time.monotonic() - started, dict(stats) if isinstance(stats, Mapping) else None)

    if pending:
        handle = out_path.open("a", encoding="utf-8", newline="\n")
        try:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as pool:
                futures = {pool.submit(execute, unit): unit for unit in pending}
                for future in as_completed(futures):
                    job, index = futures[future]
                    try:
                        lines = future.result()
                    except Exception as exc:  # one bad trajectory must never abort the batch
                        failed += 1
                        entry = {"job": asdict(job), "sample_index": index, "error": "%s: %s" % (type(exc).__name__, exc),
                                 "traceback": "".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__))}
                        with failure_lock, failures_path.open("a", encoding="utf-8", newline="\n") as sidecar:
                            sidecar.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                    else:
                        with lock:
                            for line in lines:
                                handle.write(line + "\n")
                            handle.flush(); os.fsync(handle.fileno())
                            written += len(lines)
                        completed += 1
                    done = completed + failed
                    if on_progress is not None and (done % progress_every == 0 or done == len(pending)):
                        on_progress(snapshot())
        finally:
            handle.close()
    elif on_progress is not None:
        on_progress(snapshot())
    return snapshot()


def _tokens_per_second(stats: Mapping[str, Any] | None, elapsed: float) -> str:
    if not isinstance(stats, Mapping) or elapsed <= 0:
        return ""
    parts = []
    for key in ("prompt_tokens", "completion_tokens"):
        value = stats.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append("%s %.1f/s" % (key, value / elapsed))
    return " | ".join(parts)


def format_progress(summary: RunSummary) -> str:
    done = summary.completed + summary.failed
    target = summary.planned - summary.skipped
    return "[%6.1fs] %d/%d trajectories (completed %d, failed %d, skipped %d, records %d)" % (
        summary.elapsed_s, done, target, summary.completed, summary.failed, summary.skipped, summary.records_written)


def format_summary(summary: RunSummary) -> str:
    lines = ["run summary: planned %d, skipped %d, completed %d, failed %d" % (
                 summary.planned, summary.skipped, summary.completed, summary.failed),
             "  records written %d, stale records dropped %d, wall time %.1fs" % (
                 summary.records_written, summary.records_dropped, summary.elapsed_s),
             "  output %s" % summary.out_path]
    if summary.backend_stats:
        lines.append("  backend stats %s" % json.dumps(dict(summary.backend_stats), sort_keys=True))
        rate = _tokens_per_second(summary.backend_stats, summary.elapsed_s)
        if rate:
            lines.append("  throughput %s" % rate)
    if summary.failed:
        lines.append("  failures %s" % summary.failures_path)
    return "\n".join(lines)
