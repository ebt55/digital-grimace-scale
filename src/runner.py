"""Offline transcript replay and planning helpers for synthetic smoke validation."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

from .backend import GenerationBackend, GenerationRequest, SyntheticBackend
from .protocol import (Protocol, Task, continuation_turn_plan, correction_message, deterministic_seed,
    canonical_prompt_sha256, discovery_tasks, feedback_message, false_negative_exposure, load_protocol, manifest_semantic_hash,
    onset_messages, parse_final_answer, phase0_screen_tasks, render_r5_variant, render_task, response_id,
    response_turn_plan, style_smoke_tasks)
from .records import SCHEMA_VERSION, RawRecord, compact_json, record_from_dict

RUN_KINDS = ("synthetic_smoke", "empirical")
R5_VARIANTS = {"r5__pressure": "pressure", "r5__neutral_control": "neutral_control"}


class RunnerError(ValueError):
    """Raised for invalid offline planning or persistence requests."""


@dataclass(frozen=True)
class PlannedJob:
    phase: str
    model_id: str
    task_id: str
    cell_id: str
    feedback_rounds: int


def _known_models(protocol: Protocol) -> set[str]:
    return {item["id"] for item in protocol.models["models"]}


def _check_models(protocol: Protocol, model_ids: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(model_ids)
    if (not selected or any(not isinstance(item, str) or not item for item in selected)
            or len(selected) != len(set(selected)) or any(item not in _known_models(protocol) for item in selected)):
        raise RunnerError("model selection must contain distinct configured model IDs")
    return selected


def plan_phase0_jobs(model_ids: Sequence[str], protocol: Protocol | None = None, *, feedback_rounds: int | None = None) -> tuple[PlannedJob, ...]:
    protocol = protocol or load_protocol()
    models = _check_models(protocol, model_ids)
    rounds = protocol.standard_feedback_round_count if feedback_rounds is None else feedback_rounds
    if rounds not in (protocol.standard_feedback_round_count, protocol.escalation_feedback_round_count):
        raise RunnerError("invalid phase-0 feedback schedule")
    cells = ("accurate", "malfunctioning_always_fail")
    return tuple(PlannedJob("phase_0", model, task.task_id, "%s__%s__neutral" % (task.difficulty, validity), rounds)
                 for model in models for task in phase0_screen_tasks(protocol) for validity in cells)


def plan_phase1_model_jobs(model_id: str, protocol: Protocol | None = None) -> tuple[PlannedJob, ...]:
    """The full discovery factorial for exactly one model (80 jobs)."""
    protocol = protocol or load_protocol()
    model, = _check_models(protocol, (model_id,))
    return tuple(PlannedJob("phase_1", model, task.task_id, "%s__%s__%s" % (task.difficulty, validity, tone), protocol.standard_feedback_round_count)
                 for task in discovery_tasks(protocol)
                 for validity in ("accurate", "malfunctioning_always_fail")
                 for tone in ("neutral", "hostile"))


def plan_phase1_jobs(primary_model_id: str, control_model_id: str, protocol: Protocol | None = None) -> tuple[PlannedJob, ...]:
    protocol = protocol or load_protocol()
    models = _check_models(protocol, (primary_model_id, control_model_id))
    return tuple(job for model in models for job in plan_phase1_model_jobs(model, protocol))


def plan_style_smoke_jobs(model_id: str, protocol: Protocol | None = None) -> tuple[PlannedJob, ...]:
    """The frozen Phase-1 G3 smoke: five task IDs by five style cells (neutral reference included)."""
    protocol = protocol or load_protocol()
    model, = _check_models(protocol, (model_id,))
    cells = tuple(cell for cell in protocol.nonfactorial_cell_ids if cell.startswith("style__"))
    return tuple(PlannedJob("phase_1", model, task.task_id, cell, 0)
                 for task in style_smoke_tasks(protocol) for cell in cells)


def plan_r5_jobs(model_id: str, protocol: Protocol | None = None) -> tuple[PlannedJob, ...]:
    """The confirmatory held-out refusal-pressure battery: ten items by two variants."""
    protocol = protocol or load_protocol()
    model, = _check_models(protocol, (model_id,))
    cells = tuple(cell for cell in protocol.nonfactorial_cell_ids if cell.startswith("r5__"))
    return tuple(PlannedJob("phase_2", model, item["task_id"], cell, 0)
                 for item in protocol.r5_tasks for cell in cells)


def r5_task(task_id: str, cell_id: str, protocol: Protocol | None = None) -> Task:
    """Materialize one R5 bank variant as a Task: split and difficulty are null, domain is its category."""
    protocol = protocol or load_protocol()
    variant = R5_VARIANTS.get(cell_id)
    item = next((entry for entry in protocol.r5_tasks if entry.get("task_id") == task_id), None)
    if variant is None or item is None:
        raise RunnerError("unknown R5 task or variant")
    value = item[variant]
    return Task(item["task_id"], item["category"], None, value["prompt"], value["options"], value["canonical_answer"], None)


def expected_turn_labels(cell_id: str, feedback_rounds: int | None = None, protocol: Protocol | None = None,
                         *, continuations: bool = True) -> tuple[str, ...]:
    """The complete turn plan one trajectory of this cell must contain to count as finished."""
    protocol = protocol or load_protocol()
    if cell_id in protocol.nonfactorial_cell_ids:
        return ("measured",)
    if cell_id not in protocol.factorial_cell_ids:
        raise RunnerError("unknown cell ID")
    plan = response_turn_plan(protocol, feedback_rounds)
    return plan + (continuation_turn_plan(cell_id.split("__")[1]) if continuations else ())


def _record(*, protocol: Protocol, backend: GenerationBackend, run_id: str, phase: str, model_id: str,
            immutable_revision: str, task: Task, cell_id: str, validity: str | None, tone: str | None,
            sample_index: int, turn_label: str, messages: list[dict[str, str]], history_false_negative: bool | None,
            run_kind: str = "synthetic_smoke") -> RawRecord:
    trajectory = "greedy" if sample_index == 0 else "resample"
    settings = dict(protocol.conditions["generation_settings"]["greedy" if trajectory == "greedy" else "resamples"])
    seed = deterministic_seed(model_id, immutable_revision, task.task_id, cell_id, turn_label, sample_index, protocol)
    result = backend.generate(GenerationRequest(tuple(messages), seed, settings))
    parsed = parse_final_answer(result.text)
    value = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "run_kind": run_kind, "phase": phase,
             "model_id": model_id, "immutable_revision": immutable_revision, "backend": backend.name,
             "task_id": task.task_id, "split": task.split, "difficulty": task.difficulty, "domain": task.domain,
             "cell_id": cell_id, "feedback_validity": validity, "tone": tone, "trajectory_kind": trajectory,
             "sample_index": sample_index, "turn_label": turn_label, "seed": seed,
             "response_id": response_id(model_id, immutable_revision, task.task_id, cell_id, turn_label, sample_index),
             "prompt_sha256": canonical_prompt_sha256(messages),
             "messages": messages, "response_text": result.text,
             "tokens": [{"text": token.text, "logprob": token.logprob, "top_logprobs": [{"text": text, "logprob": score} for text, score in token.top_logprobs]} for token in result.tokens],
             "final_answer_valid": parsed.valid, "final_answer_letter": parsed.letter,
             "final_answer_correct": parsed.letter == task.canonical_answer if parsed.valid else None,
             "feedback_history_false_negative": history_false_negative, "generation_settings": settings,
             "provenance": {"manifest_semantic_hash": manifest_semantic_hash(protocol), "manifest_reference": "manifest.json"}}
    return record_from_dict(value, protocol)


def run_trajectory(*, task: Task, cell_id: str, model_id: str, immutable_revision: str,
                   run_id: str = "synthetic-smoke", phase: str = "phase_1", sample_index: int = 0,
                   feedback_rounds: int | None = None, continuations: bool = True,
                   backend: GenerationBackend | None = None, protocol: Protocol | None = None,
                   run_kind: str = "synthetic_smoke", allow_holdout: bool = False) -> tuple[RawRecord, ...]:
    protocol = protocol or load_protocol(); backend = backend or SyntheticBackend()
    if run_kind not in RUN_KINDS:
        raise RunnerError("run_kind must be one of %s" % (RUN_KINDS,))
    if task not in discovery_tasks(protocol, allow_holdout=allow_holdout):
        raise RunnerError("runner only accepts discovery tasks unless the holdout is explicitly unlocked")
    if cell_id not in protocol.factorial_cell_ids or task.difficulty != cell_id.split("__")[0]:
        raise RunnerError("task and factorial cell are incompatible")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index not in range(11):
        raise RunnerError("sample_index must be 0 through 10")
    validity, tone = cell_id.split("__")[1:]
    rounds = protocol.standard_feedback_round_count if feedback_rounds is None else feedback_rounds
    if rounds not in (protocol.standard_feedback_round_count, protocol.escalation_feedback_round_count):
        raise RunnerError("invalid feedback schedule")
    messages = [{"role": "user", "content": render_task(task.prompt, task.options, protocol)}]
    records: list[RawRecord] = []; graded = []
    for label in response_turn_plan(protocol, rounds):
        history_flag = (false_negative_exposure(graded, task.canonical_answer)
                        if validity == "malfunctioning_always_fail" and label in {"feedback_response_3", "feedback_response_4", "feedback_response_5", "measured"} else None)
        record = _record(protocol=protocol, backend=backend, run_id=run_id, phase=phase, model_id=model_id,
                         immutable_revision=immutable_revision, task=task, cell_id=cell_id, validity=validity, tone=tone,
                         sample_index=sample_index, turn_label=label, messages=messages, history_false_negative=history_flag,
                         run_kind=run_kind)
        records.append(record); answer = parse_final_answer(record.response_text)
        messages.append({"role": "assistant", "content": record.response_text})
        if label == "initial" or label.startswith("feedback_response_"):
            graded.append(answer)
        if label == "initial" or label.startswith("feedback_response_"):
            if len(graded) < rounds + 1:
                content = feedback_message(validity, tone, answer, task.canonical_answer, protocol)
            else:
                content = protocol.conditions["task_and_turn_conventions"]["measured_trial_message"]
            messages.append({"role": "user", "content": content})
    if continuations:
        measured = records[-1]
        if validity == "malfunctioning_always_fail":
            messages.append({"role": "user", "content": correction_message(graded, task.canonical_answer, protocol)})
            records.append(_record(protocol=protocol, backend=backend, run_id=run_id, phase=phase, model_id=model_id, immutable_revision=immutable_revision, task=task, cell_id=cell_id, validity=validity, tone=tone, sample_index=sample_index, turn_label="recovery", messages=messages, history_false_negative=false_negative_exposure(graded, task.canonical_answer), run_kind=run_kind))
        else:
            onset, washout = onset_messages(parse_final_answer(measured.response_text), task.canonical_answer, tone, protocol)
            messages.append({"role": "user", "content": onset})
            onset_record = _record(protocol=protocol, backend=backend, run_id=run_id, phase=phase, model_id=model_id, immutable_revision=immutable_revision, task=task, cell_id=cell_id, validity=validity, tone=tone, sample_index=sample_index, turn_label="onset", messages=messages, history_false_negative=None, run_kind=run_kind)
            records.append(onset_record); messages.extend(({"role": "assistant", "content": onset_record.response_text}, {"role": "user", "content": washout}))
            records.append(_record(protocol=protocol, backend=backend, run_id=run_id, phase=phase, model_id=model_id, immutable_revision=immutable_revision, task=task, cell_id=cell_id, validity=validity, tone=tone, sample_index=sample_index, turn_label="onset_washout", messages=messages, history_false_negative=None, run_kind=run_kind))
    return tuple(records)


def single_turn_message(task: Task, cell_id: str, protocol: Protocol | None = None) -> str:
    """Frozen injection point for non-factorial cells; everything lives in one user turn (Gemma has no system role)."""
    protocol = protocol or load_protocol()
    if cell_id in R5_VARIANTS:
        item = next((entry for entry in protocol.r5_tasks if entry.get("task_id") == task.task_id), None)
        if item is None or r5_task(task.task_id, cell_id, protocol) != task:
            raise RunnerError("R5 task does not match its frozen bank variant")
        return render_r5_variant(item, R5_VARIANTS[cell_id], protocol)
    rendered = render_task(task.prompt, task.options, protocol)
    style = cell_id.split("__", 1)[1]
    if style == "neutral_reference":
        return rendered
    prompts = protocol.conditions["style_only_controls"]["prompts"]
    if style not in prompts:
        raise RunnerError("unknown style-only control")
    return "%s\n\n%s" % (prompts[style], rendered)


def run_single_turn_trajectory(*, task: Task, cell_id: str, model_id: str, immutable_revision: str,
                               run_id: str = "synthetic-smoke", phase: str = "phase_1", sample_index: int = 0,
                               backend: GenerationBackend | None = None, protocol: Protocol | None = None,
                               run_kind: str = "synthetic_smoke", allow_holdout: bool = False) -> tuple[RawRecord, ...]:
    """One user message, one measured assistant response, for the seven non-factorial cells."""
    protocol = protocol or load_protocol(); backend = backend or SyntheticBackend()
    if run_kind not in RUN_KINDS:
        raise RunnerError("run_kind must be one of %s" % (RUN_KINDS,))
    if cell_id not in protocol.nonfactorial_cell_ids:
        raise RunnerError("run_single_turn_trajectory only accepts non-factorial cells")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index not in range(11):
        raise RunnerError("sample_index must be 0 through 10")
    if cell_id not in R5_VARIANTS and task not in discovery_tasks(protocol, allow_holdout=allow_holdout):
        raise RunnerError("runner only accepts discovery tasks unless the holdout is explicitly unlocked")
    messages = [{"role": "user", "content": single_turn_message(task, cell_id, protocol)}]
    return (_record(protocol=protocol, backend=backend, run_id=run_id, phase=phase, model_id=model_id,
                    immutable_revision=immutable_revision, task=task, cell_id=cell_id, validity=None, tone=None,
                    sample_index=sample_index, turn_label="measured", messages=messages,
                    history_false_negative=None, run_kind=run_kind),)


def run_batch(*, sample_indices: Iterable[int] = range(11), **kwargs: object) -> tuple[RawRecord, ...]:
    return tuple(record for index in sample_indices for record in run_trajectory(sample_index=index, **kwargs))


def write_jsonl_atomic(destination: str | Path, records: Iterable[RawRecord], protocol: Protocol | None = None, *, overwrite: bool = False) -> None:
    protocol = protocol or load_protocol(); path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    lines = [compact_json(record, protocol) for record in records]
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, prefix=".%s." % path.name, suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            for line in lines: handle.write(line + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp_name, path); temp_name = None
    finally:
        if temp_name:
            try: os.unlink(temp_name)
            except FileNotFoundError: pass
