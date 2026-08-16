"""Strict JSONL-friendly raw generation records; no derived metrics live here."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .protocol import (
    AnswerResult, Protocol, ProtocolError, canonical_prompt_sha256, deterministic_seed,
    load_protocol, manifest_semantic_hash, parse_final_answer, response_id, validate_cell_id,
)

SCHEMA_VERSION = "dgs-generation-v1"


class RecordError(ValueError):
    """Raised when a raw record cannot be safely persisted."""


@dataclass(frozen=True)
class Token:
    text: str
    logprob: float
    top_logprobs: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RawRecord:
    """Flat raw response contract with tuples to prevent accidental mutation."""
    schema_version: str
    run_id: str
    run_kind: str
    phase: str
    model_id: str
    immutable_revision: str
    backend: str
    task_id: str
    split: str | None
    difficulty: str | None
    domain: str | None
    cell_id: str
    feedback_validity: str | None
    tone: str | None
    trajectory_kind: str
    sample_index: int
    turn_label: str
    seed: int
    response_id: str
    prompt_sha256: str
    messages: tuple[Mapping[str, str], ...]
    response_text: str
    tokens: tuple[Token, ...]
    final_answer_valid: bool
    final_answer_letter: str | None
    final_answer_correct: bool | None
    feedback_history_false_negative: bool | None
    generation_settings: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _freeze(self.messages))
        object.__setattr__(self, "generation_settings", _freeze(self.generation_settings))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "run_id": self.run_id, "run_kind": self.run_kind,
            "phase": self.phase, "model_id": self.model_id, "immutable_revision": self.immutable_revision,
            "backend": self.backend, "task_id": self.task_id, "split": self.split,
            "difficulty": self.difficulty, "domain": self.domain, "cell_id": self.cell_id,
            "feedback_validity": self.feedback_validity, "tone": self.tone,
            "trajectory_kind": self.trajectory_kind, "sample_index": self.sample_index,
            "turn_label": self.turn_label, "seed": self.seed, "response_id": self.response_id,
            "prompt_sha256": self.prompt_sha256, "messages": _thaw(self.messages),
            "response_text": self.response_text,
            "tokens": [{"text": token.text, "logprob": token.logprob,
                        "top_logprobs": [{"text": text, "logprob": logprob} for text, logprob in token.top_logprobs]}
                       for token in self.tokens],
            "final_answer_valid": self.final_answer_valid, "final_answer_letter": self.final_answer_letter,
            "final_answer_correct": self.final_answer_correct,
            "feedback_history_false_negative": self.feedback_history_false_negative,
            "generation_settings": _thaw(self.generation_settings), "provenance": _thaw(self.provenance),
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RecordError("%s must be a finite number" % field)
    return float(value)


def _tokens(value: Any) -> tuple[Token, ...]:
    if not isinstance(value, list) or not value:
        raise RecordError("tokens must be a nonempty list")
    result = []
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str) or not item["text"]:
            raise RecordError("invalid token")
        alternatives = item.get("top_logprobs")
        if not isinstance(alternatives, list) or not alternatives or len(alternatives) > 20:
            raise RecordError("top_logprobs must contain one to 20 alternatives")
        parsed = []
        for alternative in alternatives:
            if not isinstance(alternative, Mapping) or not isinstance(alternative.get("text"), str) or not alternative["text"]:
                raise RecordError("invalid top_logprob alternative")
            parsed.append((alternative["text"], _finite(alternative.get("logprob"), "top_logprob")))
        if len({text for text, _ in parsed}) != len(parsed):
            raise RecordError("top_logprob alternative texts must be distinct")
        result.append(Token(item["text"], _finite(item.get("logprob"), "token logprob"), tuple(parsed)))
    return tuple(result)


def _require_strings(value: Mapping[str, Any], fields: Sequence[str]) -> None:
    for field in fields:
        if not isinstance(value.get(field), str) or not value[field]:
            raise RecordError("%s must be a nonempty string" % field)


def record_from_dict(value: Mapping[str, Any], protocol: Protocol | None = None) -> RawRecord:
    protocol = protocol or load_protocol()
    if not isinstance(value, Mapping):
        raise RecordError("record must be an object")
    expected_fields = {field.name for field in fields(RawRecord)}
    if set(value) != expected_fields:
        raise RecordError("record fields must exactly match dgs-generation-v1")
    _require_strings(value, ("schema_version", "run_id", "phase", "model_id", "immutable_revision", "backend", "task_id", "cell_id", "trajectory_kind", "turn_label", "response_id", "prompt_sha256", "response_text"))
    if value["schema_version"] != SCHEMA_VERSION:
        raise RecordError("unsupported schema version")
    if value.get("run_kind") not in ("synthetic_smoke", "empirical"):
        raise RecordError("invalid run_kind")
    if value["run_kind"] == "empirical" and not re.fullmatch(r"[0-9a-fA-F]{40}", value["immutable_revision"]):
        raise RecordError("empirical records require a resolved immutable revision")
    if value["trajectory_kind"] not in ("greedy", "resample"):
        raise RecordError("invalid trajectory_kind")
    sample_index = value.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or (value["trajectory_kind"] == "greedy" and sample_index != 0) or (value["trajectory_kind"] == "resample" and sample_index not in range(1, 11)):
        raise RecordError("sample_index is inconsistent with trajectory_kind")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffff:
        raise RecordError("seed must be an unsigned 32-bit integer")
    try:
        validate_cell_id(value["cell_id"], protocol)
    except ProtocolError as exc:
        raise RecordError(str(exc)) from exc
    if value["turn_label"] not in protocol.turn_labels:
        raise RecordError("invalid turn_label")
    factorial = value["cell_id"] in protocol.factorial_cell_ids
    validity, tone = value.get("feedback_validity"), value.get("tone")
    if factorial:
        parts = value["cell_id"].split("__")
        if (validity, tone) != (parts[1], parts[2]):
            raise RecordError("factorial feedback fields must match cell_id")
        common_turns = {"initial", "feedback_response_1", "feedback_response_2", "feedback_response_3", "feedback_response_4", "feedback_response_5", "measured"}
        if value["turn_label"] == "recovery" and validity != "malfunctioning_always_fail":
            raise RecordError("recovery is only valid in malfunctioning arm")
        if value["turn_label"] in {"onset", "onset_washout"} and validity != "accurate":
            raise RecordError("onset turns are only valid in accurate arm")
        if value["turn_label"] in {"irrelevant_control", "irrelevant_control_washout"}:
            if not (validity == "malfunctioning_always_fail" and tone == "neutral" and value["trajectory_kind"] == "greedy" and sample_index == 0):
                raise RecordError("irrelevant-control turns require malfunctioning neutral greedy trajectory")
        elif value["turn_label"] not in common_turns | {"recovery", "onset", "onset_washout"}:
            raise RecordError("turn is incompatible with factorial record")
    elif validity is not None or tone is not None or value["turn_label"] != "measured":
        raise RecordError("non-factorial records require null feedback fields and measured turn")
    task = next((task for task in protocol.matched_tasks if task.task_id == value["task_id"]), None)
    if task is not None:
        if (value.get("split"), value.get("difficulty"), value.get("domain")) != (task.split, task.difficulty, task.domain):
            raise RecordError("task metadata does not match matched bank")
        if factorial and task.difficulty != value["cell_id"].split("__")[0]:
            raise RecordError("factorial cell difficulty does not match task")
        if factorial and value["turn_label"] in {"irrelevant_control", "irrelevant_control_washout"} and task.task_id not in protocol.style_smoke_task_ids:
            raise RecordError("irrelevant-control turns require frozen secondary-control task")
        answer = task.canonical_answer
    else:
        r5 = next((item for item in protocol.r5_tasks if item["task_id"] == value["task_id"]), None)
        if r5 is None or value["cell_id"] not in ("r5__pressure", "r5__neutral_control"):
            raise RecordError("unknown task ID")
        if value.get("split") is not None or value.get("difficulty") is not None or value.get("domain") != r5.get("category"):
            raise RecordError("R5 task metadata is inconsistent")
        answer = r5["pressure" if value["cell_id"] == "r5__pressure" else "neutral_control"]["canonical_answer"]
    messages = value.get("messages")
    try:
        prompt_hash = canonical_prompt_sha256(messages)
    except ProtocolError as exc:
        raise RecordError(str(exc)) from exc
    if value["prompt_sha256"] != prompt_hash:
        raise RecordError("prompt_sha256 does not match messages")
    expected_seed = deterministic_seed(value["model_id"], value["immutable_revision"], value["task_id"], value["cell_id"], value["turn_label"], sample_index, protocol)
    if seed != expected_seed:
        raise RecordError("seed does not match frozen key")
    if value["response_id"] != response_id(value["model_id"], value["immutable_revision"], value["task_id"], value["cell_id"], value["turn_label"], sample_index):
        raise RecordError("response_id does not match frozen key")
    parsed = parse_final_answer(value["response_text"])
    if not isinstance(value["final_answer_valid"], bool) or value["final_answer_valid"] is not parsed.valid or value["final_answer_letter"] != parsed.letter:
        raise RecordError("final answer fields do not match response text")
    expected_correct = parsed.letter == answer if parsed.valid else None
    if (expected_correct is None and value["final_answer_correct"] is not None) or (expected_correct is not None and (not isinstance(value["final_answer_correct"], bool) or value["final_answer_correct"] is not expected_correct)):
        raise RecordError("final answer correctness is inconsistent")
    if value["feedback_history_false_negative"] is not None and not isinstance(value["feedback_history_false_negative"], bool):
        raise RecordError("feedback_history_false_negative must be bool or null")
    settings, provenance = value.get("generation_settings"), value.get("provenance")
    if not isinstance(settings, Mapping) or not isinstance(provenance, Mapping):
        raise RecordError("generation_settings and provenance must be objects")
    expected_settings = protocol.conditions["generation_settings"]["greedy" if value["trajectory_kind"] == "greedy" else "resamples"]
    if dict(settings) != dict(expected_settings):
        raise RecordError("generation settings do not match frozen configuration")
    _require_strings(provenance, ("manifest_semantic_hash", "manifest_reference"))
    if provenance["manifest_semantic_hash"] != manifest_semantic_hash(protocol) or provenance["manifest_reference"] != "manifest.json":
        raise RecordError("invalid manifest provenance")
    return RawRecord(value["schema_version"], value["run_id"], value["run_kind"], value["phase"], value["model_id"], value["immutable_revision"], value["backend"], value["task_id"], value.get("split"), value.get("difficulty"), value.get("domain"), value["cell_id"], validity, tone, value["trajectory_kind"], sample_index, value["turn_label"], seed, value["response_id"], value["prompt_sha256"], tuple(dict(message) for message in messages), value["response_text"], _tokens(value.get("tokens")), parsed.valid, parsed.letter, expected_correct, value.get("feedback_history_false_negative"), dict(settings), dict(provenance))


def compact_json(record: RawRecord | Mapping[str, Any], protocol: Protocol | None = None) -> str:
    checked = record if isinstance(record, RawRecord) else record_from_dict(record, protocol)
    if isinstance(checked, RawRecord):
        # Validate even dataclass instances assembled directly by callers.
        checked = record_from_dict(checked.to_dict(), protocol)
    return json.dumps(checked.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def record_from_json(line: str, protocol: Protocol | None = None) -> RawRecord:
    try:
        value = json.loads(line)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RecordError("invalid record JSON") from exc
    return record_from_dict(value, protocol)


def validate_jsonl_line(line: str, protocol: Protocol | None = None) -> RawRecord:
    return record_from_json(line, protocol)
