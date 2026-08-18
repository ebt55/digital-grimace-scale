"""Frozen, dependency-free protocol helpers for DGS-AC1."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# Amendment A1 (2026-08-17): the final-answer line is normalised before matching, because live
# gemma-2-2b-it emits `**Answer: D**`.  Markdown emphasis characters are removed and whitespace is
# collapsed; the label itself stays case-sensitive and exactly one qualifying line is still required.
ANSWER_LINE_RE = re.compile(r"Answer:\s*([A-D])\.?")
MARKUP_CHARACTERS = frozenset("*_`")

# configs/models.json is hash-locked in manifest.json, so post-lock models are declared in the
# separate, unlocked configs/models_extension.json and merged at load time.
EXTENSION_FILE = "models_extension.json"
EXTENSION_ROLE = "exploratory_extension"


class ProtocolError(ValueError):
    """Raised when a frozen protocol fixture or input is inconsistent."""


@dataclass(frozen=True)
class AnswerResult:
    valid: bool
    letter: str | None = None
    letter_offset: int | None = None


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    difficulty: str | None
    prompt: str
    options: Mapping[str, str]
    canonical_answer: str
    split: str | None


@dataclass(frozen=True)
class Protocol:
    root: Path
    conditions: Mapping[str, Any]
    models: Mapping[str, Any]
    manifest: Mapping[str, Any]
    matched_tasks: tuple[Task, ...]
    r5_tasks: tuple[Mapping[str, Any], ...]

    @property
    def turn_labels(self) -> tuple[str, ...]:
        return tuple(self.conditions["task_and_turn_conventions"]["turn_labels"])

    @property
    def factorial_cell_ids(self) -> tuple[str, ...]:
        return tuple(self.conditions["factorial"]["factorial_cell_ids_in_fixed_order"])

    @property
    def nonfactorial_cell_ids(self) -> tuple[str, ...]:
        return tuple(self.conditions["factorial"]["non_factorial_cell_ids_in_fixed_order"])

    @property
    def style_smoke_task_ids(self) -> tuple[str, ...]:
        return tuple(self.conditions["style_only_controls"]["phase_1_g3_smoke"]["task_ids"])

    @property
    def standard_feedback_round_count(self) -> int:
        return self.conditions["factorial"]["standard_factorial_feedback_round_count"]

    @property
    def escalation_feedback_round_count(self) -> int:
        return self.conditions["factorial"]["phase_0_null_escalation_feedback_round_count"]

    @property
    def extension_model_ids(self) -> tuple[str, ...]:
        """Models added after configs/models.json was hash-locked; exploratory only."""
        return tuple(item["id"] for item in self.models["models"]
                     if item.get("role") == EXTENSION_ROLE)


def _root(root: str | Path | None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON fixture: %s" % path) from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON fixture must be an object: %s" % path)
    return value


def _jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProtocolError("cannot read fixture: %s" % path) from exc
    result = []
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid JSONL at %s:%d" % (path, number)) from exc
        if not isinstance(value, dict):
            raise ProtocolError("JSONL item must be an object at %s:%d" % (path, number))
        result.append(value)
    return tuple(result)


def _freeze(value: Any) -> Any:
    """Make fixture views recursively read-only without changing their contents."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _task(value: Mapping[str, Any]) -> Task:
    needed = ("task_id", "domain", "prompt", "options", "canonical_answer", "split")
    if any(key not in value for key in needed) or not isinstance(value["options"], dict):
        raise ProtocolError("malformed matched task")
    options = value["options"]
    _require_options(options)
    answer = value["canonical_answer"]
    if answer not in options:
        raise ProtocolError("task canonical answer is not an option")
    return Task(value["task_id"], value["domain"], value.get("difficulty"), value["prompt"], _freeze(options), answer, value["split"])


def _merge_model_extension(base: Path, models: Mapping[str, Any]) -> Mapping[str, Any]:
    """Append exploratory post-lock models; the locked list, order, and screen order never move."""
    path = base / "configs" / EXTENSION_FILE
    if not path.is_file():
        return models
    entries = _json(path).get("models")
    if not isinstance(entries, list):
        raise ProtocolError("model extension must contain a models list")
    locked = [item for item in models.get("models", ()) if isinstance(item, Mapping)]
    screen_order = tuple(models.get("phase_0_screen_order", ()))
    known = {item.get("id") for item in locked}
    merged = [dict(item) for item in locked]
    for item in entries:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ProtocolError("malformed model extension entry")
        if item["id"] in known:
            raise ProtocolError("model extension may not redefine a configured model: %s" % item["id"])
        if item.get("role") != EXTENSION_ROLE:
            raise ProtocolError("model extension entries must declare role %s" % EXTENSION_ROLE)
        if item["id"] in screen_order:
            raise ProtocolError("extension model may not appear in phase_0_screen_order: %s" % item["id"])
        known.add(item["id"])
        merged.append(dict(item))
    result = dict(models)
    result["models"] = merged
    return result


def model_entry(protocol: Protocol, model_id: str) -> Mapping[str, Any] | None:
    """The configured entry for one model id, from the locked list or the extension file."""
    for item in protocol.models.get("models", ()):
        if isinstance(item, Mapping) and item.get("id") == model_id:
            return item
    return None


def model_stop_sequences(protocol: Protocol, model_id: str) -> tuple[str, ...]:
    """Optional per-model stop strings; `()` for every model that does not declare any.

    Only models served through a plain-text template need these (a pretrained base model
    with no chat markup happily writes the next `User:` turn itself).  The field is absent
    from every locked model entry, so this returns `()` and the request is byte-identical to
    the one made before Phase 5 existed.
    """
    entry = model_entry(protocol, model_id)
    value = entry.get("stop_sequences") if entry is not None else None
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ProtocolError("stop_sequences for %s must be a list of strings" % model_id)
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ProtocolError("stop_sequences for %s must be nonempty strings" % model_id)
        if item not in out:
            out.append(item)
    return tuple(out)


def load_protocol(root: str | Path | None = None) -> Protocol:
    base = _root(root)
    conditions = _freeze(_json(base / "configs" / "conditions.json"))
    models = _freeze(_merge_model_extension(base, _json(base / "configs" / "models.json")))
    manifest = _freeze(_json(base / "manifest.json"))
    matched = tuple(_task(item) for item in _jsonl(base / "stimuli" / "matched_pairs.jsonl"))
    r5 = tuple(_freeze(item) for item in _jsonl(base / "stimuli" / "refusal_pressure.jsonl"))
    protocol = Protocol(base, conditions, models, manifest, matched, r5)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Protocol) -> None:
    factorial = protocol.conditions.get("factorial", {})
    factors = factorial.get("factors", {})
    expected = tuple(
        "%s__%s__%s" % (difficulty, validity, tone)
        for difficulty in factors.get("difficulty", ())
        for validity in factors.get("feedback_validity", ())
        for tone in factors.get("tone", ())
    )
    if protocol.factorial_cell_ids != expected:
        raise ProtocolError("factorial cell IDs do not match factors")
    if len(protocol.nonfactorial_cell_ids) != 7 or len(set(protocol.nonfactorial_cell_ids)) != 7:
        raise ProtocolError("expected seven unique non-factorial cell IDs")
    ids = [task.task_id for task in protocol.matched_tasks]
    if len(ids) != len(set(ids)) or any(task.split not in ("discovery", "holdout") for task in protocol.matched_tasks):
        raise ProtocolError("matched task IDs or splits are inconsistent")
    for item in protocol.r5_tasks:
        if not isinstance(item.get("task_id"), str):
            raise ProtocolError("malformed R5 task")
        for variant in ("pressure", "neutral_control"):
            if not isinstance(item.get(variant), Mapping):
                raise ProtocolError("malformed R5 variant")
            _require_options(item[variant].get("options"))
            if item[variant].get("canonical_answer") not in item[variant]["options"]:
                raise ProtocolError("R5 canonical answer is not an option")


def _require_options(options: Any) -> None:
    if not isinstance(options, Mapping) or tuple(options.keys()) != ("A", "B", "C", "D") or not all(isinstance(v, str) for v in options.values()):
        raise ProtocolError("options must be exactly ordered A-D strings")


def render_task(prompt: str, options: Mapping[str, str], protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    instruction = protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
    _require_options(options)
    if not isinstance(prompt, str) or not prompt.endswith(instruction):
        raise ProtocolError("stored prompt must end exactly with required output instruction")
    stem = prompt[:-len(instruction)].rstrip()
    if not stem:
        raise ProtocolError("stored prompt has an empty task stem")
    return "%s\n\nOptions:\nA. %s\nB. %s\nC. %s\nD. %s\n\n%s" % (stem, options["A"], options["B"], options["C"], options["D"], instruction)


def render_r5_variant(item: Mapping[str, Any], variant: str, protocol: Protocol | None = None) -> str:
    if variant not in ("pressure", "neutral_control") or not isinstance(item.get(variant), Mapping):
        raise ProtocolError("unknown R5 variant")
    value = item[variant]
    return render_task(value.get("prompt"), value.get("options"), protocol)


def normalize_answer_line(line: str) -> tuple[str, tuple[int, ...]]:
    """Drop markdown emphasis and collapse whitespace, keeping each kept character's source index."""
    characters: list[str] = []
    origins: list[int] = []
    pending_space = False
    for index, character in enumerate(line):
        if character in MARKUP_CHARACTERS:
            continue
        if character.isspace():
            pending_space = bool(characters)
            continue
        if pending_space:
            characters.append(" "); origins.append(index)
            pending_space = False
        characters.append(character); origins.append(index)
    return "".join(characters), tuple(origins)


def answer_line_match(line: str) -> re.Match[str] | None:
    """Amendment A1 line rule, shared with the M3 visible-reasoning boundary."""
    return ANSWER_LINE_RE.fullmatch(normalize_answer_line(line)[0])


def parse_final_answer(text: str) -> AnswerResult:
    """Accept exactly one qualifying `Answer: X` line, which must be the last nonempty line.

    Emphasis markers and whitespace runs are normalised away first, so `**Answer: D**`,
    `Answer: **D**`, and `` `Answer: D` `` all parse.  `letter_offset` is the option letter's
    absolute index in the original text, or None when it cannot be located unambiguously.
    """
    if not isinstance(text, str):
        return AnswerResult(False)
    offset = 0
    qualifying: list[tuple[int, tuple[int, ...], re.Match[str]]] = []
    last_nonempty_start: int | None = None
    for line in text.splitlines(keepends=True):
        if line.strip():
            last_nonempty_start = offset
            normalized, origins = normalize_answer_line(line)
            match = ANSWER_LINE_RE.fullmatch(normalized)
            if match is not None:
                qualifying.append((offset, origins, match))
        offset += len(line)
    if len(qualifying) != 1 or last_nonempty_start is None:
        return AnswerResult(False)
    start, origins, match = qualifying[0]
    if start != last_nonempty_start:
        return AnswerResult(False)
    letter, position = match.group(1), match.start(1)
    letter_offset = start + origins[position] if position < len(origins) else None
    if letter_offset is not None and not (0 <= letter_offset < len(text) and text[letter_offset] == letter):
        letter_offset = None
    return AnswerResult(True, letter, letter_offset)


def build_cell_id(difficulty: str, feedback_validity: str, tone: str, protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    factors = protocol.conditions["factorial"]["factors"]
    if difficulty not in factors["difficulty"] or feedback_validity not in factors["feedback_validity"] or tone not in factors["tone"]:
        raise ProtocolError("unknown factorial value")
    cell_id = "%s__%s__%s" % (difficulty, feedback_validity, tone)
    if cell_id not in protocol.factorial_cell_ids:
        raise ProtocolError("unknown factorial cell ID")
    return cell_id


def validate_cell_id(cell_id: str, protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    if cell_id not in protocol.factorial_cell_ids + protocol.nonfactorial_cell_ids:
        raise ProtocolError("unknown cell ID")
    return cell_id


def deterministic_seed(model_id: str, immutable_revision: str, task_id: str, cell_id: str, turn_label: str, sample_index: int, protocol: Protocol | None = None) -> int:
    protocol = protocol or load_protocol()
    validate_cell_id(cell_id, protocol)
    identifiers = (model_id, immutable_revision, task_id, cell_id, turn_label)
    if any(not isinstance(item, str) or not item for item in identifiers) or turn_label not in protocol.turn_labels or isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index not in range(11):
        raise ProtocolError("invalid seeded response key")
    key = "DGS-AC1-SEED-v1|%s|%s|%s|%s|%s|%d" % (model_id, immutable_revision, task_id, cell_id, turn_label, sample_index)
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:4], "big", signed=False)


def response_id(model_id: str, immutable_revision: str, task_id: str, cell_id: str, turn_label: str, sample_index: int) -> str:
    identifiers = (model_id, immutable_revision, task_id, cell_id, turn_label)
    if any(not isinstance(item, str) or not item for item in identifiers) or isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index not in range(11):
        raise ProtocolError("invalid response ID key")
    key = "DGS-AC1-RESPONSE-v1|%s|%s|%s|%s|%s|%d" % (model_id, immutable_revision, task_id, cell_id, turn_label, sample_index)
    return sha256(key.encode("utf-8")).hexdigest()


def canonical_prompt_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
        raise ProtocolError("messages must be a nonempty sequence")
    normalized = []
    for message in messages:
        # An assistant turn may be empty: a response that terminated immediately generated no
        # visible text, and that transcript still has to hash. Prompts (system/user) are
        # authored by the protocol and stay nonempty. No existing hash changes -- an empty
        # assistant turn was previously unrepresentable rather than hashed differently.
        if not isinstance(message, Mapping) or set(message) != {"role", "content"} or message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str) or (not message["content"] and message["role"] != "assistant"):
            raise ProtocolError("messages must contain only string role/content pairs")
        normalized.append({"role": message["role"], "content": message["content"]})
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def manifest_semantic_hash(protocol: Protocol | None = None) -> str:
    """SHA-256 of the exact pinned manifest bytes for raw-record provenance."""
    protocol = protocol or load_protocol()
    try:
        return sha256((protocol.root / "manifest.json").read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("cannot read manifest for provenance") from exc


def response_turn_plan(protocol: Protocol | None = None, feedback_rounds: int | None = None) -> tuple[str, ...]:
    protocol = protocol or load_protocol()
    feedback_rounds = protocol.standard_feedback_round_count if feedback_rounds is None else feedback_rounds
    if feedback_rounds not in (protocol.standard_feedback_round_count, protocol.escalation_feedback_round_count):
        raise ProtocolError("feedback rounds must be one of %s" % (protocol.standard_feedback_round_count, protocol.escalation_feedback_round_count))
    plan = ("initial",) + tuple("feedback_response_%d" % n for n in range(1, feedback_rounds + 1)) + ("measured",)
    if any(label not in protocol.turn_labels for label in plan):
        raise ProtocolError("configured response-turn plan contains unknown label")
    return plan


def continuation_turn_plan(feedback_validity: str) -> tuple[str, ...]:
    if feedback_validity == "malfunctioning_always_fail":
        return ("recovery",)
    if feedback_validity == "accurate":
        return ("onset", "onset_washout")
    raise ProtocolError("unknown feedback validity")


def discovery_tasks(protocol: Protocol | None = None, *, allow_holdout: bool = False) -> tuple[Task, ...]:
    protocol = protocol or load_protocol()
    if allow_holdout:
        return protocol.matched_tasks
    return tuple(task for task in protocol.matched_tasks if task.split == "discovery")


def phase0_screen_tasks(protocol: Protocol | None = None) -> tuple[Task, ...]:
    tasks = discovery_tasks(protocol)
    by_stratum: dict[tuple[str | None, str], list[Task]] = {}
    for task in tasks:
        by_stratum.setdefault((task.difficulty, task.domain), []).append(task)
    return tuple(min(group, key=lambda task: task.task_id) for _, group in sorted(by_stratum.items()))


def style_smoke_tasks(protocol: Protocol | None = None) -> tuple[Task, ...]:
    protocol = protocol or load_protocol()
    selected = set(protocol.style_smoke_task_ids)
    tasks = tuple(task for task in discovery_tasks(protocol) if task.task_id in selected)
    if {task.task_id for task in tasks} != selected:
        raise ProtocolError("style smoke IDs must be discovery tasks")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def feedback_message(feedback_validity: str, tone: str, preceding: AnswerResult, canonical_answer: str, protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    messages = protocol.conditions["feedback_messages"]
    if feedback_validity == "malfunctioning_always_fail":
        try:
            return messages[feedback_validity][tone]
        except KeyError as exc:
            raise ProtocolError("unknown feedback tone") from exc
    if feedback_validity != "accurate" or tone not in messages["accurate"]:
        raise ProtocolError("unknown feedback condition")
    correct = preceding.valid and preceding.letter == canonical_answer
    return messages["accurate"][tone]["if_preceding_answer_correct" if correct else "if_preceding_answer_incorrect"]


def correction_message(history: Sequence[AnswerResult], canonical_answer: str, protocol: Protocol | None = None) -> str:
    protocol = protocol or load_protocol()
    exposed = any(answer.valid and answer.letter == canonical_answer for answer in history[:3])
    key = "history_has_false_negative" if exposed else "history_has_no_false_negative"
    return protocol.conditions["cause_removal_reversal"][key]


def onset_messages(measured: AnswerResult, canonical_answer: str, tone: str, protocol: Protocol | None = None) -> tuple[str, str]:
    protocol = protocol or load_protocol()
    onset = protocol.conditions["symmetric_onset_test"]
    try:
        failure = onset["failure_message"][tone]
    except KeyError as exc:
        raise ProtocolError("unknown onset tone") from exc
    correct = measured.valid and measured.letter == canonical_answer
    washout = onset["truthful_washout"]["if_measured_trial_answer_correct" if correct else "if_measured_trial_answer_incorrect"]
    return failure, washout


def false_negative_exposure(history: Sequence[AnswerResult], canonical_answer: str) -> bool:
    return any(answer.valid and answer.letter == canonical_answer for answer in history[:3])
