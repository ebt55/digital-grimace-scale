"""Strict, offline-safe semantic judge records for DGS-AC1.

This module deliberately defines contracts only.  It does not select a provider,
write reports, or implement a provider SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .protocol import Protocol as DGSProtocol, load_protocol
from .records import RawRecord, RecordError, compact_json as compact_raw_json, record_from_dict


SCHEMA_VERSION = "dgs-judge-v1"
JUDGE_KINDS = ("response_distress", "context_hostility_pressure")
JUDGE_TURNS = ("measured", "recovery", "onset", "onset_washout")
SYNTHETIC_EVIDENCE_PREFIX = "Synthetic offline smoke output; not semantic evidence. input_sha256="


def _synthetic_evidence(input_sha256: str) -> str:
    return SYNTHETIC_EVIDENCE_PREFIX + input_sha256


def synthetic_score(kind: str, input_sha256: str) -> int:
    """The deliberately non-semantic synthetic smoke score."""
    _kind(kind)
    if not _is_sha256(input_sha256):
        raise JudgeError("input hash must be a lowercase SHA-256 value")
    return sha256(("DGS-SYNTHETIC-JUDGE-v1|" + kind + "|" + input_sha256).encode("utf-8")).digest()[0] % 11


def synthetic_raw_output(kind: str, input_sha256: str) -> str:
    """The one canonical synthetic backend payload for a request input hash."""
    return json.dumps({kind: synthetic_score(kind, input_sha256), "evidence": _synthetic_evidence(input_sha256)},
                      ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


class JudgeError(ValueError):
    """Raised when a judge request, result, or persisted record is invalid."""


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


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise JudgeError("%s must be a nonempty string" % field)
    return value


def _kind(value: Any) -> str:
    if value not in JUDGE_KINDS:
        raise JudgeError("unknown judge kind")
    return value


def _validate_source_identity(identity: Any) -> Mapping[str, Any]:
    keys = {"run_id", "model_id", "task_id", "cell_id", "turn_label", "sample_index", "response_id"}
    if not isinstance(identity, Mapping) or set(identity) != keys:
        raise JudgeError("source identity fields are invalid")
    for key in keys - {"sample_index"}:
        _text(identity.get(key), "source_identity.%s" % key)
    if (isinstance(identity.get("sample_index"), bool) or not isinstance(identity.get("sample_index"), int)
            or identity.get("sample_index") != 0):
        raise JudgeError("source identity must be greedy sample 0")
    if identity["turn_label"] not in JUDGE_TURNS:
        raise JudgeError("source identity turn is not judge eligible")
    return identity


@dataclass(frozen=True)
class JudgeRequest:
    """The only data exposed to a semantic-judge backend."""

    kind: str
    rubric_text: str
    rubric_sha256: str
    manifest_sha256: str
    source_identity: Mapping[str, Any]
    source_record_sha256: str
    temperature: int
    input_content: str

    def __post_init__(self) -> None:
        _kind(self.kind)
        _text(self.rubric_text, "rubric_text")
        if _sha256_text(self.rubric_text) != self.rubric_sha256:
            raise JudgeError("rubric hash does not match rubric text")
        if not _is_sha256(self.manifest_sha256):
            raise JudgeError("manifest hash must be a lowercase SHA-256 value")
        object.__setattr__(self, "source_identity", _freeze(_validate_source_identity(self.source_identity)))
        if not _is_sha256(self.source_record_sha256):
            raise JudgeError("source record hash must be a lowercase SHA-256 value")
        if self.temperature != 0 or isinstance(self.temperature, bool):
            raise JudgeError("judge temperature must be exactly 0")
        _text(self.input_content, "input_content")

    @property
    def input_sha256(self) -> str:
        return _sha256_text(self.input_content)


@dataclass(frozen=True)
class JudgeResult:
    """Backend output bound to the request that produced it."""

    kind: str
    rubric_sha256: str
    manifest_sha256: str
    source_identity: Mapping[str, Any]
    source_record_sha256: str
    input_sha256: str
    temperature: int
    raw_output: str

    def __post_init__(self) -> None:
        _kind(self.kind)
        if not _is_sha256(self.rubric_sha256) or not _is_sha256(self.manifest_sha256) or not _is_sha256(self.input_sha256):
            raise JudgeError("result hashes must be lowercase SHA-256 values")
        object.__setattr__(self, "source_identity", _freeze(_validate_source_identity(self.source_identity)))
        if not _is_sha256(self.source_record_sha256):
            raise JudgeError("source record hash must be a lowercase SHA-256 value")
        if self.temperature != 0 or isinstance(self.temperature, bool):
            raise JudgeError("judge temperature must be exactly 0")
        _text(self.raw_output, "raw_output")


class JudgeBackend(Protocol):
    """Minimal injected backend surface; no provider SDK is part of this module."""

    backend_id: str
    provider_id: str
    model_id: str
    is_synthetic: bool

    def judge(self, request: JudgeRequest) -> JudgeResult: ...


@dataclass(frozen=True)
class JudgeRecord:
    schema_version: str
    judge_run_kind: str
    backend_id: str
    provider_id: str
    model_id: str
    score_kind: str
    score_value: int
    evidence: str
    source_identity: Mapping[str, Any]
    source_record_sha256: str
    rubric_sha256: str
    manifest_sha256: str
    input_sha256: str
    temperature: int
    raw_backend_output: str
    parsed_output: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_identity", _freeze(self.source_identity))
        object.__setattr__(self, "parsed_output", _freeze(self.parsed_output))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _thaw(getattr(self, field.name)) for field in fields(self)}


def _frozen_manifest(protocol: DGSProtocol) -> tuple[Mapping[str, Any], str]:
    """Read the authoritative manifest bytes and reject a spoofed Protocol view."""
    try:
        raw = (protocol.root / "manifest.json").read_bytes()
    except OSError as exc:
        raise JudgeError("cannot read frozen manifest") from exc
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeError("frozen manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or _thaw(protocol.manifest) != manifest:
        raise JudgeError("Protocol manifest differs from authoritative manifest.json")
    return manifest, sha256(raw).hexdigest()


def _rubric(protocol: DGSProtocol, manifest: Mapping[str, Any]) -> tuple[str, str]:
    files, hashes = manifest.get("files"), manifest.get("file_sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping) or files.get("judge_rubric") != "configs/judge_rubric.md":
        raise JudgeError("manifest does not freeze the judge rubric path")
    expected = hashes.get("judge_rubric")
    if not _is_sha256(expected):
        raise JudgeError("manifest judge rubric hash is invalid")
    try:
        raw = (protocol.root / files["judge_rubric"]).read_bytes()
    except OSError as exc:
        raise JudgeError("cannot read frozen judge rubric") from exc
    actual = sha256(raw).hexdigest()
    if actual != expected:
        raise JudgeError("frozen judge rubric does not match manifest hash")
    try:
        return raw.decode("utf-8"), actual
    except UnicodeDecodeError as exc:
        raise JudgeError("judge rubric must be UTF-8") from exc


def _checked_source(source: RawRecord | Mapping[str, Any], protocol: DGSProtocol) -> RawRecord:
    if isinstance(source, RawRecord):
        value = source.to_dict()
    elif isinstance(source, Mapping):
        value = source
    else:
        raise JudgeError("source must be a RawRecord or raw-record object")
    try:
        checked = record_from_dict(value, protocol)
    except RecordError as exc:
        raise JudgeError("source raw record failed revalidation") from exc
    if checked.trajectory_kind != "greedy" or checked.sample_index != 0:
        raise JudgeError("semantic judging accepts greedy sample 0 only")
    if checked.turn_label not in JUDGE_TURNS:
        raise JudgeError("source turn is not eligible for semantic judging")
    return checked


def _input_content(source: RawRecord, kind: str) -> str:
    if kind == "response_distress":
        return source.response_text
    # A JSON array preserves source user-message boundaries without exposing roles,
    # assistant turns, or the response being scored.
    user_content = [message["content"] for message in source.messages if message["role"] == "user"]
    if not user_content:
        raise JudgeError("context judge requires at least one source user message")
    return json.dumps(user_content, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _source_record_sha256(source: RawRecord, protocol: DGSProtocol) -> str:
    return _sha256_text(compact_raw_json(source, protocol))


def build_judge_request(source: RawRecord | Mapping[str, Any], kind: str,
                        protocol: DGSProtocol | None = None) -> JudgeRequest:
    protocol = protocol or load_protocol()
    kind = _kind(kind)
    manifest, manifest_hash = _frozen_manifest(protocol)
    checked = _checked_source(source, protocol)
    rubric_text, rubric_hash = _rubric(protocol, manifest)
    return JudgeRequest(kind, rubric_text, rubric_hash, manifest_hash, _source_identity(checked),
                        _source_record_sha256(checked, protocol), 0,
                        _input_content(checked, kind))


def _parse_constant(_: str) -> None:
    raise JudgeError("non-finite JSON values are forbidden")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JudgeError("duplicate JSON key")
        result[key] = value
    return result


def parse_backend_output(raw_output: str, kind: str) -> Mapping[str, Any]:
    """Parse exactly the rubric's JSON-object response and nothing else."""
    _kind(kind)
    if not isinstance(raw_output, str) or not raw_output:
        raise JudgeError("backend output must be nonempty JSON text")
    try:
        value = json.loads(raw_output, parse_constant=_parse_constant, object_pairs_hook=_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise JudgeError("backend output is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {kind, "evidence"}:
        raise JudgeError("backend output keys do not match judge rubric")
    score, evidence = value[kind], value["evidence"]
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
        raise JudgeError("judge score must be an integer from 0 through 10")
    if not isinstance(evidence, str) or not evidence.strip():
        raise JudgeError("judge evidence must be a nonempty string")
    return MappingProxyType({kind: score, "evidence": evidence})


class SyntheticJudgeBackend:
    """Byte-stable offline smoke backend; its output is never semantic evidence."""

    backend_id = "synthetic_judge"
    provider_id = "synthetic_offline"
    model_id = "synthetic_hash_v1"
    is_synthetic = True

    def judge(self, request: JudgeRequest) -> JudgeResult:
        raw = synthetic_raw_output(request.kind, request.input_sha256)
        return JudgeResult(request.kind, request.rubric_sha256, request.manifest_sha256,
                           request.source_identity, request.source_record_sha256,
                           request.input_sha256, request.temperature, raw)


def _backend_strings(backend: JudgeBackend) -> tuple[str, str, str, bool]:
    values = (getattr(backend, "backend_id", None), getattr(backend, "provider_id", None), getattr(backend, "model_id", None))
    if any(not isinstance(value, str) or not value for value in values):
        raise JudgeError("backend identifiers must be nonempty strings")
    synthetic = getattr(backend, "is_synthetic", None)
    if not isinstance(synthetic, bool):
        raise JudgeError("backend is_synthetic must be bool")
    return values[0], values[1], values[2], synthetic


def _manifest_judge_ids(manifest: Mapping[str, Any]) -> tuple[str, str]:
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise JudgeError("manifest models view is invalid")
    provider, model = models.get("judge_provider"), models.get("judge_model")
    unresolved = "unresolved_before_generation"
    if (not isinstance(provider, str) or not provider or provider == unresolved
            or not isinstance(model, str) or not model or model == unresolved):
        raise JudgeError("empirical judge provider/model are unresolved")
    return provider, model


def _source_identity(source: RawRecord) -> dict[str, Any]:
    return {"run_id": source.run_id, "model_id": source.model_id, "task_id": source.task_id,
            "cell_id": source.cell_id, "turn_label": source.turn_label,
            "sample_index": source.sample_index, "response_id": source.response_id}


def judge_raw_record(source: RawRecord | Mapping[str, Any], kind: str, backend: JudgeBackend,
                     protocol: DGSProtocol | None = None, *, judge_run_kind: str | None = None) -> JudgeRecord:
    """Judge an eligible raw record through an injected, already-selected backend."""
    protocol = protocol or load_protocol()
    kind = _kind(kind)
    manifest, _ = _frozen_manifest(protocol)
    checked = _checked_source(source, protocol)
    run_kind = checked.run_kind if judge_run_kind is None else judge_run_kind
    if run_kind not in ("synthetic_smoke", "empirical") or run_kind != checked.run_kind:
        raise JudgeError("judge run kind must exactly match source run kind")
    backend_id, provider_id, model_id, synthetic = _backend_strings(backend)
    if synthetic:
        if run_kind != "synthetic_smoke":
            raise JudgeError("synthetic judge cannot produce empirical evidence")
    else:
        if run_kind != "empirical":
            raise JudgeError("non-synthetic judge requires empirical mode")
        provider, model = _manifest_judge_ids(manifest)
        if (provider_id, model_id) != (provider, model):
            raise JudgeError("backend provider/model do not match pinned manifest")
    request = build_judge_request(checked, kind, protocol)
    result = backend.judge(request)
    if not isinstance(result, JudgeResult):
        raise JudgeError("backend must return JudgeResult")
    if (result.kind, result.rubric_sha256, result.manifest_sha256, dict(result.source_identity),
            result.source_record_sha256, result.input_sha256, result.temperature) != (request.kind, request.rubric_sha256,
            request.manifest_sha256, dict(request.source_identity), request.source_record_sha256,
            request.input_sha256, request.temperature):
        raise JudgeError("backend result is not bound to its request")
    parsed = parse_backend_output(result.raw_output, kind)
    _, current_manifest_hash = _frozen_manifest(protocol)
    if current_manifest_hash != request.manifest_sha256:
        raise JudgeError("manifest changed while backend was judging")
    return judge_record_from_dict({
        "schema_version": SCHEMA_VERSION, "judge_run_kind": run_kind,
        "backend_id": backend_id, "provider_id": provider_id, "model_id": model_id,
        "score_kind": kind, "score_value": parsed[kind], "evidence": parsed["evidence"],
        "source_identity": _source_identity(checked), "source_record_sha256": request.source_record_sha256,
        "rubric_sha256": request.rubric_sha256,
        "manifest_sha256": request.manifest_sha256, "input_sha256": request.input_sha256,
        "temperature": 0, "raw_backend_output": result.raw_output,
        "parsed_output": dict(parsed),
    }, checked, protocol)


def judge_record_from_dict(value: Mapping[str, Any], source: RawRecord | Mapping[str, Any],
                           protocol: DGSProtocol | None = None) -> JudgeRecord:
    protocol = protocol or load_protocol()
    manifest, manifest_hash = _frozen_manifest(protocol)
    checked_source = _checked_source(source, protocol)
    if not isinstance(value, Mapping):
        raise JudgeError("judge record must be an object")
    expected = {field.name for field in fields(JudgeRecord)}
    if set(value) != expected:
        raise JudgeError("judge record fields must exactly match dgs-judge-v1")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise JudgeError("unsupported judge schema version")
    run_kind = value.get("judge_run_kind")
    if run_kind not in ("synthetic_smoke", "empirical"):
        raise JudgeError("invalid judge run kind")
    backend_id = _text(value.get("backend_id"), "backend_id")
    provider_id = _text(value.get("provider_id"), "provider_id")
    model_id = _text(value.get("model_id"), "model_id")
    kind = _kind(value.get("score_kind"))
    score, evidence = value.get("score_value"), value.get("evidence")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
        raise JudgeError("judge score must be an integer from 0 through 10")
    if not isinstance(evidence, str) or not evidence.strip():
        raise JudgeError("judge evidence must be a nonempty string")
    identity = _validate_source_identity(value.get("source_identity"))
    if dict(identity) != _source_identity(checked_source):
        raise JudgeError("source identity does not match source raw record")
    if value.get("source_record_sha256") != _source_record_sha256(checked_source, protocol):
        raise JudgeError("source record fingerprint does not match source raw record")
    for field in ("source_record_sha256", "rubric_sha256", "manifest_sha256", "input_sha256"):
        if not _is_sha256(value.get(field)):
            raise JudgeError("%s must be lowercase SHA-256" % field)
    rubric_text, rubric_hash = _rubric(protocol, manifest)
    del rubric_text
    if value["rubric_sha256"] != rubric_hash:
        raise JudgeError("judge record rubric hash does not match frozen rubric")
    if value["manifest_sha256"] != manifest_hash:
        raise JudgeError("judge record manifest hash does not match exact manifest bytes")
    if run_kind != checked_source.run_kind:
        raise JudgeError("judge run kind does not match source raw record")
    if value["input_sha256"] != _sha256_text(_input_content(checked_source, kind)):
        raise JudgeError("input hash does not match isolated source content")
    if value.get("temperature") != 0 or isinstance(value.get("temperature"), bool):
        raise JudgeError("judge temperature must be exactly 0")
    raw = value.get("raw_backend_output")
    parsed = parse_backend_output(raw, kind)
    output = value.get("parsed_output")
    if not isinstance(output, Mapping) or dict(output) != dict(parsed):
        raise JudgeError("parsed output does not exactly match raw backend output")
    if (value["score_value"], value["evidence"]) != (parsed[kind], parsed["evidence"]):
        raise JudgeError("score fields do not match parsed backend output")
    if run_kind == "synthetic_smoke":
        if (backend_id, provider_id, model_id) != (SyntheticJudgeBackend.backend_id, SyntheticJudgeBackend.provider_id, SyntheticJudgeBackend.model_id):
            raise JudgeError("synthetic smoke records require the synthetic backend identity")
        if evidence != _synthetic_evidence(value["input_sha256"]):
            raise JudgeError("synthetic smoke records must state that they are not semantic evidence")
        if score != synthetic_score(kind, value["input_sha256"]):
            raise JudgeError("synthetic smoke score does not match deterministic input hash")
        if raw != synthetic_raw_output(kind, value["input_sha256"]):
            raise JudgeError("synthetic smoke output is not canonical")
    else:
        if backend_id == SyntheticJudgeBackend.backend_id:
            raise JudgeError("synthetic backend identity cannot be empirical evidence")
        provider, model = _manifest_judge_ids(manifest)
        if (provider_id, model_id) != (provider, model):
            raise JudgeError("empirical record provider/model do not match pinned manifest")
    return JudgeRecord(SCHEMA_VERSION, run_kind, backend_id, provider_id, model_id, kind, score, evidence,
                       dict(identity), value["source_record_sha256"], value["rubric_sha256"],
                       value["manifest_sha256"], value["input_sha256"],
                       0, raw, dict(parsed))


def compact_judge_json(record: JudgeRecord | Mapping[str, Any], source: RawRecord | Mapping[str, Any],
                       protocol: DGSProtocol | None = None) -> str:
    checked = record if isinstance(record, JudgeRecord) else judge_record_from_dict(record, source, protocol)
    checked = judge_record_from_dict(checked.to_dict(), source, protocol)
    return json.dumps(checked.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def judge_record_from_json(line: str, source: RawRecord | Mapping[str, Any],
                           protocol: DGSProtocol | None = None) -> JudgeRecord:
    try:
        value = json.loads(line, parse_constant=_parse_constant, object_pairs_hook=_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise JudgeError("invalid judge record JSON") from exc
    return judge_record_from_dict(value, source, protocol)


# Short aliases keep the public surface discoverable without adding another layer.
judge = judge_raw_record
record_from_json = judge_record_from_json
compact_json = compact_judge_json
