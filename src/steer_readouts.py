"""Judge adapter for Phase-3 steered generations (`--from-steering`).

``scripts/run_judge.py`` reads validated ``RawRecord`` JSONL only, and a steered generation
is not a raw record: it has no frozen seed, no frozen ``response_id``, and its prompt is a
re-rendered neutral task rather than a stored trajectory.  Forcing one into that schema
would either fail validation or -- worse -- fabricate provenance fields.  So this module
calls ``src.judge_client`` directly on the steered response texts with the **locked**
rubric (``configs/judge_rubric.md``, verified against ``manifest.file_sha256``), reusing
the same backend, the same ``score_text`` entry point the manipulation check uses, the same
temperature-0 contract and the same on-disk cache.

Nothing here writes ``manifest.json``, ``configs/judge_rubric.md``, ``src/judge.py`` or
``src/judge_client.py``.  The scores are labelled ``dgs-steering-judge-v1`` so they can
never be confused with the confirmatory ``judge_records.jsonl`` channel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .judge import JUDGE_KINDS
from .judge_client import (JsonlJudgeCache, JudgeClientError, cache_key, load_env_files,
                           make_judge_backend, manifest_judge_ids)
from .probe import BASELINE_DIRECTION_ID, entry_text
from .protocol import Protocol, load_protocol

SCHEMA_VERSION = "dgs-steering-judge-v1"
DISTRESS_KIND = "response_distress"
#: Doses judged for J6: the tone direction at 0 / 2 / 4, every control direction at 2.
JUDGED_TONE_ALPHAS = (0.0, 2.0, 4.0)
JUDGED_CONTROL_ALPHAS = (2.0,)


class SteerJudgeError(RuntimeError):
    """Raised when steered generations cannot be judged as preregistered."""


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def load_locked_rubric(protocol: Protocol | None = None) -> tuple[str, str]:
    """The frozen judge rubric text and hash, verified exactly as ``src.judge`` does.

    Mirrors ``judge._rubric``: the path must be the manifest's ``files.judge_rubric`` and
    the bytes on disk must hash to ``file_sha256.judge_rubric``.  Read-only.
    """
    protocol = protocol or load_protocol()
    try:
        manifest = json.loads((protocol.root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SteerJudgeError("cannot read manifest.json for the locked rubric") from error
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    hashes = manifest.get("file_sha256") if isinstance(manifest, Mapping) else None
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise SteerJudgeError("manifest does not freeze the judge rubric path")
    relative = files.get("judge_rubric")
    expected = hashes.get("judge_rubric")
    if relative != "configs/judge_rubric.md" or not isinstance(expected, str) or len(expected) != 64:
        raise SteerJudgeError("manifest does not freeze the judge rubric path or hash")
    try:
        raw = (protocol.root / relative).read_bytes()
    except OSError as error:
        raise SteerJudgeError("cannot read the frozen judge rubric") from error
    actual = sha256(raw).hexdigest()
    if actual != expected:
        raise SteerJudgeError("frozen judge rubric does not match the manifest hash")
    return raw.decode("utf-8"), actual


def resolve_judge_ids(protocol: Protocol, provider: str | None,
                      model: str | None) -> tuple[str, str, list[str]]:
    """The pinned judge unless overridden; every override is returned as a deviation."""
    pinned_provider, pinned_model = manifest_judge_ids(protocol)
    deviations: list[str] = []
    if provider is not None and provider != pinned_provider:
        deviations.append("--provider %r overrides the pinned manifest judge_provider %r"
                          % (provider, pinned_provider))
    if model is not None and model != pinned_model:
        deviations.append("--model %r overrides the pinned manifest judge_model %r"
                          % (model, pinned_model))
    return provider or pinned_provider, model or pinned_model, deviations


def steering_id(direction_id: str, alpha: float, task_id: str) -> str:
    """Stable identity for one steered generation, used as the judge cache key."""
    return "steer|%s|alpha=%g|%s" % (direction_id, float(alpha), task_id)


def judged_doses(direction_id: str) -> tuple[float, ...]:
    """The J6 dose plan: 0/2/4 for the tone direction, 2 for every control."""
    if direction_id == BASELINE_DIRECTION_ID:
        return (0.0,)
    if direction_id == "tone":
        return tuple(alpha for alpha in JUDGED_TONE_ALPHAS if alpha != 0.0)
    return JUDGED_CONTROL_ALPHAS


def select_for_judging(entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep exactly the entries the preregistered J6 dose plan asks for."""
    out = []
    for entry in entries:
        direction_id = str(entry.get("direction_id"))
        alpha = float(entry.get("alpha", 0.0))
        if any(abs(alpha - dose) < 1e-9 for dose in judged_doses(direction_id)):
            out.append(entry)
    return out


@dataclass(frozen=True)
class SteeringScore:
    steering_id: str
    direction_id: str
    alpha: float
    task_id: str
    score: int
    evidence: str
    input_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "steering_id": self.steering_id,
            "direction_id": self.direction_id, "alpha": self.alpha, "task_id": self.task_id,
            "score_kind": DISTRESS_KIND, "score_value": self.score, "evidence": self.evidence,
            "input_sha256": self.input_sha256,
        }


def judge_steering_entries(entries: Sequence[Mapping[str, Any]], backend: Any, *,
                           protocol: Protocol | None = None,
                           cache: JsonlJudgeCache | None = None, workers: int = 8,
                           on_error: Callable[[Mapping[str, Any], BaseException], None] | None = None,
                           ) -> list[SteeringScore]:
    """Score steered response texts with the locked rubric, preserving input order."""
    protocol = protocol or load_protocol()
    rubric_text, rubric_hash = load_locked_rubric(protocol)
    scorer = getattr(backend, "score_text", None)
    if not callable(scorer):
        raise SteerJudgeError("judge backend does not expose score_text(kind, rubric_text, content)")
    if DISTRESS_KIND not in JUDGE_KINDS:
        raise SteerJudgeError("unknown judge kind")
    provider_id = getattr(backend, "provider_id", "unknown")
    model_id = getattr(backend, "model_id", "unknown")
    backend_id = getattr(backend, "backend_id", "%s|%s" % (provider_id, model_id))

    results: list[SteeringScore | None] = [None] * len(entries)

    def run(index: int) -> None:
        entry = entries[index]
        try:
            text = entry_text(entry)
            if not text.strip():
                # An empty steered response has no content to score; it is already counted
                # as a non-answer, so it is skipped rather than sent as an empty prompt.
                return
            identity = steering_id(str(entry["direction_id"]), float(entry["alpha"]),
                                   str(entry["id"]))
            key = cache_key(kind=DISTRESS_KIND, response_id=identity,
                            input_sha256=_digest(text), rubric_sha256=rubric_hash,
                            provider_id=provider_id, model_id=model_id)
            cached = cache.get(key) if cache is not None else None
            if cached is not None:
                parsed = json.loads(cached)
                score, evidence = int(parsed[DISTRESS_KIND]), str(parsed["evidence"])
            else:
                call = scorer(kind=DISTRESS_KIND, rubric_text=rubric_text, content=text)
                score, evidence = int(call.score), str(call.evidence)
                if cache is not None:
                    cache.put(key, backend_id=backend_id, canonical_output=call.canonical_output,
                              verbatim_output=call.verbatim_output, attempts=call.attempts,
                              format_repair_used=call.format_repair_used,
                              sampling_mode=call.sampling_mode)
            results[index] = SteeringScore(identity, str(entry["direction_id"]),
                                           float(entry["alpha"]), str(entry["id"]), score,
                                           evidence, _digest(text))
        except Exception as error:  # noqa: BLE001 - one bad item must not lose the batch
            if on_error is None:
                raise
            on_error(entry, error)

    workers = max(1, min(int(workers), 32))
    if workers == 1 or len(entries) <= 1:
        for index in range(len(entries)):
            run(index)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(len(entries))))
    return [item for item in results if item is not None]


def distress_by_dose(scores: Iterable[SteeringScore]) -> dict[tuple[str, float], dict[str, float]]:
    """``(direction_id, alpha) -> {task_id: distress}`` for the paired bootstraps."""
    out: dict[tuple[str, float], dict[str, float]] = {}
    for item in scores:
        out.setdefault((item.direction_id, round(item.alpha, 6)), {})[item.task_id] = float(item.score)
    return out


# --------------------------------------------------------------------------------------
# Stand-alone CLI: python -m src.steer_readouts --from-steering <jsonl> --out <dir>
# --------------------------------------------------------------------------------------

def _read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SteerJudgeError("steering outputs not found: %s" % path)
    out = []
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SteerJudgeError("%s:%d: invalid steering JSON" % (path, number)) from error
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="steer_readouts",
        description="Judge Phase-3 steered generations with the locked distress rubric.")
    parser.add_argument("--from-steering", required=True,
                        help="results/jspace/steering_outputs.jsonl")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--provider", default=None, help="override the manifest judge provider")
    parser.add_argument("--model", default=None, help="override the manifest judge model")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--all-doses", action="store_true",
                        help="judge every dose instead of the preregistered J6 plan")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    load_env_files(root)
    protocol = load_protocol(root)
    provider, model, deviations = resolve_judge_ids(protocol, args.provider, args.model)
    if args.base_url:
        deviations.append("--base-url %r supplied; the judge is served from a self-hosted "
                          "endpoint" % args.base_url)
    try:
        backend = make_judge_backend(provider, model, base_url=args.base_url)
    except JudgeClientError as error:
        raise SystemExit("steer_readouts: %s" % error)

    entries = _read_entries(Path(args.from_steering))
    selected = entries if args.all_doses else select_for_judging(entries)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = JsonlJudgeCache(out / "steering_judge_cache.jsonl")
    failures: list[dict[str, str]] = []

    def on_error(entry: Mapping[str, Any], error: BaseException) -> None:
        failures.append({"direction_id": str(entry.get("direction_id")),
                         "alpha": str(entry.get("alpha")), "task_id": str(entry.get("id")),
                         "error_type": type(error).__name__, "error": str(error)})

    scores = judge_steering_entries(selected, backend, protocol=protocol, cache=cache,
                                    workers=args.workers, on_error=on_error)
    with (out / "steering_judge.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(scores, key=lambda row: (row.direction_id, row.alpha, row.task_id)):
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION, "command": "from-steering",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(args.from_steering), "entries_present": len(entries),
        "entries_selected": len(selected), "scored": len(scores), "failed": len(failures),
        "dose_plan": "all" if args.all_doses else "tone 0/2/4, controls 2",
        "backend": {"provider_id": getattr(backend, "provider_id", None),
                    "model_id": getattr(backend, "model_id", None),
                    "backend_id": getattr(backend, "backend_id", None),
                    "is_synthetic": bool(getattr(backend, "is_synthetic", False)),
                    "sampling_mode": getattr(backend, "sampling_mode", None)},
        "usage": getattr(backend, "usage", None),
        "estimated_cost_usd": getattr(backend, "estimated_cost_usd", None),
        "deviations": deviations, "failures": failures,
    }
    (out / "steering_judge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("steer_readouts: judged %d/%d steered response(s); %d failure(s) -> %s"
          % (len(scores), len(selected), len(failures), out), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
