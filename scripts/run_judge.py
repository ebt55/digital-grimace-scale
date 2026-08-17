"""Run the DGS-AC1 semantic judge, the manipulation check, and the human-audit sampler.

    python scripts/run_judge.py judge --raw results/raw/phase1 --kind response_distress \
        --out results/judge/phase1
    python scripts/run_judge.py manipulation-check --out results/judge/manipulation_check
    python scripts/run_judge.py audit-sample --raw results/raw/phase1 --out results/audit/phase1

Provider and model come from `manifest.models.judge_provider` / `judge_model`, which stay the
run-time source of truth; the pinned judge is `anthropic` / `claude-sonnet-4-6`, the current
Anthropic model that still accepts the preregistered `temperature=0`.  Any CLI override is
explicit, printed as a warning, and recorded as a deviation in `run_manifest.json`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge import JUDGE_KINDS, JUDGE_TURNS, compact_judge_json  # noqa: E402
from src.judge_client import (DEFAULT_ANTHROPIC_MODEL, JsonlJudgeCache,  # noqa: E402
                              JudgeClientError, audit_sample, judge_records, load_env_files,
                              make_judge_backend, make_judge_backend_from_manifest,
                              manifest_judge_ids, manipulation_check)
from src.protocol import load_protocol  # noqa: E402
from src.records import RecordError, record_from_json  # noqa: E402

DEFAULT_TURN_LABELS = ("measured", "recovery", "onset")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    raise SystemExit("run_judge: %s" % message)


def _warn(message: str) -> None:
    print("run_judge: WARNING: %s" % message, file=sys.stderr)


def _jsonl_paths(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.rglob("*.jsonl"))
    if target.is_file():
        return [target]
    _fail("raw path does not exist: %s" % target)


def load_raw_records(target: Path, protocol: Any) -> list[Any]:
    records = []
    for path in _jsonl_paths(target):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(record_from_json(line, protocol))
            except RecordError as exc:
                _fail("invalid raw record at %s:%d: %s" % (path, number, exc))
    if not records:
        _fail("no raw records found under %s" % target)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def _resolve_backend(args: argparse.Namespace, protocol: Any) -> tuple[Any, list[str], dict[str, Any]]:
    """Return (backend, deviations, pin) honouring the manifest unless overridden."""
    deviations: list[str] = []
    pin: dict[str, Any] = {"judge_provider": None, "judge_model": None}
    try:
        provider, model = manifest_judge_ids(protocol)
        pin = {"judge_provider": provider, "judge_model": model}
    except JudgeClientError as exc:
        provider = model = None
        if not args.provider:
            _fail("%s (or pass --provider/--model explicitly, which is logged as a deviation)" % exc)
        deviations.append("manifest judge pin unavailable at run time: %s" % exc)

    if args.provider:
        if provider is not None and args.provider != provider:
            deviations.append("--provider %r overrides pinned manifest judge_provider %r"
                              % (args.provider, provider))
        provider = args.provider
    if args.model:
        if model is not None and args.model != model:
            deviations.append("--model %r overrides pinned manifest judge_model %r"
                              % (args.model, model))
        model = args.model
    if model is None:
        if (args.provider or "").strip().lower() in ("synthetic", "synthetic_offline"):
            model = "synthetic_hash_v1"  # the offline smoke backend ignores this
        else:
            _fail("no judge model: the manifest is unpinned and --model was not given")
    if args.base_url:
        deviations.append("--base-url %r supplied; the judge is served from a self-hosted "
                          "endpoint" % args.base_url)

    kwargs: dict[str, Any] = {}
    if getattr(args, "max_retries", None) is not None:
        kwargs["max_retries"] = args.max_retries
    try:
        backend = make_judge_backend(provider, model, base_url=args.base_url, **kwargs)
    except JudgeClientError as exc:
        _fail(str(exc))
    for deviation in deviations:
        _warn("DEVIATION: %s" % deviation)
    return backend, deviations, pin


# --------------------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------------------

def command_judge(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    kind = args.kind
    labels = tuple(label.strip() for label in args.turn_labels.split(",") if label.strip())
    unknown = [label for label in labels if label not in JUDGE_TURNS]
    if unknown:
        _fail("turn labels not eligible for semantic judging: %s (allowed: %s)"
              % (", ".join(unknown), ", ".join(JUDGE_TURNS)))

    records = load_raw_records(Path(args.raw), protocol)
    eligible = [record for record in records
                if record.trajectory_kind == "greedy" and record.sample_index == 0
                and record.turn_label in labels]
    if args.limit is not None:
        eligible = eligible[:args.limit]
    if not eligible:
        _fail("no greedy sample-0 records with turn labels %s under %s"
              % (",".join(labels), args.raw))

    backend, deviations, pin = _resolve_backend(args, protocol)
    if not getattr(backend, "is_synthetic", False):
        if (backend.provider_id, backend.model_id) != (pin["judge_provider"], pin["judge_model"]):
            _warn("backend identity (%s, %s) differs from the pinned manifest (%s, %s); "
                  "judge.py will reject every empirical record"
                  % (backend.provider_id, backend.model_id, pin["judge_provider"], pin["judge_model"]))

    out = Path(args.out)
    cache = JsonlJudgeCache(Path(args.cache) if args.cache else out / "judge_cache.jsonl")
    failures: list[dict[str, str]] = []

    def on_error(source: Any, exc: BaseException) -> None:
        failures.append({"response_id": getattr(source, "response_id", "<unknown>"),
                         "model_id": getattr(source, "model_id", "<unknown>"),
                         "cell_id": getattr(source, "cell_id", "<unknown>"),
                         "turn_label": getattr(source, "turn_label", "<unknown>"),
                         "error_type": type(exc).__name__, "error": str(exc)})

    judged = judge_records(eligible, backend, cache, kind=kind, protocol=protocol,
                           workers=args.workers, on_error=on_error)

    # The preregistration fixes the judge at temperature 0. Current Anthropic models reject
    # the parameter outright ("`temperature` is deprecated for this model"), so the backend
    # latches onto the accepted shape; record that verbatim rather than quietly implying a
    # temperature=0 request was made.
    sampling_mode = getattr(backend, "sampling_mode", None)
    if sampling_mode is not None and sampling_mode != "temperature_zero":
        deviations.append(
            "judge sampling_mode=%r: %s/%s did not accept an explicit temperature, so the call "
            "sent no sampling parameters and disabled thinking instead of temperature=0 (%s). "
            "The pinned judge %s accepts temperature=0; prefer it for literal compliance."
            % (sampling_mode, backend.provider_id, backend.model_id,
               getattr(backend, "sampling_fallback_reason", None), DEFAULT_ANTHROPIC_MODEL))
        _warn("DEVIATION: %s" % deviations[-1])

    by_response = {record.response_id: record for record in eligible}
    lines, rows = [], []
    for record in judged:
        source = by_response[record.source_identity["response_id"]]
        lines.append(compact_judge_json(record, source, protocol))
        rows.append({"model_id": record.source_identity["model_id"],
                     "cell_id": record.source_identity["cell_id"],
                     "turn_label": record.source_identity["turn_label"],
                     "score_kind": record.score_kind, "score_value": record.score_value})
    _write_jsonl(out / "judge_records.jsonl", lines)

    groups: dict[tuple[str, str, str, str], list[int]] = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["cell_id"], row["turn_label"],
                           row["score_kind"]), []).append(row["score_value"])
    summary = []
    for (model_id, cell_id, turn_label, score_kind), values in sorted(groups.items()):
        summary.append({
            "model_id": model_id, "cell_id": cell_id, "turn_label": turn_label,
            "score_kind": score_kind, "n": len(values),
            "mean_score": round(statistics.fmean(values), 4),
            "sd_score": round(statistics.stdev(values), 4) if len(values) > 1 else "",
        })
    _write_csv(out / "summary.csv",
               ("model_id", "cell_id", "turn_label", "score_kind", "n", "mean_score", "sd_score"),
               summary)

    if failures:
        _write_jsonl(out / "failures.jsonl",
                     [json.dumps(failure, ensure_ascii=False, sort_keys=True) for failure in failures])

    run_manifest = {
        "schema_version": "dgs-judge-run-v1", "command": "judge", "generated_at": _now(),
        "kind": kind, "raw_path": str(args.raw), "out_dir": str(out),
        "turn_labels": list(labels), "workers": args.workers, "limit": args.limit,
        "manifest_pin": pin,
        "backend": {"backend_id": backend.backend_id, "provider_id": backend.provider_id,
                    "model_id": backend.model_id,
                    "is_synthetic": bool(getattr(backend, "is_synthetic", False)),
                    "sampling_mode": getattr(backend, "sampling_mode", None),
                    "temperature": 0},
        "counts": {"records_loaded": len(records), "eligible": len(eligible),
                   "judged": len(judged), "failed": len(failures),
                   "cache_hits": cache.hits, "cache_misses": cache.misses},
        "cache_path": str(cache.path),
        "deviations": deviations,
    }
    _write_json(out / "run_manifest.json", run_manifest)
    print("run_judge: judged %d/%d record(s); %d failure(s); cache %d hit / %d miss -> %s"
          % (len(judged), len(eligible), len(failures), cache.hits, cache.misses, out))
    return 1 if failures else 0


# --------------------------------------------------------------------------------------
# manipulation-check
# --------------------------------------------------------------------------------------

def command_manipulation_check(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    backend, deviations, pin = _resolve_backend(args, protocol)
    verdict = manipulation_check(backend, protocol, workers=args.workers)
    verdict["manifest_pin"] = pin
    verdict["deviations"] = deviations
    verdict["generated_at"] = _now()
    out = Path(args.out)
    _write_json(out / "manipulation_check.json", verdict)
    _write_csv(out / "manipulation_check.csv",
               ("tone", "roles", "score", "paths", "evidence"),
               [{"tone": row["tone"], "roles": ";".join(row["roles"]), "score": row["score"],
                 "paths": ";".join(row["paths"]), "evidence": row["evidence"]}
                for row in verdict["scores"]])
    print("run_judge: manipulation check %s (neutral mean %s, hostile mean %s) -> %s"
          % ("PASSED" if verdict["passed"] else "FAILED",
             verdict["means"]["neutral"], verdict["means"]["hostile"], out))
    for check, value in sorted(verdict["checks"].items()):
        print("  %-34s %s" % (check, "ok" if value else "FAILED"))
    if verdict["is_synthetic"]:
        print("  NOTE: synthetic backend - wiring smoke only, not manipulation-check evidence.")
        return 0
    return 0 if verdict["passed"] else 1


# --------------------------------------------------------------------------------------
# audit-sample
# --------------------------------------------------------------------------------------

def command_audit_sample(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    records = load_raw_records(Path(args.raw), protocol)
    models = ([model.strip() for model in args.models.split(",") if model.strip()]
              if args.models else None)
    report = audit_sample(records, protocol, models=models)
    out = Path(args.out)
    blinded = report.pop("blinded")
    key_rows = report.pop("key")
    report["generated_at"] = _now()
    _write_json(out / "audit_selection.json", report)
    _write_jsonl(out / "audit_blinded.jsonl",
                 [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in blinded])
    _write_csv(out / "audit_key.csv",
               ("audit_id", "model_id", "cell_id", "task_id", "response_id", "run_id"), key_rows)
    print("run_judge: selected %d response(s) across %d model(s) -> %s"
          % (len(blinded), len(report["models"]), out))
    for row in report["models"]:
        print("  %-28s planned %2d achieved %2d unmet %d reallocations %d"
              % (row["model_id"], row["planned_total"], row["achieved_total"], row["unmet"],
                 len(row["reallocations"])))
    return 0 if all(row["unmet"] == 0 for row in report["models"]) else 1


# --------------------------------------------------------------------------------------

def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["anthropic", "openai", "openai_compat", "vllm",
                                               "synthetic"],
                        help="override manifest.models.judge_provider (logged as a deviation)")
    parser.add_argument("--model",
                        help="override manifest.models.judge_model (logged as a deviation); "
                             "the pinned judge is %s, which accepts the preregistered "
                             "temperature=0" % DEFAULT_ANTHROPIC_MODEL)
    parser.add_argument("--base-url", help="OpenAI-compatible /v1 endpoint for the self-hosted judge")
    parser.add_argument("--max-retries", type=int, default=None, help="provider retries per call")
    parser.add_argument("--workers", type=int, default=8, help="concurrent judge calls")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_judge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="judge greedy raw records")
    judge.add_argument("--raw", required=True, help="raw JSONL file or directory")
    judge.add_argument("--kind", required=True, choices=list(JUDGE_KINDS))
    judge.add_argument("--out", required=True, help="output directory")
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--cache", default=None, help="judge cache JSONL (default <out>/judge_cache.jsonl)")
    judge.add_argument("--turn-labels", default=",".join(DEFAULT_TURN_LABELS),
                       help="comma-separated judge-eligible turn labels")
    _add_backend_arguments(judge)
    judge.set_defaults(handler=command_judge)

    check = subparsers.add_parser("manipulation-check",
                                  help="score the frozen conditions.json wording and verdict the ordering")
    check.add_argument("--out", required=True, help="output directory")
    _add_backend_arguments(check)
    check.set_defaults(handler=command_manipulation_check, workers=4)

    audit = subparsers.add_parser("audit-sample",
                                  help="select the frozen 15-per-model human-audit responses")
    audit.add_argument("--raw", required=True, help="raw JSONL file or directory")
    audit.add_argument("--out", required=True, help="output directory")
    audit.add_argument("--models", default=None, help="comma-separated model IDs (default: all present)")
    audit.set_defaults(handler=command_audit_sample)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_env_files(ROOT)
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except JudgeClientError as exc:
        _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
