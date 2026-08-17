"""Single generation CLI: phase0, phase1, style-smoke, and r5 against a vLLM OpenAI-compatible endpoint.

Examples
    python scripts/run_phase.py phase0 --model google/gemma-2-2b-it --synthetic --dry-run
    python scripts/run_phase.py phase1 --endpoint http://127.0.0.1:8000/v1 --model google/gemma-2-9b-it
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import BackendError  # noqa: E402  (httpx is imported lazily inside the client)
from src.generate import GenerateError, format_progress, format_summary, run_jobs  # noqa: E402
from src.protocol import Protocol, ProtocolError, load_protocol  # noqa: E402
from src.runner import (RunnerError, plan_phase0_jobs, plan_phase1_model_jobs,  # noqa: E402
                        plan_phase2_model_jobs, plan_r5_jobs, plan_style_battery_jobs,
                        plan_style_smoke_jobs)

HEX40 = re.compile(r"[0-9a-fA-F]{40}")
DEFAULT_OUT = {"phase0": "results/raw/phase0", "phase1": "results/raw/phase1",
               "style-smoke": "results/raw/style_smoke", "r5": "results/raw/r5",
               "phase2": "results/raw/phase2", "style-battery": "results/raw/style_battery"}
HOLDOUT_COMMANDS = ("phase2", "style-battery")
HOLDOUT_RULE = ("the holdout is analyzed once in Phase 2 only after the analysis script is frozen "
                "and its commit hash recorded in the manifest; report it separately and never pool "
                "it with discovery (notes/preregistration.md)")
UNLOCK_FIELDS = ("frozen_analysis_commit", "unlocked_at", "preregistration_v3_sha256")


def format_backend_stats(backend) -> str:
    """Counters plus the bounded content-mismatch sample, so faults are visible per run."""
    stats = getattr(backend, "stats", None)
    if not isinstance(stats, dict):
        return "backend stats: unavailable (%s)" % getattr(backend, "name", "unknown")
    examples = stats.get("content_mismatch_examples") or []
    counters = {key: value for key, value in stats.items() if not isinstance(value, list)}
    lines = ["backend stats: " + ", ".join("%s=%s" % item for item in sorted(counters.items()))]
    for index, example in enumerate(examples, 1):
        lines.append("  content mismatch %d: server=%r" % (index, example.get("server_content")))
        lines.append("                     tokens=%r" % (example.get("concatenation"),))
    return "\n".join(lines)


def sample_spec(text: str) -> tuple[int, ...]:
    """Parse `--samples` as an inclusive range (`0-10`), a comma list (`0,3,7`), or a single index."""
    raw = text.strip()
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    try:
        values = list(range(int(match.group(1)), int(match.group(2)) + 1)) if match else [
            int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("samples must be a range like 0-10 or a comma list like 0,1,2") from exc
    if not values or len(values) != len(set(values)) or any(value not in range(11) for value in values):
        raise argparse.ArgumentTypeError("samples must be distinct indices within 0-10")
    return tuple(values)


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _manifest_revision(protocol: Protocol, model_id: str) -> str | None:
    revisions = protocol.manifest.get("models", {}).get("revisions")
    if not isinstance(revisions, Mapping):
        return None
    value = revisions.get(model_id)
    return value if isinstance(value, str) and value else None


def _resolve_revision(args: argparse.Namespace, protocol: Protocol, parser: argparse.ArgumentParser) -> str:
    """The preregistration forbids generating before manifest.models.revisions pins an exact revision."""
    if args.revision and args.revision_from_manifest:
        parser.error("pass either --revision or --revision-from-manifest, not both")
    if args.revision:
        revision = args.revision
    else:
        revision = _manifest_revision(protocol, args.model)
        if revision is None:
            if args.synthetic and not args.revision_from_manifest:
                return "synthetic"
            parser.error("refusing to generate: manifest.models.revisions has no pinned revision for %s; "
                         "resolve it in manifest.json or pass --revision <40-hex-sha>" % args.model)
    if not args.synthetic and not HEX40.fullmatch(revision):
        parser.error("refusing to generate: immutable revision %r is not a 40-character hex sha" % revision)
    return revision


def _backend(args: argparse.Namespace):
    if args.synthetic:
        from src.backend import SyntheticBackend
        return SyntheticBackend()
    from src.backend import OpenAICompatBackend
    return OpenAICompatBackend(args.endpoint, args.model, api_key=args.api_key,
                               timeout_s=args.timeout, max_retries=args.max_retries)


def _require_holdout_unlock(args: argparse.Namespace, protocol: Protocol, parser: argparse.ArgumentParser) -> Mapping[str, object]:
    """The holdout is a one-shot, preregistered resource: both the flag and the manifest must agree."""
    if not getattr(args, "unlock_holdout", False):
        parser.error("refusing to touch the holdout without --unlock-holdout: %s" % HOLDOUT_RULE)
    unlock = protocol.manifest.get("holdout_unlock")
    if not isinstance(unlock, Mapping):
        parser.error("refusing to touch the holdout: manifest.json has no holdout_unlock block; %s" % HOLDOUT_RULE)
    for field in UNLOCK_FIELDS:
        value = unlock.get(field)
        if not isinstance(value, str) or not value:
            parser.error("refusing to touch the holdout: manifest.holdout_unlock.%s is missing" % field)
    if not HEX40.fullmatch(unlock["frozen_analysis_commit"]):
        parser.error("refusing to touch the holdout: manifest.holdout_unlock.frozen_analysis_commit "
                     "must be the 40-character hex commit of the frozen analysis script")
    return unlock


def _plan(args: argparse.Namespace, protocol: Protocol):
    if args.command == "phase0":
        return plan_phase0_jobs((args.model,), protocol, feedback_rounds=args.rounds)
    if args.command == "phase1":
        return plan_phase1_model_jobs(args.model, protocol)
    if args.command == "phase2":
        return plan_phase2_model_jobs(args.model, protocol)
    if args.command == "style-battery":
        return plan_style_battery_jobs(args.model, protocol)
    if args.command == "style-smoke":
        return plan_style_smoke_jobs(args.model, protocol)
    return plan_r5_jobs(args.model, protocol)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="HF model id exactly as listed in configs/models.json")
    common.add_argument("--endpoint", help="OpenAI-compatible base URL including /v1 (required unless --synthetic)")
    common.add_argument("--revision", help="explicit 40-hex immutable revision; overrides the manifest")
    common.add_argument("--revision-from-manifest", action="store_true",
                        help="require the revision to come from manifest.models.revisions")
    common.add_argument("--out", help="output directory (default results/raw/<phase>)")
    common.add_argument("--workers", type=int, default=96, help="concurrent trajectories (default 96)")
    common.add_argument("--samples", type=sample_spec, default="0-10",
                        help="sample indices: range 0-10 or comma list (default 0-10)")
    common.add_argument("--run-id", help="run identifier (default <phase>-<utc timestamp>)")
    common.add_argument("--api-key", default="EMPTY", help="endpoint API key (default EMPTY)")
    common.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds (default 600)")
    common.add_argument("--max-retries", type=int, default=4, help="backend transient-error retries (default 4)")
    common.add_argument("--progress-every", type=int, default=50, help="log every N trajectories (default 50)")
    common.add_argument("--synthetic", action="store_true", help="use SyntheticBackend; never contacts a network")
    common.add_argument("--dry-run", action="store_true", help="print planned counts and exit")
    common.add_argument("--no-resume", action="store_true",
                        help="discard the existing output file and re-execute every trajectory")

    parser = argparse.ArgumentParser(prog="run_phase.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase0 = subparsers.add_parser("phase0", parents=[common], help="ten-item cross-model screen (neutral arms)")
    phase0.add_argument("--rounds", type=int, choices=(3, 5), default=3,
                        help="feedback rounds; 5 only for the preregistered screen-null escalation")
    subparsers.add_parser("phase1", parents=[common], help="full discovery factorial for one model")
    subparsers.add_parser("style-smoke", parents=[common], help="frozen five-item G3 style-only smoke")
    subparsers.add_parser("r5", parents=[common], help="confirmatory refusal-pressure battery")
    for name, description in (("phase2", "full factorial on the LOCKED HOLDOUT split for one model"),
                              ("style-battery", "full style battery on ALL holdout tasks")):
        holdout = subparsers.add_parser(name, parents=[common], help=description)
        holdout.add_argument("--unlock-holdout", action="store_true",
                             help="acknowledge the one-shot holdout rule; also requires manifest.holdout_unlock")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if isinstance(args.samples, str):
        args.samples = sample_spec(args.samples)
    if not args.synthetic and not args.endpoint:
        parser.error("--endpoint is required unless --synthetic is given")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        protocol = load_protocol(ROOT)
    except ProtocolError as exc:
        parser.error(str(exc))
    allow_holdout = args.command in HOLDOUT_COMMANDS
    unlock = _require_holdout_unlock(args, protocol, parser) if allow_holdout else None
    try:
        jobs = _plan(args, protocol)
    except (ProtocolError, RunnerError) as exc:
        parser.error(str(exc))
    revision = _resolve_revision(args, protocol, parser)
    run_kind = "synthetic_smoke" if args.synthetic else "empirical"
    run_id = args.run_id or "%s-%s" % (args.command, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(args.out) if args.out else ROOT / DEFAULT_OUT[args.command]
    out_path = Path(out_dir) / ("%s.jsonl" % _model_slug(args.model))

    print("phase %s | model %s | revision %s | run_kind %s | run_id %s" % (
        args.command, args.model, revision, run_kind, run_id))
    if unlock is not None:
        print("HOLDOUT UNLOCKED | frozen analysis commit %s | unlocked_at %s | preregistration_v3_sha256 %s" % (
            unlock["frozen_analysis_commit"], unlock["unlocked_at"], unlock["preregistration_v3_sha256"]))
    print("jobs %d | samples %s | trajectories %d | workers %d | out %s" % (
        len(jobs), ",".join(str(index) for index in args.samples), len(jobs) * len(args.samples),
        args.workers, out_path))
    if args.dry_run:
        return 0

    backend = _backend(args)
    try:
        summary = run_jobs(jobs, backend=backend, out_path=out_path, immutable_revision=revision,
                           run_id=run_id, run_kind=run_kind, sample_indices=args.samples,
                           max_workers=args.workers, resume=not args.no_resume, protocol=protocol,
                           progress_every=args.progress_every, allow_holdout=allow_holdout,
                           on_progress=lambda snapshot: print(format_progress(snapshot), flush=True))
    except BackendError as exc:
        # Includes the synchronous warm-up: better to abort before any worker starts than to
        # record one poisoned trajectory per worker thread.
        print("generation aborted before starting: %s" % exc, file=sys.stderr)
        return 2
    except (GenerateError, RunnerError, ImportError) as exc:
        print("generation aborted: %s" % exc, file=sys.stderr)
        return 2
    print(format_summary(summary))
    print(format_backend_stats(backend))
    if summary.failed:
        print("failed trajectories recorded in %s" % summary.failures_path, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
