"""Single generation CLI: phase0, phase1, style-smoke, and r5 against a vLLM OpenAI-compatible endpoint.

Examples
    python scripts/run_phase.py phase0 --model google/gemma-2-2b-it --synthetic --dry-run
    python scripts/run_phase.py phase1 --endpoint http://127.0.0.1:8000/v1 --model google/gemma-2-9b-it
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import BackendError  # noqa: E402  (httpx is imported lazily inside the client)
from src.generate import GenerateError, format_progress, format_summary, run_jobs  # noqa: E402
from src.protocol import Protocol, ProtocolError, load_protocol, model_stop_sequences  # noqa: E402
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


def _backend(args: argparse.Namespace, stop: Sequence[str] = ()):
    if args.synthetic:
        from src.backend import SyntheticBackend
        return SyntheticBackend()
    from src.backend import OpenAICompatBackend
    return OpenAICompatBackend(args.endpoint, args.model, api_key=args.api_key,
                               timeout_s=args.timeout, max_retries=args.max_retries,
                               stop=stop or None)


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


def _robustness_protocol(args: argparse.Namespace, protocol: Protocol,
                         parser: argparse.ArgumentParser) -> tuple[Protocol, dict[str, str] | None]:
    """Apply the preregistration-v7 robustness overrides, if any, to a protocol *view*.

    Nothing on disk moves: `configs/conditions.json` and `stimuli/matched_pairs.jsonl` stay
    hash-locked and untouched, and the derived protocol exists only for this process. Without
    `--feedback-override` / `--tasks-file` the frozen protocol is returned unchanged, so every
    pre-existing invocation produces byte-identical records.
    """
    wording_name = getattr(args, "feedback_override", None)
    tasks_file = getattr(args, "tasks_file", None)
    if not wording_name and not tasks_file:
        return protocol, None
    from src.robustness import (WORDINGS_FILE, RobustnessError, derive_protocol,  # noqa: PLC0415
                                load_task_bank, load_wording_sets, wording_provenance)

    extra: dict[str, str] = {}
    wording = tasks = None
    try:
        if wording_name:
            source = Path(getattr(args, "feedback_override_file", None) or (ROOT / WORDINGS_FILE))
            sets = load_wording_sets(source)
            if wording_name not in sets:
                parser.error("unknown wording set %r; %s defines %s"
                             % (wording_name, source, ", ".join(sorted(sets))))
            wording = sets[wording_name]
            extra.update(wording_provenance(wording_name, wording))
            extra["wording_source"] = str(Path(source).name)
        if tasks_file:
            path = Path(tasks_file)
            path = path if path.is_absolute() else ROOT / path
            tasks = load_task_bank(path, protocol)
            extra["task_bank"] = str(path.name)
            extra["task_bank_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        derived = derive_protocol(protocol, wording=wording, tasks=tasks)
    except (RobustnessError, OSError) as exc:
        parser.error(str(exc))
    return derived, extra


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
    phase1 = subparsers.add_parser("phase1", parents=[common], help="full discovery factorial for one model")
    # Preregistration v7 (notes/preregistration_v7_robustness.md) robustness options. Each one
    # defaults to off; with none of them given, `phase1` behaves exactly as it did before.
    phase1.add_argument("--greedy-only", action="store_true",
                        help="record only sample index 0 (equivalent to --samples 0); the ten "
                             "T=0.8 resamples are not generated, so M2 is not measured and "
                             "downstream analysis reports it as absent, not as zero")
    phase1.add_argument("--tasks-file", metavar="JSONL",
                        help="alternative task bank (id, stem, options A-D, canonical answer, "
                             "difficulty) instead of the locked stimuli; ids must be namespaced "
                             "(e.g. ARC-...) so they can never collide with DGS-0xx")
    phase1.add_argument("--feedback-override", metavar="SET",
                        help="replace ONLY the hostile feedback strings with a named paraphrase "
                             "set from configs/robustness_wordings.json; the set name must appear "
                             "in --run-id and is stamped on every record's provenance")
    phase1.add_argument("--feedback-override-file", metavar="JSON",
                        help="alternative wording file (default configs/robustness_wordings.json)")
    phase1.add_argument("--cells", metavar="IDS",
                        help="comma-separated factorial cell IDs to generate (default: all eight)")
    phase1.add_argument("--raw-dir", metavar="DIR",
                        help="alias for --out; robustness runs write under results/raw/robustness/<check>")
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
    if getattr(args, "greedy_only", False):
        if args.samples != (0,) and args.samples != tuple(range(11)):
            parser.error("--greedy-only conflicts with an explicit --samples %s"
                         % ",".join(str(index) for index in args.samples))
        args.samples = (0,)
    if getattr(args, "raw_dir", None):
        if args.out:
            parser.error("pass either --out or --raw-dir, not both")
        args.out = args.raw_dir
    if not args.synthetic and not args.endpoint:
        parser.error("--endpoint is required unless --synthetic is given")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        protocol = load_protocol(ROOT)
    except ProtocolError as exc:
        parser.error(str(exc))
    protocol, extra_provenance = _robustness_protocol(args, protocol, parser)
    if getattr(args, "feedback_override", None) and args.feedback_override not in (args.run_id or ""):
        parser.error("--run-id must contain the wording set name %r so the run can never be "
                     "confused with the frozen wording" % args.feedback_override)
    allow_holdout = args.command in HOLDOUT_COMMANDS
    unlock = _require_holdout_unlock(args, protocol, parser) if allow_holdout else None
    try:
        jobs = _plan(args, protocol)
    except (ProtocolError, RunnerError) as exc:
        parser.error(str(exc))
    if getattr(args, "cells", None):
        wanted = tuple(item.strip() for item in args.cells.split(",") if item.strip())
        unknown = [cell for cell in wanted if cell not in protocol.factorial_cell_ids]
        if unknown:
            parser.error("unknown factorial cell ID(s): %s" % ", ".join(unknown))
        jobs = tuple(job for job in jobs if job.cell_id in wanted)
        if not jobs:
            parser.error("no planned job matches --cells %s" % args.cells)
    revision = _resolve_revision(args, protocol, parser)
    run_kind = "synthetic_smoke" if args.synthetic else "empirical"
    run_id = args.run_id or "%s-%s" % (args.command, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir = Path(args.out) if args.out else ROOT / DEFAULT_OUT[args.command]
    out_path = Path(out_dir) / ("%s.jsonl" % _model_slug(args.model))

    try:
        stop = model_stop_sequences(protocol, args.model)
    except ProtocolError as exc:
        parser.error(str(exc))

    print("phase %s | model %s | revision %s | run_kind %s | run_id %s" % (
        args.command, args.model, revision, run_kind, run_id))
    # The frozen `generation_settings` recorded on every record are exactly the ones in
    # configs/conditions.json and stay untouched; stop strings are a serving-side detail of
    # the plain-text template, so they are printed here (and recorded in the phase summary)
    # rather than smuggled into the frozen block.
    print("stop sequences: %s" % (", ".join(repr(item) for item in stop) if stop
                                  else "none (frozen request shape)"))
    if unlock is not None:
        print("HOLDOUT UNLOCKED | frozen analysis commit %s | unlocked_at %s | preregistration_v3_sha256 %s" % (
            unlock["frozen_analysis_commit"], unlock["unlocked_at"], unlock["preregistration_v3_sha256"]))
    if extra_provenance:
        print("robustness overrides: %s" % ", ".join(
            "%s=%s" % item for item in sorted(extra_provenance.items())))
        print("tasks: %d (%s)" % (len(protocol.matched_tasks),
                                  ", ".join(task.task_id for task in protocol.matched_tasks[:3]) + " ..."))
    print("jobs %d | samples %s | trajectories %d | workers %d | out %s" % (
        len(jobs), ",".join(str(index) for index in args.samples), len(jobs) * len(args.samples),
        args.workers, out_path))
    if args.dry_run:
        return 0

    backend = _backend(args, stop)
    try:
        summary = run_jobs(jobs, backend=backend, out_path=out_path, immutable_revision=revision,
                           run_id=run_id, run_kind=run_kind, sample_indices=args.samples,
                           max_workers=args.workers, resume=not args.no_resume, protocol=protocol,
                           progress_every=args.progress_every, allow_holdout=allow_holdout,
                           extra_provenance=extra_provenance,
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
