"""Remove trajectories poisoned by the empty-stream placeholder token.

A transient server fault (an empty stream on the first request per worker connection) was
briefly recorded as a legitimate empty response, producing records whose entire token trace is
the placeholder ``[{"text":"","logprob":0.0,"top_logprobs":[{"text":"","logprob":0.0}]}]``.
Those are fabricated, not model output.

Because a trajectory is only meaningful as a whole -- later turns were conditioned on the
placeholder turn -- this removes *every* record of any affected trajectory, keyed by
(model_id, task_id, cell_id, sample_index). Re-running `run_phase.py` with the same --run-id
then regenerates exactly those trajectories, because resume keeps only complete ones.

Usage (PowerShell, from the repo root)::

    .venv\\Scripts\\python.exe scripts/purge_placeholder_trajectories.py --raw results/raw --dry-run
    .venv\\Scripts\\python.exe scripts/purge_placeholder_trajectories.py --raw results/raw
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.records import jsonl_lines  # noqa: E402

# The exact shape written for an empty stream. Matched structurally, never by substring, so a
# real one-token response that happens to be empty-ish is untouched.
PLACEHOLDER_TOKENS = [{"text": "", "logprob": 0.0, "top_logprobs": [{"text": "", "logprob": 0.0}]}]
KEY_FIELDS = ("model_id", "task_id", "cell_id", "sample_index")


def is_placeholder(record: Any) -> bool:
    return isinstance(record, dict) and record.get("tokens") == PLACEHOLDER_TOKENS


def trajectory_key(record: dict) -> tuple:
    return tuple(record.get(field) for field in KEY_FIELDS)


def raw_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(path for path in target.rglob("*.jsonl") if not path.name.endswith(".failures.jsonl"))
    raise SystemExit("no such file or directory: %s" % target)


def rewrite_atomic(path: Path, lines: Sequence[str]) -> None:
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".%s." % path.name, suffix=".tmp", delete=False)
    temporary = handle.name
    try:
        with handle:
            for line in lines:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except OSError: pass


def purge_file(path: Path, *, dry_run: bool) -> tuple[int, int, int]:
    """Return (trajectories purged, records removed, records kept)."""
    lines = [line for line in jsonl_lines(path.read_text(encoding="utf-8")) if line.strip()]
    parsed: list[tuple[str, dict | None]] = []
    poisoned: set[tuple] = set()
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parsed.append((line, None))  # unreadable lines are preserved untouched
            continue
        parsed.append((line, record))
        if is_placeholder(record):
            poisoned.add(trajectory_key(record))
    if not poisoned:
        return 0, 0, len(lines)
    kept = [line for line, record in parsed
            if record is None or trajectory_key(record) not in poisoned]
    removed = len(lines) - len(kept)
    if not dry_run:
        rewrite_atomic(path, kept)
    return len(poisoned), removed, len(kept)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", required=True, help="a raw JSONL file or a directory of them")
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    files = raw_files(Path(args.raw))
    total_trajectories = total_records = 0
    for path in files:
        trajectories, removed, kept = purge_file(path, dry_run=args.dry_run)
        total_trajectories += trajectories
        total_records += removed
        if trajectories:
            print("%s: %d trajectories purged, %d records removed, %d kept"
                  % (path, trajectories, removed, kept))
    verb = "would purge" if args.dry_run else "purged"
    print("%s %d trajectories (%d records) across %d file(s)"
          % (verb, total_trajectories, total_records, len(files)))
    if total_trajectories and not args.dry_run:
        print("re-run run_phase.py with the SAME --run-id to regenerate them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
