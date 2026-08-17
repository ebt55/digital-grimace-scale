"""Phase-0 screen: raw JSONL -> committed metric rows -> primary/control selection.

Usage:
    .venv\\Scripts\\python.exe scripts/screen_phase0.py \\
        --raw results/raw/phase0 --out results/summaries/phase0

A screen null is a result, not an error: the script still exits 0 and writes the
selection with its explicit null label.  It exits non-zero only when the raw data
cannot be read or contains no Phase-0 endpoints at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extract import LoadIssue, build_metric_rows, load_records, write_summaries  # noqa: E402
from src.pipeline import phase0_screen, render_phase0_markdown  # noqa: E402
from src.protocol import load_protocol  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the preregistered Phase-0 screen.")
    parser.add_argument("--raw", default="results/raw/phase0", help="raw JSONL file or directory")
    parser.add_argument("--out", default="results/summaries/phase0", help="committed summary directory")
    parser.add_argument("--strict", action="store_true", help="fail instead of reporting malformed raw lines")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT)
    issues: list[LoadIssue] = []
    records = load_records(args.raw, protocol=protocol, issues=None if args.strict else issues)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    if not records:
        print("no raw records under %s" % args.raw, file=sys.stderr)
        return 2
    rows = build_metric_rows(records, protocol=protocol)
    phase0_rows = [row for row in rows if row.phase == "phase_0"]
    if not phase0_rows:
        print("no phase_0 endpoints in %s" % args.raw, file=sys.stderr)
        return 2
    written = write_summaries(rows, args.out)
    result = phase0_screen(rows, protocol=protocol)
    out = Path(args.out)
    payload = {
        "raw_source": str(args.raw),
        "skipped_line_count": len(issues),
        "record_count": len(records),
        "screen": result.to_dict(),
    }
    (out / "screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "screen.md").write_text(render_phase0_markdown(result), encoding="utf-8", newline="\n")
    for path in list(written.values()) + [out / "screen.json", out / "screen.md"]:
        print("wrote %s" % path)
    selection = result.selection
    print("status=%s primary=%s control=%s screen_null=%s" % (
        selection.status, selection.primary_model_id, selection.control_model_id, selection.screen_null))
    if selection.status == "blocked":
        print("blocked: %s" % selection.blocked_reason, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
