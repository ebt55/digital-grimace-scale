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

from src.extract import LoadIssue, build_metric_rows, iter_records, write_summaries  # noqa: E402
from src.pipeline import AMENDED_RULES, FROZEN_RULES, phase0_screen, render_phase0_markdown  # noqa: E402
from src.protocol import load_protocol  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the preregistered Phase-0 screen.")
    parser.add_argument("--raw", default="results/raw/phase0", help="raw JSONL file or directory")
    parser.add_argument("--out", default="results/summaries/phase0", help="committed summary directory")
    parser.add_argument("--strict", action="store_true", help="fail instead of reporting malformed raw lines")
    parser.add_argument(
        "--no-amendments", action="store_true",
        help="make the frozen preregistered rules authoritative (no A2 item exclusion, no A3 pooled-SD fallback); "
             "both selections are reported either way",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT)
    issues: list[LoadIssue] = []
    counter = [0]

    def stream():
        for record in iter_records(args.raw, protocol=protocol, issues=None if args.strict else issues):
            counter[0] += 1
            yield record

    # Streamed: raw files are far too large to materialise as a list of records.
    rows = build_metric_rows(stream(), protocol=protocol)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    if not counter[0]:
        print("no raw records under %s" % args.raw, file=sys.stderr)
        return 2
    phase0_rows = [row for row in rows if row.phase == "phase_0"]
    if not phase0_rows:
        print("no phase_0 endpoints in %s" % args.raw, file=sys.stderr)
        return 2
    amendments = FROZEN_RULES if args.no_amendments else AMENDED_RULES
    result = phase0_screen(rows, protocol=protocol, amendments=amendments)
    written = write_summaries(
        rows, args.out,
        excluded_items={model: [item.task_id for item in items] for model, items in result.item_exclusions.items()},
    )
    out = Path(args.out)
    payload = {
        "raw_source": str(args.raw),
        "skipped_line_count": len(issues),
        "record_count": counter[0],
        "amendments_authoritative": result.amendments_authoritative,
        "screen": result.to_dict(),
    }
    (out / "screen.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "screen.md").write_text(render_phase0_markdown(result), encoding="utf-8", newline="\n")
    for path in list(written.values()) + [out / "screen.json", out / "screen.md"]:
        print("wrote %s" % path)
    for label, selection in (("frozen ", result.frozen_selection), ("amended", result.amended_selection)):
        print("%s: status=%s primary=%s control=%s screen_null=%s" % (
            label, selection.status, selection.primary_model_id, selection.control_model_id, selection.screen_null))
    for model_id in sorted(result.item_exclusions):
        print("A2 excluded for %s: %s" % (
            model_id, ", ".join(item.task_id for item in result.item_exclusions[model_id])))
    selection = result.selection
    print("authoritative (%s): status=%s primary=%s control=%s" % (
        "amended" if result.amendments_authoritative else "frozen",
        selection.status, selection.primary_model_id, selection.control_model_id))
    if selection.status == "blocked":
        print("blocked: %s" % selection.blocked_reason, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
