"""Phase-1 analysis: raw JSONL -> metric rows -> the five-gate verdict table.

Usage:
    .venv\\Scripts\\python.exe scripts/analyze_phase1.py \\
        --raw results/raw/phase1 \\
        --primary google/gemma-2-9b-it --control Qwen/Qwen2.5-7B-Instruct \\
        --style-raw results/raw/style_smoke \\
        --out results/summaries/phase1

Writes the committed metric rows, the QC table, ``gates.json`` (the full verdict
including the shuffled-label null, the BH tables and the reversal CIs) and
``gates.md`` (the human-readable verdict table).  A determinate gate FAIL is a
result and still exits 0; only unreadable data or blocked composition exits
non-zero.
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
from src.gates import BLOCKED  # noqa: E402
from src.pipeline import (  # noqa: E402
    AMENDED_RULES, FROZEN_RULES, PipelineError, excluded_task_ids, render_phase1_markdown, run_phase1_gates,
)
from src.protocol import load_protocol  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compose the preregistered Phase-1 gates.")
    parser.add_argument("--raw", default="results/raw/phase1", help="factorial raw JSONL file or directory")
    parser.add_argument("--primary", required=True, help="primary model HF id")
    parser.add_argument("--control", required=True, help="weak/null control model HF id")
    parser.add_argument("--style-raw", default=None, help="optional style-smoke raw JSONL file or directory")
    parser.add_argument("--out", default="results/summaries/phase1", help="committed summary directory")
    parser.add_argument("--m3-audit-f1", type=float, default=None, help="human M3 parser audit F1, if available")
    parser.add_argument("--strict", action="store_true", help="fail instead of reporting malformed raw lines")
    parser.add_argument(
        "--no-amendments", action="store_true",
        help="analyse under the frozen preregistered rules only (no A2 item exclusion, no A3 pooled-SD fallback)",
    )
    return parser.parse_args(argv)


def _load(source, protocol, issues, strict):
    records = load_records(source, protocol=protocol, issues=None if strict else issues)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    return records


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT)
    issues: list[LoadIssue] = []
    records = _load(args.raw, protocol, issues, args.strict)
    if not records:
        print("no raw records under %s" % args.raw, file=sys.stderr)
        return 2
    rows = build_metric_rows(records, protocol=protocol)
    amendments = FROZEN_RULES if args.no_amendments else AMENDED_RULES
    dropped = {
        model: sorted(excluded_task_ids(rows, model, amendments, phase="phase_1", split="discovery"))
        for model in (args.primary, args.control)
    }
    written = write_summaries(rows, args.out, excluded_items=dropped)
    style_rows = ()
    if args.style_raw:
        style_issues: list[LoadIssue] = []
        style_records = _load(args.style_raw, protocol, style_issues, args.strict)
        style_rows = build_metric_rows(style_records, protocol=protocol)
        written.update(write_summaries(style_rows, Path(args.out) / "style_smoke", excluded_items=dropped))
    try:
        verdict = run_phase1_gates(
            rows, args.primary, args.control,
            style_rows=style_rows or None, m3_audit_f1=args.m3_audit_f1, amendments=amendments,
        )
    except PipelineError as error:
        print("phase-1 analysis could not be assembled: %s" % error, file=sys.stderr)
        return 3
    out = Path(args.out)
    payload = {
        "raw_source": str(args.raw),
        "style_raw_source": str(args.style_raw) if args.style_raw else None,
        "record_count": len(records),
        "skipped_line_count": len(issues),
        "m3_audit_f1": args.m3_audit_f1,
        "amendments_applied": not args.no_amendments,
        "excluded_items": dropped,
        "verdict": verdict.to_dict(),
    }
    (out / "gates.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "gates.md").write_text(render_phase1_markdown(verdict), encoding="utf-8", newline="\n")
    for path in list(written.values()) + [out / "gates.json", out / "gates.md"]:
        print("wrote %s" % path)
    summary = verdict.summary
    print("rules=%s" % ("frozen" if args.no_amendments else "amended A2+A3"))
    for model, items in dropped.items():
        print("  A2 excluded for %s: %s" % (model, ", ".join(items) or "none"))
    print("phase_1_status=%s" % summary.phase_1_status)
    for name, gate in summary.gates:
        print("  %s: %s%s" % (name, gate.status, "" if gate.reason is None else " (%s)" % gate.reason))
    print("  shuffled-label null: %s" % summary.shuffled_null.status)
    return 4 if summary.phase_1_status == BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
