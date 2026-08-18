"""Score the preregistered Phase-1 human audit of the LLM judge (descriptive only).

Usage:
    .venv\\Scripts\\python.exe scripts/score_audit.py \\
        --audit-dir results/audit/phase1 \\
        --judge results/summaries/judge/phase1 \\
        --out results/summaries/judge/

Joins the blinded human scores to the pinned judge's `response_distress` scores on
`response_id`, and writes `human_audit.md` and `human_audit.json`.  The reported
Spearman correlation, MAE and within-2-point agreement are descriptive per
`notes/preregistration.md`; they are not gates and nothing here can fail a phase.
Exit is non-zero only when the audit inputs cannot be read or joined.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import (  # noqa: E402
    ANNOTATOR_NOTE, BOOTSTRAP_RESAMPLES, WITHIN_TOLERANCE, AuditError, build_report,
    render_markdown, report_payload,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Descriptive human-vs-judge agreement for the Phase-1 judge audit.")
    parser.add_argument("--audit-dir", default="results/audit/phase1",
                        help="directory holding audit_selection.json, audit_key.csv, audit_scores.csv")
    parser.add_argument("--judge", action="append", default=[], metavar="PATH",
                        help="judge_records.jsonl file or a directory containing one (repeatable)")
    parser.add_argument("--out", default="results/summaries/judge/",
                        help="output directory for human_audit.md / human_audit.json")
    parser.add_argument("--tolerance", type=int, default=WITHIN_TOLERANCE,
                        help="agreement tolerance in rubric points (preregistered: 2)")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES,
                        help="seeded bootstrap resamples for the descriptive Spearman CI")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    judge_paths = args.judge or ["results/summaries/judge/phase1"]
    try:
        report = build_report(audit_dir=args.audit_dir, judge_paths=judge_paths,
                              tolerance=args.tolerance, resamples=args.resamples,
                              annotator_note=ANNOTATOR_NOTE)
    except AuditError as exc:
        print("score_audit: %s" % exc, file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = report_payload(report)
    (out / "human_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (out / "human_audit.md").write_text(render_markdown(report), encoding="utf-8")

    overall = report.overall
    rho = "undefined" if overall.spearman.rho is None else "%.3f" % overall.spearman.rho
    print("score_audit: %d item(s); overall Spearman %s, MAE %.3f, within-%d %.3f -> %s"
          % (len(report.items), rho, overall.mae, overall.tolerance,
             overall.within_tolerance, out))
    for group in report.per_model:
        group_rho = "undefined" if group.spearman.rho is None else "%.3f" % group.spearman.rho
        print("  %-30s n=%d Spearman %s MAE %.3f within-%d %.3f"
              % (group.model_id, group.n, group_rho, group.mae, group.tolerance,
                 group.within_tolerance))
    print("  descriptive only per the preregistration; not a gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
