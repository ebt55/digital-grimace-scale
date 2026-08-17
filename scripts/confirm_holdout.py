"""Preregistration v3: the one permitted iteration loop, run once on the holdout.

Frozen before the holdout is analysed; its commit hash is recorded in
`manifest.json` under `holdout_unlock.confirmatory_script_commit`.

Usage:
    .venv\\Scripts\\python.exe scripts/confirm_holdout.py \\
        --raw results/raw/phase2 \\
        --style-raw results/raw/style_battery \\
        --judge results/summaries/judge/phase2/judge_records.jsonl \\
        --discovery results/summaries/phase1/exploratory/paired_contrasts.csv \\
        --out results/summaries/phase2

To exercise the code path without touching the holdout, add
``--dry-run-discovery``: it reads the Phase-1 discovery raw and judge records
instead and labels every output "DRY RUN ON DISCOVERY - NOT CONFIRMATORY".

Exit codes: 0 the analysis ran (SUCCESS or FAIL is a result, not an error),
2 inputs unreadable, 3 the analysis could not be assembled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confirm import (  # noqa: E402
    ConfirmError, load_discovery_contrasts, load_judge_scores, render_confirm_markdown,
    run_confirmation,
)
from src.extract import LoadIssue, build_metric_rows, iter_records, write_summaries, write_table  # noqa: E402
from src.pipeline import AMENDED_RULES, FROZEN_RULES  # noqa: E402
from src.protocol import load_protocol  # noqa: E402

DRY_RUN_RAW = "results/raw/phase1"
DRY_RUN_STYLE = "results/raw/style_smoke"
DRY_RUN_JUDGE = "results/summaries/judge/phase1/judge_records.jsonl"
DRY_RUN_OUT = "results/summaries/_confirm_dryrun"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the frozen preregistration-v3 holdout confirmation.")
    parser.add_argument("--raw", default="results/raw/phase2", help="factorial raw JSONL directory")
    parser.add_argument("--style-raw", default="results/raw/style_battery", help="style battery raw JSONL directory")
    parser.add_argument("--judge", default="results/summaries/judge/phase2/judge_records.jsonl",
                        help="judge records JSONL (distress scores by response_id)")
    parser.add_argument("--discovery", default="results/summaries/phase1/exploratory/paired_contrasts.csv",
                        help="discovery exploratory contrasts, printed beside each holdout estimate (read-only)")
    parser.add_argument("--out", default="results/summaries/phase2", help="committed summary directory")
    parser.add_argument("--no-amendments", action="store_true",
                        help="analyse under the frozen preregistered rules only (A2/A3/A4 off)")
    parser.add_argument("--strict", action="store_true", help="fail instead of reporting malformed raw lines")
    parser.add_argument("--dry-run-discovery", action="store_true",
                        help="point at the Phase-1 discovery data to exercise the code path; output is "
                             "labelled DRY RUN ON DISCOVERY - NOT CONFIRMATORY")
    return parser.parse_args(argv)


def _rows(source, protocol, strict, label):
    issues: list[LoadIssue] = []
    counter = [0]

    def stream():
        for record in iter_records(source, protocol=protocol, issues=None if strict else issues):
            counter[0] += 1
            yield record

    rows = build_metric_rows(stream(), protocol=protocol)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    print("%s: %d records -> %d endpoints (%d skipped lines)" % (label, counter[0], len(rows), len(issues)))
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    dry_run = bool(args.dry_run_discovery)
    raw = DRY_RUN_RAW if dry_run else args.raw
    style_raw = DRY_RUN_STYLE if dry_run else args.style_raw
    judge_path = DRY_RUN_JUDGE if dry_run else args.judge
    out = Path(DRY_RUN_OUT if dry_run else args.out)
    split = "discovery" if dry_run else "holdout"
    protocol = load_protocol(ROOT)

    if not Path(raw).exists():
        print("factorial raw not found: %s" % raw, file=sys.stderr)
        return 2
    rows = _rows(raw, protocol, args.strict, "factorial")
    style_rows = ()
    if Path(style_raw).exists():
        style_rows = _rows(style_raw, protocol, args.strict, "style battery")
    else:
        print("style battery not found at %s; H10 will be unavailable" % style_raw, file=sys.stderr)
    try:
        judge = load_judge_scores(judge_path)
    except ConfirmError as error:
        print("%s" % error, file=sys.stderr)
        return 2
    print("judge distress scores: %d" % len(judge))
    discovery = load_discovery_contrasts(args.discovery)

    label = "Preregistration v3 - holdout confirmation"
    if dry_run:
        label = "DRY RUN ON DISCOVERY - NOT CONFIRMATORY (preregistration v3 code path)"
    try:
        result = run_confirmation(
            rows, style_rows, judge, split=split, discovery=discovery,
            amendments=FROZEN_RULES if args.no_amendments else AMENDED_RULES,
            label=label, dry_run=dry_run,
        )
    except ConfirmError as error:
        print("confirmation could not be assembled: %s" % error, file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    written = write_summaries(rows, out, excluded_items={
        model_id: [item.task_id for item in items] for model_id, items in result.item_exclusions.items()})
    if style_rows:
        written.update(write_summaries(style_rows, out / "style_battery"))
    payload = {
        "label": label, "dry_run": dry_run, "split": split,
        "raw_source": str(raw), "style_raw_source": str(style_raw), "judge_source": str(judge_path),
        "discovery_source": str(args.discovery),
        "preregistration": "notes/preregistration_v3.md",
        "iteration_status": result.iteration_status,
        "result": result.to_dict(),
    }
    (out / "confirm.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (out / "confirm.md").write_text(render_confirm_markdown(result), encoding="utf-8", newline="\n")
    hypothesis_rows = [
        {
            "hypothesis_id": item.hypothesis_id, "contrast": item.contrast, "outcome": item.outcome,
            "stratum": item.stratum, "prediction": item.prediction, "discovery": item.discovery,
            "estimate": item.result.estimate, "ci95_lower": item.result.ci95_lower,
            "ci95_upper": item.result.ci95_upper, "n_items": item.result.n_items,
            "n_pairs": item.result.n_pairs, "supported": item.supported,
            "p_two_sided": item.result.p_two_sided, "bh_adjusted_p": item.adjusted_p,
            "unavailable_reason": item.result.unavailable_reason,
        }
        for item in result.hypotheses
    ]
    write_table(out / "hypotheses.csv", tuple(hypothesis_rows[0]), hypothesis_rows)
    null_rows = [
        {
            "hypothesis_id": item.hypothesis_id, "estimate": item.result.estimate,
            "ci95_lower": item.result.ci95_lower, "ci95_upper": item.result.ci95_upper,
            "n_items": item.result.n_items, "supported": item.supported,
        }
        for item in result.null_outcomes
    ]
    write_table(out / "shuffled_null.csv", tuple(null_rows[0]), null_rows)
    for path in list(written.values()) + [out / "confirm.json", out / "confirm.md",
                                          out / "hypotheses.csv", out / "shuffled_null.csv"]:
        print("wrote %s" % path)
    if dry_run:
        print("DRY RUN ON DISCOVERY - NOT CONFIRMATORY")
    print("iteration_status=%s" % result.iteration_status)
    for item in result.hypotheses:
        print("  %-4s %-8s supported=%-5s %s" % (
            item.hypothesis_id, item.outcome, item.supported,
            "%.3f [%s, %s]" % (
                item.result.estimate,
                "%.3f" % item.result.ci95_lower if item.result.ci95_lower is not None else "n/a",
                "%.3f" % item.result.ci95_upper if item.result.ci95_upper is not None else "n/a",
            ) if item.result.estimate is not None else "unavailable (%s)" % item.result.unavailable_reason))
    print("  H7 (boundary) supported=%s; H10 (style) supported=%s" % (result.h7_supported, result.h10_supported))
    print("  permutation null: real_count=%d/%d over %s, null_p=%.4f -> %s" % (
        result.null_check.real_count, len(result.null_check.family),
        ",".join(result.null_check.family), result.null_check.null_p,
        "PASSES" if result.null_check.passes else "FAILS"))
    print("  permutation histogram: %s" % dict(result.null_check.histogram))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
