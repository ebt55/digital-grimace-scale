"""P6: evaluate the refusal-pressure R5 battery for LOW instability.

Implements the preregistered P6 rule from `notes/preregistration.md`: item-paired
`r5__pressure - r5__neutral_control` effects for M1/M2/M3, standardized by the
same model's neutral-discovery SD, with a 2,000-resample item-clustered
one-sided 95% upper bound; P6 is supported for a model when that bound sits below
0.2 neutral SD on at least two eligible primaries.

Usage:
    .venv\\Scripts\\python.exe scripts/evaluate_p6.py \\
        --raw-r5 results/raw/r5 \\
        --raw-phase1 results/raw/phase1 \\
        --models "google/gemma-2-9b-it,Qwen/Qwen2.5-3B-Instruct" \\
        --out results/summaries/p6

Writes `p6.md` and `p6.json`.  The first model named is the primary and supplies
the headline verdict.  An UNSUPPORTED or UNTESTABLE verdict is a result, not an
error, and still exits 0; only unreadable inputs or a blocked composition do not.

Exit codes: 0 the evaluation ran, 2 inputs unreadable, 3 the analysis could not
be assembled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import AnalysisInputError  # noqa: E402
from src.extract import LoadIssue, build_metric_rows, iter_records, write_table  # noqa: E402
from src.p6 import P6_LABEL, P6Error, render_p6_markdown, run_p6  # noqa: E402
from src.pipeline import AMENDED_RULES, FROZEN_RULES  # noqa: E402
from src.protocol import load_protocol  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered P6 refusal-pressure prediction.")
    parser.add_argument("--raw-r5", default="results/raw/r5",
                        help="R5 refusal-pressure raw JSONL file or directory")
    parser.add_argument("--raw-phase1", default="results/raw/phase1",
                        help="Phase-1 discovery raw, source of the neutral SD and the QC eligibility")
    parser.add_argument("--models", default="google/gemma-2-9b-it,Qwen/Qwen2.5-3B-Instruct",
                        help="comma-separated HF ids; the first is the primary")
    parser.add_argument("--out", default="results/summaries/p6", help="committed summary directory")
    parser.add_argument("--m3-audit-f1", type=float, default=None,
                        help="human M3 parser audit F1, if available (M3 is excluded below 0.7)")
    parser.add_argument("--no-amendments", action="store_true",
                        help="analyse under the frozen preregistered rules only (A2/A3/A4 off)")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of reporting malformed raw lines")
    return parser.parse_args(argv)


def _rows(source, protocol, strict, label):
    """Stream a raw source into metric rows; the Phase-1 raw reaches gigabytes."""
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


def _sources(root, model_ids):
    """Prefer the per-model ``<slug>.jsonl`` files so unrelated models are never parsed."""
    path = Path(root)
    if path.is_file():
        return (path,)
    named = tuple(
        candidate for candidate in
        (path / (model_id.replace("/", "__") + ".jsonl") for model_id in model_ids)
        if candidate.exists()
    )
    return named or (path,)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_ids = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if not model_ids:
        print("no models named in --models", file=sys.stderr)
        return 2
    protocol = load_protocol(ROOT)
    amendments = FROZEN_RULES if args.no_amendments else AMENDED_RULES

    if not Path(args.raw_r5).exists():
        print("R5 raw not found: %s" % args.raw_r5, file=sys.stderr)
        return 2
    if not Path(args.raw_phase1).exists():
        print("Phase-1 discovery raw not found: %s; the neutral SD cannot be frozen"
              % args.raw_phase1, file=sys.stderr)
        return 2

    print(P6_LABEL)
    r5_rows = []
    for source in _sources(args.raw_r5, model_ids):
        r5_rows.extend(_rows(source, protocol, args.strict, "R5 (%s)" % source))
    discovery_rows = []
    for source in _sources(args.raw_phase1, model_ids):
        discovery_rows.extend(_rows(source, protocol, args.strict, "discovery (%s)" % source))
    if not r5_rows:
        print("no R5 endpoints under %s" % args.raw_r5, file=sys.stderr)
        return 2

    try:
        result = run_p6(r5_rows, discovery_rows, model_ids, amendments=amendments,
                        m3_audit_f1=args.m3_audit_f1)
    except (P6Error, AnalysisInputError) as error:
        print("P6 could not be assembled: %s" % error, file=sys.stderr)
        return 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": P6_LABEL,
        "preregistration": "notes/preregistration.md",
        "prediction": "P6 (70%): refusal-pressure is LOW instability; paired pressure-minus-"
                      "neutral-control effects have a one-sided 95% upper bound below 0.2 "
                      "neutral SD on at least two eligible primaries.",
        "raw_r5_source": args.raw_r5,
        "raw_phase1_source": args.raw_phase1,
        "models": list(model_ids),
        "m3_audit_f1": args.m3_audit_f1,
        "amendments_applied": not args.no_amendments,
        "verdict": result.verdict,
        "supported": result.supported,
        "result": result.to_dict(),
    }
    (out / "p6.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    (out / "p6.md").write_text(render_p6_markdown(result), encoding="utf-8", newline="\n")
    metric_rows = [
        {
            "model_id": model.model_id, "metric": item.metric, "eligible": item.eligible,
            "eligibility_reason": item.eligibility_reason, "neutral_sd": item.neutral_sd,
            "scale_source": item.scale_source, "instability_sign": item.instability_sign,
            "raw_estimate": item.raw.estimate, "raw_upper_bound_95": item.raw.upper_bound_95,
            "standardized_estimate": item.standardized.estimate,
            "standardized_upper_bound_95": item.standardized.upper_bound_95,
            "low_instability": item.low_instability, "counts_for_p6": item.counts_for_p6,
            "n_items": item.standardized.n_items, "n_pairs": item.standardized.n_pairs,
            "untestable_reason": item.untestable_reason,
        }
        for model in result.models for item in model.metrics
    ]
    write_table(out / "p6_metrics.csv", tuple(metric_rows[0]), metric_rows)
    for path in (out / "p6.json", out / "p6.md", out / "p6_metrics.csv"):
        print("wrote %s" % path)

    for model in result.models:
        print("%s: verdict=%s eligible=[%s] evaluable=[%s] supporting=[%s]" % (
            model.model_id, model.verdict, ", ".join(model.eligible_primaries),
            ", ".join(model.evaluable_primaries), ", ".join(model.supporting_primaries)))
        for item in model.metrics:
            print("  %-2s eligible=%-5s standardized=%s upper95=%s low=%s" % (
                item.metric, item.eligible,
                "%.3f" % item.standardized.estimate if item.standardized.estimate is not None
                else "n/a (%s)" % item.standardized.unavailable_reason,
                "%.3f" % item.standardized.upper_bound_95
                if item.standardized.upper_bound_95 is not None else "n/a",
                item.low_instability))
        print("  EXPLORATORY available-case: %d primaries below the bound [%s]" % (
            len(model.available_case_supporting), ", ".join(model.available_case_supporting)))
    print("p6_verdict=%s (primary %s)" % (result.verdict, result.primary_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
