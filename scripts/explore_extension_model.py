"""EXPLORATORY EXTENSION - not preregistered.

Run a third-family model through the *shape* of the preregistration-v3 contrast
table, on discovery and holdout separately, and print it beside the primary
model's confirmed holdout result.  This script cannot change a confirmatory
verdict: it reads `results/summaries/phase2/confirm.json` read-only and writes
only under its own ``--out`` directory.

Usage:
    .venv\\Scripts\\python.exe scripts/explore_extension_model.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --raw-phase1 results/raw/phase1 \\
        --raw-phase2 results/raw/phase2 \\
        --judge-phase1 results/summaries/judge/phase1_llama/judge_records.jsonl \\
        --judge-phase2 results/summaries/judge/phase2_llama/judge_records.jsonl \\
        --discovery-contrasts results/summaries/phase1/exploratory/paired_contrasts.csv \\
        --primary-confirm results/summaries/phase2/confirm.json \\
        --out results/summaries/extension/meta-llama__Llama-3.1-8B-Instruct

Every input is optional in practice: a split whose raw directory is absent is
reported as unavailable and the run continues with whatever splits exist.  Raw
files are read one line at a time (they reach gigabytes), and when
``<raw-dir>/<slug>.jsonl`` exists only that file is parsed.

Exit codes: 0 the exploration ran, 2 no split could be analysed at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confirm import ConfirmError, load_discovery_contrasts, load_judge_scores  # noqa: E402
from src.extension import (  # noqa: E402
    EXTENSION_LABEL, ExtensionError, analyse_split, derive_family, load_primary_holdout,
    model_raw_source, model_slug, render_extension_markdown, run_extension, unavailable_split,
)
from src.extract import LoadIssue, build_metric_rows, iter_records  # noqa: E402
from src.pipeline import AMENDED_RULES, FROZEN_RULES  # noqa: E402
from src.protocol import load_protocol  # noqa: E402

SPLIT_SOURCES = (("discovery", "raw_phase1", "judge_phase1"), ("holdout", "raw_phase2", "judge_phase2"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="EXPLORATORY EXTENSION (not preregistered): run one extra model through "
                    "the v3 contrast shapes on both splits.")
    parser.add_argument("--model", required=True, metavar="HF_ID", help="extension model HF id")
    parser.add_argument("--family", default=None,
                        help="model family label; derived from the HF id when omitted")
    parser.add_argument("--raw-phase1", default="results/raw/phase1",
                        help="discovery factorial raw JSONL file or directory")
    parser.add_argument("--raw-phase2", default="results/raw/phase2",
                        help="holdout factorial raw JSONL file or directory")
    parser.add_argument("--judge-phase1", default=None, help="discovery judge records JSONL")
    parser.add_argument("--judge-phase2", default=None, help="holdout judge records JSONL")
    parser.add_argument("--discovery-contrasts",
                        default="results/summaries/phase1/exploratory/paired_contrasts.csv",
                        help="Phase-1 exploratory contrast table, read-only cross-check")
    parser.add_argument("--primary-confirm", default="results/summaries/phase2/confirm.json",
                        help="committed confirmatory holdout result, quoted read-only")
    parser.add_argument("--out", default=None,
                        help="output directory (default results/summaries/extension/<slug>)")
    parser.add_argument("--discovery-only", action="store_true",
                        help="analyse the discovery split alone and report holdout as not run "
                             "(Phase 5's base-model denominator generates no holdout at all); "
                             "the holdout raw source is never opened")
    parser.add_argument("--no-amendments", action="store_true",
                        help="analyse under the frozen rules only (A2/A3/A4 off)")
    parser.add_argument("--strict", action="store_true", help="fail instead of reporting malformed raw lines")
    return parser.parse_args(argv)


def _rows(source, protocol, strict, label):
    """Stream a raw source into metric rows; the raw files never fit in memory."""
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


def _judge(path, label):
    """Judge scores for one split; an absent file leaves the distress contrasts empty."""
    if path is None:
        print("%s: no judge records supplied; distress contrasts will be unavailable" % label,
              file=sys.stderr)
        return {}
    if not Path(path).exists():
        print("%s: judge records not found at %s; distress contrasts will be unavailable"
              % (label, path), file=sys.stderr)
        return {}
    try:
        scores = load_judge_scores(path)
    except ConfirmError as error:
        print("%s: %s; distress contrasts will be unavailable" % (label, error), file=sys.stderr)
        return {}
    print("%s: %d distress scores" % (label, len(scores)))
    return scores


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT)
    amendments = FROZEN_RULES if args.no_amendments else AMENDED_RULES
    out = Path(args.out) if args.out else ROOT / "results" / "summaries" / "extension" / model_slug(args.model)
    print(EXTENSION_LABEL)
    print("extension model: %s (family %s)" % (args.model, args.family or derive_family(args.model)))

    splits = {}
    for split, raw_attribute, judge_attribute in SPLIT_SOURCES:
        raw = getattr(args, raw_attribute)
        judge_path = getattr(args, judge_attribute)
        if args.discovery_only and split == "holdout":
            # Declared, not inferred from a missing file: this model was never generated on
            # the holdout, so its holdout column reads "not run" rather than "unavailable".
            print("holdout: --discovery-only; split not run for this model", file=sys.stderr)
            splits[split] = unavailable_split(split, "not_run_discovery_only")
            continue
        if raw is None or not Path(raw).exists():
            print("%s: raw not found at %s; split skipped" % (split, raw), file=sys.stderr)
            splits[split] = unavailable_split(split, "raw_source_absent", raw_source=raw,
                                              judge_source=judge_path)
            continue
        source = model_raw_source(raw, args.model)
        rows = _rows(source, protocol, args.strict, "%s (%s)" % (split, source))
        judge = _judge(judge_path, "%s judge" % split)
        try:
            splits[split] = analyse_split(
                rows, judge, model_id=args.model, split=split, amendments=amendments,
                raw_source=source, judge_source=judge_path,
            )
        except (ConfirmError, ExtensionError) as error:
            print("%s: could not be analysed: %s" % (split, error), file=sys.stderr)
            splits[split] = unavailable_split(split, "analysis_failed", raw_source=source,
                                              judge_source=judge_path)

    if not any(item.available for item in splits.values()):
        print("no split could be analysed for %s" % args.model, file=sys.stderr)
        return 2

    try:
        primary_model, primary_hypotheses = load_primary_holdout(args.primary_confirm)
    except ExtensionError as error:
        print("%s; the primary comparison column will be empty" % error, file=sys.stderr)
        primary_model, primary_hypotheses = None, {}
    if not primary_hypotheses:
        print("primary confirmation unavailable at %s; comparison column will read n/a"
              % args.primary_confirm, file=sys.stderr)
    discovery_contrasts = load_discovery_contrasts(args.discovery_contrasts)

    result = run_extension(
        model_id=args.model, splits=splits, primary_hypotheses=primary_hypotheses,
        primary_model=primary_model, primary_confirm_source=args.primary_confirm,
        discovery_contrasts=discovery_contrasts, family=args.family, amendments=amendments,
    )

    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": EXTENSION_LABEL,
        "preregistered": False,
        "confirmatory": False,
        "note": "Exploratory extension model, not named in notes/preregistration_v3.md. "
                "No result here supports, refutes or amends a preregistered claim.",
        "model_id": args.model,
        "raw_phase1_source": args.raw_phase1,
        "raw_phase2_source": None if args.discovery_only else args.raw_phase2,
        "judge_phase1_source": args.judge_phase1,
        "judge_phase2_source": None if args.discovery_only else args.judge_phase2,
        "discovery_only": bool(args.discovery_only),
        "discovery_contrasts_source": args.discovery_contrasts,
        "primary_confirm_source": args.primary_confirm,
        "amendments_applied": not args.no_amendments,
        "verdict": result.verdict,
        "result": result.to_dict(),
    }
    (out / "extension.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    (out / "extension.md").write_text(
        render_extension_markdown(result), encoding="utf-8", newline="\n")
    print("wrote %s" % (out / "extension.json"))
    print("wrote %s" % (out / "extension.md"))

    shown = "discovery" if args.discovery_only else "holdout"
    for item in result.comparisons:
        outcome = item.discovery if args.discovery_only else item.holdout
        print("  %-4s %-10s %s supported=%-5s %s" % (
            item.hypothesis_id, item.outcome, shown,
            outcome.supported if outcome is not None else "n/a",
            "%.3f [%s, %s]" % (
                outcome.result.estimate,
                "%.3f" % outcome.result.ci95_lower if outcome.result.ci95_lower is not None else "n/a",
                "%.3f" % outcome.result.ci95_upper if outcome.result.ci95_upper is not None else "n/a",
            ) if outcome is not None and outcome.result.estimate is not None
            else "unavailable (%s)" % (outcome.result.unavailable_reason if outcome else "split_absent")))
    print(result.verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
