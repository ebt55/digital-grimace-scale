"""EXPLORATORY - preregistration v6, Phase 5: the base-model denominator.

Runs the discovery-split contrasts for the two Phase-5 columns -- the pretrained base model
`google/gemma-2-9b` and the format control `google/gemma-2-9b-it+plain`, both served through
the same plain-text template -- and prints them beside the primary model's already published
chat-template numbers.  The primary column is **quoted read-only** from
`results/summaries/phase1/exploratory/paired_contrasts.csv`; nothing here recomputes it and
nothing here can change a Phase-1 or Phase-2 verdict.

Usage (PowerShell, from the repo root)::

    .venv\\Scripts\\python.exe scripts/run_phase5.py analyze `
        --base-raw results/raw/phase1/google__gemma-2-9b.jsonl `
        --control-raw "results/raw/phase1/google__gemma-2-9b-it+plain.jsonl" `
        --base-judge results/summaries/judge/phase5_base/judge_records.jsonl `
        --control-judge results/summaries/judge/phase5_itplain/judge_records.jsonl `
        --out results/summaries/phase5

Every input except `--base-raw` is optional: a missing judge file leaves the distress
channel unavailable and L5 "not estimable", and a missing control raw leaves the it+plain
column empty.  Raw files reach gigabytes, so they are streamed one line at a time.

Exit codes: 0 the analysis ran, 2 the base model's raw could not be analysed.
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
from src.extension import (EXTENSION_HYPOTHESES, ExtensionError, analyse_split,  # noqa: E402
                           model_raw_source, model_slug)
from src.extract import LoadIssue, build_metric_rows, iter_records  # noqa: E402
from src.phase5 import (BASE_MODEL, CONTROL_MODEL, HEADLINE_IDS, PHASE5_LABEL,  # noqa: E402
                        PRIMARY_MODEL, cell_rates, feasibility, hostile_onset_distress,
                        non_answer_character, outcome_map, paired_distress_difference,
                        primary_reference, render_phase5_markdown, summary_payload, verdicts)
from src.pipeline import AMENDED_RULES, FROZEN_RULES  # noqa: E402
from src.protocol import load_protocol  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="EXPLORATORY (preregistration v6): Phase-5 base-model denominator.")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="contrasts, feasibility, distress and L1-L5")
    analyze.add_argument("--base-model", default=BASE_MODEL)
    analyze.add_argument("--control-model", default=CONTROL_MODEL)
    analyze.add_argument("--primary-model", default=PRIMARY_MODEL)
    analyze.add_argument("--base-raw", default="results/raw/phase1")
    analyze.add_argument("--control-raw", default="results/raw/phase1")
    analyze.add_argument("--base-judge", default=None)
    analyze.add_argument("--control-judge", default=None)
    analyze.add_argument("--discovery-contrasts",
                         default="results/summaries/phase1/exploratory/paired_contrasts.csv",
                         help="published Phase-1 exploratory table; the primary column")
    analyze.add_argument("--out", default="results/summaries/phase5")
    analyze.add_argument("--no-amendments", action="store_true",
                         help="analyse under the frozen rules only (A2/A3/A4 off)")
    analyze.add_argument("--strict", action="store_true",
                         help="fail instead of reporting malformed raw lines")
    return parser.parse_args(argv)


def _rows(source, protocol, strict, label):
    """Stream one raw source into metric rows; the raw files never fit in memory."""
    issues: list[LoadIssue] = []
    counter = [0]

    def stream():
        for record in iter_records(source, protocol=protocol, issues=None if strict else issues):
            counter[0] += 1
            yield record

    rows = build_metric_rows(stream(), protocol=protocol)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    print("%s: %d records -> %d endpoints (%d skipped lines)"
          % (label, counter[0], len(rows), len(issues)))
    return rows


def _judge(path, label):
    if path is None:
        print("%s: no judge records supplied; distress unavailable" % label, file=sys.stderr)
        return {}
    if not Path(path).exists():
        print("%s: judge records not found at %s; distress unavailable" % (label, path),
              file=sys.stderr)
        return {}
    try:
        scores = load_judge_scores(path)
    except ConfirmError as error:
        print("%s: %s; distress unavailable" % (label, error), file=sys.stderr)
        return {}
    print("%s: %d distress scores" % (label, len(scores)))
    return scores


def _analyse(rows, judge, model_id, protocol, amendments, source, judge_path):
    try:
        return analyse_split(rows, judge, model_id=model_id, split="discovery",
                             amendments=amendments, raw_source=source, judge_source=judge_path)
    except (ConfirmError, ExtensionError) as error:
        print("%s: could not be analysed: %s" % (model_id, error), file=sys.stderr)
        return None


def _column(model_id, raw, judge_path, protocol, amendments, strict):
    """(rows, judge, SplitAnalysis|None, rule_set) for one model's discovery split.

    Amendment A2 excludes an item whose own accurate+neutral baseline resamples are mostly
    invalid.  A model that almost never produces a parseable answer therefore has *every*
    item excluded, which would leave nothing to describe at all -- not even the non-answer
    channel the v6 feasibility clause says to discuss.  So when A2 empties the split, the
    column falls back to the frozen rules (available-case, no item exclusion) and says so;
    the number of items A2 would have removed is reported either way.
    """
    if raw is None or not Path(raw).exists():
        print("%s: raw not found at %s; column skipped" % (model_id, raw), file=sys.stderr)
        return (), {}, None, None
    source = model_raw_source(raw, model_id)
    rows = _rows(source, protocol, strict, "%s (%s)" % (model_id, source))
    judge = _judge(judge_path, "%s judge" % model_id)
    split = _analyse(rows, judge, model_id, protocol, amendments, source, judge_path)
    rule_set = "amended" if amendments is AMENDED_RULES else "frozen"
    if split is not None and split.available:
        return rows, judge, split, rule_set
    reason = "analysis_failed" if split is None else split.unavailable_reason
    if amendments is FROZEN_RULES:
        print("%s: no analysable discovery rows (%s)" % (model_id, reason), file=sys.stderr)
        return rows, judge, None, None
    print("%s: amended rules leave nothing to analyse (%s); falling back to the frozen rules "
          "(no A2 item exclusion, available-case)" % (model_id, reason), file=sys.stderr)
    fallback = _analyse(rows, judge, model_id, protocol, FROZEN_RULES, source, judge_path)
    if fallback is None or not fallback.available:
        print("%s: no analysable discovery rows under either rule set" % model_id, file=sys.stderr)
        return rows, judge, None, None
    return rows, judge, fallback, "frozen_fallback_after_a2_excluded_every_item"


def command_analyze(args) -> int:
    protocol = load_protocol(ROOT)
    amendments = FROZEN_RULES if args.no_amendments else AMENDED_RULES
    print(PHASE5_LABEL)

    base_rows, base_judge, base_split, base_rules = _column(
        args.base_model, args.base_raw, args.base_judge, protocol, amendments, args.strict)
    if base_split is None:
        print("the base model has no analysable discovery split; nothing to report", file=sys.stderr)
        return 2
    control_rows, control_judge, control_split, control_rules = _column(
        args.control_model, args.control_raw, args.control_judge, protocol, amendments, args.strict)

    base_feasibility = feasibility(base_split, args.base_model)
    control_feasibility = (None if control_split is None
                           else feasibility(control_split, args.control_model))
    base_distress = hostile_onset_distress(base_rows, base_judge, args.base_model)
    control_distress = hostile_onset_distress(control_rows, control_judge, args.control_model)
    difference = (paired_distress_difference(base_distress, control_distress,
                                             label="hostile_onset_distress|%s|%s"
                                                   % (args.base_model, args.control_model))
                  if base_distress and control_distress else None)

    base_outcomes = outcome_map(base_split)
    control_outcomes = outcome_map(control_split) if control_split is not None else {}
    primary = primary_reference(load_discovery_contrasts(args.discovery_contrasts),
                                args.primary_model)
    table = verdicts(base_feasibility=base_feasibility, base_outcomes=base_outcomes,
                     control_outcomes=control_outcomes, distress_difference=difference,
                     base_distress_n=len(base_distress), control_distress_n=len(control_distress))

    payload = summary_payload(
        base_model=args.base_model, control_model=args.control_model,
        primary_model=args.primary_model, amendments=amendments,
        base_split=base_split, control_split=control_split,
        base_feasibility=base_feasibility, control_feasibility=control_feasibility,
        primary=primary, distress_difference=difference,
        base_distress=base_distress, control_distress=control_distress,
        verdict_table=table, rule_sets={"base_plain": base_rules, "it_plain": control_rules},
        non_answer_shape={
            "base_plain": non_answer_character(base_rows, args.base_model),
            "it_plain": non_answer_character(control_rows, args.control_model),
        },
        sources={"base_raw": str(args.base_raw), "control_raw": str(args.control_raw),
                 "base_judge": args.base_judge, "control_judge": args.control_judge,
                 "discovery_contrasts": args.discovery_contrasts},
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase5.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    (out / "phase5.md").write_text(render_phase5_markdown(payload), encoding="utf-8", newline="\n")
    _write_cell_rates(out / "cell_valid_rates.csv",
                      [(args.base_model, base_split)]
                      + ([(args.control_model, control_split)] if control_split else []))
    print("wrote %s" % (out / "phase5.json"))
    print("wrote %s" % (out / "phase5.md"))

    for key in HEADLINE_IDS:
        print("  %-4s base %-28s it+plain %-28s it(chat) %s" % (
            key, _text(base_outcomes.get(key)), _text(control_outcomes.get(key)),
            _primary_text(primary.get(key))))
    for item in table:
        print("  %-3s %-14s %s" % (item.prediction_id, item.outcome, item.evidence[:150]))
    return 0


def _text(result) -> str:
    if result is None or result.estimate is None:
        return "n/a"
    if result.ci95_lower is None:
        return "%.3f (no CI)" % result.estimate
    return "%.3f [%.3f, %.3f]" % (result.estimate, result.ci95_lower, result.ci95_upper)


def _primary_text(row) -> str:
    if not row:
        return "n/a"
    return "%.3f [%.3f, %.3f]" % (row["estimate"], row["ci95_lower"], row["ci95_upper"])


def _write_cell_rates(path: Path, columns) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_id", "cell_id", "turn_label", "n_items", "n_valid",
                         "valid_answer_rate", "non_answer_rate"))
        for _model_id, split in columns:
            for rate in cell_rates(split):
                writer.writerow((
                    rate.model_id, rate.cell_id, rate.turn_label, rate.n_items,
                    "" if rate.n_valid is None else rate.n_valid,
                    "" if rate.valid_rate is None else "%.6f" % rate.valid_rate,
                    "" if rate.non_answer_rate is None else "%.6f" % rate.non_answer_rate))
    print("wrote %s" % path)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "analyze":
        return command_analyze(args)
    raise SystemExit("unknown command: %s" % args.command)


if __name__ == "__main__":
    raise SystemExit(main())
