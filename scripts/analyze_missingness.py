"""Missing-data sensitivity analysis for the M1 contrasts (offline, no GPU, no judge).

M1 is reported available-case; non-answers concentrate in the hostile cells, so
the item-paired contrasts are computed on the items that answered in both cells.
This script asks whether the published effects survive three alternative
treatments of the missing values, and reports the tipping point.

Usage:
    .venv\\Scripts\\python.exe scripts/analyze_missingness.py \\
        --discovery-rows results/summaries/phase1/metric_rows.csv \\
        --holdout-rows results/summaries/phase2/metric_rows.csv \\
        --out results/summaries/missingness

Amendments, matching the published analyses exactly:

* **discovery** (`google/gemma-2-9b-it`, `Qwen/Qwen2.5-3B-Instruct`): no A2 item
  exclusion, because the published discovery numbers are the Phase-1
  *exploratory* contrast table, which applies no quality-control exclusion.
* **holdout**: A2 on, as in the frozen confirmatory script.  A2 excluded no
  holdout item for either model, so this is a no-op on the numbers.
* **extension arm** (optional third family): A2 on for both splits, as in
  `scripts/explore_extension_model.py`.

The extension arm has no committed per-item table, so `--extra-raw-discovery` /
`--extra-raw-holdout` re-extract it from the raw JSONL with the same frozen
parser (greedy sample 0 carries M1).  Omit them and the arm is skipped.

Exit codes: 0 the analysis ran, 2 inputs unreadable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extension import model_raw_source  # noqa: E402
from src.extract import LoadIssue, build_metric_rows, iter_records, read_metric_rows, write_table  # noqa: E402
from src.missingness import (  # noqa: E402
    CONTROL_MODEL, PER_ITEM_COLUMNS, PRIMARY_MODEL, MissingnessError, load_discovery_published,
    load_extension_published, load_holdout_published, per_item_rows, render_markdown, run_missingness,
)
from src.pipeline import item_exclusions  # noqa: E402
from src.protocol import load_protocol  # noqa: E402

LABEL = "M1 missing-data sensitivity analysis (available-case, zero-imputation, worst-case bounds, tipping point)"
DEFAULT_EXTENSION = "meta-llama/Llama-3.1-8B-Instruct"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M1 missing-data sensitivity analysis.")
    parser.add_argument("--discovery-rows", default="results/summaries/phase1/metric_rows.csv",
                        help="committed Phase-1 discovery metric rows")
    parser.add_argument("--holdout-rows", default="results/summaries/phase2/metric_rows.csv",
                        help="committed Phase-2 holdout metric rows")
    parser.add_argument("--primary", default=PRIMARY_MODEL)
    parser.add_argument("--control", default=CONTROL_MODEL)
    parser.add_argument("--discovery-published",
                        default="results/summaries/phase1/exploratory/paired_contrasts.csv",
                        help="published discovery contrasts, read-only reproduction check")
    parser.add_argument("--holdout-published", default="results/summaries/phase2/hypotheses.csv",
                        help="published holdout contrasts, read-only reproduction check")
    parser.add_argument("--extra-model", default=DEFAULT_EXTENSION,
                        help="optional third-family model for an exploratory extension arm")
    parser.add_argument("--extra-raw-discovery", default=None,
                        help="raw JSONL file or directory holding the extension model's discovery run")
    parser.add_argument("--extra-raw-holdout", default=None,
                        help="raw JSONL file or directory holding the extension model's holdout run")
    parser.add_argument("--extra-rows", default=None, nargs=2, metavar=("DISCOVERY_CSV", "HOLDOUT_CSV"),
                        help="already-extracted extension metric rows, instead of re-reading raw")
    parser.add_argument("--extra-published",
                        default="results/summaries/extension/meta-llama__Llama-3.1-8B-Instruct/extension.json",
                        help="committed extension result, read-only reproduction check")
    parser.add_argument("--out", default="results/summaries/missingness")
    return parser.parse_args(argv)


def _extract(source, protocol, label):
    """Stream one raw source into metric rows; the raw files reach gigabytes."""
    issues: list[LoadIssue] = []
    rows = build_metric_rows(iter_records(source, protocol=protocol, issues=issues), protocol=protocol)
    print("%s: %d endpoints (%d skipped lines)" % (label, len(rows), len(issues)))
    return rows


def _extension_rows(args, protocol):
    """The optional third-family rows, from committed CSVs or from raw."""
    if args.extra_rows:
        return {split: read_metric_rows(path)
                for split, path in zip(("discovery", "holdout"), args.extra_rows)
                if Path(path).exists()}
    out = {}
    for split, raw in (("discovery", args.extra_raw_discovery), ("holdout", args.extra_raw_holdout)):
        if not raw or not Path(raw).exists():
            continue
        source = model_raw_source(raw, args.extra_model)
        out[split] = _extract(source, protocol, "%s %s" % (args.extra_model, split))
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(ROOT)
    for path in (args.discovery_rows, args.holdout_rows):
        if not Path(path).exists():
            print("metric rows not found: %s" % path, file=sys.stderr)
            return 2
    rows_by_split = {
        "discovery": list(read_metric_rows(args.discovery_rows)),
        "holdout": list(read_metric_rows(args.holdout_rows)),
    }
    models = {"primary": args.primary, "control": args.control}
    published = {}
    for key, reference in load_discovery_published(args.discovery_published).items():
        published[("discovery", key[0], key[1])] = reference
    for key, reference in load_holdout_published(args.holdout_published, models).items():
        published[("holdout", key[0], key[1])] = reference

    # A2 follows each published analysis: off on discovery (the published
    # discovery numbers are the exploratory table, which excludes nothing), on
    # for the holdout confirmation and for the extension arm on both splits.
    excluded_by = {}
    for model_id in models.values():
        excluded_by[(model_id, "discovery")] = ()
        excluded_by[(model_id, "holdout")] = tuple(
            item.task_id for item in item_exclusions(rows_by_split["holdout"], model_id, split="holdout"))

    extension_rows = _extension_rows(args, protocol) if args.extra_model else {}
    if extension_rows:
        models["extension"] = args.extra_model
        for split, rows in extension_rows.items():
            rows_by_split.setdefault(split, [])
            rows_by_split[split] = list(rows_by_split[split]) + list(rows)
            excluded_by[(args.extra_model, split)] = tuple(
                item.task_id for item in item_exclusions(rows, args.extra_model, split=split))
        if args.extra_published:
            published.update(load_extension_published(args.extra_published, args.extra_model))
    else:
        print("no extension rows supplied; the third-family arm is skipped", file=sys.stderr)

    amendment_note = {
        "a2_discovery": "Amendment A2 (treatment-blind item exclusion) is **off** on discovery, "
                        "matching the published exploratory contrast table, which applies no "
                        "quality-control exclusion.",
        "a2_holdout": "A2 is **on** for the holdout, as in the frozen confirmatory script; it "
                      "excluded no holdout item for either model (%s), so it changes nothing." % (
                          ", ".join("%s: %s" % (model_id.split("/")[-1],
                                                ", ".join(excluded_by[(model_id, "holdout")]) or "none")
                                    for model_id in models.values()
                                    if (model_id, "holdout") in excluded_by)),
    }
    if "extension" in models:
        excluded = sorted({task for split in ("discovery", "holdout")
                           for task in excluded_by.get((args.extra_model, split), ())})
        amendment_note["a2_extension"] = (
            "For the exploratory extension arm `%s`, A2 is **on** for both splits, as in the "
            "committed extension run (excluded: %s)." % (args.extra_model, ", ".join(excluded) or "none"))

    try:
        report = run_missingness(
            rows_by_split, models=models, excluded_by=excluded_by, published=published, label=LABEL,
            sources={
                "discovery metric rows": args.discovery_rows,
                "holdout metric rows": args.holdout_rows,
                "published discovery contrasts": args.discovery_published,
                "published holdout contrasts": args.holdout_published,
                **({"extension model": args.extra_model} if "extension" in models else {}),
                **({"published extension contrasts": args.extra_published}
                   if "extension" in models and args.extra_published else {}),
            },
            amendment_note=amendment_note,
        )
    except MissingnessError as error:
        print("sensitivity analysis could not be assembled: %s" % error, file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m1_missingness.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    (out / "m1_missingness.json").write_text(
        json.dumps({"label": LABEL, "preregistered": False, "confirmatory": False,
                    "note": "Sensitivity analysis. The committed available-case estimates are "
                            "unchanged; nothing here amends a preregistered result.",
                    "result": report.to_dict()},
                   indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    items = per_item_rows(report, rows_by_split)
    write_table(out / "m1_missingness_items.csv", PER_ITEM_COLUMNS,
                [{column: row.get(column) for column in PER_ITEM_COLUMNS} for row in items])
    for name in ("m1_missingness.md", "m1_missingness.json", "m1_missingness_items.csv"):
        print("wrote %s" % (out / name))

    for outcome in report.outcomes:
        available = outcome.treatment("available_case")
        zero = outcome.treatment("zero_imputation")
        print("  %-11s %-9s %-24s %-22s %-22s delta=%s  %s" % (
            outcome.contrast_id, outcome.split, outcome.model_id.split("/")[-1],
            _short(available), _short(zero),
            "n/a" if outcome.tipping.delta is None else "%.2f" % outcome.tipping.delta,
            outcome.verdict))
    return 0


def _short(treatment) -> str:
    if treatment is None or treatment.result.estimate is None:
        return "unavailable"
    if treatment.result.ci95_lower is None:
        return "%.2f [no CI]" % treatment.result.estimate
    return "%.2f [%.2f, %.2f]" % (
        treatment.result.estimate, treatment.result.ci95_lower, treatment.result.ci95_upper)


if __name__ == "__main__":
    raise SystemExit(main())
