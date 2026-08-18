"""Preregistration v7 robustness checks: item selection for S, and the W/S/G verdict table.

EXPLORATORY ROBUSTNESS - not a confirmatory analysis. Nothing here can change a preregistered
verdict; the three checks ask whether the judge-free M1 / non-answer signature survives other
hostile wordings (W), a five-times larger fresh item bank (S) and a 27B model (G).

Two subcommands::

    # freeze which fresh ARC items enter check S (hash rank, minus the Phase-4 capability set)
    .venv\\Scripts\\python.exe scripts/analyze_robustness.py select-items \\
        --bank results/dpo/fresh_items.jsonl \\
        --used results/raw/phase4_capability/fresh_items_used.jsonl \\
        --out results/raw/robustness/S/items_used.jsonl

    # compute every contrast and the nine verdicts
    .venv\\Scripts\\python.exe scripts/analyze_robustness.py analyze \\
        --reference results/summaries/phase1/metric_rows.csv \\
        --w-raw results/raw/robustness --s-raw results/raw/robustness/S \\
        --g-raw results/raw/robustness/G --out results/summaries/robustness

Every input is optional: a check whose raw directory is absent is reported as *not run* rather
than as a null result, and its three verdicts read "not estimable". All contrasts are item-paired
with a seeded, 2,000-resample item-clustered bootstrap. All three checks are greedy-only, so M2 -
and therefore the M2-valued H8 contrast - is never measured and is always reported as not
estimable rather than as zero.

Exit codes: 0 the analysis ran, 2 nothing could be analysed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.confirm import ConfirmError, load_judge_scores  # noqa: E402
from src.extract import LoadIssue, build_metric_rows, iter_records, read_metric_rows  # noqa: E402
from src.protocol import load_protocol  # noqa: E402
from src.robustness import (  # noqa: E402
    CONTRASTS_BY_ID, HYPOTHESIS_SHAPED, NOT_ESTIMABLE, PARSEABLE_FLOOR, PREREGISTRATION,
    W_CONTRASTS, Estimate, RobustnessError, derive_protocol, estimate_all, estimate_contrast,
    feasible, index_rows, load_task_bank, load_wording_sets, manipulation_band, non_answer_by_cell,
    parseable_rate, reparse_diagnostic, verdict_g1, verdict_g2, verdict_g3, verdict_s1,
    verdict_s2, verdict_s3, verdict_w1, verdict_w2, verdict_w3, select_bank_items, WORDINGS_FILE,
)

KOLKATA = timezone(timedelta(hours=5, minutes=30))
WORDING_SETS = ("W1", "W2", "W3")
REFERENCE_MODEL = "google/gemma-2-9b-it"
LABEL = "EXPLORATORY ROBUSTNESS - preregistration v7; changes no confirmatory verdict"


def _now() -> str:
    return datetime.now(KOLKATA).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit("invalid JSONL at %s:%d: %s" % (path, number, error))
    return out


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ==========================================================================
# select-items
# ==========================================================================

def command_select_items(args: argparse.Namespace) -> int:
    bank = _read_jsonl(ROOT / args.bank if not Path(args.bank).is_absolute() else Path(args.bank))
    if not bank:
        print("no bank items at %s" % args.bank, file=sys.stderr)
        return 2
    used_rows, used_source = [], None
    for candidate in (args.used or ()):
        path = Path(candidate)
        path = path if path.is_absolute() else ROOT / path
        rows = _read_jsonl(path)
        if rows:
            used_rows, used_source = rows, str(path)
            break
    used_ids = [str(row.get("item_id") or row.get("task_id") or row.get("id") or "") for row in used_rows]
    chosen, provenance = select_bank_items(bank, used_ids=used_ids, per_difficulty=args.per_difficulty)
    provenance["bank"] = args.bank
    provenance["used_source"] = used_source
    provenance["used_items_found"] = len(used_ids)
    provenance["generated_at"] = _now()
    provenance["preregistration"] = PREREGISTRATION
    if used_source is None:
        provenance["deviation"] = (
            "the Phase-4 capability set had not been frozen when check S was selected, so no items "
            "were removed for it; the bank is firewalled against the locked tasks and DPO training "
            "either way, and the overlap is reported rather than assumed away")
        print("WARNING: %s" % provenance["deviation"], file=sys.stderr)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(out.with_suffix(".provenance.json"), provenance)
    print("selected %d item(s): %s" % (len(chosen), provenance["selected_per_difficulty"]))
    print("shortfall: %s" % provenance["shortfall_per_difficulty"])
    print("wrote %s" % out)
    return 0


# ==========================================================================
# analyze
# ==========================================================================

def _model_source(root: Path, model_id: str) -> Path:
    """Prefer ``<dir>/<slug>.jsonl`` so a sibling file (the frozen item list) is never parsed."""
    candidate = root / ("%s.jsonl" % model_id.replace("/", "__"))
    return candidate if candidate.exists() else root


def _rows_from_raw(source: Path, protocol, label: str):
    """Stream one raw directory or file into metric rows; raw files reach gigabytes."""
    issues: list[LoadIssue] = []
    counter = [0]

    def stream():
        for record in iter_records(source, protocol=protocol, issues=issues):
            counter[0] += 1
            yield record

    rows = build_metric_rows(stream(), protocol=protocol)
    for issue in issues[:5]:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    print("%s: %d record(s) -> %d endpoint(s) (%d skipped line(s))"
          % (label, counter[0], len(rows), len(issues)))
    return rows


def _estimates_payload(estimates, keys) -> list[dict]:
    return [estimates[key].to_dict() for key in keys if key in estimates]


def _check_w(args, protocol, reference_index, reference_estimates) -> dict:
    """Three paraphrase sets against the frozen neutral cells of the same 20 discovery items."""
    root = Path(args.w_raw) if Path(args.w_raw).is_absolute() else ROOT / args.w_raw
    sets, estimable, cells, rates = {}, {}, {}, {}
    try:
        wordings = load_wording_sets(ROOT / WORDINGS_FILE)
    except RobustnessError as error:
        print("cannot read wording sets: %s" % error, file=sys.stderr)
        wordings = {}
    for name in WORDING_SETS:
        directory = root / name
        if not directory.exists():
            print("W: %s not run (%s absent)" % (name, directory), file=sys.stderr)
            continue
        derived = derive_protocol(protocol, wording=wordings[name]) if name in wordings else protocol
        rows = _rows_from_raw(_model_source(directory, args.reference_model), derived, "W %s" % name)
        if not rows:
            continue
        index = {**reference_index, **index_rows(rows, args.reference_model)}
        sets[name] = estimate_all(index, index, "W|%s" % name, W_CONTRASTS)
        # Feasibility is decided on the neutral cells, which W does not regenerate; the hostile
        # cells this set DID generate are reported beside it.
        neutral_rate, _ = parseable_rate(
            [row for row in reference_index.values()], tone="neutral")
        hostile_rate, hostile_n = parseable_rate(rows, tone="hostile")
        estimable[name] = feasible(neutral_rate)
        rates[name] = {"neutral_parseable_rate_reference": neutral_rate,
                       "hostile_parseable_rate": hostile_rate, "hostile_endpoints": hostile_n}
        cells[name] = non_answer_by_cell(rows)
    if not sets:
        return {"run": False, "reason": "no wording-set raw directory under %s" % root}
    return {
        "run": True, "raw_root": str(root), "sets": sorted(sets),
        "wordings": {name: dict(wordings[name]) for name in sorted(sets) if name in wordings},
        "estimates": {name: _estimates_payload(sets[name], W_CONTRASTS) for name in sorted(sets)},
        "frozen_reference": _estimates_payload(reference_estimates, W_CONTRASTS),
        "parseable_rates": rates,
        "non_answer_by_cell": {name: list(cells[name]) for name in sorted(cells)},
        "_estimates": sets, "_estimable": estimable,
    }


def _check_s(args, protocol) -> dict:
    root = Path(args.s_raw) if Path(args.s_raw).is_absolute() else ROOT / args.s_raw
    if not root.exists():
        return {"run": False, "reason": "raw directory absent: %s" % root}
    items_path = Path(args.s_items) if Path(args.s_items).is_absolute() else ROOT / args.s_items
    try:
        tasks = load_task_bank(items_path, protocol)
    except RobustnessError as error:
        return {"run": False, "reason": "cannot read the S item bank: %s" % error}
    derived = derive_protocol(protocol, tasks=tasks)
    rows = _rows_from_raw(_model_source(root, args.reference_model), derived, "S")
    if not rows:
        return {"run": False, "reason": "no endpoints under %s" % root}
    index = index_rows(rows, args.reference_model)
    estimates = estimate_all(index, index, "S")
    rate, endpoints = parseable_rate(rows, tone="neutral")
    counts = {"easy": len({task.task_id for task in tasks if task.difficulty == "easy"}),
              "hard": len({task.task_id for task in tasks if task.difficulty == "hard"})}
    provenance = items_path.with_suffix(".provenance.json")
    return {
        "run": True, "raw_root": str(root), "items_source": str(items_path),
        "items_provenance": json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None,
        "n_items": len(tasks), "items_per_difficulty": counts,
        "neutral_parseable_rate": rate, "neutral_endpoints": endpoints,
        "estimable": feasible(rate),
        "estimates": _estimates_payload(estimates, tuple(CONTRASTS_BY_ID)),
        "non_answer_by_cell": list(non_answer_by_cell(rows)),
        "_estimates": estimates, "_estimable": feasible(rate),
    }


def _check_g(args, protocol) -> dict:
    root = Path(args.g_raw) if Path(args.g_raw).is_absolute() else ROOT / args.g_raw
    if not root.exists():
        return {"run": False, "reason": "raw directory absent: %s" % root}
    source = _model_source(root, args.g_model)
    rows = _rows_from_raw(source, protocol, "G")
    if not rows:
        return {"run": False, "reason": "no endpoints under %s" % root}
    diagnostic = reparse_diagnostic(
        (record.response_text, record.final_answer_valid)
        for record in iter_records(source, protocol=protocol, issues=[])
        if record.turn_label == "measured" and record.sample_index == 0)
    index = index_rows(rows, args.g_model)
    estimates = estimate_all(index, index, "G")
    rate, endpoints = parseable_rate(rows, tone="neutral")
    judge, judge_source = {}, None
    if args.g_judge:
        path = Path(args.g_judge) if Path(args.g_judge).is_absolute() else ROOT / args.g_judge
        try:
            judge, judge_source = load_judge_scores(path), str(path)
        except ConfirmError as error:
            print("G: %s; the distress channel will be unavailable" % error, file=sys.stderr)
    distress = [judge[row.response_id] for row in rows
                if row.cell_kind == "factorial" and row.turn_label == "onset"
                and row.feedback_validity == "accurate" and row.tone == "hostile"
                and row.response_id in judge]
    return {
        "run": True, "raw_root": str(root), "model_id": args.g_model,
        "neutral_parseable_rate": rate, "neutral_endpoints": endpoints,
        "estimable": feasible(rate),
        "estimates": _estimates_payload(estimates, tuple(CONTRASTS_BY_ID)),
        "non_answer_by_cell": list(non_answer_by_cell(rows)),
        "trailing_marker_diagnostic": diagnostic,
        "judge_source": judge_source, "n_judge_scores": len(judge),
        "hostile_onset_distress_mean": sum(distress) / len(distress) if distress else None,
        "hostile_onset_distress_n": len(distress),
        "_estimates": estimates, "_estimable": feasible(rate),
    }


def _manipulation(args) -> dict:
    """The six new strings' rubric scores beside the frozen counterparts they replace."""
    out: dict = {"band": 1.5, "rows": [], "source": None, "frozen_source": None}
    frozen_scores: dict[str, int] = {}
    if args.frozen_manipulation:
        path = Path(args.frozen_manipulation) if Path(args.frozen_manipulation).is_absolute() else ROOT / args.frozen_manipulation
        if path.exists():
            out["frozen_source"] = str(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("scores", ()):
                if row.get("tone") == "hostile":
                    role = "correct" if "feedback_accurate_correct" in (row.get("roles") or ()) else "incorrect"
                    frozen_scores[role] = row.get("score")
    out["frozen_scores"] = frozen_scores
    if not args.manipulation:
        return out
    path = Path(args.manipulation) if Path(args.manipulation).is_absolute() else ROOT / args.manipulation
    if not path.exists():
        print("manipulation check not found at %s" % path, file=sys.stderr)
        return out
    out["source"] = str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("scores", ()):
        paths = row.get("paths") or []
        label = paths[0] if paths else ""
        # `<set>.<role>`; compared segment-wise because "incorrect".endswith("correct") is True.
        set_name, _, tail = label.partition(".")
        role = "correct" if tail == "correct" else "incorrect"
        frozen = frozen_scores.get(role)
        out["rows"].append({
            "set": set_name, "role": role, "text": row.get("text"), "score": row.get("score"),
            "frozen_counterpart_score": frozen,
            "within_band": manipulation_band(row.get("score"), frozen),
            "evidence": row.get("evidence"),
        })
    out["rows"].sort(key=lambda item: (item["set"], item["role"]))
    return out


def _verdicts(w, s, g, reference_estimates) -> list:
    out = []
    if w.get("run"):
        out.append(verdict_w1(w["_estimates"], w["_estimable"]))
        out.append(verdict_w2(w["_estimates"], reference_estimates.get("TONE_ACC_POOLED")))
        out.append(verdict_w3(w["_estimates"]))
    else:
        from src.robustness import _verdict  # noqa: PLC0415

        for key in ("W-1", "W-2", "W-3"):
            out.append(_verdict(key, NOT_ESTIMABLE, "check W was not run: %s" % w.get("reason", "absent"), {}))
    if s.get("run"):
        out.append(verdict_s1(s["_estimates"], s["_estimable"]))
        out.append(verdict_s2(s["_estimates"], reference_estimates, s["_estimable"]))
        out.append(verdict_s3(s["_estimates"], reference_estimates))
    else:
        from src.robustness import _verdict  # noqa: PLC0415

        for key in ("S-1", "S-2", "S-3"):
            out.append(_verdict(key, NOT_ESTIMABLE, "check S was not run: %s" % s.get("reason", "absent"), {}))
    if g.get("run"):
        out.append(verdict_g1(g["_estimates"], g["_estimable"]))
        out.append(verdict_g2(g["_estimates"], reference_estimates, g["_estimable"]))
        out.append(verdict_g3(g.get("hostile_onset_distress_mean"), g.get("hostile_onset_distress_n", 0)))
    else:
        out.append(verdict_g1({}, False, ran=False))
        out.append(verdict_g2({}, {}, False, ran=False))
        out.append(verdict_g3(None, 0, ran=False))
    return out


def _interval(row) -> str:
    if row is None or row.get("estimate") is None:
        return "n/a"
    if row.get("ci95_lower") is None:
        return "%.3f (no CI)" % row["estimate"]
    return "%.3f [%.3f, %.3f]" % (row["estimate"], row["ci95_lower"], row["ci95_upper"])


def _by_id(rows) -> dict:
    return {row["contrast_id"]: row for row in rows}


def render_markdown(payload: dict) -> str:
    w, s, g = payload["checks"]["W"], payload["checks"]["S"], payload["checks"]["G"]
    reference = _by_id(payload["reference"]["estimates"])
    lines = [
        "# Robustness checks W / S / G (preregistration v7)",
        "",
        "> **%s**" % LABEL,
        "> The three checks ask whether the judge-free M1 / non-answer signature survives other",
        "> hostile wordings (W), a five-times larger fresh item bank (S) and a 27B model (G).",
        "> All three are **greedy-only**: no T = 0.8 resamples were generated, so M2 - and with it",
        "> the M2-valued H8 contrast - is *not measured* and is reported as not estimable, never",
        "> as zero. Robustness of a behavioural signature across wording, items and scale is a",
        "> claim about the measurement; it licenses no claim about experience.",
        "",
        "- preregistration: `%s`" % payload["preregistration"],
        "- generated: %s" % payload["generated_at"],
        "- reference run (frozen wording, 20 locked discovery items): `%s`, `%s`" % (
            payload["reference"]["model_id"], payload["reference"]["source"]),
        "- bootstrap: %d item-clustered resamples, key `%s`" % (
            payload["bootstrap_resamples"], payload["bootstrap_key"]),
        "- feasibility clause: a wording, bank or model whose neutral-cell parseable-answer rate",
        "  is below %.0f%% has its M1 contrasts reported as *not estimable*." % (100 * PARSEABLE_FLOOR),
        "",
        "## Verdicts",
        "",
        "| ID | prediction | conf. | verdict | detail |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in payload["verdicts"]:
        lines.append("| **%s** | %s | %d%% | **%s** | %s |" % (
            item["id"], item["prediction"].replace("|", "\\|"), item["confidence_percent"],
            item["verdict"], str(item["detail"]).replace("|", "\\|")))

    lines += ["", "## W - hostile-wording paraphrases (`%s`, 20 discovery tasks, hostile cells only)"
              % payload["reference"]["model_id"], ""]
    if not w.get("run"):
        lines += ["Not run: %s" % w.get("reason", "absent"), ""]
    else:
        lines += ["The neutral cells are the existing frozen-wording discovery greedy records; only",
                  "the four hostile cells were re-generated per set.", "",
                  "| set | when incorrect / malfunctioning / onset | when correct |",
                  "| --- | --- | --- |"]
        for name in w["sets"]:
            wording = w["wordings"].get(name, {})
            lines.append("| %s | %s | %s |" % (name, wording.get("incorrect", "-"), wording.get("correct", "-")))
        lines += ["", "| contrast | stratum | frozen wording | %s |" % " | ".join(w["sets"]),
                  "| --- | --- | --- | %s" % " ".join("--- |" for _ in w["sets"])]
        by_set = {name: _by_id(w["estimates"][name]) for name in w["sets"]}
        for key in W_CONTRASTS:
            definition = CONTRASTS_BY_ID[key]
            cells = [_interval(by_set[name].get(key)) for name in w["sets"]]
            lines.append("| %s (%s) | %s | %s | %s |" % (
                key, definition.metric, definition.stratum.replace("|", "\\|"),
                _interval(reference.get(key)), " | ".join(cells)))
        lines += ["", "| set | reference neutral parseable rate | hostile parseable rate | hostile endpoints |",
                  "| --- | ---: | ---: | ---: |"]
        for name in w["sets"]:
            rate = w["parseable_rates"].get(name, {})
            lines.append("| %s | %s | %s | %d |" % (
                name,
                "%.3f" % rate["neutral_parseable_rate_reference"] if rate.get("neutral_parseable_rate_reference") is not None else "-",
                "%.3f" % rate["hostile_parseable_rate"] if rate.get("hostile_parseable_rate") is not None else "-",
                rate.get("hostile_endpoints", 0)))
        lines.append("")

    lines += ["## Manipulation check - the six new strings on the frozen context-hostility rubric", ""]
    manipulation = payload["manipulation"]
    if not manipulation.get("rows"):
        lines += ["Not run (no manipulation-check output supplied).", ""]
    else:
        lines += ["Feasibility: within +/- %.1f rubric points of the frozen counterpart. A miss is"
                  % manipulation["band"],
                  "reported, not fixed.", "",
                  "| set | role | score | frozen counterpart | within +/- 1.5 | string |",
                  "| --- | --- | ---: | ---: | :---: | --- |"]
        for row in manipulation["rows"]:
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                row["set"], row["role"], row["score"], row["frozen_counterpart_score"],
                {True: "yes", False: "**no**", None: "-"}[row["within_band"]],
                str(row["text"]).replace("|", "\\|")))
        lines.append("")

    lines += ["## S - item scale (`%s`, fresh ARC bank)" % payload["reference"]["model_id"], ""]
    if not s.get("run"):
        lines += ["Not run: %s" % s.get("reason", "absent"), ""]
    else:
        provenance = s.get("items_provenance") or {}
        shortfall = provenance.get("shortfall_per_difficulty") or {}
        lines += ["- items: **%d** (%s); shortfall against the preregistered 50+50: %s" % (
            s["n_items"],
            ", ".join("%d %s" % (count, name) for name, count in sorted(s["items_per_difficulty"].items())),
            ", ".join("%d %s" % (count, name) for name, count in sorted(shortfall.items())) or "none"),
            "- neutral-cell parseable-answer rate: %s over %d endpoints (%s)" % (
                "%.3f" % s["neutral_parseable_rate"] if s["neutral_parseable_rate"] is not None else "n/a",
                s["neutral_endpoints"], "estimable" if s["estimable"] else "**below the feasibility floor**"),
            ""]
        if provenance.get("deviation"):
            lines += ["> Deviation: %s" % provenance["deviation"], ""]
        lines += ["| contrast | stratum | fresh bank (%d items) | 20-item discovery | CI width fresh / discovery |"
                  % s["n_items"],
                  "| --- | --- | --- | --- | --- |"]
        fresh = _by_id(s["estimates"])
        for key in HYPOTHESIS_SHAPED:
            definition = CONTRASTS_BY_ID[key]
            left, right = fresh.get(key), reference.get(key)
            widths = "%s / %s" % (
                "%.3f" % left["ci95_width"] if left and left.get("ci95_width") is not None else "-",
                "%.3f" % right["ci95_width"] if right and right.get("ci95_width") is not None else "-")
            lines.append("| %s (%s) | %s | %s | %s | %s |" % (
                key, definition.metric, definition.stratum.replace("|", "\\|"),
                _interval(left), _interval(right), widths))
        lines.append("")

    lines += ["## G - model scale (`%s`, 20 discovery tasks)" % (g.get("model_id") or "google/gemma-2-27b-it"), ""]
    if not g.get("run"):
        lines += ["Not run: %s" % g.get("reason", "absent"), ""]
    else:
        lines += ["- neutral-cell parseable-answer rate: %s over %d endpoints (%s)" % (
            "%.3f" % g["neutral_parseable_rate"] if g["neutral_parseable_rate"] is not None else "n/a",
            g["neutral_endpoints"], "estimable" if g["estimable"] else "**below the feasibility floor**"),
            "- mean distress at hostile onset: %s over %d judged endpoints" % (
                "%.3f" % g["hostile_onset_distress_mean"] if g["hostile_onset_distress_mean"] is not None else "n/a",
                g["hostile_onset_distress_n"]),
            ""]
        diagnostic = g.get("trailing_marker_diagnostic") or {}
        if diagnostic.get("n_with_trailing_markers"):
            lines += [
                "> **Instrument note (diagnostic only, no verdict depends on it).** %d of %d measured"
                % (diagnostic["n_with_trailing_markers"], diagnostic["n_endpoints"]),
                "> greedy responses carry a trailing `<end_of_turn>` / `<eos>` marker inside",
                "> `response_text`. vLLM streams those markers as logprob entries that never appear in",
                "> `message.content`, and `src.backend` trims them only when the token trace is a",
                "> literal prefix of that content; this model interleaves a plain newline between the",
                "> two markers, so the prefix rule cannot fire. The frozen Amendment-A1 rule then",
                "> rejects the response because a nonempty line follows `Answer: X`. Parseable rate",
                "> under the frozen parser **%.3f**; with the trailing marker run removed it would be"
                % (diagnostic["frozen_parseable_rate"] or 0.0),
                "> **%.3f** (%d responses recovered). The frozen parser was NOT replaced and no"
                % (diagnostic["stripped_parseable_rate"] or 0.0, diagnostic["n_recovered_by_stripping"]),
                "> contrast, rate or verdict below is computed on the stripped text; the M1 channel is",
                "> reported as *not estimable* exactly as the preregistration's feasibility clause says.",
                ""]
        lines += [
            "| contrast | stratum | 27B | 9B (same 20 items) |",
            "| --- | --- | --- | --- |"]
        big = _by_id(g["estimates"])
        for key in HYPOTHESIS_SHAPED:
            definition = CONTRASTS_BY_ID[key]
            lines.append("| %s (%s) | %s | %s | %s |" % (
                key, definition.metric, definition.stratum.replace("|", "\\|"),
                _interval(big.get(key)), _interval(reference.get(key))))
        lines.append("")
        if payload.get("distress_comparison"):
            lines += ["| model | mean distress at hostile onset | endpoints |", "| --- | ---: | ---: |"]
            for row in payload["distress_comparison"]:
                lines.append("| `%s` | %s | %d |" % (
                    row["model_id"],
                    "%.3f" % row["mean"] if row["mean"] is not None else "-", row["n"]))
            lines.append("")

    lines += ["## Non-answer rate by cell (no exclusions applied)", "",
              "| check | cell | endpoint | items | non-answer rate | mean M1 (n) |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    blocks = []
    for name in (w.get("sets") or ()):
        blocks.append(("W %s" % name, w["non_answer_by_cell"].get(name, ())))
    if s.get("run"):
        blocks.append(("S", s["non_answer_by_cell"]))
    if g.get("run"):
        blocks.append(("G", g["non_answer_by_cell"]))
    if not blocks:
        lines.append("| - | none | - | - | - | - |")
    for label, records in blocks:
        for record in records:
            lines.append("| %s | %s | %s | %d | %.3f | %s |" % (
                label, record["cell_id"], record["turn_label"], record["n_items"],
                record["non_answer_rate"],
                "%.3f (%d)" % (record["mean_m1"], record["n_m1"]) if record["mean_m1"] is not None else "-"))
    lines.append("")
    return "\n".join(lines)


def command_analyze(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    reference_path = Path(args.reference) if Path(args.reference).is_absolute() else ROOT / args.reference
    if not reference_path.exists():
        print("reference metric rows not found: %s" % reference_path, file=sys.stderr)
        return 2
    reference_rows = [row for row in read_metric_rows(reference_path)
                      if row.model_id == args.reference_model and row.split == "discovery"]
    reference_index = index_rows(reference_rows, args.reference_model)
    reference_estimates = estimate_all(reference_index, reference_index, "reference|%s" % args.reference_model)
    reference_rate, reference_endpoints = parseable_rate(reference_rows, tone="neutral")
    print("reference: %d endpoint(s) for %s; neutral parseable %s"
          % (len(reference_rows), args.reference_model,
             "%.3f" % reference_rate if reference_rate is not None else "n/a"))

    w = _check_w(args, protocol, reference_index, reference_estimates)
    s = _check_s(args, protocol)
    g = _check_g(args, protocol)
    verdicts = _verdicts(w, s, g, reference_estimates)

    distress_comparison = []
    for model_id, judge_path in (("google/gemma-2-9b-it", args.reference_judge),
                                 ("google/gemma-2-2b-it", args.reference_judge)):
        if not judge_path:
            continue
        path = Path(judge_path) if Path(judge_path).is_absolute() else ROOT / judge_path
        try:
            scores = load_judge_scores(path)
        except ConfirmError:
            continue
        values = [scores[row.response_id] for row in read_metric_rows(reference_path)
                  if row.model_id == model_id and row.cell_kind == "factorial"
                  and row.turn_label == "onset" and row.feedback_validity == "accurate"
                  and row.tone == "hostile" and row.response_id in scores]
        distress_comparison.append({"model_id": model_id, "n": len(values),
                                    "mean": sum(values) / len(values) if values else None})
    if g.get("run"):
        distress_comparison.insert(0, {"model_id": g["model_id"], "n": g["hostile_onset_distress_n"],
                                       "mean": g["hostile_onset_distress_mean"]})

    payload = {
        "label": LABEL, "preregistered": True, "confirmatory": False,
        "preregistration": PREREGISTRATION, "generated_at": _now(),
        "greedy_only": True,
        "m2_note": "All three checks are greedy-only: the ten T=0.8 resamples were never generated, "
                   "so M2 and the M2-valued H8 contrast are not measured and are reported as not "
                   "estimable rather than as zero.",
        "bootstrap_resamples": 2000, "bootstrap_key": "DGS-AC1-ROBUSTNESS-v1",
        "reference": {"model_id": args.reference_model, "source": str(reference_path),
                      "n_endpoints": len(reference_rows),
                      "neutral_parseable_rate": reference_rate,
                      "neutral_endpoints": reference_endpoints,
                      "estimates": _estimates_payload(reference_estimates, tuple(CONTRASTS_BY_ID))},
        "manipulation": _manipulation(args),
        "distress_comparison": distress_comparison,
        "checks": {"W": w, "S": s, "G": g},
        "verdicts": [item.to_dict() for item in verdicts],
    }
    for check in payload["checks"].values():
        for key in ("_estimates", "_estimable"):
            check.pop(key, None)

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    _write_json(out / "robustness.json", payload)
    _write_text(out / "robustness.md", render_markdown(payload))
    print("wrote %s" % (out / "robustness.json"))
    print("wrote %s" % (out / "robustness.md"))
    for item in payload["verdicts"]:
        print("  %-4s %-15s %s" % (item["id"], item["verdict"], item["detail"]))
    return 0


# ==========================================================================

# ==========================================================================
# audit-special-tokens (amendment A6 blast-radius diagnostic; changes nothing)
# ==========================================================================

TURN_ORDER = ("initial", "feedback_response_1", "feedback_response_2", "feedback_response_3",
              "feedback_response_4", "feedback_response_5", "measured", "recovery",
              "onset", "onset_washout", "irrelevant_control", "irrelevant_control_washout")
AUDIT_SOURCES = (
    ("phase1_discovery", "results/raw/phase1/google__gemma-2-9b-it.jsonl", "google/gemma-2-9b-it"),
    ("phase2_holdout", "results/raw/phase2/google__gemma-2-9b-it.jsonl", "google/gemma-2-9b-it"),
    ("phase4_dpo_A", "results/raw/phase4/google__gemma-2-9b-it+dpo-A.jsonl", "google/gemma-2-9b-it+dpo-A"),
    ("phase4_dpo_B", "results/raw/phase4/google__gemma-2-9b-it+dpo-B.jsonl", "google/gemma-2-9b-it+dpo-B"),
)


def _audit_split(path: Path, label: str, n_examples: int = 5) -> dict:
    """One raw file: which responses end in a marker run, and what stripping them would do.

    Records are read with `json.loads` rather than through `record_from_json`, because the
    question is about the stored bytes, not about re-validating 5,720 records per file.
    """
    from src.protocol import (SPECIAL_TOKEN_STRINGS, parse_final_answer,  # noqa: PLC0415
                              strip_trailing_special_tokens)

    n_records = 0
    affected: list[dict] = []
    measured_greedy: dict[str, list[dict]] = {}
    measured_resamples: dict[tuple, dict[str, int]] = {}
    conversations: dict[tuple, list[str]] = {}
    examples: list[dict] = []
    model_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            n_records += 1
            record = json.loads(raw)
            text = record["response_text"]
            model_ids.add(record["model_id"])
            stripped = strip_trailing_special_tokens(text)
            hit = stripped != text and any(item in text[len(stripped):] for item in SPECIAL_TOKEN_STRINGS)
            frozen_valid = bool(record["final_answer_valid"])
            greedy = record["sample_index"] == 0 and record["trajectory_kind"] == "greedy"
            if record["turn_label"] == "measured" and greedy:
                measured_greedy.setdefault(record["cell_id"], []).append({
                    "task_id": record["task_id"], "frozen_valid": frozen_valid,
                    "stripped_valid": parse_final_answer(text, strip_special_tokens=True).valid,
                    "affected": hit,
                })
            # M2 is the one confirmatory channel built from resamples, and its frozen rule needs
            # ALL TEN measured resamples valid -- so a single flipped resample can move it.
            if record["turn_label"] == "measured" and not greedy and record["trajectory_kind"] == "resample":
                counts = measured_resamples.setdefault(
                    (record["task_id"], record["cell_id"]), {"n": 0, "frozen_valid": 0, "stripped_valid": 0})
                counts["n"] += 1
                counts["frozen_valid"] += int(frozen_valid)
                counts["stripped_valid"] += int(parse_final_answer(text, strip_special_tokens=True).valid)
            if not hit:
                continue
            stripped_valid = parse_final_answer(text, strip_special_tokens=True).valid
            bucket = ("frozen_valid_already" if frozen_valid
                      else "would_flip" if stripped_valid else "still_invalid")
            entry = {
                "cell_id": record["cell_id"], "turn_label": record["turn_label"],
                "sample_kind": "greedy" if greedy else "resample",
                "sample_index": record["sample_index"], "task_id": record["task_id"],
                "tone": record.get("tone"), "feedback_validity": record.get("feedback_validity"),
                "outcome": bucket, "trailing_text": text[len(stripped):],
            }
            affected.append(entry)
            conversations.setdefault(
                (record["run_id"], record["task_id"], record["cell_id"], record["sample_index"]),
                []).append(record["turn_label"])
            if len(examples) < n_examples and greedy:
                tokens = record.get("tokens") or []
                examples.append({
                    "cell_id": record["cell_id"], "turn_label": record["turn_label"],
                    "task_id": record["task_id"],
                    "response_tail": text[-70:],
                    "last_token_texts": [token["text"] for token in tokens[-8:]],
                    "n_tokens": len(tokens),
                })

    def tally(rows, key):
        out: dict[str, int] = {}
        for row in rows:
            out[str(row[key])] = out.get(str(row[key]), 0) + 1
        return dict(sorted(out.items()))

    flips = [row for row in affected if row["outcome"] == "would_flip"]
    grouped: dict[tuple, dict[str, int]] = {}
    for row in affected:
        key = (row["cell_id"], row["turn_label"], row["sample_kind"])
        cell = grouped.setdefault(key, {"n_affected": 0, "would_flip": 0,
                                        "still_invalid": 0, "frozen_valid_already": 0})
        cell["n_affected"] += 1
        cell[row["outcome"]] += 1

    cells = []
    for cell_id in sorted(measured_greedy):
        rows = measured_greedy[cell_id]
        cells.append({
            "cell_id": cell_id, "n_endpoints": len(rows),
            "n_affected": sum(1 for row in rows if row["affected"]),
            "n_would_flip": sum(1 for row in rows if row["affected"] and not row["frozen_valid"]
                                and row["stripped_valid"]),
            "non_answer_rate_frozen": sum(0.0 if row["frozen_valid"] else 1.0 for row in rows) / len(rows),
            "non_answer_rate_stripped": sum(0.0 if row["stripped_valid"] else 1.0 for row in rows) / len(rows),
        })

    m2_gained = [
        {"task_id": key[0], "cell_id": key[1],
         "frozen_valid": value["frozen_valid"], "stripped_valid": value["stripped_valid"]}
        for key, value in sorted(measured_resamples.items())
        if value["n"] == 10 and value["frozen_valid"] < 10 and value["stripped_valid"] == 10
    ]
    turn_trajectory: dict[tuple, dict[str, int]] = {}
    for row in affected:
        key = (row["turn_label"], row["sample_kind"])
        cell = turn_trajectory.setdefault(key, {"n_affected": 0, "would_flip": 0})
        cell["n_affected"] += 1
        cell["would_flip"] += int(row["outcome"] == "would_flip")

    first_turn: dict[str, int] = {}
    for labels in conversations.values():
        earliest = min(labels, key=lambda item: TURN_ORDER.index(item) if item in TURN_ORDER else 99)
        first_turn[earliest] = first_turn.get(earliest, 0) + 1
    n_conversations = len(conversations)
    not_measured = sum(count for label, count in first_turn.items() if label != "measured")

    return {
        "label": label, "source": str(path), "model_ids": sorted(model_ids),
        "n_records": n_records, "n_affected": len(affected),
        "n_would_flip": len(flips),
        "n_still_invalid": sum(1 for row in affected if row["outcome"] == "still_invalid"),
        "n_frozen_valid_already": sum(1 for row in affected if row["outcome"] == "frozen_valid_already"),
        "affected_by_sample_kind": tally(affected, "sample_kind"),
        "affected_by_turn_label": tally(affected, "turn_label"),
        "affected_by_cell_turn_sample": [
            {"cell_id": key[0], "turn_label": key[1], "sample_kind": key[2], **value}
            for key, value in sorted(grouped.items())],
        "affected_by_turn_and_trajectory": [
            {"turn_label": key[0], "sample_kind": key[1], **value}
            for key, value in sorted(turn_trajectory.items())],
        "measured_greedy_by_cell": cells,
        "m2_exposure": {
            "n_measured_item_cells": len(measured_resamples),
            "n_m2_missing_frozen": sum(1 for value in measured_resamples.values()
                                       if value["n"] != 10 or value["frozen_valid"] < 10),
            "n_m2_gained_by_strip": len(m2_gained),
            "item_cells_gained": m2_gained,
            "note": "M2's frozen rule needs all ten measured resamples valid, so one flipped "
                    "resample can move an item-cell from M2-missing to M2-present.",
        },
        "would_flip_by_tone": tally(flips, "tone"),
        "would_flip_by_validity": tally(flips, "feedback_validity"),
        "would_flip_by_turn_label": tally(flips, "turn_label"),
        "would_flip_by_sample_kind": tally(flips, "sample_kind"),
        "propagation": {
            "n_affected_conversations": n_conversations,
            "first_affected_turn_histogram": dict(sorted(first_turn.items())),
            "n_first_affected_not_measured": not_measured,
            "fraction_first_affected_not_measured": not_measured / n_conversations if n_conversations else None,
        },
        "token_examples": examples,
    }


def _render_audit(payload: dict) -> str:
    lines = [
        "# Trailing special-token audit (amendment A6 blast radius)",
        "",
        "> **Diagnostic only. Nothing here changes a verdict, a table or a figure.** A6 was NOT",
        "> adopted: its precondition (zero occurrences in previously analysed models) fails, and",
        "> this audit measures by how much. Every number below is computed with the frozen",
        "> Amendment-A1 parser; the \"stripped\" column shows what A6 *would* do if adopted.",
        "",
        "- generated: %s" % payload["generated_at"],
        "- strings audited: %s" % ", ".join("`%s`" % item for item in payload["special_token_strings"]),
        "- \"affected\" = `response_text` ends in a trailing run of those strings.",
        "- \"would flip\" = frozen parse invalid AND the stripped parse yields a well-formed",
        "  `Answer: X` final line, i.e. the answer line sits immediately before the trailing run.",
        "",
        "## Headline",
        "",
        "| split | records | affected | would flip | no answer line anyway | already valid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in payload["splits"]:
        lines.append("| `%s` | %d | **%d** | **%d** | %d | %d |" % (
            split["label"], split["n_records"], split["n_affected"], split["n_would_flip"],
            split["n_still_invalid"], split["n_frozen_valid_already"]))

    for split in payload["splits"]:
        lines += [
            "",
            "## %s (`%s`)" % (split["label"], ", ".join(split["model_ids"])),
            "",
            "- source: `%s`" % split["source"],
            "- affected by trajectory: %s" % (", ".join(
                "%s %d" % item for item in split["affected_by_sample_kind"].items()) or "none"),
            "- would-flip by tone: %s" % (", ".join(
                "**%s %d**" % item for item in split["would_flip_by_tone"].items()) or "none"),
            "- would-flip by feedback arm: %s" % (", ".join(
                "%s %d" % item for item in split["would_flip_by_validity"].items()) or "none"),
            "- would-flip by endpoint: %s" % (", ".join(
                "%s %d" % item for item in split["would_flip_by_turn_label"].items()) or "none"),
            "",
            "### (a) affected responses by endpoint x trajectory (the decision-relevant view)",
            "",
            "The confirmatory M1 contrasts read the **greedy** row of `measured` (H1/H2a/H2b),",
            "`onset` (H3a/H3b), `onset_washout` (H4a/H4b) and `recovery` (H5). M2 (H8) reads the",
            "ten **measured resamples** and needs all ten valid.",
            "",
            "| endpoint | trajectory | affected | would flip |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in split["affected_by_turn_and_trajectory"]:
            emphasis = "**" if (row["would_flip"] and row["sample_kind"] == "greedy") else ""
            lines.append("| %s | %s | %d | %s%d%s |" % (
                row["turn_label"], row["sample_kind"], row["n_affected"],
                emphasis, row["would_flip"], emphasis))
        exposure = split["m2_exposure"]
        lines += [
            "",
            "### M2 exposure (H8): item-cells whose ten measured resamples become all-valid",
            "",
            "- measured item-cells: %d; M2 missing under the frozen rule: %d" % (
                exposure["n_measured_item_cells"], exposure["n_m2_missing_frozen"]),
            "- item-cells that would GAIN an M2 value under A6: **%d**%s" % (
                exposure["n_m2_gained_by_strip"],
                (" (" + ", ".join("%s/%s" % (item["task_id"], item["cell_id"])
                                  for item in exposure["item_cells_gained"]) + ")")
                if exposure["item_cells_gained"] else ""),
            "",
            "### (a, detail) affected responses by cell x endpoint x trajectory",
            "",
            "| cell | endpoint | trajectory | affected | would flip | still invalid | already valid |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in split["affected_by_cell_turn_sample"]:
            lines.append("| %s | %s | %s | %d | %d | %d | %d |" % (
                row["cell_id"], row["turn_label"], row["sample_kind"], row["n_affected"],
                row["would_flip"], row["still_invalid"], row["frozen_valid_already"]))
        lines += [
            "",
            "### (f) measured greedy trials per cell: non-answer rate with and without the strip",
            "",
            "| cell | endpoints | affected | would flip | non-answer FROZEN | non-answer STRIPPED |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in split["measured_greedy_by_cell"]:
            lines.append("| %s | %d | %d | %d | **%.3f** | **%.3f** |" % (
                row["cell_id"], row["n_endpoints"], row["n_affected"], row["n_would_flip"],
                row["non_answer_rate_frozen"], row["non_answer_rate_stripped"]))
        propagation = split["propagation"]
        lines += [
            "",
            "### (d) propagation within a conversation",
            "",
            "- affected conversations (run x item x cell x sample): **%d**" % propagation["n_affected_conversations"],
            "- first affected turn is NOT the measured turn: **%d** (**%s**)" % (
                propagation["n_first_affected_not_measured"],
                "%.3f" % propagation["fraction_first_affected_not_measured"]
                if propagation["fraction_first_affected_not_measured"] is not None else "n/a"),
            "- first affected turn histogram: %s" % ", ".join(
                "%s %d" % item for item in propagation["first_affected_turn_histogram"].items()),
        ]
        lines += ["", "### (e) exact token pieces of the trailing run", ""]
        if not split["token_examples"]:
            lines.append("No affected greedy response in this split.")
        for example in split["token_examples"]:
            lines += [
                "- `%s` / %s / %s (%d tokens)" % (
                    example["task_id"], example["cell_id"], example["turn_label"], example["n_tokens"]),
                "  - tail: `%s`" % example["response_tail"].replace("\n", "\\n").replace("`", "'"),
                "  - last 8 token texts: %s" % ", ".join(
                    "`%s`" % token.replace("\n", "\\n").replace("`", "'") for token in example["last_token_texts"]),
            ]
    lines.append("")
    return "\n".join(lines)


def command_audit_special_tokens(args: argparse.Namespace) -> int:
    from src.protocol import SPECIAL_TOKEN_STRINGS  # noqa: PLC0415

    splits = []
    for label, relative, _model in AUDIT_SOURCES:
        path = ROOT / relative
        if not path.exists():
            print("%s: absent (%s)" % (label, path), file=sys.stderr)
            continue
        print("scanning %s ..." % label, flush=True)
        split = _audit_split(path, label)
        print("  %d records, %d affected, %d would flip"
              % (split["n_records"], split["n_affected"], split["n_would_flip"]), flush=True)
        splits.append(split)
    if not splits:
        print("no audit source found", file=sys.stderr)
        return 2
    payload = {
        "label": "DIAGNOSTIC ONLY - amendment A6 blast radius; no verdict, table or figure changes",
        "generated_at": _now(),
        "amendment": "A6 (notes/amendments.md); NOT adopted - its zero-occurrence precondition fails",
        "special_token_strings": list(SPECIAL_TOKEN_STRINGS),
        "splits": splits,
    }
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    _write_json(out / "special_token_audit.json", payload)
    _write_text(out / "special_token_audit.md", _render_audit(payload))
    print("wrote %s" % (out / "special_token_audit.json"))
    print("wrote %s" % (out / "special_token_audit.md"))
    return 0


# ==========================================================================
# audit-bogus-verdicts (does the marker artefact contaminate the ACCURATE arm?)
# ==========================================================================
#
# src/protocol.py::feedback_message, accurate arm:
#     correct = preceding.valid and preceding.letter == canonical_answer
#     return messages["accurate"][tone][
#         "if_preceding_answer_correct" if correct else "if_preceding_answer_incorrect"]
# and configs/conditions.json, generation_settings.trajectory_unit.invalid_branching:
#     "In the accurate arm, an invalid immediately preceding answer is treated as incorrect for
#      feedback branching."
# So an UNPARSEABLE preceding answer is graded incorrect. A response that ends
# `Answer: <correct letter>` followed by a trailing `<end_of_turn>`/`<eos>` run therefore receives
# a failure message inside the truthful arm: a bogus failure verdict.
GRADED_TURNS = ("initial", "feedback_response_1", "feedback_response_2", "feedback_response_3")
FEEDBACK_TURNS_ORDER = ("feedback_response_1", "feedback_response_2", "feedback_response_3")
SENSITIVITY_CONTRASTS = ("H1", "H1_hard", "H2a", "H2b", "TONE_ACC_POOLED", "NONANSWER_ACC_POOLED")
BOGUS_RAW = (
    ("phase1_discovery", "results/raw/phase1", "results/summaries/phase1/metric_rows.csv"),
    ("phase2_holdout", "results/raw/phase2", "results/summaries/phase2/metric_rows.csv"),
    ("phase4_dpo", "results/raw/phase4", None),
)
PRIMARY = "google/gemma-2-9b-it"


def _reduced_greedy_records(path: Path):
    """Stream the greedy sample-0 records of one raw file, keeping only the fields needed here.

    The byte pre-filter skips the ten resample lines per trajectory without parsing them, and the
    token trace is dropped immediately, so a gigabyte file costs a few hundred small dicts.
    """
    needle = b'"sample_index":0,'
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip() or needle not in raw:
                continue
            record = json.loads(raw)
            if record.get("trajectory_kind") != "greedy" or record.get("sample_index") != 0:
                continue
            messages = record.get("messages") or []
            yield {
                "model_id": record["model_id"], "task_id": record["task_id"],
                "cell_id": record["cell_id"], "turn_label": record["turn_label"],
                "tone": record.get("tone"), "feedback_validity": record.get("feedback_validity"),
                "response_text": record["response_text"],
                "final_answer_valid": bool(record["final_answer_valid"]),
                "final_answer_letter": record.get("final_answer_letter"),
                "last_user_message": next(
                    (item["content"] for item in reversed(messages) if item.get("role") == "user"), None),
            }


def _classify_verdicts(records, protocol) -> dict:
    """Grade every accurate-arm greedy conversation against the message it actually received."""
    from src.protocol import (parse_final_answer,  # noqa: PLC0415
                              strip_trailing_special_tokens)

    canonical = {task.task_id: task.canonical_answer for task in protocol.matched_tasks}
    accurate = protocol.conditions["feedback_messages"]["accurate"]
    onset_failure = protocol.conditions["symmetric_onset_test"]["failure_message"]
    washout = protocol.conditions["symmetric_onset_test"]["truthful_washout"]

    by_conversation: dict[tuple, dict[str, dict]] = {}
    models: set[str] = set()
    for record in records:
        models.add(record["model_id"])
        by_conversation.setdefault(
            (record["model_id"], record["task_id"], record["cell_id"]), {})[record["turn_label"]] = record

    verdicts: list[dict] = []
    contaminated: dict[tuple, dict] = {}
    onset_unconditional = {"checked": 0, "matched_tone_failure": 0, "other": 0}
    washout_rows: list[dict] = []
    for key, turns in sorted(by_conversation.items()):
        model_id, task_id, cell_id = key
        difficulty, validity, tone = cell_id.split("__")
        if validity != "accurate":
            continue
        answer = canonical.get(task_id)
        for index, turn in enumerate(FEEDBACK_TURNS_ORDER):
            record = turns.get(turn)
            preceding = turns.get(GRADED_TURNS[index])
            if record is None or preceding is None:
                continue
            issued = record["last_user_message"]
            if issued == accurate[tone]["if_preceding_answer_incorrect"]:
                verdict = "incorrect"
            elif issued == accurate[tone]["if_preceding_answer_correct"]:
                verdict = "correct"
            else:
                verdict = "other"
            text = preceding["response_text"]
            marker = strip_trailing_special_tokens(text) != text
            stripped = parse_final_answer(text, strip_special_tokens=True)
            frozen_valid = preceding["final_answer_valid"]
            if verdict != "incorrect":
                bucket = "graded_correct" if verdict == "correct" else "unrecognised_message"
            elif marker and not frozen_valid and stripped.valid and stripped.letter == answer:
                bucket = "bogus_failure"
            elif marker and not frozen_valid and stripped.valid:
                bucket = "marker_true_failure_wrong_letter"
            elif marker and not stripped.valid:
                bucket = "marker_no_answer_line"
            elif not frozen_valid:
                bucket = "non_answer_without_marker"
            elif preceding["final_answer_letter"] != answer:
                bucket = "true_failure"
            else:
                bucket = "UNEXPECTED_correct_graded_incorrect"
            verdicts.append({
                "model_id": model_id, "task_id": task_id, "cell_id": cell_id,
                "difficulty": difficulty, "tone": tone, "round": index + 1,
                "graded_turn": GRADED_TURNS[index], "verdict": verdict, "bucket": bucket,
            })
            if bucket == "bogus_failure":
                entry = contaminated.setdefault(
                    (model_id, task_id, cell_id),
                    {"model_id": model_id, "task_id": task_id, "cell_id": cell_id,
                     "difficulty": difficulty, "tone": tone, "n_bogus": 0, "first_round": index + 1})
                entry["n_bogus"] += 1
                entry["first_round"] = min(entry["first_round"], index + 1)
        # (5) the onset failure message is issued unconditionally; verify that against the transcript
        onset = turns.get("onset")
        if onset is not None:
            onset_unconditional["checked"] += 1
            if onset["last_user_message"] == onset_failure[tone]:
                onset_unconditional["matched_tone_failure"] += 1
            else:
                onset_unconditional["other"] += 1
        # (5) the washout message DOES depend on parsing the measured answer
        measured, wash = turns.get("measured"), turns.get("onset_washout")
        if measured is not None and wash is not None:
            text = measured["response_text"]
            marker = strip_trailing_special_tokens(text) != text
            stripped = parse_final_answer(text, strip_special_tokens=True)
            frozen_valid = measured["final_answer_valid"]
            frozen_correct = bool(frozen_valid and measured["final_answer_letter"] == answer)
            issued_correct = wash["last_user_message"] == washout["if_measured_trial_answer_correct"]
            truly_correct = bool(stripped.valid and stripped.letter == answer)
            washout_rows.append({
                "model_id": model_id, "task_id": task_id, "cell_id": cell_id,
                "measured_marker_terminated": marker,
                "frozen_valid": frozen_valid, "frozen_correct": frozen_correct,
                "stripped_correct": truly_correct, "issued_correct_washout": issued_correct,
                "misgraded_by_marker": bool(marker and not frozen_valid and truly_correct),
                "misgraded_any_cause": bool(issued_correct != truly_correct),
            })
    return {
        "models": sorted(models), "verdicts": verdicts,
        "contaminated": sorted(contaminated.values(), key=lambda item: (item["cell_id"], item["task_id"])),
        "onset_unconditional": onset_unconditional, "washout_rows": washout_rows,
    }


def _verdict_tables(result: dict, model_id: str) -> dict:
    rows = [row for row in result["verdicts"] if row["model_id"] == model_id]
    contaminated = [row for row in result["contaminated"] if row["model_id"] == model_id]
    buckets: dict[str, int] = {}
    by_cell: dict[str, dict[str, int]] = {}
    for row in rows:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
        cell = by_cell.setdefault(row["cell_id"], {})
        cell[row["bucket"]] = cell.get(row["bucket"], 0) + 1
    contaminated_by_cell: dict[str, dict] = {}
    for row in contaminated:
        entry = contaminated_by_cell.setdefault(
            row["cell_id"], {"n_conversations": 0, "task_ids": [], "first_rounds": {}})
        entry["n_conversations"] += 1
        entry["task_ids"].append(row["task_id"])
        key = str(row["first_round"])
        entry["first_rounds"][key] = entry["first_rounds"].get(key, 0) + 1
    washout = [row for row in result["washout_rows"] if row["model_id"] == model_id]
    return {
        "model_id": model_id,
        "n_graded_feedback_verdicts": len(rows),
        "buckets": dict(sorted(buckets.items())),
        "by_cell": {cell: dict(sorted(value.items())) for cell, value in sorted(by_cell.items())},
        "contaminated_conversations": contaminated,
        "contaminated_by_cell": dict(sorted(contaminated_by_cell.items())),
        "n_contaminated_conversations": len(contaminated),
        "washout": {
            "n_accurate_conversations": len(washout),
            "n_measured_marker_terminated": sum(1 for row in washout if row["measured_marker_terminated"]),
            "n_misgraded_by_marker": sum(1 for row in washout if row["misgraded_by_marker"]),
            "n_misgraded_any_cause": sum(1 for row in washout if row["misgraded_any_cause"]),
            "n_measured_unparseable_frozen": sum(1 for row in washout if not row["frozen_valid"]),
        },
    }


def _sensitivity(metric_path: Path, model_id: str, contaminated: list, label: str) -> dict:
    """Recompute the tone and validity contrasts with contaminated conversations removed."""
    rows = [row for row in read_metric_rows(metric_path) if row.model_id == model_id]
    index = index_rows(rows, model_id)
    excluded = {(row["task_id"], row["cell_id"]) for row in contaminated}
    kept = {key: value for key, value in index.items() if (key[0], key[1]) not in excluded}
    seed = "bogus|%s" % label
    out = []
    for key in SENSITIVITY_CONTRASTS:
        full = estimate_contrast(index, index, CONTRASTS_BY_ID[key], seed)
        trimmed = estimate_contrast(kept, kept, CONTRASTS_BY_ID[key], seed)
        out.append({
            "contrast_id": key, "metric": CONTRASTS_BY_ID[key].metric,
            "stratum": CONTRASTS_BY_ID[key].stratum,
            "all_conversations": full.to_dict(), "excluding_contaminated": trimmed.to_dict(),
            "items_dropped": full.n_items - trimmed.n_items,
        })
    return {"source": str(metric_path), "n_excluded_conversations": len(excluded), "contrasts": out}


def _render_bogus(payload: dict) -> str:
    def interval(row):
        if row is None or row.get("estimate") is None:
            return "n/a"
        if row.get("ci95_lower") is None:
            return "%.3f (no CI)" % row["estimate"]
        return "%.3f [%.3f, %.3f]" % (row["estimate"], row["ci95_lower"], row["ci95_upper"])

    lines = [
        "# Bogus failure verdicts in the ACCURATE arm (trailing special-token artefact)",
        "",
        "> **Diagnostic only. No verdict, table or figure changes.** Amendment A6 was not adopted.",
        "",
        "## (1) The grading rule, quoted",
        "",
        "`src/protocol.py::feedback_message`, accurate arm:",
        "",
        "```python",
        "correct = preceding.valid and preceding.letter == canonical_answer",
        'return messages["accurate"][tone][',
        '    "if_preceding_answer_correct" if correct else "if_preceding_answer_incorrect"]',
        "```",
        "",
        "and `configs/conditions.json`, `generation_settings.trajectory_unit.invalid_branching`:",
        "",
        "> \"In the accurate arm, an invalid immediately preceding answer is treated as incorrect",
        "> for feedback branching.\"",
        "",
        "So an unparseable preceding answer is graded **incorrect**. A response ending",
        "`Answer: <correct letter>` followed by a trailing `<end_of_turn>`/`<eos>` run therefore",
        "receives a failure message inside the truthful arm - a **bogus failure verdict**. The",
        "verdict actually issued is read from the stored transcript (the user turn preceding each",
        "feedback response), not re-derived, so these counts are what the model really saw.",
        "",
        "## (2) Graded feedback verdicts by outcome",
        "",
        "| split | model | verdicts | **bogus failure** | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in payload["splits"]:
        for table in split["models"]:
            buckets = table["buckets"]
            lines.append("| %s | `%s` | %d | **%d** | %d | %d | %d | %d | %d |" % (
                split["label"], table["model_id"], table["n_graded_feedback_verdicts"],
                buckets.get("bogus_failure", 0), buckets.get("marker_true_failure_wrong_letter", 0),
                buckets.get("marker_no_answer_line", 0), buckets.get("non_answer_without_marker", 0),
                buckets.get("true_failure", 0), buckets.get("graded_correct", 0)))

    for split in payload["splits"]:
        primary = next((item for item in split["models"] if item["model_id"] == split["primary_model"]), None)
        if primary is None:
            continue
        lines += [
            "",
            "## %s - `%s`" % (split["label"], primary["model_id"]),
            "",
            "### (2, per cell) bogus verdicts by accurate cell",
            "",
            "| cell | bogus failure | marker, wrong letter | marker, no answer line | non-answer (no marker) | true failure | graded correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for cell, buckets in primary["by_cell"].items():
            lines.append("| %s | **%d** | %d | %d | %d | %d | %d |" % (
                cell, buckets.get("bogus_failure", 0),
                buckets.get("marker_true_failure_wrong_letter", 0),
                buckets.get("marker_no_answer_line", 0), buckets.get("non_answer_without_marker", 0),
                buckets.get("true_failure", 0), buckets.get("graded_correct", 0)))
        lines += [
            "",
            "### (3) contaminated conversations (>= 1 bogus verdict), out of 10 per cell",
            "",
            "| cell | contaminated | items | first bogus round |",
            "| --- | ---: | --- | --- |",
        ]
        if not primary["contaminated_by_cell"]:
            lines.append("| - | 0 | none | - |")
        for cell, entry in primary["contaminated_by_cell"].items():
            lines.append("| %s | **%d** / 10 | %s | %s |" % (
                cell, entry["n_conversations"], ", ".join(entry["task_ids"]),
                ", ".join("round %s x%d" % item for item in sorted(entry["first_rounds"].items()))))
        washout = primary["washout"]
        lines += [
            "",
            "### (5) onset and washout",
            "",
            "- onset failure message issued unconditionally: **%d/%d** accurate conversations received"
            % (split["onset_unconditional"]["matched_tone_failure"], split["onset_unconditional"]["checked"]),
            "  the tone-matched failure string verbatim (%d other). Confirmed: no contamination path."
            % split["onset_unconditional"]["other"],
            "- washout message depends on parsing the measured answer: measured greedy answers",
            "  terminated by a marker **%d**; mis-graded washouts caused by a marker **%d**;" % (
                washout["n_measured_marker_terminated"], washout["n_misgraded_by_marker"]),
            "  mis-graded from any cause **%d**; measured answers unparseable under the frozen rule **%d**." % (
                washout["n_misgraded_any_cause"], washout["n_measured_unparseable_frozen"]),
        ]
        sensitivity = split.get("sensitivity")
        if sensitivity:
            lines += [
                "",
                "### (4) sensitivity: contrasts excluding contaminated conversations",
                "",
                "Item-paired: an item leaves a contrast when the conversation on **either** side is",
                "contaminated. Item bootstrap, 2,000 resamples, same seed on both columns, so the",
                "only difference is which items enter. Point estimates in the \"all conversations\"",
                "column reproduce the published table exactly; its interval can differ in the last",
                "decimal because this audit uses its own bootstrap seed, so compare the two columns",
                "here with each other rather than with the published interval.",
                "",
                "- conversations excluded: **%d**" % sensitivity["n_excluded_conversations"],
                "",
                "| contrast | metric | stratum | all conversations | excluding contaminated | items dropped |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
            for row in sensitivity["contrasts"]:
                lines.append("| %s | %s | %s | %s | %s | %d |" % (
                    row["contrast_id"], row["metric"], row["stratum"].replace("|", "\\|"),
                    interval(row["all_conversations"]), interval(row["excluding_contaminated"]),
                    row["items_dropped"]))
    lines.append("")
    return "\n".join(lines)


def command_audit_bogus_verdicts(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    splits = []
    for label, raw_relative, metric_relative in BOGUS_RAW:
        directory = ROOT / raw_relative
        if not directory.is_dir():
            print("%s: absent (%s)" % (label, directory), file=sys.stderr)
            continue
        records = []
        for path in sorted(directory.glob("*.jsonl")):
            if path.name.endswith(".failures.jsonl"):
                continue
            print("scanning %s / %s ..." % (label, path.name), flush=True)
            records.extend(_reduced_greedy_records(path))
        if not records:
            continue
        result = _classify_verdicts(records, protocol)
        primary_model = PRIMARY if PRIMARY in result["models"] else result["models"][0]
        tables = [_verdict_tables(result, model_id) for model_id in result["models"]]
        split = {
            "label": label, "raw_source": str(directory), "primary_model": primary_model,
            "models": tables, "onset_unconditional": result["onset_unconditional"],
        }
        if metric_relative and (ROOT / metric_relative).exists():
            contaminated = [row for row in result["contaminated"] if row["model_id"] == primary_model]
            split["sensitivity"] = _sensitivity(ROOT / metric_relative, primary_model, contaminated, label)
        splits.append(split)
        for table in tables:
            print("  %-34s bogus=%-4d contaminated conversations=%d"
                  % (table["model_id"], table["buckets"].get("bogus_failure", 0),
                     table["n_contaminated_conversations"]), flush=True)
    if not splits:
        print("no raw source found", file=sys.stderr)
        return 2
    payload = {
        "label": "DIAGNOSTIC ONLY - does the trailing-marker artefact contaminate the accurate arm?",
        "generated_at": _now(),
        "grading_rule": "src/protocol.py::feedback_message -- correct = preceding.valid and "
                        "preceding.letter == canonical_answer; conditions.json invalid_branching: "
                        "'In the accurate arm, an invalid immediately preceding answer is treated "
                        "as incorrect for feedback branching.'",
        "verdict_source": "the user turn stored in each feedback response's own transcript",
        "splits": splits,
    }
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    _write_json(out / "bogus_verdict_audit.json", payload)
    _write_text(out / "bogus_verdict_audit.md", _render_bogus(payload))
    print("wrote %s" % (out / "bogus_verdict_audit.json"))
    print("wrote %s" % (out / "bogus_verdict_audit.md"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_robustness.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-items", help="freeze the fresh items that enter check S")
    select.add_argument("--bank", default="results/dpo/fresh_items.jsonl")
    select.add_argument("--used", action="append", default=None,
                        help="items already used elsewhere (repeatable; first existing file wins)")
    select.add_argument("--per-difficulty", type=int, default=50)
    select.add_argument("--out", default="results/raw/robustness/S/items_used.jsonl")
    select.set_defaults(handler=command_select_items)

    analyze = subparsers.add_parser("analyze", help="compute every contrast and the nine verdicts")
    analyze.add_argument("--reference", default="results/summaries/phase1/metric_rows.csv",
                         help="committed Phase-1 metric rows, the frozen-wording 20-item reference")
    analyze.add_argument("--reference-model", default=REFERENCE_MODEL)
    analyze.add_argument("--reference-judge", default="results/summaries/judge/phase1/judge_records.jsonl")
    analyze.add_argument("--w-raw", default="results/raw/robustness",
                         help="directory holding one subdirectory per wording set")
    analyze.add_argument("--s-raw", default="results/raw/robustness/S")
    analyze.add_argument("--s-items", default="results/raw/robustness/S/items_used.jsonl")
    analyze.add_argument("--g-raw", default="results/raw/robustness/G")
    analyze.add_argument("--g-model", default="google/gemma-2-27b-it")
    analyze.add_argument("--g-judge", default=None)
    analyze.add_argument("--manipulation", default=None,
                         help="manipulation_check.json for the six new strings")
    analyze.add_argument("--frozen-manipulation",
                         default="results/summaries/manipulation_check/manipulation_check.json")
    analyze.add_argument("--out", default="results/summaries/robustness")
    analyze.set_defaults(handler=command_analyze)

    audit = subparsers.add_parser(
        "audit-special-tokens",
        help="DIAGNOSTIC: which stored responses end in a trailing special-token run, and what "
             "amendment A6 would do to them (changes no verdict, table or figure)")
    audit.add_argument("--out", default="results/summaries/robustness")
    audit.set_defaults(handler=command_audit_special_tokens)

    bogus = subparsers.add_parser(
        "audit-bogus-verdicts",
        help="DIAGNOSTIC: accurate-arm failure verdicts issued because a correct answer line was "
             "hidden behind a trailing special-token run, and the tone contrasts without them")
    bogus.add_argument("--out", default="results/summaries/robustness")
    bogus.set_defaults(handler=command_audit_bogus_verdicts)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
