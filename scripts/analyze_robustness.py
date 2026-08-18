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
    W_CONTRASTS, Estimate, RobustnessError, derive_protocol, estimate_all, feasible,
    index_rows, load_task_bank, load_wording_sets, manipulation_band, non_answer_by_cell,
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
