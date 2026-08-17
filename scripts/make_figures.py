"""Regenerate F1, F2, F4 and the FH holdout figures from the committed summaries.

Usage:
    .venv\\Scripts\\python.exe scripts/make_figures.py \\
        --summaries results/summaries --out results/figures

Figures never read ``results/raw``: everything comes from ``screen.json``,
``gates.json``, ``hypotheses.csv`` and ``confirm.json``, so a fresh clone of the
repository reproduces every panel.  Colours are the Okabe-Ito colourblind-safe
palette; each figure is written as both PNG and SVG.  Any figure whose summary
is absent is skipped with a message rather than crashing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"
METRICS = ("M1", "M2", "M3")
METRIC_COLOURS = {"M1": BLUE, "M2": VERMILLION, "M3": GREEN}
INTERPRETATION_NOTE = (
    "Higher = more instability (M1 sign-aligned). Discovery only; not evidence of experience."
)


def _read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save(figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def figure_f1(screen, out_dir: Path):
    """F1 -- Phase-0 standardised screen deltas per model x metric."""
    models = screen["screen"]["selection"]["models"]
    model_ids = sorted(models)
    if not model_ids:
        return []
    figure, axes = plt.subplots(figsize=(1.6 + 1.5 * len(model_ids), 4.4))
    width = 0.26
    for index, metric in enumerate(METRICS):
        offsets = [position + (index - 1) * width for position in range(len(model_ids))]
        values = [models[model_id]["metrics"][metric]["signed_delta"] for model_id in model_ids]
        drawn = [0.0 if value is None else value for value in values]
        axes.bar(offsets, drawn, width, label=metric, color=METRIC_COLOURS[metric],
                 edgecolor="black", linewidth=0.4)
        for offset, value in zip(offsets, values):
            if value is None:
                axes.text(offset, 0.0, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xticks(range(len(model_ids)))
    axes.set_xticklabels([_short(model_id) for model_id in model_ids], rotation=20, ha="right", fontsize=9)
    axes.set_ylabel("sign-aligned screen delta (neutral SD)")
    selection = screen["screen"]["selection"]
    axes.set_title("F1 - Phase-0 screen: false-failure minus accurate\nstatus=%s primary=%s control=%s" % (
        selection["status"], _short(selection["primary_model_id"] or "none"),
        _short(selection["control_model_id"] or "none")), fontsize=10)
    for model_index, model_id in enumerate(model_ids):
        if models[model_id]["coherent"]:
            axes.text(model_index, axes.get_ylim()[1] * 0.96, "coherent", ha="center", fontsize=7, color=PURPLE)
    axes.legend(title="metric", fontsize=8, title_fontsize=8)
    figure.supxlabel(INTERPRETATION_NOTE, fontsize=7, color="#444444")
    figure.tight_layout()
    return _save(figure, out_dir, "F1_phase0_screen_deltas")


def _coefficients(verdict, effect):
    out = {}
    for model_id, analysis in verdict["models"].items():
        for metric, result in analysis["real_g1"].items():
            coefficient = None if result is None else result.get(effect)
            if coefficient is None:
                continue
            sign = -1.0 if metric == "M1" else 1.0
            low, high = coefficient["ci95"]
            out[(model_id, metric)] = (
                sign * coefficient["coefficient"],
                sign * (low if sign > 0 else high),
                sign * (high if sign > 0 else low),
                coefficient["adjusted_p"],
            )
    return out


def figure_f2(gates, out_dir: Path):
    """F2 -- Phase-1 adjusted validity and tone effects with 95% CIs, per model."""
    verdict = gates["verdict"]
    model_ids = [verdict["primary_model_id"], verdict["control_model_id"]]
    model_ids = [model_id for model_id in model_ids if model_id in verdict["models"]]
    if not model_ids:
        return []
    figure, axes_pair = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for axes, effect, colour, label in zip(
        axes_pair, ("validity", "tone"), (BLUE, ORANGE),
        ("false-failure feedback", "hostile tone"),
    ):
        coefficients = _coefficients(verdict, effect)
        positions, labels, groups = [], [], []
        position = 0
        for model_id in model_ids:
            groups.append((position + (len(METRICS) - 1) / 2, _short(model_id)))
            for metric in METRICS:
                entry = coefficients.get((model_id, metric))
                labels.append(metric)
                positions.append(position)
                if entry is not None:
                    value, low, high, adjusted = entry
                    axes.errorbar(
                        position, value, yerr=[[value - low], [high - value]], fmt="o",
                        color=colour if adjusted is None or adjusted >= 0.01 else VERMILLION,
                        markersize=6, capsize=4, linewidth=1.6,
                    )
                    if adjusted is not None and adjusted < 0.01:
                        axes.annotate("BH p<.01", (position, high), textcoords="offset points",
                                      xytext=(0, 7), ha="center", fontsize=7, color=VERMILLION,
                                      annotation_clip=False)
                else:
                    axes.text(position, 0.0, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
                position += 1
            position += 0.8
        axes.axhline(0.0, color="black", linewidth=0.8)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, fontsize=8)
        axes.margins(y=0.16)
        for centre, name in groups:
            axes.annotate(name, (centre, 0), xycoords=("data", "axes fraction"), xytext=(0, -26),
                          textcoords="offset points", ha="center", fontsize=8, annotation_clip=False)
        axes.set_title("F2 - adjusted %s effect" % label, fontsize=10)
        axes.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axes_pair[0].set_ylabel("sign-aligned effect (neutral SD)")
    figure.suptitle(
        "Phase-1 discovery: metric ~ feedback_validity + tone + difficulty + correctness + length + (1|item)",
        fontsize=10)
    figure.supxlabel(INTERPRETATION_NOTE, fontsize=7, color="#444444")
    figure.tight_layout()
    return _save(figure, out_dir, "F2_phase1_adjusted_effects")


def figure_f4(gates, out_dir: Path):
    """F4 -- cause-removal reversal: accurate vs malfunctioning vs recovery."""
    verdict = gates["verdict"]
    model_ids = [model_id for model_id in (verdict["primary_model_id"], verdict["control_model_id"])
                 if model_id in verdict["models"]]
    endpoints = (
        ("measured_accurate", "accurate baseline", GREEN),
        ("measured_malfunctioning", "measured (false failure)", VERMILLION),
        ("post_correction_malfunctioning", "recovery (after correction)", SKY),
    )
    figure, axes_row = plt.subplots(1, max(1, len(model_ids)), figsize=(5.6 * max(1, len(model_ids)), 4.6), squeeze=False)
    drew = False
    for axes, model_id in zip(axes_row[0], model_ids):
        profiles = {item["metric_name"]: item for item in verdict["models"][model_id]["reversal"]}
        width = 0.26
        for index, (key, label, colour) in enumerate(endpoints):
            offsets, values, lower, upper = [], [], [], []
            for metric_index, metric in enumerate(METRICS):
                profile = profiles.get(metric)
                estimate = None if profile is None else profile[key]
                if estimate is None or estimate["value"] is None:
                    continue
                offsets.append(metric_index + (index - 1) * width)
                values.append(estimate["value"])
                lower.append(estimate["value"] - estimate["ci95_lower"])
                upper.append(estimate["ci95_upper"] - estimate["value"])
            if not offsets:
                continue
            drew = True
            axes.bar(offsets, values, width, label=label, color=colour, edgecolor="black", linewidth=0.4)
            axes.errorbar(offsets, values, yerr=[lower, upper], fmt="none", ecolor="black", capsize=3, linewidth=1.0)
        axes.axhline(0.0, color="black", linewidth=0.8)
        axes.set_xticks(range(len(METRICS)))
        axes.set_xticklabels(METRICS)
        axes.set_title("F4 - reversal, %s" % _short(model_id), fontsize=10)
        axes.set_ylabel("sign-aligned z (neutral SD)")
        axes.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    if not drew:
        plt.close(figure)
        return []
    axes_row[0][0].legend(fontsize=8)
    figure.supxlabel(
        "False-negative-eligible subset; 2,000 item-clustered bootstrap resamples. " + INTERPRETATION_NOTE,
        fontsize=7, color="#444444")
    figure.tight_layout()
    return _save(figure, out_dir, "F4_cause_removal_reversal")


def _read_csv(path: Path):
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def number(value):
        try:
            return float(value) if value != "" else None
        except ValueError:
            return value

    return [{key: number(value) for key, value in row.items()} for row in rows]


ENDPOINT_ORDER = ("measured", "recovery", "onset", "onset_washout")
ENDPOINT_COLOURS = {"measured": VERMILLION, "recovery": SKY, "onset": PURPLE, "onset_washout": GREEN}


def figure_exploratory(summary, out_dir: Path, model_id: str):
    """EXPLORATORY - M1 and non-answer rate by cell and endpoint for one model."""
    rows = [row for row in summary if row["model_id"] == model_id and row["cell_kind"] == "factorial"]
    if not rows:
        return []
    difficulties = [value for value in ("easy", "hard") if any(row["difficulty"] == value for row in rows)]
    figure, axes_grid = plt.subplots(2, len(difficulties), figsize=(6.6 * len(difficulties), 8.2), squeeze=False)
    for column, difficulty in enumerate(difficulties):
        arms = sorted({row["cell_id"] for row in rows if row["difficulty"] == difficulty})
        for panel, (field, label) in enumerate((("m1", "M1 margin (raw)"), ("non_answer_rate", "non-answer rate"))):
            axes = axes_grid[panel][column]
            width = 0.2
            for index, endpoint in enumerate(ENDPOINT_ORDER):
                offsets, values, lower, upper = [], [], [], []
                for position, cell_id in enumerate(arms):
                    match = next((row for row in rows if row["cell_id"] == cell_id and row["turn_label"] == endpoint), None)
                    if match is None or match["mean_" + field] is None:
                        continue
                    offsets.append(position + (index - 1.5) * width)
                    values.append(match["mean_" + field])
                    low, high = match.get("ci95_lower_" + field), match.get("ci95_upper_" + field)
                    lower.append(match["mean_" + field] - low if low is not None else 0.0)
                    upper.append(high - match["mean_" + field] if high is not None else 0.0)
                if not offsets:
                    continue
                axes.bar(offsets, values, width, label=endpoint, color=ENDPOINT_COLOURS[endpoint],
                         edgecolor="black", linewidth=0.4)
                axes.errorbar(offsets, values, yerr=[lower, upper], fmt="none", ecolor="black",
                              capsize=2.5, linewidth=0.9)
            axes.set_xticks(range(len(arms)))
            axes.set_xticklabels([cell.replace(difficulty + "__", "").replace("_always_fail", "") for cell in arms],
                                 rotation=18, ha="right", fontsize=8)
            axes.set_ylabel(label)
            axes.axhline(0.0, color="black", linewidth=0.8)
            axes.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
            axes.set_title("%s - %s items" % (label, difficulty), fontsize=10)
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    axes_grid[0][-1].legend(handles, labels, fontsize=8, title="endpoint", title_fontsize=8,
                            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    figure.suptitle("EXPLORATORY - %s by cell and endpoint (%s)\nno QC exclusion, no confirmatory status"
                    % (_short(model_id), "item bootstrap 95% CI"), fontsize=11)
    figure.supxlabel("Descriptive only. Raw values, every endpoint present in the raw data.",
                     fontsize=7, color="#444444")
    figure.tight_layout()
    return _save(figure, out_dir, "FX_exploratory_%s_by_endpoint" % _short(model_id).replace(".", "_"))


# --------------------------------------------------------------------------
# FH -- preregistration-v3 holdout confirmation (built from phase2 summaries)
# --------------------------------------------------------------------------

CHECK = "✓"
MIDDOT = "·"
# Mirrors src.confirm.H10_VIOLATION_FRACTION: a style prompt violates H10 when it
# reproduces at least half the H1 effect (and its 95% CI upper bound is below 0).
H10_VIOLATION_FRACTION = 0.5
DISCOVERY_COLOUR, HOLDOUT_COLOUR = ORANGE, BLUE
# Small multiples: the four outcomes are on incomparable scales.
HOLDOUT_PANELS = (
    ("m1", "M1 hypotheses", "M1 margin difference (nats)"),
    ("distress", "Distress", "distress difference (judge points)"),
    ("m2", "M2", "M2 resample-disagreement difference (proportion)"),
    ("non_answer", "Non-answer rate", "non-answer rate difference (proportion)"),
)
CONTRAST_PREFIXES = ("M1, ", "M2, ", "M3, ", "Distress, ", "Non-answer rate, ")
ESTIMATE_WITH_CI = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*$")
HOLDOUT_NOTE = (
    "Discovery = Phase-1 exploratory paired contrast; holdout = frozen confirmatory analysis "
    "(item-clustered bootstrap 95% CI). Bold row label + " + CHECK
    + " = the preregistered decision rule was met on the holdout."
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def _float(value):
    """Numbers only: _read_csv leaves unparseable cells as strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_estimate(text):
    """Parse a ``value [lower, upper]`` cell; None when the cell is prose."""
    match = ESTIMATE_WITH_CI.match(str(text if text is not None else ""))
    if match is None:
        return None
    return tuple(float(group) for group in match.groups())


def _predicted_direction(prediction) -> int:
    """-1, +1, or 0 when no single direction is predicted (the H7 no-effect rule)."""
    text = " ".join(str(prediction if prediction is not None else "").split()).lower()
    if "includes 0" in text:
        return 0
    if text.startswith("<"):
        return -1
    if text.startswith(">"):
        return 1
    if "point <= 0" in text:
        return -1
    return 0


def _hypothesis_rows(summaries: Path):
    """Normalise the phase-2 hypotheses from hypotheses.csv, or confirm.json."""
    rows = []
    for row in _read_csv(summaries / "phase2" / "hypotheses.csv"):
        rows.append({
            "id": row.get("hypothesis_id"), "contrast": row.get("contrast"),
            "outcome": row.get("outcome"), "stratum": row.get("stratum"),
            "prediction": row.get("prediction"), "discovery": row.get("discovery"),
            "estimate": _float(row.get("estimate")), "lower": _float(row.get("ci95_lower")),
            "upper": _float(row.get("ci95_upper")), "n_items": _float(row.get("n_items")),
            "supported": _as_bool(row.get("supported")),
            "unavailable_reason": row.get("unavailable_reason"),
        })
    if rows:
        return rows
    confirm = _read(summaries / "phase2" / "confirm.json")
    for item in ((confirm or {}).get("result") or {}).get("hypotheses", []):
        result = item.get("result") or {}
        rows.append({
            "id": item.get("hypothesis_id"), "contrast": item.get("contrast"),
            "outcome": item.get("outcome"), "stratum": item.get("stratum"),
            "prediction": item.get("prediction"), "discovery": item.get("discovery"),
            "estimate": _float(result.get("estimate")), "lower": _float(result.get("ci95_lower")),
            "upper": _float(result.get("ci95_upper")), "n_items": _float(result.get("n_items")),
            "supported": _as_bool(item.get("supported")),
            "unavailable_reason": result.get("unavailable_reason"),
        })
    return rows


def _style_rows(confirm):
    """Normalise the machine-readable H10 style battery from confirm.json."""
    rows = []
    for item in ((confirm or {}).get("result") or {}).get("style", []) or []:
        result = item.get("result") or {}
        rows.append({
            "id": item.get("style_id"), "estimate": _float(result.get("estimate")),
            "lower": _float(result.get("ci95_lower")), "upper": _float(result.get("ci95_upper")),
            "n_items": _float(result.get("n_items")), "violates": _as_bool(item.get("violates")),
        })
    return rows


def _h1_holdout_estimate(rows):
    for row in rows:
        if row.get("id") == "H1":
            return row.get("estimate")
    return None


def _contrast_label(contrast) -> str:
    text = str(contrast if contrast is not None else "")
    for prefix in CONTRAST_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _row_label(row) -> str:
    head = "%s %s" % (row["id"], CHECK) if row["supported"] else str(row["id"])
    return "%s  %s  %s\n%s" % (head, MIDDOT, row["stratum"], _contrast_label(row["contrast"]))


def _row_note(row) -> str:
    n_items = row.get("n_items")
    lines = ["n = %d" % int(n_items) if n_items is not None else "n = n/a"]
    prediction = " ".join(str(row.get("prediction") or "").split())
    if not prediction:
        prediction = "prediction n/a"
    elif len(prediction) <= 12:
        prediction = "predicts %s" % prediction
    lines += textwrap.wrap(prediction, 22)
    return "\n".join(lines)


def _draw_hypothesis_panel(axes, rows, title, xlabel):
    values = [0.0]
    for row in rows:
        values += [value for value in (row["estimate"], row["lower"], row["upper"]) if value is not None]
        discovery = _parse_estimate(row["discovery"])
        if discovery is not None:
            values += list(discovery)
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    left, right = low - 0.10 * span, high + 0.30 * span
    axes.set_xlim(left, right)
    axes.set_ylim(len(rows) - 0.5, -0.5)
    for position, row in enumerate(rows):
        direction = _predicted_direction(row["prediction"])
        if direction:
            edge = right if direction > 0 else left
            axes.add_patch(Rectangle((0.0, position - 0.42), edge, 0.84, facecolor=GREEN,
                                     alpha=0.10, edgecolor="none", zorder=0))
        discovery = _parse_estimate(row["discovery"])
        if discovery is None:
            axes.annotate("discovery reported as cell means, not on this scale",
                          (0.012, position - 0.17), xycoords=("axes fraction", "data"),
                          fontsize=6.2, style="italic", color="#666666", ha="left", va="center")
        else:
            estimate, lower, upper = discovery
            axes.errorbar(estimate, position - 0.17,
                          xerr=[[estimate - lower], [upper - estimate]], fmt="o",
                          color=DISCOVERY_COLOUR, markerfacecolor="white",
                          markeredgecolor=DISCOVERY_COLOUR, markeredgewidth=1.1,
                          markersize=5.0, elinewidth=1.0, capsize=2.2, zorder=3)
        estimate, lower, upper = row["estimate"], row["lower"], row["upper"]
        if estimate is None:
            axes.annotate("holdout unavailable (%s)" % (row["unavailable_reason"] or "n/a"),
                          (0.012, position + 0.17), xycoords=("axes fraction", "data"),
                          fontsize=6.4, color=VERMILLION, ha="left", va="center")
        else:
            error = None if lower is None or upper is None else [[estimate - lower], [upper - estimate]]
            axes.errorbar(estimate, position + 0.17, xerr=error, fmt="o", color=HOLDOUT_COLOUR,
                          markersize=6.5, elinewidth=2.6, capsize=3.4, zorder=4)
        axes.annotate(_row_note(row), (0.992, position), xycoords=("axes fraction", "data"),
                      fontsize=6.6, color="#555555", ha="right", va="center", linespacing=1.3)
    axes.axvline(0.0, color="black", linewidth=0.9, zorder=2)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([_row_label(row) for row in rows], fontsize=7.0, linespacing=1.35)
    for label, row in zip(axes.get_yticklabels(), rows):
        label.set_color("#111111" if row["supported"] else "#555555")
        if row["supported"]:
            label.set_fontweight("bold")
    axes.tick_params(axis="y", length=0)
    axes.tick_params(axis="x", labelsize=7)
    axes.set_xlabel(xlabel, fontsize=7.5)
    axes.set_title(title, fontsize=9.5, loc="left")
    axes.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.55, zorder=1)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def figure_holdout_forest(rows, out_dir: Path):
    """FH -- every preregistration-v3 hypothesis: discovery beside locked holdout."""
    known = {outcome for outcome, _, _ in HOLDOUT_PANELS}
    panels = [(name, xlabel, [row for row in rows if row["outcome"] == outcome])
              for outcome, name, xlabel in HOLDOUT_PANELS]
    panels = [panel for panel in panels if panel[2]]
    leftover = [row for row in rows if row["outcome"] not in known]
    if leftover:
        panels.append(("other outcomes", "estimate", leftover))
    if not panels:
        return []
    ratios = [len(panel[2]) + 1.4 for panel in panels]
    figure, axes_column = plt.subplots(
        len(panels), 1, figsize=(12.0, 0.42 * sum(ratios) + 1.6),
        gridspec_kw={"height_ratios": ratios}, squeeze=False)
    for index, (axes, (name, xlabel, panel_rows)) in enumerate(zip(axes_column[:, 0], panels)):
        _draw_hypothesis_panel(axes, panel_rows, "%s  %s  %s" % (chr(ord("A") + index), MIDDOT, name), xlabel)
    handles = [
        Line2D([0], [0], color=DISCOVERY_COLOUR, marker="o", markerfacecolor="white",
               markeredgecolor=DISCOVERY_COLOUR, markeredgewidth=1.1, markersize=5.0,
               linewidth=1.0, label="discovery (exploratory, Phase 1)"),
        Line2D([0], [0], color=HOLDOUT_COLOUR, marker="o", markersize=6.5, linewidth=2.6,
               label="locked holdout (confirmatory, Phase 2)"),
        Patch(facecolor=GREEN, alpha=0.10, edgecolor="none", label="predicted direction"),
    ]
    figure.suptitle(
        "Preregistration v3 hypotheses: discovery (exploratory) vs locked holdout (confirmatory)",
        fontsize=12.5, y=0.992)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.963),
                  ncol=3, fontsize=8, frameon=False)
    figure.text(0.5, 0.008, HOLDOUT_NOTE, ha="center", fontsize=7, color="#444444")
    figure.tight_layout(rect=(0.0, 0.026, 1.0, 0.945))
    return _save(figure, out_dir, "FH_holdout_forest")


def figure_holdout_style_battery(styles, h1_estimate, model_id, out_dir: Path):
    """FH -- H10: no style prompt reproduces half the H1 effect on M1."""
    if not styles:
        return []
    threshold = None if h1_estimate is None else -H10_VIOLATION_FRACTION * abs(h1_estimate)
    values = [0.0] if threshold is None else [0.0, threshold]
    for style in styles:
        values += [value for value in (style["estimate"], style["lower"], style["upper"])
                   if value is not None]
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    left, right = low - 0.20 * span, high + 0.30 * span
    figure, axes = plt.subplots(figsize=(11.0, 0.58 * len(styles) + 2.8))
    axes.set_xlim(left, right)
    axes.set_ylim(len(styles) - 0.5, -0.5)
    if threshold is not None:
        axes.add_patch(Rectangle((left, -0.5), threshold - left, len(styles), facecolor=VERMILLION,
                                 alpha=0.09, edgecolor="none", zorder=0))
        axes.axvline(threshold, color=VERMILLION, linestyle="--", linewidth=1.5, zorder=2)
        axes.annotate("violates H10", (threshold, len(styles) - 0.54), xytext=(-7, 0),
                      textcoords="offset points", ha="right", va="bottom", fontsize=8,
                      fontweight="bold", color=VERMILLION)
    axes.axvline(0.0, color="black", linestyle="--", linewidth=1.1, zorder=2)
    for position, style in enumerate(styles):
        colour = VERMILLION if style["violates"] else HOLDOUT_COLOUR
        estimate, lower, upper = style["estimate"], style["lower"], style["upper"]
        if estimate is None:
            axes.annotate("unavailable", (0.012, position), xycoords=("axes fraction", "data"),
                          fontsize=7, color=VERMILLION, ha="left", va="center")
        else:
            error = None if lower is None or upper is None else [[estimate - lower], [upper - estimate]]
            axes.errorbar(estimate, position, xerr=error, fmt="o", color=colour, markersize=7.0,
                          elinewidth=2.6, capsize=4.0, zorder=4)
        note = "n = %s  %s  violates H10: %s" % (
            "n/a" if style["n_items"] is None else int(style["n_items"]), MIDDOT,
            "yes" if style["violates"] else "no")
        axes.annotate(note, (0.992, position), xycoords=("axes fraction", "data"), fontsize=7,
                      color="#555555", ha="right", va="center")
    axes.set_yticks(range(len(styles)))
    axes.set_yticklabels([str(style["id"]).replace("style__", "").replace("_", " ")
                          for style in styles], fontsize=8.5)
    axes.tick_params(axis="y", length=0)
    axes.set_xlabel("M1: style prompt − neutral style reference (nats), paired by item", fontsize=9)
    axes.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.55, zorder=1)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    handles = [
        Line2D([0], [0], color=HOLDOUT_COLOUR, marker="o", markersize=7.0, linewidth=2.6,
               label="holdout estimate, 95% CI"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.1, label="0 (no style effect)"),
        Line2D([0], [0], color=VERMILLION, linestyle="--", linewidth=1.5,
               label=("threshold: −0.5 × |H1 holdout| = %s nats"
                      % ("n/a" if threshold is None else ("%.2f" % threshold).replace("-", "−")))),
    ]
    figure.suptitle("H10 style battery: no style prompt reproduces half the H1 effect%s"
                    % ("" if not model_id else " (%s, holdout)" % _short(model_id)), fontsize=12)
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3,
                  fontsize=8, frameon=False)
    figure.text(0.5, 0.012,
                "A prompt violates H10 only if its point estimate is at or left of the dashed "
                "threshold AND its 95% CI upper bound is below 0. H10 holds when no prompt violates it.",
                ha="center", fontsize=7, color="#444444")
    figure.tight_layout(rect=(0.0, 0.155, 1.0, 0.935))
    return _save(figure, out_dir, "FH_holdout_style_battery")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate DGS figures from committed summaries.")
    parser.add_argument("--summaries", default="results/summaries", help="committed summary root")
    parser.add_argument("--out", default="results/figures", help="figure output directory")
    parser.add_argument("--exploratory-model", default=None,
                        help="model for the exploratory endpoint figure (default: the Phase-1 primary)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summaries, out_dir = Path(args.summaries), Path(args.out)
    written: list[Path] = []
    screen = _read(summaries / "phase0" / "screen.json")
    if screen is None:
        print("skipping F1: %s not found" % (summaries / "phase0" / "screen.json"), file=sys.stderr)
    else:
        written += figure_f1(screen, out_dir)
    gates = _read(summaries / "phase1" / "gates.json")
    if gates is None:
        print("skipping F2/F4: %s not found" % (summaries / "phase1" / "gates.json"), file=sys.stderr)
    else:
        written += figure_f2(gates, out_dir)
        written += figure_f4(gates, out_dir)
    summary = _read_csv(summaries / "phase1" / "exploratory" / "cell_endpoint_summary.csv")
    if not summary:
        print("skipping the exploratory figure: no exploratory cell summary found", file=sys.stderr)
    else:
        model_id = args.exploratory_model
        if model_id is None:
            model_id = gates["verdict"]["primary_model_id"] if gates else summary[0]["model_id"]
        written += figure_exploratory(summary, out_dir, model_id)
    hypotheses = _hypothesis_rows(summaries)
    if not hypotheses:
        print("skipping FH_holdout_forest: no %s (and no phase2/confirm.json) found"
              % (summaries / "phase2" / "hypotheses.csv"), file=sys.stderr)
    else:
        written += figure_holdout_forest(hypotheses, out_dir)
    confirm = _read(summaries / "phase2" / "confirm.json")
    styles = _style_rows(confirm)
    if not styles:
        print("skipping FH_holdout_style_battery: no H10 style battery in %s"
              % (summaries / "phase2" / "confirm.json"), file=sys.stderr)
    else:
        primary = ((confirm or {}).get("result") or {}).get("models", {}).get("primary")
        written += figure_holdout_style_battery(
            styles, _h1_holdout_estimate(hypotheses), primary, out_dir)
    for path in written:
        print("wrote %s" % path)
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
