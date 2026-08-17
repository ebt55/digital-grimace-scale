"""Regenerate F1, F2 and F4 purely from the committed summaries.

Usage:
    .venv\\Scripts\\python.exe scripts/make_figures.py \\
        --summaries results/summaries --out results/figures

Figures never read ``results/raw``: everything comes from ``screen.json`` and
``gates.json``, so a fresh clone of the repository reproduces every panel.
Colours are the Okabe-Ito colourblind-safe palette; each figure is written as
both PNG and SVG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate DGS figures from committed summaries.")
    parser.add_argument("--summaries", default="results/summaries", help="committed summary root")
    parser.add_argument("--out", default="results/figures", help="figure output directory")
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
    for path in written:
        print("wrote %s" % path)
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
