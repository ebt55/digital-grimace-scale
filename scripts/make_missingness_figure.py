"""F13 -- the primary model's M1 contrasts under four treatments of the missing values.

Usage:
    .venv\\Scripts\\python.exe scripts/make_missingness_figure.py \\
        --summary results/summaries/missingness/m1_missingness.json \\
        --out results/figures

Reads only the committed sensitivity summary, never the raw data, so the figure
regenerates from a fresh clone.  House style follows `scripts/make_figures.py`:
Okabe-Ito colours, PNG and SVG at 200 dpi, an interpretation note under the axes.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Okabe-Ito, as in scripts/make_figures.py.
BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"
MIDDOT = "·"
SPLITS = ("discovery", "holdout")
# (treatment, label, colour, marker, vertical offset inside the contrast band)
TREATMENT_STYLE = (
    ("available_case", "available-case (published)", BLUE, "o", -0.21),
    ("zero_imputation", "zero-imputation (non-answer = 0 nats)", ORANGE, "s", -0.07),
    ("manski_lower", "worst-case bound: most negative", PURPLE, "<", 0.07),
    ("manski_upper", "worst-case bound: most positive", PURPLE, ">", 0.21),
)
NOTE = (
    "Item-paired M1 differences, 2,000-resample item-clustered bootstrap 95% CI. Bounds impute every missing "
    "value in one cell at the extreme of the model's neutral-accurate measured M1 and the other cell at the "
    "opposite extreme; the shaded band is the resulting identification region. delta = the constant margin every "
    "missing treated-cell trial would need for the CI to include 0. Sensitivity analysis, not a confirmatory result."
)


def _save(figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def _treatment(outcome, name):
    for item in outcome["treatments"]:
        if item["treatment"] == name:
            return item
    return None


def _values(outcomes) -> list[float]:
    out = [0.0]
    for outcome in outcomes:
        for item in outcome["treatments"]:
            result = item["result"]
            out += [value for value in (result["estimate"], result["ci95_lower"], result["ci95_upper"])
                    if value is not None]
    return out


def _label(outcome) -> str:
    return "%s\n%s" % (outcome["contrast_id"], outcome["stratum"])


def _note(outcome) -> str:
    counts = outcome["counts"]
    tipping = outcome["tipping"]
    if tipping["delta"] is None:
        delta = "no missing treated value" if tipping["reason"] == "no_missing_treated_values" else "delta n/a"
    elif tipping["reason"]:
        delta = "CI already includes 0"
    else:
        delta = "delta = %.1f nats" % tipping["delta"]
    return "n = %d of %d  %s  missing %dT / %dR  %s  %s" % (
        counts["n_available"], counts["n_pairable"], MIDDOT,
        counts["n_treated_missing"], counts["n_reference_missing"], MIDDOT, delta)


def _draw_panel(axes, outcomes, title, left, right):
    axes.set_xlim(left, right)
    axes.set_ylim(len(outcomes) - 0.5, -0.5)
    for position, outcome in enumerate(outcomes):
        lower, upper = _treatment(outcome, "manski_lower"), _treatment(outcome, "manski_upper")
        if lower is not None and upper is not None:
            low = lower["result"]["estimate"]
            high = upper["result"]["estimate"]
            if low is not None and high is not None:
                axes.barh(position, high - low, left=low, height=0.62, color=PURPLE, alpha=0.10,
                          edgecolor="none", zorder=0)
        for name, _text, colour, marker, offset in TREATMENT_STYLE:
            item = _treatment(outcome, name)
            if item is None:
                continue
            result = item["result"]
            estimate = result["estimate"]
            if estimate is None:
                continue
            error = None
            if result["ci95_lower"] is not None and result["ci95_upper"] is not None:
                error = [[estimate - result["ci95_lower"]], [result["ci95_upper"] - estimate]]
            filled = name in ("available_case", "zero_imputation")
            axes.errorbar(
                estimate, position + offset, xerr=error, fmt=marker, color=colour,
                markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                markeredgewidth=1.1, markersize=6.4 if filled else 6.8,
                elinewidth=2.2 if filled else 1.2, capsize=3.0 if filled else 2.0,
                zorder=4 if filled else 3)
        axes.annotate(_note(outcome), (0.992, position), xycoords=("axes fraction", "data"),
                      fontsize=6.6, color="#555555", ha="right", va="center")
        if position:
            axes.axhline(position - 0.5, color="#DDDDDD", linewidth=0.6, zorder=0)
    axes.axvline(0.0, color="black", linewidth=0.9, zorder=2)
    axes.set_yticks(range(len(outcomes)))
    axes.set_yticklabels([_label(outcome) for outcome in outcomes], fontsize=7.6)
    axes.tick_params(axis="y", length=0)
    axes.set_title(title, fontsize=10, loc="left")
    axes.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.55, zorder=1)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def figure_missingness(payload, out_dir: Path, *, role: str = "primary"):
    """F13 -- four treatments of the missing M1 values, discovery beside holdout."""
    result = payload.get("result") or {}
    outcomes = [item for item in result.get("outcomes", ()) if item.get("role") == role]
    if not outcomes:
        return []
    model_id = outcomes[0]["model_id"]
    panels = [(split, [item for item in outcomes if item["split"] == split]) for split in SPLITS]
    panels = [panel for panel in panels if panel[1]]
    if not panels:
        return []
    values = _values(outcomes)
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    left, right = low - 0.06 * span, high + 0.36 * span
    rows = max(len(panel[1]) for panel in panels)
    figure, axes_column = plt.subplots(
        len(panels), 1, figsize=(11.0, 0.72 * rows * len(panels) + 2.6), squeeze=False)
    for index, (axes, (split, panel_rows)) in enumerate(zip(axes_column[:, 0], panels)):
        title = "%s  %s  %s" % (
            chr(ord("A") + index), MIDDOT,
            "discovery split (exploratory, Phase 1)" if split == "discovery"
            else "locked holdout (confirmatory, Phase 2)")
        _draw_panel(axes, panel_rows, title, left, right)
        axes.set_xlabel("M1 difference (nats), item-paired", fontsize=9)
    handles = [
        Line2D([0], [0], color=colour, marker=marker, markersize=6.4,
               markerfacecolor=colour if name in ("available_case", "zero_imputation") else "white",
               markeredgecolor=colour, linewidth=2.0, label=text)
        for name, text, colour, marker, _offset in TREATMENT_STYLE
    ]
    handles.append(Patch(facecolor=PURPLE, alpha=0.10, edgecolor="none",
                         label="identification region (between the two bounds)"))
    figure.suptitle("F13 %s M1 contrasts under four treatments of the missing values %s %s"
                    % (MIDDOT, MIDDOT, model_id), fontsize=12.0, y=0.995)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.966), ncol=3,
                  fontsize=7.6, frameon=False)
    figure.text(0.5, 0.004, "\n".join(_wrap(NOTE, 132)), ha="center", fontsize=6.4, color="#444444")
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.925))
    return _save(figure, out_dir, "F13_m1_missingness_bounds")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draw F13 from the committed missingness summary.")
    parser.add_argument("--summary", default="results/summaries/missingness/m1_missingness.json")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--role", default="primary", help="which model arm to draw")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.summary)
    if not path.exists():
        print("missingness summary not found: %s; F13 skipped" % path, file=sys.stderr)
        return 0
    written = figure_missingness(json.loads(path.read_text(encoding="utf-8")), Path(args.out),
                                 role=args.role)
    if not written:
        print("no %s outcomes in %s; F13 skipped" % (args.role, path), file=sys.stderr)
        return 0
    for item in written:
        print("wrote %s" % item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
