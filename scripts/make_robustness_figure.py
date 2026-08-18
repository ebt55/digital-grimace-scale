"""Regenerate F12 from the committed robustness summary.

Usage:
    .venv\\Scripts\\python.exe scripts/make_robustness_figure.py \\
        --summary results/summaries/robustness/robustness.json --out results/figures

Like every other figure in this project, F12 never reads ``results/raw``: it is drawn entirely
from ``robustness.json``, so a fresh clone reproduces it. Colours are the Okabe-Ito
colourblind-safe palette and each panel is written as both PNG and SVG. A check that was not run
is drawn as an explicit "not run" panel rather than silently omitted.
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

BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"
SET_COLOURS = {"frozen": "black", "W1": BLUE, "W2": VERMILLION, "W3": GREEN}
S_CONTRASTS = ("H1", "H2a", "H2b", "TONE_ACC_POOLED", "H3a", "H3b")
NOTE = ("EXPLORATORY ROBUSTNESS (preregistration v7), greedy-only: M2 and the M2-valued H8 are not "
        "measured. Negative M1 = less certain. Not evidence of experience.")


def _by_id(rows):
    return {row["contrast_id"]: row for row in rows or ()}


def _blank(axes, message: str) -> None:
    axes.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#666666",
              transform=axes.transAxes)
    axes.set_xticks([]); axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def _forest(axes, entries, title: str, xlabel: str) -> None:
    """entries: list of (label, colour, row-or-None) drawn top to bottom."""
    positions = list(range(len(entries)))[::-1]
    for position, (label, colour, row) in zip(positions, entries):
        if row is None or row.get("estimate") is None:
            axes.text(0.02, position, "not estimable", fontsize=7, color="#666666",
                      va="center", transform=axes.get_yaxis_transform())
            continue
        estimate = row["estimate"]
        lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
        if lower is not None and upper is not None:
            axes.plot([lower, upper], [position, position], color=colour, linewidth=2.0,
                      solid_capstyle="butt", zorder=2)
            for bound in (lower, upper):
                axes.plot([bound, bound], [position - 0.14, position + 0.14], color=colour,
                          linewidth=1.2, zorder=2)
        axes.plot([estimate], [position], marker="o", markersize=5.5, color=colour,
                  markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    axes.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    axes.set_yticks(positions)
    axes.set_yticklabels([label for label, _, _ in entries], fontsize=8)
    axes.set_xlabel(xlabel, fontsize=8)
    axes.set_title(title, fontsize=9.5)
    axes.tick_params(axis="x", labelsize=7)
    axes.grid(axis="x", linewidth=0.3, alpha=0.4)


def _panel_w(axes_m1, axes_na, payload) -> None:
    check = payload["checks"]["W"]
    frozen = _by_id(payload["reference"]["estimates"])
    if not check.get("run"):
        _blank(axes_m1, "W not run\n%s" % check.get("reason", "")); _blank(axes_na, "W not run")
        return
    per_set = {name: _by_id(check["estimates"][name]) for name in check["sets"]}
    for axes, key, title, xlabel in (
            (axes_m1, "TONE_ACC_POOLED", "A. W - pooled accurate-arm tone effect (M1)", "hostile - neutral M1 (nats)"),
            (axes_na, "NONANSWER_ACC_POOLED", "B. W - pooled tone effect on non-answer rate", "hostile - neutral non-answer rate")):
        entries = [("frozen wording", SET_COLOURS["frozen"], frozen.get(key))]
        entries += [(name, SET_COLOURS.get(name, PURPLE), per_set[name].get(key)) for name in check["sets"]]
        _forest(axes, entries, title, xlabel)


def _panel_pair(axes, payload, check_key: str, title: str, left_label: str) -> None:
    check = payload["checks"][check_key]
    if not check.get("run"):
        _blank(axes, "%s not run\n%s" % (check_key, check.get("reason", "")))
        return
    left = _by_id(check["estimates"])
    right = _by_id(payload["reference"]["estimates"])
    labels, positions = [], []
    for index, key in enumerate(S_CONTRASTS):
        position = len(S_CONTRASTS) - index
        labels.append(key); positions.append(position)
        for row, colour, offset in ((left.get(key), VERMILLION, 0.16), (right.get(key), BLUE, -0.16)):
            if row is None or row.get("estimate") is None:
                continue
            lower, upper = row.get("ci95_lower"), row.get("ci95_upper")
            if lower is not None and upper is not None:
                axes.plot([lower, upper], [position + offset] * 2, color=colour, linewidth=1.8,
                          solid_capstyle="butt", zorder=2)
            axes.plot([row["estimate"]], [position + offset], marker="o", markersize=5,
                      color=colour, markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    axes.axvline(0.0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    axes.set_yticks(positions); axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("estimate (M1 nats; H8 is M2 and not measured)", fontsize=8)
    if check.get("estimable") is False:
        title += "\nM1 NOT ESTIMABLE: parseable-answer rate %.2f, below the 50%% feasibility floor" % (
            check.get("neutral_parseable_rate") or 0.0)
        axes.text(0.5, 0.5, "no %s estimate:\nfeasibility floor" % left_label, ha="center", va="center",
                  fontsize=9, color=VERMILLION, transform=axes.transAxes, alpha=0.75)
    axes.set_title(title, fontsize=9.5)
    axes.tick_params(axis="x", labelsize=7)
    axes.grid(axis="x", linewidth=0.3, alpha=0.4)
    axes.legend(handles=[Line2D([], [], color=VERMILLION, marker="o", linewidth=1.8, label=left_label),
                         Line2D([], [], color=BLUE, marker="o", linewidth=1.8,
                                label="20 locked items, %s" % payload["reference"]["model_id"].split("/")[-1])],
                fontsize=7, loc="best")


def figure_f12(payload, out_dir: Path):
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    _panel_w(axes[0][0], axes[0][1], payload)
    _panel_pair(axes[1][0], payload, "S", "C. S - fresh %s-item ARC bank vs the 20 locked items"
                % (payload["checks"]["S"].get("n_items", "?")),
                "fresh bank (%s items)" % (payload["checks"]["S"].get("n_items", "?")))
    _panel_pair(axes[1][1], payload, "G", "D. G - %s vs %s"
                % ((payload["checks"]["G"].get("model_id") or "27B").split("/")[-1],
                   payload["reference"]["model_id"].split("/")[-1]), "27B")
    verdicts = "  ".join("%s %s" % (item["id"], item["verdict"]) for item in payload["verdicts"])
    figure.suptitle("F12 - robustness of the M1 / non-answer signature to wording, items and scale\n%s"
                    % verdicts, fontsize=10)
    figure.supxlabel(NOTE, fontsize=7, color="#444444")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / ("F12_robustness.%s" % suffix)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default="results/summaries/robustness/robustness.json")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)
    summary = Path(args.summary) if Path(args.summary).is_absolute() else ROOT / args.summary
    if not summary.exists():
        print("robustness summary not found: %s" % summary, file=sys.stderr)
        return 2
    payload = json.loads(summary.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    for path in figure_f12(payload, out_dir):
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
