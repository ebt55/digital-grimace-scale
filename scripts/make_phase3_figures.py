"""Regenerate the Phase-3 figures from the committed Phase-3 summaries.

Usage:
    .venv\\Scripts\\python.exe scripts/make_phase3_figures.py \\
        --summaries results/summaries/phase3 --out results/figures

Like ``scripts/make_figures.py`` these panels never read ``results/raw`` or
``results/jspace``: everything comes from ``localization.json`` and ``steering.json``, so a
fresh clone reproduces them.  Colours are the Okabe-Ito colourblind-safe palette and each
figure is written as both PNG and SVG.  A figure whose summary is absent is skipped with a
message rather than crashing.

* **F5** - discovery leave-one-task-out AUC by layer for tone and validity, with the
  once-only holdout evaluation marked at the discovery-chosen layer ``L*``.
* **F6** - paired change in M1 against steering dose, one line per direction, with
  item-bootstrap 95% intervals; degenerate doses are marked.
* **F7** - EXPLORATORY: the same dose-response for the tone direction recomputed at other
  layers, with the confirmatory layer drawn alongside. It decides no preregistered verdict.
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

BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#999999")
DIRECTION_STYLE = {
    "tone": (VERMILLION, "-", "o", 2.2, 1.0),
    "unrelated_style": (BLUE, "--", "s", 1.6, 0.95),
    "random1": (GREY, ":", "^", 1.0, 0.7),
    "random2": (GREY, ":", "v", 1.0, 0.7),
    "random3": (GREY, ":", "<", 1.0, 0.7),
    "random4": (GREY, ":", ">", 1.0, 0.7),
    "random5": (GREY, ":", "d", 1.0, 0.7),
}
INTERPRETATION_NOTE = (
    "A condition-linked internal variable with causal influence on the output signature. "
    "Not evidence of experience."
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


def figure_f5(localization, out_dir: Path):
    """F5 -- discovery LOO AUC by layer, with the holdout point at L*."""
    tone = [(row["layer"], row["auc"]) for row in localization["discovery_loo_auc"]["tone"]
            if row["auc"] is not None]
    validity = [(row["layer"], row["auc"]) for row in localization["discovery_loo_auc"]["validity"]
                if row["auc"] is not None]
    if not tone:
        return []
    chosen = localization["chosen_layer"]
    holdout = localization["holdout_auc_at_chosen_layer"]
    figure, axes = plt.subplots(figsize=(8.6, 4.4))
    axes.plot([layer for layer, _ in tone], [auc for _, auc in tone], color=VERMILLION,
              linewidth=2.0, marker="o", markersize=3.0, label="tone (hostile vs neutral), discovery LOO")
    if validity:
        axes.plot([layer for layer, _ in validity], [auc for _, auc in validity], color=BLUE,
                  linewidth=1.6, marker="s", markersize=2.8, linestyle="--",
                  label="validity (malfunctioning vs accurate), discovery LOO")
    axes.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
    axes.axhline(0.80, color=GREEN, linewidth=0.8, linestyle="--")
    axes.axvline(chosen, color=PURPLE, linewidth=1.0, alpha=0.7)
    handles = []
    for label, value, colour, marker in (("tone", holdout.get("tone"), VERMILLION, "*"),
                                         ("validity", holdout.get("validity"), BLUE, "P")):
        if value is None:
            continue
        axes.plot([chosen], [value], marker=marker, markersize=13 if marker == "*" else 9,
                  color=colour, markeredgecolor="black", markeredgewidth=0.6, linestyle="none")
        handles.append(Line2D([], [], marker=marker, color=colour, linestyle="none",
                              markeredgecolor="black", markeredgewidth=0.6,
                              markersize=11 if marker == "*" else 8,
                              label="%s, holdout at L* (once)" % label))
    axes.set_xlabel("residual-stream layer (0 = embeddings)")
    axes.set_ylabel("AUC")
    axes.set_ylim(0.0, 1.02)
    axes.set_title("F5  Pre-response linear decodability by layer\n"
                   "L* = %d chosen on discovery tone AUC alone; 0.80 = J1 discovery bar" % chosen)
    existing, labels = axes.get_legend_handles_labels()
    axes.legend(existing + handles, labels + [item.get_label() for item in handles],
                fontsize=8, loc="lower right", framealpha=0.95)
    axes.grid(axis="y", alpha=0.25)
    figure.text(0.01, -0.03, INTERPRETATION_NOTE, fontsize=7.5, color="#444444")
    return _save(figure, out_dir, "F5_phase3_auc_by_layer")


def figure_f6(steering, out_dir: Path):
    """F6 -- paired change in M1 against dose, one line per direction."""
    rows = steering.get("doses") or []
    by_direction: dict[str, list] = {}
    for row in rows:
        if row["direction_id"] == "baseline":
            continue
        by_direction.setdefault(row["direction_id"], []).append(row)
    if not by_direction:
        return []
    figure, axes = plt.subplots(figsize=(7.6, 4.6))
    annotated: set[float] = set()
    # Random controls first so the tone line and its interval are drawn on top of them.
    for direction_id in sorted(by_direction, key=lambda name: (not name.startswith("random"),
                                                               name != "unrelated_style", name)):
        colour, style, marker, width, alpha = DIRECTION_STYLE.get(
            direction_id, (ORANGE, "-.", "x", 1.0, 0.8))
        ordered = sorted(by_direction[direction_id], key=lambda row: row["alpha"])
        doses = [0.0] + [row["alpha"] for row in ordered]
        values = [0.0] + [row["m1_delta"]["estimate"] if row["m1_delta"]["estimate"] is not None
                          else float("nan") for row in ordered]
        axes.plot(doses, values, color=colour, linestyle=style, marker=marker, linewidth=width,
                  markersize=4.5, alpha=alpha,
                  label=direction_id if not direction_id.startswith("random") else None)
        for row in ordered:
            lower, upper = row["m1_delta"]["ci95_lower"], row["m1_delta"]["ci95_upper"]
            estimate = row["m1_delta"]["estimate"]
            if lower is None or upper is None or estimate is None:
                continue
            axes.plot([row["alpha"], row["alpha"]], [lower, upper], color=colour,
                      linewidth=1.4 if direction_id == "tone" else 0.8, alpha=alpha)
            if row["degenerate"] and row["alpha"] not in annotated:
                annotated.add(row["alpha"])
                axes.annotate("degenerate dose", (row["alpha"], estimate),
                              textcoords="offset points", xytext=(-4, 8), fontsize=7,
                              ha="right", color=colour)
    axes.axhline(0.0, color="black", linewidth=0.8)
    norms = steering.get("direction_norms") or {}
    axes.set_xlabel("dose alpha  (added at L* = %d: alpha x d, ||d|| = %s; controls matched)"
                    % (steering["layer"],
                       "%.1f" % norms["tone_direction_norm"]
                       if norms.get("tone_direction_norm") is not None else "n/a"))
    axes.set_ylabel("M1(alpha) - M1(0), nats  (paired by item)")
    axes.set_title("F6  Direction-specificity steering on neutral holdout items\n"
                   "error bars: 2,000-resample item-clustered 95% CI")
    handles, labels = axes.get_legend_handles_labels()
    handles.append(Line2D([], [], color=GREY, linestyle=":", marker="^", markersize=4,
                          label="5 random matched-norm directions"))
    labels.append("5 random matched-norm directions")
    axes.legend(handles, labels, fontsize=8, loc="best", framealpha=0.95)
    axes.grid(axis="y", alpha=0.25)
    figure.text(0.01, -0.03, INTERPRETATION_NOTE, fontsize=7.5, color="#444444")
    return _save(figure, out_dir, "F6_phase3_steering_dose_response")


LAYER_COLOURS = (VERMILLION, GREEN, PURPLE, ORANGE, SKY)


def figure_f7(sweep, out_dir: Path):
    """F7 -- EXPLORATORY: paired change in M1 against dose, one line per steered layer."""
    by_layer: dict[int, list] = {}
    controls: dict[int, list] = {}
    for row in sweep["doses"]:
        name = row["direction_id"]
        if name.startswith("tone_L"):
            by_layer.setdefault(int(name.split("_L")[1]), []).append(row)
        elif name.startswith("random_L"):
            controls.setdefault(int(name.split("_L")[1].split("_")[0]), []).append(row)
    confirm_layer = int(sweep["confirmatory_layer"])
    by_layer.setdefault(confirm_layer, []).extend(
        row for row in sweep.get("confirmatory_tone_doses") or [] if row["alpha"] > 0)
    if not by_layer:
        return []
    figure, axes = plt.subplots(figsize=(7.8, 4.8))
    broken = []  # (alpha, layer): the dose has no parseable answer at all, so no M1 exists
    for index, layer in enumerate(sorted(by_layer)):
        colour = LAYER_COLOURS[index % len(LAYER_COLOURS)]
        ordered = sorted(by_layer[layer], key=lambda row: row["alpha"])
        confirmatory = layer == confirm_layer
        drawn = [row for row in ordered if row["m1_delta"]["estimate"] is not None]
        axes.plot([0.0] + [row["alpha"] for row in drawn],
                  [0.0] + [row["m1_delta"]["estimate"] for row in drawn],
                  color=colour, linestyle="--" if confirmatory else "-",
                  marker="s" if confirmatory else "o", linewidth=2.0, markersize=5,
                  label="layer %d%s" % (layer, " (confirmatory L*)" if confirmatory else ""))
        for row in ordered:
            if row["m1_delta"]["estimate"] is None:
                broken.append((row["alpha"], layer, colour, "tone"))
                continue
            lower, upper = row["m1_delta"]["ci95_lower"], row["m1_delta"]["ci95_upper"]
            if lower is not None and upper is not None:
                axes.plot([row["alpha"]] * 2, [lower, upper], color=colour, linewidth=1.5)
            if row["degenerate"]:
                axes.annotate("degenerate", (row["alpha"], row["m1_delta"]["estimate"]),
                              textcoords="offset points", xytext=(-4, 8), fontsize=7,
                              ha="right", color=colour)
        for row in controls.get(layer, ()):
            if row["m1_delta"]["estimate"] is None:
                broken.append((row["alpha"], layer, GREY, "control"))
                continue
            axes.plot([row["alpha"]], [row["m1_delta"]["estimate"]], marker="x", markersize=7,
                      color=GREY, linestyle="none")
            lower, upper = row["m1_delta"]["ci95_lower"], row["m1_delta"]["ci95_upper"]
            if lower is not None and upper is not None:
                axes.plot([row["alpha"]] * 2, [lower, upper], color=GREY, linewidth=0.8, alpha=0.7)
    axes.axhline(0.0, color="black", linewidth=0.8)
    if broken:
        # A dose where every item is a non-answer has no M1 to plot at all; say so on the axis
        # rather than dropping it silently.  One label per (dose, layer), listing which arms.
        bottom = axes.get_ylim()[0]
        grouped: dict[tuple[float, int], tuple[str, set]] = {}
        for alpha, layer, colour, kind in broken:
            entry = grouped.setdefault((alpha, layer), (colour, set()))
            entry[1].add(kind)
        for offset, key in enumerate(sorted(grouped)):
            (alpha, layer), (colour, kinds) = key, grouped[key]
            axes.annotate("L%d %s at alpha %g:\n100%% non-answer, no M1"
                          % (layer, " + ".join(sorted(kinds)), alpha),
                          (alpha, bottom), textcoords="offset points",
                          xytext=(-8, 12 + 26 * offset), fontsize=7, ha="right", color=colour)
    axes.set_xlabel("dose alpha  (alpha x d, d recomputed at each layer; controls matched to ||d||)")
    axes.set_ylabel("M1(alpha) - M1(0), nats  (paired by item)")
    axes.set_title("F7  EXPLORATORY tone steering by layer\n"
                   "changes no preregistered verdict; error bars: item-clustered 95% CI")
    handles, labels = axes.get_legend_handles_labels()
    handles.append(Line2D([], [], color=GREY, marker="x", linestyle="none",
                          label="random matched-norm controls"))
    labels.append("random matched-norm controls")
    axes.legend(handles, labels, fontsize=8, loc="best", framealpha=0.95)
    axes.grid(axis="y", alpha=0.25)
    figure.text(0.01, -0.03, INTERPRETATION_NOTE, fontsize=7.5, color="#444444")
    return _save(figure, out_dir, "F7_phase3_layer_sweep_exploratory")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="make_phase3_figures", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summaries", default="results/summaries/phase3")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)
    summaries = (ROOT / args.summaries) if not Path(args.summaries).is_absolute() else Path(args.summaries)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    written = []
    localization = _read(summaries / "localization.json")
    if localization is None:
        print("make_phase3_figures: no localization.json under %s; F5 skipped" % summaries)
    else:
        written.extend(figure_f5(localization, out_dir))
    steering = _read(summaries / "steering.json")
    if steering is None:
        print("make_phase3_figures: no steering.json under %s; F6 skipped" % summaries)
    else:
        written.extend(figure_f6(steering, out_dir))
    sweep = _read(summaries / "steering_layer_sweep_exploratory.json")
    if sweep is None:
        print("make_phase3_figures: no exploratory layer sweep under %s; F7 skipped" % summaries)
    else:
        written.extend(figure_f7(sweep, out_dir))
    for path in written:
        print("make_phase3_figures: wrote %s" % path)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
