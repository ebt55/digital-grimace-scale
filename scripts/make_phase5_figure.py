"""Regenerate F11 (Phase-5 base-model denominator) from the committed Phase-5 summary.

Usage:
    .venv\\Scripts\\python.exe scripts/make_phase5_figure.py \\
        --summaries results/summaries --out results/figures

Like `scripts/make_figures.py`, this never reads `results/raw`: everything comes from
`results/summaries/phase5/phase5.json`, so a fresh clone reproduces the panels.  Colours are
the Okabe-Ito colourblind-safe palette and each figure is written as PNG and SVG.  An absent
summary is skipped with a message rather than crashing.
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
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
HYPOTHESES = ("H1", "H2a", "H2b")
HYPOTHESIS_LABELS = {
    "H1": "H1  false failure\n(mal - acc, neutral, easy)",
    "H2a": "H2a  hostile tone\n(hostile - neutral, easy)",
    "H2b": "H2b  hostile tone\n(hostile - neutral, hard)",
}
COLUMNS = (("base_plain", "base + plain", BLUE),
           ("it_plain", "it + plain", VERMILLION),
           ("it_chat_template_published", "it + chat (published)", GREEN))
INTERPRETATION_NOTE = (
    "Negative = the answer margin collapses under the manipulation. Discovery split only, "
    "exploratory; provenance is not evidence of experience."
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


def estimates(payload, hypothesis_id: str):
    """[(label, colour, estimate, lower, upper)] for one hypothesis, in column order."""
    out = []
    for key, label, colour in COLUMNS:
        column = payload["columns"].get(key) or {}
        entry = (column.get("hypotheses", column) or {}).get(hypothesis_id)
        if not entry or entry.get("estimate") is None:
            out.append((label, colour, None, None, None))
            continue
        out.append((label, colour, float(entry["estimate"]),
                    entry.get("ci95_lower"), entry.get("ci95_upper")))
    return out


def measured_rates(payload):
    """[(cell, base valid rate, it+plain valid rate)] at the measured endpoint."""
    base = {item["cell_id"]: item for item in (payload["columns"]["base_plain"].get("cells") or ())
            if item["turn_label"] == "measured"}
    control = {item["cell_id"]: item
               for item in (payload["columns"]["it_plain"].get("cells") or ())
               if item["turn_label"] == "measured"}
    return [(cell_id, base[cell_id].get("valid_answer_rate"),
             (control.get(cell_id) or {}).get("valid_answer_rate"))
            for cell_id in sorted(set(base) | set(control))]


def figure_f11(payload, out_dir: Path):
    """F11 -- three rendering columns per contrast, plus the feasibility panel."""
    figure, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [2.25, 1.0]})
    effects, feasibility_axes = axes

    # When v6's feasibility gate fires, the base column's bars rest on one or two paired items
    # and are NOT estimates; they are drawn hatched and said so, so the panel cannot be read as
    # "the base model shows a small effect".
    base_estimable = ((payload["columns"]["base_plain"].get("feasibility") or {})
                      .get("m1_estimable", True))
    gap, width = 1.0, 0.24
    ticks, labels = [], []
    for index, hypothesis_id in enumerate(HYPOTHESES):
        centre = index * gap
        ticks.append(centre)
        labels.append(HYPOTHESIS_LABELS[hypothesis_id])
        for offset, (label, colour, estimate, lower, upper) in enumerate(
                estimates(payload, hypothesis_id)):
            position = centre + (offset - 1) * width
            hatched = offset == 0 and not base_estimable
            if estimate is None:
                effects.text(position, 0.0, "n/a", ha="center", va="bottom", fontsize=7,
                             rotation=90, color=GREY)
                continue
            effects.bar(position, estimate, width * 0.9,
                        color="none" if hatched else colour, edgecolor=colour if hatched else "black",
                        hatch="///" if hatched else None, linewidth=0.9 if hatched else 0.4,
                        label=(label + " - NOT ESTIMABLE" if hatched else label)
                        if index == 0 else None)
            if lower is not None and upper is not None:
                effects.errorbar(position, estimate, yerr=[[estimate - lower], [upper - estimate]],
                                 fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    effects.axhline(0.0, color="black", linewidth=0.8)
    effects.set_xticks(ticks)
    effects.set_xticklabels(labels, fontsize=8)
    effects.set_ylabel("paired M1 difference (nats), 95% item-clustered CI")
    effects.set_title("F11 - does the M1 signature exist before instruction tuning?\n"
                      "discovery split, rendering held constant across the first two columns",
                      fontsize=10)
    effects.legend(fontsize=8, loc="lower left")
    if not base_estimable:
        effects.text(0.985, 0.03,
                     "hatched: base model fails the v6 feasibility gate - one paired item, no CI",
                     transform=effects.transAxes, fontsize=7, color=GREY, ha="right", va="bottom")

    rows = measured_rates(payload)
    if rows:
        positions = range(len(rows))
        height = 0.38
        feasibility_axes.barh([p + height / 2 for p in positions],
                              [0.0 if row[1] is None else row[1] for row in rows], height,
                              color=BLUE, edgecolor="black", linewidth=0.4, label="base + plain")
        feasibility_axes.barh([p - height / 2 for p in positions],
                              [0.0 if row[2] is None else row[2] for row in rows], height,
                              color=VERMILLION, edgecolor="black", linewidth=0.4,
                              label="it + plain")
        feasibility_axes.set_yticks(list(positions))
        feasibility_axes.set_yticklabels([row[0].replace("malfunctioning_always_fail", "mal")
                                          for row in rows], fontsize=7)
        feasibility_axes.set_xlim(0.0, 1.05)
        feasibility_axes.axvline(0.5, color=GREY, linestyle="--", linewidth=0.9)
        feasibility_axes.text(0.5, len(rows) - 0.4, " v6 gate", fontsize=7, color=GREY,
                              va="top")
        feasibility_axes.set_xlabel("parseable `Answer: X` rate (measured, greedy)")
        feasibility_axes.set_title("feasibility per cell", fontsize=10)
        feasibility_axes.legend(fontsize=8, loc="lower right")
    else:
        feasibility_axes.axis("off")

    figure.supxlabel(INTERPRETATION_NOTE, fontsize=7, color="#444444")
    figure.tight_layout()
    return _save(figure, out_dir, "F11_base_denominator")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summaries", default="results/summaries")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)

    summaries, out_dir = Path(args.summaries), Path(args.out)
    payload = _read(summaries / "phase5" / "phase5.json")
    if payload is None:
        print("phase5 summary not found under %s; nothing to draw" % summaries, file=sys.stderr)
        return 1
    for path in figure_f11(payload, out_dir):
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
