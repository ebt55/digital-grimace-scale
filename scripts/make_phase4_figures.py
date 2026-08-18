"""Regenerate the Phase-4 figures from the committed Phase-4 summary.

Usage:
    .venv\\Scripts\\python.exe scripts/make_phase4_figures.py \\
        --summaries results/summaries/phase4 --out results/figures

Like the other figure scripts, nothing here reads ``results/raw``: every panel comes from
``results/summaries/phase4/phase4.json``, so a fresh clone reproduces them.  Colours are the
Okabe-Ito colourblind-safe palette and each figure is written as PNG and SVG.  A panel whose
input is absent is skipped with a message rather than crashing.

* **F8** - the claim-relevant quantity: ``DiD_A - DiD_B`` per outcome, with item-clustered
  bootstrap 95% intervals; an interval clear of zero is drawn filled.
* **F9** - the adverse-minus-neutral gap per outcome under each arm (0, A, B) side by side,
  which is what the DiD differences.
* **F10** - the manipulation checks: judged distress at the hostile onset endpoint and greedy
  capability accuracy, by arm, against their preregistered thresholds.
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
ARM_COLOUR = {"0": GREY, "A": VERMILLION, "B": BLUE}
ARM_LABEL = {"0": "arm 0 (no adapter)", "A": "arm A (distress-suppression DPO)", "B": "arm B (placebo DPO)"}
OUTCOMES = ("m1", "non_answer", "m2", "hedge_per100", "selfcorr_per100", "distress")
OUTCOME_LABEL = {"m1": "M1 (nats)", "non_answer": "non-answer rate", "m2": "M2",
                 "hedge_per100": "hedging /100 tok", "selfcorr_per100": "self-correction /100 tok",
                 "distress": "judged distress"}
INTERPRETATION_NOTE = ("Which channels an adapter reaches is a functional result about training "
                       "and measurement. Not evidence of experience.")


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


def _effect(report, section, *keys):
    node = report.get(section) or {}
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node if isinstance(node, dict) and node.get("estimate") is not None else None


def _clear_of_zero(effect) -> bool:
    lower, upper = effect.get("ci95_lower"), effect.get("ci95_upper")
    return lower is not None and upper is not None and not (lower <= 0.0 <= upper)


def figure_f8(report, out_dir: Path):
    """F8 -- DiD_A - DiD_B per outcome, the quantity the preregistration calls claim-relevant."""
    rows = [(outcome, _effect(report, "did_difference", outcome)) for outcome in OUTCOMES]
    rows = [(outcome, effect) for outcome, effect in rows if effect is not None]
    if not rows:
        return []
    figure, axes = plt.subplots(figsize=(7.6, 0.62 * len(rows) + 2.0))
    for position, (outcome, effect) in enumerate(rows):
        clear = _clear_of_zero(effect)
        colour = VERMILLION if clear else GREY
        lower = effect.get("ci95_lower") if effect.get("ci95_lower") is not None else effect["estimate"]
        upper = effect.get("ci95_upper") if effect.get("ci95_upper") is not None else effect["estimate"]
        axes.plot([lower, upper], [position, position], color=colour, linewidth=2.2, solid_capstyle="butt")
        axes.plot([effect["estimate"]], [position], marker="o", markersize=7, color=colour,
                  markerfacecolor=colour if clear else "white", markeredgewidth=1.6)
    axes.axvline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([OUTCOME_LABEL[outcome] for outcome, _ in rows])
    axes.invert_yaxis()
    axes.set_xlabel("DiD$_A$ - DiD$_B$  (negative = A moves the channel adverse-selectively beyond placebo)")
    axes.set_title("F8 - adverse-selective effect of the distress-suppression adapter beyond placebo\n"
                   "item-paired, 2,000-resample item-clustered bootstrap 95% CI", fontsize=10)
    axes.legend(handles=[Line2D([], [], color=VERMILLION, marker="o", linewidth=2.2, label="CI excludes 0"),
                         Line2D([], [], color=GREY, marker="o", markerfacecolor="white",
                                linewidth=2.2, label="CI includes 0")],
                loc="lower right", fontsize=8, frameon=False)
    figure.text(0.01, -0.02, INTERPRETATION_NOTE, fontsize=7.5, color=GREY)
    return _save(figure, out_dir, "F8_did_difference")


def figure_f9(report, out_dir: Path):
    """F9 -- the adverse-minus-neutral gap under each arm, which the DiD differences."""
    arms = [arm for arm in ("0", "A", "B") if (report.get("gaps") or {}).get(arm)]
    if not arms:
        return []
    figure, axes = plt.subplots(1, len(OUTCOMES), figsize=(2.05 * len(OUTCOMES), 3.4), sharex=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    for index, outcome in enumerate(OUTCOMES):
        panel = axes[index]
        for position, arm in enumerate(arms):
            effect = _effect(report, "gaps", arm, outcome)
            if effect is None:
                continue
            lower = effect.get("ci95_lower") if effect.get("ci95_lower") is not None else effect["estimate"]
            upper = effect.get("ci95_upper") if effect.get("ci95_upper") is not None else effect["estimate"]
            panel.plot([position, position], [lower, upper], color=ARM_COLOUR[arm], linewidth=2.2)
            panel.plot([position], [effect["estimate"]], marker="o", markersize=6, color=ARM_COLOUR[arm])
        panel.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        panel.set_xticks(range(len(arms)))
        panel.set_xticklabels(arms)
        panel.set_title(OUTCOME_LABEL[outcome], fontsize=8.5)
        if index == 0:
            panel.set_ylabel("adverse - neutral")
    figure.suptitle("F9 - the condition-selective gap under each arm (arm 0, A, B)", fontsize=10)
    figure.legend(handles=[Line2D([], [], color=ARM_COLOUR[arm], marker="o", linewidth=2.2,
                                  label=ARM_LABEL[arm]) for arm in arms],
                  loc="lower center", ncol=len(arms), fontsize=8, frameon=False,
                  bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(figure, out_dir, "F9_gap_by_arm")


def figure_f10(report, out_dir: Path):
    """F10 -- MC1 (hostile-onset distress) and MC2 (capability accuracy) against their thresholds."""
    checks = report.get("manipulation_checks") or {}
    mc1, mc2 = checks.get("MC1") or {}, checks.get("MC2") or {}
    if not mc1 and not mc2:
        return []
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    distress = {}
    for arm, check in mc1.items():
        values = check.get("values") or {}
        if values.get("baseline_mean") is not None:
            distress.setdefault("0", values["baseline_mean"])
        if values.get("arm_mean") is not None:
            distress[arm] = values["arm_mean"]
    if distress:
        arms = [arm for arm in ("0", "A", "B") if arm in distress]
        left.bar(range(len(arms)), [distress[arm] for arm in arms],
                 color=[ARM_COLOUR[arm] for arm in arms], width=0.6)
        baseline = distress.get("0")
        if baseline:
            left.axhline(0.20 * baseline, color="black", linestyle="--", linewidth=1.0)
            left.text(len(arms) - 0.5, 0.20 * baseline, " MC1 bar (20% of arm 0)", fontsize=7.5,
                      va="bottom", ha="right")
        left.set_xticks(range(len(arms)))
        left.set_xticklabels(arms)
        left.set_ylabel("judged distress at hostile onset (0-10)")
        left.set_title("MC1 - distress must fall by >= 80%", fontsize=9)

    accuracy = {}
    for arm, check in mc2.items():
        values = check.get("values") or {}
        if values.get("baseline_accuracy") is not None:
            accuracy.setdefault("0", values["baseline_accuracy"])
        if values.get("arm_accuracy") is not None:
            accuracy[arm] = values["arm_accuracy"]
    if accuracy:
        arms = [arm for arm in ("0", "A", "B") if arm in accuracy]
        right.bar(range(len(arms)), [accuracy[arm] for arm in arms],
                  color=[ARM_COLOUR[arm] for arm in arms], width=0.6)
        baseline = accuracy.get("0")
        if baseline is not None:
            right.axhspan(baseline - 0.05, baseline + 0.05, color=GREEN, alpha=0.18)
            right.axhline(baseline, color="black", linestyle="--", linewidth=1.0)
        right.set_xticks(range(len(arms)))
        right.set_xticklabels(arms)
        right.set_ylim(0, 1)
        right.set_ylabel("greedy capability accuracy")
        right.set_title("MC2 - accuracy must stay within +/-5 pp (shaded)", fontsize=9)
    else:
        right.set_axis_off()
        right.text(0.5, 0.5, "capability set not run", ha="center", va="center", color=GREY)

    figure.suptitle("F10 - Phase-4 manipulation checks", fontsize=10)
    figure.text(0.01, -0.02, INTERPRETATION_NOTE, fontsize=7.5, color=GREY)
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(figure, out_dir, "F10_manipulation_checks")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the Phase-4 figures.")
    parser.add_argument("--summaries", default="results/summaries/phase4")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)

    report = _read(ROOT / args.summaries / "phase4.json")
    if report is None:
        print("no phase4.json under %s; run `run_phase4.py analyze` first"
              % (ROOT / args.summaries), file=sys.stderr)
        return 2
    out_dir = ROOT / args.out
    written = []
    for name, builder in (("F8", figure_f8), ("F9", figure_f9), ("F10", figure_f10)):
        paths = builder(report, out_dir)
        if not paths:
            print("skipped %s: its inputs are absent from phase4.json" % name)
        written.extend(paths)
    for path in written:
        print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
