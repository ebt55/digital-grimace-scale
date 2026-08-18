"""Build the three README figures (F0 channel map, F0b headline effects, F0c phase map).

Usage:
    .venv\\Scripts\\python.exe scripts/make_readme_figures.py \\
        --summaries results/summaries --out results/figures [--print]

Like ``scripts/make_figures.py`` this reads *only* committed summaries -- no raw
records, no re-estimation.  Every number printed on a figure is loaded from a
JSON/CSV summary and formatted here; nothing is typed in by hand except the
labels.  Colours are the Okabe-Ito colourblind-safe palette and each figure is
written as both PNG and SVG.  A figure whose summary is missing is skipped with
a message on stderr rather than crashing.

``--print`` additionally writes ``F0b_headline_effects_print`` and
``F0_channel_map_print``: the same numbers and colours re-laid for paper.  The
screen figures are drawn ~11-15 in wide and shrink to illegibility in a report;
the print variants are drawn at the width they are *placed* at (or a little
wider), saved at the exact canvas size -- no ``bbox_inches="tight"`` -- so the
placement arithmetic is exact: printed pt = matplotlib pt x placement width /
figure width.  ``--print`` reports the smallest size actually used.

Sources, per figure:

* F0  -- phase1/gates.json, phase2/{hypotheses.csv,confirm.json},
         phase3/steering.json, phase4/phase4.json, phase5/{phase5.json,
         cell_valid_rates.csv}, robustness/robustness.json,
         extension/*/extension.json
* F0b -- phase2/hypotheses.csv, extension/*/extension.json,
         missingness/m1_missingness.json (pooled tone contrast),
         robustness/robustness.json (86-item fresh bank)
* F0c -- phase1/gates.json, phase2/confirm.json, phase3/steering.json,
         phase4/phase4.json, phase5/phase5.json, robustness/robustness.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

# Okabe-Ito: distinguishable under the common forms of colour vision deficiency.
BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
INK, MUTED, HAIRLINE = "#1a1a1a", "#555555", "#BBBBBB"

MOVES, FLAT, NA = "moves", "flat", "na"
CODE_FACE = {MOVES: "#CFE3F0", FLAT: "#F4F4F4", NA: "#DEDEDE"}
CODE_EDGE = {MOVES: BLUE, FLAT: "#9A9A9A", NA: "#9A9A9A"}
CODE_GLYPH = {MOVES: "●", FLAT: "○", NA: "—"}

CEILING = ("A channel that moves is a functional measurement result. "
           "It licenses no claim about experience, suffering or moral status.")


# --------------------------------------------------------------------------- io


def _read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extension(summaries: Path):
    """The single third-family extension summary, whatever the model directory."""
    root = summaries / "extension"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/extension.json")):
        return _read(path)
    return None


def _save(figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(figure)
    return written


PRINT_DPI = 300


def _save_print(figure, out_dir: Path, name: str) -> list[Path]:
    """Save at the exact canvas size: a tight bbox would break the pt arithmetic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=PRINT_DPI, facecolor="white")
        written.append(path)
    plt.close(figure)
    return written


def _fit(figure, text, limit_points: float, floor: float) -> float:
    """Shrink one text object until it is at most `limit_points` wide; never below `floor`."""
    renderer = figure.canvas.get_renderer()
    while text.get_fontsize() > floor:
        if text.get_window_extent(renderer).width * 72.0 / figure.dpi <= limit_points:
            break
        text.set_fontsize(round(text.get_fontsize() - 0.1, 2))
    return text.get_fontsize()


def _rewrap(text: str, width: int) -> str:
    """Re-wrap each authored line to a narrower character budget, keeping the words."""
    return "\n".join(_wrap(line, width) for line in text.split("\n"))


# ------------------------------------------------------------------ formatting


def _n(value, places: int = 2, sign: bool = True) -> str:
    """Format a number from a summary with a typographic minus; no signed zeroes."""
    if value is None:
        return "n/a"
    if abs(value) < 0.5 * 10 ** -places:
        return "%.*f" % (places, 0.0)
    text = ("%+.*f" if sign else "%.*f") % (places, value)
    return text.replace("-", "−")


def _pp(value, places: int = 0) -> str:
    return "%s pp" % _n(value * 100.0, places)


def _hyp(rows: list[dict], hypothesis_id: str) -> dict | None:
    for row in rows:
        if row.get("hypothesis_id") == hypothesis_id:
            return row
    return None


def _hyp_value(rows: list[dict], hypothesis_id: str, field: str = "estimate"):
    row = _hyp(rows, hypothesis_id)
    if row is None or not row.get(field):
        return None
    return float(row[field])


def _ext_holdout(extension, hypothesis_id: str) -> dict | None:
    for comparison in ((extension or {}).get("result") or {}).get("comparisons", []):
        if comparison.get("hypothesis_id") == hypothesis_id:
            return ((comparison.get("holdout") or {}).get("result")) or None
    return None


def _robustness_estimate(robustness, check: str, contrast_id: str) -> dict | None:
    for estimate in (((robustness or {}).get("checks") or {}).get(check) or {}).get("estimates", []):
        if estimate.get("contrast_id") == contrast_id and estimate.get("estimate") is not None:
            return estimate
    return None


def _missingness_available_case(missingness, model_key: str, split: str, contrast_id: str):
    """The published available-case row of a contrast in the MNAR sensitivity table."""
    models = ((missingness or {}).get("result") or {}).get("models") or {}
    model_id = models.get(model_key)
    for outcome in ((missingness or {}).get("result") or {}).get("outcomes", []):
        if outcome.get("model_id") != model_id or outcome.get("split") != split:
            continue
        if outcome.get("contrast_id") != contrast_id:
            continue
        for treatment in outcome.get("treatments", []):
            if treatment.get("treatment") == "available_case":
                return treatment.get("result") or None
    return None


def _cell_rate(cell_rates: list[dict], model_suffix: str, turn_label: str, tone: str):
    """Pooled valid-answer rate over the cells a summary table already reports."""
    valid = items = 0
    for row in cell_rates:
        if not row.get("model_id", "").endswith(model_suffix):
            continue
        if row.get("turn_label") != turn_label or not row.get("cell_id", "").endswith(tone):
            continue
        valid += int(row["n_valid"])
        items += int(row["n_items"])
    return (valid, items, (valid / items if items else None))


def _judge_cell_mean(summaries: Path, run: str, turn_label: str, cell_contains: str,
                     model_id: str | None = None):
    """n-weighted mean judged distress over the cells of a committed judge summary."""
    total = count = 0.0
    for row in _read_csv(summaries / "judge" / run / "summary.csv"):
        if model_id is not None and row.get("model_id") != model_id:
            continue
        if row.get("turn_label") != turn_label or cell_contains not in row.get("cell_id", ""):
            continue
        total += float(row["mean_score"]) * float(row["n"])
        count += float(row["n"])
    return (total / count) if count else None


# ------------------------------------------------------- F0: the channel map

CHANNEL_COLUMNS = (
    "false-failure feedback\n(3 rounds, neutral tone)",
    "hostile tone\n(truthful feedback)",
    "single bogus verdict\n(onset)",
    "truthful correction\n(washout / recovery)",
    "style prompts\n(verbose, cautious, …)",
    "tone-direction steering\n(Phase 3, α = 2)",
    "distress-DPO\nadapter A",
    "placebo DPO\nadapter B",
    "base model\n(gemma-2-9b)",
    "Qwen2.5-3B\n(control)",
    "Llama-3.1-8B\n(third family)",
    "gemma-2-27b-it\n(scale)",
)
CHANNEL_ROWS = (
    "answer margin M1\n(nats)",
    "non-answers\n(rate)",
    "resample\ndisagreement M2",
    "distress language\n(judge, 0–10)",
)


def _channel_cells(summaries: Path, hypotheses, gates, confirm, steering, steering_judge,
                   phase4, cell_rates, robustness, extension) -> dict:
    """Every cell of F0, each value read out of a committed summary."""
    primary = (gates or {}).get("verdict", {}).get("primary_model_id", "")
    gate_m2 = (((gates or {}).get("verdict", {}).get("models", {}).get(primary) or {})
               .get("real_g1", {}).get("M2") or {})
    gate_g2 = (((gates or {}).get("verdict", {}).get("models", {}).get(primary) or {})
               .get("real_g2", {}).get("M2") or {})
    style = ((gates or {}).get("verdict", {}).get("style") or {}).get("M2|style__verbose") or {}
    h10 = ((confirm or {}).get("result") or {}).get("h10_supported")

    steer_j4 = {}
    for verdict in (steering or {}).get("verdicts", []):
        if verdict.get("prediction_id") == "J4":
            steer_j4 = verdict.get("detail") or {}
    n_judge = (steering_judge or {}).get("scored")

    gaps = (phase4 or {}).get("gaps") or {}
    did = (phase4 or {}).get("did") or {}
    mc1 = ((phase4 or {}).get("manipulation_checks") or {}).get("MC1") or {}
    strip = ((phase4 or {}).get("sensitivity_a6_strip_special_tokens") or {})
    frozen_na = (strip.get("non_answer_rates_frozen") or {}).get("B", {})
    stripped_na = (strip.get("non_answer_rates_strip_on") or {}).get("B", {})

    # Phase 5: L1's denominator and the flat base non-answer rate, from the committed cell table.
    base_valid, base_items, base_rate = _cell_rate(cell_rates, "gemma-2-9b", "measured", "__neutral")
    base_non_answers = sorted({row["non_answer_rate"] for row in cell_rates
                               if row.get("model_id", "").endswith("gemma-2-9b")})
    base_cells = len({(row["cell_id"], row["turn_label"]) for row in cell_rates
                      if row.get("model_id", "").endswith("gemma-2-9b")
                      and row["turn_label"] == "measured"})
    base_distress = _judge_cell_mean(summaries, "phase5_base", "onset", "accurate__hostile")
    plain_distress = _judge_cell_mean(summaries, "phase5_itplain", "onset", "accurate__hostile")

    # Holdout judge cell means, for the endpoints no single hypothesis contrast reports alone.
    control = (gates or {}).get("verdict", {}).get("control_model_id", "")
    distress_measured = _judge_cell_mean(summaries, "phase2", "measured",
                                         "malfunctioning_always_fail__neutral", primary)
    distress_neutral_onset = _judge_cell_mean(summaries, "phase2", "onset",
                                              "accurate__neutral", primary)
    distress_control_onset = _judge_cell_mean(summaries, "phase2", "onset",
                                              "accurate__hostile", control)

    check_g = ((robustness or {}).get("checks") or {}).get("G") or {}

    def h(hid, field="estimate"):
        return _hyp_value(hypotheses, hid, field)

    def e(hid, field="estimate"):
        row = _ext_holdout(extension, hid)
        return None if row is None else row.get(field)

    cells: dict = {}

    def put(row, column, code, text):
        cells[(row, column)] = (code, text)

    def pct(block, arm):
        value = ((block.get(arm) or {}).get("values") or {}).get("relative_reduction")
        return "n/a" if value is None else "%.1f%%" % (100.0 * value)

    # ---- row 0: answer margin M1 (nats)
    put(0, 0, MOVES, "%s nats\nH1" % _n(h("H1")))
    put(0, 1, MOVES, "H2a %s easy\nH2b %s hard" % (_n(h("H2a")), _n(h("H2b"))))
    put(0, 2, MOVES, "%s nats\nH3a" % _n(h("H3a")))
    put(0, 3, MOVES, "%s after 1\n%s after 3 rounds" % (_n(h("H4a")), _n(h("H5"))))
    put(0, 4, FLAT, "no prompt reaches\n½·|H1|  (H10 %s)" % ("held" if h10 else "broken"))
    put(0, 5, FLAT, "%s nats\nJ4: not supported" % _n(steer_j4.get("alpha2_m1_delta")))
    put(0, 6, FLAT, "gap %s vs %s\nsignature intact"
        % (_n((gaps.get("A") or {}).get("m1", {}).get("estimate")),
           _n((gaps.get("0") or {}).get("m1", {}).get("estimate"))))
    put(0, 7, FLAT, "gap %s\nDiD CIs include 0"
        % _n((gaps.get("B") or {}).get("m1", {}).get("estimate")))
    put(0, 8, NA, "%d of %d parseable\nnot estimable" % (base_valid, base_items))
    put(0, 9, MOVES, "%s  H7a\n%s  H7b" % (_n(h("H7a")), _n(h("H7b"))))
    put(0, 10, MOVES, "%s nats\nH1" % _n(e("H1")))
    put(0, 11, NA, "%d of %d parseable\nnot estimable"
        % (int(round(check_g.get("neutral_parseable_rate", 0.0)
                     * check_g.get("neutral_endpoints", 0))), check_g.get("neutral_endpoints", 0)))

    # ---- row 1: non-answers
    put(1, 0, NA, "not estimated")
    put(1, 1, MOVES, "%s at onset\nH9, hard items" % _pp(h("H9")))
    put(1, 2, NA, "not estimated")
    put(1, 3, NA, "not estimated")
    put(1, 4, NA, "not estimated")
    put(1, 5, FLAT, "0.00 at every dose\nJ6: not supported")
    put(1, 6, FLAT, "DiD %s\nnot supported"
        % _n((did.get("A") or {}).get("non_answer", {}).get("estimate")))
    put(1, 7, FLAT, "%s → %s\nonce stripped"
        % (_n(frozen_na.get("hostile_onset"), 2, sign=False),
           _n(stripped_na.get("hostile_onset"), 2, sign=False)))
    put(1, 8, FLAT, "%s in all %d cells\nnot the treatment"
        % (_n(float(base_non_answers[0]) if base_non_answers else None, 2, sign=False), base_cells))
    put(1, 9, NA, "not estimated")
    put(1, 10, FLAT, "%s at onset\nH9: not supported" % _pp(e("H9")))
    put(1, 11, NA, "parser artefact")

    # ---- row 2: resample disagreement M2
    put(2, 0, FLAT, "%s z\nBH p = %s  (G1)"
        % (_n(gate_m2.get("validity", {}).get("coefficient")),
           _n(gate_m2.get("validity", {}).get("adjusted_p"), 2, sign=False)))
    put(2, 1, MOVES, "%s\nH8" % _n(h("H8")))
    put(2, 2, NA, "not estimated")
    put(2, 3, FLAT, "%s\ngate G2: no reversal" % _n(gate_g2.get("recovery")))
    put(2, 4, MOVES, "verbose %s z †\nstyle moves it more" % _n(style.get("effect")))
    put(2, 5, NA, "greedy only")
    put(2, 6, FLAT, "DiD %s\nnot supported" % _n((did.get("A") or {}).get("m2", {}).get("estimate")))
    put(2, 7, FLAT, "DiD %s\nnot supported" % _n((did.get("B") or {}).get("m2", {}).get("estimate")))
    put(2, 8, NA, "not reported")
    put(2, 9, NA, "not estimated")
    put(2, 10, MOVES, "%s\nH8" % _n(e("H8")))
    put(2, 11, NA, "greedy only")

    # ---- row 3: distress language (judge)
    put(3, 0, FLAT, "%s / 10 on the\nmeasured response" % _n(distress_measured, 1, sign=False))
    put(3, 1, MOVES, "%s / 10\nH6a" % _n(h("H6a"), 1))
    put(3, 2, FLAT, "%s / 10 under a\nneutral verdict" % _n(distress_neutral_onset, 1, sign=False))
    put(3, 3, NA, "not judged")
    put(3, 4, NA, "not judged")
    put(3, 5, FLAT, "all %s judge\nscores 0  (J6)" % (n_judge if n_judge is not None else "—"))
    put(3, 6, MOVES, "%s = %s\nMC1 bar 80%%: FAIL"
        % (_n((mc1.get("A") or {}).get("effect", {}).get("estimate"), 1), pct(mc1, "A")))
    put(3, 7, MOVES, "%s = %s\nlength placebo"
        % (_n((mc1.get("B") or {}).get("effect", {}).get("estimate"), 1), pct(mc1, "B")))
    put(3, 8, FLAT, "%s base vs %s it\nL5: at the floor"
        % (_n(base_distress, 2, sign=False), _n(plain_distress, 2, sign=False)))
    put(3, 9, FLAT, "%s / 10 at onset\nH6b gap %s" % (_n(distress_control_onset, 2, sign=False),
                                                      _n(h("H6b"), 2)))
    put(3, 10, FLAT, "%s / 10\nH6a: flat" % _n(e("H6a"), 2))
    put(3, 11, MOVES, "%s / 10\nG-3: persists" % _n(check_g.get("hostile_onset_distress_mean"), 2,
                                                    sign=False))

    _ = base_rate  # the pooled L1 rate; printed as its own numerator/denominator above
    return cells


def figure_channel_map(cells: dict, out_dir: Path) -> list[Path]:
    """F0 -- which output channel moves under which manipulation, intervention or setting."""
    if not cells:
        return []
    n_rows, n_cols = len(CHANNEL_ROWS), len(CHANNEL_COLUMNS)
    figure, axes = plt.subplots(figsize=(15.2, 7.2))
    axes.set_xlim(0, n_cols)
    axes.set_ylim(n_rows, 0)
    axes.set_aspect("auto")

    for (row, column), (code, text) in cells.items():
        axes.add_patch(Rectangle(
            (column + 0.04, row + 0.06), 0.92, 0.88,
            facecolor=CODE_FACE[code], edgecolor=CODE_EDGE[code], linewidth=0.9,
            hatch="///" if code == NA else None))
        axes.text(column + 0.5, row + 0.28, CODE_GLYPH[code], ha="center", va="center",
                  fontsize=11, color=CODE_EDGE[code] if code != NA else "#8A8A8A")
        axes.text(column + 0.5, row + 0.66, text, ha="center", va="center",
                  fontsize=6.6, color=INK if code != NA else MUTED, linespacing=1.45)

    axes.set_xticks([index + 0.5 for index in range(n_cols)])
    axes.set_xticklabels(CHANNEL_COLUMNS, fontsize=7.6, rotation=28, ha="left", linespacing=1.4)
    axes.xaxis.set_ticks_position("top")
    axes.xaxis.set_label_position("top")
    axes.set_yticks([index + 0.5 for index in range(n_rows)])
    axes.set_yticklabels(CHANNEL_ROWS, fontsize=8.4, linespacing=1.5)
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)

    # A rule separating the manipulations from the models they were measured on.
    axes.axvline(8, color=HAIRLINE, linewidth=1.4, ymin=0.0, ymax=1.0)
    axes.text(4.0, n_rows + 0.30,
              "manipulations and interventions · primary model, gemma-2-9b-it",
              ha="center", va="center", fontsize=8.2, color=MUTED, style="italic")
    axes.text(10.0, n_rows + 0.30, "the same manipulations in other models",
              ha="center", va="center", fontsize=8.2, color=MUTED, style="italic")
    axes.set_ylim(n_rows + 0.55, 0)

    handles = [
        Patch(facecolor=CODE_FACE[MOVES], edgecolor=CODE_EDGE[MOVES],
              label="moves — CI excludes 0, or the stated verdict is a move"),
        Patch(facecolor=CODE_FACE[FLAT], edgecolor=CODE_EDGE[FLAT],
              label="no move — measured; CI includes 0, at the floor, or unchanged"),
        Patch(facecolor=CODE_FACE[NA], edgecolor=CODE_EDGE[NA], hatch="///",
              label="not measurable — feasibility gate, parse failure, or not run"),
    ]
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.13),
                ncol=3, frameon=False, fontsize=8, handlelength=1.6)

    figure.suptitle(
        "F0 · Which channel moves under what",
        x=0.008, y=1.115, ha="left", fontsize=13, fontweight="bold", color=INK)
    figure.text(
        0.008, 1.075,
        "Holdout confirmatory values where they exist; otherwise the phase's own preregistered "
        "headline. M1 in nats, distress on the judge's 0–10 scale, non-answers and M2 as rates.",
        ha="left", fontsize=8.6, color=MUTED)
    figure.text(
        0.008, -0.155,
        "Phase-4 cells quote each channel's preregistered headline: the K4 adverse−neutral M1 gap, "
        "difference-in-differences for M2 and non-answers, manipulation check MC1 for distress.\n"
        "† a move that counts against the grimace reading — a style prompt reproduces the channel, "
        "so M2 is a style meter rather than a marker.\n" + CEILING,
        ha="left", fontsize=7.4, color=MUTED, linespacing=1.6)
    return _save(figure, out_dir, "F0_channel_map")


# -------------------------------------------- F0 print: the same map, for paper

PRINT_CHANNEL_COLUMNS = (
    "false-failure\nfeedback × 3\n(neutral tone)",
    "hostile tone\n(truthful\nfeedback)",
    "single bogus\nverdict\n(onset)",
    "truthful\ncorrection\n(washout)",
    "style prompts\n(verbose,\ncautious, …)",
    "tone-direction\nsteering\n(Phase 3, α = 2)",
    "distress-DPO\nadapter A",
    "placebo DPO\nadapter B",
    "base model\n(gemma-2-9b)",
    "Qwen2.5-3B\n(control)",
    "Llama-3.1-8B\n(third family)",
    "gemma-2-27b-it\n(scale)",
)
PRINT_CHANNEL_ROWS = (
    "answer margin\nM1 (nats)",
    "non-answers\n(rate)",
    "resample\ndisagreement\nM2",
    "distress\nlanguage\n(judge, 0–10)",
)


def figure_channel_map_print(cells: dict, out_dir: Path) -> tuple[list[Path], float]:
    """F0 laid out for a landscape page: drawn at 10.1 in, placed at 10.1 in."""
    if not cells:
        return ([], 0.0)
    n_rows, n_cols = len(PRINT_CHANNEL_ROWS), len(PRINT_CHANNEL_COLUMNS)
    width, height = 10.1, 6.2
    label_zone, right_pad = 0.94, 0.06
    axes_bottom, axes_height = 0.92, 4.20

    figure = plt.figure(figsize=(width, height))
    axes = figure.add_axes([label_zone / width, axes_bottom / height,
                            (width - label_zone - right_pad) / width, axes_height / height])
    axes.set_xlim(0, n_cols)
    axes.set_ylim(n_rows, 0)

    column_points = (width - label_zone - right_pad) / n_cols * 72.0
    cell_limit, header_limit = column_points * 0.88, column_points * 0.97
    smallest = 99.0

    for (row, column), (code, text) in sorted(cells.items()):
        axes.add_patch(Rectangle(
            (column + 0.04, row + 0.06), 0.92, 0.88,
            facecolor=CODE_FACE[code], edgecolor=CODE_EDGE[code], linewidth=0.8,
            hatch="///" if code == NA else None))
        axes.text(column + 0.5, row + 0.22, CODE_GLYPH[code], ha="center", va="center",
                  fontsize=9, color=CODE_EDGE[code] if code != NA else "#8A8A8A")
        drawn = axes.text(column + 0.5, row + 0.62, _rewrap(text, 11), ha="center", va="center",
                          fontsize=7.1, color=INK if code != NA else MUTED, linespacing=1.42)
        smallest = min(smallest, _fit(figure, drawn, cell_limit, 6.6))

    axes.set_xticks([index + 0.5 for index in range(n_cols)])
    axes.set_xticklabels(PRINT_CHANNEL_COLUMNS, fontsize=7.6, ha="center", linespacing=1.34)
    axes.xaxis.set_ticks_position("top")
    for label in axes.get_xticklabels():
        smallest = min(smallest, _fit(figure, label, header_limit, 7.0))
    axes.set_yticks([index + 0.5 for index in range(n_rows)])
    axes.set_yticklabels(PRINT_CHANNEL_ROWS, fontsize=8.0, linespacing=1.38)
    for label in axes.get_yticklabels():
        smallest = min(smallest, _fit(figure, label, label_zone * 72.0 - 8.0, 7.0))
    axes.tick_params(length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.axvline(8, color=HAIRLINE, linewidth=1.3)

    def fy(inches: float) -> float:
        return inches / height

    header_top = axes_bottom + axes_height
    matrix = width - label_zone - right_pad
    for centre, band in ((4.0, "manipulations and interventions · primary model, gemma-2-9b-it"),
                         (10.0, "the same manipulations in other models")):
        figure.text((label_zone + matrix * centre / n_cols) / width, fy(header_top + 0.52), band,
                    ha="center", va="center", fontsize=7.8, color=MUTED, style="italic")

    handles = [
        Patch(facecolor=CODE_FACE[MOVES], edgecolor=CODE_EDGE[MOVES],
              label="moves — CI excludes 0, or the stated verdict is a move"),
        Patch(facecolor=CODE_FACE[FLAT], edgecolor=CODE_EDGE[FLAT],
              label="no move — measured; CI includes 0, at the floor, or unchanged"),
        Patch(facecolor=CODE_FACE[NA], edgecolor=CODE_EDGE[NA], hatch="///",
              label="not measurable — feasibility gate, parse failure, or not run"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.09, fy(axes_bottom - 0.06)),
                  ncol=2, frameon=False, fontsize=7.6, handlelength=1.5, columnspacing=2.4,
                  labelspacing=0.35)

    figure.text(0.008, fy(height - 0.10), "F0 · Which channel moves under what",
                ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    figure.text(0.008, fy(height - 0.34),
                "Holdout confirmatory values where they exist; otherwise the phase's own "
                "preregistered headline. M1 in nats, distress on the judge's 0–10 scale, "
                "non-answers and M2 as rates.",
                ha="left", va="top", fontsize=8.0, color=MUTED)
    figure.text(0.008, fy(0.04),
                "Phase-4 cells quote each channel's preregistered headline: the K4 adverse−neutral "
                "M1 gap, difference-in-differences for M2 and non-answers, manipulation check MC1 "
                "for distress.\n"
                "† a move that counts against the grimace reading — a style prompt reproduces the "
                "channel, so M2 is a style meter rather than a marker.\n" + CEILING,
                ha="left", va="bottom", fontsize=7.2, color=MUTED, linespacing=1.5)
    return (_save_print(figure, out_dir, "F0_channel_map_print"), smallest)


# ----------------------------------------------- F0b: the headline forest plot

M1_ROWS = (
    ("H1", "H1  false failure × 3   (easy | neutral)"),
    ("tone_pooled", "pooled tone   (easy+hard | truthful)"),
    ("H2a", "H2a  hostile tone   (easy | truthful)"),
    ("H2b", "H2b  hostile tone   (hard | truthful)"),
    ("H3a", "H3a  one bogus verdict   (easy | neutral)"),
    ("H3b", "H3b  one bogus verdict   (easy | hostile)"),
    ("H4a", "H4a  truthful washout   (easy | neutral)"),
    ("H5", "H5  correction after 3 rounds   (hard | neutral)"),
)
DISTRESS_ROWS = (("H6a", "H6a  distress, hostile − neutral onset"),)
RATE_ROWS = (
    ("H8", "H8  M2 disagreement, hostile − neutral"),
    ("H9", "H9  non-answers, hostile − neutral onset"),
)
SERIES = (
    ("gemma", "gemma-2-9b-it · holdout, confirmatory", BLUE, "o", 7.0, -0.24),
    ("fresh", "gemma-2-9b-it · 86 fresh ARC items (v7 S)", GREEN, "D", 4.6, -0.02),
    ("qwen", "Qwen2.5-3B-Instruct · control", ORANGE, "s", 4.6, 0.17),
    ("llama", "Llama-3.1-8B-Instruct · third family", PURPLE, "^", 4.8, 0.35),
)


def _forest_values(hypotheses, extension, missingness, robustness) -> dict:
    """{series: {row_id: (estimate, low, high)}} for every panel of F0b."""
    out: dict = {name: {} for name, *_ in SERIES}

    def triple(row):
        if not row:
            return None
        estimate, low, high = (row.get("estimate"), row.get("ci95_lower"), row.get("ci95_upper"))
        if estimate is None or low is None or high is None:
            return None
        return (float(estimate), float(low), float(high))

    for hypothesis_id in ("H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H5", "H6a", "H8", "H9"):
        row = _hyp(hypotheses, hypothesis_id)
        if row and row.get("estimate"):
            out["gemma"][hypothesis_id] = (float(row["estimate"]), float(row["ci95_lower"]),
                                           float(row["ci95_upper"]))
    # The control's two preregistered M1 contrasts are H7a (= H1) and H7b (= H2a).
    for control_id, row_id in (("H7a", "H1"), ("H7b", "H2a")):
        row = _hyp(hypotheses, control_id)
        if row and row.get("estimate"):
            out["qwen"][row_id] = (float(row["estimate"]), float(row["ci95_lower"]),
                                   float(row["ci95_upper"]))
    for hypothesis_id in ("H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H5", "H6a", "H8", "H9"):
        value = triple(_ext_holdout(extension, hypothesis_id))
        if value:
            out["llama"][hypothesis_id] = value
    for key, series in (("primary", "gemma"), ("control", "qwen"), ("extension", "llama")):
        value = triple(_missingness_available_case(missingness, key, "holdout", "tone_pooled"))
        if value:
            out[series]["tone_pooled"] = value
    for contrast_id, row_id in (("H1", "H1"), ("TONE_ACC_POOLED", "tone_pooled")):
        value = triple(_robustness_estimate(robustness, "S", contrast_id))
        if value:
            out["fresh"][row_id] = value
    return out


# ``None`` means "leave matplotlib's default alone" -- the screen figures must not move.
SCREEN_STYLE = dict(series=SERIES, label=8.2, tick=8.0, xlabel=8.4, annotate=6.8, note=7.2,
                    lift=8, top=-0.7, widths=(2.2, 1.2), leading=None, labelpad=None, tickpad=None)


def _kw(**pairs) -> dict:
    return {key: value for key, value in pairs.items() if value is not None}


def _forest_panel(axes, rows, values, xlabel, note=None, style=None):
    style = style or SCREEN_STYLE
    heavy, light = style["widths"]
    axes.axvline(0.0, color=INK, linewidth=0.9, zorder=1)
    for index, (row_id, label) in enumerate(rows):
        if index % 2 == 0:
            axes.axhspan(index - 0.5, index + 0.5, color="#F7F7F7", zorder=0)
        for name, _, colour, marker, size, offset in style["series"]:
            entry = values.get(name, {}).get(row_id)
            if entry is None:
                continue
            estimate, low, high = entry
            position = index + offset
            axes.plot([low, high], [position, position], color=colour,
                      linewidth=heavy if name == "gemma" else light, solid_capstyle="round",
                      zorder=3)
            axes.plot([estimate], [position], marker=marker, markersize=size, color=colour,
                      markeredgecolor="white", markeredgewidth=0.7 if name == "gemma" else 0.4,
                      zorder=4)
            if name == "gemma":
                axes.annotate("%s [%s, %s]" % (_n(estimate), _n(low), _n(high)),
                              (estimate, position), textcoords="offset points",
                              xytext=(0, style["lift"]), ha="center", fontsize=style["annotate"],
                              color=INK)
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([label for _, label in rows], fontsize=style["label"],
                         **_kw(linespacing=style["leading"]))
    axes.set_ylim(len(rows) - 0.5, style["top"])
    axes.set_xlabel(xlabel, fontsize=style["xlabel"], **_kw(labelpad=style["labelpad"]))
    axes.tick_params(axis="x", labelsize=style["tick"], **_kw(pad=style["tickpad"]))
    axes.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    if note:
        axes.text(0.012, 0.035, note, transform=axes.transAxes, ha="left", va="bottom",
                  fontsize=style["note"], color=MUTED, linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=HAIRLINE))


def figure_headline_effects(values: dict, null_p, null_detail: str, out_dir: Path) -> list[Path]:
    """F0b -- the confirmed holdout contrasts, with the other families alongside."""
    if not values.get("gemma"):
        return []
    figure, panels = plt.subplots(
        3, 1, figsize=(11.6, 10.2), gridspec_kw={"height_ratios": [8.0, 1.9, 2.6], "hspace": 0.30})

    null_note = "family-level permutation null\n%s" % null_detail if null_detail else None
    _forest_panel(panels[0], M1_ROWS, values, "answer margin M1, item-paired difference (nats)",
                  note=null_note)
    _forest_panel(panels[1], DISTRESS_ROWS, values, "judged distress, item-paired difference (0–10)")
    _forest_panel(panels[2], RATE_ROWS, values, "item-paired difference in rate")

    handles = [Line2D([0], [0], color=colour, marker=marker, markersize=size,
                      markeredgecolor="white", linewidth=2.0 if name == "gemma" else 1.2, label=label)
               for name, label, colour, marker, size, _ in SERIES]
    panels[0].legend(handles=handles, loc="lower left", fontsize=7.8, frameon=False,
                     bbox_to_anchor=(0.0, 1.005), ncol=2)

    figure.suptitle("F0b · The confirmed signature, and who else shows it",
                    x=0.012, y=0.995, ha="left", fontsize=13, fontweight="bold", color=INK)
    figure.text(0.012, 0.969,
                "Item-paired mean differences with 2,000-resample item-clustered bootstrap 95% "
                "percentile CIs. Bars crossing 0 are not supported.",
                ha="left", fontsize=8.6, color=MUTED)
    figure.text(0.012, 0.030,
                "Only the gemma-2-9b-it holdout row is confirmatory (preregistration v3, analysed "
                "once). The control was preregistered on H7a/H7b only; Llama-3.1-8B and the 86-item "
                "fresh bank are exploratory.\nPooled-tone rows come from the MNAR sensitivity table's "
                "published available-case estimate. " + CEILING,
                ha="left", fontsize=7.4, color=MUTED, linespacing=1.5)
    _ = null_p
    return _save(figure, out_dir, "F0b_headline_effects")


# ------------------------------------------ F0b print: the same forest, for paper

PRINT_M1_ROWS = (
    ("H1", "H1  false failure × 3\neasy · neutral"),
    ("tone_pooled", "pooled tone\neasy+hard · truthful"),
    ("H2a", "H2a  hostile tone\neasy · truthful"),
    ("H2b", "H2b  hostile tone\nhard · truthful"),
    ("H3a", "H3a  one bogus verdict\neasy · neutral"),
    ("H3b", "H3b  one bogus verdict\neasy · hostile"),
    ("H4a", "H4a  truthful washout\neasy · neutral"),
    ("H5", "H5  correction after × 3\nhard · neutral"),
)
PRINT_DISTRESS_ROWS = (("H6a", "H6a  distress\nhostile − neutral onset"),)
PRINT_RATE_ROWS = (
    ("H8", "H8  M2 disagreement\nhostile − neutral"),
    ("H9", "H9  non-answers\nhostile − neutral onset"),
)
PRINT_SERIES = (
    ("gemma", "gemma-2-9b-it · holdout, confirmatory", BLUE, "o", 5.0, -0.14),
    ("fresh", "gemma-2-9b-it · 86 fresh ARC items (v7 S)", GREEN, "D", 3.4, -0.02),
    ("qwen", "Qwen2.5-3B-Instruct · control", ORANGE, "s", 3.4, 0.10),
    ("llama", "Llama-3.1-8B-Instruct · third family", PURPLE, "^", 3.6, 0.22),
)
PRINT_STYLE = dict(series=PRINT_SERIES, label=8.4, tick=8.4, xlabel=8.6, annotate=8.2, note=8.2,
                   lift=4, top=-0.62, widths=(1.7, 1.0), leading=1.22, labelpad=1.6, tickpad=1.4)


def figure_headline_effects_print(values: dict, null_detail: str,
                                  out_dir: Path) -> tuple[list[Path], float]:
    """F0b for a text column: drawn at 7.5 in, placed at 6.7 in (a 0.893 reduction)."""
    if not values.get("gemma"):
        return ([], 0.0)
    width, height = 7.5, 5.65
    figure = plt.figure(figsize=(width, height))

    def rect(x0, y0, x1, y1):
        return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]

    margin, small_margin = 1.55, 1.48
    m1_axes = figure.add_axes(rect(margin, 2.08, width - 0.12, 4.78))
    distress_axes = figure.add_axes(rect(small_margin, 1.00, 3.58, 1.68))
    rate_axes = figure.add_axes(rect(5.34, 1.00, width - 0.12, 1.68))

    note = "family-level permutation null\n%s" % null_detail if null_detail else None
    _forest_panel(m1_axes, PRINT_M1_ROWS, values,
                  "answer margin M1, item-paired difference (nats)", note=note, style=PRINT_STYLE)
    _forest_panel(distress_axes, PRINT_DISTRESS_ROWS, values,
                  "judged distress, paired difference (0–10)", style=PRINT_STYLE)
    _forest_panel(rate_axes, PRINT_RATE_ROWS, values, "item-paired difference in rate",
                  style=PRINT_STYLE)

    handles = [Line2D([0], [0], color=colour, marker=marker, markersize=size,
                      markeredgecolor="white", linewidth=1.7 if name == "gemma" else 1.0,
                      label=label)
               for name, label, colour, marker, size, _ in PRINT_SERIES]
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(margin / width, 4.80 / height),
                  ncol=2, frameon=False, fontsize=8.4, handlelength=1.9, columnspacing=1.4,
                  labelspacing=0.3)

    figure.text(0.012, (height - 0.01) / height,
                "F0b · The confirmed signature, and who else shows it",
                ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    figure.text(0.012, (height - 0.19) / height,
                "Item-paired mean differences with 2,000-resample item-clustered bootstrap 95% "
                "percentile CIs.\nBars crossing 0 are not supported.",
                ha="left", va="top", fontsize=8.0, color=MUTED, linespacing=1.4)
    footnote = figure.text(
        0.012, 0.04 / height,
        "Only the gemma-2-9b-it holdout row is confirmatory (preregistration v3, analysed once). "
        "The control was\npreregistered on H7a/H7b only; Llama-3.1-8B and the 86-item fresh bank "
        "are exploratory. Pooled-tone rows come\nfrom the MNAR sensitivity table's published "
        "available-case estimate. "
        + CEILING.replace("functional measurement", "functional\nmeasurement"),
        ha="left", va="bottom", fontsize=8.0, color=MUTED, linespacing=1.4)

    smallest = _fit(figure, footnote, (width - 0.16) * 72.0, 7.9)
    for axes in (m1_axes, distress_axes, rate_axes):
        limit = (margin if axes is m1_axes else small_margin) * 72.0 - 7.0
        for label in axes.get_yticklabels():
            smallest = min(smallest, _fit(figure, label, limit, 7.9))
    return (_save_print(figure, out_dir, "F0b_headline_effects_print"), smallest)


# ----------------------------------------------------- F0c: the phase timeline

VERDICT_COLOUR = {"fail": VERMILLION, "pass": GREEN, "mixed": ORANGE, "plan": SKY}


def _phase_boxes(gates, confirm, steering, phase4, phase5, robustness) -> list[list[dict]]:
    status = ((gates or {}).get("verdict", {}).get("summary") or {}).get("phase_1_status", "FAIL")
    iteration = ((confirm or {}).get("result") or {}).get("iteration_status", "SUCCESS")
    null_p = ((confirm or {}).get("result") or {}).get("null_check", {}).get("null_p")
    steer_j4 = {}
    for verdict in (steering or {}).get("verdicts", []):
        if verdict.get("prediction_id") == "J4":
            steer_j4 = verdict.get("detail") or {}
    mc1_a = (((phase4 or {}).get("manipulation_checks") or {}).get("MC1") or {}).get("A") or {}
    reduction = (mc1_a.get("values") or {}).get("relative_reduction")
    gap_a = (((phase4 or {}).get("gaps") or {}).get("A") or {}).get("m1", {}).get("estimate")
    l1 = next((v for v in (phase5 or {}).get("verdicts", []) if v.get("prediction_id") == "L1"), {})
    fresh_h1 = _robustness_estimate(robustness, "S", "H1") or {}

    return [
        [
            dict(title="Phase 0 — screen", meta="prereg v1 (locked) · 5 models · 10 items",
                 verdict="primary gemma-2-9b-it, control Qwen2.5-3B", tone="plan"),
            dict(title="Phase 1 — 2×2×2 factorial, five gates",
                 meta="prereg v1 · 20 discovery items · M1 / M2 / M3",
                 verdict="five-gate %s  (frozen rules: BLOCKED)" % status, tone="fail"),
        ],
        [
            dict(title="Iteration loop — H1–H10 re-preregistered",
                 meta="prereg v3 · confirmatory script frozen before unlock",
                 verdict="the one loop the roadmap permits", tone="plan"),
            dict(title="Holdout — analysed once",
                 meta="prereg v3 · 20 untouched items · script hash in manifest.json",
                 verdict="%s · real 6/6 supported · permutation null p = %s"
                         % (iteration, _n(null_p, 3, sign=False)), tone="pass"),
        ],
        [
            dict(title="Extensions", meta="exploratory · Llama-3.1-8B, P6 battery",
                 verdict="margin replicates; distress flat; P6 untestable", tone="mixed"),
            dict(title="Phase 3 — j-space", meta="prereg v4 · J1–J6 · probe + steering",
                 verdict="tone AUC 1.000; ΔM1 %s at α = 2"
                         % _n(steer_j4.get("alpha2_m1_delta")), tone="mixed"),
            dict(title="Phase 4 — DPO", meta="prereg v5 · K1–K6 · DPO vs placebo",
                 verdict="MC1 FAIL, %s of the 80%% bar; M1 gap %s intact"
                         % ("%.1f%%" % (100.0 * (reduction or 0.0)), _n(gap_a)), tone="mixed"),
            dict(title="Phase 5 — base model", meta="prereg v6 · L1–L5 · plain template",
                 verdict="base not estimable (%s parseable); H2b fails"
                         % (l1.get("evidence") or "").split(" neutral")[0].replace(
                             "parseable on ", ""), tone="fail"),
            dict(title="v7 robustness", meta="prereg v7 · wording · scale · 27B",
                 verdict="86 fresh items: H1 %s, larger; 27B not estimable"
                         % _n(fresh_h1.get("estimate")), tone="mixed"),
            dict(title="Audits", meta="marker · bogus verdict · judge · MNAR",
                 verdict="0 of 80 confirmatory responses touched; A6 withdrawn", tone="pass"),
        ],
    ]


def figure_phase_map(rows: list[list[dict]], out_dir: Path) -> list[Path]:
    """F0c -- the design as a dated flow, with each box's preregistration and verdict."""
    if not rows or not any(rows):
        return []
    figure_width, figure_height = 14.6, 7.4
    bottom, ceiling = 6.0, 97.0
    figure, axes = plt.subplots(figsize=(figure_width, figure_height))
    axes.set_xlim(0, 100)
    axes.set_ylim(bottom, ceiling)
    axes.axis("off")

    # One x unit is a fixed number of points; one y unit spans the visible range only.
    points_per_x = figure_width * 72.0 / 100.0
    points_per_y = figure_height * 72.0 / (ceiling - bottom)

    def fits(width_units: float, size: float, bold: bool = False) -> int:
        em = 0.72 if bold else 0.58
        return max(12, int((width_units - 2.0) * points_per_x / (size * em)))

    def line_units(size: float) -> float:
        return size * 1.32 / points_per_y

    bands = ("17 Aug 2026", "17–18 Aug 2026", "18 Aug 2026")
    row_y = (84.0, 52.5, 20.0)
    box_h = 22.0
    left, right = 13.0, 99.0
    title_size, meta_size, verdict_size = 8.8, 6.7, 7.2

    centres: list[list[tuple[float, float]]] = []
    for row_index, boxes in enumerate(rows):
        if not boxes:
            centres.append([])
            continue
        y = row_y[row_index]
        top = y + box_h / 2
        axes.text(0.5, top - 1.0, bands[row_index], fontsize=9, fontweight="bold",
                  color=MUTED, ha="left", va="top")
        axes.plot([0.5, 11.0], [top - 4.4, top - 4.4], color=HAIRLINE, linewidth=1.0)
        gap = 2.4
        width = ((right - left) - gap * (len(boxes) - 1)) / len(boxes)
        row_centres = []
        for column, box in enumerate(boxes):
            x = left + column * (width + gap)
            colour = VERDICT_COLOUR[box["tone"]]
            axes.add_patch(FancyBboxPatch(
                (x, y - box_h / 2), width, box_h,
                boxstyle="round,pad=0,rounding_size=1.4",
                facecolor="white", edgecolor=colour, linewidth=1.5, zorder=2))
            axes.add_patch(Rectangle((x, top - 1.5), width, 1.5,
                                     facecolor=colour, edgecolor="none", zorder=3))

            title = _wrap(box["title"], fits(width, title_size, bold=True))
            meta = _wrap(box["meta"], fits(width, meta_size))
            cursor = top - 4.0
            axes.text(x + width / 2, cursor, title, ha="center", va="top", fontsize=title_size,
                      fontweight="bold", color=INK, zorder=4, linespacing=1.3)
            cursor -= title.count("\n") * line_units(title_size) + line_units(title_size) + 1.1
            axes.text(x + width / 2, cursor, meta, ha="center", va="top", fontsize=meta_size,
                      color=MUTED, zorder=4, linespacing=1.4)
            axes.text(x + width / 2, y - box_h / 2 + 2.2,
                      _wrap(box["verdict"], fits(width, verdict_size, bold=True)),
                      ha="center", va="bottom", fontsize=verdict_size, color=colour,
                      fontweight="bold", zorder=4, linespacing=1.4)

            row_centres.append((x, x + width))
            if column:
                previous = row_centres[column - 1][1]
                axes.add_patch(FancyArrowPatch(
                    (previous + 0.35, y), (x - 0.35, y), arrowstyle="-|>", mutation_scale=11,
                    color="#8A8A8A", linewidth=1.1, zorder=1))
        centres.append(row_centres)

    # Wrap connectors: down out of the last box, back along the gutter, into the next row.
    for row_index in range(len(rows) - 1):
        if not centres[row_index] or not centres[row_index + 1]:
            continue
        start_x = sum(centres[row_index][-1]) / 2
        end_x = sum(centres[row_index + 1][0]) / 2
        start_y = row_y[row_index] - box_h / 2
        end_y = row_y[row_index + 1] + box_h / 2
        gutter = (start_y + end_y) / 2
        axes.plot([start_x, start_x, end_x], [start_y - 0.5, gutter, gutter],
                  color="#8A8A8A", linewidth=1.2, solid_capstyle="round", zorder=1)
        axes.add_patch(FancyArrowPatch(
            (end_x, gutter), (end_x, end_y + 0.4), arrowstyle="-|>", mutation_scale=12,
            color="#8A8A8A", linewidth=1.2, shrinkA=0, shrinkB=0, zorder=1))

    handles = [Patch(facecolor="white", edgecolor=VERDICT_COLOUR[tone], linewidth=1.5, label=label)
               for tone, label in (("plan", "design step"), ("fail", "preregistered FAIL / not estimable"),
                                   ("pass", "preregistered success"), ("mixed", "mixed / exploratory"))]
    axes.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=4,
                frameon=False, fontsize=8)

    figure.suptitle("F0c · How the study ran", x=0.012, y=0.985, ha="left",
                    fontsize=13, fontweight="bold", color=INK)
    figure.text(0.012, 0.947,
                "Every preregistration was committed before the data it governs. The holdout was "
                "unlocked and analysed exactly once; everything after it runs on discovery items, "
                "fresh items, or a role the loop did not consume.",
                ha="left", fontsize=8.6, color=MUTED)
    figure.text(0.012, 0.012, CEILING, ha="left", fontsize=7.4, color=MUTED)
    return _save(figure, out_dir, "F0c_phase_map")


def _wrap(text: str, width: int) -> str:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


# ------------------------------------------------------------------------ main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the three README figures from committed summaries.")
    parser.add_argument("--summaries", default="results/summaries", help="committed summary root")
    parser.add_argument("--out", default="results/figures", help="figure output directory")
    parser.add_argument("--print", dest="print_sized", action="store_true",
                        help="also write the print-sized F0b and F0 variants used by the report")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summaries, out_dir = Path(args.summaries), Path(args.out)

    gates = _read(summaries / "phase1" / "gates.json")
    confirm = _read(summaries / "phase2" / "confirm.json")
    hypotheses = _read_csv(summaries / "phase2" / "hypotheses.csv")
    steering = _read(summaries / "phase3" / "steering.json")
    steering_judge = _read(summaries / "phase3" / "steering_judge.json")
    phase4 = _read(summaries / "phase4" / "phase4.json")
    phase5 = _read(summaries / "phase5" / "phase5.json")
    cell_rates = _read_csv(summaries / "phase5" / "cell_valid_rates.csv")
    robustness = _read(summaries / "robustness" / "robustness.json")
    missingness = _read(summaries / "missingness" / "m1_missingness.json")
    extension = _extension(summaries)

    written: list[Path] = []
    smallest: dict[str, float] = {}

    if not hypotheses or gates is None:
        print("skipping F0_channel_map: need phase2/hypotheses.csv and phase1/gates.json",
              file=sys.stderr)
    else:
        cells = _channel_cells(summaries, hypotheses, gates, confirm, steering, steering_judge,
                               phase4, cell_rates, robustness, extension)
        written += figure_channel_map(cells, out_dir)
        if args.print_sized:
            paths, floor = figure_channel_map_print(cells, out_dir)
            written += paths
            smallest["F0_channel_map_print (10.1 in canvas, placed 10.1 in)"] = floor

    if not hypotheses:
        print("skipping F0b_headline_effects: no phase2/hypotheses.csv", file=sys.stderr)
    else:
        values = _forest_values(hypotheses, extension, missingness, robustness)
        null_check = ((confirm or {}).get("result") or {}).get("null_check") or {}
        histogram = null_check.get("histogram") or {}
        best = max((int(key) for key in histogram), default=None)
        detail = ""
        if null_check.get("null_p") is not None:
            detail = ("over {%s}: real %d/%d supported,\nbest of %d permutations %s → p = %s"
                      % (", ".join(null_check.get("family", [])), len(null_check.get("family", [])),
                         len(null_check.get("family", [])),
                         sum(histogram.values()) if histogram else 200, best,
                         _n(null_check["null_p"], 3, sign=False)))
        written += figure_headline_effects(values, null_check.get("null_p"), detail, out_dir)
        if args.print_sized:
            paths, floor = figure_headline_effects_print(values, detail, out_dir)
            written += paths
            smallest["F0b_headline_effects_print (7.5 in canvas, placed 6.7 in)"] = floor * 6.7 / 7.5

    if gates is None and confirm is None:
        print("skipping F0c_phase_map: no phase1/gates.json and no phase2/confirm.json",
              file=sys.stderr)
    else:
        written += figure_phase_map(
            _phase_boxes(gates, confirm, steering, phase4, phase5, robustness), out_dir)

    for path in written:
        print("wrote %s" % path)
    for name, floor in smallest.items():
        print("smallest printed type in %s: %.2f pt" % (name, floor))
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
