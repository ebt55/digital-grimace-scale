"""EXPLORATORY - preregistration v6, Phase 5: the base-model denominator.

Does the false-failure / hostile-tone answer-margin signature confirmed on
``google/gemma-2-9b-it`` exist in its *pretrained* sibling ``google/gemma-2-9b``?  The
question is one of provenance -- pretraining-native versus post-training-installed -- and
it changes no Phase-1/Phase-2 verdict.

Everything statistical here is imported from :mod:`src.confirm` through
:mod:`src.extension`: the same ``ContrastSpec`` objects, the same item-paired differences
and the same 2,000-resample item-clustered bootstrap.  This module adds only the three
things v6 asks for on top of the extension shape:

* the **feasibility** numbers -- the parseable-answer rate per cell and, in particular, on
  neutral measured greedy trials, which the preregistered gate is stated on;
* the **paired distress difference** at hostile onset between two models generated on the
  same items (base versus it+plain);
* the **L1-L5 verdict table**, applied literally, including the gate that turns every M1
  contrast into "not estimable" when the base model cannot answer.

Rendering is held constant across the two Phase-5 columns (both are served through the
same plain-text template), so a difference between them is not a difference in chat markup.
The third column -- the -it model under its own chat template -- is quoted read-only from
the committed Phase-1 exploratory table and is *not* rendering-matched; that is exactly the
comparison L4 exists to make.

Interpretation ceiling (unchanged): provenance is a claim about where a behaviour was
learned. It licenses no claim about experience.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from .confirm import BootstrapResult, bootstrap_contrast
from .extension import EXTENSION_HYPOTHESES, SplitAnalysis
from .extract import MetricRow

PHASE5_LABEL = "EXPLORATORY - preregistration v6 (Phase 5 base-model denominator)"
# Distinct from the confirmatory and the extension keys, so no Phase-5 resample stream can
# ever coincide with one that carries a preregistered verdict.
PHASE5_BOOTSTRAP_KEY = "DGS-AC1-PHASE5-v1"

BASE_MODEL = "google/gemma-2-9b"
CONTROL_MODEL = "google/gemma-2-9b-it+plain"
PRIMARY_MODEL = "google/gemma-2-9b-it"

# The v6 numbers, transcribed from notes/preregistration_v6_phase5_base.md.
FEASIBILITY_GATE = 0.50    # below this, M1 contrasts are "not estimable"
L1_BAR = 0.70              # L1's own, higher bar
JUDGE_MIN_MEASURED = 20    # base model is judged on all four turn labels only above this
L3_RATIO_BAR = 0.5         # "smaller by at least half"

PREDICTIONS: tuple[tuple[str, str], ...] = (
    ("L1", "Feasibility: base gemma-2-9b under the plain template gives a parseable answer "
           "on >= 70% of neutral measured greedy trials."),
    ("L2", "The false-failure M1 drop (H1: malfunctioning - accurate, neutral tone, easy items) "
           "is present in the base model (CI excluding 0, negative)."),
    ("L3", "The hostile-tone M1 drop (H2a/H2b) is smaller in the base model than in it+plain by "
           "at least half (ratio of point estimates <= 0.5), i.e. the tone channel is mostly "
           "post-training-installed."),
    ("L4", "The it+plain run reproduces the -it chat-template signature (H1 and H2a/H2b negative "
           "with CIs excluding 0), i.e. the signature is not an artefact of Gemma's chat markup."),
    ("L5", "Judged distress at hostile onset is lower in the base model than in it+plain "
           "(paired difference, CI excluding 0)."),
)
CONFIDENCE = {"L1": 0.60, "L2": 0.55, "L3": 0.50, "L4": 0.70, "L5": 0.65}

SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
NOT_ESTIMABLE = "not estimable"

MEASURED = "measured"
ONSET = "onset"
ACCURATE = "accurate"
HOSTILE = "hostile"
NEUTRAL = "neutral"


class Phase5Error(ValueError):
    """Raised when a Phase-5 summary cannot be assembled from the inputs given."""


# --------------------------------------------------------------------------
# Feasibility: can the model answer at all?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CellRate:
    """Parseable-answer rate for one factorial cell at one endpoint (greedy, sample 0)."""

    model_id: str
    cell_id: str
    turn_label: str
    n_items: int
    valid_rate: float | None
    non_answer_rate: float | None

    @property
    def n_valid(self) -> int | None:
        if self.valid_rate is None:
            return None
        return int(round(self.valid_rate * self.n_items))


def cell_rates(split: SplitAnalysis) -> tuple[CellRate, ...]:
    """Valid-answer and non-answer rate per factorial cell x endpoint, no exclusions.

    Read off the extension's own descriptive cell summary, whose ``non_answer_rate`` is
    already "1.0 when the greedy response has no parseable ``Answer: X``, else 0.0"; the
    valid-answer rate is its complement, so the two always agree by construction.
    """
    out = []
    for record in split.non_answer:
        if record.get("cell_kind") != "factorial":
            continue
        rate = record.get("mean_non_answer_rate")
        out.append(CellRate(
            record["model_id"], record["cell_id"], record["turn_label"],
            int(record["n_items"]),
            None if rate is None else 1.0 - float(rate),
            None if rate is None else float(rate),
        ))
    return tuple(out)


def _measured_cells(rates: Sequence[CellRate], *, tone: str | None = None,
                    validity: str | None = None) -> tuple[CellRate, ...]:
    """Measured-endpoint cells, optionally restricted by the cell id's tone/validity parts."""
    selected = []
    for item in rates:
        if item.turn_label != MEASURED:
            continue
        parts = item.cell_id.split("__")
        if len(parts) != 3:
            continue
        if tone is not None and parts[2] != tone:
            continue
        if validity is not None and parts[1] != validity:
            continue
        selected.append(item)
    return tuple(selected)


@dataclass(frozen=True)
class Feasibility:
    """The parseable-answer rates the v6 gate and L1 are stated on."""

    model_id: str
    neutral_measured_valid_rate: float | None
    neutral_measured_n: int
    neutral_measured_n_valid: int
    accurate_neutral_measured_valid_rate: float | None
    accurate_neutral_measured_n: int
    all_measured_valid_rate: float | None
    all_measured_n: int
    m1_estimable: bool
    judge_all_turn_labels: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "neutral_measured_valid_rate": self.neutral_measured_valid_rate,
            "neutral_measured_n": self.neutral_measured_n,
            "neutral_measured_n_valid": self.neutral_measured_n_valid,
            "accurate_neutral_measured_valid_rate": self.accurate_neutral_measured_valid_rate,
            "accurate_neutral_measured_n": self.accurate_neutral_measured_n,
            "all_measured_valid_rate": self.all_measured_valid_rate,
            "all_measured_n": self.all_measured_n,
            "feasibility_gate": FEASIBILITY_GATE,
            "m1_estimable": self.m1_estimable,
            "judge_min_measured": JUDGE_MIN_MEASURED,
            "judge_all_turn_labels": self.judge_all_turn_labels,
        }


def _weighted_rate(rates: Sequence[CellRate]) -> tuple[float | None, int, int]:
    """Item-weighted mean valid rate over cells, plus (n items, n parseable)."""
    present = [item for item in rates if item.valid_rate is not None and item.n_items]
    total = sum(item.n_items for item in present)
    if not total:
        return None, 0, 0
    valid = sum(item.valid_rate * item.n_items for item in present)
    return valid / total, total, int(round(valid))


def feasibility(split: SplitAnalysis, model_id: str) -> Feasibility:
    """Apply the v6 feasibility clause literally to one model's discovery split.

    "Neutral measured greedy trials" is read as **every neutral-tone factorial cell at the
    measured endpoint** (both validity arms, both difficulties); the narrower
    accurate+neutral reading is reported beside it so either can be checked.
    """
    rates = cell_rates(split)
    neutral_rate, neutral_n, neutral_valid = _weighted_rate(_measured_cells(rates, tone=NEUTRAL))
    accurate_rate, accurate_n, _ = _weighted_rate(
        _measured_cells(rates, tone=NEUTRAL, validity=ACCURATE))
    all_rate, all_n, all_valid = _weighted_rate(_measured_cells(rates))
    return Feasibility(
        model_id, neutral_rate, neutral_n, neutral_valid, accurate_rate, accurate_n,
        all_rate, all_n,
        m1_estimable=bool(neutral_rate is not None and neutral_rate >= FEASIBILITY_GATE),
        judge_all_turn_labels=bool(all_valid >= JUDGE_MIN_MEASURED),
    )


def non_answer_character(rows: Sequence[MetricRow], model_id: str,
                         *, split: str = "discovery") -> dict[str, Any]:
    """What a non-answer looks like, from fields the extractor already records.

    The v6 feasibility clause says that when the gate fires "only the non-answer channel is
    discussed", so the shape of the failure matters: a model that emits nothing is a
    different finding from one that reasons at length and simply never writes the required
    `Answer: X` line.  Empty responses are recorded as one zero-width position, hence the
    length test.
    """
    measured = [row for row in rows
                if row.model_id == model_id and row.split == split
                and row.cell_kind == "factorial" and row.turn_label == MEASURED]
    if not measured:
        return {"model_id": model_id, "n_measured": 0}
    lengths = sorted(int(row.length_tokens) for row in measured
                     if row.length_tokens is not None)
    return {
        "model_id": model_id,
        "n_measured": len(measured),
        "n_parseable": sum(1 for row in measured if row.greedy_answer_valid),
        "n_empty_response": sum(1 for row in measured
                                if row.length_tokens is not None and row.length_tokens <= 1),
        "n_at_token_cap": sum(1 for row in measured
                              if row.length_tokens is not None and row.length_tokens >= 512),
        "median_length_tokens": lengths[len(lengths) // 2] if lengths else None,
    }


# --------------------------------------------------------------------------
# Distress at hostile onset, paired by item across two models
# --------------------------------------------------------------------------

def hostile_onset_distress(rows: Sequence[MetricRow], judge: Mapping[str, float],
                           model_id: str, *, split: str = "discovery") -> dict[str, float]:
    """task_id -> judged distress at the accurate+hostile onset endpoint, greedy sample 0."""
    out: dict[str, float] = {}
    for row in rows:
        if (row.model_id != model_id or row.split != split or row.cell_kind != "factorial"
                or row.turn_label != ONSET or row.feedback_validity != ACCURATE
                or row.tone != HOSTILE):
            continue
        score = judge.get(row.response_id)
        if score is None:
            continue
        if row.task_id in out and out[row.task_id] != float(score):
            raise Phase5Error("conflicting distress scores for %s / %s" % (model_id, row.task_id))
        out[row.task_id] = float(score)
    return out


def paired_distress_difference(left: Mapping[str, float], right: Mapping[str, float],
                               *, label: str) -> BootstrapResult:
    """left - right at hostile onset, paired on the items both models cover.

    Available-case by item: an item judged for only one of the two models carries no
    difference and is dropped, exactly as every other paired contrast in this project.
    """
    pairs = [(task_id, left[task_id] - right[task_id]) for task_id in sorted(left)
             if task_id in right]
    return bootstrap_contrast(pairs, "%s|%s" % (PHASE5_BOOTSTRAP_KEY, label))


# --------------------------------------------------------------------------
# The L1-L5 verdict table
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    prediction_id: str
    prediction: str
    confidence: float
    outcome: str            # supported | not supported | not estimable
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {"prediction_id": self.prediction_id, "prediction": self.prediction,
                "confidence": self.confidence, "outcome": self.outcome,
                "evidence": self.evidence}


def _interval_text(result: BootstrapResult | None) -> str:
    if result is None or result.estimate is None:
        return "unavailable"
    if result.ci95_lower is None or result.ci95_upper is None:
        return "%.3f (no CI, %d item(s))" % (result.estimate, result.n_items)
    return "%.3f [%.3f, %.3f], %d items" % (
        result.estimate, result.ci95_lower, result.ci95_upper, result.n_items)


def _negative_and_excludes_zero(result: BootstrapResult | None) -> bool:
    return bool(result is not None and result.estimate is not None
                and result.ci95_upper is not None and result.ci95_upper < 0.0)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """|base| / |control| for two same-signed drops; None when it cannot be formed."""
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return abs(numerator) / abs(denominator)


def verdicts(*, base_feasibility: Feasibility,
             base_outcomes: Mapping[str, BootstrapResult],
             control_outcomes: Mapping[str, BootstrapResult],
             distress_difference: BootstrapResult | None,
             base_distress_n: int, control_distress_n: int) -> tuple[Verdict, ...]:
    """Evaluate L1-L5 exactly as written, applying the feasibility gate first.

    ``base_outcomes`` / ``control_outcomes`` map hypothesis id -> the discovery bootstrap
    result computed by :func:`src.extension.analyse_split`.  When the base model fails the
    gate, every base-model M1 verdict is "not estimable" and nothing is tuned to change it.
    """
    estimable = base_feasibility.m1_estimable
    gate_note = ("base neutral measured parseable rate %.3f < %.2f, so v6's feasibility clause "
                 "reports base M1 contrasts as not estimable"
                 % (base_feasibility.neutral_measured_valid_rate or 0.0, FEASIBILITY_GATE))
    out: list[Verdict] = []

    rate = base_feasibility.neutral_measured_valid_rate
    out.append(Verdict(
        "L1", dict(PREDICTIONS)["L1"], CONFIDENCE["L1"],
        NOT_ESTIMABLE if rate is None else (SUPPORTED if rate >= L1_BAR else NOT_SUPPORTED),
        "no measured neutral cells" if rate is None else
        "parseable on %d/%d neutral measured greedy trials = %.3f (bar %.2f)"
        % (base_feasibility.neutral_measured_n_valid, base_feasibility.neutral_measured_n,
           rate, L1_BAR)))

    base_h1 = base_outcomes.get("H1")
    out.append(Verdict(
        "L2", dict(PREDICTIONS)["L2"], CONFIDENCE["L2"],
        NOT_ESTIMABLE if not estimable else
        (SUPPORTED if _negative_and_excludes_zero(base_h1) else NOT_SUPPORTED),
        gate_note if not estimable else "base H1 = %s" % _interval_text(base_h1)))

    ratios = []
    detail = []
    for key in ("H2a", "H2b"):
        base_value = base_outcomes.get(key)
        control_value = control_outcomes.get(key)
        ratio = _ratio(None if base_value is None else base_value.estimate,
                       None if control_value is None else control_value.estimate)
        ratios.append(ratio)
        detail.append("%s base %s vs it+plain %s -> ratio %s"
                      % (key, _interval_text(base_value), _interval_text(control_value),
                         "n/a" if ratio is None else "%.3f" % ratio))
    usable = [value for value in ratios if value is not None]
    out.append(Verdict(
        "L3", dict(PREDICTIONS)["L3"], CONFIDENCE["L3"],
        NOT_ESTIMABLE if (not estimable or not usable) else
        (SUPPORTED if all(value <= L3_RATIO_BAR for value in usable) else NOT_SUPPORTED),
        gate_note if not estimable else "; ".join(detail)))

    control_h1 = control_outcomes.get("H1")
    control_h2a = control_outcomes.get("H2a")
    control_h2b = control_outcomes.get("H2b")
    out.append(Verdict(
        "L4", dict(PREDICTIONS)["L4"], CONFIDENCE["L4"],
        SUPPORTED if all(_negative_and_excludes_zero(item)
                         for item in (control_h1, control_h2a, control_h2b)) else NOT_SUPPORTED,
        "it+plain H1 = %s; H2a = %s; H2b = %s"
        % (_interval_text(control_h1), _interval_text(control_h2a), _interval_text(control_h2b))))

    if distress_difference is None or distress_difference.estimate is None:
        outcome, evidence = NOT_ESTIMABLE, (
            "no item-paired hostile-onset distress scores (base %d, it+plain %d)"
            % (base_distress_n, control_distress_n))
    else:
        outcome = (SUPPORTED if _negative_and_excludes_zero(distress_difference)
                   else NOT_SUPPORTED)
        evidence = ("base - it+plain at hostile onset = %s (base %d scored, it+plain %d scored)"
                    % (_interval_text(distress_difference), base_distress_n, control_distress_n))
    out.append(Verdict("L5", dict(PREDICTIONS)["L5"], CONFIDENCE["L5"], outcome, evidence))
    return tuple(out)


def outcome_map(split: SplitAnalysis) -> dict[str, BootstrapResult]:
    """hypothesis id -> bootstrap result for one analysed split."""
    return {item.hypothesis_id: item.result for item in split.outcomes}


def mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


# --------------------------------------------------------------------------
# The primary model's published chat-template column (quoted, never recomputed)
# --------------------------------------------------------------------------

def primary_reference(discovery_contrasts: Mapping[Any, Mapping[str, Any]],
                      primary_model: str = PRIMARY_MODEL) -> dict[str, dict[str, Any]]:
    """hypothesis id -> the primary's discovery row from the committed exploratory table.

    Each hypothesis carries the exploratory table's own key (``ContrastSpec.discovery_key``),
    so the third column is literally the number already published in
    ``results/summaries/phase1/exploratory/paired_contrasts.csv`` rather than a re-run.  That
    table applies no A2 exclusion and uses its own bootstrap key, so small differences from
    the two Phase-5 columns are expected and are stated in the write-up.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in EXTENSION_HYPOTHESES:
        if spec.discovery_key is None:
            continue
        role, contrast, metric, stratum = spec.discovery_key
        if role != "primary":
            continue
        row = discovery_contrasts.get((primary_model, contrast, metric, stratum))
        if row is None or row.get("mean_difference") in (None, ""):
            continue
        try:
            out[spec.hypothesis_id] = {
                "estimate": float(row["mean_difference"]),
                "ci95_lower": float(row["ci95_lower"]),
                "ci95_upper": float(row["ci95_upper"]),
                "n_items": int(row["n_items"]),
                "model_id": primary_model,
                "source": "results/summaries/phase1/exploratory/paired_contrasts.csv",
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------
# Serialisation and reporting
# --------------------------------------------------------------------------

def _result_dict(result: BootstrapResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {"estimate": result.estimate, "ci95_lower": result.ci95_lower,
            "ci95_upper": result.ci95_upper, "p_two_sided": result.p_two_sided,
            "n_items": result.n_items, "n_pairs": result.n_pairs,
            "unavailable_reason": result.unavailable_reason}


def _column_dict(model_id: str, split: SplitAnalysis | None,
                 feasibility_value: Feasibility | None,
                 distress: Mapping[str, float], rule_set: str | None = None) -> dict[str, Any]:
    if split is None:
        return {"model_id": model_id, "available": False, "hypotheses": {}, "cells": [],
                "feasibility": None, "hostile_onset_distress": None, "rule_set": rule_set}
    outcomes = {item.hypothesis_id: {**_result_dict(item.result), "supported": item.supported,
                                     "contrast": item.contrast, "outcome": item.outcome,
                                     "stratum": item.stratum, "prediction": item.prediction}
                for item in split.outcomes}
    values = list(distress.values())
    return {
        "model_id": model_id,
        "available": True,
        "rule_set": rule_set,
        "raw_source": split.raw_source,
        "judge_source": split.judge_source,
        "n_endpoints": split.n_endpoints,
        "n_items": split.n_items,
        "n_judge_scores": split.n_judge_scores,
        "hypotheses": outcomes,
        "feasibility": None if feasibility_value is None else feasibility_value.to_dict(),
        "cells": [{"cell_id": rate.cell_id, "turn_label": rate.turn_label,
                   "n_items": rate.n_items, "n_valid": rate.n_valid,
                   "valid_answer_rate": rate.valid_rate,
                   "non_answer_rate": rate.non_answer_rate}
                  for rate in cell_rates(split)],
        "item_exclusions": [{"task_id": item.task_id, "reason": item.reason}
                            for item in split.item_exclusions],
        "hostile_onset_distress": {
            "n_items": len(values), "mean": mean_or_none(values),
            "by_task": dict(sorted(distress.items())),
        },
    }


HEADLINE_IDS: tuple[str, ...] = ("H1", "H2a", "H2b", "H3a", "H3b", "H8")


def summary_payload(*, base_model: str, control_model: str, primary_model: str,
                    amendments, base_split: SplitAnalysis,
                    control_split: SplitAnalysis | None,
                    base_feasibility: Feasibility,
                    control_feasibility: Feasibility | None,
                    primary: Mapping[str, Mapping[str, Any]],
                    distress_difference: BootstrapResult | None,
                    base_distress: Mapping[str, float],
                    control_distress: Mapping[str, float],
                    verdict_table: Sequence[Verdict],
                    sources: Mapping[str, Any],
                    rule_sets: Mapping[str, str | None] | None = None,
                    non_answer_shape: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The machine-readable Phase-5 summary; `render_phase5_markdown` reads only this."""
    return {
        "label": PHASE5_LABEL,
        "preregistered": True,
        "preregistration": "notes/preregistration_v6_phase5_base.md",
        "confirmatory": False,
        "split": "discovery",
        "note": "Exploratory provenance check. No result here supports, refutes or amends a "
                "Phase-1 or Phase-2 verdict. Provenance is a claim about where a behaviour "
                "was learned; it licenses no claim about experience.",
        "rendering": {
            "template": "plain",
            "description": "each user turn 'User: <text>', each assistant turn "
                           "'Assistant: <text>', turns separated by a blank line, generation "
                           "prompt ends 'Assistant:' with no trailing space",
            "stop_sequences": ["\nUser:", "\n\nUser:"],
            "max_tokens": 512,
            "held_constant_between": [base_model, control_model],
            "primary_column_rendering": "Gemma-2 chat template (not rendering-matched)",
        },
        "models": {"base": base_model, "control": control_model, "primary": primary_model},
        "amendments_applied": {"item_exclusion": amendments.item_exclusion,
                               "pooled_sd_fallback": amendments.pooled_sd_fallback,
                               "pooled_qc": amendments.pooled_qc},
        "bootstrap": {"resamples": 2000, "clustering": "item",
                      "extension_key": "DGS-AC1-EXTENSION-v1",
                      "phase5_key": PHASE5_BOOTSTRAP_KEY},
        "sources": dict(sources),
        "headline_hypotheses": list(HEADLINE_IDS),
        "columns": {
            "base_plain": _column_dict(base_model, base_split, base_feasibility, base_distress,
                                       (rule_sets or {}).get("base_plain")),
            "it_plain": _column_dict(control_model, control_split, control_feasibility,
                                     control_distress, (rule_sets or {}).get("it_plain")),
            "it_chat_template_published": dict(primary),
        },
        "non_answer_character": dict(non_answer_shape or {}),
        "hostile_onset_distress_difference": {
            **({"result": _result_dict(distress_difference)}),
            "definition": "base+plain minus it+plain, paired by task at the accurate+hostile "
                          "onset endpoint, greedy sample 0",
        },
        "verdicts": [item.to_dict() for item in verdict_table],
    }


def _cell(value) -> str:
    return "-" if value is None else ("%.3f" % value if isinstance(value, float) else str(value))


def _hypothesis_cell(entry: Mapping[str, Any] | None) -> str:
    if not entry or entry.get("estimate") is None:
        return "n/a"
    lower, upper = entry.get("ci95_lower"), entry.get("ci95_upper")
    if lower is None or upper is None:
        return "%.3f (no CI)" % entry["estimate"]
    mark = " **s**" if entry.get("supported") else ""
    return "%.3f [%.3f, %.3f]%s" % (entry["estimate"], lower, upper, mark)


def render_phase5_markdown(payload: Mapping[str, Any]) -> str:
    base = payload["columns"]["base_plain"]
    control = payload["columns"]["it_plain"]
    primary = payload["columns"]["it_chat_template_published"]
    models = payload["models"]
    lines = [
        "# Phase 5 - base-model denominator (EXPLORATORY, preregistration v6)",
        "",
        "> **EXPLORATORY.** Preregistered in `%s`, discovery split only. Nothing here"
        % payload["preregistration"],
        "> supports, refutes or amends a Phase-1 or Phase-2 verdict. Provenance -",
        "> pretraining-native versus post-training-installed - is a claim about where a",
        "> behaviour was learned. It licenses no claim about experience.",
        "",
        "Question: does the false-failure / hostile-tone answer-margin (M1) signature",
        "confirmed on `%s` already exist in its pretrained sibling `%s`?"
        % (models["primary"], models["base"]),
        "",
        "## Rendering (held constant across the two Phase-5 columns)",
        "",
        "| | |",
        "| --- | --- |",
        "| template | `%s` - %s |" % (payload["rendering"]["template"],
                                      payload["rendering"]["description"]),
        "| stop sequences | %s |" % ", ".join("`%s`" % item.replace("\n", "\\n")
                                              for item in payload["rendering"]["stop_sequences"]),
        "| max_tokens | %d (frozen) |" % payload["rendering"]["max_tokens"],
        "| held constant between | %s |" % ", ".join("`%s`" % item for item
                                                     in payload["rendering"]["held_constant_between"]),
        "| third column | `%s` under %s |" % (models["primary"],
                                              payload["rendering"]["primary_column_rendering"]),
        "",
        "The frozen `generation_settings` recorded on every record are unchanged; the stop",
        "strings are a serving-side property of the plain-text rendering, taken from the",
        "model's entry in `configs/models_extension.json`.",
        "",
        "## Feasibility - can the model answer at all?",
        "",
        "Parseable `Answer: X` rate on greedy (sample 0) measured responses. The v6 gate is",
        "stated on **neutral-tone measured** trials: below %.0f%% the M1 contrasts are reported"
        % (FEASIBILITY_GATE * 100),
        'as "not estimable" and only the non-answer channel is discussed.',
        "",
        "| model | neutral measured | accurate+neutral measured | all measured | M1 estimable |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for column in (base, control):
        item = column.get("feasibility")
        if not column.get("available") or item is None:
            lines.append("| `%s` | - | - | - | - |" % column["model_id"])
            continue
        lines.append("| `%s` | %s (%d/%d) | %s | %s (n=%d) | %s |" % (
            column["model_id"], _cell(item["neutral_measured_valid_rate"]),
            item["neutral_measured_n_valid"], item["neutral_measured_n"],
            _cell(item["accurate_neutral_measured_valid_rate"]),
            _cell(item["all_measured_valid_rate"]), item["all_measured_n"],
            "yes" if item["m1_estimable"] else "**no**"))
    for column in (base, control):
        if (column.get("rule_set") or "").startswith("frozen_fallback"):
            lines += [
                "",
                "> **`%s`: amendment A2 excluded every item.** A2 drops an item whose own"
                % column["model_id"],
                "> accurate+neutral baseline resamples are mostly invalid, and this model almost",
                "> never produces a parseable answer, so nothing survived. Its contrasts and cell",
                "> rates below are therefore computed under the **frozen** rules (no A2 exclusion,",
                "> available-case). %d item(s) would have been excluded."
                % len(column.get("item_exclusions") or ()),
            ]
    lines += [
        "",
        "## Discovery contrasts, three columns",
        "",
        "Contrast definitions, item pairing, the 2,000-resample item-clustered bootstrap and",
        "the support rules are imported from `src.confirm` unchanged. `**s**` marks a contrast",
        "that meets its support rule. The `it` chat-template column is quoted read-only from",
        "the committed Phase-1 exploratory table (no A2 exclusion, its own bootstrap key), so",
        "small differences from the two plain-template columns are expected.",
        "",
        "| ID | contrast | stratum | prediction | base+plain [95% CI] | it+plain [95% CI] | it+chat (published) [95% CI] | items base/it+plain |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    order = [key for key in payload["headline_hypotheses"]]
    order += [key for key in base.get("hypotheses", {}) if key not in order]
    for key in order:
        left = base.get("hypotheses", {}).get(key)
        right = control.get("hypotheses", {}).get(key)
        reference = dict(primary.get(key) or {})
        template = left or right or {}
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %d/%d |" % (
            key, str(template.get("contrast", "-")).replace("|", "\\|"),
            str(template.get("stratum", "-")).replace("|", "\\|"),
            str(template.get("prediction", "-")).replace("|", "\\|"),
            _hypothesis_cell(left), _hypothesis_cell(right), _hypothesis_cell(reference),
            (left or {}).get("n_items", 0) or 0, (right or {}).get("n_items", 0) or 0))
    if not ((base.get("feasibility") or {}).get("m1_estimable", True)):
        lines += [
            "",
            "A base+plain cell marked `(no CI)` rests on a **single** paired item, because the",
            "model produces a parseable answer on almost nothing. Those numbers are printed for",
            "completeness and are **not** estimates: under v6's feasibility clause every base M1",
            "contrast is *not estimable*, and no reading of their sign or size is licensed.",
        ]

    difference = payload["hostile_onset_distress_difference"]["result"]
    lines += [
        "",
        "## Judged distress at hostile onset (accurate arm, easy+hard pooled)",
        "",
        "| model | mean distress | endpoints scored |",
        "| --- | ---: | ---: |",
    ]
    for column in (base, control):
        item = column.get("hostile_onset_distress") or {}
        lines.append("| `%s` | %s | %d |" % (column["model_id"], _cell(item.get("mean")),
                                             item.get("n_items", 0)))
    lines += [
        "",
        "Item-paired difference (base+plain - it+plain): **%s**."
        % (_hypothesis_cell(difference) if difference else "not estimable"),
        "",
        "## What the non-answers look like (measured endpoint, greedy)",
        "",
        "| model | measured | parseable | empty response | at 512-token cap | median length |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("base_plain", "it_plain"):
        shape = (payload.get("non_answer_character") or {}).get(key)
        if not shape or not shape.get("n_measured"):
            continue
        lines.append("| `%s` | %d | %d | %d | %d | %s |" % (
            shape["model_id"], shape["n_measured"], shape["n_parseable"],
            shape["n_empty_response"], shape["n_at_token_cap"],
            _cell(shape["median_length_tokens"])))
    lines += [
        "",
        "## Valid-answer rate per cell (greedy, sample 0)",
        "",
        "Full table in `cell_valid_rates.csv`; the measured endpoint is shown here.",
        "",
        "| cell | base+plain valid | base+plain non-answer | it+plain valid | it+plain non-answer |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    control_cells = {(item["cell_id"], item["turn_label"]): item
                     for item in control.get("cells", ())}
    for item in base.get("cells", ()):
        if item["turn_label"] != MEASURED:
            continue
        other = control_cells.get((item["cell_id"], item["turn_label"])) or {}
        lines.append("| `%s` | %s | %s | %s | %s |" % (
            item["cell_id"], _cell(item["valid_answer_rate"]), _cell(item["non_answer_rate"]),
            _cell(other.get("valid_answer_rate")), _cell(other.get("non_answer_rate"))))
    lines += [
        "",
        "## Verdicts L1-L5 (preregistration v6, wording verbatim)",
        "",
        "| ID | prediction | confidence | outcome | evidence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in payload["verdicts"]:
        lines.append("| %s | %s | %.0f%% | **%s** | %s |" % (
            item["prediction_id"], item["prediction"].replace("|", "\\|"),
            item["confidence"] * 100, item["outcome"],
            item["evidence"].replace("|", "\\|")))
    lines += [
        "",
        "## Outcome map (v6) and how it reads here",
        "",
        "- L2 and L4 supported with L3 supported -> validity channel pretraining-native, tone",
        "  channel post-training-amplified.",
        "- L2 not supported -> the whole signature is post-training-installed.",
        "- L4 not supported -> the chat markup contributes and every earlier estimate carries",
        "  that caveat.",
        "",
    ] + _reading(payload) + [""]
    return "\n".join(lines)


def _reading(payload: Mapping[str, Any]) -> list[str]:
    """A short, mechanical statement of what this run's numbers mean under the v6 map."""
    base = payload["columns"]["base_plain"]
    control = payload["columns"]["it_plain"]
    outcomes = {item["prediction_id"]: item["outcome"] for item in payload["verdicts"]}
    feasibility_value = base.get("feasibility") or {}
    rate = feasibility_value.get("neutral_measured_valid_rate")
    shape = (payload.get("non_answer_character") or {}).get("base_plain") or {}
    lines = ["**Reading.**"]
    if not feasibility_value.get("m1_estimable", True):
        lines.append(
            "- The denominator question is **not answered**: the base model produces a parseable "
            "`Answer: X` on only %s of neutral measured greedy trials, below v6's %.0f%% gate, so "
            "L2 and L3 are *not estimable* rather than negative. A base model that cannot be "
            "measured on M1 is not evidence that the signature is absent before instruction "
            "tuning; it is evidence that this instrument needs an instruction-followed format."
            % ("%.0f%%" % (rate * 100) if rate is not None else "n/a", FEASIBILITY_GATE * 100))
        if shape.get("n_measured"):
            lines.append(
                "- The non-answer channel is **flat**: the parseable rate is identical in every "
                "one of the eight factorial cells, so the failure tracks the item and the format, "
                "not the treatment. %d of %d measured responses were empty and the median measured "
                "response was %s tokens."
                % (shape.get("n_empty_response", 0), shape["n_measured"],
                   _cell(shape.get("median_length_tokens"))))
    reproduced = [key for key in ("H1", "H2a", "H2b")
                  if _negative_and_excludes_zero_dict((control.get("hypotheses") or {}).get(key))]
    missed = [key for key in ("H1", "H2a", "H2b") if key not in reproduced]
    if control.get("available"):
        if outcomes.get("L4") == SUPPORTED:
            lines.append(
                "- L4 is **supported**: H1, H2a and H2b all reproduce under the plain template, so "
                "the signature is not an artefact of Gemma's chat markup.")
        else:
            lines.append(
                "- L4 is **not supported**, but only through %s: %s reproduce%s under the plain "
                "template with intervals excluding zero, while %s do%s not. The chat markup is "
                "therefore implicated in the %s contrast specifically, not in the signature as a "
                "whole - and every estimate for that contrast carries the caveat."
                % (", ".join(missed) or "no contrast", ", ".join(reproduced) or "no contrast",
                   "s" if len(reproduced) == 1 else "", ", ".join(missed) or "none",
                   "es" if len(missed) == 1 else "", ", ".join(missed) or "affected"))
    difference = payload["hostile_onset_distress_difference"]["result"]
    if outcomes.get("L5") == SUPPORTED and difference:
        lines.append(
            "- L5 is **supported**: judged distress at hostile onset is %.3f lower in the base "
            "model %s. The semantic distress channel is post-training-installed on this evidence; "
            "note that it is measured on the same responses whose answer format the base model "
            "does not follow."
            % (abs(difference["estimate"]),
               "[%.3f, %.3f]" % (difference["ci95_lower"], difference["ci95_upper"])
               if difference.get("ci95_lower") is not None else "(no CI)"))
    lines.append(
        "- Interpretation ceiling: provenance is a claim about where a behaviour was learned. "
        "None of this licenses a claim about experience.")
    return lines


def _negative_and_excludes_zero_dict(entry: Mapping[str, Any] | None) -> bool:
    return bool(entry and entry.get("estimate") is not None
                and entry.get("ci95_upper") is not None and entry["ci95_upper"] < 0.0)
