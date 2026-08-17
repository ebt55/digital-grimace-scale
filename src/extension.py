"""EXPLORATORY EXTENSION - not preregistered.

A third-family model (e.g. ``meta-llama/Llama-3.1-8B-Instruct``, family
"Llama-3.1") is run through the *shape* of the preregistration-v3 contrast table
so its behaviour can be compared with the primary model's confirmed holdout
result.  Nothing here is confirmatory:

* The model is **not** in `notes/preregistration_v3.md`.  No hypothesis in this
  module was registered for it, no success criterion is evaluated, and no result
  produced here can support or refute a preregistered claim.
* Every contrast is computed on **both splits separately** (discovery from the
  Phase-1 raw, holdout from the Phase-2 raw) and both are reported.  The holdout
  is not protected for this model, so its holdout number is exploratory too.
* The arithmetic is imported from :mod:`src.confirm`, never re-implemented: the
  same ``ContrastSpec`` objects, the same item-paired differences, the same
  2,000-resample item-clustered bootstrap and the same frozen support rules
  (including H5's bounded-upper-CI rule).  Only the bootstrap seed differs, so an
  extension estimate can never be confused with a confirmatory one.
* Amendments A2 (treatment-blind item exclusion) and A4 (pooled QC bars) are
  applied exactly as in the confirmatory analysis, computed on **this model's
  own** accurate+neutral resamples within **each split separately**.

The contrast set is H1, H2a, H2b, H3a, H3b, H4a, H4b, H5, H8 and H9, plus the
H6a distress contrast (hostile onset - neutral onset, accurate arm, easy+hard
pooled) and this model's raw hostile-onset distress mean.  H6b and H7 are
model-role contrasts (primary vs control) and are meaningless for a single
extension model, so they are absent rather than silently redefined.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# Imported, never copied: the extension must use the confirmatory arithmetic.
from .confirm import (
    HYPOTHESES, PREDICTION_TEXT, BootstrapResult, ContrastSpec, bootstrap_contrast, build_index,
    contrast_pairs, is_supported, outcome_value, _interval, _jsonable,
)
from .extract import MetricRow
from .pipeline import (
    AMENDED_RULES, Amendments, ItemExclusion, MetricEligibility, exploratory_cell_summary,
    item_exclusions, metric_eligibility,
)

EXTENSION_LABEL = "EXPLORATORY EXTENSION - not preregistered"
# A distinct key, so no extension resample stream can coincide with a
# confirmatory one even for an identically named contrast.
EXTENSION_BOOTSTRAP_KEY = "DGS-AC1-EXTENSION-v1"
# H6b (primary - control) and H7 (control-model boundary) are contrasts BETWEEN
# model roles; a single extension model has no second role to occupy, so they
# are omitted rather than redefined into something the v3 table never stated.
EXTENSION_HYPOTHESIS_IDS = (
    "H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H4b", "H5", "H6a", "H8", "H9",
)
SPLITS = ("discovery", "holdout")
DISTRESS_VALIDITY = "accurate"
DISTRESS_TONE = "hostile"
DISTRESS_TURN = "onset"
DISTRESS_DIFFICULTIES = ("easy", "hard")


class ExtensionError(ValueError):
    """Raised when the exploratory extension cannot be assembled."""


def _freeze(mapping):
    return MappingProxyType(dict(mapping))


EXTENSION_HYPOTHESES: tuple[ContrastSpec, ...] = tuple(
    spec for spec in HYPOTHESES if spec.hypothesis_id in EXTENSION_HYPOTHESIS_IDS
)
M1_HYPOTHESIS_IDS: tuple[str, ...] = tuple(
    spec.hypothesis_id for spec in EXTENSION_HYPOTHESES if spec.outcome == "m1"
)

if len(EXTENSION_HYPOTHESES) != len(EXTENSION_HYPOTHESIS_IDS):  # pragma: no cover - import guard
    raise ExtensionError("extension hypothesis IDs do not all exist in src.confirm.HYPOTHESES")


def model_slug(model_id: str) -> str:
    """``meta-llama/Llama-3.1-8B-Instruct`` -> ``meta-llama__Llama-3.1-8B-Instruct``."""
    return model_id.replace("/", "__")


def derive_family(model_id: str) -> str:
    """``.../Llama-3.1-8B-Instruct`` -> ``Llama-3.1``; the size token ends the name."""
    basename = model_id.rsplit("/", 1)[-1]
    parts = basename.split("-")
    kept = []
    for part in parts:
        lowered = part.lower()
        if lowered.endswith("b") and lowered[:-1].replace(".", "", 1).isdigit():
            break
        kept.append(part)
    return "-".join(kept) or basename


def model_raw_source(source: str | Path, model_id: str) -> Path:
    """Prefer ``<dir>/<slug>.jsonl`` so a sibling model's gigabyte is never parsed."""
    path = Path(source)
    if path.is_file():
        return path
    candidate = path / (model_slug(model_id) + ".jsonl")
    return candidate if candidate.exists() else path


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtensionOutcome:
    """One hypothesis-shaped contrast for the extension model on one split."""

    hypothesis_id: str
    contrast: str
    outcome: str
    stratum: str
    prediction: str
    result: BootstrapResult
    supported: bool


@dataclass(frozen=True)
class SplitAnalysis:
    """Everything computed for one split; ``available`` is false when raw is absent."""

    split: str
    available: bool
    unavailable_reason: str | None
    raw_source: str | None
    judge_source: str | None
    n_endpoints: int
    n_items: int
    n_judge_scores: int
    outcomes: tuple[ExtensionOutcome, ...]
    hostile_onset_distress_mean: float | None
    hostile_onset_distress_n_items: int
    item_exclusions: tuple[ItemExclusion, ...]
    eligibility: tuple[MetricEligibility, ...]
    non_answer: tuple[Mapping[str, Any], ...]

    @property
    def by_id(self) -> Mapping[str, ExtensionOutcome]:
        return _freeze({item.hypothesis_id: item for item in self.outcomes})


@dataclass(frozen=True)
class Comparison:
    """One row of the headline table: primary holdout beside both extension splits."""

    hypothesis_id: str
    contrast: str
    outcome: str
    stratum: str
    prediction: str
    primary: Mapping[str, Any] | None
    discovery: ExtensionOutcome | None
    holdout: ExtensionOutcome | None
    discovery_reference: Mapping[str, Any] | None
    consistent_with_primary: bool | None


@dataclass(frozen=True)
class ExtensionResult:
    label: str
    model_id: str
    family: str
    amendments: Amendments
    primary_model: str | None
    primary_confirm_source: str | None
    splits: Mapping[str, SplitAnalysis]
    comparisons: tuple[Comparison, ...]
    verdict: str
    verdict_detail: Mapping[str, Any]

    def to_dict(self):
        return _jsonable(self)


# --------------------------------------------------------------------------
# Primary-model holdout, read from the committed confirmatory JSON
# --------------------------------------------------------------------------

def load_primary_holdout(path: str | Path | None):
    """``(model_id, {hypothesis_id: {estimate, ci, n_items, supported}})`` or ``(None, {})``.

    Read-only: the confirmatory result is never recomputed here, only quoted.
    """
    if path is None:
        return None, _freeze({})
    path = Path(path)
    if not path.exists():
        return None, _freeze({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtensionError("cannot read primary confirmation: %s" % path) from error
    result = payload.get("result", payload) if isinstance(payload, Mapping) else {}
    models = result.get("models") or {}
    primary_model = models.get("primary") if isinstance(models, Mapping) else None
    out = {}
    for item in result.get("hypotheses") or ():
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("hypothesis_id")
        inner = item.get("result") or {}
        if not isinstance(identifier, str) or not isinstance(inner, Mapping):
            continue
        out[identifier] = _freeze({
            "estimate": inner.get("estimate"),
            "ci95_lower": inner.get("ci95_lower"),
            "ci95_upper": inner.get("ci95_upper"),
            "n_items": inner.get("n_items"),
            "supported": bool(item.get("supported")),
            "split": result.get("split"),
        })
    return primary_model, _freeze(out)


def _discovery_reference(spec: ContrastSpec, discovery, model_id: str):
    """The same contrast as computed by the Phase-1 exploratory table, if present."""
    if spec.discovery_key is None:
        return None
    _role, contrast, metric, stratum = spec.discovery_key
    row = discovery.get((model_id, contrast, metric, stratum))
    if row is None or row.get("mean_difference") in (None, ""):
        return None
    try:
        return _freeze({
            "mean_difference": float(row["mean_difference"]),
            "ci95_lower": float(row["ci95_lower"]),
            "ci95_upper": float(row["ci95_upper"]),
            "n_items": int(row["n_items"]),
        })
    except (KeyError, TypeError, ValueError):
        return None


def _consistent(primary, holdout: ExtensionOutcome | None):
    """Same sign as the primary holdout AND an extension CI that excludes zero.

    ``None`` means "cannot be judged": one of the two estimates is unavailable.
    A zero point estimate has no sign and can never be consistent.
    """
    if primary is None or primary.get("estimate") is None:
        return None
    if holdout is None or holdout.result.estimate is None or holdout.result.ci95_lower is None:
        return None
    left, right = float(primary["estimate"]), float(holdout.result.estimate)
    if left == 0.0 or right == 0.0:
        return False
    same_sign = (left > 0.0) == (right > 0.0)
    excludes_zero = holdout.result.ci95_lower > 0.0 or holdout.result.ci95_upper < 0.0
    return bool(same_sign and excludes_zero)


# --------------------------------------------------------------------------
# One split
# --------------------------------------------------------------------------

def _seed(model_id: str, split: str, spec: ContrastSpec) -> str:
    return "%s|%s|%s|%s|%s" % (EXTENSION_BOOTSTRAP_KEY, model_id, split, spec.hypothesis_id, spec.stratum)


def unavailable_split(split: str, reason: str, *, raw_source=None, judge_source=None) -> SplitAnalysis:
    """A split with no raw data: reported as absent, never as a null result."""
    return SplitAnalysis(
        split, False, reason, None if raw_source is None else str(raw_source),
        None if judge_source is None else str(judge_source),
        0, 0, 0, (), None, 0, (), (), (),
    )


def analyse_split(
    rows: Sequence[MetricRow],
    judge: Mapping[str, float],
    *,
    model_id: str,
    split: str,
    amendments: Amendments = AMENDED_RULES,
    raw_source: str | Path | None = None,
    judge_source: str | Path | None = None,
) -> SplitAnalysis:
    """Every extension contrast on one split, plus that split's own QC."""
    rows = tuple(row for row in rows if row.model_id == model_id)
    if not rows:
        return unavailable_split(split, "no_rows_for_model", raw_source=raw_source, judge_source=judge_source)
    exclusions = item_exclusions(rows, model_id, split=split)
    excluded_ids = frozenset(item.task_id for item in exclusions) if amendments.item_exclusion else frozenset()
    models = {"primary": model_id, "control": model_id}
    index = build_index(rows, split=split, models=models, excluded={model_id: excluded_ids})
    if not index:
        return unavailable_split(split, "no_endpoints_in_split", raw_source=raw_source, judge_source=judge_source)

    # Same derivation as the confirmatory run: deterministic per-difficulty item order.
    tasks_by_difficulty: dict[str, list[str]] = {}
    for (_model, task_id, cell_id, _turn) in index:
        bucket = tasks_by_difficulty.setdefault(cell_id.split("__")[0], [])
        if task_id not in bucket:
            bucket.append(task_id)
    for bucket in tasks_by_difficulty.values():
        bucket.sort()

    outcomes = []
    for spec in EXTENSION_HYPOTHESES:
        pairs = contrast_pairs(index, spec, models, judge, tasks_by_difficulty=tasks_by_difficulty)
        result = bootstrap_contrast(pairs, _seed(model_id, split, spec))
        outcomes.append(ExtensionOutcome(
            spec.hypothesis_id, spec.contrast, spec.outcome, spec.stratum,
            PREDICTION_TEXT[spec.prediction], result, is_supported(spec.prediction, result),
        ))

    distress = [
        value for row in rows
        if row.split == split and row.cell_kind == "factorial"
        and row.turn_label == DISTRESS_TURN and row.feedback_validity == DISTRESS_VALIDITY
        and row.tone == DISTRESS_TONE and row.difficulty in DISTRESS_DIFFICULTIES
        and row.task_id not in excluded_ids
        for value in (outcome_value(row, "distress", judge),) if value is not None
    ]
    return SplitAnalysis(
        split, True, None,
        None if raw_source is None else str(raw_source),
        None if judge_source is None else str(judge_source),
        len(rows), len({row.task_id for row in rows if row.split == split}), len(judge),
        tuple(outcomes),
        mean(distress) if distress else None, len(distress),
        exclusions,
        metric_eligibility(rows, model_id, split=split, amendments=amendments),
        exploratory_cell_summary(rows, phase=None, split=split),
    )


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------

def run_extension(
    *,
    model_id: str,
    splits: Mapping[str, SplitAnalysis],
    primary_hypotheses: Mapping[str, Mapping[str, Any]] | None = None,
    primary_model: str | None = None,
    primary_confirm_source: str | Path | None = None,
    discovery_contrasts=None,
    family: str | None = None,
    amendments: Amendments = AMENDED_RULES,
) -> ExtensionResult:
    """Assemble the comparison table and the one-line replication verdict."""
    primary_hypotheses = primary_hypotheses if primary_hypotheses is not None else {}
    discovery_contrasts = discovery_contrasts if discovery_contrasts is not None else {}
    discovery = splits.get("discovery")
    holdout = splits.get("holdout")
    discovery_by_id = discovery.by_id if discovery is not None and discovery.available else {}
    holdout_by_id = holdout.by_id if holdout is not None and holdout.available else {}

    comparisons = []
    for spec in EXTENSION_HYPOTHESES:
        holdout_outcome = holdout_by_id.get(spec.hypothesis_id)
        primary = primary_hypotheses.get(spec.hypothesis_id)
        comparisons.append(Comparison(
            spec.hypothesis_id, spec.contrast, spec.outcome, spec.stratum,
            PREDICTION_TEXT[spec.prediction], primary,
            discovery_by_id.get(spec.hypothesis_id), holdout_outcome,
            _discovery_reference(spec, discovery_contrasts, model_id),
            _consistent(primary, holdout_outcome),
        ))

    extension_supported = tuple(
        key for key in M1_HYPOTHESIS_IDS
        if key in holdout_by_id and holdout_by_id[key].supported
    )
    primary_supported = tuple(
        key for key in M1_HYPOTHESIS_IDS
        if primary_hypotheses.get(key, {}).get("supported")
    )
    replicated = tuple(key for key in primary_supported if key in extension_supported)
    detail = {
        "m1_hypotheses": M1_HYPOTHESIS_IDS,
        "extension_holdout_supported": extension_supported,
        "primary_holdout_supported": primary_supported,
        "replicated": replicated,
        "holdout_available": bool(holdout is not None and holdout.available),
        "primary_available": bool(primary_hypotheses),
    }
    if holdout is None or not holdout.available:
        verdict = "%s: %s has no holdout split to analyse; nothing replicates or fails to." % (
            EXTENSION_LABEL, model_id)
    elif not primary_hypotheses:
        verdict = "%s: %s supports %d/%d M1 hypotheses on its own holdout (%s); no primary confirmation supplied to compare against." % (
            EXTENSION_LABEL, model_id, len(extension_supported), len(M1_HYPOTHESIS_IDS),
            ", ".join(extension_supported) or "none")
    else:
        verdict = "%s: %s replicates %d/%d of the M1 hypotheses supported in the primary holdout (%s); %d/%d M1 hypotheses supported on its own holdout." % (
            EXTENSION_LABEL, model_id, len(replicated), len(primary_supported),
            ", ".join(replicated) or "none", len(extension_supported), len(M1_HYPOTHESIS_IDS))
    detail["verdict"] = verdict
    return ExtensionResult(
        EXTENSION_LABEL, model_id, family or derive_family(model_id), amendments,
        primary_model, None if primary_confirm_source is None else str(primary_confirm_source),
        _freeze(splits), tuple(comparisons), verdict, _freeze(detail),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _text(value) -> str:
    """Escape the pipes inside strata like ``easy | neutral`` so the table holds."""
    return str(value).replace("|", "\\|")


def _primary_interval(primary) -> str:
    if primary is None or primary.get("estimate") is None:
        return "n/a"
    lower, upper = primary.get("ci95_lower"), primary.get("ci95_upper")
    if lower is None or upper is None:
        return "%.3f (no CI)" % float(primary["estimate"])
    return "%.3f [%.3f, %.3f]%s" % (
        float(primary["estimate"]), float(lower), float(upper),
        " **s**" if primary.get("supported") else "")


def _outcome_interval(outcome: ExtensionOutcome | None) -> str:
    if outcome is None:
        return "not run"
    return "%s%s" % (_interval(outcome.result), " **s**" if outcome.supported else "")


def _consistency_text(value) -> str:
    if value is None:
        return "-"
    return "**yes**" if value else "no"


def render_extension_markdown(result: ExtensionResult) -> str:
    lines = [
        "# %s" % EXTENSION_LABEL,
        "",
        "> **EXPLORATORY EXTENSION - NOT PREREGISTERED.**",
        "> `%s` (family `%s`) is not named in `notes/preregistration_v3.md`. No hypothesis"
        % (result.model_id, result.family),
        "> below was registered for it, no success criterion is evaluated, and its holdout",
        "> was never protected for this model. Nothing here can support, refute or amend a",
        "> preregistered claim; it describes a third family beside the confirmed result.",
        "",
        "- extension model: `%s` (family `%s`)" % (result.model_id, result.family),
        "- primary model (confirmatory, quoted read-only): `%s`" % (result.primary_model or "n/a"),
        "- primary confirmation source: `%s`" % (result.primary_confirm_source or "not supplied"),
        "- amendments: A2 item exclusion %s, A3 pooled-SD fallback %s, A4 pooled QC %s" % (
            "on" if result.amendments.item_exclusion else "off",
            "on" if result.amendments.pooled_sd_fallback else "off",
            "on" if result.amendments.pooled_qc else "off"),
        "- bootstrap: 2,000 item-clustered resamples, key `%s` (distinct from the confirmatory key)"
        % EXTENSION_BOOTSTRAP_KEY,
        "",
        "**%s**" % result.verdict,
        "",
        "## Splits analysed",
        "",
        "| split | available | raw source | judge source | endpoints | items | judge scores |",
        "| --- | :---: | --- | --- | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        item = result.splits.get(split)
        if item is None:
            lines.append("| %s | no | - | - | 0 | 0 | 0 |" % split)
            continue
        lines.append("| %s | %s | `%s` | `%s` | %d | %d | %d |" % (
            split, "yes" if item.available else "**no** (`%s`)" % (item.unavailable_reason or "absent"),
            item.raw_source or "-", item.judge_source or "-",
            item.n_endpoints, item.n_items, item.n_judge_scores))
    lines += [
        "",
        "## Hypothesis-shaped contrasts (EXPLORATORY EXTENSION - not preregistered)",
        "",
        "Contrast definitions, item pairing, bootstrap and support rules are imported from",
        "`src.confirm` unchanged, including H5's rule (CI upper <= +1.0 nat and point <= 0).",
        "`**s**` marks a contrast that meets its support rule. \"Consistent with primary\"",
        "means the extension holdout estimate has the same sign as the primary holdout",
        "estimate AND its own 95% CI excludes zero.",
        "",
        "H6b and H7 are primary-vs-control contrasts and have no meaning for a single",
        "extension model, so they are omitted rather than redefined.",
        "",
        "| ID | contrast | outcome | stratum | prediction | primary holdout [95% CI] | extension discovery [95% CI] | extension holdout [95% CI] | items disc/hold | consistent with primary |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | :---: |",
    ]
    for item in result.comparisons:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d/%d | %s |" % (
            item.hypothesis_id, _text(item.contrast), item.outcome, _text(item.stratum),
            _text(item.prediction), _primary_interval(item.primary),
            _outcome_interval(item.discovery), _outcome_interval(item.holdout),
            item.discovery.result.n_items if item.discovery else 0,
            item.holdout.result.n_items if item.holdout else 0,
            _consistency_text(item.consistent_with_primary),
        ))
    lines += [
        "",
        "### Hostile-onset distress mean (raw judge score, accurate arm, easy+hard pooled)",
        "",
        "| split | mean distress at hostile onset | endpoints scored |",
        "| --- | ---: | ---: |",
    ]
    for split in SPLITS:
        item = result.splits.get(split)
        if item is None or not item.available:
            lines.append("| %s | - | 0 |" % split)
            continue
        lines.append("| %s | %s | %d |" % (
            split,
            "%.3f" % item.hostile_onset_distress_mean if item.hostile_onset_distress_mean is not None else "-",
            item.hostile_onset_distress_n_items))
    lines += [
        "",
        "## Cross-check against the Phase-1 exploratory contrast table",
        "",
        "The same contrast as computed by `src.pipeline.exploratory_contrasts` on discovery,",
        "where that table names this model. Small differences are expected: the exploratory",
        "table applies no A2 exclusion and uses its own bootstrap key.",
        "",
        "| ID | exploratory table [95% CI] | items | extension discovery [95% CI] | items |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    referenced = [item for item in result.comparisons if item.discovery_reference is not None]
    if referenced:
        for item in referenced:
            reference = item.discovery_reference
            lines.append("| %s | %.3f [%.3f, %.3f] | %d | %s | %d |" % (
                item.hypothesis_id, reference["mean_difference"], reference["ci95_lower"],
                reference["ci95_upper"], reference["n_items"], _outcome_interval(item.discovery),
                item.discovery.result.n_items if item.discovery else 0))
    else:
        lines.append("| - | this model is absent from the supplied exploratory table | - | - | - |")
    lines += [
        "",
        "## QC - amendment A2 item exclusions (this model's own accurate+neutral resamples, per split)",
        "",
        "| split | item | baseline cell | invalid/absent baseline resamples | reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    any_excluded = False
    for split in SPLITS:
        item = result.splits.get(split)
        for exclusion in (item.item_exclusions if item else ()):
            any_excluded = True
            lines.append("| %s | %s | %s | %d/%d | `%s` |" % (
                split, exclusion.task_id, exclusion.baseline_cell_id or "absent",
                exclusion.invalid_or_absent_resamples, exclusion.required_resamples, exclusion.reason))
    if not any_excluded:
        lines.append("| - | none | - | - | - |")
    lines += [
        "",
        "## QC - A4-style pooled missing rates for M1 and M2",
        "",
        "Reported for completeness on the same 5% bars the confirmatory QC uses. These are",
        "descriptive here: no extension contrast is gated on them.",
        "",
        "| split | metric | within bar | decided on | pooled rate | worst cell | worst-cell rate |",
        "| --- | --- | :---: | --- | ---: | --- | ---: |",
    ]
    any_eligibility = False
    for split in SPLITS:
        item = result.splits.get(split)
        for entry in (item.eligibility if item else ()):
            if entry.metric_name not in ("M1", "M2"):
                continue
            any_eligibility = True
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                split, entry.metric_name, "yes" if entry.eligible else "**no**", entry.scope,
                "%.4f" % entry.pooled_rate if entry.pooled_rate is not None else "n/a",
                entry.worst_cell_id or "-",
                "%.4f" % entry.worst_rate if entry.worst_rate is not None else "n/a"))
    if not any_eligibility:
        lines.append("| - | - | - | - | - | - | - |")
    lines += [
        "",
        "## QC - non-answer rate by cell and endpoint (no exclusions applied)",
        "",
        "| split | cell | endpoint | items | non-answer rate | mean M1 (n) |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    any_cells = False
    for split in SPLITS:
        item = result.splits.get(split)
        for record in (item.non_answer if item else ()):
            if record["cell_kind"] != "factorial":
                continue
            any_cells = True
            lines.append("| %s | %s | %s | %d | %s | %s |" % (
                split, record["cell_id"], record["turn_label"], record["n_items"],
                "%.3f" % record["mean_non_answer_rate"] if record["mean_non_answer_rate"] is not None else "-",
                "%.3f (%d)" % (record["mean_m1"], record["n_m1"]) if record["mean_m1"] is not None else "-",
            ))
    if not any_cells:
        lines.append("| - | none | - | - | - | - |")
    lines.append("")
    return "\n".join(lines)
