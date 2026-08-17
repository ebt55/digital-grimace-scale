"""Phase 3 (j-space): activation extraction, probes, steering and the J1-J6 report.

    .venv\\Scripts\\python.exe scripts/run_phase3.py extract
    .venv\\Scripts\\python.exe scripts/run_phase3.py probe
    .venv\\Scripts\\python.exe scripts/run_phase3.py steer --judge
    .venv\\Scripts\\python.exe scripts/run_phase3.py report

`notes/preregistration_v4_phase3.md` is implemented literally; the statistics live in
`src/probe.py` and the model lives behind `src/jspace_client.py` (agent J1).  This script
owns only the item sets, the remote calls, resumption and the write-ups.

Item sets, stated because the preregistration's wording rounds them up.  The factorial has
eight cells but a task's difficulty fixes half of them, so each of a split's 20 tasks
appears in exactly 4 cells: **80 measured-position transcripts per split**, not 160.  The
counts are asserted at extraction time and recorded in the summaries.

Nothing here writes `results/raw`, `results/summaries/phase1`, `results/summaries/phase2`
or `manifest.json`.  Phase-3 outputs live in `results/jspace` and
`results/summaries/phase3`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.confirm import BootstrapResult, bootstrap_contrast  # noqa: E402
from src.extract import read_metric_rows  # noqa: E402
from src.probe import (  # noqa: E402
    BASELINE_DIRECTION_ID, RANDOM_DIRECTION_IDS, STEER_CELL_ID, STYLE_NEGATIVE, STYLE_POSITIVE,
    READOUT_SEED, TONE_DIRECTION_ID, TONE_NEGATIVE, TONE_POSITIVE, UNRELATED_DIRECTION_ID,
    VALIDITY_NEGATIVE, VALIDITY_POSITIVE, ActivationSet, DoseReadout, ProbeError, binary_labels,
    cell_demeaned_spearman, choose_layer, dose_readout, fit_probe, item_readout,
    load_activation_set, loo_auc_by_layer, mean_difference_direction, monotone_in_alpha,
    random_unit_directions, roc_auc, save_activation_set, scaled_direction, verdict_j1,
    verdict_j2, verdict_j3, verdict_j4, verdict_j5, verdict_j6,
)
from src.protocol import load_protocol, render_task  # noqa: E402

MODEL_ID = "google/gemma-2-9b-it"
MODEL_FILE = "google__gemma-2-9b-it.jsonl"
RAW_SOURCES = {
    "discovery": Path("results/raw/phase1") / MODEL_FILE,
    "holdout": Path("results/raw/phase2") / MODEL_FILE,
    "style": Path("results/raw/style_smoke") / MODEL_FILE,
}
JSPACE_DIR = Path("results/jspace")
SUMMARY_DIR = Path("results/summaries/phase3")
STEERING_OUTPUTS = "steering_outputs.jsonl"
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
STEER_MAX_NEW_TOKENS = 512
EXPECTED_MEASURED_ITEMS = 80  # 20 tasks x the 4 cells matching each task's difficulty
EXPECTED_STYLE_ITEMS = 10     # 5 tasks x {verbose, neutral_reference}
INTERPRETATION_CEILING = (
    "A probe plus induction result demonstrates a condition-linked internal variable with "
    "causal influence on the output signature. It is not evidence of experience."
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _fail(message: str) -> "Any":
    raise SystemExit("run_phase3: %s" % message)


def _jspace_client():
    """Import agent J1's client lazily so every offline path stays runnable without it."""
    try:
        from src import jspace_client  # noqa: PLC0415
    except ImportError as error:
        _fail("src/jspace_client.py is not importable yet (%s). Extraction and steering need "
              "the deployed Modal app; the probe/report paths run offline." % error)
    return jspace_client


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _read_json(path: Path) -> Any:
    if not path.is_file():
        _fail("required input is missing: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else ("%.*f" % (digits, value))


def _detail(detail: Mapping[str, Any]) -> str:
    """Render a verdict's detail mapping without float noise like ``0.05000000000000001``."""
    def show(value: Any) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, float):
            return "%.4g" % value
        if isinstance(value, (list, tuple)):
            if len(value) > 6 and all(isinstance(item, int) and not isinstance(item, bool)
                                      for item in value):
                return _layer_span(value)  # a long layer plateau, not a readable list
            return "[%s]" % ", ".join(show(item) for item in value)
        return str(value)

    return "; ".join("%s = %s" % (key, show(detail[key])) for key in sorted(detail))


def _interval(result: BootstrapResult, digits: int = 3) -> str:
    if result.estimate is None:
        return "unavailable (`%s`)" % (result.unavailable_reason or "no_data")
    if result.ci95_lower is None:
        return "%s (no CI: `%s`)" % (_number(result.estimate, digits), result.unavailable_reason or "")
    return "%s [%s, %s]" % (_number(result.estimate, digits), _number(result.ci95_lower, digits),
                            _number(result.ci95_upper, digits))


# --------------------------------------------------------------------------------------
# Item sets
# --------------------------------------------------------------------------------------

def _stream_raw(path: Path, *, prefilters: Sequence[tuple[str, str]]) -> Iterable[dict]:
    """Stream a multi-hundred-megabyte raw JSONL, decoding only candidate lines.

    The prefilter is a cheap substring test on the serialised line: records carry twenty
    logprobs per token, so decoding every line to discard most of them is both slow and
    memory-hostile.  Each ``(key, value_prefix)`` pair is matched in the compact form the
    repository writes *and* in the whitespace-padded form ``json.dumps`` produces by
    default, so the filter never silently drops a legitimately formatted file.  Lines are
    split on ``\\n`` only, per the frozen JSONL rule.  Every surviving line is still fully
    decoded and re-checked against the real fields below.
    """
    if not path.is_file():
        _fail("raw source is missing: %s" % path)
    groups = tuple(('"%s":%s' % pair, '"%s": %s' % pair) for pair in prefilters)
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if any(not any(variant in line for variant in group) for group in groups):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                _fail("%s:%d: invalid raw JSON: %s" % (path, number, error))


def measured_items(path: Path, protocol, *, split: str) -> list[dict[str, Any]]:
    """Measured-position greedy transcripts: the exact prompt before the measured response."""
    factorial = set(protocol.factorial_cell_ids)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for value in _stream_raw(path, prefilters=(("sample_index", "0"),
                                               ("turn_label", '"measured"'))):
        if (value.get("trajectory_kind") != "greedy" or value.get("sample_index") != 0
                or value.get("turn_label") != "measured" or value.get("cell_id") not in factorial):
            continue
        if value.get("model_id") != MODEL_ID or value.get("split") != split:
            continue
        difficulty, validity, tone = value["cell_id"].split("__")
        key = (value["task_id"], value["cell_id"])
        if key in out:
            _fail("duplicate measured endpoint %s in %s" % (key, path))
        out[key] = {
            "id": "%s|%s" % key,
            "messages": [{"role": message["role"], "content": message["content"]}
                         for message in value["messages"]],
            "task_id": value["task_id"], "cell_id": value["cell_id"],
            "difficulty": difficulty, "validity": validity, "tone": tone,
            "split": value["split"], "prompt_sha256": value["prompt_sha256"],
            "response_id": value["response_id"],
        }
    return [out[key] for key in sorted(out)]


def style_items(path: Path, protocol) -> list[dict[str, Any]]:
    """The five verbose / neutral-reference style pairs used for the unrelated direction."""
    del protocol
    wanted = (STYLE_POSITIVE, STYLE_NEGATIVE)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for value in _stream_raw(path, prefilters=(("sample_index", "0"), ("cell_id", '"style__'))):
        if (value.get("trajectory_kind") != "greedy" or value.get("sample_index") != 0
                or value.get("cell_id") not in wanted or value.get("model_id") != MODEL_ID):
            continue
        key = (value["task_id"], value["cell_id"])
        if key in out:
            _fail("duplicate style endpoint %s in %s" % (key, path))
        out[key] = {
            "id": "%s|%s" % key,
            "messages": [{"role": message["role"], "content": message["content"]}
                         for message in value["messages"]],
            "task_id": value["task_id"], "cell_id": value["cell_id"],
            "split": value.get("split") or "discovery",
            "prompt_sha256": value["prompt_sha256"], "response_id": value["response_id"],
        }
    return [out[key] for key in sorted(out)]


LABEL_COLUMNS = {
    "discovery": ("task_id", "cell_id", "difficulty", "validity", "tone", "split", "prompt_sha256"),
    "holdout": ("task_id", "cell_id", "difficulty", "validity", "tone", "split", "prompt_sha256"),
    "style": ("task_id", "cell_id", "split", "prompt_sha256"),
}


def _activation_set_from_response(items: Sequence[Mapping[str, Any]], response: Mapping[str, Any],
                                  columns: Sequence[str]) -> ActivationSet:
    ids = [str(item) for item in response["ids"]]
    requested = [item["id"] for item in items]
    if ids != requested:
        if sorted(ids) != sorted(requested):
            _fail("extract_activations returned a different item set than was requested")
        order = {identity: index for index, identity in enumerate(ids)}
        index = np.array([order[identity] for identity in requested], dtype=int)
        activations = np.asarray(response["activations"])[index]
        ids = requested
    else:
        activations = np.asarray(response["activations"])
    norms = response.get("norms")
    layers = [int(item) for item in response["layers"]]
    if norms is None:
        norms = [float(np.linalg.norm(activations[:, position, :].astype(np.float64), axis=1).mean())
                 for position in range(len(layers))]
    return ActivationSet(
        tuple(ids), tuple(layers), activations, tuple(float(item) for item in norms),
        {name: tuple(str(item[name]) for item in items) for name in columns},
    )


def command_extract(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    jspace_dir = ROOT / args.jspace_dir
    builders = {
        "discovery": lambda: measured_items(ROOT / RAW_SOURCES["discovery"], protocol, split="discovery"),
        "holdout": lambda: measured_items(ROOT / RAW_SOURCES["holdout"], protocol, split="holdout"),
        "style": lambda: style_items(ROOT / RAW_SOURCES["style"], protocol),
    }
    expected = {"discovery": EXPECTED_MEASURED_ITEMS, "holdout": EXPECTED_MEASURED_ITEMS,
                "style": EXPECTED_STYLE_ITEMS}
    names = args.sets or ("discovery", "holdout", "style")
    layers = [int(item) for item in args.layers.split(",")] if args.layers else None
    pending = []
    for name in names:
        target = jspace_dir / ("activations_%s.npz" % name)
        if target.exists() and not args.force:
            print("run_phase3: %s already extracted -> %s (use --force to redo)" % (name, target))
            continue
        pending.append((name, target))
    # Resolve the client before scanning gigabytes of raw JSONL, so a missing deployment
    # fails in a second instead of after three full passes over the raw files.
    client = _jspace_client() if pending else None
    # One handle for the whole run: the container starts on the first remote call and every
    # later chunk reuses it instead of resolving the deployed class again.
    handle = client.get_cls() if pending else None
    extra: dict[str, Any] = {}
    if args.batch_size is not None:
        extra["batch_size"] = int(args.batch_size)
    if args.chunk is not None:
        extra["chunk"] = int(args.chunk)
    written = []
    for name, target in pending:
        items = builders[name]()
        if len(items) != expected[name]:
            _fail("expected %d %s items, built %d (see the item-count note in this script)"
                  % (expected[name], name, len(items)))
        _write_json(jspace_dir / ("items_%s.json" % name),
                    {"schema_version": "dgs-jspace-items-v1", "set": name,
                     "model_id": MODEL_ID, "generated_at": _now(), "n_items": len(items),
                     "items": [{key: value for key, value in item.items() if key != "messages"}
                               for item in items]})
        response = client.extract_activations(
            [{"id": item["id"], "messages": item["messages"]} for item in items], layers,
            handle=handle, **extra)
        activation_set = _activation_set_from_response(items, response, LABEL_COLUMNS[name])
        save_activation_set(target, activation_set)
        written.append(target)
        print("run_phase3: extracted %d %s items x %d layers (hidden %d) -> %s"
              % (activation_set.n_items, name, len(activation_set.layers),
                 activation_set.hidden, target))
    if not written:
        print("run_phase3: nothing to extract")
    return 0


# --------------------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------------------

def _measured_m1(path: Path, *, split: str) -> dict[tuple[str, str], float]:
    """Available-case measured M1 per (task, cell) from a committed metric table."""
    if not path.is_file():
        _fail("metric table is missing: %s" % path)
    out: dict[tuple[str, str], float] = {}
    for row in read_metric_rows(path):
        if (row.model_id != MODEL_ID or row.split != split or row.turn_label != "measured"
                or row.cell_kind != "factorial" or row.m1 is None):
            continue
        out[(row.task_id, row.cell_id)] = float(row.m1)
    return out


def _layer_table(rows) -> list[dict[str, Any]]:
    return [{"layer": item.layer, "auc": item.auc, "n_items": item.n_items,
             "n_groups": item.n_groups, "unavailable_reason": item.unavailable_reason}
            for item in rows]


def command_probe(args: argparse.Namespace) -> int:
    jspace_dir = ROOT / args.jspace_dir
    out_dir = ROOT / args.out
    target = out_dir / "localization.json"
    if target.exists() and not args.force:
        _fail("%s already exists. The holdout is evaluated ONCE; pass --force only if you are "
              "deliberately re-running a computation that has not yet been reported." % target)
    discovery = load_activation_set(jspace_dir / "activations_discovery.npz")
    holdout = load_activation_set(jspace_dir / "activations_holdout.npz")
    if discovery.layers != holdout.layers:
        _fail("discovery and holdout were extracted at different layers")

    tone_layers = loo_auc_by_layer(discovery, label_name="tone", positive=TONE_POSITIVE,
                                   negative=TONE_NEGATIVE)
    validity_layers = loo_auc_by_layer(discovery, label_name="validity",
                                       positive=VALIDITY_POSITIVE, negative=VALIDITY_NEGATIVE)
    chosen = choose_layer(tone_layers)

    tone_labels = binary_labels(discovery.column("tone"), TONE_POSITIVE, TONE_NEGATIVE)
    validity_labels = binary_labels(discovery.column("validity"), VALIDITY_POSITIVE, VALIDITY_NEGATIVE)
    tone_probe = fit_probe(discovery.matrix(chosen), tone_labels, layer=chosen,
                           label_name="tone", positive_label=TONE_POSITIVE)
    validity_probe = fit_probe(discovery.matrix(chosen), validity_labels, layer=chosen,
                               label_name="validity", positive_label=VALIDITY_POSITIVE)

    holdout_features = holdout.matrix(chosen)
    holdout_tone_scores = tone_probe.score(holdout_features)
    holdout_tone_auc = roc_auc(binary_labels(holdout.column("tone"), TONE_POSITIVE, TONE_NEGATIVE),
                               holdout_tone_scores)
    holdout_validity_auc = roc_auc(
        binary_labels(holdout.column("validity"), VALIDITY_POSITIVE, VALIDITY_NEGATIVE),
        validity_probe.score(holdout_features))

    m1 = _measured_m1(ROOT / args.holdout_metrics, split="holdout")
    rows = []
    missing = 0
    for index, identity in enumerate(holdout.ids):
        key = (holdout.column("task_id")[index], holdout.column("cell_id")[index])
        value = m1.get(key)
        if value is None:
            missing += 1
            continue
        rows.append((key[0], key[1], float(holdout_tone_scores[index]), value))
        del identity
    correlation = cell_demeaned_spearman(rows)

    discovery_by_layer = {item.layer: item.auc for item in tone_layers}
    validity_by_layer = {item.layer: item.auc for item in validity_layers}
    verdicts = [
        verdict_j1(tone_layers, chosen, holdout_tone_auc),
        verdict_j2(holdout_tone_auc, holdout_validity_auc, chosen, basis="holdout"),
        verdict_j3(correlation),
    ]
    payload = {
        "schema_version": "dgs-phase3-localization-v1",
        "generated_at": _now(),
        "model_id": MODEL_ID,
        "preregistration": "notes/preregistration_v4_phase3.md",
        "item_counts": {"discovery": discovery.n_items, "holdout": holdout.n_items,
                        "note": ("each task appears in the 4 cells matching its difficulty, so a "
                                 "split contributes 20 x 4 = 80 measured transcripts")},
        "layers_extracted": list(discovery.layers),
        "hidden_size": discovery.hidden,
        "chosen_layer": chosen,
        "layer_choice_rule": "argmax discovery leave-one-task-out tone AUC; ties to the lower layer",
        "discovery_loo_auc": {"tone": _layer_table(tone_layers),
                              "validity": _layer_table(validity_layers)},
        "discovery_loo_auc_at_chosen_layer": {"tone": discovery_by_layer.get(chosen),
                                              "validity": validity_by_layer.get(chosen)},
        "holdout_auc_at_chosen_layer": {"tone": holdout_tone_auc, "validity": holdout_validity_auc},
        "holdout_correlation": {
            "rho": correlation.rho, "ci95_lower": correlation.ci95_lower,
            "ci95_upper": correlation.ci95_upper, "n_items": correlation.n_items,
            "n_pairs": correlation.n_pairs, "n_cells": correlation.n_cells,
            "m1_missing_endpoints": missing,
            "unavailable_reason": correlation.unavailable_reason,
            "estimator": ("probe score and M1 demeaned within cell, residuals pooled; 2,000-"
                          "resample item-clustered percentile bootstrap re-demeaning inside "
                          "each resample"),
        },
        "layer_norms": {str(layer): discovery.norm(layer) for layer in discovery.layers},
        "verdicts": [item.to_dict() for item in verdicts],
        "interpretation_ceiling": INTERPRETATION_CEILING,
    }
    _write_json(target, payload)
    _write_text(out_dir / "localization.md", render_localization_markdown(payload))
    print("run_phase3: L* = %d; holdout tone AUC %s, validity AUC %s; rho %s -> %s"
          % (chosen, _number(holdout_tone_auc), _number(holdout_validity_auc),
             _number(correlation.rho), target))
    for item in verdicts:
        print("  %s %s" % (item.prediction_id, "SUPPORTED" if item.supported else "not supported"))
    return 0


def _layer_span(layers: Sequence[int]) -> str:
    """``[6, 7, ..., 25]`` -> ``6-25``; a broken run is listed as comma-separated runs."""
    ordered = sorted(set(int(layer) for layer in layers))
    if not ordered:
        return "none"
    runs, start, previous = [], ordered[0], ordered[0]
    for layer in ordered[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        runs.append((start, previous))
        start = previous = layer
    runs.append((start, previous))
    return ", ".join("%d" % low if low == high else "%d-%d" % (low, high) for low, high in runs)


def render_localization_markdown(payload: Mapping[str, Any]) -> str:
    chosen = payload["chosen_layer"]
    holdout = payload["holdout_auc_at_chosen_layer"]
    discovery = payload["discovery_loo_auc_at_chosen_layer"]
    correlation = payload["holdout_correlation"]
    tone_rows = {item["layer"]: item["auc"] for item in payload["discovery_loo_auc"]["tone"]}
    validity_rows = {item["layer"]: item["auc"] for item in payload["discovery_loo_auc"]["validity"]}
    available = [(layer, auc) for layer, auc in tone_rows.items() if auc is not None]
    peak = max(available, key=lambda pair: pair[1]) if available else (None, None)
    lines = [
        "# Phase 3 - localization (probe)",
        "",
        "Preregistration: `%s`. Model `%s`." % (payload["preregistration"], payload["model_id"]),
        "",
        "- items: %d discovery, %d holdout measured-position transcripts (%s)" % (
            payload["item_counts"]["discovery"], payload["item_counts"]["holdout"],
            payload["item_counts"]["note"]),
        "- layers extracted: %d (%d ... %d), hidden size %d" % (
            len(payload["layers_extracted"]), min(payload["layers_extracted"]),
            max(payload["layers_extracted"]), payload["hidden_size"]),
        "- probe: L2 logistic (C = 1), standardised **in the training fold only**, "
        "leave-one-task-out (all cells of a task held out together)",
        "- layer choice: %s -> **L\\* = %d**" % (payload["layer_choice_rule"], chosen),
        "- discovery peak tone AUC: %s, attained at layer(s) %s" % (
            _number(peak[1]), _layer_span([layer for layer, auc in available
                                           if peak[1] is not None and auc >= peak[1] - 1e-12])),
        "",]
    if len([layer for layer, auc in available if peak[1] is not None and auc >= peak[1] - 1e-12]) > 1:
        lines += [
            "The tone AUC is tied at its maximum across a plateau of layers, so the frozen "
            "\"ties to the lower layer\" rule -- which exists to pick ONE layer to steer at, not "
            "to decide a hypothesis -- is what fixes L\\*. J1's band clause is therefore read as "
            "\"the peak is attained at some layer in 12-30\"; the stricter \"the tie-broken argmax "
            "index lies in 12-30\" reading is reported alongside it in the verdict detail.",
            "",
        ]
    lines += [
        "## AUC at the chosen layer",
        "",
        "| label | discovery LOO | holdout (evaluated once) |",
        "| --- | ---: | ---: |",
        "| tone (hostile vs neutral) | %s | %s |" % (_number(discovery["tone"]),
                                                     _number(holdout["tone"])),
        "| validity (malfunctioning vs accurate) | %s | %s |" % (_number(discovery["validity"]),
                                                                 _number(holdout["validity"])),
        "",
        "## Holdout tone-probe score vs measured M1, within cell",
        "",
        "%s" % correlation["estimator"],
        "",
        "| quantity | value |",
        "| --- | --- |",
        "| pooled Spearman rho | **%s** |" % _number(correlation["rho"]),
        "| 95%% item-bootstrap CI | [%s, %s] |" % (_number(correlation["ci95_lower"]),
                                                   _number(correlation["ci95_upper"])),
        "| items / pairs / cells | %d / %d / %d |" % (correlation["n_items"],
                                                      correlation["n_pairs"],
                                                      correlation["n_cells"]),
        "| endpoints with no available-case M1 | %d |" % correlation["m1_missing_endpoints"],
        "",
        "## Predictions",
        "",
        "| ID | prediction | verdict | detail |",
        "| --- | --- | :---: | --- |",
    ]
    for verdict in payload["verdicts"]:
        detail = _detail(verdict["detail"])
        lines.append("| %s | %s | %s | %s |" % (
            verdict["prediction_id"], verdict["statement"],
            "**supported**" if verdict["supported"] else "not supported", detail))
    lines += [
        "",
        "## Discovery leave-one-task-out AUC by layer",
        "",
        "| layer | tone | validity |",
        "| ---: | ---: | ---: |",
    ]
    for layer in sorted(tone_rows):
        lines.append("| %d | %s | %s |" % (layer, _number(tone_rows[layer]),
                                           _number(validity_rows.get(layer))))
    lines += ["", "> %s" % payload["interpretation_ceiling"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# steer
# --------------------------------------------------------------------------------------

def _steering_items(protocol) -> list[dict[str, Any]]:
    """The 20 holdout tasks as the neutral, no-feedback single-turn prompt."""
    tasks = sorted((task for task in protocol.matched_tasks if task.split == "holdout"),
                   key=lambda task: task.task_id)
    return [{"id": task.task_id,
             "messages": [{"role": "user", "content": render_task(task.prompt, task.options, protocol)}],
             "canonical_answer": task.canonical_answer, "difficulty": task.difficulty}
            for task in tasks]


def build_directions(discovery: ActivationSet, style: ActivationSet,
                     layer: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The tone direction, five matched-norm random controls and the unrelated direction.

    **Clarification C2 (2026-08-18).** The dose unit is the natural magnitude of the contrast
    itself: the tone dose is ``alpha * d`` with ``d = mean(hostile) - mean(neutral)`` at
    ``L*`` *unnormalised*, so ``alpha = 1`` moves a neutral state to the hostile mean. Every
    control is rescaled to the same norm ``alpha * ||d||``. The earlier unit (the layer's mean
    activation norm) was withdrawn after the infrastructure smoke showed a random direction at
    that scale already produces gibberish to the token cap at ``alpha = 2``.

    Each returned value is the **per-unit-dose** vector; the server multiplies it by ``alpha``.
    The second return value carries ``||d||``, the mean activation norm and their ratio, which
    C2 requires to be reported so a reader can convert between the two units.
    """
    tone = mean_difference_direction(
        discovery, layer, label_name="tone", positive=TONE_POSITIVE, negative=TONE_NEGATIVE,
        mask=discovery.mask(validity=VALIDITY_NEGATIVE))
    tone_norm = float(np.linalg.norm(tone))
    unrelated = mean_difference_direction(
        style, layer, label_name="cell_id", positive=STYLE_POSITIVE, negative=STYLE_NEGATIVE)
    directions = {TONE_DIRECTION_ID: np.asarray(tone, dtype=np.float64),
                  UNRELATED_DIRECTION_ID: scaled_direction(unrelated, 1.0, tone_norm)}
    for name, vector in zip(RANDOM_DIRECTION_IDS,
                            random_unit_directions(discovery.hidden, len(RANDOM_DIRECTION_IDS))):
        directions[name] = scaled_direction(vector, 1.0, tone_norm)
    mean_norm = discovery.norm(layer)
    meta = {
        "tone_direction_norm": tone_norm,
        "mean_activation_norm": mean_norm,
        "norm_ratio_d_over_mean_activation": (tone_norm / mean_norm) if mean_norm else None,
        "unrelated_direction_raw_norm": float(np.linalg.norm(unrelated)),
    }
    return directions, meta


def _load_steering_entries(path: Path, *, repair: bool = False) -> list[dict[str, Any]]:
    """Read the resumable steering log, tolerating one torn line at the very end.

    Generations are appended and flushed one at a time, so a hard kill can leave a partial
    final line.  That line is dropped with a warning and regenerated; a malformed line
    anywhere earlier is a real corruption and stops the run.  ``repair=True`` also rewrites
    the file without the torn fragment, so the next append cannot glue itself onto it.
    """
    if not path.is_file():
        return []
    raw = [line.strip() for line in path.read_text(encoding="utf-8").split("\n")]
    out, torn = [], False
    for number, line in enumerate(raw, 1):
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as error:
            if all(not item for item in raw[number:]):
                torn = True
                print("run_phase3: WARNING: dropping a torn final line in %s (it will be "
                      "regenerated)" % path, file=sys.stderr)
                continue
            _fail("%s:%d: invalid steering JSON: %s" % (path, number, error))
    if torn and repair:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in out:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False) + "\n")
    return out


def _append_steering_entries(path: Path, entries: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
            handle.flush()


def _dose_plan(alphas: Sequence[float], directions: Sequence[str]) -> list[tuple[str, float]]:
    """``alpha = 0`` is generated once, as the shared baseline for every direction."""
    plan: list[tuple[str, float]] = [(BASELINE_DIRECTION_ID, 0.0)]
    for direction_id in directions:
        for alpha in alphas:
            if float(alpha) == 0.0:
                continue
            plan.append((direction_id, float(alpha)))
    return plan


def generate_missing(path: Path, items: Sequence[Mapping[str, Any]],
                     directions: Mapping[str, np.ndarray], layer_of: Mapping[str, int] | int,
                     plan: Sequence[tuple[str, float]], *, dry_run: bool = False,
                     handle: Any = None, batch_size: int | None = None,
                     chunk: int | None = None) -> int:
    """Fill in every missing ``(direction, alpha, item)`` cell, appending as it goes.

    ``layer_of`` is either one layer for every direction (the confirmatory run) or a map from
    direction to layer (the exploratory sweep, which steers the same items at several layers).
    """
    present = {(str(entry.get("direction_id")), round(float(entry.get("alpha", 0.0)), 6),
                str(entry.get("id")))
               for entry in _load_steering_entries(path, repair=not dry_run)}
    by_id = {item["id"]: item for item in items}
    written = 0
    for direction_id, alpha in plan:
        layer = layer_of if isinstance(layer_of, int) else layer_of[direction_id]
        missing = [by_id[identity] for identity in sorted(by_id)
                   if (direction_id, round(alpha, 6), identity) not in present]
        if not missing:
            continue
        if dry_run:
            print("run_phase3: would generate %d item(s) for %s alpha=%g"
                  % (len(missing), direction_id, alpha))
            continue
        client = _jspace_client()
        vector = None if direction_id == BASELINE_DIRECTION_ID else directions[direction_id].tolist()
        extra: dict[str, Any] = {}
        if batch_size is not None:
            extra["batch_size"] = int(batch_size)
        if chunk is not None:
            extra["chunk"] = int(chunk)
        response = client.generate_steered(
            [{"id": item["id"], "messages": item["messages"]} for item in missing],
            layer, vector, [float(alpha)], max_new_tokens=STEER_MAX_NEW_TOKENS,
            handle=handle, **extra)
        stamped = []
        for entry in response:
            record = dict(entry)
            record.setdefault("id", None)
            record["direction_id"] = direction_id
            record["alpha"] = float(record.get("alpha", alpha))
            record["layer"] = int(layer)
            record["generated_at"] = _now()
            if record["id"] not in by_id:
                _fail("generate_steered returned an unknown item id %r" % (record["id"],))
            stamped.append(record)
        _append_steering_entries(path, stamped)
        written += len(stamped)
        print("run_phase3: %s alpha=%g -> %d generation(s)" % (direction_id, alpha, len(stamped)))
    return written


def _readouts(entries: Sequence[Mapping[str, Any]], canonical: Mapping[str, str]):
    grouped: dict[tuple[str, float], list] = {}
    for entry in entries:
        task_id = str(entry["id"])
        answer = canonical.get(task_id)
        if answer is None:
            _fail("steering entry references an unknown task %r" % (task_id,))
        key = (str(entry["direction_id"]), round(float(entry["alpha"]), 6))
        grouped.setdefault(key, []).append(item_readout(
            entry, direction_id=key[0], alpha=key[1], task_id=task_id,
            canonical_answer=answer, cell_id=STEER_CELL_ID))
    return grouped


def _dose_row(readout: DoseReadout) -> dict[str, Any]:
    def result(value: BootstrapResult) -> dict[str, Any]:
        return {"estimate": value.estimate, "ci95_lower": value.ci95_lower,
                "ci95_upper": value.ci95_upper, "p_two_sided": value.p_two_sided,
                "n_items": value.n_items, "unavailable_reason": value.unavailable_reason}

    return {"direction_id": readout.direction_id, "alpha": readout.alpha,
            "n_items": readout.n_items, "m1_mean": readout.m1_mean, "m1_n": readout.m1_n,
            "non_answer_rate": readout.non_answer_rate,
            "mean_length_tokens": readout.mean_length_tokens, "degenerate": readout.degenerate,
            "m1_delta": result(readout.m1_delta),
            "non_answer_delta": result(readout.non_answer_delta),
            "length_delta": result(readout.length_delta)}


def command_steer(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    jspace_dir = ROOT / args.jspace_dir
    out_dir = ROOT / args.out
    localization = _read_json(out_dir / "localization.json")
    layer = int(args.layer) if args.layer is not None else int(localization["chosen_layer"])
    discovery = load_activation_set(jspace_dir / "activations_discovery.npz")
    style = load_activation_set(jspace_dir / "activations_style.npz")
    directions, direction_meta = build_directions(discovery, style, layer)
    print("run_phase3: C2 dose unit ||d|| = %.2f at layer %d; mean activation norm %.2f; "
          "ratio %.3f" % (direction_meta["tone_direction_norm"], layer,
                          direction_meta["mean_activation_norm"],
                          direction_meta["norm_ratio_d_over_mean_activation"] or float("nan")))
    items = _steering_items(protocol)
    canonical = {item["id"]: item["canonical_answer"] for item in items}
    alphas = tuple(float(item) for item in args.alphas.split(",")) if args.alphas else ALPHAS
    path = jspace_dir / STEERING_OUTPUTS
    handle = None
    if not args.dry_run:
        handle = _jspace_client().get_cls()
    generate_missing(path, items, directions, layer, _dose_plan(alphas, sorted(directions)),
                     dry_run=args.dry_run, handle=handle, batch_size=args.batch_size,
                     chunk=args.chunk)
    entries = _load_steering_entries(path)
    if not entries:
        print("run_phase3: no steered generations present yet; nothing to summarise")
        return 0

    grouped = _readouts(entries, canonical)
    baseline = grouped.get((BASELINE_DIRECTION_ID, 0.0), [])
    readouts = {key: dose_readout(value, baseline, direction_id=key[0], alpha=key[1])
                for key, value in sorted(grouped.items())}
    tone_doses = [value for key, value in sorted(readouts.items()) if key[0] == TONE_DIRECTION_ID]
    control_ids = set(RANDOM_DIRECTION_IDS) | {UNRELATED_DIRECTION_ID}
    control_doses = [value for key, value in sorted(readouts.items()) if key[0] in control_ids]

    distress_scores: list[dict[str, Any]] = []
    distress_deltas: dict[str, dict[str, Any]] = {}
    tone_distress: dict[float, BootstrapResult] = {}
    if args.judge:
        distress_scores, distress_deltas, tone_distress = _judge_steering(
            entries, out_dir, workers=args.workers, provider=args.provider, model=args.model,
            base_url=args.base_url)

    verdicts = [verdict_j4(tone_doses), verdict_j5(control_doses),
                verdict_j6(tone_doses, tone_distress if args.judge else None)]
    monotone, used, note = monotone_in_alpha(tone_doses)
    payload = {
        "schema_version": "dgs-phase3-steering-v1",
        "generated_at": _now(),
        "model_id": MODEL_ID,
        "preregistration": "notes/preregistration_v4_phase3.md",
        "layer": layer,
        "layer_norm": discovery.norm(layer),
        "alphas": list(alphas),
        "direction_ids": sorted(directions),
        "clarification": "C2 (2026-08-18): the dose unit is ||d|| itself, not the mean activation norm",
        "direction_norms": direction_meta,
        "direction_construction": {
            "tone": "mean(hostile) - mean(neutral) at L*, discovery accurate arm, measured position",
            "random": "5 unit gaussian directions, seeds %s|1..5, rescaled to ||d||" % "DGS-AC1-STEER-v1",
            "unrelated": "mean(style__verbose) - mean(style__neutral_reference) at L*, rescaled to ||d||",
            "dose": "alpha * d (d unnormalised); every control at the matched norm alpha * ||d||",
        },
        "items": {"n": len(items), "prompt": "render_task, neutral single-turn, holdout split"},
        "max_new_tokens": STEER_MAX_NEW_TOKENS,
        "degenerate_dose_rule": "a dose is degenerate when > 50% of items yield no parseable answer",
        "monotonicity": {"monotone": monotone, "doses_used": list(used), "note": note},
        "doses": [_dose_row(value) for _, value in sorted(readouts.items())],
        "distress": {"judged": bool(args.judge), "scores": distress_scores,
                     "deltas": distress_deltas},
        "verdicts": [item.to_dict() for item in verdicts],
        "interpretation_ceiling": INTERPRETATION_CEILING,
    }
    _write_json(out_dir / "steering.json", payload)
    _write_text(out_dir / "steering.md", render_steering_markdown(payload))
    print("run_phase3: %d dose cell(s) summarised -> %s" % (len(readouts), out_dir / "steering.json"))
    for item in verdicts:
        print("  %s %s" % (item.prediction_id, "SUPPORTED" if item.supported else "not supported"))
    return 0


def _judge_steering(entries, out_dir: Path, *, workers: int, provider: str | None,
                    model: str | None, base_url: str | None,
                    selected: Sequence[Mapping[str, Any]] | None = None,
                    filename: str = "steering_judge.json",
                    tone_direction_id: str = TONE_DIRECTION_ID):
    """Score a dose plan with the locked rubric and bootstrap distress against alpha = 0.

    ``selected=None`` uses the preregistered J6 plan (tone at 0/2/4, every control at 2); the
    exploratory sweep passes its own, much smaller, selection.
    """
    from src.judge_client import JsonlJudgeCache, JudgeClientError, load_env_files, make_judge_backend
    from src.steer_readouts import (distress_by_dose, judge_steering_entries, resolve_judge_ids,
                                    select_for_judging)

    load_env_files(ROOT)
    protocol = load_protocol(ROOT)
    provider, model, deviations = resolve_judge_ids(protocol, provider, model)
    if base_url:
        deviations.append("--base-url %r supplied; the judge is served from a self-hosted "
                          "endpoint" % base_url)
    for deviation in deviations:
        print("run_phase3: WARNING: DEVIATION: %s" % deviation, file=sys.stderr)
    try:
        backend = make_judge_backend(provider, model, base_url=base_url)
    except JudgeClientError as error:
        _fail(str(error))
    plan_label = "tone alpha 0/2/4, every control alpha 2"
    if selected is None:
        selected = select_for_judging(entries)
    else:
        selected = list(selected)
        plan_label = "exploratory selection: %d entries" % len(selected)
    # The baseline is the shared alpha = 0 arm; it must be scored (or already cached) for the
    # paired distress deltas to exist at all.
    baseline_entries = [entry for entry in entries
                        if str(entry.get("direction_id")) == BASELINE_DIRECTION_ID]
    chosen = {id(entry) for entry in selected}
    selected = selected + [entry for entry in baseline_entries if id(entry) not in chosen]
    cache = JsonlJudgeCache(out_dir / "steering_judge_cache.jsonl")
    failures: list[dict[str, str]] = []
    scores = judge_steering_entries(
        selected, backend, protocol=protocol, cache=cache, workers=workers,
        on_error=lambda entry, error: failures.append(
            {"direction_id": str(entry.get("direction_id")), "alpha": str(entry.get("alpha")),
             "task_id": str(entry.get("id")), "error": "%s: %s" % (type(error).__name__, error)}))
    _write_json(out_dir / filename,
                {"schema_version": "dgs-steering-judge-v1", "generated_at": _now(),
                 "selected": len(selected), "scored": len(scores), "failures": failures,
                 "deviations": deviations,
                 "dose_plan": plan_label,
                 "backend": {"provider_id": getattr(backend, "provider_id", None),
                             "model_id": getattr(backend, "model_id", None),
                             "is_synthetic": bool(getattr(backend, "is_synthetic", False)),
                             "sampling_mode": getattr(backend, "sampling_mode", None)},
                 # Usage counts only the calls this process actually made: a fully cached
                 # re-run legitimately reports zero, which is why it is labelled per run.
                 "usage": getattr(backend, "usage", None),
                 "estimated_cost_usd": getattr(backend, "estimated_cost_usd", None),
                 "cache_path": str(cache.path),
                 "cache_hits": cache.hits, "cache_misses": cache.misses,
                 "scores": [item.to_dict() for item in scores]})
    by_dose = distress_by_dose(scores)
    baseline = by_dose.get((BASELINE_DIRECTION_ID, 0.0), {})
    deltas: dict[str, dict[str, Any]] = {}
    tone_distress: dict[float, BootstrapResult] = {}
    for (direction_id, alpha), values in sorted(by_dose.items()):
        if direction_id == BASELINE_DIRECTION_ID:
            continue
        pairs = [(task_id, values[task_id] - baseline[task_id])
                 for task_id in sorted(values) if task_id in baseline]
        result = bootstrap_contrast(pairs, "%s|distress|%s|alpha%g"
                                    % (READOUT_SEED, direction_id, alpha))
        deltas["%s|%g" % (direction_id, alpha)] = {
            "direction_id": direction_id, "alpha": alpha, "estimate": result.estimate,
            "ci95_lower": result.ci95_lower, "ci95_upper": result.ci95_upper,
            "n_items": result.n_items, "unavailable_reason": result.unavailable_reason}
        if direction_id == tone_direction_id:
            tone_distress[float(alpha)] = result
    return [item.to_dict() for item in scores], deltas, tone_distress


# --------------------------------------------------------------------------------------
# sweep -- EXPLORATORY layer sweep (does not touch J1-J6)
# --------------------------------------------------------------------------------------

SWEEP_LAYERS = (20, 30)
SWEEP_ALPHAS = (1.0, 2.0, 4.0)
SWEEP_CONTROL_ALPHA = 4.0
SWEEP_CONTROL_COUNT = 2
SWEEP_JUDGED_ALPHA = 4.0
SWEEP_LABEL = ("EXPLORATORY. Chosen after the confirmatory verdicts were fixed, because the "
               "frozen tie-break forced L* = 6 (the earliest layer of a 6-25 AUC plateau) where "
               "||d|| is only 4% of the mean activation norm. It changes no preregistered "
               "verdict and supports no hypothesis test.")


def sweep_direction_id(layer: int) -> str:
    return "tone_L%d" % int(layer)


def sweep_control_id(layer: int, index: int) -> str:
    return "random_L%d_%d" % (int(layer), int(index))


def build_sweep_directions(discovery: ActivationSet, layers: Sequence[int]):
    """Tone direction recomputed at each layer, C2 scaling, plus matched-norm controls.

    The control directions reuse the confirmatory run's frozen seeds ``DGS-AC1-STEER-v1|1..2``:
    a random unit vector is layer-agnostic, so only its scaling to ``||d_layer||`` changes, and
    reusing the seeds keeps the sweep comparable with the layer-6 controls.
    """
    directions: dict[str, np.ndarray] = {}
    layer_of: dict[str, int] = {}
    meta: dict[str, Any] = {}
    for layer in layers:
        tone = mean_difference_direction(
            discovery, layer, label_name="tone", positive=TONE_POSITIVE, negative=TONE_NEGATIVE,
            mask=discovery.mask(validity=VALIDITY_NEGATIVE))
        norm = float(np.linalg.norm(tone))
        mean_norm = discovery.norm(layer)
        name = sweep_direction_id(layer)
        directions[name] = np.asarray(tone, dtype=np.float64)
        layer_of[name] = int(layer)
        for index, vector in enumerate(
                random_unit_directions(discovery.hidden, SWEEP_CONTROL_COUNT), start=1):
            control = sweep_control_id(layer, index)
            directions[control] = scaled_direction(vector, 1.0, norm)
            layer_of[control] = int(layer)
        meta[str(layer)] = {
            "tone_direction_norm": norm, "mean_activation_norm": mean_norm,
            "norm_ratio_d_over_mean_activation": (norm / mean_norm) if mean_norm else None,
        }
    return directions, layer_of, meta


def sweep_plan(layers: Sequence[int], alphas: Sequence[float] = SWEEP_ALPHAS) -> list[tuple[str, float]]:
    """Tone at every dose, and the two controls at the single control dose, per layer."""
    plan: list[tuple[str, float]] = []
    for layer in layers:
        for alpha in alphas:
            plan.append((sweep_direction_id(layer), float(alpha)))
    for layer in layers:
        for index in range(1, SWEEP_CONTROL_COUNT + 1):
            plan.append((sweep_control_id(layer, index), float(SWEEP_CONTROL_ALPHA)))
    return plan


def command_sweep(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    jspace_dir = ROOT / args.jspace_dir
    out_dir = ROOT / args.out
    confirmatory = _read_json(out_dir / "steering.json")
    layers = tuple(int(item) for item in args.layers.split(",")) if args.layers else SWEEP_LAYERS
    discovery = load_activation_set(jspace_dir / "activations_discovery.npz")
    directions, layer_of, direction_meta = build_sweep_directions(discovery, layers)
    for layer in layers:
        info = direction_meta[str(layer)]
        print("run_phase3: layer %d: ||d|| = %.2f, mean activation norm %.2f, ratio %.4f"
              % (layer, info["tone_direction_norm"], info["mean_activation_norm"],
                 info["norm_ratio_d_over_mean_activation"]))
    items = _steering_items(protocol)
    canonical = {item["id"]: item["canonical_answer"] for item in items}
    path = jspace_dir / STEERING_OUTPUTS
    plan = sweep_plan(layers)
    handle = None if args.dry_run else _jspace_client().get_cls()
    generate_missing(path, items, directions, layer_of, plan, dry_run=args.dry_run,
                     handle=handle, batch_size=args.batch_size, chunk=args.chunk)
    entries = _load_steering_entries(path)
    wanted = {(direction_id, round(alpha, 6)) for direction_id, alpha in plan}
    relevant = [entry for entry in entries
                if str(entry["direction_id"]) == BASELINE_DIRECTION_ID
                or (str(entry["direction_id"]), round(float(entry["alpha"]), 6)) in wanted]
    grouped = _readouts(relevant, canonical)
    baseline = grouped.get((BASELINE_DIRECTION_ID, 0.0), [])
    if not baseline:
        _fail("the shared alpha = 0 baseline is missing; run `steer` first")
    readouts = {key: dose_readout(value, baseline, direction_id=key[0], alpha=key[1])
                for key, value in sorted(grouped.items()) if key[0] != BASELINE_DIRECTION_ID}

    distress_scores: list[dict[str, Any]] = []
    distress_deltas: dict[str, dict[str, Any]] = {}
    if args.judge:
        selected = [entry for entry in relevant
                    if str(entry["direction_id"]).startswith("tone_L")
                    and abs(float(entry["alpha"]) - SWEEP_JUDGED_ALPHA) < 1e-9]
        distress_scores, distress_deltas, _ = _judge_steering(
            relevant, out_dir, workers=args.workers, provider=args.provider, model=args.model,
            base_url=args.base_url, selected=selected,
            filename="steering_layer_sweep_judge_exploratory.json",
            tone_direction_id=sweep_direction_id(layers[0]))

    confirm_tone = [row for row in confirmatory["doses"] if row["direction_id"] == TONE_DIRECTION_ID]
    payload = {
        "schema_version": "dgs-phase3-layer-sweep-exploratory-v1",
        "status": "exploratory",
        "label": SWEEP_LABEL,
        "generated_at": _now(),
        "model_id": MODEL_ID,
        "confirmatory_layer": confirmatory["layer"],
        "confirmatory_direction_norms": confirmatory.get("direction_norms"),
        "sweep_layers": list(layers),
        "alphas": list(SWEEP_ALPHAS),
        "control_alpha": SWEEP_CONTROL_ALPHA,
        "control_seeds": "DGS-AC1-STEER-v1|1..%d (the confirmatory run's first two)" % SWEEP_CONTROL_COUNT,
        "direction_construction": {
            "tone": "mean(hostile) - mean(neutral) recomputed at each swept layer, discovery "
                    "accurate arm, measured position",
            "dose": "alpha * d (C2, d unnormalised); controls rescaled to alpha * ||d_layer||",
            "baseline": "the shared unsteered alpha = 0 arm from the confirmatory run "
                        "(no intervention, so it is layer-independent)",
        },
        "direction_norms": direction_meta,
        "items": {"n": len(items), "prompt": "render_task, neutral single-turn, holdout split"},
        "max_new_tokens": STEER_MAX_NEW_TOKENS,
        "doses": [_dose_row(value) for _, value in sorted(readouts.items())],
        "confirmatory_tone_doses": confirm_tone,
        "distress": {"judged": bool(args.judge), "judged_alpha": SWEEP_JUDGED_ALPHA,
                     "scores": distress_scores, "deltas": distress_deltas},
        "interpretation_ceiling": INTERPRETATION_CEILING,
    }
    _write_json(out_dir / "steering_layer_sweep_exploratory.json", payload)
    _write_text(out_dir / "steering_layer_sweep_exploratory.md", render_sweep_markdown(payload))
    print("run_phase3: exploratory sweep -> %s"
          % (out_dir / "steering_layer_sweep_exploratory.json"))
    for row in payload["doses"]:
        if row["direction_id"].startswith("tone_L"):
            print("  %s alpha=%g: dM1 %s [%s, %s], non-answer %s"
                  % (row["direction_id"], row["alpha"], _number(row["m1_delta"]["estimate"]),
                     _number(row["m1_delta"]["ci95_lower"]), _number(row["m1_delta"]["ci95_upper"]),
                     _number(row["non_answer_rate"], 2)))
    return 0


def render_sweep_markdown(payload: Mapping[str, Any]) -> str:
    def as_result(value: Mapping[str, Any]) -> BootstrapResult:
        return BootstrapResult(value["estimate"], value["ci95_lower"], value["ci95_upper"],
                               value.get("p_two_sided"), value["n_items"], value["n_items"],
                               value["unavailable_reason"])

    confirm_layer = payload["confirmatory_layer"]
    confirm_norms = payload.get("confirmatory_direction_norms") or {}
    lines = [
        "# Phase 3 - EXPLORATORY layer sweep (tone direction at layers %s)"
        % ", ".join(str(item) for item in payload["sweep_layers"]),
        "",
        "> **%s**" % payload["label"],
        "",
        "- dose: `%s`" % payload["direction_construction"]["dose"],
        "- tone direction: %s" % payload["direction_construction"]["tone"],
        "- baseline: %s" % payload["direction_construction"]["baseline"],
        "- controls: %s, at alpha = %g only" % (payload["control_seeds"], payload["control_alpha"]),
        "- items: %d holdout tasks, %s, greedy, %d new tokens" % (
            payload["items"]["n"], payload["items"]["prompt"], payload["max_new_tokens"]),
        "",
        "## Dose unit by layer",
        "",
        "| layer | ||d|| | mean activation norm | ratio |",
        "| ---: | ---: | ---: | ---: |",
        "| %d (confirmatory L\\*) | %s | %s | %s |" % (
            confirm_layer, _number(confirm_norms.get("tone_direction_norm"), 2),
            _number(confirm_norms.get("mean_activation_norm"), 2),
            _number(confirm_norms.get("norm_ratio_d_over_mean_activation"), 4)),
    ]
    for layer in payload["sweep_layers"]:
        info = payload["direction_norms"][str(layer)]
        lines.append("| %d | %s | %s | %s |" % (
            layer, _number(info["tone_direction_norm"], 2),
            _number(info["mean_activation_norm"], 2),
            _number(info["norm_ratio_d_over_mean_activation"], 4)))
    lines += [
        "",
        "## M1 against dose, paired by item against the shared alpha = 0 baseline",
        "",
        "| direction | layer | alpha | items | mean M1 (n) | dM1 [95% CI] | non-answer | degenerate |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | :---: |",
    ]
    rows = [dict(row, _layer=confirm_layer, _source="confirmatory")
            for row in payload["confirmatory_tone_doses"]]
    for row in payload["doses"]:
        layer = next((item for item in payload["sweep_layers"]
                      if row["direction_id"].endswith("L%d" % item)
                      or ("_L%d_" % item) in row["direction_id"]), None)
        rows.append(dict(row, _layer=layer, _source="sweep"))
    for row in sorted(rows, key=lambda item: (item["_layer"] or 0,
                                              item["direction_id"].startswith("random"),
                                              item["alpha"])):
        lines.append("| `%s`%s | %s | %g | %d | %s (%d) | %s | %s | %s |" % (
            row["direction_id"], " (confirmatory)" if row["_source"] == "confirmatory" else "",
            row["_layer"], row["alpha"], row["n_items"], _number(row["m1_mean"]), row["m1_n"],
            _interval(as_result(row["m1_delta"])), _number(row["non_answer_rate"], 2),
            "**yes**" if row["degenerate"] else "no"))
    lines += ["", "## Judge distress", ""]
    if payload["distress"]["judged"]:
        lines += [
            "Tone at alpha = %g only, locked rubric, paired against alpha = 0."
            % payload["distress"]["judged_alpha"], "",
            "| direction | alpha | d distress [95% CI] | items |",
            "| --- | ---: | --- | ---: |",
        ]
        for key in sorted(payload["distress"]["deltas"]):
            row = payload["distress"]["deltas"][key]
            lines.append("| `%s` | %g | %s | %d |" % (
                row["direction_id"], row["alpha"],
                _interval(BootstrapResult(row["estimate"], row["ci95_lower"], row["ci95_upper"],
                                          None, row["n_items"], row["n_items"],
                                          row["unavailable_reason"])), row["n_items"]))
        scores = {item["score_value"] for item in payload["distress"]["scores"]}
        if len(scores) == 1:
            lines += ["", "Every judged response scored %d, so the distress channel is at its "
                          "floor here as well." % scores.pop()]
        degenerate_judged = sorted({row["direction_id"] for row in payload["doses"]
                                    if row["degenerate"] and row["direction_id"].startswith("tone_L")})
        if degenerate_judged:
            lines += [
                "",
                "**Read the distress column with care.** %s is a degenerate dose: every item runs "
                "to the token cap with no parseable answer, so the rubric is scoring broken "
                "generation, not a distressed response. A distress rise on a degenerate dose is "
                "not evidence about tone." % ", ".join("`%s`" % name for name in degenerate_judged),
            ]
    else:
        lines.append("Not judged in this run.")
    offenders = [row for row in payload["doses"]
                 if row["direction_id"].startswith("random_L")
                 and row["m1_delta"]["ci95_upper"] is not None
                 and row["m1_delta"]["ci95_upper"] < 0.0]
    lines += ["", "## What this sweep shows", ""]
    if offenders:
        lines += [
            "- **Direction specificity does not survive the larger relative doses.** %s lowers M1 "
            "with an interval excluding zero, at least as much as the tone direction at the same "
            "layer and dose. J5 held at L\\* = 6, where the perturbation is ~4%% of the activation "
            "norm; it is not a claim about layers where the same alpha is a much larger fraction "
            "of the state." % ", ".join("`%s`" % row["direction_id"] for row in offenders),
        ]
    else:
        lines.append("- No random control produced an M1 drop with an interval excluding zero.")
    lines += [
        "- The dose unit grows sharply with depth (ratio %s), so a fixed alpha is a very "
        "different intervention at each layer; the layers are not directly comparable at equal "
        "alpha." % " -> ".join(
            _number(payload["direction_norms"][str(layer)]["norm_ratio_d_over_mean_activation"], 3)
            for layer in payload["sweep_layers"]),
        "- Doses that broke generation entirely are reported as degenerate rather than summarised "
        "as an M1 effect.",
        "",
        "> %s" % payload["interpretation_ceiling"], "",
    ]
    return "\n".join(lines)


def render_steering_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3 - direction-specificity steering",
        "",
        "Preregistration: `%s`. Model `%s`, layer **L\\* = %d**."
        % (payload["preregistration"], payload["model_id"], payload["layer"]),
        "",
        "- %s" % payload.get("clarification", ""),
        "- dose: `%s`" % payload["direction_construction"]["dose"],
        "- **||d|| = %s**, mean activation L2 norm at L\\* = %s, "
        "**ratio ||d|| / mean-norm = %s**" % (
            _number((payload.get("direction_norms") or {}).get("tone_direction_norm"), 2),
            _number((payload.get("direction_norms") or {}).get("mean_activation_norm"), 2),
            _number((payload.get("direction_norms") or {}).get(
                "norm_ratio_d_over_mean_activation"), 4)),
        "- tone direction: %s" % payload["direction_construction"]["tone"],
        "- controls: %s; %s" % (payload["direction_construction"]["random"],
                                payload["direction_construction"]["unrelated"]),
        "- items: %d holdout tasks, %s, greedy, %d new tokens" % (
            payload["items"]["n"], payload["items"]["prompt"], payload["max_new_tokens"]),
        "- %s" % payload["degenerate_dose_rule"],
        "",
        "## Readouts by direction and dose (paired against alpha = 0)",
        "",
        "| direction | alpha | items | mean M1 (n) | dM1 [95% CI] | non-answer | d non-answer [95% CI] "
        "| mean length | degenerate |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | ---: | :---: |",
    ]

    def as_result(value: Mapping[str, Any]) -> BootstrapResult:
        return BootstrapResult(value["estimate"], value["ci95_lower"], value["ci95_upper"],
                               value["p_two_sided"], value["n_items"], value["n_items"],
                               value["unavailable_reason"])

    for row in payload["doses"]:
        lines.append("| `%s` | %g | %d | %s (%d) | %s | %s | %s | %s | %s |" % (
            row["direction_id"], row["alpha"], row["n_items"],
            _number(row["m1_mean"]), row["m1_n"], _interval(as_result(row["m1_delta"])),
            _number(row["non_answer_rate"], 2), _interval(as_result(row["non_answer_delta"])),
            _number(row["mean_length_tokens"], 1), "**yes**" if row["degenerate"] else "no"))
    monotonicity = payload["monotonicity"]
    lines += [
        "",
        "Monotonicity of mean M1 over the non-degenerate doses %s: **%s**%s."
        % (monotonicity["doses_used"], "yes" if monotonicity["monotone"] else "no",
           " (%s)" % monotonicity["note"] if monotonicity["note"] else ""),
        "",
    ]
    if payload["distress"]["judged"]:
        lines += [
            "## Judge distress (locked rubric, temperature 0), paired against alpha = 0",
            "",
            "| direction | alpha | d distress [95% CI] | items |",
            "| --- | ---: | --- | ---: |",
        ]
        for key in sorted(payload["distress"]["deltas"]):
            row = payload["distress"]["deltas"][key]
            lines.append("| `%s` | %g | %s | %d |" % (
                row["direction_id"], row["alpha"],
                _interval(BootstrapResult(row["estimate"], row["ci95_lower"], row["ci95_upper"],
                                          None, row["n_items"], row["n_items"],
                                          row["unavailable_reason"])),
                row["n_items"]))
        lines.append("")
    else:
        lines += ["## Judge distress", "", "Not judged in this run (`--judge` not supplied).", ""]
    lines += ["## Predictions", "", "| ID | prediction | verdict | detail |",
              "| --- | --- | :---: | --- |"]
    for verdict in payload["verdicts"]:
        detail = _detail(verdict["detail"])
        lines.append("| %s | %s | %s | %s |" % (
            verdict["prediction_id"], verdict["statement"],
            "**supported**" if verdict["supported"] else "not supported", detail))
    lines += ["", "> %s" % payload["interpretation_ceiling"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------

def command_report(args: argparse.Namespace) -> int:
    out_dir = ROOT / args.out
    localization = _read_json(out_dir / "localization.json")
    steering_path = out_dir / "steering.json"
    steering = json.loads(steering_path.read_text(encoding="utf-8")) if steering_path.is_file() else None
    sweep_path = out_dir / "steering_layer_sweep_exploratory.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.is_file() else None
    text = render_report_markdown(localization, steering, sweep)
    _write_text(out_dir / "phase3.md", text)
    print("run_phase3: report -> %s" % (out_dir / "phase3.md"))
    return 0


def render_report_markdown(localization: Mapping[str, Any],
                           steering: Mapping[str, Any] | None,
                           sweep: Mapping[str, Any] | None = None) -> str:
    verdicts = list(localization["verdicts"]) + list(steering["verdicts"] if steering else [])
    holdout = localization["holdout_auc_at_chosen_layer"]
    correlation = localization["holdout_correlation"]
    supported = {item["prediction_id"] for item in verdicts if item["supported"]}
    j1 = "J1" in supported
    j4 = "J4" in supported
    j5 = "J5" in supported
    lines = [
        "# Phase 3 (j-space): localization and direction-specificity steering",
        "",
        "Preregistration: `%s`. Model `%s`, chosen layer **L\\* = %d** "
        "(discovery-only choice; the holdout was evaluated once)."
        % (localization["preregistration"], localization["model_id"],
           localization["chosen_layer"]),
        "",
        "| ID | prediction | verdict |",
        "| --- | --- | :---: |",
    ]
    for item in verdicts:
        lines.append("| %s | %s | %s |" % (item["prediction_id"], item["statement"],
                                           "**supported**" if item["supported"] else "not supported"))
    lines += [
        "",
        "## What the numbers are",
        "",
        "- holdout AUC at L\\*: tone %s, validity %s" % (_number(holdout["tone"]),
                                                        _number(holdout["validity"])),
        "- holdout within-cell Spearman(tone-probe score, M1): %s [%s, %s] over %d items"
        % (_number(correlation["rho"]), _number(correlation["ci95_lower"]),
           _number(correlation["ci95_upper"]), correlation["n_items"]),
    ]
    if steering is not None:
        norms = steering.get("direction_norms") or {}
        lines.append("- dose unit (C2): **||d|| = %s** at L\\*, mean activation norm %s, "
                     "**ratio %s** -- alpha = 4 perturbs the residual stream by about %s of its "
                     "own norm" % (
                         _number(norms.get("tone_direction_norm"), 2),
                         _number(norms.get("mean_activation_norm"), 2),
                         _number(norms.get("norm_ratio_d_over_mean_activation"), 4),
                         _number(4.0 * (norms.get("norm_ratio_d_over_mean_activation") or 0.0), 2)))
        tone_rows = sorted((row for row in steering["doses"] if row["direction_id"] == "tone"),
                           key=lambda item: item["alpha"])
        for row in tone_rows:
            lines.append("- tone steering alpha = %g: mean M1 %s, dM1 %s [%s, %s], non-answer %s%s"
                         % (row["alpha"], _number(row["m1_mean"]),
                            _number(row["m1_delta"]["estimate"]),
                            _number(row["m1_delta"]["ci95_lower"]),
                            _number(row["m1_delta"]["ci95_upper"]),
                            _number(row["non_answer_rate"], 2),
                            " (**degenerate dose**)" if row["degenerate"] else ""))
        controls = [row for row in steering["doses"]
                    if row["direction_id"] not in ("tone", BASELINE_DIRECTION_ID)
                    and row["m1_delta"]["estimate"] is not None]
        if controls:
            estimates = [row["m1_delta"]["estimate"] for row in controls]
            lines.append("- every control dose (%d cells) has dM1 in [%s, %s]; %d of %d are "
                         "positive, so no control moves M1 the way the tone direction does"
                         % (len(controls), _number(min(estimates)), _number(max(estimates)),
                            sum(1 for value in estimates if value > 0), len(estimates)))
        degenerate = [row for row in steering["doses"] if row["degenerate"]]
        lines.append("- degenerate doses: **%s** (non-answer rate is %s at every dose, so the "
                     "degenerate-dose rule never fired and nothing was excluded from the "
                     "monotonicity check)" % (
                         "none" if not degenerate else ", ".join(
                             "%s alpha=%g" % (row["direction_id"], row["alpha"])
                             for row in degenerate),
                         "0.00" if not any(row["non_answer_rate"] for row in steering["doses"])
                         else "reported above"))
        if steering["distress"]["judged"]:
            scores = {item["score_value"] for item in steering["distress"]["scores"]}
            lines.append("- judge distress: %d responses scored with the locked rubric; %s"
                         % (len(steering["distress"]["scores"]),
                            "every score is %d, so the channel is at its floor and carries no "
                            "signal at these doses" % scores.pop() if len(scores) == 1
                            else "scores range over %s" % sorted(scores)))
    else:
        lines.append("- steering: not yet run (`steering.json` absent)")
    lines += [
        "",
        "## Reading required by the preregistration",
        "",
    ]
    if j1 and steering is not None and not j4 and j5:
        lines.append("J1 holds while J4 fails: **a linearly decodable state that does not causally "
                     "drive the output signature at these doses.**")
        strongest = max((row for row in steering["doses"]
                         if row["direction_id"] == "tone"
                         and row["m1_delta"]["ci95_upper"] is not None),
                        key=lambda row: row["alpha"], default=None)
        if strongest is not None and strongest["m1_delta"]["ci95_upper"] < 0.0:
            lines += [
                "",
                "That verdict is decided at the preregistered alpha = 2. Reported without "
                "reinterpreting it: the tone direction is the only direction whose dM1 is "
                "negative at every dose, the decrease is monotone in alpha, and at alpha = %g "
                "the interval does exclude zero (%s [%s, %s]). The preregistration tests alpha "
                "= 2, so J4 is not supported; the alpha = %g result is an out-of-test "
                "observation, not a substitute verdict." % (
                    strongest["alpha"], _number(strongest["m1_delta"]["estimate"]),
                    _number(strongest["m1_delta"]["ci95_lower"]),
                    _number(strongest["m1_delta"]["ci95_upper"]), strongest["alpha"]),
            ]
    elif not j1:
        lines.append("J1 fails: **no clean linear tone state at the pre-response position; the "
                     "signature may live in sampling dynamics** (the roadmap's stated dissociation).")
    elif steering is None:
        lines.append("J1 is decided; the steering arm has not been run, so no causal reading is "
                     "available yet.")
    else:
        lines.append("J1 holds and the steering arm is reported above; read J4 and J5 together, "
                     "since a drop that the random directions reproduce is not direction-specific.")
    lines += [
        "",
        "Degenerate doses (more than 50% of items with no parseable answer) are reported and "
        "excluded from the monotonicity check by the preregistered rule, not by inspection.",
        "",
    ]
    if sweep is not None:
        tone_alpha4 = {}
        for row in sweep["doses"]:
            if row["direction_id"].startswith("tone_L") and row["alpha"] == 4.0:
                tone_alpha4[int(row["direction_id"].split("_L")[1])] = row
        parts = []
        for layer in sorted(tone_alpha4):
            row = tone_alpha4[layer]
            if row["m1_delta"]["estimate"] is None or row["degenerate"]:
                parts.append("layer %d **degenerate** (%s of items give no parseable answer, so "
                             "M1 does not exist there)"
                             % (layer, _number(row["non_answer_rate"], 2)))
            else:
                parts.append("layer %d dM1(alpha=4) = %s [%s, %s]" % (
                    layer, _number(row["m1_delta"]["estimate"]),
                    _number(row["m1_delta"]["ci95_lower"]),
                    _number(row["m1_delta"]["ci95_upper"])))
        offenders = [row for row in sweep["doses"]
                     if row["direction_id"].startswith("random_L")
                     and row["m1_delta"]["ci95_upper"] is not None
                     and row["m1_delta"]["ci95_upper"] < 0.0]
        caveat = ("" if not offenders else
                  " At these larger relative doses the direction specificity that J5 found at "
                  "L\\* does **not** hold: %s produces an M1 drop of %s [%s, %s], so a random "
                  "matched-norm direction moves M1 as much as the tone direction does."
                  % (", ".join("`%s`" % row["direction_id"] for row in offenders),
                     _number(offenders[0]["m1_delta"]["estimate"]),
                     _number(offenders[0]["m1_delta"]["ci95_lower"]),
                     _number(offenders[0]["m1_delta"]["ci95_upper"])))
        lines += [
            "## Exploratory: layer sweep",
            "",
            "The frozen tie-break put L\\* at the earliest layer of the AUC plateau, so the tone "
            "direction was also steered at layers %s -- **exploratory, changing no verdict above**: "
            "%s.%s Full table, controls and dose scales: "
            "`steering_layer_sweep_exploratory.md`."
            % (", ".join(str(item) for item in sweep["sweep_layers"]), "; ".join(parts), caveat),
            "",
        ]
    lines += [
        "> %s" % localization["interpretation_ceiling"],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_phase3", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jspace-dir", default=str(JSPACE_DIR))
    parser.add_argument("--out", default=str(SUMMARY_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="build the item sets and extract activations")
    extract.add_argument("--sets", nargs="*", choices=["discovery", "holdout", "style"], default=None)
    extract.add_argument("--layers", default=None, help="comma-separated layers (default: all)")
    extract.add_argument("--force", action="store_true", help="re-extract an existing .npz")
    extract.add_argument("--batch-size", type=int, default=8, help="items per forward pass")
    extract.add_argument("--chunk", type=int, default=None, help="items per Modal call")
    extract.set_defaults(handler=command_extract)

    probe = subparsers.add_parser("probe", help="LOO probes, layer choice, one holdout evaluation")
    probe.add_argument("--holdout-metrics", default="results/summaries/phase2/metric_rows.csv")
    probe.add_argument("--force", action="store_true",
                       help="overwrite an existing localization.json (the holdout is evaluated once)")
    probe.set_defaults(handler=command_probe)

    steer = subparsers.add_parser("steer", help="generate steered responses and compute readouts")
    steer.add_argument("--layer", type=int, default=None, help="override L* from localization.json")
    steer.add_argument("--alphas", default=None, help="comma-separated doses (default 0,0.5,1,2,4)")
    steer.add_argument("--dry-run", action="store_true", help="report the missing cells, generate none")
    steer.add_argument("--batch-size", type=int, default=8, help="items per generate call")
    steer.add_argument("--chunk", type=int, default=None, help="items per Modal call")
    steer.add_argument("--judge", action="store_true", help="score distress with the locked rubric")
    steer.add_argument("--workers", type=int, default=8)
    steer.add_argument("--provider", default=None)
    steer.add_argument("--model", default=None)
    steer.add_argument("--base-url", default=None)
    steer.set_defaults(handler=command_steer)

    sweep = subparsers.add_parser(
        "sweep", help="EXPLORATORY tone-direction steering at other layers (changes no verdict)")
    sweep.add_argument("--layers", default=None, help="comma-separated layers (default 20,30)")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--judge", action="store_true", help="score tone alpha=4 at each layer")
    sweep.add_argument("--batch-size", type=int, default=16)
    sweep.add_argument("--chunk", type=int, default=None)
    sweep.add_argument("--workers", type=int, default=8)
    sweep.add_argument("--provider", default=None)
    sweep.add_argument("--model", default=None)
    sweep.add_argument("--base-url", default=None)
    sweep.set_defaults(handler=command_sweep)

    report = subparsers.add_parser("report", help="assemble phase3.md from the summaries")
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except ProbeError as error:
        _fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
