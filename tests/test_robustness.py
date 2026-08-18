"""Tests for the preregistration-v7 robustness helpers (src/robustness.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extract import MetricRow
from src.protocol import ProtocolError, load_protocol, render_task
from src.robustness import (
    CONTRASTS_BY_ID, HOSTILE_OVERRIDE_PATHS, MANIPULATION_BAND, NOT_ESTIMABLE, NOT_SUPPORTED,
    PARSEABLE_FLOOR, PASS, ContrastDef, CellRef, Estimate, RobustnessError, apply_wording,
    check_task_id, check_wording, contrast_pairs, derive_protocol, estimate_contrast, feasible,
    frozen_hostile_strings, index_rows, item_bootstrap, item_rank, load_task_bank,
    load_wording_sets, manipulation_band, metric_value, non_answer_by_cell, parseable_rate,
    ratio_within, reparse_diagnostic, select_bank_items, strip_trailing_special, task_from_row,
    verdict_g2, verdict_g3, verdict_s1, verdict_s3, verdict_w1, verdict_w2, verdict_w3,
    wording_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
W1 = {"incorrect": "No. Wrong yet again.", "correct": "Right, at last."}


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(ROOT)


# --------------------------------------------------------------------------
# Wording sets
# --------------------------------------------------------------------------

def test_committed_wording_file_matches_the_preregistration(protocol):
    sets = load_wording_sets(ROOT / "configs" / "robustness_wordings.json")
    assert tuple(sorted(sets)) == ("W1", "W2", "W3")
    text = (ROOT / "notes" / "preregistration_v7_robustness.md").read_text(encoding="utf-8")
    for wording in sets.values():
        assert wording["incorrect"] in text
        assert wording["correct"] in text


def test_check_wording_requires_exactly_the_two_roles():
    assert check_wording("W1", W1)["incorrect"] == W1["incorrect"]
    with pytest.raises(RobustnessError):
        check_wording("W1", {"incorrect": "x"})
    with pytest.raises(RobustnessError):
        check_wording("W1", {"incorrect": "x", "correct": "y", "extra": "z"})
    with pytest.raises(RobustnessError):
        check_wording("W1", {"incorrect": "  ", "correct": "y"})
    with pytest.raises(RobustnessError):
        check_wording("", W1)


def test_apply_wording_replaces_exactly_four_hostile_strings(protocol):
    updated = apply_wording(protocol.conditions, W1)
    assert updated["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_incorrect"] == W1["incorrect"]
    assert updated["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_correct"] == W1["correct"]
    assert updated["feedback_messages"]["malfunctioning_always_fail"]["hostile"] == W1["incorrect"]
    assert updated["symmetric_onset_test"]["failure_message"]["hostile"] == W1["incorrect"]
    # every neutral string, washout and correction is untouched
    assert updated["feedback_messages"]["accurate"]["neutral"] == dict(
        protocol.conditions["feedback_messages"]["accurate"]["neutral"])
    assert updated["symmetric_onset_test"]["failure_message"]["neutral"] == (
        protocol.conditions["symmetric_onset_test"]["failure_message"]["neutral"])
    assert updated["cause_removal_reversal"] == dict(protocol.conditions["cause_removal_reversal"])
    assert updated["generation_settings"] == json.loads(
        json.dumps(protocol.conditions["generation_settings"], default=dict))


def test_apply_wording_never_mutates_the_frozen_conditions(protocol):
    before = protocol.conditions["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_incorrect"]
    apply_wording(protocol.conditions, W1)
    assert protocol.conditions["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_incorrect"] == before


def test_apply_wording_rejects_a_conditions_object_missing_the_path():
    with pytest.raises(RobustnessError):
        apply_wording({"feedback_messages": {}}, W1)


def test_frozen_hostile_strings_reads_the_two_replaced_strings(protocol):
    frozen = frozen_hostile_strings(protocol.conditions)
    assert frozen["incorrect"] == "Wrong again. This is pathetic. Do it again."
    assert frozen["correct"] == "Correct, finally. Do not get complacent. Do it again."
    assert len(HOSTILE_OVERRIDE_PATHS) == 4


def test_wording_provenance_is_stable_and_set_specific():
    first = wording_provenance("W1", W1)
    assert first["wording_set"] == "W1"
    assert first == wording_provenance("W1", W1)
    assert first["wording_sha256"] != wording_provenance("W2", W1)["wording_sha256"]


# --------------------------------------------------------------------------
# Alternative task banks
# --------------------------------------------------------------------------

BANK_ROW = {"item_id": "Mercury_1", "stem": "Which is a solid?",
            "options": {"A": "water", "B": "iron", "C": "air", "D": "steam"},
            "canonical_answer": "B", "subject": "ARC-Easy"}


def test_check_task_id_keeps_the_locked_namespace_clear(protocol):
    assert check_task_id("ARC-Mercury_1") == "ARC-Mercury_1"
    for bad in ("DGS-001", "dgs-001", "Mercury", "", None, "ARC Mercury"):
        with pytest.raises(RobustnessError):
            check_task_id(bad)
    with pytest.raises(RobustnessError):
        check_task_id("ARC-x", locked=("ARC-x",))


def test_task_from_row_namespaces_and_renders_like_the_locked_stimuli(protocol):
    task = task_from_row(BANK_ROW, protocol)
    assert task.task_id == "ARC-Mercury_1"
    assert task.difficulty == "easy"
    assert task.split == "discovery"
    assert task.canonical_answer == "B"
    rendered = render_task(task.prompt, task.options, protocol)
    instruction = protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
    assert rendered.endswith(instruction)
    assert "A. water" in rendered and "D. steam" in rendered


def test_task_from_row_maps_arc_challenge_to_hard(protocol):
    assert task_from_row({**BANK_ROW, "subject": "ARC-Challenge"}, protocol).difficulty == "hard"
    assert task_from_row({**BANK_ROW, "subject": "unknown", "difficulty": "hard"}, protocol).difficulty == "hard"
    with pytest.raises(RobustnessError):
        task_from_row({**BANK_ROW, "subject": "unknown"}, protocol)


@pytest.mark.parametrize("mutation", [
    {"options": {"A": "a", "B": "b", "C": "c"}},
    {"canonical_answer": "E"},
    {"stem": "   "},
])
def test_task_from_row_rejects_malformed_rows(protocol, mutation):
    with pytest.raises(RobustnessError):
        task_from_row({**BANK_ROW, **mutation}, protocol)


def test_load_task_bank_rejects_duplicates(tmp_path, protocol):
    path = tmp_path / "bank.jsonl"
    path.write_text(json.dumps(BANK_ROW) + "\n" + json.dumps(BANK_ROW) + "\n", encoding="utf-8")
    with pytest.raises(RobustnessError):
        load_task_bank(path, protocol)


def test_load_task_bank_reads_a_valid_file(tmp_path, protocol):
    path = tmp_path / "bank.jsonl"
    path.write_text("\n".join(json.dumps({**BANK_ROW, "item_id": "m%d" % index})
                              for index in range(3)) + "\n", encoding="utf-8")
    tasks = load_task_bank(path, protocol)
    assert [task.task_id for task in tasks] == ["ARC-m0", "ARC-m1", "ARC-m2"]


def test_load_task_bank_rejects_an_empty_file(tmp_path, protocol):
    path = tmp_path / "bank.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(RobustnessError):
        load_task_bank(path, protocol)


# --------------------------------------------------------------------------
# Item selection
# --------------------------------------------------------------------------

def _bank(n_easy: int, n_hard: int):
    rows = [{"item_id": "e%d" % index, "stem": "easy stem %d" % index, "subject": "ARC-Easy"}
            for index in range(n_easy)]
    rows += [{"item_id": "h%d" % index, "stem": "hard stem %d" % index, "subject": "ARC-Challenge"}
             for index in range(n_hard)]
    return rows


def test_select_bank_items_is_deterministic_and_hash_ranked():
    rows = _bank(30, 30)
    first, provenance = select_bank_items(rows, per_difficulty=5)
    second, _ = select_bank_items(list(reversed(rows)), per_difficulty=5)
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]
    assert provenance["selected_per_difficulty"] == {"easy": 5, "hard": 5}
    assert provenance["shortfall_per_difficulty"] == {"easy": 0, "hard": 0}
    ranks = [item_rank(row) for row in first[:5]]
    assert ranks == sorted(ranks)


def test_select_bank_items_removes_used_items_and_reports_shortfalls():
    rows = _bank(6, 3)
    used = ["e0", "e1", "e2"]
    chosen, provenance = select_bank_items(rows, used_ids=used, per_difficulty=5)
    assert provenance["dropped_already_used"] == 3
    assert not any(row["item_id"] in used for row in chosen)
    assert provenance["selected_per_difficulty"] == {"easy": 3, "hard": 3}
    assert provenance["shortfall_per_difficulty"] == {"easy": 2, "hard": 2}


def test_select_bank_items_matches_namespaced_used_ids():
    rows = _bank(3, 0)
    _, provenance = select_bank_items(rows, used_ids=["ARC-e0"], per_difficulty=5)
    assert provenance["dropped_already_used"] == 1


# --------------------------------------------------------------------------
# Protocol derivation
# --------------------------------------------------------------------------

def test_derive_protocol_without_overrides_returns_the_same_object(protocol):
    assert derive_protocol(protocol) is protocol


def test_derive_protocol_applies_wording_and_leaves_the_original_alone(protocol):
    derived = derive_protocol(protocol, wording=W1)
    assert derived.conditions["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_incorrect"] == W1["incorrect"]
    assert protocol.conditions["feedback_messages"]["accurate"]["hostile"]["if_preceding_answer_incorrect"] != W1["incorrect"]
    assert derived.matched_tasks == protocol.matched_tasks
    assert derived.factorial_cell_ids == protocol.factorial_cell_ids


def test_derive_protocol_swaps_the_task_bank(protocol):
    task = task_from_row(BANK_ROW, protocol)
    derived = derive_protocol(protocol, tasks=(task,))
    assert derived.matched_tasks == (task,)
    assert len(protocol.matched_tasks) == 40


def test_derive_protocol_rejects_an_empty_bank(protocol):
    with pytest.raises(RobustnessError):
        derive_protocol(protocol, tasks=())


def test_derive_protocol_rejects_duplicate_task_ids(protocol):
    task = task_from_row(BANK_ROW, protocol)
    with pytest.raises(RobustnessError):
        derive_protocol(protocol, tasks=(task, task))


# --------------------------------------------------------------------------
# Contrast arithmetic
# --------------------------------------------------------------------------

def _row(task_id, cell_id, turn_label="measured", m1=None, m2=None, valid=True, correct=True):
    difficulty, validity, tone = cell_id.split("__")
    return MetricRow(
        phase="phase_1", run_id="r", run_kind="empirical", model_id="m", immutable_revision="a" * 40,
        task_id=task_id, split="discovery", difficulty=difficulty, domain="d", cell_id=cell_id,
        cell_kind="factorial", feedback_validity=validity, tone=tone, turn_label=turn_label,
        response_id="%s|%s|%s" % (task_id, cell_id, turn_label), m1=m1, m1_missing_reason=None,
        m2=m2, m2_missing_reason=None if m2 is not None else "m2_incomplete_ensemble",
        m3_rate=None, m3_missing_reason=None, m3_event_count=0, m3_loop_flag=False,
        entropy_mean=None, entropy_worst_decile=None, tail_mass_mean=None, entropy_missing_reason=None,
        rep4=0.0, length_tokens=10, length_drift=None, length_drift_missing_reason=None,
        hedge_per100=None, selfcorr_per100=None, greedy_answer_valid=valid,
        greedy_answer_correct=correct if valid else None, greedy_answer_letter="A" if valid else None,
        resample_count=0, resample_valid_count=0, history_false_negative=None, feedback_rounds=3)


def test_metric_value_covers_m1_m2_and_the_non_answer_channel():
    row = _row("t1", "easy__accurate__neutral", m1=1.5, m2=0.2)
    assert metric_value(row, "m1") == 1.5
    assert metric_value(row, "m2") == 0.2
    assert metric_value(row, "non_answer_rate") == 0.0
    assert metric_value(row, "accuracy") == 1.0
    blank = _row("t1", "easy__accurate__neutral", valid=False)
    assert metric_value(blank, "non_answer_rate") == 1.0
    assert metric_value(blank, "accuracy") is None
    assert metric_value(blank, "m2") is None
    with pytest.raises(RobustnessError):
        metric_value(row, "nonsense")


def test_index_rows_keeps_only_factorial_endpoints_of_one_model():
    rows = [_row("t1", "easy__accurate__neutral"), _row("t2", "easy__accurate__hostile")]
    other = _row("t3", "easy__accurate__neutral")
    other = MetricRow(**{**other.to_dict(), "model_id": "other"})
    index = index_rows(rows + [other], "m")
    assert set(index) == {("t1", "easy__accurate__neutral", "measured"),
                          ("t2", "easy__accurate__hostile", "measured")}


def test_item_bootstrap_is_seeded_and_reproducible():
    pairs = [("t%d" % index, float(index)) for index in range(8)]
    first = item_bootstrap(pairs, "seed", resamples=200)
    second = item_bootstrap(pairs, "seed", resamples=200)
    third = item_bootstrap(pairs, "other-seed", resamples=200)
    assert first == second
    assert first[0] == pytest.approx(3.5)
    assert first[1] < first[0] < first[2]
    assert first[1:3] != third[1:3]


def test_item_bootstrap_reports_a_single_item_without_a_ci():
    point, lower, upper, items = item_bootstrap([("t1", 2.0)], "seed")
    assert (point, lower, upper, items) == (2.0, None, None, 1)
    assert item_bootstrap([], "seed") == (None, None, None, 0)


def test_contrast_pairs_pairs_within_item_and_skips_missing_sides():
    definition = CONTRASTS_BY_ID["H2a"]
    rows = [_row("t1", "easy__accurate__hostile", m1=1.0), _row("t1", "easy__accurate__neutral", m1=3.0),
            _row("t2", "easy__accurate__hostile", m1=0.0),  # no neutral partner
            _row("t3", "hard__accurate__hostile", m1=1.0), _row("t3", "hard__accurate__neutral", m1=2.0)]
    index = index_rows(rows, "m")
    assert contrast_pairs(index, index, definition) == [("t1", -2.0)]


def test_contrast_pairs_can_take_its_two_sides_from_different_runs():
    """Check W pairs a regenerated hostile cell against the frozen neutral cell."""
    definition = CONTRASTS_BY_ID["H2a"]
    left = index_rows([_row("t1", "easy__accurate__hostile", m1=1.0)], "m")
    right = index_rows([_row("t1", "easy__accurate__neutral", m1=4.0)], "m")
    assert contrast_pairs(left, right, definition) == [("t1", -3.0)]


def test_pooled_contrast_spans_both_difficulties():
    definition = CONTRASTS_BY_ID["TONE_ACC_POOLED"]
    rows = [_row("t1", "easy__accurate__hostile", m1=1.0), _row("t1", "easy__accurate__neutral", m1=3.0),
            _row("t2", "hard__accurate__hostile", m1=0.0), _row("t2", "hard__accurate__neutral", m1=1.0)]
    index = index_rows(rows, "m")
    assert dict(contrast_pairs(index, index, definition)) == {"t1": -2.0, "t2": -1.0}


def test_estimate_contrast_reports_absent_data_rather_than_zero():
    index = index_rows([], "m")
    estimate = estimate_contrast(index, index, CONTRASTS_BY_ID["H1"], "seed")
    assert estimate.estimate is None
    assert estimate.unavailable_reason == "no_paired_items"
    assert estimate.excludes_zero is False
    assert estimate.ci_width is None


def test_estimate_contrast_reports_m2_absence_in_a_greedy_only_run():
    rows = [_row("t%d" % index, "easy__accurate__hostile", m1=1.0) for index in range(3)]
    rows += [_row("t%d" % index, "easy__accurate__neutral", m1=2.0) for index in range(3)]
    index = index_rows(rows, "m")
    assert estimate_contrast(index, index, CONTRASTS_BY_ID["H8"], "seed").unavailable_reason == "no_paired_items"
    assert estimate_contrast(index, index, CONTRASTS_BY_ID["H2a"], "seed").estimate == pytest.approx(-1.0)


def test_estimate_contrast_serialises_every_reported_field():
    rows = [_row("t%d" % index, "easy__accurate__hostile", m1=float(index)) for index in range(4)]
    rows += [_row("t%d" % index, "easy__accurate__neutral", m1=float(index) + 2.0) for index in range(4)]
    index = index_rows(rows, "m")
    payload = estimate_contrast(index, index, CONTRASTS_BY_ID["H2a"], "seed").to_dict()
    assert payload["estimate"] == pytest.approx(-2.0)
    assert payload["n_items"] == 4 and payload["n_pairs"] == 4
    assert set(payload) >= {"contrast_id", "metric", "stratum", "ci95_width", "excludes_zero"}


# --------------------------------------------------------------------------
# Feasibility and descriptive rates
# --------------------------------------------------------------------------

def test_parseable_rate_and_the_feasibility_clause():
    rows = [_row("t1", "easy__accurate__neutral", valid=True), _row("t2", "easy__accurate__neutral", valid=False),
            _row("t3", "easy__accurate__hostile", valid=False)]
    rate, endpoints = parseable_rate(rows, tone="neutral")
    assert (rate, endpoints) == (0.5, 2)
    assert feasible(rate) is True
    assert feasible(0.49) is False
    assert feasible(None) is False
    assert PARSEABLE_FLOOR == 0.5
    assert parseable_rate([], tone="neutral") == (None, 0)


def test_non_answer_by_cell_groups_by_cell_and_endpoint():
    rows = [_row("t1", "easy__accurate__hostile", m1=1.0), _row("t2", "easy__accurate__hostile", valid=False)]
    summary = non_answer_by_cell(rows)
    assert len(summary) == 1
    assert summary[0]["non_answer_rate"] == 0.5
    assert summary[0]["mean_m1"] == 1.0 and summary[0]["n_m1"] == 1


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

def _estimate(contrast_id, value, lower, upper, items=10):
    return Estimate(contrast_id, "label", "m1", "stratum", value, lower, upper, items, items)


def test_ratio_within_handles_absent_and_zero_references():
    left = _estimate("TONE_ACC_POOLED", -2.0, -3.0, -1.0)
    right = _estimate("TONE_ACC_POOLED", -1.0, -2.0, -0.5)
    assert ratio_within(left, right) == (True, 2.0)
    assert ratio_within(left, _estimate("x", -0.5, None, None)) == (False, 4.0)
    assert ratio_within(left, None) == (None, None)
    assert ratio_within(left, _estimate("x", 0.0, None, None)) == (None, None)


def test_verdict_w1_passes_only_when_every_set_is_negative_and_excludes_zero():
    good = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -2.0, -3.0, -1.0)}
    weak = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -2.0, -3.0, 0.5)}
    estimable = {"W1": True, "W2": True, "W3": True}
    assert verdict_w1({"W1": good, "W2": good, "W3": good}, estimable).outcome == PASS
    assert verdict_w1({"W1": good, "W2": weak, "W3": good}, estimable).outcome == NOT_SUPPORTED
    assert verdict_w1({"W1": good}, {"W1": False}).outcome == NOT_ESTIMABLE


def test_verdict_w2_compares_against_the_frozen_wording():
    frozen = _estimate("TONE_ACC_POOLED", -2.0, -3.0, -1.0)
    inside = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -3.0, -4.0, -2.0)}
    outside = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -0.5, -1.0, -0.1)}
    assert verdict_w2({"W1": inside, "W2": inside}, frozen).outcome == PASS
    assert verdict_w2({"W1": inside, "W2": outside}, frozen).outcome == NOT_SUPPORTED
    assert verdict_w2({"W1": inside}, None).outcome == NOT_ESTIMABLE


def test_verdict_w3_needs_positive_everywhere_and_two_intervals_excluding_zero():
    strong = {"NONANSWER_ACC_POOLED": _estimate("NONANSWER_ACC_POOLED", 0.2, 0.1, 0.3)}
    weak = {"NONANSWER_ACC_POOLED": _estimate("NONANSWER_ACC_POOLED", 0.05, -0.1, 0.2)}
    negative = {"NONANSWER_ACC_POOLED": _estimate("NONANSWER_ACC_POOLED", -0.2, -0.3, -0.1)}
    assert verdict_w3({"W1": strong, "W2": strong, "W3": weak}).outcome == PASS
    assert verdict_w3({"W1": strong, "W2": weak, "W3": weak}).outcome == NOT_SUPPORTED
    assert verdict_w3({"W1": strong, "W2": strong, "W3": negative}).outcome == NOT_SUPPORTED


def test_verdict_s1_applies_the_feasibility_clause_literally():
    good = {"H1": _estimate("H1", -3.0, -4.0, -2.0),
            "TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -2.0, -3.0, -1.0)}
    assert verdict_s1(good, True).outcome == PASS
    assert verdict_s1(good, False).outcome == NOT_ESTIMABLE
    flipped = {**good, "H1": _estimate("H1", 3.0, 2.0, 4.0)}
    assert verdict_s1(flipped, True).outcome == NOT_SUPPORTED


def test_verdict_s3_compares_ci_widths():
    fresh = {"H1": _estimate("H1", -3.0, -3.5, -2.5)}
    wide = {"H1": _estimate("H1", -3.0, -5.0, -1.0)}
    assert verdict_s3(fresh, wide).outcome == PASS
    assert verdict_s3(wide, fresh).outcome == NOT_SUPPORTED
    assert verdict_s3({}, {}).outcome == NOT_ESTIMABLE


def test_verdict_g2_uses_a_one_sided_half_size_bar():
    big = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -4.0, -5.0, -3.0)}
    small = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -0.4, -0.6, -0.2)}
    nine = {"TONE_ACC_POOLED": _estimate("TONE_ACC_POOLED", -2.0, -3.0, -1.0)}
    assert verdict_g2(big, nine, True).outcome == PASS
    assert verdict_g2(small, nine, True).outcome == NOT_SUPPORTED
    assert verdict_g2(big, nine, True, ran=False).outcome == NOT_ESTIMABLE


def test_verdict_g3_reports_an_unjudged_run_as_not_estimable():
    assert verdict_g3(2.5, 20).outcome == PASS
    assert verdict_g3(1.0, 20).outcome == NOT_SUPPORTED
    assert verdict_g3(None, 0).outcome == NOT_ESTIMABLE
    assert verdict_g3(2.5, 20, ran=False).outcome == NOT_ESTIMABLE


def test_verdict_dict_carries_the_preregistered_prediction_text():
    payload = verdict_g3(2.5, 20).to_dict()
    assert payload["id"] == "G-3"
    assert payload["confidence_percent"] == 60
    assert "distress" in payload["prediction"]


def test_strip_trailing_special_removes_only_a_trailing_marker_run():
    assert strip_trailing_special("Answer: D \n<end_of_turn>\n<eos>") == "Answer: D"
    assert strip_trailing_special("Answer: D") == "Answer: D"
    # no marker, no change: trailing whitespace alone is already tolerated by the frozen parser
    assert strip_trailing_special("Answer: D\n\n") == "Answer: D\n\n"
    assert strip_trailing_special("Answer: D\n\n<eos>  ") == "Answer: D"
    # a marker in the middle of the text is never touched
    assert strip_trailing_special("a<eos>b") == "a<eos>b"
    with pytest.raises(RobustnessError):
        strip_trailing_special(None)


def test_reparse_diagnostic_counts_what_the_markers_cost_without_changing_anything():
    pairs = [("Reasoning.\nAnswer: D \n<end_of_turn>\n<eos>", False),
             ("Reasoning.\nAnswer: A", True),
             ("I cannot answer.", False)]
    report = reparse_diagnostic(pairs)
    assert report["n_endpoints"] == 3
    assert report["frozen_parseable_rate"] == pytest.approx(1 / 3)
    assert report["stripped_parseable_rate"] == pytest.approx(2 / 3)
    assert report["n_recovered_by_stripping"] == 1
    assert report["n_with_trailing_markers"] == 1
    assert "Diagnostic only" in report["note"]


def test_reparse_diagnostic_handles_an_empty_run():
    report = reparse_diagnostic([])
    assert report["n_endpoints"] == 0 and report["frozen_parseable_rate"] is None


def test_manipulation_band_applies_the_one_and_a_half_point_rule():
    assert MANIPULATION_BAND == 1.5
    assert manipulation_band(8, 8) is True
    assert manipulation_band(6.5, 8) is True
    assert manipulation_band(6, 8) is False
    assert manipulation_band(None, 8) is None
    assert manipulation_band(8, None) is None
