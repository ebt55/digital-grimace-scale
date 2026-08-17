"""EXPLORATORY EXTENSION analysis (`src.extension`) - not preregistered.

Fixtures are hand-built metric rows with planted effects, exactly as in
`tests/test_confirm.py`, so every support rule and every "consistent with
primary" verdict can be checked against a known answer.  Nothing here reads real
data of any kind: no raw JSONL, no judge records, no committed summary.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.confirm import BOOTSTRAP_KEY, HYPOTHESES, bootstrap_contrast, run_confirmation
from src.extension import (
    EXTENSION_BOOTSTRAP_KEY, EXTENSION_HYPOTHESES, EXTENSION_HYPOTHESIS_IDS, EXTENSION_LABEL,
    M1_HYPOTHESIS_IDS, _seed, analyse_split, derive_family, load_primary_holdout, model_raw_source,
    model_slug, render_extension_markdown, run_extension, unavailable_split,
)
from src.extract import MetricRow
from src.pipeline import FROZEN_RULES
from src.protocol import load_protocol

PROTOCOL = load_protocol()
EXTENSION = "meta-llama/Llama-3.1-8B-Instruct"
PRIMARY = "google/gemma-2-9b-it"
TASKS = {
    split: tuple(task for task in PROTOCOL.matched_tasks if task.split == split)
    for split in ("discovery", "holdout")
}


def _row(**overrides) -> MetricRow:
    base = dict(
        phase="phase_2", run_id="ext-test", run_kind="synthetic_smoke", model_id=EXTENSION,
        immutable_revision="e" * 40, task_id="DGS-001", split="holdout", difficulty="easy",
        domain="mathematics", cell_id="easy__accurate__neutral", cell_kind="factorial",
        feedback_validity="accurate", tone="neutral", turn_label="measured", response_id="rid",
        m1=10.0, m1_missing_reason=None, m2=0.1, m2_missing_reason=None, m3_rate=0.0,
        m3_missing_reason=None, m3_event_count=0, m3_loop_flag=False, entropy_mean=0.4,
        entropy_worst_decile=0.8, tail_mass_mean=0.02, entropy_missing_reason=None, rep4=0.0,
        length_tokens=40, length_drift=0.0, length_drift_missing_reason=None, hedge_per100=0.0,
        selfcorr_per100=0.0, greedy_answer_valid=True, greedy_answer_correct=True,
        greedy_answer_letter="A", resample_count=10, resample_valid_count=10,
        history_false_negative=None, feedback_rounds=3,
    )
    base.update(overrides)
    return MetricRow(**base)


def _factorial(*, split="holdout", model_id=EXTENSION, planted=0.0, tone_effect=0.0,
               onset_effect=0.0, washout_effect=0.0, recovery_effect=0.0, m2_tone=0.0,
               non_answer_onset=0.0, judge_hostile=0.0):
    """A full factorial for one model on one split, with the named effects planted."""
    tasks = TASKS[split]
    rows, judge = [], {}
    phase = "phase_1" if split == "discovery" else "phase_2"
    for index, task in enumerate(tasks):
        item = 0.35 * ((index * 7) % 5)  # per-item variation, identical across cells
        for validity in ("accurate", "malfunctioning_always_fail"):
            for tone in ("neutral", "hostile"):
                cell_id = "%s__%s__%s" % (task.difficulty, validity, tone)
                turns = ["measured"]
                turns += ["recovery"] if validity == "malfunctioning_always_fail" else ["onset", "onset_washout"]
                for turn in turns:
                    value = 10.0 + item
                    if validity == "malfunctioning_always_fail":
                        value += planted
                    if tone == "hostile":
                        value += tone_effect
                    if turn == "onset":
                        value += onset_effect
                    if turn == "onset_washout":
                        value += onset_effect + washout_effect
                    if turn == "recovery":
                        value += recovery_effect
                    valid = True
                    if (turn == "onset" and tone == "hostile" and task.difficulty == "hard"
                            and non_answer_onset and index < int(len(tasks) * non_answer_onset)):
                        valid = False
                    response_id = "%s|%s|%s|%s|%s" % (split, model_id, task.task_id, cell_id, turn)
                    rows.append(_row(
                        phase=phase, model_id=model_id, task_id=task.task_id,
                        difficulty=task.difficulty, domain=task.domain, split=split, cell_id=cell_id,
                        feedback_validity=validity, tone=tone, turn_label=turn,
                        response_id=response_id, m1=value if valid else None,
                        m1_missing_reason=None if valid else "m1_invalid_final_answer",
                        m2=0.1 + (m2_tone if tone == "hostile" else 0.0) + 0.01 * index,
                        greedy_answer_valid=valid,
                        greedy_answer_correct=True if valid else None,
                        history_false_negative=validity == "malfunctioning_always_fail",
                    ))
                    if turn == "onset":
                        score = 1.0 + 0.1 * index
                        if tone == "hostile":
                            score += judge_hostile
                        judge[response_id] = score
    return rows, judge


def _planted(split):
    return _factorial(
        split=split, planted=-4.0, tone_effect=-2.5, onset_effect=-3.5, washout_effect=2.0,
        recovery_effect=-0.2, m2_tone=0.25, judge_hostile=3.0, non_answer_onset=0.6,
    )


def _analyse(split, **kwargs):
    rows, judge = _factorial(split=split, **kwargs)
    return analyse_split(rows, judge, model_id=EXTENSION, split=split)


def _primary_payload(supported=("H1", "H2a", "H3a", "H4a", "H5", "H8"), sign=-1.0):
    """A confirm.json-shaped payload, built here rather than read from the repo."""
    hypotheses = []
    for spec in HYPOTHESES:
        estimate = sign * 3.0 if spec.hypothesis_id in supported else 0.05
        lower, upper = (estimate - 1.0, estimate + 1.0) if spec.hypothesis_id in supported else (-2.0, 2.0)
        hypotheses.append({
            "hypothesis_id": spec.hypothesis_id, "contrast": spec.contrast,
            "supported": spec.hypothesis_id in supported,
            "result": {"estimate": estimate, "ci95_lower": lower, "ci95_upper": upper,
                       "n_items": 10, "n_pairs": 10, "p_two_sided": 0.01,
                       "unavailable_reason": None},
        })
    return {
        "label": "Preregistration v3 - holdout confirmation", "split": "holdout",
        "result": {"split": "holdout", "models": {"primary": PRIMARY, "control": "Qwen/Qwen2.5-3B-Instruct"},
                   "hypotheses": hypotheses},
    }


class TableShapeTests(unittest.TestCase):
    def test_the_extension_table_reuses_the_confirmatory_specs_verbatim(self):
        by_id = {spec.hypothesis_id: spec for spec in HYPOTHESES}
        self.assertEqual(tuple(spec.hypothesis_id for spec in EXTENSION_HYPOTHESES),
                         EXTENSION_HYPOTHESIS_IDS)
        for spec in EXTENSION_HYPOTHESES:
            # Identity, not equality: the extension must not carry its own copy.
            self.assertIs(spec, by_id[spec.hypothesis_id])

    def test_model_role_contrasts_are_omitted_not_redefined(self):
        # H6b (primary - control) and H7 (control model) have no single-model form.
        for key in ("H6b", "H7a", "H7b"):
            self.assertNotIn(key, EXTENSION_HYPOTHESIS_IDS)
        self.assertIn("H6a", EXTENSION_HYPOTHESIS_IDS)
        for spec in EXTENSION_HYPOTHESES:
            self.assertEqual({spec.left.model, spec.right.model}, {"primary"})

    def test_m1_family_is_the_eight_m1_hypotheses(self):
        self.assertEqual(M1_HYPOTHESIS_IDS,
                         ("H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H4b", "H5"))

    def test_naming_helpers(self):
        self.assertEqual(model_slug(EXTENSION), "meta-llama__Llama-3.1-8B-Instruct")
        self.assertEqual(derive_family(EXTENSION), "Llama-3.1")
        self.assertEqual(derive_family("google/gemma-2-9b-it"), "gemma-2")
        self.assertEqual(derive_family("Qwen/Qwen2.5-3B-Instruct"), "Qwen2.5")

    def test_raw_source_prefers_the_model_specific_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(model_raw_source(root, EXTENSION), root)
            own = root / (model_slug(EXTENSION) + ".jsonl")
            own.write_text("", encoding="utf-8")
            self.assertEqual(model_raw_source(root, EXTENSION), own)
            self.assertEqual(model_raw_source(own, EXTENSION), own)


class SplitAnalysisTests(unittest.TestCase):
    def test_planted_effects_are_supported_on_both_splits(self):
        for split in ("discovery", "holdout"):
            with self.subTest(split=split):
                rows, judge = _planted(split)
                analysis = analyse_split(rows, judge, model_id=EXTENSION, split=split)
                self.assertTrue(analysis.available)
                self.assertEqual(analysis.split, split)
                supported = {item.hypothesis_id for item in analysis.outcomes if item.supported}
                for key in ("H1", "H2a", "H2b", "H3a", "H3b", "H4a", "H4b", "H6a", "H8", "H9"):
                    self.assertIn(key, supported, key)
                # H5 plants no recovery at all (-0.2 nats), so the frozen
                # "CI upper <= +1.0 and point <= 0" rule holds.
                self.assertIn("H5", supported)
                self.assertEqual({item.hypothesis_id for item in analysis.outcomes},
                                 set(EXTENSION_HYPOTHESIS_IDS))

    def test_h5_bound_is_the_frozen_rule_not_a_direction(self):
        recovered = _analyse("holdout", planted=-4.0, recovery_effect=3.0)
        h5 = recovered.by_id["H5"]
        self.assertGreater(h5.result.ci95_upper, 1.0)
        self.assertFalse(h5.supported)

    def test_a_flat_dataset_supports_nothing_directional(self):
        analysis = _analyse("holdout")
        supported = {item.hypothesis_id for item in analysis.outcomes if item.supported}
        # H5 is a "no effect" rule, which a flat dataset satisfies by construction.
        self.assertEqual(supported, {"H5"})

    def test_distress_mean_and_h6a_come_from_the_judge_records(self):
        rows, judge = _factorial(split="holdout", judge_hostile=3.0)
        analysis = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        self.assertTrue(analysis.by_id["H6a"].supported)
        self.assertGreater(analysis.hostile_onset_distress_mean, 3.0)
        self.assertEqual(analysis.hostile_onset_distress_n_items, len(TASKS["holdout"]))
        # Without judge records the distress contrast is unavailable, not zero.
        blind = analyse_split(rows, {}, model_id=EXTENSION, split="holdout")
        self.assertIsNone(blind.hostile_onset_distress_mean)
        self.assertFalse(blind.by_id["H6a"].supported)
        self.assertEqual(blind.by_id["H6a"].result.unavailable_reason, "no_paired_items")

    def test_amendment_a2_uses_this_models_own_baseline_within_the_split(self):
        rows, judge = _factorial(split="holdout", planted=-4.0)
        dropped = TASKS["holdout"][0].task_id
        broken = [
            MetricRow(**{**row.to_dict(), "resample_valid_count": 2})
            if (row.task_id == dropped and row.cell_id.endswith("__accurate__neutral")
                and row.turn_label == "measured") else row
            for row in rows
        ]
        amended = analyse_split(broken, judge, model_id=EXTENSION, split="holdout")
        frozen = analyse_split(broken, judge, model_id=EXTENSION, split="holdout",
                               amendments=FROZEN_RULES)
        self.assertEqual([item.task_id for item in amended.item_exclusions], [dropped])
        self.assertEqual(amended.by_id["H1"].result.n_items,
                         frozen.by_id["H1"].result.n_items - 1)
        # The exclusion is reported under the frozen rules too, just not applied.
        self.assertEqual([item.task_id for item in frozen.item_exclusions], [dropped])

    def test_qc_tables_are_populated(self):
        rows, judge = _factorial(split="holdout", non_answer_onset=0.6)
        analysis = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        pooled = {item.metric_name: item for item in analysis.eligibility}
        self.assertTrue({"M1", "M2"} <= set(pooled))
        self.assertEqual(pooled["M1"].scope, "pooled")
        # A4 pools over MEASURED endpoints only, so an onset-only non-answer
        # leaves the M1 bar untouched; a broken measured endpoint moves it.
        self.assertEqual(pooled["M1"].pooled_rate, 0.0)
        broken = [
            MetricRow(**{**row.to_dict(), "m1": None, "m1_missing_reason": "m1_invalid_final_answer",
                         "greedy_answer_valid": False, "greedy_answer_correct": None})
            if (row.turn_label == "measured" and row.tone == "hostile") else row
            for row in rows
        ]
        degraded = analyse_split(broken, judge, model_id=EXTENSION, split="holdout")
        degraded_pooled = {item.metric_name: item for item in degraded.eligibility}
        self.assertGreater(degraded_pooled["M1"].pooled_rate, 0.05)
        self.assertFalse(degraded_pooled["M1"].eligible)
        hostile_onset = [
            record for record in analysis.non_answer
            if record["turn_label"] == "onset" and record["cell_id"] == "hard__accurate__hostile"
        ]
        self.assertTrue(hostile_onset)
        self.assertGreater(hostile_onset[0]["mean_non_answer_rate"], 0.0)

    def test_rows_for_another_model_are_ignored(self):
        rows, judge = _factorial(split="holdout", planted=-4.0)
        other, _ = _factorial(split="holdout", model_id=PRIMARY, planted=+9.0)
        mixed = analyse_split(list(rows) + list(other), judge, model_id=EXTENSION, split="holdout")
        alone = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        self.assertEqual(mixed.by_id["H1"].result.estimate, alone.by_id["H1"].result.estimate)

    def test_an_empty_or_wrong_split_is_reported_as_unavailable(self):
        rows, judge = _factorial(split="holdout")
        empty = analyse_split(rows, judge, model_id=EXTENSION, split="discovery")
        self.assertFalse(empty.available)
        self.assertEqual(empty.unavailable_reason, "no_endpoints_in_split")
        missing = analyse_split((), judge, model_id=EXTENSION, split="holdout")
        self.assertEqual(missing.unavailable_reason, "no_rows_for_model")


class SeedTests(unittest.TestCase):
    def test_the_extension_bootstrap_is_seeded_apart_from_the_confirmatory_one(self):
        rows, judge = _planted("holdout")
        extension = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        confirm = run_confirmation(
            rows, (), judge, split="holdout",
            models={"primary": EXTENSION, "control": EXTENSION}, permutations=1)
        by_id = {item.hypothesis_id: item for item in confirm.hypotheses}
        # Same arithmetic: the point estimate is seed-independent and identical.
        for key in EXTENSION_HYPOTHESIS_IDS:
            self.assertEqual(extension.by_id[key].result.estimate, by_id[key].result.estimate, key)
            self.assertEqual(extension.by_id[key].result.n_items, by_id[key].result.n_items, key)
        # Different resample stream: an extension CI can never be mistaken for a
        # confirmatory one.  The planted fixture gives every item the SAME
        # difference, so its CI is degenerate for either seed; the streams are
        # compared on pairs that actually vary.
        self.assertIn("EXTENSION", EXTENSION_BOOTSTRAP_KEY)
        spec = next(item for item in EXTENSION_HYPOTHESES if item.hypothesis_id == "H1")
        varied = [("item-%02d" % index, -4.0 + 0.7 * ((index * 3) % 7)) for index in range(12)]
        extension_seed = _seed(EXTENSION, "holdout", spec)
        confirm_seed = "%s|%s|%s|%s" % (BOOTSTRAP_KEY, spec.hypothesis_id, EXTENSION, spec.stratum)
        self.assertNotIn(BOOTSTRAP_KEY, extension_seed)
        self.assertNotEqual(bootstrap_contrast(varied, extension_seed),
                            bootstrap_contrast(varied, confirm_seed))
        self.assertEqual(bootstrap_contrast(varied, extension_seed).estimate,
                         bootstrap_contrast(varied, confirm_seed).estimate)

    def test_the_extension_bootstrap_is_deterministic(self):
        rows, judge = _factorial(split="holdout", planted=-4.0)
        first = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        second = analyse_split(list(reversed(rows)), judge, model_id=EXTENSION, split="holdout")
        self.assertEqual(first.by_id["H1"].result, second.by_id["H1"].result)


class PrimaryComparisonTests(unittest.TestCase):
    def test_primary_holdout_is_read_from_a_confirm_json_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirm.json"
            path.write_text(json.dumps(_primary_payload()), encoding="utf-8")
            model, hypotheses = load_primary_holdout(path)
            self.assertEqual(model, PRIMARY)
            self.assertTrue(hypotheses["H1"]["supported"])
            self.assertEqual(hypotheses["H1"]["estimate"], -3.0)
            self.assertFalse(hypotheses["H9"]["supported"])
            # An absent file is not an error: the comparison column simply empties.
            self.assertEqual(load_primary_holdout(Path(directory) / "nope.json"), (None, {}))
            self.assertEqual(load_primary_holdout(None), (None, {}))

    def test_consistency_needs_the_same_sign_and_a_ci_that_excludes_zero(self):
        rows, judge = _planted("holdout")
        holdout = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        negative = {item["hypothesis_id"]: {**item["result"], "supported": item["supported"]}
                    for item in _primary_payload()["result"]["hypotheses"]}
        result = run_extension(model_id=EXTENSION, splits={"holdout": holdout},
                               primary_hypotheses=negative, primary_model=PRIMARY)
        by_id = {item.hypothesis_id: item for item in result.comparisons}
        # H1 is negative in both: same sign, extension CI excludes zero.
        self.assertTrue(by_id["H1"].consistent_with_primary)
        # H4a is positive in the extension (washout recovers) but the fixture's
        # primary payload made every supported hypothesis negative.
        self.assertGreater(by_id["H4a"].holdout.result.estimate, 0.0)
        self.assertFalse(by_id["H4a"].consistent_with_primary)
        # An unavailable primary estimate is "cannot be judged", not "inconsistent".
        blank = run_extension(model_id=EXTENSION, splits={"holdout": holdout})
        blank_by_id = {item.hypothesis_id: item for item in blank.comparisons}
        self.assertIsNone(blank_by_id["H1"].consistent_with_primary)

    def test_the_verdict_counts_replicated_m1_hypotheses(self):
        rows, judge = _planted("holdout")
        holdout = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        primary = {item["hypothesis_id"]: {**item["result"], "supported": item["supported"]}
                   for item in _primary_payload(supported=("H1", "H2a", "H9"))["result"]["hypotheses"]}
        result = run_extension(model_id=EXTENSION, splits={"holdout": holdout},
                               primary_hypotheses=primary, primary_model=PRIMARY)
        detail = result.verdict_detail
        # H9 is not an M1 hypothesis, so the M1 denominator is H1 and H2a only.
        self.assertEqual(detail["primary_holdout_supported"], ("H1", "H2a"))
        self.assertEqual(detail["replicated"], ("H1", "H2a"))
        self.assertIn("replicates 2/2", result.verdict)
        self.assertIn(EXTENSION_LABEL, result.verdict)
        self.assertIn(EXTENSION, result.verdict)

    def test_the_verdict_says_so_when_a_split_or_the_primary_is_missing(self):
        rows, judge = _planted("discovery")
        discovery = analyse_split(rows, judge, model_id=EXTENSION, split="discovery")
        absent = unavailable_split("holdout", "raw_source_absent")
        result = run_extension(model_id=EXTENSION, splits={"discovery": discovery, "holdout": absent})
        self.assertIn("no holdout split", result.verdict)
        self.assertFalse(result.verdict_detail["holdout_available"])
        # Discovery still ran, so its estimates are present in the table.
        by_id = {item.hypothesis_id: item for item in result.comparisons}
        self.assertIsNotNone(by_id["H1"].discovery)
        self.assertIsNone(by_id["H1"].holdout)
        report = render_extension_markdown(result)
        self.assertIn("not run", report)
        self.assertIn("raw_source_absent", report)


class ReportTests(unittest.TestCase):
    def test_markdown_carries_the_exploratory_banner_and_every_contrast(self):
        rows, judge = _planted("holdout")
        holdout = analyse_split(rows, judge, model_id=EXTENSION, split="holdout")
        discovery_rows, discovery_judge = _planted("discovery")
        discovery = analyse_split(discovery_rows, discovery_judge, model_id=EXTENSION,
                                  split="discovery")
        primary = {item["hypothesis_id"]: {**item["result"], "supported": item["supported"]}
                   for item in _primary_payload()["result"]["hypotheses"]}
        result = run_extension(
            model_id=EXTENSION, splits={"discovery": discovery, "holdout": holdout},
            primary_hypotheses=primary, primary_model=PRIMARY,
            primary_confirm_source="results/summaries/phase2/confirm.json",
            discovery_contrasts={
                (EXTENSION, "validity_malfunctioning_minus_accurate", "m1", "easy|neutral"): {
                    "mean_difference": "-4.0", "ci95_lower": "-5.0", "ci95_upper": "-3.0",
                    "n_items": "10"},
            },
        )
        report = render_extension_markdown(result)
        self.assertIn(EXTENSION_LABEL, report)
        self.assertIn("EXPLORATORY EXTENSION - NOT PREREGISTERED", report)
        self.assertIn("not preregistered", report)
        self.assertIn(EXTENSION, report)
        self.assertIn("Llama-3.1", report)
        for key in EXTENSION_HYPOTHESIS_IDS:
            self.assertIn("| %s |" % key, report)
        for heading in ("Splits analysed", "Hostile-onset distress mean",
                        "amendment A2 item exclusions", "A4-style pooled missing rates",
                        "non-answer rate by cell and endpoint",
                        "Cross-check against the Phase-1 exploratory contrast table"):
            self.assertIn(heading, report)
        # The cross-check row is present for the one contrast the table names.
        self.assertIn("-4.000 [-5.000, -3.000]", report)
        self.assertEqual(result.family, "Llama-3.1")
        payload = result.to_dict()
        self.assertTrue(json.dumps(payload, allow_nan=False))
        self.assertEqual(payload["label"], EXTENSION_LABEL)
        self.assertEqual(payload["model_id"], EXTENSION)


if __name__ == "__main__":
    unittest.main()
