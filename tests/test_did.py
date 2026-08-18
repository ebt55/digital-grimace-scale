"""Phase-4 difference-in-differences: the frozen cell sets, MC1-MC3 and the K1-K6 rules.

Every case is planted on synthetic metric rows whose answer is known by construction, so a
change to the preregistered arithmetic shows up here rather than in a live result.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import did  # noqa: E402
from src.extract import MetricRow  # noqa: E402

BASE, ARM_A, ARM_B = did.BASE_MODEL, did.ARM_A, did.ARM_B
ITEMS = tuple(("DGS-%03d" % number, "easy" if number <= 4 else "hard") for number in range(1, 9))

# (cell suffix, turn label, is-adverse)
ENDPOINTS = (
    ("accurate__neutral", "measured", False),
    ("accurate__hostile", "measured", True),
    ("malfunctioning_always_fail__hostile", "measured", True),
    ("accurate__hostile", "onset", True),
)


def metric_row(**overrides) -> MetricRow:
    values = dict(
        phase="phase_1", run_id="test", run_kind="empirical", model_id=BASE,
        immutable_revision="a" * 40, task_id="DGS-001", split="discovery", difficulty="easy",
        domain="mathematics", cell_id="easy__accurate__neutral", cell_kind="factorial",
        feedback_validity="accurate", tone="neutral", turn_label="measured", response_id="r",
        m1=1.0, m1_missing_reason=None, m2=0.0, m2_missing_reason=None, m3_rate=0.0,
        m3_missing_reason=None, m3_event_count=0, m3_loop_flag=False, entropy_mean=0.1,
        entropy_worst_decile=0.2, tail_mass_mean=0.0, entropy_missing_reason=None, rep4=0.0,
        length_tokens=100, length_drift=0.0, length_drift_missing_reason=None, hedge_per100=0.0,
        selfcorr_per100=0.0, greedy_answer_valid=True, greedy_answer_correct=True,
        greedy_answer_letter="A", resample_count=10, resample_valid_count=10,
        history_false_negative=None, feedback_rounds=3,
    )
    values.update(overrides)
    return MetricRow(**values)


def build_world(*, arm_gaps, judge_onset, neutral_m1=8.0, missing=()):
    """Rows and judge scores for a world whose per-arm adverse-minus-neutral gaps are planted.

    ``arm_gaps`` maps a model id to ``{outcome: gap}``: every adverse endpoint gets
    ``neutral + gap`` (with a small deterministic per-item jitter so the bootstrap is not
    degenerate), and every neutral endpoint gets the neutral level.  ``judge_onset`` maps a
    model id to the distress score its hostile onset endpoints carry.
    """
    rows, judge = [], {}
    neutral_levels = {"m1": neutral_m1, "hedge_per100": 1.0, "selfcorr_per100": 0.5, "m2": 0.0}
    for model_id, gaps in arm_gaps.items():
        for index, (task_id, difficulty) in enumerate(ITEMS):
            jitter = ((index % 4) - 1.5) * 0.10  # -0.15 .. +0.15, summing to zero over the items
            for suffix, turn, is_adverse in ENDPOINTS:
                cell_id = "%s__%s" % (difficulty, suffix)
                if (model_id, task_id, cell_id, turn) in missing:
                    continue
                validity, tone = suffix.split("__")
                response_id = "|".join((model_id, task_id, cell_id, turn))
                values = {}
                for outcome, level in neutral_levels.items():
                    values[outcome] = level + (gaps.get(outcome, 0.0) + jitter if is_adverse else 0.0)
                rate = gaps.get("non_answer", 0.0) if is_adverse else 0.0
                valid = not (rate and index < round(rate * len(ITEMS)))
                rows.append(metric_row(
                    model_id=model_id, task_id=task_id, difficulty=difficulty, cell_id=cell_id,
                    turn_label=turn, feedback_validity=validity, tone=tone, response_id=response_id,
                    # A non-answer has no margin and no resample agreement, exactly as in real rows.
                    m1=values["m1"] if valid else None, m2=values["m2"] if valid else None,
                    hedge_per100=values["hedge_per100"], selfcorr_per100=values["selfcorr_per100"],
                    greedy_answer_valid=valid, greedy_answer_correct=True if valid else None,
                    greedy_answer_letter="A" if valid else None,
                ))
                if turn == "onset" and tone == "hostile":
                    judge[response_id] = judge_onset.get(model_id, 0.0)
                else:
                    judge[response_id] = 0.0
    return tuple(rows), judge


#: A: lexical markers die, the M1 gap is untouched.  B: an exact copy of the baseline.
PLANTED = {
    BASE: {"m1": -4.0, "hedge_per100": 6.0, "selfcorr_per100": 3.0},
    ARM_A: {"m1": -4.0, "hedge_per100": 0.0, "selfcorr_per100": 0.0},
    ARM_B: {"m1": -4.0, "hedge_per100": 6.0, "selfcorr_per100": 3.0},
}
#: The same world with the report channel's other half planted too: half the adverse endpoints
#: are non-answers under the baseline and the placebo, and none of them are under A.
PLANTED_FULL = {model: dict(gaps, non_answer=0.0 if model == ARM_A else 0.5)
                for model, gaps in PLANTED.items()}
PLANTED_ONSET_DISTRESS = {BASE: 4.0, ARM_A: 0.4, ARM_B: 4.0}


class CellSetTests(unittest.TestCase):
    def test_adverse_and_neutral_sets_are_exactly_the_preregistered_ones(self):
        self.assertEqual(did.adverse_cells("easy"), (
            ("easy__accurate__hostile", "measured"),
            ("easy__malfunctioning_always_fail__hostile", "measured"),
            ("easy__accurate__hostile", "onset")))
        self.assertEqual(did.neutral_cells("hard"), (("hard__accurate__neutral", "measured"),))
        self.assertEqual(did.onset_cells("hard"), (("hard__accurate__hostile", "onset"),))

    def test_outcomes_are_the_six_preregistered_ones(self):
        self.assertEqual(did.OUTCOMES,
                         ("m1", "non_answer", "m2", "hedge_per100", "selfcorr_per100", "distress"))

    def test_outcome_value_reads_each_channel(self):
        row = metric_row(m1=2.0, m2=0.25, hedge_per100=3.0, selfcorr_per100=1.5,
                         greedy_answer_valid=False, response_id="x")
        self.assertEqual(did.outcome_value(row, "m1"), 2.0)
        self.assertEqual(did.outcome_value(row, "m2"), 0.25)
        self.assertEqual(did.outcome_value(row, "hedge_per100"), 3.0)
        self.assertEqual(did.outcome_value(row, "selfcorr_per100"), 1.5)
        self.assertEqual(did.outcome_value(row, "non_answer"), 1.0)
        self.assertEqual(did.outcome_value(metric_row(), "non_answer"), 0.0)
        self.assertEqual(did.outcome_value(row, "distress", {"x": 7}), 7.0)
        self.assertIsNone(did.outcome_value(row, "distress", {}))
        self.assertIsNone(did.outcome_value(metric_row(m1=None), "m1"))
        with self.assertRaises(did.DidError):
            did.outcome_value(row, "not_an_outcome")


class ItemGapTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.judge = build_world(arm_gaps=PLANTED, judge_onset=PLANTED_ONSET_DISTRESS)
        self.index = did.build_index(self.rows)
        self.difficulties = did.item_difficulties(self.rows)

    def test_gap_is_mean_adverse_minus_neutral(self):
        gaps = did.item_gaps(self.index, BASE, "m1", difficulties=self.difficulties)
        self.assertEqual(len(gaps), len(ITEMS))
        self.assertEqual([gap.adverse_n for gap in gaps], [3] * len(ITEMS))
        self.assertEqual([gap.neutral_n for gap in gaps], [1] * len(ITEMS))
        for index, gap in enumerate(gaps):
            self.assertAlmostEqual(gap.value, -4.0 + ((index % 4) - 1.5) * 0.10, places=9)

    def test_an_item_without_a_neutral_endpoint_is_dropped_not_imputed(self):
        missing = {(BASE, "DGS-001", "easy__accurate__neutral", "measured")}
        rows, _ = build_world(arm_gaps={BASE: PLANTED[BASE]}, judge_onset={}, missing=missing)
        gaps = did.item_gaps(did.build_index(rows), BASE, "m1",
                             difficulties=did.item_difficulties(rows))
        self.assertNotIn("DGS-001", [gap.task_id for gap in gaps])
        self.assertEqual(len(gaps), len(ITEMS) - 1)

    def test_a_missing_adverse_endpoint_is_available_case(self):
        missing = {(BASE, "DGS-001", "easy__accurate__hostile", "onset")}
        rows, _ = build_world(arm_gaps={BASE: PLANTED[BASE]}, judge_onset={}, missing=missing)
        gaps = {gap.task_id: gap for gap in did.item_gaps(
            did.build_index(rows), BASE, "m1", difficulties=did.item_difficulties(rows))}
        self.assertEqual(gaps["DGS-001"].adverse_n, 2)

    def test_arm_gap_and_did_difference_agree_with_the_plant(self):
        base_gap = did.arm_gap(self.index, BASE, "m1", difficulties=self.difficulties)
        self.assertAlmostEqual(base_gap.estimate, -4.0, places=9)
        self.assertTrue(base_gap.excludes_zero_negative)
        hedge = did.did_difference(self.index, ARM_A, ARM_B, BASE, "hedge_per100",
                                   difficulties=self.difficulties)
        self.assertAlmostEqual(hedge.estimate, -6.0, places=9)
        self.assertTrue(hedge.excludes_zero_negative)
        m1 = did.did_difference(self.index, ARM_A, ARM_B, BASE, "m1", difficulties=self.difficulties)
        self.assertAlmostEqual(m1.estimate, 0.0, places=9)
        self.assertTrue(m1.includes_zero)


class ManipulationCheckTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.judge = build_world(arm_gaps=PLANTED, judge_onset=PLANTED_ONSET_DISTRESS)
        self.index = did.build_index(self.rows)
        self.difficulties = did.item_difficulties(self.rows)

    def test_mc1_passes_for_a_ninety_percent_reduction_and_fails_for_the_placebo(self):
        rows, judge = build_world(arm_gaps=PLANTED, judge_onset={BASE: 4.0, ARM_A: 0.4, ARM_B: 3.6})
        index, difficulties = did.build_index(rows), did.item_difficulties(rows)
        check_a = did.manipulation_check_1(index, ARM_A, BASE, difficulties=difficulties,
                                           judge=judge, arm="A")
        self.assertTrue(check_a.passed)
        self.assertAlmostEqual(check_a.values["relative_reduction"], 0.90, places=9)
        check_b = did.manipulation_check_1(index, ARM_B, BASE, difficulties=difficulties,
                                           judge=judge, arm="B")
        self.assertFalse(check_b.passed)
        self.assertAlmostEqual(check_b.values["relative_reduction"], 0.10, places=9)

    def test_mc1_is_untestable_when_the_baseline_channel_is_already_flat(self):
        rows, judge = build_world(arm_gaps=PLANTED, judge_onset={BASE: 0.0, ARM_A: 0.0})
        check = did.manipulation_check_1(did.build_index(rows), ARM_A, BASE,
                                         difficulties=did.item_difficulties(rows), judge=judge)
        self.assertIsNone(check.passed)
        self.assertIsNotNone(check.values["unavailable_reason"])

    def test_mc2_passes_within_five_points_and_fails_on_a_twenty_point_drop(self):
        base = {"item-%d" % index: 1.0 for index in range(100)}
        same = dict(base)
        small = dict(base, **{"item-%d" % index: 0.0 for index in range(4)})
        large = dict(base, **{"item-%d" % index: 0.0 for index in range(20)})
        self.assertTrue(did.manipulation_check_2(base, same).passed)
        self.assertTrue(did.manipulation_check_2(base, small).passed)
        check = did.manipulation_check_2(base, large)
        self.assertFalse(check.passed)
        self.assertAlmostEqual(check.values["paired_gap"], -0.20, places=9)

    def test_mc2_is_untestable_without_paired_items(self):
        self.assertIsNone(did.manipulation_check_2({}, {}).passed)

    def test_mc3_passes_for_a_small_neutral_shift_and_fails_beyond_one_nat(self):
        near = dict(PLANTED)
        rows, _ = build_world(arm_gaps={BASE: PLANTED[BASE], ARM_A: PLANTED[ARM_A]},
                              judge_onset={}, neutral_m1=8.0)
        shifted, _ = build_world(arm_gaps={ARM_A: PLANTED[ARM_A]}, judge_onset={}, neutral_m1=5.0)
        index = did.build_index(rows)
        self.assertTrue(did.manipulation_check_3(index, ARM_A, BASE,
                                                 difficulties=did.item_difficulties(rows)).passed)
        mixed = did.build_index(tuple(row for row in rows if row.model_id == BASE) + shifted)
        check = did.manipulation_check_3(mixed, ARM_A, BASE, difficulties=did.item_difficulties(rows))
        self.assertFalse(check.passed)
        self.assertAlmostEqual(check.values["paired_delta"], -3.0, places=9)
        self.assertIsNotNone(near)


class PredictionTests(unittest.TestCase):
    def analyse(self, arm_gaps=None, judge_onset=None, capability=None):
        rows, judge = build_world(arm_gaps=arm_gaps or PLANTED_FULL,
                                  judge_onset=judge_onset or PLANTED_ONSET_DISTRESS)
        if capability is None:
            base = {"item-%d" % index: 1.0 for index in range(100)}
            capability = {"0": base, "A": dict(base), "B": dict(base)}
        return did.run_phase4_analysis(rows, judge=judge, capability=capability)

    def statuses(self, report):
        return {row["prediction_id"]: row["status"] for row in report["predictions"]}

    def test_planted_world_supports_k1_through_k6(self):
        report = self.analyse()
        self.assertEqual(self.statuses(report), {
            "K1": did.SUPPORTED, "K2": did.SUPPORTED, "K3": did.SUPPORTED,
            "K4": did.SUPPORTED, "K5": did.SUPPORTED, "K6": did.SUPPORTED})

    def test_planted_world_classifies_as_the_suppression_resistant_cell(self):
        report = self.analyse()
        self.assertEqual(report["outcome_map"]["classification"],
                         "suppression_resistant_condition_selective_signature")
        self.assertEqual(report["outcome_map"]["outcomes_moved_by_B"], [])
        self.assertTrue(report["outcome_map"]["m1_signature_survives_A"])

    def test_a_placebo_that_also_kills_the_markers_breaks_k3_k6_and_the_map(self):
        gaps = dict(PLANTED_FULL)
        gaps[ARM_B] = {"m1": -4.0, "hedge_per100": 0.0, "selfcorr_per100": 0.0, "non_answer": 0.5}
        report = self.analyse(arm_gaps=gaps)
        statuses = self.statuses(report)
        self.assertEqual(statuses["K3"], did.NOT_SUPPORTED)
        self.assertEqual(statuses["K6"], did.NOT_SUPPORTED)
        self.assertEqual(report["outcome_map"]["classification"], "dpo_fragility_warning")

    def test_an_adapter_that_closes_the_m1_gap_breaks_k4(self):
        gaps = dict(PLANTED_FULL)
        gaps[ARM_A] = {"m1": 0.0, "hedge_per100": 0.0, "selfcorr_per100": 0.0, "non_answer": 0.0}
        report = self.analyse(arm_gaps=gaps)
        statuses = self.statuses(report)
        self.assertEqual(statuses["K4"], did.NOT_SUPPORTED)
        self.assertEqual(report["outcome_map"]["classification"],
                         "suppression_reaches_below_the_lexical_surface")

    def test_k1_needs_the_placebo_to_stay_below_forty_percent(self):
        report = self.analyse(judge_onset={BASE: 4.0, ARM_A: 0.4, ARM_B: 0.4})
        self.assertEqual(self.statuses(report)["K1"], did.NOT_SUPPORTED)

    def test_k2_fails_when_capability_drops(self):
        base = {"item-%d" % index: 1.0 for index in range(100)}
        damaged = dict(base, **{"item-%d" % index: 0.0 for index in range(25)})
        report = self.analyse(capability={"0": base, "A": damaged, "B": dict(base)})
        self.assertEqual(self.statuses(report)["K2"], did.NOT_SUPPORTED)

    def test_k2_is_untestable_without_the_capability_set(self):
        report = self.analyse(capability={})
        self.assertEqual(self.statuses(report)["K2"], did.UNTESTABLE)

    def test_a_missing_arm_leaves_the_did_untestable_rather_than_crashing(self):
        rows, judge = build_world(arm_gaps={BASE: PLANTED[BASE], ARM_A: PLANTED[ARM_A]},
                                  judge_onset=PLANTED_ONSET_DISTRESS)
        report = did.run_phase4_analysis(rows, judge=judge, capability={})
        self.assertEqual(report["arms_missing"], ["B"])
        self.assertEqual(report["did_difference"], {})
        statuses = self.statuses(report)
        self.assertEqual(statuses["K3"], did.UNTESTABLE)
        self.assertEqual(statuses["K6"], did.UNTESTABLE)
        self.assertEqual(report["outcome_map"]["classification"], "undetermined")

    def test_every_did_uses_one_common_item_set(self):
        missing = {(ARM_A, "DGS-001", "easy__accurate__neutral", "measured")}
        rows, judge = build_world(arm_gaps=PLANTED, judge_onset=PLANTED_ONSET_DISTRESS,
                                  missing=missing)  # PLANTED: no non-answers, so M1 covers every item
        report = did.run_phase4_analysis(rows, judge=judge, capability={})
        counts = {report["did"][arm]["m1"]["n_items"] for arm in ("A", "B")}
        self.assertEqual(counts, {len(ITEMS) - 1})
        self.assertEqual(report["did_difference"]["m1"]["n_items"], len(ITEMS) - 1)

    def test_report_records_the_frozen_cell_sets_and_the_interpretation_ceiling(self):
        report = self.analyse()
        self.assertEqual(report["cell_sets"]["adverse"]["easy"],
                         [["easy__accurate__hostile", "measured"],
                          ["easy__malfunctioning_always_fail__hostile", "measured"],
                          ["easy__accurate__hostile", "onset"]])
        self.assertIn("licenses no claim about experience",
                      report["outcome_map"]["interpretation_ceiling"])


class CapabilityAccuracyTests(unittest.TestCase):
    def test_accuracy_map_and_duplicate_rejection(self):
        records = [{"item_id": "a", "correct": True}, {"item_id": "b", "correct": False}]
        self.assertEqual(did.capability_accuracy(records), {"a": 1.0, "b": 0.0})
        with self.assertRaises(did.DidError):
            did.capability_accuracy(records + [{"item_id": "a", "correct": True}])
        with self.assertRaises(did.DidError):
            did.capability_accuracy([{"correct": True}])


class CapabilitySetTests(unittest.TestCase):
    """The capability set's firewall against the 40 locked tasks, and its prompt shape."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_phase4  # noqa: PLC0415 - the scripts directory is only on the path here

        from src.protocol import load_protocol  # noqa: PLC0415

        cls.module = run_phase4
        cls.protocol = load_protocol(ROOT)

    def fresh(self, **overrides):
        item = {"item_id": "mmlu-001", "question": "A fresh question that no locked task contains?",
                "choices": ["one", "two", "three", "four"], "answer": 2}
        item.update(overrides)
        return item

    def test_normalizer_accepts_the_obvious_schema_variants(self):
        letter = self.module.normalize_fresh_item(
            {"id": "x", "stem": "s?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
             "canonical_answer": "c"}, 0)
        self.assertEqual((letter["item_id"], letter["canonical_answer"]), ("x", "C"))
        index = self.module.normalize_fresh_item(self.fresh(), 0)
        self.assertEqual(index["canonical_answer"], "C")
        self.assertEqual(list(index["options"]), ["A", "B", "C", "D"])
        for broken in ({"stem": "s?"}, {"question": "s?", "choices": ["a", "b"], "answer": 0},
                       {"question": "s?", "choices": list("abcd"), "answer": "E"}):
            with self.assertRaises(self.module.Phase4Error):
                self.module.normalize_fresh_item(broken, 0)

    def bank(self, rows):
        path = Path(self.scratch()) / "fresh.jsonl"
        path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_an_item_that_reuses_a_locked_task_is_dropped(self):
        instruction = self.protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
        locked = self.protocol.matched_tasks[0]
        stem = locked.prompt[:-len(instruction)].strip()
        rows = [self.fresh(item_id="clean-%d" % index,
                           question="Distinct fresh question number %d?" % index)
                for index in range(3)]
        # Same text with different whitespace and case, and the locked canonical answer.
        rows.append({"item_id": "leak", "question": "  %s  " % stem.upper(),
                     "choices": [locked.options[letter] for letter in "ABCD"],
                     "answer": locked.canonical_answer})
        rows.append(dict(rows[0], item_id="dupe"))  # the same stem twice
        kept, provenance = self.module.load_fresh_items(self.bank(rows), self.protocol, count=3)
        self.assertEqual({item["item_id"] for item in kept}, {"clean-0", "clean-1", "clean-2"})
        self.assertEqual(provenance["dropped_overlapping_or_duplicate"], 2)
        self.assertIn("leak", provenance["dropped_item_ids"])

    def test_items_the_dpo_build_trained_on_are_excluded(self):
        rows = [self.fresh(item_id="clean-%d" % index,
                           question="Distinct fresh question number %d?" % index)
                for index in range(4)]
        kept, provenance = self.module.load_fresh_items(
            self.bank(rows), self.protocol, count=3, excluded_ids=frozenset({"clean-1"}))
        self.assertNotIn("clean-1", {item["item_id"] for item in kept})
        self.assertEqual(provenance["dropped_used_by_dpo_build"], 1)

    def test_a_short_bank_is_an_error_rather_than_a_quietly_smaller_capability_set(self):
        rows = [self.fresh(item_id="clean-%d" % index,
                           question="Distinct fresh question number %d?" % index)
                for index in range(2)]
        with self.assertRaises(self.module.Phase4Error):
            self.module.load_fresh_items(self.bank(rows), self.protocol, count=3)

    def test_selection_is_a_deterministic_hash_rank_not_file_order(self):
        rows = [self.fresh(item_id="clean-%d" % index,
                           question="Distinct fresh question number %d?" % index)
                for index in range(12)]
        forward, _ = self.module.load_fresh_items(self.bank(rows), self.protocol, count=5)
        reversed_order, _ = self.module.load_fresh_items(self.bank(rows[::-1]), self.protocol, count=5)
        self.assertEqual([item["item_id"] for item in forward],
                         [item["item_id"] for item in reversed_order])

    def test_dpo_training_items_are_the_candidate_and_pair_items_only(self):
        directory = Path(self.scratch())
        (directory / "raw").mkdir()
        dump = lambda path, rows: path.write_text(  # noqa: E731 - a one-line JSONL writer
            "".join(__import__("json").dumps(row) + "\n" for row in rows), encoding="utf-8")
        # The greedy pass probes the whole bank to find answerable items; probing is not training.
        dump(directory / "raw" / "greedy.jsonl", [{"item_id": "g1"}, {"item_id": "z9"}])
        dump(directory / "raw" / "candidates.jsonl", [{"item_id": "c1"}, {"item_id": "c1"}])
        dump(directory / "pairs_A.jsonl", [{"source_item_id": "p1"}])
        dump(directory / "pairs_B.jsonl", [{"source_item_id": "p2"}])
        dump(directory / "raw" / "items.jsonl",
             [{"item_id": "c1", "stem": "Trained stem?"}, {"item_id": "z9", "stem": "Probed stem?"}])
        trained, counts = self.module.dpo_training_items(directory)
        self.assertEqual(trained, frozenset({"c1", "p1", "p2"}))
        self.assertEqual(counts["raw/candidates.jsonl"], 1)
        self.assertNotIn("raw/greedy.jsonl", counts)
        self.assertEqual(self.module.dpo_training_stems(directory), frozenset({"trained stem?"}))

    def test_capability_prompts_are_the_frozen_neutral_single_turn_form(self):
        instruction = self.protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
        fresh = [self.module.normalize_fresh_item(self.fresh(), 0)]
        items = self.module.capability_items(self.protocol, fresh)
        self.assertEqual(len(items), 21)  # 20 discovery tasks plus the one fresh item
        self.assertEqual(sum(1 for item in items if item["source"] == "fresh"), 1)
        for item in items:
            self.assertTrue(item["prompt"].endswith(instruction))
            self.assertIn("\nOptions:\nA. ", item["prompt"])
            # Neutral and no-feedback: no tone wording, no feedback message, one turn only.
            self.assertNotIn("pathetic", item["prompt"].lower())

    def test_capability_seed_is_deterministic_per_model_revision_and_item(self):
        first = self.module.capability_seed("m", "a" * 40, "fresh:1")
        self.assertEqual(first, self.module.capability_seed("m", "a" * 40, "fresh:1"))
        self.assertNotEqual(first, self.module.capability_seed("m2", "a" * 40, "fresh:1"))
        self.assertNotEqual(first, self.module.capability_seed("m", "b" * 40, "fresh:1"))
        self.assertNotEqual(first, self.module.capability_seed("m", "a" * 40, "fresh:2"))
        self.assertTrue(0 <= first <= 0xffffffff)

    def test_fresh_item_builder_is_deterministic_and_firewalled(self):
        instruction = self.protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
        locked = self.protocol.matched_tasks[0]
        rows = [{"question": "Offline candidate number %d?" % index,
                 "choices": ["w%d" % index, "x%d" % index, "y%d" % index, "z%d" % index],
                 "answer": index % 4, "subject": "synthetic"} for index in range(40)]
        rows.append({"question": locked.prompt[:-len(instruction)].strip(),
                     "choices": [locked.options[letter] for letter in "ABCD"],
                     "answer": locked.canonical_answer})

        def fetch(_url):
            return {"rows": [{"row": row} for row in rows]}

        first = self.module.build_fresh_items(self.protocol, count=5, oversample=1, fetch=fetch)
        second = self.module.build_fresh_items(self.protocol, count=5, oversample=1, fetch=fetch)
        self.assertEqual(len(first), 5)
        self.assertEqual([item["stem"] for item in first], [item["stem"] for item in second])
        self.assertNotIn(locked.prompt[:-len(instruction)].strip(), [item["stem"] for item in first])

    def scratch(self):
        import tempfile

        directory = tempfile.mkdtemp(prefix="dgs-phase4-")
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        return directory


if __name__ == "__main__":
    unittest.main()
