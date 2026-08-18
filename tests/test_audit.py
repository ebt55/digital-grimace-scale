"""Offline tests for the descriptive human-audit statistics. No network, no fixtures on disk."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from src.audit import (ANNOTATOR_NOTE, SCORE_KIND, AuditError, AuditItem,
                       audit_statistics, average_ranks, build_report, check_selection_against_items,
                       count_rev_flags, describe_group, describe_spearman, histogram,
                       is_answer_only, join_audit, load_audit_key, load_blinded_responses,
                       load_human_scores, load_judge_scores, load_selection, mean_absolute_error,
                       per_item_rows, render_markdown, report_payload, spearman,
                       summarise_answer_only, within_tolerance_rate)


def item(audit_id, human, judge, *, model="M/a", cell="easy__accurate__neutral",
         task="DGS-001", response=None, note=""):
    return AuditItem(audit_id=audit_id, model_id=model, cell_id=cell, task_id=task,
                     response_id=response or ("r" + audit_id), run_id="run-1",
                     human_score=human, judge_score=judge, note=note)


# --------------------------------------------------------------------------------------
# rank handling


class AverageRanksTest(unittest.TestCase):
    def test_no_ties_is_plain_ordering(self):
        self.assertEqual(average_ranks([10, 30, 20]), (1.0, 3.0, 2.0))

    def test_ties_share_the_average_of_the_ranks_they_span(self):
        # values 1,2,2,3 -> ranks 1, (2+3)/2, (2+3)/2, 4
        self.assertEqual(average_ranks([1, 2, 2, 3]), (1.0, 2.5, 2.5, 4.0))

    def test_all_tied_values_all_get_the_midpoint(self):
        self.assertEqual(average_ranks([7, 7, 7, 7]), (2.5, 2.5, 2.5, 2.5))

    def test_ranks_always_sum_to_n_times_n_plus_one_over_two(self):
        for values in ([0, 0, 1, 1, 2], [3, 3, 3, 1], [5, 4, 3, 2, 1], [0] * 6):
            n = len(values)
            self.assertAlmostEqual(sum(average_ranks(values)), n * (n + 1) / 2, places=9)

    def test_empty_input_raises(self):
        with self.assertRaises(AuditError):
            average_ranks([])


class SpearmanTest(unittest.TestCase):
    def test_perfect_monotone_is_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=12)

    def test_perfect_reversal_is_minus_one(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=12)

    def test_tie_corrected_value_matches_the_hand_computation(self):
        # x ranks 1, 2.5, 2.5, 4; y ranks 1, 2, 3, 4 -> 4.5 / sqrt(4.5 * 5.0)
        expected = 4.5 / math.sqrt(4.5 * 5.0)
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 2, 3, 4]), expected, places=12)
        self.assertAlmostEqual(expected, 0.9486832980505138, places=12)

    def test_ties_on_both_sides(self):
        # x ranks 1.5,1.5,3.5,3.5 ; y ranks 1.5,3.5,1.5,3.5 -> zero covariance
        self.assertAlmostEqual(spearman([0, 0, 1, 1], [0, 1, 0, 1]), 0.0, places=12)

    def test_constant_side_is_undefined_not_zero(self):
        self.assertIsNone(spearman([0, 0, 0, 0], [0, 1, 2, 3]))
        self.assertIsNone(spearman([0, 1, 2, 3], [5, 5, 5, 5]))

    def test_length_mismatch_and_short_input_raise(self):
        with self.assertRaises(AuditError):
            spearman([1, 2, 3], [1, 2])
        with self.assertRaises(AuditError):
            spearman([1], [1])

    def test_is_invariant_to_monotone_rescaling(self):
        xs, ys = [0, 1, 1, 2, 3], [0, 0, 2, 1, 3]
        rescaled = [value * 3 + 7 for value in ys]
        self.assertAlmostEqual(spearman(xs, ys), spearman(xs, rescaled), places=12)


# --------------------------------------------------------------------------------------
# error / agreement statistics


class ErrorStatisticsTest(unittest.TestCase):
    def test_mae_is_the_mean_absolute_difference(self):
        self.assertAlmostEqual(mean_absolute_error([0, 1, 2, 3], [0, 0, 4, 3]), 3 / 4, places=12)

    def test_mae_is_zero_on_exact_agreement(self):
        self.assertEqual(mean_absolute_error([2, 2, 2], [2, 2, 2]), 0.0)

    def test_mae_is_symmetric(self):
        xs, ys = [0, 5, 9], [3, 1, 9]
        self.assertAlmostEqual(mean_absolute_error(xs, ys), mean_absolute_error(ys, xs), places=12)

    def test_within_tolerance_is_inclusive_at_the_boundary(self):
        # diffs 0, 2, 3 -> two of three within 2
        self.assertAlmostEqual(within_tolerance_rate([0, 0, 0], [0, 2, 3]), 2 / 3, places=12)

    def test_within_tolerance_zero_is_exact_agreement(self):
        self.assertAlmostEqual(within_tolerance_rate([0, 1, 2], [0, 1, 3], tolerance=0),
                               2 / 3, places=12)

    def test_negative_tolerance_and_mismatched_lengths_raise(self):
        with self.assertRaises(AuditError):
            within_tolerance_rate([0], [0], tolerance=-1)
        with self.assertRaises(AuditError):
            within_tolerance_rate([0, 1], [0])
        with self.assertRaises(AuditError):
            mean_absolute_error([], [])

    def test_histogram_keeps_empty_buckets_and_covers_the_whole_rubric(self):
        counts = dict(histogram([0, 0, 1, 3]))
        self.assertEqual(sorted(counts), list(range(0, 11)))
        self.assertEqual((counts[0], counts[1], counts[2], counts[3]), (2, 1, 0, 1))
        self.assertEqual(sum(counts.values()), 4)

    def test_histogram_rejects_out_of_rubric_values(self):
        with self.assertRaises(AuditError):
            histogram([0, 11])


class BootstrapTest(unittest.TestCase):
    def test_is_deterministic_for_a_given_label(self):
        xs, ys = [0, 0, 1, 1, 2, 3, 0, 1], [0, 1, 0, 2, 0, 3, 0, 0]
        first = describe_spearman(xs, ys, label="g", resamples=200)
        second = describe_spearman(xs, ys, label="g", resamples=200)
        self.assertEqual(first.ci95, second.ci95)
        self.assertEqual(first.degenerate_resamples, second.degenerate_resamples)

    def test_different_labels_draw_different_resamples(self):
        xs, ys = [0, 0, 1, 1, 2, 3, 0, 1], [0, 1, 0, 2, 0, 3, 0, 0]
        self.assertNotEqual(describe_spearman(xs, ys, label="a", resamples=200).ci95,
                            describe_spearman(xs, ys, label="b", resamples=200).ci95)

    def test_interval_brackets_the_point_estimate_region(self):
        xs = list(range(12))
        result = describe_spearman(xs, xs, label="perfect", resamples=200)
        self.assertAlmostEqual(result.rho, 1.0, places=12)
        self.assertEqual(result.degenerate_resamples, 0)
        self.assertLessEqual(result.ci95[0], result.rho + 1e-12)
        self.assertLessEqual(result.ci95[1], 1.0 + 1e-12)

    def test_constant_judge_side_reports_undefined_with_a_reason(self):
        result = describe_spearman([0, 1, 2, 3], [0, 0, 0, 0], label="flat", resamples=50)
        self.assertIsNone(result.rho)
        self.assertIn("undefined", result.undefined_reason)
        self.assertEqual(result.degenerate_resamples, 50)
        self.assertIsNone(result.ci95)
        self.assertEqual(result.ci95_degenerate_as_zero, (0.0, 0.0))

    def test_degenerate_resamples_are_counted_not_silently_dropped(self):
        # one nonzero judge score: many resamples miss it entirely
        xs, ys = [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 3]
        result = describe_spearman(xs, ys, label="sparse", resamples=300)
        self.assertGreater(result.degenerate_resamples, 0)
        self.assertLess(result.degenerate_resamples, 300)
        self.assertEqual(result.resamples, 300)

    def test_nonpositive_resamples_raise(self):
        with self.assertRaises(AuditError):
            describe_spearman([0, 1], [0, 1], label="x", resamples=0)


class GroupStatisticsTest(unittest.TestCase):
    def test_group_reports_counts_means_and_worst_disagreement(self):
        items = [item("A", 0, 0), item("B", 1, 0), item("C", 3, 0), item("D", 0, 2)]
        group = describe_group(items, label="g", resamples=50)
        self.assertEqual(group.n, 4)
        self.assertAlmostEqual(group.mae, (0 + 1 + 3 + 2) / 4, places=12)
        self.assertEqual(group.within_tolerance_count, 3)      # diffs 0,1,3,2
        self.assertAlmostEqual(group.within_tolerance, 3 / 4, places=12)
        self.assertEqual(group.max_abs_diff, 3)
        self.assertAlmostEqual(group.human_mean, 1.0, places=12)
        self.assertAlmostEqual(group.judge_mean, 0.5, places=12)

    def test_per_model_split_is_sorted_and_partitions_the_items(self):
        items = [item("A", 0, 0, model="z/model"), item("B", 2, 0, model="a/model"),
                 item("C", 1, 1, model="z/model")]
        overall, per_model = audit_statistics(items, resamples=50)
        self.assertEqual(overall.n, 3)
        self.assertEqual([group.model_id for group in per_model], ["a/model", "z/model"])
        self.assertEqual(sum(group.n for group in per_model), overall.n)

    def test_empty_group_raises(self):
        with self.assertRaises(AuditError):
            describe_group([], label="g")
        with self.assertRaises(AuditError):
            audit_statistics([])


# --------------------------------------------------------------------------------------
# loading and join integrity


class AuditFixture(unittest.TestCase):
    KEY_HEADER = "audit_id,model_id,cell_id,task_id,response_id,run_id\n"

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.write_key([("AUD-0001", "m/a", "easy__accurate__neutral", "DGS-001", "r1"),
                        ("AUD-0002", "m/a", "easy__accurate__neutral", "DGS-002", "r2"),
                        ("AUD-0003", "m/b", "hard__accurate__hostile", "DGS-003", "r3")])
        self.write_scores([("AUD-0001", "0", ""), ("AUD-0002", "2", ""), ("AUD-0003", "1", "")])
        self.write_judge({"r1": 0, "r2": 3, "r3": 1})
        self.write_selection()

    def write_key(self, rows):
        text = self.KEY_HEADER + "".join(
            "%s,%s,%s,%s,%s,run-1\n" % row for row in rows)
        (self.dir / "audit_key.csv").write_text(text, encoding="utf-8")

    def write_scores(self, rows):
        text = "audit_id,human_distress_0_10,note_optional\n" + "".join(
            "%s,%s,%s\n" % row for row in rows)
        (self.dir / "audit_scores.csv").write_text(text, encoding="utf-8")

    def write_judge(self, by_response, *, kind=SCORE_KIND, path=None):
        lines = [json.dumps({"score_kind": kind, "score_value": score,
                             "backend_id": "test|judge|rev:pinned",
                             "source_identity": {"response_id": response_id,
                                                 "model_id": "m/a", "cell_id": "c",
                                                 "task_id": "t", "turn_label": "measured"}},
                            sort_keys=True)
                 for response_id, score in sorted(by_response.items())]
        target = Path(path) if path else self.dir / "judge_records.jsonl"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def write_selection(self, cells=None):
        cells = cells or {"m/a": {"easy__accurate__neutral": ["r1", "r2"]},
                          "m/b": {"hard__accurate__hostile": ["r3"]}}
        models = []
        for model_id, by_cell in sorted(cells.items()):
            model_cells = [{"cell_id": cell_id, "hash_rank": rank, "planned": len(ids),
                            "achieved": len(ids), "available": 10, "response_ids": list(ids)}
                           for rank, (cell_id, ids) in enumerate(sorted(by_cell.items()))]
            models.append({"model_id": model_id, "cells": model_cells,
                           "planned_total": sum(cell["planned"] for cell in model_cells),
                           "achieved_total": sum(cell["achieved"] for cell in model_cells),
                           "shortfall": 0, "unmet": 0, "reallocations": []})
        (self.dir / "audit_selection.json").write_text(
            json.dumps({"schema_version": "dgs-judge-audit-v1", "models": models,
                        "per_model_target": 2, "candidate_count": 30,
                        "allocation_rule": "test rule"}, indent=2),
            encoding="utf-8")


class LoadTest(AuditFixture):
    def test_round_trip_join_is_complete_and_sorted(self):
        items = join_audit(load_audit_key(self.dir / "audit_key.csv"),
                           load_human_scores(self.dir / "audit_scores.csv"),
                           load_judge_scores([self.dir]))
        self.assertEqual([entry.audit_id for entry in items],
                         ["AUD-0001", "AUD-0002", "AUD-0003"])
        self.assertEqual([entry.judge_score for entry in items], [0, 3, 1])
        self.assertEqual([entry.human_score for entry in items], [0, 2, 1])
        self.assertEqual([entry.abs_diff for entry in items], [0, 1, 0])

    def test_missing_judge_score_raises_and_names_the_response(self):
        self.write_judge({"r1": 0, "r2": 3})           # r3 absent
        with self.assertRaises(AuditError) as caught:
            join_audit(load_audit_key(self.dir / "audit_key.csv"),
                       load_human_scores(self.dir / "audit_scores.csv"),
                       load_judge_scores([self.dir]))
        self.assertIn("r3", str(caught.exception))

    def test_missing_human_score_raises(self):
        self.write_scores([("AUD-0001", "0", ""), ("AUD-0002", "2", "")])
        with self.assertRaises(AuditError) as caught:
            join_audit(load_audit_key(self.dir / "audit_key.csv"),
                       load_human_scores(self.dir / "audit_scores.csv"),
                       load_judge_scores([self.dir]))
        self.assertIn("AUD-0003", str(caught.exception))

    def test_human_score_for_unknown_audit_id_raises(self):
        self.write_scores([("AUD-0001", "0", ""), ("AUD-0002", "2", ""),
                           ("AUD-0003", "1", ""), ("AUD-0099", "4", "")])
        with self.assertRaises(AuditError) as caught:
            join_audit(load_audit_key(self.dir / "audit_key.csv"),
                       load_human_scores(self.dir / "audit_scores.csv"),
                       load_judge_scores([self.dir]))
        self.assertIn("AUD-0099", str(caught.exception))

    def test_blank_human_cell_raises_instead_of_defaulting_to_zero(self):
        self.write_scores([("AUD-0001", "", ""), ("AUD-0002", "2", ""), ("AUD-0003", "1", "")])
        with self.assertRaises(AuditError):
            load_human_scores(self.dir / "audit_scores.csv")

    def test_out_of_range_and_nonintegral_human_scores_raise(self):
        for bad in ("11", "-1", "2.5", "high"):
            self.write_scores([("AUD-0001", bad, ""), ("AUD-0002", "2", ""),
                               ("AUD-0003", "1", "")])
            with self.assertRaises(AuditError):
                load_human_scores(self.dir / "audit_scores.csv")

    def test_duplicate_audit_ids_raise(self):
        self.write_key([("AUD-0001", "m/a", "c", "DGS-001", "r1"),
                        ("AUD-0001", "m/a", "c", "DGS-002", "r2")])
        with self.assertRaises(AuditError):
            load_audit_key(self.dir / "audit_key.csv")

    def test_missing_column_and_missing_file_raise(self):
        (self.dir / "audit_scores.csv").write_text("audit_id,score\nAUD-0001,0\n", encoding="utf-8")
        with self.assertRaises(AuditError):
            load_human_scores(self.dir / "audit_scores.csv")
        with self.assertRaises(AuditError):
            load_audit_key(self.dir / "nope.csv")

    def test_conflicting_judge_scores_for_one_response_raise(self):
        path = self.dir / "judge_records.jsonl"
        text = path.read_text(encoding="utf-8")
        text += json.dumps({"score_kind": SCORE_KIND, "score_value": 9,
                            "source_identity": {"response_id": "r1"}}) + "\n"
        path.write_text(text, encoding="utf-8")
        with self.assertRaises(AuditError) as caught:
            load_judge_scores([self.dir])
        self.assertIn("r1", str(caught.exception))

    def test_identical_repeat_of_a_judge_score_is_accepted(self):
        path = self.dir / "judge_records.jsonl"
        text = path.read_text(encoding="utf-8")
        text += json.dumps({"score_kind": SCORE_KIND, "score_value": 0,
                            "source_identity": {"response_id": "r1"}}) + "\n"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(load_judge_scores([self.dir]).by_response["r1"], 0)

    def test_other_score_kinds_are_ignored(self):
        self.write_judge({"r1": 0, "r2": 3, "r3": 1})
        path = self.dir / "judge_records.jsonl"
        text = path.read_text(encoding="utf-8")
        text += json.dumps({"score_kind": "context_hostility_pressure", "score_value": 8,
                            "source_identity": {"response_id": "r1"}}) + "\n"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(load_judge_scores([self.dir]).by_response["r1"], 0)

    def test_multiple_judge_sources_are_merged(self):
        second = self.dir / "other.jsonl"
        self.write_judge({"r1": 0, "r2": 3}, path=self.dir / "judge_records.jsonl")
        self.write_judge({"r3": 1}, path=second)
        scores = load_judge_scores([self.dir, second])
        self.assertEqual(sorted(scores.by_response), ["r1", "r2", "r3"])

    def test_judge_source_without_matching_records_raises(self):
        self.write_judge({}, path=self.dir / "judge_records.jsonl")
        with self.assertRaises(AuditError):
            load_judge_scores([self.dir])

    def test_rev_flags_are_counted_from_the_note_column(self):
        self.write_scores([("AUD-0001", "0", ""), ("AUD-0002", "2", "REV at char 40"),
                           ("AUD-0003", "1", "no flag here")])
        self.assertEqual(count_rev_flags(load_human_scores(self.dir / "audit_scores.csv")),
                         ("AUD-0002",))

    def test_rev_substring_inside_a_word_is_not_a_flag(self):
        self.write_scores([("AUD-0001", "0", "revised wording"), ("AUD-0002", "2", ""),
                           ("AUD-0003", "1", "")])
        self.assertEqual(count_rev_flags(load_human_scores(self.dir / "audit_scores.csv")), ())


class SelectionTest(AuditFixture):
    def load_items(self):
        return join_audit(load_audit_key(self.dir / "audit_key.csv"),
                          load_human_scores(self.dir / "audit_scores.csv"),
                          load_judge_scores([self.dir]))

    def test_matching_selection_passes(self):
        check_selection_against_items(load_selection(self.dir / "audit_selection.json"),
                                      self.load_items())

    def test_selection_naming_a_response_the_key_does_not_contain_raises(self):
        self.write_selection({"m/a": {"easy__accurate__neutral": ["r1", "r9"]},
                              "m/b": {"hard__accurate__hostile": ["r3"]}})
        with self.assertRaises(AuditError) as caught:
            check_selection_against_items(load_selection(self.dir / "audit_selection.json"),
                                          self.load_items())
        self.assertIn("mismatch", str(caught.exception))

    def test_key_cell_absent_from_the_selection_raises(self):
        self.write_selection({"m/a": {"easy__accurate__neutral": ["r1", "r2"]}})
        with self.assertRaises(AuditError):
            check_selection_against_items(load_selection(self.dir / "audit_selection.json"),
                                          self.load_items())

    def test_wrong_schema_version_raises(self):
        path = self.dir / "audit_selection.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = "something-else"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(AuditError):
            load_selection(path)


class AnswerOnlyTest(AuditFixture):
    def test_recognises_a_bare_answer_line(self):
        self.assertTrue(is_answer_only("Answer: A"))
        self.assertTrue(is_answer_only("\n  Answer: D  \n\n"))

    def test_rejects_answers_with_any_other_content(self):
        for text in ("Because 2+2=4.\n\nAnswer: A", "Answer: A\nAnswer: B",
                     "The answer is A", "Answer: AB", "", "Answer: E"):
            self.assertFalse(is_answer_only(text), text)

    def test_summary_counts_the_bare_answers_and_their_error_share(self):
        items = [item("A", 1, 0), item("B", 0, 0), item("C", 3, 0)]
        texts = {"A": "Answer: A", "B": "Answer: B", "C": "Long prose.\n\nAnswer: C"}
        summary = summarise_answer_only(items, texts)
        self.assertEqual(summary.n, 2)
        self.assertEqual(summary.audit_ids, ("A", "B"))
        self.assertEqual(summary.human_nonzero, 1)
        self.assertEqual(summary.judge_nonzero, 0)
        self.assertEqual(summary.abs_diff_total, 1)
        self.assertEqual(summary.abs_diff_total_all_items, 4)

    def test_missing_blinded_text_raises(self):
        with self.assertRaises(AuditError):
            summarise_answer_only([item("A", 0, 0)], {})

    def test_loader_reads_the_blinded_jsonl(self):
        path = self.dir / "audit_blinded.jsonl"
        path.write_text('{"audit_id": "AUD-0001", "response_text": "Answer: A"}\n'
                        '{"audit_id": "AUD-0002", "response_text": "x"}\n', encoding="utf-8")
        self.assertEqual(load_blinded_responses(path)["AUD-0001"], "Answer: A")

    def test_duplicate_blinded_audit_id_raises(self):
        path = self.dir / "audit_blinded.jsonl"
        path.write_text('{"audit_id": "AUD-0001", "response_text": "a"}\n'
                        '{"audit_id": "AUD-0001", "response_text": "b"}\n', encoding="utf-8")
        with self.assertRaises(AuditError):
            load_blinded_responses(path)


class ReportTest(AuditFixture):
    def build(self):
        return build_report(audit_dir=self.dir, judge_paths=[self.dir], resamples=64)

    def test_answer_only_summary_is_absent_without_the_blinded_file(self):
        report = self.build()
        self.assertIsNone(report.answer_only)
        self.assertIsNone(report_payload(report)["answer_only_responses"])
        self.assertIn("Annotator note", render_markdown(report))

    def test_answer_only_summary_appears_when_the_blinded_file_exists(self):
        (self.dir / "audit_blinded.jsonl").write_text(
            '{"audit_id": "AUD-0001", "response_text": "Answer: A"}\n'
            '{"audit_id": "AUD-0002", "response_text": "prose\\n\\nAnswer: B"}\n'
            '{"audit_id": "AUD-0003", "response_text": "Answer: C"}\n', encoding="utf-8")
        report = self.build()
        self.assertEqual(report.answer_only.n, 2)
        self.assertEqual(report_payload(report)["answer_only_responses"]["n"], 2)
        self.assertIn("bare `Answer: X` lines", render_markdown(report))

    def test_report_payload_carries_the_verbatim_note_and_descriptive_framing(self):
        payload = report_payload(self.build())
        self.assertEqual(payload["annotator_note_verbatim"], ANNOTATOR_NOTE)
        self.assertTrue(payload["not_a_gate"])
        self.assertEqual(payload["status"], "descriptive_only")
        self.assertEqual(payload["n_items"], 3)
        self.assertEqual(payload["join_key"], "response_id")
        self.assertEqual(len(payload["items"]), 3)
        self.assertEqual(payload["rev_flagged_count"], 0)
        self.assertEqual(json.loads(json.dumps(payload))["n_items"], 3)

    def test_markdown_states_it_is_not_a_gate_and_lists_every_item(self):
        report = self.build()
        text = render_markdown(report)
        self.assertIn("not a gate", text)
        self.assertIn(ANNOTATOR_NOTE, text)
        for entry in report.items:
            self.assertIn(entry.audit_id, text)
        self.assertIn("Spearman", text)

    def test_undefined_spearman_is_surfaced_in_the_markdown(self):
        self.write_judge({"r1": 0, "r2": 0, "r3": 0})
        text = render_markdown(self.build())
        self.assertIn("undefined", text)

    def test_markdown_names_items_outside_the_tolerance_band(self):
        self.write_scores([("AUD-0001", "0", ""), ("AUD-0002", "2", ""), ("AUD-0003", "9", "")])
        text = render_markdown(self.build())          # AUD-0003: human 9 vs judge 1
        self.assertIn("Items outside the within-2 band (1 of 3)", text)
        self.assertIn("AUD-0003 human 9 vs judge 1 (human higher by 8)", text)

    def test_markdown_says_so_when_everything_agrees_within_tolerance(self):
        self.assertIn("Every item agreed within 2 points.", render_markdown(self.build()))

    def test_per_item_rows_expose_the_full_response_id(self):
        rows = per_item_rows(self.build().items)
        self.assertEqual([row["response_id"] for row in rows], ["r1", "r2", "r3"])
        self.assertEqual([row["abs_diff"] for row in rows], [0, 1, 0])

    def test_build_report_rejects_an_incomplete_audit(self):
        self.write_judge({"r1": 0, "r2": 3})
        with self.assertRaises(AuditError):
            self.build()


if __name__ == "__main__":
    unittest.main()
