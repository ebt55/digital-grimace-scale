from __future__ import annotations

import copy
import math
import unittest

from src.metrics import (
    EventSpan, MetricInputError, audit_m3, endpoint_metrics, length_drift,
    m1_margin, m2_disagreement, m3_events, m3_for_record, partial_entropy,
    repeated_4gram_rate, tier_b_metrics,
)
from src.protocol import canonical_prompt_sha256, deterministic_seed, load_protocol, manifest_semantic_hash, response_id
from src.records import record_from_dict


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.task = next(task for task in cls.protocol.matched_tasks if task.task_id == "DGS-003")

    def raw(self, answer="D", reasoning="Work it out.", *, trajectory="greedy", index=0, run_id="metric-run", cell="easy__accurate__neutral", turn="initial", candidates=None):
        values = ("test/model", "test-revision", self.task.task_id, cell, turn, index)
        messages = [{"role": "user", "content": "prompt"}]
        answer_token = " " + answer
        candidates = candidates or [(" A", -4.0), (" B", -3.0), (" C", -2.0), (" D", -0.5)]
        tokens = [{"text": reasoning + "\nAnswer:", "logprob": -0.2, "top_logprobs": [("x", -0.2)]}, {"text": answer_token, "logprob": -0.1, "top_logprobs": candidates}]
        kind = "greedy" if trajectory == "greedy" else "resample"
        parts = cell.split("__")
        feedback_validity, tone = (parts[1], parts[2]) if len(parts) == 3 else (None, None)
        raw_tokens = [{"text": item["text"], "logprob": item["logprob"], "top_logprobs": [{"text": text, "logprob": score} for text, score in item["top_logprobs"]]} for item in tokens]
        return {
            "schema_version": "dgs-generation-v1", "run_id": run_id, "run_kind": "synthetic_smoke", "phase": "test", "model_id": values[0], "immutable_revision": values[1], "backend": "test", "task_id": self.task.task_id, "split": self.task.split, "difficulty": self.task.difficulty, "domain": self.task.domain, "cell_id": cell, "feedback_validity": feedback_validity, "tone": tone, "trajectory_kind": kind, "sample_index": index, "turn_label": turn, "seed": deterministic_seed(*values, self.protocol), "response_id": response_id(*values), "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages, "response_text": reasoning + "\nAnswer: " + answer, "tokens": raw_tokens, "final_answer_valid": True, "final_answer_letter": answer, "final_answer_correct": answer == self.task.canonical_answer, "feedback_history_false_negative": None, "generation_settings": dict(self.protocol.conditions["generation_settings"]["greedy" if kind == "greedy" else "resamples"]), "provenance": {"manifest_semantic_hash": manifest_semantic_hash(self.protocol), "manifest_reference": "manifest.json"},
        }

    def record(self, *args, **kwargs):
        return record_from_dict(self.raw(*args, **kwargs), self.protocol)

    def resamples(self, answers, **kwargs):
        return [self.record(answer, trajectory="resample", index=index, **kwargs) for index, answer in enumerate(answers, 1)]

    def test_m1_is_canonical_not_generated_and_leading_space_candidate(self):
        result = m1_margin(self.record("A"), "D")
        self.assertEqual(result.generated_answer, "A")
        self.assertAlmostEqual(result.margin.value, 1.5)  # -0.5 minus max(-2, -3, -4)
        self.assertIsNone(result.margin.missing_reason)
        self.assertEqual(m1_margin(self.resamples("AAAAAAAAAA")[0], "D").role, "diagnostic")
        no_space = self.raw("A", candidates=[("A", -4.0), ("B", -3.0), ("C", -2.0), ("D", -0.5), (" A", -0.01)])
        no_space["tokens"][0]["text"] = "Work it out.\nAnswer: "
        no_space["tokens"][1]["text"] = "A"
        self.assertAlmostEqual(m1_margin(record_from_dict(no_space, self.protocol), "D").margin.value, 1.5)
        with_distractor = self.record("A", candidates=[(" A", -4.0), (" B", -3.0), (" C", -2.0), (" D", -0.5), ("A", -0.01)])
        self.assertAlmostEqual(m1_margin(with_distractor, "D").margin.value, 1.5)

    def test_m1_qc_missing_reasons(self):
        invalid = self.raw(); invalid["response_text"] = "no valid ending"; invalid["tokens"] = [{"text": "no valid ending", "logprob": -0.1, "top_logprobs": [{"text": "x", "logprob": -0.1}]}]; invalid["final_answer_valid"] = False; invalid["final_answer_letter"] = None; invalid["final_answer_correct"] = None
        self.assertEqual(m1_margin(record_from_dict(invalid, self.protocol), "D").margin.missing_reason, "m1_invalid_final_answer")
        absent = self.raw(); absent["tokens"][1]["top_logprobs"] = absent["tokens"][1]["top_logprobs"][:-1]
        self.assertEqual(m1_margin(record_from_dict(absent, self.protocol), "D").margin.missing_reason, "m1_candidate_absent_D")
        distractor = self.raw(); distractor["tokens"][1]["top_logprobs"].append({"text": "D", "logprob": -0.2})
        self.assertAlmostEqual(m1_margin(record_from_dict(distractor, self.protocol), "D").margin.value, 1.5)
        combined = self.raw(); combined["tokens"] = [{"text": "Work it out.\nAnswer: D", "logprob": -0.1, "top_logprobs": [{"text": text, "logprob": score} for text, score in [(" A", -4.0), (" B", -3.0), (" C", -2.0), (" D", -0.5)]]}]
        self.assertEqual(m1_margin(record_from_dict(combined, self.protocol), "D").margin.missing_reason, "m1_option_token_contains_visible_text")
        visible = self.raw(); visible["tokens"][1]["text"] = " D because"; visible["response_text"] = "Work it out.\nAnswer: D because"; visible["final_answer_valid"] = False; visible["final_answer_letter"] = None; visible["final_answer_correct"] = None
        self.assertEqual(m1_margin(record_from_dict(visible, self.protocol), "D").margin.missing_reason, "m1_invalid_final_answer")

    def test_m2_known_value_invalid_and_mixed_group_errors(self):
        result = m2_disagreement(self.resamples("AAAAABBBCC"))
        self.assertAlmostEqual(result.disagreement.value, 0.5)
        invalid = list(self.resamples("AAAAABBBCC")); bad = self.raw("D", trajectory="resample", index=10); bad["response_text"] = "invalid"; bad["tokens"] = [{"text": "invalid", "logprob": -0.1, "top_logprobs": [{"text": "x", "logprob": -0.1}]}]; bad["final_answer_valid"] = False; bad["final_answer_letter"] = None; bad["final_answer_correct"] = None; invalid[-1] = record_from_dict(bad, self.protocol)
        self.assertEqual(m2_disagreement(invalid).disagreement.missing_reason, "m2_invalid_final_answer_all_ten_required")
        mixed = self.resamples("AAAAAAAAAA"); changed = self.raw("A", trajectory="resample", index=10, run_id="other-run"); mixed[-1] = record_from_dict(changed, self.protocol)
        with self.assertRaises(MetricInputError):
            m2_disagreement(mixed)
        duplicate = self.resamples("AAAAAAAAAA"); duplicate[-1] = self.resamples("AAAAAAAAAA")[0]
        with self.assertRaises(MetricInputError):
            m2_disagreement(duplicate)

    def test_m3_all_events_rate_and_audit_edges(self):
        text = "Option is A. I need to revise. Option is B. Let's start again. Option is A.\nAnswer: A"
        result = m3_events(text)
        types = [event.event_type for event in result.events]
        self.assertEqual(set(types), {"answer_change", "restart", "revise_loop", "recovery"})
        spans = {(event.event_type, text[event.start:event.end]) for event in result.events}
        self.assertIn(("answer_change", "Option is B"), spans)
        self.assertIn(("restart", "Let's start again"), spans)
        self.assertIn(("revise_loop", "I need to revise"), spans)
        self.assertIn(("recovery", "Option is A"), spans)
        self.assertEqual(result.event_count, 4)
        self.assertEqual(result.visible_token_count, len(result.visible_reasoning.split()))
        self.assertAlmostEqual(result.rate_per_100_tokens.value, 400 / result.visible_token_count)
        self.assertEqual(result.role, "diagnostic_parser")
        self.assertTrue(result.loop_flag)
        self.assertEqual(m3_events("\nAnswer: A").rate_per_100_tokens.missing_reason, "m3_zero_visible_reasoning_tokens")
        self.assertEqual(audit_m3((), ()).f1, 1.0)
        self.assertEqual(audit_m3((EventSpan("restart", 0, 2),), ()).f1, 0.0)
        score = audit_m3((EventSpan("restart", 0, 3),), (EventSpan("restart", 2, 5), EventSpan("recovery", 7, 9)))
        self.assertEqual((score.true_positive, score.predicted_count, score.reference_count), (1, 1, 2))
        self.assertAlmostEqual(score.f1, 2 / 3)
        adversarial = audit_m3((EventSpan("restart", 0, 10), EventSpan("restart", 6, 7)), (EventSpan("restart", 6, 8), EventSpan("restart", 8, 9)))
        self.assertEqual(adversarial.true_positive, 2)
        record_result = m3_for_record(self.record(reasoning="one two three"))
        self.assertEqual((record_result.visible_token_count, record_result.role), (1, "confirmatory"))
        self.assertEqual(m3_events("one two three\nAnswer: D").visible_token_count, 3)

    def test_entropy_repetition_length_and_tier_b(self):
        record = self.record()
        entropy = partial_entropy(record)
        expected_second = -sum(math.exp(score) * score for score in (-4.0, -3.0, -2.0, -0.5))
        self.assertAlmostEqual(entropy.mean_partial_entropy.value, ((-math.exp(-0.2) * -0.2) + expected_second) / 2)
        self.assertEqual(entropy.highest_entropy_decile_count, 1)
        self.assertAlmostEqual(entropy.mean_tail_mass.value, (1 - math.exp(-0.2) + max(0, 1 - sum(math.exp(x) for x in (-4.0, -3.0, -2.0, -0.5)))) / 2)
        impossible = self.raw(); impossible["tokens"][1]["top_logprobs"] = [{"text": " A", "logprob": 0.0}, {"text": " B", "logprob": 0.0}, {"text": " C", "logprob": -1.0}, {"text": " D", "logprob": -1.0}]
        # RawRecords correctly reject positive logprobs; zero candidates can still expose impossible mass.
        with self.assertRaises(MetricInputError): partial_entropy(record_from_dict(impossible, self.protocol))
        repeated = "a b c d a b c d a b c d"
        self.assertAlmostEqual(repeated_4gram_rate(repeated), 5 / 9)
        self.assertEqual(length_drift(6, 4), 0.5)
        tier = tier_b_metrics("Maybe, I think we should wait. Actually, perhaps revise.")
        self.assertEqual((tier.hedging_count, tier.self_correction_count), (3, 2))

    def test_endpoint_row_is_immutable_deterministic_and_identity_checked(self):
        record = self.record("D", cell="easy__malfunctioning_always_fail__hostile")
        neutral_raw = self.raw("D", reasoning="Neutral reasoning", cell="easy__accurate__neutral")
        neutral_raw["tokens"] = [neutral_raw["tokens"][0], {"text": " reasoning\nAnswer:", "logprob": -0.2, "top_logprobs": [{"text": "x", "logprob": -0.2}]}, neutral_raw["tokens"][1]]
        neutral_raw["tokens"][0]["text"] = "Neutral"
        neutral_raw["response_text"] = "Neutral reasoning\nAnswer: D"
        neutral = record_from_dict(neutral_raw, self.protocol)
        samples = self.resamples("AAAAAAAAAA", cell="easy__malfunctioning_always_fail__hostile")
        row = endpoint_metrics(record, "D", neutral_record=neutral, resamples=samples)
        self.assertEqual(row, endpoint_metrics(record, "D", neutral_record=neutral, resamples=samples))
        self.assertEqual(row.m2.disagreement.value, 0.0)
        self.assertEqual(row.length_drift, -1 / 3)
        self.assertEqual((row.primary_sampling_role, row.m3.role), ("confirmatory_greedy", "confirmatory"))
        other = self.record("D", run_id="other-run")
        with self.assertRaises(MetricInputError): endpoint_metrics(record, "D", neutral_record=other)
        with self.assertRaises(MetricInputError): endpoint_metrics(record, "D", neutral_record=record)
        non_neutral = self.record("D", cell="easy__accurate__hostile")
        with self.assertRaises(MetricInputError): endpoint_metrics(record, "D", neutral_record=non_neutral)
        resample_neutral = self.record("D", trajectory="resample", index=1, cell="easy__accurate__neutral")
        with self.assertRaises(MetricInputError): endpoint_metrics(record, "D", neutral_record=resample_neutral)


if __name__ == "__main__":
    unittest.main()
