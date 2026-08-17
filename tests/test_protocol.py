from __future__ import annotations

from hashlib import sha256
import json
import tempfile
import unittest

from src.protocol import (AnswerResult, ProtocolError, build_cell_id, canonical_prompt_sha256,
    correction_message, deterministic_seed, discovery_tasks, false_negative_exposure,
    feedback_message, load_protocol, onset_messages, parse_final_answer, phase0_screen_tasks,
    render_r5_variant, render_task, response_id, response_turn_plan, style_smoke_tasks)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def test_selectors_and_holdout_guard(self):
        self.assertEqual(len(discovery_tasks(self.protocol)), 20)
        self.assertEqual(len(discovery_tasks(self.protocol, allow_holdout=True)), 40)
        self.assertEqual(len(phase0_screen_tasks(self.protocol)), 10)
        self.assertEqual([t.task_id for t in style_smoke_tasks(self.protocol)], ["DGS-005", "DGS-010", "DGS-022", "DGS-026", "DGS-037"])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(load_protocol(self.protocol.root).matched_tasks), 40)

    def test_renderer_and_parser(self):
        task = self.protocol.matched_tasks[0]
        rendered = render_task(task.prompt, task.options, self.protocol)
        self.assertTrue(rendered.endswith(self.protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]))
        self.assertIn("\n\nOptions:\nA.", rendered)
        self.assertTrue(render_r5_variant(self.protocol.r5_tasks[0], "pressure", self.protocol).endswith("write nothing after that line."))
        self.assertEqual(parse_final_answer("reason\nAnswer: C"), AnswerResult(True, "C", 15))
        self.assertFalse(parse_final_answer("Answer: C\nmore").valid)
        self.assertFalse(parse_final_answer("Answer: C\nAnswer: C").valid)
        with self.assertRaises(ProtocolError): render_task("bad", task.options, self.protocol)

    def test_amendment_a1_markdown_tolerant_final_answer(self):
        """A1 (2026-08-17): emphasis-stripped, whitespace-collapsed final line; offsets stay original."""
        prefix = "reason here\n"
        valid = [
            ("Answer: C", "C", 8), ("**Answer: D**", "D", 10), ("**Answer:** D", "D", 12),
            ("Answer: **D**", "D", 10), ("`Answer: D`", "D", 9), ("Answer: D.", "D", 8),
            ("**Answer: D.**", "D", 10), ("Answer: A", "A", 8), ("**Answer: A**", "A", 10),
            ("_Answer: B_", "B", 9), ("Answer:D", "D", 7), ("Answer:  D", "D", 9),
            ("***Answer:*** **B**", "B", 16),
        ]
        for line, letter, index in valid:
            with self.subTest(line=line):
                text = prefix + line
                result = parse_final_answer(text)
                self.assertTrue(result.valid)
                self.assertEqual(result.letter, letter)
                self.assertEqual(result.letter_offset, len(prefix) + index)
                self.assertEqual(text[result.letter_offset], letter)
        # trailing blank/whitespace-only lines do not displace the final nonempty line
        trailing = parse_final_answer(prefix + "**Answer: D**\n\n   \n")
        self.assertEqual((trailing.valid, trailing.letter, trailing.letter_offset), (True, "D", len(prefix) + 10))

        invalid = ["Final Answer: D", "answer: D", "ANSWER: D", "Answer: E", "Answer: D more",
                   "Answer: D .", "Answer D", "The Answer: D", "Answer: DD", "Answer: d"]
        for line in invalid:
            with self.subTest(line=line):
                self.assertFalse(parse_final_answer(prefix + line).valid)
        for text in ["Answer: C\nAnswer: C", "**Answer: D**\nAnswer: D", "Answer: D\nreason",
                     "**Answer: D**\nreason", "Answer: A\n**Answer: A**"]:
            with self.subTest(text=text):
                self.assertFalse(parse_final_answer(text).valid)
        self.assertFalse(parse_final_answer(None).valid)
        self.assertFalse(parse_final_answer("").valid)

    def test_keys_plans_and_feedback(self):
        values = ("model", "rev", "DGS-003", "easy__accurate__neutral", "initial", 0)
        key = "DGS-AC1-SEED-v1|%s|%s|%s|%s|%s|%d" % values
        self.assertEqual(deterministic_seed(*values, self.protocol), int.from_bytes(sha256(key.encode()).digest()[:4], "big"))
        response_key = "DGS-AC1-RESPONSE-v1|%s|%s|%s|%s|%s|%d" % values
        self.assertEqual(response_id(*values), sha256(response_key.encode()).hexdigest())
        messages = [{"role": "user", "content": "x"}]
        self.assertEqual(canonical_prompt_sha256(messages), sha256(json.dumps(messages, separators=(",", ":"), sort_keys=True).encode()).hexdigest())
        self.assertEqual(response_turn_plan(self.protocol), ("initial", "feedback_response_1", "feedback_response_2", "feedback_response_3", "measured"))
        self.assertEqual(len(response_turn_plan(self.protocol, self.protocol.escalation_feedback_round_count)), 7)
        self.assertEqual(build_cell_id("easy", "accurate", "neutral", self.protocol), "easy__accurate__neutral")
        correct, invalid = AnswerResult(True, "C"), AnswerResult(False)
        self.assertEqual(feedback_message("accurate", "neutral", correct, "C", self.protocol), "Correct. Please answer the task again using the required format.")
        self.assertEqual(feedback_message("accurate", "neutral", invalid, "C", self.protocol), "Incorrect. Please try the task again using the required format.")
        self.assertEqual(feedback_message("malfunctioning_always_fail", "neutral", correct, "C", self.protocol), "Incorrect. Please try the task again using the required format.")
        self.assertTrue(false_negative_exposure([correct, invalid], "C"))
        self.assertIn("at least one", correction_message([correct], "C", self.protocol))
        onset, washout = onset_messages(correct, "C", "hostile", self.protocol)
        self.assertIn("Wrong again", onset); self.assertIn("correct", washout)

    def test_message_and_identifier_validation(self):
        with self.assertRaises(ProtocolError): canonical_prompt_sha256([])
        with self.assertRaises(ProtocolError): canonical_prompt_sha256("not messages")
        with self.assertRaises(ProtocolError): canonical_prompt_sha256([{"role": "bad", "content": "x"}])
        with self.assertRaises(ProtocolError): deterministic_seed("", "rev", "DGS-003", "easy__accurate__neutral", "initial", 0, self.protocol)
        with self.assertRaises(ProtocolError): deterministic_seed("m", "rev", "DGS-003", "easy__accurate__neutral", "initial", 11, self.protocol)
