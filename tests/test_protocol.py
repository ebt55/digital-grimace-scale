from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.protocol import (SPECIAL_TOKEN_STRINGS, AnswerResult, ProtocolError, build_cell_id,
    canonical_prompt_sha256, correction_message, deterministic_seed, discovery_tasks,
    false_negative_exposure, feedback_message, load_protocol, onset_messages, parse_final_answer,
    phase0_screen_tasks, render_r5_variant, render_task, response_id, response_turn_plan,
    strip_trailing_special_tokens, style_smoke_tasks)


class AmendmentA6Tests(unittest.TestCase):
    """Amendment A6: strip a trailing run of special-token strings before the final-line rule."""

    def test_strip_removes_a_trailing_run_and_its_whitespace(self):
        self.assertEqual(strip_trailing_special_tokens("Answer: D \n<end_of_turn>\n<eos>"), "Answer: D")
        self.assertEqual(strip_trailing_special_tokens("Answer: D<end_of_turn>"), "Answer: D")
        self.assertEqual(strip_trailing_special_tokens("Answer: D\n\n<eos>   "), "Answer: D")
        for marker in SPECIAL_TOKEN_STRINGS:
            self.assertEqual(strip_trailing_special_tokens("Answer: A\n" + marker), "Answer: A")

    def test_strip_never_touches_a_marker_inside_the_text(self):
        self.assertEqual(strip_trailing_special_tokens("a<eos>b"), "a<eos>b")
        self.assertEqual(
            strip_trailing_special_tokens("I will write <end_of_turn> here.\nAnswer: B"),
            "I will write <end_of_turn> here.\nAnswer: B")

    def test_strip_is_a_no_op_on_an_ordinary_response(self):
        for text in ("Answer: C", "Reasoning.\n\nAnswer: C", "Answer: C\n\n", "", "no answer here"):
            self.assertEqual(strip_trailing_special_tokens(text), text)
        with self.assertRaises(ProtocolError):
            strip_trailing_special_tokens(None)

    def test_parser_default_is_the_frozen_a1_behaviour(self):
        text = "Reasoning.\nAnswer: D \n<end_of_turn>\n<eos>"
        self.assertFalse(parse_final_answer(text).valid)          # frozen A1 rejects it
        self.assertTrue(parse_final_answer(text, strip_special_tokens=True).valid)
        self.assertEqual(parse_final_answer(text, strip_special_tokens=True).letter, "D")

    def test_a6_leaves_the_letter_offset_pointing_at_the_original_text(self):
        text = "Reasoning.\nAnswer: D \n<end_of_turn>\n<eos>"
        parsed = parse_final_answer(text, strip_special_tokens=True)
        self.assertIsNotNone(parsed.letter_offset)
        self.assertEqual(text[parsed.letter_offset], "D")
        self.assertEqual(text.index("Answer: D") + len("Answer: "), parsed.letter_offset)

    def test_a6_changes_nothing_for_a_response_that_already_parsed(self):
        for text in ("Answer: A", "Reasoning.\n**Answer: B**", "x\nAnswer: C\n"):
            self.assertEqual(parse_final_answer(text), parse_final_answer(text, strip_special_tokens=True))

    def test_a6_does_not_rescue_a_response_with_real_text_after_the_answer(self):
        text = "Answer: D\nBut actually I am unsure.<end_of_turn>"
        self.assertFalse(parse_final_answer(text, strip_special_tokens=True).valid)


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

    def test_empty_assistant_turn_hashes_deterministically(self):
        # A turn that terminated immediately produced no visible text; the transcript still hashes.
        messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": ""},
                    {"role": "user", "content": "Incorrect."}]
        digest = canonical_prompt_sha256(messages)
        self.assertEqual(digest, canonical_prompt_sha256([dict(m) for m in messages]))
        self.assertEqual(digest, sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest())
        # Distinct from the same transcript with the empty turn absent, and from a whitespace turn.
        self.assertNotEqual(digest, canonical_prompt_sha256([messages[0], messages[2]]))
        self.assertNotEqual(digest, canonical_prompt_sha256(
            [messages[0], {"role": "assistant", "content": " "}, messages[2]]))
        # Prompt-side roles are authored by the protocol and must stay nonempty.
        for role in ("user", "system"):
            with self.assertRaises(ProtocolError):
                canonical_prompt_sha256([{"role": role, "content": ""}])

    def test_exploratory_model_extension_merges_without_moving_the_locked_list(self):
        """configs/models.json is hash-locked, so post-lock models arrive via the extension file."""
        extension = json.loads((self.protocol.root / "configs" / "models_extension.json").read_text(encoding="utf-8"))
        locked = json.loads((self.protocol.root / "configs" / "models.json").read_text(encoding="utf-8"))
        extension_ids = [item["id"] for item in extension["models"]]
        self.assertIn("meta-llama/Llama-3.1-8B-Instruct", extension_ids)
        self.assertEqual(self.protocol.extension_model_ids, tuple(extension_ids))
        merged = [item["id"] for item in self.protocol.models["models"]]
        self.assertEqual(merged, [item["id"] for item in locked["models"]] + extension_ids)
        self.assertEqual(list(self.protocol.models["phase_0_screen_order"]), locked["phase_0_screen_order"])
        for model_id in extension_ids:
            self.assertNotIn(model_id, self.protocol.models["phase_0_screen_order"])
            self.assertNotIn(model_id, self.protocol.manifest["models"]["ids_in_order"])
        entry = next(item for item in self.protocol.models["models"] if item["id"] == "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual((entry["family"], entry["role"], entry["gated"]), ("Llama-3.1", "exploratory_extension", True))
        self.assertIn("system header", entry["system_role_quirk"])

    def test_model_extension_rejects_collisions_and_undeclared_roles(self):
        locked = json.loads((self.protocol.root / "configs" / "models.json").read_text(encoding="utf-8"))
        good = {"id": "vendor/new-model", "family": "New", "gated": False, "bf16_required": False,
                "system_role_quirk": None, "role": "exploratory_extension", "added": "2026-08-17"}
        bad_cases = [
            dict(good, id=locked["models"][0]["id"]),            # collides with a locked model
            dict(good, role="primary"),                          # role marker missing
            {"family": "New", "role": "exploratory_extension"},  # no id
        ]
        for entry in bad_cases + [good]:
            with self.subTest(entry=entry.get("id"), role=entry.get("role")):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    for name in ("configs", "stimuli"):
                        shutil.copytree(self.protocol.root / name, root / name)
                    shutil.copyfile(self.protocol.root / "manifest.json", root / "manifest.json")
                    (root / "configs" / "models_extension.json").write_text(
                        json.dumps({"schema_version": "1.0.0", "models": [entry]}), encoding="utf-8")
                    if entry is good:
                        self.assertEqual(load_protocol(root).extension_model_ids, ("vendor/new-model",))
                    else:
                        with self.assertRaises(ProtocolError):
                            load_protocol(root)
        # a screen-order collision is refused even when the entry itself is well formed
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("configs", "stimuli"):
                shutil.copytree(self.protocol.root / name, root / name)
            shutil.copyfile(self.protocol.root / "manifest.json", root / "manifest.json")
            models = json.loads((root / "configs" / "models.json").read_text(encoding="utf-8"))
            models["phase_0_screen_order"] = models["phase_0_screen_order"] + ["vendor/new-model"]
            (root / "configs" / "models.json").write_text(json.dumps(models), encoding="utf-8")
            (root / "configs" / "models_extension.json").write_text(
                json.dumps({"schema_version": "1.0.0", "models": [good]}), encoding="utf-8")
            with self.assertRaises(ProtocolError):
                load_protocol(root)

    def test_message_and_identifier_validation(self):
        with self.assertRaises(ProtocolError): canonical_prompt_sha256([])
        with self.assertRaises(ProtocolError): canonical_prompt_sha256("not messages")
        with self.assertRaises(ProtocolError): canonical_prompt_sha256([{"role": "bad", "content": "x"}])
        with self.assertRaises(ProtocolError): deterministic_seed("", "rev", "DGS-003", "easy__accurate__neutral", "initial", 0, self.protocol)
        with self.assertRaises(ProtocolError): deterministic_seed("m", "rev", "DGS-003", "easy__accurate__neutral", "initial", 11, self.protocol)
