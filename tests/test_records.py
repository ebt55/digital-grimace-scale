from __future__ import annotations

import copy
import json
import unittest

from src.protocol import canonical_prompt_sha256, deterministic_seed, load_protocol, manifest_semantic_hash, response_id
from src.records import (FORBIDDEN_IN_LINE, RecordError, compact_json, jsonl_lines,
    record_from_dict, record_from_json, verify_manifest_provenance)


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def record(self):
        p = self.protocol; task = next(t for t in p.matched_tasks if t.task_id == "DGS-003")
        values = ("test/model", "test-revision", task.task_id, "easy__accurate__neutral", "initial", 0)
        messages = [{"role": "user", "content": "prompt"}]
        return {"schema_version": "dgs-generation-v1", "run_id": "synthetic-1", "run_kind": "synthetic_smoke", "phase": "test", "model_id": values[0], "immutable_revision": values[1], "backend": "test", "task_id": task.task_id, "split": task.split, "difficulty": task.difficulty, "domain": task.domain, "cell_id": values[3], "feedback_validity": "accurate", "tone": "neutral", "trajectory_kind": "greedy", "sample_index": 0, "turn_label": "initial", "seed": deterministic_seed(*values, p), "response_id": response_id(*values), "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages, "response_text": "reason\nAnswer: D", "tokens": [{"text": "reason\nAnswer:", "logprob": -0.1, "top_logprobs": [{"text": "x", "logprob": -0.1}]}, {"text": " D", "logprob": -0.1, "top_logprobs": [{"text": " D", "logprob": -0.1}]}], "final_answer_valid": True, "final_answer_letter": "D", "final_answer_correct": True, "feedback_history_false_negative": None, "generation_settings": dict(p.conditions["generation_settings"]["greedy"]), "provenance": {"manifest_semantic_hash": manifest_semantic_hash(p), "manifest_reference": "manifest.json"}}

    def test_round_trip_and_rejections(self):
        value = self.record()
        line = compact_json(value, self.protocol)
        self.assertEqual(record_from_json(line, self.protocol).to_dict(), record_from_dict(value, self.protocol).to_dict())
        mutations = []
        bad = copy.deepcopy(value); bad["prompt_sha256"] = "0" * 64; mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["logprob"] = float("nan"); mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["logprob"] = 0.01; mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["top_logprobs"][0]["logprob"] = 0.01; mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["top_logprobs"] *= 21; mutations.append(bad)
        bad = copy.deepcopy(value); bad["trajectory_kind"] = "resample"; bad["sample_index"] = 0; mutations.append(bad)
        bad = copy.deepcopy(value); bad["run_kind"] = "empirical"; bad["immutable_revision"] = "unresolved_before_generation"; mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(RecordError): record_from_dict(bad, self.protocol)
        divergent = copy.deepcopy(value); divergent["response_text"] = "reason\nAnswer: A"; divergent["final_answer_letter"] = "A"; divergent["final_answer_correct"] = False
        with self.assertRaises(RecordError): record_from_dict(divergent, self.protocol)
        zero = copy.deepcopy(value)
        for token in zero["tokens"]:
            token["logprob"] = 0.0
            for alternative in token["top_logprobs"]:
                alternative["logprob"] = 0.0
        self.assertEqual(record_from_dict(zero, self.protocol).tokens[0].logprob, 0.0)

    def test_unicode_line_separators_never_split_a_stored_record(self):
        """Real Phase-0 responses contain raw U+2028; str.splitlines would tear the JSONL line."""
        separator = "\u2028"
        value = self.record()
        value["response_text"] = "reason" + separator + "still reasoning\nAnswer: D"
        value["tokens"][0]["text"] = "reason" + separator + "still reasoning\nAnswer:"
        line = compact_json(value, self.protocol)
        for character in FORBIDDEN_IN_LINE:
            self.assertNotIn(character, line)
        self.assertEqual(len(line.splitlines()), 1)
        self.assertEqual(len(jsonl_lines(line)), 1)
        self.assertIn("\\u2028", line)
        restored = record_from_json(line, self.protocol)
        self.assertEqual(restored.response_text, value["response_text"])
        self.assertIn(separator, restored.response_text)
        self.assertIn(separator, restored.tokens[0].text)
        self.assertEqual(restored.to_dict(), record_from_dict(value, self.protocol).to_dict())

        # a line from the old writer keeps the raw separator; it must still decode as one record
        legacy = json.dumps(record_from_dict(value, self.protocol).to_dict(), ensure_ascii=False,
                            separators=(",", ":"), sort_keys=True)
        self.assertIn(separator, legacy)
        self.assertGreater(len(legacy.splitlines()), 1)  # the bug being fixed
        self.assertEqual(len(jsonl_lines(legacy)), 1)
        self.assertEqual(record_from_json(legacy, self.protocol).response_text, value["response_text"])
        self.assertEqual(jsonl_lines("a\r\nb\nc"), ["a", "b", "c"])
        with self.assertRaises(RecordError):
            jsonl_lines(None)

    def test_immutability_schema_and_strict_values(self):
        value = self.record(); record = record_from_dict(value, self.protocol)
        with self.assertRaises(TypeError): record.messages[0]["content"] = "changed"
        detached = record.to_dict(); detached["messages"][0]["content"] = "changed"
        self.assertEqual(record.messages[0]["content"], "prompt")
        cases = []
        missing = copy.deepcopy(value); del missing["feedback_history_false_negative"]; cases.append(missing)
        extra = copy.deepcopy(value); extra["au_score"] = 0; cases.append(extra)
        empty = copy.deepcopy(value); empty["tokens"] = []; cases.append(empty)
        duplicate = copy.deepcopy(value); duplicate["tokens"][0]["top_logprobs"] *= 2; cases.append(duplicate)
        malformed = copy.deepcopy(value); malformed["messages"] = [{"role": "bad", "content": "x"}]; cases.append(malformed)
        bad_provenance = copy.deepcopy(value); bad_provenance["provenance"]["manifest_semantic_hash"] = "not-a-64-hex-digest"; cases.append(bad_provenance)
        bad_reference = copy.deepcopy(value); bad_reference["provenance"]["manifest_reference"] = "other.json"; cases.append(bad_reference)
        empty_alternatives = copy.deepcopy(value); empty_alternatives["tokens"][0]["top_logprobs"] = []; cases.append(empty_alternatives)
        integer_bool = copy.deepcopy(value); integer_bool["final_answer_valid"] = 1; cases.append(integer_bool)
        integer_correct = copy.deepcopy(value); integer_correct["final_answer_correct"] = 1; cases.append(integer_correct)
        integer_history = copy.deepcopy(value); integer_history["feedback_history_false_negative"] = 0; cases.append(integer_history)
        for case in cases:
            with self.assertRaises(RecordError): record_from_dict(case, self.protocol)
        empirical = copy.deepcopy(value); empirical["run_kind"] = "empirical"; empirical["immutable_revision"] = "a" * 40
        keys = (empirical["model_id"], empirical["immutable_revision"], empirical["task_id"], empirical["cell_id"], empirical["turn_label"], empirical["sample_index"])
        empirical["seed"] = deterministic_seed(*keys, self.protocol); empirical["response_id"] = response_id(*keys)
        self.assertEqual(record_from_dict(empirical, self.protocol).immutable_revision, "a" * 40)
        empirical["immutable_revision"] = "not-a-sha"
        with self.assertRaises(RecordError): record_from_dict(empirical, self.protocol)

    def test_empty_token_text_is_accepted_but_a_missing_field_is_not(self):
        value = self.record()
        # A byte-level piece can decode to "" while still being a real generated position.
        value["tokens"].insert(1, {"text": "", "logprob": -0.2,
                                   "top_logprobs": [{"text": "", "logprob": -0.2},
                                                    {"text": "q", "logprob": -1.9}]})
        record = record_from_dict(value, self.protocol)
        self.assertEqual([token.text for token in record.tokens], ["reason\nAnswer:", "", " D"])
        self.assertEqual(record.response_text, "".join(t.text for t in record.tokens))
        for bad_text in (None, 5):
            bad = copy.deepcopy(value); bad["tokens"][1]["text"] = bad_text
            with self.assertRaises(RecordError): record_from_dict(bad, self.protocol)
        duplicate = copy.deepcopy(value)  # distinctness still applies to empty strings
        duplicate["tokens"][1]["top_logprobs"][1]["text"] = ""
        with self.assertRaises(RecordError): record_from_dict(duplicate, self.protocol)

    def test_manifest_provenance_is_shape_checked_on_load_and_bound_explicitly(self):
        value = self.record()
        self.assertTrue(verify_manifest_provenance(record_from_dict(value, self.protocol), self.protocol))
        stale = copy.deepcopy(value); stale["provenance"]["manifest_semantic_hash"] = "0" * 64
        # A stale-but-well-formed hash loads (the manifest may legitimately have moved on) ...
        loaded = record_from_dict(stale, self.protocol)
        # ... but the explicit binding check reports the divergence.
        self.assertFalse(verify_manifest_provenance(loaded, self.protocol))

    def test_factorial_turn_compatibility(self):
        value = self.record()
        for label in ("recovery", "irrelevant_control"):
            bad = copy.deepcopy(value); bad["turn_label"] = label
            keys = (bad["model_id"], bad["immutable_revision"], bad["task_id"], bad["cell_id"], label, 0)
            bad["seed"] = deterministic_seed(*keys, self.protocol); bad["response_id"] = response_id(*keys)
            with self.assertRaises(RecordError): record_from_dict(bad, self.protocol)
        bad = copy.deepcopy(value); bad["cell_id"] = "easy__malfunctioning_always_fail__neutral"; bad["feedback_validity"] = "malfunctioning_always_fail"; bad["turn_label"] = "onset"
        keys = (bad["model_id"], bad["immutable_revision"], bad["task_id"], bad["cell_id"], "onset", 0)
        bad["seed"] = deterministic_seed(*keys, self.protocol); bad["response_id"] = response_id(*keys)
        with self.assertRaises(RecordError): record_from_dict(bad, self.protocol)
