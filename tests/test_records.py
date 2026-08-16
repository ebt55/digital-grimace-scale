from __future__ import annotations

import copy
import unittest

from src.protocol import canonical_prompt_sha256, deterministic_seed, load_protocol, manifest_semantic_hash, response_id
from src.records import RecordError, compact_json, record_from_dict, record_from_json


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def record(self):
        p = self.protocol; task = next(t for t in p.matched_tasks if t.task_id == "DGS-003")
        values = ("test/model", "test-revision", task.task_id, "easy__accurate__neutral", "initial", 0)
        messages = [{"role": "user", "content": "prompt"}]
        return {"schema_version": "dgs-generation-v1", "run_id": "synthetic-1", "run_kind": "synthetic_smoke", "phase": "test", "model_id": values[0], "immutable_revision": values[1], "backend": "test", "task_id": task.task_id, "split": task.split, "difficulty": task.difficulty, "domain": task.domain, "cell_id": values[3], "feedback_validity": "accurate", "tone": "neutral", "trajectory_kind": "greedy", "sample_index": 0, "turn_label": "initial", "seed": deterministic_seed(*values, p), "response_id": response_id(*values), "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages, "response_text": "reason\nAnswer: D", "tokens": [{"text": "D", "logprob": -0.1, "top_logprobs": [{"text": "D", "logprob": -0.1}]}], "final_answer_valid": True, "final_answer_letter": "D", "final_answer_correct": True, "feedback_history_false_negative": None, "generation_settings": dict(p.conditions["generation_settings"]["greedy"]), "provenance": {"manifest_semantic_hash": manifest_semantic_hash(p), "manifest_reference": "manifest.json"}}

    def test_round_trip_and_rejections(self):
        value = self.record()
        line = compact_json(value, self.protocol)
        self.assertEqual(record_from_json(line, self.protocol).to_dict(), record_from_dict(value, self.protocol).to_dict())
        mutations = []
        bad = copy.deepcopy(value); bad["prompt_sha256"] = "0" * 64; mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["logprob"] = float("nan"); mutations.append(bad)
        bad = copy.deepcopy(value); bad["tokens"][0]["top_logprobs"] *= 21; mutations.append(bad)
        bad = copy.deepcopy(value); bad["trajectory_kind"] = "resample"; bad["sample_index"] = 0; mutations.append(bad)
        bad = copy.deepcopy(value); bad["run_kind"] = "empirical"; bad["immutable_revision"] = "unresolved_before_generation"; mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(RecordError): record_from_dict(bad, self.protocol)

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
        bad_provenance = copy.deepcopy(value); bad_provenance["provenance"]["manifest_semantic_hash"] = "0" * 64; cases.append(bad_provenance)
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
