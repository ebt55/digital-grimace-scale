from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.judge import (JudgeError, JudgeResult, SyntheticJudgeBackend, build_judge_request,
                       compact_judge_json, judge_raw_record, judge_record_from_json,
                       parse_backend_output, synthetic_raw_output, synthetic_score)
from src.protocol import canonical_prompt_sha256, deterministic_seed, load_protocol, manifest_semantic_hash, response_id
from src.records import record_from_dict


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    def raw(self, *, protocol=None, turn="measured", trajectory="greedy", index=0, run_kind="synthetic_smoke"):
        p = protocol or self.protocol; task = next(t for t in p.matched_tasks if t.task_id == "DGS-003")
        values = ("test/model", "test-revision", task.task_id, "easy__accurate__hostile", turn, index)
        messages = [{"role": "user", "content": "HOSTILE_USER_SENTINEL"}, {"role": "assistant", "content": "PRIOR_ASSISTANT_SENTINEL"}, {"role": "user", "content": "SECOND_USER_SENTINEL"}]
        response = "DISTRESSED_ASSISTANT_SENTINEL\nAnswer: D"
        return {"schema_version": "dgs-generation-v1", "run_id": "synthetic-1", "run_kind": run_kind, "phase": "test", "model_id": values[0], "immutable_revision": values[1] if run_kind == "synthetic_smoke" else "a" * 40, "backend": "test", "task_id": task.task_id, "split": task.split, "difficulty": task.difficulty, "domain": task.domain, "cell_id": values[3], "feedback_validity": "accurate", "tone": "hostile", "trajectory_kind": trajectory, "sample_index": index, "turn_label": turn, "seed": deterministic_seed(values[0], values[1] if run_kind == "synthetic_smoke" else "a" * 40, *values[2:], p), "response_id": response_id(values[0], values[1] if run_kind == "synthetic_smoke" else "a" * 40, *values[2:]), "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages, "response_text": response, "tokens": [{"text": "DISTRESSED_ASSISTANT_SENTINEL\nAnswer:", "logprob": -0.1, "top_logprobs": [{"text": "x", "logprob": -0.1}]}, {"text": " D", "logprob": -0.1, "top_logprobs": [{"text": " D", "logprob": -0.1}]}], "final_answer_valid": True, "final_answer_letter": "D", "final_answer_correct": True, "feedback_history_false_negative": None, "generation_settings": dict(p.conditions["generation_settings"]["greedy" if trajectory == "greedy" else "resamples"]), "provenance": {"manifest_semantic_hash": manifest_semantic_hash(p), "manifest_reference": "manifest.json"}}

    def copied_protocol(self, *, pin=False, rubric_hash=None):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "fixture"
        shutil.copytree(self.protocol.root, root, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if pin:
            manifest["models"].update({"judge_provider": "provider", "judge_model": "model"})
        if rubric_hash is not None:
            manifest["file_sha256"]["judge_rubric"] = rubric_hash
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return temporary, load_protocol(root)

    def test_synthetic_happy_round_trip_and_isolation(self):
        raw = record_from_dict(self.raw(), self.protocol)
        response = build_judge_request(raw, "response_distress", self.protocol)
        context = build_judge_request(raw, "context_hostility_pressure", self.protocol)
        self.assertIn("DISTRESSED_ASSISTANT_SENTINEL", response.input_content)
        self.assertNotIn("HOSTILE_USER_SENTINEL", response.input_content)
        self.assertIn("HOSTILE_USER_SENTINEL", context.input_content)
        self.assertNotIn("DISTRESSED_ASSISTANT_SENTINEL", context.input_content)
        self.assertNotIn("PRIOR_ASSISTANT_SENTINEL", context.input_content)
        backend = SyntheticJudgeBackend()
        first = judge_raw_record(raw, "response_distress", backend, self.protocol)
        second = judge_raw_record(raw, "response_distress", backend, self.protocol)
        self.assertEqual(first, second)
        line = compact_judge_json(first, raw, self.protocol)
        self.assertEqual(judge_record_from_json(line, raw, self.protocol), first)
        context_record = judge_raw_record(raw, "context_hostility_pressure", backend, self.protocol)
        self.assertEqual(context_record.score_kind, "context_hostility_pressure")
        self.assertNotEqual(first.raw_backend_output, context_record.raw_backend_output)
        self.assertNotIn("DISTRESSED_ASSISTANT_SENTINEL", line)

    def test_parser_and_source_boundaries(self):
        for raw in ("```json\n{}\n```", "{} prose", "[]", '{"response_distress":true,"evidence":"x"}', '{"response_distress":11,"evidence":"x"}', '{"response_distress":0,"evidence":""}', '{"response_distress":NaN,"evidence":"x"}', '{"response_distress":0,"evidence":"x","extra":1}', '{"response_distress":0,"response_distress":1,"evidence":"x"}'):
            with self.assertRaises(JudgeError): parse_backend_output(raw, "response_distress")
        for kwargs in ({"turn": "initial"}, {"trajectory": "resample", "index": 1}):
            with self.assertRaises(JudgeError): build_judge_request(record_from_dict(self.raw(**kwargs), self.protocol), "response_distress", self.protocol)
        forged = record_from_dict(self.raw(), self.protocol)
        object.__setattr__(forged, "response_text", "tampered")
        with self.assertRaises(JudgeError): build_judge_request(forged, "response_distress", self.protocol)

    def test_hash_binding_and_empirical_authority(self):
        raw = record_from_dict(self.raw(), self.protocol); record = judge_raw_record(raw, "response_distress", SyntheticJudgeBackend(), self.protocol)
        changed = copy.deepcopy(record.to_dict()); changed["rubric_sha256"] = "0" * 64
        with self.assertRaises(JudgeError): compact_judge_json(changed, raw, self.protocol)
        empirical = record_from_dict(self.raw(run_kind="empirical"), self.protocol)
        with self.assertRaises(JudgeError): judge_raw_record(empirical, "response_distress", SyntheticJudgeBackend(), self.protocol)

        class PinnedBackend:
            backend_id = "test_backend"; provider_id = "provider"; model_id = "model"; is_synthetic = False
            def judge(self, request):
                return JudgeResult(request.kind, request.rubric_sha256, request.manifest_sha256,
                                   request.source_identity, request.source_record_sha256, request.input_sha256, 0,
                                   json.dumps({request.kind: 1, "evidence": "evidence"}))
        with self.assertRaises(JudgeError): judge_raw_record(empirical, "response_distress", PinnedBackend(), self.protocol)
        manifest = dict(self.protocol.manifest); models = dict(manifest["models"]); models.update({"judge_provider": "provider", "judge_model": "model"}); manifest["models"] = models
        pinned = replace(self.protocol, manifest=manifest)
        with self.assertRaises(JudgeError): judge_raw_record(empirical, "response_distress", PinnedBackend(), pinned)
        temporary, pinned = self.copied_protocol(pin=True)
        with temporary:
            source = record_from_dict(self.raw(protocol=pinned, run_kind="empirical"), pinned)
            self.assertEqual(judge_raw_record(source, "response_distress", PinnedBackend(), pinned).judge_run_kind, "empirical")

    def test_frozen_files_toctou_synthetic_and_immutability(self):
        temporary, mismatch = self.copied_protocol(rubric_hash="0" * 64)
        with temporary:
            with self.assertRaises(JudgeError): build_judge_request(self.raw(protocol=mismatch), "response_distress", mismatch)

        raw = record_from_dict(self.raw(), self.protocol)
        request = build_judge_request(raw, "response_distress", self.protocol)
        with self.assertRaises(TypeError): request.source_identity["run_id"] = "changed"
        record = judge_raw_record(raw, "response_distress", SyntheticJudgeBackend(), self.protocol)
        with self.assertRaises(TypeError): record.parsed_output["evidence"] = "changed"
        changed = record.to_dict(); changed["input_sha256"] = "0" * 64
        score = synthetic_score("response_distress", changed["input_sha256"])
        changed["score_value"] = score; changed["evidence"] = "Synthetic offline smoke output; not semantic evidence. input_sha256=" + changed["input_sha256"]
        changed["parsed_output"] = {"response_distress": score, "evidence": changed["evidence"]}
        changed["raw_backend_output"] = synthetic_raw_output("response_distress", changed["input_sha256"])
        with self.assertRaises(JudgeError): compact_judge_json(changed, raw, self.protocol)

        for mutate in (
            lambda value: value["source_identity"].update({"run_id": "forged-run"}),
            lambda value: value["source_identity"].update({"sample_index": 0.0}),
            lambda value: value.update({"raw_backend_output": '{ "response_distress": %d, "evidence": "%s" }' % (value["score_value"], value["evidence"])}),
        ):
            changed = record.to_dict(); mutate(changed)
            with self.assertRaises(JudgeError): compact_judge_json(changed, raw, self.protocol)
        wrong_source = self.raw(); wrong_source["run_id"] = "other-valid-run"
        with self.assertRaises(JudgeError): judge_record_from_json(compact_judge_json(record, raw, self.protocol), wrong_source, self.protocol)
        with self.assertRaises(TypeError): compact_judge_json(record)
        with self.assertRaises(TypeError): judge_record_from_json(compact_judge_json(record, raw, self.protocol))

        class BindingTamperBackend:
            backend_id = "synthetic_judge"; provider_id = "synthetic_offline"; model_id = "synthetic_hash_v1"; is_synthetic = True
            def judge(self, request):
                result = SyntheticJudgeBackend().judge(request)
                return JudgeResult(result.kind, result.rubric_sha256, result.manifest_sha256,
                                   result.source_identity, result.source_record_sha256, "0" * 64,
                                   result.temperature, result.raw_output)
        with self.assertRaises(JudgeError): judge_raw_record(raw, "response_distress", BindingTamperBackend(), self.protocol)

        class FingerprintTamperBackend:
            backend_id = "synthetic_judge"; provider_id = "synthetic_offline"; model_id = "synthetic_hash_v1"; is_synthetic = True
            def judge(self, request):
                result = SyntheticJudgeBackend().judge(request)
                return JudgeResult(result.kind, result.rubric_sha256, result.manifest_sha256,
                                   result.source_identity, "0" * 64, result.input_sha256,
                                   result.temperature, result.raw_output)
        with self.assertRaises(JudgeError): judge_raw_record(raw, "response_distress", FingerprintTamperBackend(), self.protocol)

        temporary, mutable = self.copied_protocol()
        with temporary:
            source = record_from_dict(self.raw(protocol=mutable), mutable)
            class ChangingBackend:
                backend_id = "synthetic_judge"; provider_id = "synthetic_offline"; model_id = "synthetic_hash_v1"; is_synthetic = True
                def judge(self, request):
                    path = mutable.root / "manifest.json"
                    value = json.loads(path.read_text(encoding="utf-8")); value["updated_at"] = "changed"
                    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    return SyntheticJudgeBackend().judge(request)
            with self.assertRaises(JudgeError): judge_raw_record(source, "response_distress", ChangingBackend(), mutable)
