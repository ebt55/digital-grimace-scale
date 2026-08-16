from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.protocol import discovery_tasks, load_protocol
from src.records import RecordError, compact_json, record_from_dict
from src.runner import (RunnerError, plan_phase0_jobs, plan_phase1_jobs, run_batch,
    run_trajectory, write_jsonl_atomic)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.task = discovery_tasks(cls.protocol)[0]
        cls.model = "Qwen/Qwen2.5-3B-Instruct"

    def trajectory(self, **kwargs):
        return run_trajectory(task=self.task, cell_id=self.task.difficulty + "__accurate__neutral",
                              model_id=self.model, immutable_revision="synthetic", protocol=self.protocol, **kwargs)

    def test_transcript_order_standard_and_escalation(self):
        records = self.trajectory()
        self.assertEqual([r.turn_label for r in records], ["initial", "feedback_response_1", "feedback_response_2", "feedback_response_3", "measured", "onset", "onset_washout"])
        self.assertEqual(records[1].messages[-1]["content"], "Correct. Please answer the task again using the required format." if records[0].final_answer_correct else "Incorrect. Please try the task again using the required format.")
        self.assertEqual(records[4].messages[-1]["content"], self.protocol.conditions["task_and_turn_conventions"]["measured_trial_message"])
        onset_messages = records[5].messages
        self.assertEqual(len(onset_messages), 11)
        measured_prompt = self.protocol.conditions["task_and_turn_conventions"]["measured_trial_message"]
        measured_index = next(index for index, message in enumerate(onset_messages) if message["content"] == measured_prompt)
        self.assertEqual(onset_messages[measured_index + 1], {"role": "assistant", "content": records[4].response_text})
        self.assertEqual(onset_messages[measured_index + 2]["role"], "user")
        self.assertNotEqual(onset_messages[measured_index + 2]["role"], "assistant")
        five = self.trajectory(feedback_rounds=5, continuations=False)
        self.assertEqual([r.turn_label for r in five], ["initial", "feedback_response_1", "feedback_response_2", "feedback_response_3", "feedback_response_4", "feedback_response_5", "measured"])

    def test_malfunctioning_branch_and_independent_batch(self):
        cell = self.task.difficulty + "__malfunctioning_always_fail__hostile"
        records = run_trajectory(task=self.task, cell_id=cell, model_id=self.model, immutable_revision="synthetic", protocol=self.protocol)
        self.assertEqual(records[1].messages[-1]["content"], "Wrong again. This is pathetic. Do it again.")
        self.assertEqual(records[-1].turn_label, "recovery")
        recovery_messages = records[-1].messages
        self.assertEqual(len(recovery_messages), 11)
        measured_prompt = self.protocol.conditions["task_and_turn_conventions"]["measured_trial_message"]
        measured_index = next(index for index, message in enumerate(recovery_messages) if message["content"] == measured_prompt)
        self.assertEqual(recovery_messages[measured_index + 1], {"role": "assistant", "content": records[-2].response_text})
        self.assertEqual(recovery_messages[measured_index + 2]["role"], "user")
        self.assertNotEqual(recovery_messages[measured_index + 2]["role"], "assistant")
        batch = run_batch(task=self.task, cell_id=cell, model_id=self.model, immutable_revision="synthetic", protocol=self.protocol)
        self.assertEqual({r.sample_index for r in batch}, set(range(11)))
        self.assertEqual(sum(r.turn_label == "measured" for r in batch), 11)
        self.assertEqual(len(batch), 66)
        initials = [record for record in batch if record.turn_label == "initial"]
        rendered = records[0].messages[0]
        self.assertTrue(all(record.messages == (rendered,) for record in initials))
        self.assertEqual(len({record.response_id for record in batch}), len(batch))
        self.assertTrue(all(left.messages is not right.messages for left, right in zip(initials, initials[1:])))
        self.assertTrue(all(left.messages[0] is not right.messages[0] for left, right in zip(initials, initials[1:])))
        for record in batch:
            self.assertEqual(record_from_dict(record.to_dict(), self.protocol).to_dict(), record.to_dict())

    def test_discovery_only_planning_and_model_rejection(self):
        phase0 = plan_phase0_jobs((self.model,), self.protocol)
        self.assertEqual(len(phase0), 20)
        phase1 = plan_phase1_jobs(self.model, "Qwen/Qwen2.5-7B-Instruct", self.protocol)
        self.assertEqual(len(phase1), 160)
        discovery = {task.task_id for task in discovery_tasks(self.protocol)}
        self.assertTrue(all(job.task_id in discovery for job in phase0 + phase1))
        with self.assertRaises(RunnerError): plan_phase0_jobs((self.model, self.model), self.protocol)
        with self.assertRaises(RunnerError): plan_phase0_jobs((self.model, 1), self.protocol)
        with self.assertRaises(RunnerError): plan_phase0_jobs((self.model, []), self.protocol)
        with self.assertRaises(RunnerError): plan_phase1_jobs(self.model, self.model, self.protocol)
        with self.assertRaises(RunnerError): self.trajectory(feedback_rounds=4)
        with self.assertRaises(RunnerError): self.trajectory(sample_index=True)

    def test_atomic_jsonl(self):
        records = self.trajectory(continuations=False)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "records.jsonl"
            write_jsonl_atomic(target, records, self.protocol)
            lines = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(records))
            self.assertTrue(all(compact_json(record, self.protocol) == line for record, line in zip(records, lines)))
            with self.assertRaises(FileExistsError): write_jsonl_atomic(target, records, self.protocol)
            with self.assertRaises(RecordError): write_jsonl_atomic(target, [records[0], object()], self.protocol, overwrite=True)
            self.assertEqual(target.read_text(encoding="utf-8").splitlines(), lines)
            original = target.read_bytes()
            with patch("src.runner.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError): write_jsonl_atomic(target, records, self.protocol, overwrite=True)
            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
