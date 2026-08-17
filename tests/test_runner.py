from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.protocol import discovery_tasks, load_protocol, render_task, style_smoke_tasks
from src.records import RecordError, compact_json, record_from_dict
from src.runner import (RunnerError, expected_turn_labels, holdout_tasks, plan_phase0_jobs,
    plan_phase1_jobs, plan_phase1_model_jobs, plan_phase2_model_jobs, plan_r5_jobs,
    plan_style_battery_jobs, plan_style_smoke_jobs, r5_task, run_batch,
    run_single_turn_trajectory, run_trajectory, write_jsonl_atomic)

EMPIRICAL_REVISION = "0123456789abcdef0123456789abcdef01234567"


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

    def test_run_kind_and_holdout_guard(self):
        empirical = run_trajectory(task=self.task, cell_id=self.task.difficulty + "__accurate__neutral",
                                   model_id=self.model, immutable_revision=EMPIRICAL_REVISION,
                                   protocol=self.protocol, run_kind="empirical", continuations=False)
        self.assertTrue(all(record.run_kind == "empirical" for record in empirical))
        self.assertTrue(all(record_from_dict(record.to_dict(), self.protocol) == record for record in empirical))
        self.assertTrue(all(record.run_kind == "synthetic_smoke" for record in self.trajectory(continuations=False)))
        with self.assertRaises(RecordError):  # empirical records demand a resolved 40-hex revision
            self.trajectory(run_kind="empirical", continuations=False)
        with self.assertRaises(RunnerError):
            self.trajectory(run_kind="dress_rehearsal")
        holdout = next(task for task in discovery_tasks(self.protocol, allow_holdout=True) if task.split == "holdout")
        arguments = dict(task=holdout, cell_id=holdout.difficulty + "__accurate__neutral", model_id=self.model,
                         immutable_revision="synthetic", protocol=self.protocol, continuations=False)
        with self.assertRaises(RunnerError):
            run_trajectory(**arguments)
        unlocked = run_trajectory(allow_holdout=True, **arguments)
        self.assertEqual(unlocked[0].split, "holdout")

    def test_run_batch_accepts_sample_indices(self):
        batch = run_batch(task=self.task, cell_id=self.task.difficulty + "__accurate__neutral", model_id=self.model,
                          immutable_revision="synthetic", protocol=self.protocol, continuations=False,
                          sample_indices=(0, 4, 9))
        self.assertEqual([record.sample_index for record in batch], [0] * 5 + [4] * 5 + [9] * 5)
        self.assertEqual({record.trajectory_kind for record in batch}, {"greedy", "resample"})

    def test_single_turn_style_wording(self):
        task = style_smoke_tasks(self.protocol)[0]
        rendered = render_task(task.prompt, task.options, self.protocol)
        prompts = self.protocol.conditions["style_only_controls"]["prompts"]
        for cell, prefix in [("style__neutral_reference", None)] + [("style__" + name, text) for name, text in prompts.items()]:
            record, = run_single_turn_trajectory(task=task, cell_id=cell, model_id=self.model,
                                                 immutable_revision="synthetic", protocol=self.protocol)
            self.assertEqual(record.messages[0]["content"], rendered if prefix is None else "%s\n\n%s" % (prefix, rendered))
            self.assertEqual((record.turn_label, record.feedback_validity, record.tone), ("measured", None, None))
            self.assertEqual((record.split, record.difficulty, record.domain), (task.split, task.difficulty, task.domain))
            self.assertIsNone(record.feedback_history_false_negative)
            self.assertEqual(record_from_dict(record.to_dict(), self.protocol).to_dict(), record.to_dict())

    def test_single_turn_rejects_factorial_cells_and_bad_samples(self):
        task = style_smoke_tasks(self.protocol)[0]
        arguments = dict(task=task, model_id=self.model, immutable_revision="synthetic", protocol=self.protocol)
        with self.assertRaises(RunnerError):
            run_single_turn_trajectory(cell_id=task.difficulty + "__accurate__neutral", **arguments)
        with self.assertRaises(RunnerError):
            run_single_turn_trajectory(cell_id="style__enthusiastic", sample_index=11, **arguments)
        with self.assertRaises(RunnerError):
            run_single_turn_trajectory(cell_id="style__enthusiastic", sample_index=True, **arguments)
        with self.assertRaises(RunnerError):
            run_single_turn_trajectory(cell_id="r5__pressure", **arguments)  # matched task is not an R5 item

    def test_phase2_planners_cover_only_the_holdout_split(self):
        tasks = holdout_tasks(self.protocol)
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(task.split == "holdout" for task in tasks))
        factorial = plan_phase2_model_jobs(self.model, self.protocol)
        battery = plan_style_battery_jobs(self.model, self.protocol)
        self.assertEqual((len(factorial), len(battery)), (80, 100))
        holdout_ids = {task.task_id for task in tasks}
        discovery_ids = {task.task_id for task in discovery_tasks(self.protocol)}
        for jobs in (factorial, battery):
            self.assertEqual({job.phase for job in jobs}, {"phase_2"})
            self.assertEqual({job.task_id for job in jobs}, holdout_ids)
            self.assertFalse({job.task_id for job in jobs} & discovery_ids)
        self.assertEqual(len({job.cell_id for job in factorial}), 8)
        self.assertEqual({job.cell_id for job in battery}, set(self.protocol.nonfactorial_cell_ids[:5]))
        self.assertTrue(all(job.cell_id.startswith("style__") for job in battery))
        # every planned job must be executable only under an explicit holdout unlock
        job = factorial[0]
        task = next(item for item in tasks if item.task_id == job.task_id)
        arguments = dict(task=task, cell_id=job.cell_id, model_id=self.model, immutable_revision="synthetic",
                         protocol=self.protocol, phase=job.phase, continuations=False)
        with self.assertRaises(RunnerError): run_trajectory(**arguments)
        self.assertEqual(run_trajectory(allow_holdout=True, **arguments)[0].split, "holdout")
        with self.assertRaises(RunnerError): plan_phase2_model_jobs("not/a-model", self.protocol)
        with self.assertRaises(RunnerError): plan_style_battery_jobs("not/a-model", self.protocol)

    def test_exploratory_extension_model_is_plannable_in_every_phase(self):
        model = "meta-llama/Llama-3.1-8B-Instruct"
        self.assertIn(model, self.protocol.extension_model_ids)
        counts = {"phase0": len(plan_phase0_jobs((model,), self.protocol)),
                  "phase1": len(plan_phase1_model_jobs(model, self.protocol)),
                  "phase2": len(plan_phase2_model_jobs(model, self.protocol)),
                  "style_smoke": len(plan_style_smoke_jobs(model, self.protocol)),
                  "style_battery": len(plan_style_battery_jobs(model, self.protocol)),
                  "r5": len(plan_r5_jobs(model, self.protocol))}
        self.assertEqual(counts, {"phase0": 20, "phase1": 80, "phase2": 80,
                                  "style_smoke": 25, "style_battery": 100, "r5": 20})
        self.assertEqual({job.model_id for job in plan_phase1_model_jobs(model, self.protocol)}, {model})
        # it is exploratory only: it never joins the frozen Phase-0 screen order
        self.assertNotIn(model, self.protocol.models["phase_0_screen_order"])
        records = run_trajectory(task=self.task, cell_id=self.task.difficulty + "__accurate__neutral",
                                 model_id=model, immutable_revision="synthetic", protocol=self.protocol,
                                 continuations=False)
        self.assertEqual({record.model_id for record in records}, {model})
        with self.assertRaises(RunnerError): plan_phase1_model_jobs("meta-llama/Llama-3.1-70B-Instruct", self.protocol)

    def test_non_factorial_planners_and_turn_plans(self):
        style, r5 = plan_style_smoke_jobs(self.model, self.protocol), plan_r5_jobs(self.model, self.protocol)
        self.assertEqual((len(style), len(r5)), (25, 20))
        self.assertTrue(all(job.feedback_rounds == 0 for job in style + r5))
        self.assertEqual(len(plan_phase1_model_jobs(self.model, self.protocol)), 80)
        self.assertEqual(plan_phase1_jobs(self.model, "Qwen/Qwen2.5-7B-Instruct", self.protocol)[:80],
                         plan_phase1_model_jobs(self.model, self.protocol))
        self.assertEqual(expected_turn_labels("style__verbose", None, self.protocol), ("measured",))
        self.assertEqual(expected_turn_labels("easy__accurate__neutral", 3, self.protocol),
                         ("initial", "feedback_response_1", "feedback_response_2", "feedback_response_3", "measured", "onset", "onset_washout"))
        self.assertEqual(expected_turn_labels("hard__malfunctioning_always_fail__hostile", 5, self.protocol)[-1], "recovery")
        self.assertEqual(len(expected_turn_labels("hard__malfunctioning_always_fail__hostile", 5, self.protocol)), 8)
        with self.assertRaises(RunnerError): expected_turn_labels("nonsense__cell", None, self.protocol)
        with self.assertRaises(RunnerError): r5_task("R5-001", "style__verbose", self.protocol)
        with self.assertRaises(RunnerError): r5_task("DGS-001", "r5__pressure", self.protocol)
        with self.assertRaises(RunnerError): plan_style_smoke_jobs("not/a-model", self.protocol)
