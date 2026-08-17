from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

from src.backend import SyntheticBackend
from src.generate import GenerateError, format_summary, run_jobs
from src.protocol import deterministic_seed, discovery_tasks, load_protocol, render_r5_variant
from src.records import record_from_json
from src.runner import (PlannedJob, plan_r5_jobs, plan_style_smoke_jobs, r5_task,
                        run_single_turn_trajectory)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "synthetic"


class CountingBackend:
    """SyntheticBackend wrapper that counts calls and can fail one designated trajectory."""

    name = "synthetic"

    def __init__(self, fail_seeds: frozenset[int] = frozenset()):
        self._inner = SyntheticBackend()
        self._lock = threading.Lock()
        self._fail_seeds = fail_seeds
        self.calls = 0
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def generate(self, request):
        with self._lock:
            self.calls += 1
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += sum(len(message["content"]) for message in request.messages)
            self.stats["completion_tokens"] += 2
        if request.seed in self._fail_seeds:
            raise RuntimeError("simulated backend failure")
        return self._inner.generate(request)


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.tasks = discovery_tasks(cls.protocol)[:2]
        cls.jobs = tuple(PlannedJob("phase_1", MODEL, task.task_id,
                                    "%s__%s__neutral" % (task.difficulty, validity),
                                    cls.protocol.standard_feedback_round_count)
                         for task in cls.tasks
                         for validity in ("accurate", "malfunctioning_always_fail"))
        cls.samples = (0, 1, 2)
        # accurate: initial + 3 feedback + measured + onset + washout; malfunctioning: ... + recovery
        cls.turns = {"accurate": 7, "malfunctioning_always_fail": 6}
        cls.expected_records = len(cls.samples) * sum(cls.turns[job.cell_id.split("__")[1]] for job in cls.jobs)

    def run_plan(self, out_path, backend, *, jobs=None, samples=None, **kwargs):
        return run_jobs(self.jobs if jobs is None else jobs, backend=backend, out_path=out_path, immutable_revision=REVISION,
                        run_id="generate-test", run_kind="synthetic_smoke",
                        sample_indices=self.samples if samples is None else samples,
                        protocol=self.protocol, **kwargs)

    def read(self, path):
        return [record_from_json(line, self.protocol) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_full_run_then_resume_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.jsonl"
            first = CountingBackend()
            summary = self.run_plan(out, first, max_workers=4)
            self.assertEqual((summary.planned, summary.skipped, summary.completed, summary.failed), (12, 0, 12, 0))
            self.assertEqual(summary.records_written, self.expected_records)
            records = self.read(out)
            self.assertEqual(len(records), self.expected_records)
            self.assertEqual(len({record.response_id for record in records}), self.expected_records)
            self.assertTrue(all(record.run_kind == "synthetic_smoke" for record in records))
            self.assertEqual(first.calls, self.expected_records)
            self.assertFalse(summary.failures_path.exists())
            self.assertIn("planned 12", format_summary(summary))

            second = CountingBackend()
            resumed = self.run_plan(out, second, max_workers=4)
            self.assertEqual((resumed.planned, resumed.skipped, resumed.completed, resumed.failed), (12, 12, 0, 0))
            self.assertEqual((second.calls, resumed.records_dropped), (0, 0))
            self.assertEqual(len(self.read(out)), self.expected_records)

    def test_resume_reruns_only_the_truncated_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.jsonl"
            self.run_plan(out, CountingBackend(), max_workers=4)
            lines = out.read_text(encoding="utf-8").splitlines()
            victim = record_from_json(lines[-1], self.protocol)
            out.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            backend = CountingBackend()
            resumed = self.run_plan(out, backend, max_workers=4)
            turns = self.turns[victim.cell_id.split("__")[1]]
            self.assertEqual((resumed.completed, resumed.skipped, resumed.failed), (1, 11, 0))
            self.assertEqual(resumed.records_dropped, turns - 1)
            self.assertEqual(backend.calls, turns)
            records = self.read(out)
            self.assertEqual(len(records), self.expected_records)
            self.assertEqual(len({record.response_id for record in records}), self.expected_records)
            rerun = [record for record in records
                     if (record.task_id, record.cell_id, record.sample_index) == (victim.task_id, victim.cell_id, victim.sample_index)]
            self.assertEqual(len(rerun), turns)

    def test_one_failing_trajectory_is_isolated_in_the_sidecar(self):
        target = self.jobs[0]
        seed = deterministic_seed(MODEL, REVISION, target.task_id, target.cell_id, "initial", 1, self.protocol)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.jsonl"
            summary = self.run_plan(out, CountingBackend(frozenset({seed})), max_workers=4)
            self.assertEqual((summary.completed, summary.failed), (11, 1))
            self.assertFalse(summary.ok)
            entries = [json.loads(line) for line in summary.failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(entries), 1)
            self.assertEqual(set(entries[0]), {"job", "sample_index", "error", "traceback"})
            self.assertEqual((entries[0]["job"]["task_id"], entries[0]["sample_index"]), (target.task_id, 1))
            self.assertIn("simulated backend failure", entries[0]["error"])
            self.assertIn("RuntimeError", entries[0]["traceback"])
            records = self.read(out)
            self.assertEqual(len(records), self.expected_records - self.turns["accurate"])
            self.assertFalse(any((r.task_id, r.cell_id, r.sample_index) == (target.task_id, target.cell_id, 1) for r in records))
            # the failed trajectory is simply pending again on the next resume
            retry = self.run_plan(out, CountingBackend(), max_workers=4)
            self.assertEqual((retry.skipped, retry.completed, retry.failed), (11, 1, 0))
            self.assertFalse(retry.failures_path.exists())
            self.assertEqual(len(self.read(out)), self.expected_records)

    def test_concurrency_does_not_change_the_result_set(self):
        with tempfile.TemporaryDirectory() as directory:
            serial, parallel = Path(directory) / "one.jsonl", Path(directory) / "eight.jsonl"
            self.run_plan(serial, CountingBackend(), max_workers=1)
            self.run_plan(parallel, CountingBackend(), max_workers=8)
            self.assertEqual(sorted(record.response_id for record in self.read(serial)),
                             sorted(record.response_id for record in self.read(parallel)))
            self.assertEqual(sorted(record.prompt_sha256 for record in self.read(serial)),
                             sorted(record.prompt_sha256 for record in self.read(parallel)))

    def test_non_factorial_jobs_run_single_turn_and_planners_are_frozen(self):
        style = plan_style_smoke_jobs(MODEL, self.protocol)
        r5 = plan_r5_jobs(MODEL, self.protocol)
        self.assertEqual((len(style), len(r5)), (25, 20))
        self.assertEqual(len({job.cell_id for job in style}), 5)
        self.assertEqual({job.task_id for job in style}, set(self.protocol.style_smoke_task_ids))
        self.assertEqual({job.cell_id for job in r5}, {"r5__pressure", "r5__neutral_control"})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "nonfactorial.jsonl"
            summary = self.run_plan(out, CountingBackend(), jobs=style[:3] + r5[:2], samples=(0, 1), max_workers=4)
            self.assertEqual((summary.planned, summary.completed, summary.records_written), (10, 10, 10))
            records = self.read(out)
            self.assertTrue(all(record.turn_label == "measured" for record in records))
            self.assertTrue(all(record.feedback_validity is None and record.tone is None for record in records))
            self.assertTrue(all(len(record.messages) == 1 for record in records))
            self.assertEqual({record.sample_index for record in records}, {0, 1})
            self.assertEqual({record.trajectory_kind for record in records}, {"greedy", "resample"})

    def test_r5_records_validate_and_carry_bank_metadata(self):
        task = r5_task("R5-001", "r5__pressure", self.protocol)
        self.assertEqual((task.split, task.difficulty), (None, None))
        record, = run_single_turn_trajectory(task=task, cell_id="r5__pressure", model_id=MODEL,
                                             immutable_revision=REVISION, protocol=self.protocol, phase="phase_2")
        self.assertEqual((record.split, record.difficulty, record.domain), (None, None, task.domain))
        item = next(entry for entry in self.protocol.r5_tasks if entry["task_id"] == "R5-001")
        self.assertEqual(record.messages[0]["content"], render_r5_variant(item, "pressure", self.protocol))
        self.assertEqual(record_from_json(json.dumps(record.to_dict()), self.protocol).to_dict(), record.to_dict())

    def test_invalid_plans_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.jsonl"
            with self.assertRaises(GenerateError):
                self.run_plan(out, CountingBackend(), jobs=())
            with self.assertRaises(GenerateError):
                self.run_plan(out, CountingBackend(), samples=(0, 11))
            with self.assertRaises(GenerateError):
                self.run_plan(out, CountingBackend(), samples=(0,), max_workers=0)
            with self.assertRaises(GenerateError):
                self.run_plan(out, CountingBackend(), jobs=self.jobs + (self.jobs[0],), samples=(0,))
            self.assertFalse(out.exists())
