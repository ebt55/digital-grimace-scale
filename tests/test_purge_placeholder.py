"""Purging trajectories poisoned by the empty-stream placeholder token."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "purge_placeholder_trajectories", ROOT / "scripts" / "purge_placeholder_trajectories.py")
assert SPEC and SPEC.loader
PURGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PURGE)

PLACEHOLDER = [{"text": "", "logprob": 0.0, "top_logprobs": [{"text": "", "logprob": 0.0}]}]
REAL = [{"text": "Answer: C", "logprob": -0.1, "top_logprobs": [{"text": "Answer: C", "logprob": -0.1}]}]
# A genuine EOS-only empty response: empty text, but a real distribution -- must survive.
GENUINE_EMPTY = [{"text": "", "logprob": -0.03,
                  "top_logprobs": [{"text": "<end_of_turn>", "logprob": -0.03}]}]


def record(model="m", task="DGS-003", cell="easy__accurate__neutral", sample=0, turn="initial",
           tokens=None):
    return {"model_id": model, "task_id": task, "cell_id": cell, "sample_index": sample,
            "turn_label": turn, "tokens": tokens if tokens is not None else REAL}


class PurgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
        return path

    def read(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]

    def test_whole_affected_trajectory_is_removed_and_others_survive(self):
        path = self.write("raw.jsonl", [
            record(sample=0, turn="initial", tokens=PLACEHOLDER),   # poisoned
            record(sample=0, turn="measured"),                      # same trajectory
            record(sample=1, turn="initial"),                       # different sample
            record(task="DGS-004", turn="initial"),                 # different task
        ])
        trajectories, removed, kept = PURGE.purge_file(path, dry_run=False)
        self.assertEqual((trajectories, removed, kept), (1, 2, 2))
        survivors = {(r["task_id"], r["sample_index"]) for r in self.read(path)}
        self.assertEqual(survivors, {("DGS-003", 1), ("DGS-004", 0)})

    def test_genuine_eos_only_empty_response_is_not_purged(self):
        path = self.write("raw.jsonl", [record(tokens=GENUINE_EMPTY), record(turn="measured")])
        self.assertEqual(PURGE.purge_file(path, dry_run=False), (0, 0, 2))
        self.assertEqual(len(self.read(path)), 2)

    def test_dry_run_reports_without_writing(self):
        records = [record(turn="initial", tokens=PLACEHOLDER), record(turn="measured")]
        path = self.write("raw.jsonl", records)
        before = path.read_bytes()
        self.assertEqual(PURGE.purge_file(path, dry_run=True), (1, 2, 0))
        self.assertEqual(path.read_bytes(), before)

    def test_unparseable_lines_are_preserved(self):
        path = self.write("raw.jsonl", [record(turn="measured")])
        path.write_text(path.read_text(encoding="utf-8") + "not json\n", encoding="utf-8", newline="\n")
        PURGE.purge_file(path, dry_run=False)
        self.assertIn("not json", path.read_text(encoding="utf-8"))

    def test_cli_walks_a_directory_and_skips_failure_sidecars(self):
        (self.root / "phase0").mkdir()
        self.write("phase0/a.jsonl", [record(turn="initial", tokens=PLACEHOLDER)])
        self.write("phase0/b.jsonl", [record(turn="measured")])
        self.write("phase0/a.jsonl.failures.jsonl", [record(turn="initial", tokens=PLACEHOLDER)])
        files = PURGE.raw_files(self.root)
        self.assertEqual([p.name for p in files], ["a.jsonl", "b.jsonl"])
        self.assertEqual(PURGE.main(["--raw", str(self.root)]), 0)
        self.assertEqual(self.read(self.root / "phase0" / "a.jsonl"), [])
        self.assertEqual(len(self.read(self.root / "phase0" / "b.jsonl")), 1)
