"""Regression checks for the pre-generation preregistration firewall."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "verify_preregistration.py"
SPEC = importlib.util.spec_from_file_location("verify_preregistration", VERIFIER_PATH)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class PreregistrationTests(unittest.TestCase):
    def test_locked_firewall_verifies(self) -> None:
        self.assertEqual(VERIFIER.verify(ROOT), [])

    def test_tooling_results_are_ignored_but_real_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory)
            for relative in ("manifest.json", "digital-grimace-scale-full-roadmap-build-guide.md", "stimuli", "configs", "notes"):
                source = ROOT / relative
                target = copied / relative
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)

            fake = copied / ".venv" / "lib" / "python3.12" / "tests" / "results" / "foo.csv"
            fake.parent.mkdir(parents=True)
            fake.write_text("tooling fixture")
            self.assertEqual(VERIFIER.verify(copied), [])

            real = copied / "results" / "raw" / "foo.jsonl"
            real.parent.mkdir(parents=True)
            real.write_text("{}\n")
            errors = VERIFIER.verify(copied)
            self.assertTrue(any(error.replace("\\", "/").endswith("results/raw/foo.jsonl") for error in errors))

    def test_verifier_is_cwd_independent_and_needs_no_generation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, str(VERIFIER_PATH)],
                cwd=temporary_directory,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("PREREGISTRATION VERIFICATION PASSED", completed.stdout)

    def test_tampered_hashed_fixture_is_rejected_in_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory)
            for relative in ("manifest.json", "digital-grimace-scale-full-roadmap-build-guide.md", "stimuli", "configs", "notes"):
                source = ROOT / relative
                target = copied / relative
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            fixture = copied / "stimuli" / "matched_pairs.jsonl"
            fixture.write_bytes(fixture.read_bytes().replace(b"DGS-001", b"DGS-001", 1) + b" ")
            errors = VERIFIER.verify(copied)
            self.assertTrue(any("raw SHA-256 mismatch: matched_bank" in error for error in errors))
            manifest = copied / "manifest.json"
            value = __import__("json").loads(manifest.read_text())
            del value["file_sha256"]["r5_bank"]
            manifest.write_text(__import__("json").dumps(value))
            self.assertTrue(any("file_sha256 keys" in error for error in VERIFIER.verify(copied)))


if __name__ == "__main__":
    unittest.main()
