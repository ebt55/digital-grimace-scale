"""Regression checks for the pre-generation preregistration firewall."""

from __future__ import annotations

import importlib.util
import json
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

ROADMAP = "digital-grimace-scale-full-roadmap-build-guide.md"
FIREWALL_INPUTS = ("manifest.json", ROADMAP, "stimuli", "configs", "notes")


def copy_firewall_inputs(target: Path) -> None:
    """Copy the hash-locked inputs into `target`.

    The roadmap is committed by hash only and is not distributed, so a checkout may not have it;
    everything else is mandatory and is copied unconditionally.
    """
    for relative in FIREWALL_INPUTS:
        source = ROOT / relative
        if relative == ROADMAP and not source.is_file():
            continue
        destination = target / relative
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


class PreregistrationTests(unittest.TestCase):
    def test_locked_firewall_verifies(self) -> None:
        self.assertEqual(VERIFIER.verify(ROOT), [])

    def test_tooling_results_are_ignored_but_real_results_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory)
            copy_firewall_inputs(copied)

            # The no-artifacts sweep is the pre-generation firewall, so pin the copy to that
            # state; the committed manifest has since moved on to generation_status "ready".
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generation_status"] = "not_started"
            manifest["models"] = {"ids_in_order": manifest["models"]["ids_in_order"],
                                  "revisions": "unresolved_before_generation",
                                  "judge_provider": "unresolved_before_generation",
                                  "judge_model": "unresolved_before_generation"}
            manifest.pop("preflight", None)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

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
            copy_firewall_inputs(copied)
            fixture = copied / "stimuli" / "matched_pairs.jsonl"
            fixture.write_bytes(fixture.read_bytes().replace(b"DGS-001", b"DGS-001", 1) + b" ")
            errors = VERIFIER.verify(copied)
            self.assertTrue(any("raw SHA-256 mismatch: matched_bank" in error for error in errors))
            manifest = copied / "manifest.json"
            value = __import__("json").loads(manifest.read_text())
            del value["file_sha256"]["r5_bank"]
            manifest.write_text(__import__("json").dumps(value))
            self.assertTrue(any("file_sha256 keys" in error for error in VERIFIER.verify(copied)))

    def test_absent_roadmap_is_a_notice_but_a_modified_one_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory)
            copy_firewall_inputs(copied)
            roadmap = copied / ROADMAP
            roadmap.unlink(missing_ok=True)

            # Not distributed: the inventory and the frozen hash stay in manifest.json, and the
            # firewall passes with a notice rather than a missing-file failure.
            notices: list[str] = []
            self.assertEqual(VERIFIER.verify(copied, notices=notices), [])
            expected = json.loads((copied / "manifest.json").read_text(encoding="utf-8"))["file_sha256"]["roadmap"]
            self.assertEqual(len(notices), 1)
            self.assertIn("locked file 'roadmap' is not distributed with the repository", notices[0])
            self.assertIn(expected, notices[0])

            # Present but not the frozen bytes: still a hard failure, exactly as before.
            roadmap.write_text("not the authors' planning document\n", encoding="utf-8")
            errors = VERIFIER.verify(copied)
            self.assertTrue(any("raw SHA-256 mismatch: roadmap" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
