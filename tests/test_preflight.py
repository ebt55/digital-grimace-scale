"""Preflight pins revisions and judge identity without touching any frozen file."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / ("%s.py" % name))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load("preflight")
VERIFIER = _load("verify_preregistration")

RESOLVABLE = {
    "google/gemma-2-2b-it": "a" * 40,
    "google/gemma-2-9b-it": "b" * 40,
    "Qwen/Qwen2.5-3B-Instruct": "C" * 40,
    "Qwen/Qwen2.5-7B-Instruct": "d" * 40,
}
GATED = "meta-llama/Llama-3.2-3B-Instruct"


class GatedRepoError(Exception):
    """Mirrors huggingface_hub's gated-repo failure by class name."""


class FakeInfo:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class FakeApi:
    """`shas` decides metadata access; `blocked_files` decides weight-file access."""

    def __init__(self, shas=None, blocked_files=()) -> None:
        self.shas = dict(RESOLVABLE if shas is None else shas)
        self.blocked_files = set(blocked_files)
        self.calls: list[str] = []
        self.file_calls: list[str] = []

    def model_info(self, model_id: str) -> FakeInfo:
        self.calls.append(model_id)
        if model_id not in self.shas:
            raise GatedRepoError("403 gated repo: %s" % model_id)
        return FakeInfo(self.shas[model_id])

    def dgs_probe_file(self, model_id: str, filename: str) -> object:
        self.file_calls.append(model_id)
        if model_id in self.blocked_files:
            raise GatedRepoError("403 cannot access gated repo file: %s/%s" % (model_id, filename))
        return object()


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in ("manifest.json", "digital-grimace-scale-full-roadmap-build-guide.md",
                         "stimuli", "configs", "notes"):
            source = ROOT / relative
            destination = self.root / relative
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        # Rewind the copy to the pre-generation sentinel state so these tests describe what
        # preflight does, not whatever the committed manifest happens to be pinned to today.
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["generation_status"] = "not_started"
        manifest["models"] = {"ids_in_order": manifest["models"]["ids_in_order"],
                              "revisions": PREFLIGHT.UNRESOLVED,
                              "judge_provider": PREFLIGHT.UNRESOLVED,
                              "judge_model": PREFLIGHT.UNRESOLVED}
        manifest.pop("preflight", None)
        (self.root / "manifest.json").write_text(PREFLIGHT.dump_manifest(manifest),
                                                 encoding="utf-8", newline="\n")
        self.locked = {path: hashlib.sha256((self.root / path).read_bytes()).hexdigest()
                       for path in manifest["files"].values()}

    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def run_preflight(self, *args: str, api=None, probe=None) -> tuple[int, str]:
        stream = io.StringIO()
        resolver = PREFLIGHT.HubResolver(api=api if api is not None else FakeApi())
        code = PREFLIGHT.run(["--root", str(self.root), *args], resolver=resolver, probe=probe, stream=stream)
        return code, stream.getvalue()

    def assert_locked_untouched(self) -> None:
        for path, digest in self.locked.items():
            self.assertEqual(hashlib.sha256((self.root / path).read_bytes()).hexdigest(), digest, path)

    # -- core behaviour ----------------------------------------------------
    def test_dump_manifest_reproduces_the_committed_manifest_byte_for_byte(self) -> None:
        original = (ROOT / "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(PREFLIGHT.dump_manifest(json.loads(original)), original)

    def test_resolves_revisions_records_unavailable_and_leaves_locked_files_alone(self) -> None:
        code, output = self.run_preflight("--models", *RESOLVABLE, GATED, "--generation-status", "ready")
        self.assertEqual(code, 0, output)
        manifest = self.manifest()
        self.assertEqual(manifest["models"]["revisions"], {key: value.lower() for key, value in RESOLVABLE.items()})
        self.assertIn(GATED, manifest["models"]["unavailable"])
        self.assertTrue(manifest["models"]["unavailable"][GATED].startswith("hf_403_no_license_"))
        self.assertEqual(manifest["generation_status"], "ready")
        self.assertEqual(manifest["models"]["judge_provider"], "anthropic")
        self.assertEqual(manifest["models"]["judge_model"], "claude-sonnet-4-6")
        self.assertNotEqual(manifest["updated_at"], manifest["created_at"])
        self.assert_locked_untouched()

    def test_public_metadata_with_gated_weights_is_marked_unavailable(self) -> None:
        # meta-llama serves a public model card and a real sha, then 403s every file HEAD.
        api = FakeApi(shas={**RESOLVABLE, GATED: "e" * 40}, blocked_files={GATED})
        code, output = self.run_preflight("--models", *RESOLVABLE, GATED,
                                          "--generation-status", "ready", api=api)
        self.assertEqual(code, 0, output)
        models = self.manifest()["models"]
        self.assertNotIn(GATED, models["revisions"])
        self.assertIn(GATED, models["unavailable"])
        self.assertEqual(sorted(api.file_calls), sorted(RESOLVABLE) + [GATED])

    def test_untouched_manifest_content_is_preserved_exactly(self) -> None:
        before = self.manifest()
        self.run_preflight("--models", *RESOLVABLE, "--generation-status", "ready")
        after = self.manifest()
        for key in set(before) - {"models", "generation_status", "updated_at"}:
            self.assertEqual(after[key], before[key], key)

    def test_verifier_passes_before_and_after_preflight(self) -> None:
        self.assertEqual(VERIFIER.verify(self.root), [])
        self.run_preflight("--models", *RESOLVABLE, GATED, "--generation-status", "ready")
        self.assertEqual(VERIFIER.verify(self.root), [])

    def test_verifier_rejects_short_revisions_once_generation_has_started(self) -> None:
        self.run_preflight("--models", *RESOLVABLE, GATED, "--generation-status", "ready")
        manifest = self.manifest()
        manifest["models"]["revisions"]["google/gemma-2-2b-it"] = "main"
        (self.root / "manifest.json").write_text(PREFLIGHT.dump_manifest(manifest), encoding="utf-8", newline="\n")
        self.assertTrue(any("40-hex" in problem for problem in VERIFIER.verify(self.root)))

    def test_verifier_rejects_a_model_that_is_neither_pinned_nor_unavailable(self) -> None:
        self.run_preflight("--models", *RESOLVABLE, "--generation-status", "ready")
        self.assertTrue(any("neither pinned nor marked unavailable" in problem
                            for problem in VERIFIER.verify(self.root)))

    def test_idempotent_second_run_writes_nothing(self) -> None:
        self.run_preflight("--models", *RESOLVABLE, GATED, "--generation-status", "ready")
        first = (self.root / "manifest.json").read_bytes()
        code, output = self.run_preflight("--models", *RESOLVABLE, GATED, "--generation-status", "ready")
        self.assertEqual(code, 0, output)
        self.assertIn("already pinned", output)
        self.assertEqual((self.root / "manifest.json").read_bytes(), first)

    def test_dry_run_prints_but_does_not_write(self) -> None:
        before = (self.root / "manifest.json").read_bytes()
        code, output = self.run_preflight("--models", *RESOLVABLE, "--generation-status", "ready", "--dry-run")
        self.assertEqual(code, 0, output)
        self.assertIn("dry run", output)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)

    def test_judge_identity_is_overridable(self) -> None:
        self.run_preflight("--models", "google/gemma-2-2b-it", "--judge-provider", "openai",
                           "--judge-model", "gpt-x", "--judge-revision", "2026-08-01")
        models = self.manifest()["models"]
        self.assertEqual((models["judge_provider"], models["judge_model"], models["judge_revision"]),
                         ("openai", "gpt-x", "2026-08-01"))

    def test_letter_token_check_recorded_from_live_endpoint(self) -> None:
        seen: list[tuple[str, str, str]] = []

        def probe(base_url, model, api_key):
            seen.append((base_url, model, api_key))
            return {"A": True, "B": True, "C": True, "D": False}

        self.run_preflight("--models", "google/gemma-2-2b-it", "--endpoint", "https://x.modal.run/v1",
                           probe=probe)
        check = self.manifest()["preflight"]["letter_token_check"]
        self.assertEqual(seen, [("https://x.modal.run/v1", "google/gemma-2-2b-it", "EMPTY")])
        self.assertEqual(check["results"], {"A": True, "B": True, "C": True, "D": False})
        self.assertIs(check["all_single_tokens"], False)

    def test_non_access_resolution_failures_abort_without_writing(self) -> None:
        class Broken:
            def model_info(self, model_id):
                raise RuntimeError("network down")

        before = (self.root / "manifest.json").read_bytes()
        code, output = self.run_preflight("--models", "google/gemma-2-2b-it", api=Broken())
        self.assertEqual(code, 1)
        self.assertIn("preflight failed", output)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)

    def test_frozen_file_drift_aborts_before_any_write(self) -> None:
        drifted = self.root / "configs" / "models.json"
        drifted.write_text(drifted.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        before = (self.root / "manifest.json").read_bytes()
        code, output = self.run_preflight("--models", "google/gemma-2-2b-it")
        self.assertEqual(code, 1)
        self.assertIn("frozen file changed", output)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)

    def test_check_keys_reports_presence_only(self) -> None:
        (self.root / ".env").write_text("ANTHROPIC_API_KEY=sk-secret-value\n", encoding="utf-8")
        code, output = self.run_preflight("--models", "google/gemma-2-2b-it", "--check-keys")
        self.assertEqual(code, 0, output)
        self.assertIn("ANTHROPIC_API_KEY=dotenv", output)
        self.assertIn("OPENAI_API_KEY=absent", output)
        self.assertNotIn("sk-secret-value", output)
