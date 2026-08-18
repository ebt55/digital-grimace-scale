"""The container re-imports serve_modal.py with the deploy shell's environment gone.

Regression guard for the bug where every app served google/gemma-2-2b-it because the
module-level DGS_MODEL_ID read fell back to its default inside the container.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "serve_modal.py"


class _Chain:
    """Accepts any attribute access, call, or decoration; enough to import the module."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _Chain()


def load_serve_modal(environment: dict[str, str]):
    """Import serve_modal.py under a controlled environment, with modal/HF stubbed out."""
    modal_stub = types.ModuleType("modal")
    for attribute in ("Image", "Volume", "Secret", "App", "concurrent", "web_server"):
        setattr(modal_stub, attribute, _Chain())
    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.get_token = lambda: "stub-token"  # never a real secret in tests

    spec = importlib.util.spec_from_file_location("serve_modal_under_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    saved = {name: sys.modules.get(name) for name in ("modal", "huggingface_hub")}
    sys.modules["modal"] = modal_stub
    sys.modules["huggingface_hub"] = hub_stub
    try:
        with mock.patch.dict("os.environ", environment, clear=True):
            spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


DEPLOY_ENVIRONMENT = {
    "DGS_MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",
    "DGS_REVISION": "a" * 40,
    "DGS_GPU": "L40S",
}


class ConfigResolutionTests(unittest.TestCase):
    def test_deploy_pass_reads_the_ambient_environment(self):
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        self.assertEqual(module.MODEL_ID, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(module.REVISION, "a" * 40)
        self.assertEqual(module.GPU, "L40S")
        self.assertEqual(module.APP_NAME, "dgs-vllm-qwen2-5-7b-instruct")
        self.assertEqual(module.CONFIG["source"], "environment")

    def test_container_reimport_without_the_deploy_environment_uses_baked_values(self):
        deploy = load_serve_modal(DEPLOY_ENVIRONMENT)
        # The container only ever sees what the image baked in -- no DGS_* names at all.
        container = load_serve_modal(dict(deploy.BAKED_ENVIRONMENT))
        self.assertEqual(container.MODEL_ID, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(container.REVISION, "a" * 40)
        self.assertEqual(container.GPU, "L40S")
        self.assertEqual(container.APP_NAME, "dgs-vllm-qwen2-5-7b-instruct")
        self.assertEqual(container.CONFIG["source"], "baked")
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", container.vllm_command())
        self.assertIn("--revision", container.vllm_command())

    def test_container_reimport_with_no_environment_at_all_falls_back_to_the_default(self):
        # This is exactly what used to happen in production: every app served the default.
        container = load_serve_modal({})
        self.assertEqual(container.MODEL_ID, "google/gemma-2-2b-it")
        self.assertEqual(container.CONFIG["source"], "environment")

    def test_baked_values_win_over_conflicting_ambient_values(self):
        module = load_serve_modal({
            "DGS_MODEL_ID": "google/gemma-2-2b-it", "DGS_GPU": "A10G",
            "DGS_BAKED_MODEL_ID": "google/gemma-2-9b-it", "DGS_BAKED_GPU": "L40S",
            "DGS_BAKED_REVISION": "b" * 40, "DGS_BAKED_VLLM_VERSION": "0.26.0",
        })
        self.assertEqual(module.MODEL_ID, "google/gemma-2-9b-it")
        self.assertEqual(module.GPU, "L40S")

    def test_image_bakes_every_setting_the_container_needs(self):
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        self.assertEqual(set(module.BAKED_ENVIRONMENT), {
            "DGS_BAKED_MODEL_ID", "DGS_BAKED_REVISION", "DGS_BAKED_GPU",
            "DGS_BAKED_VLLM_VERSION", "DGS_BAKED_ATTENTION_BACKEND"})
        for key, value in module.BAKED_ENVIRONMENT.items():
            self.assertIn(key, module.image_environment)
            self.assertEqual(module.image_environment[key], value)


class GpuAndNamingTests(unittest.TestCase):
    def test_app_name_and_default_gpu_track_the_model(self):
        module = load_serve_modal({})
        for model_id, name, gpu in (
            ("google/gemma-2-2b-it", "dgs-vllm-gemma-2-2b-it", "A10G"),
            ("Qwen/Qwen2.5-3B-Instruct", "dgs-vllm-qwen2-5-3b-instruct", "A10G"),
            ("google/gemma-2-9b-it", "dgs-vllm-gemma-2-9b-it", "L40S"),
            ("Qwen/Qwen2.5-7B-Instruct", "dgs-vllm-qwen2-5-7b-instruct", "L40S"),
        ):
            self.assertEqual(module.app_name(model_id), name)
            self.assertEqual(module.default_gpu(model_id), gpu)

    def test_distinct_models_never_share_an_app_name(self):
        module = load_serve_modal({})
        models = ("google/gemma-2-2b-it", "google/gemma-2-9b-it",
                  "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(len({module.app_name(m) for m in models}), len(models))


LOCAL_ENVIRONMENT = {
    "DGS_MODEL_PATH": "/adapters/A/merged",
    "DGS_SERVED_NAME": "google/gemma-2-9b-it+dpo-A",
    "DGS_REVISION": "c" * 40,
}


class LocalMergedWeightsTests(unittest.TestCase):
    """Phase 4 serves merged DPO weights from a volume under a name of its own."""

    def test_local_weights_deploy_resolves_path_name_app_and_gpu(self):
        module = load_serve_modal(LOCAL_ENVIRONMENT)
        self.assertEqual(module.MODEL_PATH, "/adapters/A/merged")
        self.assertEqual(module.SERVED_NAME, "google/gemma-2-9b-it+dpo-A")
        self.assertEqual(module.APP_NAME, "dgs-vllm-gemma-2-9b-it-dpo-a")
        self.assertEqual(module.GPU, "L40S")  # derived from the served name, not the unused default id

    def test_vllm_serves_the_directory_under_the_advertised_name_and_drops_the_revision(self):
        command = load_serve_modal(LOCAL_ENVIRONMENT).vllm_command()
        self.assertEqual(command[:2], ["vllm", "serve"])
        self.assertEqual(command[2], "/adapters/A/merged")
        self.assertEqual(command[command.index("--served-model-name") + 1],
                         "google/gemma-2-9b-it+dpo-A")
        # A local directory has no Hugging Face revision; passing one aborts the server.
        self.assertNotIn("--revision", command)

    def test_the_adapter_volume_is_mounted_only_in_local_weights_mode(self):
        local = load_serve_modal(LOCAL_ENVIRONMENT)
        self.assertEqual(set(local.VOLUMES), {local.HF_CACHE, local.ADAPTER_MOUNT})
        hub = load_serve_modal(DEPLOY_ENVIRONMENT)
        self.assertEqual(set(hub.VOLUMES), {hub.HF_CACHE})

    def test_container_reimport_reproduces_the_local_weights_configuration(self):
        deploy = load_serve_modal(LOCAL_ENVIRONMENT)
        container = load_serve_modal(dict(deploy.BAKED_ENVIRONMENT))
        self.assertEqual(container.CONFIG["source"], "baked")
        self.assertEqual(container.MODEL_PATH, "/adapters/A/merged")
        self.assertEqual(container.SERVED_NAME, "google/gemma-2-9b-it+dpo-A")
        self.assertEqual(container.APP_NAME, deploy.APP_NAME)
        self.assertEqual(container.vllm_command(), deploy.vllm_command())

    def test_a_hugging_face_deploy_bakes_exactly_what_it_baked_before_phase_4(self):
        # The image environment is part of the image's identity: adding keys unconditionally
        # would rebuild -- and thus re-download -- every existing model's image.
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        self.assertEqual(set(module.BAKED_ENVIRONMENT), {
            "DGS_BAKED_MODEL_ID", "DGS_BAKED_REVISION", "DGS_BAKED_GPU",
            "DGS_BAKED_VLLM_VERSION", "DGS_BAKED_ATTENTION_BACKEND"})
        self.assertIsNone(module.MODEL_PATH)
        self.assertEqual(module.SERVED_NAME, module.MODEL_ID)
        self.assertIn("--revision", module.vllm_command())

    def test_half_a_local_configuration_is_refused(self):
        for environment in ({"DGS_MODEL_PATH": "/adapters/A/merged"},
                            {"DGS_SERVED_NAME": "google/gemma-2-9b-it+dpo-A"}):
            with self.assertRaises(RuntimeError):
                load_serve_modal(environment)

    def test_a_path_outside_the_volume_mount_is_refused(self):
        with self.assertRaises(RuntimeError):
            load_serve_modal({"DGS_MODEL_PATH": "/root/merged",
                              "DGS_SERVED_NAME": "google/gemma-2-9b-it+dpo-A"})


class ServedModelGuardTests(unittest.TestCase):
    def response(self, ids):
        payload = ('{"data": [%s]}' % ", ".join('{"id": "%s"}' % i for i in ids)).encode()

        class Response:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return payload
        return Response()

    def process(self, poll=None):
        return mock.Mock(poll=mock.Mock(return_value=poll), returncode=poll)

    def test_matching_served_id_passes(self):
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        with mock.patch("urllib.request.urlopen", return_value=self.response(["Qwen/Qwen2.5-7B-Instruct"])):
            module._assert_serving_intended_model(self.process())

    def test_wrong_served_id_raises(self):
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        with mock.patch("urllib.request.urlopen", return_value=self.response(["google/gemma-2-2b-it"])):
            with self.assertRaises(RuntimeError) as caught:
                module._assert_serving_intended_model(self.process())
        self.assertIn("served model mismatch", str(caught.exception))

    def test_dead_server_process_raises(self):
        module = load_serve_modal(DEPLOY_ENVIRONMENT)
        with self.assertRaises(RuntimeError) as caught:
            module._assert_serving_intended_model(self.process(poll=1))
        self.assertIn("exited with code", str(caught.exception))

    def test_local_weights_guard_checks_the_served_name(self):
        module = load_serve_modal(LOCAL_ENVIRONMENT)
        with mock.patch("urllib.request.urlopen",
                        return_value=self.response(["google/gemma-2-9b-it+dpo-A"])):
            module._assert_serving_intended_model(self.process())
        # The base model answering under its own name is exactly the mislabelling to catch.
        with mock.patch("urllib.request.urlopen", return_value=self.response(["google/gemma-2-9b-it"])):
            with self.assertRaises(RuntimeError) as caught:
                module._assert_serving_intended_model(self.process())
        self.assertIn("served model mismatch", str(caught.exception))
