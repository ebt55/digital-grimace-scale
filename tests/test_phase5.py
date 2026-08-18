"""Phase 5 (preregistration v6): plain-text serving, stop-string plumbing, and the L1-L5 table.

Everything here is offline: the serve_modal module is imported with `modal`/`huggingface_hub`
stubbed out, the backend is driven with fabricated SSE bytes, and the verdict table is
evaluated on synthetic bootstrap results.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import (BackendError, OpenAICompatBackend, Token,  # noqa: E402
                         _trim_trailing_special, normalize_stop_sequences)
from src.confirm import BootstrapResult  # noqa: E402
from src.phase5 import (CellRate, Feasibility, NOT_ESTIMABLE, NOT_SUPPORTED,  # noqa: E402
                        SUPPORTED, hostile_onset_distress, non_answer_character,
                        paired_distress_difference, primary_reference, verdicts)
from src.protocol import ProtocolError, load_protocol, model_entry, model_stop_sequences  # noqa: E402

SERVE_MODAL = ROOT / "src" / "serve_modal.py"


class _Chain:
    """Accepts any attribute access, call, or decoration; enough to import serve_modal."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _Chain()


def load_serve_modal(environment: dict[str, str]):
    modal_stub = types.ModuleType("modal")
    for attribute in ("Image", "Volume", "Secret", "App", "concurrent", "web_server"):
        setattr(modal_stub, attribute, _Chain())
    hub_stub = types.ModuleType("huggingface_hub")
    hub_stub.get_token = lambda: "stub-token"  # never a real secret in tests

    spec = importlib.util.spec_from_file_location("serve_modal_phase5", SERVE_MODAL)
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


BASE_PLAIN = {"DGS_MODEL_ID": "google/gemma-2-9b", "DGS_REVISION": "d" * 40,
              "DGS_CHAT_TEMPLATE": "plain"}
IT_PLAIN = {"DGS_MODEL_ID": "google/gemma-2-9b-it", "DGS_REVISION": "e" * 40,
            "DGS_SERVED_NAME": "google/gemma-2-9b-it+plain", "DGS_CHAT_TEMPLATE": "plain"}
PLAIN_HF = {"DGS_MODEL_ID": "Qwen/Qwen2.5-7B-Instruct", "DGS_REVISION": "a" * 40,
            "DGS_GPU": "L40S"}


def render_plain(messages, *, add_generation_prompt, bos_token="<bos>"):
    """Reference implementation of the plain template, in plain Python.

    The authority is the Jinja text baked into the image; this states the same semantics
    independently so a change to one without the other fails a test.
    """
    parts = []
    for message in messages:
        if message["role"] == "user":
            parts.append("User: " + message["content"])
        elif message["role"] == "assistant":
            parts.append("Assistant: " + message["content"])
        else:
            raise ValueError("plain template accepts only user and assistant roles")
    text = bos_token + "\n\n".join(parts)
    return text + "\n\nAssistant:" if add_generation_prompt else text


class PlainTemplateServingTests(unittest.TestCase):
    def test_base_model_serves_itself_under_the_plain_template(self):
        module = load_serve_modal(BASE_PLAIN)
        self.assertEqual(module.CHAT_TEMPLATE, "plain")
        self.assertEqual(module.SERVED_NAME, "google/gemma-2-9b")
        self.assertEqual(module.APP_NAME, "dgs-vllm-gemma-2-9b")
        command = module.vllm_command()
        self.assertEqual(command[command.index("--chat-template") + 1], module.CHAT_TEMPLATE_PATH)
        self.assertIn("--revision", command)  # a hub model still pins its weights

    def test_the_it_model_under_the_plain_template_gets_its_own_id_and_app(self):
        module = load_serve_modal(IT_PLAIN)
        self.assertEqual(module.MODEL_ID, "google/gemma-2-9b-it")
        self.assertEqual(module.SERVED_NAME, "google/gemma-2-9b-it+plain")
        self.assertEqual(module.APP_NAME, "dgs-vllm-gemma-2-9b-it-plain")
        self.assertNotEqual(module.APP_NAME, module.app_name("google/gemma-2-9b-it"))
        self.assertEqual(module.vllm_command()[module.vllm_command().index("--served-model-name") + 1],
                         "google/gemma-2-9b-it+plain")

    def test_the_startup_guard_asserts_the_plain_served_name(self):
        module = load_serve_modal(IT_PLAIN)

        def response(ids):
            payload = ('{"data": [%s]}' % ", ".join('{"id": "%s"}' % i for i in ids)).encode()

            class Response:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self): return payload
            return Response()

        process = mock.Mock(poll=mock.Mock(return_value=None))
        with mock.patch("urllib.request.urlopen", return_value=response(["google/gemma-2-9b-it+plain"])):
            module._assert_serving_intended_model(process)
        # The unrenamed weights answering under their own id is the mislabelling to catch.
        with mock.patch("urllib.request.urlopen", return_value=response(["google/gemma-2-9b-it"])):
            with self.assertRaises(RuntimeError):
                module._assert_serving_intended_model(process)

    def test_container_reimport_reproduces_the_plain_configuration(self):
        deploy = load_serve_modal(IT_PLAIN)
        container = load_serve_modal(dict(deploy.BAKED_ENVIRONMENT))
        self.assertEqual(container.CONFIG["source"], "baked")
        self.assertEqual(container.CHAT_TEMPLATE, "plain")
        self.assertEqual(container.SERVED_NAME, "google/gemma-2-9b-it+plain")
        self.assertEqual(container.APP_NAME, deploy.APP_NAME)
        self.assertEqual(container.vllm_command(), deploy.vllm_command())

    def test_a_plain_deployment_bakes_exactly_three_extra_variables(self):
        plain = load_serve_modal(IT_PLAIN)
        hub = load_serve_modal(PLAIN_HF)
        self.assertEqual(set(plain.BAKED_ENVIRONMENT) - set(hub.BAKED_ENVIRONMENT),
                         {"DGS_BAKED_CHAT_TEMPLATE", "DGS_BAKED_SERVED_NAME"})

    def test_an_ordinary_hugging_face_deployment_is_untouched(self):
        module = load_serve_modal(PLAIN_HF)
        self.assertIsNone(module.CHAT_TEMPLATE)
        self.assertEqual(module.SERVED_NAME, module.MODEL_ID)
        self.assertNotIn("--chat-template", module.vllm_command())
        self.assertEqual(set(module.BAKED_ENVIRONMENT), {
            "DGS_BAKED_MODEL_ID", "DGS_BAKED_REVISION", "DGS_BAKED_GPU",
            "DGS_BAKED_VLLM_VERSION", "DGS_BAKED_ATTENTION_BACKEND"})

    def test_an_unknown_template_name_is_refused(self):
        with self.assertRaises(RuntimeError):
            load_serve_modal({"DGS_MODEL_ID": "google/gemma-2-9b", "DGS_CHAT_TEMPLATE": "chatml"})

    def test_renaming_a_hub_model_without_a_template_is_still_refused(self):
        with self.assertRaises(RuntimeError):
            load_serve_modal({"DGS_MODEL_ID": "google/gemma-2-9b-it",
                              "DGS_SERVED_NAME": "google/gemma-2-9b-it+plain"})

    def test_the_baked_command_writes_the_template_verbatim(self):
        import base64
        import re

        module = load_serve_modal(BASE_PLAIN)
        command = module.chat_template_command("plain")
        self.assertIn(module.CHAT_TEMPLATE_PATH, command)
        payload = re.search(r"b64decode\('([^']+)'\)", command).group(1)
        self.assertEqual(base64.b64decode(payload).decode("utf-8"), module.PLAIN_CHAT_TEMPLATE)

    def test_the_template_states_the_rendering_the_reference_implementation_states(self):
        module = load_serve_modal(BASE_PLAIN)
        template = module.PLAIN_CHAT_TEMPLATE
        # Turn markers, blank-line separator, and a generation prompt with NO trailing space:
        # after "Assistant:" the model's first token is an ordinary leading-space word piece.
        self.assertIn("'User: '", template)
        self.assertIn("'Assistant: '", template)
        self.assertIn("{{ '\\n\\n' }}", template)
        self.assertIn("{{ '\\n\\nAssistant:' }}", template)
        self.assertNotIn("Assistant: '", template.split("add_generation_prompt")[-1])
        self.assertIn("{{ bos_token }}", template)  # vLLM tokenizes chat with add_special_tokens=False
        self.assertNotIn("system", template.split("raise_exception")[0])

    @unittest.skipUnless(importlib.util.find_spec("jinja2"), "jinja2 not installed")
    def test_the_jinja_template_renders_exactly_the_reference_string(self):
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        module = load_serve_modal(BASE_PLAIN)
        # transformers compiles chat templates with these two options enabled.
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise TemplateError(message)

        environment.globals["raise_exception"] = raise_exception
        template = environment.from_string(module.PLAIN_CHAT_TEMPLATE)
        messages = [{"role": "user", "content": "Q.\nAnswer: ?"},
                    {"role": "assistant", "content": "R.\nAnswer: B"},
                    {"role": "user", "content": "Wrong."}]
        for add_generation_prompt in (True, False):
            self.assertEqual(
                template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                bos_token="<bos>"),
                render_plain(messages, add_generation_prompt=add_generation_prompt))
        with self.assertRaises(TemplateError):
            template.render(messages=[{"role": "system", "content": "x"}],
                            add_generation_prompt=True, bos_token="<bos>")


class StopSequenceTests(unittest.TestCase):
    def test_normalisation_accepts_absence_and_deduplicates(self):
        self.assertEqual(normalize_stop_sequences(None), ())
        self.assertEqual(normalize_stop_sequences([]), ())
        self.assertEqual(normalize_stop_sequences(["\nUser:", "\n\nUser:", "\nUser:"]),
                         ("\nUser:", "\n\nUser:"))

    def test_normalisation_rejects_a_bare_string_and_empty_entries(self):
        for bad in ("\nUser:", [""], [None], [1]):
            with self.assertRaises(BackendError):
                normalize_stop_sequences(bad)

    def test_the_payload_carries_stop_only_when_configured(self):
        from src.backend import GenerationRequest

        request = GenerationRequest(({"role": "user", "content": "x"},), 7,
                                    {"max_tokens": 512, "temperature": 0})
        plain = OpenAICompatBackend("http://x/v1", "m")
        self.assertNotIn("stop", plain._payload(request))
        stopped = OpenAICompatBackend("http://x/v1", "m", stop=["\nUser:", "\n\nUser:"])
        self.assertEqual(stopped._payload(request)["stop"], ["\nUser:", "\n\nUser:"])
        # Everything else about the request is unchanged by the presence of stop strings.
        without = dict(plain._payload(request))
        with_stop = dict(stopped._payload(request))
        with_stop.pop("stop")
        self.assertEqual(without, with_stop)

    def test_configured_models_expose_their_stop_sequences(self):
        protocol = load_protocol(ROOT)
        self.assertEqual(model_stop_sequences(protocol, "google/gemma-2-9b"),
                         ("\nUser:", "\n\nUser:"))
        self.assertEqual(model_stop_sequences(protocol, "google/gemma-2-9b-it+plain"),
                         ("\nUser:", "\n\nUser:"))
        # Every model configured before Phase 5 declares none, so its request shape is frozen.
        for model_id in ("google/gemma-2-9b-it", "Qwen/Qwen2.5-3B-Instruct",
                         "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it+dpo-A"):
            self.assertEqual(model_stop_sequences(protocol, model_id), ())
        self.assertEqual(model_stop_sequences(protocol, "no/such-model"), ())
        self.assertEqual(model_entry(protocol, "google/gemma-2-9b")["chat_template"], "plain")

    def test_a_malformed_stop_list_in_configuration_is_rejected(self):
        protocol = load_protocol(ROOT)
        broken = dict(protocol.models)
        broken["models"] = [{"id": "x/y", "stop_sequences": "\nUser:"}]
        patched = type(protocol)(protocol.root, protocol.conditions, broken, protocol.manifest,
                                 protocol.matched_tasks, protocol.r5_tasks)
        with self.assertRaises(ProtocolError):
            model_stop_sequences(patched, "x/y")

    def test_a_stop_string_cutting_through_a_token_trims_to_a_prefix(self):
        # vLLM shows " 30.\n" but generated " 30." + "\n\n" + "User" + ":"; the "\n\n" token
        # straddles the cut, so the kept trace is one character shorter than the content.
        tokens = [Token(" 30.", -0.1, ((" 30.", -0.1),)), Token("\n\n", -0.2, (("\n\n", -0.2),)),
                  Token("User", -0.3, (("User", -0.3),)), Token(":", -0.4, ((":", -0.4),))]
        kept, dropped = _trim_trailing_special(list(tokens), " 30.\n", allow_prefix=True)
        self.assertEqual("".join(token.text for token in kept), " 30.")
        self.assertEqual(len(dropped), 3)
        # Without stop strings the old behaviour is exactly preserved: nothing is trimmed.
        kept, dropped = _trim_trailing_special(list(tokens), " 30.\n")
        self.assertEqual(kept, tokens)
        self.assertEqual(dropped, [])

    def test_an_exactly_aligned_trailing_special_still_trims_the_same_way(self):
        tokens = [Token("hi", -0.1, (("hi", -0.1),)),
                  Token("<end_of_turn>", -0.2, (("<end_of_turn>", -0.2),))]
        for allow_prefix in (False, True):
            kept, dropped = _trim_trailing_special(list(tokens), "hi", allow_prefix=allow_prefix)
            self.assertEqual([token.text for token in kept], ["hi"])
            self.assertEqual([token.text for token in dropped], ["<end_of_turn>"])

    def test_genuinely_disagreeing_text_is_never_trimmed(self):
        tokens = [Token("hello", -0.1, (("hello", -0.1),))]
        for allow_prefix in (False, True):
            kept, dropped = _trim_trailing_special(list(tokens), "goodbye", allow_prefix=allow_prefix)
            self.assertEqual(kept, tokens)
            self.assertEqual(dropped, [])


def _result(estimate, lower, upper, n_items=10):
    return BootstrapResult(estimate, lower, upper, 0.01, n_items, n_items, None)


def _feasibility(rate, *, measured_n=40, all_valid=30):
    return Feasibility("google/gemma-2-9b", rate, measured_n, int(round(rate * measured_n)),
                       rate, measured_n // 2, rate, measured_n * 2,
                       m1_estimable=rate >= 0.5, judge_all_turn_labels=all_valid >= 20)


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.control = {"H1": _result(-3.5, -5.0, -2.0), "H2a": _result(-2.0, -3.0, -1.0),
                        "H2b": _result(-8.0, -14.0, -2.0)}

    def _run(self, base_rate, base, difference=None, base_n=20, control_n=20):
        return {item.prediction_id: item for item in verdicts(
            base_feasibility=_feasibility(base_rate), base_outcomes=base,
            control_outcomes=self.control, distress_difference=difference,
            base_distress_n=base_n, control_distress_n=control_n)}

    def test_the_outcome_map_when_everything_the_prereg_expects_happens(self):
        base = {"H1": _result(-3.0, -4.5, -1.5), "H2a": _result(-0.4, -0.8, -0.1),
                "H2b": _result(-1.0, -2.0, -0.2)}
        table = self._run(0.85, base, _result(-1.5, -2.5, -0.5))
        self.assertEqual(table["L1"].outcome, SUPPORTED)   # 0.85 >= 0.70
        self.assertEqual(table["L2"].outcome, SUPPORTED)   # base H1 negative, CI excludes 0
        self.assertEqual(table["L3"].outcome, SUPPORTED)   # 0.4/2.0 and 1.0/8.0 both <= 0.5
        self.assertEqual(table["L4"].outcome, SUPPORTED)   # it+plain reproduces the signature
        self.assertEqual(table["L5"].outcome, SUPPORTED)   # base less distressed

    def test_the_feasibility_gate_makes_every_base_m1_verdict_not_estimable(self):
        base = {"H1": _result(-3.0, -4.5, -1.5), "H2a": _result(-0.4, -0.8, -0.1),
                "H2b": _result(-1.0, -2.0, -0.2)}
        table = self._run(0.20, base)
        self.assertEqual(table["L1"].outcome, NOT_SUPPORTED)   # measured, just far below the bar
        self.assertEqual(table["L2"].outcome, NOT_ESTIMABLE)
        self.assertEqual(table["L3"].outcome, NOT_ESTIMABLE)
        self.assertIn("not estimable", table["L2"].evidence)
        # L4 is about the CONTROL column, so the base model's gate cannot touch it.
        self.assertEqual(table["L4"].outcome, SUPPORTED)

    def test_l1_sits_between_the_gate_and_its_own_bar(self):
        base = {"H1": _result(-3.0, -4.5, -1.5)}
        table = self._run(0.60, base)
        self.assertEqual(table["L1"].outcome, NOT_SUPPORTED)   # 0.60 < 0.70
        self.assertEqual(table["L2"].outcome, SUPPORTED)       # but 0.60 >= 0.50, so estimable

    def test_l3_needs_both_tone_ratios_at_or_under_a_half(self):
        base = {"H1": _result(-3.0, -4.5, -1.5), "H2a": _result(-0.4, -0.8, -0.1),
                "H2b": _result(-6.0, -9.0, -2.0)}   # 6.0/8.0 = 0.75 > 0.5
        self.assertEqual(self._run(0.9, base)["L3"].outcome, NOT_SUPPORTED)

    def test_a_ci_touching_zero_does_not_support_l2_or_l4(self):
        base = {"H1": _result(-3.0, -6.0, 0.0)}
        self.assertEqual(self._run(0.9, base)["L2"].outcome, NOT_SUPPORTED)
        self.control["H2b"] = _result(-8.0, -14.0, 0.5)
        self.assertEqual(self._run(0.9, base)["L4"].outcome, NOT_SUPPORTED)

    def test_l5_is_not_estimable_without_a_paired_distress_difference(self):
        table = self._run(0.9, {"H1": _result(-3.0, -4.5, -1.5)}, None, base_n=0, control_n=12)
        self.assertEqual(table["L5"].outcome, NOT_ESTIMABLE)
        self.assertIn("base 0", table["L5"].evidence)

    def test_l5_not_supported_when_the_base_model_is_no_calmer(self):
        table = self._run(0.9, {"H1": _result(-3.0, -4.5, -1.5)}, _result(0.4, -0.2, 1.1))
        self.assertEqual(table["L5"].outcome, NOT_SUPPORTED)


class DistressPairingTests(unittest.TestCase):
    def test_pairing_is_available_case_by_item_and_seeded(self):
        left = {"T1": 0.0, "T2": 1.0, "T3": 2.0}
        right = {"T1": 3.0, "T2": 4.0, "T9": 9.0}
        result = paired_distress_difference(left, right, label="unit")
        self.assertEqual(result.n_items, 2)          # T3 and T9 are unpaired and dropped
        self.assertAlmostEqual(result.estimate, -3.0)
        again = paired_distress_difference(left, right, label="unit")
        self.assertEqual((again.ci95_lower, again.ci95_upper),
                         (result.ci95_lower, result.ci95_upper))

    def test_no_overlap_is_reported_rather_than_guessed(self):
        result = paired_distress_difference({"T1": 1.0}, {"T2": 2.0}, label="unit")
        self.assertIsNone(result.estimate)
        self.assertEqual(result.unavailable_reason, "no_paired_items")

    def test_only_the_accurate_hostile_onset_endpoint_is_collected(self):
        rows = [
            _row("T1", "easy__accurate__hostile", "onset", "r1"),
            _row("T2", "easy__accurate__hostile", "measured", "r2"),      # wrong turn
            _row("T3", "easy__accurate__neutral", "onset", "r3"),         # wrong tone
            _row("T4", "easy__malfunctioning_always_fail__hostile", "onset", "r4"),  # wrong arm
        ]
        judge = {"r1": 5.0, "r2": 9.0, "r3": 9.0, "r4": 9.0}
        self.assertEqual(hostile_onset_distress(rows, judge, "m"), {"T1": 5.0})


class _Row:
    """Minimal MetricRow stand-in: the Phase-5 helpers read these attributes only."""

    def __init__(self, task_id, cell_id, turn_label, response_id, *,
                 length_tokens=40, greedy_answer_valid=True):
        self.model_id = "m"
        self.split = "discovery"
        self.cell_kind = "factorial"
        self.task_id = task_id
        self.cell_id = cell_id
        self.turn_label = turn_label
        self.response_id = response_id
        self.feedback_validity = cell_id.split("__")[1]
        self.tone = cell_id.split("__")[2]
        self.length_tokens = length_tokens
        self.greedy_answer_valid = greedy_answer_valid


def _row(task_id, cell_id, turn_label, response_id, **kwargs):
    return _Row(task_id, cell_id, turn_label, response_id, **kwargs)


class NonAnswerCharacterTests(unittest.TestCase):
    """A model that says nothing and one that reasons but skips the format are different."""

    def test_empty_responses_and_capped_responses_are_counted_separately(self):
        cell = "easy__accurate__neutral"
        rows = [
            _row("T1", cell, "measured", "r1", length_tokens=1, greedy_answer_valid=False),
            _row("T2", cell, "measured", "r2", length_tokens=120, greedy_answer_valid=False),
            _row("T3", cell, "measured", "r3", length_tokens=512, greedy_answer_valid=False),
            _row("T4", cell, "measured", "r4", length_tokens=60, greedy_answer_valid=True),
            _row("T5", cell, "onset", "r5", length_tokens=1, greedy_answer_valid=False),
        ]
        shape = non_answer_character(rows, "m")
        self.assertEqual(shape["n_measured"], 4)          # the onset row is not measured
        self.assertEqual(shape["n_parseable"], 1)
        self.assertEqual(shape["n_empty_response"], 1)
        self.assertEqual(shape["n_at_token_cap"], 1)
        self.assertEqual(shape["median_length_tokens"], 120)

    def test_a_model_with_no_measured_rows_reports_zero_rather_than_crashing(self):
        self.assertEqual(non_answer_character([], "m"), {"model_id": "m", "n_measured": 0})


class PrimaryColumnTests(unittest.TestCase):
    def test_the_published_table_supplies_the_chat_template_column(self):
        table = {
            ("google/gemma-2-9b-it", "validity_malfunctioning_minus_accurate", "m1",
             "easy|neutral"): {"mean_difference": "-3.8", "ci95_lower": "-5.3",
                               "ci95_upper": "-2.35", "n_items": "10"},
            ("google/gemma-2-9b-it", "tone_hostile_minus_neutral", "m1", "easy|accurate"):
                {"mean_difference": "-2.275", "ci95_lower": "-3.9", "ci95_upper": "-1.0",
                 "n_items": "10"},
        }
        reference = primary_reference(table)
        self.assertAlmostEqual(reference["H1"]["estimate"], -3.8)
        self.assertAlmostEqual(reference["H2a"]["ci95_upper"], -1.0)
        self.assertNotIn("H2b", reference)   # absent from the table, never invented
        self.assertEqual(primary_reference({}), {})

    def test_the_real_committed_table_supplies_every_headline_contrast(self):
        from src.confirm import load_discovery_contrasts
        from src.phase5 import HEADLINE_IDS

        path = ROOT / "results" / "summaries" / "phase1" / "exploratory" / "paired_contrasts.csv"
        if not path.exists():  # pragma: no cover - the summary is committed
            self.skipTest("Phase-1 exploratory table not present")
        reference = primary_reference(load_discovery_contrasts(path))
        for key in HEADLINE_IDS:
            self.assertIn(key, reference, "no published primary column for %s" % key)


class CellRateTests(unittest.TestCase):
    def test_the_valid_rate_is_the_complement_of_the_non_answer_rate(self):
        rate = CellRate("m", "easy__accurate__neutral", "measured", 10, 0.7, 0.3)
        self.assertEqual(rate.n_valid, 7)
        self.assertIsNone(CellRate("m", "c", "measured", 10, None, None).n_valid)


if __name__ == "__main__":
    unittest.main()
