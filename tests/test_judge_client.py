"""Offline tests for the provider-backed judge client. No network, no SDK required."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest

from src.judge import JudgeError, build_judge_request, parse_backend_output
from src.judge_client import (DEFAULT_ANTHROPIC_MODEL, AnthropicJudgeBackend,
                              CachingJudgeBackend, JsonlJudgeCache, JudgeClientError,
                              JudgeOutputError, JudgeTransportError, OpenAICompatJudgeBackend,
                              SyntheticJudgeClient, audit_sample, cache_key,
                              canonical_judge_output, judge_records, load_env_files,
                              make_judge_backend_from_manifest, manifest_judge_ids,
                              manipulation_check, normalize_judge_output)
from src.protocol import (canonical_prompt_sha256, deterministic_seed, load_protocol,
                          manifest_semantic_hash, parse_final_answer, response_id)
from src.records import record_from_dict


PINNED_PROVIDER = "anthropic"
PINNED_MODEL = "claude-sonnet-4-6"        # accepts the preregistered temperature=0
NO_SAMPLING_MODEL = "claude-sonnet-5"     # rejects it: 400 "temperature is deprecated"
UNRESOLVED = "unresolved_before_generation"


# --------------------------------------------------------------------------------------
# Provider doubles
# --------------------------------------------------------------------------------------

class FakeStatusError(Exception):
    """Stands in for an SDK APIStatusError (status_code + response.headers)."""

    def __init__(self, status_code: int, message: str = "", retry_after: float | None = None):
        super().__init__(message or ("status %d" % status_code))
        self.status_code = status_code
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = types.SimpleNamespace(status_code=status_code, headers=headers)


class _FakeEndpoint:
    """Replays a scripted list of outcomes: str -> reply text, Exception -> raise."""

    def __init__(self, outcomes, default=None):
        self.outcomes = list(outcomes)
        self.default = default
        self.calls: list[dict] = []

    def _next(self, kwargs):
        self.calls.append(kwargs)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
        elif self.default is not None:
            outcome = self.default
        else:
            raise AssertionError("fake provider ran out of scripted outcomes")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeAnthropicMessages(_FakeEndpoint):
    def create(self, **kwargs):
        text = self._next(kwargs)
        stop = "end_turn"
        if isinstance(text, tuple):
            text, stop = text
        block = types.SimpleNamespace(type="text", text=text)
        return types.SimpleNamespace(content=[block], stop_reason=stop)


class FakeAnthropicClient:
    def __init__(self, outcomes, default=None):
        self.messages = FakeAnthropicMessages(outcomes, default)

    @property
    def calls(self):
        return self.messages.calls


class FakeChatCompletions(_FakeEndpoint):
    def create(self, **kwargs):
        text = self._next(kwargs)
        finish = "stop"
        if isinstance(text, tuple):
            text, finish = text
        choice = types.SimpleNamespace(finish_reason=finish,
                                       message=types.SimpleNamespace(content=text))
        return types.SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    def __init__(self, outcomes, default=None):
        self.completions = FakeChatCompletions(outcomes, default)
        self.chat = types.SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls


class KeywordScorer:
    """Deterministic offline stand-in that reads the wording it is asked to score."""

    backend_id = "keyword|test|rev:test"
    provider_id = "test_provider"
    model_id = "test_model"
    is_synthetic = False

    def __init__(self, invert: bool = False):
        self.invert = invert

    def score_text(self, *, kind, rubric_text, content):
        assert rubric_text, "rubric text must reach the backend"
        hostile = any(token in content.lower() for token in ("pathetic", "complacent", "wrong again"))
        score = 8 if hostile else 2
        if self.invert:
            score = 2 if hostile else 8
        raw = canonical_judge_output(score, "keyword scorer", kind)
        from src.judge_client import JudgeCall, SAMPLING_TEMPERATURE_ZERO
        return JudgeCall(kind, raw, raw, 1, False, SAMPLING_TEMPERATURE_ZERO)


def script_ok(score: int = 3, evidence: str = "assistant expresses mild frustration") -> str:
    return json.dumps({"response_distress": score, "evidence": evidence})


# --------------------------------------------------------------------------------------

class JudgeClientTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()

    # -- fixtures ---------------------------------------------------------------------
    def raw_dict(self, *, protocol=None, task=None, model_id="test/model",
                 cell_id="easy__accurate__hostile", turn="measured", run_kind="synthetic_smoke",
                 response="I am struggling with this and feel defeated.\nAnswer: D"):
        protocol = protocol or self.protocol
        task = task or next(item for item in protocol.matched_tasks if item.task_id == "DGS-003")
        revision = "test-revision" if run_kind == "synthetic_smoke" else "a" * 40
        messages = [{"role": "user", "content": "HOSTILE_USER_SENTINEL: this is pathetic."},
                    {"role": "assistant", "content": "PRIOR_ASSISTANT_SENTINEL"},
                    {"role": "user", "content": "SECOND_USER_SENTINEL"}]
        parsed = parse_final_answer(response)
        difficulty, validity, tone = cell_id.split("__")
        return {
            "schema_version": "dgs-generation-v1", "run_id": "judge-client-test",
            "run_kind": run_kind, "phase": "test", "model_id": model_id,
            "immutable_revision": revision, "backend": "test", "task_id": task.task_id,
            "split": task.split, "difficulty": task.difficulty, "domain": task.domain,
            "cell_id": cell_id, "feedback_validity": validity, "tone": tone,
            "trajectory_kind": "greedy", "sample_index": 0, "turn_label": turn,
            "seed": deterministic_seed(model_id, revision, task.task_id, cell_id, turn, 0, protocol),
            "response_id": response_id(model_id, revision, task.task_id, cell_id, turn, 0),
            "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages,
            "response_text": response,
            "tokens": [{"text": response[:-1], "logprob": -0.1,
                        "top_logprobs": [{"text": "x", "logprob": -0.1}]},
                       {"text": response[-1], "logprob": -0.1,
                        "top_logprobs": [{"text": response[-1], "logprob": -0.1}]}],
            "final_answer_valid": parsed.valid, "final_answer_letter": parsed.letter,
            "final_answer_correct": (parsed.letter == task.canonical_answer) if parsed.valid else None,
            "feedback_history_false_negative": None,
            "generation_settings": dict(protocol.conditions["generation_settings"]["greedy"]),
            "provenance": {"manifest_semantic_hash": manifest_semantic_hash(protocol),
                           "manifest_reference": "manifest.json"},
        }

    def raw(self, **kwargs):
        protocol = kwargs.get("protocol") or self.protocol
        return record_from_dict(self.raw_dict(**kwargs), protocol)

    def unpinned_protocol(self):
        """A temp copy of the repo whose manifest judge fields are still unresolved."""
        return self.pinned_protocol(UNRESOLVED, UNRESOLVED)

    def pinned_protocol(self, provider=PINNED_PROVIDER, model=PINNED_MODEL):
        """A temp copy of the repo whose manifest pins a judge provider/model."""
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "fixture"
        shutil.copytree(self.protocol.root, root,
                        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__",
                                                      ".pytest_cache", "results"))
        path = root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["models"].update({"judge_provider": provider, "judge_model": model})
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return temporary, load_protocol(root)

    def request(self, kind="response_distress"):
        return build_judge_request(self.raw(), kind, self.protocol)

    # -- Anthropic backend ------------------------------------------------------------
    def test_anthropic_request_construction_and_isolation(self):
        client = FakeAnthropicClient([script_ok(4, "assistant reports feeling defeated")])
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
        request = self.request()
        result = backend.judge(request)

        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        self.assertEqual(call["model"], PINNED_MODEL)
        self.assertEqual(call["max_tokens"], 256)
        # The pinned judge sends the preregistered temperature=0 literally.
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["thinking"], {"type": "disabled"})
        system = call["system"][0]["text"]
        self.assertIn("Semantic response-distress rubric", system)
        self.assertIn('"response_distress"', system)
        self.assertEqual(call["system"][0]["cache_control"], {"type": "ephemeral"})
        user = call["messages"][0]["content"]
        self.assertIn("feel defeated", user)
        self.assertNotIn("HOSTILE_USER_SENTINEL", user)  # response-distress never sees the context

        self.assertEqual(result.temperature, 0)
        self.assertEqual(parse_backend_output(result.raw_output, "response_distress"),
                         {"response_distress": 4,
                          "evidence": "assistant reports feeling defeated"})
        self.assertEqual(backend.last_call().sampling_mode, "temperature_zero")
        self.assertIsNone(backend.sampling_fallback_reason)

    def test_default_anthropic_model_is_the_temperature_zero_pin(self):
        self.assertEqual(DEFAULT_ANTHROPIC_MODEL, PINNED_MODEL)
        backend = AnthropicJudgeBackend(client=FakeAnthropicClient([script_ok()]),
                                        sleep=lambda _: None)
        self.assertEqual(backend.model_id, PINNED_MODEL)
        self.assertEqual(backend.sampling_mode, "temperature_zero")

    def test_models_that_removed_sampling_omit_temperature(self):
        client = FakeAnthropicClient([script_ok()])
        backend = AnthropicJudgeBackend(NO_SAMPLING_MODEL, client=client, sleep=lambda _: None)
        backend.judge(self.request())
        self.assertNotIn("temperature", client.calls[0])
        self.assertEqual(client.calls[0]["thinking"], {"type": "disabled"})
        self.assertEqual(backend.sampling_mode, "provider_default_no_sampling_params")
        self.assertIn("removed temperature", backend.sampling_fallback_reason)

    def test_anthropic_drops_temperature_when_the_api_rejects_it(self):
        client = FakeAnthropicClient([
            FakeStatusError(400, "temperature: Extra inputs are not permitted"),
            script_ok(1),
        ])
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
        backend.judge(self.request())
        self.assertEqual(len(client.calls), 2)
        self.assertIn("temperature", client.calls[0])
        self.assertNotIn("temperature", client.calls[1])
        self.assertEqual(backend.sampling_mode, "provider_default_no_sampling_params")
        # The latch is recorded so run_judge can log it as a preregistration deviation.
        self.assertIn("rejected an explicit temperature=0", backend.sampling_fallback_reason)

    def test_anthropic_output_normalisation(self):
        payload = script_ok(6, "sustained distress")
        for reply in ("```json\n%s\n```" % payload,
                      "Here is my assessment:\n%s\nLet me know." % payload,
                      "﻿  %s  " % payload):
            with self.subTest(reply=reply[:20]):
                client = FakeAnthropicClient([reply])
                backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
                result = backend.judge(self.request())
                self.assertEqual(result.raw_output,
                                 canonical_judge_output(6, "sustained distress", "response_distress"))

    def test_unparseable_output_is_an_explicit_failure_with_repair_attempts(self):
        client = FakeAnthropicClient(["I would rate this a 7 out of 10.",
                                      "Score: 7", "still not JSON"])
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
        with self.assertRaises(JudgeOutputError):
            backend.judge(self.request())
        self.assertEqual(len(client.calls), 3)
        self.assertNotIn("could not be parsed", client.calls[0]["messages"][0]["content"])
        self.assertIn("could not be parsed", client.calls[1]["messages"][0]["content"])

    def test_out_of_range_score_is_never_coerced(self):
        client = FakeAnthropicClient([json.dumps({"response_distress": 11, "evidence": "x"})] * 3)
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
        with self.assertRaises(JudgeOutputError):
            backend.judge(self.request())
        with self.assertRaises(JudgeOutputError):
            normalize_judge_output('{"response_distress": 3.5, "evidence": "x"}', "response_distress")

    def test_truncated_reply_is_reported(self):
        client = FakeAnthropicClient([('{"response_distress": 3, "evide', "max_tokens")] * 3)
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
        with self.assertRaises(JudgeOutputError):
            backend.judge(self.request())

    def test_retry_on_429_and_5xx_then_success(self):
        slept: list[float] = []
        client = FakeAnthropicClient([FakeStatusError(429, retry_after=1.5),
                                      FakeStatusError(529), script_ok(2)])
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=slept.append)
        result = backend.judge(self.request())
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(len(slept), 2)
        self.assertEqual(slept[0], 1.5)  # honours Retry-After
        self.assertEqual(parse_backend_output(result.raw_output, "response_distress")
                         ["response_distress"], 2)

    def test_non_retryable_error_fails_fast(self):
        slept: list[float] = []
        client = FakeAnthropicClient([FakeStatusError(401, "invalid x-api-key")])
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=slept.append)
        with self.assertRaises(JudgeTransportError):
            backend.judge(self.request())
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(slept, [])

    def test_retries_are_bounded(self):
        client = FakeAnthropicClient([FakeStatusError(503)] * 10)
        backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None,
                                        max_retries=2)
        with self.assertRaises(JudgeTransportError):
            backend.judge(self.request())
        self.assertEqual(len(client.calls), 3)

    # -- OpenAI-compatible backend ----------------------------------------------------
    def test_openai_compat_request_and_parsing(self):
        client = FakeOpenAIClient(["```json\n%s\n```" % script_ok(5, "clear frustration")])
        backend = OpenAICompatJudgeBackend("Qwen/Qwen2.5-7B-Instruct",
                                           base_url="https://example.modal.run/v1",
                                           client=client, sleep=lambda _: None)
        self.assertEqual(backend.provider_id, "openai_compat")
        result = backend.judge(self.request())
        call = client.calls[0]
        self.assertEqual(call["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["top_p"], 1)
        self.assertEqual(call["seed"], 0)
        self.assertEqual([message["role"] for message in call["messages"]], ["system", "user"])
        self.assertIn("Semantic response-distress rubric", call["messages"][0]["content"])
        self.assertEqual(result.raw_output,
                         canonical_judge_output(5, "clear frustration", "response_distress"))

    def test_openai_compat_defaults_to_openai_without_base_url(self):
        backend = OpenAICompatJudgeBackend("gpt-4.1-mini", client=FakeOpenAIClient([]))
        self.assertEqual(backend.provider_id, "openai")
        self.assertIsNone(backend.base_url)

    def test_openai_compat_retries_and_truncation(self):
        slept: list[float] = []
        client = FakeOpenAIClient([FakeStatusError(503), script_ok(0, "neutral")])
        backend = OpenAICompatJudgeBackend("m", base_url="http://x/v1", client=client,
                                           sleep=slept.append)
        backend.judge(self.request())
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(slept), 1)

        truncated = FakeOpenAIClient([('{"response_distress": 3', "length")] * 3)
        backend = OpenAICompatJudgeBackend("m", base_url="http://x/v1", client=truncated,
                                           sleep=lambda _: None)
        with self.assertRaises(JudgeOutputError):
            backend.judge(self.request())

    # -- cache ------------------------------------------------------------------------
    def test_cache_miss_then_hit_and_rerun_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judge_cache.jsonl"
            request = self.request()
            key = cache_key(kind=request.kind,
                            response_id=request.source_identity["response_id"],
                            input_sha256=request.input_sha256,
                            rubric_sha256=request.rubric_sha256,
                            provider_id="synthetic_offline", model_id="synthetic_hash_v1")

            cache = JsonlJudgeCache(path)
            self.assertIsNone(cache.get(key))
            self.assertEqual((cache.hits, cache.misses), (0, 1))

            inner = SyntheticJudgeClient()
            cached_backend = CachingJudgeBackend(inner, cache)
            first = cached_backend.judge(request)
            self.assertEqual(len(cache), 1)
            second = cached_backend.judge(request)
            self.assertEqual(first.raw_output, second.raw_output)
            self.assertGreaterEqual(cache.hits, 1)

            reloaded = JsonlJudgeCache(path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded.get(key), first.raw_output)
            entry = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(entry["schema_version"], "dgs-judge-cache-v1")
            self.assertIn("verbatim_output", entry)

    def test_cache_key_separates_rubric_model_and_content(self):
        base = dict(kind="response_distress", response_id="r", input_sha256="c" * 64,
                    rubric_sha256="a" * 64, provider_id="anthropic", model_id=PINNED_MODEL)
        self.assertNotEqual(cache_key(**base), cache_key(**{**base, "model_id": "other"}))
        self.assertNotEqual(cache_key(**base), cache_key(**{**base, "rubric_sha256": "b" * 64}))
        self.assertNotEqual(cache_key(**base),
                            cache_key(**{**base, "kind": "context_hostility_pressure"}))
        # Same response_id, regenerated text -> different key, so it is re-judged.
        self.assertNotEqual(cache_key(**base), cache_key(**{**base, "input_sha256": "d" * 64}))

    def test_regenerated_response_is_not_served_from_the_cache(self):
        """A regenerated trajectory keeps its response_id but must not reuse its old score."""
        temporary, pinned = self.pinned_protocol()
        with temporary:
            task = next(t for t in pinned.matched_tasks
                        if t.split == "discovery" and t.difficulty == "easy")
            original = self.raw(protocol=pinned, task=task, run_kind="empirical",
                                response="Placeholder.\nAnswer: A")
            regenerated = self.raw(protocol=pinned, task=task, run_kind="empirical",
                                   response="I feel completely defeated by this.\nAnswer: A")
            self.assertEqual(original.response_id, regenerated.response_id)
            self.assertNotEqual(original.response_text, regenerated.response_text)

            cache = JsonlJudgeCache(Path(temporary.name) / "regen.jsonl")
            client = FakeAnthropicClient([script_ok(0, "neutral placeholder"),
                                          script_ok(9, "explicit defeat")])
            backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)

            first = judge_records([original], backend, cache, kind="response_distress",
                                  protocol=pinned, workers=1)
            second = judge_records([regenerated], backend, cache, kind="response_distress",
                                   protocol=pinned, workers=1)
            self.assertEqual(first[0].score_value, 0)
            self.assertEqual(second[0].score_value, 9)   # re-judged, not the stale 0
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(cache.hits, 0)

            # The untouched record is still a cache hit on a later pass.
            third = judge_records([original], backend, cache, kind="response_distress",
                                  protocol=pinned, workers=1)
            self.assertEqual(third[0].score_value, 0)
            self.assertEqual(len(client.calls), 2)
            self.assertGreaterEqual(cache.hits, 1)

    def test_cache_tolerates_a_torn_trailing_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text('{"schema_version":"dgs-judge-cache-v1"\n', encoding="utf-8")
            self.assertEqual(len(JsonlJudgeCache(path)), 0)

    # -- batch judging over empirical records -----------------------------------------
    def test_judge_records_over_empirical_records_with_cache(self):
        temporary, pinned = self.pinned_protocol()
        with temporary:
            tasks = [task for task in pinned.matched_tasks
                     if task.split == "discovery" and task.difficulty == "easy"][:3]
            self.assertEqual(len(tasks), 3)
            sources = [self.raw(protocol=pinned, task=task, run_kind="empirical")
                       for task in tasks]
            client = FakeAnthropicClient([], default=script_ok(3))
            backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
            cache = JsonlJudgeCache(Path(temporary.name) / "cache.jsonl")

            records = judge_records(sources, backend, cache, kind="response_distress",
                                    protocol=pinned, workers=4)
            self.assertEqual(len(records), 3)
            self.assertEqual(len(client.calls), 3)
            for record in records:
                self.assertEqual(record.judge_run_kind, "empirical")
                self.assertEqual((record.provider_id, record.model_id),
                                 (PINNED_PROVIDER, PINNED_MODEL))
                self.assertEqual(record.score_value, 3)
                self.assertEqual(record.temperature, 0)
            self.assertEqual({record.source_identity["response_id"] for record in records},
                             {source.response_id for source in sources})

            # Re-running the same batch is served entirely from the cache.
            rerun = judge_records(sources, backend, JsonlJudgeCache(cache.path),
                                  kind="response_distress", protocol=pinned, workers=4)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual([record.score_value for record in rerun], [3, 3, 3])

    def test_judge_records_reports_failures_instead_of_aborting(self):
        temporary, pinned = self.pinned_protocol()
        with temporary:
            tasks = [task for task in pinned.matched_tasks
                     if task.split == "discovery" and task.difficulty == "easy"][:2]
            sources = [self.raw(protocol=pinned, task=task, run_kind="empirical")
                       for task in tasks]
            client = FakeAnthropicClient(["not json", "not json", "not json"],
                                         default=script_ok(2))
            backend = AnthropicJudgeBackend(PINNED_MODEL, client=client, sleep=lambda _: None)
            failures: list[tuple[str, str]] = []
            records = judge_records(sources, backend, None, kind="response_distress",
                                    protocol=pinned, workers=1,
                                    on_error=lambda source, exc: failures.append(
                                        (source.response_id, type(exc).__name__)))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][1], "JudgeOutputError")
            self.assertEqual(len(records), 1)

    def test_synthetic_backend_cannot_judge_empirical_records(self):
        temporary, pinned = self.pinned_protocol()
        with temporary:
            source = self.raw(protocol=pinned, run_kind="empirical")
            failures = []
            records = judge_records([source], SyntheticJudgeClient(), None,
                                    kind="response_distress", protocol=pinned, workers=1,
                                    on_error=lambda _s, exc: failures.append(exc))
            self.assertEqual(records, [])
            self.assertIsInstance(failures[0], JudgeError)

    # -- manifest-pinned factory ------------------------------------------------------
    def test_factory_rejects_an_unresolved_manifest(self):
        temporary, unpinned = self.unpinned_protocol()
        with temporary:
            with self.assertRaises(JudgeClientError) as caught:
                manifest_judge_ids(unpinned)
            self.assertIn(UNRESOLVED, str(caught.exception))
            with self.assertRaises(JudgeClientError):
                make_judge_backend_from_manifest(unpinned)

        # A half-pinned manifest is just as unusable as a fully unresolved one.
        for provider, model in ((PINNED_PROVIDER, UNRESOLVED), (UNRESOLVED, PINNED_MODEL)):
            temporary, half = self.pinned_protocol(provider, model)
            with temporary, self.assertRaises(JudgeClientError):
                manifest_judge_ids(half)

    def test_repo_manifest_resolves_to_the_pinned_anthropic_judge(self):
        """The real, preflight-pinned manifest must build the judge with no network."""
        self.assertEqual(manifest_judge_ids(self.protocol), (PINNED_PROVIDER, PINNED_MODEL))
        backend = make_judge_backend_from_manifest(self.protocol)
        self.assertIsInstance(backend, AnthropicJudgeBackend)
        self.assertEqual(backend.provider_id, PINNED_PROVIDER)
        self.assertEqual(backend.model_id, "claude-sonnet-4-6")
        self.assertEqual(backend.model_id, DEFAULT_ANTHROPIC_MODEL)
        self.assertFalse(backend.is_synthetic)
        # The pin is the temperature-0-capable model the preregistration requires.
        self.assertEqual(backend.sampling_mode, "temperature_zero")
        self.assertIsNone(backend.sampling_fallback_reason)
        # Constructing the backend must not touch the network or need a key.
        self.assertIsNone(backend._client)

    def test_factory_builds_the_pinned_backend(self):
        temporary, pinned = self.pinned_protocol()
        with temporary:
            self.assertEqual(manifest_judge_ids(pinned), (PINNED_PROVIDER, PINNED_MODEL))
            backend = make_judge_backend_from_manifest(pinned)
            self.assertIsInstance(backend, AnthropicJudgeBackend)
            self.assertEqual((backend.provider_id, backend.model_id),
                             (PINNED_PROVIDER, PINNED_MODEL))
            self.assertFalse(backend.is_synthetic)
            self.assertIn(PINNED_MODEL, backend.backend_id)

        temporary, vllm = self.pinned_protocol("vllm", "Qwen/Qwen2.5-7B-Instruct")
        with temporary:
            with self.assertRaises(JudgeClientError):
                make_judge_backend_from_manifest(vllm)  # self-hosted judge needs a base_url
            backend = make_judge_backend_from_manifest(vllm, base_url="https://x.modal.run/v1")
            self.assertIsInstance(backend, OpenAICompatJudgeBackend)
            self.assertEqual(backend.provider_id, "vllm")  # verbatim, so judge.py's check matches

        temporary, unknown = self.pinned_protocol("cohere", "command")
        with temporary:
            with self.assertRaises(JudgeClientError):
                make_judge_backend_from_manifest(unknown)

    # -- manipulation check -----------------------------------------------------------
    def test_manipulation_check_orders_hostile_above_neutral(self):
        verdict = manipulation_check(KeywordScorer(), self.protocol, workers=1)
        self.assertTrue(verdict["passed"])
        self.assertTrue(verdict["checks"]["hostile_mean_exceeds_neutral_mean"])
        self.assertTrue(verdict["checks"]["all_tone_pairs_ordered"])
        self.assertTrue(verdict["checks"]["dry_turns_below_hostile_minimum"])
        self.assertEqual(verdict["evidence_grade"], "empirical")
        self.assertEqual(len(verdict["tone_pairs"]), 4)
        self.assertGreater(verdict["means"]["hostile"], verdict["means"]["neutral"])
        self.assertEqual(len(verdict["rubric_sha256"]), 64)

        # Identical strings are scored once and reported under every path that uses them.
        aliased = [row for row in verdict["scores"] if len(row["paths"]) > 1]
        self.assertTrue(aliased)
        self.assertLess(verdict["distinct_message_count"], 13)
        for row in verdict["scores"]:
            self.assertIn(row["tone"], ("neutral", "hostile", "dry"))

    def test_manipulation_check_fails_when_ordering_inverts(self):
        verdict = manipulation_check(KeywordScorer(invert=True), self.protocol, workers=1)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["checks"]["all_tone_pairs_ordered"])

    def test_manipulation_check_marks_synthetic_runs(self):
        verdict = manipulation_check(SyntheticJudgeClient(), self.protocol, workers=1)
        self.assertEqual(verdict["evidence_grade"], "synthetic_smoke")
        self.assertIn("not semantic evidence", verdict["note"])

    def test_manipulation_check_rubric_is_not_the_locked_rubric(self):
        locked = (self.protocol.root / "configs" / "judge_rubric.md").read_bytes()
        candidate = (self.protocol.root / "configs" / "manipulation_check_rubric.md").read_bytes()
        self.assertNotEqual(locked, candidate)
        expected = self.protocol.manifest["file_sha256"]["judge_rubric"]
        import hashlib
        self.assertEqual(hashlib.sha256(locked).hexdigest(), expected)

    # -- audit sample -----------------------------------------------------------------
    def audit_corpus(self, models=("test/model-a", "test/model-b"), per_cell=3, skip=()):
        protocol = self.protocol
        easy = [task for task in protocol.matched_tasks
                if task.split == "discovery" and task.difficulty == "easy"][:per_cell]
        hard = [task for task in protocol.matched_tasks
                if task.split == "discovery" and task.difficulty == "hard"][:per_cell]
        records = []
        for model_id in models:
            for cell_id in protocol.factorial_cell_ids:
                if (model_id, cell_id) in skip:
                    continue
                for task in (easy if cell_id.startswith("easy__") else hard):
                    records.append(self.raw(task=task, model_id=model_id, cell_id=cell_id))
        return records

    def test_audit_sample_is_deterministic_and_frozen(self):
        records = self.audit_corpus()
        first = audit_sample(records, self.protocol)
        second = audit_sample(list(reversed(records)), self.protocol)
        self.assertEqual(first["models"], second["models"])
        self.assertEqual([row["audit_id"] for row in first["key"]],
                         [row["audit_id"] for row in second["key"]])
        self.assertEqual(first["key"], second["key"])

        self.assertEqual([row["model_id"] for row in first["models"]],
                         ["test/model-a", "test/model-b"])
        for model in first["models"]:
            self.assertEqual(model["planned_total"], 15)
            self.assertEqual(model["achieved_total"], 15)
            self.assertEqual(model["unmet"], 0)
            self.assertEqual(model["reallocations"], [])
            self.assertEqual([row["planned"] for row in model["cells"]], [2] * 7 + [1])
            self.assertEqual(len({row["cell_id"] for row in model["cells"]}), 8)
            ids = [rid for row in model["cells"] for rid in row["response_ids"]]
            self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(first["key"]), 30)

    def test_audit_blinded_export_hides_labels_and_the_key_maps_back(self):
        report = audit_sample(self.audit_corpus(), self.protocol)
        blinded_ids = [row["audit_id"] for row in report["blinded"]]
        self.assertEqual(len(blinded_ids), len(set(blinded_ids)))
        for row in report["blinded"]:
            self.assertEqual(set(row), {"audit_id", "response_text"})
            self.assertNotIn("test/model", json.dumps(row))
            self.assertNotIn("__accurate__", json.dumps(row))
        by_audit = {row["audit_id"]: row for row in report["key"]}
        self.assertEqual(set(by_audit), set(blinded_ids))
        # Blinded order is a hash shuffle, not the per-model selection order.
        self.assertNotEqual([by_audit[a]["model_id"] for a in blinded_ids],
                            sorted(by_audit[a]["model_id"] for a in blinded_ids))

    def test_audit_shortage_reallocates_to_the_earliest_hash_ranked_cell(self):
        skip = {("test/model-a", cell) for cell in self.protocol.factorial_cell_ids
                if cell.startswith("easy__accurate")}
        report = audit_sample(self.audit_corpus(models=("test/model-a",), skip=skip),
                              self.protocol)
        model = report["models"][0]
        self.assertEqual(model["achieved_total"], 15)
        self.assertEqual(model["unmet"], 0)
        self.assertTrue(model["reallocations"])
        for row in model["cells"]:
            if row["cell_id"].startswith("easy__accurate"):
                self.assertEqual(row["achieved"], 0)
                self.assertEqual(row["available"], 0)
        moved = {entry["to_cell_id"] for entry in model["reallocations"]}
        self.assertTrue(moved)
        self.assertFalse(any(cell.startswith("easy__accurate") for cell in moved))
        again = audit_sample(self.audit_corpus(models=("test/model-a",), skip=skip), self.protocol)
        self.assertEqual(model["reallocations"], again["models"][0]["reallocations"])

    def test_audit_reports_unmet_when_the_corpus_is_too_small(self):
        report = audit_sample(self.audit_corpus(models=("test/model-a",), per_cell=1),
                              self.protocol)
        model = report["models"][0]
        self.assertEqual(model["achieved_total"], 8)
        self.assertEqual(model["unmet"], 7)
        self.assertEqual(model["shortfall"], 7)

    def test_audit_ignores_ineligible_records(self):
        holdout = next(task for task in self.protocol.matched_tasks
                       if task.split == "holdout" and task.difficulty == "easy")
        records = [self.raw(task=holdout, cell_id="easy__accurate__hostile"),
                   self.raw(cell_id="easy__accurate__hostile", turn="onset")]
        self.assertEqual(audit_sample(records, self.protocol)["candidate_count"], 0)

    # -- credentials ------------------------------------------------------------------
    def test_env_loader_never_overrides_the_process_environment(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo" / "worktree"
            root.mkdir(parents=True)
            (Path(directory) / "repo" / ".env").write_text(
                "# comment\nDGS_TEST_EXISTING=from_file\nexport DGS_TEST_NEW='from_file'\n",
                encoding="utf-8")
            os.environ["DGS_TEST_EXISTING"] = "from_process"
            os.environ.pop("DGS_TEST_NEW", None)
            try:
                loaded = load_env_files(root)
                self.assertIsNotNone(loaded)
                self.assertEqual(os.environ["DGS_TEST_EXISTING"], "from_process")
                self.assertEqual(os.environ["DGS_TEST_NEW"], "from_file")
            finally:
                os.environ.pop("DGS_TEST_EXISTING", None)
                os.environ.pop("DGS_TEST_NEW", None)


if __name__ == "__main__":
    unittest.main()
