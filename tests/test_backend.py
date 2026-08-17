from __future__ import annotations

import json
import math
import threading
import unittest
from unittest import mock

from src.backend import (BackendError, GenerationRequest, OpenAICompatBackend, SyntheticBackend,
                         normalize_alternatives, probe_letter_tokens)
from src.metrics import m1_margin, m3_for_record, partial_entropy
from src.protocol import discovery_tasks, load_protocol
from src.records import Token, record_from_dict
from src.runner import run_single_turn_trajectory, run_trajectory


def sse(*chunks: dict) -> list[str]:
    # ensure_ascii=False mirrors vLLM's serializer, which leaves U+2028/U+2029 unescaped.
    return ["data: " + json.dumps(chunk, ensure_ascii=False) for chunk in chunks] + ["", "data: [DONE]"]


def token_chunk(text: str, logprob: float, alternatives, finish: str | None = None) -> dict:
    return {"object": "chat.completion.chunk", "choices": [{
        "index": 0, "delta": {"content": text}, "finish_reason": finish,
        "logprobs": {"content": [{"token": text, "logprob": logprob, "top_logprobs": [
            {"token": candidate, "logprob": score} for candidate, score in alternatives]}]}}]}


def usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"object": "chat.completion.chunk", "choices": [],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


SAMPLED_PROBABILITY = 0.55
DUPLICATE_PROBABILITY = 0.02


def letter_distribution(letter: str) -> list[tuple[str, float]]:
    """A realistic top-k slice over ` A`..` D` that keeps total mass below one."""
    sampled = letter.strip()
    others = [option for option in "ABCD" if option != sampled]
    probabilities = {sampled: SAMPLED_PROBABILITY, others[0]: 0.25, others[1]: 0.12, others[2]: 0.05}
    return [(" %s" % option, math.log(probabilities[option])) for option in "ABCD"]


def answer_stream(letter: str = " C") -> list[str]:
    """A three-token reply whose final token is the option letter with duplicate alternatives."""
    # A second token id that decodes to the same string, as vLLM can genuinely return.
    duplicated = letter_distribution(letter) + [(letter, math.log(DUPLICATE_PROBABILITY))]
    return sse(
        token_chunk("Reasoning.\n", math.log(0.90), [("Reasoning.\n", math.log(0.90)), ("Because", math.log(0.08))]),
        token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
        token_chunk(letter, math.log(SAMPLED_PROBABILITY), duplicated, finish="stop"),
        usage_chunk(41, 3),
    )


class FakeResponse:
    """Serves the SSE payload as byte chunks, the way httpx's iter_bytes does."""

    def __init__(self, status_code: int, lines: list[str], text: str = "", chunk_size: int = 7) -> None:
        self.status_code = status_code
        self.text = text
        self.request = object()
        self._payload = "".join("%s\n" % line for line in lines).encode("utf-8")
        self._chunk_size = chunk_size

    def read(self) -> bytes:
        return self.text.encode("utf-8")

    def iter_bytes(self):
        # Deliberately tiny chunks so line framing must survive split boundaries.
        for start in range(0, len(self._payload), self._chunk_size):
            yield self._payload[start:start + self._chunk_size]


class FakeStream:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> FakeResponse:
        return self._response

    def __exit__(self, *_: object) -> bool:
        return False


class FakeClient:
    """Stands in for httpx.Client; `handler` maps a request payload to a FakeResponse."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self._lock = threading.Lock()
        self.payloads: list[dict] = []

    def stream(self, method: str, url: str, json=None) -> FakeStream:
        with self._lock:
            self.payloads.append(json)
            index = len(self.payloads) - 1
        return FakeStream(self._handler(json, index))

    def close(self) -> None:
        pass


def backend_with(handler, **kwargs) -> OpenAICompatBackend:
    backend = OpenAICompatBackend("http://fake.local/v1", "test/model", max_retries=kwargs.pop("max_retries", 4), **kwargs)
    backend.close()
    backend._client = FakeClient(handler)
    return backend


REQUEST = GenerationRequest(({"role": "user", "content": "task"},), 7,
                            {"temperature": 0, "top_p": 1, "max_logprobs": 20, "max_tokens": 512})


class SyntheticBackendTests(unittest.TestCase):
    def test_repeatable_reconstructible_response(self):
        request = GenerationRequest(({"role": "user", "content": "task"},), 9, {"temperature": 0})
        backend = SyntheticBackend()
        first = backend.generate(request); second = backend.generate(request)
        self.assertEqual(first, second)
        self.assertEqual(first.text, "".join(token.text for token in first.tokens))
        self.assertRegex(first.text, r"\nAnswer: [A-D]$")
        answer = first.tokens[-1]
        self.assertEqual({text for text, _ in answer.top_logprobs}, {"A", "B", "C", "D"})
        self.assertTrue(all(1 <= len(token.top_logprobs) <= 20 for token in first.tokens))


class NormalizationTests(unittest.TestCase):
    def test_duplicate_alternatives_merge_by_log_sum_exp(self):
        merged = normalize_alternatives(" C", math.log(0.5), [(" C", math.log(0.2)), (" C", math.log(0.3)), (" A", math.log(0.1))])
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(dict(merged)[" C"], math.log(0.5), places=12)
        self.assertAlmostEqual(dict(merged)[" A"], math.log(0.1), places=12)

    def test_sampled_token_added_when_server_omits_it(self):
        merged = normalize_alternatives(" D", -5.0, [(" A", -0.1), (" B", -0.2)])
        self.assertIn(" D", dict(merged))
        self.assertEqual(dict(merged)[" D"], -5.0)

    def test_truncates_to_twenty_but_never_drops_the_sampled_token(self):
        crowd = [("t%02d" % index, -0.01 * index) for index in range(25)]
        merged = normalize_alternatives(" D", -40.0, crowd)
        self.assertEqual(len(merged), 20)
        self.assertIn(" D", dict(merged))
        self.assertEqual(len({text for text, _ in merged}), 20)

    def test_sorted_descending_and_non_positive(self):
        merged = normalize_alternatives("x", -1.0, [("a", -3.0), ("b", -0.5), ("c", 1e-9)])
        self.assertEqual([score for _, score in merged], sorted((score for _, score in merged), reverse=True))
        self.assertTrue(all(score <= 0.0 for _, score in merged))
        self.assertEqual(dict(merged)["c"], 0.0)


class OpenAICompatBackendTests(unittest.TestCase):
    def test_streaming_accumulation_text_matches_token_concatenation(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, answer_stream()))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "Reasoning.\nAnswer: C")
        self.assertEqual(result.text, "".join(token.text for token in result.tokens))
        self.assertEqual([token.text for token in result.tokens], ["Reasoning.\n", "Answer:", " C"])
        self.assertEqual(backend.stats["requests"], 1)
        self.assertEqual(backend.stats["prompt_tokens"], 41)
        self.assertEqual(backend.stats["completion_tokens"], 3)
        self.assertEqual(backend.stats["content_mismatches"], 0)

    def test_request_payload_maps_settings_without_mutating_them(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, answer_stream()))
        settings = dict(REQUEST.settings) | {"count": 10, "temperature": 0.8}
        snapshot = dict(settings)
        backend.generate(GenerationRequest(REQUEST.messages, 7, settings))
        self.assertEqual(settings, snapshot)  # the frozen configuration must survive untouched
        payload = backend._client.payloads[0]
        self.assertEqual(payload["temperature"], 0.8)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["top_logprobs"], 20)
        self.assertIs(payload["logprobs"], True)
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["seed"], 7)
        self.assertNotIn("count", payload)

    def test_duplicate_answer_alternatives_collapse_for_m1(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, answer_stream()))
        answer = backend.generate(REQUEST).tokens[-1]
        texts = [text for text, _ in answer.top_logprobs]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(set(texts), {" A", " B", " C", " D"})
        self.assertAlmostEqual(dict(answer.top_logprobs)[" C"],
                               math.log(SAMPLED_PROBABILITY + DUPLICATE_PROBABILITY), places=12)

    def test_retries_transient_failures_then_succeeds(self):
        def handler(payload, index):
            if index == 0:
                return FakeResponse(503, [], text="service unavailable")
            if index == 1:
                return FakeResponse(429, [], text="rate limited")
            return FakeResponse(200, answer_stream())

        backend = backend_with(handler)
        with mock.patch("src.backend.time.sleep") as sleep:
            result = backend.generate(REQUEST)
        self.assertEqual(result.text, "Reasoning.\nAnswer: C")
        self.assertEqual(backend.stats["retries"], 2)
        self.assertEqual(backend.stats["requests"], 1)
        self.assertEqual(sleep.call_count, 2)

    def test_exhausted_retries_raise_backend_error(self):
        backend = backend_with(lambda payload, index: FakeResponse(500, [], text="boom"), max_retries=2)
        with mock.patch("src.backend.time.sleep"):
            with self.assertRaises(BackendError):
                backend.generate(REQUEST)
        self.assertEqual(backend.stats["retries"], 2)

    def test_client_errors_fail_fast(self):
        backend = backend_with(lambda payload, index: FakeResponse(404, [], text="no such model"))
        with mock.patch("src.backend.time.sleep") as sleep:
            with self.assertRaises(BackendError):
                backend.generate(REQUEST)
        self.assertEqual(sleep.call_count, 0)

    def test_content_chunk_without_logprobs_is_rejected(self):
        naked = {"object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"content": "oops"}, "finish_reason": None}]}
        backend = backend_with(lambda payload, index: FakeResponse(200, sse(naked)))
        with self.assertRaises(BackendError):
            backend.generate(REQUEST)

    def test_content_free_chunks_are_skipped(self):
        role_only = {"object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "logprobs": None,
                                  "finish_reason": None}]}
        backend = backend_with(lambda payload, index: FakeResponse(200, [json.dumps(role_only).join(("data: ", ""))] + answer_stream()))
        self.assertEqual(backend.generate(REQUEST).text, "Reasoning.\nAnswer: C")

    def test_content_mismatch_is_counted_not_fatal(self):
        divergent = sse(
            token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
            {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "XX"},
             "logprobs": {"content": [{"token": " C", "logprob": math.log(SAMPLED_PROBABILITY),
                                       "top_logprobs": [{"token": candidate, "logprob": score} for candidate, score in letter_distribution(" C")]}]},
             "finish_reason": "stop"}]})
        backend = backend_with(lambda payload, index: FakeResponse(200, divergent))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "Answer: C")
        self.assertEqual(backend.stats["content_mismatches"], 1)

    def test_concurrent_generate_calls_are_thread_safe(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, answer_stream()))
        outputs: list[str] = []
        lock = threading.Lock()

        def worker():
            text = backend.generate(REQUEST).text
            with lock:
                outputs.append(text)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(outputs, ["Reasoning.\nAnswer: C"] * 16)
        self.assertEqual(backend.stats["requests"], 16)
        self.assertEqual(backend.stats["completion_tokens"], 48)


class RecordIntegrationTests(unittest.TestCase):
    def test_mocked_backend_tokens_survive_the_full_record_contract(self):
        protocol = load_protocol()
        task = discovery_tasks(protocol)[0]
        backend = backend_with(lambda payload, index: FakeResponse(200, answer_stream(" %s" % task.canonical_answer)))
        records = run_trajectory(task=task, cell_id="%s__malfunctioning_always_fail__neutral" % task.difficulty,
                                 model_id="test/model", immutable_revision="b" * 40, backend=backend, protocol=protocol)
        self.assertEqual(len(records), 6)  # initial + 3 feedback + measured + recovery
        for record in records:
            round_tripped = record_from_dict(record.to_dict(), protocol)
            self.assertEqual(round_tripped.response_text, "".join(t.text for t in round_tripped.tokens))
            self.assertTrue(all(1 <= len(t.top_logprobs) <= 20 for t in round_tripped.tokens))
        measured = next(record for record in records if record.turn_label == "measured")
        margin = m1_margin(measured, protocol=protocol)
        self.assertIsNotNone(margin.margin.value, margin.margin.missing_reason)
        self.assertIsNotNone(partial_entropy(measured).mean_partial_entropy.value)

    def test_an_empty_token_mid_response_still_yields_m1_and_counts_for_entropy(self):
        protocol = load_protocol()
        task = discovery_tasks(protocol)[0]
        letter = " %s" % task.canonical_answer

        def handler(payload, index):
            return FakeResponse(200, sse(
                token_chunk("Reason", math.log(0.90), [("Reason", math.log(0.90))]),
                # zero-width position sitting between the reasoning and the answer line
                token_chunk("", math.log(0.80), [("", math.log(0.80)), ("x", math.log(0.15))]),
                token_chunk("\nAnswer:", math.log(0.95), [("\nAnswer:", math.log(0.95))]),
                token_chunk(letter, math.log(SAMPLED_PROBABILITY), letter_distribution(letter),
                            finish="stop")))

        backend = backend_with(handler)
        records = run_trajectory(task=task, cell_id="%s__malfunctioning_always_fail__neutral" % task.difficulty,
                                 model_id="test/model", immutable_revision="b" * 40, backend=backend,
                                 protocol=protocol)
        measured = next(record for record in records if record.turn_label == "measured")
        self.assertEqual([t.text for t in measured.tokens], ["Reason", "", "\nAnswer:", letter])
        self.assertEqual(measured.response_text, "Reason\nAnswer:%s" % letter)
        margin = m1_margin(measured, protocol=protocol)
        # The zero-width token must not shift the character offset of the option token.
        self.assertIsNotNone(margin.margin.value, margin.margin.missing_reason)
        self.assertEqual(margin.option_token_index, 3)
        entropy = partial_entropy(measured)
        self.assertEqual(entropy.position_count, 4)  # the empty position is counted
        self.assertIsNotNone(entropy.mean_partial_entropy.value)


class EmptyTokenTests(unittest.TestCase):
    """vLLM emits byte-level pieces that decode to "" but still carry a real logprob."""

    def stream_with_empty_token(self):
        return sse(
            token_chunk("Reason", math.log(0.90), [("Reason", math.log(0.90))]),
            token_chunk("", math.log(0.80), [("", math.log(0.80)), ("x", math.log(0.15))]),
            token_chunk("\nAnswer:", math.log(0.95), [("\nAnswer:", math.log(0.95))]),
            token_chunk(" C", math.log(SAMPLED_PROBABILITY), letter_distribution(" C"), finish="stop"),
        )

    def test_empty_token_is_kept_as_a_generated_position(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, self.stream_with_empty_token()))
        result = backend.generate(REQUEST)
        self.assertEqual([token.text for token in result.tokens], ["Reason", "", "\nAnswer:", " C"])
        self.assertEqual(result.text, "Reason\nAnswer: C")
        empty = result.tokens[1]
        self.assertAlmostEqual(empty.logprob, math.log(0.80), places=12)
        self.assertIn("", dict(empty.top_logprobs))

    def test_empty_alternative_text_survives_normalization(self):
        merged = normalize_alternatives("", math.log(0.5), [("", math.log(0.2)), ("", math.log(0.3)),
                                                            ("a", math.log(0.1))])
        self.assertAlmostEqual(dict(merged)[""], math.log(0.5), places=12)
        self.assertEqual(len(merged), 2)

    def test_missing_or_non_string_token_is_still_an_error(self):
        for value in (None, 7):
            chunk = {"object": "chat.completion.chunk", "choices": [{
                "index": 0, "delta": {"content": "x"}, "finish_reason": None,
                "logprobs": {"content": [{"token": value, "logprob": -0.1,
                                          "top_logprobs": [{"token": "x", "logprob": -0.1}]}]}}]}
            backend = backend_with(lambda payload, index, c=chunk: FakeResponse(200, sse(c)))
            with self.assertRaises(BackendError):
                backend.generate(REQUEST)


class EmptyResponseTests(unittest.TestCase):
    """An immediately-terminated turn is a legitimate outcome, not a crash."""

    def eos_only_stream(self):
        return sse({"object": "chat.completion.chunk", "choices": [{
            "index": 0, "delta": {}, "finish_reason": "stop",
            "logprobs": {"content": [{"token": "<end_of_turn>", "logprob": math.log(0.97),
                                      "top_logprobs": [{"token": "<end_of_turn>", "logprob": math.log(0.97)},
                                                       {"token": "The", "logprob": math.log(0.02)}]}]}}]})

    def test_eos_only_response_becomes_one_zero_width_position(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, self.eos_only_stream()))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "")
        self.assertEqual(len(result.tokens), 1)
        token = result.tokens[0]
        self.assertEqual(token.text, "")
        # The EOS distribution is carried over -- it says how sure the model was about stopping.
        self.assertAlmostEqual(token.logprob, math.log(0.97), places=12)
        self.assertEqual(dict(token.top_logprobs).keys(), {"<end_of_turn>", "The"})
        self.assertEqual(backend.stats["empty_responses"], 1)
        self.assertEqual(backend.stats["trailing_special_tokens"], 1)

    def test_stream_with_no_logprob_entries_at_all_still_returns_a_record(self):
        bare = sse({"object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]})
        backend = backend_with(lambda payload, index: FakeResponse(200, bare))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "")
        self.assertEqual(result.tokens, (Token("", 0.0, (("", 0.0),)),))
        self.assertEqual(backend.stats["empty_responses"], 1)
        self.assertEqual(backend.stats["trailing_special_tokens"], 0)

    def test_empty_response_records_and_parses_as_an_invalid_final_answer(self):
        protocol = load_protocol()
        task = discovery_tasks(protocol)[0]
        backend = backend_with(lambda payload, index: FakeResponse(200, self.eos_only_stream()))
        # Single-turn path: a multi-turn replay additionally needs protocol.py to accept an
        # empty assistant turn, which is outside this module's contract (see lab log).
        records = run_single_turn_trajectory(task=task, cell_id="style__neutral_reference",
                                             model_id="test/model", immutable_revision="b" * 40,
                                             backend=backend, protocol=protocol)
        measured = records[0]
        self.assertEqual(measured.response_text, "")
        self.assertFalse(measured.final_answer_valid)
        self.assertIsNone(measured.final_answer_letter)
        round_tripped = record_from_dict(measured.to_dict(), protocol)
        self.assertEqual(round_tripped.response_text, "".join(t.text for t in round_tripped.tokens))
        self.assertEqual(m1_margin(measured, protocol=protocol).margin.missing_reason,
                         "m1_invalid_final_answer")
        self.assertEqual(m3_for_record(measured).rate_per_100_tokens.missing_reason,
                         "m3_zero_visible_reasoning_tokens")
        self.assertEqual(partial_entropy(measured).position_count, 1)

    def test_multi_turn_trajectory_survives_an_empty_response_on_turn_two(self):
        protocol = load_protocol()
        task = discovery_tasks(protocol)[0]
        letter = " %s" % task.canonical_answer

        def handler(payload, index):
            if index == 1:  # the second assistant turn terminates immediately
                return FakeResponse(200, self.eos_only_stream())
            return FakeResponse(200, answer_stream(letter))

        backend = backend_with(handler)
        records = run_trajectory(task=task, cell_id="%s__accurate__neutral" % task.difficulty,
                                 model_id="test/model", immutable_revision="b" * 40,
                                 backend=backend, protocol=protocol)
        self.assertEqual(len(records), 7)  # initial + 3 feedback + measured + onset + washout
        for record in records:
            round_tripped = record_from_dict(record.to_dict(), protocol)
            self.assertEqual(round_tripped.response_text,
                             "".join(t.text for t in round_tripped.tokens))
        empty = records[1]
        self.assertEqual(empty.response_text, "")
        self.assertFalse(empty.final_answer_valid)
        self.assertIsNone(empty.final_answer_letter)
        self.assertIsNone(empty.final_answer_correct)
        # Every other turn still parses, so the empty turn did not corrupt the replay.
        self.assertTrue(all(record.final_answer_valid for record in records if record is not empty))
        # In the accurate arm an invalid preceding answer branches as incorrect.
        incorrect = protocol.conditions["feedback_messages"]["accurate"]["neutral"]["if_preceding_answer_incorrect"]
        self.assertEqual(records[2].messages[-1]["content"], incorrect)
        self.assertEqual(backend.stats["empty_responses"], 1)


class SseFramingTests(unittest.TestCase):
    def test_unicode_line_separators_inside_json_do_not_split_a_chunk(self):
        # U+2028 is legal unescaped inside a JSON string, and str.splitlines() breaks on it;
        # framing on raw newline bytes must not.
        exotic = "a bc"
        stream = sse(
            token_chunk(exotic, math.log(0.9), [(exotic, math.log(0.9))]),
            token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
            token_chunk(" C", math.log(SAMPLED_PROBABILITY), letter_distribution(" C"), finish="stop"),
        )
        # str.splitlines() shatters this single SSE line; framing on newline bytes must not.
        self.assertGreater(len(stream[0].splitlines()), 1)
        self.assertEqual(stream[0].count("\n"), 0)
        backend = backend_with(lambda payload, index: FakeResponse(200, stream))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "%sAnswer: C" % exotic)
        self.assertEqual(result.tokens[0].text, exotic)

    def test_multibyte_characters_split_across_byte_chunks_survive(self):
        emoji = "🎉→é"
        stream = sse(
            token_chunk(emoji, math.log(0.9), [(emoji, math.log(0.9))]),
            token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
            token_chunk(" C", math.log(SAMPLED_PROBABILITY), letter_distribution(" C"), finish="stop"),
        )
        backend = backend_with(lambda payload, index: FakeResponse(200, stream, chunk_size=3))
        self.assertEqual(backend.generate(REQUEST).text, "%sAnswer: C" % emoji)


class TrailingSpecialTokenTests(unittest.TestCase):
    """vLLM streams <end_of_turn> as a logprob entry with no matching content delta."""

    def stream_with_eos(self):
        letters = letter_distribution(" C")
        return sse(
            token_chunk("Reasoning.\n", math.log(0.90), [("Reasoning.\n", math.log(0.90))]),
            token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
            token_chunk(" C", math.log(SAMPLED_PROBABILITY), letters),
            {"object": "chat.completion.chunk", "choices": [{
                "index": 0, "delta": {}, "finish_reason": "stop",
                "logprobs": {"content": [{"token": "<end_of_turn>", "logprob": math.log(0.99),
                                          "top_logprobs": [{"token": "<end_of_turn>", "logprob": math.log(0.99)}]}]}}]},
        )

    def test_end_of_turn_token_is_trimmed_and_counted(self):
        backend = backend_with(lambda payload, index: FakeResponse(200, self.stream_with_eos()))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "Reasoning.\nAnswer: C")
        self.assertEqual([token.text for token in result.tokens], ["Reasoning.\n", "Answer:", " C"])
        self.assertEqual(backend.stats["trailing_special_tokens"], 1)
        self.assertEqual(backend.stats["content_mismatches"], 0)

    def test_genuine_divergence_is_not_trimmed(self):
        divergent = sse(
            token_chunk("Answer:", math.log(0.95), [("Answer:", math.log(0.95))]),
            {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "ZZ"},
             "logprobs": {"content": [{"token": " C", "logprob": math.log(SAMPLED_PROBABILITY),
                                       "top_logprobs": [{"token": t, "logprob": s} for t, s in letter_distribution(" C")]}]},
             "finish_reason": "stop"}]})
        backend = backend_with(lambda payload, index: FakeResponse(200, divergent))
        result = backend.generate(REQUEST)
        self.assertEqual(result.text, "Answer: C")  # concatenation wins; nothing silently dropped
        self.assertEqual(backend.stats["trailing_special_tokens"], 0)
        self.assertEqual(backend.stats["content_mismatches"], 1)


class ProbeLetterTokenTests(unittest.TestCase):
    def _client(self, stem_ids, per_letter, decoded):
        class Response:
            def __init__(self, body): self._body = body
            def raise_for_status(self): return None
            def json(self): return self._body

        class Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def post(self, url, json=None):
                if url.endswith("/tokenize"):
                    prompt = json["prompt"]
                    if prompt == "Answer:":
                        return Response({"count": len(stem_ids), "tokens": list(stem_ids)})
                    return Response({"tokens": per_letter[prompt[-1]]})
                return Response({"prompt": decoded.get(tuple(json["tokens"]), "?")})
        return Client

    def test_single_letter_tokens_detected(self):
        stem = [100, 200]
        per_letter = {letter: stem + [300 + index] for index, letter in enumerate("ABCD")}
        # /tokenize reports the vocabulary piece; only /detokenize gives the decoded " A".
        decoded = {(300 + index,): " %s" % letter for index, letter in enumerate("ABCD")}
        with mock.patch("httpx.Client", self._client(stem, per_letter, decoded)):
            self.assertEqual(probe_letter_tokens("http://fake.local/v1", "m"),
                             {"A": True, "B": True, "C": True, "D": True})

    def test_multi_token_or_mismatched_letters_reported_false(self):
        stem = [100, 200]
        per_letter = {letter: stem + [300, 301] for letter in "ABC"}  # two tokens, not one
        per_letter["D"] = stem + [400]
        decoded = {(400,): "D"}  # single token, but no leading space
        with mock.patch("httpx.Client", self._client(stem, per_letter, decoded)):
            self.assertEqual(probe_letter_tokens("http://fake.local/v1", "m"),
                             {"A": False, "B": False, "C": False, "D": False})
