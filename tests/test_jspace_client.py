"""Offline tests for the Phase 3 j-space client. No network, no GPU, no `modal` import.

Every remote call is a fake, so what is under test is exactly the local half: chunking and
concatenation, the npz round trip, and the token-dict to `records.Token` conversion that lets a
steered generation be read by `src.metrics.m1_margin` like any vLLM record.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from src.jspace_client import (HIDDEN_SIZE, JSpaceClientError, MODEL_ID, NUM_HIDDEN_STATES,
                               REVISION, extract_activations, generate_steered, load_npz,
                               merge_activation_chunks, save_npz, to_tokens, token_dicts)
from src.records import Token


def make_items(count: int, prefix: str = "task") -> list[dict]:
    return [{"id": "%s-%02d" % (prefix, index),
             "messages": [{"role": "user", "content": "question %d" % index}]}
            for index in range(count)]


class FakeExtractor:
    """Stands in for `JSpace.extract_activations.remote`, with deterministic activations."""

    def __init__(self, layers: list[int] | None = None, hidden: int = 4) -> None:
        self.layers = layers or [0, 1, 2]
        self.hidden = hidden
        self.calls: list[tuple[list[dict], list[int] | None]] = []
        self.batch_sizes: list[int | None] = []

    def __call__(self, items, layers=None, batch_size=None):
        self.calls.append((list(items), layers))
        self.batch_sizes.append(batch_size)
        chosen = list(layers) if layers is not None else self.layers
        values = np.arange(len(items) * len(chosen) * self.hidden, dtype=np.float32)
        block = values.reshape(len(items), len(chosen), self.hidden) + 100.0 * len(self.calls)
        norms = np.linalg.norm(block, axis=-1).mean(axis=0)
        return {
            "ids": [item["id"] for item in items],
            "layers": chosen,
            "activations": block.astype(np.float16),
            "norms": [float(value) for value in norms],
            "prompt_tokens": [7 + index for index in range(len(items))],
            "hidden_size": self.hidden,
            "num_layers": len(chosen) - 1,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "peak_abs_activation": float(np.abs(block).max()),
            "float16_overflow": 0,
            "seconds": 1.5,
        }


class FakeGenerator:
    """Stands in for `JSpace.generate_steered.remote`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, items, layer, direction, alphas, max_new_tokens=512, top_k_logprobs=20,
                 batch_size=None):
        self.calls.append({"ids": [item["id"] for item in items], "layer": layer,
                           "direction": direction, "alphas": list(alphas),
                           "max_new_tokens": max_new_tokens, "top_k_logprobs": top_k_logprobs,
                           "batch_size": batch_size})
        entries = []
        for item in items:
            for alpha in alphas:
                entries.append({
                    "id": item["id"], "alpha": float(alpha), "layer": layer,
                    "decoder_layer": layer - 1, "steered": direction is not None and alpha != 0,
                    "text": "Answer: A", "finish": "eos", "generated_tokens": 3,
                    "nonfinite_steps": 0,
                    "tokens": [
                        {"text": "Answer:", "logprob": -0.2,
                         "top_logprobs": [{"text": "Answer:", "logprob": -0.2}]},
                        {"text": " A", "logprob": -0.1,
                         "top_logprobs": [{"text": " A", "logprob": -0.1},
                                          {"text": " B", "logprob": -2.3}]},
                    ],
                })
        return entries


class ChunkConcatenationTests(unittest.TestCase):
    def test_activation_chunks_are_concatenated_in_order(self):
        items = make_items(7)
        remote = FakeExtractor()
        result = extract_activations(items, [0, 1, 2], chunk=3, remote=remote, progress=False)
        self.assertEqual([len(call[0]) for call in remote.calls], [3, 3, 1])
        self.assertEqual(result["ids"], [item["id"] for item in items])
        self.assertEqual(result["layers"], [0, 1, 2])
        self.assertEqual(result["activations"].shape, (7, 3, 4))
        self.assertEqual(result["activations"].dtype, np.float16)
        self.assertEqual(result["hidden_size"], 4)
        self.assertEqual(len(result["prompt_tokens"]), 7)

    def test_concatenated_rows_keep_their_chunk_values(self):
        remote = FakeExtractor()
        result = extract_activations(make_items(4), [0, 1, 2], chunk=3, remote=remote, progress=False)
        first_chunk = np.asarray(remote(make_items(3), [0, 1, 2])["activations"])  # 4th call
        # Row 3 came from the second call, whose offset is 200.0; rows 0-2 from the first (100.0).
        self.assertAlmostEqual(float(result["activations"][0, 0, 0]), 100.0)
        self.assertAlmostEqual(float(result["activations"][3, 0, 0]), 200.0)
        self.assertEqual(first_chunk.shape, (3, 3, 4))

    def test_norms_are_item_weighted_not_a_mean_of_means(self):
        layers = [0, 1]
        chunks = [
            {"ids": ["a", "b", "c"], "layers": layers, "activations": np.zeros((3, 2, 4), np.float16),
             "norms": [10.0, 20.0], "prompt_tokens": [1, 1, 1]},
            {"ids": ["d"], "layers": layers, "activations": np.zeros((1, 2, 4), np.float16),
             "norms": [2.0, 4.0], "prompt_tokens": [1]},
        ]
        merged = merge_activation_chunks(chunks)
        self.assertAlmostEqual(merged["norms"][0], (10.0 * 3 + 2.0) / 4)
        self.assertAlmostEqual(merged["norms"][1], (20.0 * 3 + 4.0) / 4)

    def test_layer_disagreement_between_chunks_is_fatal(self):
        chunks = [
            {"ids": ["a"], "layers": [0, 1], "activations": np.zeros((1, 2, 4), np.float16),
             "norms": [1.0, 1.0]},
            {"ids": ["b"], "layers": [0, 2], "activations": np.zeros((1, 2, 4), np.float16),
             "norms": [1.0, 1.0]},
        ]
        with self.assertRaises(JSpaceClientError):
            merge_activation_chunks(chunks)

    def test_shape_mismatch_and_duplicate_ids_are_fatal(self):
        with self.assertRaises(JSpaceClientError):
            merge_activation_chunks([{"ids": ["a", "b"], "layers": [0],
                                      "activations": np.zeros((1, 1, 4), np.float16),
                                      "norms": [1.0]}])
        duplicated = [{"ids": ["a"], "layers": [0], "activations": np.zeros((1, 1, 4), np.float16),
                       "norms": [1.0]}] * 2
        with self.assertRaises(JSpaceClientError):
            merge_activation_chunks(duplicated)

    def test_a_chunk_from_another_model_is_fatal(self):
        with self.assertRaises(JSpaceClientError):
            merge_activation_chunks([{"ids": ["a"], "layers": [0], "norms": [1.0],
                                      "activations": np.zeros((1, 1, 4), np.float16),
                                      "model_id": "google/gemma-2-2b-it"}])

    def test_generation_chunks_cover_every_item_and_dose(self):
        items = make_items(5)
        remote = FakeGenerator()
        direction = [0.5] * HIDDEN_SIZE
        entries = generate_steered(items, 21, direction, [0.0, 2.0], chunk=2, remote=remote,
                                   progress=False)
        self.assertEqual([len(call["ids"]) for call in remote.calls], [2, 2, 1])
        self.assertEqual(len(entries), 10)
        self.assertEqual({entry["id"] for entry in entries}, {item["id"] for item in items})
        self.assertEqual(sorted({entry["alpha"] for entry in entries}), [0.0, 2.0])
        self.assertTrue(all(call["layer"] == 21 for call in remote.calls))
        self.assertTrue(all(call["alphas"] == [0.0, 2.0] for call in remote.calls))
        self.assertTrue(all(len(call["direction"]) == HIDDEN_SIZE for call in remote.calls))

    def test_batch_size_is_forwarded_only_when_asked_for(self):
        default = FakeExtractor()
        extract_activations(make_items(2), [0, 1, 2], chunk=2, remote=default, progress=False)
        self.assertEqual(default.batch_sizes, [None])
        tuned = FakeExtractor()
        extract_activations(make_items(2), [0, 1, 2], chunk=2, batch_size=8, remote=tuned,
                            progress=False)
        self.assertEqual(tuned.batch_sizes, [8])
        generator = FakeGenerator()
        generate_steered(make_items(2), 21, None, [0.0], chunk=2, batch_size=8, remote=generator,
                         progress=False)
        self.assertEqual([call["batch_size"] for call in generator.calls], [8])

    def test_a_wrong_length_direction_never_reaches_the_gpu(self):
        remote = FakeGenerator()
        with self.assertRaises(JSpaceClientError):
            generate_steered(make_items(2), 21, [1.0, 2.0], [0.0], remote=remote, progress=False)
        self.assertEqual(remote.calls, [])

    def test_a_short_remote_reply_is_fatal(self):
        def truncating(items, layer, direction, alphas, max_new_tokens=512, top_k_logprobs=20):
            return [{"id": items[0]["id"], "alpha": alphas[0], "text": "", "tokens": []}]

        with self.assertRaises(JSpaceClientError):
            generate_steered(make_items(2), 21, None, [0.0, 1.0], chunk=2, remote=truncating,
                             progress=False)

    def test_an_unknown_id_in_the_reply_is_fatal(self):
        def mislabelling(items, layer, direction, alphas, max_new_tokens=512, top_k_logprobs=20):
            return [{"id": "someone-else", "alpha": alphas[0], "text": "", "tokens": []}]

        with self.assertRaises(JSpaceClientError):
            generate_steered(make_items(1), 21, None, [0.0], chunk=1, remote=mislabelling,
                             progress=False)


class NpzRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)

    def test_round_trip_preserves_arrays_and_metadata(self):
        result = extract_activations(make_items(5), [0, 1, 2], chunk=2,
                                     remote=FakeExtractor(), progress=False)
        path = save_npz(self.directory / "nested" / "activations.npz", result)
        self.assertTrue(path.exists())
        loaded = load_npz(path)
        self.assertEqual(loaded["ids"], result["ids"])
        self.assertEqual(loaded["layers"], result["layers"])
        self.assertEqual(loaded["activations"].dtype, np.float16)
        np.testing.assert_array_equal(loaded["activations"], result["activations"])
        for index, value in enumerate(result["norms"]):
            self.assertAlmostEqual(loaded["norms"][index], value, places=6)
        self.assertEqual(loaded["model_id"], MODEL_ID)
        self.assertEqual(loaded["revision"], REVISION)
        self.assertEqual(loaded["schema"], "dgs-jspace-activations-v1")

    def test_unicode_ids_survive_the_round_trip(self):
        result = {"ids": ["easy__accurate__neutral|t-01", "hard__x|t-02"], "layers": [7],
                  "activations": np.ones((2, 1, 3), dtype=np.float16), "norms": [1.5]}
        loaded = load_npz(save_npz(self.directory / "ids.npz", result))
        self.assertEqual(loaded["ids"], result["ids"])

    def test_saving_an_inconsistent_payload_is_refused(self):
        with self.assertRaises(JSpaceClientError):
            save_npz(self.directory / "bad.npz",
                     {"ids": ["a", "b"], "layers": [0],
                      "activations": np.zeros((1, 1, 3), np.float16), "norms": [1.0]})


class TokenConversionTests(unittest.TestCase):
    def entry(self, **overrides):
        base = {
            "id": "task-01", "alpha": 0.0, "text": "Answer: A", "finish": "eos",
            "tokens": [
                {"text": "Answer:", "logprob": -0.25,
                 "top_logprobs": [{"text": "Answer:", "logprob": -0.25},
                                  {"text": "The", "logprob": -1.5}]},
                {"text": " A", "logprob": -0.1,
                 "top_logprobs": [{"text": " A", "logprob": -0.1}, {"text": " B", "logprob": -2.0},
                                  {"text": " C", "logprob": -3.0}, {"text": " D", "logprob": -4.0}]},
            ],
        }
        base.update(overrides)
        return base

    def test_entry_becomes_a_records_token_trace(self):
        tokens = to_tokens(self.entry())
        self.assertEqual(len(tokens), 2)
        self.assertTrue(all(isinstance(token, Token) for token in tokens))
        self.assertEqual("".join(token.text for token in tokens), "Answer: A")
        self.assertEqual(tokens[1].text, " A")
        self.assertEqual([text for text, _ in tokens[1].top_logprobs], [" A", " B", " C", " D"])

    def test_text_that_disagrees_with_the_trace_is_fatal(self):
        with self.assertRaises(JSpaceClientError):
            to_tokens(self.entry(text="Answer: B"))

    def test_duplicate_alternative_texts_merge_by_log_sum_exp(self):
        entry = self.entry(text=" A", tokens=[{
            "text": " A", "logprob": -0.6931471805599453,
            "top_logprobs": [{"text": " A", "logprob": -0.6931471805599453},
                             {"text": " A", "logprob": -0.6931471805599453},
                             {"text": " B", "logprob": -3.0}]}])
        tokens = to_tokens(entry)
        texts = [text for text, _ in tokens[0].top_logprobs]
        self.assertEqual(len(texts), len(set(texts)))
        merged = dict(tokens[0].top_logprobs)[" A"]
        # log(exp(-ln2) + exp(-ln2)) == 0
        self.assertAlmostEqual(merged, 0.0, places=9)
        self.assertLessEqual(merged, 0.0)

    def test_more_than_twenty_alternatives_are_truncated_around_the_sampled_token(self):
        alternatives = [{"text": "t%02d" % index, "logprob": -0.01 * index} for index in range(30)]
        alternatives.append({"text": " A", "logprob": -9.0})  # sampled, far down the ranking
        entry = self.entry(text=" A", tokens=[{"text": " A", "logprob": -9.0,
                                               "top_logprobs": alternatives}])
        tokens = to_tokens(entry)
        self.assertEqual(len(tokens[0].top_logprobs), 20)
        self.assertIn(" A", [text for text, _ in tokens[0].top_logprobs])
        self.assertTrue(all(value <= 0.0 for _, value in tokens[0].top_logprobs))

    def test_an_empty_eos_only_turn_is_one_zero_width_position(self):
        entry = self.entry(text="", tokens=[{"text": "", "logprob": -0.05,
                                             "top_logprobs": [{"text": "<end_of_turn>", "logprob": -0.05},
                                                              {"text": "I", "logprob": -3.2}]}])
        tokens = to_tokens(entry)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].text, "")
        self.assertEqual(len(tokens[0].top_logprobs), 3)  # the sampled "" is guaranteed a slot

    def test_non_finite_logprobs_are_floored_rather_than_recorded(self):
        entry = self.entry(text=" A", tokens=[{
            "text": " A", "logprob": -0.1,
            "top_logprobs": [{"text": " A", "logprob": -0.1},
                             {"text": " B", "logprob": float("-inf")}]}])
        tokens = to_tokens(entry)
        self.assertTrue(all(math.isfinite(value) for _, value in tokens[0].top_logprobs))
        self.assertEqual(dict(tokens[0].top_logprobs)[" B"], -9999.0)

    def test_a_positive_logprob_is_refused(self):
        with self.assertRaises(JSpaceClientError):
            to_tokens(self.entry(tokens=[{"text": "Answer: A", "logprob": float("inf"),
                                          "top_logprobs": [{"text": "Answer: A", "logprob": 0.0}]}]))

    def test_token_dicts_match_the_records_json_shape(self):
        payload = token_dicts(self.entry())
        self.assertEqual("".join(position["text"] for position in payload), "Answer: A")
        for position in payload:
            self.assertEqual(set(position), {"text", "logprob", "top_logprobs"})
            self.assertTrue(1 <= len(position["top_logprobs"]) <= 20)
            texts = [alternative["text"] for alternative in position["top_logprobs"]]
            self.assertEqual(len(texts), len(set(texts)))


class ConstantsTests(unittest.TestCase):
    def test_layer_space_constants_describe_gemma_2_9b(self):
        self.assertEqual((HIDDEN_SIZE, NUM_HIDDEN_STATES), (3584, 43))
        self.assertEqual(MODEL_ID, "google/gemma-2-9b-it")
        self.assertRegex(REVISION, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
