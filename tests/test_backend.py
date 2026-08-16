from __future__ import annotations

import unittest

from src.backend import GenerationRequest, SyntheticBackend


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
