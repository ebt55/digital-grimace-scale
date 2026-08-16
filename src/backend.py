"""Deterministic, offline-only generation backend for validation smoke runs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Protocol

from .records import Token


@dataclass(frozen=True)
class GenerationRequest:
    messages: tuple[Mapping[str, str], ...]
    seed: int
    settings: Mapping[str, object]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens: tuple[Token, ...]


class GenerationBackend(Protocol):
    """Minimal generation surface consumed by the frozen transcript runner."""

    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class SyntheticBackend:
    """A byte-stable backend that never contacts a model or network."""

    name = "synthetic"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = repr((tuple((m["role"], m["content"]) for m in request.messages),
                        request.seed, tuple(sorted(request.settings.items())))).encode("utf-8")
        digest = sha256(payload).digest()
        answer = "ABCD"[digest[0] % 4]
        prefix = "Synthetic deterministic reasoning.\nAnswer: "
        alternatives = tuple((letter, -0.05 - index) for index, letter in enumerate("ABCD"))
        chosen = next(score for letter, score in alternatives if letter == answer)
        tokens = (Token(prefix, -0.01, ((prefix, -0.01),)), Token(answer, chosen, alternatives))
        return GenerationResult(prefix + answer, tokens)
