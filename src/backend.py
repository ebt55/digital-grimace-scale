"""Generation backends: an offline deterministic stub and a vLLM OpenAI-compatible client."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
import threading
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from .records import Token

# Tolerance for float noise around the log-probability ceiling of zero.
LOGPROB_EPSILON = 1e-6
# Floor substituted for non-finite alternatives so records.py stays satisfiable.
LOGPROB_FLOOR = -9999.0
MAX_ALTERNATIVES = 20


class BackendError(RuntimeError):
    """Raised when a generation backend cannot produce a usable response."""


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


def _logaddexp(first: float, second: float) -> float:
    """Numerically stable log(exp(first) + exp(second)) for merging duplicate tokens."""
    high, low = (first, second) if first >= second else (second, first)
    if math.isinf(low):
        return high
    return high + math.log1p(math.exp(low - high))


def _clean_logprob(value: Any) -> tuple[float, bool]:
    """Coerce a server log probability into the finite, non-positive range records.py demands."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendError("log probability must be a number, got %r" % (value,))
    number = float(value)
    if math.isnan(number):
        raise BackendError("log probability must not be NaN")
    if math.isinf(number):
        # -inf is a legitimate vLLM/OpenAI encoding of "vanishing probability".
        if number > 0:
            raise BackendError("log probability must not be +inf")
        return LOGPROB_FLOOR, True
    if number > 0:
        if number > LOGPROB_EPSILON:
            raise BackendError("log probability must be less than or equal to zero, got %r" % number)
        return 0.0, False
    return number, False


def normalize_alternatives(sampled_text: str, sampled_logprob: float,
                           alternatives: Iterable[tuple[str, float]],
                           *, limit: int = MAX_ALTERNATIVES) -> tuple[tuple[str, float], ...]:
    """Merge duplicate texts, guarantee the sampled token, sort descending, truncate to `limit`.

    vLLM can return several distinct token IDs that decode to the same string; records.py
    requires distinct alternative texts, so duplicates are combined by log-sum-exp.
    """
    if limit < 1:
        raise BackendError("alternative limit must be positive")
    merged: dict[str, float] = {}
    for text, logprob in alternatives:
        if not isinstance(text, str) or not text:
            continue
        merged[text] = _logaddexp(merged[text], logprob) if text in merged else logprob
    if sampled_text not in merged:
        merged[sampled_text] = sampled_logprob
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    kept = ranked[:limit]
    if all(text != sampled_text for text, _ in kept):
        # The sampled token must survive truncation even when `limit` alternatives outrank it.
        kept[-1] = (sampled_text, merged[sampled_text])
        kept.sort(key=lambda item: (-item[1], item[0]))
    return tuple((text, min(0.0, logprob)) for text, logprob in kept)


def _sse_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Frame a Server-Sent Events stream on newline bytes only.

    httpx's `iter_lines` splits on everything `str.splitlines` treats as a break -- including
    U+2028/U+2029 and U+0085, which JSON permits *unescaped* inside a string. A generated
    token containing one of those characters therefore chops a `data:` line in half and the
    JSON parse fails. Splitting the raw bytes on 0x0A cannot misfire that way, because no
    UTF-8 continuation byte is 0x0A.
    """
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while True:
            index = buffer.find(b"\n")
            if index < 0:
                break
            line, buffer = buffer[:index], buffer[index + 1:]
            yield line.decode("utf-8", errors="replace").strip()
    if buffer.strip():
        yield buffer.decode("utf-8", errors="replace").strip()


def _trim_trailing_special(tokens: list[Token], content: str) -> tuple[list[Token], int]:
    """Drop trailing tokens the server never showed in `message.content`.

    vLLM streams the end-of-turn token as a logprob entry with no matching content delta, so
    a naive concatenation appends `<end_of_turn>` to the response. That extra line makes the
    frozen `Answer: X` parser reject an otherwise-valid response, so trim it -- but only when
    the visible content is an exact prefix, never when the two genuinely disagree.
    """
    if not content:
        return tokens, 0
    joined = "".join(token.text for token in tokens)
    if joined == content or not joined.startswith(content):
        return tokens, 0
    trimmed = list(tokens)
    while trimmed and len("".join(token.text for token in trimmed)) > len(content):
        trimmed.pop()
    if trimmed and "".join(token.text for token in trimmed) == content:
        return trimmed, len(tokens) - len(trimmed)
    return tokens, 0


def _endpoint_root(base_url: str) -> str:
    """Strip the OpenAI `/v1` suffix so server-level routes such as /tokenize resolve."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")].rstrip("/") if trimmed.endswith("/v1") else trimmed


class OpenAICompatBackend:
    """Streaming client for a vLLM (or any OpenAI-compatible) chat-completions server.

    Token traces are normalized so every response satisfies the raw-record contract:
    one to twenty distinct alternatives per position, all finite and non-positive, and
    `text` exactly equal to the concatenated token trace.
    """

    def __init__(self, base_url: str, model: str, *, api_key: str = "EMPTY",
                 timeout_s: float = 600.0, max_retries: int = 4,
                 name: str = "vllm_openai_compat") -> None:
        import httpx  # imported lazily so offline test runs never need the dependency

        if not isinstance(base_url, str) or not base_url.strip():
            raise BackendError("base_url must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise BackendError("model must be a nonempty string")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise BackendError("max_retries must be a nonnegative integer")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = max_retries
        self._api_key = api_key
        self._include_usage = True
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {"requests": 0, "retries": 0, "content_mismatches": 0,
                                      "prompt_tokens": 0, "completion_tokens": 0,
                                      "nonfinite_logprobs": 0, "truncated": 0,
                                      "trailing_special_tokens": 0}
        # ~100 worker threads share this client, so the pool must not become the bottleneck.
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout_s, connect=60.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
            headers={"Authorization": "Bearer %s" % api_key, "Accept": "text/event-stream"},
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _bump(self, field: str, amount: int = 1) -> None:
        with self._lock:
            self.stats[field] = self.stats.get(field, 0) + amount

    # -- request construction ---------------------------------------------
    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        settings = dict(request.settings)  # never mutate the frozen configuration
        top_logprobs = settings.get("max_logprobs", MAX_ALTERNATIVES)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message["role"], "content": message["content"]}
                         for message in request.messages],
            "temperature": settings.get("temperature", 0),
            "top_p": settings.get("top_p", 1),
            "max_tokens": settings.get("max_tokens"),
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "seed": request.seed,
            "stream": True,
            "n": 1,
        }
        if self._include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- streaming ---------------------------------------------------------
    def _stream_once(self, payload: Mapping[str, Any]) -> tuple[list[Token], str, dict[str, int]]:
        import httpx

        tokens: list[Token] = []
        content_parts: list[str] = []
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        url = "%s/chat/completions" % self.base_url
        with self._client.stream("POST", url, json=dict(payload)) as response:
            if response.status_code >= 400:
                response.read()
                raise httpx.HTTPStatusError("HTTP %d: %s" % (response.status_code, response.text[:500]),
                                            request=response.request, response=response)
            for line in _sse_lines(response.iter_bytes()):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise BackendError("malformed stream chunk: %s" % data[:200]) from exc
                if isinstance(chunk.get("usage"), Mapping):
                    for key in ("prompt_tokens", "completion_tokens"):
                        value = chunk["usage"].get(key)
                        if isinstance(value, int) and not isinstance(value, bool):
                            usage[key] = value
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                entries = self._logprob_entries(choice)
                if entries is None:
                    # A content-bearing chunk without logprobs would silently corrupt the trace.
                    if isinstance(piece, str) and piece:
                        raise BackendError("stream chunk carried content but no logprob information")
                    continue
                for entry in entries:
                    tokens.append(self._token(entry))
        if not tokens:
            raise BackendError("server returned no tokens with logprobs")
        if finish_reason == "length":
            self._bump("truncated")
        content = "".join(content_parts)
        tokens, dropped = _trim_trailing_special(tokens, content)
        if dropped:
            self._bump("trailing_special_tokens", dropped)
        return tokens, content, usage

    @staticmethod
    def _logprob_entries(choice: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, Mapping):
            return None
        entries = logprobs.get("content")
        if entries is None:
            entries = logprobs.get("tokens")
        if not isinstance(entries, list) or not entries:
            return None
        return [entry for entry in entries if isinstance(entry, Mapping)]

    def _token(self, entry: Mapping[str, Any]) -> Token:
        text = entry.get("token")
        if not isinstance(text, str) or not text:
            raise BackendError("logprob entry is missing a nonempty token string")
        logprob, floored = _clean_logprob(entry.get("logprob"))
        if floored:
            self._bump("nonfinite_logprobs")
        raw: list[tuple[str, float]] = []
        for alternative in entry.get("top_logprobs") or ():
            if not isinstance(alternative, Mapping):
                continue
            candidate = alternative.get("token")
            if not isinstance(candidate, str) or not candidate:
                continue
            score, was_floored = _clean_logprob(alternative.get("logprob"))
            if was_floored:
                self._bump("nonfinite_logprobs")
            raw.append((candidate, score))
        return Token(text, logprob, normalize_alternatives(text, logprob, raw))

    # -- public surface ----------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        import httpx

        payload = self._payload(request)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                tokens, content, usage = self._stream_once(payload)
            except (httpx.TransportError, httpx.HTTPStatusError, BackendError) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    if status == 400 and self._include_usage and "stream_options" in exc.response.text:
                        # Older OpenAI-compatible servers reject stream_options; drop it and retry.
                        self._include_usage = False
                        payload = self._payload(request)
                        continue
                    if status < 500 and status != 429:
                        raise BackendError("generation failed permanently: %s" % exc) from exc
                elif isinstance(exc, BackendError):
                    raise
                last = exc
                if attempt >= self.max_retries:
                    break
                self._bump("retries")
                delay = min(30.0, 1.5 * (2 ** attempt)) * (0.5 + random.random())
                time.sleep(delay)
                continue
            self._bump("requests")
            self._bump("prompt_tokens", usage.get("prompt_tokens", 0))
            self._bump("completion_tokens", usage.get("completion_tokens", 0))
            text = "".join(token.text for token in tokens)
            if content and content != text:
                self._bump("content_mismatches")
            return GenerationResult(text, tuple(tokens))
        raise BackendError("generation failed after %d attempts: %s" % (self.max_retries + 1, last))


def probe_letter_tokens(base_url: str, model: str, api_key: str = "EMPTY",
                        *, timeout_s: float = 120.0) -> dict[str, bool]:
    """Verify each in-context option letter is a single ` X` token for this model's tokenizer.

    Preregistration requires that "Answer: A".."Answer: D" each end in exactly one token whose
    decoded text is the letter with its leading space, so the M1 margin reads four comparable
    candidates from a single logprob distribution.
    """
    import httpx

    root = _endpoint_root(base_url)
    headers = {"Authorization": "Bearer %s" % api_key}
    result: dict[str, bool] = {}
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        def tokenize(prompt: str) -> list[int]:
            response = client.post("%s/tokenize" % root, json={
                "model": model, "prompt": prompt, "add_special_tokens": False})
            response.raise_for_status()
            ids = response.json().get("tokens")
            if not isinstance(ids, list) or not ids:
                raise BackendError("tokenize returned no tokens for %r" % prompt)
            return ids

        def detokenize(ids: Sequence[int]) -> str:
            response = client.post("%s/detokenize" % root,
                                   json={"model": model, "tokens": list(ids)})
            response.raise_for_status()
            text = response.json().get("prompt")
            if not isinstance(text, str):
                raise BackendError("detokenize returned no prompt string")
            return text

        stem_ids = tokenize("Answer:")
        for letter in "ABCD":
            ids = tokenize("Answer: %s" % letter)
            single = len(ids) == len(stem_ids) + 1 and ids[: len(stem_ids)] == stem_ids
            # Detokenize rather than reading token_strs: /tokenize returns raw vocabulary
            # pieces ("▁A" for SentencePiece), while chat completions report the decoded
            # text (" A") that the M1 extractor matches against.
            result[letter] = bool(single) and detokenize(ids[-1:]) == " %s" % letter
    return result
