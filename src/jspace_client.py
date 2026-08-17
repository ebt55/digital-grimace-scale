"""Local driver for the Phase 3 j-space Modal app (`src/jspace_modal.py`).

Nothing here touches a GPU: it chunks payloads, calls the deployed Modal class, concatenates
the pieces, persists activations as `.npz`, and converts the returned token dicts into
`records.Token` so `src.metrics.m1_margin` can read a steered generation exactly like a vLLM one.

Deploy the app once (PowerShell, from the repo root)::

    C:\\...\\.venv\\Scripts\\python.exe -m modal deploy src/jspace_modal.py

then, from Python::

    from src import jspace_client as jc
    handle = jc.get_cls()                       # deployed class, containers start on first call
    result = jc.extract_activations(items, None, handle=handle)
    jc.save_npz("results/phase3/activations_discovery.npz", result)

and stop the app when the batch is done -- GPU time is the only real cost in this project::

    C:\\...\\.venv\\Scripts\\python.exe -m modal app stop dgs-jspace-gemma-2-9b-it --yes

Layer indices are hidden-state indices throughout, exactly as `jspace_modal` documents: 0 is the
embedding output, i in 1..41 is the output of decoder block i - 1, 42 is the final normed state.
`generate_steered(layer=i)` steers the stream that `extract_activations` reports at index i.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backend import LOGPROB_FLOOR, MAX_ALTERNATIVES, normalize_alternatives
from .records import Token

# Keep in sync with src/jspace_modal.py (imported separately so tests never need `modal`).
APP_NAME = "dgs-jspace-gemma-2-9b-it"
CLASS_NAME = "JSpace"
MODEL_ID = "google/gemma-2-9b-it"
REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
HIDDEN_SIZE = 3584
NUM_LAYERS = 42
NUM_HIDDEN_STATES = NUM_LAYERS + 1

DEFAULT_EXTRACT_CHUNK = 32
DEFAULT_GENERATE_CHUNK = 10
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TOP_K = 20
NPZ_SCHEMA = "dgs-jspace-activations-v1"


class JSpaceClientError(RuntimeError):
    """Raised when a remote payload does not have the shape Phase 3 depends on."""


# -- deployment ------------------------------------------------------------------------------
def deploy_command(python: str = "python") -> str:
    """The exact deploy command; deploying is free, only invocations cost GPU time."""
    return "%s -m modal deploy src/jspace_modal.py" % python


def stop_command(python: str = "python") -> str:
    """Stop the app once a batch finishes (containers also scale to zero after 5 idle minutes)."""
    return "%s -m modal app stop %s --yes" % (python, APP_NAME)


def get_cls(app_name: str = APP_NAME, class_name: str = CLASS_NAME) -> Any:
    """A handle on the deployed class; the first remote call starts the container."""
    import modal  # imported lazily so offline analysis and tests never need it

    return modal.Cls.from_name(app_name, class_name)()


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise JSpaceClientError("chunk size must be a positive integer")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        raise JSpaceClientError("items must be a nonempty sequence")
    return [items[start:start + size] for start in range(0, len(items), size)]


# -- activations -----------------------------------------------------------------------------
def merge_activation_chunks(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Concatenate per-chunk activation payloads into one, checking every chunk agrees.

    The per-layer norms are means, so they combine as an item-count weighted average -- not a
    mean of means, which would misweight a short final chunk.
    """
    if not results:
        raise JSpaceClientError("no activation chunks to merge")
    layers = [int(value) for value in results[0]["layers"]]
    ids: list[str] = []
    blocks: list[np.ndarray] = []
    prompt_tokens: list[int] = []
    weighted = np.zeros(len(layers), dtype=np.float64)
    counted = 0
    peak = 0.0
    overflow = 0
    seconds = 0.0
    for index, result in enumerate(results):
        if [int(value) for value in result["layers"]] != layers:
            raise JSpaceClientError("chunk %d returned layers %s, expected %s"
                                    % (index, list(result["layers"]), layers))
        for field, expected in (("model_id", MODEL_ID), ("revision", REVISION)):
            actual = result.get(field)
            if actual is not None and actual != expected:
                raise JSpaceClientError("chunk %d reports %s=%r, expected %r"
                                        % (index, field, actual, expected))
        block = np.asarray(result["activations"])
        chunk_ids = [str(value) for value in result["ids"]]
        if block.ndim != 3 or block.shape[0] != len(chunk_ids) or block.shape[1] != len(layers):
            raise JSpaceClientError("chunk %d has activations of shape %s for %d ids and %d layers"
                                    % (index, block.shape, len(chunk_ids), len(layers)))
        if blocks and block.shape[2] != blocks[0].shape[2]:
            raise JSpaceClientError("chunk %d has hidden size %d, expected %d"
                                    % (index, block.shape[2], blocks[0].shape[2]))
        norms = np.asarray(result["norms"], dtype=np.float64)
        if norms.shape != (len(layers),):
            raise JSpaceClientError("chunk %d returned %d norms for %d layers"
                                    % (index, norms.size, len(layers)))
        ids.extend(chunk_ids)
        blocks.append(block)
        prompt_tokens.extend(int(value) for value in result.get("prompt_tokens", ()))
        weighted += norms * len(chunk_ids)
        counted += len(chunk_ids)
        peak = max(peak, float(result.get("peak_abs_activation", 0.0)))
        overflow += int(result.get("float16_overflow", 0))
        seconds += float(result.get("seconds", 0.0))
    if len(set(ids)) != len(ids):
        raise JSpaceClientError("duplicate item ids across chunks")
    activations = np.concatenate(blocks, axis=0)
    return {
        "ids": ids,
        "layers": layers,
        "activations": activations,
        "norms": (weighted / counted).tolist(),
        "prompt_tokens": prompt_tokens,
        "hidden_size": int(activations.shape[2]),
        "model_id": MODEL_ID,
        "revision": REVISION,
        "peak_abs_activation": peak,
        "float16_overflow": overflow,
        "remote_seconds": seconds,
    }


def extract_activations(items: Sequence[Mapping[str, Any]], layers: Sequence[int] | None = None,
                        *, chunk: int = DEFAULT_EXTRACT_CHUNK, batch_size: int | None = None,
                        handle: Any = None,
                        remote: Callable[..., Mapping[str, Any]] | None = None,
                        progress: bool = True) -> dict[str, Any]:
    """Residual state at the final prompt token for every item, in chunks of `chunk`.

    `layers=None` means all 43 hidden states. `chunk` is how many items travel in one Modal
    call; `batch_size` (default: the app's own 8) is how many share one forward pass. Returns
    the merged payload: `ids`, `layers`, `activations` (float16, `(n_items, n_layers, hidden)`),
    per-layer mean `norms`, and metadata.
    """
    call = remote if remote is not None else (handle or get_cls()).extract_activations.remote
    selected = None if layers is None else [int(value) for value in layers]
    sizing = {} if batch_size is None else {"batch_size": int(batch_size)}
    batches = _chunks(list(items), chunk)
    results = []
    started = time.time()
    for index, batch in enumerate(batches, start=1):
        tick = time.time()
        results.append(call(list(batch), selected, **sizing))
        _progress(progress, "  activations chunk %d/%d (%d items) %.1fs"
                  % (index, len(batches), len(batch), time.time() - tick))
    merged = merge_activation_chunks(results)
    merged["wall_seconds"] = time.time() - started
    _progress(progress, "  activations: %s in %.1fs"
              % (merged["activations"].shape, merged["wall_seconds"]))
    return merged


# -- steered generation ----------------------------------------------------------------------
def generate_steered(items: Sequence[Mapping[str, Any]], layer: int,
                     direction: Sequence[float] | None, alphas: Sequence[float],
                     *, chunk: int = DEFAULT_GENERATE_CHUNK,
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                     top_k_logprobs: int = DEFAULT_TOP_K, batch_size: int | None = None,
                     handle: Any = None,
                     remote: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
                     progress: bool = True) -> list[dict[str, Any]]:
    """Greedy generation at each dose in `alphas`, in chunks of `chunk` items.

    `direction` is used as given: scaling it by `1/||d||` and the layer's mean activation norm
    is the caller's decision, and the preregistration's dose is defined on that scaled vector.
    `direction=None` requests the unsteered baseline at every dose. `batch_size` (default: the
    app's own 4) sets how many items share one `generate` call; throughput scales with it.
    """
    call = remote if remote is not None else (handle or get_cls()).generate_steered.remote
    sizing = {} if batch_size is None else {"batch_size": int(batch_size)}
    doses = [float(alpha) for alpha in alphas]
    if not doses:
        raise JSpaceClientError("alphas must be a nonempty sequence")
    vector = None
    if direction is not None:
        vector = [float(value) for value in np.asarray(direction, dtype=np.float64).ravel()]
        if len(vector) != HIDDEN_SIZE:
            raise JSpaceClientError("direction has %d entries, expected %d"
                                    % (len(vector), HIDDEN_SIZE))
        if not all(math.isfinite(value) for value in vector):
            raise JSpaceClientError("direction contains a non-finite entry")
    batches = _chunks(list(items), chunk)
    entries: list[dict[str, Any]] = []
    started = time.time()
    for index, batch in enumerate(batches, start=1):
        tick = time.time()
        returned = call(list(batch), int(layer), vector, doses, int(max_new_tokens),
                        int(top_k_logprobs), **sizing)
        expected = len(batch) * len(doses)
        if not isinstance(returned, Sequence) or len(returned) != expected:
            raise JSpaceClientError("chunk %d returned %s entries, expected %d"
                                    % (index, "no" if returned is None else len(returned), expected))
        wanted = {str(item["id"]) for item in batch}
        for entry in returned:
            if str(entry.get("id")) not in wanted:
                raise JSpaceClientError("chunk %d returned an unexpected id %r"
                                        % (index, entry.get("id")))
            entries.append(dict(entry))
        produced = sum(int(entry.get("generated_tokens", 0)) for entry in returned)
        elapsed = time.time() - tick
        _progress(progress, "  steering chunk %d/%d (%d items x %d alphas) %.1fs, %d tokens (%.1f tok/s)"
                  % (index, len(batches), len(batch), len(doses), elapsed, produced,
                     produced / max(elapsed, 1e-9)))
    _progress(progress, "  steering: %d generations in %.1fs" % (len(entries), time.time() - started))
    return entries


# -- persistence -----------------------------------------------------------------------------
def save_npz(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Persist a merged activation payload; metadata rides along as a JSON blob."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    activations = np.asarray(result["activations"])
    ids = [str(value) for value in result["ids"]]
    layers = [int(value) for value in result["layers"]]
    if activations.ndim != 3 or activations.shape[0] != len(ids) or activations.shape[1] != len(layers):
        raise JSpaceClientError("activations of shape %s do not match %d ids and %d layers"
                                % (activations.shape, len(ids), len(layers)))
    metadata = {key: value for key, value in result.items()
                if key not in {"ids", "layers", "activations", "norms"}}
    metadata["schema"] = NPZ_SCHEMA
    np.savez_compressed(
        destination,
        ids=np.asarray(ids, dtype=np.str_),
        layers=np.asarray(layers, dtype=np.int64),
        activations=activations,
        norms=np.asarray(result.get("norms", []), dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
    )
    return destination


def load_npz(path: str | Path) -> dict[str, Any]:
    """Inverse of `save_npz`; never unpickles (every stored array is a plain dtype)."""
    with np.load(Path(path), allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        result = {
            "ids": [str(value) for value in stored["ids"].tolist()],
            "layers": [int(value) for value in stored["layers"].tolist()],
            "activations": stored["activations"],
            "norms": [float(value) for value in stored["norms"].tolist()],
        }
    result.update(metadata)
    return result


# -- token conversion ------------------------------------------------------------------------
def _clean_logprob(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JSpaceClientError("%s must be a number, got %r" % (field, value))
    number = float(value)
    if math.isnan(number):
        raise JSpaceClientError("%s must not be NaN" % field)
    if math.isinf(number):
        if number > 0:
            raise JSpaceClientError("%s must not be +inf" % field)
        return LOGPROB_FLOOR
    return min(0.0, number)


def to_tokens(entry: Mapping[str, Any]) -> tuple[Token, ...]:
    """Convert one steered-generation entry into the `records.Token` trace it stands for.

    Alternatives go through `backend.normalize_alternatives`, so duplicate texts are merged by
    log-sum-exp and the trace can never carry more than twenty of them or lose the sampled
    token -- the same normalisation every vLLM record went through. The concatenated token text
    must equal `entry["text"]`, which is the invariant `records.record_from_dict` enforces.
    """
    if not isinstance(entry, Mapping):
        raise JSpaceClientError("entry must be a mapping")
    text = entry.get("text")
    if not isinstance(text, str):
        raise JSpaceClientError("entry has no text")
    positions = entry.get("tokens")
    if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)) or not positions:
        raise JSpaceClientError("entry %r has an empty token trace" % entry.get("id"))
    tokens: list[Token] = []
    for index, position in enumerate(positions):
        if not isinstance(position, Mapping) or not isinstance(position.get("text"), str):
            raise JSpaceClientError("position %d has no token text" % index)
        logprob = _clean_logprob(position.get("logprob"), "token logprob")
        raw: list[tuple[str, float]] = []
        for alternative in position.get("top_logprobs") or ():
            if not isinstance(alternative, Mapping) or not isinstance(alternative.get("text"), str):
                raise JSpaceClientError("position %d has an invalid alternative" % index)
            raw.append((alternative["text"], _clean_logprob(alternative.get("logprob"), "top_logprob")))
        alternatives = normalize_alternatives(position["text"], logprob, raw,
                                              limit=MAX_ALTERNATIVES)
        tokens.append(Token(position["text"], logprob, alternatives))
    joined = "".join(token.text for token in tokens)
    if joined != text:
        raise JSpaceClientError("entry %r text does not match its token trace (%r vs %r)"
                                % (entry.get("id"), text[:80], joined[:80]))
    return tuple(tokens)


def token_dicts(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """`to_tokens` in the JSON shape `records.record_from_dict` expects under "tokens"."""
    return [{"text": token.text, "logprob": token.logprob,
             "top_logprobs": [{"text": text, "logprob": logprob}
                              for text, logprob in token.top_logprobs]}
            for token in to_tokens(entry)]
