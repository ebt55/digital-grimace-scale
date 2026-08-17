"""Modal app for Phase 3 (j-space): residual-stream activations and steered generation.

Phase 3 needs two things vLLM cannot give us: the hidden state at the final prompt token for
every layer, and greedy generation with a fixed vector added to one layer's residual stream.
Both are one `transformers` forward hook away, so this app runs the pinned `google/gemma-2-9b-it`
weights directly in torch rather than behind an OpenAI-compatible server.

Deploy (PowerShell, from the repo root)::

    C:\\...\\.venv\\Scripts\\python.exe -m modal deploy src/jspace_modal.py

Smoke it (against the *deployed* app, so no second container is paid for)::

    C:\\...\\.venv\\Scripts\\python.exe -m modal run src/jspace_modal.py::smoke --deployed

Stop it when a batch finishes (idle containers also scale to zero after `scaledown_window`,
but GPU time is the only real cost in this project)::

    C:\\...\\.venv\\Scripts\\python.exe -m modal app stop dgs-jspace-gemma-2-9b-it --yes

Conventions that the analysis depends on, all of them deliberate:

*Layer indices are hidden-state indices.* `extract_activations` returns `hidden_states[i]` for
each requested `i`, exactly as `transformers` numbers them: 0 is the (scaled) embedding output,
`i` in 1..41 is the output of decoder block `i - 1`, and 42 is the final RMS-normed state.
`generate_steered(layer=i)` uses **the same numbering**: it hooks decoder block `i - 1`, so the
added vector first appears at `hidden_states[i]` -- the very stream the direction was measured
in. Passing the probe's chosen layer straight through is therefore correct, with no off-by-one.

*Eager attention.* gemma-2 softcaps both attention and final logits; `attn_implementation="eager"`
is the only implementation that applies the attention softcap exactly.

*Stop tokens.* This revision's `generation_config.json` lists only `<eos>` (id 1), but the
instruction-tuned model ends its turn with `<end_of_turn>` (id 107). Both are treated as EOS,
otherwise every greedy generation would run to `max_new_tokens` and emit a fresh `<start_of_turn>`.

*Token traces.* Per-position dicts mirror `src/backend.py` exactly -- `text`, `logprob`,
top-20 `top_logprobs` with duplicate texts merged by log-sum-exp, trailing EOS dropped, an
immediately-terminated turn kept as one zero-width position -- so `src.jspace_client.to_tokens`
turns them into `records.Token` and `src.metrics.m1_margin` reads them unchanged.

The Hugging Face token is read from the local login at deploy time and passed as an anonymous
per-deployment secret. It is never printed, never written to a named Modal secret, never committed.
"""
from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import modal

MINUTES = 60

MODEL_ID = "google/gemma-2-9b-it"
# Pinned in manifest.json; the same weights Phases 0-2 were generated from.
REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
APP_NAME = "dgs-jspace-gemma-2-9b-it"
CLASS_NAME = "JSpace"
# L40S (48 GB) holds bf16 weights plus all-layer hidden states comfortably; A100-40GB is the
# fallback Modal falls back to when no L40S is free.
GPU = ["L40S", "A100-40GB"]
HF_CACHE = "/root/.cache/huggingface"

# Asserted against the loaded config at container start, so a silent model swap cannot pass.
HIDDEN_SIZE = 3584
NUM_LAYERS = 42
MAX_PROMPT_TOKENS = 8192  # gemma-2 max_position_embeddings

DEFAULT_TOP_K = 20
DEFAULT_EXTRACT_BATCH = 8
DEFAULT_GENERATE_BATCH = 4
DEFAULT_MAX_NEW_TOKENS = 512
# Mirrors src/backend.py: the floor substituted for a non-finite log probability.
LOGPROB_FLOOR = -9999.0
FLOAT16_MAX = 65504.0
TURN_END_TOKEN = "<end_of_turn>"

# transformers 4.57.x is pinned deliberately over the 5.x line: 4.x builds the hidden-state
# tuple inside `Gemma2Model.forward`, so `hidden_states[i]` has the documented meaning this
# file's whole layer convention rests on. 5.x moved that plumbing into a capture decorator whose
# per-layer semantics we could only confirm by burning GPU time.
TORCH_VERSION = "2.9.1"
TRANSFORMERS_VERSION = "4.57.6"
ACCELERATE_VERSION = "1.14.0"
NUMPY_VERSION = "2.4.6"


def _hugging_face_token() -> str:
    """Read the local Hugging Face login without ever surfacing its value.

    Modal re-imports this module inside the container, where there is no local login -- but the
    secret has already been injected as HF_TOKEN by then, so the environment read wins. The
    hard failure is reserved for the deploy pass, where a missing token means gated gemma-2
    weights would never download.
    """
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        try:
            from huggingface_hub import get_token

            token = (get_token() or "").strip()
        except ImportError:
            token = ""
    if not token and not os.environ.get("MODAL_TASK_ID"):
        raise RuntimeError(
            "no Hugging Face token found. Log in once with `huggingface-cli login` (the token "
            "lands in %USERPROFILE%\\.cache\\huggingface\\token) or set HF_TOKEN in this shell. "
            "gemma-2 weights are gated and will not download without it.")
    return token


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==%s" % TORCH_VERSION,
        "transformers==%s" % TRANSFORMERS_VERSION,
        "accelerate==%s" % ACCELERATE_VERSION,
        "numpy==%s" % NUMPY_VERSION,
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_CACHE": "%s/hub" % HF_CACHE,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Provenance, readable from `modal shell` / container logs.
        "DGS_JSPACE_MODEL_ID": MODEL_ID,
        "DGS_JSPACE_REVISION": REVISION,
    })
)

# The same persistent weight cache the vLLM apps fill: gemma-2-9b-it is already in it.
hf_cache = modal.Volume.from_name("dgs-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


# -- payload validation (pure, runs container-side) --------------------------------------------
def _validated_items(items: Any) -> list[dict[str, Any]]:
    """Accept only `{"id": str, "messages": [{"role": str, "content": str}, ...]}` items."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or not items:
        raise ValueError("items must be a nonempty sequence of dicts")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError("item %d must be a mapping" % index)
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("item %d has no id" % index)
        if identifier in seen:
            raise ValueError("duplicate item id %r" % identifier)
        seen.add(identifier)
        messages = item.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
            raise ValueError("item %r has no messages" % identifier)
        cleaned = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("item %r has a non-mapping message" % identifier)
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("item %r has a message without string role/content" % identifier)
            if role == "system":
                raise ValueError("item %r uses a system role, which gemma-2's template rejects"
                                 % identifier)
            cleaned.append({"role": role, "content": content})
        parsed.append({"id": identifier, "messages": cleaned})
    return parsed


def _validated_layers(layers: Any, num_layers: int) -> list[int]:
    """`None` means every hidden state, 0 (embeddings) through `num_layers` (final norm)."""
    if layers is None:
        return list(range(num_layers + 1))
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers:
        raise ValueError("layers must be None or a nonempty sequence of ints")
    result = []
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("layer indices must be integers, got %r" % (layer,))
        if not 0 <= layer <= num_layers:
            raise ValueError("layer %d is outside 0..%d" % (layer, num_layers))
        result.append(int(layer))
    if len(set(result)) != len(result):
        raise ValueError("layer indices must be distinct")
    return result


def _logaddexp(first: float, second: float) -> float:
    """Numerically stable log(exp(first) + exp(second)); mirrors src/backend.py."""
    import math

    high, low = (first, second) if first >= second else (second, first)
    if low == float("-inf"):
        return high
    return high + math.log1p(math.exp(low - high))


def _steering_hook(delta: Any):
    """Add a fixed vector to a decoder block's residual output at every position."""

    def hook(module: Any, inputs: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            if not output:
                return output
            return (output[0] + delta,) + tuple(output[1:])
        return output + delta

    return hook


@app.cls(
    image=image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hugging_face_token()})],
    timeout=2 * 60 * MINUTES,
    scaledown_window=5 * MINUTES,
    startup_timeout=15 * MINUTES,
    min_containers=0,
    max_containers=1,
)
class JSpace:
    """One warm gemma-2-9b-it in bf16, with eager attention and the model's own chat template."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.time()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        # LEFT padding, so the final prompt token is the last position in every batched row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.bfloat16,
            attn_implementation="eager",  # gemma-2 attention softcapping is exact only here
            device_map="cuda:0",
        )
        self.model.eval()
        if self.model.dtype != torch.bfloat16:
            raise RuntimeError("model loaded as %s, expected bfloat16" % self.model.dtype)
        if self.model.config._attn_implementation != "eager":
            raise RuntimeError("attn_implementation is %r, expected 'eager'"
                               % self.model.config._attn_implementation)
        self.device = next(self.model.parameters()).device
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        if (self.hidden_size, self.num_layers) != (HIDDEN_SIZE, NUM_LAYERS):
            raise RuntimeError("loaded a model with hidden_size=%d num_layers=%d, expected %d/%d"
                               % (self.hidden_size, self.num_layers, HIDDEN_SIZE, NUM_LAYERS))
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_ids = self._resolve_eos_ids()
        self._decoded: dict[int, str] = {}
        self.load_seconds = time.time() - started
        print("loaded %s @ %s in %.1fs on %s (hidden=%d layers=%d eos=%s)"
              % (MODEL_ID, REVISION[:12], self.load_seconds, self.device, self.hidden_size,
                 self.num_layers, sorted(self.eos_ids)))

    # -- helpers -------------------------------------------------------------------------------
    def _resolve_eos_ids(self) -> tuple[int, ...]:
        configured = getattr(self.model.generation_config, "eos_token_id", None)
        if configured is None:
            configured = []
        elif isinstance(configured, int):
            configured = [configured]
        ids = {int(value) for value in configured}
        turn_end = self.tokenizer.convert_tokens_to_ids(TURN_END_TOKEN)
        if isinstance(turn_end, int) and turn_end >= 0 and turn_end != self.tokenizer.unk_token_id:
            ids.add(int(turn_end))
        if not ids:
            raise RuntimeError("no EOS token ids resolved for %s" % MODEL_ID)
        return tuple(sorted(ids))

    def _decode_id(self, token_id: int) -> str:
        text = self._decoded.get(token_id)
        if text is None:
            text = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            self._decoded[token_id] = text
        return text

    def _decoder_layers(self) -> Any:
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            decoder = self.model.get_decoder()
            layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("could not locate the decoder layer list on %s" % type(self.model))
        return layers

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        """The model's own chat template, with the generation prompt appended."""
        return self.tokenizer.apply_chat_template(
            [dict(message) for message in messages], tokenize=False, add_generation_prompt=True)

    def _encode(self, texts: Sequence[str], ids: Sequence[str]) -> tuple[Any, Any]:
        # add_special_tokens=False: gemma-2's template already emits <bos>, and a second one
        # would silently shift every activation.
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                                 add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError("prompt of %d tokens exceeds %d (items %s)"
                             % (input_ids.shape[1], MAX_PROMPT_TOKENS, list(ids)))
        return input_ids.to(self.device), encoded["attention_mask"].to(self.device)

    def _final_token_states(self, texts: Sequence[str], ids: Sequence[str],
                            layers: Sequence[int]) -> tuple[Any, list[int]]:
        """Hidden state at the final prompt token, (n, len(layers), hidden) float32 on the CPU.

        Splits the batch in half and retries on CUDA OOM, so one long prompt cannot lose a run.
        """
        torch = self.torch
        try:
            input_ids, attention_mask = self._encode(texts, ids)
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states
            if hidden is None or len(hidden) != self.num_layers + 1:
                raise RuntimeError("expected %d hidden states, got %s"
                                   % (self.num_layers + 1, None if hidden is None else len(hidden)))
            # LEFT padding, so position -1 is the final real prompt token for every row.
            stacked = torch.stack([hidden[index][:, -1, :].float() for index in layers], dim=1)
            result = stacked.cpu()
            counts = [int(value) for value in attention_mask.sum(dim=1).tolist()]
            del outputs, hidden, stacked, input_ids, attention_mask
            return result, counts
        except torch.cuda.OutOfMemoryError:
            if len(texts) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(texts) // 2
            first, first_counts = self._final_token_states(texts[:half], ids[:half], layers)
            second, second_counts = self._final_token_states(texts[half:], ids[half:], layers)
            return torch.cat([first, second], dim=0), first_counts + second_counts

    def _token_dict(self, token_id: int, log_probabilities: Any, top_k: int) -> dict[str, Any]:
        """One position of the trace, shaped exactly like src/backend.py's Token payload."""
        torch = self.torch
        values, indices = torch.topk(log_probabilities, k=min(top_k, log_probabilities.numel()))
        merged: dict[str, float] = {}
        for value, index in zip(values.tolist(), indices.tolist()):
            text = self._decode_id(int(index))
            merged[text] = _logaddexp(merged[text], float(value)) if text in merged else float(value)
        chosen_text = self._decode_id(int(token_id))
        chosen_logprob = float(log_probabilities[int(token_id)])
        if chosen_text not in merged:
            merged[chosen_text] = chosen_logprob
        ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        kept = ranked[:top_k]
        if all(text != chosen_text for text, _ in kept):
            kept[-1] = (chosen_text, merged[chosen_text])
            kept.sort(key=lambda item: (-item[1], item[0]))
        return {
            "text": chosen_text,
            "logprob": min(0.0, chosen_logprob),
            "top_logprobs": [{"text": text, "logprob": min(0.0, value)} for text, value in kept],
        }

    def _log_probabilities(self, row: Any) -> tuple[Any, bool]:
        """log_softmax of one step's processed scores, forced finite and non-positive."""
        torch = self.torch
        scores = row.float()
        clean = bool(torch.isfinite(scores).all())
        values = torch.log_softmax(scores, dim=-1)
        values = torch.nan_to_num(values, nan=LOGPROB_FLOOR, posinf=0.0, neginf=LOGPROB_FLOOR)
        return values.clamp(max=0.0), clean

    # -- remote methods ------------------------------------------------------------------------
    @modal.method()
    def info(self) -> dict[str, Any]:
        """Cheap warm-up ping that also reports what the container actually loaded."""
        return {
            "model_id": MODEL_ID, "revision": REVISION, "hidden_size": self.hidden_size,
            "num_layers": self.num_layers, "num_hidden_states": self.num_layers + 1,
            "eos_token_ids": list(self.eos_ids), "pad_token_id": self.pad_token_id,
            "device": str(self.device), "dtype": str(self.model.dtype),
            "attn_implementation": self.model.config._attn_implementation,
            "transformers": TRANSFORMERS_VERSION, "torch": TORCH_VERSION,
            "load_seconds": self.load_seconds,
        }

    @modal.method()
    def extract_activations(self, items: list[dict], layers: list[int] | None = None,
                            batch_size: int = DEFAULT_EXTRACT_BATCH) -> dict[str, Any]:
        """Residual-stream state at the final prompt token, for every requested layer.

        `layers=None` means all of them: index 0 is the embedding output (`hidden_states[0]`),
        index i in 1..41 is the output of decoder block i - 1, index 42 is the final normed
        state. Returns float16 activations of shape (n_items, len(layers), hidden_size) plus the
        per-layer mean L2 norm (computed in float32, before the cast).
        """
        import numpy as np

        started = time.time()
        parsed = _validated_items(items)
        chosen = _validated_layers(layers, self.num_layers)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        ids = [item["id"] for item in parsed]
        texts = [self._render(item["messages"]) for item in parsed]

        blocks, prompt_tokens = [], []
        for start in range(0, len(texts), batch_size):
            stop = start + batch_size
            block, counts = self._final_token_states(texts[start:stop], ids[start:stop], chosen)
            blocks.append(block)
            prompt_tokens.extend(counts)
        states = self.torch.cat(blocks, dim=0)  # (n, len(chosen), hidden) float32, CPU

        norms = states.norm(dim=-1)  # (n, len(chosen))
        peak = float(states.abs().max())
        overflow = int((states.abs() > FLOAT16_MAX).sum())
        activations = states.clamp(-FLOAT16_MAX, FLOAT16_MAX).to(self.torch.float16).numpy()
        seconds = time.time() - started
        return {
            "ids": ids,
            "layers": chosen,
            "activations": np.ascontiguousarray(activations),
            "norms": [float(value) for value in norms.mean(dim=0).tolist()],
            "prompt_tokens": prompt_tokens,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "peak_abs_activation": peak,
            "float16_overflow": overflow,
            "seconds": seconds,
        }

    @modal.method()
    def generate_steered(self, items: list[dict], layer: int, direction: list[float] | None,
                         alphas: list[float], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                         top_k_logprobs: int = DEFAULT_TOP_K,
                         batch_size: int = DEFAULT_GENERATE_BATCH) -> list[dict]:
        """Greedy generation with `alpha * direction` added to one layer's residual stream.

        `layer` is a hidden-state index in the same space `extract_activations` uses, so the
        vector first appears exactly where it was measured; decoder block `layer - 1` carries
        the hook, and the addition applies at every position, prompt and generated alike.
        `direction=None` (or `alpha == 0`) is the unsteered baseline. `direction` is used as
        given -- scaling by ||d|| and the activation norm is the caller's job.
        """
        import copy

        torch = self.torch
        parsed = _validated_items(items)
        if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= self.num_layers:
            raise ValueError("layer must be a hidden-state index in 1..%d" % self.num_layers)
        if not isinstance(alphas, Sequence) or isinstance(alphas, (str, bytes)) or not alphas:
            raise ValueError("alphas must be a nonempty sequence")
        doses = [float(alpha) for alpha in alphas]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if isinstance(top_k_logprobs, bool) or not isinstance(top_k_logprobs, int) or not 1 <= top_k_logprobs <= 20:
            raise ValueError("top_k_logprobs must be an integer in 1..20")
        vector = None
        if direction is not None:
            vector = torch.tensor([float(value) for value in direction], dtype=torch.float32,
                                  device=self.device)
            if vector.numel() != self.hidden_size:
                raise ValueError("direction has %d entries, expected %d"
                                 % (vector.numel(), self.hidden_size))
            if not bool(torch.isfinite(vector).all()):
                raise ValueError("direction contains a non-finite entry")

        # Greedy, and only greedy: copying the model's generation config keeps gemma-2's hybrid
        # cache and pad/eos wiring while stripping every sampling parameter it ships with.
        settings = copy.deepcopy(self.model.generation_config)
        settings._from_model_config = False  # an explicit config, not a legacy model-config echo
        settings.do_sample = False
        settings.num_beams = 1
        settings.temperature = None
        settings.top_p = None
        settings.top_k = None
        settings.max_new_tokens = max_new_tokens
        settings.eos_token_id = list(self.eos_ids)
        settings.pad_token_id = self.pad_token_id
        settings.return_dict_in_generate = True
        settings.output_scores = True
        # gemma-2 ships `cache_implementation="hybrid"`, which routes generate onto a static
        # cache and torch.compile. transformers 4.57 unsets that default itself (the dynamic
        # sliding-window cache replaced it), but only on the model's own config -- this copy
        # keeps it. Left in place it costs ~5 minutes of recompilation *per steering hook*,
        # because every dose installs a fresh closure. Unset it: dynamic cache, no compile.
        settings.cache_implementation = None

        block = self._decoder_layers()[layer - 1]
        order = {item["id"]: index for index, item in enumerate(parsed)}
        results: list[dict[str, Any]] = []
        for alpha in doses:
            steered = vector is not None and alpha != 0.0
            handle = None
            if steered:
                delta = (vector * alpha).to(self.model.dtype)
                handle = block.register_forward_hook(_steering_hook(delta))
            try:
                for start in range(0, len(parsed), batch_size):
                    batch = parsed[start:start + batch_size]
                    results.extend(self._generate_batch(batch, alpha, layer, steered, settings,
                                                        top_k_logprobs))
            finally:
                if handle is not None:
                    handle.remove()
        results.sort(key=lambda entry: (order[entry["id"]], entry["alpha"]))
        return results

    def _generate_batch(self, batch: Sequence[Mapping[str, Any]], alpha: float, layer: int,
                        steered: bool, settings: Any, top_k: int) -> list[dict[str, Any]]:
        torch = self.torch
        started = time.time()
        ids = [item["id"] for item in batch]
        texts = [self._render(item["messages"]) for item in batch]
        input_ids, attention_mask = self._encode(texts, ids)
        prompt_length = int(input_ids.shape[1])
        with torch.inference_mode():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          generation_config=settings)
        generated = outputs.sequences[:, prompt_length:]
        scores = outputs.scores
        seconds = time.time() - started

        entries: list[dict[str, Any]] = []
        for row, item in enumerate(batch):
            tokens: list[dict[str, Any]] = []
            finish = "length"
            nonfinite = 0
            steps = 0
            terminator: dict[str, Any] | None = None
            for step in range(len(scores)):
                token_id = int(generated[row, step])
                values, clean = self._log_probabilities(scores[step][row])
                if not clean:
                    nonfinite += 1
                position = self._token_dict(token_id, values, top_k)
                steps += 1
                if token_id in self.eos_ids:
                    finish, terminator = "eos", position
                    break
                tokens.append(position)
            text = "".join(position["text"] for position in tokens)
            if not tokens:
                # A turn that terminated immediately still occupies one generated position, and
                # its distribution says how sure the model was about stopping (cf. backend.py).
                empty = {"text": "", "logprob": 0.0, "top_logprobs": [{"text": "", "logprob": 0.0}]}
                if terminator is not None:
                    empty = {"text": "", "logprob": terminator["logprob"],
                             "top_logprobs": terminator["top_logprobs"]}
                tokens = [empty]
            entries.append({
                "id": item["id"],
                "alpha": alpha,
                "layer": layer,
                "decoder_layer": layer - 1,
                "steered": steered,
                "text": text,
                "tokens": tokens,
                "finish": finish,
                "prompt_tokens": prompt_length,
                "generated_tokens": steps,  # decoding steps consumed, EOS included
                "nonfinite_steps": nonfinite,
                "batch_seconds": seconds,
                "batch_size": len(batch),
            })
        del outputs, scores, generated, input_ids, attention_mask
        torch.cuda.empty_cache()
        return entries


@app.local_entrypoint()
def smoke(deployed: bool = False, max_new_tokens: int = 96) -> None:
    """Both methods on two tiny prompts; prints shapes, layer norms and timings.

    `--deployed` runs against the already-deployed app instead of an ephemeral copy, so the
    smoke costs one container start rather than two.
    """
    import numpy as np

    handle = modal.Cls.from_name(APP_NAME, CLASS_NAME)() if deployed else JSpace()
    items = [
        {"id": "smoke-1", "messages": [{"role": "user", "content": "Say hello in one short sentence."}]},
        {"id": "smoke-2", "messages": [{"role": "user", "content": "Name one primary colour."}]},
    ]

    started = time.time()
    details = handle.info.remote()
    print("info (%.1fs): %s" % (time.time() - started, details))

    started = time.time()
    activations = handle.extract_activations.remote(items, None)
    elapsed = time.time() - started
    array = activations["activations"]
    print("extract_activations: shape=%s dtype=%s layers=%d ids=%s"
          % (array.shape, array.dtype, len(activations["layers"]), activations["ids"]))
    print("   %.2fs wall (%.2fs remote, %.2fs/item), prompt tokens %s, peak |a| %.1f, fp16 overflow %d"
          % (elapsed, activations["seconds"], elapsed / len(items), activations["prompt_tokens"],
             activations["peak_abs_activation"], activations["float16_overflow"]))
    norms = activations["norms"]
    print("   mean L2 norm: layer0=%.1f layer21=%.1f layer42=%.1f" % (norms[0], norms[21], norms[42]))

    layer = 21
    generator = np.random.default_rng(0)
    raw = generator.normal(size=activations["hidden_size"])
    direction = (raw / np.linalg.norm(raw) * norms[layer]).tolist()
    zero = [0.0] * activations["hidden_size"]

    for label, vector in (("random", direction), ("zero", zero)):
        started = time.time()
        generations = handle.generate_steered.remote(items, layer, vector, [0.0, 2.0],
                                                     max_new_tokens, 20, 2)
        elapsed = time.time() - started
        produced = sum(entry["generated_tokens"] for entry in generations)
        print("generate_steered[%s]: %d entries, %d tokens, %.1fs (%.1f tok/s)"
              % (label, len(generations), produced, elapsed, produced / max(elapsed, 1e-9)))
        for entry in generations:
            preview = entry["text"][:110].replace("\n", "\\n")
            print("   %s alpha=%.1f finish=%s tokens=%d nonfinite=%d | %s"
                  % (entry["id"], entry["alpha"], entry["finish"], len(entry["tokens"]),
                     entry["nonfinite_steps"], preview))
            first = entry["tokens"][0]
            print("      first position: text=%r logprob=%.3f alternatives=%d"
                  % (first["text"], first["logprob"], len(first["top_logprobs"])))
