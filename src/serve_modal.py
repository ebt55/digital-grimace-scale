"""Modal app serving one vLLM OpenAI-compatible endpoint per experiment model.

The roadmap's generation stack is vLLM on Modal with `--max-logprobs 20` behind an
OpenAI-compatible client (see `src/backend.py`). One deployable app per model, selected by
environment variables read at deploy time, so the app name and served model always agree.

Deploy (PowerShell, from the repo root)::

    $env:DGS_MODEL_ID='google/gemma-2-2b-it'
    C:\\...\\.venv\\Scripts\\python.exe -m modal deploy src/serve_modal.py
    # optional: $env:DGS_REVISION='<40-hex sha>'   pins the exact weights revision
    # optional: $env:DGS_GPU='L40S'                overrides the GPU heuristic
    # optional: $env:DGS_VLLM_VERSION='0.27.1'     overrides the pinned vLLM release
    # optional: $env:DGS_ATTENTION_BACKEND='FLASHINFER'  if gemma-2 logit softcapping errors

Deploy prints the web URL; the OpenAI base URL is that URL with `/v1` appended::

    OpenAICompatBackend("https://<workspace>--dgs-vllm-gemma-2-2b-it-serve.modal.run/v1",
                        "google/gemma-2-2b-it")

Check it is up (PowerShell)::

    Invoke-RestMethod "https://<...>.modal.run/v1/models"

Or run the bundled probe against an ephemeral deployment::

    C:\\...\\.venv\\Scripts\\python.exe -m modal run src/serve_modal.py::smoke

Stop it (idle GPUs also scale to zero after `scaledown_window`, but stop explicitly when a
generation batch finishes -- GPU time is the only real cost in this project)::

    C:\\...\\.venv\\Scripts\\python.exe -m modal app stop dgs-vllm-gemma-2-2b-it
    C:\\...\\.venv\\Scripts\\python.exe -m modal app list      # confirm nothing is running

GPU guidance (bf16 weights plus KV cache at `--max-model-len 8192`):

===========================  ==========  ==========================================
model size                   default     recommendation
===========================  ==========  ==========================================
<= 4B (gemma-2-2b, Qwen 3B)  A10G 24GB   A10G is comfortable and the cheapest option
7B-9B (gemma-2-9b, Qwen 7B)  L40S 48GB   L40S, or A100-40GB for more throughput
===========================  ==========  ==========================================

The Hugging Face token is read from the local login at deploy time and passed as an
anonymous per-deployment secret. It is never printed, never written to a named Modal
secret, and never committed.
"""
from __future__ import annotations

import os
import re
import subprocess

import modal

PORT = 8000
MINUTES = 60
HF_CACHE = "/root/.cache/huggingface"
MAX_LOGPROBS = 20
MAX_MODEL_LEN = 8192

MODEL_ID = (os.environ.get("DGS_MODEL_ID") or "google/gemma-2-2b-it").strip()
REVISION = (os.environ.get("DGS_REVISION") or "").strip() or None
VLLM_VERSION = (os.environ.get("DGS_VLLM_VERSION") or "0.26.0").strip()
ATTENTION_BACKEND = (os.environ.get("DGS_ATTENTION_BACKEND") or "").strip() or None


def app_name(model_id: str) -> str:
    """Derive a stable Modal app name from a Hugging Face id: dgs-vllm-gemma-2-2b-it."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    return "dgs-vllm-%s" % (slug or "model")


def parameter_billions(model_id: str) -> float | None:
    """Best-effort parameter count from the model id, used only to pick a default GPU."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", model_id.lower())
    return float(match.group(1)) if match else None


def default_gpu(model_id: str) -> str:
    size = parameter_billions(model_id)
    if size is None or size <= 4:
        return "A10G"
    if size <= 10:
        return "L40S"
    return "A100-40GB"


GPU = (os.environ.get("DGS_GPU") or "").strip() or default_gpu(MODEL_ID)
APP_NAME = app_name(MODEL_ID)


def _hugging_face_token() -> str:
    """Read the local Hugging Face login without ever surfacing its value."""
    try:
        from huggingface_hub import get_token
    except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
        raise RuntimeError("huggingface_hub is required to read the local HF token") from exc
    token = (os.environ.get("HF_TOKEN") or get_token() or "").strip()
    if not token:
        raise RuntimeError(
            "no Hugging Face token found. Log in once with `huggingface-cli login` (the token "
            "lands in %%USERPROFILE%%\\.cache\\huggingface\\token) or set HF_TOKEN in this shell. "
            "gemma-2 weights are gated and will not download without it.")
    return token


def vllm_command() -> list[str]:
    """Server flags the preregistration depends on: bf16, top-20 logprobs, fixed seed."""
    command = [
        "vllm", "serve", MODEL_ID,
        "--served-model-name", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-logprobs", str(MAX_LOGPROBS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--enable-prefix-caching",
        "--gpu-memory-utilization", "0.90",
        "--seed", "0",
        "--uvicorn-log-level", "info",
    ]
    if REVISION:
        command += ["--revision", REVISION]
    return command


image_environment = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": "%s/hub" % HF_CACHE,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_LOGGING_LEVEL": "INFO",
}
if ATTENTION_BACKEND:
    # gemma-2 needs an attention backend with logit softcapping; recent vLLM picks one itself.
    image_environment["VLLM_ATTENTION_BACKEND"] = ATTENTION_BACKEND

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==%s" % VLLM_VERSION)
    .env(image_environment)
)

# Persistent weight cache, as the roadmap prescribes: pay the download once per model.
hf_cache = modal.Volume.from_name("dgs-hf-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hugging_face_token()})],
    timeout=4 * 60 * MINUTES,
    scaledown_window=5 * MINUTES,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=200)
@modal.web_server(port=PORT, startup_timeout=15 * MINUTES)
def serve() -> None:
    command = vllm_command()
    print("launching:", " ".join(command))
    subprocess.Popen(command)


@app.local_entrypoint()
def smoke(timeout_s: int = 15 * MINUTES) -> None:
    """Wait for the server, then print /v1/models and the four letter-token checks."""
    import json
    import time
    import urllib.error
    import urllib.request

    base_url = "%s/v1" % serve.get_web_url().rstrip("/")
    print("model:   %s%s" % (MODEL_ID, " @ %s" % REVISION if REVISION else ""))
    print("gpu:     %s" % GPU)
    print("vllm:    %s" % VLLM_VERSION)
    print("base_url:%s" % base_url)
    deadline = time.time() + timeout_s
    while True:
        try:
            with urllib.request.urlopen("%s/models" % base_url, timeout=60) as response:
                print("/v1/models:", json.dumps(json.loads(response.read())))
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if time.time() >= deadline:
                raise RuntimeError("server did not become ready within %ds" % timeout_s) from exc
            time.sleep(10)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.backend import probe_letter_tokens

    print("letter tokens:", probe_letter_tokens(base_url, MODEL_ID))
