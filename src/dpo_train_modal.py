"""Modal app that QLoRA-DPO-trains gemma-2-9b-it on one Phase 4 arm and merges the adapter.

Preregistration v5 fixes the recipe and it is implemented literally here: TRL `DPOTrainer`,
4-bit NF4 base, LoRA r = 16 / alpha = 32 / dropout 0.05 on every attention and MLP projection,
beta = 0.1, lr 5e-6, 2 epochs, effective batch 8, seed 0, identical for arm A and arm B.  The
reference policy is the base model reached by disabling the adapter (`ref_model=None` with
peft), so no second copy of the weights is loaded.

Each run writes two things to the `dgs-adapters` volume:

* ``/adapters/<arm>/lora``   -- the LoRA adapter as trained,
* ``/adapters/<arm>/merged`` -- the adapter merged into bf16 base weights plus the tokenizer,
  which is what vLLM serves for the Phase 4 evaluation.

Run it through `scripts/train_dpo.py`, which passes the pairs file and stores the returned
manifest under ``results/dpo/train_<arm>.json``::

    C:\\...\\.venv\\Scripts\\python.exe scripts/train_dpo.py --arm A --pairs results/dpo/pairs_A.jsonl

`preflight` is a CPU-only entry point that imports the stack and builds the exact configs, so
an API mistake surfaces in seconds instead of after a GPU cold start::

    C:\\...\\.venv\\Scripts\\python.exe -m modal run src/dpo_train_modal.py::preflight
"""
from __future__ import annotations

import os
from typing import Any

import modal

APP_NAME = "dgs-dpo-gemma-2-9b-it"
BASE_MODEL_ID = "google/gemma-2-9b-it"
BASE_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"

GPU = os.environ.get("DGS_DPO_GPU", "A100-40GB")
MINUTES = 60
HF_CACHE = "/root/.cache/huggingface"
ADAPTER_ROOT = "/adapters"

# Preregistration v5 hyperparameters. Nothing here may differ between arm A and arm B.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")
DPO_BETA = 0.1
LEARNING_RATE = 5e-6
NUM_EPOCHS = 2.0
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4  # PER_DEVICE_BATCH * GRAD_ACCUM == effective batch 8
SEED = 0
MAX_LENGTH = 1536  # prompt (~400 tok) + a full 512-token completion, with headroom

# Current stable releases as of 2026-08-18; pinned so both arms train on identical software.
PINS = {
    "torch": "2.13.0",
    "transformers": "5.15.0",
    "trl": "1.10.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
}


def _shared_guard() -> str:
    """A shared secret both sides derive from the local Hugging Face login, never from the repo.

    The HTTPS fallback below is a public `*.modal.run` URL, so it carries a guard token: the
    deploying shell bakes this digest into a Modal secret and `scripts/train_dpo.py` recomputes
    it from the same local login at call time.  The token is a one-way digest, so it neither
    reveals nor transports the Hugging Face token itself, and nothing is written to the repo.
    """
    from hashlib import sha256

    return sha256(("DGS-PHASE4-TRAIN-GUARD|%s" % _hugging_face_token()).encode("utf-8")).hexdigest()


def _hugging_face_token() -> str:
    """Read the local Hugging Face login without ever surfacing its value (as `serve_modal` does)."""
    try:
        from huggingface_hub import get_token
    except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
        raise RuntimeError("huggingface_hub is required to read the local HF token") from exc
    token = (os.environ.get("HF_TOKEN") or get_token() or "").strip()
    if not token:
        raise RuntimeError(
            "no Hugging Face token found. Log in once with `huggingface-cli login` or set "
            "HF_TOKEN in this shell; gemma-2 weights are gated.")
    return token


train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==%s" % PINS["torch"],
        "transformers==%s" % PINS["transformers"],
        "trl==%s" % PINS["trl"],
        "peft==%s" % PINS["peft"],
        "bitsandbytes==%s" % PINS["bitsandbytes"],
        "accelerate==%s" % PINS["accelerate"],
        "datasets==%s" % PINS["datasets"],
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.1",
        "fastapi[standard]==0.122.0",
    )
    .env({
        "HF_HOME": HF_CACHE,
        "HF_HUB_CACHE": "%s/hub" % HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

# Shared with `src/serve_modal.py`: the gemma-2-9b weights are downloaded exactly once.
hf_cache = modal.Volume.from_name("dgs-hf-cache", create_if_missing=True)
adapters = modal.Volume.from_name("dgs-adapters", create_if_missing=True)

app = modal.App(APP_NAME)


def _sha256_file(path: str) -> str:
    from hashlib import sha256

    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_configs() -> dict[str, Any]:
    """The exact BitsAndBytes / LoRA / DPO configuration both arms use.

    Importable on its own so `preflight` can construct it on a CPU container and prove the
    installed TRL/peft accept every argument before a GPU is ever allocated.
    """
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import DPOConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )

    def make_args(output_dir: str, epochs: float = NUM_EPOCHS) -> "DPOConfig":
        return DPOConfig(
            output_dir=output_dir,
            beta=DPO_BETA,
            loss_type=["sigmoid"],
            max_length=MAX_LENGTH,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            # transformers 5.x dropped `warmup_ratio`; `warmup_steps` in [0, 1) is that ratio.
            warmup_steps=0.1,
            num_train_epochs=epochs,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            gradient_checkpointing=True,
            bf16=True,
            optim="adamw_torch",
            seed=SEED,
            data_seed=SEED,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "eager",
                               "revision": BASE_REVISION},
        )

    return {"quantization": quantization, "lora": lora, "make_args": make_args}


@app.function(image=train_image, timeout=15 * MINUTES)
def preflight() -> dict[str, Any]:
    """CPU-only import/config check: catches an API mismatch before any GPU time is spent."""
    import importlib.metadata as metadata

    from trl import DPOTrainer  # noqa: F401 - imported to prove the symbol exists

    configs = build_configs()
    args = configs["make_args"]("/tmp/preflight")
    versions = {name: metadata.version(name) for name in
                ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets")}
    return {
        "versions": versions,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "loss_type": args.loss_type,
        "lora_target_modules": sorted(configs["lora"].target_modules),
        "lora_r_alpha_dropout": [configs["lora"].r, configs["lora"].lora_alpha,
                                 configs["lora"].lora_dropout],
        "quantization": configs["quantization"].to_dict().get("bnb_4bit_quant_type"),
    }


@app.function(
    image=train_image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache, ADAPTER_ROOT: adapters},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hugging_face_token()})],
    timeout=3 * 60 * MINUTES,
)
def train(pairs_jsonl_text: str, arm: str, epochs: float = NUM_EPOCHS) -> dict[str, Any]:
    """Train one arm and return its manifest (paths, adapter sha256, training summary)."""
    import gc
    import json
    import os as _os
    import shutil
    import time

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    if arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B', got %r" % (arm,))
    started = time.time()

    records = [json.loads(line) for line in pairs_jsonl_text.splitlines() if line.strip()]
    if not records:
        raise ValueError("no pair records were supplied")
    arms = {record.get("arm") for record in records}
    if arms != {arm}:
        raise ValueError("pairs file carries arm(s) %s but this run is arm %r" % (sorted(arms), arm))
    examples = [{"prompt": [{"role": message["role"], "content": message["content"]}
                            for message in record["prompt"]],
                 "chosen": [{"role": "assistant", "content": record["chosen"]}],
                 "rejected": [{"role": "assistant", "content": record["rejected"]}]}
                for record in records]
    dataset = Dataset.from_list(examples)
    print("arm %s: %d preference pairs" % (arm, len(dataset)), flush=True)

    arm_root = _os.path.join(ADAPTER_ROOT, arm)
    lora_dir = _os.path.join(arm_root, "lora")
    merged_dir = _os.path.join(arm_root, "merged")
    work_dir = "/tmp/dpo-%s" % arm
    for path in (lora_dir, merged_dir):
        if _os.path.isdir(path):
            shutil.rmtree(path)
    _os.makedirs(arm_root, exist_ok=True)

    configs = build_configs()
    args = configs["make_args"](work_dir, epochs)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_REVISION)

    trainer = DPOTrainer(
        model=BASE_MODEL_ID,
        ref_model=None,  # the reference policy is the adapter-disabled base model
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        quantization_config=configs["quantization"],
        peft_config=configs["lora"],
    )
    result = trainer.train()

    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    steps = [{"step": entry.get("step"), "loss": entry.get("loss"),
              "rewards_accuracies": entry.get("rewards/accuracies"),
              "rewards_margins": entry.get("rewards/margins"),
              "learning_rate": entry.get("learning_rate")}
             for entry in history]
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters()
                    if parameter.requires_grad)

    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    adapters.commit()
    print("saved adapter -> %s" % lora_dir, flush=True)

    # Free the 4-bit training copy before the bf16 merge so the two never coexist on the GPU.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, revision=BASE_REVISION, dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cpu")
    merged = PeftModel.from_pretrained(base, lora_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    adapters.commit()
    print("saved merged bf16 model -> %s" % merged_dir, flush=True)

    adapter_weights = _os.path.join(lora_dir, "adapter_model.safetensors")
    merged_files = sorted(name for name in _os.listdir(merged_dir))
    manifest = {
        "schema_version": "dgs-dpo-train-v1",
        "arm": arm,
        "app_name": APP_NAME,
        "gpu": GPU,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "adapter_volume": "dgs-adapters",
        "lora_path": lora_dir,
        "merged_path": merged_dir,
        "merged_files": merged_files,
        "adapter_sha256": _sha256_file(adapter_weights),
        "adapter_config_sha256": _sha256_file(_os.path.join(lora_dir, "adapter_config.json")),
        "pairs": len(dataset),
        "trainable_parameters": trainable,
        "hyperparameters": {
            "lora_r": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES), "beta": DPO_BETA,
            "learning_rate": LEARNING_RATE, "epochs": epochs,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch_size": PER_DEVICE_BATCH * GRAD_ACCUM,
            "seed": SEED, "max_length": MAX_LENGTH, "quantization": "nf4_double_4bit",
            "loss_type": "sigmoid", "ref_model": "peft_adapter_disabled_base",
        },
        "pins": PINS,
        "training": {
            "steps": int(result.global_step),
            "train_runtime_s": round(float(result.metrics.get("train_runtime", 0.0)), 1),
            "final_loss": history[-1].get("loss") if history else None,
            "mean_reward_accuracy": (
                round(sum(entry["rewards_accuracies"] for entry in steps
                          if entry["rewards_accuracies"] is not None)
                      / max(1, sum(1 for entry in steps
                                   if entry["rewards_accuracies"] is not None)), 4)),
            "final_reward_accuracy": next(
                (entry["rewards_accuracies"] for entry in reversed(steps)
                 if entry["rewards_accuracies"] is not None), None),
            "log_history": steps,
        },
        "wall_clock_s": round(time.time() - started, 1),
    }
    print(json.dumps({key: value for key, value in manifest.items() if key != "training"},
                     indent=2, sort_keys=True), flush=True)
    return manifest


@app.function(
    image=train_image,
    gpu=GPU,
    volumes={HF_CACHE: hf_cache, ADAPTER_ROOT: adapters},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": _hugging_face_token(),
                                     "DGS_TRAIN_GUARD": _shared_guard()})],
    timeout=3 * 60 * MINUTES,
)
@modal.fastapi_endpoint(method="POST", docs=False)
def train_web(payload: dict):
    """HTTPS fallback for `train`, used when Modal's gRPC invocation path is unreachable.

    `*.modal.run` reaches this machine reliably while `*.modal.com` gRPC handshakes stall (see
    `scripts/train_dpo.py`), so the same training body is exposed over plain HTTPS.  The reply
    is a newline-delimited stream: a heartbeat object every few seconds while training runs,
    then one final object carrying the manifest (or the error).  Streaming keeps the connection
    productive for the whole run instead of betting on a single long-silent response.
    """
    import json as _json
    import os as _os
    import queue
    import threading

    from fastapi.responses import JSONResponse, StreamingResponse

    if payload.get("token") != _os.environ.get("DGS_TRAIN_GUARD"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    arm = payload.get("arm")
    pairs = payload.get("pairs_jsonl_text")
    if arm not in ("A", "B") or not isinstance(pairs, str) or not pairs.strip():
        return JSONResponse({"error": "arm must be 'A' or 'B' and pairs_jsonl_text nonempty"},
                            status_code=400)
    epochs = float(payload.get("epochs", NUM_EPOCHS))

    done: "queue.Queue[None]" = queue.Queue()
    outcome: dict[str, Any] = {}

    def work() -> None:
        try:
            outcome["manifest"] = train.local(pairs, arm, epochs)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller as JSON
            outcome["error"] = "%s: %s" % (type(exc).__name__, exc)
        finally:
            done.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        waited = 0
        while True:
            try:
                done.get(timeout=10)
                break
            except queue.Empty:
                waited += 10
                yield _json.dumps({"status": "running", "arm": arm, "elapsed_s": waited}) + "\n"
        yield _json.dumps(outcome) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.local_entrypoint()
def preflight_local() -> None:
    """`modal run src/dpo_train_modal.py::preflight_local` -- print the CPU preflight result."""
    import json

    print(json.dumps(preflight.remote(), indent=2, sort_keys=True))
