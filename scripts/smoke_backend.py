"""Live smoke test of a deployed vLLM endpoint against the frozen protocol.

Runs, in order: GET /v1/models, the four in-context letter-token checks, one full greedy
trajectory through `src.runner.run_trajectory`, M1 and partial entropy on the measured
record, and finally N concurrent trajectories to measure aggregate throughput.

Writes nothing: raw records stay in memory, so no generation artifact ever lands in the
preregistration firewall's path.

Usage (PowerShell, from the repo root)::

    C:\\...\\.venv\\Scripts\\python.exe scripts/smoke_backend.py `
        --base-url https://<workspace>--dgs-vllm-gemma-2-2b-it-serve.modal.run/v1 `
        --model google/gemma-2-2b-it --revision <40-hex sha> --concurrency 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import OpenAICompatBackend, probe_letter_tokens  # noqa: E402
from src.metrics import TAIL_MASS_TOLERANCE, MetricInputError, m1_margin, partial_entropy  # noqa: E402
from src.protocol import discovery_tasks, load_protocol  # noqa: E402
from src.runner import run_trajectory  # noqa: E402

CELL_ID = "easy__malfunctioning_always_fail__neutral"
OPTION_TOKEN = re.compile(r"(\s*)([A-D])(\s*)")


def max_mass_excess(record) -> float:
    """Largest amount by which a position's top-20 probabilities exceed one (fp32 noise)."""
    worst = 0.0
    for token in record.tokens:
        worst = max(worst, sum(math.exp(logprob) for _, logprob in token.top_logprobs) - 1.0)
    return worst


def letter_candidates(record) -> dict[str, Any]:
    """Independent of parser validity: are all four letters in the top-20 at the answer position?

    m1_margin reports 'missing' both when the logprob channel is broken and when the response
    simply does not end in the frozen `Answer: X` line. This separates the two.
    """
    text = "".join(token.text for token in record.tokens)
    prefix = "Answer: "
    position = text.rfind(prefix)
    if position < 0:
        return {"found": False, "reason": "no 'Answer: ' in response"}
    offset = position + len(prefix)
    cursor = 0
    for token in record.tokens:
        end = cursor + len(token.text)
        if cursor <= offset < end:
            boundary = OPTION_TOKEN.fullmatch(token.text)
            if boundary is None:
                return {"found": False, "reason": "option token carries visible text: %r" % token.text}
            leading, letter, trailing = boundary.groups()
            present = {option: any(candidate == leading + option + trailing
                                   for candidate, _ in token.top_logprobs) for option in "ABCD"}
            return {"found": True, "token": token.text, "letter": letter,
                    "alternatives": len(token.top_logprobs), "present": present,
                    "all_four": all(present.values())}
        cursor = end
    return {"found": False, "reason": "answer offset not inside any token"}


def get_models(base_url: str, deadline_s: float = 1200.0) -> Any:
    """Poll /v1/models until the container has cold-started and loaded the weights."""
    url = "%s/models" % base_url.rstrip("/")
    deadline = time.time() + deadline_s
    attempt = 0
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - cold start surfaces as timeouts and 5xx alike
            if time.time() >= deadline:
                raise RuntimeError("endpoint not ready after %.0fs" % deadline_s) from exc
            print("     waiting for cold start (attempt %d): %s" % (attempt, type(exc).__name__))
            time.sleep(15)


def completion_tokens(backend: OpenAICompatBackend) -> int:
    return int(backend.stats.get("completion_tokens", 0))


def trajectory(task, backend, model_id: str, revision: str, sample_index: int, protocol) -> Sequence[Any]:
    kwargs: dict[str, Any] = {
        "task": task, "cell_id": CELL_ID, "model_id": model_id, "immutable_revision": revision,
        "sample_index": sample_index, "backend": backend, "protocol": protocol,
        "run_id": "dgs-live-smoke", "phase": "phase_0",
    }
    # Agent B is adding a run_kind parameter concurrently; use it only once it exists.
    if "run_kind" in inspect.signature(run_trajectory).parameters:
        kwargs["run_kind"] = "empirical"
    return run_trajectory(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test against a deployed vLLM endpoint")
    parser.add_argument("--base-url", required=True, help="e.g. https://xxx.modal.run/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="40-hex Hugging Face sha")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--warmup-timeout-s", type=float, default=1200.0,
                        help="how long to wait for a Modal cold start (weights download + load)")
    args = parser.parse_args()

    protocol = load_protocol(ROOT)
    task = next(item for item in discovery_tasks(protocol) if item.difficulty == "easy")

    print("=" * 78)
    print("(a) GET /v1/models")
    started = time.perf_counter()
    served = get_models(args.base_url, args.warmup_timeout_s)
    print("     ready in %.0fs" % (time.perf_counter() - started))
    print("    ", json.dumps(served)[:400])

    print("(b) letter-token check")
    letters = probe_letter_tokens(args.base_url, args.model, args.api_key, timeout_s=300.0)
    print("    ", letters, "-> all single tokens:", all(letters.values()))

    backend = OpenAICompatBackend(args.base_url, args.model, api_key=args.api_key,
                                  timeout_s=args.timeout_s)
    try:
        print("(c) one greedy trajectory: task=%s cell=%s" % (task.task_id, CELL_ID))
        started = time.perf_counter()
        records = trajectory(task, backend, args.model, args.revision, 0, protocol)
        greedy_seconds = time.perf_counter() - started
        greedy_tokens = completion_tokens(backend)
        print("     turns:", [record.turn_label for record in records])
        print("     answers:", [record.final_answer_letter for record in records])
        print("     valid:", [record.final_answer_valid for record in records])
        print("     last 200 chars of the measured response:")
        measured = next(record for record in records if record.turn_label == "measured")
        print("      ", repr(measured.response_text[-200:]))

        print("(d) metrics on the measured record")
        margin = m1_margin(measured, protocol=protocol)
        print("     M1 margin:", margin.margin.value, "| missing_reason:", margin.margin.missing_reason)
        print("     canonical:", margin.canonical_answer, "| generated:", margin.generated_answer,
              "| option_token_index:", margin.option_token_index)
        print("     M1 present:", margin.margin.value is not None)
        print("     letter candidates at the answer position:", letter_candidates(measured))
        excess = max_mass_excess(measured)
        print("     max top-logprob mass excess: %.3e (metrics tolerance %.0e)"
              % (excess, TAIL_MASS_TOLERANCE))
        try:
            entropy = partial_entropy(measured)
            print("     mean partial entropy:", entropy.mean_partial_entropy.value,
                  "| highest decile:", entropy.highest_entropy_decile_mean.value,
                  "| mean tail mass:", entropy.mean_tail_mass.value)
        except MetricInputError as exc:
            # Reported, not swallowed: fp32 logprobs from a real server routinely sum to
            # 1 + ~1e-7, which the current 1e-9 tolerance in src/metrics.py rejects.
            print("     partial_entropy REJECTED the measured record: %s" % exc)
            print("     -> src/metrics.py TAIL_MASS_TOLERANCE (1e-9) is tighter than fp32 noise;"
                  " it needs to be about 1e-6 for live vLLM logprobs.")

        print("(e) throughput")
        print("     greedy: %d completion tokens in %.1fs = %.1f tok/s (%d turns)"
              % (greedy_tokens, greedy_seconds, greedy_tokens / max(greedy_seconds, 1e-9), len(records)))

        print("(f) %d concurrent trajectories" % args.concurrency)
        before_tokens = completion_tokens(backend)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(trajectory, task, backend, args.model, args.revision, index + 1, protocol)
                       for index in range(args.concurrency)]
            batches = [future.result() for future in futures]
        concurrent_seconds = time.perf_counter() - started
        concurrent_tokens = completion_tokens(backend) - before_tokens
        print("     trajectories: %d, records: %d" % (len(batches), sum(len(batch) for batch in batches)))
        print("     concurrent: %d completion tokens in %.1fs = %.1f tok/s"
              % (concurrent_tokens, concurrent_seconds,
                 concurrent_tokens / max(concurrent_seconds, 1e-9)))
        print("     speedup vs single-stream: %.1fx"
              % ((concurrent_tokens / max(concurrent_seconds, 1e-9))
                 / max(greedy_tokens / max(greedy_seconds, 1e-9), 1e-9)))
        print("backend stats:", backend.stats)
    finally:
        backend.close()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
