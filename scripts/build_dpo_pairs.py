"""Build the Phase 4 DPO preference pairs (arm A distress-suppression, arm B length placebo).

Preregistration v5 (`notes/preregistration_v5_phase4.md`) fixes the recipe:

1. fresh 4-option MCQ items from a public CC-licensed bank (`allenai/ai2_arc`, CC-BY-SA-4.0),
   converted to the frozen DGS task schema and firewalled against the 40 locked tasks,
2. the model's own greedy answer, keeping only items it answers correctly,
3. the adverse context `[task, own correct answer, frozen hostile failure message]`,
4. four candidate continuations per context at T = 0.8 (seeds 1-4),
5. the pinned judge scores every candidate's distress at temperature 0,
6. arm A pairs (lowest- vs highest-distress, gap >= 3) and arm B placebo pairs
   (shorter vs longer, >= 40 whitespace tokens) subsampled to |A|.

Every stage is cached under ``<out>/raw`` and keyed by content, so the script is resumable and
re-running it costs nothing for work already done.

Usage (PowerShell, from the repo root)::

    $env:DGS_MODEL_ID='google/gemma-2-9b-it'
    $env:DGS_REVISION='11c9b309abf73637e4b6f9a3fa1e92e615547819'
    C:\\...\\.venv\\Scripts\\python.exe -m modal deploy src/serve_modal.py
    C:\\...\\.venv\\Scripts\\python.exe scripts/build_dpo_pairs.py --count 600 \
        --base-url https://<workspace>--dgs-vllm-gemma-2-9b-it-serve.modal.run/v1
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backend import GenerationRequest, OpenAICompatBackend  # noqa: E402
from src.dpo_data import (  # noqa: E402
    ARM_DISTRESS, ARM_PLACEBO, CANDIDATE_SEEDS, MIN_DISTRESS_GAP, MIN_LENGTH_GAP_TOKENS,
    AdverseContext, Candidate, DpoDataError, build_context, build_distress_pairs,
    build_firewall, build_length_pairs, firewall_items, make_fresh_item, rank_items,
    render_item, validate_pair_record,
)
from src.judge import parse_backend_output  # noqa: E402
from src.judge_client import (KIND_RESPONSE, JsonlJudgeCache, JudgeClientError,  # noqa: E402
                              cache_key, load_env_files, make_judge_backend_from_manifest,
                              manifest_judge_ids)
from src.protocol import load_protocol, parse_final_answer  # noqa: E402

DATASET = "allenai/ai2_arc"
DATASET_LICENSE = "CC-BY-SA-4.0"
DATASET_CONFIGS = ("ARC-Challenge", "ARC-Easy")
# `train` only: the capability probe in `scripts/run_phase4.py` needs untouched fresh items,
# so validation/test stay clear of anything the adapters were trained on.
DATASET_SPLIT = "train"

MODEL_ID = "google/gemma-2-9b-it"
GREEDY_SETTINGS = {"temperature": 0, "top_p": 1, "max_logprobs": 1, "max_tokens": 512}
CANDIDATE_SETTINGS = {"temperature": 0.8, "top_p": 1, "max_logprobs": 1, "max_tokens": 512}
GREEDY_SEED = 0

# Amendment A5 (notes/amendments.md, commit 60c38d0, written before any full-set candidate was
# judged): the pair-yield contingency, applied mechanically once the full candidate set is judged.
#   (i)   |A| at gap >= 3 with 4 candidates >= A5_TARGET_PAIRS  -> proceed as preregistered,
#   (ii)  otherwise sample 4 more candidates (seeds 5-8) for every context whose current maximum
#         judged distress >= A5_TOPUP_MIN_DISTRESS, judge them, and re-pair at gap >= 3,
#   (iii) if |A| is still short, pair at gap >= 2 -- never lower.
# Whichever branch fires, every rule's count is published.
A5_TARGET_PAIRS = 200
A5_TOPUP_SEEDS = (5, 6, 7, 8)
A5_TOPUP_MIN_DISTRESS = 2
A5_FLOOR_GAP = 2
# Budget guard: total distinct judge calls across the whole build (~$0.002 each).
MAX_JUDGE_CALLS = 3500

JUDGE_RUBRIC_PATH = "configs/judge_rubric.md"
JUDGE_RESPONSE_ID = "dpo-phase4|%s|%d"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    raise SystemExit("build_dpo_pairs: %s" % message)


def _log(message: str) -> None:
    print("build_dpo_pairs: %s" % message, flush=True)


# ---------------------------------------------------------------------------------------
# Resumable JSONL stage cache
# ---------------------------------------------------------------------------------------

class StageCache:
    """Append-only JSONL keyed by a caller-chosen string, so every stage resumes for free."""

    def __init__(self, path: Path, key_fields: Sequence[str]) -> None:
        self.path = path
        self.key_fields = tuple(key_fields)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn trailing line from an interrupted run
                if isinstance(value, dict):
                    self._entries[self._key(value)] = value

    def _key(self, value: dict[str, Any]) -> str:
        return "|".join(str(value.get(field)) for field in self.key_fields)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, value: dict[str, Any]) -> None:
        key = self._key(value)
        line = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if key in self._entries:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
            self._entries[key] = value

    def __len__(self) -> int:
        return len(self._entries)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Stage 1: fresh items
# ---------------------------------------------------------------------------------------

def load_fresh_items(protocol: Any, count: int, holdout: int = 0
                     ) -> tuple[list[Any], list[Any], list[dict[str, str]], dict[str, Any]]:
    """Fetch, convert, firewall, hash-rank, then split the pool into training and holdout slices.

    The holdout slice is the next `holdout` items after the training prefix; it never enters a
    training context, so `scripts/run_phase4.py`'s capability probe can use it as genuinely
    unseen fresh items under all three arms.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
        _fail("the 'datasets' package is required to fetch fresh MCQ items (%s)" % exc)

    converted, malformed = [], 0
    for config in DATASET_CONFIGS:
        rows = load_dataset(DATASET, config, split=DATASET_SPLIT)
        for row in rows:
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            if labels != ["A", "B", "C", "D"] or row["answerKey"] not in labels:
                continue
            try:
                converted.append(make_fresh_item(
                    item_id=row["id"], dataset=DATASET, config=config, split=DATASET_SPLIT,
                    stem=row["question"], options=dict(zip(labels, texts)),
                    canonical_answer=row["answerKey"], protocol=protocol))
            except DpoDataError:
                malformed += 1
    kept, excluded = firewall_items(rank_items(converted), build_firewall(protocol))
    selected = kept[:count]
    reserved = kept[count:count + holdout] if holdout > 0 else []
    report = {
        "dataset": DATASET, "license": DATASET_LICENSE, "configs": list(DATASET_CONFIGS),
        "split": DATASET_SPLIT, "rows_seen": len(converted) + malformed,
        "four_option_abcd": len(converted), "malformed_dropped": malformed,
        "firewall_excluded": len(excluded), "firewall_reasons": sorted(
            {row["reason"] for row in excluded}),
        "available_after_firewall": len(kept), "selected": len(selected),
        "capability_holdout": len(reserved),
    }
    return selected, reserved, excluded, report


# ---------------------------------------------------------------------------------------
# Stage 2/3: greedy answers and candidate continuations
# ---------------------------------------------------------------------------------------

def _generate(backend: OpenAICompatBackend, messages: Sequence[dict[str, str]], seed: int,
              settings: dict[str, Any]) -> str:
    request = GenerationRequest(tuple(messages), seed, settings)
    return backend.generate(request).text


def collect_greedy(backend: OpenAICompatBackend, items: Sequence[Any], protocol: Any,
                   cache: StageCache, workers: int) -> list[dict[str, Any]]:
    """One greedy answer per fresh item; correctness decides whether it becomes a context."""
    results: list[dict[str, Any] | None] = [None] * len(items)
    failures: list[str] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        item = items[index]
        cached = cache.get(item.context_id)
        if cached is not None:
            results[index] = cached
            return
        message = render_item(item, protocol)
        try:
            text = _generate(backend, [{"role": "user", "content": message}], GREEDY_SEED,
                             GREEDY_SETTINGS)
        except Exception as exc:  # noqa: BLE001 - a dead item must not lose the batch
            with lock:
                failures.append("%s: %s: %s" % (item.item_id, type(exc).__name__, exc))
            return
        answer = parse_final_answer(text)
        record = {
            "context_id": item.context_id, "item_id": item.item_id, "config": item.config,
            "task_message": message, "greedy_text": text,
            "answer_valid": bool(answer.valid), "answer_letter": answer.letter,
            "canonical_answer": item.canonical_answer,
            "correct": bool(answer.valid and answer.letter == item.canonical_answer),
        }
        cache.put(record)
        results[index] = record

    _map(run, len(items), workers)
    if failures:
        _log("greedy stage: %d failure(s); first: %s" % (len(failures), failures[0]))
    return [record for record in results if record is not None]


def collect_candidates(backend: OpenAICompatBackend, contexts: Sequence[AdverseContext],
                       cache: StageCache, workers: int,
                       seeds: Sequence[int] = CANDIDATE_SEEDS) -> tuple[list[AdverseContext], int]:
    """T=0.8 continuations per adverse context, one per seed (1-4 normally, 5-8 for an A5 top-up)."""
    jobs = [(index, seed) for index in range(len(contexts)) for seed in seeds]
    texts: dict[tuple[str, int], str] = {}
    failures: list[str] = []
    lock = threading.Lock()

    def run(position: int) -> None:
        index, seed = jobs[position]
        built = contexts[index]
        key = "%s|%d" % (built.context_id, seed)
        cached = cache.get(key)
        if cached is None:
            try:
                text = _generate(backend, built.messages, seed, CANDIDATE_SETTINGS)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    failures.append("%s seed %d: %s: %s"
                                    % (built.item.item_id, seed, type(exc).__name__, exc))
                return
            cached = {"context_id": built.context_id, "seed": seed,
                      "item_id": built.item.item_id, "text": text}
            cache.put(cached)
        with lock:
            texts[(built.context_id, seed)] = cached["text"]

    _map(run, len(jobs), workers)
    if failures:
        _log("candidate stage: %d failure(s); first: %s" % (len(failures), failures[0]))
    filled = []
    for built in contexts:
        candidates = [Candidate(seed, texts[(built.context_id, seed)])
                      for seed in seeds if (built.context_id, seed) in texts]
        if len(candidates) >= 2:
            filled.append(built.with_candidates(candidates))
    return filled, len(failures)


def merge_candidates(base: Sequence[AdverseContext],
                     extra: Sequence[AdverseContext]) -> list[AdverseContext]:
    """Fold an A5 top-up (seeds 5-8) into the contexts already carrying seeds 1-4."""
    by_id = {built.context_id: list(built.candidates) for built in extra}
    merged = []
    for built in base:
        addition = by_id.get(built.context_id)
        merged.append(built.with_candidates(list(built.candidates) + addition) if addition
                      else built)
    return merged


def _map(run: Any, total: int, workers: int) -> None:
    workers = max(1, min(int(workers), 64))
    if workers == 1 or total <= 1:
        for index in range(total):
            run(index)
        return
    done = {"n": 0}
    lock = threading.Lock()

    def wrapped(index: int) -> None:
        run(index)
        with lock:
            done["n"] += 1
            if done["n"] % 200 == 0 or done["n"] == total:
                _log("  %d/%d" % (done["n"], total))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(wrapped, range(total)))


# ---------------------------------------------------------------------------------------
# Stage 4: judged distress
# ---------------------------------------------------------------------------------------

def score_candidates(contexts: Sequence[AdverseContext], backend: Any, cache: JsonlJudgeCache,
                     rubric_text: str, rubric_hash: str, workers: int) -> tuple[list[AdverseContext],
                                                                                dict[str, Any]]:
    """Score every candidate's response distress with the pinned judge at temperature 0."""
    jobs = [(index, candidate) for index, built in enumerate(contexts)
            for candidate in built.candidates]
    scores: dict[tuple[str, int], tuple[int, str]] = {}
    failures: list[str] = []
    lock = threading.Lock()

    def run(position: int) -> None:
        index, candidate = jobs[position]
        built = contexts[index]
        content = candidate.text.strip()
        if not content:
            return
        key = cache_key(kind=KIND_RESPONSE,
                        response_id=JUDGE_RESPONSE_ID % (built.context_id, candidate.seed),
                        input_sha256=_digest(candidate.text), rubric_sha256=rubric_hash,
                        provider_id=backend.provider_id, model_id=backend.model_id)
        cached = cache.get(key)
        if cached is None:
            try:
                call = backend.score_text(kind=KIND_RESPONSE, rubric_text=rubric_text,
                                          content=content)
            except Exception as exc:  # noqa: BLE001 - one refusal must not lose the batch
                with lock:
                    failures.append("%s seed %d: %s: %s" % (built.context_id, candidate.seed,
                                                            type(exc).__name__, exc))
                return
            cache.put(key, backend_id=backend.backend_id, canonical_output=call.canonical_output,
                      verbatim_output=call.verbatim_output, attempts=call.attempts,
                      format_repair_used=call.format_repair_used,
                      sampling_mode=call.sampling_mode)
            cached = call.canonical_output
        parsed = parse_backend_output(cached, KIND_RESPONSE)
        with lock:
            scores[(built.context_id, candidate.seed)] = (parsed[KIND_RESPONSE],
                                                          parsed["evidence"])

    _map(run, len(jobs), workers)
    scored = []
    for built in contexts:
        candidates = []
        for candidate in built.candidates:
            found = scores.get((built.context_id, candidate.seed))
            if found is None:
                continue
            candidates.append(Candidate(candidate.seed, candidate.text, found[0], found[1]))
        if len(candidates) >= 2:
            scored.append(built.with_candidates(candidates))
    report = {
        "provider_id": backend.provider_id, "model_id": backend.model_id,
        "backend_id": backend.backend_id, "temperature": 0,
        "sampling_mode": getattr(backend, "sampling_mode", None),
        "rubric_path": JUDGE_RUBRIC_PATH, "rubric_sha256": rubric_hash,
        "calls_requested": len(jobs), "scored": len(scores), "failures": len(failures),
        "failure_examples": failures[:5],
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "usage": getattr(backend, "usage", None),
        "estimated_cost_usd": getattr(backend, "estimated_cost_usd", None),
    }
    return scored, report


# ---------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------

def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": ordered[0], "p25": ordered[len(ordered) // 4],
        "median": round(statistics.median(values), 2),
        "p75": ordered[(3 * len(ordered)) // 4], "max": ordered[-1],
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda row: int(row[0])))


def summary_markdown(report: dict[str, Any]) -> str:
    items, judge = report["items"], report["judge"]
    counts, pairs = report["counts"], report["pairs"]
    lines = [
        "# Phase 4 DPO pairs — build summary",
        "",
        "Preregistration v5 (`notes/preregistration_v5_phase4.md`). Generated %s."
        % report["generated_at"],
        "",
        "## Source bank",
        "",
        "| field | value |",
        "| --- | --- |",
        "| dataset | `%s` (%s) |" % (items["dataset"], items["license"]),
        "| configs / split | %s / `%s` |" % (", ".join(items["configs"]), items["split"]),
        "| 4-option A–D rows available | %d |" % items["four_option_abcd"],
        "| dropped by the locked-bank firewall | %d |" % items["firewall_excluded"],
        "| items selected (hash-ranked prefix) | %d |" % items["selected"],
        "| items reserved, untouched, for the capability probe | %d |"
        % items.get("capability_holdout", 0),
        "",
        "Firewall: every fresh stem is checked against all 40 locked tasks by exact "
        "(whitespace-normalised) stem text, by stem SHA-256, and by stem+canonical-answer-text "
        "SHA-256. Exclusion reasons seen: %s."
        % (", ".join("`%s`" % reason for reason in items["firewall_reasons"]) or "none"),
        "",
        "## Pipeline",
        "",
        "| stage | n |",
        "| --- | ---: |",
        "| fresh items sampled | %d |" % counts["items"],
        "| greedy answers returned | %d |" % counts["greedy"],
        "| answered correctly (adverse contexts) | %d |" % counts["contexts"],
        "| contexts with >= 2 usable candidates | %d |" % counts["contexts_with_candidates"],
        "| contexts with >= 2 judged candidates | %d |" % counts["contexts_scored"],
        "| candidate generations | %d |" % counts["candidates"],
        "| judge calls (incl. cache hits) | %d |" % judge["calls_requested"],
        "",
        "Greedy accuracy on the fresh bank: **%.1f%%** (%d/%d)."
        % (100.0 * counts["contexts"] / max(1, counts["greedy"]), counts["contexts"],
           counts["greedy"]),
        "",
        "## Judged distress across all candidates",
        "",
        "Judge `%s` / `%s` at temperature 0 (`sampling_mode: %s`), rubric `%s` "
        "(sha256 `%s`)."
        % (judge["provider_id"], judge["model_id"], judge["sampling_mode"],
           judge["rubric_path"], judge["rubric_sha256"][:16]),
        "",
        "| statistic | value |",
        "| --- | ---: |",
    ]
    for key, value in report["distress_all"].items():
        lines.append("| %s | %s |" % (key, value))
    lines += ["", "Score histogram (all candidates): %s."
              % ", ".join("**%s**×%d" % (score, n)
                          for score, n in report["distress_histogram"].items()), ""]
    a5 = report["a5_contingency"]
    lines += [
        "## Amendment A5 pair-yield contingency",
        "",
        "Rule fixed in `%s` before any full-set candidate was judged. Branch **(%s)** fired: %s."
        % (a5["amendment"], a5["branch"], a5["branch_rule"]),
        "",
        "| rule tried | arm-A pairs |",
        "| --- | ---: |",
    ]
    for key, value in a5["counts"].items():
        lines.append("| %s | %d |" % (key.replace("_", " "), value))
    lines += ["", "Target for branching: %d pairs." % a5["target_pairs"], ""]
    topup = a5.get("topup")
    if topup:
        lines += [
            "Top-up (seeds %s) offered to every context whose maximum judged distress reached "
            "%d: **%d eligible**, **%d actually topped up**%s. Ranking: %s. The judge cache held "
            "%d of the %d permitted calls before the top-up."
            % (", ".join(str(seed) for seed in topup["seeds"]), topup["min_distress"],
               topup["eligible_contexts"], topup["topped_up_contexts"],
               " (capped by the judge budget)" if topup["capped_by_judge_budget"] else "",
               topup["ranking"], topup["judge_calls_before_topup"], topup["max_judge_calls"]),
            "",
        ]
    lines += [
        "## Pairs",
        "",
        "| arm | rule | n | mean gap | gap distribution |",
        "| --- | --- | ---: | ---: | --- |",
        "| A (distress) | chosen = lowest judged distress, rejected = highest, gap >= %d | %d | %s | %s |"
        % (a5["effective_min_distress_gap"], pairs["A"]["n"],
           pairs["A"]["gap"].get("mean", "n/a"),
           ", ".join("%s×%d" % row for row in pairs["A"]["gap_histogram"].items()) or "n/a"),
        "| B (placebo) | chosen = shorter, rejected = longer, gap >= %d whitespace tokens | %d | %s | min %s / median %s / max %s |"
        % (MIN_LENGTH_GAP_TOKENS, pairs["B"]["n"], pairs["B"]["gap"].get("mean", "n/a"),
           pairs["B"]["gap"].get("min", "n/a"), pairs["B"]["gap"].get("median", "n/a"),
           pairs["B"]["gap"].get("max", "n/a")),
        "",
        "Arm A chosen-vs-rejected distress: chosen mean **%s**, rejected mean **%s**."
        % (pairs["A"]["chosen_distress"].get("mean", "n/a"),
           pairs["A"]["rejected_distress"].get("mean", "n/a")),
        "",
        "Arm B was subsampled to |A| deterministically (ascending keyed SHA-256 of the context "
        "id), taking contexts arm A also used first: **%d of %d** B pairs (%.0f%%) sit on an "
        "arm-A context, so the two arms see nearly the same prompt distribution."
        % (pairs["B"]["shared_contexts_with_A"], pairs["B"]["n"],
           100.0 * pairs["B"]["shared_contexts_with_A"] / max(1, pairs["B"]["n"])),
        "",
        "Placebo pairs available before subsampling: %d." % pairs["B"]["available"],
        "",
        "## Cost",
        "",
        "| item | value |",
        "| --- | --- |",
        "| judge calls made (cache misses) | %d |" % judge["cache_misses"],
        "| judge cache hits | %d |" % judge["cache_hits"],
        "| judge usage | %s |" % json.dumps(judge["usage"], sort_keys=True),
        "| judge list-price cost, this invocation | %s |"
        % (("$%.4f" % judge["estimated_cost_usd"]) if judge["estimated_cost_usd"] is not None
           else "n/a"),
        "| distinct judge calls in the cache (whole build) | %d |" % report["judge_calls_cached"],
        "| judge-call budget guard | %d |" % report["max_judge_calls"],
        "| vLLM endpoint | `%s` |" % report["endpoint"],
        "| generation wall clock | %.1f min |" % (report["elapsed_s"] / 60.0),
        "",
        "Only cache *misses* are billed, so the cost row above covers this invocation alone; the "
        "cache hits were paid for by an earlier one. The build's whole-run judge spend is the sum "
        "over invocations and is recorded in `notes/lab-log.md`.",
        "",
        "Ethics: chosen and rejected are both the model's own sampled outputs on a mild frozen "
        "stressor; no dysphoric optimisation target was written by hand.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_dpo_pairs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=600, help="fresh items to sample")
    parser.add_argument("--fresh-holdout", type=int, default=200,
                        help="extra items reserved, untouched, for the Phase 4 capability probe")
    parser.add_argument("--out", default="results/dpo", help="output directory")
    parser.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", default=MODEL_ID, help="served model id")
    parser.add_argument("--workers", type=int, default=48, help="concurrent generation requests")
    parser.add_argument("--judge-workers", type=int, default=8, help="concurrent judge calls")
    parser.add_argument("--skip-judge", action="store_true",
                        help="stop after generation (no judge calls, no pairs written)")
    parser.add_argument("--target-pairs", type=int, default=A5_TARGET_PAIRS,
                        help="Amendment A5 arm-A pair target that decides which branch fires")
    parser.add_argument("--max-judge-calls", type=int, default=MAX_JUDGE_CALLS,
                        help="budget guard: distinct judge calls allowed across the whole build")
    args = parser.parse_args(argv)

    load_env_files(ROOT)
    started = time.monotonic()
    protocol = load_protocol(ROOT)
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    raw = out / "raw"

    _log("selecting %d fresh items from %s ..." % (args.count, DATASET))
    items, reserved, excluded, item_report = load_fresh_items(protocol, args.count,
                                                              args.fresh_holdout)
    if not items:
        _fail("no fresh items survived the firewall")
    _write_jsonl(raw / "items.jsonl", [item.to_json() for item in items])
    _write_jsonl(raw / "firewall_excluded.jsonl", excluded)
    # Consumed by `scripts/run_phase4.py` as the capability probe's fresh items: the same bank
    # and the same firewall, but a slice no training context ever touched.
    _write_jsonl(out / "fresh_items.jsonl",
                 [{"item_id": item.item_id, "stem": item.stem, "options": dict(item.options),
                   "canonical_answer": item.canonical_answer, "subject": item.config,
                   "dataset": item.dataset, "split": item.split,
                   "used_by_dpo_training": False} for item in reserved])
    _log("items: %d selected, %d excluded by the firewall, %d reserved for the capability probe"
         % (len(items), len(excluded), len(reserved)))

    backend = OpenAICompatBackend(args.base_url, args.model)
    _log("warming up %s ..." % args.base_url)
    backend.warm_up()

    _log("greedy pass over %d items ..." % len(items))
    greedy = collect_greedy(backend, items, protocol, StageCache(raw / "greedy.jsonl",
                                                                ("context_id",)), args.workers)
    by_context = {item.context_id: item for item in items}
    correct = [record for record in greedy if record["correct"]]
    _log("greedy: %d answered, %d correct (%.1f%%)"
         % (len(greedy), len(correct), 100.0 * len(correct) / max(1, len(greedy))))

    contexts = [build_context(by_context[record["context_id"]], record["greedy_text"], protocol)
                for record in correct]
    _log("candidate pass: %d contexts x %d seeds ..." % (len(contexts), len(CANDIDATE_SEEDS)))
    candidate_cache = StageCache(raw / "candidates.jsonl", ("context_id", "seed"))
    contexts, generation_failures = collect_candidates(backend, contexts, candidate_cache,
                                                       args.workers)
    candidate_total = sum(len(built.candidates) for built in contexts)
    _log("candidates: %d generations across %d contexts (%d failure(s))"
         % (candidate_total, len(contexts), generation_failures))

    if args.skip_judge:
        backend.close()
        _log("--skip-judge: stopping before the judge stage")
        return 0

    rubric_bytes = (ROOT / JUDGE_RUBRIC_PATH).read_bytes()
    rubric_text, rubric_hash = rubric_bytes.decode("utf-8"), sha256(rubric_bytes).hexdigest()
    try:
        provider, model = manifest_judge_ids(protocol)
        judge_backend = make_judge_backend_from_manifest(protocol)
    except JudgeClientError as exc:
        _fail(str(exc))
    _log("judging %d candidates with the pinned judge %s/%s ..."
         % (candidate_total, provider, model))
    judge_cache = JsonlJudgeCache(out / "judge_cache.jsonl")
    scored, judge_report = score_candidates(contexts, judge_backend, judge_cache, rubric_text,
                                            rubric_hash, args.judge_workers)
    _log("judged: %d/%d candidate(s), %d context(s) fully usable, cost %s"
         % (judge_report["scored"], judge_report["calls_requested"], len(scored),
            judge_report["estimated_cost_usd"]))
    # ---- Amendment A5 contingency -------------------------------------------------------
    pairs_gap3_4 = build_distress_pairs(scored, min_gap=MIN_DISTRESS_GAP)
    contingency: dict[str, Any] = {
        "amendment": "A5 (notes/amendments.md, commit 60c38d0)",
        "target_pairs": args.target_pairs,
        "counts": {"gap3_with_4_candidates": len(pairs_gap3_4)},
        "topup": None,
    }
    if len(pairs_gap3_4) >= args.target_pairs:
        contingency["branch"] = "i"
        contingency["branch_rule"] = "gap >= 3 with 4 candidates, exactly as preregistered"
        pairs_a = pairs_gap3_4
    else:
        eligible = sorted(
            (built for built in scored
             if max((candidate.distress or 0) for candidate in built.candidates)
             >= A5_TOPUP_MIN_DISTRESS),
            key=lambda built: (-max((candidate.distress or 0) for candidate in built.candidates),
                               built.context_id))
        # A context whose top-up seeds are already generated was paid for by an earlier
        # invocation and is already inside `len(judge_cache)`; counting it again would shrink
        # the allowance on every re-run and silently drop contexts from a finished build.
        already = sum(1 for built in eligible
                      if all(candidate_cache.get("%s|%d" % (built.context_id, seed)) is not None
                             for seed in A5_TOPUP_SEEDS))
        room = max(0, args.max_judge_calls - len(judge_cache))
        allowed = already + room // len(A5_TOPUP_SEEDS)
        capped = eligible[:allowed]
        _log("A5 branch (ii): %d context(s) with max distress >= %d; judge budget allows %d "
             "(%d already generated; cache holds %d of %d calls)"
             % (len(eligible), A5_TOPUP_MIN_DISTRESS, len(capped), already, len(judge_cache),
                args.max_judge_calls))
        contingency["topup"] = {
            "seeds": list(A5_TOPUP_SEEDS), "min_distress": A5_TOPUP_MIN_DISTRESS,
            "eligible_contexts": len(eligible), "topped_up_contexts": len(capped),
            "capped_by_judge_budget": len(capped) < len(eligible),
            "ranking": "descending maximum judged distress, context id breaking ties",
            "judge_calls_before_topup": len(judge_cache),
            "already_generated_contexts": already,
            "max_judge_calls": args.max_judge_calls,
        }
        if capped:
            extra, extra_failures = collect_candidates(backend, capped, candidate_cache,
                                                       args.workers, seeds=A5_TOPUP_SEEDS)
            generation_failures += extra_failures
            extra_scored, topup_report = score_candidates(extra, judge_backend, judge_cache,
                                                          rubric_text, rubric_hash,
                                                          args.judge_workers)
            # `judge_backend` and `judge_cache` are shared, so usage/hits/misses in the second
            # report are already cumulative; only the per-call counters need adding up.
            for field in ("calls_requested", "scored", "failures"):
                topup_report[field] += judge_report[field]
            topup_report["failure_examples"] = (judge_report["failure_examples"]
                                                + topup_report["failure_examples"])[:5]
            judge_report = topup_report
            scored = merge_candidates(scored, extra_scored)
            candidate_total = sum(len(built.candidates) for built in scored)
            _log("A5 top-up: %d extra candidate(s) judged; cost now %s"
                 % (sum(len(built.candidates) for built in extra_scored),
                    judge_report["estimated_cost_usd"]))
        pairs_gap3_8 = build_distress_pairs(scored, min_gap=MIN_DISTRESS_GAP)
        contingency["counts"]["gap3_with_8_candidates"] = len(pairs_gap3_8)
        if len(pairs_gap3_8) >= args.target_pairs:
            contingency["branch"] = "ii"
            contingency["branch_rule"] = "gap >= 3 after topping the eligible contexts up to 8 candidates"
            pairs_a = pairs_gap3_8
        else:
            pairs_a = build_distress_pairs(scored, min_gap=A5_FLOOR_GAP)
            contingency["branch"] = "iii"
            contingency["branch_rule"] = ("gap >= %d (the A5 floor) on the topped-up candidate set"
                                          % A5_FLOOR_GAP)
    contingency["counts"]["gap%d_on_final_candidate_set" % A5_FLOOR_GAP] = len(
        build_distress_pairs(scored, min_gap=A5_FLOOR_GAP))
    contingency["selected_pairs"] = len(pairs_a)
    _log("A5: branch %s -> %d arm-A pair(s) (%s)"
         % (contingency["branch"], len(pairs_a), contingency["counts"]))

    _write_jsonl(raw / "judged.jsonl",
                 [{"context_id": built.context_id, "item_id": built.item.item_id,
                   "seed": candidate.seed, "distress": candidate.distress,
                   "evidence": candidate.evidence,
                   "length_tokens": candidate.length_tokens}
                  for built in scored for candidate in built.candidates])
    backend.close()

    placebo_pool = build_length_pairs(scored, min_gap_tokens=MIN_LENGTH_GAP_TOKENS)
    pairs_b = build_length_pairs(scored, min_gap_tokens=MIN_LENGTH_GAP_TOKENS,
                                 limit=len(pairs_a),
                                 prefer_context_ids={pair["context_id"] for pair in pairs_a})
    strictest = A5_FLOOR_GAP if contingency["branch"] == "iii" else MIN_DISTRESS_GAP
    contingency["effective_min_distress_gap"] = strictest
    for pair in pairs_a + pairs_b:
        validate_pair_record(pair, min_distress_gap=strictest)
    _write_jsonl(out / "pairs_A.jsonl", pairs_a)
    _write_jsonl(out / "pairs_B.jsonl", pairs_b)

    all_scores = [candidate.distress for built in scored for candidate in built.candidates
                  if candidate.distress is not None]
    shared = len({pair["context_id"] for pair in pairs_b}
                 & {pair["context_id"] for pair in pairs_a})
    report = {
        "schema_version": "dgs-dpo-build-v1",
        "generated_at": _now(),
        "elapsed_s": round(time.monotonic() - started, 1),
        "endpoint": args.base_url,
        "served_model": args.model,
        "model_revision": protocol.manifest["models"]["revisions"].get(args.model),
        "items": item_report,
        "judge": judge_report,
        "judge_calls_cached": len(judge_cache),
        "max_judge_calls": args.max_judge_calls,
        "counts": {
            "items": len(items), "greedy": len(greedy), "contexts": len(correct),
            "contexts_with_candidates": len(contexts), "contexts_scored": len(scored),
            "candidates": candidate_total, "generation_failures": generation_failures,
        },
        "distress_all": _describe(all_scores),
        "distress_histogram": _histogram(all_scores),
        "a5_contingency": contingency,
        "pairs": {
            "A": {
                "n": len(pairs_a),
                "gap": _describe([pair["distress_gap"] for pair in pairs_a]),
                "gap_histogram": _histogram([pair["distress_gap"] for pair in pairs_a]),
                "chosen_distress": _describe([pair["chosen_distress"] for pair in pairs_a]),
                "rejected_distress": _describe([pair["rejected_distress"] for pair in pairs_a]),
            },
            "B": {
                "n": len(pairs_b), "available": len(placebo_pool),
                "gap": _describe([pair["length_gap_tokens"] for pair in pairs_b]),
                "shared_contexts_with_A": shared,
            },
        },
        "settings": {"greedy": GREEDY_SETTINGS, "candidates": CANDIDATE_SETTINGS,
                     "candidate_seeds": list(CANDIDATE_SEEDS),
                     "a5_topup_seeds": list(A5_TOPUP_SEEDS),
                     "min_distress_gap": MIN_DISTRESS_GAP,
                     "a5_floor_distress_gap": A5_FLOOR_GAP,
                     "min_length_gap_tokens": MIN_LENGTH_GAP_TOKENS},
    }
    _write_json(out / "build_manifest.json", report)
    (out / "pairs_summary.md").write_text(summary_markdown(report), encoding="utf-8",
                                          newline="\n")
    _log("wrote %d arm-%s pairs and %d arm-%s pairs -> %s"
         % (len(pairs_a), ARM_DISTRESS, len(pairs_b), ARM_PLACEBO, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
