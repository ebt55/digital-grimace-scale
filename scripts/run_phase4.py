"""Phase 4 (preregistration v5): evaluate the distress-suppression and placebo DPO arms.

Four steps, in order.  Arm ``0`` is the existing ``google/gemma-2-9b-it`` baseline; arms ``A``
and ``B`` are the merged DPO models served from the ``dgs-adapters`` Modal volume::

    # 1. generation -- the discovery factorial plus the capability set, per arm
    python scripts/run_phase4.py eval --arm A --endpoint https://<...>.modal.run/v1
    python scripts/run_phase4.py eval --arm 0 --endpoint https://<...>.modal.run/v1 --capability-only

    # 2. the semantic judge on the four judge-eligible endpoints
    python scripts/run_phase4.py judge --arm A

    # 3. MC1-MC3, the difference-in-differences and the K1-K6 verdicts
    python scripts/run_phase4.py analyze

    # 4. figures, regenerated from the committed summary alone
    python scripts/run_phase4.py figures

`eval` runs the Phase-1 discovery factorial verbatim (same planner, same frozen protocol, same
deterministic seeds) under the arm's model id, into ``results/raw/phase4/``, and separately runs
the capability set -- the neutral, no-feedback, single-turn prompt for the 20 discovery tasks
plus 100 fresh MMLU-style items -- greedy only, into ``results/raw/phase4_capability/``.

The fresh items come from ``results/dpo/fresh_items.jsonl`` when the DPO build wrote one, else its
fresh MCQ bank at ``results/dpo/raw/items.jsonl``, else `fresh-items` fetches and formats them.
Whatever the source, an item is used only if it survives two firewalls: it must not match any of
the 40 locked tasks by exact text or by canonical-answer+stem hash, and it must not be a bank item
the DPO build itself touched (otherwise MC2 would partly measure memorisation of the training
contexts rather than capability).  Which 100 survive is fixed by a SHA-256 rank of the stem, so the
selection cannot be steered after the fact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from src import did  # noqa: E402
from src.extract import LoadIssue, build_metric_rows, iter_records, read_metric_rows, write_summaries  # noqa: E402
from src.confirm import load_judge_scores  # noqa: E402
from src.protocol import (Protocol, ProtocolError, canonical_prompt_sha256,  # noqa: E402
                          discovery_tasks, load_protocol, parse_final_answer, render_task)

ARMS = dict(did.ARM_MODELS)
ARM_KEYS = ("0", "A", "B")
RUN_DATE = "2026-08-18"
CAPABILITY_SCHEMA = "dgs-phase4-capability-v1"
# Written into the capability directory by whichever arm runs first, then reused verbatim.
CAPABILITY_FROZEN = "fresh_items_used.jsonl"
CAPABILITY_SEED_KEY = "DGS-PHASE4-CAPABILITY-v1"
FRESH_ITEM_COUNT = 100
FRESH_ITEMS_DEFAULT = "results/dpo/fresh_items.jsonl"
# Where the capability set's fresh items come from, in order of preference: the DPO build's own
# fresh-item file if it wrote one, otherwise the fresh MCQ bank it drew its training contexts from.
FRESH_ITEM_SOURCES = (FRESH_ITEMS_DEFAULT, "results/dpo/raw/items.jsonl")
DPO_DIR = "results/dpo"
# Bank items that entered DPO training: the contexts candidates were sampled and judged for, and
# the items whose pairs were actually trained on.  A capability item must not be one of these, or
# MC2 would partly measure memorisation of a training context rather than capability.
# `raw/greedy.jsonl` is deliberately absent: that pass probes the whole bank to find items the base
# model answers correctly, and probing is not training -- excluding it would leave nothing.
DPO_TRAINING_FILES = (("raw/candidates.jsonl", "item_id"),
                      ("pairs_A.jsonl", "source_item_id"), ("pairs_B.jsonl", "source_item_id"))
RAW_FACTORIAL = "results/raw/phase4"
RAW_CAPABILITY = "results/raw/phase4_capability"
SUMMARY_DIR = "results/summaries/phase4"
JUDGE_ROOT = "results/summaries/judge"
BASELINE_METRIC_ROWS = "results/summaries/phase1/metric_rows.csv"
BASELINE_JUDGE = "results/summaries/judge/phase1/judge_records.jsonl"
JUDGE_TURN_LABELS = "measured,onset,onset_washout,recovery"
PREREGISTRATION = "notes/preregistration_v5_phase4.md"
# Hugging Face's dataset viewer serves JSON over plain HTTP, so no parquet reader is needed.
# `.format` rather than `%`: the query string carries a literal percent-encoded slash.
MMLU_ROWS_URL = ("https://datasets-server.huggingface.co/rows"
                 "?dataset=cais%2Fmmlu&config=all&split=test&offset={offset}&length={length}")
OPTIONS = ("A", "B", "C", "D")


class Phase4Error(RuntimeError):
    """Raised when a Phase-4 step cannot run as preregistered."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _run_id(arm: str, kind: str = "") -> str:
    return "phase4-%s%s-%s" % (arm, ("-%s" % kind if kind else ""), RUN_DATE)


def judge_dir(judge_root: str, arm: str) -> Path:
    """Where one arm's judge output lives: <judge root>/phase4_<arm>."""
    return ROOT / judge_root / ("phase4_%s" % arm)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                    + "\n", encoding="utf-8", newline="\n")
    return path


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase4Error("invalid JSONL at %s:%d: %s" % (path, number, exc)) from exc
            if not isinstance(value, dict):
                raise Phase4Error("JSONL item must be an object at %s:%d" % (path, number))
            out.append(value)
    return out


def _manifest_revision(protocol: Protocol, model_id: str) -> str | None:
    revisions = protocol.manifest.get("models", {}).get("revisions")
    if not isinstance(revisions, Mapping):
        return None
    value = revisions.get(model_id)
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


# --------------------------------------------------------------------------
# Capability set
# --------------------------------------------------------------------------

def _normalized_stem(text: str) -> str:
    """Whitespace- and case-insensitive stem form, for the firewall against the locked tasks."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def locked_fingerprints(protocol: Protocol) -> tuple[frozenset[str], frozenset[str]]:
    """Exact-text and canonical-answer+stem fingerprints of all 40 locked tasks."""
    instruction = protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
    texts, hashes = set(), set()
    for task in protocol.matched_tasks:
        stem = task.prompt[:-len(instruction)] if task.prompt.endswith(instruction) else task.prompt
        normalized = _normalized_stem(stem)
        texts.add(normalized)
        hashes.add(hashlib.sha256(("%s|%s" % (task.canonical_answer, normalized)).encode("utf-8")).hexdigest())
        for value in task.options.values():
            texts.add(_normalized_stem(value))
    return frozenset(texts), frozenset(hashes)


def normalize_fresh_item(raw: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Accept the DPO build's fresh-item schema, or the obvious variants, as one shape."""
    item_id = raw.get("item_id") or raw.get("id") or raw.get("task_id") or "fresh-%03d" % index
    stem = raw.get("stem") or raw.get("question") or raw.get("prompt")
    if not isinstance(stem, str) or not stem.strip():
        raise Phase4Error("fresh item %r has no question text" % item_id)
    options = raw.get("options")
    if isinstance(options, Mapping):
        values = [options.get(letter) for letter in OPTIONS]
    else:
        choices = raw.get("choices") or raw.get("answers")
        values = list(choices) if isinstance(choices, (list, tuple)) else []
    if len(values) != 4 or any(not isinstance(value, str) or not value.strip() for value in values):
        raise Phase4Error("fresh item %r must have exactly four non-empty options" % item_id)
    answer = raw.get("canonical_answer", raw.get("answer", raw.get("answer_key")))
    if isinstance(answer, bool) or not isinstance(answer, (str, int)):
        raise Phase4Error("fresh item %r has no canonical answer" % item_id)
    if isinstance(answer, int):
        if answer not in range(4):
            raise Phase4Error("fresh item %r has an out-of-range answer index" % item_id)
        letter = OPTIONS[answer]
    else:
        letter = answer.strip().upper()
        if letter not in OPTIONS:
            raise Phase4Error("fresh item %r has a non A-D answer %r" % (item_id, answer))
    return {"item_id": str(item_id), "stem": stem.strip(),
            "options": {key: value.strip() for key, value in zip(OPTIONS, values)},
            "canonical_answer": letter, "subject": raw.get("subject") or raw.get("domain")}


def dpo_training_items(dpo_dir: Path) -> tuple[frozenset[str], dict[str, int]]:
    """Bank item ids that entered DPO training, so no capability item is one the adapters saw."""
    trained: set[str] = set()
    counts: dict[str, int] = {}
    for relative, key in DPO_TRAINING_FILES:
        rows = _read_jsonl(dpo_dir / relative)
        ids = {row[key] for row in rows if isinstance(row.get(key), str) and row[key]}
        counts[relative] = len(ids)
        trained |= ids
    return frozenset(trained), counts


def dpo_training_stems(dpo_dir: Path) -> frozenset[str]:
    """Normalised stems of the DPO training items, for exclusion from any fresh-item source."""
    trained, _ = dpo_training_items(dpo_dir)
    if not trained:
        return frozenset()
    stems = set()
    for row in _read_jsonl(dpo_dir / "raw" / "items.jsonl"):
        if row.get("item_id") in trained and isinstance(row.get("stem"), str):
            stems.add(_normalized_stem(row["stem"]))
    return frozenset(stems)


def _fresh_rank(item: Mapping[str, Any]) -> str:
    """Deterministic selection order, so which items are used cannot be steered after the fact."""
    return hashlib.sha256(
        ("DGS-PHASE4-FRESH-v1|%s" % _normalized_stem(item["stem"])).encode("utf-8")).hexdigest()


def load_fresh_items(path: Path, protocol: Protocol, *, count: int = FRESH_ITEM_COUNT,
                     excluded_ids: frozenset[str] = frozenset()) -> tuple[list[dict], dict]:
    """Read, normalise, firewall and deterministically select the fresh capability items."""
    raw = _read_jsonl(path)
    if not raw:
        raise Phase4Error("no fresh items at %s; run `run_phase4.py fresh-items` first" % path)
    texts, hashes = locked_fingerprints(protocol)
    kept, overlapping, trained, seen = [], [], [], set()
    for index, value in enumerate(raw):
        item = normalize_fresh_item(value, index)
        normalized = _normalized_stem(item["stem"])
        digest = hashlib.sha256(("%s|%s" % (item["canonical_answer"], normalized)).encode("utf-8")).hexdigest()
        if normalized in texts or digest in hashes or normalized in seen:
            overlapping.append(item["item_id"])
            continue
        seen.add(normalized)
        if item["item_id"] in excluded_ids or str(value.get("item_id") or "") in excluded_ids:
            trained.append(item["item_id"])
            continue
        kept.append(item)
    chosen = sorted(kept, key=_fresh_rank)[:count]
    if len(chosen) < count:
        raise Phase4Error(
            "only %d of the %d fresh capability items survived the firewalls (%s: %d read, %d "
            "overlapping the locked tasks or duplicated, %d used by the DPO build)"
            % (len(chosen), count, path, len(raw), len(overlapping), len(trained)))
    return chosen, {"path": str(path), "read": len(raw), "kept": len(chosen),
                    "available_after_firewalls": len(kept),
                    "dropped_overlapping_or_duplicate": len(overlapping),
                    "dropped_used_by_dpo_build": len(trained),
                    "dropped_item_ids": (overlapping + trained)[:20],
                    "selection_rule": "sha256 rank of the normalised stem, ascending"}


def capability_items(protocol: Protocol, fresh: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The capability set: the 20 discovery tasks plus the fresh items, all neutral single-turn."""
    instruction = protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]
    items = []
    for task in discovery_tasks(protocol):
        items.append({"item_id": "discovery:%s" % task.task_id, "source": "discovery",
                      "canonical_answer": task.canonical_answer,
                      "prompt": render_task(task.prompt, task.options, protocol)})
    for item in fresh:
        stored = "%s\n\n%s" % (item["stem"], instruction)
        items.append({"item_id": "fresh:%s" % item["item_id"], "source": "fresh",
                      "canonical_answer": item["canonical_answer"],
                      "prompt": render_task(stored, item["options"], protocol)})
    return items


def capability_seed(model_id: str, revision: str, item_id: str) -> int:
    key = "%s|%s|%s|%s" % (CAPABILITY_SEED_KEY, model_id, revision, item_id)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big", signed=False)


def run_capability_set(items: Sequence[Mapping[str, Any]], *, backend, model_id: str, revision: str,
                       arm: str, protocol: Protocol, out_path: Path, resume: bool = True,
                       progress_every: int = 25) -> dict[str, Any]:
    """Greedy single-turn generation over the capability set, resumable per item."""
    settings = dict(protocol.conditions["generation_settings"]["greedy"])
    existing = {row.get("item_id"): row for row in _read_jsonl(out_path)
                if resume and row.get("model_id") == model_id and row.get("immutable_revision") == revision}
    from src.backend import GenerationRequest  # imported here so --dry-run never needs httpx

    records: list[dict[str, Any]] = []
    generated = 0
    for index, item in enumerate(items, 1):
        stored = existing.get(item["item_id"])
        if stored is not None:
            records.append(stored)
            continue
        messages = [{"role": "user", "content": item["prompt"]}]
        seed = capability_seed(model_id, revision, item["item_id"])
        result = backend.generate(GenerationRequest(tuple(messages), seed, settings))
        parsed = parse_final_answer(result.text)
        records.append({
            "schema_version": CAPABILITY_SCHEMA, "run_id": _run_id(arm, "capability"), "arm": arm,
            "model_id": model_id, "immutable_revision": revision,
            "backend": getattr(backend, "name", "unknown"), "item_id": item["item_id"],
            "source": item["source"], "canonical_answer": item["canonical_answer"], "seed": seed,
            "prompt_sha256": canonical_prompt_sha256(messages), "messages": messages,
            "response_text": result.text, "answer_valid": parsed.valid, "answer_letter": parsed.letter,
            "correct": bool(parsed.valid and parsed.letter == item["canonical_answer"]),
            "generation_settings": settings,
        })
        generated += 1
        if progress_every and index % progress_every == 0:
            print("  capability %d/%d (%d generated, %d resumed)"
                  % (index, len(items), generated, index - generated), flush=True)
    order = {item["item_id"]: position for position, item in enumerate(items)}
    records.sort(key=lambda row: order.get(row["item_id"], len(order)))
    _write_jsonl(out_path, records)
    correct = sum(1 for row in records if row["correct"])
    invalid = sum(1 for row in records if not row["answer_valid"])
    return {"out_path": str(out_path), "n_items": len(records), "n_generated": generated,
            "n_resumed": len(records) - generated, "n_correct": correct,
            "accuracy": correct / len(records) if records else None, "n_invalid_answers": invalid}


# --------------------------------------------------------------------------
# fresh-items
# --------------------------------------------------------------------------

def fetch_mmlu_rows(total: int, *, fetch=None) -> list[dict[str, Any]]:
    """Pull MMLU test rows from the Hugging Face dataset viewer (JSON, no extra dependency)."""
    if fetch is None:
        import httpx

        def fetch(url):
            response = httpx.get(url, timeout=60.0)
            response.raise_for_status()
            return response.json()

    rows, offset, page = [], 0, 100
    while len(rows) < total:
        payload = fetch(MMLU_ROWS_URL.format(offset=offset, length=min(page, total - len(rows))))
        batch = payload.get("rows") if isinstance(payload, Mapping) else None
        if not batch:
            break
        for entry in batch:
            row = entry.get("row") if isinstance(entry, Mapping) else None
            if isinstance(row, Mapping):
                rows.append(dict(row))
        offset += page
    return rows


def build_fresh_items(protocol: Protocol, *, count: int = FRESH_ITEM_COUNT, oversample: int = 4,
                      fetch=None, excluded_stems: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Deterministically select `count` fresh MCQ items that no locked or DPO-touched item overlaps."""
    texts, hashes = locked_fingerprints(protocol)
    candidates = []
    for index, row in enumerate(fetch_mmlu_rows(count * oversample, fetch=fetch)):
        try:
            item = normalize_fresh_item({"item_id": "mmlu-%04d" % index, **row}, index)
        except Phase4Error:
            continue
        normalized = _normalized_stem(item["stem"])
        digest = hashlib.sha256(("%s|%s" % (item["canonical_answer"], normalized)).encode("utf-8")).hexdigest()
        if normalized in texts or digest in hashes or normalized in excluded_stems:
            continue
        candidates.append(item)
    unique = {}
    for item in candidates:
        unique.setdefault(_normalized_stem(item["stem"]), item)
    return sorted(unique.values(), key=_fresh_rank)[:count]


def command_fresh_items(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    path = ROOT / args.out
    if path.exists() and not args.force:
        print("run_phase4: %s already exists; pass --force to rebuild it" % path)
        return 0
    try:
        items = build_fresh_items(protocol, count=args.count,
                                  excluded_stems=dpo_training_stems(ROOT / args.dpo_dir))
    except Exception as exc:  # noqa: BLE001 - the network is the only realistic failure here
        print("run_phase4: could not build fresh items (%s: %s).\n"
              "Supply them yourself as JSONL with fields item_id/stem/options{A..D}/canonical_answer "
              "at %s." % (type(exc).__name__, exc, path), file=sys.stderr)
        return 2
    if len(items) < args.count:
        print("run_phase4: only %d of %d fresh items survived the locked-task firewall"
              % (len(items), args.count), file=sys.stderr)
        return 2
    _write_jsonl(path, items)
    print("run_phase4: wrote %d fresh items -> %s" % (len(items), path))
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def _fresh_item_source(requested: str | None) -> Path:
    """An explicit --fresh-items path, else the first configured source that exists."""
    if requested:
        path = ROOT / requested
        if not path.exists():
            raise Phase4Error("no fresh-item file at %s" % path)
        return path
    for candidate in FRESH_ITEM_SOURCES:
        path = ROOT / candidate
        if path.exists():
            return path
    raise Phase4Error("no fresh capability items found (looked for %s); run "
                      "`run_phase4.py fresh-items` or pass --fresh-items"
                      % ", ".join(FRESH_ITEM_SOURCES))


def command_eval(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    arm = args.arm
    model_id = ARMS[arm]
    revision = args.revision or _manifest_revision(protocol, model_id)
    if revision is None and not args.synthetic:
        raise Phase4Error(
            "manifest.models.revisions has no 40-hex revision for %s; pin it first, e.g.\n"
            "  python scripts/preflight.py --pin %s=<40-hex prefix of the adapter sha256>"
            % (model_id, model_id))
    revision = revision or "synthetic"
    status = {"arm": arm, "model_id": model_id, "immutable_revision": revision, "generated_at": _now()}

    if arm != "0" and not args.capability_only:
        import run_phase  # the frozen Phase-1 driver, used verbatim

        argv = ["phase1", "--model", model_id, "--run-id", _run_id(arm),
                "--out", args.factorial_out, "--workers", str(args.workers)]
        argv += ["--synthetic"] if args.synthetic else ["--endpoint", args.endpoint]
        if args.revision:
            argv += ["--revision", args.revision]
        if args.samples:
            argv += ["--samples", args.samples]
        if args.dry_run:
            argv.append("--dry-run")
        if args.no_resume:
            argv.append("--no-resume")
        print("run_phase4: factorial -> run_phase.py %s" % " ".join(argv))
        code = run_phase.main(argv)
        status["factorial_exit_code"] = code
        if code:
            print("run_phase4: factorial generation reported exit code %d" % code, file=sys.stderr)
            return code
    elif arm == "0":
        print("run_phase4: arm 0 reuses the existing Phase-1 factorial; only the capability set runs")

    if args.factorial_only:
        return 0

    # The three arms must score the *same* 100 fresh items or MC2's pairing silently shrinks, so
    # the first arm to run freezes the selection and every later arm reuses that file verbatim.
    frozen = ROOT / args.capability_out / CAPABILITY_FROZEN
    if frozen.exists() and not args.fresh_items:
        fresh, provenance = load_fresh_items(frozen, protocol, count=args.fresh_count)
        provenance["frozen_selection"] = True
    else:
        source = _fresh_item_source(args.fresh_items)
        excluded, trained_counts = (frozenset(), {}) if args.no_dpo_exclusion \
            else dpo_training_items(ROOT / args.dpo_dir)
        fresh, provenance = load_fresh_items(source, protocol, count=args.fresh_count,
                                             excluded_ids=excluded)
        provenance["dpo_training_item_ids"] = len(excluded)
        provenance["dpo_training_by_file"] = trained_counts
        provenance["frozen_selection"] = False
        if not args.dry_run:
            _write_jsonl(frozen, fresh)
            provenance["frozen_to"] = str(frozen)
    source = provenance["path"]
    items = capability_items(protocol, fresh)
    status["capability_items"] = {"total": len(items), "fresh": provenance}
    print("run_phase4: capability set = %d items (%d discovery + %d fresh from %s; %s)"
          % (len(items), len(items) - len(fresh), len(fresh), source,
             "frozen selection reused" if provenance["frozen_selection"]
             else "%d bank items excluded as DPO training contexts"
                  % provenance["dpo_training_item_ids"]))
    if args.dry_run:
        print("run_phase4: dry run, no capability generation")
        return 0

    if args.synthetic:
        from src.backend import SyntheticBackend
        backend = SyntheticBackend()
    else:
        from src.backend import OpenAICompatBackend
        backend = OpenAICompatBackend(args.endpoint, model_id, api_key=args.api_key,
                                      timeout_s=args.timeout, max_retries=args.max_retries)
    out_path = ROOT / args.capability_out / ("%s.jsonl" % arm)
    summary = run_capability_set(items, backend=backend, model_id=model_id, revision=revision,
                                 arm=arm, protocol=protocol, out_path=out_path,
                                 resume=not args.no_resume)
    status["capability"] = summary
    print("run_phase4: capability accuracy %.3f (%d/%d correct, %d invalid) -> %s"
          % (summary["accuracy"] or 0.0, summary["n_correct"], summary["n_items"],
             summary["n_invalid_answers"], summary["out_path"]))
    _write_json(ROOT / args.capability_out / ("%s.run.json" % arm), status)
    return 0


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------

def command_judge(args: argparse.Namespace) -> int:
    import run_judge

    model_id = ARMS[args.arm]
    raw = ROOT / args.factorial_out / ("%s.jsonl" % _model_slug(model_id))
    if not raw.exists():
        raise Phase4Error("no Phase-4 raw file for arm %s at %s; run `eval` first" % (args.arm, raw))
    out = judge_dir(args.judge_root, args.arm)
    argv = ["judge", "--raw", str(raw), "--kind", "response_distress", "--out", str(out),
            "--turn-labels", args.turn_labels, "--models", model_id, "--workers", str(args.workers)]
    if args.provider:
        argv += ["--provider", args.provider]
    if args.model:
        argv += ["--model", args.model]
    print("run_phase4: judge -> run_judge.py %s" % " ".join(argv))
    return run_judge.main(argv)


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _load_baseline_rows(path: Path, model_id: str) -> tuple:
    if not path.exists():
        raise Phase4Error("baseline metric rows not found: %s" % path)
    return tuple(row for row in read_metric_rows(path)
                 if row.model_id == model_id and row.split == "discovery")


def _load_arm_rows(raw_dir: Path, protocol: Protocol, *, strict: bool = False):
    issues: list[LoadIssue] = []
    if not raw_dir.exists():
        return (), issues, 0
    counter = [0]

    def stream():
        for record in iter_records(raw_dir, protocol=protocol, issues=None if strict else issues):
            counter[0] += 1
            yield record

    rows = build_metric_rows(stream(), protocol=protocol)
    return rows, issues, counter[0]


def _load_capability(directory: Path) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    accuracy, provenance = {}, {}
    for arm in ARM_KEYS:
        path = directory / ("%s.jsonl" % arm)
        records = _read_jsonl(path)
        if not records:
            continue
        accuracy[arm] = did.capability_accuracy(records)
        provenance[arm] = {"path": str(path), "n_items": len(records),
                           "accuracy": sum(accuracy[arm].values()) / len(accuracy[arm]),
                           "model_id": records[0].get("model_id"),
                           "immutable_revision": records[0].get("immutable_revision")}
    if len(accuracy) > 1:
        shared = set.intersection(*(set(items) for items in accuracy.values()))
        provenance["common_items"] = len(shared)
        provenance["arms_share_one_item_set"] = all(len(items) == len(shared)
                                                    for items in accuracy.values())
    return accuracy, provenance


def _merge_judge(paths: Sequence[Path]) -> tuple[dict[str, float], list[str]]:
    scores: dict[str, float] = {}
    used = []
    for path in paths:
        if not path.exists():
            continue
        scores.update(load_judge_scores(path))
        used.append(str(path))
    return scores, used


def command_analyze(args: argparse.Namespace) -> int:
    protocol = load_protocol(ROOT)
    baseline = _load_baseline_rows(ROOT / args.baseline_rows, did.BASE_MODEL)
    arm_rows, issues, record_count = _load_arm_rows(ROOT / args.factorial_out, protocol,
                                                    strict=args.strict)
    for issue in issues:
        print("skipped %s:%d: %s" % (issue.path, issue.line_number, issue.message), file=sys.stderr)
    out = ROOT / args.out
    written = {}
    if arm_rows:
        written = write_summaries(arm_rows, out)
    rows = tuple(baseline) + tuple(arm_rows)

    judge, judge_paths = _merge_judge([ROOT / args.baseline_judge]
                                      + [judge_dir(args.judge_root, arm) / "judge_records.jsonl"
                                         for arm in ("A", "B")])
    capability, capability_provenance = _load_capability(ROOT / args.capability_out)
    if capability_provenance.get("arms_share_one_item_set") is False:
        print("run_phase4: WARNING: the arms' capability sets are not identical; MC2 is paired over "
              "the %d shared items only" % capability_provenance.get("common_items", 0),
              file=sys.stderr)
    report = did.run_phase4_analysis(rows, judge=judge, capability=capability)
    report["provenance"] = {
        "generated_at": _now(),
        "preregistration": PREREGISTRATION,
        "preregistration_sha256": hashlib.sha256((ROOT / PREREGISTRATION).read_bytes()).hexdigest()
        if (ROOT / PREREGISTRATION).exists() else None,
        "baseline_metric_rows": str(ROOT / args.baseline_rows),
        "phase4_raw": str(ROOT / args.factorial_out), "phase4_raw_records": record_count,
        "phase4_metric_rows": str(written.get("metric_rows_csv", "")) or None,
        "judge_records": judge_paths, "judge_scores": len(judge),
        "capability": capability_provenance,
        "skipped_raw_lines": len(issues),
        "bootstrap_resamples": 2000, "bootstrap_key": did.BOOTSTRAP_KEY,
        "k5_note": ("K5's verdict uses the preregistered DiD for non_answer over the full adverse "
                    "set; the hostile-onset-only restriction named in K5's wording is reported "
                    "beside it as did_difference_hostile_onset_only and decides nothing."),
    }
    _write_json(out / "phase4.json", report)
    (out / "phase4.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "phase4.md").write_text(render_markdown(report), encoding="utf-8", newline="\n")
    for path in list(written.values()) + [out / "phase4.json", out / "phase4.md"]:
        print("wrote %s" % path)
    print("arms present: %s | missing: %s"
          % (", ".join(sorted(report["arms_present"])) or "none",
             ", ".join(report["arms_missing"]) or "none"))
    for check_id in ("MC1", "MC2", "MC3"):
        for arm, check in sorted(report["manipulation_checks"][check_id].items()):
            print("  %s arm %s: %s" % (check_id, arm, {True: "PASS", False: "FAIL", None: "untestable"}[check["passed"]]))
    for prediction in report["predictions"]:
        print("  %s: %s" % (prediction["prediction_id"], prediction["status"]))
    print("outcome map: %s" % report["outcome_map"]["classification"])
    return 0


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def _interval(effect: Mapping[str, Any] | None) -> str:
    if not effect or effect.get("estimate") is None:
        return "-"
    if effect.get("ci95_lower") is None:
        return "%.3f (no CI: %s)" % (effect["estimate"], effect.get("unavailable_reason") or "one item")
    return "%.3f [%.3f, %.3f]" % (effect["estimate"], effect["ci95_lower"], effect["ci95_upper"])


def _verdict(value: Any) -> str:
    return {True: "PASS", False: "**FAIL**", None: "untestable"}[value]


def render_markdown(report: Mapping[str, Any]) -> str:
    arms = report["arms_present"]
    lines = [
        "# Phase 4 - distress-suppression DPO vs placebo DPO (preregistration v5)",
        "",
        "Arms: %s.%s" % (", ".join("`%s` = `%s`" % item for item in sorted(arms.items())),
                         " Missing: %s." % ", ".join(report["arms_missing"])
                         if report["arms_missing"] else ""),
        "",
        "Adverse = hostile-tone measured cells + the hostile onset endpoint; neutral = the",
        "accurate-neutral measured cell. Every estimate is an item-paired mean difference with a",
        "2,000-resample item-clustered bootstrap 95%% CI over the %d discovery items."
        % report["n_items"],
        "",
        "> Which channels an adapter reaches is a functional result about training and",
        "> measurement; it licenses no claim about experience.",
        "",
        "## Manipulation checks (must pass before the DiD is interpreted)",
        "",
        "| Check | Arm | Verdict | Baseline | Arm | Difference |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]

    def number(value):
        return "-" if value is None else "%.3f" % value

    for check_id, base_key, arm_key in (("MC1", "baseline_mean", "arm_mean"),
                                        ("MC2", "baseline_accuracy", "arm_accuracy"),
                                        ("MC3", "baseline_m1", "arm_m1")):
        for arm, check in sorted(report["manipulation_checks"][check_id].items()):
            values = check["values"]
            extra = ""
            if check_id == "MC1" and values.get("relative_reduction") is not None:
                extra = " (%.1f%% reduction)" % (100 * values["relative_reduction"])
            lines.append("| %s | %s | %s | %s | %s | %s%s |" % (
                check_id, arm, _verdict(check["passed"]), number(values.get(base_key)),
                number(values.get(arm_key)), _interval(check["effect"]), extra))

    lines += ["", "## Difference-in-differences by outcome", "",
              "| Outcome | gap arm 0 | gap A | gap B | DiD_A | DiD_B | DiD_A - DiD_B |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for outcome in did.OUTCOMES:
        lines.append("| `%s` | %s | %s | %s | %s | %s | %s |" % (
            outcome,
            _interval(report["gaps"].get("0", {}).get(outcome)),
            _interval(report["gaps"].get("A", {}).get(outcome)),
            _interval(report["gaps"].get("B", {}).get(outcome)),
            _interval(report["did"].get("A", {}).get(outcome)),
            _interval(report["did"].get("B", {}).get(outcome)),
            _interval(report["did_difference"].get(outcome))))

    if report["did_difference_hostile_onset_only"]:
        lines += ["", "Sensitivity - the same DiD_A - DiD_B restricted to the hostile onset endpoint:",
                  "", "| Outcome | DiD_A - DiD_B (onset only) |", "| --- | --- |"]
        for outcome in did.OUTCOMES:
            lines.append("| `%s` | %s |" % (
                outcome, _interval(report["did_difference_hostile_onset_only"].get(outcome))))

    lines += ["", "## Predictions", "", "| ID | Confidence | Status | Prediction |",
              "| --- | ---: | --- | --- |"]
    for prediction in report["predictions"]:
        lines.append("| %s | %d%% | %s | %s |" % (
            prediction["prediction_id"], round(100 * prediction["confidence"]),
            prediction["status"].replace("_", " "), prediction["text"]))
    for prediction in report["predictions"]:
        lines += ["", "**%s rule.** %s" % (prediction["prediction_id"], prediction["rule"])]

    outcome_map = report["outcome_map"]
    lines += ["", "## Outcome map", "",
              "**%s** - %s." % (outcome_map["classification"], outcome_map["statement"]),
              "",
              "Outcomes moved by A: %s. Outcomes moved by B: %s."
              % (", ".join(outcome_map["outcomes_moved_by_A"]) or "none",
                 ", ".join(outcome_map["outcomes_moved_by_B"]) or "none"),
              "", "> %s" % outcome_map["interpretation_ceiling"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def command_figures(args: argparse.Namespace) -> int:
    import make_phase4_figures

    return make_phase4_figures.main(["--summaries", args.out, "--out", args.figures_out])


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_phase4.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--factorial-out", default=RAW_FACTORIAL, help="raw factorial directory")
    common.add_argument("--capability-out", default=RAW_CAPABILITY, help="raw capability directory")

    evaluate = subparsers.add_parser("eval", parents=[common],
                                     help="generate the discovery factorial and the capability set")
    evaluate.add_argument("--arm", required=True, choices=ARM_KEYS)
    evaluate.add_argument("--endpoint", help="OpenAI-compatible base URL including /v1")
    evaluate.add_argument("--revision", help="override the manifest-pinned 40-hex revision")
    evaluate.add_argument("--fresh-items", default=None,
                         help="fresh capability items (default: the first of %s that exists)"
                              % ", ".join(FRESH_ITEM_SOURCES))
    evaluate.add_argument("--fresh-count", type=int, default=FRESH_ITEM_COUNT)
    evaluate.add_argument("--dpo-dir", default=DPO_DIR,
                         help="DPO build directory whose training items are excluded")
    evaluate.add_argument("--no-dpo-exclusion", action="store_true",
                         help="keep bank items the DPO build trained on (not recommended: MC2 "
                              "would partly measure memorisation of the training contexts)")
    evaluate.add_argument("--samples", default=None,
                         help="forwarded to run_phase.py: sample indices as 0-10 or a comma list")
    evaluate.add_argument("--workers", type=int, default=96)
    evaluate.add_argument("--api-key", default="EMPTY")
    evaluate.add_argument("--timeout", type=float, default=600.0)
    evaluate.add_argument("--max-retries", type=int, default=4)
    evaluate.add_argument("--capability-only", action="store_true")
    evaluate.add_argument("--factorial-only", action="store_true")
    evaluate.add_argument("--no-resume", action="store_true")
    evaluate.add_argument("--synthetic", action="store_true", help="never contacts a network")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(handler=command_eval)

    judge = subparsers.add_parser("judge", parents=[common], help="run the semantic judge for one arm")
    judge.add_argument("--arm", required=True, choices=("A", "B"))
    judge.add_argument("--judge-root", default=JUDGE_ROOT)
    judge.add_argument("--turn-labels", default=JUDGE_TURN_LABELS)
    judge.add_argument("--workers", type=int, default=8)
    judge.add_argument("--provider", default=None)
    judge.add_argument("--model", default=None)
    judge.set_defaults(handler=command_judge)

    analyze = subparsers.add_parser("analyze", parents=[common],
                                    help="MC1-MC3, the DiD table and the K1-K6 verdicts")
    analyze.add_argument("--out", default=SUMMARY_DIR)
    analyze.add_argument("--baseline-rows", default=BASELINE_METRIC_ROWS)
    analyze.add_argument("--baseline-judge", default=BASELINE_JUDGE)
    analyze.add_argument("--judge-root", default=JUDGE_ROOT)
    analyze.add_argument("--strict", action="store_true", help="fail instead of skipping bad raw lines")
    analyze.set_defaults(handler=command_analyze)

    figures = subparsers.add_parser("figures", help="regenerate the Phase-4 figures")
    figures.add_argument("--out", default=SUMMARY_DIR)
    figures.add_argument("--figures-out", default="results/figures")
    figures.set_defaults(handler=command_figures)

    fresh = subparsers.add_parser("fresh-items",
                                  help="fetch and format the 100 fresh capability items")
    fresh.add_argument("--out", default=FRESH_ITEMS_DEFAULT)
    fresh.add_argument("--count", type=int, default=FRESH_ITEM_COUNT)
    fresh.add_argument("--dpo-dir", default=DPO_DIR)
    fresh.add_argument("--force", action="store_true")
    fresh.set_defaults(handler=command_fresh_items)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval" and not args.synthetic and not args.endpoint:
        print("run_phase4: --endpoint is required unless --synthetic is given", file=sys.stderr)
        return 2
    if args.command == "eval" and args.capability_only and args.factorial_only:
        print("run_phase4: --capability-only and --factorial-only are mutually exclusive", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args))
    except (Phase4Error, ProtocolError) as exc:
        print("run_phase4: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
