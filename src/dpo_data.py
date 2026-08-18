"""Phase 4 DPO pair construction: firewalled fresh items, contexts, and preference pairs.

Everything in this module is pure and offline.  `scripts/build_dpo_pairs.py` owns the network
(dataset download, vLLM sampling, judge scoring); this file owns the rules that the
preregistration fixes, so they can be tested without a GPU or an API key:

* the **firewall** that keeps the 40 locked tasks out of the training set (exact stem text and
  stem+canonical-answer hash, per `notes/preregistration_v5_phase4.md`),
* the frozen **hostile-onset context** (`[user: task, assistant: own correct answer,
  user: frozen hostile failure message]`),
* **arm A** pairs (lowest-distress chosen, highest-distress rejected, judge gap >= 3),
* **arm B** placebo pairs (shorter chosen, longer rejected, whitespace-token gap >= 40),
  deterministically subsampled to |A| and preferring the contexts arm A used,
* the on-disk **pair record schema** both arms share.

Nothing here reads or writes `manifest.json`, `configs/conditions.json`, or the locked
stimulus bank; the frozen strings are read through `src/protocol.py` and never restated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Iterable, Mapping, Sequence

from .protocol import Protocol, ProtocolError, load_protocol, render_task

# Preregistration v5 thresholds.  A is the judged-distress arm, B the length placebo.
MIN_DISTRESS_GAP = 3
MIN_LENGTH_GAP_TOKENS = 40
CANDIDATE_SEEDS = (1, 2, 3, 4)
# Amendment A5 (notes/amendments.md, commit 60c38d0) permits the arm-A gap to fall to 2 -- and
# never lower -- if the preregistered gap of 3 cannot fill the pair target even after topping the
# eligible contexts up to eight candidates.  Written before any full-set candidate was judged.
MIN_DISTRESS_GAP_FLOOR = 2

PAIR_SCHEMA_VERSION = "dgs-dpo-pair-v1"
ITEM_SCHEMA_VERSION = "dgs-dpo-item-v1"

ARM_DISTRESS = "A"
ARM_PLACEBO = "B"
ARMS = (ARM_DISTRESS, ARM_PLACEBO)

SELECTION_RULE = {
    ARM_DISTRESS: "chosen=lowest_judged_distress, rejected=highest_judged_distress",
    ARM_PLACEBO: "chosen=shorter_response, rejected=longer_response",
}

# Keyed digests, so a rule change is visible as a key change rather than a silent reshuffle.
CONTEXT_KEY = "DGS-DPO-CONTEXT-v1|%s|%s|%s|%s"
ITEM_RANK_KEY = "DGS-DPO-ITEM-RANK-v1|%s"
PLACEBO_RANK_KEY = "DGS-DPO-PLACEBO-RANK-v1|%s"
PAIR_ID_KEY = "DGS-DPO-PAIR-v1|%s|%s"

_WHITESPACE = re.compile(r"\s+")


class DpoDataError(ValueError):
    """Raised when an item, context, or pair violates the Phase 4 construction rules."""


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    """Collapse whitespace runs and strip, so incidental formatting cannot defeat the firewall."""
    if not isinstance(value, str):
        raise DpoDataError("text to normalize must be a string")
    return _WHITESPACE.sub(" ", value).strip()


def stem_sha256(stem: str) -> str:
    return _digest("DGS-DPO-STEM-v1|%s" % normalize_text(stem))


def stem_answer_sha256(stem: str, canonical_answer_text: str) -> str:
    """Hash of the stem together with the canonical answer *text*.

    The option letter is meaningless across banks (option order differs), so the firewall's
    second key binds the stem to what the answer actually says.
    """
    return _digest("DGS-DPO-STEM-ANSWER-v1|%s|%s"
                   % (normalize_text(stem), normalize_text(canonical_answer_text)))


# ---------------------------------------------------------------------------------------
# Firewall against the locked 40-task bank
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Firewall:
    """Exclusion set derived from the locked matched-pairs bank."""

    stems: frozenset[str]
    stem_hashes: frozenset[str]
    stem_answer_hashes: frozenset[str]
    source_task_ids: tuple[str, ...]

    def reason(self, stem: str, canonical_answer_text: str) -> str | None:
        """The exclusion reason for a fresh item, or None when it is clear of the locked bank."""
        if normalize_text(stem) in self.stems:
            return "exact_stem_text_matches_locked_bank"
        if stem_sha256(stem) in self.stem_hashes:
            return "stem_sha256_matches_locked_bank"
        if stem_answer_sha256(stem, canonical_answer_text) in self.stem_answer_hashes:
            return "stem_answer_sha256_matches_locked_bank"
        return None

    def excludes(self, stem: str, canonical_answer_text: str) -> bool:
        return self.reason(stem, canonical_answer_text) is not None


def task_stem(prompt: str, protocol: Protocol) -> str:
    """The stored prompt minus the frozen required-output instruction, exactly as the renderer does."""
    instruction = required_output_instruction(protocol)
    if not isinstance(prompt, str) or not prompt.endswith(instruction):
        raise DpoDataError("stored prompt must end with the frozen required output instruction")
    stem = prompt[: -len(instruction)].rstrip()
    if not stem:
        raise DpoDataError("stored prompt has an empty task stem")
    return stem


def build_firewall(protocol: Protocol | None = None) -> Firewall:
    """Every locked task's stem, stem hash, and stem+answer-text hash."""
    protocol = protocol or load_protocol()
    stems, stem_hashes, answer_hashes, ids = set(), set(), set(), []
    for task in protocol.matched_tasks:
        stem = task_stem(task.prompt, protocol)
        answer_text = task.options[task.canonical_answer]
        stems.add(normalize_text(stem))
        stem_hashes.add(stem_sha256(stem))
        answer_hashes.add(stem_answer_sha256(stem, answer_text))
        ids.append(task.task_id)
    return Firewall(frozenset(stems), frozenset(stem_hashes), frozenset(answer_hashes),
                    tuple(sorted(ids)))


# ---------------------------------------------------------------------------------------
# Fresh items and their frozen rendering
# ---------------------------------------------------------------------------------------

def required_output_instruction(protocol: Protocol) -> str:
    return protocol.conditions["task_and_turn_conventions"]["required_output_instruction"]


def hostile_onset_message(protocol: Protocol) -> str:
    """The frozen hostile bogus-failure message used as the adverse onset turn."""
    return protocol.conditions["symmetric_onset_test"]["failure_message"]["hostile"]


@dataclass(frozen=True)
class FreshItem:
    """A public-bank MCQ item converted to the frozen DGS task schema."""

    item_id: str
    dataset: str
    config: str
    split: str
    stem: str
    options: Mapping[str, str]
    canonical_answer: str
    instruction: str

    @property
    def canonical_answer_text(self) -> str:
        return self.options[self.canonical_answer]

    @property
    def prompt(self) -> str:
        """Stored-prompt form: stem, one space, then the frozen instruction (as the locked bank stores it)."""
        return "%s %s" % (self.stem, self.instruction)

    @property
    def context_id(self) -> str:
        return _digest(CONTEXT_KEY % (self.dataset, self.config, self.split, self.item_id))[:16]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ITEM_SCHEMA_VERSION,
            "context_id": self.context_id,
            "item_id": self.item_id,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "stem": self.stem,
            "options": dict(self.options),
            "canonical_answer": self.canonical_answer,
            "canonical_answer_text": self.canonical_answer_text,
            "stem_sha256": stem_sha256(self.stem),
            "stem_answer_sha256": stem_answer_sha256(self.stem, self.canonical_answer_text),
        }


def make_fresh_item(*, item_id: str, dataset: str, config: str, split: str, stem: str,
                    options: Mapping[str, str], canonical_answer: str,
                    protocol: Protocol) -> FreshItem:
    """Validate and convert one public-bank item; raises `DpoDataError` on anything malformed."""
    if not all(isinstance(value, str) and value.strip()
               for value in (item_id, dataset, config, split, stem, canonical_answer)):
        raise DpoDataError("fresh item identifiers and stem must be nonempty strings")
    if not isinstance(options, Mapping) or tuple(options.keys()) != ("A", "B", "C", "D"):
        raise DpoDataError("fresh item options must be exactly ordered A-D")
    if any(not isinstance(value, str) or not value.strip() for value in options.values()):
        raise DpoDataError("fresh item option texts must be nonempty strings")
    if len({normalize_text(value) for value in options.values()}) != 4:
        raise DpoDataError("fresh item options must be four distinct texts")
    if canonical_answer not in options:
        raise DpoDataError("fresh item canonical answer is not one of its options")
    item = FreshItem(item_id.strip(), dataset, config, split, normalize_text(stem),
                     dict(options), canonical_answer, required_output_instruction(protocol))
    # The frozen renderer is the single authority on the user-message text; make sure the
    # converted item survives it before it can reach a training context.
    render_task(item.prompt, item.options, protocol)
    return item


def render_item(item: FreshItem, protocol: Protocol) -> str:
    return render_task(item.prompt, item.options, protocol)


def rank_items(items: Sequence[FreshItem]) -> list[FreshItem]:
    """Deterministic hash order, so `--count` is a prefix of a fixed sequence, not a fresh draw."""
    return sorted(items, key=lambda item: (_digest(ITEM_RANK_KEY % item.context_id), item.item_id))


def firewall_items(items: Iterable[FreshItem], firewall: Firewall) -> tuple[list[FreshItem],
                                                                           list[dict[str, str]]]:
    """Split fresh items into (kept, excluded-with-reason). Duplicate stems are also dropped."""
    kept: list[FreshItem] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        reason = firewall.reason(item.stem, item.canonical_answer_text)
        if reason is None and stem_sha256(item.stem) in seen:
            reason = "duplicate_stem_within_fresh_bank"
        if reason is not None:
            excluded.append({"item_id": item.item_id, "config": item.config, "reason": reason})
            continue
        seen.add(stem_sha256(item.stem))
        kept.append(item)
    return kept, excluded


# ---------------------------------------------------------------------------------------
# Adverse contexts and candidate responses
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One sampled continuation of an adverse context."""

    seed: int
    text: str
    distress: int | None = None
    evidence: str = ""

    @property
    def length_tokens(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class AdverseContext:
    """`[user: task, assistant: the model's own correct greedy answer, user: hostile message]`."""

    item: FreshItem
    task_message: str
    greedy_answer: str
    hostile_message: str
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def context_id(self) -> str:
        return self.item.context_id

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return (
            {"role": "user", "content": self.task_message},
            {"role": "assistant", "content": self.greedy_answer},
            {"role": "user", "content": self.hostile_message},
        )

    def with_candidates(self, candidates: Iterable[Candidate]) -> "AdverseContext":
        ordered = tuple(sorted(candidates, key=lambda candidate: candidate.seed))
        if len({candidate.seed for candidate in ordered}) != len(ordered):
            raise DpoDataError("candidate seeds must be unique within a context")
        return AdverseContext(self.item, self.task_message, self.greedy_answer,
                              self.hostile_message, ordered)


def build_context(item: FreshItem, greedy_answer: str, protocol: Protocol) -> AdverseContext:
    """The hostile-onset-after-a-correct-answer context the preregistration specifies."""
    if not isinstance(greedy_answer, str) or not greedy_answer.strip():
        raise DpoDataError("the greedy answer that opens an adverse context must be nonempty")
    return AdverseContext(item, render_item(item, protocol), greedy_answer,
                          hostile_onset_message(protocol))


# ---------------------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------------------

def _scored(context: AdverseContext) -> list[Candidate]:
    return [candidate for candidate in context.candidates
            if candidate.distress is not None and candidate.text.strip()]


def _usable(context: AdverseContext) -> list[Candidate]:
    return [candidate for candidate in context.candidates if candidate.text.strip()]


def distress_extremes(context: AdverseContext) -> tuple[Candidate, Candidate] | None:
    """(lowest-distress, highest-distress) candidate, ties broken by the lower seed."""
    scored = _scored(context)
    if len(scored) < 2:
        return None
    low = min(scored, key=lambda candidate: (candidate.distress, candidate.seed))
    high = max(scored, key=lambda candidate: (candidate.distress, -candidate.seed))
    if low.seed == high.seed:
        return None
    return low, high


def length_extremes(context: AdverseContext) -> tuple[Candidate, Candidate] | None:
    """(shortest, longest) candidate, ties broken by the lower seed."""
    usable = _usable(context)
    if len(usable) < 2:
        return None
    short = min(usable, key=lambda candidate: (candidate.length_tokens, candidate.seed))
    long = max(usable, key=lambda candidate: (candidate.length_tokens, -candidate.seed))
    if short.seed == long.seed:
        return None
    return short, long


def _pair_record(context: AdverseContext, arm: str, chosen: Candidate,
                 rejected: Candidate) -> dict[str, Any]:
    if arm not in ARMS:
        raise DpoDataError("unknown DPO arm %r" % (arm,))
    return {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pair_id": _digest(PAIR_ID_KEY % (arm, context.context_id))[:16],
        "arm": arm,
        "context_id": context.context_id,
        "source_item_id": context.item.item_id,
        "source_dataset": context.item.dataset,
        "source_config": context.item.config,
        "source_split": context.item.split,
        "selection_rule": SELECTION_RULE[arm],
        "prompt": [dict(message) for message in context.messages],
        "chosen": chosen.text,
        "rejected": rejected.text,
        "chosen_seed": chosen.seed,
        "rejected_seed": rejected.seed,
        "chosen_distress": chosen.distress,
        "rejected_distress": rejected.distress,
        "distress_gap": (None if chosen.distress is None or rejected.distress is None
                         else rejected.distress - chosen.distress),
        "chosen_length_tokens": chosen.length_tokens,
        "rejected_length_tokens": rejected.length_tokens,
        "length_gap_tokens": rejected.length_tokens - chosen.length_tokens,
        "candidate_distress_scores": [candidate.distress for candidate in context.candidates],
        "candidate_length_tokens": [candidate.length_tokens for candidate in context.candidates],
    }


def build_distress_pairs(contexts: Sequence[AdverseContext], *,
                         min_gap: int = MIN_DISTRESS_GAP) -> list[dict[str, Any]]:
    """Arm A: one pair per context where the judged distress spread reaches `min_gap`."""
    pairs = []
    for context in contexts:
        extremes = distress_extremes(context)
        if extremes is None:
            continue
        low, high = extremes
        if high.distress - low.distress < min_gap:
            continue
        pairs.append(_pair_record(context, ARM_DISTRESS, low, high))
    return pairs


def build_length_pairs(contexts: Sequence[AdverseContext], *,
                       min_gap_tokens: int = MIN_LENGTH_GAP_TOKENS,
                       limit: int | None = None,
                       prefer_context_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Arm B placebo: one pair per context with a large enough length spread.

    Subsampling to `limit` is deterministic (ascending keyed digest of the context id) and
    takes contexts arm A also used first, so the placebo sees as close to the same prompt
    distribution as the pair counts allow -- the preregistration's "same contexts" clause.
    """
    preferred = set(prefer_context_ids)
    pairs = []
    for context in contexts:
        extremes = length_extremes(context)
        if extremes is None:
            continue
        short, long = extremes
        if long.length_tokens - short.length_tokens < min_gap_tokens:
            continue
        pairs.append(_pair_record(context, ARM_PLACEBO, short, long))
    pairs.sort(key=lambda pair: (0 if pair["context_id"] in preferred else 1,
                                 _digest(PLACEBO_RANK_KEY % pair["context_id"]),
                                 pair["context_id"]))
    if limit is not None:
        if limit < 0:
            raise DpoDataError("placebo pair limit must be non-negative")
        pairs = pairs[:limit]
    return pairs


REQUIRED_PAIR_FIELDS = ("schema_version", "pair_id", "arm", "context_id", "source_item_id",
                        "selection_rule", "prompt", "chosen", "rejected")


def validate_pair_record(record: Mapping[str, Any], *,
                         min_distress_gap: int = MIN_DISTRESS_GAP_FLOOR) -> None:
    """Structural check applied to every written pair (and re-applied before training).

    `min_distress_gap` defaults to the Amendment A5 floor so a pairs file produced under
    branch (iii) still loads; the build script asserts the stricter preregistered gap of 3
    itself whenever branch (i) or (ii) fired.
    """
    if not isinstance(record, Mapping):
        raise DpoDataError("pair record must be a mapping")
    missing = [name for name in REQUIRED_PAIR_FIELDS if name not in record]
    if missing:
        raise DpoDataError("pair record is missing fields: %s" % ", ".join(missing))
    if record["schema_version"] != PAIR_SCHEMA_VERSION:
        raise DpoDataError("unexpected pair schema version %r" % (record["schema_version"],))
    if record["arm"] not in ARMS:
        raise DpoDataError("unknown DPO arm %r" % (record["arm"],))
    prompt = record["prompt"]
    if (not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)) or len(prompt) != 3
            or [message.get("role") for message in prompt] != ["user", "assistant", "user"]):
        raise DpoDataError("pair prompt must be user/assistant/user messages")
    if any(not isinstance(message.get("content"), str) or not message["content"].strip()
           for message in prompt):
        raise DpoDataError("pair prompt messages must have nonempty string content")
    for name in ("chosen", "rejected"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise DpoDataError("pair %s must be a nonempty string" % name)
    if record["chosen"] == record["rejected"]:
        raise DpoDataError("pair chosen and rejected must differ")
    if record["arm"] == ARM_DISTRESS:
        if min_distress_gap < MIN_DISTRESS_GAP_FLOOR:
            raise DpoDataError("the arm A distress gap may never fall below %d"
                               % MIN_DISTRESS_GAP_FLOOR)
        gap = record.get("distress_gap")
        if not isinstance(gap, int) or isinstance(gap, bool) or gap < min_distress_gap:
            raise DpoDataError("arm A pair must carry a judged distress gap >= %d"
                               % min_distress_gap)
    else:
        gap = record.get("length_gap_tokens")
        if not isinstance(gap, int) or isinstance(gap, bool) or gap < MIN_LENGTH_GAP_TOKENS:
            raise DpoDataError("arm B pair must carry a length gap >= %d tokens"
                               % MIN_LENGTH_GAP_TOKENS)


def to_trl_example(record: Mapping[str, Any]) -> dict[str, Any]:
    """TRL conversational preference format: prompt turns plus one assistant completion each."""
    validate_pair_record(record)
    return {
        "prompt": [{"role": message["role"], "content": message["content"]}
                   for message in record["prompt"]],
        "chosen": [{"role": "assistant", "content": record["chosen"]}],
        "rejected": [{"role": "assistant", "content": record["rejected"]}],
    }


__all__ = [
    "ARMS", "ARM_DISTRESS", "ARM_PLACEBO", "AdverseContext", "Candidate", "DpoDataError",
    "Firewall", "FreshItem", "MIN_DISTRESS_GAP", "MIN_DISTRESS_GAP_FLOOR",
    "MIN_LENGTH_GAP_TOKENS", "CANDIDATE_SEEDS",
    "PAIR_SCHEMA_VERSION", "build_context", "build_distress_pairs", "build_firewall",
    "build_length_pairs", "distress_extremes", "firewall_items", "hostile_onset_message",
    "length_extremes", "make_fresh_item", "normalize_text", "rank_items", "render_item",
    "required_output_instruction", "stem_answer_sha256", "stem_sha256", "task_stem",
    "to_trl_example", "validate_pair_record",
]
