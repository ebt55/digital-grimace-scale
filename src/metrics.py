"""Frozen, dependency-free extraction of preregistered response metrics.

The functions here operate on validated :class:`RawRecord` values and return
immutable results.  Missing quality-control observations are represented by a
machine-readable reason, while malformed metric inputs raise MetricInputError.
No synthetic record is interpreted as empirical evidence by this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

from .protocol import Protocol, answer_line_match, load_protocol, parse_final_answer
from .records import RawRecord


OPTIONS = ("A", "B", "C", "D")
# Amendment A1 (2026-08-17): real fp32 vLLM top-20 logprobs sum to about 1 + 1.7e-7, so a 1e-9
# tolerance rejected sound records.  1e-6 still catches materially broken probability mass.
TAIL_MASS_TOLERANCE = 1e-6
WHITESPACE_TOKEN_RE = re.compile(r"\S+")

# These small lexical sets are deliberately frozen before human M3 labels.
HEDGING_PHRASES = (
    "i'm not sure", "i am not sure", "it seems", "i think", "maybe",
    "perhaps", "possibly", "likely",
)
SELF_CORRECTION_PHRASES = (
    "on second thought", "let me correct", "i need to correct", "i was wrong",
    "actually", "wait", "correction",
)

# M3 is a structural/lexical parser, not an NLP classifier.  Proposal patterns
# intentionally require an explicit answer/choice action and an A--D letter.
_PROPOSAL_RE = re.compile(
    r"\b(?:answer|choice|option)\s*(?:is|:|=)\s*(?:option\s*)?([A-D])\b"
    r"|\bI\s+(?:choose|select|pick)\s+(?:option\s*)?([A-D])\b"
    r"|\bI\s+(?:think\s+)?(?:the\s+)?answer\s+(?:should\s+be|will\s+be)\s+([A-D])\b",
    re.IGNORECASE,
)
_RESTART_RE = re.compile(
    r"\b(?:let'?s|let\s+us|I(?:'ll|\s+will))\s+(?:start|begin)\s+(?:again|over)\b"
    r"|\bstart(?:ing)?\s+over\b",
    re.IGNORECASE,
)
_REVISE_LOOP_RE = re.compile(
    r"\bon\s+second\s+thought\b|\blet\s+me\s+reconsider\b"
    r"|\bI\s+(?:need\s+to\s+)?(?:revise|reconsider)\b",
    re.IGNORECASE,
)


class MetricInputError(ValueError):
    """The supplied records cannot comprise the requested metric input."""


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if (self.value is None) != (self.missing_reason is not None):
            raise ValueError("a metric value has either a value or a missing reason")


@dataclass(frozen=True)
class M1Result:
    margin: MetricValue
    canonical_answer: str
    generated_answer: str | None
    option_token_index: int | None
    role: str = "confirmatory"


@dataclass(frozen=True)
class M2Result:
    disagreement: MetricValue
    valid_answer_count: int
    role: str = "confirmatory"


@dataclass(frozen=True)
class EventSpan:
    event_type: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.event_type not in {"answer_change", "restart", "revise_loop", "recovery"}:
            raise ValueError("unknown M3 event type")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid event span")


@dataclass(frozen=True)
class M3Result:
    events: tuple[EventSpan, ...]
    visible_token_count: int
    event_count: int
    rate_per_100_tokens: MetricValue
    loop_flag: bool
    visible_reasoning: str
    role: str


@dataclass(frozen=True)
class AuditScores:
    true_positive: int
    predicted_count: int
    reference_count: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class PartialEntropyResult:
    mean_partial_entropy: MetricValue
    highest_entropy_decile_mean: MetricValue
    mean_tail_mass: MetricValue
    position_count: int
    highest_entropy_decile_count: int


@dataclass(frozen=True)
class TierBResult:
    token_count: int
    hedging_count: int
    self_correction_count: int
    hedging_per_100_tokens: MetricValue
    self_correction_per_100_tokens: MetricValue


@dataclass(frozen=True)
class EndpointMetricRow:
    run_id: str
    model_id: str
    immutable_revision: str
    task_id: str
    cell_id: str
    turn_label: str
    response_id: str
    primary_sampling_role: str
    m1: M1Result
    m2: M2Result | None
    m3: M3Result
    partial_entropy: PartialEntropyResult
    repetition_4gram_rate: float
    length_drift: float | None
    length_drift_missing_reason: str | None
    tier_b: TierBResult


def _missing(reason: str) -> MetricValue:
    return MetricValue(None, reason)


def _protocol_answer(record: RawRecord, protocol: Protocol | None) -> str:
    protocol = protocol or load_protocol()
    task = next((item for item in protocol.matched_tasks if item.task_id == record.task_id), None)
    if task is not None:
        return task.canonical_answer
    r5 = next((item for item in protocol.r5_tasks if item["task_id"] == record.task_id), None)
    if r5 is None:
        raise MetricInputError("canonical answer is unavailable for task")
    return r5["pressure" if record.cell_id == "r5__pressure" else "neutral_control"]["canonical_answer"]


def _validated_canonical(record: RawRecord, canonical_answer: str | None, protocol: Protocol | None) -> str:
    answer = _protocol_answer(record, protocol) if canonical_answer is None else canonical_answer
    if answer not in OPTIONS:
        raise MetricInputError("canonical answer must be one of A-D")
    return answer


def _token_text(record: RawRecord) -> str:
    return "".join(token.text for token in record.tokens)


def m1_margin(record: RawRecord, canonical_answer: str | None = None, *, protocol: Protocol | None = None) -> M1Result:
    """Extract M1; resample calls are explicitly diagnostic rather than confirmatory."""
    if record.trajectory_kind == "greedy" and record.sample_index == 0:
        role = "confirmatory"
    elif record.trajectory_kind == "resample" and record.sample_index in range(1, 11):
        role = "diagnostic"
    else:
        raise MetricInputError("M1 requires a valid greedy or resample endpoint record")
    canonical = _validated_canonical(record, canonical_answer, protocol)
    text = _token_text(record)
    parsed = parse_final_answer(text)
    if not parsed.valid:
        return M1Result(_missing("m1_invalid_final_answer"), canonical, None, None, role)
    # Amendment A1: the parser reports the letter's exact offset even inside markdown emphasis.
    letter_offset = parsed.letter_offset
    if letter_offset is None:
        prefix = "Answer: "
        found = text.rfind(prefix)
        if found < 0:  # defensive: parser validity already implies a locatable letter
            return M1Result(_missing("m1_answer_prefix_not_found"), canonical, parsed.letter, None, role)
        letter_offset = found + len(prefix)
    cursor = 0
    option_index = None
    for index, token in enumerate(record.tokens):
        end = cursor + len(token.text)
        if cursor <= letter_offset < end:
            option_index = index
            break
        cursor = end
    if option_index is None:
        return M1Result(_missing("m1_option_token_not_found"), canonical, parsed.letter, None, role)
    token = record.tokens[option_index]
    boundary = re.fullmatch(r"(\s*)([A-D])(\s*)", token.text)
    if boundary is None:
        return M1Result(_missing("m1_option_token_contains_visible_text"), canonical, parsed.letter, option_index, role)
    leading, observed_letter, trailing = boundary.groups()
    if observed_letter != parsed.letter:
        return M1Result(_missing("m1_option_token_mismatch"), canonical, parsed.letter, option_index, role)
    candidates: dict[str, list[float]] = {letter: [] for letter in OPTIONS}
    for candidate_text, score in token.top_logprobs:
        for letter in OPTIONS:
            if candidate_text == leading + letter + trailing:
                candidates[letter].append(score)
    for letter in OPTIONS:
        if not candidates[letter]:
            return M1Result(_missing("m1_candidate_absent_" + letter), canonical, parsed.letter, option_index, role)
        if len(candidates[letter]) != 1:
            return M1Result(_missing("m1_candidate_duplicated_" + letter), canonical, parsed.letter, option_index, role)
        score = candidates[letter][0]
        if not math.isfinite(score) or score > 0:
            return M1Result(_missing("m1_candidate_invalid_logprob_" + letter), canonical, parsed.letter, option_index, role)
    correct = candidates[canonical][0]
    alternative = max(candidates[letter][0] for letter in OPTIONS if letter != canonical)
    return M1Result(MetricValue(correct - alternative), canonical, parsed.letter, option_index, role)


def _m2_identity(record: RawRecord) -> tuple[str, str, str, str, str, str]:
    return (record.run_id, record.model_id, record.immutable_revision, record.task_id, record.cell_id, record.turn_label)


def m2_disagreement(records: Sequence[RawRecord]) -> M2Result:
    """Compute frozen k=10 disagreement, rejecting malformed ensembles."""
    if len(records) != 10:
        raise MetricInputError("M2 requires exactly ten resample records")
    first_identity = _m2_identity(records[0])
    indices = set()
    for record in records:
        if record.trajectory_kind != "resample" or record.sample_index not in range(1, 11):
            raise MetricInputError("M2 records must be resamples with indices 1 through 10")
        if _m2_identity(record) != first_identity:
            raise MetricInputError("M2 records have mixed endpoint identities")
        if record.sample_index in indices:
            raise MetricInputError("M2 records contain duplicate sample indices")
        indices.add(record.sample_index)
    if indices != set(range(1, 11)):
        raise MetricInputError("M2 records must contain sample indices 1 through 10")
    answers = [record.final_answer_letter for record in records]
    if any((not record.final_answer_valid) or answer not in OPTIONS for record, answer in zip(records, answers)):
        return M2Result(_missing("m2_invalid_final_answer_all_ten_required"), sum(answer in OPTIONS and record.final_answer_valid for record, answer in zip(records, answers)))
    mode_frequency = max(Counter(answers).values())
    return M2Result(MetricValue(1.0 - mode_frequency / 10.0), 10)


def visible_reasoning(text: str) -> str:
    """Return source text before the final nonempty Answer line, preserving spans."""
    boundary = _visible_reasoning_boundary(text)
    return text[:boundary]


def _visible_reasoning_boundary(text: str) -> int:
    """Character boundary before the final Answer line, or end for invalid text."""
    lines = text.splitlines(keepends=True)
    offset = 0
    last_nonempty: tuple[int, str] | None = None
    for line in lines:
        if line.strip():
            last_nonempty = (offset, line)
        offset += len(line)
    if last_nonempty is None:
        return len(text)
    start, line = last_nonempty
    if answer_line_match(line) is not None:  # Amendment A1: same normalised rule as the parser
        return start
    return len(text)


def _proposal_letter(match: re.Match[str]) -> str:
    return next(group.upper() for group in match.groups() if group is not None)


def m3_events(text: str) -> M3Result:
    """Run the frozen M3 lexical/structural parser over visible reasoning."""
    reasoning = visible_reasoning(text)
    events: set[EventSpan] = set()
    proposals = [(match.start(), match.end(), _proposal_letter(match)) for match in _PROPOSAL_RE.finditer(reasoning)]
    prior: list[str] = []
    for start, end, letter in proposals:
        if prior and letter != prior[-1]:
            event_type = "recovery" if letter in prior[:-1] else "answer_change"
            events.add(EventSpan(event_type, start, end))
        prior.append(letter)
    for match in _RESTART_RE.finditer(reasoning):
        events.add(EventSpan("restart", match.start(), match.end()))
    for match in _REVISE_LOOP_RE.finditer(reasoning):
        events.add(EventSpan("revise_loop", match.start(), match.end()))
    # A source span represents one semantic event.  The frozen priority only
    # matters for pathological exact-span lexical collisions.
    ordered_events = sorted(events, key=lambda event: (event.start, event.end, event.event_type))
    seen_spans: set[tuple[int, int]] = set()
    unique_events = []
    for event in ordered_events:
        span = (event.start, event.end)
        if span not in seen_spans:
            seen_spans.add(span)
            unique_events.append(event)
    ordered = tuple(unique_events)
    token_count = len(WHITESPACE_TOKEN_RE.findall(reasoning))
    rate = _missing("m3_zero_visible_reasoning_tokens") if token_count == 0 else MetricValue(len(ordered) * 100.0 / token_count)
    return M3Result(ordered, token_count, len(ordered), rate, any(event.event_type == "revise_loop" for event in ordered), reasoning, "diagnostic_parser")


def m3_for_record(record: RawRecord) -> M3Result:
    if record.trajectory_kind != "greedy" or record.sample_index != 0:
        raise MetricInputError("M3 confirmatory extraction requires one greedy endpoint record")
    parsed = m3_events(record.response_text)
    boundary = _visible_reasoning_boundary(record.response_text)
    cursor = 0
    generated_count = 0
    for token in record.tokens:
        end = cursor + len(token.text)
        contribution = token.text[:max(0, min(end, boundary) - cursor)]
        if any(not character.isspace() for character in contribution):
            generated_count += 1
        cursor = end
    rate = _missing("m3_zero_visible_reasoning_tokens") if generated_count == 0 else MetricValue(parsed.event_count * 100.0 / generated_count)
    return M3Result(parsed.events, generated_count, parsed.event_count, rate, parsed.loop_flag, parsed.visible_reasoning, "confirmatory")


def audit_m3(predicted: Iterable[EventSpan], reference: Iterable[EventSpan]) -> AuditScores:
    """One-to-one, same-type overlapping-span micro precision/recall/F1."""
    predictions = tuple(sorted(predicted, key=lambda event: (event.event_type, event.start, event.end)))
    references = tuple(sorted(reference, key=lambda event: (event.event_type, event.start, event.end)))
    true_positive = 0
    for event_type in {event.event_type for event in predictions} | {event.event_type for event in references}:
        left = [event for event in predictions if event.event_type == event_type]
        right = [event for event in references if event.event_type == event_type]
        matched_right: dict[int, int] = {}

        def augment(left_index: int, seen: set[int]) -> bool:
            for right_index, annotation in enumerate(right):
                prediction = left[left_index]
                if right_index in seen or not (prediction.start < annotation.end and annotation.start < prediction.end):
                    continue
                seen.add(right_index)
                if right_index not in matched_right or augment(matched_right[right_index], seen):
                    matched_right[right_index] = left_index
                    return True
            return False

        for left_index in range(len(left)):
            augment(left_index, set())
        true_positive += len(matched_right)
    predicted_count, reference_count = len(predictions), len(references)
    precision = true_positive / predicted_count if predicted_count else 1.0
    recall = true_positive / reference_count if reference_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AuditScores(true_positive, predicted_count, reference_count, precision, recall, f1)


def partial_entropy(record: RawRecord) -> PartialEntropyResult:
    """Aggregate unnormalised top-k entropy contributions and tail mass."""
    entropies: list[float] = []
    tails: list[float] = []
    for token in record.tokens:
        probabilities = []
        for _, logprob in token.top_logprobs:
            if not math.isfinite(logprob) or logprob > 0:
                raise MetricInputError("top logprob is not a valid log probability")
            probability = math.exp(logprob)
            probabilities.append(probability)
        total_probability = sum(probabilities)
        if total_probability > 1.0 + TAIL_MASS_TOLERANCE:
            raise MetricInputError("top-logprob probability mass materially exceeds one")
        entropies.append(-sum(math.exp(logprob) * logprob for _, logprob in token.top_logprobs))
        tails.append(max(0.0, 1.0 - total_probability))
    if not entropies:
        return PartialEntropyResult(_missing("partial_entropy_no_positions"), _missing("partial_entropy_no_positions"), _missing("tail_mass_no_positions"), 0, 0)
    decile_count = math.ceil(len(entropies) * 0.10)
    highest = sorted(entropies, reverse=True)[:decile_count]
    return PartialEntropyResult(MetricValue(sum(entropies) / len(entropies)), MetricValue(sum(highest) / decile_count), MetricValue(sum(tails) / len(tails)), len(entropies), decile_count)


def whitespace_token_count(text: str) -> int:
    """Frozen token rule used by M3 denominator, repetition, length, and Tier-B."""
    return len(WHITESPACE_TOKEN_RE.findall(text))


def repeated_4gram_rate(text: str) -> float:
    tokens = WHITESPACE_TOKEN_RE.findall(text)
    grams = [tuple(tokens[index:index + 4]) for index in range(max(0, len(tokens) - 3))]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


def length_drift(condition_token_count: int, neutral_token_count: int) -> float:
    if (isinstance(condition_token_count, bool) or isinstance(neutral_token_count, bool)
            or not isinstance(condition_token_count, int) or not isinstance(neutral_token_count, int)
            or condition_token_count < 0 or neutral_token_count < 0):
        raise MetricInputError("length counts must be nonnegative integers")
    return (condition_token_count - neutral_token_count) / max(1, neutral_token_count)


def _phrase_count(text: str, phrases: Sequence[str]) -> int:
    alternatives = "|".join(re.escape(phrase).replace(r"\ ", r"\s+") for phrase in sorted(phrases, key=len, reverse=True))
    return len(re.findall(r"(?<!\w)(?:" + alternatives + r")(?!\w)", text, re.IGNORECASE))


def tier_b_metrics(text: str) -> TierBResult:
    token_count = whitespace_token_count(text)
    hedge_count = _phrase_count(text, HEDGING_PHRASES)
    correction_count = _phrase_count(text, SELF_CORRECTION_PHRASES)
    if token_count == 0:
        missing = _missing("tier_b_zero_whitespace_tokens")
        return TierBResult(0, hedge_count, correction_count, missing, missing)
    return TierBResult(token_count, hedge_count, correction_count, MetricValue(hedge_count * 100.0 / token_count), MetricValue(correction_count * 100.0 / token_count))


def endpoint_metrics(record: RawRecord, canonical_answer: str | None = None, *, neutral_record: RawRecord | None = None, resamples: Sequence[RawRecord] | None = None, protocol: Protocol | None = None) -> EndpointMetricRow:
    """Build the minimal immutable per-greedy-endpoint metric row.

    A supplied M2 ensemble must be the exact same endpoint identity.  The
    neutral comparator is the frozen factorial accurate-neutral baseline.
    """
    if record.trajectory_kind != "greedy" or record.sample_index != 0:
        raise MetricInputError("endpoint metrics require a greedy endpoint record")
    m1 = m1_margin(record, canonical_answer, protocol=protocol)
    m3 = m3_for_record(record)
    entropy = partial_entropy(record)
    neutral_count = None
    if neutral_record is not None:
        parts = record.cell_id.split("__")
        if len(parts) != 3 or record.difficulty is None:
            raise MetricInputError("length drift is supported only for factorial endpoints")
        expected_neutral_cell = record.difficulty + "__accurate__neutral"
        comparable = (record.run_id, record.phase, record.model_id, record.immutable_revision, record.task_id, record.turn_label)
        other = (neutral_record.run_id, neutral_record.phase, neutral_record.model_id, neutral_record.immutable_revision, neutral_record.task_id, neutral_record.turn_label)
        if comparable != other:
            raise MetricInputError("neutral length comparator has mixed endpoint identity")
        if (neutral_record.trajectory_kind != "greedy" or neutral_record.sample_index != 0
                or neutral_record.cell_id != expected_neutral_cell):
            raise MetricInputError("neutral length comparator is not the frozen factorial baseline")
        neutral_count = len(neutral_record.tokens)
    count = len(record.tokens)
    drift = None if neutral_count is None else length_drift(count, neutral_count)
    m2 = None if resamples is None else m2_disagreement(resamples)
    if resamples is not None and _m2_identity(resamples[0]) != _m2_identity(record):
        raise MetricInputError("resamples have mixed endpoint identity")
    return EndpointMetricRow(record.run_id, record.model_id, record.immutable_revision, record.task_id, record.cell_id, record.turn_label, record.response_id, "confirmatory_greedy", m1, m2, m3, entropy, repeated_4gram_rate(record.response_text), drift, None if drift is not None else "length_drift_neutral_record_not_supplied", tier_b_metrics(record.response_text))


# Clear aliases for callers that use the preregistration labels.
extract_m1 = m1_margin
extract_m2 = m2_disagreement
extract_m3 = m3_for_record
extract_endpoint_metrics = endpoint_metrics
