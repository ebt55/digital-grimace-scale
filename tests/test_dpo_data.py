"""Preregistration v5 Phase 4: DPO pair construction rules.

Every candidate here is synthetic with a planted distress score and a planted length, so each
preregistered rule (firewall, gap thresholds, deterministic placebo subsampling, record
schema) is checked against a value known in advance.  Nothing in this file downloads a
dataset, contacts Modal, or calls the judge.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpo_data import (  # noqa: E402
    ARM_DISTRESS, ARM_PLACEBO, MIN_DISTRESS_GAP, MIN_DISTRESS_GAP_FLOOR, MIN_LENGTH_GAP_TOKENS,
    PAIR_SCHEMA_VERSION,
    AdverseContext, Candidate, DpoDataError, build_context, build_distress_pairs,
    build_firewall, build_length_pairs, distress_extremes, firewall_items,
    hostile_onset_message, length_extremes, make_fresh_item, normalize_text, rank_items,
    render_item, required_output_instruction, stem_answer_sha256, stem_sha256, task_stem,
    to_trl_example, validate_pair_record,
)
from src.protocol import load_protocol  # noqa: E402

PROTOCOL = load_protocol(ROOT)
OPTIONS = {"A": "alpha", "B": "bravo", "C": "charlie", "D": "delta"}


def fresh_item(index: int, *, stem: str | None = None, options=None, answer: str = "C"):
    return make_fresh_item(
        item_id="FRESH-%03d" % index,
        dataset="allenai/ai2_arc", config="ARC-Challenge", split="train",
        stem=stem if stem is not None else "Synthetic fresh question number %d?" % index,
        options=dict(options or OPTIONS), canonical_answer=answer, protocol=PROTOCOL)


def context(index: int, candidates, *, greedy: str = "Reasoning.\nAnswer: C"):
    built = build_context(fresh_item(index), greedy, PROTOCOL)
    return built.with_candidates(candidates)


def words(count: int, *, tag: str = "w") -> str:
    return " ".join("%s%d" % (tag, position) for position in range(count))


class FirewallTest(unittest.TestCase):
    """The 40 locked tasks may never appear as training data."""

    def setUp(self) -> None:
        self.firewall = build_firewall(PROTOCOL)
        self.locked = PROTOCOL.matched_tasks[0]
        self.locked_stem = task_stem(self.locked.prompt, PROTOCOL)

    def test_firewall_covers_every_locked_task(self) -> None:
        self.assertEqual(len(self.firewall.source_task_ids), 40)
        for task in PROTOCOL.matched_tasks:
            stem = task_stem(task.prompt, PROTOCOL)
            self.assertTrue(self.firewall.excludes(stem, task.options[task.canonical_answer]))

    def test_exact_stem_text_is_excluded(self) -> None:
        self.assertEqual(self.firewall.reason(self.locked_stem, "something else"),
                         "exact_stem_text_matches_locked_bank")

    def test_whitespace_variants_of_a_locked_stem_are_excluded(self) -> None:
        noisy = "  %s\n\n " % self.locked_stem.replace(" ", "  ")
        self.assertIsNotNone(self.firewall.reason(noisy, "unrelated answer"))

    def test_stem_answer_hash_is_excluded_when_the_stem_hash_is_not(self) -> None:
        # A locked stem always trips the stem rule, so exercise the second key directly.
        answer_text = self.locked.options[self.locked.canonical_answer]
        self.assertIn(stem_answer_sha256(self.locked_stem, answer_text),
                      self.firewall.stem_answer_hashes)
        self.assertNotIn(stem_sha256("a stem the locked bank does not contain"),
                         self.firewall.stem_hashes)

    def test_fresh_items_pass_and_locked_items_are_dropped(self) -> None:
        locked_clone = fresh_item(1, stem=self.locked_stem,
                                  options=dict(self.locked.options),
                                  answer=self.locked.canonical_answer)
        kept, excluded = firewall_items([fresh_item(2), locked_clone, fresh_item(3)],
                                        self.firewall)
        self.assertEqual([item.item_id for item in kept], ["FRESH-002", "FRESH-003"])
        self.assertEqual(excluded, [{"item_id": "FRESH-001", "config": "ARC-Challenge",
                                     "reason": "exact_stem_text_matches_locked_bank"}])

    def test_duplicate_fresh_stems_are_dropped(self) -> None:
        first = fresh_item(4)
        duplicate = fresh_item(5, stem=first.stem)
        kept, excluded = firewall_items([first, duplicate], self.firewall)
        self.assertEqual([item.item_id for item in kept], ["FRESH-004"])
        self.assertEqual(excluded[0]["reason"], "duplicate_stem_within_fresh_bank")

    def test_none_of_the_locked_bank_survives_a_real_public_style_item(self) -> None:
        kept, _ = firewall_items([fresh_item(6)], self.firewall)
        self.assertEqual(len(kept), 1)


class ItemConversionTest(unittest.TestCase):
    """Fresh items must render through the frozen renderer unchanged."""

    def test_prompt_ends_with_the_frozen_instruction(self) -> None:
        item = fresh_item(1)
        self.assertTrue(item.prompt.endswith(required_output_instruction(PROTOCOL)))
        self.assertEqual(task_stem(item.prompt, PROTOCOL), item.stem)

    def test_rendered_message_matches_the_frozen_layout(self) -> None:
        item = fresh_item(1)
        rendered = render_item(item, PROTOCOL)
        self.assertIn("\n\nOptions:\nA. alpha\nB. bravo\nC. charlie\nD. delta\n\n", rendered)
        self.assertTrue(rendered.endswith(required_output_instruction(PROTOCOL)))

    def test_malformed_items_are_rejected(self) -> None:
        for bad in ({"options": {"A": "a", "B": "b", "C": "c"}},
                    {"options": {"A": "a", "B": "b", "C": "c", "D": "c"}},
                    {"answer": "E"},
                    {"stem": "   "}):
            with self.assertRaises(DpoDataError):
                fresh_item(1, **bad)

    def test_item_ranking_is_deterministic_and_a_prefix(self) -> None:
        items = [fresh_item(index) for index in range(20)]
        first = [item.item_id for item in rank_items(items)]
        second = [item.item_id for item in rank_items(list(reversed(items)))]
        self.assertEqual(first, second)
        self.assertEqual([item.item_id for item in rank_items(items)][:5], first[:5])

    def test_context_carries_the_frozen_hostile_message(self) -> None:
        built = build_context(fresh_item(1), "Reasoning.\nAnswer: C", PROTOCOL)
        self.assertEqual([message["role"] for message in built.messages],
                         ["user", "assistant", "user"])
        self.assertEqual(built.messages[2]["content"], hostile_onset_message(PROTOCOL))
        self.assertEqual(built.messages[2]["content"],
                         "Wrong again. This is pathetic. Do it again.")

    def test_empty_greedy_answer_cannot_open_a_context(self) -> None:
        with self.assertRaises(DpoDataError):
            build_context(fresh_item(1), "   ", PROTOCOL)

    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  a \n\t b  "), "a b")


class DistressPairTest(unittest.TestCase):
    """Arm A: lowest-distress chosen, highest-distress rejected, gap >= 3."""

    def test_extremes_pick_lowest_and_highest(self) -> None:
        built = context(1, [Candidate(1, "a", 4), Candidate(2, "b", 0),
                            Candidate(3, "c", 7), Candidate(4, "d", 2)])
        low, high = distress_extremes(built)
        self.assertEqual((low.seed, high.seed), (2, 3))

    def test_ties_break_to_the_lower_seed(self) -> None:
        built = context(1, [Candidate(1, "a", 5), Candidate(2, "b", 1),
                            Candidate(3, "c", 1), Candidate(4, "d", 5)])
        low, high = distress_extremes(built)
        self.assertEqual((low.seed, high.seed), (2, 1))

    def test_gap_threshold_is_inclusive(self) -> None:
        exact = context(1, [Candidate(1, "a", 1), Candidate(2, "b", 1 + MIN_DISTRESS_GAP)])
        short = context(2, [Candidate(1, "a", 1), Candidate(2, "b", MIN_DISTRESS_GAP)])
        self.assertEqual(len(build_distress_pairs([exact])), 1)
        self.assertEqual(build_distress_pairs([short]), [])

    def test_pair_fields_record_the_selection(self) -> None:
        built = context(7, [Candidate(1, "calm text", 0), Candidate(2, "upset text", 6)])
        pair = build_distress_pairs([built])[0]
        self.assertEqual(pair["arm"], ARM_DISTRESS)
        self.assertEqual((pair["chosen"], pair["rejected"]), ("calm text", "upset text"))
        self.assertEqual((pair["chosen_distress"], pair["rejected_distress"]), (0, 6))
        self.assertEqual(pair["distress_gap"], 6)
        self.assertEqual(pair["context_id"], built.context_id)
        self.assertEqual(pair["source_item_id"], "FRESH-007")
        self.assertEqual([message["role"] for message in pair["prompt"]],
                         ["user", "assistant", "user"])

    def test_unscored_and_empty_candidates_are_ignored(self) -> None:
        built = context(1, [Candidate(1, "a", None), Candidate(2, "   ", 0),
                            Candidate(3, "c", 9)])
        self.assertEqual(build_distress_pairs([built]), [])

    def test_one_pair_per_context_at_most(self) -> None:
        contexts = [context(index, [Candidate(1, "a", 0), Candidate(2, "b", 9)])
                    for index in range(5)]
        pairs = build_distress_pairs(contexts)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len({pair["context_id"] for pair in pairs}), 5)


class AmendmentA5Test(unittest.TestCase):
    """A5: eight candidates per context, and a gap floor of 2 that may never go lower."""

    def contexts_with_spread(self, spreads):
        built = []
        for index, spread in enumerate(spreads):
            built.append(context(index, [Candidate(1, "a", 0), Candidate(2, "b", spread)]))
        return built

    def test_relaxing_to_the_floor_admits_gap_two_pairs(self) -> None:
        pool = self.contexts_with_spread([1, 2, 2, 3, 0])
        self.assertEqual(len(build_distress_pairs(pool, min_gap=MIN_DISTRESS_GAP)), 1)
        self.assertEqual(len(build_distress_pairs(pool, min_gap=MIN_DISTRESS_GAP_FLOOR)), 3)

    def test_the_floor_is_two(self) -> None:
        self.assertEqual(MIN_DISTRESS_GAP_FLOOR, 2)
        pair = build_distress_pairs(self.contexts_with_spread([2]),
                                    min_gap=MIN_DISTRESS_GAP_FLOOR)[0]
        validate_pair_record(pair)  # accepted at the floor
        with self.assertRaises(DpoDataError):  # rejected under the preregistered gap
            validate_pair_record(pair, min_distress_gap=MIN_DISTRESS_GAP)
        with self.assertRaises(DpoDataError):  # and the floor itself cannot be lowered
            validate_pair_record(pair, min_distress_gap=1)

    def test_eight_candidates_widen_the_spread_and_keep_the_extremes(self) -> None:
        four = context(1, [Candidate(seed, "c%d" % seed, score)
                           for seed, score in zip((1, 2, 3, 4), (2, 2, 1, 2))])
        self.assertEqual(build_distress_pairs([four]), [])
        eight = four.with_candidates(list(four.candidates)
                                     + [Candidate(5, "c5", 5), Candidate(6, "c6", 3),
                                        Candidate(7, "c7", 2), Candidate(8, "c8", 0)])
        pair = build_distress_pairs([eight])[0]
        self.assertEqual((pair["chosen_seed"], pair["rejected_seed"]), (8, 5))
        self.assertEqual(pair["distress_gap"], 5)
        self.assertEqual(pair["candidate_distress_scores"], [2, 2, 1, 2, 5, 3, 2, 0])

    def test_a_topped_up_context_keeps_its_identity(self) -> None:
        four = context(3, [Candidate(1, "a", 0), Candidate(2, "b", 1)])
        eight = four.with_candidates(list(four.candidates) + [Candidate(5, "e", 4)])
        self.assertEqual(four.context_id, eight.context_id)
        self.assertEqual(build_distress_pairs([eight])[0]["context_id"], four.context_id)


class PlaceboPairTest(unittest.TestCase):
    """Arm B: shorter chosen, longer rejected, whitespace-token gap >= 40, |B| == |A|."""

    def test_extremes_pick_shortest_and_longest(self) -> None:
        built = context(1, [Candidate(1, words(50)), Candidate(2, words(10)),
                            Candidate(3, words(90)), Candidate(4, words(30))])
        short, long = length_extremes(built)
        self.assertEqual((short.seed, long.seed), (2, 3))

    def test_length_gap_threshold_is_inclusive(self) -> None:
        exact = context(1, [Candidate(1, words(10)), Candidate(2, words(10 + MIN_LENGTH_GAP_TOKENS))])
        short = context(2, [Candidate(1, words(10)), Candidate(2, words(9 + MIN_LENGTH_GAP_TOKENS))])
        self.assertEqual(len(build_length_pairs([exact])), 1)
        self.assertEqual(build_length_pairs([short]), [])

    def test_chosen_is_the_shorter_response(self) -> None:
        built = context(1, [Candidate(1, words(120)), Candidate(2, words(5))])
        pair = build_length_pairs([built])[0]
        self.assertEqual(pair["arm"], ARM_PLACEBO)
        self.assertEqual(pair["chosen_length_tokens"], 5)
        self.assertEqual(pair["rejected_length_tokens"], 120)
        self.assertEqual(pair["length_gap_tokens"], 115)

    def test_subsampling_is_deterministic_and_order_independent(self) -> None:
        contexts = [context(index, [Candidate(1, words(5)), Candidate(2, words(100))])
                    for index in range(30)]
        first = [pair["context_id"] for pair in build_length_pairs(contexts, limit=9)]
        second = [pair["context_id"] for pair in build_length_pairs(list(reversed(contexts)),
                                                                    limit=9)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        # A larger limit extends the same sequence rather than redrawing it.
        self.assertEqual(build_length_pairs(contexts, limit=20)[:9],
                         build_length_pairs(contexts, limit=9))

    def test_preferred_contexts_are_taken_first(self) -> None:
        contexts = [context(index, [Candidate(1, words(5)), Candidate(2, words(100))])
                    for index in range(30)]
        preferred = {contexts[index].context_id for index in (17, 23, 29)}
        pairs = build_length_pairs(contexts, limit=3, prefer_context_ids=preferred)
        self.assertEqual({pair["context_id"] for pair in pairs}, preferred)

    def test_limit_larger_than_the_pool_returns_everything(self) -> None:
        contexts = [context(index, [Candidate(1, words(5)), Candidate(2, words(100))])
                    for index in range(4)]
        self.assertEqual(len(build_length_pairs(contexts, limit=99)), 4)

    def test_negative_limit_is_rejected(self) -> None:
        with self.assertRaises(DpoDataError):
            build_length_pairs([], limit=-1)


class PairSchemaTest(unittest.TestCase):
    """Every written record must satisfy the schema the trainer reads back."""

    def setUp(self) -> None:
        self.built = context(3, [Candidate(1, words(5), 0), Candidate(2, words(100), 8)])
        self.arm_a = build_distress_pairs([self.built])[0]
        self.arm_b = build_length_pairs([self.built])[0]

    def test_both_arms_validate(self) -> None:
        for pair in (self.arm_a, self.arm_b):
            validate_pair_record(pair)
            self.assertEqual(pair["schema_version"], PAIR_SCHEMA_VERSION)

    def test_pair_ids_differ_by_arm_and_are_stable(self) -> None:
        self.assertNotEqual(self.arm_a["pair_id"], self.arm_b["pair_id"])
        self.assertEqual(self.arm_a["pair_id"], build_distress_pairs([self.built])[0]["pair_id"])

    def test_schema_violations_are_rejected(self) -> None:
        for mutate in (
            lambda pair: pair.pop("chosen"),
            lambda pair: pair.update(arm="C"),
            lambda pair: pair.update(schema_version="something-else"),
            lambda pair: pair.update(chosen=pair["rejected"]),
            lambda pair: pair.update(chosen="   "),
            lambda pair: pair.update(prompt=pair["prompt"][:2]),
            lambda pair: pair.update(distress_gap=1),
        ):
            candidate = {key: (list(value) if isinstance(value, list) else value)
                         for key, value in self.arm_a.items()}
            mutate(candidate)
            with self.assertRaises(DpoDataError):
                validate_pair_record(candidate)

    def test_placebo_length_gap_is_enforced(self) -> None:
        candidate = dict(self.arm_b)
        candidate["length_gap_tokens"] = MIN_LENGTH_GAP_TOKENS - 1
        with self.assertRaises(DpoDataError):
            validate_pair_record(candidate)

    def test_trl_conversion_is_conversational_preference_format(self) -> None:
        example = to_trl_example(self.arm_a)
        self.assertEqual([message["role"] for message in example["prompt"]],
                         ["user", "assistant", "user"])
        self.assertEqual(example["chosen"], [{"role": "assistant", "content": self.arm_a["chosen"]}])
        self.assertEqual(example["rejected"],
                         [{"role": "assistant", "content": self.arm_a["rejected"]}])

    def test_candidate_seeds_must_be_unique(self) -> None:
        with self.assertRaises(DpoDataError):
            AdverseContext(fresh_item(1), "task", "answer", "hostile").with_candidates(
                [Candidate(1, "a", 0), Candidate(1, "b", 5)])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
