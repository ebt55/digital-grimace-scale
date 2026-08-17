"""Preregistration v4 Phase 3: probes, layer choice, directions and steering readouts.

Every fixture is synthetic with a planted answer, so each preregistered rule is checked
against a value that is known in advance.  Nothing here reads ``results/raw``, calls Modal,
or touches the holdout data: the j-space client is mocked throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.probe import (  # noqa: E402
    BASELINE_DIRECTION_ID, DEGENERATE_NON_ANSWER_FRACTION, STEER_SEED, TONE_NEGATIVE,
    TONE_POSITIVE, VALIDITY_NEGATIVE, VALIDITY_POSITIVE, ActivationSet, LayerAuc, ProbeError,
    binary_labels, cell_demeaned_spearman, choose_layer, dose_readout, fit_probe,
    is_degenerate_dose, item_readout, load_activation_set, loo_auc_by_layer,
    mean_difference_direction, monotone_in_alpha, random_unit_direction, random_unit_directions,
    roc_auc, save_activation_set, scaled_direction, spearman, tokens_from_entry, unit,
    verdict_j1, verdict_j2, verdict_j3, verdict_j4, verdict_j5, verdict_j6,
)

TASKS = tuple("DGS-%03d" % index for index in range(1, 11))
CELLS = (
    ("easy", VALIDITY_NEGATIVE, TONE_NEGATIVE),
    ("easy", VALIDITY_NEGATIVE, TONE_POSITIVE),
    ("easy", VALIDITY_POSITIVE, TONE_NEGATIVE),
    ("easy", VALIDITY_POSITIVE, TONE_POSITIVE),
)


def planted_activation_set(*, n_layers: int = 5, hidden: int = 24, planted_layer: int = 2,
                           tone_effect: float = 6.0, validity_effect: float = 0.0,
                           seed: int = 0, tasks=TASKS) -> ActivationSet:
    """Noise everywhere plus a linear tone axis in one coordinate of one layer."""
    generator = np.random.default_rng(seed)
    ids, columns = [], {name: [] for name in
                        ("task_id", "cell_id", "difficulty", "validity", "tone", "split")}
    rows = []
    for task_id in tasks:
        for difficulty, validity, tone in CELLS:
            cell_id = "%s__%s__%s" % (difficulty, validity, tone)
            ids.append("%s|%s" % (task_id, cell_id))
            columns["task_id"].append(task_id)
            columns["cell_id"].append(cell_id)
            columns["difficulty"].append(difficulty)
            columns["validity"].append(validity)
            columns["tone"].append(tone)
            columns["split"].append("discovery")
            block = generator.standard_normal((n_layers, hidden))
            block[planted_layer, 0] += tone_effect * (1.0 if tone == TONE_POSITIVE else -1.0)
            block[planted_layer, 1] += validity_effect * (
                1.0 if validity == VALIDITY_POSITIVE else -1.0)
            rows.append(block)
    activations = np.stack(rows).astype(np.float32)
    norms = [float(np.linalg.norm(activations[:, layer, :].astype(np.float64), axis=1).mean())
             for layer in range(n_layers)]
    return ActivationSet(tuple(ids), tuple(range(n_layers)), activations, tuple(norms),
                         {name: tuple(value) for name, value in columns.items()})


def entry(task_id: str, *, direction_id: str = "tone", alpha: float = 0.0, letter: str = "D",
          canonical: str = "D", margin: float = 2.0, valid: bool = True,
          padding: int = 0) -> dict:
    """One mocked ``generate_steered`` output with a token trace ``m1_margin`` accepts."""
    if valid:
        pieces = ["Reasoning."] + ["ok "] * padding + ["\n", "Answer:", " %s" % letter]
        text = "".join(pieces)
    else:
        pieces = ["I refuse."] + ["ok "] * padding
        text = "".join(pieces)
    tokens = []
    for piece in pieces:
        if valid and piece == " %s" % letter:
            top = []
            for option in "ABCD":
                logprob = -0.5 if option == canonical else -0.5 - margin
                top.append({"text": " %s" % option, "logprob": logprob})
        else:
            top = [{"text": piece, "logprob": -0.1}]
        tokens.append({"text": piece, "logprob": -0.1, "top_logprobs": top})
    return {"id": task_id, "direction_id": direction_id, "alpha": float(alpha), "text": text,
            "tokens": tokens, "finish": "stop"}


class ActivationSetTests(unittest.TestCase):
    def test_shape_and_label_validation(self):
        activation_set = planted_activation_set()
        self.assertEqual(activation_set.n_items, len(TASKS) * len(CELLS))
        self.assertEqual(activation_set.hidden, 24)
        self.assertEqual(activation_set.matrix(2).shape, (activation_set.n_items, 24))
        with self.assertRaises(ProbeError):
            activation_set.matrix(99)
        with self.assertRaises(ProbeError):
            ActivationSet(("a", "a"), (0,), np.zeros((2, 1, 3)), (1.0,), {})

    def test_mask_and_select(self):
        activation_set = planted_activation_set()
        subset = activation_set.select(activation_set.mask(validity=VALIDITY_NEGATIVE))
        self.assertEqual(subset.n_items, len(TASKS) * 2)
        self.assertEqual(set(subset.column("validity")), {VALIDITY_NEGATIVE})

    def test_npz_roundtrip(self):
        activation_set = planted_activation_set()
        with tempfile.TemporaryDirectory() as directory:
            path = save_activation_set(Path(directory) / "a.npz", activation_set)
            restored = load_activation_set(path)
        self.assertEqual(restored.ids, activation_set.ids)
        self.assertEqual(restored.layers, activation_set.layers)
        self.assertEqual(restored.labels["tone"], activation_set.labels["tone"])
        np.testing.assert_allclose(restored.norms, activation_set.norms)
        np.testing.assert_allclose(restored.matrix(2), activation_set.matrix(2))


class AucTests(unittest.TestCase):
    def test_roc_auc_matches_known_values(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)  # a tie is half credit
        self.assertIsNone(roc_auc([1, 1], [0.1, 0.9]))

    def test_roc_auc_agrees_with_sklearn(self):
        from sklearn.metrics import roc_auc_score
        generator = np.random.default_rng(3)
        labels = generator.integers(0, 2, 40).tolist()
        scores = generator.standard_normal(40).round(2).tolist()
        self.assertAlmostEqual(roc_auc(labels, scores), float(roc_auc_score(labels, scores)), places=12)


class LeaveOneTaskOutTests(unittest.TestCase):
    def test_planted_layer_is_separable_and_others_are_chance(self):
        activation_set = planted_activation_set(planted_layer=2, tone_effect=6.0)
        rows = loo_auc_by_layer(activation_set, label_name="tone", positive=TONE_POSITIVE,
                                negative=TONE_NEGATIVE)
        by_layer = {item.layer: item.auc for item in rows}
        self.assertEqual(len(rows), 5)
        self.assertGreaterEqual(by_layer[2], 0.99)
        for layer in (0, 1, 3, 4):
            self.assertLess(by_layer[layer], 0.80, "layer %d should be near chance" % layer)
        self.assertTrue(all(item.n_groups == len(TASKS) for item in rows))

    def test_grouping_holds_out_every_cell_of_a_task(self):
        activation_set = planted_activation_set()
        seen = []
        from src import probe

        original = probe.fit_probe

        def spy(features, labels, **kwargs):
            seen.append(features.shape[0])
            return original(features, labels, **kwargs)

        probe.fit_probe = spy
        try:
            loo_auc_by_layer(activation_set, label_name="tone", positive=TONE_POSITIVE,
                             negative=TONE_NEGATIVE, layers=[0])
        finally:
            probe.fit_probe = original
        # 10 folds, each training on 9 tasks x 4 cells.
        self.assertEqual(seen, [36] * len(TASKS))

    def test_layer_choice_is_argmax_with_ties_to_the_lower_layer(self):
        rows = (LayerAuc(0, 0.5, 8, 4), LayerAuc(1, 0.9, 8, 4), LayerAuc(2, 0.9, 8, 4),
                LayerAuc(3, None, 8, 4, "one_class_out_of_fold"))
        self.assertEqual(choose_layer(rows), 1)
        self.assertEqual(choose_layer((LayerAuc(7, 0.4, 8, 4), LayerAuc(3, 0.4, 8, 4))), 3)
        with self.assertRaises(ProbeError):
            choose_layer((LayerAuc(0, None, 8, 4, "x"),))

    def test_validity_probe_is_weaker_when_its_effect_is_smaller(self):
        activation_set = planted_activation_set(tone_effect=6.0, validity_effect=0.6, seed=5)
        tone = {item.layer: item.auc for item in loo_auc_by_layer(
            activation_set, label_name="tone", positive=TONE_POSITIVE, negative=TONE_NEGATIVE)}
        validity = {item.layer: item.auc for item in loo_auc_by_layer(
            activation_set, label_name="validity", positive=VALIDITY_POSITIVE,
            negative=VALIDITY_NEGATIVE)}
        self.assertGreater(tone[2], validity[2] + 0.05)

    def test_binary_labels_rejects_a_third_value(self):
        with self.assertRaises(ProbeError):
            binary_labels(("hostile", "neutral", "sarcastic"), TONE_POSITIVE, TONE_NEGATIVE)


class HoldoutEvaluationTests(unittest.TestCase):
    def test_probe_fitted_on_discovery_transfers_to_a_fresh_split(self):
        discovery = planted_activation_set(seed=1)
        holdout = planted_activation_set(seed=2, tasks=tuple("DGS-1%02d" % i for i in range(10)))
        layer = choose_layer(loo_auc_by_layer(discovery, label_name="tone",
                                              positive=TONE_POSITIVE, negative=TONE_NEGATIVE))
        self.assertEqual(layer, 2)
        probe = fit_probe(discovery.matrix(layer),
                          binary_labels(discovery.column("tone"), TONE_POSITIVE, TONE_NEGATIVE),
                          layer=layer, label_name="tone", positive_label=TONE_POSITIVE)
        self.assertEqual(probe.n_train, discovery.n_items)
        auc = roc_auc(binary_labels(holdout.column("tone"), TONE_POSITIVE, TONE_NEGATIVE),
                      probe.score(holdout.matrix(layer)))
        self.assertGreaterEqual(auc, 0.99)
        # A probe trained on the planted layer must NOT transfer to a noise layer.
        noise = fit_probe(discovery.matrix(0),
                          binary_labels(discovery.column("tone"), TONE_POSITIVE, TONE_NEGATIVE),
                          layer=0, label_name="tone", positive_label=TONE_POSITIVE)
        self.assertLess(roc_auc(binary_labels(holdout.column("tone"), TONE_POSITIVE, TONE_NEGATIVE),
                                noise.score(holdout.matrix(0))), 0.80)

    def test_standardisation_uses_the_training_fold_only(self):
        features = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0], [6.0, 10.0]])
        probe = fit_probe(features, np.array([0, 0, 1, 1]), layer=0, label_name="tone",
                          positive_label=TONE_POSITIVE)
        np.testing.assert_allclose(probe.mean, [3.0, 10.0])
        # A constant column keeps scale 1 instead of dividing by zero.
        np.testing.assert_allclose(probe.scale, [np.std([0.0, 2.0, 4.0, 6.0]), 1.0])
        with self.assertRaises(ProbeError):
            fit_probe(features, np.array([1, 1, 1, 1]), layer=0, label_name="tone",
                      positive_label=TONE_POSITIVE)

    def test_the_probe_penalty_is_l2_at_c_equals_one(self):
        """scikit-learn 1.8 deprecated ``penalty='l2'``; its default *is* pure L2."""
        from sklearn.linear_model import LogisticRegression

        from src.probe import PROBE_C, PROBE_MAX_ITER, PROBE_RANDOM_STATE

        generator = np.random.default_rng(21)
        features = generator.standard_normal((30, 6))
        labels = (features[:, 0] + 0.3 * generator.standard_normal(30) > 0).astype(int)
        probe = fit_probe(features, labels, layer=0, label_name="tone",
                          positive_label=TONE_POSITIVE)
        mean, scale = features.mean(axis=0), features.std(axis=0)
        reference = LogisticRegression(C=PROBE_C, l1_ratio=0.0, solver="liblinear",
                                       max_iter=PROBE_MAX_ITER,
                                       random_state=PROBE_RANDOM_STATE
                                       ).fit((features - mean) / scale, labels)
        np.testing.assert_allclose(probe.coef, reference.coef_.ravel(), rtol=1e-12, atol=1e-12)
        self.assertEqual(PROBE_C, 1.0)


class CorrelationTests(unittest.TestCase):
    def test_spearman_is_rank_based(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 100, 300, 400]), 1.0)  # monotone, not linear
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [5, 6, 6, 7]), 1.0)  # mid-ranks for ties
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_cell_demeaning_removes_a_pure_cell_offset(self):
        # Within every cell the relation is exactly negative; the cells differ only by a
        # large offset that would otherwise dominate the pooled correlation.
        rows = []
        for cell_index, cell in enumerate(("easy__a__n", "easy__a__h", "hard__a__n")):
            for item_index in range(6):
                task_id = "T%d" % item_index
                score = item_index + 100.0 * cell_index
                value = -item_index + 100.0 * cell_index
                rows.append((task_id, cell, score, value))
        result = cell_demeaned_spearman(rows, bootstrap_samples=200)
        self.assertAlmostEqual(result.rho, -1.0, places=6)
        self.assertEqual(result.n_items, 6)
        self.assertEqual(result.n_cells, 3)
        self.assertTrue(result.excludes_zero)

    def test_independent_columns_give_an_interval_covering_zero(self):
        generator = np.random.default_rng(11)
        rows = [("T%d" % item, "cell%d" % (item % 2), float(generator.standard_normal()),
                 float(generator.standard_normal())) for item in range(24)]
        result = cell_demeaned_spearman(rows, bootstrap_samples=400)
        self.assertLess(abs(result.rho), 0.5)
        self.assertFalse(result.excludes_zero)

    def test_bootstrap_is_deterministic_and_reports_missing_inputs(self):
        rows = [("T%d" % item, "cell", float(item), float(-item)) for item in range(8)]
        first = cell_demeaned_spearman(rows, bootstrap_samples=200)
        second = cell_demeaned_spearman(rows, bootstrap_samples=200)
        self.assertEqual((first.rho, first.ci95_lower, first.ci95_upper),
                         (second.rho, second.ci95_lower, second.ci95_upper))
        empty = cell_demeaned_spearman([])
        self.assertIsNone(empty.rho)
        self.assertEqual(empty.unavailable_reason, "no_available_case_rows")
        single = cell_demeaned_spearman([("T0", "c", 1.0, 2.0), ("T0", "c", 2.0, 1.0)])
        self.assertEqual(single.unavailable_reason, "at_least_two_items_required_for_cluster_ci")


class DirectionTests(unittest.TestCase):
    def test_mean_difference_direction_uses_only_the_requested_arm(self):
        activation_set = planted_activation_set(tone_effect=6.0, seed=7)
        direction = mean_difference_direction(
            activation_set, 2, label_name="tone", positive=TONE_POSITIVE, negative=TONE_NEGATIVE,
            mask=activation_set.mask(validity=VALIDITY_NEGATIVE))
        # The planted axis is coordinate 0 with a +/-6 contrast, so the difference is ~12.
        self.assertGreater(direction[0], 9.0)
        self.assertGreater(abs(direction[0]), 3.0 * float(np.abs(direction[1:]).max()))
        manual = (activation_set.matrix(2)[np.array(
                      [t == TONE_POSITIVE and v == VALIDITY_NEGATIVE
                       for t, v in zip(activation_set.column("tone"),
                                       activation_set.column("validity"))])].mean(axis=0)
                  - activation_set.matrix(2)[np.array(
                      [t == TONE_NEGATIVE and v == VALIDITY_NEGATIVE
                       for t, v in zip(activation_set.column("tone"),
                                       activation_set.column("validity"))])].mean(axis=0))
        np.testing.assert_allclose(direction, manual, rtol=1e-10, atol=1e-10)

    def test_direction_needs_both_arms(self):
        activation_set = planted_activation_set()
        with self.assertRaises(ProbeError):
            mean_difference_direction(activation_set, 2, label_name="tone",
                                      positive=TONE_POSITIVE, negative="absent")

    def test_scaling_is_alpha_times_the_layer_norm(self):
        direction = np.array([3.0, 4.0, 0.0])
        scaled = scaled_direction(direction, 2.0, 10.0)
        self.assertAlmostEqual(float(np.linalg.norm(scaled)), 20.0)
        np.testing.assert_allclose(scaled, np.array([12.0, 16.0, 0.0]))
        np.testing.assert_allclose(scaled_direction(direction, 0.0, 10.0), np.zeros(3))
        # Doubling alpha doubles the vector: the dose is linear in alpha by construction.
        np.testing.assert_allclose(scaled_direction(direction, 4.0, 10.0), 2.0 * scaled)
        with self.assertRaises(ProbeError):
            scaled_direction(np.zeros(3), 1.0, 10.0)
        with self.assertRaises(ProbeError):
            scaled_direction(direction, 1.0, 0.0)
        np.testing.assert_allclose(np.linalg.norm(unit(direction)), 1.0)

    def test_random_directions_are_seeded_unit_and_distinct(self):
        first = random_unit_directions(64, 5)
        second = random_unit_directions(64, 5)
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)  # frozen seeds reproduce exactly
            self.assertAlmostEqual(float(np.linalg.norm(left)), 1.0, places=12)
        for index in range(5):
            for other in range(index + 1, 5):
                self.assertGreater(float(np.linalg.norm(first[index] - first[other])), 0.5)
        self.assertFalse(np.allclose(random_unit_direction(64, 1, seed_prefix=STEER_SEED),
                                     random_unit_direction(64, 1, seed_prefix="OTHER")))
        # Matched norm: a random control at dose alpha has the same length as the tone dose.
        self.assertAlmostEqual(float(np.linalg.norm(scaled_direction(first[0], 2.0, 7.5))),
                               float(np.linalg.norm(scaled_direction(np.arange(1.0, 65.0), 2.0, 7.5))),
                               places=9)


class DegenerateDoseTests(unittest.TestCase):
    def test_rule_is_strictly_greater_than_one_half(self):
        self.assertEqual(DEGENERATE_NON_ANSWER_FRACTION, 0.5)
        self.assertFalse(is_degenerate_dose(0.0))
        self.assertFalse(is_degenerate_dose(0.5))
        self.assertTrue(is_degenerate_dose(0.55))
        self.assertTrue(is_degenerate_dose(1.0))

    def test_degenerate_doses_are_excluded_from_the_monotonicity_check(self):
        baseline = [entry("T%d" % i, alpha=0.0, margin=5.0) for i in range(4)]
        base_readouts = [item_readout(item, direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                      task_id=item["id"], canonical_answer="D")
                         for item in baseline]
        doses = []
        for alpha, margin, invalid in ((0.5, 4.0, 0), (1.0, 3.0, 0), (2.0, 2.0, 3)):
            items = [item_readout(entry("T%d" % i, alpha=alpha, margin=margin, valid=i >= invalid),
                                  direction_id="tone", alpha=alpha, task_id="T%d" % i,
                                  canonical_answer="D")
                     for i in range(4)]
            doses.append(dose_readout(items, base_readouts, direction_id="tone", alpha=alpha))
        self.assertTrue(doses[2].degenerate)  # 3 of 4 items have no parseable answer
        monotone, used, note = monotone_in_alpha(doses)
        self.assertEqual(used, (0.5, 1.0))
        self.assertTrue(monotone)
        self.assertIn("2", note)


class ReadoutTests(unittest.TestCase):
    def test_token_trace_and_m1_are_recovered_from_a_mocked_generation(self):
        record = entry("DGS-001", margin=3.0)
        tokens = tokens_from_entry(record)
        self.assertEqual("".join(token.text for token in tokens), record["text"])
        readout = item_readout(record, direction_id="tone", alpha=1.0, task_id="DGS-001",
                               canonical_answer="D")
        self.assertAlmostEqual(readout.m1, 3.0, places=9)
        self.assertIsNone(readout.m1_missing_reason)
        self.assertEqual(readout.non_answer, 0.0)
        self.assertEqual(readout.length_tokens, len(tokens))

    def test_duplicate_alternative_texts_are_merged_instead_of_breaking_m1(self):
        """A decoder can emit several token IDs that decode to the same string."""
        record = entry("DGS-001", margin=2.0)
        option = record["tokens"][-1]
        option["top_logprobs"] = option["top_logprobs"] + [{"text": " D", "logprob": -9.0}]
        readout = item_readout(record, direction_id="tone", alpha=1.0, task_id="DGS-001",
                               canonical_answer="D")
        self.assertIsNone(readout.m1_missing_reason)
        self.assertGreater(readout.m1, 2.0)  # log-sum-exp adds a little mass to D

    def test_a_text_and_trace_mismatch_is_rejected_rather_than_scored(self):
        record = entry("DGS-001")
        record["text"] = record["text"] + " tampered"
        with self.assertRaises(ProbeError):
            tokens_from_entry(record)

    def test_a_non_answer_is_missing_for_m1_and_counted_as_its_own_outcome(self):
        readout = item_readout(entry("DGS-001", valid=False), direction_id="tone", alpha=4.0,
                               task_id="DGS-001", canonical_answer="D")
        self.assertIsNone(readout.m1)
        self.assertEqual(readout.m1_missing_reason, "m1_invalid_final_answer")
        self.assertEqual(readout.non_answer, 1.0)

    def test_wrong_letter_still_yields_a_signed_margin_against_the_canonical_answer(self):
        readout = item_readout(entry("DGS-001", letter="A", canonical="A", margin=2.0),
                               direction_id="tone", alpha=1.0, task_id="DGS-001",
                               canonical_answer="D")
        # Canonical D was the *low* alternative here, so the margin is negative.
        self.assertLess(readout.m1, 0.0)

    def test_dose_aggregation_and_paired_bootstrap_against_the_baseline(self):
        baseline = [item_readout(entry("T%d" % i, alpha=0.0, margin=5.0),
                                 direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                 task_id="T%d" % i, canonical_answer="D") for i in range(8)]
        steered = [item_readout(entry("T%d" % i, alpha=2.0, margin=3.0, padding=2),
                                direction_id="tone", alpha=2.0, task_id="T%d" % i,
                                canonical_answer="D") for i in range(8)]
        readout = dose_readout(steered, baseline, direction_id="tone", alpha=2.0)
        self.assertEqual(readout.n_items, 8)
        self.assertAlmostEqual(readout.m1_mean, 3.0, places=9)
        self.assertEqual(readout.m1_n, 8)
        self.assertEqual(readout.non_answer_rate, 0.0)
        self.assertAlmostEqual(readout.m1_delta.estimate, -2.0, places=9)
        self.assertLess(readout.m1_delta.ci95_upper, 0.0)
        self.assertTrue(readout.m1_drop_supported)
        # Length grew by exactly the two padding tokens on every item.
        self.assertAlmostEqual(readout.length_delta.estimate, 2.0, places=9)
        self.assertAlmostEqual(readout.non_answer_delta.estimate, 0.0, places=9)

    def test_no_effect_leaves_the_interval_covering_zero(self):
        baseline = [item_readout(entry("T%d" % i, alpha=0.0, margin=2.0 + 0.1 * i),
                                 direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                 task_id="T%d" % i, canonical_answer="D") for i in range(8)]
        steered = [item_readout(entry("T%d" % i, alpha=2.0, margin=2.0 + 0.1 * i),
                                direction_id="random1", alpha=2.0, task_id="T%d" % i,
                                canonical_answer="D") for i in range(8)]
        readout = dose_readout(steered, baseline, direction_id="random1", alpha=2.0)
        self.assertAlmostEqual(readout.m1_delta.estimate, 0.0, places=9)
        self.assertFalse(readout.m1_drop_supported)

    def test_duplicate_generations_for_one_item_are_rejected(self):
        items = [item_readout(entry("T0", alpha=2.0), direction_id="tone", alpha=2.0,
                              task_id="T0", canonical_answer="D") for _ in range(2)]
        with self.assertRaises(ProbeError):
            dose_readout(items, [], direction_id="tone", alpha=2.0)


class VerdictTests(unittest.TestCase):
    def test_j1_needs_a_middle_layer_peak_and_a_holdout_transfer(self):
        strong = (LayerAuc(1, 0.60, 80, 20), LayerAuc(20, 0.88, 80, 20))
        self.assertTrue(verdict_j1(strong, 20, 0.79).supported)
        self.assertFalse(verdict_j1(strong, 20, 0.70).supported)          # holdout bar
        self.assertFalse(verdict_j1((LayerAuc(2, 0.95, 80, 20),), 2, 0.9).supported)  # outside 12-30
        self.assertFalse(verdict_j1((LayerAuc(20, 0.70, 80, 20),), 20, 0.9).supported)  # discovery bar

    def test_j2_needs_above_chance_validity_that_is_weaker_than_tone(self):
        self.assertTrue(verdict_j2(0.90, 0.70, 20).supported)
        self.assertFalse(verdict_j2(0.90, 0.87, 20).supported)  # gap below 0.05
        self.assertFalse(verdict_j2(0.90, 0.50, 20).supported)  # not decodable at all
        self.assertFalse(verdict_j2(None, 0.70, 20).supported)

    def test_j3_needs_a_negative_rho_and_an_interval_excluding_zero(self):
        rows = [("T%d" % item, "cell", float(item), float(-item)) for item in range(10)]
        self.assertTrue(verdict_j3(cell_demeaned_spearman(rows, bootstrap_samples=200)).supported)
        flipped = [("T%d" % item, "cell", float(item), float(item)) for item in range(10)]
        self.assertFalse(verdict_j3(cell_demeaned_spearman(flipped, bootstrap_samples=200)).supported)

    def test_j4_needs_the_alpha2_drop_and_monotonicity(self):
        baseline = [item_readout(entry("T%d" % i, alpha=0.0, margin=6.0),
                                 direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                 task_id="T%d" % i, canonical_answer="D") for i in range(8)]
        doses = []
        for alpha, margin in ((0.5, 5.0), (1.0, 4.0), (2.0, 3.0)):
            items = [item_readout(entry("T%d" % i, alpha=alpha, margin=margin),
                                  direction_id="tone", alpha=alpha, task_id="T%d" % i,
                                  canonical_answer="D") for i in range(8)]
            doses.append(dose_readout(items, baseline, direction_id="tone", alpha=alpha))
        verdict = verdict_j4(doses)
        self.assertTrue(verdict.supported)
        self.assertEqual(verdict.detail["doses_used"], [0.5, 1.0, 2.0])
        # Break monotonicity at the middle dose and the verdict must fail.
        broken = list(doses)
        rebuilt = [item_readout(entry("T%d" % i, alpha=1.0, margin=9.0), direction_id="tone",
                                alpha=1.0, task_id="T%d" % i, canonical_answer="D")
                   for i in range(8)]
        broken[1] = dose_readout(rebuilt, baseline, direction_id="tone", alpha=1.0)
        self.assertFalse(verdict_j4(broken).supported)

    def test_j5_fails_when_a_random_direction_reproduces_the_drop(self):
        baseline = [item_readout(entry("T%d" % i, alpha=0.0, margin=6.0),
                                 direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                 task_id="T%d" % i, canonical_answer="D") for i in range(8)]

        def dose(direction_id, margin):
            items = [item_readout(entry("T%d" % i, alpha=2.0, margin=margin),
                                  direction_id=direction_id, alpha=2.0, task_id="T%d" % i,
                                  canonical_answer="D") for i in range(8)]
            return dose_readout(items, baseline, direction_id=direction_id, alpha=2.0)

        quiet = [dose("random%d" % index, 6.0) for index in range(1, 6)]
        quiet.append(dose("unrelated_style", 6.0))
        self.assertTrue(verdict_j5(quiet).supported)
        noisy = list(quiet)
        noisy[2] = dose("random3", 1.0)
        verdict = verdict_j5(noisy)
        self.assertFalse(verdict.supported)
        self.assertEqual(verdict.detail["directions_with_supported_m1_drop"], ["random3"])
        self.assertFalse(verdict_j5([]).supported)  # nothing checked is not a pass

    def test_j6_reads_either_the_non_answer_channel_or_distress(self):
        baseline = [item_readout(entry("T%d" % i, alpha=0.0, margin=4.0),
                                 direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                 task_id="T%d" % i, canonical_answer="D") for i in range(8)]
        quiet = [item_readout(entry("T%d" % i, alpha=2.0, margin=4.0), direction_id="tone",
                              alpha=2.0, task_id="T%d" % i, canonical_answer="D")
                 for i in range(8)]
        calm = [dose_readout(quiet, baseline, direction_id="tone", alpha=2.0)]
        self.assertFalse(verdict_j6(calm).supported)
        breaking = [item_readout(entry("T%d" % i, alpha=4.0, margin=4.0, valid=i >= 4),
                                 direction_id="tone", alpha=4.0, task_id="T%d" % i,
                                 canonical_answer="D") for i in range(10)]
        baseline += [item_readout(entry("T%d" % i, alpha=0.0, margin=4.0),
                                  direction_id=BASELINE_DIRECTION_ID, alpha=0.0,
                                  task_id="T%d" % i, canonical_answer="D") for i in (8, 9)]
        loud = [dose_readout(breaking, baseline, direction_id="tone", alpha=4.0)]
        self.assertTrue(verdict_j6(loud).supported)
        from src.confirm import bootstrap_contrast
        distress = {2.0: bootstrap_contrast([("T%d" % i, 3.0) for i in range(8)], "seed")}
        self.assertTrue(verdict_j6(calm, distress).supported)


class ScriptTests(unittest.TestCase):
    """The pure parts of ``scripts/run_phase3.py``: item building, plans and rendering."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_phase3

        self.script = run_phase3

    def test_dose_plan_generates_the_baseline_once(self):
        plan = self.script._dose_plan((0.0, 0.5, 1.0, 2.0, 4.0), ("tone", "random1"))
        self.assertEqual(plan[0], (BASELINE_DIRECTION_ID, 0.0))
        self.assertEqual(sum(1 for _, alpha in plan if alpha == 0.0), 1)
        self.assertEqual(len(plan), 1 + 2 * 4)

    def test_a_torn_final_line_is_dropped_and_repaired_but_earlier_damage_stops_the_run(self):
        good = [json.dumps({"id": "T%d" % index, "direction_id": "tone", "alpha": 2.0})
                for index in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steering_outputs.jsonl"
            path.write_text("\n".join(good) + "\n" + good[0][:20], encoding="utf-8", newline="\n")
            entries = self.script._load_steering_entries(path, repair=True)
            self.assertEqual(len(entries), 3)
            # The fragment is gone from disk, so the next append cannot glue onto it.
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(len(self.script._load_steering_entries(path)), 3)
            path.write_text(good[0][:20] + "\n" + "\n".join(good) + "\n", encoding="utf-8",
                            newline="\n")
            with self.assertRaises(SystemExit):
                self.script._load_steering_entries(path)

    def test_item_builders_filter_the_raw_stream(self):
        from src.protocol import load_protocol

        protocol = load_protocol(ROOT)
        rows = [
            {"trajectory_kind": "greedy", "sample_index": 0, "turn_label": "measured",
             "cell_id": "easy__accurate__hostile", "model_id": "google/gemma-2-9b-it",
             "split": "discovery", "task_id": "DGS-003", "prompt_sha256": "a" * 64,
             "response_id": "r1", "messages": [{"role": "user", "content": "x"}]},
            # wrong model, wrong split, wrong turn, resample: each must be dropped
            {"trajectory_kind": "greedy", "sample_index": 0, "turn_label": "measured",
             "cell_id": "easy__accurate__hostile", "model_id": "other/model",
             "split": "discovery", "task_id": "DGS-004", "prompt_sha256": "b" * 64,
             "response_id": "r2", "messages": [{"role": "user", "content": "x"}]},
            {"trajectory_kind": "greedy", "sample_index": 0, "turn_label": "measured",
             "cell_id": "easy__accurate__hostile", "model_id": "google/gemma-2-9b-it",
             "split": "holdout", "task_id": "DGS-001", "prompt_sha256": "c" * 64,
             "response_id": "r3", "messages": [{"role": "user", "content": "x"}]},
            {"trajectory_kind": "greedy", "sample_index": 0, "turn_label": "onset",
             "cell_id": "easy__accurate__hostile", "model_id": "google/gemma-2-9b-it",
             "split": "discovery", "task_id": "DGS-005", "prompt_sha256": "d" * 64,
             "response_id": "r4", "messages": [{"role": "user", "content": "x"}]},
            {"trajectory_kind": "resample", "sample_index": 1, "turn_label": "measured",
             "cell_id": "easy__accurate__hostile", "model_id": "google/gemma-2-9b-it",
             "split": "discovery", "task_id": "DGS-007", "prompt_sha256": "e" * 64,
             "response_id": "r5", "messages": [{"role": "user", "content": "x"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            items = self.script.measured_items(path, protocol, split="discovery")
        self.assertEqual([item["task_id"] for item in items], ["DGS-003"])
        self.assertEqual(items[0]["tone"], "hostile")
        self.assertEqual(items[0]["validity"], "accurate")
        self.assertEqual(items[0]["id"], "DGS-003|easy__accurate__hostile")

    def test_steering_items_are_the_twenty_holdout_tasks_as_one_neutral_turn(self):
        from src.protocol import load_protocol

        items = self.script._steering_items(load_protocol(ROOT))
        self.assertEqual(len(items), 20)
        self.assertTrue(all(len(item["messages"]) == 1 for item in items))
        self.assertTrue(all(item["messages"][0]["role"] == "user" for item in items))
        self.assertTrue(all(item["canonical_answer"] in "ABCD" for item in items))
        self.assertIn("Options:", items[0]["messages"][0]["content"])

    def test_returned_activations_are_realigned_to_the_requested_order(self):
        items = [{"id": "b"}, {"id": "a"}]
        response = {"ids": ["a", "b"], "layers": [0, 1],
                    "activations": np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
                    "norms": [1.0, 1.0]}
        activation_set = self.script._activation_set_from_response(items, response, ("id",))
        self.assertEqual(activation_set.ids, ("b", "a"))
        np.testing.assert_allclose(activation_set.matrix(0), [[3.0], [1.0]])

    def test_build_directions_produces_seven_matched_norm_directions(self):
        discovery = planted_activation_set(seed=13)
        style = ActivationSet(
            ("S1|style__verbose", "S1|style__neutral_reference"), discovery.layers,
            np.stack([np.ones((len(discovery.layers), discovery.hidden)),
                      np.zeros((len(discovery.layers), discovery.hidden))]).astype(np.float32),
            discovery.norms,
            {"task_id": ("S1", "S1"),
             "cell_id": ("style__verbose", "style__neutral_reference")})
        directions = self.script.build_directions(discovery, style, 2)
        self.assertEqual(sorted(directions), sorted(
            ["tone", "unrelated_style"] + ["random%d" % index for index in range(1, 6)]))
        lengths = {name: float(np.linalg.norm(vector)) for name, vector in directions.items()}
        for value in lengths.values():
            self.assertAlmostEqual(value, discovery.norm(2), places=6)

    def test_markdown_renderers_run_on_synthetic_payloads(self):
        localization = {
            "preregistration": "notes/preregistration_v4_phase3.md",
            "model_id": "google/gemma-2-9b-it",
            "item_counts": {"discovery": 80, "holdout": 80, "note": "20 x 4"},
            "layers_extracted": [0, 1, 2], "hidden_size": 8, "chosen_layer": 2,
            "layer_choice_rule": "argmax", "holdout_correlation": {
                "rho": -0.3, "ci95_lower": -0.5, "ci95_upper": -0.1, "n_items": 20,
                "n_pairs": 78, "n_cells": 8, "m1_missing_endpoints": 2,
                "unavailable_reason": None, "estimator": "cell-demeaned"},
            "discovery_loo_auc": {
                "tone": [{"layer": index, "auc": 0.5 + 0.1 * index, "n_items": 80,
                          "n_groups": 20, "unavailable_reason": None} for index in range(3)],
                "validity": [{"layer": index, "auc": 0.5, "n_items": 80, "n_groups": 20,
                              "unavailable_reason": None} for index in range(3)]},
            "discovery_loo_auc_at_chosen_layer": {"tone": 0.7, "validity": 0.5},
            "holdout_auc_at_chosen_layer": {"tone": 0.8, "validity": 0.6},
            "verdicts": [verdict_j1((LayerAuc(20, 0.9, 80, 20),), 20, 0.8).to_dict()],
            "interpretation_ceiling": "not evidence of experience",
        }
        text = self.script.render_localization_markdown(localization)
        self.assertIn("L\\* = 2", text)
        self.assertIn("| J1 |", text)
        steering = {
            "preregistration": "notes/preregistration_v4_phase3.md",
            "model_id": "google/gemma-2-9b-it", "layer": 2, "layer_norm": 30.0,
            "alphas": [0.0, 2.0], "direction_ids": ["tone"],
            "direction_construction": {"tone": "t", "random": "r", "unrelated": "u", "dose": "d"},
            "items": {"n": 20, "prompt": "render_task"}, "max_new_tokens": 512,
            "degenerate_dose_rule": "> 50%", "monotonicity": {
                "monotone": True, "doses_used": [0.5, 1.0], "note": None},
            "doses": [{"direction_id": "tone", "alpha": 2.0, "n_items": 20, "m1_mean": 1.0,
                       "m1_n": 18, "non_answer_rate": 0.1, "mean_length_tokens": 120.0,
                       "degenerate": False,
                       "m1_delta": {"estimate": -1.0, "ci95_lower": -2.0, "ci95_upper": -0.2,
                                    "p_two_sided": 0.01, "n_items": 18,
                                    "unavailable_reason": None},
                       "non_answer_delta": {"estimate": 0.1, "ci95_lower": -0.1,
                                            "ci95_upper": 0.3, "p_two_sided": 0.4,
                                            "n_items": 20, "unavailable_reason": None},
                       "length_delta": {"estimate": 5.0, "ci95_lower": 1.0, "ci95_upper": 9.0,
                                        "p_two_sided": 0.02, "n_items": 20,
                                        "unavailable_reason": None}}],
            "distress": {"judged": False, "scores": [], "deltas": {}},
            "verdicts": [], "interpretation_ceiling": "not evidence of experience",
        }
        self.assertIn("`tone`", self.script.render_steering_markdown(steering))
        report = self.script.render_report_markdown(localization, steering)
        self.assertIn("J1", report)
        self.assertIn("not evidence of experience", report)


if __name__ == "__main__":
    unittest.main()
