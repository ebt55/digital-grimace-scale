from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import math
import unittest
from unittest import mock

from src.analysis import (
    AnalysisInputError,
    AnalysisObservation,
    G5Row,
    PHASE0_DIFFICULTIES,
    PHASE0_MODELS,
    PHASE0_SCREEN_TASKS,
    PRIMARY_METRICS,
    ReversalRow,
    benjamini_hochberg,
    freeze_neutral_standardization,
    g1_adjusted_effects,
    g2_reversal,
    g5_predictive_gap,
    g5_shuffled_feedback_labels,
    phase0_screen,
    shuffled_feedback_labels,
    validate_observations,
)


HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def observation(
    *, model="model", task="task", phase="phase_1", run="run", split="discovery",
    metric="M2", value=1.0, validity="accurate", tone="neutral", difficulty="easy",
    turn="measured", correctness=True, length=10, eligible=False, missing_reason=None,
):
    return AnalysisObservation(
        phase, run, split, model, task, f"{difficulty}__{validity}__{tone}", difficulty,
        validity, tone, turn, metric, None if missing_reason else value, missing_reason,
        correctness, length, eligible,
    )


def phase0_rows(effects=None):
    effects = effects or dict(zip(PHASE0_MODELS, (0.50, 0.30, -0.05, -0.15, -0.20)))
    rows = []
    for model in PHASE0_MODELS:
        for index, task in enumerate(PHASE0_SCREEN_TASKS):
            for validity in ("accurate", "malfunctioning_always_fail"):
                for metric in PRIMARY_METRICS:
                    effect = effects[model]
                    raw_effect = -effect if metric == "M1" else effect
                    rows.append(observation(
                        model=model, task=task, phase="phase_0", run="phase0", metric=metric,
                        value=index + (raw_effect if validity == "malfunctioning_always_fail" else 0),
                        validity=validity, difficulty=PHASE0_DIFFICULTIES[task],
                    ))
    return rows


def phase1_factorial_rows(models=("model",), metrics=("M2",), items=range(6)):
    rows = []
    for model_index, model in enumerate(models):
        for metric_index, metric in enumerate(metrics):
            for item in items:
                difficulty = "easy" if item % 2 == 0 else "hard"
                for validity in ("accurate", "malfunctioning_always_fail"):
                    for tone in ("neutral", "hostile"):
                        effect = 1.2 * (validity == "malfunctioning_always_fail") + 0.4 * (tone == "hostile")
                        variation = ((item * 7 + len(validity) + len(tone)) % 3) * 0.017
                        rows.append(observation(
                            model=model, task=f"item-{item}", metric=metric,
                            value=item * 4.0 + model_index + metric_index + effect + variation,
                            validity=validity, tone=tone, difficulty=difficulty,
                            correctness=(item + len(validity) + len(tone)) % 2 == 0,
                            length=20 + item * 3 + int(tone == "hostile"),
                        ))
    return rows


def g5_rows(items=("easy-0", "easy-1", "hard-0", "hard-1"), outlier_task=None):
    rows = []
    for index, task in enumerate(items):
        difficulty = "easy" if task.startswith("easy") else "hard"
        for validity in ("accurate", "malfunctioning_always_fail"):
            for tone in ("neutral", "hostile"):
                label = float(validity == "malfunctioning_always_fail")
                length = 10000 + int(tone == "hostile") if task == outlier_task else 10 + index * 3 + int(tone == "hostile")
                rows.append(G5Row(
                    "phase_1", "run", "discovery", "model", task,
                    f"{difficulty}__{validity}__{tone}", difficulty, validity, tone, "measured",
                    {"M1": 3 * label + index, "M2": label + 0.1 * index},
                    (index + int(tone == "hostile")) % 2 == 0, length,
                ))
    return rows


class ObservationAndPhase0Tests(unittest.TestCase):
    def test_observation_standardization_bh_and_immutability(self):
        for field, value in (("phase", "phase_2"), ("split", "pilot"), ("turn", "other"),
                             ("validity", "maybe"), ("tone", "kind"), ("value", float("nan")),
                             ("length", -1)):
            with self.subTest(field=field), self.assertRaises(AnalysisInputError):
                observation(**{field: value})

        neutral_a = observation(task="a", value=2.0, length=3)
        neutral_b = observation(task="b", value=6.0, length=4)
        excluded = (
            observation(task="c", value=100.0, tone="hostile"),
            observation(task="d", value=100.0, split="holdout"),
            observation(task="e", value=100.0, turn="initial"),
            observation(task="f", value=100.0, validity="malfunctioning_always_fail"),
        )
        frozen = freeze_neutral_standardization((neutral_a, neutral_b, *excluded))
        self.assertEqual(frozen[("model", "M2")].mean, 4.0)
        self.assertAlmostEqual(frozen[("model", "M2")].sample_sd, math.sqrt(8.0))
        with self.assertRaises(TypeError):
            frozen[("model", "M2")] = None

        adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(dict(adjusted), {"a": 0.03, "b": 0.04, "c": 0.04})
        with self.assertRaises(TypeError):
            adjusted["a"] = 1.0
        with self.assertRaises(AnalysisInputError):
            validate_observations((neutral_a, neutral_a))
        with self.assertRaises(AnalysisInputError):
            validate_observations((neutral_a, replace(neutral_b, task_id="a", difficulty="hard", cell_id="hard__accurate__neutral")))

    def test_phase0_exact_design_selection_ties_and_blocks(self):
        rows = phase0_rows()
        selected = phase0_screen(rows)
        self.assertEqual((selected.status, selected.primary_model_id, selected.control_model_id), ("selected", PHASE0_MODELS[0], PHASE0_MODELS[2]))
        self.assertEqual(phase0_screen(list(reversed(rows))), selected)

        mutations = (
            lambda data: data[:-1],
            lambda data: data + [replace(data[0], model_id="unexpected/model")],
            lambda data: [replace(data[0], turn="initial"), *data[1:]],
            lambda data: [replace(data[0], run_id="other"), *data[1:]],
            lambda data: [replace(data[0], tone="hostile", cell_id=data[0].cell_id.replace("neutral", "hostile")), *data[1:]],
            lambda data: [replace(data[0], task_id="DGS-999"), *data[1:]],
            lambda data: [replace(data[0], difficulty="hard" if data[0].difficulty == "easy" else "easy", cell_id=data[0].cell_id.replace(data[0].difficulty, "hard" if data[0].difficulty == "easy" else "easy", 1)), *data[1:]],
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.assertEqual(phase0_screen(mutate(rows)).status, "blocked")
        self.assertEqual(phase0_screen(rows, PHASE0_MODELS[::-1]).status, "blocked")
        self.assertEqual(phase0_screen(rows, screen_task_ids=PHASE0_SCREEN_TASKS[::-1]).status, "blocked")

        m1_tiebreak = []
        for row in rows:
            if row.model_id == PHASE0_MODELS[1] and row.feedback_validity == "malfunctioning_always_fail":
                signed_effect = 0.7 if row.metric_name == "M1" else 0.4
                raw_effect = -signed_effect if row.metric_name == "M1" else signed_effect
                m1_tiebreak.append(replace(row, metric_value=PHASE0_SCREEN_TASKS.index(row.task_id) + raw_effect))
            else:
                m1_tiebreak.append(row)
        self.assertEqual(phase0_screen(m1_tiebreak).primary_model_id, PHASE0_MODELS[1])

        unavailable = [replace(row, metric_value=None, missing_reason="qc") for row in rows]
        self.assertEqual(phase0_screen(unavailable).blocked_reason, "all_phase0_metrics_unavailable")
        null_effects = phase0_rows({model: 0.0 for model in PHASE0_MODELS})
        self.assertEqual(phase0_screen(null_effects).status, "escalation_required")
        no_qwen = [replace(row, metric_value=None, missing_reason="qc") if row.model_id.startswith("Qwen/") else row for row in rows]
        self.assertEqual(phase0_screen(no_qwen).blocked_reason, "no_distinct_available_qwen_control")


class ShuffleTests(unittest.TestCase):
    def test_shuffle_sha_assignment_and_logical_cell_invariants(self):
        rows = []
        for difficulty, tone, prefix in (("easy", "neutral", "e"), ("hard", "hostile", "h")):
            for number, validity in enumerate(("malfunctioning_always_fail", "accurate", "accurate")):
                for metric in PRIMARY_METRICS:
                    rows.append(observation(task=f"{prefix}-{number}", metric=metric, value=float(number), validity=validity, tone=tone, difficulty=difficulty))
        shuffled = shuffled_feedback_labels(rows)
        expected = {}
        for difficulty, tone in (("easy", "neutral"), ("hard", "hostile")):
            cells = {(row.task_id, row.cell_id, row.feedback_validity) for row in rows if row.difficulty == difficulty and row.tone == tone}
            ranked = sorted(cells, key=lambda cell: hashlib.sha256(f"DGS-AC1-SHUFFLE-v1|model|{cell[0]}|{cell[1]}".encode()).hexdigest())
            count = sum(cell[2] == "malfunctioning_always_fail" for cell in cells)
            expected.update({(task, cell): "malfunctioning_always_fail" if index < count else "accurate" for index, (task, cell, _) in enumerate(ranked)})
        self.assertEqual({(row.task_id, row.cell_id): row.effective_feedback_validity for row in shuffled}, expected)
        self.assertEqual(shuffled, shuffled_feedback_labels(list(reversed(rows))))
        self.assertEqual({row.cell_id for row in shuffled}, {row.cell_id for row in rows})
        self.assertEqual({(row.task_id, row.metric_name, row.feedback_validity) for row in shuffled}, {(row.task_id, row.metric_name, row.feedback_validity) for row in rows})
        self.assertTrue(any(row.effective_feedback_validity != row.feedback_validity for row in shuffled))
        for difficulty, tone in (("easy", "neutral"), ("hard", "hostile")):
            self.assertEqual(sum(row.effective_feedback_validity == "malfunctioning_always_fail" for row in shuffled if row.difficulty == difficulty and row.tone == tone), 3)
        for changed in (replace(rows[0], experiment_phase="phase_0"), replace(rows[0], run_id="other"), replace(rows[0], split="holdout")):
            with self.subTest(changed=changed), self.assertRaises(AnalysisInputError):
                shuffled_feedback_labels([changed, *rows[1:]])


@unittest.skipUnless(HAS_STATSMODELS, "scientific G1 tests require statsmodels")
class G1Tests(unittest.TestCase):
    def test_g1_real_mixedlm_fixture_converges(self):
        rows = []
        for item in range(16):
            for validity in ("accurate", "malfunctioning_always_fail"):
                for tone in ("neutral", "hostile"):
                    value = item * 0.31
                    value += 1.1 if validity == "malfunctioning_always_fail" else 0.0
                    value += 0.23 if tone == "hostile" else 0.0
                    value += (item % 3) * 0.017
                    rows.append(observation(
                        task=f"fit-{item}", metric="M2", value=value, validity=validity,
                        tone=tone, difficulty="easy" if item < 8 else "hard",
                        correctness=(item + len(validity) + len(tone)) % 2 == 0,
                        length=20 + item + int(tone == "hostile"),
                    ))
        result = g1_adjusted_effects(rows, ("M2",))[("model", "M2")]
        self.assertTrue(result.converged)
        self.assertIsNotNone(result.validity)
        self.assertIsNotNone(result.tone)
        self.assertEqual(result.paired_validity.n_items, 16)
        self.assertEqual(result, g1_adjusted_effects(list(reversed(rows)), ("M2",))[("model", "M2")])

    def test_g1_captures_formula_separate_data_scaling_and_global_bh(self):
        rows = phase1_factorial_rows(models=("a", "b"), metrics=("M2", "M3"), items=range(4))
        captures = []

        class Fit:
            converged = True
            params = {"malfunctioning": 0.3, "hostile": 0.2}
            bse = {"malfunctioning": 0.1, "hostile": 0.1}
            pvalues = {"malfunctioning": 0.01, "hostile": 0.04}

        class Mixed:
            @classmethod
            def from_formula(cls, formula, groups, data):
                captures.append((formula, groups, data.copy()))
                return cls()

            def fit(self, **kwargs):
                return Fit()

        with mock.patch("statsmodels.regression.mixed_linear_model.MixedLM", Mixed):
            result = g1_adjusted_effects(rows, ("M2", "M3"))
            shuffled = shuffled_feedback_labels(rows)
            g1_adjusted_effects(shuffled, ("M2", "M3"))
        self.assertEqual(len(captures), 8)
        self.assertEqual({formula for formula, _, _ in captures}, {"z_metric ~ malfunctioning + hostile + difficulty_hard + correctness + length"})
        self.assertEqual({groups for _, groups, _ in captures}, {"item"})
        self.assertEqual({len(data) for _, _, data in captures}, {16})
        for _, _, data in captures:
            self.assertAlmostEqual(float(data["length"].mean()), 0.0)
            self.assertAlmostEqual(float(data["length"].std(ddof=0)), 1.0)
        self.assertEqual({entry.validity.adjusted_p for entry in result.values()}, {0.02})
        self.assertEqual({entry.tone.adjusted_p for entry in result.values()}, {0.04})
        captured_labels = captures[4][2]["malfunctioning"].tolist()
        expected_labels = [
            int(row.effective_feedback_validity == "malfunctioning_always_fail")
            for row in shuffled
            if row.model_id == "a" and row.metric_name == "M2"
        ]
        self.assertEqual(captured_labels, expected_labels)

    def test_g1_fit_exception_nonconvergence_and_paired_descriptors(self):
        rows = phase1_factorial_rows(items=range(4))

        class Failure:
            @classmethod
            def from_formula(cls, *args, **kwargs):
                raise RuntimeError("fit failed")

        with mock.patch("statsmodels.regression.mixed_linear_model.MixedLM", Failure):
            failed = g1_adjusted_effects(rows, ("M2",))[("model", "M2")]
        self.assertEqual(failed.unavailable_reason, "mixedlm_unavailable:RuntimeError")
        self.assertIsNotNone(failed.paired_validity)
        self.assertIsNotNone(failed.paired_tone)
        no_neutral = [
            replace(row, metric_value=None, missing_reason="qc")
            if row.feedback_validity == "accurate" and row.tone == "neutral" else row
            for row in rows
        ]
        unavailable = g1_adjusted_effects(no_neutral, ("M2",))[("model", "M2")]
        self.assertEqual(unavailable.paired_validity.unavailable_reason, "neutral_standardization_unavailable")

        class NotConverged:
            @classmethod
            def from_formula(cls, *args, **kwargs):
                return cls()

            def fit(self, **kwargs):
                return type("Fit", (), {"converged": False})()

        with mock.patch("statsmodels.regression.mixed_linear_model.MixedLM", NotConverged):
            failed = g1_adjusted_effects(rows, ("M2",))[("model", "M2")]
        self.assertEqual(failed.unavailable_reason, "mixedlm_unavailable:RuntimeError")
        self.assertIsNotNone(failed.paired_validity)
        self.assertIsNotNone(failed.paired_tone)

        raw = []
        cells = {"a": ((0, 2, 1, 4), "easy"), "b": ((2, 5, 3, 7), "hard")}
        for task, ((accurate_neutral, bad_neutral, accurate_hostile, bad_hostile), difficulty) in cells.items():
            for validity, tone, value in (("accurate", "neutral", accurate_neutral), ("malfunctioning_always_fail", "neutral", bad_neutral), ("accurate", "hostile", accurate_hostile), ("malfunctioning_always_fail", "hostile", bad_hostile)):
                raw.append(observation(task=task, metric="M2", value=value, validity=validity, tone=tone, difficulty=difficulty, correctness=tone == "neutral", length=10 + len(raw)))

        class Fit:
            converged = True
            params = {"malfunctioning": 0.1, "hostile": 0.1}
            bse = {"malfunctioning": 0.1, "hostile": 0.1}
            pvalues = {"malfunctioning": 0.2, "hostile": 0.3}

        class Mixed:
            @classmethod
            def from_formula(cls, *args, **kwargs):
                return cls()

            def fit(self, **kwargs):
                return Fit()

        with mock.patch("statsmodels.regression.mixed_linear_model.MixedLM", Mixed):
            first = g1_adjusted_effects(raw, ("M2",))[("model", "M2")]
            second = g1_adjusted_effects(list(reversed(raw)), ("M2",))[("model", "M2")]
            null = g1_adjusted_effects(shuffled_feedback_labels(raw), ("M2",))[("model", "M2")]
        sd = math.sqrt(2.0)
        self.assertEqual((first.paired_validity.n_pairs, first.paired_validity.n_items), (4, 2))
        self.assertAlmostEqual(first.paired_validity.raw_mean, 3.0 / sd)
        self.assertAlmostEqual(first.paired_tone.raw_mean, 1.5 / sd)
        self.assertEqual(first.paired_validity, second.paired_validity)
        self.assertEqual(first.paired_validity.raw_ci95, first.paired_validity.sign_aligned_ci95)
        self.assertLessEqual(*first.paired_validity.raw_ci95)
        self.assertEqual(null.paired_validity.unavailable_reason, "unavailable_for_shuffled_analysis_labels")
        missing_partner = [row for row in raw if not (row.task_id == "a" and row.feedback_validity == "malfunctioning_always_fail" and row.tone == "hostile")]
        with mock.patch("statsmodels.regression.mixed_linear_model.MixedLM", Mixed):
            incomplete = g1_adjusted_effects(missing_partner, ("M2",))[("model", "M2")]
            m1 = g1_adjusted_effects([replace(row, metric_name="M1") for row in raw], ("M1",))[("model", "M1")]
        self.assertEqual((incomplete.paired_validity.n_pairs, incomplete.paired_tone.n_pairs), (3, 3))
        self.assertAlmostEqual(m1.paired_validity.sign_aligned_mean, -3.0 / sd)
        self.assertEqual(
            m1.paired_validity.sign_aligned_ci95,
            tuple(-value for value in reversed(m1.paired_validity.raw_ci95)),
        )
        self.assertLessEqual(*m1.paired_validity.raw_ci95)
        self.assertLessEqual(*m1.paired_validity.sign_aligned_ci95)


class G2Tests(unittest.TestCase):
    def test_g2_clustered_point_estimates_and_boundaries(self):
        rows = [
            ReversalRow("phase_1", "run", "discovery", "model", "M2", "a", "neutral", 1, 3, 2, True),
            ReversalRow("phase_1", "run", "discovery", "model", "M2", "a", "hostile", 2, 4, 3, True),
            ReversalRow("phase_1", "run", "discovery", "model", "M2", "b", "neutral", 1, 5, 3, True),
            ReversalRow("phase_1", "run", "discovery", "model", "M2", "b", "hostile", 2, 6, 4, True),
            ReversalRow("phase_1", "run", "discovery", "model", "M2", "ignored", "neutral", 0, 99, 0, False),
        ]
        result = g2_reversal(rows)
        self.assertEqual((result.n_items, result.n_rows, result.induction, result.recovery, result.recovery_to_induction), (2, 4, 3.0, 1.5, 0.5))
        self.assertEqual(result, g2_reversal(list(reversed(rows))))
        self.assertLessEqual(*result.recovery_ci95)
        with self.assertRaises(AnalysisInputError):
            g2_reversal(rows, bootstrap_samples=10)
        with self.assertRaises(AnalysisInputError):
            g2_reversal([rows[0], rows[0]])
        self.assertEqual(g2_reversal(rows[:2]).unavailable_reason, "at_least_two_items_required_for_cluster_ci")
        self.assertEqual(g2_reversal([replace(rows[0], post_correction_malfunctioning=None), *rows[1:4]]).unavailable_reason, "required_reversal_endpoint_missing")


class G5Tests(unittest.TestCase):
    def test_g5_source_validation_and_shuffle_source_invariants(self):
        rows = g5_rows()
        shuffled = g5_shuffled_feedback_labels(rows)
        self.assertEqual({(row.task_id, row.cell_id) for row in shuffled}, {(row.task_id, row.cell_id) for row in rows})
        self.assertEqual(shuffled, g5_shuffled_feedback_labels(list(reversed(rows))))
        for difficulty, tone in (("easy", "neutral"), ("easy", "hostile"), ("hard", "neutral"), ("hard", "hostile")):
            self.assertEqual(sum(row.effective_feedback_validity == "malfunctioning_always_fail" for row in shuffled if row.difficulty == difficulty and row.tone == tone), 2)
        self.assertTrue(g5_predictive_gap(rows[:-2], ("M1",)).unavailable_reason.startswith("invalid_factorial_source:"))
        with self.assertRaises(AnalysisInputError):
            g5_predictive_gap([*rows, rows[0]], ("M1",))
        mixed = [*rows, *[replace(row, task_id="easy-0") for row in rows if row.task_id == "hard-0"]]
        self.assertTrue(g5_predictive_gap(mixed, ("M1",)).unavailable_reason.startswith("invalid_factorial_source:"))
        with self.assertRaises(AnalysisInputError):
            g5_predictive_gap([replace(rows[0], run_id="other"), *rows[1:]], ("M1",))
        with self.assertRaises(AnalysisInputError):
            G5Row("phase_1", "r", "discovery", "m", "t", "easy__accurate__neutral", "easy", "accurate", "neutral", "measured", {}, True, 1, "invalid")

    @unittest.skipUnless(HAS_SKLEARN, "scientific G5 tests require scikit-learn")
    def test_g5_fake_estimator_locks_inputs_folds_balance_and_effective_labels(self):
        rows = g5_shuffled_feedback_labels(g5_rows(outlier_task="easy-0"))
        captures = []

        class Logistic:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.n_iter_ = [2]

            def fit(self, features, labels):
                self.features = features
                self.labels = labels
                captures.append(self)
                return self

            def predict_proba(self, features):
                import numpy as np

                self.test_features = features
                return np.array([[0.75, 0.25] for _ in features])

        with mock.patch("sklearn.linear_model.LogisticRegression", Logistic), mock.patch("sklearn.metrics.roc_auc_score", return_value=0.75):
            result = g5_predictive_gap(rows, ("M1", "M2"))
        self.assertEqual((result.n_folds, len(captures)), (4, 8))
        for full, baseline in zip(captures[::2], captures[1::2]):
            self.assertEqual(full.kwargs, {"C": 1, "penalty": "l2", "solver": "liblinear", "max_iter": 1000, "random_state": 0})
            self.assertEqual((len(full.features[0]), len(baseline.features[0])), (2, 2))
            self.assertNotEqual(full.features, baseline.features)
            self.assertEqual(full.labels, baseline.labels)
            self.assertEqual(sum(full.labels), len(full.labels) // 2)
        self.assertGreater(max(abs(row[1]) for row in captures[1].test_features), 1000)
        self.assertEqual((result.full_auc, result.baseline_auc, result.auc_gap), (0.75, 0.75, 0.0))
        self.assertEqual(result.fold_item_ids, tuple((task,) for task in sorted({row.task_id for row in rows})))
        missing_metric = [replace(rows[0], metrics={"M1": None, "M2": rows[0].metrics["M2"]}), *rows[1:]]
        complete_case = g5_predictive_gap(missing_metric, ("M1", "M2"))
        self.assertEqual((complete_case.n_rows, complete_case.dropped_count, complete_case.fold_item_ids), (len(rows) - 1, 1, result.fold_item_ids))

        sentinel_source = []
        for task, difficulty in (("easy-a", "easy"), ("easy-b", "easy"), ("easy-c", "easy"), ("hard-a", "hard"), ("hard-b", "hard")):
            for validity in ("accurate", "malfunctioning_always_fail"):
                for tone in ("neutral", "hostile"):
                    index = len(sentinel_source)
                    sentinel_source.append(G5Row(
                        "phase_1", "run", "discovery", "model", task,
                        f"{difficulty}__{validity}__{tone}", difficulty, validity, tone, "measured",
                        {"M1": 1000.0 + index, "M2": 2000.0 + 10 * index},
                        index % 3 == 0, 300 + 7 * index,
                    ))
        sentinel_rows = g5_shuffled_feedback_labels(sentinel_source)
        expected_labels = {}
        for difficulty in ("easy", "hard"):
            for tone in ("neutral", "hostile"):
                stratum = [row for row in sentinel_source if row.difficulty == difficulty and row.tone == tone]
                ranked = sorted(stratum, key=lambda row: hashlib.sha256(
                    f"DGS-AC1-SHUFFLE-v1|model|{row.task_id}|{row.cell_id}".encode()
                ).hexdigest())
                count = sum(row.feedback_validity == "malfunctioning_always_fail" for row in stratum)
                expected_labels.update({
                    (row.task_id, row.cell_id): "malfunctioning_always_fail" if index < count else "accurate"
                    for index, row in enumerate(ranked)
                })
        self.assertEqual(
            {(row.task_id, row.cell_id): row.effective_feedback_validity for row in sentinel_rows},
            expected_labels,
        )

        captures.clear()
        with mock.patch("sklearn.linear_model.LogisticRegression", Logistic), mock.patch("sklearn.metrics.roc_auc_score", return_value=0.5):
            g5_predictive_gap(sentinel_rows, ("M1", "M2"))
        full, baseline = captures[:2]
        fold = "easy-a"
        remaining = [row for row in sentinel_rows if row.task_id != fold]
        classes = {
            label: sorted(
                (row for row in remaining if int(row.effective_feedback_validity == "malfunctioning_always_fail") == label),
                key=lambda row: (row.task_id, row.cell_id),
            )
            for label in (0, 1)
        }
        self.assertNotEqual(len(classes[0]), len(classes[1]))
        retained_count = min(len(classes[0]), len(classes[1]))
        retained = sorted(classes[0][:retained_count] + classes[1][:retained_count], key=lambda row: (row.task_id, row.cell_id))
        retained_keys = [(row.task_id, row.cell_id) for row in retained]

        def standardized(rows, values):
            columns = list(zip(*values))
            averages = [sum(column) / len(column) for column in columns]
            deviations = [math.sqrt(sum((value - average) ** 2 for value in column) / len(column)) for column, average in zip(columns, averages)]
            return [[0.0 if deviation == 0 else (value - average) / deviation for value, average, deviation in zip(row, averages, deviations)] for row in values]

        expected_full = standardized(retained, [[row.metrics["M1"], row.metrics["M2"]] for row in retained])
        expected_baseline = standardized(retained, [[float(row.correctness), float(row.generated_response_tokens)] for row in retained])
        self.assertEqual(full.features, expected_full)
        self.assertEqual(baseline.features, expected_baseline)
        self.assertEqual(full.labels, [int(row.effective_feedback_validity == "malfunctioning_always_fail") for row in retained])
        self.assertEqual(full.labels, baseline.labels)
        self.assertEqual(retained_keys, sorted(retained_keys))

    @unittest.skipUnless(HAS_SKLEARN, "scientific G5 tests require scikit-learn")
    def test_g5_structured_failures_real_auc_and_row_order(self):
        rows = g5_rows()

        class Stalled:
            def __init__(self, **kwargs):
                self.n_iter_ = [1000]

            def fit(self, *args):
                return self

        with mock.patch("sklearn.linear_model.LogisticRegression", Stalled):
            self.assertEqual(g5_predictive_gap(rows, ("M1",)).unavailable_reason, "logistic_nonconvergence")

        class Raising:
            def __init__(self, **kwargs):
                pass

            def fit(self, *args):
                raise RuntimeError("failed")

        with mock.patch("sklearn.linear_model.LogisticRegression", Raising):
            self.assertEqual(g5_predictive_gap(rows, ("M1",)).unavailable_reason, "logistic_fit_failed:RuntimeError")

        one_class = [replace(row, analysis_feedback_validity="accurate" if row.task_id == "easy-0" else "malfunctioning_always_fail") for row in rows]
        self.assertEqual(g5_predictive_gap(one_class, ("M1",)).unavailable_reason, "one_class_training_fold")

        class Nonfinite:
            def __init__(self, **kwargs):
                self.n_iter_ = [2]

            def fit(self, *args):
                return self

            def predict_proba(self, features):
                import numpy as np

                return np.array([[float("nan"), float("nan")] for _ in features])

        with mock.patch("sklearn.linear_model.LogisticRegression", Nonfinite):
            self.assertEqual(g5_predictive_gap(rows, ("M1",)).unavailable_reason, "nonfinite_probabilities")

        actual = g5_predictive_gap(rows, ("M1", "M2"))
        self.assertIsNone(actual.unavailable_reason)
        from sklearn.metrics import roc_auc_score

        by_key = {(row.task_id, row.cell_id): row for row in rows}
        labels, full, baseline = [], [], []
        for key, probabilities in actual.heldout_probabilities.items():
            labels.append(int(by_key[key].effective_feedback_validity == "malfunctioning_always_fail"))
            full.append(probabilities[0])
            baseline.append(probabilities[1])
        self.assertAlmostEqual(actual.full_auc, roc_auc_score(labels, full))
        self.assertAlmostEqual(actual.baseline_auc, roc_auc_score(labels, baseline))
        self.assertEqual(actual, g5_predictive_gap(list(reversed(rows)), ("M1", "M2")))
