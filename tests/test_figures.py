"""The FH holdout figures are built from committed summaries and skip when absent."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / ("%s.py" % name))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIGURES = _load("make_figures")

HEADER = ("hypothesis_id,contrast,outcome,stratum,prediction,discovery,estimate,ci95_lower,"
          "ci95_upper,n_items,n_pairs,supported,p_two_sided,bh_adjusted_p,unavailable_reason")
ROWS = (
    'H1,"M1, malfunctioning - accurate (measured)",m1,easy | neutral,< 0,'
    '"-3.800 [-5.297, -2.350]",-2.9,-3.97,-1.84,10,10,true,0.0,0.0,',
    'H6a,"Distress, hostile onset - neutral onset (accurate)",distress,easy+hard pooled,> 0,'
    '"+2.6 (easy), +4.7 (hard) cell means",3.2,2.1,4.3,20,20,true,0.0,0.0,',
    'H8,"M2, hostile - neutral (measured)",m2,easy | accurate,> 0,'
    '"0.257 [0.100, 0.386]",0.283,0.167,0.4,6,6,true,0.0,0.0,',
    'H9,"Non-answer rate, hostile onset - neutral onset (accurate)",non_answer,hard,> 0,'
    '"+0.20 [0.00, +0.50]",0.6,0.3,0.9,10,10,false,0.0,0.0,',
)
CONFIRM = {
    "result": {
        "models": {"primary": "google/gemma-2-9b-it", "control": "Qwen/Qwen2.5-3B-Instruct"},
        "style": [
            {"style_id": "style__enthusiastic", "violates": False,
             "result": {"estimate": -2.18, "ci95_lower": -6.28, "ci95_upper": 1.55,
                        "n_items": 20, "n_pairs": 20, "unavailable_reason": None}},
            {"style_id": "style__verbose", "violates": True,
             "result": {"estimate": -4.0, "ci95_lower": -6.0, "ci95_upper": -1.0,
                        "n_items": 18, "n_pairs": 18, "unavailable_reason": None}},
        ],
        "hypotheses": [
            {"hypothesis_id": "H1", "contrast": "M1, malfunctioning - accurate (measured)",
             "outcome": "m1", "stratum": "easy | neutral", "prediction": "< 0",
             "discovery": "-3.800 [-5.297, -2.350]", "supported": True,
             "result": {"estimate": -2.9, "ci95_lower": -3.97, "ci95_upper": -1.84,
                        "n_items": 10, "n_pairs": 10, "unavailable_reason": None}},
        ],
    },
}


@contextlib.contextmanager
def _summaries(*, hypotheses=True, confirm=True):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        phase2 = root / "summaries" / "phase2"
        phase2.mkdir(parents=True)
        if hypotheses:
            (phase2 / "hypotheses.csv").write_text("\n".join((HEADER,) + ROWS) + "\n", encoding="utf-8")
        if confirm:
            (phase2 / "confirm.json").write_text(json.dumps(CONFIRM), encoding="utf-8")
        yield root / "summaries", root / "figures"


class ParsingTest(unittest.TestCase):
    def test_estimate_with_ci_is_parsed(self):
        self.assertEqual(FIGURES._parse_estimate("-3.800 [-5.297, -2.350]"), (-3.8, -5.297, -2.35))
        self.assertEqual(FIGURES._parse_estimate("+0.20 [0.00, +0.50]"), (0.2, 0.0, 0.5))

    def test_prose_discovery_is_not_an_estimate(self):
        self.assertIsNone(FIGURES._parse_estimate("+2.6 (easy), +4.7 (hard) cell means"))
        self.assertIsNone(FIGURES._parse_estimate("3.8 vs 0.85 cell means"))
        self.assertIsNone(FIGURES._parse_estimate(None))

    def test_predicted_direction(self):
        self.assertEqual(FIGURES._predicted_direction("< 0"), -1)
        self.assertEqual(FIGURES._predicted_direction("> 0"), 1)
        self.assertEqual(FIGURES._predicted_direction("CI upper <= +1.0 nat and point <= 0"), -1)
        # The H7 boundary rule predicts no effect, so it gets no directional shading.
        self.assertEqual(FIGURES._predicted_direction("CI includes 0 or is positive"), 0)
        self.assertEqual(FIGURES._predicted_direction(""), 0)

    def test_rows_are_normalised_from_the_csv(self):
        with _summaries() as (summaries, _):
            rows = FIGURES._hypothesis_rows(summaries)
        self.assertEqual([row["id"] for row in rows], ["H1", "H6a", "H8", "H9"])
        self.assertEqual([row["supported"] for row in rows], [True, True, True, False])
        self.assertEqual(rows[0]["estimate"], -2.9)
        self.assertEqual(rows[0]["n_items"], 10.0)
        self.assertEqual(FIGURES._h1_holdout_estimate(rows), -2.9)

    def test_rows_fall_back_to_confirm_json(self):
        with _summaries(hypotheses=False) as (summaries, _):
            rows = FIGURES._hypothesis_rows(summaries)
        self.assertEqual([row["id"] for row in rows], ["H1"])
        self.assertTrue(rows[0]["supported"])
        self.assertEqual(rows[0]["estimate"], -2.9)

    def test_style_rows_carry_the_violation_flag(self):
        rows = FIGURES._style_rows(CONFIRM)
        self.assertEqual([row["id"] for row in rows], ["style__enthusiastic", "style__verbose"])
        self.assertEqual([row["violates"] for row in rows], [False, True])
        self.assertEqual(FIGURES._style_rows(None), [])
        self.assertEqual(FIGURES._style_rows({}), [])


class RenderTest(unittest.TestCase):
    def test_both_holdout_figures_are_written(self):
        with _summaries() as (summaries, out_dir):
            rows = FIGURES._hypothesis_rows(summaries)
            written = FIGURES.figure_holdout_forest(rows, out_dir)
            written += FIGURES.figure_holdout_style_battery(
                FIGURES._style_rows(json.loads((summaries / "phase2" / "confirm.json").read_text())),
                FIGURES._h1_holdout_estimate(rows), "google/gemma-2-9b-it", out_dir)
            names = sorted(path.name for path in written)
            self.assertEqual(names, [
                "FH_holdout_forest.png", "FH_holdout_forest.svg",
                "FH_holdout_style_battery.png", "FH_holdout_style_battery.svg"])
            for path in written:
                self.assertTrue(path.exists() and path.stat().st_size > 0)

    def test_empty_inputs_return_nothing(self):
        with _summaries() as (_, out_dir):
            self.assertEqual(FIGURES.figure_holdout_forest([], out_dir), [])
            self.assertEqual(FIGURES.figure_holdout_style_battery([], -2.9, "m", out_dir), [])

    def test_missing_phase2_skips_instead_of_crashing(self):
        with _summaries(hypotheses=False, confirm=False) as (summaries, out_dir):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = FIGURES.main(["--summaries", str(summaries), "--out", str(out_dir)])
        self.assertEqual(status, 2)
        self.assertIn("skipping FH_holdout_forest", stderr.getvalue())
        self.assertIn("skipping FH_holdout_style_battery", stderr.getvalue())
        self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
