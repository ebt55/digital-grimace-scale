"""The three README figures build from the committed summaries and skip when absent."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "results" / "summaries"
NAMES = ("F0_channel_map", "F0b_headline_effects", "F0c_phase_map")
PRINT_NAMES = ("F0_channel_map_print", "F0b_headline_effects_print")


def _png_size(path: Path) -> tuple[int, int]:
    """(width, height) in pixels, straight out of the PNG IHDR chunk."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", header[16:24])


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / ("%s.py" % name))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIGURES = _load("make_readme_figures")


class FormattingTest(unittest.TestCase):
    def test_numbers_carry_a_typographic_minus_and_no_signed_zero(self):
        self.assertEqual(FIGURES._n(-2.8999998737637727), "−2.90")
        self.assertEqual(FIGURES._n(1.843750142409408), "+1.84")
        self.assertEqual(FIGURES._n(-0.0013, 2), "0.00")
        self.assertEqual(FIGURES._n(0.1, 2, sign=False), "0.10")
        self.assertEqual(FIGURES._n(None), "n/a")

    def test_rates_render_as_percentage_points(self):
        self.assertEqual(FIGURES._pp(0.6), "+60 pp")
        self.assertEqual(FIGURES._pp(0.2), "+20 pp")


class CommittedSummariesTest(unittest.TestCase):
    """The real summaries must produce all three figures, with sourced numbers."""

    @classmethod
    def setUpClass(cls):
        if not (SUMMARIES / "phase2" / "hypotheses.csv").exists():
            raise unittest.SkipTest("committed summaries are not present")
        cls._directory = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._directory.name) / "figures"
        cls.status = FIGURES.main(["--summaries", str(SUMMARIES), "--out", str(cls.out_dir)])

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_all_three_figures_are_written_as_png_and_svg(self):
        self.assertEqual(self.status, 0)
        written = sorted(path.name for path in self.out_dir.iterdir())
        self.assertEqual(written, sorted("%s.%s" % (name, suffix)
                                         for name in NAMES for suffix in ("png", "svg")))
        for path in self.out_dir.iterdir():
            self.assertGreater(path.stat().st_size, 0, path.name)

    def test_the_channel_map_covers_every_row_and_column(self):
        cells = self._cells()
        self.assertEqual(len(cells), len(FIGURES.CHANNEL_ROWS) * len(FIGURES.CHANNEL_COLUMNS))
        codes = {code for code, _ in cells.values()}
        self.assertEqual(codes, {FIGURES.MOVES, FIGURES.FLAT, FIGURES.NA})

    def test_channel_map_headline_numbers_match_the_summaries(self):
        cells = self._cells()
        # M1 x false failure is holdout H1; M1 x hostile tone is H2a/H2b.
        self.assertIn("−2.90 nats", cells[(0, 0)][1])
        self.assertIn("−16.13", cells[(0, 1)][1])
        self.assertIn("−7.87", cells[(0, 1)][1])
        # Distress moves under hostile tone (H6a) and under adapter A (MC1), nowhere else on that row.
        self.assertEqual(cells[(3, 1)][0], FIGURES.MOVES)
        self.assertIn("+3.2 / 10", cells[(3, 1)][1])
        self.assertIn("65.8%", cells[(3, 6)][1])
        # The base model and the 27B run are not measurable on M1.
        self.assertEqual(cells[(0, 8)][0], FIGURES.NA)
        self.assertEqual(cells[(0, 11)][0], FIGURES.NA)

    def test_forest_values_include_every_series(self):
        values = FIGURES._forest_values(
            FIGURES._read_csv(SUMMARIES / "phase2" / "hypotheses.csv"),
            FIGURES._extension(SUMMARIES),
            FIGURES._read(SUMMARIES / "missingness" / "m1_missingness.json"),
            FIGURES._read(SUMMARIES / "robustness" / "robustness.json"))
        self.assertEqual(values["gemma"]["H1"][0], -2.8999998737637727)
        self.assertIn("tone_pooled", values["gemma"])
        self.assertIn("tone_pooled", values["fresh"])
        self.assertIn("H1", values["qwen"])
        self.assertIn("H6a", values["llama"])

    def _cells(self):
        return FIGURES._channel_cells(
            SUMMARIES,
            FIGURES._read_csv(SUMMARIES / "phase2" / "hypotheses.csv"),
            FIGURES._read(SUMMARIES / "phase1" / "gates.json"),
            FIGURES._read(SUMMARIES / "phase2" / "confirm.json"),
            FIGURES._read(SUMMARIES / "phase3" / "steering.json"),
            FIGURES._read(SUMMARIES / "phase3" / "steering_judge.json"),
            FIGURES._read(SUMMARIES / "phase4" / "phase4.json"),
            FIGURES._read_csv(SUMMARIES / "phase5" / "cell_valid_rates.csv"),
            FIGURES._read(SUMMARIES / "robustness" / "robustness.json"),
            FIGURES._extension(SUMMARIES))


class PrintSizedFiguresTest(unittest.TestCase):
    """``--print`` adds the two report figures; without it nothing extra is written."""

    @classmethod
    def setUpClass(cls):
        if not (SUMMARIES / "phase2" / "hypotheses.csv").exists():
            raise unittest.SkipTest("committed summaries are not present")
        cls._directory = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._directory.name) / "figures"
        cls.status = FIGURES.main(
            ["--summaries", str(SUMMARIES), "--out", str(cls.out_dir), "--print"])

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_print_writes_both_variants_as_png_and_svg(self):
        self.assertEqual(self.status, 0)
        written = sorted(path.name for path in self.out_dir.iterdir())
        self.assertEqual(written, sorted("%s.%s" % (name, suffix)
                                         for name in NAMES + PRINT_NAMES
                                         for suffix in ("png", "svg")))
        for name in PRINT_NAMES:
            for suffix in ("png", "svg"):
                path = self.out_dir / ("%s.%s" % (name, suffix))
                self.assertGreater(path.stat().st_size, 0, path.name)

    def test_print_pngs_are_the_exact_canvas_at_300_dpi(self):
        # No tight bbox: the saved size must be figsize x 300, or the pt arithmetic in the
        # report ("drawn at 7.5 in, placed at 6.7 in") stops being checkable.
        self.assertEqual(_png_size(self.out_dir / "F0b_headline_effects_print.png"), (2250, 1695))
        self.assertEqual(_png_size(self.out_dir / "F0_channel_map_print.png"), (3030, 1860))

    def test_print_type_never_falls_below_seven_points_at_placement(self):
        floors = [FIGURES.PRINT_STYLE[key] for key in ("label", "tick", "xlabel", "annotate", "note")]
        self.assertGreaterEqual(min(floors) * 6.7 / 7.5, 7.0)  # F0b: drawn 7.5 in, placed 6.7 in
        self.assertGreaterEqual(7.0 * 10.1 / 10.1, 7.0)        # F0: drawn and placed at 10.1 in

    def test_the_screen_figures_are_untouched_by_the_print_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "figures"
            FIGURES.main(["--summaries", str(SUMMARIES), "--out", str(plain)])
            for name in NAMES:
                self.assertEqual((plain / ("%s.png" % name)).read_bytes(),
                                 (self.out_dir / ("%s.png" % name)).read_bytes(), name)


class MissingSummariesTest(unittest.TestCase):
    def test_absent_summaries_skip_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "summaries"
            empty.mkdir()
            out_dir = Path(directory) / "figures"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = FIGURES.main(["--summaries", str(empty), "--out", str(out_dir)])
        self.assertEqual(status, 2)
        self.assertIn("skipping F0_channel_map", stderr.getvalue())
        self.assertIn("skipping F0b_headline_effects", stderr.getvalue())
        self.assertIn("skipping F0c_phase_map", stderr.getvalue())
        self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
