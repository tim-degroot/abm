import unittest
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "code"))

from abm.code.core.utility import risk_adjusted_growth, logit_choice, logit_probabilities


class TestRiskAdjustedGrowth(unittest.TestCase):
    def test_risk_neutral(self):
        result = risk_adjusted_growth(
            expected_growth=0.01, expected_volatility=0.1, risk_loading=0.0
        )
        self.assertAlmostEqual(result, 0.01)

    def test_risk_adjusts_downward(self):
        result = risk_adjusted_growth(
            expected_growth=0.01, expected_volatility=0.1, risk_loading=0.5
        )
        self.assertAlmostEqual(result, -0.04)

    def test_negative_volatility_raises(self):
        with self.assertRaises(ValueError):
            risk_adjusted_growth(expected_growth=0.01, expected_volatility=-0.1, risk_loading=0.5)

    def test_negative_risk_loading_raises(self):
        with self.assertRaises(ValueError):
            risk_adjusted_growth(expected_growth=0.01, expected_volatility=0.1, risk_loading=-0.5)


class TestLogitChoice(unittest.TestCase):
    def test_picks_highest_value(self):
        rng = np.random.default_rng(42)
        values = {"a": 10.0, "b": 100.0, "c": 1.0}
        choices = [logit_choice(values, rng) for _ in range(100)]
        # "b" should dominate
        self.assertGreater(sum(1 for c in choices if c == "b"), 80)

    def test_all_infinite_returns_fallback(self):
        rng = np.random.default_rng(42)
        values = {"hold": float("-inf"), "sell": float("-inf"), "none": float("-inf")}
        result = logit_choice(values, rng)
        self.assertEqual(result, "hold")

    def test_all_infinite_no_fallback(self):
        rng = np.random.default_rng(42)
        values = {"a": float("-inf"), "b": float("-inf")}
        result = logit_choice(values, rng)
        self.assertIn(result, ["a", "b"])

    def test_single_finite_option(self):
        rng = np.random.default_rng(42)
        values = {"a": float("-inf"), "b": 50.0, "c": float("-inf")}
        result = logit_choice(values, rng)
        self.assertEqual(result, "b")


class TestLogitProbabilities(unittest.TestCase):
    def test_sums_to_one(self):
        values = {"a": 10.0, "b": 100.0, "c": 1.0}
        probs = logit_probabilities(values)
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_highest_value_highest_prob(self):
        values = {"a": 10.0, "b": 100.0, "c": 1.0}
        probs = logit_probabilities(values)
        self.assertGreater(probs["b"], probs["a"])
        self.assertGreater(probs["a"], probs["c"])

    def test_all_equal(self):
        values = {"a": 5.0, "b": 5.0, "c": 5.0}
        probs = logit_probabilities(values)
        for p in probs.values():
            self.assertAlmostEqual(p, 1.0 / 3)

    def test_all_infinite(self):
        values = {"a": float("-inf"), "b": float("-inf")}
        probs = logit_probabilities(values)
        for p in probs.values():
            self.assertAlmostEqual(p, 0.5)


if __name__ == "__main__":
    unittest.main()
