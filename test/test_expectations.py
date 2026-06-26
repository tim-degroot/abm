import unittest
from pathlib import Path
import sys

import numpy as np

_ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from code.expectations import (
    adaptive_update,
    growth_signal,
    volatility_signal,
    institutional_price_forecast,
    _design_matrix,
    _target_vector,
    _fallback_price_change,
)


class TestAdaptiveUpdate(unittest.TestCase):
    def test_ewma_high_persistence(self):
        result = adaptive_update(current=0.01, signal=0.02, delta=0.9)
        self.assertAlmostEqual(result, 0.9 * 0.01 + 0.1 * 0.02)

    def test_delta_zero_discards_prior(self):
        result = adaptive_update(current=0.5, signal=0.02, delta=0.0)
        self.assertAlmostEqual(result, 0.02)

    def test_delta_one_ignores_signal(self):
        result = adaptive_update(current=0.5, signal=0.02, delta=1.0)
        self.assertAlmostEqual(result, 0.5)


class TestGrowthSignal(unittest.TestCase):
    def test_constant_series(self):
        result = growth_signal([100.0, 100.0, 100.0], window=5)
        self.assertAlmostEqual(result, 0.0)

    def test_positive_growth(self):
        result = growth_signal([100.0, 110.0, 121.0], window=5)
        self.assertAlmostEqual(result, 0.1, delta=1e-6)

    def test_empty_history(self):
        result = growth_signal([], window=5)
        self.assertAlmostEqual(result, 0.0)

    def test_short_history(self):
        result = growth_signal([50.0], window=5)
        self.assertAlmostEqual(result, 0.0)


class TestVolatilitySignal(unittest.TestCase):
    def test_constant_series(self):
        result = volatility_signal([100.0, 100.0, 100.0, 100.0], window=5)
        self.assertAlmostEqual(result, 0.0)

    def test_volatility_increases_with_variance(self):
        low = volatility_signal([100, 101, 100, 101, 100, 101], window=10)
        high = volatility_signal([100, 110, 90, 110, 90, 110], window=10)
        self.assertGreater(high, low)

    def test_too_short_returns_zero(self):
        result = volatility_signal([100.0, 100.0], window=5)
        self.assertAlmostEqual(result, 0.0)


class TestDesignMatrix(unittest.TestCase):
    def test_shape(self):
        history = [
            {
                "price": 200_000,
                "rent": 1000,
                "volume": 5,
                "macro": "Neutral",
                "avg_ltv": 0.85,
                "inst_share": 0.3,
            },
            {
                "price": 201_000,
                "rent": 1010,
                "volume": 6,
                "macro": "Boom",
                "avg_ltv": 0.86,
                "inst_share": 0.31,
            },
            {
                "price": 202_000,
                "rent": 1020,
                "volume": 7,
                "macro": "Recession",
                "avg_ltv": 0.84,
                "inst_share": 0.29,
            },
        ]
        X = _design_matrix(history)
        # 3 time-points → 2 transitions → 2 rows, 8 features
        self.assertEqual(X.shape, (2, 8))

    def test_single_entry(self):
        X = _design_matrix([{"price": 200_000, "rent": 1000, "volume": 5, "macro": "Neutral"}])
        self.assertEqual(len(X), 0)


class TestTargetVector(unittest.TestCase):
    def test_price_change(self):
        history = [
            {"price": 200_000},
            {"price": 201_000},
            {"price": 203_000},
        ]
        y = _target_vector(history)
        np.testing.assert_array_almost_equal(y, [1000, 2000])


class TestFallbackPriceChange(unittest.TestCase):
    def test_single_entry(self):
        result = _fallback_price_change([{"price": 200_000}])
        self.assertAlmostEqual(result, 0.0)

    def test_median_change(self):
        result = _fallback_price_change(
            [
                {"price": 200_000},
                {"price": 201_000},
                {"price": 203_000},
            ]
        )
        self.assertAlmostEqual(result, 1500.0)  # median of [1000, 2000]

    def test_empty_returns_zero(self):
        result = _fallback_price_change([])
        self.assertAlmostEqual(result, 0.0)


class TestInstitutionalPriceForecast(unittest.TestCase):
    def test_fallback_for_short_history(self):
        result = institutional_price_forecast(
            [{"price": 200_000}, {"price": 201_000}],
            window=5,
        )
        self.assertAlmostEqual(result, 1000.0)

    def test_ols_returns_finite(self):
        history = []
        for i in range(10):
            history.append(
                {
                    "price": float(200_000 + i * 1000),
                    "rent": float(1000 + i * 10),
                    "volume": 5 + i,
                    "avg_ltv": 0.85,
                    "inst_share": 0.3,
                    "macro": "Neutral",
                }
            )
        result = institutional_price_forecast(history, window=5)
        self.assertTrue(np.isfinite(result))

    def test_underdetermined_falls_back(self):
        history = []
        for i in range(7):  # 6 transitions but 8 features → underdetermined
            history.append(
                {
                    "price": float(200_000 + i * 1000),
                    "rent": float(1000 + i * 10),
                    "volume": 5 + i,
                    "avg_ltv": 0.85,
                    "inst_share": 0.3,
                    "macro": "Neutral",
                }
            )
        # window = 8 means window+1 = 9, so we have 7 < 9 → fallback
        result = institutional_price_forecast(history, window=8)
        self.assertAlmostEqual(result, _fallback_price_change(history))


class TestInstitutionalRentGrowthSignal(unittest.TestCase):
    def test_delegates_to_growth_signal(self):
        from code.expectations import institutional_rent_growth_signal

        result = institutional_rent_growth_signal(
            [{"rent": 1000}, {"rent": 1010}, {"rent": 1020}],
            window=5,
        )
        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main()
