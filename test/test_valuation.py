import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "code"))

from code.valuation import (
    housing_consumption_value,
    estimate_market_rent,
    _dcf_price,
    household_buy_wtp,
    household_btl_wtp,
    institutional_wtp,
    household_rent_wtp,
)


class TestHousingConsumptionValue(unittest.TestCase):
    def test_median_quality(self):
        val = housing_consumption_value(
            quality=0.0, quality_value_scale=200, base_housing_value=800
        )
        self.assertAlmostEqual(val, 800)

    def test_positive_quality(self):
        val = housing_consumption_value(
            quality=1.0, quality_value_scale=200, base_housing_value=800
        )
        self.assertAlmostEqual(val, 1000)

    def test_negative_quality(self):
        val = housing_consumption_value(
            quality=-1.0, quality_value_scale=200, base_housing_value=800
        )
        self.assertAlmostEqual(val, 600)

    def test_clips_below_zero(self):
        val = housing_consumption_value(
            quality=-10.0, quality_value_scale=200, base_housing_value=800
        )
        self.assertAlmostEqual(val, 0.0)


class TestEstimateMarketRent(unittest.TestCase):
    def test_median_quality(self):
        rent = estimate_market_rent(quality=0.0, avg_market_rent=1000, quality_sensitivity=0.3)
        self.assertAlmostEqual(rent, 1000)

    def test_positive_quality(self):
        rent = estimate_market_rent(quality=1.0, avg_market_rent=1000, quality_sensitivity=0.3)
        self.assertAlmostEqual(rent, 1300)

    def test_zero_avg_rent_returns_zero(self):
        rent = estimate_market_rent(quality=1.0, avg_market_rent=0.0, quality_sensitivity=0.3)
        self.assertAlmostEqual(rent, 0.0)


class TestDCFPrice(unittest.TestCase):
    def test_matches_brute_force(self):
        benefit = 1000.0
        discount = 0.003
        benefit_growth = 0.001
        price_growth = 0.002
        anchor = 200_000.0
        horizon = 360

        closed_form = _dcf_price(benefit, discount, benefit_growth, price_growth, anchor, horizon)

        def brute_force():
            pv = 0.0
            for t in range(horizon):
                b = benefit * (1 + benefit_growth) ** t
                pv += b / (1 + discount) ** t
            terminal = anchor * (1 + price_growth) ** horizon / (1 + discount) ** horizon
            return pv + terminal

        expected = brute_force()
        self.assertAlmostEqual(closed_form, expected, delta=abs(expected) * 1e-10)

    def test_zero_discount_flat(self):
        price = _dcf_price(
            benefit_flow=1000.0,
            discount_rate=0.0,
            benefit_growth=0.0,
            price_growth=0.0,
            price_anchor=200_000.0,
            horizon=240,
        )
        self.assertAlmostEqual(price, 1000.0 * 240 + 200_000.0)


class TestHouseholdBuyWTP(unittest.TestCase):
    def test_credit_ceiling_clips(self):
        wtp = household_buy_wtp(
            quality=1.0,
            quality_value_scale=200,
            base_housing_value=800,
            mortgage_rate=0.00308,
            risk_adjusted_price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=50_000.0,
            horizon=360,
        )
        self.assertAlmostEqual(wtp, 50_000.0)

    def test_returns_positive(self):
        wtp = household_buy_wtp(
            quality=0.5,
            quality_value_scale=200,
            base_housing_value=800,
            mortgage_rate=0.00308,
            risk_adjusted_price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=500_000.0,
            horizon=360,
        )
        self.assertGreater(wtp, 0)


class TestHouseholdBtlWTP(unittest.TestCase):
    def test_credit_ceiling_clips(self):
        wtp = household_btl_wtp(
            quality=1.0,
            quality_sensitivity=0.3,
            base_rent=1000,
            funding_rate=0.008,
            risk_adjusted_rent_growth=0.001,
            risk_adjusted_price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=30_000.0,
            horizon=360,
        )
        self.assertAlmostEqual(wtp, 30_000.0)

    def test_returns_positive(self):
        wtp = household_btl_wtp(
            quality=0.5,
            quality_sensitivity=0.3,
            base_rent=1000,
            funding_rate=0.008,
            risk_adjusted_rent_growth=0.001,
            risk_adjusted_price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=500_000.0,
            horizon=360,
        )
        self.assertGreater(wtp, 0)


class TestInstitutionalWTP(unittest.TestCase):
    def test_credit_ceiling_clips(self):
        wtp = institutional_wtp(
            quality=1.0,
            quality_sensitivity=0.3,
            base_rent=1000,
            funding_rate=0.0045,
            required_return=0.0015,
            rent_growth=0.001,
            price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=20_000.0,
            horizon=360,
        )
        self.assertAlmostEqual(wtp, 20_000.0)

    def test_returns_positive(self):
        wtp = institutional_wtp(
            quality=0.5,
            quality_sensitivity=0.3,
            base_rent=1000,
            funding_rate=0.0045,
            required_return=0.0015,
            rent_growth=0.001,
            price_growth=0.001,
            price_anchor=200_000.0,
            credit_ceiling=500_000.0,
            horizon=360,
        )
        self.assertGreater(wtp, 0)


class TestHouseholdRentWTP(unittest.TestCase):
    def test_benefit_constrained(self):
        # benefit = 800 + 200*0.5 = 900; ceiling = 50K/12 * 0.4 = 1667; min = 900
        wtp = household_rent_wtp(
            quality=0.5,
            quality_value_scale=200,
            base_housing_value=800,
            income=50_000,
            dti_limit=0.4,
        )
        self.assertAlmostEqual(wtp, 900)

    def test_income_constrained(self):
        # benefit = 800 + 200*1.0 = 1000; ceiling = 12K/12 * 0.4 = 400; min = 400
        wtp = household_rent_wtp(
            quality=1.0,
            quality_value_scale=200,
            base_housing_value=800,
            income=12_000,
            dti_limit=0.4,
        )
        self.assertAlmostEqual(wtp, 400)

    def test_zero_income(self):
        wtp = household_rent_wtp(
            quality=0.5,
            quality_value_scale=200,
            base_housing_value=800,
            income=0,
            dti_limit=0.4,
        )
        self.assertAlmostEqual(wtp, 0.0)


if __name__ == "__main__":
    unittest.main()
