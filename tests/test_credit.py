"""
Tests for CreditEnvironment (monthly timestep).

Validates:
  1. Rejection  — people who do not meet criteria are rejected
  2. Acceptance — people who exceed criteria are approved
  3. Companies  — different params (LTV, rate, scale) all respected
  4. Multi-home — outstanding principal amortises; existing mortgage
                  reduces capacity for a second purchase
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from credit import CreditEnvironment


class TestRejection(unittest.TestCase):
    """People who do NOT meet criteria should be rejected."""

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_deposit_constraint_rejects_low_cash(self):
        cash = 10_000.0   # deposit needed = 30_000 at LTV=0.85
        self.assertFalse(
            self.credit.is_feasible(price=200_000.0, cash=cash, monthly_income=10_000.0)
        )

    def test_income_constraint_rejects_low_income(self):
        monthly_income = 1_000.0  # cannot service 170k mortgage
        self.assertFalse(
            self.credit.is_feasible(price=200_000.0, cash=100_000.0, monthly_income=monthly_income)
        )

    def test_rental_affordability_rejects_high_rent(self):
        self.assertFalse(
            self.credit.is_rental_affordable(monthly_rent=2_000.0, monthly_income=3_000.0)
        )


class TestAcceptance(unittest.TestCase):
    """People who EXCEED criteria should be approved."""

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_deposit_constraint_passes_high_cash(self):
        self.assertTrue(
            self.credit.is_feasible(price=200_000.0, cash=200_000.0, monthly_income=3_000.0)
        )

    def test_income_constraint_passes_high_income(self):
        self.assertTrue(
            self.credit.is_feasible(price=200_000.0, cash=30_001.0, monthly_income=20_000.0)
        )

    def test_rental_affordability_passes_low_rent(self):
        self.assertTrue(
            self.credit.is_rental_affordable(monthly_rent=800.0, monthly_income=6_000.0)
        )


class TestCompaniesAndInstitutions(unittest.TestCase):
    """Different credit parameters (LTV, rate, cash scale) are respected."""

    def test_stricter_ltv_needs_more_deposit(self):
        default = CreditEnvironment(ltv_limit=0.85)
        strict = CreditEnvironment(ltv_limit=0.60)
        price, cash, income = 200_000.0, 50_000.0, 10_000.0
        self.assertTrue(default.is_feasible(price, cash, income))    # deposit = 30k
        self.assertFalse(strict.is_feasible(price, cash, income))    # deposit = 80k

    def test_higher_rate_makes_deal_unaffordable(self):
        cheap = CreditEnvironment(mortgage_rate=0.0025)    # 3% p.a.
        expensive = CreditEnvironment(mortgage_rate=0.00833)  # 10% p.a.
        price, cash, income = 200_000.0, 100_000.0, 4_000.0
        self.assertTrue(cheap.is_feasible(price, cash, income))
        self.assertFalse(expensive.is_feasible(price, cash, income))

    def test_institution_scale_cash_works(self):
        self.assertTrue(
            CreditEnvironment().is_feasible(
                price=10_000_000.0, cash=20_000_000.0, monthly_income=1_000_000.0
            )
        )


class TestMultiHomeAndLandlord(unittest.TestCase):
    """Outstanding principal amortisation and cumulative constraint on
    second-property purchase."""

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_outstanding_principal_starts_at_full(self):
        bal = self.credit.outstanding_principal(200_000.0, 0.85, 0)
        self.assertAlmostEqual(bal, 170_000.0)

    def test_outstanding_principal_declines(self):
        bal_60 = self.credit.outstanding_principal(200_000.0, 0.85, 60)
        bal_120 = self.credit.outstanding_principal(200_000.0, 0.85, 120)
        self.assertGreater(bal_60, bal_120)

    def test_outstanding_principal_repaid_at_term(self):
        bal = self.credit.outstanding_principal(200_000.0, 0.85, 300)
        self.assertAlmostEqual(bal, 0.0)

    def test_existing_mortgage_reduces_second_home_capacity(self):
        income = 8_000.0
        cash = 150_000.0
        price_first = 250_000.0

        unconstrained = self.credit.max_affordable_price(cash, income)

        deposit_first = price_first * (1.0 - self.credit.ltv_limit)
        cash_remaining = cash - deposit_first
        annual_pmt_first = self.credit.monthly_mortgage_payment(price_first)
        income_remaining = income - annual_pmt_first / self.credit.dti_limit
        constrained = self.credit.max_affordable_price(cash_remaining, income_remaining)

        self.assertLess(constrained, unconstrained)


if __name__ == "__main__":
    unittest.main()
