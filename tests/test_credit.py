"""
Tests for CreditEnvironment.

Covers rejection (does not meet criteria) and acceptance (exceeds criteria),
including scenarios for companies / institutions and people with more than one
home (multi-property / landlord situations).
"""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from credit import CreditEnvironment


class TestCreditRejection(unittest.TestCase):
    """People who do NOT meet the credit criteria should be rejected."""

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_deposit_constraint_rejects_low_cash(self):
        """Low cash means deposit cannot be covered."""
        price = 200_000.0
        cash = 10_000.0   # deposit needed = 30_000 at LTV=0.85
        income = 100_000.0  # income alone would suffice
        self.assertFalse(self.credit.is_feasible(price, cash, income))

    def test_income_constraint_rejects_low_income(self):
        """Low income means mortgage payments cannot be serviced."""
        price = 200_000.0
        cash = 100_000.0   # deposit easily covered
        income = 10_000.0   # insufficient to service mortgage
        self.assertFalse(self.credit.is_feasible(price, cash, income))

    def test_both_constraints_reject(self):
        """Both cash and income are too low."""
        price = 500_000.0
        cash = 5_000.0
        income = 15_000.0
        self.assertFalse(self.credit.is_feasible(price, cash, income))

    def test_rental_affordability_rejects_high_rent(self):
        """Monthly rent exceeding the income fraction is not affordable."""
        monthly_rent = 2_000.0
        annual_income = 30_000.0
        self.assertFalse(
            self.credit.is_rental_affordable(monthly_rent, annual_income)
        )

    def test_zero_cash_rejects(self):
        """Zero cash means no deposit, so rejected unless LTV=100%."""
        price = 200_000.0
        cash = 0.0
        income = 200_000.0
        self.assertFalse(self.credit.is_feasible(price, cash, income))

    def test_zero_income_rejects(self):
        """Zero income means no mortgage service capacity."""
        price = 200_000.0
        cash = 200_000.0
        income = 0.0
        self.assertFalse(self.credit.is_feasible(price, cash, income))


class TestCreditAcceptance(unittest.TestCase):
    """People who EXCEED the criteria should be approved."""

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_deposit_constraint_passes_high_cash(self):
        """Ample cash easily covers the deposit."""
        price = 200_000.0
        cash = 200_000.0   # way more than 30k deposit
        income = 35_000.0
        self.assertTrue(self.credit.is_feasible(price, cash, income))

    def test_income_constraint_passes_high_income(self):
        """High income easily services the mortgage."""
        price = 200_000.0
        cash = 30_001.0   # slightly above minimum deposit (FP-safe)
        income = 200_000.0  # easily services mortgage
        self.assertTrue(self.credit.is_feasible(price, cash, income))

    def test_both_constraints_pass(self):
        """High cash AND high income = easily affordable."""
        price = 1_000_000.0
        cash = 500_000.0
        income = 500_000.0
        self.assertTrue(self.credit.is_feasible(price, cash, income))

    def test_rental_affordability_passes_low_rent(self):
        """Low rent relative to income is affordable."""
        monthly_rent = 800.0
        annual_income = 60_000.0
        self.assertTrue(
            self.credit.is_rental_affordable(monthly_rent, annual_income)
        )

    def test_boundary_deposit_exact(self):
        """Price exactly at the deposit ceiling is feasible."""
        cash = 30_000.0
        income = 200_000.0
        max_from_deposit = cash / (1.0 - self.credit.ltv_limit)  # 30k/0.15 = 200k
        self.assertTrue(
            self.credit.is_feasible(max_from_deposit, cash, income)
        )

    def test_boundary_deposit_one_pound_over(self):
        """Price one pound over deposit ceiling is NOT feasible."""
        cash = 30_000.0
        income = 200_000.0
        max_from_deposit = cash / (1.0 - self.credit.ltv_limit)
        self.assertFalse(
            self.credit.is_feasible(max_from_deposit + 1.0, cash, income)
        )


class TestCompaniesAndInstitutions(unittest.TestCase):
    """
    Companies / institutions face different credit parameters:
    - They use their own LTV (inst_ltv = 0.60, btl_ltv = 0.75)
    - They have very large cash reserves
    - They use different funding rates

    We validate that CreditEnvironment correctly handles these
    different parameter configurations.
    """

    def setUp(self):
        self.default_credit = CreditEnvironment()

    def test_stricter_ltv_reduces_affordability(self):
        """Lower LTV (institutional-style) means bigger deposit needed."""
        inst_credit = CreditEnvironment(ltv_limit=0.60)
        price = 200_000.0
        cash = 50_000.0   # 25% deposit — enough for 0.85 LTV but not 0.60
        income = 100_000.0
        # Deposit needed at LTV=0.60: 80_000; at LTV=0.85: 30_000
        self.assertTrue(self.default_credit.is_feasible(price, cash, income))
        self.assertFalse(inst_credit.is_feasible(price, cash, income))

    def test_higher_funding_rate_reduces_affordability(self):
        """Higher mortgage rate → larger payments → less affordable."""
        cheap_credit = CreditEnvironment(mortgage_rate=0.03)
        expensive_credit = CreditEnvironment(mortgage_rate=0.10)
        price = 200_000.0
        cash = 100_000.0
        income = 40_000.0
        self.assertTrue(cheap_credit.is_feasible(price, cash, income))
        self.assertFalse(expensive_credit.is_feasible(price, cash, income))

    def test_large_institution_can_buy_anything(self):
        """Institution-scale cash overwhelms any deposit constraint."""
        inst_cash = 20_000_000.0
        inst_income = 10_000_000.0
        price = 10_000_000.0
        self.assertTrue(
            self.default_credit.is_feasible(price, inst_cash, inst_income)
        )

    def test_max_affordable_price_with_inst_ltv(self):
        """max_affordable_price respects a lower LTV cap."""
        inst_credit = CreditEnvironment(ltv_limit=0.60)
        cash = 100_000.0
        income = 200_000.0
        max_price = inst_credit.max_affordable_price(cash, income)
        deposit_ceiling = cash / (1.0 - 0.60)  # 250k
        self.assertAlmostEqual(max_price, min(deposit_ceiling, max_price))

    def test_btl_rate_vs_household_rate(self):
        """BTL funding rate (higher) reduces max affordable price."""
        household_credit = CreditEnvironment(mortgage_rate=0.05)
        btl_credit = CreditEnvironment(mortgage_rate=0.06)
        cash = 100_000.0
        income = 80_000.0
        self.assertGreater(
            household_credit.max_affordable_price(cash, income),
            btl_credit.max_affordable_price(cash, income),
        )


class TestMultiHomeAndLandlord(unittest.TestCase):
    """
    People with more than one home (landlords) face additional
    financial constraints: outstanding principal on existing
    mortgages, extra deposit requirements, and cumulative
    payment obligations that reduce affordability.
    """

    def setUp(self):
        self.credit = CreditEnvironment()

    def test_outstanding_principal_starts_at_full_amount(self):
        """At origination (t=0), full principal is outstanding."""
        principal = self.credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=0
        )
        self.assertAlmostEqual(principal, 200_000.0 * 0.85)

    def test_outstanding_principal_declines_over_time(self):
        """Principal decreases each year as payments are made."""
        t1 = self.credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=5
        )
        t2 = self.credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=10
        )
        self.assertGreater(t1, t2)

    def test_outstanding_principal_zero_after_full_term(self):
        """After full loan term, principal is fully repaid."""
        remaining = self.credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=25
        )
        self.assertAlmostEqual(remaining, 0.0)

    def test_outstanding_principal_clamped_at_zero(self):
        """Years beyond term still return zero."""
        remaining = self.credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=30
        )
        self.assertAlmostEqual(remaining, 0.0)

    def test_outstanding_principal_zero_rate(self):
        """Test outstanding principal at 0% mortgage rate."""
        credit = CreditEnvironment(mortgage_rate=0.0)
        principal = 200_000.0 * 0.85
        remaining = credit.outstanding_principal(
            original_price=200_000.0, ltv=0.85, years_elapsed=10
        )
        expected = principal - (principal / 25.0) * 10
        self.assertAlmostEqual(remaining, expected)

    def test_existing_mortgage_reduces_affordable_price(self):
        """
        An agent who already services a mortgage on one property
        has reduced capacity for a second property because the
        mortgage payment consumes part of their DTI allowance.
        """
        price_first = 200_000.0
        annual_payment = self.credit.annual_mortgage_payment(price_first)
        income = 60_000.0

        remaining_capacity = self.credit.dti_limit * income - annual_payment
        max_new_principal = 0.0
        r = self.credit.mortgage_rate
        n = self.credit.loan_term_years
        if r > 0:
            max_new_principal = remaining_capacity * (
                (1 + r) ** n - 1
            ) / (r * (1 + r) ** n)
        max_new_price = max_new_principal / self.credit.ltv_limit

        # Without the first mortgage, max price should be higher
        max_price_no_first = self.credit.max_affordable_price(
            cash=100_000.0, annual_income=income
        )

        # With the first mortgage and reduced cash, max should be lower
        max_price_with_first = self.credit.max_affordable_price(
            cash=100_000.0 - price_first * (1.0 - self.credit.ltv_limit),
            annual_income=income,
        )

        # The income-burdened max should be <= the available max from credit
        from_income = min(max_new_price, max_price_with_first)
        self.assertLess(from_income, max_price_no_first)

    def test_annual_mortgage_payment_scales_with_ltv(self):
        """Higher LTV → larger loan → larger annual payment."""
        payment_low_ltv = self.credit.annual_mortgage_payment(
            price=200_000.0, ltv=0.70
        )
        payment_high_ltv = self.credit.annual_mortgage_payment(
            price=200_000.0, ltv=0.85
        )
        self.assertGreater(payment_high_ltv, payment_low_ltv)

    def test_landlord_payment_burden_reduces_buying_power(self):
        """
        A landlord with an existing rental property mortgage
        has less income capacity left to qualify for another
        purchase.
        """
        income = 80_000.0
        cash = 150_000.0
        price_first = 250_000.0

        annual_payment_first = self.credit.annual_mortgage_payment(price_first)
        dti_burden = annual_payment_first / income

        # Max price for a second property given remaining DTI
        remaining_dti_fraction = self.credit.dti_limit - dti_burden
        r = self.credit.mortgage_rate
        n = self.credit.loan_term_years
        if remaining_dti_fraction > 0:
            max_payment_second = remaining_dti_fraction * income
            if r > 0:
                max_principal_second = max_payment_second * (
                    (1 + r) ** n - 1
                ) / (r * (1 + r) ** n)
            else:
                max_principal_second = max_payment_second * n
            income_ceiling = max_principal_second / self.credit.ltv_limit
        else:
            income_ceiling = 0.0

        # Deposit ceiling after first purchase
        deposit_left = cash - price_first * (1.0 - self.credit.ltv_limit)
        deposit_ceiling = (
            deposit_left / (1.0 - self.credit.ltv_limit)
            if deposit_left > 0
            else 0.0
        )

        constrained_max = min(deposit_ceiling, income_ceiling)

        # Without the first property, max should be higher
        unconstrained_max = self.credit.max_affordable_price(cash, income)
        self.assertLess(constrained_max, unconstrained_max)


class TestEdgeCases(unittest.TestCase):
    """Boundary and edge cases for the credit environment."""

    def test_zero_mortgage_rate(self):
        """r=0: payments are simple principal / n."""
        credit = CreditEnvironment(mortgage_rate=0.0)
        payment = credit.annual_mortgage_payment(price=200_000.0)
        expected = (200_000.0 * 0.85) / 25.0
        self.assertAlmostEqual(payment, expected)

    def test_near_zero_ltv_limit(self):
        """Very low LTV means mostly cash purchase."""
        credit = CreditEnvironment(ltv_limit=0.01)
        max_price = credit.max_affordable_price(cash=50_000.0, annual_income=1e9)
        expected = 50_000.0 / (1.0 - 0.01)
        self.assertAlmostEqual(max_price, expected)

    def test_full_ltv_limit(self):
        """LTV=1 means no deposit needed — only income constraint."""
        credit = CreditEnvironment(ltv_limit=1.0)
        max_price = credit.max_affordable_price(cash=0.0, annual_income=50_000.0)
        self.assertGreater(max_price, 0.0)

    def test_max_affordable_deposit_binds(self):
        """When cash is low, deposit constraint is the binding ceiling."""
        credit = CreditEnvironment()
        max_price = credit.max_affordable_price(cash=15_000.0, annual_income=1e9)
        expected = 15_000.0 / (1.0 - 0.85)
        self.assertAlmostEqual(max_price, expected)

    def test_max_affordable_income_binds(self):
        """When income is low, income constraint is the binding ceiling."""
        credit = CreditEnvironment()
        max_price = credit.max_affordable_price(cash=1e9, annual_income=20_000.0)
        r, n = credit.mortgage_rate, credit.loan_term_years
        max_payment = credit.dti_limit * 20_000.0
        max_principal = max_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        expected = max_principal / credit.ltv_limit
        self.assertAlmostEqual(max_price, expected)

    def test_annual_mortgage_payment_formula(self):
        """Verify the annuity formula matches manual computation."""
        credit = CreditEnvironment(mortgage_rate=0.05, loan_term_years=25)
        price = 200_000.0
        ltv = 0.85
        principal = price * ltv  # 170_000
        r, n = 0.05, 25
        expected = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        self.assertAlmostEqual(
            credit.annual_mortgage_payment(price), expected
        )

    def test_rental_affordability_boundary(self):
        """Rent exactly at the fraction boundary is affordable."""
        credit = CreditEnvironment()
        monthly_rent = 1_000.0
        annual_income = monthly_rent * 12 / 0.35  # ~34_285.71
        self.assertTrue(
            credit.is_rental_affordable(monthly_rent, annual_income)
        )
        self.assertFalse(
            credit.is_rental_affordable(
                monthly_rent + 0.01, annual_income
            )
        )

    def test_default_parameters(self):
        """Defaults match the config defaults."""
        credit = CreditEnvironment()
        self.assertEqual(credit.mortgage_rate, 0.05)
        self.assertEqual(credit.ltv_limit, 0.85)
        self.assertEqual(credit.dti_limit, 0.35)
        self.assertEqual(credit.loan_term_years, 25)


class TestOutstandingPrincipal(unittest.TestCase):
    """Detailed tests for the amortisation formula."""

    def setUp(self):
        self.credit = CreditEnvironment(mortgage_rate=0.05, loan_term_years=25)

    def test_full_amortisation_schedule(self):
        """Principal declines monotonically over the full term."""
        p0 = self.credit.outstanding_principal(200_000.0, 0.85, 0)
        balances = [
            self.credit.outstanding_principal(200_000.0, 0.85, t)
            for t in range(26)
        ]
        for i in range(len(balances) - 1):
            self.assertGreaterEqual(balances[i], balances[i + 1])
        self.assertAlmostEqual(balances[0], p0)
        self.assertAlmostEqual(balances[25], 0.0)

    def test_zero_ltv_outstanding(self):
        """LTV=0 means no loan, so outstanding is always 0."""
        for t in range(0, 30, 5):
            bal = self.credit.outstanding_principal(200_000.0, 0.0, t)
            self.assertAlmostEqual(bal, 0.0)

    def test_higher_rate_slows_repayment(self):
        """Higher rate → more goes to interest early → higher balance mid-term."""
        slow = CreditEnvironment(mortgage_rate=0.03)
        fast = CreditEnvironment(mortgage_rate=0.07)
        for t in [5, 10, 15, 20]:
            bal_slow = slow.outstanding_principal(200_000.0, 0.85, t)
            bal_fast = fast.outstanding_principal(200_000.0, 0.85, t)
            self.assertGreater(bal_fast, bal_slow,
                f"Expected higher rate to have higher balance at t={t}")


if __name__ == "__main__":
    unittest.main()
