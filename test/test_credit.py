import unittest
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "code"))

from code.credit import CreditEnvironment


class TestOriginationLTV(unittest.TestCase):
    def setUp(self):
        self.credit = CreditEnvironment()

    def test_buy_ltv(self):
        self.assertEqual(self.credit.origination_ltv("buy"), self.credit.ltv_limit)

    def test_btl_ltv(self):
        self.assertEqual(self.credit.origination_ltv("buy-to-let"), self.credit.btl_ltv)

    def test_institution_ltv(self):
        self.assertEqual(self.credit.origination_ltv("acquire"), self.credit.inst_ltv)
        self.assertEqual(self.credit.origination_ltv("institution"), self.credit.inst_ltv)

    def test_unknown_purpose_raises(self):
        with self.assertRaises(ValueError):
            self.credit.origination_ltv("unknown")


class TestFundingRate(unittest.TestCase):
    def setUp(self):
        self.credit = CreditEnvironment()

    def test_buy_rate(self):
        self.assertEqual(self.credit.funding_rate("buy"), self.credit.mortgage_rate)

    def test_btl_rate(self):
        self.assertEqual(self.credit.funding_rate("buy-to-let"), self.credit.btl_funding_rate)

    def test_institution_rate(self):
        self.assertEqual(self.credit.funding_rate("acquire"), self.credit.inst_funding_rate)

    def test_unknown_purpose_raises(self):
        with self.assertRaises(ValueError):
            self.credit.funding_rate("unknown")


class TestMortgagePayment(unittest.TestCase):
    def test_standard_payment(self):
        credit = CreditEnvironment(mortgage_rate=0.00308, loan_term_months=300)
        payment = credit.monthly_mortgage_payment(price=200_000, ltv=0.9, rate=0.00308)
        # principal = 180,000, r = 0.00308, n = 300
        # P * r * (1+r)^n / ((1+r)^n - 1)
        self.assertAlmostEqual(payment, 200_000 * 0.9 * 0.00308 * 1.00308**300 / (1.00308**300 - 1))

    def test_zero_rate_payment(self):
        credit = CreditEnvironment(mortgage_rate=0.0, loan_term_months=240)
        payment = credit.monthly_mortgage_payment(price=200_000, ltv=0.9, rate=0.0)
        self.assertAlmostEqual(payment, 180_000 / 240)

    def test_payment_is_positive(self):
        credit = CreditEnvironment(mortgage_rate=0.00308, loan_term_months=300)
        payment = credit.monthly_mortgage_payment(price=100_000, ltv=0.8, rate=0.00308)
        self.assertGreater(payment, 0)


class TestOutstandingPrincipal(unittest.TestCase):
    def test_full_term_zero_balance(self):
        credit = CreditEnvironment(mortgage_rate=0.00308, loan_term_months=300)
        balance = credit.outstanding_principal(
            original_price=200_000, ltv=0.9, months_elapsed=300, rate=0.00308
        )
        self.assertAlmostEqual(balance, 0.0, delta=1.0)

    def test_zero_rate_amortisation(self):
        credit = CreditEnvironment(mortgage_rate=0.0, loan_term_months=240)
        balance = credit.outstanding_principal(
            original_price=240_000, ltv=1.0, months_elapsed=120, rate=0.0
        )
        self.assertAlmostEqual(balance, 120_000)

    def test_initial_balance_equals_principal(self):
        credit = CreditEnvironment(mortgage_rate=0.00308, loan_term_months=300)
        balance = credit.outstanding_principal(
            original_price=200_000, ltv=0.9, months_elapsed=0, rate=0.00308
        )
        self.assertAlmostEqual(balance, 180_000)


class TestHouseholdMaxPrice(unittest.TestCase):
    def test_deposit_constrained(self):
        credit = CreditEnvironment(ltv_limit=0.9, dti_limit=0.4, mortgage_rate=0.00308)
        # cash=20K, ltv=0.9 → deposit_ceiling = 20K / 0.1 = 200K
        # income=100K → max payment = 0.4 * 100K/12 = 3333 → max principal for 300mo at 0.308% = 658K → income_ceiling = 658K/0.9 = 731K
        # min = 200K
        max_price = credit.household_max_price(cash=20_000, annual_income=100_000)
        self.assertAlmostEqual(max_price, 200_000)

    def test_income_constrained(self):
        credit = CreditEnvironment(ltv_limit=0.9, dti_limit=0.4, mortgage_rate=0.00308)
        # cash=100K → deposit_ceiling = 100K / 0.1 = 1M
        # income=30K → max payment = 0.4 * 30K/12 = 1000 → principal ~ 196K → ceiling ~ 218K
        max_price = credit.household_max_price(cash=100_000, annual_income=30_000)
        self.assertLess(max_price, 300_000)
        self.assertGreater(max_price, 100_000)

    def test_cannot_exceed_deposit(self):
        credit = CreditEnvironment(ltv_limit=0.9)
        max_price = credit.household_max_price(cash=10_000, annual_income=1_000_000)
        self.assertAlmostEqual(max_price, 100_000)


class TestInstitutionMaxPrice(unittest.TestCase):
    def test_basic_ceiling(self):
        credit = CreditEnvironment(inst_ltv=0.6)
        max_price = credit.institution_max_price(cash=100_000)
        self.assertAlmostEqual(max_price, 250_000)


class TestMaxPriceDispatch(unittest.TestCase):
    def setUp(self):
        self.credit = CreditEnvironment()

    def test_dispatches_to_household(self):
        result = self.credit.max_price("buy", cash=50_000, annual_income=50_000)
        expected = self.credit.household_max_price(50_000, 50_000)
        self.assertAlmostEqual(result, expected)

    def test_dispatches_to_btl(self):
        result = self.credit.max_price("buy-to-let", cash=50_000, annual_income=50_000)
        expected = self.credit.btl_max_price(50_000, 50_000)
        self.assertAlmostEqual(result, expected)

    def test_dispatches_to_institution(self):
        result = self.credit.max_price("acquire", cash=50_000)
        expected = self.credit.institution_max_price(50_000)
        self.assertAlmostEqual(result, expected)

    def test_unknown_purpose_raises(self):
        with self.assertRaises(ValueError):
            self.credit.max_price("unknown", cash=0)


if __name__ == "__main__":
    unittest.main()
