"""
Credit environment.

Evaluates borrowing capacity and purchase feasibility.
Contains no agent behavior — purely financial constraint logic.

Two independent credit constraints are checked:
  1. Deposit constraint:  (1 - LTV) * price <= cash
  2. Income constraint:   annual_mortgage_payment <= DTI * annual_income

Both must be satisfied for a purchase to be feasible.

The loan term is used throughout: it determines the annual amortising
payment which drives the income constraint, and it governs how quickly
the outstanding mortgage principal amortises — which determines the
net cash a seller receives when they sell at a gain or loss.
"""


class CreditEnvironment:
    """
    Encapsulates credit conditions for a given period.

    mortgage_rate  : annual rate, e.g. 0.05 for 5%
    ltv_limit      : maximum loan-to-value, e.g. 0.85
    dti_limit      : maximum annual debt-service to income ratio, e.g. 0.35
    loan_term_years: amortisation period; drives payment size and balance rundown
    """

    def __init__(
        self, mortgage_rate=0.05, ltv_limit=0.85, dti_limit=0.35, loan_term_years=25
    ):
        self.mortgage_rate = mortgage_rate
        self.ltv_limit = ltv_limit
        self.dti_limit = dti_limit
        self.loan_term_years = loan_term_years

    def annual_mortgage_payment(self, price, ltv=None):
        """
        Annual mortgage payment using the standard annuity formula.
        Uses ltv_limit if ltv not specified.
        """
        if ltv is None:
            ltv = self.ltv_limit
        principal = ltv * price
        r = self.mortgage_rate
        n = self.loan_term_years
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def outstanding_principal(self, original_price, ltv, years_elapsed):
        """
        Remaining mortgage principal after years_elapsed periods of payments.

        Uses the standard amortisation formula. The outstanding balance
        is used to compute net sale proceeds: seller receives
            sale_price - outstanding_principal
        which may be negative (negative equity) if sale_price < balance.

        original_price : price at which the property was purchased
        ltv            : loan-to-value at origination
        years_elapsed  : integer number of annual periods since purchase
        """
        P = original_price * ltv
        r = self.mortgage_rate
        n = self.loan_term_years
        t = min(years_elapsed, n)
        if r == 0:
            return max(0.0, P - (P / n) * t)
        # Amortisation formula: balance after t payments
        balance = P * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)
        return max(0.0, balance)

    def max_affordable_price(self, cash, annual_income):
        """
        Maximum price satisfying both deposit and income constraints.
        Returns the binding (lower) ceiling.
        """
        # Deposit constraint
        deposit_ceiling = (
            cash / (1.0 - self.ltv_limit) if self.ltv_limit < 1.0 else float("inf")
        )

        # Income constraint: solve payment(price) = dti * income for price
        max_payment = self.dti_limit * annual_income
        r = self.mortgage_rate
        n = self.loan_term_years
        if r == 0:
            max_principal = max_payment * n
        else:
            max_principal = max_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / self.ltv_limit

        return min(deposit_ceiling, income_ceiling)

    def is_feasible(self, price, cash, annual_income):
        """True if the agent can finance a purchase at this price."""
        return price <= self.max_affordable_price(cash, annual_income)

    def is_rental_affordable(self, monthly_rent, annual_income, rent_fraction=0.35):
        """Rent must not exceed rent_fraction of annual income."""
        return monthly_rent * 12 <= rent_fraction * annual_income
