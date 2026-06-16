class CreditEnvironment:
    """
    mortgage_rate    : monthly rate, e.g. 0.004167 for 5% p.a.
    ltv_limit        : maximum loan-to-value, e.g. 0.85
    dti_limit        : maximum monthly debt-service to income ratio, e.g. 0.35
    loan_term_months : amortisation period in months; drives payment size
                       and balance rundown
    """

    def __init__(  # all hardcoded, these should come from config
        self,
        mortgage_rate=0.004167,
        ltv_limit=0.85,
        dti_limit=0.35,
        loan_term_months=300,
    ):
        self.mortgage_rate = mortgage_rate
        self.ltv_limit = ltv_limit
        self.dti_limit = dti_limit
        self.loan_term_months = loan_term_months

    def monthly_mortgage_payment(self, price, ltv=None):
        """
        Monthly mortgage payment using the standard annuity formula.
        Uses ltv_limit if ltv not specified.
        """
        if ltv is None:
            ltv = self.ltv_limit
        principal = ltv * price
        r = self.mortgage_rate
        n = self.loan_term_months
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def outstanding_principal(self, original_price, ltv, months_elapsed):
        """
        Remaining mortgage principal after months_elapsed periods of payments.

        The outstanding balance is used to compute net sale proceeds: seller receives
            sale_price - outstanding_principal
        which may be negative (negative equity) if sale_price < balance.

        original_price : price at which the property was purchased
        ltv            : loan-to-value at origination
        months_elapsed : integer number of monthly periods since purchase
        """
        P = original_price * ltv
        r = self.mortgage_rate
        n = self.loan_term_months
        t = min(months_elapsed, n)
        if r == 0:
            return max(0.0, P - (P / n) * t)
        balance = P * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)
        return max(0.0, balance)

    def max_affordable_price(self, cash, monthly_income):
        """
        Maximum price satisfying both deposit and income constraints.
        """
        deposit_ceiling = cash / (1.0 - self.ltv_limit) if self.ltv_limit < 1.0 else float("inf")

        max_payment = self.dti_limit * monthly_income
        r = self.mortgage_rate
        n = self.loan_term_months
        if r == 0:
            max_principal = max_payment * n
        else:
            max_principal = max_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / self.ltv_limit

        return min(deposit_ceiling, income_ceiling)

    def is_feasible(
        self, price, cash, monthly_income
    ):  # how does this come into play, technically there are no listings.
        """True if the agent can finance a purchase at this price."""
        return price <= self.max_affordable_price(cash, monthly_income)

    def is_rental_affordable(self, monthly_rent, monthly_income, rent_fraction=0.35):  # hmm...
        """Rent must not exceed rent_fraction of monthly income."""
        return monthly_rent <= rent_fraction * monthly_income
