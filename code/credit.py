from config import CreditConfig


class CreditEnvironment:
    def __init__(self, **kwargs):
        cfg = CreditConfig()
        self.mortgage_rate = kwargs.get("mortgage_rate", cfg.mortgage_rate)
        self.ltv_limit = kwargs.get("ltv_limit", cfg.ltv_limit)
        self.dti_limit = kwargs.get("dti_limit", cfg.dti_limit)
        self.loan_term_months = kwargs.get("loan_term_months", cfg.loan_term_months)
        self.inst_funding_rate = kwargs.get("inst_funding_rate", cfg.inst_funding_rate)
        self.inst_ltv = kwargs.get("inst_ltv", cfg.inst_ltv)

    def _update(self):
        self.__init__()  # re-read config values in case they have changed

    def monthly_mortgage_payment(
        self, price, r, n
    ):  # r moved to param since either mortgage or inst rate depending on agent
        ltv = self.ltv_limit
        principal = ltv * price
        n = self.loan_term_months
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def outstanding_principal(
        self, original_price, ltv, months_elapsed, r
    ):  # r moved to param since either mortgage or inst rate depending on agent
        P = original_price * ltv
        n = self.loan_term_months
        t = min(months_elapsed, n)
        if r == 0:
            return max(0.0, P - (P / n) * t)
        balance = P * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)
        return max(0.0, balance)

    def max_affordable_price(
        self, cash, monthly_income, dti_limit, ltv_limit, mortgage_rate, loan_term_months
    ):
        deposit_ceiling = cash / (1.0 - ltv_limit)
        max_payment = dti_limit * monthly_income
        r = mortgage_rate
        n = loan_term_months
        if r == 0:
            max_principal = max_payment * n
        else:
            max_principal = max_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / ltv_limit

        return min(deposit_ceiling, income_ceiling)
