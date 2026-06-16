from config import CreditConfig


class CreditEnvironment:
    def __init__(self):
        self.mortgage_rate = CreditConfig().mortgage_rate
        self.ltv_limit = CreditConfig().ltv_limit
        self.dti_limit = CreditConfig().dti_limit
        self.loan_term_months = CreditConfig().loan_term_months
        self.inst_funding_rate = CreditConfig().inst_funding_rate
        self.inst_ltv = CreditConfig().inst_ltv

    def _update(self):
        self.__init__()  # re-read config values in case they have changed

    def monthly_mortgage_payment(self, price):
        ltv = self.ltv_limit
        principal = ltv * price
        r = self.mortgage_rate
        n = self.loan_term_months
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def outstanding_principal(self, original_price, ltv, months_elapsed):
        P = original_price * ltv
        r = self.mortgage_rate
        n = self.loan_term_months
        t = min(months_elapsed, n)
        if r == 0:
            return max(0.0, P - (P / n) * t)
        balance = P * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)
        return max(0.0, balance)

    def max_affordable_price(self, cash, monthly_income):
        deposit_ceiling = cash / (1.0 - self.ltv_limit)
        max_payment = self.dti_limit * monthly_income
        r = self.mortgage_rate
        n = self.loan_term_months
        if r == 0:
            max_principal = max_payment * n
        else:
            max_principal = max_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / self.ltv_limit

        return min(deposit_ceiling, income_ceiling)
