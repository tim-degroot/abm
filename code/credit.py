"""
The credit environment.
"""

from __future__ import annotations

from config import CreditConfig


class CreditEnvironment:
    """Mutable snapshot of credit conditions; policies may replace it mid-run."""

    def __init__(self, **kwargs):
        cfg = CreditConfig()
        self.mortgage_rate = kwargs.get("mortgage_rate", cfg.mortgage_rate)
        self.ltv_limit = kwargs.get("ltv_limit", cfg.ltv_limit)
        self.dti_limit = kwargs.get("dti_limit", cfg.dti_limit)
        self.loan_term_months = kwargs.get("loan_term_months", cfg.loan_term_months)
        self.btl_funding_rate = kwargs.get("btl_funding_rate", cfg.btl_funding_rate)
        self.btl_ltv = kwargs.get("btl_ltv", cfg.btl_ltv)
        self.inst_funding_rate = kwargs.get("inst_funding_rate", cfg.inst_funding_rate)
        self.inst_ltv = kwargs.get("inst_ltv", cfg.inst_ltv)

    # -- per-class origination LTV (the policy/config lever, not random) --------

    def origination_ltv(self, purpose: str) -> float:
        """LTV at which a new loan is originated, by purchase purpose."""
        if purpose == "buy":
            return self.ltv_limit
        if purpose == "buy-to-let":
            return self.btl_ltv
        if purpose in ("acquire", "institution"):
            return self.inst_ltv
        raise ValueError(f"Unknown purpose for origination LTV: {purpose!r}")

    def funding_rate(self, purpose: str) -> float:
        """Monthly financing rate by purchase purpose."""
        if purpose == "buy":
            return self.mortgage_rate
        if purpose == "buy-to-let":
            return self.btl_funding_rate
        if purpose in ("acquire", "institution"):
            return self.inst_funding_rate
        raise ValueError(f"Unknown purpose for funding rate: {purpose!r}")

    # -- amortising-loan mechanics (rate is mandatory) --------------------------

    def monthly_mortgage_payment(self, price: float, ltv: float, rate: float) -> float:
        principal = ltv * price
        n = self.loan_term_months
        r = rate
        if r == 0:
            return principal / n
        return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

    def outstanding_principal(
        self, original_price: float, ltv: float, months_elapsed: int, rate: float
    ) -> float:
        P = original_price * ltv
        n = self.loan_term_months
        r = rate
        t = min(months_elapsed, n)
        if r == 0:
            return max(0.0, P - (P / n) * t)
        balance = P * ((1 + r) ** n - (1 + r) ** t) / ((1 + r) ** n - 1)
        return max(0.0, balance)

    # -- affordability ceilings (one per class) ---------------------------------

    def household_max_price(self, cash: float, annual_income: float, existing_monthly_payments: float = 0.0) -> float:
        """
        Max price a household can pay: min of deposit and DTI ceilings.
        """
        ltv = self.ltv_limit
        deposit_ceiling = cash / (1.0 - ltv) if ltv < 1.0 else float("inf")
        monthly_income = annual_income / 12.0
        max_new_payment = max(0.0, self.dti_limit * monthly_income - existing_monthly_payments)
        r = self.mortgage_rate
        n = self.loan_term_months
        if r == 0:
            max_principal = max_new_payment * n
        else:
            max_principal = max_new_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / ltv if ltv > 0 else float("inf")
        return max(0.0, min(deposit_ceiling, income_ceiling))

    def btl_max_price(self, cash: float, annual_income: float = 0.0, existing_monthly_payments: float = 0.0) -> float:
        """
        Max price for a buy-to-let purchase: min of deposit and DTI ceilings.
        """
        ltv = self.btl_ltv
        deposit_ceiling = cash / (1.0 - ltv) if ltv < 1.0 else float("inf")
        monthly_income = annual_income / 12.0
        max_new_payment = max(0.0, self.dti_limit * monthly_income - existing_monthly_payments)
        r = self.btl_funding_rate
        n = self.loan_term_months
        if r == 0:
            max_principal = max_new_payment * n
        else:
            max_principal = max_new_payment * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        income_ceiling = max_principal / ltv if ltv > 0 else float("inf")
        return max(0.0, min(deposit_ceiling, income_ceiling))

    def institution_max_price(self, cash: float) -> float:
        """Max price for an institution (deposit constraint at inst_ltv)."""
        if self.inst_ltv >= 1.0:
            return float("inf")
        return max(0.0, cash / (1.0 - self.inst_ltv))

    def max_price(self, purpose: str, cash: float, annual_income: float = 0.0, existing_monthly_payments: float = 0.0) -> float:
        """Unified affordability ceiling by purpose."""
        if purpose == "buy":
            return self.household_max_price(cash, annual_income, existing_monthly_payments)
        if purpose == "buy-to-let":
            return self.btl_max_price(cash, annual_income, existing_monthly_payments)
        if purpose in ("acquire", "institution"):
            return self.institution_max_price(cash)
        raise ValueError(f"Unknown purpose for max price: {purpose!r}")


__all__ = ["CreditEnvironment"]
