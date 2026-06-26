"""
Designed credit-shock experiments, applied through the policy layer.
Usage:
    model = HousingModel(config=cfg, policy=RateIncrease(step=240))
"""

from __future__ import annotations
from code.core.credit import CreditEnvironment


class NoPolicy:
    """Baseline: no intervention."""

    def on_step_start(self, model):
        pass


class _ScheduledCreditShock(NoPolicy):
    """Apply pre-shock credit levels, then a permanent shift at a scheduled step."""

    def __init__(self, step: int, overrides: dict, label: str = "credit shock", initial=None):
        self.step = step
        self.overrides = overrides
        self.label = label
        self.initial = initial
        self._initialised = False
        self._applied = False

    def on_step_start(self, model):
        shock = not self._applied and model.steps >= self.step
        initialise = not self._initialised and self.initial is not None
        if initialise or shock:
            overrides = self.overrides if shock else self.initial
            current = {
                "mortgage_rate": model.credit.mortgage_rate,
                "ltv_limit": model.credit.ltv_limit,
                "dti_limit": model.credit.dti_limit,
                "loan_term_months": model.credit.loan_term_months,
                "btl_funding_rate": model.credit.btl_funding_rate,
                "btl_ltv": model.credit.btl_ltv,
                "inst_funding_rate": model.credit.inst_funding_rate,
                "inst_ltv": model.credit.inst_ltv,
            }
            current.update(overrides)
            model.credit = CreditEnvironment(**current)
            self._initialised = True
            if shock:
                self._applied = True
                print(f"  [SHOCK @ step {model.steps}] {self.label}: {self.overrides}")


def RateIncrease(step=240, rate=0.08 / 12):
    """Gamal interest-rate rise: 3.7% to 8% annually."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": rate, "btl_funding_rate": rate, "inst_funding_rate": rate},
        label="rate 3.7% -> 8% annually",
        initial={
            "mortgage_rate": 0.037 / 12,
            "btl_funding_rate": 0.037 / 12,
            "inst_funding_rate": 0.037 / 12,
        },
    )


def RateDecrease(step=240, rate=0.037 / 12):
    """Gamal interest-rate decline: 8% to 3.7% annually."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": rate, "btl_funding_rate": rate, "inst_funding_rate": rate},
        label="rate 8% -> 3.7% annually",
        initial={
            "mortgage_rate": 0.08 / 12,
            "btl_funding_rate": 0.08 / 12,
            "inst_funding_rate": 0.08 / 12,
        },
    )


def LTVTighten(step=240, ltv=0.69, btl_ltv=0.69, inst_ltv=0.69):
    """Gamal LTV decline: 90% to 69%."""
    return _ScheduledCreditShock(
        step,
        {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label="LTV 90% -> 69%",
        initial={"ltv_limit": 0.90, "btl_ltv": 0.90, "inst_ltv": 0.90},
    )


def LTVLoosen(step=240, ltv=0.74, btl_ltv=0.74, inst_ltv=0.74):
    """Gamal LTV rise: 60% to 74%."""
    return _ScheduledCreditShock(
        step,
        {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label="LTV 60% -> 74%",
        initial={"ltv_limit": 0.60, "btl_ltv": 0.60, "inst_ltv": 0.60},
    )


def CreditTightening(step=240):
    """Combined tightening: higher rate, lower LTV, lower DTI."""
    return _ScheduledCreditShock(
        step,
        {
            "mortgage_rate": 0.006667,
            "ltv_limit": 0.80,
            "dti_limit": 0.30,
            "btl_funding_rate": 0.0075,
            "inst_funding_rate": 0.0065,
            "btl_ltv": 0.65,
            "inst_ltv": 0.50,
        },
        label="combined tightening",
    )


# Backwards-compatible alias for the old experiment name in run.py.
CreditShockPolicy = CreditTightening


EXPERIMENTS = {
    "rate-up": RateIncrease,
    "rate-down": RateDecrease,
    "ltv-tighten": LTVTighten,
    "ltv-loosen": LTVLoosen,
    "tightening": CreditTightening,
}


__all__ = [
    "NoPolicy",
    "RateIncrease",
    "RateDecrease",
    "LTVTighten",
    "LTVLoosen",
    "CreditTightening",
    "CreditShockPolicy",
    "EXPERIMENTS",
]
