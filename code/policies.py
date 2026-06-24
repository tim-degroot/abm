"""Designed credit-shock experiments, applied through the policy layer.

There is no stochastic macro transition; the macro regime is fixed per run and
credit changes are deterministic, scheduled shocks. Each policy rewrites the
model's CreditEnvironment at a scheduled step, touching all the affected levers
consistently. This is what the report's "mirror experiments" use.

Usage:
    model = HousingModel(config=cfg, policy=RateShock(step=240, delta=0.003))
"""

from __future__ import annotations

from credit import CreditEnvironment


class NoPolicy:
    """Baseline: no intervention."""

    def on_step_start(self, model):
        pass


class _ScheduledCreditShock(NoPolicy):
    """Apply a one-off set of credit-field overrides at a scheduled step."""

    def __init__(self, step: int, overrides: dict, label: str = "credit shock"):
        self.step = step
        self.overrides = overrides
        self.label = label
        self._applied = False

    def on_step_start(self, model):
        if not self._applied and model.steps >= self.step:
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
            current.update(self.overrides)
            model.credit = CreditEnvironment(**current)
            self._applied = True
            print(f"  [SHOCK @ step {model.steps}] {self.label}: {self.overrides}")


def RateIncrease(step=240, delta=0.0025):
    """Raise the mortgage rate (and BTL/funding spreads move with it)."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": 0.00308 + delta,
         "btl_funding_rate": 0.005 + delta,
         "inst_funding_rate": 0.0045 + delta},
        label=f"rate +{delta:.4f}/mo",
    )


def RateDecrease(step=240, delta=0.0015):
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": max(0.0, 0.00308 - delta),
         "btl_funding_rate": max(0.0, 0.005 - delta),
         "inst_funding_rate": max(0.0, 0.0045 - delta)},
        label=f"rate -{delta:.4f}/mo",
    )


def LTVTighten(step=240, ltv=0.75, btl_ltv=0.60, inst_ltv=0.50):
    return _ScheduledCreditShock(
        step, {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label=f"LTV tighten -> {ltv}",
    )


def LTVLoosen(step=240, ltv=0.95, btl_ltv=0.85, inst_ltv=0.70):
    return _ScheduledCreditShock(
        step, {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label=f"LTV loosen -> {ltv}",
    )


def CreditTightening(step=240):
    """Combined tightening: higher rate, lower LTV, lower DTI."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": 0.006667, "ltv_limit": 0.80, "dti_limit": 0.30,
         "btl_funding_rate": 0.0075, "inst_funding_rate": 0.0065,
         "btl_ltv": 0.65, "inst_ltv": 0.50},
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
    "NoPolicy", "RateIncrease", "RateDecrease", "LTVTighten", "LTVLoosen",
    "CreditTightening", "CreditShockPolicy", "EXPERIMENTS",
]
