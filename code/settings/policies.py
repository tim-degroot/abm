"""
Designed credit-shock experiments, applied through the policy layer.
Usage:
    model = HousingModel(config=cfg, policy=RateIncrease(step=240))
"""

from __future__ import annotations

from code.core.credit import CreditEnvironment
from code.settings.policy_config import (
    HIGH_RATE,
    LOW_RATE,
    OO_LTV_HIGH,
    OO_LTV_LOW,
    OO_LTV_LOOSE,
    OO_LTV_TIGHT,
    SCENARIOS,
)


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


class _ScheduledCreditMacroShock(_ScheduledCreditShock):
    """Bundle credit and macro shifts to test realistic scenario packages."""

    def __init__(
        self,
        step: int,
        overrides: dict,
        label: str = "credit + macro shock",
        initial=None,
        macro: str | None = None,
        initial_macro: str | None = "Neutral",
    ):
        super().__init__(step, overrides, label, initial)
        self.macro = macro
        self.initial_macro = initial_macro
        self._macro_initialised = False

    def on_step_start(self, model):
        if not self._macro_initialised and self.initial_macro is not None:
            model.current_macro_state = self.initial_macro
            self._macro_initialised = True
        if not self._applied and model.steps >= self.step and self.macro is not None:
            model.current_macro_state = self.macro
        super().on_step_start(model)


# Single-channel Gamal shocks are retained as validation experiments.
def RateIncrease(step=240, rate=HIGH_RATE):
    """Gamal interest-rate rise: 3.7% to 8% annually."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": rate, "btl_funding_rate": rate, "inst_funding_rate": rate},
        label="rate 3.7% -> 8% annually",
        initial={
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE,
            "inst_funding_rate": LOW_RATE,
        },
    )


def RateDecrease(step=240, rate=LOW_RATE):
    """Gamal interest-rate decline: 8% to 3.7% annually."""
    return _ScheduledCreditShock(
        step,
        {"mortgage_rate": rate, "btl_funding_rate": rate, "inst_funding_rate": rate},
        label="rate 8% -> 3.7% annually",
        initial={
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE,
            "inst_funding_rate": HIGH_RATE,
        },
    )


def LTVTighten(step=240, ltv=OO_LTV_TIGHT, btl_ltv=OO_LTV_TIGHT, inst_ltv=OO_LTV_TIGHT):
    """Gamal LTV decline: 90% to 69%."""
    return _ScheduledCreditShock(
        step,
        {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label="LTV 90% -> 69%",
        initial={"ltv_limit": OO_LTV_HIGH, "btl_ltv": OO_LTV_HIGH, "inst_ltv": OO_LTV_HIGH},
    )


def LTVLoosen(step=240, ltv=OO_LTV_LOOSE, btl_ltv=OO_LTV_LOOSE, inst_ltv=OO_LTV_LOOSE):
    """Gamal LTV rise: 60% to 74%."""
    return _ScheduledCreditShock(
        step,
        {"ltv_limit": ltv, "btl_ltv": btl_ltv, "inst_ltv": inst_ltv},
        label="LTV 60% -> 74%",
        initial={"ltv_limit": OO_LTV_LOW, "btl_ltv": OO_LTV_LOW, "inst_ltv": OO_LTV_LOW},
    )


def _macro_scenario(name: str, step: int):
    """Build one bundled macro-credit scenario from policy_config.py."""
    scenario = SCENARIOS[name]
    return _ScheduledCreditMacroShock(
        step,
        scenario["overrides"],
        label=scenario["label"],
        initial=scenario["initial"],
        macro=scenario["macro"],
        initial_macro=scenario.get("initial_macro", "Neutral"),
    )


def RecessionEasingCrunch(step=240):
    """Rate cuts meet tighter underwriting in recession; offsetting-channel test."""
    return _macro_scenario("recession-easing-crunch", step)


def BoomCreditExpansion(step=240):
    """Boom, cheaper funding, and looser constraints; full credit-upswing test."""
    return _macro_scenario("boom-credit-expansion", step)


def RecessionCreditCrunch(step=240):
    """Recession plus tighter rates, leverage, and DTI; severe stress test."""
    return _macro_scenario("recession-credit-crunch", step)


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
    "recession-easing-crunch": RecessionEasingCrunch,
    "boom-credit-expansion": BoomCreditExpansion,
    "recession-credit-crunch": RecessionCreditCrunch,
    "tightening": CreditTightening,
}


__all__ = [
    "NoPolicy",
    "RateIncrease",
    "RateDecrease",
    "LTVTighten",
    "LTVLoosen",
    "RecessionEasingCrunch",
    "BoomCreditExpansion",
    "RecessionCreditCrunch",
    "CreditTightening",
    "CreditShockPolicy",
    "EXPERIMENTS",
]
