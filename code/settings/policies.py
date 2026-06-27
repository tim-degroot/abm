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


class _ScheduledCreditMacroShock(_ScheduledCreditShock):
    """Pair a credit package with a macro-regime shift for scenario experiments."""

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


# Scenario anchors: Gamal shocks + BTL spread/LTV from Gamal's UK BTL discussion.
LOW_RATE = 0.037 / 12  # Gamal rate
HIGH_RATE = 0.08 / 12  # Gamal shock
BTL_RATE_SPREAD = 0.0038 / 12  # Gamal BTL spread
OO_LTV_TIGHT = 0.69  # Gamal median low
OO_LTV_HIGH = 0.90  # Gamal band
OO_LTV_LOW = 0.60  # Gamal band
OO_LTV_LOOSE = 0.74  # Gamal median high
BTL_LTV_TIGHT = 0.60  # Gamal BTL deposit
BTL_LTV_LOOSE = 0.75  # Gamal BTL deposit
INST_LTV_SCENARIO = 0.60  # model default
DTI_TIGHT = 0.33  # Gamal affordability
DTI_LOOSE = 0.40  # model default


def RecessionEasingCrunch(step=240):
    """Rates ease in recession, but underwriting tightens; tests offsetting channels."""
    return _ScheduledCreditMacroShock(
        step,
        {
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_TIGHT,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
        label="recession + rate cut, LTV/DTI crunch",
        initial={
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_HIGH,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
        macro="Recession",
    )


def BoomCreditExpansion(step=240):
    """Boom plus cheaper funding and looser constraints; tests a full credit upswing."""
    return _ScheduledCreditMacroShock(
        step,
        {
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_LOOSE,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
        label="boom + rate cut, LTV/DTI loosening",
        initial={
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_LOW,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
        macro="Boom",
    )


def RecessionCreditCrunch(step=240):
    """Recession plus tighter credit across rates, leverage, and DTI; severe stress."""
    return _ScheduledCreditMacroShock(
        step,
        {
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_TIGHT,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
        label="recession + rate rise, LTV/DTI crunch",
        initial={
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_HIGH,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
        macro="Recession",
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
