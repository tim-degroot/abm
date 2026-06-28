"""
Shared constants for policy experiments.

Rates are monthly because model steps are monthly.
"""

# Run design
N_RUNS = 20
SHOCK_STEP = 240
PRE_SHOCK_MONTHS = 60
POST_SHOCK_MONTHS = 120
ROLLING_WINDOW = 12
WORKERS = 16

POLICIES_TO_RUN = [
    "rate-up",
    "rate-down",
    "ltv-tighten",
    "ltv-loosen",
    "recession-easing-crunch",
    "boom-credit-expansion",
    "recession-credit-crunch",
]

# Scenario anchors
LOW_RATE = 0.037 / 12  # Gamal rate
HIGH_RATE = 0.08 / 12  # Gamal shock
BTL_RATE_SPREAD = 0.0038 / 12  # Gamal OO-BTL spread

OO_LTV_TIGHT = 0.69  # Gamal median low
OO_LTV_HIGH = 0.90  # Gamal band
OO_LTV_LOW = 0.60  # Gamal band
OO_LTV_LOOSE = 0.74  # Gamal median high

BTL_LTV_TIGHT = 0.60  # Gamal BTL deposit
BTL_LTV_LOOSE = 0.75  # Gamal BTL deposit

INST_LTV_SCENARIO = 0.60  # model default
DTI_TIGHT = 0.33  # Gamal affordability
DTI_LOOSE = 0.40  # model default


SCENARIOS = {
    "recession-easing-crunch": {
        "label": "recession + rate cut, LTV/DTI crunch",
        "initial_macro": "Neutral",
        "macro": "Recession",
        "initial": {
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_HIGH,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
        "overrides": {
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_TIGHT,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
    },
    "boom-credit-expansion": {
        "label": "boom + rate cut, LTV/DTI loosening",
        "initial_macro": "Neutral",
        "macro": "Boom",
        "initial": {
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_LOW,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
        "overrides": {
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_LOOSE,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
    },
    "recession-credit-crunch": {
        "label": "recession + rate rise, LTV/DTI crunch",
        "initial_macro": "Neutral",
        "macro": "Recession",
        "initial": {
            "mortgage_rate": LOW_RATE,
            "btl_funding_rate": LOW_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": LOW_RATE,
            "ltv_limit": OO_LTV_HIGH,
            "btl_ltv": BTL_LTV_LOOSE,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_LOOSE,
        },
        "overrides": {
            "mortgage_rate": HIGH_RATE,
            "btl_funding_rate": HIGH_RATE + BTL_RATE_SPREAD,
            "inst_funding_rate": HIGH_RATE,
            "ltv_limit": OO_LTV_TIGHT,
            "btl_ltv": BTL_LTV_TIGHT,
            "inst_ltv": INST_LTV_SCENARIO,
            "dti_limit": DTI_TIGHT,
        },
    },
}
