"""

It calculates Willingness To Pay (WTP) for different agents.
The WTP is the price at which the agent's surplus over their outside option equals zero.

The buyer's theoretical WTP is capped by what they can actually finance: min(wtp_surplus, credit.max_affordable_price(...)).
Applied by the caller, not inside household_wtp.

"""


def household_wtp(
    quality_value: float,
    expected_capital_gain: float,
    outside_option: float,
    mortgage_rate: float,
    ltv: float,
) -> float:
    """p_max = (E[Δp] + q_k − V_outside) / (r_m·L).

    The quality score q_k lifts the owner-occupier ceiling directly; the mortgage rate and LTV
    compress it. The outside option V_outside is the value of the best available rental."""
    ...


def investor_wtp(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
) -> float:
    """p_max = (R − φ + E[Δp]) / (r_f·L). Pass r_f^BTL for landlords, r_f for institutions.

    Since r_f < r_f^BTL, institutions always have a higher price ceiling than private landlords
    for the same property at the same rent and expectations. Quality q_k does not enter investor
    ceilings directly; it enters only through R, the achievable rent, which is increasing in q_k."""
    ...


