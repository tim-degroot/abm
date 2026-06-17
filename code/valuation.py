"""
Valuation: willingness-to-pay (WTP) formulas.

Maximum WTP is the price at which an agent's surplus over its outside
option is zero.

Owner-occupier:
  p_max = (q_k + E[Δp] - V_outside) / (r_m * L)

Private landlord / institutional investor:
  p_max = (R - φ + E[Δp]) / (r_f * L)
"""


def household_wtp(
    quality_value: float,
    capital_gain: float,
    outside_option_value: float,
    mortgage_rate: float,
    ltv: float,
    credit_ceiling: float,
) -> float:
    """Owner-occupier WTP.

    Parameters
    ----------
    quality_value        : q_k, monthly value of housing consumption
    capital_gain         : E[Δp], expected monthly £ appreciation
    outside_option_value : V_outside, monthly value of the rental alternative
    mortgage_rate        : r_m (monthly)
    ltv                  : L, loan-to-value ratio
    credit_ceiling       : max affordable price from deposit/DTI constraints

    Returns
    -------
    p_max clipped to [0, credit_ceiling].
    """
    numerator = quality_value + capital_gain - outside_option_value
    denominator = mortgage_rate * ltv
    if denominator <= 0.0:
        return 0.0
    p_max = numerator / denominator
    return max(0.0, min(p_max, credit_ceiling))


def investor_wtp(
    net_rent: float,
    capital_gain: float,
    funding_rate: float,
    ltv: float,
) -> float:
    """Private landlord or institutional investor WTP.

    Parameters
    ----------
    net_rent     : R - φ, expected monthly net rent
    capital_gain : E[Δp], expected monthly £ appreciation
    funding_rate : r_f (or r_f^BTL, monthly)
    ltv          : L, loan-to-value ratio

    Returns
    -------
    p_max (>= 0). Credit constraints are applied by the caller.
    """
    numerator = net_rent + capital_gain
    denominator = funding_rate * ltv
    if denominator <= 0.0:
        return 0.0
    p_max = numerator / denominator
    return max(0.0, p_max)


def household_max_rent(income: float, max_rent_income_ratio: float) -> float:
    """Maximum monthly rent bid from the affordability ceiling."""
    if income <= 0.0:
        return 0.0
    return income * max_rent_income_ratio


def estimate_market_rent(
    quality: float,
    avg_market_rent: float,
    quality_sensitivity: float,
) -> float:
    """Monthly rent a property could command, based on its quality.

    Rent scales linearly with quality around the market average.
    """
    if avg_market_rent <= 0.0:
        return 0.0
    return avg_market_rent * (1.0 + quality_sensitivity * quality)
