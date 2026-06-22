"""
Valuation: willingness-to-pay (WTP) formulas.

Maximum WTP is the price at which an agent's surplus over their outside option is zero.
"""


def household_wtp(
    quality_value: float,
    outside_option_value: float,
    mortgage_rate: float,
    expected_growth: float,
    credit_ceiling: float,
) -> float:
    """Owner-occupier WTP.

    Perpetuity valuation: the price P that satisfies
        q_k + g·P - r_m·P - V_out = 0  ⇒  P = (q_k - V_out) / (r_m - g)

    Parameters
    ----------
    quality_value        : q_k, monthly value of housing consumption
    outside_option_value : V_outside, monthly value of the rental alternative
    mortgage_rate        : r_m (monthly)
    expected_growth      : g, expected monthly price growth rate
    credit_ceiling       : max affordable price from deposit/DTI constraints

    Returns
    -------
    p_max clipped to [0, credit_ceiling].
    """
    numerator = quality_value - outside_option_value
    denominator = mortgage_rate - expected_growth
    if denominator <= 0.0:
        return credit_ceiling
    p_max = numerator / denominator
    return max(0.0, min(p_max, credit_ceiling))


def investor_wtp(
    net_rent: float,
    funding_rate: float,
    expected_growth: float,
) -> float:
    """Private landlord or institutional investor WTP.

    Perpetuity valuation:
        P = net_rent / (r_f - g)

    Parameters
    ----------
    net_rent        : R - φ, expected monthly net rent
    funding_rate    : r_f (or r_f^BTL, monthly)
    expected_growth : g, expected monthly price growth rate

    Returns
    -------
    p_max (≥ 0). Credit constraints are applied by the caller.
    """
    numerator = net_rent
    denominator = funding_rate - expected_growth
    if denominator <= 0.0:
        return float("inf")
    return max(0.0, numerator / denominator)


def household_rent_wtp(
    quality_value: float,
    base_rent: float,
    income: float,
    max_rent_income_ratio: float,
) -> float:
    """Maximum monthly rent a household is willing to pay for a property.

    The household will pay up to the quality premium plus the baseline
    market rent, subject to an affordability ceiling.

    Parameters
    ----------
    quality_value         : q_k, monthly value of this property's quality
    base_rent             : outside-option monthly rent for a baseline unit
    income                : annual household income
    max_rent_income_ratio : maximum fraction of monthly income for rent

    Returns
    -------
    Monthly rent bid clipped to [0, affordability ceiling].
    """
    if income <= 0.0:
        return 0.0
    wtp = quality_value + base_rent
    ceiling = (income / 12.0) * max_rent_income_ratio
    return max(0.0, min(wtp, ceiling))


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
