"""
Valuation module.

Agents do not contain valuation logic.
They call functions here to obtain willingness-to-pay (WTP).

Two fundamentally distinct valuation models:

  Household (owner-occupier logic):
    WTP driven by quality consumption + expected capital gain - financing cost.
    Credit-constrained ceiling applied last.

  Institution (yield-investor logic):
    WTP driven by expected rental income + capital gain - financing cost.
    No quality consumption term; no credit ceiling (effectively unconstrained).

This asymmetry is the architectural foundation of the marginal-pricer mechanism.
The two ceilings respond differently to changes in mortgage rates, rent levels,
and credit conditions, causing regime switches in who sets prices.
"""

# ---------------------------------------------------------------------------
# Household WTP
# ---------------------------------------------------------------------------


def household_wtp(
    quality,
    expected_price_growth,
    mortgage_rate,
    ltv,
    outside_option_value,
    credit_ceiling,
):
    """
    Maximum willingness-to-pay for a household bidding as owner-occupier.

    Derived from the P&L surplus equation:
        Pi_H = E[dp] + q - V_outside - r_m * L * p = 0
        => p_max = (E[dp] + q - V_outside) / (r_m * L)

    quality              : float, standardised property quality
    expected_price_growth: float, annual expected price appreciation (fraction)
    mortgage_rate        : float, annual mortgage rate
    ltv                  : float, loan-to-value ratio (e.g. 0.85)
    outside_option_value : float, value of best available rental alternative
    credit_ceiling       : float, max affordable price from CreditEnvironment

    Returns WTP >= 0, capped at credit_ceiling.
    """
    financing_cost_factor = mortgage_rate * ltv
    if financing_cost_factor <= 0:
        # degenerate: no financing cost means unconstrained (use credit ceiling)
        return credit_ceiling

    numerator = expected_price_growth + quality - outside_option_value
    wtp = numerator / financing_cost_factor

    # Apply credit constraint
    wtp = min(wtp, credit_ceiling)
    return max(wtp, 0.0)


# ---------------------------------------------------------------------------
# Institution WTP
# ---------------------------------------------------------------------------


def institution_wtp(
    expected_rent, operating_cost_fraction, expected_price_growth, funding_rate, ltv
):
    """
    Maximum willingness-to-pay for an institutional investor.

    Derived from:
        Pi_I = R - phi - r_f * L * p + E[dp] = 0
        => p_max = (R - phi + E[dp]) / (r_f * L)

    expected_rent           : float, expected annual rent income
    operating_cost_fraction : float, operating costs as fraction of rent (e.g. 0.15)
    expected_price_growth   : float, annual expected capital appreciation
    funding_rate            : float, institution's annual funding rate
    ltv                     : float, leverage ratio

    Returns WTP >= 0. No credit ceiling applied (institutions are
    effectively unconstrained relative to individual properties).
    """
    net_rent = expected_rent * (1.0 - operating_cost_fraction)
    financing_cost_factor = funding_rate * ltv
    if financing_cost_factor <= 0:
        return float("inf")

    wtp = (net_rent + expected_price_growth) / financing_cost_factor
    return max(wtp, 0.0)


# ---------------------------------------------------------------------------
# Rental valuation
# ---------------------------------------------------------------------------


def household_max_rent(income, rent_income_fraction=0.35):
    """
    Maximum monthly rent a household is willing/able to pay.

    Simple affordability ceiling: rent_income_fraction of monthly income.
    """
    return income * rent_income_fraction / 12.0


def estimate_market_rent(quality, base_rent, quality_sensitivity=0.3):
    """
    Estimate achievable market rent for a property given its quality.

    Used by landlords and institutions when valuing a property
    for rental purposes before observing actual auction outcomes.

    base_rent          : float, average rent in the zone/market
    quality_sensitivity: float, elasticity of rent to quality
    """
    return base_rent * (1.0 + quality_sensitivity * quality)


# ---------------------------------------------------------------------------
# Outside option
# ---------------------------------------------------------------------------


def renter_outside_option(avg_rent, income):
    """
    Value of the outside option for a renter (staying as renter).

    Expressed as a negative flow cost: the agent avoids paying rent
    by owning. Here we return a simple normalised value.

    For the first implementation we use a reduced-form proxy:
    outside option value = avg_rent / income (rent-burden measure).
    This enters household_wtp as the opportunity cost of not buying.
    """
    if income <= 0:
        return 0.0
    return avg_rent / income
