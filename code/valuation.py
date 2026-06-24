"""Willingness-to-pay (WTP) formulas.

Valuation uses a *linearised user-cost* model rather than a Gordon-growth
perpetuity. For a required (discount) rate r, a monthly benefit flow b, and an
expected capital-gain money-flow c, the price at which the agent's user cost
equals its benefit is

    P = (b + c) / r.

The expected capital-gain flow is c = g_adj * P_anchor, where P_anchor is the
property's current market value (its estimated_value) and g_adj is the expected
monthly price-growth rate, risk-adjusted for households. This is deliberately
NOT the Gordon form b / (r - g): putting g in the denominator makes the price
diverge as g -> r and is numerically unstable. Here capital gains enter once, as
a bounded flow in the numerator, and the denominator is just the discount rate.
(So we either keep g in the denominator OR as a numerator flow, never both; we
choose the numerator-flow form.)

Per the agreed design:
  * Owner-occupiers capitalise a *housing-consumption* flow (always positive)
    plus their expected capital gain. The rental alternative is handled at the
    action-choice stage, not inside the asset value.
  * Landlords / institutions capitalise net rent plus expected capital gain.
  * Risk-averse households reduce the capital-gain flow via g_adj = g - gamma*sigma,
    where sigma is the volatility of the GROWTH RATE. Institutions are
    risk-neutral and use raw g.
"""

from __future__ import annotations


def housing_consumption_value(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,
) -> float:
    """Monthly housing-consumption value (money) of a home of standardised quality.

    `base_housing_value` is the value of a median (q = 0) home; the quality term
    shifts it up or down. Clamped at 0 so a very low-quality home is never a
    negative consumption flow (standardised quality is mean-zero, so ~half of q
    is negative; this is the fix for the old negative-WTP bug).
    """
    return max(0.0, base_housing_value + quality_value_scale * quality)


def estimate_market_rent(
    quality: float,
    avg_market_rent: float,
    quality_sensitivity: float,
) -> float:
    """Monthly rent a property of this quality could command."""
    if avg_market_rent <= 0.0:
        return 0.0
    return max(0.0, avg_market_rent * (1.0 + quality_sensitivity * quality))


def _user_cost_price(
    benefit_flow: float,
    discount_rate: float,
    expected_growth: float,
    price_anchor: float,
) -> float:
    """P = (benefit_flow + expected_growth * price_anchor) / discount_rate.

    `expected_growth` may be negative (expected depreciation), which lowers WTP.
    Returns 0 for a non-positive total flow; never diverges.
    """
    if discount_rate <= 0.0:
        return float("inf")
    capital_gain_flow = expected_growth * price_anchor
    total_flow = benefit_flow + capital_gain_flow
    if total_flow <= 0.0:
        return 0.0
    return total_flow / discount_rate


def household_buy_wtp(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,
    mortgage_rate: float,
    expected_growth: float,
    price_anchor: float,
    credit_ceiling: float,
) -> float:
    """Owner-occupier WTP (user-cost form). `expected_growth` is risk-adjusted."""
    benefit = housing_consumption_value(quality, quality_value_scale, base_housing_value)
    p_max = _user_cost_price(benefit, mortgage_rate, expected_growth, price_anchor)
    return max(0.0, min(p_max, credit_ceiling))


def household_btl_wtp(
    quality: float,
    quality_sensitivity: float,
    base_rent: float,
    funding_rate: float,
    expected_growth: float,
    price_anchor: float,
    credit_ceiling: float,
) -> float:
    """Household buy-to-let WTP: net rent + capital gain, over the BTL rate."""
    net_rent = estimate_market_rent(quality, base_rent, quality_sensitivity)
    p_max = _user_cost_price(net_rent, funding_rate, expected_growth, price_anchor)
    return max(0.0, min(p_max, credit_ceiling))


def institutional_wtp(
    quality: float,
    quality_sensitivity: float,
    base_rent: float,
    funding_rate: float,
    required_return: float,
    expected_growth: float,
    price_anchor: float,
    credit_ceiling: float = float("inf"),
) -> float:
    """Institutional WTP: net rent + capital gain, over (funding + required return).

    Institutions are risk-neutral, so `expected_growth` is not risk-adjusted.
    """
    net_rent = estimate_market_rent(quality, base_rent, quality_sensitivity)
    effective_rate = funding_rate + required_return
    p_max = _user_cost_price(net_rent, effective_rate, expected_growth, price_anchor)
    return max(0.0, min(p_max, credit_ceiling))


def household_rent_wtp(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,
    income: float,
    dti_limit: float,
) -> float:
    """Maximum monthly rent a household bids: housing-consumption value capped by
    an affordability ceiling (the DTI limit reused as a rent-to-income cap, so
    there is no separate max_rent_income_ratio parameter).
    """
    if income <= 0.0:
        return 0.0
    benefit = housing_consumption_value(quality, quality_value_scale, base_housing_value)
    ceiling = (income / 12.0) * dti_limit
    return max(0.0, min(benefit, ceiling))


__all__ = [
    "housing_consumption_value",
    "estimate_market_rent",
    "household_buy_wtp",
    "household_btl_wtp",
    "institutional_wtp",
    "household_rent_wtp",
]
