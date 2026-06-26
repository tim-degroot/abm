"""
Willingness-to-pay (WTP) formulas using a time-bounded DCF.
"""

from __future__ import annotations
import math


def housing_consumption_value(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,
) -> float:
    """Monthly housing-consumption value of a home of standardised quality, in money."""
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


def _dcf_price(
    benefit_flow: float,
    discount_rate: float,
    benefit_growth: float,
    price_growth: float,
    price_anchor: float,
    horizon: int,
) -> float:
    """
    Maximum price implied by a time-bounded DCF.

        P = b0 * annuity_factor(r, g, T) + price_anchor * ((1+g)/(1+r))^T

    Uses the geometric-series closed form instead of a loop over horizon.
    """
    d = (1.0 + benefit_growth) / (1.0 + discount_rate)
    if abs(d - 1.0) < 1e-12:
        annuity = benefit_flow * horizon
    else:
        annuity = benefit_flow * (1.0 - d**horizon) / (1.0 - d)
    terminal = price_anchor * ((1.0 + price_growth) / (1.0 + discount_rate)) ** horizon
    return annuity + terminal


def household_buy_wtp(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,  # for quality
    mortgage_rate: float,
    risk_adjusted_price_growth: float,
    price_anchor: float,
    credit_ceiling: float,
    horizon: int,
) -> float:
    """
    Owner-occupier WTP.
    """
    benefit = housing_consumption_value(quality, quality_value_scale, base_housing_value)
    p_max = _dcf_price(
        benefit_flow=benefit,
        discount_rate=mortgage_rate,
        benefit_growth=0.0,  # no qual growth for consumption benefit
        price_growth=risk_adjusted_price_growth,
        price_anchor=price_anchor,
        horizon=horizon,
    )
    return max(0.0, min(p_max, credit_ceiling))


def household_btl_wtp(
    quality: float,
    quality_sensitivity: float,
    base_rent: float,
    funding_rate: float,
    risk_adjusted_rent_growth: float,
    risk_adjusted_price_growth: float,
    price_anchor: float,
    credit_ceiling: float,
    horizon: int,
) -> float:
    """
    Household buy-to-let WTP.
    """
    net_rent = estimate_market_rent(quality, base_rent, quality_sensitivity)
    p_max = _dcf_price(
        benefit_flow=net_rent,
        discount_rate=funding_rate,
        benefit_growth=risk_adjusted_rent_growth,
        price_growth=risk_adjusted_price_growth,
        price_anchor=price_anchor,
        horizon=horizon,
    )

    return max(0.0, min(p_max, credit_ceiling))


def institutional_wtp(
    quality: float,
    quality_sensitivity: float,
    base_rent: float,
    funding_rate: float,
    required_return: float,
    rent_growth: float,
    price_growth: float,
    price_anchor: float,
    credit_ceiling: float,
    horizon: int,
) -> float:
    """
    Institutional investor WTP.

    Institutions are risk-neutral, so `expected_growth` is not risk-adjusted.
    """
    net_rent = estimate_market_rent(quality, base_rent, quality_sensitivity)
    effective_rate = funding_rate + required_return
    p_max = _dcf_price(
        benefit_flow=net_rent,
        discount_rate=effective_rate,
        benefit_growth=rent_growth,
        price_growth=price_growth,
        price_anchor=price_anchor,
        horizon=horizon,
    )

    return max(0.0, min(p_max, credit_ceiling))


def household_rent_wtp(
    quality: float,
    quality_value_scale: float,
    base_housing_value: float,
    income: float,
    dti_limit: float,
) -> float:
    """
    Maximum monthly rent a household bids.
    """
    if income <= 0.0:
        return 0.0
    benefit = housing_consumption_value(quality, quality_value_scale, base_housing_value)
    ceiling = (
        income / 12.0
    ) * dti_limit  # used as an equivalent affordability ceiling, not literally a DTI limit
    return max(0.0, min(benefit, ceiling))


__all__ = [
    "housing_consumption_value",
    "estimate_market_rent",
    "household_buy_wtp",
    "household_btl_wtp",
    "institutional_wtp",
    "household_rent_wtp",
]
