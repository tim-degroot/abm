"""
Valuation: willingness-to-pay (WTP) formulas.

Maximum WTP is the price at which an agent's surplus over its outside
option is exactly zero.
"""


def household_wtp(
    quality_value,
    capital_gain,
    outside_option_value,
    mortgage_rate,
    ltv,
    credit_ceiling,
    *,
    max_price_to_income,  # remove
    income,
):
    """
    quality_value        : q_k, monthly £ value of the home's quality consumption
    capital_gain         : E[dp], expected monthly £ price appreciation
    outside_option_value : V_outside, monthly £ value of the renter alternative
    mortgage_rate        : r_m (monthly)      ltv : L, loan-to-value
    credit_ceiling       : max affordable price from the credit constraints
    max_price_to_income  : fundamentals ceiling = this * income
    income               : bidder's monthly income
    """
    denom = mortgage_rate * ltv
    if denom <= 0:
        raw = float("inf")
    else:
        raw = (quality_value + capital_gain - outside_option_value) / denom

    ceiling_bound = income_ceiling < raw

    price = min(raw, credit_ceiling, income_ceiling)
    return max(0.0, price), ceiling_bound


def investor_wtp(
    monthly_net_rent,
    capital_gain,
    funding_rate,
    ltv,
    *,
    max_price_to_rent,
    expected_monthly_rent,
):
    """
    Break-even price for a yield investor — private landlord OR institution
    (plan §11). Same formula for both; they differ only in funding_rate
    (r_f^BTL for landlords, r_f for institutions, with r_f < r_f^BTL):

        p_max = ( R - phi + E[dp] ) / ( r_f * L )

    capped at a fundamentals ceiling — a price-to-rent multiple of the
    property's expected monthly rent. As in household_wtp, this is a hard
    SAFETY NET on the final bid anchored to the asset's real income capacity,
    not a constraint on beliefs.

    monthly_net_rent      : R - phi, expected monthly rent net of operating costs (£)
    capital_gain          : E[dp], expected monthly £ price appreciation
    funding_rate          : r_f (or r_f^BTL, monthly)      ltv : L, loan-to-value
    max_price_to_rent     : fundamentals ceiling = this * expected_monthly_rent
    expected_monthly_rent : R, expected GROSS monthly rent of the property (£)

    Returns (wtp, ceiling_bound) where ceiling_bound is True iff the
    fundamentals (price-to-rent) ceiling sits below the formula-computed price.
    """
    denom = funding_rate * ltv
    if denom <= 0:
        raw = float("inf")
    else:
        raw = (monthly_net_rent + capital_gain) / denom

    rent_ceiling = max_price_to_rent * expected_monthly_rent
    ceiling_bound = rent_ceiling < raw

    return max(0.0, min(raw, rent_ceiling)), ceiling_bound


def expected_capital_gain(
    mode,
    market_price,
    *,
    fixed_level,
    growth_signal,
    growth_min,
    growth_max,
):
    """
    Expected per-period capital gain E[dp], in £, for the WTP numerator (plan §11).

    The naive form `expected_price_growth * market_price` creates an explosive
    feedback loop. Two configurable modes break that loop:

    - "fixed_level":   a constant £ level per period (`fixed_level`), entirely
                       independent of the price.

    - "bounded_growth": g * market_price, with g = clamp(growth_signal,
                        growth_min, growth_max). g is sourced from the
                        rent-growth / macro signal, never the realised-price EMA.

    market_price : the current market price level the gain is taken against (£).
    """
    if mode == "fixed_level":
        return fixed_level
    if mode == "bounded_growth":
        g = min(growth_max, max(growth_min, growth_signal))
        return g * market_price
    raise ValueError(
        f"Unknown capital_gain_mode {mode!r}; expected 'fixed_level' or 'bounded_growth'."
    )


def estimate_market_rent(quality, base_rent, quality_sensitivity):
    """Expected market rent for a property: base rent scaled up with quality."""
    return base_rent * (1.0 + quality_sensitivity * quality)


def household_max_rent(income, rent_income_fraction):
    """Most a household will pay in monthly rent: a share of monthly income."""
    return income * rent_income_fraction
