"""
Valuation: willingness-to-pay (WTP) formulas.

Agents hold no valuation logic; they call these functions. Maximum WTP is the
price at which an agent's surplus over its outside option is exactly zero
(plan §11). Every agent shares the same P&L object (plan §2.1):

    Pi = CF + E[dp] - FC          (cash flow + expected capital gain - financing)

What differs is the source of cash flow and the financing cost, giving two
break-even forms. All money terms are annual £.

Owner-occupier — household_wtp (plan §6, §11):
    p_max = ( q_k + E[dp] - V_outside ) / ( r_m * L )
  The cash flow is the consumption value of the home's quality (imputed rent
  cancels between owning and renting, so it does not appear). Credit-capped.

Yield investor — investor_wtp (plan §11), used by BOTH private landlords and
institutions (same formula, only the funding rate differs):
    p_max = ( R - phi + E[dp] ) / ( r_f * L )
  The cash flow is net rental income. Institutions fund at r_f; private
  landlords at r_f^BTL, with r_f < r_f^BTL (plan §6), so for the same property
  institutions always have the higher ceiling.

Symbols (plan §6, §11):
    q_k       quality consumption value of the home (owner-occupiers only)
    E[dp]     expected capital gain, in £ (here exogenous: growth * current value)
    V_outside value of the best rental alternative (set to 0 for now)
    r_m       mortgage rate           L   loan-to-value (leverage)
    R         annual rental income    phi operating costs
    r_f       investor funding rate (r_f for institutions, r_f^BTL for landlords)

Because r_m, r_f and r_f^BTL move differently with credit conditions, the two
ceilings rise and fall at different rates — this is what drives the
marginal-pricer regime switches (plan §2).
"""


def household_wtp(
    quality_value,
    capital_gain,
    outside_option_value,
    mortgage_rate,
    ltv,
    credit_ceiling,
    *,
    max_price_to_income,
    income,
):
    """
    Owner-occupier break-even price (plan §6, §11):

        p_max = ( q_k + E[dp] - V_outside ) / ( r_m * L )

    then capped at the household's credit ceiling (plan §9: whichever of the
    deposit or income constraint binds first), and finally at a fundamentals
    ceiling — a price-to-income multiple of the bidder's income. The
    fundamentals ceiling is a hard SAFETY NET on the final bid: it guarantees no
    bid can detach from the bidder's real economic capacity regardless of what
    the expectation/belief terms produce. It is not a constraint on beliefs.

    quality_value        : q_k, annual £ value of the home's quality consumption
    capital_gain         : E[dp], expected annual £ price appreciation
    outside_option_value : V_outside, annual £ value of the renter alternative
    mortgage_rate        : r_m            ltv : L, loan-to-value
    credit_ceiling       : max affordable price from the credit constraints
    max_price_to_income  : fundamentals ceiling = this * income
    income               : bidder's annual income (£)

    Returns (wtp, ceiling_bound) where ceiling_bound is True iff the
    fundamentals (price-to-income) ceiling sits below the formula-computed price
    — i.e. the safety net, not economics, is capping the raw computed WTP.
    """
    denom = mortgage_rate * ltv
    if denom <= 0:
        raw = float("inf")
    else:
        raw = (quality_value + capital_gain - outside_option_value) / denom

    income_ceiling = max_price_to_income * income
    ceiling_bound = income_ceiling < raw

    price = min(raw, credit_ceiling, income_ceiling)
    return max(0.0, price), ceiling_bound


def investor_wtp(
    annual_net_rent,
    capital_gain,
    funding_rate,
    ltv,
    *,
    max_price_to_rent,
    expected_annual_rent,
):
    """
    Break-even price for a yield investor — private landlord OR institution
    (plan §11). Same formula for both; they differ only in funding_rate
    (r_f^BTL for landlords, r_f for institutions, with r_f < r_f^BTL):

        p_max = ( R - phi + E[dp] ) / ( r_f * L )

    capped at a fundamentals ceiling — a price-to-rent multiple of the
    property's expected gross annual rent. As in household_wtp, this is a hard
    SAFETY NET on the final bid anchored to the asset's real income capacity,
    not a constraint on beliefs.

    annual_net_rent      : R - phi, expected annual rent net of operating costs (£)
    capital_gain         : E[dp], expected annual £ price appreciation
    funding_rate         : r_f (or r_f^BTL)      ltv : L, loan-to-value
    max_price_to_rent    : fundamentals ceiling = this * expected_annual_rent
    expected_annual_rent : R, expected GROSS annual rent of the property (£)

    Returns (wtp, ceiling_bound) where ceiling_bound is True iff the
    fundamentals (price-to-rent) ceiling sits below the formula-computed price.
    """
    denom = funding_rate * ltv
    if denom <= 0:
        raw = float("inf")
    else:
        raw = (annual_net_rent + capital_gain) / denom

    rent_ceiling = max_price_to_rent * expected_annual_rent
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
    feedback loop: realised price -> growth signal -> expectation -> WTP ->
    realised price, with no anchor (it produced £10M houses). Two configurable
    modes break that loop (config: valuation.capital_gain_mode):

    - "fixed_level":   a constant £ level per period (`fixed_level`), entirely
                       independent of the price. Matches plan §11, where E[dp] is
                       a modest per-period £ amount, not growth * price.

    - "bounded_growth": keep the proportional form `g * market_price` (prices do
                       grow roughly proportionally), but (a) CLAMP g to
                       [growth_min, growth_max], and (b) SOURCE g from
                       `growth_signal` — the rent-growth / macro signal — never
                       from the realised-price EMA. Because g no longer depends on
                       the realised price, the price cannot drive its own
                       expectation and the loop is broken. (Until the macro state
                       machine lands, `growth_signal` is the agent's adaptive
                       rent-growth expectation, which is income- not price-driven.)

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
    return income * rent_income_fraction / 12.0
