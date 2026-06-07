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
):
    """
    Owner-occupier break-even price (plan §6, §11):

        p_max = ( q_k + E[dp] - V_outside ) / ( r_m * L )

    then capped at the household's credit ceiling (plan §9: whichever of the
    deposit or income constraint binds first).

    quality_value        : q_k, annual £ value of the home's quality consumption
    capital_gain         : E[dp], expected annual £ price appreciation
    outside_option_value : V_outside, annual £ value of the renter alternative
    mortgage_rate        : r_m            ltv : L, loan-to-value
    credit_ceiling       : max affordable price from the credit constraints
    """
    denom = mortgage_rate * ltv
    if denom <= 0:
        return credit_ceiling
    price = (quality_value + capital_gain - outside_option_value) / denom
    return max(0.0, min(price, credit_ceiling))


def investor_wtp(annual_net_rent, capital_gain, funding_rate, ltv):
    """
    Break-even price for a yield investor — private landlord OR institution
    (plan §11). Same formula for both; they differ only in funding_rate
    (r_f^BTL for landlords, r_f for institutions, with r_f < r_f^BTL):

        p_max = ( R - phi + E[dp] ) / ( r_f * L )

    annual_net_rent : R - phi, expected annual rent net of operating costs (£)
    capital_gain    : E[dp], expected annual £ price appreciation
    funding_rate    : r_f (or r_f^BTL)      ltv : L, loan-to-value
    """
    denom = funding_rate * ltv
    if denom <= 0:
        return float("inf")
    return max(0.0, (annual_net_rent + capital_gain) / denom)


def estimate_market_rent(quality, base_rent, quality_sensitivity):
    """Expected market rent for a property: base rent scaled up with quality."""
    return base_rent * (1.0 + quality_sensitivity * quality)


def household_max_rent(income, rent_income_fraction):
    """Most a household will pay in monthly rent: a share of monthly income."""
    return income * rent_income_fraction / 12.0
