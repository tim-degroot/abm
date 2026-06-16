"""
Valuation: willingness-to-pay (WTP) formulas.

Maximum WTP is the price at which an agent's surplus over its outside
option is zero.
"""


def household_wtp():
    """
    quality_value        : q_k, monthly value of the home's quality consumption
    capital_gain         : E[dp], expected monthly price appreciation
    outside_option_value : V_outside, monthly value of the renter alternative
    mortgage_rate        : r_m (monthly)      ltv : L, loan-to-value
    credit_ceiling       : max affordable price from the credit constraints
    income               : bidder's monthly income
    """
    pass


def investor_wtp():
    """
    monthly_net_rent      : R - phi, expected monthly rent net of operating costs
    capital_gain          : E[dp], expected monthly £ price appreciation
    funding_rate          : r_f (or r_f^BTL, monthly)      ltv : L, loan-to-value
    expected_monthly_rent : R, expected GROSS monthly rent of the property
    """
    pass
