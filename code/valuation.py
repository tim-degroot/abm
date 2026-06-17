"""
Valuation: P&L and willingness-to-pay (monetary, £/month).

Two price objects:
  - Π = CF + E[Δp] − FC, the per-period P&L by agent class (§6)
  - p_max, the WTP ceiling derived from Π (§11): the feasible-set bound and
    the truthful Vickrey bid

These are NOT logit inputs (that's utility.py, which reuses the Π below).
None of these functions take an agent: the caller passes the agent's own
numbers (its expectations, income, credit ceiling).

Rates and flows are monthly (§27).
"""

# ---------------------------------------------------------------------------
# P&L:  Π = CF + E[Δp] − FC   (§6)
# ---------------------------------------------------------------------------


def pnl_owner_occupier(
    expected_capital_gain: float,
    mortgage_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Π_H = E[Δp] − r_m·L·p."""
    ...


def pnl_landlord(
    net_rent: float,
    expected_capital_gain: float,
    btl_funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Π_L = (R − φ) − r_f^BTL·L·p + E[Δp]."""
    ...


def pnl_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Π_I = (R − φ) − r_f·L·p + E[Δp].  r_f < r_f^BTL."""
    ...


# ---------------------------------------------------------------------------
# WTP ceiling:  p_max   (§11)
# ---------------------------------------------------------------------------
# Price where surplus over the outside option hits zero, capped by credit.
# Returns (p_max, binding) where binding names the constraint that bound.


def household_wtp(
    quality_value: float,
    expected_capital_gain: float,
    outside_option: float,
    mortgage_rate: float,
    ltv: float,
    credit_ceiling: float,
    income: float,
) -> tuple[float, str]:
    """Owner-occupier ceiling: (E[Δp] + q_k − V_outside) / (r_m·L), capped at
    `credit_ceiling`. binding ∈ {"surplus", "deposit", "dti"}."""
    ...


def estimate_market_rent(quality: float, avg_market_rent: float, quality_sensitivity: float) -> float:
    """R(q_k): quality-adjusted rent for one property. Scales avg_market_rent by
    the property's relative quality; quality_sensitivity controls elasticity."""
    ...


def household_max_rent(income: float, max_rent_share: float) -> float:
    """Stage-3 rental bid ceiling: the most a household will pay per month.
    max_rent_share comes from config (e.g. 0.35)."""
    ...


def investor_wtp(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    credit_ceiling: float | None = None,
) -> tuple[float, str]:
    """Investor ceiling: (R − φ + E[Δp]) / (r_f·L). Landlords pass r_f^BTL,
    institutions pass r_f. `credit_ceiling` caps constrained bidders."""
    ...
