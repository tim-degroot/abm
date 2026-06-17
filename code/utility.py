"""
Utility: converts P&L into logit inputs.

Flow per action: compute Π (P&L) → compute V (value) → compute ΔV (surplus over outside
option) → apply U (risk curvature) → feed logit.

Sections:
  - P&L helpers     raw profit/loss per agent type
  - Value V         monetary value of an outcome
  - ΔV per action   surplus = V − V_outside, the logit input
  - U(ΔV)           risk curvature (CRRA for households, identity for institutions)
  - Logit           action and property selection mechanism
"""

from typing import Hashable, Mapping

# ---------------------------------------------------------------------------
# P&L helpers  (internal — called by value and delta_v functions below)
# ---------------------------------------------------------------------------


def _pnl_owner_occupier(
    expected_capital_gain: float,
    mortgage_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for an owner-occupier.
    Gain from expected price growth minus monthly mortgage cost.
    Π_H = E[Δp] − r_m·L·p, where r_m = mortgage rate, L = LTV ratio.
    Imputed rent does not appear here because it is common to both owning and renting
    and cancels in the surplus comparison."""
    ...


def _pnl_landlord(
    net_rent: float,
    expected_capital_gain: float,
    btl_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for a private landlord.
    Rental income plus expected price growth minus mortgage cost.
    Π_L = R − φ − r_f^BTL·L·p + E[Δp]"""
    ...


def _pnl_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for an institutional investor.
    Same structure as landlord but cheaper funding rate.
    Π_I = R − φ − r_f·L·p + E[Δp]"""
    ...


# ---------------------------------------------------------------------------
# Value V  — monetary value before risk curvature
# ---------------------------------------------------------------------------


def value_owner_occupier(
    quality_value: float,
    expected_capital_gain: float,
    mortgage_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of owning and living in a property.
    Quality adds direct consumption value on top of P&L.
    V^OO = q_k + Π_H"""
    ...


def value_landlord(
    net_rent: float,
    expected_capital_gain: float,
    btl_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of owning and renting out a property.
    Quality does not enter directly — only through the rent R.
    V^LL = Π_L"""
    ...


def value_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of a property for an institutional investor.
    Risk-neutral: value equals expected P&L, no curvature applied.
    V^INST = E[Π_I]"""
    ...


# ---------------------------------------------------------------------------
# ΔV per action — the logit inputs  (Stage 1)
# ---------------------------------------------------------------------------
# ΔV = V(action) − V(outside option).
# Institutions use risk-free return as the outside option for every action.
# Households use their best feasible rental as the outside option.


# TODO: outside options and action sets for households and private landlords need
# clarification before implementing their delta_v functions:
#   - tenant: is rent the zero baseline (ΔV_rent=0) or does V_outside differ?
#   - owner: does "rent" mean selling and becoming a tenant? what is V_outside for "stay"?
#   - landlord: outside option for buy-to-let = risk-free or best rental?
#   - landlord "occupy" action: too rare to model?

# --- Institutional investor ---


def delta_v_acquire(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
    cash: float,
    risk_free_rate: float,
) -> float:
    """Surplus from buying a property over investing the same cash at the risk-free rate.
    ΔV_acquire = E[Π_I] − r_f·cash"""
    ...


def delta_v_hold(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
    market_value: float,
    risk_free_rate: float,
) -> float:
    """Surplus from continuing to own a property over liquidating it at market value.
    ΔV_hold = E[Π_I] − r_f·market_value. Caller sums over the full portfolio to get V̄_hold."""
    ...


def delta_v_sell_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
    market_value: float,
    risk_free_rate: float,
) -> float:
    """Surplus from selling. Symmetric opposite of hold: sell is preferred when
    the property earns less than the risk-free rate.
    ΔV_sell = −ΔV_hold"""
    ...


# ---------------------------------------------------------------------------
# U(ΔV) — risk curvature
# ---------------------------------------------------------------------------


def crra(delta_v: float, gamma: float) -> float:
    """Applies risk aversion to a surplus value.
    U = sign(ΔV)·|ΔV|^(1−γ) / (1−γ).
    γ=0 returns ΔV unchanged (risk-neutral, used for institutions)."""
    ...


def apply_loss_aversion(
    sell_value: float,
    sale_price: float,
    purchase_anchor: float,
    loss_aversion: float,
) -> float:
    """If an agent sells a property for less than they originally paid, the perceived value of that sale is reduced by a penalty. This penalty is proportional to the loss.
    Institutions do not pay this penalty (λ=1).
    V̄^sell = sell_value − λ·max(p_0 − p, 0)."""
    ...


# ---------------------------------------------------------------------------
# Logit — action and property selection
# ---------------------------------------------------------------------------


def logit_choice(values: Mapping[Hashable, float], beta: float, rng) -> Hashable:
    """Picks one option probabilistically. Higher value = more likely to be chosen.
    Pr(k) = exp(β·V_k) / Σ exp(β·V_k'). Infeasible options get V = −inf → zero probability."""
    ...
