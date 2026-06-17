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

import numpy as np
from typing import Hashable, Mapping
from dataclasses import dataclass

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
    and cancels in the surplus comparison.
    """
    return expected_capital_gain - mortgage_rate * ltv * price


def _pnl_landlord(
    net_rent: float,
    expected_capital_gain: float,
    btl_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for a private landlord.
    Rental income plus expected price growth minus mortgage cost.
    Π_L = R − φ − r_f^BTL·L·p + E[Δp]
    """
    return net_rent + expected_capital_gain - btl_rate * ltv * price


def _pnl_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for an institutional investor.
    Same structure as landlord but cheaper funding rate.
    Π_I = R − φ − r_f·L·p + E[Δp]
    """
    return net_rent + expected_capital_gain - funding_rate * ltv * price


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
    V^OO = q_k + Π_H
    """
    return quality_value + _pnl_owner_occupier(expected_capital_gain, mortgage_rate, ltv, price)


def value_landlord(
    net_rent: float,
    expected_capital_gain: float,
    btl_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of owning and renting out a property.
    Quality does not enter directly — only through the rent R.
    V^LL = Π_L
    """
    return _pnl_landlord(net_rent, expected_capital_gain, btl_rate, ltv, price)


def value_institution(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of a property for an institutional investor.
    Risk-neutral: value equals expected P&L, no curvature applied.
    V^INST = E[Π_I]
    """
    return _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)


# ---------------------------------------------------------------------------
# ΔV per action — the logit inputs  (Stage 1)
# ---------------------------------------------------------------------------
# ΔV = V(action) − V(outside option).
# Institutions use risk-free return as the outside option for every action.
# Households use their best feasible rental as the outside option.

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
    ΔV_acquire = E[Π_I] − r_f·cash
    = (net_rent + E[Δp] − r_f·L·p) − r_f·cash
    """
    pnl = _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)
    return pnl - risk_free_rate * cash


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
    ΔV_hold = E[Π_I] − r_f·market_value
    Caller sums over the full portfolio to get V̄_hold.
    """
    pnl = _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)
    return pnl - risk_free_rate * market_value


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
    ΔV_sell = −ΔV_hold
    """
    return -delta_v_hold(net_rent, expected_capital_gain, funding_rate, ltv, price, market_value, risk_free_rate)


# ---------------------------------------------------------------------------
# U(ΔV) — risk curvature
# ---------------------------------------------------------------------------


def crra(delta_v: float, gamma: float) -> float:
    """Applies risk aversion to a surplus value.
    U = sign(ΔV)·|ΔV|^(1−γ) / (1−γ).
    γ=0 returns ΔV unchanged (risk-neutral, used for institutions).
    Returns -inf for non-positive surplus (infeasible action).
    """
    return crra_utility(delta_v, gamma)


def crra_utility(surplus: float, gamma: float) -> float:
    """CRRA utility of a surplus.

    U(ΔV) = (ΔV)^(1-γ) / (1-γ)   for γ ≠ 1
    U(ΔV) = ln(ΔV)                for γ = 1

    Returns -inf for non-positive surplus (infeasible action).
    """
    if surplus <= 0.0:
        return -np.inf
    if gamma == 0.0:
        return surplus
    if abs(gamma - 1.0) < 1e-12:
        return float(np.log(surplus))
    return float(surplus ** (1.0 - gamma)) / (1.0 - gamma)


def household_action_value(
    property_value: float,
    outside_option_value: float,
    gamma: float,
) -> float:
    """Utility of an action for a household.

    Computes U(ΔV) where ΔV = property_value - outside_option_value.
    """
    surplus = property_value - outside_option_value
    return crra_utility(surplus, gamma)


def institutional_action_value(expected_profit: float) -> float:
    """Utility of an action for an institution.

    Institutions are risk-neutral: U = E[Π].
    """
    return expected_profit


def apply_loss_aversion(
    sell_value: float,
    sale_price: float,
    purchase_anchor: float,
    loss_aversion: float,
) -> float:
    """If an agent sells a property for less than they originally paid, the
    perceived value of that sale is reduced by a penalty proportional to the loss.
    Institutions do not pay this penalty (λ=1).
    V̄^sell = sell_value − λ·max(p_0 − p, 0).
    """
    penalty = loss_aversion * max(purchase_anchor - sale_price, 0.0)
    return sell_value - penalty


# ---------------------------------------------------------------------------
# Logit — action and property selection
# ---------------------------------------------------------------------------


@dataclass
class DecisionContext:
    """Context passed to property_value for property-level evaluation."""
    avg_market_rent: float
    purchase_candidates: list
    rental_candidates: tuple = ()


def property_value(agent, prop, ctx: DecisionContext) -> float:
    """Evaluate a property for an agent. Delegates to the agent's WTP method."""
    return agent._wtp_for_property(prop, ctx.avg_market_rent)


def logit_choice(values: Mapping[Hashable, float], beta: float, rng) -> Hashable:
    """Picks one option probabilistically. Higher value = more likely to be chosen.
    Pr(k) = exp(β·V_k) / Σ exp(β·V_k'). Infeasible options get V = −inf → zero probability.
    """
    labels = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)

    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        probs = np.zeros(len(vals))
    else:
        shifted = np.where(finite_mask, vals * beta - vals[finite_mask].max() * beta, -np.inf)
        exp_v = np.where(finite_mask, np.exp(np.clip(shifted, -500, 0)), 0.0)
        total = exp_v.sum()
        probs = exp_v / total if total > 0 else np.zeros(len(vals))

    return rng.choice(list(labels), p=list(probs))
