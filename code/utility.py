"""
Utility: converts P&L into logit inputs.
"""

import numpy as np
import math
from typing import Hashable, Mapping
from dataclasses import dataclass

def _pnl_owner_occupier(
    expected_capital_gain: float,
    mortgage_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Monthly P&L for an owner-occupier.
    Gain from expected price growth minus monthly mortgage cost.
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
    """
    return net_rent + expected_capital_gain - funding_rate * ltv * price


def value_owner_occupier(
    quality_value: float,
    expected_capital_gain: float,
    mortgage_rate: float,
    ltv: float,
    price: float,
) -> float:
    """Value of owning and living in a property.
    Quality adds direct consumption value on top of P&L.
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
    """
    return _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)

# Institutions use risk-free return as the outside option for every action.
# Households use their best feasible rental as the outside option.

def delta_v_acquire(
    net_rent: float,
    expected_capital_gain: float,
    funding_rate: float,
    ltv: float,
    price: float,
    risk_free_rate: float,
) -> float:
    """Surplus from buying a property over the risk-free return on the equity deployed.
    ΔV_acquire = E[Π_I] − r_f·(1−L)·p
    """
    pnl = _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)
    equity = (1.0 - ltv) * price
    return pnl - risk_free_rate * equity


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
<<<<<<< HEAD
=======
    Equity freed on sale = market_value − L·price (proceeds after loan paydown).
    ΔV_hold = E[Π_I] − r_f·(market_value − L·price)
>>>>>>> 35069bc8f4117bf27b5458021b25a7653b5596fc
    """
    pnl = _pnl_institution(net_rent, expected_capital_gain, funding_rate, ltv, price)
    proceeds = market_value - ltv * price
    if proceeds < 0.0:
        return pnl  # negative equity → sale crystallises a loss, so hold is always better
    return pnl - risk_free_rate * proceeds


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
    """
    return -delta_v_hold(net_rent, expected_capital_gain, funding_rate, ltv, price, market_value, risk_free_rate)


def crra(delta_v: float, gamma: float) -> float:
    """Applies risk aversion to a surplus value.
    """
    return crra_utility(delta_v, gamma)


def crra_utility(surplus: float, gamma: float) -> float:
    """CRRA utility of a surplus.
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
    """
    surplus = property_value - outside_option_value
    return crra_utility(surplus, gamma)


def institutional_action_value(expected_profit: float) -> float:
    """Utility of an action for an institution.
    """
    return expected_profit


def apply_loss_aversion(
    sell_value: float,
    sale_price: float,
    purchase_anchor: float,
    loss_aversion: float,
) -> float:
    """If a household sells a property for less than they originally paid, the
    perceived value of that sale is reduced by a penalty proportional to the loss.
    """
    penalty = loss_aversion * max(purchase_anchor - sale_price, 0.0)
    return sell_value - penalty


def risk_adjusted_growth(
    expected_growth: float,
    expected_volatility: float,
    risk_loading: float,
) -> float:
    """Reduced form household risk
    """
    values = {
        "expected_growth": expected_growth,
        "expected_volatility": expected_volatility,
        "risk_loading": risk_loading,
    }

    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if expected_volatility < 0.0:
        raise ValueError("expected_volatility must be non-negative")

    if risk_loading < 0.0:
        raise ValueError("risk_loading must be non-negative")

    return expected_growth - risk_loading * expected_volatility

@dataclass
class DecisionContext:
    """Context passed to property_value for property-level evaluation."""
    purchase_candidates: list
    rental_candidates: tuple = ()


def property_value(agent, prop, ctx: DecisionContext) -> float:
    """Evaluate a property for an agent. Delegates to the agent's WTP method."""
    return agent._wtp_for_property(prop)


def logit_choice(values: Mapping[Hashable, float], rng, beta: float = 1.0) -> Hashable:
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
