"""
Utility: the logit inputs, built from valuation.py's P&L.

The flow: value V (money) → surplus ΔV = V − V_outside → U(ΔV) → logit.

  - value_*       raw monetary value of an outcome (§5, §6)
  - crra          risk-aversion curvature U(ΔV) (§5, §8)
  - delta_v_*     ΔV per action, the logit inputs (§10)
  - logit_*       the choice mechanism; agent-agnostic (§10)

This module imports valuation.py, never the reverse. Agents enter here (the
value functions read agent state); they never enter valuation.py.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Mapping, Sequence

if TYPE_CHECKING:
    from agents import HouseholdAgent, InstitutionalAgent
    from properties import Property

    Agent = HouseholdAgent | InstitutionalAgent


@dataclass(frozen=True)
class DecisionContext:
    """The market snapshot one agent values all its actions against."""

    avg_market_rent: float
    purchase_candidates: Sequence["Property"]
    rental_candidates: Sequence["Property"]


# ---------------------------------------------------------------------------
# Value V (money)   (§5, §6)
# ---------------------------------------------------------------------------
# Value before risk curvature. ε (taste shock) is added once, in the
# logit, never both (§30).


def value_owner_occupier(quality_value: float, pnl: float) -> float:
    """V^OO = q_k + Π_H. `pnl` from valuation.pnl_owner_occupier."""
    ...


def value_landlord(pnl: float) -> float:
    """V^LL = Π_L. Quality enters only through rent."""
    ...


def value_institution(expected_pnl: float) -> float:
    """V^INST = E[Π_I]. Risk-neutral, so no curvature later (§8)."""
    ...


# Outside option: the alternative to buying, subtracted to form ΔV. Households
# can rent; investors can only hold cash — different baselines (§6, §11).


def household_outside_option(agent: "Agent", ctx: DecisionContext) -> float:
    """Household's alternative to buying: value of its best feasible rental."""
    ...


def investor_outside_option(agent: "Agent", ctx: DecisionContext) -> float:
    """Investor's alternative to buying: holding cash at the risk-free rate."""
    ...


# ---------------------------------------------------------------------------
# U(ΔV)   (§5, §8)
# ---------------------------------------------------------------------------


def crra(delta_v: float, gamma: float) -> float:
    """Signed CRRA: U = sign(ΔV)·|ΔV|^(1−γ)/(1−γ). ΔV may be negative (financing
    cost can exceed expected gain). γ=0 → identity (§8)."""
    ...


def apply_loss_aversion(
    sell_value: float,
    sale_price: float,
    purchase_anchor: float,
    loss_aversion: float,
) -> float:
    """V̄^sell = sell_value − λ·max(p_0 − p, 0) (§5). λ>1; institutions exempt."""
    ...


# ---------------------------------------------------------------------------
# ΔV per action — the logit inputs   (§10)
# ---------------------------------------------------------------------------
#


def rental_property_value(agent: "Agent", prop: "Property", ctx: DecisionContext) -> float:
    """Stage-2 scorer for rental candidates: quality consumption minus rent burden,
    no capital gain term. Feeds the logit that picks which rental to bid on."""
    ...


def property_value(agent: "Agent", prop: "Property", ctx: DecisionContext) -> float:
    """V_ik: this agent's value of property `prop` (§10 Stage 2). Scores the
    property logit; delta_v_buy maxes it over candidates."""
    ...


def delta_v_buy(agent: "Agent", ctx: DecisionContext) -> float:
    """Best buy value over feasible candidates, minus outside option.
    Owner-occupier value if buying a home, landlord value if buy-to-let.
    −inf if nothing is credit-feasible."""
    ...


def delta_v_rent(agent: "Agent", ctx: DecisionContext) -> float:
    """Value of the best feasible rental vs outside option (§10)."""
    ...


def delta_v_stay(agent: "Agent", ctx: DecisionContext) -> float:
    """Continuation value of holding the current home (owners, §10)."""
    ...


def delta_v_sell(agent: "Agent", prop: "Property", ctx: DecisionContext) -> float:
    """Value of selling `prop`, loss-aversion adjusted for households (§13)."""
    ...


def delta_v_let(agent: "Agent", prop: "Property", ctx: DecisionContext) -> float:
    """Value of letting `prop`: Π_L stream, ownership continued (§12)."""
    ...


def action_value(agent: "Agent", action: str, ctx: DecisionContext) -> float:
    """V̄^a: the right delta_v_* for `action`, curved by crra at the agent's γ.
    This is the scalar the Stage-1 logit consumes. −inf if infeasible (§10)."""
    ...


def feasible_actions(agent: "Agent", ctx: DecisionContext) -> list[str]:
    """The agent's action set A_i (§10): tenant {buy, rent}; owner adds stay,
    sell, let; institution {acquire, sell, hold}."""
    ...


# ---------------------------------------------------------------------------
# Logit (mechanism)   (§10)
# ---------------------------------------------------------------------------
# Reused for both action choice (Stage 1) and property choice (Stage 2). Sees
# only a value vector, never the agent. 


def logit_choice(values: Mapping[Hashable, float], beta: float, rng) -> Hashable:
    """Draw one key by multinomial logit: Pr(k) = exp(β·V_k) / Σ exp(β·V_k')."""
