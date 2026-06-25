"""
Agent definitions: households and institutional investors.
"""

from __future__ import annotations

import mesa
import valuation as val
from utility import risk_adjusted_growth, logit_choice, logit_probabilities


# ---------------------------------------------------------------------------
# Mixin: shared balance-sheet mechanics
# ---------------------------------------------------------------------------

class _BalanceSheetMixin:
    """
    Mortgage book and net-worth accounting shared by both agent classes.
    """

    def _init_balance_sheet(self):
        self._mortgages: dict[int, tuple[float, float, int, float]] = {}
        self._housing_asset_value: float = 0.0

    @property
    def gross_housing_assets(self) -> float:
        return self._housing_asset_value

    @property
    def mortgage_debt(self) -> float:
        credit = self.model.credit
        return float(
            sum(
                credit.outstanding_principal(orig, ltv, held, rate)
                for orig, ltv, held, rate in self._mortgages.values()
            )
        )

    @property
    def housing_equity(self) -> float:
        return self.gross_housing_assets - self.mortgage_debt

    @property
    def net_worth(self) -> float:
        return self.cash + self.housing_equity

    def mortgage_payment_due(self) -> float:
        credit = self.model.credit
        return float(
            sum(
                credit.monthly_mortgage_payment(orig, ltv, rate)
                for orig, ltv, _held, rate in self._mortgages.values()
            )
        )

    def service_mortgages(self):
        credit = self.model.credit
        updated = {}
        for pid, (orig, ltv, held, rate) in self._mortgages.items():
            self.cash -= credit.monthly_mortgage_payment(orig, ltv, rate)
            updated[pid] = (orig, ltv, held + 1, rate)
        self._mortgages = updated


# ---------------------------------------------------------------------------
# HouseholdAgent
# ---------------------------------------------------------------------------

class HouseholdAgent(_BalanceSheetMixin, mesa.Agent):
    """A household - owner-occupier / renter / landlord roles are derived from state."""

    def __init__(self, unique_id, model, income, cash, risk_aversion, home_zone):
        super().__init__(model)
        self.income = income
        self.baseline_income = income
        self.cash = cash
        self.risk_aversion = risk_aversion
        self.home_zone = home_zone
        self.owned_properties: set[int] = set()
        self.home_property: int | None = None  # owned OR rented, or None if unhoused
        self.rental_income_monthly: float = 0.0  # total rent received this step

        ecfg = model.config.expectations
        self.expected_price_growth = ecfg.init_price_growth
        self.expected_rent_growth = ecfg.init_rent_growth
        self.expected_price_vol = ecfg.init_price_vol
        self.expected_rent_vol = ecfg.init_rent_vol

        self._init_balance_sheet()

    # -- derived roles ----------------------------------------------------------

    @property
    def is_owner_occupier(self) -> bool:
        return self.home_property is not None and self.home_property in self.owned_properties

    @property
    def is_renter(self) -> bool:
        return self.home_property is None or self.home_property not in self.owned_properties

    @property
    def is_landlord(self) -> bool:
        rented_out = self.owned_properties - (
            {self.home_property} if self.home_property in self.owned_properties else set()
        )
        return len(rented_out) > 0

    @property
    def role(self) -> str:
        parts = []
        if self.is_owner_occupier:
            parts.append("owner-occupier")
        if self.is_renter and self.home_property is not None:
            parts.append("renter")
        if self.is_landlord:
            parts.append("landlord")
        return "+".join(parts) if parts else "unhoused"

    # -- expectations (set by the model; single source of truth) ----------------

    def set_expectations(self, price_growth, rent_growth, price_vol, rent_vol):
        self.expected_price_growth = price_growth
        self.expected_rent_growth = rent_growth
        self.expected_price_vol = price_vol
        self.expected_rent_vol = rent_vol

    def _risk_adjusted_price_growth(self) -> float:
        return risk_adjusted_growth(
            self.expected_price_growth, self.expected_price_vol, self.risk_aversion
        )

    def _risk_adjusted_rent_growth(self) -> float:
        return risk_adjusted_growth(
            self.expected_rent_growth, self.expected_rent_vol, self.risk_aversion
        )

    # -- valuation --------------------------------------------------------------

    def buy_wtp(self, prop) -> float:
        """Owner-occupier WTP for living in `prop` (risk-adjusted, credit-capped)."""
        cfg = self.model.config
        credit = self.model.credit
        total_income = self.income + self.rental_income_monthly * 12
        ceiling = credit.household_max_price(self.cash, total_income, self.mortgage_payment_due())
        return val.household_buy_wtp(
            prop.quality,
            cfg.valuation.quality_value_scale,
            cfg.valuation.base_housing_value,
            credit.mortgage_rate,
            self._risk_adjusted_price_growth(),
            prop.estimated_value,
            ceiling,
            cfg.valuation.horizon
        )

    def btl_wtp(self, prop, market_rent) -> float:
        """Buy-to-let WTP (risk-adjusted on price growth, credit-capped)."""
        cfg = self.model.config
        credit = self.model.credit
        total_income = self.income + self.rental_income_monthly * 12
        ceiling = credit.btl_max_price(self.cash, total_income, self.mortgage_payment_due())
        return val.household_btl_wtp(
            prop.quality,
            cfg.valuation.quality_sensitivity,
            market_rent,
            credit.btl_funding_rate,
            self._risk_adjusted_rent_growth(),
            self._risk_adjusted_price_growth(),
            prop.estimated_value,
            ceiling,
            cfg.valuation.horizon
        )

    def rent_wtp(self, prop) -> float:
        cfg = self.model.config
        return val.household_rent_wtp(
            prop.quality,
            cfg.valuation.quality_value_scale,
            cfg.valuation.base_housing_value,
            self.income,
            self.model.credit.dti_limit,
        )

    def hold_value(self, prop, market_rent) -> float:
        """
        Uncapped value of continuing to own prop (seller's outside option).
        """
        cfg = self.model.config
        credit = self.model.credit
        if prop.id == self.home_property:
            return val.household_buy_wtp(
                prop.quality, cfg.valuation.quality_value_scale,
                cfg.valuation.base_housing_value, credit.mortgage_rate,
                self._risk_adjusted_price_growth(), prop.estimated_value, float("inf"), cfg.valuation.horizon
            )
        return val.household_btl_wtp(
            prop.quality, cfg.valuation.quality_sensitivity, market_rent,
            credit.btl_funding_rate, self._risk_adjusted_rent_growth(),
            self._risk_adjusted_price_growth(),
            prop.estimated_value, float("inf"), cfg.valuation.horizon
        )

    def reservation_price(self, prop, market_rent) -> float:
        """Sale price p_res at which V_sell(p_res) == V_hold.

        V_hold is the uncapped valuation of keeping the property, V_sell(p) is the
        sale price net of the loss-aversion penalty against the purchase anchor:
            V_sell(p) = p - lambda * max(p0 - p, 0).
        """
        v_hold = self.hold_value(prop, market_rent)
        p0 = prop.purchase_anchor_price
        lam = self.model.config.agent_init.loss_aversion
        if v_hold <= p0:
            return v_hold
        return (v_hold + (lam - 1.0) * p0) / lam

    # -- Stage 1: action choice -------------------------------------------------

    def choose_action(self, purchase_candidates, market_rent):
        """Return {property_id: 'hold'|'sell'|'let', '__agent__': 'buy'|'buy-to-let'|'rent'|'none'}.

        Per-property and agent-level actions are each chosen by logit over the
        expected value of the feasible options. A no-transaction option ('hold'
        per property, 'none' at the agent level) is always available.
        """
        rng = self.model.rng
        result = {}
        credit = self.model.credit

        # Per-property: for each owned property decide hold / sell / let.
        for pid in list(self.owned_properties):
            prop = self.model._property_map[pid]
            v_hold = self.hold_value(prop, market_rent)
            # Sale surplus over holding (net of loss aversion at the reservation).
            p_res = self.reservation_price(prop, market_rent)
            v_sell = p_res - v_hold  # >= 0 by construction of p_res; ~0 at indiff.
            values = {"hold": 0.0, "sell": v_sell}
            if pid == self.home_property:
                # Letting one's home is only sensible if not living elsewhere; we
                # allow 'let' only for non-home (already-investment) properties.
                pass
            else:
                v_let = val.estimate_market_rent(
                    prop.quality, market_rent, self.model.config.valuation.quality_sensitivity
                )
                values["let"] = v_let
            result[pid] = logit_choice(values, rng)

        # Agent-level: buy / buy-to-let / rent / none.
        values = {"none": 0.0}

        # Outside option for buying = best feasible rental's consumption value.
        best_rent_value = 0.0
        if purchase_candidates:
            best_rent_value = max(
                (val.housing_consumption_value(
                    p.quality, self.model.config.valuation.quality_value_scale,
                    self.model.config.valuation.base_housing_value)
                 for p in purchase_candidates),
                default=0.0,
            )

        buy_ceiling = credit.household_max_price(self.cash, self.income + self.rental_income_monthly * 12, self.mortgage_payment_due())
        affordable_buy = [p for p in purchase_candidates if p.estimated_value <= buy_ceiling]
        if affordable_buy:
            # Surplus of best buy
            best_buy = max(
                (self.buy_wtp(p) - p.estimated_value for p in affordable_buy), default=0.0
            )
            values["buy"] = best_buy

        btl_ceiling = credit.btl_max_price(self.cash, self.income + self.rental_income_monthly * 12, self.mortgage_payment_due())
        affordable_btl = [p for p in purchase_candidates if p.estimated_value <= btl_ceiling]
        if affordable_btl:
            best_btl = max(
                (self.btl_wtp(p, market_rent) - p.estimated_value for p in affordable_btl),
                default=0.0,
            )
            values["buy-to-let"] = best_btl

        # Renting is an option for anyone not currently owner-occupying.
        if self.is_renter:
            # Value of renting ~ best available rental consumption value minus current rent burden
            values["rent"] = best_rent_value * 0.5

        result["__agent__"] = logit_choice(values, rng)
        return result

    # -- Stage 2: property selection --------------------------------------------

    def choose_property(self, candidates, purpose, market_rent):
        if not candidates:
            return None
        if purpose == "buy":
            scores = {p: self.buy_wtp(p) for p in candidates}
        elif purpose == "buy-to-let":
            scores = {p: self.btl_wtp(p, market_rent) for p in candidates}
        else:
            raise ValueError(f"Unknown purpose: {purpose!r}")
        return logit_choice(scores, self.model.rng)

    # -- Stage 3: bidding -------------------------------------------------------

    def compute_bid(self, prop, purpose="buy", market_rent=0.0) -> float:
        if purpose == "buy":
            return self.buy_wtp(prop)
        if purpose == "buy-to-let":
            return self.btl_wtp(prop, market_rent)
        raise ValueError(f"Unknown purpose: {purpose!r}")

    def compute_rent_bid(self, prop) -> float:
        return self.rent_wtp(prop)

    # -- balance-sheet transitions ----------------------------------------------

    def acquire_property(self, prop, price, purpose="buy"):
        credit = self.model.credit
        ltv = credit.origination_ltv(purpose)
        rate = credit.funding_rate(purpose)
        deposit = price * (1.0 - ltv)
        if self.cash < deposit - 1e-9:
            raise RuntimeError(
                f"Buyer {self.unique_id} cannot cover deposit {deposit:.2f} "
                f"(cash {self.cash:.2f}); feasibility check failed."
            )
        self.cash -= deposit
        self.owned_properties.add(prop.id)
        self._mortgages[prop.id] = (price, ltv, 0, rate)
        self._housing_asset_value += prop.estimated_value

        # Move in if buying to occupy and currently not owner-occupying.
        if purpose == "buy" and not self.is_owner_occupier:
            old_home = self.home_property
            if old_home is not None and old_home != prop.id and old_home not in self.owned_properties:
                old_prop = self.model._property_map.get(old_home)
                if old_prop is not None and old_prop.occupant_id == self.unique_id:
                    old_prop.occupant_id = None
                    old_prop.listed_for_rent = True
            self.home_property = prop.id
            self.home_zone = prop.zone
            prop.occupant_id = self.unique_id
            prop.current_rent = None
        elif purpose == "buy-to-let":
            prop.listed_for_rent = True

    def release_property(self, prop, sale_price):
        if prop.id not in self.owned_properties:
            return
        if prop.id in self._mortgages:
            orig, ltv, held, rate = self._mortgages[prop.id]
            outstanding = self.model.credit.outstanding_principal(orig, ltv, held, rate)
        else:
            outstanding = 0.0
        self.cash += sale_price - outstanding
        self.owned_properties.discard(prop.id)
        self._mortgages.pop(prop.id, None)
        self._housing_asset_value = max(0.0, self._housing_asset_value - prop.estimated_value)
        if self.home_property == prop.id:
            self.home_property = None

    def move_into_rental(self, prop):
        self.home_property = prop.id
        self.home_zone = prop.zone

    def vacate_rental(self):
        if self.home_property not in self.owned_properties:
            self.home_property = None

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent
        self.rental_income_monthly += monthly_rent

    def pay_rent(self, monthly_rent):
        self.cash -= monthly_rent

    def evolve_income(self, mu, sd):
        """Apply one month of multiplicative income growth (drawn by the model)."""
        import numpy as np
        self.income = float(self.income * np.exp(self.model.rng.normal(mu, sd)))

    def distress_sale_candidates(self, market_rent):
        """Owned properties ranked from least to greatest hold value (sell cheap-to-hold first)."""
        ranked = [
            (self.hold_value(self.model._property_map[pid], market_rent),
             self.model._property_map[pid])
            for pid in self.owned_properties
        ]
        return sorted(ranked, key=lambda item: item[0])

    def step(self):
        pass


# ---------------------------------------------------------------------------
# InstitutionalAgent
# ---------------------------------------------------------------------------

class InstitutionalAgent(_BalanceSheetMixin, mesa.Agent):
    """Risk-neutral investor: yield-based valuation, cheap funding, market-wide info."""

    def __init__(self, unique_id, model, cash):
        super().__init__(model)
        self.cash = cash
        self.portfolio: set[int] = set()  # alias maintained below for compatibility

        ecfg = model.config.expectations
        self.expected_price_growth = ecfg.init_price_growth
        self.expected_rent_growth = ecfg.init_rent_growth
        # Institutions are risk-neutral
        self.expected_price_vol = ecfg.init_price_vol
        self.expected_rent_vol = ecfg.init_rent_vol

        self._init_balance_sheet()

    # Institutions hold no income; expose 0 so shared affordability code is safe.
    income = 0.0

    @property
    def owned_properties(self):
        """Alias so model code can treat both classes uniformly."""
        return self.portfolio

    def set_expectations(self, price_growth, rent_growth, price_vol, rent_vol):
        self.expected_price_growth = price_growth
        self.expected_rent_growth = rent_growth
        self.expected_price_vol = price_vol
        self.expected_rent_vol = rent_vol

    # -- valuation (risk-neutral) -----------------------------------------------

    def acquire_wtp(self, prop, market_rent) -> float:
        cfg = self.model.config
        credit = self.model.credit
        ceiling = credit.institution_max_price(self.cash)
        return val.institutional_wtp(
            prop.quality,
            cfg.valuation.quality_sensitivity,
            market_rent,
            credit.inst_funding_rate,
            cfg.agent_init.inst_required_return,
            self.expected_rent_growth,
            self.expected_price_growth,
            prop.estimated_value,
            ceiling,
            cfg.valuation.horizon
        )

    def hold_value(self, prop, market_rent) -> float:
        cfg = self.model.config
        credit = self.model.credit
        return val.institutional_wtp(
            prop.quality, cfg.valuation.quality_sensitivity, market_rent,
            credit.inst_funding_rate, cfg.agent_init.inst_required_return,
            self.expected_price_growth, self.expected_rent_growth, prop.estimated_value, float("inf"), cfg.valuation.horizon
        )

    def reservation_price(self, prop, market_rent) -> float:
        """Institutions mark to fundamentals; reservation = uncapped hold value."""
        return self.hold_value(prop, market_rent)

    # -- decisions --------------------------------------------------------------

    def choose_action(self, purchase_candidates, market_rent):
        rng = self.model.rng
        credit = self.model.credit
        result = {}

        for pid in list(self.portfolio):
            prop = self.model._property_map[pid]
            v_hold = self.hold_value(prop, market_rent)
            p_res = self.reservation_price(prop, market_rent)
            values = {"hold": 0.0, "sell": p_res - v_hold}
            result[pid] = logit_choice(values, rng)

        ceiling = credit.institution_max_price(self.cash)
        feasible = [p for p in purchase_candidates if p.estimated_value <= ceiling]
        values = {"none": 0.0}
        if feasible:
            best = max((self.acquire_wtp(p, market_rent) - p.estimated_value for p in feasible),
                       default=0.0)
            values["acquire"] = best
        result["__agent__"] = logit_choice(values, rng)
        return result

    def choose_property(self, candidates, purpose, market_rent):
        if not candidates:
            return None
        scores = {p: self.acquire_wtp(p, market_rent) for p in candidates}
        return logit_choice(scores, self.model.rng)

    def compute_bid(self, prop, purpose="acquire", market_rent=0.0) -> float:
        return self.acquire_wtp(prop, market_rent)

    # -- balance sheet ----------------------------------------------------------

    def acquire_property(self, prop, price, purpose="acquire"):
        credit = self.model.credit
        ltv = credit.inst_ltv
        rate = credit.inst_funding_rate
        deposit = price * (1.0 - ltv)
        if self.cash < deposit - 1e-9:
            raise RuntimeError(
                f"Institution {self.unique_id} cannot cover deposit {deposit:.2f}."
            )
        self.cash -= deposit
        self.portfolio.add(prop.id)
        self._mortgages[prop.id] = (price, ltv, 0, rate)
        self._housing_asset_value += prop.estimated_value
        prop.listed_for_rent = True

    def release_property(self, prop, sale_price):
        if prop.id not in self.portfolio:
            return
        if prop.id in self._mortgages:
            orig, ltv, held, rate = self._mortgages[prop.id]
            outstanding = self.model.credit.outstanding_principal(orig, ltv, held, rate)
        else:
            outstanding = 0.0
        self.cash += sale_price - outstanding
        self.portfolio.discard(prop.id)
        self._mortgages.pop(prop.id, None)
        self._housing_asset_value = max(0.0, self._housing_asset_value - prop.estimated_value)

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent

    def distress_sale_candidates(self, market_rent):
        ranked = [
            (self.hold_value(self.model._property_map[pid], market_rent),
             self.model._property_map[pid])
            for pid in self.portfolio
        ]
        return sorted(ranked, key=lambda item: item[0])

    def step(self):
        pass


__all__ = ["HouseholdAgent", "InstitutionalAgent"]
