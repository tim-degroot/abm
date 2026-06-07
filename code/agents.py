"""
Agent definitions.

  Tenant, Owner-Occupier, Landlord are NOT agent classes.
  They are economic roles derived from agent state.

  home_property : the property id where this household *resides*
                  (whether owned or rented). None = unhoused.

  owned_properties : set of property ids this household owns.

  Derived roles:
    is_owner_occupier : home_property in owned_properties
    is_renter         : housed but home_property not owned
                        (or unhoused — seeking rental)
    is_landlord       : owns properties other than home_property

  Valid simultaneous role combinations:
    renter                  - lives in rented property, owns nothing
    owner-occupier          - lives in owned property, no other holdings
    owner-occupier+landlord - lives in owned home, also owns rentals
    renter+landlord         - rents their own home, owns investment properties
    unhoused                - between homes (transitional state only)

  Mortgage tracking:
    _mortgages : dict { property_id -> (purchase_price, ltv, steps_held) }
    Each step, steps_held increments for all owned properties.
    On sale, outstanding_principal is computed from this record.

  Income dynamics:
    Income evolves each period via a log-normal shock, mean-reverting
    to the agent's baseline_income. This drives affordability changes
    that cause renters to eventually buy and owners to default or sell.
"""

import numpy as np
import mesa
from expectations import adaptive_update, init_price_expectation, init_rent_expectation
from valuation import (
    household_wtp,
    investor_wtp,
    household_max_rent,
    estimate_market_rent,
)

# All tunable parameters now live in config.py / config.toml and are read via
# self.model.config (e.g. self.model.config.agent.beta_action). Nothing tunable
# is hardcoded in this module.


def _logit_probs(scores):
    """
    Compute logit probabilities from a list of (label, score) pairs.

    Scores of -inf receive zero probability.
    Returns list of (label, probability) in same order.
    """
    labels = [s[0] for s in scores]
    values = np.array([s[1] for s in scores], dtype=float)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        probs = np.zeros(len(values))
    else:
        shifted = np.where(finite_mask, values - values[finite_mask].max(), -np.inf)
        exp_v = np.where(finite_mask, np.exp(np.clip(shifted, -500, 0)), 0.0)
        total = exp_v.sum()
        probs = exp_v / total if total > 0 else np.zeros(len(values))

    return list(zip(labels, probs))


# ---------------------------------------------------------------------------
# HouseholdAgent
# ---------------------------------------------------------------------------


class HouseholdAgent(mesa.Agent):
    """
    A household. Role is entirely derived from ownership and occupancy state.

    home_property   : id of property where household currently lives, or None
    owned_properties: set of property ids owned by this household
    _mortgages      : {prop_id: (purchase_price, ltv, steps_held)}
    """

    def __init__(
        self,
        unique_id,
        model,
        income,
        cash,
        risk_aversion,
        home_zone,
        expected_price_growth=None,
        expected_rent_growth=None,
    ):
        super().__init__(model)

        self.income = income
        self.baseline_income = income
        self.cash = cash
        self.risk_aversion = risk_aversion
        self.home_zone = home_zone  # zone of current residence

        self.owned_properties = set()
        self.home_property = None  # where agent lives (owned OR rented), or None

        ecfg = self.model.config.expectations
        self.expected_price_growth = (
            expected_price_growth
            if expected_price_growth is not None
            else init_price_expectation(ecfg.init_price_growth)
        )
        self.expected_rent_growth = (
            expected_rent_growth
            if expected_rent_growth is not None
            else init_rent_expectation(ecfg.init_rent_growth)
        )

        # Mortgage tracking: prop_id -> (purchase_price, ltv_at_origination, steps_held)
        self._mortgages = {}

        # Mark-to-market housing asset value (updated by model each step)
        self._housing_asset_value = 0.0

    # ------------------------------------------------------------------
    # Derived roles
    # ------------------------------------------------------------------

    @property
    def is_owner_occupier(self):
        """Lives in a property they own."""
        return (
            self.home_property is not None
            and self.home_property in self.owned_properties
        )

    @property
    def is_renter(self):
        """
        Lives in a rented property (home_property set but not owned),
        or is unhoused and seeking rental.
        """
        return (
            self.home_property is None
            or self.home_property not in self.owned_properties
        )

    @property
    def is_landlord(self):
        """Owns at least one property they do not live in."""
        rented_out = self.owned_properties - (
            {self.home_property}
            if self.home_property in self.owned_properties
            else set()
        )
        return len(rented_out) > 0

    @property
    def role(self):
        """Human-readable combined role string."""
        parts = []
        if self.is_owner_occupier:
            parts.append("owner-occupier")
        if self.is_renter and self.home_property is not None:
            parts.append("renter")
        if self.is_landlord:
            parts.append("landlord")
        if not parts:
            return "unhoused"
        return "+".join(parts)

    @property
    def net_worth(self):
        """Cash plus mark-to-market housing asset value."""
        return self.cash + self._housing_asset_value

    # ------------------------------------------------------------------
    # Stage 1: Action selection
    # ------------------------------------------------------------------

    def choose_action(self, purchase_candidates, avg_market_rent):
        """
        Choose action via multinomial logit.

        Returns one of: 'buy', 'rent', 'hold', 'sell', 'rent_out'
        """
        credit = self.model.credit
        acfg = self.model.config.agent
        scores = []

        # BUY — only if credit-feasible candidates exist
        affordable = [
            p
            for p in purchase_candidates
            if credit.is_feasible(p.estimated_value, self.cash, self.income)
        ]
        if affordable:
            best_wtp = max(
                self._wtp_for_property(p, avg_market_rent, credit) for p in affordable
            )
            scores.append(("buy", acfg.beta_action * best_wtp))
        else:
            scores.append(("buy", -np.inf))

        # RENT — always available; scored by rent burden
        monthly_burden = avg_market_rent / max(self.income / 12.0, 1.0)
        rent_score = acfg.beta_action * (-monthly_burden)
        scores.append(("rent", rent_score))

        # HOLD / SELL / RENT_OUT — only for owners
        if self.owned_properties:
            hold_score = acfg.beta_action * self.expected_price_growth
            scores.append(("hold", hold_score))

            sell_score = acfg.beta_action * (
                -self.expected_price_growth + acfg.sell_score_offset
            )
            scores.append(("sell", sell_score))

        # RENT_OUT home — only if owner-occupier (move out, become renter+landlord)
        if self.is_owner_occupier:
            rent_out_score = acfg.beta_action * (
                avg_market_rent
                * 12
                / max(self.model._property_map[self.home_property].estimated_value, 1.0)
                + self.expected_rent_growth
            )
            scores.append(("rent_out", rent_out_score))

        probs = _logit_probs(scores)
        actions, weights = zip(*probs)
        return self.model.random.choices(list(actions), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 2: Property selection
    # ------------------------------------------------------------------

    def choose_property(self, candidates, avg_market_rent):
        """Select among feasible candidates via logit on WTP."""
        if not candidates:
            return None
        credit = self.model.credit
        beta_property = self.model.config.agent.beta_property
        scores = [
            (p, beta_property * self._wtp_for_property(p, avg_market_rent, credit))
            for p in candidates
        ]
        probs = _logit_probs(scores)
        props, weights = zip(*probs)
        return self.model.random.choices(list(props), weights=list(weights), k=1)[0]

    def choose_rental(self, rental_candidates):
        """Select among available rentals. Prefers lower rent relative to income."""
        if not rental_candidates:
            return None
        beta_property = self.model.config.agent.beta_property
        scores = [
            (p, beta_property * (-p.estimated_value / max(self.income / 12.0, 1.0)))
            for p in rental_candidates
        ]
        probs = _logit_probs(scores)
        props, weights = zip(*probs)
        return self.model.random.choices(list(props), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 3: Bid formation
    # ------------------------------------------------------------------

    def compute_bid(self, prop, avg_market_rent):
        """Truthful WTP bid for ownership."""
        return self._wtp_for_property(prop, avg_market_rent, self.model.credit)

    def compute_rent_bid(self):
        """Maximum monthly rent bid (affordability ceiling)."""
        return household_max_rent(
            self.income, self.model.config.valuation.rent_income_fraction
        )

    # ------------------------------------------------------------------
    # Balance sheet updates
    # ------------------------------------------------------------------

    def acquire_property(self, prop, price, origination_ltv=None):
        """
        Called by model when this household wins an ownership auction.

        Deducts deposit from cash, records mortgage, moves in if unhoused.
        """
        # Use supplied origination LTV if provided; otherwise use the current
        # credit environment LTV cap — the loan will be originated at the
        # applicable LTV limit if the purchase is feasible.
        if origination_ltv is None:
            ltv = self.model.credit.ltv_limit
        else:
            ltv = origination_ltv

        deposit = price * (1.0 - ltv)

        # At this point bids should have been credit-checked; assert feasibility
        if self.cash < deposit:
            raise RuntimeError(
                f"Buyer {self.unique_id} cannot cover deposit {deposit:.2f}; "
                "bid/feasibility logic failed."
            )

        self.cash -= deposit
        self.owned_properties.add(prop.id)
        self._mortgages[prop.id] = (price, ltv, 0)  # (purchase_price, ltv, steps_held)
        self._housing_asset_value += prop.estimated_value

        # Move in if currently unhoused or renting
        if not self.is_owner_occupier:
            # Leave current rental if any (occupant_id cleared by model)
            self.home_property = prop.id
            self.home_zone = prop.zone
            prop.occupant_id = self.unique_id
            prop.current_rent = None

    def release_property(self, prop, sale_price):
        """
        Called by model when this household sells a property.

        Credits net proceeds (sale_price - outstanding_mortgage) to cash.
        Negative equity is possible: cash decreases if underwater.
        """
        if prop.id not in self.owned_properties:
            return

        # Compute outstanding mortgage balance
        if prop.id in self._mortgages:
            orig_price, ltv, steps_held = self._mortgages[prop.id]
            outstanding = self.model.credit.outstanding_principal(
                orig_price, ltv, steps_held
            )
        else:
            outstanding = 0.0

        net_proceeds = sale_price - outstanding
        self.cash += net_proceeds

        self.owned_properties.discard(prop.id)
        self._mortgages.pop(prop.id, None)
        self._housing_asset_value = max(
            0.0, self._housing_asset_value - prop.estimated_value
        )

        # If sold home, become renter (unhoused until rental clears)
        if self.home_property == prop.id:
            self.home_property = None

    def move_into_rental(self, prop):
        """
        Called by model when this household wins a rental auction.
        Updates home_property and home_zone.
        """
        # If vacating a previously rented property, that is handled by model
        self.home_property = prop.id
        self.home_zone = prop.zone

    def vacate_rental(self):
        """
        Called by model when a renter must vacate (landlord selling, etc.).
        Household becomes unhoused until next rental market clears.
        """
        if self.home_property not in self.owned_properties:
            self.home_property = None

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent

    def pay_rent(self, monthly_rent):
        self.cash -= monthly_rent

    def service_mortgages(self):
        """
        Deduct one period of mortgage payments for all owned properties.
        Called each step by the model.

        Increments steps_held on each mortgage record.
        Payment = annual_mortgage_payment / 1 (we treat each step as one year).
        """
        credit = self.model.credit
        updated = {}
        for pid, (orig_price, ltv, steps_held) in self._mortgages.items():
            payment = credit.annual_mortgage_payment(orig_price, ltv)
            self.cash -= payment
            updated[pid] = (orig_price, ltv, steps_held + 1)
        self._mortgages = updated

    def mortgage_payment_due(self):
        """Total mortgage servicing due this period."""
        credit = self.model.credit
        return float(
            sum(
                credit.annual_mortgage_payment(orig_price, ltv)
                for orig_price, ltv, _ in self._mortgages.values()
            )
        )

    def distress_sale_candidates(self, avg_market_rent):
        """Rank owned properties from least to greatest expected utility loss."""
        credit = self.model.credit
        ranked = [
            (
                self._wtp_for_property(
                    self.model._property_map[pid], avg_market_rent, credit
                ),
                self.model._property_map[pid],
            )
            for pid in self.owned_properties
        ]
        return sorted(ranked, key=lambda item: item[0])

    # ------------------------------------------------------------------
    # Income dynamics
    # ------------------------------------------------------------------

    def evolve_income(self):
        """
        Apply one period of income evolution.

        Apply multiplicative income growth draws each period (no mean
        reversion) per plan: income_t+1 = income_t * exp(shock).
        """
        acfg = self.model.config.agent
        shock = self.model.rng.normal(0.0, acfg.income_shock_sd)
        self.income = float(self.income * np.exp(shock))

    # ------------------------------------------------------------------
    # Expectation update
    # ------------------------------------------------------------------

    def update_expectations(self, price_signal, rent_signal, delta=None):
        d = delta if delta is not None else self.model.config.expectations.delta
        noise_sd = self.model.config.expectations.noise_sd
        self.expected_price_growth = adaptive_update(
            self.expected_price_growth, price_signal, d
        )
        self.expected_rent_growth = adaptive_update(
            self.expected_rent_growth, rent_signal, d
        )
        if noise_sd > 0.0:
            self.expected_price_growth += float(self.model.rng.normal(0.0, noise_sd))
            self.expected_rent_growth += float(self.model.rng.normal(0.0, noise_sd))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_market_rent, credit):
        cfg = self.model.config
        # Exogenous expected capital gain (£).
        # Use the market-level recent average price as the base for expected
        # capital gains rather than the per-property `estimated_value` to avoid
        # positive feedback where a single high transaction inflates one
        # property's value and hence everyone's WTP for it.
        market_price = (
            self.model._price_history[-1]
            if self.model._price_history
            else prop.estimated_value
        )
        capital_gain = self.expected_price_growth * market_price
        annual_rent = (
            estimate_market_rent(
                prop.quality, avg_market_rent, cfg.valuation.quality_sensitivity
            )
            * 12.0
        )
        if self.is_owner_occupier:
            # Already housed → an extra purchase is buy-to-let (landlord, plan §11).
            net_rent = annual_rent * (1.0 - cfg.valuation.operating_cost_fraction)
            wtp = investor_wtp(
                net_rent, capital_gain, cfg.credit.btl_funding_rate, cfg.credit.btl_ltv
            )
            # Private landlords are still credit-constrained (unlike institutions).
            return min(wtp, credit.max_affordable_price(self.cash, self.income))

        # Owner-occupier purchase: value = quality consumption + capital gain.
        quality_value = cfg.valuation.quality_value_scale * annual_rent

        # Outside option: value of the best available rental alternative in the
        # agent's searchable zones (annualised). If none available, fall back to
        # the current average market rent.
        zones = self.model.get_search_zones(self.home_zone)
        rental_props = [
            p
            for p in self.model.properties
            if p.zone in zones and p.id != self.home_property
        ]
        best_rent_ann = 0.0
        for rp in rental_props:
            ann = (
                estimate_market_rent(
                    rp.quality, avg_market_rent, cfg.valuation.quality_sensitivity
                )
                * 12.0
            )
            if ann > best_rent_ann:
                best_rent_ann = ann

        if best_rent_ann <= 0.0:
            best_rent_ann = avg_market_rent * 12.0

        outside_option = cfg.valuation.quality_value_scale * best_rent_ann

        return household_wtp(
            quality_value,
            capital_gain,
            outside_option,
            credit.mortgage_rate,
            credit.ltv_limit,
            credit.max_affordable_price(self.cash, self.income),
        )

    def step(self):
        """Mesa step hook. Orchestrated by model."""
        pass


# ---------------------------------------------------------------------------
# InstitutionalAgent
# ---------------------------------------------------------------------------


class InstitutionalAgent(mesa.Agent):
    """
    Institutional investor. Never occupies housing.

    Yield-based valuation, lower funding cost, effectively unconstrained.
    The only fundamentally distinct agent class.
    """

    def __init__(
        self,
        unique_id,
        model,
        cash,
        funding_rate,
        home_zone,
        expected_price_growth=None,
        expected_rent_growth=None,
    ):
        super().__init__(model)

        self.cash = cash
        self.funding_rate = funding_rate
        self.home_zone = home_zone

        self.portfolio = set()
        self._mortgages = {}

        ecfg = self.model.config.expectations
        self.expected_price_growth = (
            expected_price_growth
            if expected_price_growth is not None
            else init_price_expectation(ecfg.init_price_growth)
        )
        self.expected_rent_growth = (
            expected_rent_growth
            if expected_rent_growth is not None
            else init_rent_expectation(ecfg.init_rent_growth)
        )

        self._housing_asset_value = 0.0

    @property
    def net_worth(self):
        debt = 0.0
        credit = self.model.credit
        for orig_price, ltv, steps_held in self._mortgages.values():
            debt += credit.outstanding_principal(orig_price, ltv, steps_held)
        return self.cash + self._housing_asset_value - debt

    # ------------------------------------------------------------------
    # Stage 1: Action selection
    # ------------------------------------------------------------------

    def choose_action(self, purchase_candidates, avg_rent):
        acfg = self.model.config.agent
        scores = []

        if purchase_candidates:
            best_wtp = max(
                self._wtp_for_property(p, avg_rent) for p in purchase_candidates
            )
            scores.append(("buy", acfg.beta_action * best_wtp))
        else:
            scores.append(("buy", -np.inf))

        scores.append(("hold", acfg.beta_action * self.expected_price_growth))
        scores.append(
            (
                "sell",
                acfg.beta_action
                * (-self.expected_price_growth + acfg.inst_sell_score_offset),
            )
        )

        probs = _logit_probs(scores)
        actions, weights = zip(*probs)
        return self.model.random.choices(list(actions), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 2: Property selection
    # ------------------------------------------------------------------

    def choose_property(self, candidates, avg_rent):
        if not candidates:
            return None
        beta_property = self.model.config.agent.beta_property
        scores = [
            (p, beta_property * self._wtp_for_property(p, avg_rent)) for p in candidates
        ]
        probs = _logit_probs(scores)
        props, weights = zip(*probs)
        return self.model.random.choices(list(props), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 3: Bid formation
    # ------------------------------------------------------------------

    def compute_bid(self, prop, avg_rent):
        return self._wtp_for_property(prop, avg_rent)

    # ------------------------------------------------------------------
    # Balance sheet
    # ------------------------------------------------------------------

    def acquire_property(self, prop, price, origination_ltv=None):
        if origination_ltv is None:
            ltv = self.model.config.agent.inst_ltv
        else:
            ltv = origination_ltv

        deposit = price * (1.0 - ltv)
        if self.cash < deposit:
            raise RuntimeError(
                f"Institution {self.unique_id} cannot cover deposit {deposit:.2f}; bid/financing logic failed."
            )

        self.cash -= deposit
        self.portfolio.add(prop.id)
        self._housing_asset_value += prop.estimated_value
        self._mortgages[prop.id] = (price, ltv, 0)

    def release_property(self, prop, sale_price):
        self.portfolio.discard(prop.id)
        if prop.id in self._mortgages:
            orig_price, ltv, steps_held = self._mortgages[prop.id]
            outstanding = self.model.credit.outstanding_principal(
                orig_price, ltv, steps_held
            )
        else:
            outstanding = 0.0

        self.cash += sale_price - outstanding
        self._housing_asset_value = max(
            0.0, self._housing_asset_value - prop.estimated_value
        )
        self._mortgages.pop(prop.id, None)

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent

    def service_mortgages(self):
        credit = self.model.credit
        updated = {}
        for pid, (orig_price, ltv, steps_held) in self._mortgages.items():
            payment = credit.annual_mortgage_payment(orig_price, ltv)
            self.cash -= payment
            updated[pid] = (orig_price, ltv, steps_held + 1)
        self._mortgages = updated

    def mortgage_payment_due(self):
        """Total mortgage servicing due this period."""
        credit = self.model.credit
        return float(
            sum(
                credit.annual_mortgage_payment(orig_price, ltv)
                for orig_price, ltv, _ in self._mortgages.values()
            )
        )

    def distress_sale_candidates(self, avg_rent):
        """Rank owned properties from least to greatest expected utility loss."""
        ranked = [
            (
                self._wtp_for_property(self.model._property_map[pid], avg_rent),
                self.model._property_map[pid],
            )
            for pid in self.portfolio
        ]
        return sorted(ranked, key=lambda item: item[0])

    # ------------------------------------------------------------------
    # Expectation update
    # ------------------------------------------------------------------

    def update_expectations(self, price_signal, rent_signal, delta=None):
        d = delta if delta is not None else self.model.config.expectations.delta
        noise_sd = self.model.config.expectations.noise_sd
        self.expected_price_growth = adaptive_update(
            self.expected_price_growth, price_signal, d
        )
        self.expected_rent_growth = adaptive_update(
            self.expected_rent_growth, rent_signal, d
        )
        if noise_sd > 0.0:
            self.expected_price_growth += float(self.model.rng.normal(0.0, noise_sd))
            self.expected_rent_growth += float(self.model.rng.normal(0.0, noise_sd))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_rent):
        cfg = self.model.config
        # Use market-average price for expected capital gains (see household
        # comment above) to avoid property-specific positive feedback loops.
        market_price = (
            self.model._price_history[-1]
            if self.model._price_history
            else prop.estimated_value
        )
        capital_gain = self.expected_price_growth * market_price
        net_rent = (
            estimate_market_rent(
                prop.quality, avg_rent, cfg.valuation.quality_sensitivity
            )
            * 12.0
            * (1.0 - cfg.valuation.operating_cost_fraction)
        )
        # Discount expected capital gains using funding_rate PLUS an
        # institutional required return (risk premium) so low funding costs do
        # not by themselves generate implausibly large WTPs.
        effective_rate = float(self.funding_rate + cfg.agent.inst_required_return)
        wtp = investor_wtp(net_rent, capital_gain, effective_rate, cfg.agent.inst_ltv)
        # Prevent unbounded institutional bids by capping at a cash-derived
        # affordability ceiling: assume institutions can leverage up to
        # `inst_ltv` on purchases so max_price = cash / (1 - inst_ltv).
        inst_ltv = cfg.agent.inst_ltv
        if inst_ltv >= 1.0:
            return wtp
        max_price = self.cash / max(1e-9, (1.0 - inst_ltv))
        return min(wtp, max_price)

    def step(self):
        pass
