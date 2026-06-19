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

  Mortgage tracking:
    _mortgages : dict { property_id -> (purchase_price, ltv, steps_held) }
    Each step, steps_held increments for all owned properties.
    On sale, outstanding_principal is computed from this record.

  Income dynamics:
    Income evolves each period via a log-normal shock, mean-reverting
    to the agent's baseline_income. This drives affordability changes
    that cause renters to eventually buy and owners to default or sell.
"""

# NO UNHOUSED

import numpy as np
import mesa
from expectations import (
    adaptive_update,
    init_price_expectation,
    init_rent_expectation,
    price_growth_signal,
    rent_growth_signal,
    institutional_price_forecast,
    institutional_rent_growth_signal,
)
from valuation import (
    household_wtp,
    investor_wtp,
    household_max_rent,
    estimate_market_rent,
)
import utility
from utility import household_action_value, institutional_action_value


def _logit_probs(scores):  # We can remove it
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
        return self.home_property is not None and self.home_property in self.owned_properties

    @property
    def is_renter(self):
        """
        Lives in a rented property (home_property set but not owned),
        or is unhoused and seeking rental.
        """
        return self.home_property is None or self.home_property not in self.owned_properties

    @property
    def is_landlord(self):
        """Owns at least one property they do not live in."""
        rented_out = self.owned_properties - (
            {self.home_property} if self.home_property in self.owned_properties else set()
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
    def gross_housing_assets(self):
        # Market value of the houses the household owns.
        return self._housing_asset_value

    @property
    def mortgage_debt(self):
        # household wealth = cash + house value - remaining loan balance
        credit = self.model.credit
        return float(
            sum(
                credit.outstanding_principal(orig_price, ltv, steps_held)
                for orig_price, ltv, steps_held in self._mortgages.values()
            )
        )

    @property
    def housing_equity(self):
        return self.gross_housing_assets - self.mortgage_debt

    @property
    def net_worth(self):
        return self.cash + self.housing_equity

    # ------------------------------------------------------------------
    # Stage 1: Action selection
    # ------------------------------------------------------------------

    def choose_action(
        self, purchase_candidates, avg_market_rent
    ):  # NONE OF THE REPRESENTATIVE UTILITIES FOLLOW THE PLAN
        """
        Choose action via multinomial logit.

        Returns one of: 'buy', 'buy-to-let', 'rent', 'hold', 'sell', 'rent_out'
        """
        credit = self.model.credit
        scores = []

        # BUY (primary residence) — only if credit-feasible candidates exist
        affordable = [
            p
            for p in purchase_candidates
            if credit.max_affordable_price(self.cash, self.income) >= p.estimated_value
        ]
        if affordable:
            best_wtp = max(
                self._wtp_for_property(p, avg_market_rent, credit, purpose="buy")
                for p in affordable  # SHOULD BE EXPECTATIONS NOT MKT AVG
            )
            scores.append(("buy", best_wtp))
        else:
            scores.append(("buy", -np.inf))

        # BUY-TO-LET (investment) — same candidate pool, investor WTP
        if affordable:
            best_btl_wtp = max(
                self._wtp_for_property(p, avg_market_rent, credit, purpose="buy-to-let")
                for p in affordable
            )
            scores.append(("buy-to-let", best_btl_wtp))
        else:
            scores.append(("buy-to-let", -np.inf))

        # RENT — score depends on housing status
        if self.home_property is not None and self.is_renter:
            # Housed renter: compare current rent to market avg
            current_prop = self.model._property_map.get(self.home_property)
            if current_prop is not None and current_prop.current_rent is not None:
                current_burden = current_prop.current_rent / max(self.income, 1.0)
                market_burden = avg_market_rent / max(self.income, 1.0)
                savings = current_burden - market_burden
                moving_cost = 0.05
                rent_score = savings - moving_cost
            else:
                rent_score = -avg_market_rent / max(self.income, 1.0)
        else:
            # Unhoused or owner-occupier: score reflects affordability
            monthly_burden = avg_market_rent / max(self.income, 1.0)
            rent_score = -monthly_burden
        scores.append(("rent", rent_score))

        # HOLD / SELL / RENT_OUT — only for owners
        if self.owned_properties:
            hold_score = self.expected_price_growth
            scores.append(("hold", hold_score))

            sell_score = -self.expected_price_growth
            scores.append(("sell", sell_score))

        # RENT_OUT home — only if owner-occupier (move out, become renter+landlord) - THEY ACTUALLY HAVE TO GO TO THE RENTAL MARKET - CHECK THIS
        if self.is_owner_occupier:
            rent_out_score = (
                avg_market_rent  # EXPECTATIONS
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

    def choose_property(self, candidates, avg_market_rent, purpose="buy"):  # EXPECTATIONS
        """Select among feasible candidates via logit on WTP."""
        if not candidates:
            return None
        credit = self.model.credit
        scores = [(p, self._wtp_for_property(p, avg_market_rent, credit, purpose=purpose)) for p in candidates]
        probs = _logit_probs(scores)
        props, weights = zip(*probs)
        return self.model.random.choices(list(props), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 3: Bid formation
    # ------------------------------------------------------------------

    def compute_bid(self, prop, avg_market_rent, purpose="buy"):
        """Truthful WTP bid for ownership."""
        return self._wtp_for_property(prop, avg_market_rent, self.model.credit, purpose=purpose)

    def compute_rent_bid(self):
        """Maximum monthly rent bid (affordability ceiling)."""
        ratio = self.model.config.valuation.max_rent_income_ratio
        return household_max_rent(self.income, ratio)

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
            # Free the rental unit being vacated so it does not keep a stale
            # occupant pointing at this buyer (which would double-occupy).
            old_home = self.home_property
            if (
                old_home is not None
                and old_home != prop.id
                and old_home not in self.owned_properties
            ):
                old_prop = self.model._property_map.get(old_home)
                if old_prop is not None and old_prop.occupant_id == self.unique_id:
                    old_prop.occupant_id = None
                    old_prop.listed_for_rent = True
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
            outstanding = self.model.credit.outstanding_principal(orig_price, ltv, steps_held)
        else:
            outstanding = 0.0

        net_proceeds = sale_price - outstanding
        self.cash += net_proceeds

        self.owned_properties.discard(prop.id)
        self._mortgages.pop(prop.id, None)
        self._housing_asset_value = max(0.0, self._housing_asset_value - prop.estimated_value)

        # If sold home, become renter (unhoused until rental clears) - THIS SHOULD BE ONLY IF YOU'RE LIVING IN THE HOUSE YOURE SELLING
        # this should be working release_property() only clears home_property if self.home_property == prop.id

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
        Payment = monthly_mortgage_payment (each step is one month).
        """
        credit = self.model.credit
        updated = {}
        for pid, (orig_price, ltv, steps_held) in self._mortgages.items():
            payment = credit.monthly_mortgage_payment(orig_price, ltv)
            self.cash -= payment
            updated[pid] = (orig_price, ltv, steps_held + 1)
        self._mortgages = updated

    def mortgage_payment_due(self):
        """Total mortgage servicing due this period (monthly)."""
        credit = self.model.credit
        return float(
            sum(
                credit.monthly_mortgage_payment(orig_price, ltv)
                for orig_price, ltv, _ in self._mortgages.values()
            )
        )

    def distress_sale_candidates(self, avg_market_rent):
        """Rank owned properties from least to greatest expected utility loss."""
        credit = self.model.credit
        ranked = [
            (
                self._wtp_for_property(
                    self.model._property_map[pid], avg_market_rent, credit, purpose="buy-to-let"
                ),
                self.model._property_map[pid],
            )
            for pid in self.owned_properties
        ]
        return sorted(ranked, key=lambda item: item[0])

    # ------------------------------------------------------------------
    # Income dynamics - SHOULD BE IN ANOTHER MODULE
    # ------------------------------------------------------------------

    def evolve_income(self):
        """
        Apply one period of income evolution.
        Apply multiplicative income growth draws each period driven by the
        current macro state.
        """
        mcfg = getattr(self.model.config, "macro", None)
        state = getattr(self.model, "current_macro_state", "Neutral")
        if mcfg is None:
            raise RuntimeError(
                "Missing [macro] section in config; macro-driven income shocks are required."
            )
        match state:
            case "Boom":
                mu, sd = mcfg.boom_mean, mcfg.boom_sd
            case "Recession":
                mu, sd = mcfg.recession_mean, mcfg.recession_sd
            case _:
                mu, sd = mcfg.neutral_mean, mcfg.neutral_sd
        shock = float(self.model.rng.normal(mu, sd))

        self.income = float(self.income * np.exp(shock))

    # ------------------------------------------------------------------
    # Expectation update
    # ------------------------------------------------------------------

    def update_expectations(self, price_signal, rent_signal, delta=None):
        d = delta if delta is not None else self.model.config.expectations.delta
        noise_sd = self.model.config.expectations.household_noise_sd
        self.expected_price_growth = adaptive_update(self.expected_price_growth, price_signal, d)
        self.expected_rent_growth = adaptive_update(self.expected_rent_growth, rent_signal, d)
        if noise_sd > 0.0:
            self.expected_price_growth += float(self.model.rng.normal(0.0, noise_sd))
            self.expected_rent_growth += float(self.model.rng.normal(0.0, noise_sd))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_market_rent, credit, purpose="buy"):
        cfg = self.model.config
        monthly_rent = estimate_market_rent(
            prop.quality, avg_market_rent, cfg.valuation.quality_sensitivity
        )
        reference_price = (
            self.model._price_history[-1] if self.model._price_history else prop.estimated_value
        )
        capital_gain = self.expected_price_growth * reference_price

        if purpose == "buy":
            # Owner-occupier primary residence purchase
            quality_value = cfg.valuation.quality_value_scale * prop.quality
            zones = self.model.get_household_search_zones(self.home_zone)
            rental_props = [
                p for p in self.model.properties if p.zone in zones and p.id != self.home_property
            ]
            best_monthly_rent = 0.0
            for rp in rental_props:
                mr = estimate_market_rent(
                    rp.quality, avg_market_rent, cfg.valuation.quality_sensitivity
                )
                if mr > best_monthly_rent:
                    best_monthly_rent = mr
            if best_monthly_rent <= 0.0:
                best_monthly_rent = avg_market_rent
            outside_option = best_monthly_rent

            wtp = household_wtp(
                quality_value,
                capital_gain,
                outside_option,
                credit.mortgage_rate,
                credit.ltv_limit,
                credit.max_affordable_price(self.cash, self.income),
            )
            return wtp

        # Buy-to-let / investment purchase
        net_rent = monthly_rent
        wtp = investor_wtp(
            net_rent,
            capital_gain,
            cfg.credit.btl_funding_rate,
            cfg.credit.btl_ltv,
        )
        wtp = min(wtp, credit.max_affordable_price(self.cash, self.income))
        return wtp

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
        home_zone=None,
        expected_price_growth=None,
        expected_rent_growth=None,
    ):
        super().__init__(model)

        self.cash = cash
        self.funding_rate = funding_rate
        self.home_zone = home_zone if home_zone is not None else 0

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
    def gross_housing_assets(self):
        return self._housing_asset_value

    @property
    def mortgage_debt(self):
        credit = self.model.credit
        return float(
            sum(
                credit.outstanding_principal(orig_price, ltv, steps_held)
                for orig_price, ltv, steps_held in self._mortgages.values()
            )
        )

    @property
    def housing_equity(self):
        return self.gross_housing_assets - self.mortgage_debt

    @property
    def net_worth(self):
        return self.cash + self.housing_equity

    # ------------------------------------------------------------------
    # Stage 1: Action selection
    # ------------------------------------------------------------------

    def choose_action(self, purchase_candidates, avg_rent=None):
        ltv = self.model.credit.inst_ltv
        owned = [self.model._property_map[pid] for pid in self.portfolio]
        feasible = [p for p in purchase_candidates
                    if p.estimated_value <= self.cash / max(1e-9, 1.0 - ltv)]

        def _acquire(p):
            return utility.delta_v_acquire(
                net_rent=p.quality * self.expected_rent_growth,  # TODO: rent level
                expected_capital_gain=self.expected_price_growth * p.estimated_value,
                funding_rate=self.funding_rate, ltv=ltv, price=p.estimated_value,
                cash=self.cash, risk_free_rate=self.funding_rate,
            )

        def _hold(p):
            return utility.delta_v_hold(
                net_rent=p.quality * self.expected_rent_growth,  # TODO: rent level
                expected_capital_gain=self.expected_price_growth * p.estimated_value,
                funding_rate=self.funding_rate, ltv=ltv, price=p.estimated_value,
                market_value=p.estimated_value, risk_free_rate=self.funding_rate,
            )

        values = {
            "acquire": max((_acquire(p) for p in feasible), default=float("-inf")),
            "hold":    sum(_hold(p) for p in owned) if owned else float("-inf"),
            "sell":    max((-_hold(p) for p in owned), default=float("-inf")),
        }
        return utility.logit_choice(values, self.model.rng)

    # ------------------------------------------------------------------
    # Stage 2: Property selection
    # ------------------------------------------------------------------

    def choose_property(self, candidates, avg_rent):
        if not candidates:
            return None
        ctx = utility.DecisionContext(
            avg_market_rent=avg_rent,
            purchase_candidates=candidates,
            rental_candidates=(),
        )
        values = {p: utility.property_value(self, p, ctx) for p in candidates}
        return utility.logit_choice(values, self.model.rng)

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
            ltv = self.model.config.credit.inst_ltv
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
        if prop.id not in self.portfolio:
            return

        self.portfolio.discard(prop.id)
        if prop.id in self._mortgages:
            orig_price, ltv, steps_held = self._mortgages[prop.id]
            outstanding = self.model.credit.outstanding_principal(orig_price, ltv, steps_held)
        else:
            outstanding = 0.0

        self.cash += sale_price - outstanding
        self._housing_asset_value = max(
            0.0,
            self._housing_asset_value - prop.estimated_value,
        )
        self._mortgages.pop(prop.id, None)

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent

    def service_mortgages(self):
        credit = self.model.credit
        updated = {}
        for pid, (orig_price, ltv, steps_held) in self._mortgages.items():
            payment = credit.monthly_mortgage_payment(orig_price, ltv)
            self.cash -= payment
            updated[pid] = (orig_price, ltv, steps_held + 1)
        self._mortgages = updated

    def mortgage_payment_due(self):
        """Total mortgage servicing due this period (monthly)."""
        credit = self.model.credit
        return float(
            sum(
                credit.monthly_mortgage_payment(orig_price, ltv)
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
        cfg = self.model.config.expectations
        state_history = getattr(self.model, "_state_history", None)
        if state_history is not None and len(state_history) >= cfg.inst_forecast_window + 1:
            predicted_change = institutional_price_forecast(
                state_history, cfg.inst_forecast_window
            )
            current_price = state_history[-1].get("price", 1.0)
            self.expected_price_growth = predicted_change / max(current_price, 1e-9)
            self.expected_rent_growth = institutional_rent_growth_signal(state_history)
        else:
            d = delta if delta is not None else cfg.delta
            self.expected_price_growth = adaptive_update(
                self.expected_price_growth, price_signal, d
            )
            self.expected_rent_growth = adaptive_update(
                self.expected_rent_growth, rent_signal, d
            )

        if cfg.inst_noise_sd > 0.0:
            self.expected_price_growth += float(self.model.rng.normal(0.0, cfg.inst_noise_sd))
            self.expected_rent_growth += float(self.model.rng.normal(0.0, cfg.inst_noise_sd))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_rent):
        cfg = self.model.config
        reference_price = (
            self.model._price_history[-1] if self.model._price_history else prop.estimated_value
        )
        capital_gain = self.expected_price_growth * reference_price

        gross_monthly_rent = estimate_market_rent(
            prop.quality, avg_rent, cfg.valuation.quality_sensitivity
        )
        net_rent = gross_monthly_rent
        effective_rate = float(self.funding_rate + cfg.agent.inst_required_return)

        wtp = investor_wtp(
            net_rent,
            capital_gain,
            effective_rate,
            cfg.agent.inst_ltv,
        )
        inst_ltv = cfg.agent.inst_ltv
        if inst_ltv < 1.0:
            max_price = self.cash / max(1e-9, (1.0 - inst_ltv))
            wtp = min(wtp, max_price)

        return wtp

    def step(self):
        pass
