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
    institution_wtp,
    household_max_rent,
    estimate_market_rent,
    renter_outside_option,
)

# Logit temperature parameters
BETA_ACTION = 1.0
BETA_PROPERTY = 0.5

# Income mean-reversion speed (per period)
INCOME_REVERSION = 0.05
# Income shock volatility (log scale)
INCOME_SHOCK_SD = 0.05


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

        self.expected_price_growth = (
            expected_price_growth
            if expected_price_growth is not None
            else init_price_expectation()
        )
        self.expected_rent_growth = (
            expected_rent_growth
            if expected_rent_growth is not None
            else init_rent_expectation()
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
            scores.append(("buy", BETA_ACTION * best_wtp))
        else:
            scores.append(("buy", -np.inf))

        # RENT — always available; scored by rent burden
        monthly_burden = avg_market_rent / max(self.income / 12.0, 1.0)
        rent_score = BETA_ACTION * (-monthly_burden)
        scores.append(("rent", rent_score))

        # HOLD / SELL / RENT_OUT — only for owners
        if self.owned_properties:
            hold_score = BETA_ACTION * self.expected_price_growth
            scores.append(("hold", hold_score))

            sell_score = BETA_ACTION * (-self.expected_price_growth + 0.02)
            scores.append(("sell", sell_score))

        # RENT_OUT home — only if owner-occupier (move out, become renter+landlord)
        if self.is_owner_occupier:
            rent_out_score = BETA_ACTION * (
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
        scores = [
            (p, BETA_PROPERTY * self._wtp_for_property(p, avg_market_rent, credit))
            for p in candidates
        ]
        probs = _logit_probs(scores)
        props, weights = zip(*probs)
        return self.model.random.choices(list(props), weights=list(weights), k=1)[0]

    def choose_rental(self, rental_candidates):
        """Select among available rentals. Prefers lower rent relative to income."""
        if not rental_candidates:
            return None
        scores = [
            (p, BETA_PROPERTY * (-p.estimated_value / max(self.income / 12.0, 1.0)))
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
        return household_max_rent(self.income)

    # ------------------------------------------------------------------
    # Balance sheet updates
    # ------------------------------------------------------------------

    def acquire_property(self, prop, price):
        """
        Called by model when this household wins an ownership auction.

        Deducts deposit from cash, records mortgage, moves in if unhoused.
        """
        ltv = self.model.credit.ltv_limit
        deposit = price * (1.0 - ltv)
        self.cash -= deposit
        self.owned_properties.add(prop.id)
        self._mortgages[prop.id] = (price, ltv, 0)  # (purchase_price, ltv, steps_held)
        self._housing_asset_value += prop.estimated_value

        # Move in if currently unhoused or renting
        if not self.is_owner_occupier:
            # Leave current rental if any (occupant_id cleared by model)
            self.home_property = prop.id
            self.home_zone = prop.zone

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

    # ------------------------------------------------------------------
    # Income dynamics
    # ------------------------------------------------------------------

    def evolve_income(self):
        """
        Apply one period of income evolution.

        Log-normal shock with mean reversion toward baseline_income.
        A household whose income rises enough can afford to buy;
        one whose income falls may need to downsize or fall behind
        on mortgage payments.
        """
        shock = self.model.rng.normal(0.0, INCOME_SHOCK_SD)
        log_income = np.log(max(self.income, 1.0))
        log_baseline = np.log(max(self.baseline_income, 1.0))
        log_income_new = (
            log_income + INCOME_REVERSION * (log_baseline - log_income) + shock
        )
        self.income = float(np.exp(log_income_new))

    # ------------------------------------------------------------------
    # Expectation update
    # ------------------------------------------------------------------

    def update_expectations(self, price_signal, rent_signal, delta=None):
        d = delta if delta is not None else 0.7
        self.expected_price_growth = adaptive_update(
            self.expected_price_growth, price_signal, d
        )
        self.expected_rent_growth = adaptive_update(
            self.expected_rent_growth, rent_signal, d
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_market_rent, credit):
        outside_opt = renter_outside_option(avg_market_rent, self.income)
        credit_ceil = credit.max_affordable_price(self.cash, self.income)
        return household_wtp(
            quality=prop.quality,
            expected_price_growth=self.expected_price_growth,
            mortgage_rate=credit.mortgage_rate,
            ltv=credit.ltv_limit,
            outside_option_value=outside_opt,
            credit_ceiling=credit_ceil,
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

        self.expected_price_growth = (
            expected_price_growth
            if expected_price_growth is not None
            else init_price_expectation()
        )
        self.expected_rent_growth = (
            expected_rent_growth
            if expected_rent_growth is not None
            else init_rent_expectation()
        )

        self._housing_asset_value = 0.0

    @property
    def net_worth(self):
        return self.cash + self._housing_asset_value

    # ------------------------------------------------------------------
    # Stage 1: Action selection
    # ------------------------------------------------------------------

    def choose_action(self, purchase_candidates, avg_rent):
        scores = []

        if purchase_candidates:
            best_wtp = max(
                self._wtp_for_property(p, avg_rent) for p in purchase_candidates
            )
            scores.append(("buy", BETA_ACTION * best_wtp))
        else:
            scores.append(("buy", -np.inf))

        scores.append(("hold", BETA_ACTION * self.expected_price_growth))
        scores.append(("sell", BETA_ACTION * (-self.expected_price_growth + 0.01)))

        probs = _logit_probs(scores)
        actions, weights = zip(*probs)
        return self.model.random.choices(list(actions), weights=list(weights), k=1)[0]

    # ------------------------------------------------------------------
    # Stage 2: Property selection
    # ------------------------------------------------------------------

    def choose_property(self, candidates, avg_rent):
        if not candidates:
            return None
        scores = [
            (p, BETA_PROPERTY * self._wtp_for_property(p, avg_rent)) for p in candidates
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

    def acquire_property(self, prop, price):
        self.cash -= price
        self.portfolio.add(prop.id)
        self._housing_asset_value += prop.estimated_value

    def release_property(self, prop, sale_price):
        self.portfolio.discard(prop.id)
        self._housing_asset_value = max(
            0.0, self._housing_asset_value - prop.estimated_value
        )
        self.cash += sale_price

    def receive_rent(self, monthly_rent):
        self.cash += monthly_rent

    # ------------------------------------------------------------------
    # Expectation update
    # ------------------------------------------------------------------

    def update_expectations(self, price_signal, rent_signal, delta=None):
        d = delta if delta is not None else 0.7
        self.expected_price_growth = adaptive_update(
            self.expected_price_growth, price_signal, d
        )
        self.expected_rent_growth = adaptive_update(
            self.expected_rent_growth, rent_signal, d
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wtp_for_property(self, prop, avg_rent):
        expected_rent = estimate_market_rent(prop.quality, avg_rent)
        return institution_wtp(
            expected_rent=expected_rent * 12,
            operating_cost_fraction=0.15,
            expected_price_growth=self.expected_price_growth,
            funding_rate=self.funding_rate,
            ltv=0.60,
        )

    def step(self):
        pass
