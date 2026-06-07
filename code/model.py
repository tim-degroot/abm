"""
HousingModel — main simulation class.

Orchestrates the minimum economic loop each step:

  income_evolution
  → expectations
  → valuation (via valuation module)
  → action selection (logit, agents)
  → property selection (logit, agents)
  → bidding (WTP, agents)
  → auction (Vickrey, market layer)
  → ownership/rental transfers + balance-sheet updates
  → mortgage servicing
  → mark-to-market asset revaluation
  → expectation update

Spatial structure:
  Z = grid_rows x grid_cols zones on a 2D TOROIDAL grid (config [spatial]).
  Each agent's consideration set = its own zone + the 4 von Neumann neighbours
  (up/down/left/right), wrapping around the torus edges, so every agent faces a
  symmetric 5-zone search space with no edge effects. Properties are distributed
  as evenly as possible across zones. (See config.toml [spatial] for why the
  default is a 4x4 torus rather than plan.md's nominal Z=10.)

Initialisation (plan.md §17-18 — balance sheets DERIVED from allocations so the
accounting identity HousingAssets = HousingEquity + MortgageDebt holds by
construction):
  1. Generate housing stock; quality q_k = mu_z + nu_k, standardised; price
     anchor = base_price + price_sensitivity * q_k (base_price is a CALIBRATED
     market anchor, not arbitrary).
  2. Draw households: income (log-normal), TOTAL WEALTH (multiple of income),
     risk aversion (log-normal).
  3. Match income-ranked households to quality-ranked properties (richer get
     better). Draw an origination LTV per owner (capped at credit.ltv_limit).
     Derive: deposit = (1-LTV)*price = equity; mortgage = LTV*price;
     liquid cash = wealth - deposit.
  4. ownership_mode = "emergent" (default): a household owns only if it can
     afford the deposit AND meets the income (DTI) test; otherwise it becomes a
     renter, so the ownership rate EMERGES. ownership_mode = "target"
     (DIAGNOSTIC): force target_ownership_rate by making the wealthiest
     households owners, topping up cash if short so sheets stay feasible.
  5. Private landlords at t=0: a share of owners receive extra (let-out)
     properties with right-skewed portfolio sizes (plan §17).
  6. Institutions allocated a separate tranche of properties as rental stock.
  7. Remaining renters placed into available rental stock.
  8. Seed price and rent history from initial allocations.
"""

import numpy as np
import mesa
from mesa.datacollection import DataCollector

from config import Config, load_config
from properties import Property
from agents import HouseholdAgent, InstitutionalAgent
from credit import CreditEnvironment
from markets import OwnershipMarket, RentalMarket
from policies import NoPolicy
from expectations import price_growth_signal, rent_growth_signal
from metrics import MODEL_REPORTERS


class HousingModel(mesa.Model):
    """
    Minimal research-grade housing market ABM.

    Parameters
    ----------
    config : Config, optional
        Immutable parameter container (see config.py). Defaults to the bundled
        config.toml via load_config(). This is the single source of truth for
        every parameter and initialisation setting.
    policy : policy object, optional
        Defaults to NoPolicy.
    """

    def __init__(self, config=None, policy=None):
        self.config = config if config is not None else load_config()
        cfg = self.config

        super().__init__(seed=cfg.sim.seed)

        assert (
            cfg.sim.n_properties > cfg.sim.n_households
        ), "Need more properties than households so renters can find rentals."

        self.n_households = cfg.sim.n_households
        self.n_institutions = cfg.sim.n_institutions
        self.grid_rows = cfg.spatial.grid_rows
        self.grid_cols = cfg.spatial.grid_cols
        self.n_zones = cfg.spatial.n_zones
        self.target_ownership_rate = cfg.sim.target_ownership_rate

        self.policy = policy if policy is not None else NoPolicy()

        self.credit = CreditEnvironment(
            mortgage_rate=cfg.credit.mortgage_rate,
            ltv_limit=cfg.credit.ltv_limit,
            dti_limit=cfg.credit.dti_limit,
            loan_term_years=cfg.credit.loan_term_years,
        )

        # Spatial adjacency — 2D von Neumann torus
        self._zone_adjacency = self._build_zone_adjacency(self.grid_rows, self.grid_cols)

        # Housing stock (guaranteed zone distribution)
        self.properties = self._init_properties(cfg.sim.n_properties, self.n_zones)
        self._property_map = {p.id: p for p in self.properties}

        # Agents
        self._init_agents(cfg.sim.n_households, cfg.sim.n_institutions)

        # Ownership and tenure allocation
        self._init_ownership_and_tenure()

        # Seed market history from initial state
        self._price_history = self._seed_price_history()
        self._rent_history = []
        self._avg_rent = self._estimate_initial_rent()
        self._rent_history.append(self._avg_rent)

        # Per-step registers
        self.this_step_transactions = []
        self.this_step_rental_transactions = []

        self.all_transactions = []
        self.all_rental_transactions = []

        # Agent lookup cache (rebuilt if agents added/removed)
        self._agent_map = {a.unique_id: a for a in self.agents}

        self.datacollector = DataCollector(model_reporters=MODEL_REPORTERS)
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Spatial structure
    # ------------------------------------------------------------------

    def _build_zone_adjacency(self, rows, cols):
        """
        2D toroidal grid, von Neumann (4-neighbour) topology.

        Zone z maps to grid cell (row, col) = (z // cols, z % cols). Its
        searchable set is itself plus the 4 orthogonal neighbours, with row/col
        indices wrapped modulo rows/cols (the torus). With rows, cols >= 3 the
        four neighbours are always distinct from each other and from self, so
        every zone has exactly five searchable zones — a symmetric consideration
        set with no edge effects.

        Returns dict zone -> frozenset of searchable zones.
        """

        def zid(r, c):
            return (r % rows) * cols + (c % cols)

        adjacency = {}
        for r in range(rows):
            for c in range(cols):
                z = zid(r, c)
                adjacency[z] = frozenset(
                    {
                        z,
                        zid(r - 1, c),  # up
                        zid(r + 1, c),  # down
                        zid(r, c - 1),  # left
                        zid(r, c + 1),  # right
                    }
                )
        return adjacency

    def get_search_zones(self, home_zone):
        return self._zone_adjacency[home_zone]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_properties(self, n_properties, n_zones):
        """
        Generate housing stock with guaranteed zone coverage.

        Properties are distributed as evenly as possible across zones.
        Each zone gets at least floor(n_properties / n_zones) properties,
        with the remainder distributed one-per-zone from zone 0 up.

        Zone quality means are drawn from N(0, 0.5); property-level
        residuals from N(0, 0.5). All qualities standardised to mean 0, sd 1.

        Initial estimated_value and purchase_anchor_price are set to
        200_000 + 50_000 * quality (quality-proportional baseline).
        """
        pcfg = self.config.property_init
        zone_means = self.rng.normal(0.0, pcfg.zone_quality_sd, n_zones)

        # Distribute properties evenly: each zone gets its quota
        base = n_properties // n_zones
        remainder = n_properties % n_zones
        zone_counts = [base + (1 if z < remainder else 0) for z in range(n_zones)]

        raw_qualities = []
        zone_assignments = []
        for z, count in enumerate(zone_counts):
            for _ in range(count):
                q = zone_means[z] + self.rng.normal(0.0, pcfg.property_residual_sd)
                raw_qualities.append(q)
                zone_assignments.append(z)

        q_arr = np.array(raw_qualities)
        q_std = (q_arr - q_arr.mean()) / (q_arr.std() + 1e-9)

        base_price = pcfg.base_price
        price_sensitivity = pcfg.price_sensitivity

        props = []
        for i in range(n_properties):
            anchor = base_price + price_sensitivity * float(q_std[i])
            props.append(
                Property(
                    id=i,
                    zone=zone_assignments[i],
                    quality=float(q_std[i]),
                    owner_id=None,
                    purchase_anchor_price=anchor,
                    estimated_value=anchor,
                )
            )
        return props

    def _init_agents(self, n_households, n_institutions):
        """
        Create agents with heterogeneous attributes.

        Households: income       ~ LogNormal(log(income_median), income_sigma).
                    total wealth  ~ Uniform(wealth_income_mult_low, _high)*income.
                                    NOTE: `cash` initially holds TOTAL wealth; the
                                    deposit on any property owned at init is later
                                    subtracted in _init_ownership_and_tenure,
                                    leaving cash = liquid wealth (plan §17).
                    risk_aversion ~ LogNormal(risk_aversion_mu, _sigma).
                    home_zone assigned evenly across zones.

        Institutions: cash-rich, low funding rate.
        """
        acfg = self.config.agent_init
        incomes = self.rng.lognormal(
            np.log(acfg.income_median), acfg.income_sigma, n_households
        )
        wealth_mult = self.rng.uniform(
            acfg.wealth_income_mult_low, acfg.wealth_income_mult_high, n_households
        )
        risk_av = self.rng.lognormal(
            acfg.risk_aversion_mu, acfg.risk_aversion_sigma, n_households
        )

        for i in range(n_households):
            zone = int(i % self.n_zones)  # distribute evenly across zones
            HouseholdAgent(
                unique_id=i,
                model=self,
                income=float(incomes[i]),
                cash=float(incomes[i] * wealth_mult[i]),  # total wealth (see note)
                risk_aversion=float(risk_av[i]),
                home_zone=zone,
            )

        for j in range(n_institutions):
            zone = int(j % self.n_zones)
            InstitutionalAgent(
                unique_id=n_households + j,
                model=self,
                cash=float(self.rng.uniform(acfg.inst_cash_low, acfg.inst_cash_high)),
                funding_rate=float(
                    self.rng.uniform(
                        acfg.inst_funding_rate_low, acfg.inst_funding_rate_high
                    )
                ),
                home_zone=zone,
            )

    def _draw_origination_ltv(self):
        """Draw an origination LTV from the configured distribution, capped at
        the regulatory ceiling (plan §17 / FCA MPSD)."""
        ai = self.config.agent_init
        return min(
            self.credit.ltv_limit,
            float(self.rng.uniform(ai.ltv_dist_low, ai.ltv_dist_high)),
        )

    def _assign_property_to_owner(self, hh, prop, is_home, allow_topup):
        """
        Acquire `prop` for household `hh` at init and DERIVE the balance sheet so
        the accounting identity holds by construction (plan §17-18):
            deposit = (1 - LTV) * price  == equity at origination
            mortgage = LTV * price
            liquid cash = wealth - deposit

        Deposit-constrained households max out leverage (smallest deposit).
        Feasibility:
          - emergent mode: returns False if the deposit or the DTI/income test
            fails (household stays a renter; property left in the pool).
          - target mode (allow_topup=True): forces ownership, injecting cash to
            cover the deposit if short (diagnostic only) so liquid >= 0.
        Returns True iff the property was assigned.
        """
        price = prop.estimated_value
        ltv = self._draw_origination_ltv()
        deposit = price * (1.0 - ltv)

        # Deposit-constrained borrowers take the maximum allowed leverage.
        if hh.cash < deposit and self.credit.ltv_limit > ltv:
            ltv = self.credit.ltv_limit
            deposit = price * (1.0 - ltv)

        # Income (DTI) feasibility.
        payment = self.credit.annual_mortgage_payment(price, ltv)
        income_ok = payment <= self.credit.dti_limit * hh.income
        if not income_ok and not allow_topup:
            return False

        if hh.cash < deposit:
            if not allow_topup:
                return False
            hh.cash = deposit  # diagnostic top-up (target mode): liquid -> 0

        # Commit.
        hh.cash -= deposit
        hh.owned_properties.add(prop.id)
        hh._mortgages[prop.id] = (price, ltv, 0)  # (purchase_price, ltv, steps_held)
        hh._housing_asset_value += price
        prop.owner_id = hh.unique_id
        if is_home:
            hh.home_property = prop.id
            prop.occupant_id = hh.unique_id
        else:
            prop.listed_for_rent = True  # let-out (landlord) property
        return True

    def _init_ownership_and_tenure(self):
        """
        Allocate properties and DERIVE balance sheets (plan §17-18).

        1. Sort households by income desc, properties by quality desc, and match
           rank-to-rank (richer get better) for the top target_ownership_rate.
        2. ownership_mode = "emergent" (default): each match owns only if it
           clears the deposit AND DTI tests at its drawn LTV — so the ownership
           rate EMERGES (it is NOT pinned to the target; see TODO Model #4).
           ownership_mode = "target" (DIAGNOSTIC): force the target rate by
           topping up cash where needed so sheets stay feasible.
        3. Private landlords at t=0: the wealthiest `landlord_share` of owners
           receive extra let-out properties, portfolio size right-skewed
           (Geometric), subject to the same feasibility rules.
        4. Institutions take a tranche of remaining stock as rental supply.
        5. Remaining renters placed into available rental stock.
        6. Accounting identity verified.
        """
        cfg = self.config
        mode = cfg.sim.ownership_mode
        allow_topup = mode == "target"

        households = sorted(
            [a for a in self.agents if isinstance(a, HouseholdAgent)],
            key=lambda h: h.income,
            reverse=True,
        )
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]

        # Properties sorted best-to-worst
        props_sorted = sorted(self.properties, key=lambda p: p.quality, reverse=True)
        available = list(props_sorted)

        n_owners_target = int(self.target_ownership_rate * len(households))
        n_inst_target = int(cfg.sim.inst_ownership_share * len(self.properties))

        # --- 1-2. Primary (owner-occupier) allocation ---
        for hh in households[:n_owners_target]:
            if not available:
                break
            # Prefer a property in hh's home zone, else the best available.
            zone_match = [p for p in available if p.zone == hh.home_zone]
            prop = zone_match[0] if zone_match else available[0]
            if self._assign_property_to_owner(hh, prop, is_home=True, allow_topup=allow_topup):
                available.remove(prop)
            # else: emergent infeasible -> hh stays renter, prop stays in pool

        # --- 3. Private landlords at t=0 (right-skewed extra portfolios) ---
        ai = cfg.agent_init
        owners = [h for h in households if h.owned_properties]
        n_landlords = int(ai.landlord_share * len(owners))
        landlords = sorted(owners, key=lambda h: h.cash, reverse=True)[:n_landlords]
        for ll in landlords:
            extra = int(self.rng.geometric(ai.landlord_portfolio_geom_p))  # >= 1
            for _ in range(extra):
                if not available:
                    break
                zones = self.get_search_zones(ll.home_zone)
                zone_match = [p for p in available if p.zone in zones]
                prop = zone_match[0] if zone_match else available[0]
                if self._assign_property_to_owner(
                    ll, prop, is_home=False, allow_topup=allow_topup
                ):
                    available.remove(prop)
                else:
                    break  # can't afford more rentals; stop extending this LL

        # --- 4. Institutional stock (rental supply) ---
        inst_stock = available[:n_inst_target]
        for k, prop in enumerate(inst_stock):
            available.remove(prop)
            inst = institutions[k % len(institutions)]
            prop.owner_id = inst.unique_id
            inst.portfolio.add(prop.id)
            inst._housing_asset_value += prop.estimated_value
            prop.listed_for_rent = True

        # --- 5. Place renters into rental stock ---
        rental_pool = [
            p for p in self.properties if p.occupant_id is None and p.listed_for_rent
        ]
        rental_pool += [p for p in available if p.owner_id is None]

        renter_households = [h for h in households if h.home_property is None]
        for hh in renter_households:
            zone_match = [p for p in rental_pool if p.zone == hh.home_zone]
            prop = (
                zone_match[0]
                if zone_match
                else (rental_pool[0] if rental_pool else None)
            )
            if prop is None:
                continue  # unhoused; will enter rental market next step
            rental_pool.remove(prop)
            prop.occupant_id = hh.unique_id
            prop.listed_for_rent = False
            hh.home_property = prop.id
            hh.home_zone = prop.zone

        # --- 6. Verify accounting identity (plan §18) ---
        self._verify_accounting()

    def _verify_accounting(self, tol=1.0):
        """
        Assert the housing balance sheet is internally consistent at init:
          - total HousingAssets from properties == sum of agent housing-asset
            values (no double-counting / orphaned ownership), and
          - total mortgage debt does not exceed housing assets.
        HousingEquity is then HousingAssets - MortgageDebt by definition, so
        HousingAssets = HousingEquity + MortgageDebt holds by construction.
        """
        props_assets = sum(
            p.estimated_value for p in self.properties if p.owner_id is not None
        )
        agent_assets = sum(
            getattr(a, "_housing_asset_value", 0.0) for a in self.agents
        )
        assert abs(props_assets - agent_assets) <= tol, (
            f"Housing assets mismatch: properties={props_assets:.2f} "
            f"agents={agent_assets:.2f}"
        )

        debt = 0.0
        for a in self.agents:
            for orig, ltv, held in getattr(a, "_mortgages", {}).values():
                debt += self.credit.outstanding_principal(orig, ltv, held)
        assert debt <= props_assets + tol, (
            f"Mortgage debt {debt:.2f} exceeds housing assets {props_assets:.2f}"
        )

    def _seed_price_history(self):
        """Seed price history from initial property values."""
        allocated = [
            p.purchase_anchor_price for p in self.properties if p.owner_id is not None
        ]
        if allocated:
            return [float(np.mean(allocated))]
        return [self.config.market.fallback_price]

    def _estimate_initial_rent(self):
        """Monthly rent from the configured gross yield on median property price."""
        prices = [p.estimated_value for p in self.properties]
        return float(np.median(prices)) * self.config.market.initial_rent_yield / 12.0

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self):
        """
        One model period. Full economic loop:

          1. Policy hooks + income evolution
          2. Mark-to-market asset revaluation
          3. List properties (sellers + landlords)
          4. Submit bids (buyers + renters)
          5. Clear ownership market (Vickrey)
          6. Clear rental market (Vickrey)
          7. Apply transactions + balance-sheet updates
          8. Mortgage servicing
          9. Expectation updates
         10. Data collection
        """
        self.policy.on_step_start(self)
        self.credit = self.policy.modify_credit(self.credit)

        self.this_step_transactions = []
        self.this_step_rental_transactions = []

        # 1. Income evolution
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                agent.evolve_income()

        # 2. Mark-to-market revaluation
        self._mark_to_market()

        avg_rent = self._avg_rent

        ownership_market = OwnershipMarket(step=self.steps)
        rental_market = RentalMarket(step=self.steps)

        # 3 & 4. Listing and bidding
        self._list_properties(ownership_market, rental_market, avg_rent)
        self._submit_bids(ownership_market, rental_market, avg_rent)

        # 5 & 6. Market clearing
        sale_txns = ownership_market.clear()
        rental_txns = rental_market.clear()

        # 7. Apply transactions
        self._apply_ownership_transactions(sale_txns)
        self._apply_rental_transactions(rental_txns)

        # 8. Mortgage servicing (after sales so sold mortgages are gone)
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) and agent._mortgages:
                agent.service_mortgages()

        # 9. Expectations
        self._update_expectations()

        # 10. Data
        self.datacollector.collect(self)

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def _mark_to_market(self):
        """
        Update estimated_value on all properties and restate agent
        _housing_asset_value using current estimated values.

        Simple approach: use the average price growth expectation across
        all agents to nudge estimated values, anchored to transaction prices
        when available. This gives agents a continuously updated balance sheet.
        """
        if len(self._price_history) >= 2:
            growth = (self._price_history[-1] - self._price_history[-2]) / max(
                self._price_history[-2], 1.0
            )
        else:
            growth = 0.0

        for prop in self.properties:
            prop.estimated_value = prop.estimated_value * (1.0 + growth)

        # Restate each agent's housing asset value
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) and agent.owned_properties:
                agent._housing_asset_value = sum(
                    self._property_map[pid].estimated_value
                    for pid in agent.owned_properties
                )
            elif isinstance(agent, InstitutionalAgent) and agent.portfolio:
                agent._housing_asset_value = sum(
                    self._property_map[pid].estimated_value for pid in agent.portfolio
                )

    # ------------------------------------------------------------------
    # Market participation
    # ------------------------------------------------------------------

    def _list_properties(self, ownership_market, rental_market, avg_rent):
        """
        Owners decide each period whether to sell, rent out, or hold.

        Each owned property is evaluated independently.
        Owner-occupiers choose over: hold, sell, rent_out.
        Landlord properties default to listed_for_rent unless sell is chosen.
        """
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                purchase_candidates = self._get_purchase_candidates(agent)
                action = agent.choose_action(purchase_candidates, avg_rent)

                for pid in list(agent.owned_properties):
                    prop = self._property_map[pid]
                    prop.listed_for_sale = False
                    prop.listed_for_rent = False

                    if pid == agent.home_property:
                        # Owner-occupied home: sell or rent_out trigger listing
                        if action == "sell":
                            reservation = (
                                prop.purchase_anchor_price
                                * self.config.market.household_sell_reservation_discount
                            )
                            ownership_market.list_property(
                                pid, agent.unique_id, reservation
                            )
                            prop.listed_for_sale = True
                        elif action == "rent_out":
                            reservation_rent = self._reservation_rent(prop)
                            rental_market.list_property(
                                pid, agent.unique_id, reservation_rent
                            )
                            prop.listed_for_rent = True
                    else:
                        # Investment property: always list for rent unless selling
                        if action == "sell":
                            reservation = (
                                prop.purchase_anchor_price
                                * self.config.market.household_sell_reservation_discount
                            )
                            ownership_market.list_property(
                                pid, agent.unique_id, reservation
                            )
                            prop.listed_for_sale = True
                        else:
                            reservation_rent = self._reservation_rent(prop)
                            rental_market.list_property(
                                pid, agent.unique_id, reservation_rent
                            )
                            prop.listed_for_rent = True

            elif isinstance(agent, InstitutionalAgent):
                purchase_candidates = self._get_purchase_candidates(agent)
                action = agent.choose_action(purchase_candidates, avg_rent)

                for pid in list(agent.portfolio):
                    prop = self._property_map[pid]
                    prop.listed_for_sale = False
                    prop.listed_for_rent = False

                    if action == "sell":
                        reservation = (
                            prop.purchase_anchor_price
                            * self.config.market.inst_sell_reservation_discount
                        )
                        ownership_market.list_property(
                            pid, agent.unique_id, reservation
                        )
                        prop.listed_for_sale = True
                    else:
                        reservation_rent = self._reservation_rent(prop)
                        rental_market.list_property(
                            pid, agent.unique_id, reservation_rent
                        )
                        prop.listed_for_rent = True

    def _submit_bids(self, ownership_market, rental_market, avg_rent):
        """Buyers and renters submit bids."""
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                purchase_candidates = self._get_purchase_candidates(agent)
                action = agent.choose_action(purchase_candidates, avg_rent)

                # Purchase bid
                if action == "buy":
                    affordable = [
                        p
                        for p in purchase_candidates
                        if self.credit.is_feasible(
                            p.estimated_value, agent.cash, agent.income
                        )
                        and p.listed_for_sale
                    ]
                    chosen = agent.choose_property(affordable, avg_rent)
                    if chosen is not None:
                        bid = agent.compute_bid(chosen, avg_rent)
                        if bid > 0:
                            ownership_market.submit_bid(
                                chosen.id, agent.unique_id, bid, "household"
                            )

                # Rental bid — any renter (unhoused or choosing to rent)
                if action == "rent" or agent.home_property is None:
                    rental_candidates = self._get_rental_candidates(agent)
                    chosen = agent.choose_rental(rental_candidates)
                    if chosen is not None:
                        rent_bid = agent.compute_rent_bid()
                        if rent_bid > 0:
                            rental_market.submit_bid(
                                chosen.id, agent.unique_id, rent_bid
                            )

            elif isinstance(agent, InstitutionalAgent):
                purchase_candidates = self._get_purchase_candidates(agent)
                action = agent.choose_action(purchase_candidates, avg_rent)

                if action == "buy":
                    listed = [p for p in purchase_candidates if p.listed_for_sale]
                    chosen = agent.choose_property(listed, avg_rent)
                    if chosen is not None:
                        bid = agent.compute_bid(chosen, avg_rent)
                        if bid > 0:
                            ownership_market.submit_bid(
                                chosen.id, agent.unique_id, bid, "institution"
                            )

    # ------------------------------------------------------------------
    # Candidate set construction
    # ------------------------------------------------------------------

    def _get_purchase_candidates(self, agent):
        """Properties listed for sale in agent's search zones (not self-owned)."""
        zones = self.get_search_zones(agent.home_zone)
        return [
            p
            for p in self.properties
            if p.zone in zones and p.listed_for_sale and p.owner_id != agent.unique_id
        ]

    def _get_rental_candidates(self, agent):
        """Properties listed for rent in agent's search zones (not self-owned)."""
        zones = self.get_search_zones(agent.home_zone)
        return [
            p
            for p in self.properties
            if p.zone in zones
            and p.listed_for_rent
            and p.occupant_id is None
            and p.owner_id != agent.unique_id
        ]

    def _reservation_rent(self, prop):
        """
        Minimum rent a landlord will accept.
        Anchored to the configured gross yield on purchase anchor price.
        """
        mcfg = self.config.market
        return max(
            mcfg.min_reservation_rent,
            prop.purchase_anchor_price * mcfg.landlord_reservation_yield / 12.0,
        )

    # ------------------------------------------------------------------
    # Transaction application
    # ------------------------------------------------------------------

    def _apply_ownership_transactions(self, transactions):
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            seller = self._agent_map.get(txn.seller_id)
            buyer = self._agent_map.get(txn.buyer_id)
            if seller is None or buyer is None:
                continue

            # Vacate current occupant if it is the buyer (moving from rented)
            # — occupant_id will be reset by buyer.acquire_property
            prev_occupant_id = prop.occupant_id
            if prev_occupant_id is not None and prev_occupant_id != txn.seller_id:
                prev_occupant = self._agent_map.get(prev_occupant_id)
                if isinstance(prev_occupant, HouseholdAgent):
                    prev_occupant.vacate_rental()

            seller.release_property(prop, txn.price)

            prop.owner_id = txn.buyer_id
            prop.purchase_anchor_price = txn.price
            prop.estimated_value = txn.price
            prop.listed_for_sale = False

            buyer.acquire_property(prop, txn.price)

            self.policy.on_transaction(txn, self)

        self.this_step_transactions = transactions
        self.all_transactions.extend(transactions)

    def _apply_rental_transactions(self, transactions):
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            tenant = self._agent_map.get(txn.tenant_id)
            landlord = self._agent_map.get(txn.landlord_id)
            if tenant is None or landlord is None:
                continue

            # Evict previous occupant if any
            if prop.occupant_id is not None and prop.occupant_id != txn.tenant_id:
                prev = self._agent_map.get(prop.occupant_id)
                if isinstance(prev, HouseholdAgent):
                    prev.vacate_rental()

            prop.occupant_id = txn.tenant_id
            prop.listed_for_rent = False

            if isinstance(tenant, HouseholdAgent):
                tenant.move_into_rental(prop)
                tenant.pay_rent(txn.monthly_rent)

            if isinstance(landlord, HouseholdAgent):
                landlord.receive_rent(txn.monthly_rent)
            elif isinstance(landlord, InstitutionalAgent):
                landlord.receive_rent(txn.monthly_rent)

            self.policy.on_rental_transaction(txn, self)

        self.this_step_rental_transactions = transactions
        self.all_rental_transactions.extend(transactions)

    # ------------------------------------------------------------------
    # Expectation updates
    # ------------------------------------------------------------------

    def _update_expectations(self):
        if self.this_step_transactions:
            period_avg = float(np.mean([t.price for t in self.this_step_transactions]))
        elif self._price_history:
            period_avg = self._price_history[-1]
        else:
            period_avg = self.config.market.fallback_price
        self._price_history.append(period_avg)

        if self.this_step_rental_transactions:
            period_rent = float(
                np.mean([t.monthly_rent for t in self.this_step_rental_transactions])
            )
            self._rent_history.append(period_rent)
            self._avg_rent = period_rent
        elif self._rent_history:
            self._rent_history.append(self._rent_history[-1])

        window = self.config.expectations.signal_window
        p_signal = price_growth_signal(self._price_history[-window:])
        r_signal = rent_growth_signal(self._rent_history[-window:])

        for agent in self.agents:
            agent.update_expectations(p_signal, r_signal)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_model_state(self):
        households = [a for a in self.agents if isinstance(a, HouseholdAgent)]
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]
        owners = sum(1 for h in households if h.owned_properties)
        owner_occ = sum(1 for h in households if h.is_owner_occupier)
        renters = sum(1 for h in households if h.is_renter)
        landlords = sum(1 for h in households if h.is_landlord)
        renter_landlords = sum(1 for h in households if h.is_renter and h.is_landlord)
        return {
            "step": self.steps,
            "hh_ownership_rate": owners / max(len(households), 1),
            "owner_occupier_count": owner_occ,
            "renter_count": renters,
            "landlord_count": landlords,
            "renter_landlord_count": renter_landlords,
            "inst_property_count": sum(len(i.portfolio) for i in institutions),
            "sale_txns": len(self.this_step_transactions),
            "rental_txns": len(self.this_step_rental_transactions),
            "price_tail": self._price_history[-3:],
        }
