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
  Z zones in a RING (toroidal 1-D) topology.
  Each zone has floor(n_properties / n_zones) + overflow properties,
  ensuring every zone has enough stock for its resident agents.
  Agents search their home zone + adjacent zones only.

Initialisation:
  1. Generate housing stock: properties distributed evenly across zones.
     Each zone guaranteed >= ceil(n_households / n_zones) * 1.3 properties.
  2. Draw agent incomes from log-normal; sort ascending.
  3. Sort properties by quality ascending.
  4. Match households to properties income-rank to quality-rank (richer
     households get better properties); verify deposit feasibility.
  5. Bottom ~35% of households initialised as renters entering rental market.
  6. Institutions allocated a separate tranche of properties.
  7. Set estimated_value = purchase_anchor_price at init.
  8. Seed price and rent history from initial allocations.
"""

import numpy as np
import mesa
from mesa.datacollection import DataCollector

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
    n_households : int
        Number of household agents.
    n_institutions : int
        Number of institutional investor agents.
    n_properties : int
        Total housing stock (must be > n_households to guarantee supply).
    n_zones : int
        Number of spatial zones (ring topology).
    target_ownership_rate : float
        Fraction of households initialised as owners (~0.65).
    seed : int or None
        Random seed.
    policy : policy object, optional
        Defaults to NoPolicy.
    credit_params : dict, optional
        Overrides for CreditEnvironment.
    """

    def __init__(
        self,
        n_households=100,
        n_institutions=5,
        n_properties=130,
        n_zones=10,
        target_ownership_rate=0.65,
        seed=42,
        policy=None,
        credit_params=None,
    ):
        super().__init__(seed=seed)

        assert (
            n_properties > n_households
        ), "Need more properties than households so renters can find rentals."

        self.n_households = n_households
        self.n_institutions = n_institutions
        self.n_zones = n_zones
        self.target_ownership_rate = target_ownership_rate

        self.policy = policy if policy is not None else NoPolicy()

        cp = credit_params or {}
        self.credit = CreditEnvironment(**cp)

        # Spatial adjacency — ring topology
        self._zone_adjacency = self._build_zone_adjacency(n_zones)

        # Housing stock (guaranteed zone distribution)
        self.properties = self._init_properties(n_properties, n_zones)
        self._property_map = {p.id: p for p in self.properties}

        # Agents
        self._init_agents(n_households, n_institutions)

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

    def _build_zone_adjacency(self, n_zones):
        """
        Ring topology: zone z neighbours (z-1) and (z+1) mod n_zones.
        Returns dict zone -> frozenset of searchable zones (self + neighbours).
        """
        return {
            z: frozenset({z, (z - 1) % n_zones, (z + 1) % n_zones})
            for z in range(n_zones)
        }

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
        zone_means = self.rng.normal(0.0, 0.5, n_zones)

        # Distribute properties evenly: each zone gets its quota
        base = n_properties // n_zones
        remainder = n_properties % n_zones
        zone_counts = [base + (1 if z < remainder else 0) for z in range(n_zones)]

        raw_qualities = []
        zone_assignments = []
        for z, count in enumerate(zone_counts):
            for _ in range(count):
                q = zone_means[z] + self.rng.normal(0.0, 0.5)
                raw_qualities.append(q)
                zone_assignments.append(z)

        q_arr = np.array(raw_qualities)
        q_std = (q_arr - q_arr.mean()) / (q_arr.std() + 1e-9)

        base_price = 200_000.0
        price_sensitivity = 50_000.0

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

        Households: income ~ LogNormal(log(35000), 0.5), UK-inspired.
                    cash ~ Uniform(0.5, 2.0) * income.
                    risk_aversion ~ LogNormal(0, 0.5).
                    home_zone assigned proportionally across zones.

        Institutions: cash-rich, low funding rate.
        """
        incomes = self.rng.lognormal(np.log(35_000), 0.5, n_households)
        cash_mult = self.rng.uniform(0.5, 2.0, n_households)
        risk_av = self.rng.lognormal(0.0, 0.5, n_households)

        for i in range(n_households):
            zone = int(i % self.n_zones)  # distribute evenly across zones
            HouseholdAgent(
                unique_id=i,
                model=self,
                income=float(incomes[i]),
                cash=float(incomes[i] * cash_mult[i]),
                risk_aversion=float(risk_av[i]),
                home_zone=zone,
            )

        for j in range(n_institutions):
            zone = int(j % self.n_zones)
            InstitutionalAgent(
                unique_id=n_households + j,
                model=self,
                cash=float(self.rng.uniform(5_000_000, 20_000_000)),
                funding_rate=float(self.rng.uniform(0.02, 0.03)),
                home_zone=zone,
            )

    def _init_ownership_and_tenure(self):
        """
        Allocate properties to agents consistently with income rank.

        Strategy:
          1. Sort households by income descending (richer get better properties).
          2. Sort properties by quality descending.
          3. Top target_ownership_rate fraction of households become owners.
             Each is matched to a property in their home zone if possible,
             falling back to any available property.
          4. Check deposit feasibility; if not feasible, agent becomes renter.
          5. Remaining properties: bottom tranche to institutions as rental stock,
             rest remain unowned (available rental/purchase pool).
          6. Renters are placed in institutional or unowned properties.

        Balance sheets:
          - Owner's cash reduced by deposit (price * (1 - LTV)).
          - Owner's _housing_asset_value set to estimated_value.
          - Mortgage record created.
        """
        households = sorted(
            [a for a in self.agents if isinstance(a, HouseholdAgent)],
            key=lambda h: h.income,
            reverse=True,
        )
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]

        # Properties sorted best-to-worst
        props_sorted = sorted(self.properties, key=lambda p: p.quality, reverse=True)

        n_owners_target = int(self.target_ownership_rate * len(households))
        n_inst_target = int(0.10 * len(self.properties))

        # --- Allocate ownership ---
        available = list(props_sorted)
        owner_idx = 0
        actual_owners = 0

        for hh in households[:n_owners_target]:
            if not available:
                break
            # Prefer a property in hh's home zone
            zone_match = [p for p in available if p.zone == hh.home_zone]
            prop = zone_match[0] if zone_match else available[0]
            available.remove(prop)

            # Check deposit feasibility at this property's price
            if not self.credit.is_feasible(prop.estimated_value, hh.cash, hh.income):
                # Income-rank says this household should own, but can't afford
                # at this property price. Put them in rental pool instead.
                available.insert(0, prop)  # return property to pool
                continue

            deposit = prop.estimated_value * (1.0 - self.credit.ltv_limit)
            hh.cash -= deposit
            hh.owned_properties.add(prop.id)
            hh.home_property = prop.id
            hh._mortgages[prop.id] = (prop.estimated_value, self.credit.ltv_limit, 0)
            hh._housing_asset_value = prop.estimated_value
            prop.owner_id = hh.unique_id
            prop.occupant_id = hh.unique_id
            actual_owners += 1

        # --- Allocate institutional stock ---
        # Take from the remaining available pool (lower quality / unmatched)
        inst_stock = available[:n_inst_target]
        for k, prop in enumerate(inst_stock):
            available.remove(prop)
            inst = institutions[k % len(institutions)]
            prop.owner_id = inst.unique_id
            inst.portfolio.add(prop.id)
            inst._housing_asset_value += prop.estimated_value
            prop.listed_for_rent = True  # immediately available to rent

        # --- Place renters into rental stock ---
        # Renters are all households without a home yet.
        # They go into institutional and unowned vacant properties.
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

    def _seed_price_history(self):
        """Seed price history from initial property values."""
        allocated = [
            p.purchase_anchor_price for p in self.properties if p.owner_id is not None
        ]
        if allocated:
            return [float(np.mean(allocated))]
        return [200_000.0]

    def _estimate_initial_rent(self):
        """Monthly rent from ~4.5% gross yield on median property price."""
        prices = [p.estimated_value for p in self.properties]
        return float(np.median(prices)) * 0.045 / 12.0

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
                            reservation = prop.purchase_anchor_price * 0.95
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
                            reservation = prop.purchase_anchor_price * 0.95
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
                        reservation = prop.purchase_anchor_price * 0.97
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
        Anchored to ~4% gross yield on purchase anchor price.
        """
        return max(200.0, prop.purchase_anchor_price * 0.04 / 12.0)

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
            period_avg = 200_000.0
        self._price_history.append(period_avg)

        if self.this_step_rental_transactions:
            period_rent = float(
                np.mean([t.monthly_rent for t in self.this_step_rental_transactions])
            )
            self._rent_history.append(period_rent)
            self._avg_rent = period_rent
        elif self._rent_history:
            self._rent_history.append(self._rent_history[-1])

        p_signal = price_growth_signal(self._price_history[-5:])
        r_signal = rent_growth_signal(self._rent_history[-5:])

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
