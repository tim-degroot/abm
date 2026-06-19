"""
Main simulation class.

Orchestrates initalisation and the economic loop each step:
  income_evolution
  → expectations
  → valuation
  → action selection
  → property selection
  → bidding
  → auction
  → ownership/rental transfers + balance-sheet updates
  → mortgage servicing
  → mark-to-market asset revaluation
  → expectation update
"""

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

from config import Config
from policies import NoPolicy
from properties import Property
from credit import CreditEnvironment
from markets import OwnershipMarket, RentalMarket
from agents import HouseholdAgent, InstitutionalAgent
from expectations import price_growth_signal, rent_growth_signal
from metrics import MODEL_REPORTERS


class HousingModel(mesa.Model):
    def __init__(self, config, policy=None):
        self.config = config
        cfg = self.config

        super().__init__(rng=np.random.default_rng(cfg.sim.seed))  # set rng

        # Create grid and housing
        self.n_households = cfg.sim.n_households
        self.n_institutions = cfg.sim.n_institutions
        self.grid_rows = cfg.spatial.grid_rows
        self.grid_cols = cfg.spatial.grid_cols
        self.n_zones = cfg.spatial.n_zones
        self.house_grid_rows, self.house_grid_cols = self._house_grid_dimensions(
            cfg.sim.n_properties
        )
        self.grid = OrthogonalVonNeumannGrid(
            (self.house_grid_rows, self.house_grid_cols),
            torus=True,
            capacity=1,
            random=self.random,
        )
        self.grid.create_property_layer("house_status", default_value=0, dtype=int)
        self._zone_adjacency = self._build_zone_adjacency(self.grid_rows, self.grid_cols)
        self.properties = self._init_properties(cfg.sim.n_properties, self.n_zones)
        self._property_map = {p.id: p for p in self.properties}

        # Set policy
        self.policy = policy if policy is not None else NoPolicy()

        # Set credit environment
        self.credit = CreditEnvironment(**cfg.credit.model_dump())

        # Create agents and initial state
        self._init_agents(cfg.sim.n_households, cfg.sim.n_institutions)
        self._agent_map = {a.unique_id: a for a in self.agents}

        # Per-step registers (must exist before step 0 rental auction)
        self.this_step_transactions = []
        self.this_step_rental_transactions = []
        self.all_transactions = []
        self.all_rental_transactions = []

        self._init_ownership_and_rent()
        self._seed_history()
        self.current_macro_state = getattr(self.config.macro, "initial_state", "Neutral")

        # Debug logs
        self._debug_counts = {
            "rental_listed": 0,
            "ownership_listed": 0,
            "rental_bids_submitted": 0,
            "rental_bids_filtered": 0,
            "ownership_bids_submitted": 0,
            "ownership_bids_filtered": 0,
            "ownership_bid_samples": [],
            "rental_bid_samples": [],
        }
        self._debug_bid_log = []
        self._debug_rental_bid_log = []

        # Market state history for institutional price forecasts
        self._state_history: list[dict] = []

        self.datacollector = DataCollector(model_reporters=MODEL_REPORTERS)
        self.datacollector.collect(self)
        cfg_debug = getattr(self.config, "debug", None)
        self._debug_bid_logging = True

        self._sync_visual_grid()

    # ------------------------------------------------------------------
    # Spatial
    # ------------------------------------------------------------------

    def _build_zone_adjacency(self, rows, cols):
        """
        2D toroidal grid, von Neumann topology.
        """

        def zone_id(r, c):
            return (r % rows) * cols + (c % cols)

        adjacency = {}
        for r in range(rows):
            for c in range(cols):
                z = zone_id(r, c)
                adjacency[z] = frozenset(
                    {
                        z,
                        zone_id(r - 1, c),  # up
                        zone_id(r + 1, c),  # down
                        zone_id(r, c - 1),  # left
                        zone_id(r, c + 1),  # right
                    }
                )
        return adjacency

    def _house_grid_dimensions(self, n_properties):
        rows = int(np.floor(np.sqrt(n_properties)))
        while rows > 1 and n_properties % rows != 0:
            rows -= 1
        if rows <= 1:
            cols = int(np.ceil(np.sqrt(n_properties)))
            rows = int(np.ceil(n_properties / cols))
            return rows, cols
        cols = n_properties // rows
        return rows, cols

    def _sync_visual_grid(self):
        """Keep the toroidal house grid aligned with household occupancy."""
        for cell in self.grid.all_cells:
            for agent in list(cell.agents):
                try:
                    cell.remove_agent(agent)
                except Exception:
                    pass
                agent.cell = None
                agent.pos = None

        cells = sorted(self.grid.all_cells, key=lambda cell: cell.coordinate)
        cell_by_coord = {cell.coordinate: cell for cell in cells}

        for prop, cell in zip(self.properties, cells):
            prop.grid_coord = cell.coordinate

        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) and agent.home_property is not None:
                prop = self._property_map.get(agent.home_property)
                if prop is None or prop.grid_coord is None:
                    continue
                cell = cell_by_coord.get(prop.grid_coord)
                if cell is None:
                    continue
                cell.add_agent(agent)
                agent.cell = cell
                agent.pos = cell.coordinate

        self._update_house_status_layer(cells)

    def _update_house_status_layer(self, cells=None):
        """Encode ownership and occupancy state into a grid property layer."""
        if not hasattr(self.grid, "set_property"):
            return

        if cells is None:
            cells = sorted(self.grid.all_cells, key=lambda cell: cell.coordinate)

        self.grid.set_property("house_status", 0)

        for prop, cell in zip(self.properties, cells):
            status = self._house_status_for_property(prop)
            cell.house_status = status

    def _house_status_for_property(self, prop):
        owner = self._agent_map.get(prop.owner_id)
        if prop.listed_for_sale:
            return 6
        if prop.occupant_id is None:
            return 5
        if prop.occupant_id == prop.owner_id:
            if owner.is_landlord:
                return 2  # owner-occupied landlord household
            return 1  # owner-occupied non-landlord
        if isinstance(owner, HouseholdAgent):
            return 3  # household-owned rental occupied by tenant
        if isinstance(owner, InstitutionalAgent):
            return 4  # institution-owned rental occupied by tenant

    def get_household_search_zones(self, home_zone):
        return self._zone_adjacency[home_zone]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_properties(self, n_properties, n_zones):
        """
        Generate housing stock within zones.
        Each zone gets at least floor(n_properties / n_zones) properties,
        with the remainder distributed one-per-zone from zone 0 up.
        """
        pcfg = self.config.property_init
        zone_means = self.rng.normal(0.0, pcfg.zone_quality_sd, n_zones)

        # Distribute properties evenly
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

        base_price = pcfg.init_base_price
        price_sensitivity = pcfg.init_price_quality_sensitivity

        props = []
        for i in range(n_properties):
            anchor = base_price + price_sensitivity * float(q_std[i])
            prop = Property(
                id=i,
                zone=zone_assignments[i],
                quality=float(q_std[i]),
                owner_id=None,
                purchase_anchor_price=anchor,
                estimated_value=anchor,
            )
            props.append(prop)

        cells = sorted(self.grid.all_cells, key=lambda cell: cell.coordinate)
        for prop, cell in zip(props, cells):
            prop.grid_coord = cell.coordinate
        return props

    def _init_agents(self, n_households, n_institutions):
        """
        Create agents with heterogeneous attributes.
        """
        acfg = self.config.agent_init
        ccfg = self.config.credit
        incomes = self.rng.lognormal(np.log(acfg.income_mean), acfg.income_sigma, n_households)
        wealth_mult = self.rng.uniform(
            acfg.wealth_income_mult_low, acfg.wealth_income_mult_high, n_households
        )
        risk_av = self.rng.lognormal(acfg.risk_aversion_mu, acfg.risk_aversion_sigma, n_households)

        for i in range(n_households):
            zone = int(i % self.n_zones)  # distribute evenly across zones
            HouseholdAgent(
                unique_id=i,
                model=self,
                income=float(incomes[i]),
                cash=float(incomes[i] * wealth_mult[i]),
                risk_aversion=float(risk_av[i]),
                home_zone=zone,
            )

        for j in range(n_institutions):
            zone = int(j % self.n_zones)
            InstitutionalAgent(
                unique_id=n_households + j,
                model=self,
                cash=float(self.rng.uniform(acfg.inst_cash_low, acfg.inst_cash_high)),
                funding_rate=ccfg.inst_funding_rate,
            )

    def _draw_origination_ltv(self):
        """Draw an origination LTV from the configured distribution."""
        ai = self.config.agent_init
        return min(
            self.credit.ltv_limit,
            float(self.rng.uniform(ai.ltv_dist_low, ai.ltv_dist_high)),
        )

    def _assign_property_to_agent(self, agent, prop, is_home, agent_type):
        price = prop.estimated_value
        ltv = self._draw_origination_ltv()
        deposit = price * (1.0 - ltv)

        # Feasibility
        if agent_type == "household":
            payment = self.credit.monthly_mortgage_payment(price, ltv)
            income_ok = payment <= self.credit.dti_limit * agent.income / 12.0
            if not income_ok:
                return False

        if agent.cash < deposit:
            return False

        # Commit
        if agent_type == "household":  # this should be unified
            agent.owned_properties.add(prop.id)
        else:
            agent.portfolio.add(prop.id)

        agent.cash -= deposit
        agent._mortgages[prop.id] = (price, ltv, 0)  # (purchase_price, ltv, steps_held)
        agent._housing_asset_value += price
        prop.owner_id = agent.unique_id

        if is_home:
            agent.home_property = prop.id
            prop.occupant_id = agent.unique_id
        else:
            prop.listed_for_rent = True
        return True

    def _init_ownership_and_rent(self):
        """
        Allocate properties and derive balance sheets.
        """
        cfg = self.config

        households = sorted(
            [a for a in self.agents if isinstance(a, HouseholdAgent)],
            key=lambda h: h.income,
            reverse=True,
        )
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]

        # Ownership — random allocation per NewPlan §17
        available = list(self.properties)
        self.rng.shuffle(available)

        self.rng.shuffle(households)
        for hh in households:
            if not available:
                break
            if self.rng.random() > cfg.property_init.init_ownership_prob:
                continue
            zone_props = [p for p in available if p.zone == hh.home_zone]
            if not zone_props:
                continue
            prop = zone_props[0]
            if self._assign_property_to_agent(
                hh, prop, is_home=True, agent_type="household"
            ):
                available.remove(prop)

        for k, prop in enumerate(list(available)):
            inst = institutions[k % len(institutions)]
            if self._assign_property_to_agent(
                inst, prop, is_home=False, agent_type="institution"
            ):
                available.remove(prop)

        # Step 0 rental market — house all unhoused via auction
        renter_households = [h for h in households if h.home_property is None]
        if renter_households:
            rental_market = RentalMarket(step=0)
            for prop in self.properties:
                if prop.occupant_id is None and prop.listed_for_rent:
                    rental_market.list_property(prop.id, prop.owner_id, 0.0)
            for hh in renter_households:
                candidates = self._get_rental_candidates(hh)
                if candidates:
                    rent_bid = hh.compute_rent_bid()
                    if rent_bid > 0:
                        for c in candidates:
                            rental_market.submit_bid(c.id, hh.unique_id, rent_bid)
            rental_txns = rental_market.resolve()
            self._apply_rental_transactions(rental_txns)

        self._verify_accounting()

    def _verify_accounting(self, tol=1.0):
        """
        Assert the housing balance sheet is internally consistent at init:
          - total HousingAssets from properties == sum of agent housing-asset
            values (no double-counting / orphaned ownership), and
          - total mortgage debt does not exceed housing assets.
        """
        props_assets = sum(p.estimated_value for p in self.properties if p.owner_id is not None)
        agent_assets = sum(getattr(a, "_housing_asset_value", 0.0) for a in self.agents)
        assert abs(props_assets - agent_assets) <= tol, (
            f"Housing assets mismatch: properties={props_assets:.2f} " f"agents={agent_assets:.2f}"
        )
        debt = 0.0
        for a in self.agents:
            for orig, ltv, held in getattr(a, "_mortgages", {}).values():
                debt += self.credit.outstanding_principal(orig, ltv, held)
        assert (
            debt <= props_assets + tol
        ), f"Mortgage debt {debt:.2f} exceeds housing assets {props_assets:.2f}"

    def _seed_history(self):
        """Seed price history from initial property values."""
        allocated = [p.purchase_anchor_price for p in self.properties if p.owner_id is not None]
        if allocated:
            self._price_history = [float(np.mean(allocated))]
        else:
            self._price_history = []

    def _plan_distress_sales(self):  # too extreme
        """
        Pick forced sales for agents whose cash cannot cover mortgage servicing.

        Properties are sold in ascending expected utility-loss order. When an
        agent is under servicing pressure, every owned property is listed so
        the weakest assets can clear first and the agent can avoid default.
        """
        plans = {}
        for agent in self.agents:
            if not hasattr(agent, "mortgage_payment_due"):
                continue

            if (agent.mortgage_payment_due() - agent.cash) <= 0:
                continue

            selected = []
            for _, prop in agent.distress_sale_candidates():
                if prop.id not in getattr(
                    agent, "owned_properties", set()
                ) and prop.id not in getattr(agent, "portfolio", set()):
                    continue

                selected.append(prop.id)

            if selected:
                plans[agent.unique_id] = set(selected)

        return plans

    # ------------------------------------------------------------------
    # Main Step
    # ------------------------------------------------------------------

    def step(self):
        """
        One model period:
          1. Policy hooks
          2. Income evolution
          3. Service mortgage/rent
          4. Expire leases
          5. List properties
          6. Submit bids
          7. Clear ownership market
          8. Clear rental market
          9. Apply transactions
        """
        self.policy.on_step_start(self)
        ### Fill in update hooks

        self.this_step_transactions = []
        self.this_step_rental_transactions = []

        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                agent.evolve_income()

        self._service_rents()

        self._age_tenancies()
        self._expire_leases()

        self._mark_to_market()

        distress_sales = self._plan_distress_sales()

        ownership_market = OwnershipMarket(step=self.steps)
        rental_market = RentalMarket(step=self.steps)

        actions = {}
        for agent in self.agents:
            purchase_candidates = self._get_purchase_candidates(agent, listed_only=False)
            if isinstance(agent, HouseholdAgent):
                actions[agent.unique_id] = agent.choose_action(purchase_candidates)
            elif isinstance(agent, InstitutionalAgent):
                actions[agent.unique_id] = agent.choose_action(purchase_candidates)

        self._list_properties(ownership_market, rental_market, actions, distress_sales)
        self._submit_bids(ownership_market, rental_market, actions)

        sale_txns = ownership_market.resolve()
        rental_txns = rental_market.resolve()

        self._apply_ownership_transactions(sale_txns)
        self._apply_rental_transactions(rental_txns)

        for agent in self.agents:
            if getattr(agent, "_mortgages", None):
                agent.service_mortgages()

        self.datacollector.collect(self)
        self._sync_visual_grid()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tenant_can_pay_rent(self, tenant, monthly_rent) -> bool:
        income = getattr(tenant, "income", None)
        return monthly_rent <= income

    def _service_rents(self):
        """Collect current rents; tenants who cannot pay vacate immediately."""
        for prop in self.properties:
            if prop.occupant_id is None or prop.occupant_id == prop.owner_id:
                continue

            tenant = self._agent_map.get(prop.occupant_id)
            rent = prop.current_rent
            if not self._tenant_can_pay_rent(tenant, rent):
                prop.occupant_id = None
                prop.tenancy_months = 0
                prop.listed_for_rent = True
                tenant.vacate_rental()
                continue

            landlord = self._agent_map.get(prop.owner_id)
            tenant.pay_rent(rent)
            landlord.receive_rent(rent)

    def _age_tenancies(self):
        for prop in self.properties:
            if prop.occupant_id is None or prop.occupant_id == prop.owner_id:
                continue
            prop.tenancy_months += 1

    def _expire_leases(self):
        """
        Randomly some rentals each period, subject to a minimum lease term.
        """
        mcfg = self.config.market

        for prop in self.properties:
            if (
                prop.occupant_id is None
                or prop.owner_id is None
                or prop.occupant_id == prop.owner_id
            ):
                continue

            prop.tenancy_months += 1
            prob = (
                mcfg.normal_exit_prob
                if prop.tenancy_months >= mcfg.min_tenancy
                else mcfg.early_exit_prob
            )
            if prob <= 0 or self.rng.random() >= prob:
                continue

            tenant = self._agent_map.get(prop.occupant_id)
            prop.occupant_id = None
            prop.tenancy_months = 0
            prop.listed_for_rent = True
            tenant.vacate_rental()

    def _mark_to_market(self):
        """
        Update estimated_value on all properties, giving agents a continuously updated balance sheet.
        """
        if len(self._price_history) >= 2:  # dig into price history, may not match requirements here
            growth = (
                self.mean_price_history[-1] - self.mean_price_history[0]
            ) / self.mean_price_history[0]
        else:
            growth = 0.0

        for prop in self.properties:
            prop.estimated_value = prop.estimated_value * (1.0 + growth)

        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) and agent.owned_properties:
                agent._housing_asset_value = sum(
                    self._property_map[pid].estimated_value for pid in agent.owned_properties
                )
            elif isinstance(agent, InstitutionalAgent) and agent.portfolio:
                agent._housing_asset_value = sum(
                    self._property_map[pid].estimated_value for pid in agent.portfolio
                )

    def _list_properties(self, ownership_market, rental_market, actions, distress_sales):
        """
        Owners decide each period whether to sell, rent out, or hold.
        """
        for prop in self.properties:
            if prop.listed_for_rent:
                rental_market.list_property(prop.id, prop.owner_id, 0.0)
                self._debug_counts["rental_listed"] += 1

        for agent in self.agents:
            action = actions.get(agent.unique_id)
            forced_sales = distress_sales.get(agent.unique_id, set())

            owned = getattr(agent, "owned_properties", None) or getattr(agent, "portfolio", set())
            for pid in list(owned):
                prop = self._property_map[pid]
                prop.listed_for_sale = False

                if pid in forced_sales:
                    ownership_market.list_property(pid, agent.unique_id, 0.0)
                    prop.listed_for_sale = True
                    self._debug_counts["ownership_listed"] += 1
                    continue

                if hasattr(agent, "home_property") and pid == agent.home_property:
                    if action == "sell":
                        ownership_market.list_property(pid, agent.unique_id, 0.0)
                        prop.listed_for_sale = True
                        self._debug_counts["ownership_listed"] += 1
                    elif action == "rent_out":
                        if prop.occupant_id is None:
                            rental_market.list_property(pid, agent.unique_id, 0.0)
                            prop.listed_for_rent = True
                            self._debug_counts["rental_listed"] += 1
                else:
                    if action == "sell":
                        ownership_market.list_property(pid, agent.unique_id, 0.0)
                        prop.listed_for_sale = True
                        self._debug_counts["ownership_listed"] += 1
                    else:
                        if prop.occupant_id is None:
                            rental_market.list_property(pid, agent.unique_id, 0.0)
                            prop.listed_for_rent = True
                            self._debug_counts["rental_listed"] += 1

    def _submit_bids(self, ownership_market, rental_market, actions):
        """Buyers and renters submit bids."""
        submitted_pairs = set()
        for agent in self.agents:
            purchase_candidates = self._get_purchase_candidates(agent)
            action = actions.get(agent.unique_id)

            if action in ("buy", "buy-to-let", "acquire"):
                affordable = [
                    p
                    for p in purchase_candidates
                    if p.listed_for_sale and self._purchase_feasible(agent, p)
                ]
                if not affordable:
                    continue
                chosen = agent.choose_property(affordable)
                if chosen is None:
                    continue
                bid = min(
                    agent.compute_bid(chosen),
                    self._purchase_price_ceiling(agent),
                )
                if bid <= 0:
                    continue
                if self._debug_bid_logging:
                    self._debug_bid_log.append(
                        {
                            "step": int(self.steps),
                            "property_id": int(chosen.id),
                            "bidder_id": int(agent.unique_id),
                            "amount": float(bid),
                            "bidder_type": (
                                "household"
                                if isinstance(agent, HouseholdAgent)
                                else "institution"
                            ),
                            "cash": float(agent.cash),
                            "income": float(agent.income),
                            "expected_price_growth": float(agent.expected_price_growth),
                        }
                    )
                bidder_type = (
                    "household" if isinstance(agent, HouseholdAgent) else "institution"
                )
                ownership_market.submit_bid(
                    chosen.id, agent.unique_id, bid, bidder_type
                )
                submitted_pairs.add((agent.unique_id, chosen.id))
                self._debug_counts["ownership_bids_submitted"] += 1
                self._debug_counts["ownership_bid_samples"].append(bid)
            elif action == "rent":
                rental_candidates = self._get_rental_candidates(agent)
                if rental_candidates:
                    rent_bid = agent.compute_rent_bid()
                    if rent_bid > 0:
                        for chosen in rental_candidates:
                            if self._debug_bid_logging:
                                self._debug_rental_bid_log.append(
                                    {
                                        "step": int(self.steps),
                                        "property_id": int(chosen.id),
                                        "tenant_id": int(agent.unique_id),
                                        "amount": float(rent_bid),
                                    }
                                )
                            rental_market.submit_bid(
                                chosen.id,
                                agent.unique_id,
                                rent_bid,
                            )
                            self._debug_counts["rental_bids_submitted"] += 1
                            self._debug_counts["rental_bid_samples"].append(rent_bid)

    def _get_purchase_candidates(self, agent, listed_only=True):
        """Households search locally, institutions see all properties.

        When listed_only=True (default), filters to properties currently listed
        for sale. Set listed_only=False for Stage 1 action choice, which
        evaluates all feasible properties (NewPlan §10).
        """
        if isinstance(agent, InstitutionalAgent):
            candidates = [p for p in self.properties if p.owner_id != agent.unique_id]
        else:
            zones = self.get_household_search_zones(agent.home_zone)
            candidates = [
                p for p in self.properties
                if p.zone in zones and p.owner_id != agent.unique_id
            ]

        if listed_only:
            candidates = [p for p in candidates if p.listed_for_sale]

        return [p for p in candidates if self._purchase_feasible(agent, p)]

    def _purchase_feasible(self, agent, prop):
        """Feasibility check that includes current mortgage servicing burden."""
        current_due = 0.0
        if hasattr(agent, "mortgage_payment_due"):
            current_due = agent.mortgage_payment_due()

        if isinstance(agent, HouseholdAgent):
            ltv = self.credit.ltv_limit
            deposit = prop.estimated_value * (1.0 - ltv)
            if agent.cash < current_due + deposit:
                return False
            new_payment = self.credit.monthly_mortgage_payment(prop.estimated_value, ltv)
            return current_due + new_payment <= self.credit.dti_limit * agent.income / 12.0

        if isinstance(agent, InstitutionalAgent):
            ltv = self.credit.inst_ltv
            deposit = prop.estimated_value * (1.0 - ltv)
            return agent.cash >= current_due + deposit

        return False

    def _purchase_price_ceiling(self, agent):  # this is just a wrapper, integrate and remove
        """Highest purchase price an agent can safely bid after servicing."""
        current_due = 0.0
        if hasattr(agent, "mortgage_payment_due"):
            current_due = agent.mortgage_payment_due()
        available_cash = max(0.0, agent.cash - current_due)

        return self.credit.max_affordable_price(available_cash, agent.income)

    def _get_rental_candidates(self, agent):
        """
        Listed vacant rentals the agent can afford to bid on.

        An UNHOUSED household (no owned home AND no active lease, i.e.
        home_property is None) is not tied to any neighbourhood, so it searches
        the ENTIRE market — guaranteeing it is always an active rental searcher
        and can re-house whenever any rental supply exists. This is what stops
        agents who lose their home (distress sale, lease non-renewal) from
        silently dropping out of the market because their home zone happens to
        have no vacancies. A housed renter looking to move stays restricted to
        its local search zones.
        """
        listed = [
            p
            for p in self.properties
            if p.listed_for_rent and p.occupant_id is None and p.owner_id != agent.unique_id
        ]
        if agent.home_property is None:
            return listed
        zones = self.get_household_search_zones(agent.home_zone)
        return [p for p in listed if p.zone in zones]

    def _apply_ownership_transactions(self, transactions):
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            seller = self._agent_map.get(txn.seller_id)
            buyer = self._agent_map.get(txn.buyer_id)

            prev_occupant_id = prop.occupant_id
            if prev_occupant_id is not None and prev_occupant_id != txn.seller_id:  # evict tenant
                prev_occupant = self._agent_map.get(prev_occupant_id)
                if isinstance(prev_occupant, HouseholdAgent):
                    prev_occupant.vacate_rental()
                prop.occupant_id = None
            elif prop.occupant_id == txn.seller_id:  # move out as owner-occupier
                prop.occupant_id = None
                if isinstance(seller, HouseholdAgent):
                    seller.home_property = None

            seller.release_property(prop, txn.price)
            prop.owner_id = txn.buyer_id
            prop.purchase_anchor_price = txn.price

            prop.listed_for_sale = False
            prop.listed_for_rent = False
            prop.current_rent = None

            buyer.acquire_property(prop, txn.price, origination_ltv=txn.origination_ltv)

        self.this_step_transactions = transactions
        self.all_transactions.extend(transactions)

    def _apply_rental_transactions(self, transactions):
        applied = []
        self._rental_apply_counts = {
            "awarded": len(transactions),
            "stale_landlord": 0,
            "owner_occupied": 0,
            "rent_unaffordable": 0,
            "insufficient_cash": 0,
        }
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            if not prop.listed_for_rent:
                continue
            tenant = self._agent_map.get(txn.tenant_id)
            landlord = self._agent_map.get(txn.landlord_id)

            # Evict previous occupant if any
            if prop.occupant_id is not None and prop.occupant_id != txn.tenant_id:
                prev = self._agent_map.get(prop.occupant_id)
                if isinstance(prev, HouseholdAgent):
                    prev.vacate_rental()

            prop.occupant_id = txn.tenant_id
            prop.tenancy_months = 0
            prop.tenancy_months = 0
            prop.listed_for_rent = False
            prop.current_rent = txn.monthly_rent

            old_home = tenant.home_property
            if old_home is not None and old_home != prop.id:
                old_prop = self._property_map.get(old_home)
                if old_prop is not None and old_prop.occupant_id == tenant.unique_id:
                    old_prop.occupant_id = None
                    old_prop.tenancy_quarters = 0
                    if old_home not in tenant.owned_properties:
                        old_prop.listed_for_rent = True
            tenant.move_into_rental(prop)

            tenant.pay_rent(txn.monthly_rent)
            landlord.receive_rent(txn.monthly_rent)

            applied.append(txn)

        self.this_step_rental_transactions = applied
        self.all_rental_transactions.extend(applied)

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
        }
