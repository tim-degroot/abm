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
import random
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
        self._init_ownership_and_rent()
        self._seed_history()
        self._avg_rent = self._estimate_initial_rent()
        self.current_macro_state = getattr(self.config.macro, "initial_state", "Neutral")

        # Per-step registers
        self.this_step_transactions = []
        self.this_step_rental_transactions = []

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

        # Transaction history
        self.all_transactions = []
        self.all_rental_transactions = []

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

    def _assign_property_to_household(self, hh, prop, is_home):
        price = prop.estimated_value
        ltv = self._draw_origination_ltv()
        deposit = price * (1.0 - ltv)

        # Feasibility
        payment = self.credit.monthly_mortgage_payment(price, ltv)
        income_ok = payment <= self.credit.dti_limit * hh.income
        if not income_ok:
            return False

        if hh.cash < deposit:
            return False

        # Commit
        hh.cash -= deposit
        hh.owned_properties.add(prop.id)
        hh._mortgages[prop.id] = (price, ltv, 0)  # (purchase_price, ltv, steps_held)
        hh._housing_asset_value += price
        prop.owner_id = hh.unique_id
        if is_home:
            hh.home_property = prop.id
            prop.occupant_id = hh.unique_id
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

        # Ownership
        props_sorted = sorted(self.properties, key=lambda p: p.quality, reverse=True)
        available = list(props_sorted)
        n_properties = len(available)

        while available and (available / n_properties <= cfg.sim.target_household_share):
            hh = random.choice(households)
            zone_match = [p for p in available if p.zone == hh.home_zone]
            prop = zone_match[0]
            is_home = hh.home_property == None
            if self._assign_property_to_household(hh, prop, is_home=is_home):
                available.remove(prop)
            # else: prop stays in pool, repeat

        for k, prop in enumerate(available):
            available.remove(prop)
            inst = institutions[k % len(institutions)]
            prop.owner_id = inst.unique_id
            inst.portfolio.add(prop.id)
            inst._housing_asset_value += prop.estimated_value
            prop.listed_for_rent = True
            prop.current_rent = None

        # Rent
        rental_pool = [p for p in self.properties if p.occupant_id is None and p.listed_for_rent]
        renter_households = [h for h in households if h.home_property is None]
        for hh in renter_households:
            zone_match = [p for p in rental_pool if p.zone == hh.home_zone]
            prop = zone_match[0] if zone_match else (rental_pool[0] if rental_pool else None)
            rental_pool.remove(prop)
            prop.occupant_id = hh.unique_id
            prop.listed_for_rent = False
            prop.current_rent = prop.estimated_value * self.config.market.initial_rent_yield / 12.0
            hh.home_property = prop.id
            hh.home_zone = prop.zone

        # Accounting
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]
        vacant_unowned = [
            p for p in self.properties if p.occupant_id is None and p.owner_id is None
        ]
        if vacant_unowned:
            raise AssertionError(
                f"Init left {len(vacant_unowned)} unowned properties; all stock must have an owner."
            )

        # Also mark any remaining vacant, owned properties as listed for rent
        # so they appear in the rental market.
        for p in self.properties:
            if p.occupant_id is None and p.owner_id is not None:
                p.listed_for_rent = True
                if p.current_rent is None:
                    p.current_rent = (
                        p.estimated_value * self.config.market.initial_rent_yield / 12.0
                    )

        self._verify_accounting()  # IS THE HOUSING COST ACTUALLY BEING ASSIGNED OR THIS IS DONE NOT TOUCHING INCOMES AND WEALTHS?

    def _verify_accounting(self, tol=1.0):
        """
        Assert the housing balance sheet is internally consistent at init:
          - total HousingAssets from properties == sum of agent housing-asset
            values (no double-counting / orphaned ownership), and
          - total mortgage debt does not exceed housing assets.
        HousingEquity is then HousingAssets - MortgageDebt by definition, so
        HousingAssets = HousingEquity + MortgageDebt holds by construction.
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
        """Seed price and rent history from initial property values."""
        allocated = [p.purchase_anchor_price for p in self.properties if p.owner_id is not None]
        if allocated:
            self._price_history = [float(np.mean(allocated))]
        else:
            self._price_history = []
        self._rent_history = [self._estimate_initial_rent()]

    def _estimate_initial_rent(self):
        """Monthly rent proxy from the configured gross yield on median property price."""
        prices = [p.estimated_value for p in self.properties]
        return float(np.median(prices)) * self.config.market.initial_rent_yield / 12.0

    def _current_avg_rent(self):
        """Average monthly rent across properties that are actually rented."""
        rents = [
            p.current_rent
            for p in self.properties
            if p.occupant_id is not None
            and p.owner_id is not None
            and p.occupant_id != p.owner_id
            and p.current_rent is not None
        ]
        if rents:
            return float(np.mean(rents))
        return self._estimate_initial_rent()

    def _plan_distress_sales(self, avg_rent):
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
            for _, prop in agent.distress_sale_candidates(avg_rent):
                if prop.id not in getattr(
                    agent, "owned_properties", set()
                ) and prop.id not in getattr(agent, "portfolio", set()):
                    continue

                selected.append(prop.id)

            if selected:
                plans[agent.unique_id] = set(selected)

        return plans

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    #  UPDATE NEEDED: LISTING FOR SALE, THEN CLEAR PURCHASES, THEN LISTING RENT, THEN CLEAR RENT CREATE
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
        # Advance macro state (Markov chain) before incomes evolve so income
        # draws in this period reflect the current macro state.
        self._advance_macro_state()
        self.credit = self.policy.modify_credit(self.credit)

        self.this_step_transactions = []
        self.this_step_rental_transactions = []

        # 1. Income evolution
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                agent.evolve_income()

        # 1b. Rent servicing before turnover/listing so defaults can search now.
        self._service_rents()

        # 1c. Age active tenancies (tracks minimum lease term for lock-in).
        self._age_tenancies()

        # 2. Mark-to-market revaluation
        self._mark_to_market()

        avg_rent = self._avg_rent
        distress_sales = self._plan_distress_sales(avg_rent)

        ownership_market = OwnershipMarket(step=self.steps)
        rental_market = RentalMarket(step=self.steps)

        # Compute each agent's action once and cache it for this step to avoid
        # inconsistencies where an agent is asked twice and may change choice.
        actions = {}
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) or isinstance(agent, InstitutionalAgent):
                purchase_candidates = self._get_purchase_candidates(agent, distress_sales)
                actions[agent.unique_id] = agent.choose_action(purchase_candidates, avg_rent)

        # 3 & 4. Listing and bidding (use cached actions)
        self._list_properties(ownership_market, rental_market, avg_rent, actions, distress_sales)
        self._submit_bids(ownership_market, rental_market, avg_rent, actions, distress_sales)

        # 5 & 6. Market clearing
        sale_txns = ownership_market.resolve()
        rental_txns = rental_market.resolve(rng=self.rng)

        # 7. Apply transactions
        self._apply_ownership_transactions(sale_txns)
        self._apply_rental_transactions(rental_txns)

        # 8. Mortgage servicing (after sales so sold mortgages are gone)
        for agent in self.agents:
            if getattr(agent, "_mortgages", None):
                agent.service_mortgages()

        # 9. Expectations
        self._update_expectations()

        # 10. Data
        self.datacollector.collect(self)
        self._sync_visual_grid()

    # ------------------------------------------------------------------
    # Rent servicing
    # ------------------------------------------------------------------

    def _tenant_can_pay_rent(self, tenant, monthly_rent) -> bool:
        if tenant is None or monthly_rent is None or monthly_rent <= 0:
            return False
        income = getattr(tenant, "income", None)
        if income is None or income <= 0:
            return False
        return monthly_rent <= income

    def _service_rents(self):
        """Collect current rents; tenants who cannot pay vacate immediately."""
        for prop in self.properties:
            if (
                prop.occupant_id is None
                or prop.owner_id is None
                or prop.occupant_id == prop.owner_id
                or prop.current_rent is None
            ):
                continue

            tenant = self._agent_map.get(prop.occupant_id)
            if not isinstance(tenant, HouseholdAgent) or tenant.home_property != prop.id:
                continue

            rent = prop.current_rent
            if not self._tenant_can_pay_rent(tenant, rent):
                prop.occupant_id = None
                prop.listed_for_rent = True
                tenant.vacate_rental()
                continue

            landlord = self._agent_map.get(prop.owner_id)
            if not isinstance(landlord, (HouseholdAgent, InstitutionalAgent)):
                continue
            tenant.pay_rent(rent)
            landlord.receive_rent(rent)

    def _age_tenancies(self):
        for prop in self.properties:
            if (
                prop.occupant_id is None
                or prop.owner_id is None
                or prop.occupant_id == prop.owner_id
            ):
                continue
            prop.tenancy_months += 1

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
                    self._property_map[pid].estimated_value for pid in agent.owned_properties
                )
            elif isinstance(agent, InstitutionalAgent) and agent.portfolio:
                agent._housing_asset_value = sum(
                    self._property_map[pid].estimated_value for pid in agent.portfolio
                )

    def _advance_macro_state(self):
        """Advance the macro state xi ∈ {Boom, Neutral, Recession} using a
        Markov transition matrix supplied in `self.config.macro`.

        The config fields are read as probabilities row-wise (from-state -> to-state).
        """
        mcfg = getattr(self.config, "macro", None)
        if mcfg is None:
            return

        current = getattr(self, "current_macro_state", "Neutral")
        r = self.rng.random()

        match current:
            case "Boom":
                probs = [
                    mcfg.boom_to_boom,
                    mcfg.boom_to_neutral,
                    mcfg.boom_to_recession,
                ]
                states = ["Boom", "Neutral", "Recession"]
            case "Recession":
                probs = [
                    mcfg.recession_to_boom,
                    mcfg.recession_to_neutral,
                    mcfg.recession_to_recession,
                ]
                states = ["Boom", "Neutral", "Recession"]
            case _:
                probs = [
                    mcfg.neutral_to_boom,
                    mcfg.neutral_to_neutral,
                    mcfg.neutral_to_recession,
                ]
                states = ["Boom", "Neutral", "Recession"]

        # Cumulative selection
        cum = 0.0
        for p, s in zip(probs, states):
            cum += float(p)
            if r < cum:
                self.current_macro_state = s
                return
        # Numerical safety: fallback to last state if rounding leaves r >= cum
        self.current_macro_state = states[-1]

    # ------------------------------------------------------------------
    # Market participation
    # ------------------------------------------------------------------

    def _list_properties(self, ownership_market, rental_market, avg_rent, actions, distress_sales):
        """
        Owners decide each period whether to sell, rent out, or hold.

        Each owned property is evaluated independently.
        Owner-occupiers choose over: hold, sell, rent_out.
        Landlord properties default to listed_for_rent unless sell is chosen.
        """
        # Register any properties already flagged as listed_for_rent (e.g., from
        # initialisation) into the per-step rental market so tenants can bid.
        for prop in self.properties:
            if prop.listed_for_rent and prop.occupant_id is None and prop.owner_id is not None:
                reservation = self._reservation_rent(prop)
                rental_market.list_property(prop.id, prop.owner_id, reservation)
                self._debug_counts["rental_listed"] += 1

        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                # Use cached action to ensure consistency between listing and bidding
                action = actions.get(agent.unique_id)
                forced_sales = distress_sales.get(agent.unique_id, set())

                for pid in list(agent.owned_properties):
                    prop = self._property_map[pid]
                    prop.listed_for_sale = False

                    if pid in forced_sales:
                        ownership_market.list_property(pid, agent.unique_id, 0.0)
                        prop.listed_for_sale = True
                        self._debug_counts["ownership_listed"] += 1
                        continue

                    if pid == agent.home_property:
                        # Owner-occupied home: sell or rent_out trigger listing
                        if action == "sell":
                            reservation = self._seller_reservation(prop, agent)
                            ownership_market.list_property(pid, agent.unique_id, reservation)
                            prop.listed_for_sale = True
                            self._debug_counts["ownership_listed"] += 1
                        elif action == "rent_out":
                            # Zero reservation is fine as long as zero bids are
                            # rejected at the market layer.
                            if prop.occupant_id is None:
                                rental_market.list_property(pid, agent.unique_id, 0.0)
                                prop.listed_for_rent = True
                                self._debug_counts["rental_listed"] += 1
                    else:
                        # Investment property: always list for rent unless selling
                        if action == "sell":
                            reservation = self._seller_reservation(prop, agent)
                            ownership_market.list_property(pid, agent.unique_id, reservation)
                            prop.listed_for_sale = True
                            self._debug_counts["ownership_listed"] += 1
                        else:
                            # Investment property: list for rent at zero reserve
                            # if vacant; zero bids are rejected in the market.
                            if prop.occupant_id is None:
                                rental_market.list_property(pid, agent.unique_id, 0.0)
                                prop.listed_for_rent = True
                                self._debug_counts["rental_listed"] += 1

            elif isinstance(agent, InstitutionalAgent):
                action = actions.get(agent.unique_id)
                forced_sales = distress_sales.get(agent.unique_id, set())

                for pid in list(agent.portfolio):
                    prop = self._property_map[pid]
                    prop.listed_for_sale = False
                    prop.listed_for_rent = False

                    if pid in forced_sales:
                        ownership_market.list_property(pid, agent.unique_id, 0.0)
                        prop.listed_for_sale = True
                        self._debug_counts["ownership_listed"] += 1
                        continue

                    if action == "sell":
                        reservation = prop.purchase_anchor_price
                        ownership_market.list_property(pid, agent.unique_id, reservation)
                        prop.listed_for_sale = True
                    else:
                        # Institutions: list for rent without a reservation,
                        # but only if vacant.
                        if prop.occupant_id is None:
                            rental_market.list_property(pid, agent.unique_id, 0.0)
                            prop.listed_for_rent = True
                            self._debug_counts["rental_listed"] += 1

    def _submit_bids(self, ownership_market, rental_market, avg_rent, actions, distress_sales):
        """Buyers and renters submit bids."""
        submitted_pairs = set()
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent):
                purchase_candidates = self._get_purchase_candidates(agent, distress_sales)
                action = actions.get(agent.unique_id)

                # Purchase bid
                if action in ("buy", "buy-to-let"):
                    # Determine feasibility using the credit envelope (LTV/DTI).
                    # Count filtered candidates for diagnostics.
                    num_candidates = len(purchase_candidates)
                    affordable = [
                        p
                        for p in purchase_candidates
                        if p.listed_for_sale and self._purchase_feasible(agent, p)
                    ]
                    self._debug_counts["ownership_bids_filtered"] += num_candidates - len(
                        affordable
                    )

                    if affordable:
                        chosen = agent.choose_property(affordable, avg_rent, purpose=action)
                        if chosen is not None:
                            bid = min(
                                agent.compute_bid(chosen, avg_rent, purpose=action),
                                self._purchase_price_ceiling(agent),
                            )
                            if bid > 0:
                                # Log full bid for diagnostics (guarded)
                                if self._debug_bid_logging:
                                    self._debug_bid_log.append(
                                        {
                                            "step": int(self.steps),
                                            "property_id": int(chosen.id),
                                            "bidder_id": int(agent.unique_id),
                                            "amount": float(bid),
                                            "bidder_type": "household",
                                            "cash": float(agent.cash),
                                            "income": float(agent.income),
                                            "expected_price_growth": float(
                                                agent.expected_price_growth
                                            ),
                                        }
                                    )
                                ownership_market.submit_bid(
                                    chosen.id, agent.unique_id, bid, "household"
                                )
                                submitted_pairs.add((agent.unique_id, chosen.id))
                                self._debug_counts["ownership_bids_submitted"] += 1
                                self._debug_counts["ownership_bid_samples"].append(bid)

                # Rental bid. Who searches for a rental this step:
                #  - Unhoused households: always (they need a home).
                #  - Housed renters still inside the minimum lease term: never
                #    (locked in).
                #  - Housed renters past the minimum term: only if their action
                #    choice is "rent". They only consider options strictly better
                #    than their current home.
                #  - Others (owner-occupiers): only if they explicitly chose rent.
                if agent.home_property is None:
                    wants_rental = True
                elif self._tenant_locked_in(agent):
                    wants_rental = False
                else:
                    wants_rental = action == "rent"

                if wants_rental:
                    rental_candidates = self._get_rental_candidates(agent)
                    if agent.home_property is not None and rental_candidates:
                        current = self._property_map.get(agent.home_property)
                        if current is not None:
                            rental_candidates = [
                                p for p in rental_candidates if p.quality > current.quality
                            ]
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

            elif isinstance(agent, InstitutionalAgent):
                purchase_candidates = self._get_purchase_candidates(agent, distress_sales)
                action = actions.get(agent.unique_id)

                if action == "buy":
                    listed = [p for p in purchase_candidates if p.listed_for_sale]
                    chosen = agent.choose_property(listed, avg_rent)
                    if chosen is not None:
                        bid = min(
                            agent.compute_bid(chosen, avg_rent),
                            self._purchase_price_ceiling(agent),
                        )
                        if bid > 0:
                            # Log institutional bid (guarded)
                            if self._debug_bid_logging:
                                self._debug_bid_log.append(
                                    {
                                        "step": int(self.steps),
                                        "property_id": int(chosen.id),
                                        "bidder_id": int(agent.unique_id),
                                        "amount": float(bid),
                                        "bidder_type": "institution",
                                        "cash": float(agent.cash),
                                        "funding_rate": float(agent.funding_rate),
                                        "expected_price_growth": float(agent.expected_price_growth),
                                    }
                                )
                            ownership_market.submit_bid(
                                chosen.id,
                                agent.unique_id,
                                bid,
                                "institution",
                                self.config.agent.inst_ltv,
                            )
                            submitted_pairs.add((agent.unique_id, chosen.id))
                            self._debug_counts["ownership_bids_submitted"] += 1
                            self._debug_counts["ownership_bid_samples"].append(bid)

        # Fire-sale pass: distressed properties are exposed to all feasible
        # buyers regardless of their action choice so liquidity can actually
        # form before mortgage servicing. Keep it to one bid per buyer to
        # avoid overcommitting cash across multiple purchases.
        distressed_ids = {pid for ids in distress_sales.values() for pid in ids}
        if distressed_ids:
            for agent in self.agents:
                if any(pair[0] == agent.unique_id for pair in submitted_pairs):
                    continue

                if isinstance(agent, HouseholdAgent):
                    candidates = [
                        self._property_map[pid]
                        for pid in distressed_ids
                        if self._property_map[pid].listed_for_sale
                        and self._property_map[pid].zone
                        in self.get_household_search_zones(agent.home_zone)
                        and self._property_map[pid].owner_id != agent.unique_id
                    ]
                    if not candidates:
                        continue
                    chosen = max(
                        candidates,
                        key=lambda prop: agent.compute_bid(prop, avg_rent, purpose="buy-to-let"),
                    )
                    bid = min(
                        agent.compute_bid(chosen, avg_rent, purpose="buy-to-let"),
                        self._purchase_price_ceiling(agent),
                    )
                    if bid > 0:
                        if self._debug_bid_logging:
                            self._debug_bid_log.append(
                                {
                                    "step": int(self.steps),
                                    "property_id": int(chosen.id),
                                    "bidder_id": int(agent.unique_id),
                                    "amount": float(bid),
                                    "bidder_type": "household",
                                    "cash": float(agent.cash),
                                    "income": float(agent.income),
                                    "expected_price_growth": float(agent.expected_price_growth),
                                    "fire_sale": True,
                                }
                            )
                        ownership_market.submit_bid(chosen.id, agent.unique_id, bid, "household")
                        submitted_pairs.add((agent.unique_id, chosen.id))
                        self._debug_counts["ownership_bids_submitted"] += 1
                        self._debug_counts["ownership_bid_samples"].append(bid)

                elif isinstance(agent, InstitutionalAgent):
                    candidates = [
                        self._property_map[pid]
                        for pid in distressed_ids
                        if self._property_map[pid].listed_for_sale
                        and self._property_map[pid].owner_id != agent.unique_id
                        # Activity hurdle applies to fire-sales too: institutions
                        # only snap up distressed stock in the high-yield corner.
                        and self._inst_yield_ok(self._property_map[pid], avg_rent)
                        and self._purchase_feasible(agent, self._property_map[pid])
                    ]
                    if not candidates:
                        continue
                    chosen = max(
                        candidates,
                        key=lambda prop: agent.compute_bid(prop, avg_rent),
                    )
                    bid = min(
                        agent.compute_bid(chosen, avg_rent),
                        self._purchase_price_ceiling(agent),
                    )
                    if bid > 0:
                        if self._debug_bid_logging:
                            self._debug_bid_log.append(
                                {
                                    "step": int(self.steps),
                                    "property_id": int(chosen.id),
                                    "bidder_id": int(agent.unique_id),
                                    "amount": float(bid),
                                    "bidder_type": "institution",
                                    "cash": float(agent.cash),
                                    "funding_rate": float(agent.funding_rate),
                                    "expected_price_growth": float(agent.expected_price_growth),
                                    "fire_sale": True,
                                }
                            )
                        ownership_market.submit_bid(
                            chosen.id,
                            agent.unique_id,
                            bid,
                            "institution",
                            self.config.agent.inst_ltv,
                        )
                        submitted_pairs.add((agent.unique_id, chosen.id))
                        self._debug_counts["ownership_bids_submitted"] += 1
                        self._debug_counts["ownership_bid_samples"].append(bid)

    # ------------------------------------------------------------------
    # Candidate set construction
    # ------------------------------------------------------------------

    def _inst_expected_yield(self, prop, avg_rent):
        """
        Gross expected rental yield an institution would earn on a property:
        expected gross annual rent / current price. Drives the activity hurdle: institutions only chase the high-yield corner,
        so under loose credit (high prices ⇒ low yields) they sit out and leave
        the marginal bid to leveraged households, while under tight credit
        (depressed prices ⇒ high yields) they step in — the plan §2.3 regime
        switch.
        """
        price = prop.estimated_value
        if price <= 0:
            return 0.0
        gross_annual_rent = (  # monthly rent * 12 → annual rent for yield computation
            estimate_market_rent(prop.quality, avg_rent, self.config.valuation.quality_sensitivity)
            * 12.0
        )
        return gross_annual_rent / price

    def _inst_yield_ok(self, prop, avg_rent):
        """True iff the property clears the institutional activity hurdle."""
        return self._inst_expected_yield(prop, avg_rent) >= self.config.agent.inst_min_yield

    def _get_purchase_candidates(self, agent, distress_sales=None):
        """Households search locally; institutions see all sale listings."""
        if isinstance(agent, InstitutionalAgent):
            candidates = [
                p
                for p in self.properties
                if p.listed_for_sale and p.owner_id != agent.unique_id
                # Activity hurdle: only bid where expected yield >= inst_min_yield.
                # Below the threshold the institution does not bid at all.
                and self._inst_yield_ok(p, self._avg_rent)
            ]
        else:
            zones = self.get_household_search_zones(agent.home_zone)
            candidates = [
                p
                for p in self.properties
                if p.zone in zones and p.listed_for_sale and p.owner_id != agent.unique_id
            ]

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
            return current_due + new_payment <= self.credit.dti_limit * agent.income

        if isinstance(agent, InstitutionalAgent):
            ltv = self.config.agent.inst_ltv
            deposit = prop.estimated_value * (1.0 - ltv)
            return agent.cash >= current_due + deposit

        return False

    def _purchase_price_ceiling(self, agent):
        """Highest purchase price an agent can safely bid after servicing."""
        current_due = 0.0
        if hasattr(agent, "mortgage_payment_due"):
            current_due = agent.mortgage_payment_due()

        available_cash = max(0.0, agent.cash - current_due)
        if isinstance(agent, HouseholdAgent):
            return self.credit.max_affordable_price(available_cash, agent.income)
        if isinstance(agent, InstitutionalAgent):
            ltv = self.config.agent.inst_ltv
            if ltv >= 1.0:
                return float("inf")
            return available_cash / max(1e-9, 1.0 - ltv)
        return 0.0

    def _tenant_locked_in(self, agent):
        if agent.home_property is None or not agent.is_renter:
            return False
        home = self._property_map.get(agent.home_property)
        if home is None:
            return False
        return home.tenancy_months < self.config.market.min_lease_months

    def _get_rental_candidates(self, agent):
        """
        Listed, vacant rentals the agent can afford to bid on (never one it owns).

        An UNHOUSED household (no owned home AND no active lease, i.e.
        home_property is None) is not tied to any neighbourhood, so it searches
        the ENTIRE market — guaranteeing it is always an active rental searcher
        and can re-house whenever any rental supply exists. This is what stops
        agents who lose their home (distress sale, lease non-renewal) from
        silently dropping out of the market because their home zone happens to
        have no vacancies. A housed renter looking to move stays restricted to
        its local search zones.
        """
        if not hasattr(agent, "compute_rent_bid"):
            return []

        max_rent = agent.compute_rent_bid()
        if max_rent <= 0:
            return []

        listed = [
            p
            for p in self.properties
            if p.listed_for_rent
            and p.occupant_id is None
            and p.owner_id != agent.unique_id
            and self._reservation_rent(p) <= max_rent
        ]
        if agent.home_property is None:
            return listed  # unhoused: search the whole market
        zones = self.get_household_search_zones(agent.home_zone)
        return [p for p in listed if p.zone in zones]

    def _reservation_rent(self, prop):
        """
        Minimum monthly rent a landlord will accept.

        Anchored to the configured gross annual yield on purchase anchor price,
        converted to a monthly rent.
        """
        mcfg = self.config.market
        yield_rent = (
            prop.purchase_anchor_price
            * mcfg.landlord_reservation_yield
            / 12.0
        )
        return max(mcfg.min_reservation_rent, yield_rent)

    def _seller_reservation(self, prop, seller_agent):
        """
        Compute a seller's reservation price incorporating loss aversion.

        If the purchase anchor `p_0` exceeds a reasonable expected market price,
        the seller demands a premium proportional to the shortfall scaled by
        a loss-aversion coefficient (>1). Institutions are exempt.
        """
        cfg = self.config
        p0 = prop.purchase_anchor_price
        # Use last observed market price as expected market price (fallback to anchor)
        expected = self._price_history[-1] if self._price_history else p0

        # Institution sellers are not loss-averse
        if isinstance(seller_agent, InstitutionalAgent):
            return p0

        # Choose lambda depending on whether owner-occupier or landlord
        if isinstance(seller_agent, HouseholdAgent) and seller_agent.is_owner_occupier:
            lam = cfg.market.loss_aversion_owner
        else:
            lam = cfg.market.loss_aversion_landlord

        if p0 <= expected:
            # No nominal loss relative to expectation: accept near anchor/discount
            return p0

        # Nominal loss: increase reservation above anchor by lambda * shortfall
        shortfall = p0 - expected
        return p0 + lam * shortfall

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

            # Vacate the current tenant on sale (their home record is cleared by
            # vacate_rental). Crucially also clear occupant_id here so it cannot
            # go stale: if the buyer does not move in (e.g. an institution or a
            # landlord already housed elsewhere), the unit must read as vacant,
            # not still-occupied by a tenant who has left. A stale occupant_id
            # silently breaks the occupant/home invariant and lets two agents end
            # up "living" in one property. If the buyer is moving in,
            # acquire_property overwrites occupant_id below.
            prev_occupant_id = prop.occupant_id
            if prev_occupant_id is not None and prev_occupant_id != txn.seller_id:
                prev_occupant = self._agent_map.get(prev_occupant_id)
                if isinstance(prev_occupant, HouseholdAgent):
                    prev_occupant.vacate_rental()
                prop.occupant_id = None

            seller.release_property(prop, txn.price)

            # If the seller was OCCUPYING the unit (an owner-occupier selling the
            # home they live in — e.g. a distress sale), the eviction block above
            # was skipped because the occupant was the seller. They are now
            # unhoused: explicitly reset their housing state to "searching" and
            # clear the unit's occupancy so it does not keep the departed seller
            # as a ghost occupant. This guarantees the seller re-enters the rental
            # search next step (see _get_rental_candidates) instead of silently
            # dropping out. If the buyer moves in, acquire_property re-sets
            # occupant_id below.
            if prop.occupant_id == txn.seller_id:
                prop.occupant_id = None
                if isinstance(seller, HouseholdAgent):
                    seller.home_property = None

            prop.owner_id = txn.buyer_id
            prop.purchase_anchor_price = txn.price

            prop.listed_for_sale = False
            prop.listed_for_rent = False
            prop.current_rent = None

            buyer.acquire_property(prop, txn.price, origination_ltv=txn.origination_ltv)

            self.policy.on_transaction(txn, self)

        self.this_step_transactions = transactions
        self.all_transactions.extend(transactions)

    def _apply_rental_transactions(self, transactions):
        applied = []
        self._rental_apply_counts = {
            "awarded": len(transactions),
            "applied": 0,
            "missing_parties": 0,
            "stale_landlord": 0,
            "owner_occupied": 0,
            "non_household_tenant": 0,
            "rent_unaffordable": 0,
            "insufficient_cash": 0,
        }
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            tenant = self._agent_map.get(txn.tenant_id)
            landlord = self._agent_map.get(txn.landlord_id)
            if tenant is None or landlord is None:
                self._rental_apply_counts["missing_parties"] += 1
                continue

            # Ownership transactions are applied BEFORE rentals, so a property
            # can be both sold and let in the same step (it was listed in both
            # markets while vacant). Drop the rental award if it is now stale:
            # the landlord no longer owns the unit (it sold), or the new owner
            # moved in (owner-occupied). Otherwise a tenant would move in on top
            # of the new owner-occupier and both would claim it as home.
            if prop.owner_id != txn.landlord_id:
                self._rental_apply_counts["stale_landlord"] += 1
                continue
            if prop.occupant_id is not None and prop.occupant_id == prop.owner_id:
                self._rental_apply_counts["owner_occupied"] += 1
                continue
            if not isinstance(tenant, HouseholdAgent):
                self._rental_apply_counts["non_household_tenant"] += 1
                continue
            if not self._tenant_can_pay_rent(tenant, txn.monthly_rent):
                self._rental_apply_counts["rent_unaffordable"] += 1
                continue

            # Evict previous occupant if any
            if prop.occupant_id is not None and prop.occupant_id != txn.tenant_id:
                prev = self._agent_map.get(prop.occupant_id)
                if isinstance(prev, HouseholdAgent):
                    prev.vacate_rental()

            prop.occupant_id = txn.tenant_id
            prop.tenancy_months = 0
            prop.listed_for_rent = False
            prop.current_rent = txn.monthly_rent

            if isinstance(tenant, HouseholdAgent):
                # Free the tenant's previous residence so it cannot retain a stale
                # occupant pointer after the household moves.
                old_home = tenant.home_property
                if old_home is not None and old_home != prop.id:
                    old_prop = self._property_map.get(old_home)
                    if old_prop is not None and old_prop.occupant_id == tenant.unique_id:
                        old_prop.occupant_id = None
                        if old_home not in tenant.owned_properties:
                            old_prop.listed_for_rent = True
                tenant.move_into_rental(prop)
                tenant.pay_rent(txn.monthly_rent)

            if isinstance(landlord, HouseholdAgent):
                landlord.receive_rent(txn.monthly_rent)
            elif isinstance(landlord, InstitutionalAgent):
                landlord.receive_rent(txn.monthly_rent)

            self.policy.on_rental_transaction(txn, self)
            applied.append(txn)
            self._rental_apply_counts["applied"] += 1

        self.this_step_rental_transactions = applied
        self.all_rental_transactions.extend(applied)

    # ------------------------------------------------------------------
    # Expectation updates
    # ------------------------------------------------------------------

    def _update_expectations(self):
        min_txns_for_signal = 3

        if self.this_step_transactions and len(self.this_step_transactions) >= min_txns_for_signal:
            period_avg = float(np.median([t.price for t in self.this_step_transactions]))
        elif self._price_history:
            period_avg = self._price_history[-1]
        else:
            period_avg = None
        if period_avg is not None:
            self._price_history.append(period_avg)

        current_avg_rent = self._current_avg_rent()
        self._rent_history.append(current_avg_rent)
        self._avg_rent = current_avg_rent

        # Append market state for institutional price forecasts
        hh_agents = [a for a in self.agents if isinstance(a, HouseholdAgent)]
        avg_ltv = (
            sum(
                ltv for _, ltv, _ in sum((list(a._mortgages.values()) for a in hh_agents), [])
            )
            / max(sum(len(a._mortgages) for a in hh_agents), 1)
        )
        inst_agents = [a for a in self.agents if isinstance(a, InstitutionalAgent)]
        inst_share = sum(len(a.portfolio) for a in inst_agents) / max(len(self.properties), 1)

        self._state_history.append({
            "price": period_avg if period_avg is not None else (self._price_history[-1] if self._price_history else 0.0),
            "rent": current_avg_rent,
            "volume": len(self.this_step_transactions),
            "macro": getattr(self, "current_macro_state", "Neutral"),
            "avg_ltv": avg_ltv,
            "inst_share": inst_share,
        })

        window = self.config.expectations.signal_window
        valid_prices = [p for p in self._price_history if p is not None and p > 0]
        valid_rents = [r for r in self._rent_history if r is not None and r > 0]
        p_signal = price_growth_signal(valid_prices[-window:]) if len(valid_prices) >= 2 else 0.0
        r_signal = rent_growth_signal(valid_rents[-window:]) if len(valid_rents) >= 2 else 0.0

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
