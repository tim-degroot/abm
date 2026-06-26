"""
Main simulation: the HousingModel and its monthly step.

Step order (single, documented schedule):
  1. policy hook (apply any scheduled credit shock)
  2. income evolution (fixed macro regime)
  3. rent servicing + tenancy ageing + lease expiry
  4. mark-to-market revaluation
  5. update expectations
  6. action selection (logit over expected action values)
  7. ownership auction -> apply
  8. rental auction (losers + renters fall through) -> apply
  9. mortgage servicing
 10. record history + collect metrics
"""

from __future__ import annotations

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid

from code.settings.config import Config
from code.settings.policies import NoPolicy
from code.core.properties import Property
from code.core.credit import CreditEnvironment
from code.core.markets import OwnershipMarket, RentalMarket
from code.core.agents import HouseholdAgent, InstitutionalAgent
import code.core.expectations as exp
from code.settings.metrics import MODEL_REPORTERS


class HousingModel(mesa.Model):
    def __init__(self, config: Config | None = None, policy=None):
        self.config = config if config is not None else Config()
        cfg = self.config
        super().__init__(rng=np.random.default_rng(cfg.sim.seed))

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
        self._zone_adjacency = self._build_zone_adjacency(
            self.grid_rows, self.grid_cols, cfg.spatial.search_radius
        )
        self.properties = self._init_properties(cfg.sim.n_properties, self.n_zones)
        self._property_map = {p.id: p for p in self.properties}

        self.policy = policy if policy is not None else NoPolicy()
        self.credit = CreditEnvironment(**cfg.credit.model_dump())
        self.current_macro_state = cfg.macro.initial_state

        self._init_agents(cfg.sim.n_households, cfg.sim.n_institutions)
        self._agent_map = {a.unique_id: a for a in self.agents}

        self.this_step_transactions = []
        self.this_step_rental_transactions = []
        self.all_transactions = []
        self.all_rental_transactions = []

        self._init_ownership_and_rent()
        self._price_history: list[float] = []
        self._rent_history: list[float] = []
        self._state_history: list[dict] = []
        self._seed_history()

        self.datacollector = DataCollector(model_reporters=MODEL_REPORTERS)
        self.datacollector.collect(self)
        self._sync_visual_grid()

    # ------------------------------------------------------------------ spatial
    def _build_zone_adjacency(self, rows, cols, radius):
        """Map each zone to the set of zones within `radius` (toroidal Chebyshev)."""

        def zid(r, c):
            return (r % rows) * cols + (c % cols)

        adjacency = {}
        for r in range(rows):
            for c in range(cols):
                neigh = set()
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) + abs(dc) <= radius:  # von Neumann ball
                            neigh.add(zid(r + dr, c + dc))
                adjacency[zid(r, c)] = frozenset(neigh)
        return adjacency

    def _house_grid_dimensions(self, n):
        rows = int(np.floor(np.sqrt(n)))
        while rows > 1 and n % rows != 0:
            rows -= 1
        if rows <= 1:
            cols = int(np.ceil(np.sqrt(n)))
            return int(np.ceil(n / cols)), cols
        return rows, n // rows

    def get_household_search_zones(self, home_zone):
        return self._zone_adjacency[home_zone]

    # --------------------------------------------------------------- init stock
    def _init_properties(self, n_properties, n_zones):
        pcfg = self.config.property_init
        zone_means = self.rng.normal(0.0, pcfg.zone_quality_sd, n_zones)
        base = n_properties // n_zones
        remainder = n_properties % n_zones
        zone_counts = [base + (1 if z < remainder else 0) for z in range(n_zones)]

        raw_q, zones = [], []
        for z, count in enumerate(zone_counts):
            for _ in range(count):
                raw_q.append(zone_means[z] + self.rng.normal(0.0, pcfg.property_residual_sd))
                zones.append(z)
        q = np.array(raw_q)
        q_std = (q - q.mean()) / (q.std() + 1e-9)

        props = []
        for i in range(n_properties):
            anchor = pcfg.init_base_price + pcfg.init_price_quality_sensitivity * float(q_std[i])
            props.append(
                Property(
                    id=i,
                    zone=zones[i],
                    quality=float(q_std[i]),
                    owner_id=None,
                    purchase_anchor_price=anchor,
                    estimated_value=anchor,
                )
            )
        cells = sorted(self.grid.all_cells, key=lambda c: c.coordinate)
        for prop, cell in zip(props, cells):
            prop.grid_coord = cell.coordinate
        return props

    def _init_agents(self, n_households, n_institutions):
        acfg = self.config.agent_init
        incomes = self.rng.lognormal(np.log(acfg.income_mean), acfg.income_sigma, n_households)
        wealth_mult = self.rng.uniform(
            acfg.wealth_income_mult_low, acfg.wealth_income_mult_high, n_households
        )
        risk_av = self.rng.lognormal(acfg.risk_aversion_mu, acfg.risk_aversion_sigma, n_households)
        for i in range(n_households):
            HouseholdAgent(
                unique_id=i,
                model=self,
                income=float(incomes[i]),
                cash=float(incomes[i] * wealth_mult[i]),
                risk_aversion=float(risk_av[i]),
                home_zone=int(i % self.n_zones),
            )
        for j in range(n_institutions):
            InstitutionalAgent(
                unique_id=n_households + j,
                model=self,
                cash=float(self.rng.uniform(acfg.inst_cash_low, acfg.inst_cash_high)),
            )

    def _legacy_origination_ltv(self):
        """Random LTV for the STARTING mortgage book only (a spread of legacy loans).

        New mortgages during the run use the policy/config LTV, never this.
        """
        ai = self.config.agent_init
        return min(
            self.credit.ltv_limit, float(self.rng.uniform(ai.ltv_dist_low, ai.ltv_dist_high))
        )

    def _assign_initial_property(self, agent, prop, is_home, is_household):
        price = prop.estimated_value
        ltv = self._legacy_origination_ltv() if is_household else self.credit.inst_ltv
        rate = self.credit.mortgage_rate if is_household else self.credit.inst_funding_rate
        deposit = price * (1.0 - ltv)
        if is_household:
            payment = self.credit.monthly_mortgage_payment(price, ltv, rate)
            if payment > self.credit.dti_limit * agent.income / 12.0:
                return False
        if agent.cash < deposit:
            return False
        agent.owned_properties.add(prop.id)
        agent.cash -= deposit
        agent._mortgages[prop.id] = (price, ltv, 0, rate)
        agent._housing_asset_value += price
        prop.owner_id = agent.unique_id
        if is_home:
            agent.home_property = prop.id
            prop.occupant_id = agent.unique_id
        else:
            prop.listed_for_rent = True
        return True

    def _init_ownership_and_rent(self):
        cfg = self.config
        households = [a for a in self.agents if isinstance(a, HouseholdAgent)]
        institutions = [a for a in self.agents if isinstance(a, InstitutionalAgent)]
        self.rng.shuffle(households)

        available = list(self.properties)
        self.rng.shuffle(available)
        by_zone: dict[int, list] = {}
        for p in available:
            by_zone.setdefault(p.zone, []).append(p)

        for hh in households:
            if self.rng.random() > cfg.property_init.init_ownership_prob:
                continue
            pool = by_zone.get(hh.home_zone) or None
            prop = None
            if pool:
                prop = pool[-1]
            else:
                for z, plist in by_zone.items():
                    if plist:
                        prop = plist[-1]
                        break
            if prop is None:
                break
            if self._assign_initial_property(hh, prop, is_home=True, is_household=True):
                by_zone[prop.zone].remove(prop)
                hh.home_zone = prop.zone

        # Remaining stock -> all to institutions (no household landlords at init).
        leftover = [p for plist in by_zone.values() for p in plist if p.owner_id is None]
        self.rng.shuffle(leftover)
        for k, prop in enumerate(leftover):
            inst = institutions[k % len(institutions)]
            self._assign_initial_property(inst, prop, is_home=False, is_household=False)

        # House the still-unhoused via an initial rental auction.
        self._run_rental_market(
            [h for h in households if h.home_property is None],
            step=0,
            market_rent=self._mean_rent_or_default(),
        )
        self._verify_accounting()

    def _mean_rent_or_default(self):
        rents = [p.current_rent for p in self.properties if p.current_rent]
        if rents:
            return float(np.mean(rents))
        # default baseline rent ~ consumption value of a median home
        v = self.config.valuation
        return v.base_housing_value

    def _verify_accounting(self, tol=1.0):
        props_assets = sum(p.estimated_value for p in self.properties if p.owner_id is not None)
        agent_assets = sum(getattr(a, "_housing_asset_value", 0.0) for a in self.agents)
        assert abs(props_assets - agent_assets) <= max(
            tol, 1e-6 * props_assets
        ), f"Housing assets mismatch: properties={props_assets:.2f} agents={agent_assets:.2f}"

    def _seed_history(self):
        allocated = [p.purchase_anchor_price for p in self.properties if p.owner_id is not None]
        self._price_history = (
            [float(np.mean(allocated))]
            if allocated
            else [self.config.property_init.init_base_price]
        )
        self._rent_history = [self._mean_rent_or_default()]
        self._record_state()

    # ----------------------------------------------------------------- the step
    def step(self):
        cfg = self.config
        self.policy.on_step_start(self)

        # 2. income evolution (fixed macro regime)
        mu, sd = self._macro_income_params()
        for a in self.agents:
            if isinstance(a, HouseholdAgent):
                a.evolve_income(mu, sd)

        market_rent = self._mean_rent_or_default()

        # 3. rents, tenancies, leases
        self._service_rents(market_rent)
        self._expire_leases()

        # 4. mark to market
        self._mark_to_market()

        # precompute candidate indices for action selection
        self._build_candidate_index()

        # 5. update expectations
        self._update_expectations()

        # 6. action selection
        actions = {}
        for a in self.agents:
            cands = self._purchase_candidates(a, listed_only=False)
            actions[a.unique_id] = a.choose_action(cands, market_rent)

        # distress sales are added to the sell set
        distress = self._plan_distress_sales(market_rent)

        # 7. ownership auction
        own_market = OwnershipMarket(step=self.steps)
        self._list_for_sale(own_market, actions, distress, market_rent)
        self._sync_listed_set()
        self._submit_purchase_bids(own_market, actions, market_rent)
        sale_txns = own_market.resolve()
        self._apply_ownership_transactions(sale_txns)

        # 8. rental auction: every household left without a home this period bids
        bought = {t.buyer_id for t in self.this_step_transactions}
        renters = [
            a
            for a in self.agents
            if isinstance(a, HouseholdAgent)
            and a.unique_id not in bought
            and a.home_property is None
        ]
        self._list_for_rent(actions)
        for p in self.properties:
            if p.listed_for_rent and p.occupant_id is None:
                self._rental_by_zone[p.zone].add(p.id)
            else:
                self._rental_by_zone[p.zone].discard(p.id)
        self._run_rental_market(renters, step=self.steps, market_rent=market_rent)

        # 9. mortgage servicing
        for a in self.agents:
            if getattr(a, "_mortgages", None):
                a.service_mortgages()

        # 10. record
        self._record_state()
        self.datacollector.collect(self)
        self._sync_visual_grid()

    # ----------------------------------------------------------- macro / income
    def _macro_income_params(self):
        mcfg = self.config.macro
        state = self.current_macro_state
        if state == "Boom":
            return mcfg.boom_mean, mcfg.boom_sd
        if state == "Recession":
            return mcfg.recession_mean, mcfg.recession_sd
        return mcfg.neutral_mean, mcfg.neutral_sd

    # ----------------------------------------------------------- expectations
    def clamp(self, x):
        """Clamp a growth rate to the configured cap."""
        gcap = self.config.expectations.growth_rate_cap
        return float(max(-gcap, min(gcap, x)))

    def _update_expectations(self):
        ecfg = self.config.expectations
        s = ecfg.smoothing
        w = ecfg.signal_window

        # Global signals for institutions
        g_price = self.clamp(exp.growth_signal(self._price_history, w))
        g_rent = self.clamp(exp.growth_signal(self._rent_history, w))
        v_price = exp.volatility_signal(self._price_history, w)
        v_rent = exp.volatility_signal(self._rent_history, w)
        inst_change = exp.institutional_price_forecast(self._state_history, w)
        cur_price = self._price_history[-1] if self._price_history else 1.0
        inst_price_growth = self.clamp(inst_change / max(cur_price, 1e-9))
        inst_rent_growth = self.clamp(exp.inst_rent_growth_signal(self._state_history, w))

        # Local signals for households (bounded vision)
        zone_price_growth = self._zone_price_growth(w)
        for a in self.agents:
            if isinstance(a, HouseholdAgent):
                local_g = self.clamp(zone_price_growth.get(a.home_zone, g_price))
                npg = self.clamp(exp.adaptive_update(a.expected_price_growth, local_g, s))
                nrg = self.clamp(exp.adaptive_update(a.expected_rent_growth, g_rent, s))
                npv = exp.adaptive_update(a.expected_price_vol, v_price, s)
                nrv = exp.adaptive_update(a.expected_rent_vol, v_rent, s)
                if ecfg.household_noise_sd > 0:
                    npg = self.clamp(npg + float(self.rng.normal(0.0, ecfg.household_noise_sd)))
                    nrg = self.clamp(nrg + float(self.rng.normal(0.0, ecfg.household_noise_sd)))
                a.set_expectations(npg, nrg, max(0.0, npv), max(0.0, nrv))
            else:
                npg = self.clamp(exp.adaptive_update(a.expected_price_growth, inst_price_growth, s))
                nrg = self.clamp(exp.adaptive_update(a.expected_rent_growth, inst_rent_growth, s))
                if ecfg.inst_noise_sd > 0:
                    npg = self.clamp(npg + float(self.rng.normal(0.0, ecfg.inst_noise_sd)))
                    nrg = self.clamp(nrg + float(self.rng.normal(0.0, ecfg.inst_noise_sd)))
                a.set_expectations(npg, nrg, v_price, v_rent)

    def _zone_price_growth(self, window):
        """Per-zone recent transaction-price growth (bounded-vision signal).

        Falls back to the global growth when a zone has too few trades (at init).
        """
        out = {}
        hist = getattr(self, "_zone_price_history", None)
        if hist is None:
            return out
        for z, series in hist.items():
            out[z] = exp.growth_signal(series, window)
        return out

    # --------------------------------------------------------------- mark to mkt
    def _mark_to_market(self):
        """Nudge estimated values toward the latest MEDIAN transaction price.

        Uses the median for robustness and moves only a small fraction toward it.
        """
        if not self.this_step_transactions:
            return
        adj_pct = self.config.expectations.mark_to_market_adj_pct
        latest = float(np.median([t.price for t in self.this_step_transactions]))
        ref = float(np.median([p.estimated_value for p in self.properties]))
        if ref <= 0 or latest <= 0:
            return
        ratio = latest / ref
        adj = 1.0 + adj_pct * (ratio - 1.0)
        for p in self.properties:
            p.estimated_value = max(1.0, p.estimated_value * adj)
        for a in self.agents:
            owned = a.owned_properties
            if owned:
                a._housing_asset_value = sum(
                    self._property_map[pid].estimated_value for pid in owned
                )

    # -------------------------------------------------------------- rent / lease
    def _service_rents(self, market_rent):
        for a in self.agents:
            if isinstance(a, HouseholdAgent):
                a.rental_income_monthly = 0.0
        for prop in self.properties:
            if prop.occupant_id is None or prop.occupant_id == prop.owner_id:
                continue
            tenant = self._agent_map.get(prop.occupant_id)
            landlord = self._agent_map.get(prop.owner_id)
            rent = prop.current_rent or 0.0
            if tenant is None or rent > tenant.income:  # cannot afford -> vacate
                if tenant is not None:
                    tenant.vacate_rental()
                prop.occupant_id = None
                prop.tenancy_months = 0
                prop.listed_for_rent = True
                continue
            tenant.pay_rent(rent)
            if landlord is not None:
                landlord.receive_rent(rent)

    def _expire_leases(self):
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
            if prob > 0 and self.rng.random() < prob:
                tenant = self._agent_map.get(prop.occupant_id)
                if tenant is not None:
                    tenant.vacate_rental()
                prop.occupant_id = None
                prop.tenancy_months = 0
                prop.listed_for_rent = True

    # --------------------------------------------------------- candidate indices
    def _build_candidate_index(self):
        """Pre-compute zone→property-ID indices for candidate lookups."""
        by_zone = {z: set() for z in range(self.n_zones)}
        rental_by_zone = {z: set() for z in range(self.n_zones)}
        for p in self.properties:
            by_zone[p.zone].add(p.id)
            if p.listed_for_rent and p.occupant_id is None:
                rental_by_zone[p.zone].add(p.id)
        self._candidate_by_zone = by_zone
        self._rental_by_zone = rental_by_zone
        self._all_property_ids = set(self._property_map.keys())
        self._sync_listed_set()

    def _sync_listed_set(self):
        """Fast update of the listed-for-sale set (after _list_for_sale changes it)."""
        self._listed_set = {p.id for p in self.properties if p.listed_for_sale}

    # --------------------------------------------------------------- candidates
    def _purchase_candidates(self, agent, listed_only=True):
        if isinstance(agent, InstitutionalAgent):
            cand_ids = set(self._all_property_ids)
        else:
            zones = self.get_household_search_zones(agent.home_zone)
            cand_ids = set()
            for z in zones:
                cand_ids.update(self._candidate_by_zone[z])
        cand_ids.difference_update(agent.owned_properties)
        if listed_only:
            cand_ids.intersection_update(self._listed_set)
        return [self._property_map[pid] for pid in cand_ids]

    def _rental_candidates(self, agent):
        if not hasattr(self, "_rental_by_zone"):
            # Fallback during __init__ before index is built
            listed = [
                p
                for p in self.properties
                if p.listed_for_rent and p.occupant_id is None and p.owner_id != agent.unique_id
            ]
            if agent.home_property is None:
                return listed
            zones = self.get_household_search_zones(agent.home_zone)
            return [p for p in listed if p.zone in zones]

        if agent.home_property is None:
            cand_ids = set().union(*self._rental_by_zone.values())
        else:
            zones = self.get_household_search_zones(agent.home_zone)
            cand_ids = set()
            for z in zones:
                cand_ids.update(self._rental_by_zone[z])
        cand_ids.difference_update(agent.owned_properties)
        return [self._property_map[pid] for pid in cand_ids]

    # ----------------------------------------------------------------- distress
    def _plan_distress_sales(self, market_rent):
        """List the FEWEST properties needed to cover each agent's cash shortfall."""
        plans = {}
        for agent in self.agents:
            due = agent.mortgage_payment_due() if hasattr(agent, "mortgage_payment_due") else 0.0
            shortfall = due - agent.cash
            if shortfall <= 0:
                continue
            selected, recovered = set(), 0.0
            for hold_val, prop in agent.distress_sale_candidates(market_rent):
                if recovered >= shortfall:
                    break
                selected.add(prop.id)
                # rough recovery estimate: equity at current value
                if prop.id in agent._mortgages:
                    orig, ltv, held, rate = agent._mortgages[prop.id]
                    out = self.credit.outstanding_principal(orig, ltv, held, rate)
                else:
                    out = 0.0
                recovered += max(0.0, prop.estimated_value - out)
            if selected:
                plans[agent.unique_id] = selected
        return plans

    # --------------------------------------------------------------- listing
    def _list_for_sale(self, market, actions, distress, market_rent):
        for prop in self.properties:
            prop.listed_for_sale = False
        for agent in self.agents:
            forced = distress.get(agent.unique_id, set())
            agent_actions = actions.get(agent.unique_id, {})
            for pid in list(agent.owned_properties):
                prop = self._property_map[pid]
                if pid in forced or agent_actions.get(pid) == "sell":
                    reservation = (
                        0.0 if pid in forced else agent.reservation_price(prop, market_rent)
                    )
                    market.list_property(pid, agent.unique_id, reservation)
                    prop.listed_for_sale = True

    def _list_for_rent(self, actions):
        for agent in self.agents:
            agent_actions = actions.get(agent.unique_id, {})
            for pid in list(agent.owned_properties):
                prop = self._property_map[pid]
                if agent_actions.get(pid) == "let" and prop.occupant_id is None:
                    prop.listed_for_rent = True

    def _submit_purchase_bids(self, market, actions, market_rent):
        for agent in self.agents:
            action = actions.get(agent.unique_id, {}).get("__agent__")
            if action not in ("buy", "buy-to-let", "acquire"):
                continue
            purpose = action
            cands = self._purchase_candidates(agent, listed_only=True)
            feasible = [p for p in cands if self._purchase_feasible(agent, p, purpose)]
            if not feasible:
                continue
            chosen = agent.choose_property(feasible, purpose, market_rent)
            if chosen is None:
                continue
            bid = agent.compute_bid(chosen, purpose, market_rent)
            total_income = (
                getattr(agent, "income", 0.0) + getattr(agent, "rental_income_monthly", 0.0) * 12
            )
            existing = getattr(agent, "mortgage_payment_due", lambda: 0.0)()
            ceiling = self.credit.max_price(purpose, agent.cash, total_income, existing)
            bid = min(bid, ceiling)
            if bid <= 0:
                continue
            btype = "household" if isinstance(agent, HouseholdAgent) else "institution"
            market.submit_bid(chosen.id, agent.unique_id, bid, btype, purpose)

    def _purchase_feasible(self, agent, prop, purpose):
        total_income = (
            getattr(agent, "income", 0.0) + getattr(agent, "rental_income_monthly", 0.0) * 12
        )
        existing = getattr(agent, "mortgage_payment_due", lambda: 0.0)()
        ceiling = self.credit.max_price(purpose, agent.cash, total_income, existing)
        if ceiling <= 0:
            return False
        ltv = self.credit.origination_ltv(purpose)
        deposit = prop.estimated_value * (1.0 - ltv)
        return agent.cash >= deposit and prop.estimated_value <= ceiling

    # --------------------------------------------------------------- rental run
    def _run_rental_market(self, renters, step, market_rent):
        rental = RentalMarket(step=step)
        for prop in self.properties:
            if prop.listed_for_rent and prop.occupant_id is None:
                rental.list_property(prop.id, prop.owner_id)
        for hh in renters:
            for c in self._rental_candidates(hh):
                bid = hh.compute_rent_bid(c)
                if bid > 0:
                    rental.submit_bid(c.id, hh.unique_id, bid)
        self._apply_rental_transactions(rental.resolve())

    # ------------------------------------------------------------- apply txns
    def _apply_ownership_transactions(self, transactions):
        applied = []
        for txn in transactions:
            buyer = self._agent_map.get(txn.buyer_id)
            seller = self._agent_map.get(txn.seller_id)
            if buyer is None or seller is None:
                continue
            ltv = self.credit.origination_ltv(txn.purpose)
            deposit = txn.price * (1.0 - ltv)
            if buyer.cash < deposit - 1e-9:
                continue
            prop = self._property_map[txn.property_id]
            if prop.occupant_id is not None and prop.occupant_id != txn.seller_id:
                occ = self._agent_map.get(prop.occupant_id)
                if isinstance(occ, HouseholdAgent):
                    occ.vacate_rental()
            prop.occupant_id = None
            seller.release_property(prop, txn.price)
            prop.owner_id = txn.buyer_id
            prop.purchase_anchor_price = txn.price
            prop.listed_for_sale = False
            prop.listed_for_rent = False
            prop.current_rent = None
            buyer.acquire_property(prop, txn.price, txn.purpose)
            applied.append(txn)

        self.this_step_transactions = applied
        self.all_transactions.extend(applied)

    def _apply_rental_transactions(self, transactions):
        applied = []
        for txn in transactions:
            prop = self._property_map[txn.property_id]
            if not prop.listed_for_rent:
                continue
            tenant = self._agent_map.get(txn.tenant_id)
            landlord = self._agent_map.get(txn.landlord_id)
            if tenant is None:
                continue
            # vacate the tenant's old rented home (if any)
            old = tenant.home_property
            if old is not None and old != prop.id and old not in tenant.owned_properties:
                op = self._property_map.get(old)
                if op is not None and op.occupant_id == tenant.unique_id:
                    op.occupant_id = None
                    op.listed_for_rent = True
            prop.occupant_id = txn.tenant_id
            prop.tenancy_months = 0
            prop.listed_for_rent = False
            prop.current_rent = txn.monthly_rent
            tenant.move_into_rental(prop)
            tenant.pay_rent(txn.monthly_rent)
            if landlord is not None:
                landlord.receive_rent(txn.monthly_rent)
            applied.append(txn)
        self.this_step_rental_transactions = applied
        self.all_rental_transactions.extend(applied)

    # ------------------------------------------------------------- record state
    def _record_state(self):
        # Price index: median estimated value of the whole stock.
        self._price_history.append(float(np.median([p.estimated_value for p in self.properties])))
        rents = [
            p.current_rent
            for p in self.properties
            if p.current_rent and p.occupant_id is not None and p.occupant_id != p.owner_id
        ]
        if rents:
            self._rent_history.append(float(np.mean(rents)))

        # per-zone price index (median estimated value in the zone)
        zh = getattr(self, "_zone_price_history", None)
        if zh is None:
            zh = {z: [] for z in range(self.n_zones)}
            self._zone_price_history = zh
        zone_vals: dict[int, list] = {}
        for p in self.properties:
            zone_vals.setdefault(p.zone, []).append(p.estimated_value)
        for z in range(self.n_zones):
            if zone_vals.get(z):
                zh[z].append(float(np.median(zone_vals[z])))

        insts = [a for a in self.agents if isinstance(a, InstitutionalAgent)]
        inst_units = sum(len(i.portfolio) for i in insts)
        ltvs = [m[1] for a in self.agents for m in a._mortgages.values()]
        self._state_history.append(
            {
                "step": self.steps,
                "price": self._price_history[-1] if self._price_history else 0.0,
                "rent": self._rent_history[-1] if self._rent_history else 0.0,
                "volume": len(self.this_step_transactions),
                "macro": self.current_macro_state,
                "avg_ltv": float(np.mean(ltvs)) if ltvs else 0.0,
                "inst_share": inst_units / max(len(self.properties), 1),
            }
        )

    # ------------------------------------------------------------- visual grid
    def _sync_visual_grid(self):
        for cell in self.grid.all_cells:
            for agent in list(cell.agents):
                try:
                    cell.remove_agent(agent)
                except Exception:
                    pass
        cells = sorted(self.grid.all_cells, key=lambda c: c.coordinate)
        by_coord = {c.coordinate: c for c in cells}
        for prop, cell in zip(self.properties, cells):
            prop.grid_coord = cell.coordinate
        for agent in self.agents:
            if isinstance(agent, HouseholdAgent) and agent.home_property is not None:
                prop = self._property_map.get(agent.home_property)
                if prop is None:
                    continue
                cell = by_coord.get(prop.grid_coord)
                if cell is not None and len(cell.agents) == 0:
                    cell.add_agent(agent)

    def get_model_state(self):
        households = [a for a in self.agents if isinstance(a, HouseholdAgent)]
        return {
            "step": self.steps,
            "hh_ownership_rate": sum(1 for h in households if h.owned_properties)
            / max(len(households), 1),
            "renters": sum(1 for h in households if h.is_renter and h.home_property is not None),
            "landlords": sum(1 for h in households if h.is_landlord),
        }


__all__ = ["HousingModel"]
