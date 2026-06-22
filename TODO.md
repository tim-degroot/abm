
    sigma_hat = The household's estimated volatility of future property-price growth. 
                standard deviation of possible growth outcomes (self.model.config.expectations.household_price_growth_volatility).
                household_price_growth_volatility = perceived uncertainty penalty households apply to expected growth
                household_noise_sd != household_price_growth_volatility. 1. noise in the household's price forecasts 2. percieved volatility of actual price growth outcomes
    
    ToDo:
    1. add helper to agent to get parameters
    2. pass risk adjusted capital gain into household WTP should downstream into buy action, buy to let, propery selection, submitted bids
    3. change hold score to use risk adjusted growth instead of expected growth
    4. remove crra(), crra_utility(), household_action_value()

Agents:
 - Expectations should be updated before every action selection
 - What is ''_local_rent_estimate(self)'' and why is it not in expectations?

looks good, some things that jumped out:
- we need to put a risk free rate in macro config and a market value estimator in expectations to make them work
- we have crra which just calls crra_utility so need to merge these
- you're right, minus inf for nonpositive surplus aint right its far too restrictive (since it goes into a logit anyway the probabilties stay positive) - need to address this
- also household action value right now only addresses a buy action but there are a bunch of different alternative actions to get V for
- loss aversion and risk adjusted value look like the right idea but i dont think they're hooked in correctly, loss aversion specifically is already used to get a reservation price in model so maybe we can just ignore it here, and the risk adjustment should be applied to any stochastic variable (so rent vol, price vol etc.) before its fed into V, so that it leads to lower valuations/wtp/higher likelihood of selling etc

@Tim there's a few changes I don't understand in your latest pushes I wanted to ask you about, could you run me through these?
- Your random allocation assigns a house to every household
- You removed r from the call to monthly mortgage payment but its necessary because institutionals and households have different interest rates so it needs to be passed
- Initial rent yield has been restored as a hardcoded parameter and estimate intial rent yield is back
- Avg rent has been restored to function signature like plan distress sales even though its a var which shouldnt exist at all (people should use their expectations, not market wide means) and we're using choose action on each property but this could feasibly result in no listings (they choose to hold all of them) or insufficient listings to cover the money they need to raise (or excess listings where they liquidate everything but dont need to)
- Reservation rent has been restored when it has no function (only positive bids are allowed and any money is better than no money)

sensitivity analysis 

Missing the bounded vision and difference in signal for institutionals. extrapolation, regression on property atributes, pass szone, geommen 


Spatial structure: Z = grid_rows x grid_cols zones on a 2D TOROIDAL grid (config [spatial]). Each agent's consideration set = its own zone + the 4 von Neumann neighbours (up/down/left/right), wrapping around the torus edges, so every agent faces a symmetric 5-zone search space with no edge effects. Properties are distributed as evenly as possible across zones. (See config [spatial] for why the default is a 4x4 torus rather than plan.md's nominal Z=10.) Initialisation (plan.md §17-18 — balance sheets DERIVED from allocations so the accounting identity HousingAssets = HousingEquity + MortgageDebt holds by construction): 1. Generate housing stock; quality q_k = mu_z + nu_k, standardised; price anchor = base_price + price_sensitivity * q_k (base_price is a CALIBRATED market anchor, not arbitrary). 2. Draw households: income (log-normal), TOTAL WEALTH (multiple of income), risk aversion (log-normal). 3. Match income-ranked households to quality-ranked properties (richer get better). Draw an origination LTV per owner (capped at credit.ltv_limit). Derive: deposit = (1-LTV)*price = equity; mortgage = LTV*price; liquid cash = wealth - deposit. 4. ownership_mode = "emergent" (default): a household owns only if it can afford the deposit AND meets the income (DTI) test; otherwise it becomes a renter, so the ownership rate EMERGES. ownership_mode = "target" (DIAGNOSTIC): force target_ownership_rate by making the wealthiest households owners, topping up cash if short so sheets stay feasible. 5. Private landlords at t=0: a share of owners receive extra (let-out) properties with right-skewed portfolio sizes (plan §17). 6. Institutions allocated a separate tranche of properties as rental stock. 7. Remaining renters placed into available rental stock. 8. Seed price and rent history from initial allocations.


UPDATE NEEDED: LISTING FOR SALE, THEN CLEAR PURCHASES, THEN LISTING RENT, THEN CLEAR RENT CREATE


                voluntary_move = False
                if agent.home_property is None:
                    wants_rental = True
                elif self._tenant_locked_in(agent):
                    wants_rental = False
                else:
                    wants_rental = action == "rent"



move a lot of stuff out of model, poorly architected

        An UNHOUSED household (no owned home AND no active lease, i.e.
        home_property is None) is not tied to any neighbourhood, so it searches
        the ENTIRE market — guaranteeing it is always an active rental searcher
        and can re-house whenever any rental supply exists. This is what stops
        agents who lose their home (distress sale, lease non-renewal) from
        silently dropping out of the market because their home zone happens to
        have no vacancies. A housed renter looking to move stays restricted to
        its local search zones.

bounded rationality and loss aversion and risk aversion

expectations should be updated when used not once per step




NONE OF THE REPRESENTATIVE UTILITIES FOLLOW THE PLAN
LISTINGS SHOULD OCCUR BEFORE BUYS
~~YOU SHOULD BE ABLE TO SUBMIT BIDS TO MULTIPLE PLACES TO RENT, IN FACT ALL OF THEM SEQUENTIAL CLEARING SO EVERYONE GETS A HOUSE SO YOU DO HIGHEST AT A TIME CLEAR IT IN A RANDOM ORER~~
CAN PEOPLE RENT AND BUY AUCTIONS IN THE SAME PERIOD? 
LISTING FOR SALE, THEN CLEAR PURCHASES, THEN LISTING RENT, THEN CLEAR RENT
CREATE PARENT AGENT CLASS TO INHERIT COMMONALITIES
compute bid is ignoring expectations and just using average rent
quality not coming into the valuations or utility
candidates should already be only listed houses
representative utilities are totally wrong
check they can buy the property BEFORE and AFTER the bid
property choice has to happen once everything is listed
fire sale should stop when enough are siled
if you sell a property someone lives in you have to kick them out!
what is prop.estimated value 
what happens to properties which are not sold in a given period, do they get unliseted the next period?
is inst_ltv and required return varying with macro enviornment?
 household should have cheaper funding than institutional
 smoothing valuations   
 ewma

Addressed feedback? Papers we were supposed to read?

---

## §3 — Spatial structure
- [P0] Fix grid to plan: 5×5 zones (Z=25), 625 homes/zone, 15,625 properties. Code is 10×10 (Z=100), 120 properties.
- [P1] Add centre/suburb layout (central zone + surrounding), currently a symmetric torus with no centre.
- [P1] Implement local vs market-wide PRICE observation: households see only local transaction prices, institutions see market-wide. Now all agents use the same global signal.
- [P2] Confirm `quality_clustering` flag is exposed in config (currently only read via getattr default).

## §4 / §15 — Banks & macro regime
- [P0] Make credit parameters regime-dependent: map each macro state to (mortgage_rate, max_LTV, DTI, BTL_rate). Now macro only drives income.
- [P1] Make rents and expectations respond to macro state (plan §15 lists them; code does not).

## §5 / §8 / §11 — Risk aversion & utility
- [P0] Implement CRRA utility U(ΔV)=(ΔV)^(1-γ)/(1-γ) on the surplus, with heterogeneous γ. Now `risk_aversion` is stored but never used.
- [P1] Wire the V → surplus → U → logit flow described in §31.
- [P2] Keep loss aversion as-is (already implemented) but reconcile formula: code compares anchor vs last market price, plan compares vs bid p.

## §7 — Expectations & information asymmetry
- [P0] Implement institutional information advantage: institutions condition expectations on macro state + household leverage and forecast future household credit conditions. Now identical to households.
- [P1] Make E[Δp] heterogeneous across agents (plan §31). In `fixed_level` mode it is a constant for everyone.

## §10 / §13 — Decision architecture
- [P1] Replace heuristic Stage-1 action scores with proper V̄^a (expected value at best property) for rent/hold/sell/rent_out. Only `buy` currently uses real WTP.
- [P1] Align action sets with plan (households {buy,rent,stay}, landlords {buy-to-let,sell,hold,occupy}); code uses one shared enum.
- [P1] Implement reservation price as the indifference solution V_hold = V_sell(p_res); code uses anchor×discount heuristic.

## §11 — WTP / valuation
- [P2] Add institutional required-return reconciliation so r_f < r_f^BTL still guarantees higher institutional ceiling (§11).

## §12 — Rental market
- [P1] Implement r_max = argmax E[U(V(r))] for rent bids; code uses pure affordability (income × fraction).
- [P2] Rental property selection by utility/logit; code ranks by -price/income.

## §17 — Initialization
- [P1] Make institutional liquid capital effectively unconstrained (plan §17); code caps bids at cash/(1-ltv).

## §19 — Parameterisation 
- [P1] Tune init knobs to plausible/stylised values so emergent ownership ≈ 65% without target mode (§29 P1.5). NOT a real-data fit.
- [P2] Keep yield 4-5% / price-to-rent 20-30× only as order-of-magnitude sanity checks, not fitting targets.
- [P1] Implement credit as stylised loose/tight regimes (hand-set tuples), per §4/§15.

## §20 / §22 — Experiments & SA
- [P1] Make marginal-pricer classification three-way (owner-occupier / landlord / institution); code only splits household vs institution.
- [P1] Add metrics: share of transaction VALUE and avg valuation premium per group.
- [P1] Add `inst_min_yield` to the plan (§11/§22) — key regime-switch mechanism currently in code but absent from plan.
- [P2] Note: SA param risk_aversion_mu is inert until §5/§8 implemented.

## §28 — Timestep / cleanup
- [P2] Remove leftover `tenancy_quarters` references (model.py:849, 1688); only `tenancy_months` exists.

## Baseline pathologies (§29 P0 — blocking experiments)
- [P0] Explosive price dynamics (E[Δp] feedback).
- [P0] Rental market collapse after ~2 steps.
- [P0] Institutional dominance crowding out households.
- [P0] Initial ownership rate too low vs 65% target.
