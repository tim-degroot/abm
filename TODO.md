# TODO_new — Gap list to realize NewPlan.md


compute bid is ignoring expectations and just using average rent
quality not coming into the valuations or utility
candidates should already be only listed houses
representative utilities are totally wrong
macro growht shocks
wtf is record_bind
check they can buy the property BEFORE and AFTER the bid
property choice has to happen once everything is listed
fire sale should stop when enough are siled
if you sell a property someone lives in you have to kick them out!
what is prop.estimated value 


What is missing in the code to match NewPlan.md. Grouped by plan section.
Priority tags: [P0] blocking core claims · [P1] important · [P2] polish.

---

## §3 — Spatial structure
- [P0] Fix grid to plan: 5×5 zones (Z=25), 625 homes/zone, 15,625 properties. Code is 10×10 (Z=100), 120 properties.
- [P1] Add centre/suburb layout (central zone + surrounding), currently a symmetric torus with no centre.
- [P1] Implement local vs market-wide PRICE observation: households see only local transaction prices, institutions see market-wide. Now all agents use the same global signal.
- [P2] Confirm `quality_clustering` flag is exposed in config.toml (currently only read via getattr default).

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
- [P2] Decide quality handling: plan uses q_k directly; code uses quality_value_scale × rent. Normalise/weight per §29.7.
- [P2] Add institutional required-return reconciliation so r_f < r_f^BTL still guarantees higher institutional ceiling (§11).

## §12 — Rental market
- [P1] Implement r_max = argmax E[U(V(r))] for rent bids; code uses pure affordability (income × fraction).
- [P2] Rental property selection by utility/logit; code ranks by -price/income.

## §17 — Initialization
- [P0] Cap institutional stock at `inst_ownership_share` (0.10); code assigns ALL residual stock to institutions (causes institutional dominance, §29 P0.3).
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
