# Overall
1. Consolidate - there is a lot of code that is doing similar things in different places, and it is totally unclear which of these is being used where. Model in particular has a lot of code that belongs in other places, or overlaps with other places, and agents does too, move anything that does not *have* to live in agents or model to the appropriate module so you can actualyl see/consolidate the logic around each thing.
2. Decide where and when expectations are to be updated, and do so consistently. We have a partial update in model, and a partial update in agents, but we need to decide where the single source of truth for expectations is, and make sure it's being updated consistently.

## Agents
 - There are still enough commonalities here that a BaseAgent class would be useful to consolidate some of the code and make it clearer what is actually different between the agents.
 - ''_local_rent_estimate(self)'' should not exist, use expectations!
 - Ensure the choose property logic is only happening once everything is listed, and that the candidates are only listed houses.
 - Fire sale logic should stop when enough are sold, not just when there are no more to sell.
 - We need proper WTP calculations for rent bids not the heuristic that currently exists.

## Config
 - We need a risk free rate in the macro config to be used in valuation.

## Credit
 - Getting all kinds of Buyer X cannot cover despoit bid/feasibility logic failed errors

## Expectations
 - Where are price and rent volatility expectations, and are they feeding into utility properly via risk-adjustments?
 - Household expectations must implement *bounded vision* they can only see data from their zone.

## Markets
 - 

## Metrics
 - Make marginal-pricer classification three-way (owner-occupier / landlord / institution); code only splits household vs institution.
 - Add share of transaction VALUE 

## Model
 - 

## Plotting
 - 

## Policies
 - Build a few more experiments to investigate the variables of interest under realistic shocks:
   - interest-rate increase,
   - interest-rate decrease,
   - LTV tightening,
   - LTV loosening.
   - Credit-condition sensitivity of prices and rents.
   - Shifts in the identity of the marginal pricer.
   - Distributional effects of credit expansion.
 - Are inst_ltv, LTV, DTI, mortgage_rate and required return varying with macro enviornment?
 - Small parameter shifts (e.g., lending rules) may trigger systemic collapse/booms (invesigate).

## Properties
 - 

## Run
 - 

## Utility
 - Are we using the risk-adjusted growth rate appropriately? Is it using volatility expectations appropriately? Is the risk aversion parameter being used as the coefficient of volatility in the risk adjustment properly?
 - Should we do proper loss aversion (currently a heuristic in reservation price) something like $\bar{V}^{sell}(p) = \bar{V}^{sell,\text{fin}}(p) - \lambda \cdot \max(p_0 - p,\, 0)$? The trouble is that we never actually evaluate this over several prices we're doing everything in point estimates (that's why we're ending up with risk aversion in the way we have it), so we don't have a good way to do this. But we should think about it.
 - Nonpositive surplus should not be set to -inf - too extreme, what should be the case is that infeasible actions — those violating credit constraints or requiring properties outside the feasible set — are assigned $\bar{V} = -\infty$ and receive zero probability mass (but this should be done in the appropriate place, which is not in utility).



## Valuation
 - Implement reservation price as the indifference solution V_hold = V_sell(p_res); code uses anchor×discount heuristic.

   
## Sensitivity Analysis
 - "Global Sensitivity Analysis Plot - 1st and Total Order" (Presentation Guidelines)

