# TODO

## General
1. ~~Create a config file to store all parameters and initialisation settings, and pass this to all functions that need it.~~ ✅ Done: Moved all settings to config.toml . Now config.py loads these into an immutable Config object and passes them safely to the model, agents and run functions. If you can't find a parameter, check config.py to see if its name was changed in the code.
~~Add a requirements file~~. ✅ Done: generated requirements.txt as a fallback (uv remains the recommended setup).
3. Build visualisations for all the things we're interested in, including the spatial distribution of properties and agents, the evolution of prices and rents over time, and the distribution of wealth across agent types and income deciles.
4. Implement risk and loss aversion properly.
5. Consider adding limited vision to the model, where non-institutionals can only see a subset of the market (e.g. properties in their own and adjacent zones) when making decisions.
6. Check reservation price logic is in there.
7. Figure out whether just putting in the quality with no weight actually makes sense for renters and owner occupiers, its between 0 and 1 too small to move the needle on the decision process, and for landlords it doesn't make sense at all, they should just be looking at the price and their costs and expected rental income, so maybe we need to add a weight to the quality in the decision process or normalize to percentiles or something.
8. Figure out whether we actually need Validation at all!

## Agents

1. Correct income dynamics, there's no need for mean reversion, we can just draw from a growth rate distribution each period depending on the macro state.
2. Are the logit temperatures reasonable?
3. Adjust net worth for loan balance.
4. Review action and property selection logic for consistency with our plan and economic intuition - I think this is completely wrong right now (just some filler) and not actually what we intended, for example I don't see quality anywhere in the decision process - and ensure all actions are available to all agent types (except institutional landlords, who can only buy and sell).
5. Ensure agents can bid on multiple rental properties only, and move out if they cannot pay rent.

## Credit

...

## Expectations

...

## Markets

...

## Metrics

1. Review whether all the things we're interested in are actually being recorded here.

## Model
1. Correct the spatial structure, it should be a 2D grid (toroidal) and the allocation of properties to zones does not follow what we laid out in the plan.
2. The initialization rules are also completely ad hoc and need to be revised following the plan.

## Policies

...

## Properties

...

## Run

1. Right now completely ad hoc and prematurely tries to run experiments, let's rewrite this to just run the model in a useful way for our iteration, and then we can build out the experiments once the model is working properly.

## Valuation

1. Pretty much all of this is ad hoc placeholders and doesn't follow the plan at all, need to review and revise.
