# 1. Motivation and Research Question

Housing ABMs in the Gilbert–Gamal tradition generate complex dynamics through behavioural heuristics. Agents cross wealth thresholds, react to financial conditions, and follow empirically motivated decision rules.

This project proposes a structural alternative.

The central hypothesis is that many stylised housing-market facts emerge from interactions among agents with fundamentally different utility functions, financing structures, constraints and information sets.

The model seeks to explain:

- Credit-driven booms.
- Yield-driven price floors.
- Interest-rate sensitivity.
- Transaction-volume collapses.
- Tenure transitions.
- Distributional effects of credit expansion.
- Institutional entry into housing markets.
- Shifts in the identity of the marginal pricer.

Research Question:

> Can a model of utility-maximising agents with heterogeneous utility and profit-and-loss structures reproduce the dynamics typically attributed to behavioural housing ABMs, while providing a structural explanation of who sets prices and why?
>
> The central claim of the project is that housing-market cycles emerge because owner-occupiers, landlords and institutional investors value the same asset differently. Changes in credit conditions alter which group becomes the marginal pricer, and these shifts in marginal-pricer identity generate changes in prices, transaction volume and wealth distribution.

---

# 2. Core Theoretical Framework

## 2.1 Common Profit-and-Loss Object

All ownership decisions derive from:

$$
\Pi = CF + E[\Delta p] - FC
$$

where:

- $CF$ = expected cash flow,
- $E[\Delta p]$ = expected capital appreciation,
- $FC$ = financing cost.

The same accounting logic applies to all ownership classes.

What differs across agents is:

- the source of cash flow,
- financing costs,
- risk preferences,
- constraints,
- information sets.

## 2.2 The Marginal Pricer

The central theoretical concept is the marginal pricer.

Housing prices are not determined by a representative agent.

Instead, each class generates a valuation distribution. Market prices emerge from competition among those distributions.

At any point:

$$
g^* = \text{group supplying the winning marginal bids}
$$

The model records marginal-pricer identity each period.

Marginal-pricer identity is treated as a first-class state variable. Alongside prices, rents and transaction volume, it forms a primary output of the model and provides the central mechanism linking credit conditions to market outcomes.

This transforms the concept from a narrative explanation into an observable state variable.

## 2.3 Regime Switching

### Loose-Credit Regime

Owner-occupiers dominate.

Prices are driven primarily by affordability and leverage.

### Tight-Credit Regime

Investor demand becomes increasingly important.

Prices become more sensitive to yields and financing spreads.

### Severe Tightening

Reservation prices exceed feasible bids.

Transaction volume collapses before prices fully adjust.

---

# 3. Housing Stock

Housing supply is fixed.

There is:

- no construction,
- no demolition,
- no developer sector.

Each dwelling $k$ possesses a latent quality score:

$$
q_k
$$

representing:

- location,
- amenities,
- size,
- condition,
- neighbourhood desirability.

Standardisation:

$$
E[q]=0
$$

$$
Var(q)=1
$$

This retains heterogeneity while avoiding a large spatial state space.

---

# 4. Agent Classes

## Owner-Occupiers

Consume housing services directly.

Utility derives from:

- housing consumption,
- capital gains,
- financing costs.

## Private Landlords

Value:

- rental income,
- capital gains,
- financing costs.

Risk averse.

## Institutional Investors

Value:

- rental yields,
- capital gains,
- portfolio performance.

Risk neutral.

Possess superior information and lower funding costs.

## Banks

Reduced-form credit providers.

Control:

- mortgage rates,
- LTV limits,
- DTI limits.

Banks are not strategic agents in the baseline.

---

# 5. Utility Functions

## Owner-Occupiers

$$
V^{OO}_{ik}
=
q_k + \Pi_{ik} + \epsilon_{ik}
$$

Utility:

$$
U(V)
=
\frac{V^{1-\gamma_i}}{1-\gamma_i}
$$

where $\gamma_i$ is household-specific risk aversion.

## Private Landlords

$$
V^{LL}_{ik}
=
\Pi_{ik}+\epsilon_{ik}
$$

Utility:

$$
U(V)
=
\frac{V^{1-\gamma_j}}{1-\gamma_j}
$$

## Institutional Investors

$$
U^{INST}=E[\Pi]
$$

Institutions are risk neutral.

---

# 6. Explicit P&L Functions

## Owner-Occupiers

$$
\Pi_H
=
R^{imp}
+
E[\Delta p]
-
r_mLp
$$

where:

- $R^{imp}$ = imputed rent,
- $r_m$ = mortgage rate,
- $L$ = leverage ratio.

## Private Landlords

$$
\Pi_L
=
R
-
\phi
-
r_f^{BTL}Lp
+
E[\Delta p]
$$

## Institutional Investors

$$
\Pi_I
=
R
-
\phi
-
r_fLp
+
E[\Delta p]
$$

with

$$
r_f < r_f^{BTL}
$$

giving institutions a structural funding advantage.

---

# 7. Expectations and Information

All agents use adaptive expectations:

$$
E_{i,t}[x]
=
\delta E_{i,t-1}[x]
+
(1-\delta)S_{i,t}
$$

Differences arise from observed signals.

## Owner-Occupiers

Observe recent transaction prices.

Extrapolate adaptively.

## Private Landlords

Observe:

- prices,
- rents,
- financing conditions.

They form adaptive expectations for both rents and prices and simple expectations about future financing conditions.

## Institutional Investors

Observe:

- prices,
- rents,
- macro state,
- household leverage.

Institutions forecast future household credit conditions.

This nested dependence is one of the model's key mechanisms.

Institutions understand that future prices depend partly on future household borrowing capacity. Consequently, institutions are able to anticipate turning points in housing demand before households revise their expectations. This creates the possibility that institutional behaviour leads rather than follows changes in market conditions.

As a result, institutions may reduce bids before households adjust expectations.

---

# 8. Risk Aversion

Risk aversion is heterogeneous:

$$
\gamma_i
\sim
LogNormal(\mu_\gamma,\sigma_\gamma)
$$

Institutions:

$$
\gamma_I = 0
$$

Calibration target:

the observed share of households that remain renters despite appearing financially capable of purchasing.

---

# 9. Credit Constraints

Ownership requires satisfying two independent constraints.

## Deposit Constraint

$$
(1-LTV)p \le w
$$

## Income Constraint

$$
MortgagePayment \le \alpha y
$$

The model explicitly distinguishes:

- affordability exclusion,
- deposit exclusion.

This distinction is important because UK evidence suggests deposit constraints are often more binding than income constraints.

---

# 10. Two-Stage Decision Architecture

## Stage 1: Entry

Participation occurs if:

$$
V^{buy}>V^{outside}
$$

for at least one feasible property.

Feasible set:

$$
\mathcal K_i
=
\{k:p_k\le p_i^{max}\}
$$

If:

$$
\mathcal K_i=\emptyset
$$

the agent does not participate.

Outputs:

- participation,
- tenure choice,
- willingness-to-pay,
- feasible opportunities.

## Stage 2: Selection

Conditional on participation:

$$
Pr(k)
=
\frac{\exp(\beta V_{ik})}
{\sum_{j\in\mathcal K_i}\exp(\beta V_{ij})}
$$

The logit selects among feasible alternatives only.

The logit never determines willingness-to-pay.

---

# 11. Ownership Valuation and Bidding

Maximum willingness-to-pay:

$$
p^{max}_{ik}
=
\arg\max_p E[U(V_{ik}(p))]
$$

subject to:

- deposit constraints,
- DTI constraints,
- wealth constraints.

Submitted bid:

$$
b_{ik}=p^{max}_{ik}
$$

Ownership transactions use Vickrey auctions.

Truthful bidding is therefore optimal.

---

# 12. Rental Market

Agents unable or unwilling to purchase enter the rental market.

Rental decisions mirror ownership decisions:

1. Entry.
2. Feasible set.
3. Logit selection.
4. Rent bidding.

Maximum rent bid:

$$
r^{max}_{ik}
=
\arg\max_r E[U(V_{ik}(r))]
$$

subject to affordability constraints.

Ownership and rental decisions are intentionally symmetric.

---

# 13. Seller Behaviour and Reservation Prices

Owners compare:

$$
V^{hold}
$$

and

$$
V^{sell}
$$

List when:

$$
V^{sell}>V^{hold}
$$

Reservation prices satisfy:

$$
V^{hold}
=
V^{sell}(p^{res})
$$

This mechanism generates:

- sticky prices,
- selective selling,
- distressed sales,
- transaction-volume collapse.

No separate seller heuristics are required.

---

# 14. Market Clearing

Ownership and rental markets clear independently through Vickrey auctions.

Ownership:

- highest bidder wins,
- pays second-highest bid,
- bid must exceed reservation price.

Rental:

- highest rent bidder wins,
- pays second-highest rent bid.

Prices emerge directly from valuations.

---

# 15. Macroeconomic Environment

The economy evolves through:

$$
\xi \in \{Boom, Neutral, Recession\}
$$

Transitions follow a Markov process.

The macro state affects:

- incomes,
- rents,
- expectations,
- credit conditions.

This provides aggregate fluctuations without requiring a full macro model.

---

# 16. State Space

Households:

$$
(y,w,m,\gamma,E^p,E^r,s)
$$

Private landlords:

$$
(w,m,H,\gamma,E^p,E^r)
$$

Institutions:

$$
(H,E^p,E^r)
$$

Properties:

$$
(q,p,o)
$$

Global variables:

- macro state,
- credit conditions,
- transaction histories,
- ownership histories.

---

# 17. Initialization

1. Generate housing stock.
2. Draw quality values.
3. Assign tenure.
4. Generate mortgages.
5. Generate wealth.
6. Verify accounting identities.
7. Initialise expectations.

Balance sheets are derived from ownership allocations.

---

# 18. Accounting Consistency

Maintain:

$$
HousingAssets
=
HousingEquity + MortgageDebt
$$

and

$$
Assets = Liabilities
$$

throughout simulation.

---

# 19. Calibration Strategy

Calibrated to UK data.

Data sources:

- ONS income distributions,
- Wealth and Assets Survey,
- English Housing Survey,
- FCA mortgage statistics,
- Bank of England mortgage rates,
- ONS HPI,
- rental-yield estimates.

Calibration targets:

- ownership rates,
- rental yields,
- wealth distribution,
- leverage distribution,
- transaction volume,
- price-to-rent ratios.

---

# 20. Experiments

## Mirror Experiments

Replicate the shocks used in Gamal-style models:

- interest-rate increase,
- interest-rate decrease,
- LTV tightening,
- LTV loosening.

### Marginal-Pricer Regime Classification

For each simulation period classify the market as:

* owner-occupier dominated,
* landlord dominated,
* institution dominated.

Record:

* share of winning bids,
* share of transaction value,
* average valuation premium.

## Novel Experiments

### Endogenous Credit Tightening

Macro-induced tightening versus exogenous tightening.

### Risk-Aversion Shock

Increase aggregate risk aversion.

### Institutional Information Advantage

Remove institutional forecasting advantage and compare outcomes.

### Marginal-Pricer Identification

Track the identity of the marginal pricer over a full cycle.

### Distributional Analysis

Measure wealth accumulation across income deciles under different credit regimes.

---

# 21. Outputs

Primary outputs:

- house prices,
- rents,
- transaction volume,
- ownership rates,
- wealth distribution,
- marginal-pricer identity.

Secondary outputs:

- leverage,
- investor share,
- turnover,
- tenure transitions.

---

# 22. Sensitivity Analysis

OFAT analysis:

- interest rates,
- LTV limits,
- risk aversion,
- expectation persistence,
- income growth.

Global analysis:

- Sobol decomposition,
- variance attribution,
- interaction effects.

Key question:

> Do credit parameters dominate housing outcomes?

---

# 23. Validation

Layer 1: Stylised facts.

Layer 2: Quantitative moments.

Layer 3: Regime identification.

The model should generate empirically plausible transitions in marginal-pricer identity as credit conditions change, with corresponding effects on prices, rents, ownership rates and transaction volume.

---

# 24. Paper Structure

1. Introduction
2. Literature Review
3. Theory
4. Model Description
5. Calibration
6. Experiments
7. Sensitivity Analysis
8. Validation
9. Discussion
10. Conclusion

---

# 25. Central Contribution

The central contribution is the proposition that housing-market dynamics can be understood as shifts in the identity of the marginal pricer. Rather than treating prices as the outcome of a representative agent or a collection of behavioural heuristics, the model explains housing cycles as the result of interactions among groups with different utility functions, financing structures and information sets. Credit conditions matter because they change which group is capable of setting prices at the margin.
