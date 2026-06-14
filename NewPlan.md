# 1. Motivation and Research Question

Housing ABMs in the Gilbert–Gamal tradition generate complex dynamics through behavioural heuristics. Agents cross wealth thresholds, react to financial conditions, and follow empirically motivated decision rules.

This project proposes a structural alternative.

The central hypothesis is that many stylised housing-market facts emerge from interactions among agents with fundamentally different utility functions, financing structures, constraints and information sets.

The model seeks to explain:
- Credit-condition sensitivity of prices and rents.
- Shifts in the identity of the marginal pricer.
- Distributional effects of credit expansion.

Research Question:

> Can a model of utility-maximising, boundedly rational agents with heterogeneous utility and profit-and-loss structures reproduce the dynamics typically attributed to behavioural housing ABMs, while providing a structural explanation of who sets prices and why?
>
> The central claim of the project is that housing-market cycles emerge because owner-occupiers, landlords and institutional investors value the same asset differently. Changes in credit conditions alter which group becomes the marginal pricer, and these shifts in marginal-pricer identity generate changes in prices, transaction volume and wealth distribution.

## Emergent Phenomena to Reproduce

The model targets two specific emergent phenomena, which it aims to *reproduce* from agent interactions rather than impose by assumption:

1. **Price and rent cycles** — endogenous boom–bust fluctuations in sale prices *and* rents driven by credit conditions, not monotone trends or exogenous sentiment shocks.
2. **Shifts in marginal-pricer identity over the cycle** — the agent class supplying the winning marginal bid (owner-occupier, landlord, institution) changes systematically as credit tightens or loosens.

These are the objects of the experiments (§20) and sensitivity analysis (§22), not free outputs: the model is judged on whether they arise unprompted.

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


## Spatial Structure

Properties are arranged on a $5 \times 5$ toroidal grid of zones ($Z = 25$), with the city centre in the middle and suburbs around it. Each zone contains $25 \times 25 = 625$ homes, for a total housing stock of $15{,}625$ properties. The toroidal topology eliminates edge effects: every zone has exactly four neighbours (von Neumann neighbourhood).

Each agent has a home zone and forms their consideration set from properties in their own and adjacent zones (5-zone consideration set). Owner-occupiers and private landlords observe only local transaction prices within this neighbourhood; institutional investors observe market-wide prices.

Each property $k$ in zone $z$ has quality:

$$
q_k = \mu_z + \nu_k, \quad \nu_k \sim \mathcal{N}(0, \sigma^2_\nu)
$$

where $\mu_z$ is a fixed zone-level quality mean and $\nu_k$ is a property-specific residual. The zone component induces within-zone quality correlation, so that neighbourhood-level differences emerge from the initialisation. Standardised across all properties: $E[q] = 0$, $\text{Var}(q) = 1$. Spatial quality clustering (correlated $\mu_z$ across adjacent zones) is a configurable extension behind a `quality_clustering` flag.

---

# 4. Agent Classes

Owner-occupiers and private landlords are both household agents; they share the same objective function and differ only in tenure state and portfolio. Tenants are not a separate agent class — tenancy is a tenure *state* that household agents cycle through (unhoused → renting → owning → potentially back to renting). The objective function is identical across tenure states; what changes is the feasible action set. (Or 4 separate classes, including Tenants?)

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

Reduced-form credit providers modelled as regime-dependent parameter sets, not active agents. Each macro regime (§15) maps to a tuple of credit parameters (mortgage rate, max LTV, DTI limit, BTL funding rate).

Control:

- mortgage rates,
- LTV limits,
- DTI limits.

Banks are not strategic agents in the baseline.

---

# 5. Utility Functions

## Bounded Rationality and Cognitive Biases

Agents are not perfect optimisers. Bounded rationality enters the model in three ways: adaptive (not rational) expectations over prices and rents; logit (not argmax) choice at every decision node, reflecting limited attention and idiosyncratic preference; and local consideration sets bounded by zone, reflecting limited search.

Two cognitive biases are explicitly included.

**Risk aversion** is captured by CRRA utility curvature, with heterogeneous coefficients drawn from a log-normal distribution.

**Loss aversion** enters the seller decision via a prospect-theory term. Owner-occupiers and private landlords evaluate a potential sale relative to their nominal purchase price anchor $p_0$. Losses below that anchor are weighted more heavily than equivalent gains:

$$
\bar{V}^{sell}(p) = \bar{V}^{sell,\text{fin}}(p) - \lambda \cdot \max(p_0 - p,\, 0)
$$

where $\lambda > 1$ is the loss-aversion coefficient. Genesove and Mayer (2001) document this pattern in Boston condominium data: sellers facing nominal losses set asking prices 25–35 percent higher relative to expected value, attain 3–18 percent higher realised prices, and exhibit substantially lower sale hazard. The effect holds for both owner-occupants and small investors, though approximately twice as large for owner-occupants; accordingly, $\lambda$ is larger for owner-occupiers than for private landlords. Institutional investors are exempt: they evaluate properties against portfolio benchmarks and mark-to-market values rather than nominal purchase prices. This asymmetry generates nominal price stickiness and contributes to transaction-volume collapse in downturns — patterns that pure rational models systematically under-produce.

## Owner-Occupiers

$$
V^{OO}_{ik}
=
q_k + \Pi_{ik} + \epsilon_{ik}
$$

Utility is applied to the surplus over the outside option:

$$
U(\Delta V)
=
\frac{(\Delta V)^{1-\gamma_i}}{1-\gamma_i}
$$

where $\Delta V = V^{buy}_{ik} - V^{outside}_i$ and $\gamma_i$ is household-specific risk aversion. Applying curvature to the surplus rather than the level ensures the argument is non-negative for all properties in the feasible set $\mathcal{K}_i$, which is required for CRRA to be well-defined.

## Private Landlords

$$
V^{LL}_{ik}
=
\Pi_{ik}+\epsilon_{ik}
$$

Utility:

$$
U(\Delta V)
=
\frac{(\Delta V)^{1-\gamma_j}}{1-\gamma_j}
$$

where $\Delta V = V^{buy}_{ik} - V^{outside}_j$.

## Institutional Investors

$$
U^{INST}=E[\Pi]
$$

Institutions are risk neutral and are not subject to loss aversion.

---

# 6. Explicit P&L Functions

## Owner-Occupiers

The owner-occupier's financial surplus from ownership over the outside option of renting is:

$$
\Pi_H
=
E[\Delta p]
-
r_m L p
$$

where:

- $r_m$ = mortgage rate,
- $L$ = LTV ratio.

Imputed rent does not appear here because it is common to both owning and renting and cancels in the surplus comparison. The agent buys when the expected capital gain net of financing cost exceeds zero, subject to constraints.

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

They form adaptive expectations for both rents and prices and simple expectations about future financing conditions. Rent expectations are updated from observed rental transaction prices using the same adaptive rule, with the signal $S_{i,t}$ being the realised auction rent for the period.

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

# 10. Decision Architecture

All agents choose over a discrete action set each period. The same logit framework governs both action selection and property selection, capturing idiosyncrasy at every decision node without introducing deterministic threshold rules anywhere in the model.

## Stage 1: Action Choice

Each agent draws idiosyncratic taste shocks over their available actions and selects probabilistically via logit:

$$
Pr(a)
=
\frac{\exp(\beta_a \bar{V}^a_i)}
{\sum_{a' \in \mathcal{A}_i} \exp(\beta_a \bar{V}^{a'}_i)}
$$

where $\bar{V}^a_i$ is the expected value of action $a$ for agent $i$, evaluated at the best available property under that action, and $\mathcal{A}_i$ is the agent's feasible action set.

Action sets by agent type:

- **Households:** $\mathcal{A} = \{\text{buy}, \text{rent}, \text{stay}\}$ for current owners; $\mathcal{A} = \{\text{buy}, \text{rent}\}$ for current tenants.
- **Private landlords:** $\mathcal{A} = \{\text{buy-to-let}, \text{sell}, \text{hold}, \text{occupy}\}$.
- **Institutional investors:** $\mathcal{A} = \{\text{acquire}, \text{sell}, \text{hold}\}$.

Infeasible actions — those violating credit constraints or requiring properties outside the feasible set — are assigned $\bar{V} = -\infty$ and receive zero probability mass. No agent is ever forced into an action by a deterministic rule.

## Stage 2: Property Selection

Conditional on an action that involves a property transaction, the agent selects among feasible properties by logit:

$$
Pr(k \mid a)
=
\frac{\exp(\beta V_{ik})}
{\sum_{j \in \mathcal{K}_i} \exp(\beta V_{ij})}
$$

The feasible set $\mathcal{K}_i = \{k : p_k \leq p^{\max}_i\}$ is defined by willingness-to-pay, which is computed from the P&L functions independently of the logit.

The logit determines which property receives a bid. It never determines willingness-to-pay.

## Stage 3: Bidding

Having selected a property, the agent submits their truthful maximum willingness-to-pay as their bid. The Vickrey mechanism then determines the transaction outcome.

---

# 11. Ownership Valuation and Bidding

Maximum willingness-to-pay is the price at which the agent's surplus over their outside option equals zero. Given the P&L functions, this has a closed form for all agent types.

**Owner-occupiers** (binding constraint is surplus = 0):

$$
p^{\max}_{ik} = \frac{E[\Delta p] + q_k - V^{outside}_i + \epsilon_{ik}}{r_m L}
$$

subject to whichever of the deposit or DTI constraints binds first. The quality score $q_k$ lifts the owner-occupier ceiling directly; the mortgage rate and LTV compress it. The outside option $V^{outside}_i$ is the value of the best available rental.

**Private landlords:**

$$
p^{\max}_{L} = \frac{R - \phi + E[\Delta p]}{r_f^{BTL} L}
$$

**Institutional investors:**

$$
p^{\max}_{I} = \frac{R - \phi + E[\Delta p]}{r_f L}
$$

Since $r_f < r_f^{BTL}$, institutions always have a higher price ceiling than private landlords for the same property at the same rent and expectations. Quality $q_k$ does not enter investor ceilings directly; it enters only through $R$, the achievable rent, which is increasing in $q_k$.

These expressions make the regime-switching logic transparent: in a loose-credit environment the owner-occupier ceiling is high and rising with $E[\Delta p]$; as $r_m$ rises it collapses, while investor ceilings — anchored to rents and wholesale funding rates — are comparatively stable.

Where constraints interact in a non-linear way, $p^{\max}$ is solved numerically. The closed forms above serve as the baseline and as a check on the numerical solution.

Submitted bid:

$$
b_{ik}=p^{max}_{ik}
$$

Ownership transactions use Vickrey auctions.

Truthful bidding is therefore optimal.

---

# 12. Rental Market

Agents who choose the rent action in Stage 1 of §10 — whether because they cannot buy, choose not to buy, or are landlords listing a property for let — enter the rental market.

Rental property selection and bidding mirror the ownership process: feasible set defined by affordability, logit selection among feasible rentals, truthful maximum rent bid submitted to a Vickrey auction.

Maximum rent bid:

$$
r^{max}_{ik}
=
\arg\max_r E[U(V_{ik}(r))]
$$

subject to affordability constraints.

For landlords, the let action in $\mathcal{A}$ places their property on the rental market. The expected value of letting, $\bar{V}^{let}$, is the capitalised rental income stream net of operating costs — identical in structure to $\Pi_L$ and $\Pi_I$ but with the sale option replaced by continued ownership. This enters the Stage 1 logit alongside sell and hold, so the let-versus-sell decision is governed by the same unified framework as all other choices.

---

# 13. Seller Behaviour and Reservation Prices

Seller behaviour is a direct consequence of the action-choice logit in §10. Owners do not compare sell against hold in isolation; they choose among all available actions — sell, hold, occupy (for landlords), let — with probabilities determined by the expected value of each.

The reservation price is defined implicitly as the sale price at which $\bar{V}^{sell} = \bar{V}^{hold}$:

$$
V^{hold} = V^{sell}(p^{res})
$$

The seller accepts any bid above $p^{res}$. Below it, holding (or letting) dominates and the property is withdrawn.

Reservation prices differ by agent type in a principled way. Owner-occupiers in negative equity have a reservation price above the current market because the utility of selling — receiving negative equity and moving to the rental market — is worse than staying. Private landlords' reservation price is anchored to the capitalised rental value: they will not sell below the yield-implied price because holding and letting dominates. Institutions' reservation price incorporates the capital gain term in their P&L, making them slower to sell in downturns.

The logit over actions means that even agents whose $\bar{V}^{sell} < \bar{V}^{hold}$ in expectation will occasionally sell due to idiosyncratic shocks — capturing distressed or liquidity-driven sales without requiring a separate mechanism.

This framework generates without additional assumptions:

- sticky prices,
- selective selling,
- distressed sales,
- transaction-volume collapse.

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

In a Vickrey (second-price) auction, bidding one's true valuation is a weakly dominant strategy: because the price paid is set by the *second*-highest bid and never by the winner's own bid, no agent can improve its outcome by shading. Over-bidding only risks winning at a price above value (a loss); under-bidding only risks losing an auction the agent would have won profitably. Truthful bidding ($b_{ik} = p^{\max}_{ik}$, §11) is therefore optimal, so agents need no strategic bid-shading and prices reveal genuine valuations.

---

# 15. Macroeconomic Environment

The economy evolves through:

$$
\xi \in \{Boom, Neutral, Recession\}
$$

Transitions follow a Markov process. The macro state is implemented as a parameter-selection index — each regime maps to values for income growth rate, base mortgage rate, max LTV, and expected rent growth — rather than as code branches. Transition probabilities are rescaled for monthly time steps: diagonal entries $\approx (11+p)/12$.

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
(y, w, m, \gamma, E^p, E^r, s, z)
$$

Private landlords:

$$
(w, m, H, \gamma, E^p, E^r, z)
$$

Institutions:

$$
(H, E^p, E^r, z)
$$

Properties:

$$
(q, p, p_0, o, z)
$$

where $z$ is zone, $p_0$ is the nominal purchase price anchor used in the loss-aversion term, and $p$ is the current estimated market value.

Global variables:

- macro state,
- credit conditions,
- transaction histories,
- ownership histories.

---

# 17. Initialization

1. Generate housing stock and assign properties to zones.
2. Draw zone quality means $\mu_z$ and property residuals $\nu_k$; compute $q_k = \mu_z + \nu_k$.
3. Generate mortgages consistent with the LTV distribution calibrated to FCA Mortgage Product Sales Data.
4. Generate agent wealth consistent with income decile, drawn from ONS Wealth and Assets Survey conditional distributions.
5. Assign purchase price anchors $p_0$ to all current owners, initialised at the current property value (updated at each transaction thereafter).
6. Verify accounting identities.
7. Initialise expectations.

Balance sheets are derived from ownership allocations, not generated independently. This ordering enforces accounting consistency: total housing asset value equals the sum of equity and mortgage debt across all agents by construction.

Private landlord portfolio sizes are drawn from a right-skewed distribution consistent with the empirical concentration of private rented sectors. Institutional investors are initialised with a small number of large-portfolio agents whose aggregate holdings match the institutional share of stock; their liquid capital is treated as effectively unconstrained relative to the housing market. Household liquid wealth is set to the residual of total wealth minus housing equity for owners, and to savings consistent with their income decile for tenants.

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

# 19. Parameterisation Strategy (Scope)

**Full empirical calibration to UK micro-data is out of scope.** Parameters are set to *plausible, stylised values* that place the model in a realistic region of the parameter space — they are not fitted to any specific dataset or time period. Likewise, credit enters the model as a small set of **stylised credit regimes** (loose / tight), with hand-set parameter tuples (§4, §15), rather than a calibrated Bank-of-England rate path.

Reference anchors (used only to keep magnitudes sensible — sanity checks, not fitting targets):

- ownership rate ≈ 65%,
- gross rental yield 4–5%,
- price-to-rent ratio 20–30$\times$,
- leverage in a realistic range.

The scientific claims rest on the *qualitative* emergent phenomena of §1 (price/rent cycles and marginal-pricer shifts) and on the sensitivity analysis (§22), **not** on quantitative empirical fit. Robustness to moderate variation in these stylised parameters is verified through the SA.

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

### Institutional Information Advantage

Remove institutional conditioning on macro state and household leverage. Replace with the same adaptive expectation rule used by private landlords.

Compare the resulting cycle dynamics against the baseline. Key questions: does the regime switch still occur? Does it occur later? Does the institutional price floor hold as firmly when institutions can no longer anticipate turning points in the household credit regime?

This experiment isolates the contribution of the information asymmetry to cycle dynamics, independently of the funding-cost and risk-aversion asymmetries.

### Marginal-Pricer Identification

Track the identity of the marginal pricer over a full cycle.

### Distributional Analysis

Measure wealth accumulation across income deciles under different credit regimes.

---

# 21. Outputs

Primary outputs:

- `avg_sale_price` — average transaction price,
- `avg_rent` — average rental transaction price,
- `transaction_volume` — number of ownership transactions,
- `rental_transaction_volume` — number of rental transactions,
- `ownership_rate` — fraction of households who own,
- `institutional_ownership_share` — fraction of housing stock held by institutions,
- `household_ownership_share_of_stock` — fraction of stock held by households (note: defined differently from ownership rate),
- `household_marginal_pricer_share` — fraction of winning bids placed by households,
- `total_household_net_worth` — aggregate household wealth,
- `price_to_rent_ratio` — avg sale price / avg annual rent,
- `avg_loan_to_value` — average LTV across mortgaged properties,
- `vacancy_rate` — fraction of properties unoccupied.

---

# 22. Sensitivity Analysis

Research hypothesis: *Credit conditions dominate housing outcomes by shifting which agent class becomes the marginal pricer.*

## Core SA Inputs (11 parameters)

| # | Parameter | Config key | Channel |
|---|-----------|------------|---------|
| 1 | Mortgage rate | `credit.mortgage_rate` | Owner-occupier WTP ↓ |
| 2 | LTV limit | `credit.ltv_limit` | Deposit constraint binds |
| 3 | DTI limit | `credit.dti_limit` | Income constraint binds |
| 4 | BTL funding rate | `credit.btl_funding_rate` | Landlord WTP ↓ (§6) |
| 5 | BTL LTV | `credit.btl_ltv` | Landlord deposit constraint |
| 6 | Institutional required return | `agent.inst_required_return` | Institution activity threshold |
| 7 | Institutional LTV | `agent.inst_ltv` | Institution WTP |
| 8 | Expectation persistence δ | `expectations.delta` | Expectation momentum (§7) |
| 9 | Risk aversion μ | `agent_init.risk_aversion_mu` | Bid shading (§8) |
| 10 | Owner loss aversion λ | `market.loss_aversion_owner` | Price stickiness, volume collapse (§5) |
| 11 | Action logit temperature β | `agent.beta_action` | Noise in action choice (§10) |

All other `config.toml` parameters (spatial, property init, income/wealth distributions, lease parameters) are held at fixed, plausible (stylised) values and are not SA targets.

## Analysis Methods

OFAT analysis across all 11 parameters. Global analysis via Sobol decomposition ($N = 1024$ Saltelli samples) for variance attribution and interaction effects.

Key question:

> Do credit parameters (1–5) dominate housing outcomes?


---

# 23. Paper Structure

1. Introduction
2. Literature Review
3. Theory
4. Model Description
5. Calibration
6. Experiments
7. Sensitivity Analysis
8. Discussion
9. Conclusion

---

# 24. ABM Checklist Compliance

| Requirement                                         | Implementation                                                                                                                                 | Role in emergent outcomes                                                                                                                                     |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Discrete agents with clear identities               | Four agent classes with distinct state vectors (§4, §16)                                                                                       | Enables heterogeneous valuation distributions to coexist and compete                                                                                          |
| Internal states changing over time                  | Wealth, mortgage, equity, expectations, tenure, $p_0$ all evolve each period (§16)                                                             | Generates path dependence and wealth accumulation dynamics                                                                                                    |
| Spatial localisation                                | Zone structure with within-zone quality correlation; consideration sets bounded by zone (§3)                                                   | Generates endogenous neighbourhood quality sorting and localised price dynamics                                                                               |
| Environmental perception and interaction            | Agents observe transaction prices, rents, macro state, and credit conditions; update expectations accordingly (§7)                             | Expectation heterogeneity produces regime-dependent dynamics and leading/lagging behaviour                                                                    |
| Bounded rationality                                 | Adaptive expectations; logit choice at every decision node; zone-bounded search (§5, §7, §10)                                                  | Produces momentum, overshooting, and search frictions absent from full-information models                                                                     |
| Risk aversion and loss aversion                     | CRRA utility with heterogeneous $\gamma_i \sim \text{LogNormal}$; prospect-theory loss aversion in seller decisions with anchor $p_0$ (§5, §8) | Risk aversion generates non-participation and deposit-constrained exclusion; loss aversion generates nominal price stickiness and transaction-volume collapse |
| Learning and adaptation                             | Adaptive expectation updating each period; consideration set adjusts to feasible set (§7, §10)                                                 | Expectation dynamics generate boom-bust cycles without exogenous sentiment shocks                                                                             |
| Strategic interaction with game-theoretic structure | Vickrey auctions for both ownership and rental markets; truthful bidding is a dominant strategy (§14)                                          | Prices emerge from valuation competition; no strategic bid-shading required                                                                                   |
| No central supervisor                               | Banks are reduced-form; no price-setter or planner; prices emerge from decentralised auctions (§4, §14)                                        | Emergent pricing from decentralised competition is the model's core mechanism                                                                                 |
| Nontrivial emergent behaviour                       | Marginal-pricer regime switching, endogenous transaction-volume collapse, wealth distributional divergence across income deciles               | These are not imposed — they arise from agent interactions under changing credit conditions                                                                   |
| Robust sensitivity analysis                         | Sobol decomposition with $N = 1024$ Saltelli samples; OFAT for individual parameters (§22, §23)                                                | Establishes which parameters drive variance in prices and rents; falsifies or confirms credit-dominance hypothesis                                            |

---

# 25. Central Contribution

The central contribution is the proposition that housing-market dynamics can be understood as shifts in the identity of the marginal pricer. Rather than treating prices as the outcome of a representative agent or a collection of behavioural heuristics, the model explains housing cycles as the result of interactions among groups with different utility functions, financing structures and information sets. Credit conditions matter because they change which group is capable of setting prices at the margin.

---

# 26. Open Questions

- In the absence of shocks and births/deaths, does the model converge to an absorbing state? What is the equilibrium wealth distribution across agent types and income deciles? How does this depend on credit conditions?
- Activation order: random vs. fixed sequential. Fixed ordering gives first movers access to the full property pool — decide and document.

---

# 27. Timestep Convention

Every model period represents **one calendar month**. All flows, rates, and durations consistently use monthly units:

| Concept | Period unit | Usage |
|---------|-------------|-------|
| Mortgage payment | monthly | Annuity formula uses `r/12` periodic rate and `n×12` periods |
| Income | monthly | DTI check: `payment ≤ α × monthly_income` |
| Rent | monthly | Already monthly in implementation (no change needed) |
| Mortgage rate | per month | 0.05 p.a. → 0.004167 per month |
| Funding rate | per month | 0.03 p.a. → 0.0025 per month |
| BTL funding rate | per month | 0.06 p.a. → 0.005 per month |
| Loan term | months | 25 years → 300 months |
| `E[dp]` (expected capital gain) | £ per month | fixed_level: 2000 p.a. → ~166.67 per month |
| Price-to-income ceiling | × monthly income | 4.5 × annual → 54 × monthly (same effective cap) |
| Price-to-rent ceiling | × monthly rent | 25 × annual → 300 × monthly (same effective cap) |
| `steps_held` | months | Incremented by 1 each step |
| `tenancy_months` | months | Incremented by 1 each step (replaces `tenancy_quarters`) |

**Why monthly rather than annual**: monthly calibration aligns with real-world mortgage payment cycles, rent payment cycles, and income receipt cycles. It produces smoother price and rent series, avoids discretisation artefacts from chunky annual payments, and allows realistic lease terms (minimum 12 months, not 4 quarters). The cost is 12× more steps for the same calendar time, and all per-period growth rates and transition probabilities must be rescaled (means ÷12; standard deviations ÷√12; Markov diagonal entries ≈ (11+p)/12).

**What stays the same**: The structural P&L formulas in §6 are invariant — they are expressed as per-period equations. Only the units change. The annuity formulas, WTP denominators, cash-flow deductions, and lease-turnover logic all use the same mathematical forms with the monthly-adjusted parameters.

Config parameters with an implicit "per period" unit are rescaled accordingly (see `config.toml`). All code comments referencing "annual" have been updated to "monthly". Method names like `annual_mortgage_payment` have been renamed to `monthly_mortgage_payment`.

---

# 28. Implementation TODO

Priority-ordered. Everything below follows this plan unless there is a very good reason not to.

### P0 — Baseline Pathologies (blocking all experiments)

1. **Explosive price dynamics** — feedback loop in E[Δp] anchored to realised prices, not fundamentals. Root cause of most other pathologies.
2. **Rental market collapse** — dies after ~2 steps. Fix requires lease expiry, re-queuing unhoused agents, low-probability re-search for housed renters.
3. **Institutional dominance** — crowds out all household ownership.
4. **Initial ownership rate** too low vs. 65% UK target.

### P1 — Core Model Completion

5. Calibrate `wealth_income_mult_*`, `income_median`, and related init knobs so emergent ownership stays near 65% without `ownership_mode="target"`.
6. Replace mean-reverting income dynamics with macro growth shocks.
7. Review decision-rule weights, quality handling, and reservation-price logic against this plan (§10–§13). Quality weight needs attention: raw $q_k \in [0,1]$ is too small to move the needle; consider normalising to percentiles or adding a weight parameter. Landlord decisions should not include quality directly — only through achievable rent $R$.
8. Implement risk and loss aversion properly per §5 and §8.
9. Review logit temperatures for reasonableness.

### P2 — Diagnostics & Experiments

10. Build visualisations for prices, rents, tenure composition, and wealth by agent type.
11. Add spatial quality clustering behind `quality_clustering` flag and test.
12. Expand visual diagnostics by agent type.
13. Run mirror experiments (§20) and sensitivity analysis (§22).

---

# 29. Configuration Pruning Candidates

The following config options should be collapsed once the baseline is stable:

| Option | Action |
|--------|--------|
| `capital_gain_mode` + `capital_gain_growth_min/max` | Commit to `fixed_level` permanently |
| `ownership_mode` | Drop `"target"`, keep only `"emergent"` |
| `inst_cash_low` / `inst_cash_high` | Replace with single fixed value |
| `inst_funding_rate_low` / `inst_funding_rate_high` | Replace with single fixed value |

---

# 30. Key Design Principles

Accumulated from development — consult before making architectural decisions.

- **Proportional vs. fixed capital gains**: proportional growth (% of price) is realistic; the problem is that E[Δp] must be bounded and anchored to fundamentals rather than echoing realised prices.
- **E[Δp] is the key heterogeneous term**: subjective expectations drive agent differentiation and marginal-pricer transitions; CF and FC differences are real but more structurally determined.
- **Information asymmetry + utility differences are both necessary** for marginal-pricer transitions: utility differences enable the transition; information asymmetry produces the temporal lag that makes it gradual.
- **Bounded rationality as deliberate assumption**: owner-occupiers not connecting rising rates to falling prices is a simplifying assumption (not a behavioural law); flag as such, with relaxation as a natural extension.
- **Task 1 vs. Task 2 distinction**: Task 1 bounds the expectation term (beliefs); Task 2 imposes a hard affordability ceiling (what a bank would actually lend). Distinct mechanisms.
- **ε placement**: the idiosyncratic taste shock appears only once — either in $p^{\max}$ or in the logit, not both (double-counting).
- **V → surplus → U flow**: raw property value $V$ is computed first, then surplus $\Delta V = V - V^{outside}$, then CRRA utility $U(\Delta V)$ is applied before entering the logit.
- **Ownership rate vs. household share of stock**: differently defined fractions; do not plot on the same axis without clear labelling.

---

# 31. References

- Gamal et al. (2024), JASSS: https://www.jasss.org/23/2/7.html
- Bezemer & co-authors, "Roof or real estate? An agent-based model of housing affordability"
- Genesove & Mayer (2001), Loss aversion and seller behaviour
- Parunak et al. (1998), Bonabeau (2002) — ABM methodology
- Baptista et al. (2016), Carro et al. (2022) — housing ABMs
- Geanakoplos et al. (2012), Axtell et al. (2014) — leverage cycles
- Farmer & Foley (2009) — complexity economics
- Fagiolo et al. (2019) — ABM validation framework
- UK data: BoE rates, ONS wage growth / WAS / ASHE / HPI, PRA LTV rules, FCA MPSD, English Housing Survey, HMRC stamp duty, IMD for spatial quality