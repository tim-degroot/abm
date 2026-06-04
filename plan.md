# A Structural P&L Model of Housing Market Dynamics
## Agent-Based Model — Project Plan

---

## 1. Motivation and Research Question

Housing ABMs in the Gilbert–Gamal lineage generate complex dynamics through behavioural heuristics: agents cross wealth thresholds and respond according to empirically motivated rules. This project proposes a structural alternative grounded in a single organising principle: **agents are heterogeneous not in psychology but in the shape of their profit-and-loss (P&L) functions**.

The core claim is that the qualitative features of housing markets — credit-driven booms, yield-driven floors, regime switching between marginal pricers, and regressive distributional effects of credit expansion — emerge mechanically from the interaction of structurally distinct payoff surfaces. Bounded rationality is not necessary. The "weirdness" is in the incentive geometry, not in the agents' heads.

Gamal et al. model agents with behavioural rules: households are "relatively rich" or "relatively poor" and respond to financial conditions through threshold-based heuristics. This project asks instead: what if we give agents utility-maximising decision rules derived from their actual payoff functions? Does the model produce qualitatively different dynamics, or do the same stylised facts emerge either way?

Three refinements beyond a pure rational baseline give the model empirical purchase:

1. **Endogenous risk aversion.** Individual households draw risk preferences from a distribution; institutions are risk-neutral by construction. Risk aversion enters utility as a certainty-equivalent adjustment to the P&L.
2. **Calibrated initial conditions.** The starting allocation of housing stock across tenure types and the initial wealth distribution across income deciles are drawn from UK data. The balance sheet weights $\lambda_g$ that enter the market clearing condition are empirical inputs, not free parameters.
3. **Macroeconomic embedding.** Agent incomes evolve with a slow-moving discrete macro state, and a single representative bank sets credit conditions endogenously as a function of that state.

**Research question:** Does a model of utility-maximising agents with heterogeneous P&L functions — augmented by endogenous risk preferences and calibrated initial conditions — reproduce the dynamics of behavioural housing ABMs, and does it match empirical patterns in UK data?

---

## 2. Theoretical Framework

### 2.1 The core object: P&L surfaces

Each agent group $g \in \{H, L, I, B\}$ — owner-occupiers, private landlords, institutional investors, and banks — is defined by a profit function:

$$\Pi_g = \Pi_g(p,\, x,\, a_g)$$

where $p$ is the house price, $x$ is a vector of state variables (mortgage rate $r_m$, BTL funding rate $r_f$, LTV ratio $L$, income $y$, rent $R$, macro state $\xi$), and $a_g$ is the agent's action (buy, sell, hold, let). Each group maximises expected utility over its certainty-equivalent P&L:

$$\max_{a_g} \; \mathbb{E}\bigl[\Pi_g(p, x, a_g)\bigr] - \rho_g \cdot \mathrm{Var}\bigl[\Pi_g\bigr]$$

Demand from group $g$ is the quantity $q_g$ that solves this programme given balance sheet constraints. Market clearing determines the equilibrium price:

$$\sum_g \lambda_g D_g(p, x) = S(p, x)$$

where $\lambda_g$ is the aggregate balance sheet weight of group $g$ and $S(p,x)$ is housing supply (inelastic in the short run). The equilibrium price $p^* = p^*(\{x\}, \{\lambda_g\})$ is determined by who can bid most given current conditions — not by fundamentals per se, but by the dominant P&L gradient at the margin.

Price dynamics between clearing periods follow a tâtonnement process:

$$p_{t+1} = p_t + \kappa \bigl(D(p_t;\, \mathrm{params}_t) - S(p_t)\bigr)$$

where $\kappa$ is an adjustment speed parameter, allowing prices to be out of equilibrium within a period and producing realistic inertia.

### 2.2 The marginal pricer mechanism

The central theoretical insight: **the price at any moment is set by the group with the steepest effective demand slope given current state variables**. Formally:

$$g^* = \arg\max_g \left| \frac{\partial D_g}{\partial x_{\mathrm{binding}}} \right|$$

where $x_{\mathrm{binding}}$ is whichever state variable is currently most constraining. This produces regime switching without psychological assumptions:

- **Loose credit regime.** Households are the marginal buyer. Price is set by mortgage payment capacity: $p^*_H = \kappa \cdot y / r_m$. Household P&L is ultra-sensitive to $r_m$ and $L$; the credit gradient dominates.
- **Tight credit regime.** Institutional investors and private landlords become the marginal buyer. Price is set by cap rate logic: $p^*_I = R / (r_f + \phi)$. The yield gradient dominates.
- **Both constrained.** Volume collapses, price becomes sticky.

A crucial nested dependence: institutional P&L contains an expected capital gain term $\mathbb{E}[\Delta p]$ which is itself a function of the future household credit regime. Institutions ride the household credit cycle — their valuations are downstream of the political economy of mortgage credit, not independent anchors.

### 2.3 P&L functions by agent group

**Owner-occupiers $H$:**

Owner-occupiers maximise utility over their certainty-equivalent P&L:

$$U_H = \mathbb{E}[\Delta p] \cdot m - r_m \cdot m - \rho_i \cdot \sigma^2_{\Delta p} \cdot m$$

where $m = L \cdot p$ is the mortgage, $\rho_i \sim F_\rho$ is agent $i$'s risk aversion coefficient (§2.4), and $\sigma^2_{\Delta p}$ is local price variance estimated from recent transactions — the one bounded-rationality element in the model, justified by genuine information constraints. The final term is the certainty-equivalent cost of bearing house price risk.

The debt-to-income (DTI) constraint requires the annualised mortgage repayment $a = r_m \cdot m / (1 - (1+r_m)^{-d})$ to satisfy $a \leq \alpha \cdot y_i$, giving an effective maximum bid:

$$p^{\max}_H = \frac{\alpha \cdot y_i}{r_m} \cdot \frac{1 - (1+r_m)^{-d}}{L} - \rho_i \cdot \sigma^2_{\Delta p} \cdot L \cdot p$$

Tenants enter the purchase market when the certainty-equivalent cost of ownership falls below rent and the deposit constraint is met; they return to rental when it does not.

**Private landlords $L$:**

Private landlords access BTL mortgages at a spread over base rate (calibrated to BoE data at approximately 0.4 pp over institutional rates). Their P&L mirrors $\Pi_I$ below but with a higher funding rate, smaller portfolios (no diversification benefit), and a non-zero risk aversion drawn from the same distribution as owner-occupiers. They are an intermediate group — active in the yield regime but more credit-sensitive than pure institutional capital:

$$U_L = R(p) - r_f^{\mathrm{BTL}} \cdot p - \phi + \mathbb{E}[\Delta p] - \rho_j \cdot \sigma^2_{\Delta p} \cdot p$$

**Institutional investors $I$ (risk-neutral, wholesale funding):**

$$\Pi_I = R(p) - r_f \cdot p - \phi + \mathbb{E}[\Delta p]$$

where $R(p)$ is achievable rent on a property worth $p$, $r_f$ is the institutional funding rate, and $\phi$ covers operating costs, voids, and management. Institutions are risk-neutral ($\rho_I = 0$) by structural assumption, reflecting portfolio diversification. Their maximum bid on a pure yield basis is:

$$p^{\max}_I = \frac{R}{r_f + \phi/p}$$

The capital gain term $\mathbb{E}[\Delta p]$ is estimated adaptively from the transaction price series.

**Banks $B$ (credit supply, regulatory constraint):**

Banks do not bid on housing directly; they set $r_m$ and $L$ each period to maintain profitability under their capital constraint:

$$\Pi_B = (r_m - r_f^{\mathrm{bank}}) \cdot m - k \cdot p \cdot \mathrm{RW}$$

where $r_f^{\mathrm{bank}}$ is the bank's funding cost, $k$ is the required capital ratio, and $\mathrm{RW}$ is the risk weight on mortgage assets. When the macro state deteriorates, risk weights rise, banks tighten $L$ and raise $r_m$ spreads — endogenous credit contraction rather than a purely exogenous shock. In the default model configuration, $r_m$ and $L$ are exogenous (for comparability with Gamal et al.); bank endogeneity is activated as a toggle for the novel credit-tightening experiment (§5.2).

### 2.4 Endogenous risk aversion

Risk preferences are drawn from a log-normal distribution at agent initialisation:

$$\rho_i \sim \mathrm{LogNormal}(\mu_\rho, \sigma_\rho)$$

The log-normal ensures $\rho_i > 0$ while allowing a fat right tail of highly risk-averse agents. The parameters $(\mu_\rho, \sigma_\rho)$ are calibrated to match the observed share of income-qualified households who do not enter the ownership market — a pattern that cannot be explained by credit constraints alone.

Institutions remain risk-neutral throughout. This asymmetry — heterogeneous household risk aversion against institutional risk-neutrality — generates the key distributional result: institutional investors have a flatter effective demand curve with respect to price uncertainty and can bid more aggressively during volatile periods.

### 2.5 Macroeconomic embedding

Rather than a continuously varying macro state, the model uses three discrete macro states $\xi \in \{\mathrm{boom},\, \mathrm{neutral},\, \mathrm{recession}\}$, each associated with a fixed parameterisation of the income distribution:

$$y_{i,t} \sim \mathrm{Gamma}(k_\xi, \theta_\xi)$$

Transitions between states follow a simple Markov process with calibrated transition probabilities. Rent $R_t$ also shifts with $\xi$, coupling the rental and ownership markets through the macroeconomy. This is enough to generate the second tier of regime switching — income distribution tightening in recessions pushes marginal buyers out of the market and makes yield-based pricing dominant — without the overhead of a continuously calibrated slow-moving state variable.

### 2.6 Initial conditions

The starting state of the model is an empirical input, not a free parameter. Two components require calibration:

**Tenure allocation.** The initial distribution of housing stock across owner-occupiers, private landlords, and institutional investors is drawn from English Housing Survey data. The approximate UK baseline (circa 2015) is: 65% owner-occupied, 20% private rented (split roughly 4:1 between private landlords and institutional investors), and 15% social rented (treated as a fixed outside option). This sets the starting balance sheet weights $\lambda_g$ and avoids the unrealistic equilibrium that would emerge from an arbitrary initialisation.

**Wealth distribution by income decile.** Each owner-occupier begins with housing equity consistent with their position in the income distribution, drawn from ONS Wealth and Assets Survey data. This ensures the distributional experiments in §5.2 begin from a realistic baseline rather than an implausibly equal starting point.

---

## 3. Model Specification

### 3.1 Agents and state variables

| Agent | Key state variables |
|---|---|
| Owner-occupier $H_i$ | Income $y_i$, capital $c_i$, mortgage $m_i$, equity $e_i$, risk aversion $\rho_i$, tenure status |
| Private landlord $L_j$ | Capital $c_j$, portfolio $\{p_k\}$, BTL rate $r_f^{\mathrm{BTL}}$, rent roll $\{R_k\}$, risk aversion $\rho_j$ |
| Institutional investor $I_j$ | Capital $c_j$, portfolio $\{p_k\}$, funding rate $r_f$, operating cost $\phi$, rent roll $\{R_k\}$ |
| Bank $B$ | Capital ratio $k$, funding cost $r_f^{\mathrm{bank}}$, mortgage book $\{m_i\}$; sets $r_m$ and $L$ (endogenously when toggled) |
| Realtor | Locality radius, transaction memory, local median price estimate $\tilde{p}$ with exponential decay (as in Gamal et al.) |
| House | Price $p$, rent $r$, type, for-sale / for-rent status, age |

Global state: macro state $\xi_t \in \{\mathrm{boom}, \mathrm{neutral}, \mathrm{recession}\}$, base rate $r_f$ (exogenous policy instrument), income distribution parameters $(k_\xi, \theta_\xi)$.

### 3.2 Decision rules

Every period, each agent evaluates their certainty-equivalent utility and acts:

**Households and tenants.** Compute $U_H$ net of risk penalty. If $U_H > 0$ and DTI/LTV constraints are satisfied, enter the mortgage market. If $U_H < 0$ for $\tau$ consecutive periods, list for sale and move to the rental market. Tenants enter ownership when the certainty-equivalent cost of owning falls below rent and the deposit constraint is met.

**Private landlords.** Compute $U_L$. Enter BTL when the risk-adjusted yield exceeds the BTL funding cost. Exit (list for sale) when $U_L < 0$ for $\tau$ consecutive periods.

**Institutional investors.** Compute $\Pi_I$ with $\rho_I = 0$. Enter when the cap rate exceeds funding cost plus operating cost. Exit when it falls below.

**Banks.** In the default configuration, $r_m$ and $L$ are exogenous parameters. When the endogenous bank toggle is active, the bank solves for $(r_m, L)$ each period to maintain $\Pi_B > 0$ subject to the capital constraint; a deteriorating macro state $\xi_t$ raises $\mathrm{RW}$ and forces credit tightening.

**Price expectations.** All agents use an exponentially weighted moving average of local realtor-recorded transaction prices:

$$\mathbb{E}_t[\Delta p] = \sum_{s=0}^{T} \delta^s (p_{t-s} - p_{t-s-1})$$

This is the one bounded-rationality element in the model, producing momentum and overshooting consistent with observed UK house price cycles.

### 3.3 Market clearing

Mortgage market clears before BTL/rental market; chain resolution and realtor-based price valuation with exponential decay follow Gamal et al.'s structure. The key departure is that entry and exit thresholds are derived from P&L sign changes — the point at which utility turns negative — rather than from capital ratio heuristics.

---

## 4. Empirical Calibration

The model is calibrated to UK data; parameter choices are grounded in observable moments rather than arbitrary assumption.

| Parameter | Data source | Target moment |
|---|---|---|
| Income distribution $(k_\xi, \theta_\xi)$ | ONS Annual Survey of Hours and Earnings | Mean ≈ £30k by macro state; gamma shape |
| Initial tenure shares $\lambda_g$ | English Housing Survey | ≈ 65% OO, 20% PRS, 15% social |
| Initial wealth by income decile | ONS Wealth and Assets Survey | Median housing equity per decile |
| $r_m$ trajectory | BoE quoted mortgage rates | 2010–2024 time series |
| $r_f^{\mathrm{BTL}}$ spread | BoE BTL mortgage data | Spread over base rate ≈ 0.4 pp |
| LTV distribution | FCA Mortgage Product Sales Data | Median LTV by year |
| House price index | ONS UK HPI | Level and growth rate |
| Rental yield | ONS / Zoopla rental data | Gross yield ≈ 4–5% |
| Risk aversion $(\mu_\rho, \sigma_\rho)$ | Calibrated to match non-participation | Share of income-qualified non-buyers |
| Macro transition probabilities | UK recession dating (NBER/NIESR) | Average expansion and recession durations |

All calibration data are publicly available; no proprietary data are required.

---

## 5. Experiments

### 5.1 Mirror experiments (direct comparison with Gamal et al.)

These four shocks match Gamal et al. exactly, enabling direct qualitative comparison:

| Experiment | Shock |
|---|---|
| Interest rate rise | $r_m,\, r_f$: $3.7\% \to 8\%$ at step 300 |
| Interest rate decline | $r_m,\, r_f$: $8\% \to 3.7\%$ at step 300 |
| LTV tightening | $L$: $90\% \to 69\%$ at step 300 |
| LTV loosening | $L$: $60\% \to 74\%$ at step 300 |

Track across all experiments: median sale prices, median rents, owner/tenant ratios, market activity (transaction volume), and wealth distribution across income deciles.

### 5.2 Novel experiments enabled by the richer model

**Endogenous credit tightening.** Rather than an exogenous LTV shock, trigger a macro state transition $\xi_t: \mathrm{neutral} \to \mathrm{recession}$ and observe the bank endogenously tightening credit (bank toggle active). Does the price response and timing differ from an exogenous LTV shock of equivalent eventual magnitude? This tests whether the endogenous credit channel produces materially different dynamics — a propagation story rather than a level shift.

**Risk aversion distribution shift.** Shock $\mu_\rho$ upward (an aggregate increase in risk aversion, as in 2008) and observe the effect on market participation rates and prices. This experiment cannot be conducted in Gamal et al.'s model at all.

**Marginal pricer identification.** At each simulation step, record which group's P&L gradient is binding at the margin — that is, which group's exit or entry would most move the clearing price. Plot the marginal pricer identity over a full simulated credit cycle. This directly illustrates the core theoretical mechanism and produces a novel empirical prediction: we should observe a shift from household-dominated to yield-dominated pricing around 2022 in UK data.

**Credit subsidy distributional analysis.** Compare wealth accumulation across income deciles under loose versus tight credit regimes. This tests the core distributional claim: credit expansion functions as a wealth-proportional subsidy, disproportionately benefiting high-income households who can borrow more and capture larger capital gains, while raising prices for those without access.

---

## 6. Sensitivity Analysis

**One-factor-at-a-time (OFAT).** Vary $r_m$, $r_f$, $L$, $\phi$, $\mu_\rho$, $\sigma_\rho$, and $\bar{y}$ individually. Establish the sign and approximate magnitude of each parameter's effect on equilibrium price $p^*$ and equilibrium rent $R^*$.

**Sobol global sensitivity analysis.** On the reduced parameter set $\theta = \{r_m, r_f, L, \phi, \mu_\rho, \bar{y}\}$, compute first-order and total-order Sobol indices for variance in equilibrium price and rent. Use a Saltelli sample with $N = 1024$ (feasible at approximately 3.8 s/run via SALib).

The Sobol analysis is where the theoretical claims become falsifiable:

- Do credit parameters ($r_m$, $L$) dominate $\mathrm{Var}(p^*)$? This would confirm that housing prices primarily reflect financing structure rather than fundamentals.
- Do risk preference parameters ($\mu_\rho$, $\sigma_\rho$) show significant interaction effects with credit parameters? This would confirm the contribution of endogenous risk aversion beyond a simple level effect.
- What is the Sobol-implied ranking of policy levers by price impact?

---

## 7. Validation

Three-layer validation strategy:

**Layer 1 — Stylised facts (qualitative replication of Gamal et al.).**
The model should reproduce the same qualitative patterns:
- Interest rate rise → price decline, rental price increase
- LTV loosening → price increase, owner-occupancy rate increase
- BTL share counter-cyclical with respect to prices

**Layer 2 — Quantitative moments (calibration targets).**
Against UK data for 2010–2023:
- Price-to-rent ratio in the range 20–30×
- Gross rental yield 4–5%
- Owner-occupancy rate approximately 65%
- Price response to the 2022 rate rise: approximately 5–10% nominal decline

**Layer 3 — Regime identification (novel validation target).**
The model should exhibit household-dominated pricing in 2010–2021 (loose credit) and a shift toward yield-based pricing from 2022 (tight credit, institutional floor becomes binding). This is a validation target that Gamal et al. cannot test, as their model contains no mechanism for identifying the marginal pricer. Correspondence between the simulated regime switch and the timing of the observed UK price plateau in 2022–2023 would constitute genuine out-of-sample validation of the theoretical mechanism.

---

## 8. Paper Structure

Following JASSS format and the Dilaver & Gilbert (2023) ODD-adjacent template:

1. **Introduction** — The behavioural versus rational heterogeneity question; why P&L functions and utility rather than reduced-form heuristics; paper roadmap.
2. **Background** — Gilbert (2009) and Gamal et al. (2024) as direct lineage; Geanakoplos leverage cycles; HANK models and why they do not capture the marginal pricer mechanism; the theoretical case for structural heterogeneity.
3. **Theoretical Framework** — P&L surfaces, market clearing and tâtonnement, the marginal pricer mechanism, nested institutional dependence on the household credit regime.
4. **Model Description** — Agent types, state variables, P&L functions, endogenous risk aversion, macro embedding, initial conditions, decision rules, market clearing, temporal structure (ODD format).
5. **Calibration** — Data sources, target moments, parameter values.
6. **Experiments** — Mirror experiments (comparison with Gamal et al.) and novel experiments.
7. **Sensitivity Analysis** — OFAT results and Sobol indices, interpreted against theoretical predictions.
8. **Validation** — Three-layer validation.
9. **Discussion** — What the comparison implies; distributional results; limitations (no spatial calibration, adaptive rather than fully rational expectations, simplified bank behaviour, no mortgage vintage pass-through, no endogenous neighbourhood quality); future directions.
10. **Conclusion.**

---

## 9. Implementation Notes

**Platform.** Mesa (Python) for the ABM; SALib for Sobol analysis; pandas/NumPy for calibration and data handling.

**Computational budget.** Saltelli sample $N = 1024$ at approximately 3.8 s/run is feasible on a standard workstation. The discrete macro state adds negligible overhead. Bank endogeneity adds one feedback loop per step when toggled on.

**Modularity.** The model has a clear priority hierarchy:

The **irreducible core** is P&L-derived entry/exit rules replacing Gamal et al.'s threshold heuristics, calibrated initial tenure shares and wealth distribution, and the four mirror experiments. This alone is a publishable contribution.

The **first-tier extensions** are endogenous risk aversion (cheap to implement, directly supports the non-participation calibration and the risk aversion shift experiment) and the discrete macro state with Markov transitions (enables the recession experiment and the marginal pricer cycle). Both should be in the main model.

The **second-tier extension** is the endogenous bank ($\Pi_B$ feedback). Implement as a toggle: exogenous by default for comparability, endogenous for the credit-tightening experiment. If time is tight this can run as a single supplementary experiment rather than a core feature.

**Data.** All calibration data are publicly available (ONS, Bank of England, FCA MPS, English Housing Survey, ONS Wealth and Assets Survey). No proprietary data required.

**Division of labour.** Three largely parallelisable work streams after the model specification is locked: (i) model implementation and calibration pipeline, (ii) experiment execution and marginal pricer tracking, (iii) Sobol analysis and validation.

**Explicitly deferred to future work.** Zone-based spatial topology and neighbourhood sorting dynamics; mortgage vintage pass-through; endogenous neighbourhood quality (Schelling-type tipping dynamics). These are the natural next extensions once the core model is established and validated.

---

## 10. The Core Idea: A Pedagogical Note

*This section explains the project's animating idea in plain terms, for anyone coming to it fresh.*

### Why do housing markets behave so strangely?

Housing markets violate almost every intuition from introductory economics. Prices rise when affordability falls. Supply increases barely move prices. Interest rate changes produce asymmetric effects — small cuts cause large booms; large rises cause surprisingly modest busts. Rents and prices can diverge for years. These are not second-order puzzles; they are first-order features of one of the largest asset classes in the world.

Standard economic models struggle because they assume a representative household bidding based on the "fundamental value" of the housing service — essentially, the present discounted value of future rents. But this is not how most buyers actually bid. Most buyers ask: *how much can I borrow, and what does that imply for my monthly payment?* The binding constraint is not value but leverage — specifically, the mortgage.

### The key insight: prices reflect financing structure, not use value

A mortgage is, in economic terms, a long-dated, typically subsidised, often non-recourse call option on a real asset. If prices rise, the borrower captures all the upside beyond the deposit. If prices fall below the loan balance, the downside is bounded. The rent — or the imputed rent from owner-occupation — often covers a substantial fraction of the interest cost, meaning the option is partially self-financing.

This means the maximum price a household can rationally bid is not determined by what the house is *worth* in any fundamental sense, but by *how much leverage they can obtain at what cost*. Change the mortgage rules — lower the interest rate, raise the permitted LTV, loosen the income multiple — and you literally reshape the household's P&L function, shifting their demand curve outward and raising the equilibrium price. Because housing supply is highly inelastic in the short and medium run, this credit expansion does not produce more housing; it produces higher prices. Affordability does not improve — it may actually worsen for those at the margin of credit access.

### Three structurally distinct buyers, three different logics

The second key insight is that housing markets do not have one type of buyer — they have at least three, each operating under a fundamentally different payoff structure:

**Owner-occupiers** buy leveraged exposure to an asset they also consume. Their maximum bid is determined by credit conditions: income, the prevailing interest rate, and the permitted LTV. They are the dominant buyer in loose credit regimes and the most interest-rate-sensitive group.

**Institutional investors and private landlords** are not constrained by personal income or the owner-occupier mortgage market. They evaluate housing as a yield-generating asset: the relevant question is whether the net rental yield exceeds their funding cost. Their bid ceiling is $p = R / (r_f + \phi)$ — rent divided by the cost of capital plus operating costs. They are less sensitive to the mortgage rate and more sensitive to the rental market and their own funding conditions.

**Banks** do not bid on houses, but they determine what everyone else can bid. By setting the mortgage rate and LTV, banks effectively control the shape of the household demand curve. Their own P&L depends on the spread between lending and funding rates, subject to regulatory capital constraints. When stress rises, endogenous credit tightening amplifies the price decline — banks reduce LTV precisely when prices are falling, producing a procyclical feedback loop.

### Why the same asset behaves differently across time

At any given moment, the clearing price is set by whoever has the largest effective demand at the margin — whichever group has the steepest response to the currently binding constraint. This is what we call the **marginal pricer**.

In the UK between 2010 and 2021, cheap credit and loose lending standards meant that owner-occupiers were the marginal buyers. Prices were therefore highly sensitive to mortgage rates and LTV limits. The 2020–2021 price surge was driven almost entirely by the mortgage rate falling toward zero — the household P&L gradient was extraordinarily steep.

From 2022, as rates rose sharply, household affordability collapsed. But prices did not fall proportionally, because institutional and private landlord buyers — evaluating properties on a yield basis — provided a floor. The dominant mechanism shifted: the marginal pricer changed from credit-constrained households to yield-constrained investors. This is not a psychological story about sentiment. It is a structural consequence of which group's P&L gradient is binding.

### Why existing models miss this

Behavioural ABMs in the Gilbert–Gamal tradition capture rich dynamics through empirically motivated heuristics: households have threshold rules, landlords have rules of thumb, and the interaction of these rules generates emergent price dynamics. This is a valuable modelling approach, and this project builds directly on it.

But threshold-based heuristics make it hard to ask certain questions. Whose thresholds matter most? How does the distribution of risk aversion affect participation? What would happen if you changed the mortgage rules rather than the psychology? And crucially: *who is setting the price at the margin right now, and when will that change?*

The P&L framework answers these questions by construction. Because every entry and exit decision is derived from a clearly specified payoff function, we can identify the binding constraint for each group at each moment, track regime switches, and decompose price variance across policy parameters using global sensitivity analysis. The distributional question — who benefits from credit expansion? — also has a clean answer: those who can borrow the most gain the largest subsidy, and their gains are financed by higher prices for those who cannot.

### What this project contributes

This project does not claim that rationality is a better description of household psychology than behavioural models. It claims something more specific: that the *macro-level* dynamics of housing markets — boom-bust cycles, regime switches, yield floors, distributional divergence — are primarily driven by structural differences in the payoff functions of different buyer groups, not by the details of individual psychology. If a model of utility-maximising agents with heterogeneous P&L functions reproduces the same stylised facts as a behavioural model, that is evidence that the P&L geometry is doing the heavy lifting. It also means that policy interventions — changes to mortgage rules, capital requirements, or planning constraints — can be analysed directly in terms of how they reshape the payoff functions, without needing a model of how agents *feel* about those changes.

That is the scientific case for doing this. The practical case is that it gives policymakers a cleaner lever: change the incentive geometry, not the psychology.

---

*All data sources are publicly available. Model implementation uses Mesa (Python). Calibration data: ONS, Bank of England, FCA Mortgage Product Sales Data, English Housing Survey, ONS Wealth and Assets Survey.*

