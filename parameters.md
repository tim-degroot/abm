# Parameter Sources

## Table 1 — Parameters from Gamal et al. (2024)

Direct analogues between this model and Gamal, Elsenbroich, Gilbert, Heppenstall & Zia (2024) *"A Behavioural Agent-Based Model for Housing Markets: Impact of Financial Shocks in the UK"*, JASSS 27(4) 5. [doi:10.18564/jasss.5518](https://doi.org/10.18564/jasss.5518)

| Parameter     | Config path               | Our value           | Gamal value | Gamal source            | Notes                                                                                                                                                    |
| ------------- | ------------------------- | ------------------- | ----------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mortgage rate | `credit.mortgage_rate`    | 0.00308 (≈3.7% APR) | 3.7%        | Bank of England (2023a) | Direct match. Gamal labels this `interest-rate (I_t)`.                                                                                                   |
| LTV limit     | `credit.ltv_limit`        | 0.90                | 90%         | Bank of England (2023a) | Direct match. Gamal labels this `LTV (LTV_t)`.                                                                                                           |
| Loan term     | `credit.loan_term_months` | 240 (20 yrs)        | 25 years    | Author assumption       | Direct match. Gamal labels this `mortgage-duration`.                                                                                                     |
| Income mean   | `agent_init.income_mean`  | 36,700              | £30,000     | ONS (2023a)             | Updated. Gamal used a gamma(α=2) truncated to £15K–£45K with mean £30K citing ONS (2023a). We use the direct ONS median — see Table 2.                   |
| DTI limit     | `credit.dti_limit`        | 0.40                | 0.33 (α)    | Author assumption       | Updated. Gamal's `affordability (α)` is an author assumption with no external source cited. We use a value grounded in UK lender practice — see Table 2. |

---

## Table 2 — Parameters from UK data

| Parameter                  | Config path                                    | Value               | Source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------- | ---------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mortgage rate              | `credit.mortgage_rate`                         | 0.00308 (≈3.7% APR) | Bank of England (2023a), via Gamal (2024). Matches the BoE Bank Rate / average mortgage rate period used in their calibration.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| LTV limit                  | `credit.ltv_limit`                             | 0.90                | Bank of England (2023a), via Gamal (2024). Standard maximum LTV for UK residential mortgages.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Income mean                | `agent_init.income_mean`                       | 36,700              | Office for National Statistics (2025). *Average household income, UK: financial year ending 2024*. Median household disposable income was £36,700. [Link](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/bulletins/householddisposableincomeandinequality/financialyearending2024)                                                                                                                                                                                                                    |
| Income dispersion          | `agent_init.income_sigma`                      | 0.5                 | ONS income distribution data. A lognormal with σ=0.5 produces a Gini coefficient consistent with UK post-tax income inequality. Author assumption for the parametric form.                                                                                                                                                                                                                                                                                                                                                                              |
| DTI limit                  | `credit.dti_limit`                             | 0.40                | CeMAP mortgage underwriting guidance: "Lenders typically prefer a DTI below 40%… A typical maximum DTI accepted by UK lenders is 45% for most residential mortgages." [Source](https://cemap123.co.uk/mortgage-underwriting-cemap-students/). Also consistent with FCA post-MMR responsible lending rules.                                                                                                                                                                                                                                              |
| Base house price           | `property_init.init_base_price`                | 200,000             | Office for National Statistics (2025). *UK House Price Index: December 2024*. Terraced house UK average: £223,808; UK average excluding London: ~£226,000. [Link](https://www.gov.uk/government/statistics/uk-house-price-index-for-december-2024/uk-house-price-index-summary-december-2024). Our 200K reflects a non-London/South East calibration.                                                                                                                                                                                                   |
| Price-quality sensitivity  | `property_init.init_price_quality_sensitivity` | 25,000              | Calibrated so that ~95% of properties fall in the £160K–£240K range at ±2σ quality, covering the regional UK spread from North East (£161K) to East Midlands (£242K). Source: ONS UK HPI regional breakdown (same release as above).                                                                                                                                                                                                                                                                                                                    |
| Dwellings:households ratio | `sim.n_properties` / `sim.n_households`        | ~1.25:1             | MHCLG (2025). *Dwelling Stock Estimates, England: 31 March 2024*: 25.6M dwellings. [Link](https://www.gov.uk/government/statistics/dwelling-stock-estimates-in-england-2024/dwelling-stock-estimates-england-31-march-2024). ONS (2025). *Families and Households, UK: 2024*: 28.6M households. [Link](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024). Raw ratio ~1.07:1; adjusted upward because our model excludes social housing (~16% of stock) and needs BTL surplus. |

---

## Table 3 — Model design parameters (Author assumption)

Parameters with no external source. Labelled "Author assumption" following the convention in Gamal (2024) Table 2.

### Simulation scale

| Parameter    | Config path          | Default | Rationale                                                                                           |
| ------------ | -------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| Households   | `sim.n_households`   | 500     | Model scale. Chosen for tractability across 10 stochastic seeds × 8192 Sobol samples.               |
| Institutions | `sim.n_institutions` | 5       | Arbitrary; enough to avoid single-institution monopoly without fragmenting the BTL market.          |
| Steps        | `sim.n_steps`        | 1200    | 100 years at monthly resolution. Sufficient for shock experiments (Gamal uses 100 years post-shock). |
| Seed         | `sim.seed`           | 42      | RNG seed for reproducibility.                                                                       |

### Spatial

| Parameter      | Config path              | Default | Rationale                                                                                               |
| -------------- | ------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| Grid rows/cols | `spatial.grid_rows/cols` | 5×5     | Coarse spatial abstraction; 25 zones. A torus topology avoids edge effects.                             |
| Search radius  | `spatial.search_radius`  | 1       | Each household sees own zone + 4 von Neumann neighbours (5 zones). Matches Gamal's `search-length = 5`. |

### Property initialisation

| Parameter              | Config path                          | Default | Rationale                                                                                  |
| ---------------------- | ------------------------------------ | ------- | ------------------------------------------------------------------------------------------ |
| Zone quality SD        | `property_init.zone_quality_sd`      | 0.8     | Creates meaningful between-zone quality variation (~0.8σ between zones).                   |
| Property residual SD   | `property_init.property_residual_sd` | 0.3     | Within-zone quality variation; ~0.3σ residual after zone mean.                             |
| Initial ownership prob | `property_init.init_ownership_prob`  | 0.80    | 80% of households start as owner-occupiers; 20% enter as renters, creating natural demand. |

### Agent initialisation

| Parameter                     | Config path                              | Default               | Rationale                                                                                                               |
| ----------------------------- | ---------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Wealth multiplier range       | `agent_init.wealth_income_mult_low/high` | [0.5, 25.0]           | Cash = income × multiplier. Produces wide wealth distribution. Matches UK wealth inequality pattern.                    |
| Initial LTV distribution      | `agent_init.ltv_dist_low/high`           | [0.70, 0.85]          | Random initial LTV for legacy mortgages, capped at `credit.ltv_limit`.                                                  |
| Risk aversion params          | `agent_init.risk_aversion_mu/sigma`      | μ=1.0, σ=0.5          | Lognormal distribution of household risk aversion.                                                                      |
| Institutional cash            | `agent_init.inst_cash_low/high`          | [7.5M, 50M]           | Set so 5 institutions can absorb 350 properties at 40% deposit (~5.6M/inst at 500 HH scale). Scales with n_properties. |
| Institutional required return | `agent_init.inst_required_return`        | 0.0015 (≈1.8% annual) | Minimum monthly return on BTL properties.                                                                               |
| Loss aversion                 | `agent_init.loss_aversion`               | 1.30                  | Kahneman & Tversky (1979) endowment effect parameter.                                                                   |

### Credit (BTL & institution)

| Parameter                | Config path                | Default            | Rationale                                                                                              |
| ------------------------ | -------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| BTL funding rate         | `credit.btl_funding_rate`  | 0.008 (≈9.6% APR)  | Higher than owner-occupier rate, reflecting BTL risk premium.                                          |
| BTL LTV                  | `credit.btl_ltv`           | 0.50               | Typical UK BTL maximum LTV (lower than residential due to regulatory treatment).                       |
| Institution funding rate | `credit.inst_funding_rate` | 0.0045 (≈5.4% APR) | Between mortgage rate and BTL rate; institutions access cheaper capital than individual BTL landlords. |
| Institution LTV          | `credit.inst_ltv`          | 0.60               | Common UK institutional BTL LTV ceiling.                                                               |

### Valuation

| Parameter                | Config path                     | Default       | Rationale                                                                             |
| ------------------------ | ------------------------------- | ------------- | ------------------------------------------------------------------------------------- |
| Rent quality sensitivity | `valuation.quality_sensitivity` | 0.3           | Rent = base_rent × (1 + 0.3 × q), so a +1σ quality property commands 30% higher rent. |
| Quality value scale      | `valuation.quality_value_scale` | 250           | Consumption value of quality increment for owner-occupier WTP.                        |
| Base housing value       | `valuation.base_housing_value`  | 800 (£/month) | Monthly consumption value of a median-quality home.                                   |
| Horizon                  | `valuation.horizon`             | 480 (40 yrs)  | DCF horizon for WTP calculations.                                                     |

### Expectations

| Parameter                 | Config path                            | Default                  | Rationale                                                                                    |
| ------------------------- | -------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| EWMA smoothing            | `expectations.smoothing`               | 0.90                     | High persistence; expectations adjust slowly to new signals.                                 |
| Signal window             | `expectations.signal_window`           | 18 (months)              | Lookback for OLS growth estimates and volatility.                                            |
| Initial price/rent growth | `expectations.init_price/rent_growth`  | 0.001667 (≈2% annual)    | Calibrated to ~2% annual trend growth, consistent with long-run UK house price appreciation. |
| Initial volatility        | `expectations.init_price/rent_vol`     | 0.005 (≈1.7% monthly σ)  | Baseline expected volatility.                                                                |
| Idiosyncratic noise SD    | `expectations.inst/household_noise_sd` | Inst: 0.0003, HH: 0.0006 | Small noise added to expectations each period; households noisier than institutions.         |

### Macro

| Parameter             | Config path                            | Default       | Rationale                                                                                                    |
| --------------------- | -------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| Initial state         | `macro.initial_state`                  | "Neutral"     | No initial shock; experiments apply shocks via policies.                                                     |
| Income growth regimes | `macro.boom/neutral/recession_mean/sd` | See config.py | Monthly income growth rates and volatilities for three macro states. Calibrated to UK GDP per capita growth. |

### Market (tenancy)

| Parameter        | Config path               | Default    | Rationale                                     |
| ---------------- | ------------------------- | ---------- | --------------------------------------------- |
| Min tenancy      | `market.min_tenancy`      | 12 months  | Typical UK assured shorthold tenancy minimum. |
| Early exit prob  | `market.early_exit_prob`  | 0.05/month | Probability a tenant breaks lease early.      |
| Normal exit prob | `market.normal_exit_prob` | 0.20/month | Probability a tenancy ends at term.           |

---

## References

- Bank of England (2023a). *Bank Rate history and data*.
- FCA (2014). *Mortgage Market Review: Responsible Lending Rules*. [Link](https://www.fca.org.uk/news/press-releases/new-mortgage-rules-come-force)
- FCA (2016). *TR16/4: Embedding the Mortgage Market Review: Responsible Lending Review*. [Link](https://www.fca.org.uk/publications/thematic-reviews/tr16-4-embedding-mortgage-market-review-responsible-lending-review)
- Gamal, Y., Elsenbroich, C., Gilbert, N., Heppenstall, A. & Zia, K. (2024). *A Behavioural Agent-Based Model for Housing Markets: Impact of Financial Shocks in the UK*. JASSS 27(4) 5. [doi:10.18564/jasss.5518](https://doi.org/10.18564/jasss.5518)
- MHCLG (2025). *Dwelling Stock Estimates, England: 31 March 2024*. [Link](https://www.gov.uk/government/statistics/dwelling-stock-estimates-in-england-2024/dwelling-stock-estimates-england-31-march-2024)
- ONS (2025). *Average household income, UK: financial year ending 2024*. [Link](https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/bulletins/householddisposableincomeandinequality/financialyearending2024)
- ONS (2025). *UK House Price Index: December 2024*. [Link](https://www.gov.uk/government/statistics/uk-house-price-index-for-december-2024/uk-house-price-index-summary-december-2024)
- ONS (2025). *Families and Households, UK: 2024*. [Link](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/bulletins/familiesandhouseholds/2024)
- Kahneman, D. & Tversky, A. (1979). *Prospect Theory: An Analysis of Decision under Risk*. Econometrica 47(2), 263–292.
