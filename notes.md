# Notes

## Friday 5 June 2026

- Three main global parameters: 1) interest rate, 2) Loan-to-Value (LTV), and 3) deposit requirement

- Private Landlords and Owner-Occupants are households (and renters)
- Institutional Investors are corporations

- Owner-Occupiers only nearby recent transactions
- 5x5 (25) zones with city in the middle, suburbs around, etc with 25x25 homes inside


## Sensitivity Analysis — Parameter Candidates

Research hypothesis: *Credit conditions dominate housing outcomes by shifting which agent class becomes the marginal pricer.*

### Core SA inputs (11 parameters)

| # | Parameter | File | Channel |
|---|-----------|------|---------|
| 1 | `credit.mortgage_rate` | config.toml | Owner-occupier WTP ↓ |
| 2 | `credit.ltv_limit` | config.toml | Deposit constraint binds |
| 3 | `credit.dti_limit` | config.toml | Income constraint binds |
| 4 | `credit.btl_funding_rate` | config.toml | Landlord WTP ↓ (plan §6) |
| 5 | `credit.btl_ltv` | config.toml | Landlord deposit constraint |
| 6 | `agent.inst_required_return` | config.toml | Institution activity threshold |
| 7 | `agent.inst_ltv` | config.toml | Institution WTP |
| 8 | `expectations.delta` | config.toml | Expectation persistence (plan §22) |
| 9 | `agent_init.risk_aversion_mu` | config.toml | Risk aversion → bid shading |
| 10 | `market.loss_aversion_owner` | config.toml | Price stickiness, vol collapse |
| 11 | `agent.beta_action` | config.toml | Logit noise in action choice |

### Fixed calibration parameters (not SA targets)
Everything else in config.toml — spatial, property init, income/wealth distribution params, lease params, etc.

### Future pruning candidates (need code changes)
- `capital_gain_mode` + `capital_gain_growth_min/max`: commit to `fixed_level` permanently
- `ownership_mode`: drop `"target"` option, keep only `"emergent"`
- `inst_cash_low`/`inst_cash_high`: replace with single fixed value
- `inst_funding_rate_low`/`inst_funding_rate_high`: replace with single fixed value

---

## Metrics — Deprecation Candidates

Keep all current metrics; the following are candidates for removal once they're confirmed to add no value:

| Metric | Reason |
|--------|--------|
| `debug_rental_listed` | Debug only |
| `debug_ownership_listed` | Debug only |
| `debug_rental_bids_submitted` | Debug only |
| `debug_ownership_bids_submitted` | Debug only |
| `debug_ownership_bids_filtered` | Debug only |
| `debug_avg_ownership_bid` | Debug only |
| `avg_winning_bid` | Redundant with `avg_sale_price` |
| `total_household_cash` | Sum of a single net-worth component; only `total_household_net_worth` needed |
| `total_household_gross_housing_assets` | Component of net worth; not needed independently |
| `total_household_mortgage_debt` | Component of net worth; not needed independently |
| `total_household_housing_equity` | Component of net worth; not needed independently |
| `ceiling_bind_rate` | Diagnostic for WTP ceiling binding |
| `unhoused_households` | Debug diagnostic |
| `macro_state` | Replaced by simple income growth shock |

### Retained primary metrics (plan §21)
- `avg_sale_price`, `transaction_volume`, `ownership_rate`
- `institutional_ownership_share`, `household_ownership_share_of_stock`
- `household_marginal_pricer_share`
- `avg_rent`, `rental_transaction_volume`
- `total_household_net_worth`
- `price_to_rent_ratio` (new)
- `avg_loan_to_value` (new)
- `vacancy_rate` (new)
