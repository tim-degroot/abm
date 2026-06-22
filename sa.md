# Sensitivity Analysis

## Requirements

- "Robust sensitivity analysis with adequate samples is mandatory" (Project Requirements)
- "Test interaction range and agent count – both dramatically alter emergence timing and stability" (Presentation Guidelines)
- "Global Sensitivity Analysis Plot - 1st and Total Order" (Presentation Guidelines)
- "Critical for policy: Small parameter shifts (e.g., lending rules) may trigger systemic collapse/booms." (Presentation Guidelines)

## Method

**Sobol' (Saltelli 2002)** — variance-based global SA, 1st and total order indices with bootstrapped confidence intervals.

Total runs = $N \times (2k + 2)$, where $k$ = number of parameters, $N$ = base sample size.

- $k=6$, $N=64$ → 896 runs (~1–2 hrs on 8 cores)
- $k=6$, $N=128$ → 1792 runs (~2–4 hrs)
- To add a parameter: add one entry to `parameters:` list in config, re-run (total runs grows by $2N$ per param)

## Files

| File | Purpose |
|---|---|
| `code/sa_config.yaml` | **All tunable variables** — N, steps, parameters, responses, cores, output paths |
| `code/sensitivity.py` | Harness: sampling → parallel batch run → Sobol analysis → plot |
| `code/config.py` | Added `spatial.search_radius` field (exposes interaction range) |
| `code/model.py` | Updated `_build_zone_adjacency` to use `search_radius` with configurable Manhattan distance |
| `results/sa/sobol_indices.png` | Grouped bar chart (1st + total order per response) |
| `results/sa/sobol_results.csv` | Raw parameter values + scalar responses per run |
| `results/sa/sobol_indices.csv` | Sobol indices table (S1, S1_conf, ST, ST_conf per param per response) |

## Parameters (k=6)

| # | Parameter | Config path | Default | Distribution | Range | Mechanism |
|---|---|---|---|---|---|---|
| 1 | Agent count | `sim.n_households` | 100 | int | 100–300 | Market tightness — more buyers → competition, price growth |
| 2 | Search radius | `spatial.search_radius` | 1 (von Neumann) | int | 0–3 | Wider search → more properties bid on → price convergence |
| 3 | Mortgage rate | `credit.mortgage_rate` | 0.0025 (3% APR) | log-uniform | 0.001–0.01 (1.2–12.7% APR) | Higher rate → higher monthly payment → lower WTP |
| 4 | LTV limit | `credit.ltv_limit` | 0.85 | uniform | 0.60–0.95 | Deposit constraint → looser → more buyers, higher prices |
| 5 | Inst required return | `agent_init.inst_required_return` | 0.004 | log-uniform | 0.001–0.01 | Lower → institutions buy aggressively, market dominance |
| 6 | Loss aversion λ | `agent_init.loss_aversion` | 1.30 | uniform | 0–5 | Higher → sellers hold out → volume collapse, price stickiness |

## Response Functions

| Response | Metric | Reduce | What it captures |
|---|---|---|---|
| `mean_price` | `avg_sale_price` | nanmean (last 120 steps) | Long-run price level — "is the market expensive?" |
| `price_volatility` | `avg_sale_price` | nanstd (full run) | Market instability — "how much does it swing?" |
| `inst_share_final` | `institutional_ownership_share` | last | Market structure — "who owns the stock?" |
| `max_drawdown` | custom | peak-to-trough | Collapse severity — "how bad does it get?" |

## Configuration (`code/sa_config.yaml`)

Every tunable is in this one file:

- **`sobol.N`** — base sample size (default 64)
- **`sobol.seed`** — RNG seed for Saltelli sequence
- **`steps`** — simulation length (720 for full, 24 for quick test)
- **`parameters[]`** — list of {name, path, bounds, distribution} — add/remove entries here
- **`responses[]`** — list of {name, metric, reduce, tail, custom} — define new response functions here
- **`parallel.n_cores`** — set to -1 for all cores, or a fixed number
- **`output.dir`** — where results go (default `results/sa/`)

## Extensibility

**Add a 7th parameter:** edit `sa_config.yaml` — add one entry to the `parameters:` list. No Python code changes.

```yaml
  - name: dti_limit
    path: credit.dti_limit
    bounds: [0.20, 0.50]
    distribution: uniform
```

**Add a 5th response:** edit `sa_config.yaml` — add one entry to the `responses:` list.

**Change run length for quick tests:** set `steps: 24` in YAML, run, then set back to 720.

**Colleagues refactor the model:** only `run_single()` in `sensitivity.py` needs updating if the Model/Config API changes. The YAML config stays the same.

## Usage

```bash
# Full SA (default N=64, 720 steps, all cores)
uv run python -m code.sensitivity

# Quick smoke test (edit sa_config.yaml → steps: 24)
uv run python -m code.sensitivity
```
