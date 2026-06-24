# Housing Market Agent-Based Model

Project for Agent-Based Modelling (5284AGBM6Y)

## Quickstart Guide
```bash
git clone https://github.com/tim-degroot/abm.git
uv run run.py
```
## Requirements & Usage

- **Python:** 3.13+
- **Package Manager:** [uv](https://docs.astral.sh/uv/)
- **Dependencies:** Managed via pyproject.toml (includes MESA)

## Sensitivity Analysis (Sobol)

Sobol global sensitivity analysis with stochastic replicates. Each parameter set is evaluated across multiple model seeds to separate parametric sensitivity from stochastic noise.

### Stage 1: Generate the Saltelli sample (once)
```bash
uv run python sensitivity/main.py --generate --N 512
```
Creates `results/sensitivity/sobol_samples.csv` with N × (2k+2) rows. With k=9 parameters and N=512: 10,240 parameter sets.

### Stage 2: Evaluate — one invocation per device
Each device runs the full parameter set with a different model seed. Distributing seeds (not parameter chunks) keeps devices self-contained, with no coordination needed.

```bash
Device 1:  uv run python sensitivity/main.py --evaluate --model-seed 0
Device 2:  uv run python sensitivity/main.py --evaluate --model-seed 1
...
Device 10: uv run python sensitivity/main.py --evaluate --model-seed 9
```

Each device writes `results/sensitivity/seed_{S}.csv`. Copy these back to a central machine for aggregation.

### Stage 3: Aggregate (once)
```bash
uv run python sensitivity/main.py --aggregate
```

Averages responses across seeds per parameter set, then computes Sobol first-order (S1) and total-order (ST) indices with confidence intervals.

| Output file | Contents |
|---|---|
| `results/sensitivity/responses_avg.csv` | Seed-averaged responses (10,240 rows) |
| `results/sensitivity/sobol_indices.csv` | Sobol S1, S1_conf, ST, ST_conf per parameter × response |
| `results/sensitivity/sobol_indices.png` | Grouped bar chart |

### Smoke test (N=4, quick check)
```bash
uv run python sensitivity/main.py --generate --N 4
uv run python sensitivity/main.py --evaluate --model-seed 0 --n-cores 4
uv run python sensitivity/main.py --evaluate --model-seed 1 --n-cores 4
uv run python sensitivity/main.py --aggregate
```

### Experiments (future)
Add `--experiment <name>` to evaluate and aggregate stages to run the SA under a policy shock instead of the baseline. Response files are written to a subdirectory (e.g. `results/sensitivity/tightening/`).

## Designed credit-shock experiments

The baseline holds credit conditions fixed. Designed experiments apply a scheduled
credit shock through the policy layer:

```bash
uv run run.py --experiment rate-up         # mortgage/funding rate increase
uv run run.py --experiment rate-down
uv run run.py --experiment ltv-tighten      # lower LTV caps
uv run run.py --experiment ltv-loosen
uv run run.py --experiment tightening       # combined rate up + LTV/DTI tighten
uv run run.py --experiment rate-up --shock-step 120
```