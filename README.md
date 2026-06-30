# Who Sets the Price? A Structural Agent-Based Model of Housing Markets

Tim de Groot, Tristan Farran, George Petropoulos, and Matteo Postiferi

## Overview

This repository contains an agent-based model (ABM) of the housing market built using MESA. It aims to investigate the role of three different types of owners: owner-occupiers, private landlords, and institutions.

> **Core Research Question:** Can a model of boundedly rational, utility-maximising agents with heterogeneous financing, valuation, and information structures reproduce the dynamics typically attributed to behavioural housing ABMs, while providing a structural explanation of who sets prices and why?

A detailed description of all model parameters, their calibration sources, and sensitivity ranges can be found in [`parameters.md`](parameters.md).

## Repository Structure

```text
.
├── code/                 # Core ABM package
│   ├── core/             #   Model logic: agents, markets, credit, expectations
│   ├── settings/         #   Config, metrics, policies, sensitivity config
│   ├── sensitivity/      #   Sobol global sensitivity analysis (generate / evaluate / aggregate)
│   ├── plotting/         #   Jupyter notebooks for publication figures
│   ├── test/             #   Unit and regression tests (unittest)
│   ├── run.py            #   CLI entry point for single model runs
│   └── policy_analysis.py #   Multi-seed policy experiment runner
├── results/              # Output data, figures, and CSVs
├── parameters.md         # Detailed parameter sources and calibration
├── pyproject.toml        # Dependency management via uv
└── README.md

```

## Requirements

* **Python:** 3.13+
* **Package Manager:** [uv](https://docs.astral.sh/uv/)
* **Dependencies:** Managed via pyproject.toml (includes MESA)

## Quickstart Guide

```bash
git clone https://github.com/tim-degroot/abm.git
uv run -m code.run

```

## Tests

We use the standard `unittest` framework. To test the model components:

```bash
uv run -m unittest                 # run all tests
uv run -m unittest -v              # verbose mode
uv run -m unittest code.test.test_credit  # single module

```

## Policies

Policies form the design framework used for experiments on our model. These policies are defined in `code/settings/policies.py` where their effects and parameters can be changed. Run experiments by passing the policy name as an argument:

```bash
uv run -m code.run --experiment [policy]

```

**Examples:**

```bash
uv run -m code.run --experiment rate-up         # mortgage/funding rate increase
uv run -m code.run --experiment rate-down
uv run -m code.run --experiment ltv-tighten     # lower LTV caps
uv run -m code.run --experiment ltv-loosen
uv run -m code.run --experiment tightening      # combined rate up + LTV/DTI tighten
uv run -m code.run --experiment rate-up --shock-step 120

```

## Policy Analysis

The Policy Analysis wrapper is designed to run multiple seeds of experiments in a parallel way and generate visualizations based on these results.

This analysis can be run using the following command:

```bash
uv run -m code.policy_analysis

```

This produces `responses.csv` with all results, a response figure per experiment, and a winner-share comparison across policies in `results/policy_analysis/`. Use `--replot` to regenerate figures from an existing `responses.csv` without re-running models.

## Sensitivity Analysis (Sobol)

Three-stage Sobol global sensitivity analysis with stochastic replicates across 10 model seeds:

```bash
uv run -m code.sensitivity --generate                 # stage 1: Saltelli samples (once)
uv run -m code.sensitivity --evaluate --model-seed 0  # stage 2: per seed (repeat 0-9)
uv run -m code.sensitivity --aggregate                # stage 3: Sobol indices across seeds

```

The analysis is configured in `code/settings/sensitivity_config.yaml` — choose which parameters to vary (uniform/log-uniform/int), their bounds, and which response metrics to use. The `--aggregate` step computes Sobol indices independently per seed, then reports the mean ± 95% CI (using a t-distribution across seeds). This ensures error bars properly shrink as more stochastic replicates are added.

## Figures

Figures are generated from Jupyter notebooks in `code/plotting/`:

| Notebook | Generates |
|---|---|
| `code/plotting/single_run.ipynb` | Single-run summary plots for model validation |
| `code/plotting/policy_analysis.ipynb` | Policy response time series with confidence bands, winner-share comparisons |
| `code/plotting/sensitivity.ipynb` | Sobol heatmap, S₁ vs Sₜ scatter, thematic bar charts for ownership, prices, and welfare |

To regenerate all figures from existing results, open a notebook and run all cells. The notebooks read from `results/` and write high-resolution PNGs to `results/sensitivity/figs/` and `results/policy_analysis/`.

## High Performance Computing

Two `SBATCH` scripts are provided to run the policy analysis and sensitivity analysis on a HPC cluster. These require [uv](https://docs.astral.sh/uv/) to be installed on the cluster:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
sbatch policy_analysis.sh
```

```bash
uv run -m code.sensitivity --generate   # generates Saltelli indices; can be run on login node
sbatch sensitivity_analysis.sh          # submits the 10 seeds for sensitivity analysis run on 2 nodes at a time.
uv run -m code.sensitivity --aggregate  # Generates Sobol indices from all seeds; can be run on login node
```
