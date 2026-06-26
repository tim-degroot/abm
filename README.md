# Who Sets the Price? A Structural Agent-Based Model of Housing Markets

Tim de Groot, Tristan Farran, George Petropoulos, and Matteo Postiferi

## Overview

Agent-based model of the UK housing market that aims to investigate the role of three different types of owners: owner-occupiers, private landlords, and institutions. How do credit conditions shape market structure (who buys) and stability (price volatility)?

## Repository structure

## Requirements

- **Python:** 3.13+
- **Package Manager:** [uv](https://docs.astral.sh/uv/)
- **Dependencies:** Managed via pyproject.toml (includes MESA)

## Quickstart Guide
```bash
git clone https://github.com/tim-degroot/abm.git
uv run run.py
```

## Policies

Policies are the design framework used for experiments on our model. These policies are defined in `code/policies.py` where their effects and parameters can be changed.

Experiments can be run using:

```bash
uv run code/run.py --experiment [policy]
```

```bash
uv run run.py --experiment rate-up         # mortgage/funding rate increase
uv run run.py --experiment rate-down
uv run run.py --experiment ltv-tighten      # lower LTV caps
uv run run.py --experiment ltv-loosen
uv run run.py --experiment tightening       # combined rate up + LTV/DTI tighten
uv run run.py --experiment rate-up --shock-step 120
```

## Policy Analysis

The Policy Analysis wrapper is designed to run multiple seeds of experiments in a parallel way and generate visualizations based on these results.

Upon reviewing the configuration within the `code/policy_analysis.py` file this analysis can be run using the following command:

```bash
uv run code/policy_analysis.py
```

This produces `responses.csv` with all results and a response figure per experiment in `results/credit_shocks/`.

## Sensitivity Analysis (Sobol)

Three-stage Sobol global sensitivity analysis with stochastic replicates:

```bash
uv run python sensitivity/main.py --generate      # stage 1: Saltelli samples (once)
uv run python sensitivity/main.py --evaluate --model-seed 0  # stage 2: per seed
uv run python sensitivity/main.py --aggregate     # stage 3: Sobol indices from all seeds
```

The analysis is configured in `sensitivity/config.yaml` — choose which parameters to vary, their bounds, and which response metrics to use. After aggregation, the 2×2 grid of first-order vs total-order indices can be plotted:

```bash
uv run python sensitivity/analysis.py   # → results/sensitivity/global_sa.png
```
