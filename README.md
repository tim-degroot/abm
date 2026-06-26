# Who Sets the Price? A Structural Agent-Based Model of Housing Markets

Tim de Groot, Tristan Farran, George Petropoulos, and Matteo Postiferi

## Overview

This repository contains an agent-based model (ABM) of the UK housing market built using MESA. It aims to investigate the role of three different types of owners: owner-occupiers, private landlords, and institutions.

> **Core Research Question:** Can a model of boundedly rational, utility-maximising agents with heterogeneous financing, valuation, and information structures reproduce the dynamics typically attributed to behavioural housing ABMs, while providing a structural explanation of who sets prices and why?

## Repository Structure

```text
.
├── code/                 # Core ABM logic, agents, markets, and run scripts
├── results/              # Output figures, CSVs, and policy analysis plots
├── sensitivity/          # Scripts and configs for Sobol global sensitivity analysis
├── test/                 # Unit and regression tests
├── parameters.md         # Detailed explanation of model parameters
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
uv run run.py

```

## Tests

We use the standard `unittest` framework. To test the model components:

```bash
uv run -m unittest                  # run all tests
uv run -m unittest -v               # verbose mode
uv run -m unittest test.test_credit # single module

```

## Policies

Policies form the design framework used for experiments on our model. These policies are defined in `code/policies.py` where their effects and parameters can be changed. Run experiments by passing the policy name as an argument:

```bash
uv run code/run.py --experiment [policy]

```

**Examples:**

```bash
uv run code/run.py --experiment rate-up         # mortgage/funding rate increase
uv run code/run.py --experiment rate-down
uv run code/run.py --experiment ltv-tighten     # lower LTV caps
uv run code/run.py --experiment ltv-loosen
uv run code/run.py --experiment tightening      # combined rate up + LTV/DTI tighten
uv run code/run.py --experiment rate-up --shock-step 120

```

## Policy Analysis

The Policy Analysis wrapper is designed to run multiple seeds of experiments in a parallel way and generate visualizations based on these results.

Upon reviewing the configuration within the `code/policy_analysis.py` file this analysis can be run using the following command:

```bash
uv run code/policy_analysis.py

```

This produces `responses.csv` with all results, a response figure per experiment, and a winner-share comparison across policies in `results/policy_analysis/`. Use `--replot` to regenerate figures from an existing `responses.csv` without re-running models.

## Sensitivity Analysis (Sobol)

Three-stage Sobol global sensitivity analysis with stochastic replicates:

```bash
uv run python sensitivity/main.py --generate                 # stage 1: Saltelli samples (once)
uv run python sensitivity/main.py --evaluate --model-seed 0  # stage 2: per seed
uv run python sensitivity/main.py --aggregate                # stage 3: Sobol indices from all seeds

```

The analysis is configured in `sensitivity/config.yaml` — choose which parameters to vary, their bounds, and which response metrics to use. After aggregation, the 2×2 grid of first-order vs total-order indices can be plotted:

```bash
uv run python sensitivity/analysis.py   # → results/sensitivity/global_sa.png

```