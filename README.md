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

## Sensitivity Analysis

```bash
uv run python -m sensitivity
```

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