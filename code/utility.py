"""
    Utility: turn expected payoffs into action values and logit choices.
    Linear utility with risk-adjusted payoff. Risk aversion is implemented
    via a loading on volatility of expected returns, g -> g - gamma * sigma.
"""

from __future__ import annotations

import numpy as np
from typing import Hashable, Mapping


def risk_adjusted_growth(
    expected_growth: float,
    expected_volatility: float,
    risk_loading: float,
) -> float:
    """
    Risk adjustment: g_adj = g - gamma * sigma.
    A risk-neutral agent (gamma = 0) gets back the raw growth.
    """
    if expected_volatility < 0.0:
        raise ValueError("expected_volatility must be non-negative")
    if risk_loading < 0.0:
        raise ValueError("risk_loading must be non-negative")
    return expected_growth - risk_loading * expected_volatility


def logit_choice(
    values: Mapping[Hashable, float],
    rng,
) -> Hashable:
    """Pick one option according to logit probabilities.

    Infeasible options carry value -inf and receive zero probability.
    """
    labels = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)

    finite = np.isfinite(vals)
    if not np.any(finite):
        for fallback in ("hold", "none", "stay", "do_nothing"):
            if fallback in values:
                return fallback
        return rng.choice(labels)

    shifted = np.where(finite, (vals - vals[finite].max()), -np.inf)
    exp_v = np.where(finite, np.exp(np.clip(shifted, -500, 0)), 0.0)
    total = exp_v.sum()
    probs = exp_v / total if total > 0 else finite / finite.sum()
    return rng.choice(labels, p=probs)


def logit_probabilities(values: Mapping[Hashable, float]):
    """Return {label: probability} for diagnostics / property selection weighting."""
    labels = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        u = 1.0 / len(labels)
        return {k: u for k in labels}
    shifted = np.where(finite, (vals - vals[finite].max()), -np.inf)
    exp_v = np.where(finite, np.exp(np.clip(shifted, -500, 0)), 0.0)
    total = exp_v.sum()
    probs = exp_v / total if total > 0 else finite / finite.sum()
    return {k: float(p) for k, p in zip(labels, probs)}


__all__ = ["risk_adjusted_growth", "logit_choice", "logit_probabilities"]
