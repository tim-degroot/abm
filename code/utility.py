"""Utility: turn expected payoffs into action values and logit choices.

Design (per the report):
  * Linear utility in a *risk-adjusted* payoff. Risk aversion is NOT CRRA
    curvature; it is a loading on expected growth, g -> g - gamma * sigma. The
    risk adjustment is applied in valuation via the adjusted growth rate, so the
    action values here are linear in money.
  * A single logit choice function is used everywhere (no duplicate logits).
"""

from __future__ import annotations

import numpy as np
from typing import Hashable, Mapping


def risk_adjusted_growth(
    expected_growth: float,
    expected_volatility: float,
    risk_loading: float,
) -> float:
    """Reduced-form risk adjustment: g_adj = g - gamma * sigma.

    risk_loading (gamma) and expected_volatility (sigma) must be non-negative.
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
    beta: float = 1.0,
) -> Hashable:
    """Pick one option with probability proportional to exp(beta * value).

    Infeasible options carry value -inf and receive zero probability. If every
    option is infeasible, the lowest-impact option is returned if present
    ('hold'/'none'/'stay'), otherwise a uniform pick is made.
    """
    labels = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)

    finite = np.isfinite(vals)
    if not np.any(finite):
        for fallback in ("hold", "none", "stay", "do_nothing"):
            if fallback in values:
                return fallback
        return rng.choice(labels)

    shifted = np.where(finite, beta * (vals - vals[finite].max()), -np.inf)
    exp_v = np.where(finite, np.exp(np.clip(shifted, -500, 0)), 0.0)
    total = exp_v.sum()
    probs = exp_v / total if total > 0 else finite / finite.sum()
    return rng.choice(labels, p=probs)


def logit_probabilities(values: Mapping[Hashable, float], beta: float = 1.0):
    """Return {label: probability} for diagnostics / property selection weighting."""
    labels = list(values.keys())
    vals = np.array(list(values.values()), dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        u = 1.0 / len(labels)
        return {k: u for k in labels}
    shifted = np.where(finite, beta * (vals - vals[finite].max()), -np.inf)
    exp_v = np.where(finite, np.exp(np.clip(shifted, -500, 0)), 0.0)
    total = exp_v.sum()
    probs = exp_v / total if total > 0 else finite / finite.sum()
    return {k: float(p) for k, p in zip(labels, probs)}


__all__ = ["risk_adjusted_growth", "logit_choice", "logit_probabilities"]
