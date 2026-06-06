"""
Expectation formation module.

All agents use adaptive expectations:

    E_t = delta * E_{t-1} + (1 - delta) * Signal_t

Signals are computed from recent transaction histories maintained
by the model. This module contains no agent state — it provides
pure functions that agents call to update their own expectations.

Separating expectation logic here means future researchers can
replace the entire expectation system without touching agent code.
"""

import numpy as np


# Default smoothing parameter: higher = more inertia
DEFAULT_DELTA = 0.7


def adaptive_update(current_expectation, signal, delta=DEFAULT_DELTA):
    """
    Single adaptive expectation update.

    current_expectation : float, E_{t-1}
    signal              : float, observed signal S_t
    delta               : float in [0,1], weight on prior expectation

    Returns updated expectation E_t.
    """
    return delta * current_expectation + (1.0 - delta) * signal


def price_growth_signal(recent_prices):
    """
    Compute price growth signal from a sequence of recent average prices.

    recent_prices : list of floats, most recent last.
    Returns growth rate, or 0.0 if insufficient data.
    """
    if len(recent_prices) < 2:
        return 0.0
    p_prev = recent_prices[-2]
    p_curr = recent_prices[-1]
    if p_prev <= 0:
        return 0.0
    return (p_curr - p_prev) / p_prev


def rent_growth_signal(recent_rents):
    """
    Compute rent growth signal from a sequence of recent average rents.

    recent_rents : list of floats, most recent last.
    Returns growth rate, or 0.0 if insufficient data.
    """
    if len(recent_rents) < 2:
        return 0.0
    r_prev = recent_rents[-2]
    r_curr = recent_rents[-1]
    if r_prev <= 0:
        return 0.0
    return (r_curr - r_prev) / r_prev


def init_price_expectation(baseline_growth=0.02):
    """
    Initial expected price growth.
    Calibrated loosely to long-run UK nominal house price growth.
    """
    return baseline_growth


def init_rent_expectation(baseline_growth=0.02):
    """
    Initial expected rent growth.
    """
    return baseline_growth
