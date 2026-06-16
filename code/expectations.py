"""
Expectation formation module.
"""

import numpy as np

# NOTE: These are standalone fallback defaults only. In a running model the
# canonical values come from config.toml (expectations.delta, init_price_growth,
# init_rent_growth) and are passed in explicitly by the agents. Keep them in
# sync with config.toml if you change them.

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


def price_growth_signal(
    recent_prices,
):  # extremely coarse, need a decent extrapolation method
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
    if p_prev <= 0:
        return 0.0
    return (p_curr - p_prev) / p_prev


def rent_growth_signal(
    recent_rents,
):  # extremely coarse, need a decent extrapolation method
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


# Missing the bounded vision and difference in signal for institutionals.
