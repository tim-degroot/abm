"""Expectation formation: adaptive belief updating and price forecasting.

This module is the single source of truth for how signals are turned into
expectations. The model computes the raw signals once per step (locally per zone
for households, globally for institutions) and calls these helpers; agents store
only the resulting expectation values.

  - Households: EWMA (adaptive) extrapolation of local price/rent growth, plus an
    EWMA estimate of growth volatility used for risk adjustment.
  - Institutions: rolling-window OLS forecast of the next price change on the
    market state, plus an EWMA rent-growth signal.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core EWMA primitives
# ---------------------------------------------------------------------------

def adaptive_update(current: float, signal: float, delta: float) -> float:
    """One EWMA step: E_t = delta * E_{t-1} + (1 - delta) * signal."""
    return delta * current + (1.0 - delta) * signal


def growth_signal(history: list[float], window: int) -> float:
    """Median period-over-period growth rate over the last `window` observations.

    `history` is a list of level observations (prices or rents). Returns 0.0 when
    there is not enough data. Uses the median to be robust to outliers/None gaps.
    Note: None entries are treated as missing *observations* (skipped) but the
    growth is computed only between consecutive present values, so sparse series
    do not spuriously inflate growth.
    """
    clean = [x for x in history if x is not None and x > 0]
    if len(clean) < 2:
        return 0.0
    series = clean[-window:] if len(clean) > window else clean
    rates = [
        (series[i] - series[i - 1]) / series[i - 1]
        for i in range(1, len(series))
        if series[i - 1] > 0
    ]
    if not rates:
        return 0.0
    return float(np.median(rates))


def volatility_signal(history: list[float], window: int) -> float:
    """Std of period-over-period growth over the last `window` observations.

    Returns 0.0 when there is not enough data to estimate a spread.
    """
    clean = [x for x in history if x is not None and x > 0]
    if len(clean) < 3:
        return 0.0
    series = clean[-window:] if len(clean) > window else clean
    rates = [
        (series[i] - series[i - 1]) / series[i - 1]
        for i in range(1, len(series))
        if series[i - 1] > 0
    ]
    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))


# ---------------------------------------------------------------------------
# Institutional rolling-window OLS price forecast
# ---------------------------------------------------------------------------

def _design_matrix(state_history: list[dict]) -> np.ndarray:
    """Feature rows for the OLS, one per transition t-1 -> t.

    Features (at t-1): const, price, rent, volume, is_boom, is_recession,
    avg_ltv, inst_share.
    """
    rows = []
    for i in range(1, len(state_history)):
        prev = state_history[i - 1]
        rows.append([
            1.0,
            prev.get("price", 0.0) or 0.0,
            prev.get("rent", 0.0) or 0.0,
            prev.get("volume", 0) or 0,
            1.0 if prev.get("macro") == "Boom" else 0.0,
            1.0 if prev.get("macro") == "Recession" else 0.0,
            prev.get("avg_ltv", 0.0) or 0.0,
            prev.get("inst_share", 0.0) or 0.0,
        ])
    return np.array(rows, dtype=float)


def _target_vector(state_history: list[dict]) -> np.ndarray:
    """Price change Delta price_t = price_t - price_{t-1}."""
    targets = []
    for i in range(1, len(state_history)):
        p_t = state_history[i].get("price", 0.0) or 0.0
        p_prev = state_history[i - 1].get("price", 0.0) or 0.0
        targets.append(p_t - p_prev)
    return np.array(targets, dtype=float)


def _fallback_price_change(state_history: list[dict]) -> float:
    if len(state_history) < 2:
        return 0.0
    changes = [
        (state_history[i].get("price", 0.0) or 0.0)
        - (state_history[i - 1].get("price", 0.0) or 0.0)
        for i in range(1, len(state_history))
    ]
    return float(np.median(changes)) if changes else 0.0


def institutional_price_forecast(state_history: list[dict], window: int) -> float:
    """Predicted next-period price *change* (in money) via rolling-window OLS.

    Falls back to the median recent change when there is too little data or the
    regression is ill-conditioned.
    """
    if len(state_history) < window + 1:
        return _fallback_price_change(state_history)

    recent = state_history[-window - 1:]
    X = _design_matrix(recent)
    y = _target_vector(recent)
    if X.shape[0] < X.shape[1]:  # underdetermined
        return _fallback_price_change(state_history)

    latest = state_history[-1]
    x_pred = np.array([
        1.0,
        latest.get("price", 0.0) or 0.0,
        latest.get("rent", 0.0) or 0.0,
        latest.get("volume", 0) or 0,
        1.0 if latest.get("macro") == "Boom" else 0.0,
        1.0 if latest.get("macro") == "Recession" else 0.0,
        latest.get("avg_ltv", 0.0) or 0.0,
        latest.get("inst_share", 0.0) or 0.0,
    ], dtype=float)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return float(x_pred @ coeffs)
    except np.linalg.LinAlgError:
        return _fallback_price_change(state_history)


def institutional_rent_growth_signal(state_history: list[dict], window: int) -> float:
    """EWMA-style robust rent-growth estimate from the global rent series.

    Replaces the old one-period point estimate with a windowed median growth,
    consistent with the price-forecast treatment.
    """
    rents = [s.get("rent") for s in state_history]
    return growth_signal(rents, window)


__all__ = [
    "adaptive_update",
    "growth_signal",
    "volatility_signal",
    "institutional_price_forecast",
    "institutional_rent_growth_signal",
]
