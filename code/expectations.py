"""
Expectations: adaptive belief formation and price forecasting.

  - Households: simple adaptive (EWMA) extrapolation of local signals
  - Institutions: linear regression on market-wide data (rolling window OLS)
"""

import numpy as np


def init_price_expectation(init_value: float) -> float:
    return init_value


def init_rent_expectation(init_value: float) -> float:
    return init_value


def adaptive_update(current: float, signal: float, delta: float) -> float:
    return delta * current + (1.0 - delta) * signal


def price_growth_signal(price_history: list, window: int) -> float:
    if len(price_history) < 2:
        return 0.0
    clean = [p for p in price_history if p is not None and p > 0]
    if len(clean) < 2:
        return 0.0
    series = clean[-window:] if len(clean) > window else clean
    growth_rates = []
    for i in range(1, len(series)):
        prev = series[i - 1]
        if prev > 0:
            growth_rates.append((series[i] - prev) / prev)
    if not growth_rates:
        return 0.0
    return float(np.median(growth_rates))


def rent_growth_signal(rent_history: list[float], window: int) -> float:
    return price_growth_signal(rent_history, window)  # same computation, different series


def household_update_expectations(
    agent,
    price_signal: float,
    rent_signal: float,
    delta: float,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    new_price_growth = adaptive_update(agent.expected_price_growth, price_signal, delta)
    new_rent_growth = adaptive_update(agent.expected_rent_growth, rent_signal, delta)
    if noise_sd > 0.0:
        new_price_growth += float(rng.normal(0.0, noise_sd))
        new_rent_growth += float(rng.normal(0.0, noise_sd))
    return new_price_growth, new_rent_growth


def _build_feature_matrix(state_history: list[dict]) -> np.ndarray:
    """Build feature matrix from state history for institutional OLS.

    Features per observation (target is price_t - price_{t-1} at row t):
      const, price_{t-1}, rent_{t-1}, volume_{t-1},
      is_boom_{t-1}, is_recession_{t-1}, avg_ltv_{t-1}, inst_share_{t-1}
    """
    n = len(state_history)
    rows = []
    for i in range(1, n):
        prev = state_history[i - 1]
        rows.append([
            1.0,  # constant
            prev.get("price", 0.0),
            prev.get("rent", 0.0),
            prev.get("volume", 0),
            1.0 if prev.get("macro") == "Boom" else 0.0,
            1.0 if prev.get("macro") == "Recession" else 0.0,
            prev.get("avg_ltv", 0.0),
            prev.get("inst_share", 0.0),
        ])
    return np.array(rows)


def _build_target_vector(state_history: list[dict]) -> np.ndarray:
    """Price change as target: Δprice_t = price_t - price_{t-1}."""
    n = len(state_history)
    targets = []
    for i in range(1, n):
        price_t = state_history[i].get("price", 0.0)
        price_prev = state_history[i - 1].get("price", 0.0)
        targets.append(price_t - price_prev)
    return np.array(targets)


def institutional_price_forecast(
    state_history: list[dict],
    window: int,
) -> float:
    """Fit rolling-window OLS on market state history; predict next price change.

    Returns predicted 1-period-ahead price change (£).  Falls back to the
    average observed price change if insufficient data (< window+1 periods).
    """
    if len(state_history) < window + 1:
        return _fallback_price_change(state_history)

    X = _build_feature_matrix(state_history[-window - 1:])
    y = _build_target_vector(state_history[-window - 1:])

    # Latest complete feature vector (at t) to predict Δprice_{t+1}
    latest = state_history[-1]
    x_pred = np.array([
        1.0,
        latest.get("price", 0.0),
        latest.get("rent", 0.0),
        latest.get("volume", 0),
        1.0 if latest.get("macro") == "Boom" else 0.0,
        1.0 if latest.get("macro") == "Recession" else 0.0,
        latest.get("avg_ltv", 0.0),
        latest.get("inst_share", 0.0),
    ])

    # OLS via least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        prediction = float(x_pred @ coeffs)
    except np.linalg.LinAlgError:
        prediction = _fallback_price_change(state_history)

    return prediction


def _fallback_price_change(state_history: list[dict]) -> float:
    if len(state_history) < 2:
        return 0.0
    changes = []
    for i in range(1, len(state_history)):
        changes.append(
            state_history[i].get("price", 0.0) - state_history[i - 1].get("price", 0.0)
        )
    return float(np.median(changes)) if changes else 0.0


def institutional_rent_growth_signal(state_history: list[dict]) -> float:
    """Latest rent growth rate from state history."""
    if len(state_history) < 2:
        return 0.0
    r_prev = state_history[-2].get("rent", 0.0)
    r_curr = state_history[-1].get("rent", 0.0)
    if r_prev > 0:
        return (r_curr - r_prev) / r_prev
    return 0.0


def institutional_update_expectations(
    agent,
    state_history: list[dict],
    window: int,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Update institutional expectations using rolling OLS price forecast.

    Returns (new_expected_price_growth, new_expected_rent_growth).
    """
    predicted_change = institutional_price_forecast(state_history, window)

    # Convert level-change prediction to monthly growth rate
    current_price = state_history[-1].get("price", 1.0) if state_history else 1.0
    price_growth = predicted_change / max(current_price, 1e-9)

    # Rent expectations: use the latest observed rent growth signal
    if len(state_history) >= 2:
        r_prev = state_history[-2].get("rent", 0.0)
        r_curr = state_history[-1].get("rent", 0.0)
        rent_growth = (r_curr - r_prev) / max(r_prev, 1e-9) if r_prev > 0 else 0.0
    else:
        rent_growth = 0.0

    if noise_sd > 0.0:
        price_growth += float(rng.normal(0.0, noise_sd))
        rent_growth += float(rng.normal(0.0, noise_sd))

    return price_growth, rent_growth
