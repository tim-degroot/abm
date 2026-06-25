"""
Configuration schema for the housing-market ABM.

Every per-period rate is expressed in *monthly* units.

Sections:
  SimConfig         - run size and seed
  SpatialConfig     - zone grid (coarse spatial abstraction; houses sit in zones)
  PropertyInitConfig- housing-stock initialisation
  AgentInitConfig   - agent endowments and heterogeneity
  CreditConfig      - the credit environment (the lever the experiments vary)
  ValuationConfig   - WTP / quality-to-money conversion
  ExpectationsConfig- adaptive-expectation and volatility parameters
  MacroConfig       - fixed income-growth regime (no stochastic transition; see
                      policies.py for designed credit-shock experiments)
  MarketConfig      - tenancy / lease parameters
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_households: int = Field(1000, gt=0)
    n_institutions: int = Field(5, gt=0)
    n_properties: int = Field(1250, gt=0)
    n_steps: int = Field(240, gt=0)
    seed: int = Field(42, ge=0)


class SpatialConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # The grid is a coarse abstraction: GRID_ROWS x GRID_COLS *zones* arranged on
    # a torus (von Neumann neighbourhood). Dwellings are assigned to zones, not to
    # individual coordinates. `search_radius` is the Chebyshev/Manhattan radius of
    # the zone consideration set; 1 = own zone + 4 neighbours (the report's spec).
    grid_rows: int = Field(5, ge=3)
    grid_cols: int = Field(5, ge=3)
    search_radius: int = Field(1, ge=0)

    @property
    def n_zones(self) -> int:
        return self.grid_rows * self.grid_cols


class PropertyInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zone_quality_sd: float = Field(0.8, ge=0)
    property_residual_sd: float = Field(0.3, ge=0)
    init_base_price: float = Field(200_000.0, gt=0)
    init_price_quality_sensitivity: float = Field(20_000.0, ge=0)
    init_ownership_prob: float = Field(
        0.90, ge=0, le=1
    )  # fraction of households initially allocated an owned home.


class AgentInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    income_mean: float = Field(36_700.0, gt=0)  # YEARLY
    income_sigma: float = Field(0.5, ge=0)
    wealth_income_mult_low: float = Field(0.5, ge=0)
    wealth_income_mult_high: float = Field(25.0, ge=0)
    ltv_dist_low: float = Field(0.70, ge=0, le=1)
    ltv_dist_high: float = Field(0.85, ge=0, le=1)
    risk_aversion_mu: float = Field(1.0, ge=0)
    risk_aversion_sigma: float = Field(0.5, ge=0)
    inst_cash_low: float = Field(15_000_000.0, ge=0)
    inst_cash_high: float = Field(100_000_000.0, ge=0)
    inst_required_return: float = Field(0.0015, ge=0)
    inst_min_yield: float = Field(0.04, ge=0)
    loss_aversion: float = Field(1.30, ge=0)


class CreditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mortgage_rate: float = Field(0.00308, ge=0)
    ltv_limit: float = Field(0.9, ge=0, le=1)
    dti_limit: float = Field(0.4, ge=0, le=1)
    loan_term_months: int = Field(300, gt=0)
    btl_funding_rate: float = Field(0.008, ge=0)
    btl_ltv: float = Field(0.50, ge=0, le=1)
    inst_funding_rate: float = Field(0.0045, ge=0)
    inst_ltv: float = Field(0.60, ge=0, le=1)


class ValuationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # Rent sensitivity to quality: rent = avg_rent * (1 + quality_sensitivity * q).
    quality_sensitivity: float = Field(0.3, ge=0)
    # Conversion of standardised quality to a *monthly* consumption value (money).
    quality_value_scale: float = Field(200.0, ge=0)
    # Baseline monthly consumption value of a median (q = 0) home
    base_housing_value: float = Field(700.0, ge=0)
    horizon: int = Field(40 * 12, ge=1)  # valuation horizon


class ExpectationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # Single EWMA smoothing weight shared across agents and expectations:
    #   E_t = smoothing * E_{t-1} + (1 - smoothing) * signal
    smoothing: float = Field(0.9, ge=0, le=1)
    # Lookback window (months) for the growth/volatility signals and the OLS.
    signal_window: int = Field(12, gt=0)
    # Initial growth expectations (monthly).
    init_price_growth: float = 0.001667
    init_rent_growth: float = 0.001667
    # Initial expectations of the *volatility of the growth rate* (monthly std)
    init_price_vol: float = Field(0.005, ge=0)
    init_rent_vol: float = Field(0.005, ge=0)
    # Idiosyncratic noise added to expectations each period.
    inst_noise_sd: float = Field(0.0003, ge=0)
    household_noise_sd: float = Field(0.0006, ge=0)


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    min_tenancy: int = Field(12, ge=0)
    early_exit_prob: float = Field(0.05, ge=0, le=1)
    normal_exit_prob: float = Field(0.2, ge=0, le=1)


class MacroConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_state: Literal["Boom", "Neutral", "Recession"] = "Neutral"
    boom_mean: float = 0.0025
    boom_sd: float = Field(0.00577, ge=0)
    neutral_mean: float = 0.000833
    neutral_sd: float = Field(0.00289, ge=0)
    recession_mean: float = -0.001667
    recession_sd: float = Field(0.00866, ge=0)
    # Risk-free monthly rate, used in the institutional outside option.
    risk_free_rate: float = Field(0.00308, ge=0)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sim: SimConfig = SimConfig()
    spatial: SpatialConfig = SpatialConfig()
    property_init: PropertyInitConfig = PropertyInitConfig()
    agent_init: AgentInitConfig = AgentInitConfig()
    credit: CreditConfig = CreditConfig()
    valuation: ValuationConfig = ValuationConfig()
    expectations: ExpectationsConfig = ExpectationsConfig()
    macro: MacroConfig = MacroConfig()
    market: MarketConfig = MarketConfig()


__all__ = [
    "Config",
    "SimConfig",
    "SpatialConfig",
    "PropertyInitConfig",
    "AgentInitConfig",
    "CreditConfig",
    "ValuationConfig",
    "ExpectationsConfig",
    "MarketConfig",
    "MacroConfig",
]
