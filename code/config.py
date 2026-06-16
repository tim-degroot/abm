from __future__ import annotations

from pathlib import Path
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field


class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_households: int = Field(100, gt=0)
    n_institutions: int = Field(5, gt=0)
    n_properties: int = Field(120, gt=0)
    seed: int = 42
    n_steps: int = Field(720, gt=0)


class SpatialConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    grid_rows: int = Field(5, ge=3)
    grid_cols: int = Field(5, ge=3)

    @property
    def n_zones(self) -> int:  #
        return self.grid_rows * self.grid_cols


class PropertyInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zone_quality_sd: float = Field(0.5, ge=0)
    property_residual_sd: float = Field(0.5, ge=0)
    base_price: float = Field(200_000.0, gt=0)
    price_sensitivity: float = Field(50_000.0, ge=0)
    clustering_strength: float = Field(0.5, ge=0, le=1)


class AgentInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    income_mean: float = Field(35_000.0, gt=0)
    income_sigma: float = Field(10_000.0, ge=0)
    wealth_income_mult_low: float = Field(0, ge=0)
    wealth_income_mult_high: float = Field(25.0, ge=0)
    ltv_dist_low: float = Field(0.70, ge=0, le=1)
    ltv_dist_high: float = Field(0.85, ge=0, le=1)
    risk_aversion_mu: float = 1.0
    risk_aversion_sigma: float = Field(0.5, ge=0)
    inst_cash_low: float = Field(5_000_000.0, ge=0)
    inst_cash_high: float = Field(20_000_000.0, ge=0)  # appropriate?
    inst_required_return: float = Field(0.0025, ge=0)
    loss_aversion: float = Field(1.30, ge=0)  # is this appropriately scaled?


class CreditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mortgage_rate: float = Field(0.004167, ge=0)
    ltv_limit: float = Field(0.85, ge=0, le=1)
    dti_limit: float = Field(0.35, ge=0, le=1)
    loan_term_months: int = Field(360, gt=0)
    btl_funding_rate: float = Field(0.005, ge=0)
    btl_ltv: float = Field(0.75, ge=0, le=1)
    inst_funding_rate_low: float = Field(0.001667, ge=0)
    inst_funding_rate_high: float = Field(0.0025, ge=0)
    inst_ltv: float = Field(0.60, ge=0, le=1)


class ValuationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    quality_sensitivity: float = Field(0.3, ge=0)  # arbitrary


class ExpectationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    delta: float = Field(0.7, ge=0, le=1)  # arbitary
    init_price_growth: float = 0.001667
    init_rent_growth: float = 0.001667
    household_signal_window: int = Field(60, ge=2)
    institutional_signal_window: int = Field(120, ge=2)
    noise_sd: float = Field(0.00144, ge=0)


class MacroConfig(BaseModel):  # Need to add credit conditions
    model_config = ConfigDict(frozen=True, extra="forbid")
    initial_state: Literal["Boom", "Neutral", "Recession"] = "Neutral"
    boom_mean: float = 0.0025
    boom_sd: float = Field(0.00577, ge=0)
    neutral_mean: float = 0.000833
    neutral_sd: float = Field(0.00289, ge=0)
    recession_mean: float = -0.001667
    recession_sd: float = Field(0.00866, ge=0)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

__all__ = [
    "Config",
    "SimConfig",
    "SpatialConfig",
    "PropertyInitConfig",
    "AgentInitConfig",
    "AgentConfig",
    "CreditConfig",
    "ValuationConfig",
    "ExpectationsConfig",
    "MarketConfig",
    "MacroConfig",
    "DebugConfig",
    "load_config",
]
