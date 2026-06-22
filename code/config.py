from __future__ import annotations

from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field

class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    n_households: int = Field(250, gt=0)
    n_institutions: int = Field(5, gt=0)
    n_properties: int = Field(300, gt=0)
    n_steps: int = Field(720, gt=0)
    seed: int = 42

class SpatialConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    grid_rows: int = Field(5, ge=3)
    grid_cols: int = Field(5, ge=3)

    @property
    def n_zones(self) -> int:
        return self.grid_rows * self.grid_cols

class PropertyInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    zone_quality_sd: float = Field(0.8, ge=0)
    property_residual_sd: float = Field(0.3, ge=0)
    init_base_price: float = Field(200_000.0, gt=0)
    init_price_quality_sensitivity: float = Field(50_000.0, ge=0)
    init_ownership_prob: float = Field(0.96, ge=0, le=1)

class AgentInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    income_mean: float = Field(30_000.0, gt=0)
    income_sigma: float = Field(0.5, ge=0)
    wealth_income_mult_low: float = Field(0, ge=0)
    wealth_income_mult_high: float = Field(25.0, ge=0)
    ltv_dist_low: float = Field(0.70, ge=0, le=1)
    ltv_dist_high: float = Field(0.85, ge=0, le=1)
    risk_aversion_mu: float = 1.0
    risk_aversion_sigma: float = Field(0.5, ge=0)
    inst_cash_low: float = Field(1_500_000.0, ge=0)
    inst_cash_high: float = Field(3_500_000.0, ge=0)
    inst_required_return: float = Field(0.0003, ge=0) # 3.6% APR
    inst_min_yield: float = Field(0.04, ge=0)
    loss_aversion: float = Field(1.30, ge=0)

class CreditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mortgage_rate: float = Field(0.00308, ge=0) # 3.7% APR
    ltv_limit: float = Field(0.9, ge=0, le=1)
    dti_limit: float = Field(0.33, ge=0, le=1)
    loan_term_months: int = Field(360, gt=0)
    btl_funding_rate: float = Field(0.005, ge=0)
    btl_ltv: float = Field(0.75, ge=0, le=1)
    inst_funding_rate: float = Field(0.0045, ge=0)
    inst_ltv: float = Field(0.60, ge=0, le=1)

class ValuationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    quality_sensitivity: float = Field(0.3, ge=0)
    quality_value_scale: float = Field(2000.0, ge=0)
    max_rent_income_ratio: float = Field(0.3, ge=0, le=1)

class ExpectationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    init_price_growth: float = 0.001667
    init_rent_growth: float = 0.001667
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
