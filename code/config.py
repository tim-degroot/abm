"""
Central configuration loader.

Uses Pydantic for type/range validation per field and cross-field checks.
No manual type-checking or validation functions needed — Field() constraints
and @model_validator covers everything.

load_config() reads config.toml and returns a frozen Config object.
Partial override files work: any missing key falls back to the Python default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DEFAULT_TOML = Path(__file__).parent / "config.toml"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class SimConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_households: int = Field(100, gt=0)
    n_institutions: int = Field(5, gt=0)
    n_properties: int = Field(120, gt=0)
    target_ownership_rate: float = Field(0.65, ge=0, le=1)
    inst_ownership_share: float = Field(0.10, ge=0, le=1)
    seed: int = 42
    n_steps: int = Field(360, gt=0)
    ownership_mode: str = "emergent"


class SpatialConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    grid_rows: int = Field(4, ge=3)
    grid_cols: int = Field(4, ge=3)

    @property
    def n_zones(self) -> int:
        return self.grid_rows * self.grid_cols


class PropertyInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    zone_quality_sd: float = Field(0.5, ge=0)
    property_residual_sd: float = Field(0.5, ge=0)
    base_price: float = Field(200_000.0, gt=0)
    price_sensitivity: float = Field(50_000.0, ge=0)
    quality_clustering: bool = False
    clustering_strength: float = Field(0.5, ge=0, le=1)


class AgentInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    income_median: float = Field(35_000.0, gt=0)
    income_sigma: float = Field(0.5, ge=0)
    wealth_income_mult_low: float = Field(0.5, ge=0)
    wealth_income_mult_high: float = Field(2.0, ge=0)
    ltv_dist_low: float = Field(0.70, ge=0, le=1)
    ltv_dist_high: float = Field(0.85, ge=0, le=1)
    landlord_share: float = Field(0.10, ge=0, le=1)
    landlord_portfolio_geom_p: float = Field(0.6, gt=0, le=1)
    risk_aversion_mu: float = 0.0
    risk_aversion_sigma: float = Field(0.5, ge=0)
    inst_cash_low: float = Field(5_000_000.0, ge=0)
    inst_cash_high: float = Field(20_000_000.0, ge=0)
    inst_funding_rate_low: float = Field(0.001667, ge=0)
    inst_funding_rate_high: float = Field(0.0025, ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Self:
        if self.wealth_income_mult_low > self.wealth_income_mult_high:
            raise ValueError(
                "agent_init.wealth_income_mult_low must be <= _high"
            )
        if self.ltv_dist_low > self.ltv_dist_high:
            raise ValueError("agent_init.ltv_dist_low must be <= ltv_dist_high")
        return self


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    beta_action: float = Field(1.0, ge=0)
    beta_property: float = Field(0.5, ge=0)
    sell_score_offset: float = 0.001667
    inst_sell_score_offset: float = 0.000833
    inst_ltv: float = Field(0.60, ge=0, le=1)
    inst_required_return: float = Field(0.0025, ge=0)
    inst_min_yield: float = Field(0.05, ge=0, le=1)


class CreditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mortgage_rate: float = Field(0.004167, ge=0)
    ltv_limit: float = Field(0.85, ge=0, le=1)
    dti_limit: float = Field(0.35, ge=0, le=1)
    loan_term_months: int = Field(300, gt=0)
    btl_funding_rate: float = Field(0.005, ge=0)
    btl_ltv: float = Field(0.75, ge=0, le=1)


class ValuationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rent_income_fraction: float = Field(0.35, ge=0, le=1)
    quality_sensitivity: float = Field(0.3, ge=0)
    operating_cost_fraction: float = Field(0.15, ge=0, le=1)
    quality_value_scale: float = Field(1.0, ge=0)
    capital_gain_mode: Literal["fixed_level", "bounded_growth"] = "fixed_level"
    expected_capital_gain_level: float = 166.67
    capital_gain_growth_min: float = -0.001667
    capital_gain_growth_max: float = 0.001667
    max_price_to_income: float = Field(54.0, gt=0)
    max_price_to_rent: float = Field(300.0, gt=0)

    @model_validator(mode="after")
    def _check_growth_bounds(self) -> Self:
        if self.capital_gain_growth_min > self.capital_gain_growth_max:
            raise ValueError(
                "valuation.capital_gain_growth_min must be <= "
                "capital_gain_growth_max"
            )
        return self


class ExpectationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float = Field(0.7, ge=0, le=1)
    init_price_growth: float = 0.001667
    init_rent_growth: float = 0.001667
    signal_window: int = Field(60, ge=2)
    noise_sd: float = Field(0.00144, ge=0)


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    household_sell_reservation_discount: float = Field(0.95, ge=0, le=1)
    inst_sell_reservation_discount: float = Field(0.97, ge=0, le=1)
    landlord_reservation_yield: float = Field(0.04, ge=0)
    min_reservation_rent: float = Field(200.0, ge=0)
    initial_rent_yield: float = Field(0.045, ge=0)
    fallback_price: float = Field(200_000.0, gt=0)
    lease_expiry_prob: float = Field(0.0278, ge=0, le=1)
    min_lease_months: int = Field(12, ge=0)
    lease_early_exit_prob: float = Field(0.003, ge=0, le=1)
    renter_research_prob: float = Field(0.006, ge=0, le=1)
    loss_aversion_owner: float = Field(1.30, ge=0)
    loss_aversion_landlord: float = Field(1.15, ge=0)
    estimated_value_smooth_alpha: float = Field(1.0, ge=0, le=1)


class DebugConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_bid_logging: bool = False


class MacroConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_state: Literal["Boom", "Neutral", "Recession"] = "Neutral"
    boom_mean: float = 0.0025
    boom_sd: float = Field(0.00577, ge=0)
    neutral_mean: float = 0.000833
    neutral_sd: float = Field(0.00289, ge=0)
    recession_mean: float = -0.001667
    recession_sd: float = Field(0.00866, ge=0)
    boom_to_boom: float = Field(0.80, ge=0, le=1)
    boom_to_neutral: float = Field(0.15, ge=0, le=1)
    boom_to_recession: float = Field(0.05, ge=0, le=1)
    neutral_to_boom: float = Field(0.05, ge=0, le=1)
    neutral_to_neutral: float = Field(0.90, ge=0, le=1)
    neutral_to_recession: float = Field(0.05, ge=0, le=1)
    recession_to_boom: float = Field(0.05, ge=0, le=1)
    recession_to_neutral: float = Field(0.15, ge=0, le=1)
    recession_to_recession: float = Field(0.80, ge=0, le=1)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sim: SimConfig = SimConfig()
    spatial: SpatialConfig = SpatialConfig()
    property_init: PropertyInitConfig = PropertyInitConfig()
    agent_init: AgentInitConfig = AgentInitConfig()
    agent: AgentConfig = AgentConfig()
    credit: CreditConfig = CreditConfig()
    valuation: ValuationConfig = ValuationConfig()
    expectations: ExpectationsConfig = ExpectationsConfig()
    market: MarketConfig = MarketConfig()
    macro: MacroConfig = MacroConfig()
    debug: DebugConfig = DebugConfig()

    @model_validator(mode="after")
    def _validate(self) -> Self:
        s, sp, ai, c = self.sim, self.spatial, self.agent_init, self.credit

        if s.n_properties <= s.n_households:
            raise ValueError(
                f"n_properties ({s.n_properties}) must exceed n_households "
                f"({s.n_households}) so renters can find rentals."
            )
        if s.n_properties < sp.n_zones:
            raise ValueError(
                f"n_properties ({s.n_properties}) must be >= n_zones "
                f"({sp.n_zones}) so every zone can hold stock."
            )
        if ai.ltv_dist_high > c.ltv_limit:
            raise ValueError(
                f"agent_init.ltv_dist_high ({ai.ltv_dist_high}) must be <= "
                f"credit.ltv_limit ({c.ltv_limit}); origination LTVs would be "
                "silently clamped, distorting the distribution."
            )
        if c.btl_funding_rate < ai.inst_funding_rate_high:
            raise ValueError(
                f"credit.btl_funding_rate ({c.btl_funding_rate}) must be >= "
                f"agent_init.inst_funding_rate_high ({ai.inst_funding_rate_high}); "
                "plan §6 requires r_f^BTL > r_f."
            )
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> Config:
    """Read config.toml (or an override file) and return a validated Config."""
    toml_path = Path(path) if path is not None else _DEFAULT_TOML
    if not toml_path.exists():
        raise FileNotFoundError(f"Config file not found: {toml_path}")

    with open(toml_path, "rb") as fh:
        raw = tomllib.load(fh)

    return Config(**raw)


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
