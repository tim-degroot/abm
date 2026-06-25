"""Configuration schema for the housing-market ABM.

All parameters are plausible, stylised values (the model is not calibrated to
micro-data). Every per-period rate, flow and duration is expressed in *monthly*
units, since one model step is one calendar month.

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
    n_households: int = Field(250, gt=0)
    n_institutions: int = Field(5, gt=0)
    n_properties: int = Field(300, gt=0)
    n_steps: int = Field(720, gt=0)
    seed: int = 42


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
    init_price_quality_sensitivity: float = Field(50_000.0, ge=0)
    # Fraction of households initially allocated an owned home. Lowered from the
    # legacy 0.96 (which, combined with zone-matching failures, left the market
    # ~50% vacant) to a value consistent with the report's ~65% ownership target.
    init_ownership_prob: float = Field(0.90, ge=0, le=1)


class AgentInitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    income_mean: float = Field(30_000.0, gt=0)
    income_sigma: float = Field(0.5, ge=0)
    wealth_income_mult_low: float = Field(0.5, ge=0)
    wealth_income_mult_high: float = Field(25.0, ge=0)
    # Spread of *legacy* origination LTVs for the starting mortgage book only.
    # New mortgages during the run use the policy/credit LTV (see credit.py), not
    # a random draw (fixes the "random origination LTV" bug).
    ltv_dist_low: float = Field(0.70, ge=0, le=1)
    ltv_dist_high: float = Field(0.85, ge=0, le=1)
    # Household risk-aversion coefficient gamma ~ LogNormal(mu, sigma). Enters
    # behaviour as a risk *loading* on expected growth: g -> g - gamma * sigma_g.
    risk_aversion_mu: float = Field(1.0) # -0.3
    risk_aversion_sigma: float = Field(0.5, ge=0)
    inst_cash_low: float = Field(1_500_000.0, ge=0)
    inst_cash_high: float = Field(10_000_000.0, ge=0)
    inst_required_return: float = Field(0.0015, ge=0)  # monthly, 1.8% APR
    inst_min_yield: float = Field(0.04, ge=0)
    # Loss-aversion coefficient lambda > 1 used in the seller's reservation price.
    loss_aversion: float = Field(1.30, ge=0)


class CreditConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mortgage_rate: float = Field(0.00308, ge=0)  # monthly, ~3.7% APR
    ltv_limit: float = Field(0.9, ge=0, le=1)
    dti_limit: float = Field(0.33, ge=0, le=1)
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
    # Baseline monthly housing-consumption value of a median (q = 0) home (~rent
    # for a typical home). Calibrated so that, at a representative risk-adjusted
    # discount-minus-growth denominator (~0.004/mo), the capitalised owner-occupier
    # value sits near the ~200k price scale; credit constraints then bind for many
    # buyers. Also keeps the flow positive across the (mean-zero) quality range,
    # fixing the "negative WTP for half the stock" bug.
    base_housing_value: float = Field(700.0, ge=0)


class ExpectationsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # Single EWMA smoothing weight shared by households and institutions and by
    # the growth and volatility updates:
    #   E_t = smoothing * E_{t-1} + (1 - smoothing) * signal.
    # (There is no separate institutional delta or forecast window: institutions
    # differ only in seeing the global signal and a rolling OLS forecast, both of
    # which reuse `signal_window`.)
    smoothing: float = Field(0.9, ge=0, le=1)
    # Lookback window (months) for the growth/volatility signals and the OLS.
    signal_window: int = Field(12, gt=0)
    # Initial growth expectations (monthly).
    init_price_growth: float = 0.001667
    init_rent_growth: float = 0.001667
    # Initial expectations of the *volatility of the growth rate* (monthly std),
    # updated by the same EWMA as the growth expectations.
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
    # The macro state is fixed within a run (no stochastic transition matrix).
    # Income grows by a normal draw with the regime's mean/sd each month. Credit
    # shocks are applied separately and deterministically through the policy layer.
    initial_state: Literal["Boom", "Neutral", "Recession"] = "Neutral"
    boom_mean: float = 0.0025
    boom_sd: float = Field(0.00577, ge=0)
    neutral_mean: float = 0.000833
    neutral_sd: float = Field(0.00289, ge=0)
    recession_mean: float = -0.001667
    recession_sd: float = Field(0.00866, ge=0)
    # Risk-free monthly rate, used as the institutional outside option.
    risk_free_rate: float = Field(0.00308, ge=0) # mortgage rate, ~3.7% APR


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
