"""
Central configuration for the Housing Market ABM.

Workflow
--------
1. **Edit** `config.toml` — the single file editable by a human that contains every
   model parameter.  This is where you change values between experiments.

2. **Load** — call `load_config()` at the start of a run.  It reads the TOML,
   validates all values and returns a fully populated `Config` object.

3. **Lock** — `Config` and all its sub-configs are *frozen* dataclasses.  Once
   a run begins, no code can accidentally overwrite a parameter during the simulation;
   any attempted write raises an error immediately.

Parameter access
----------------
- **The Model:** Stores the main configuration as `model.config`. This is loaded only once when the model starts, using `load_config()`.
- **The Agents:** Can read the configuration at any time by calling `self.model.config`.
- **Helper Functions:** Do not receive the entire configuration object. Instead, you should pass them only the exact variables they need to do their math.

Testing
-------
For quick testing, you can just call `Config()` in your code. It uses built-in default values that perfectly match `config.toml`, so it works even if the file isn't on your disk.
However, for real simulations, you must always use `load_config()`. This ensures that the parameters are actually read from the TOML file and validated before the run starts.

"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

_DEFAULT_TOML = Path(__file__).parent / "config.toml"


# ---------------------------------------------------------------------------
# Sub-configs (frozen). Defaults mirror config.toml.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    n_households: int = 100
    n_institutions: int = 5
    n_properties: int = 120
    n_zones: int = 10
    target_ownership_rate: float = 0.65
    inst_ownership_share: float = 0.10
    seed: int = 42
    n_steps: int = 30


@dataclass(frozen=True)
class PropertyInitConfig:
    zone_quality_sd: float = 0.5
    property_residual_sd: float = 0.5
    base_price: float = 200_000.0
    price_sensitivity: float = 50_000.0


@dataclass(frozen=True)
class AgentInitConfig:
    income_median: float = 35_000.0
    income_sigma: float = 0.5
    cash_mult_low: float = 0.5
    cash_mult_high: float = 2.0
    risk_aversion_mu: float = 0.0
    risk_aversion_sigma: float = 0.5
    inst_cash_low: float = 5_000_000.0
    inst_cash_high: float = 20_000_000.0
    inst_funding_rate_low: float = 0.02
    inst_funding_rate_high: float = 0.03


@dataclass(frozen=True)
class AgentConfig:
    beta_action: float = 1.0
    beta_property: float = 0.5
    income_reversion: float = 0.05
    income_shock_sd: float = 0.05
    sell_score_offset: float = 0.02
    inst_sell_score_offset: float = 0.01
    inst_operating_cost_fraction: float = 0.15
    inst_ltv: float = 0.60


@dataclass(frozen=True)
class CreditConfig:
    mortgage_rate: float = 0.05
    ltv_limit: float = 0.85
    dti_limit: float = 0.35
    loan_term_years: int = 25
    rent_affordability_fraction: float = 0.35


@dataclass(frozen=True)
class ValuationConfig:
    rent_income_fraction: float = 0.35
    quality_sensitivity: float = 0.3


@dataclass(frozen=True)
class ExpectationsConfig:
    delta: float = 0.7
    init_price_growth: float = 0.02
    init_rent_growth: float = 0.02
    signal_window: int = 5


@dataclass(frozen=True)
class MarketConfig:
    household_sell_reservation_discount: float = 0.95
    inst_sell_reservation_discount: float = 0.97
    landlord_reservation_yield: float = 0.04
    min_reservation_rent: float = 200.0
    initial_rent_yield: float = 0.045
    fallback_price: float = 200_000.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

# Maps the TOML section name -> (Config attribute name, dataclass type).
_SECTIONS = {
    "sim": ("sim", SimConfig),
    "property_init": ("property_init", PropertyInitConfig),
    "agent_init": ("agent_init", AgentInitConfig),
    "agent": ("agent", AgentConfig),
    "credit": ("credit", CreditConfig),
    "valuation": ("valuation", ValuationConfig),
    "expectations": ("expectations", ExpectationsConfig),
    "market": ("market", MarketConfig),
}


@dataclass(frozen=True)
class Config:
    """Immutable container of all model parameters."""

    sim: SimConfig = SimConfig()
    property_init: PropertyInitConfig = PropertyInitConfig()
    agent_init: AgentInitConfig = AgentInitConfig()
    agent: AgentConfig = AgentConfig()
    credit: CreditConfig = CreditConfig()
    valuation: ValuationConfig = ValuationConfig()
    expectations: ExpectationsConfig = ExpectationsConfig()
    market: MarketConfig = MarketConfig()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def _build_section(name: str, cls: type, raw: dict) -> object:
    """
    Instantiate a sub-config dataclass from a TOML section dict.

    Unknown keys raise (typo protection). Missing keys fall back to the
    dataclass default. Numeric ints declared as floats in TOML (e.g. 25.0)
    are coerced to the field's declared type where that type is int.
    """
    section = raw.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(
            f"[{name}] section in config must be a table, got {type(section)}"
        )

    valid = {f.name: f for f in fields(cls)}
    unknown = set(section) - set(valid)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in [{name}] section of config."
        )

    coerced = {}
    for key, value in section.items():
        declared = valid[key].type
        if declared in ("int", int) and isinstance(value, float) and value.is_integer():
            value = int(value)
        coerced[key] = value
    return cls(**coerced)


def _validate(cfg: Config) -> None:
    """Sanity-check parameter ranges; raise ValueError on violation."""
    s, c, m = cfg.sim, cfg.credit, cfg.market

    def _pos(label, v):
        if v <= 0:
            raise ValueError(f"{label} must be > 0, got {v}")

    def _frac(label, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{label} must be in [0, 1], got {v}")

    _pos("sim.n_households", s.n_households)
    _pos("sim.n_institutions", s.n_institutions)
    _pos("sim.n_properties", s.n_properties)
    _pos("sim.n_zones", s.n_zones)
    _pos("sim.n_steps", s.n_steps)
    _frac("sim.target_ownership_rate", s.target_ownership_rate)
    _frac("sim.inst_ownership_share", s.inst_ownership_share)

    if s.n_properties <= s.n_households:
        raise ValueError(
            f"sim.n_properties ({s.n_properties}) must exceed sim.n_households "
            f"({s.n_households}) so renters can find rentals."
        )

    _frac("credit.ltv_limit", c.ltv_limit)
    _frac("credit.dti_limit", c.dti_limit)
    _frac("credit.rent_affordability_fraction", c.rent_affordability_fraction)
    if c.mortgage_rate < 0:
        raise ValueError(f"credit.mortgage_rate must be >= 0, got {c.mortgage_rate}")
    _pos("credit.loan_term_years", c.loan_term_years)

    _frac("agent.inst_operating_cost_fraction", cfg.agent.inst_operating_cost_fraction)
    _frac("agent.inst_ltv", cfg.agent.inst_ltv)
    _frac("valuation.rent_income_fraction", cfg.valuation.rent_income_fraction)
    _frac("expectations.delta", cfg.expectations.delta)
    if cfg.expectations.signal_window < 2:
        raise ValueError(
            f"expectations.signal_window must be >= 2, got {cfg.expectations.signal_window}"
        )

    _frac(
        "market.household_sell_reservation_discount",
        m.household_sell_reservation_discount,
    )
    _frac("market.inst_sell_reservation_discount", m.inst_sell_reservation_discount)


def load_config(path: str | Path | None = None) -> Config:
    """
    Load and validate a Config from a TOML file.

    path : path to a TOML file, or None to use the bundled config.toml.
    """
    toml_path = Path(path) if path is not None else _DEFAULT_TOML
    if not toml_path.exists():
        raise FileNotFoundError(f"Config file not found: {toml_path}")

    with open(toml_path, "rb") as fh:
        raw = tomllib.load(fh)

    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"Unknown section(s) {sorted(unknown_sections)} in {toml_path}."
        )

    kwargs = {
        attr: _build_section(section_name, cls, raw)
        for section_name, (attr, cls) in _SECTIONS.items()
    }
    cfg = Config(**kwargs)
    _validate(cfg)
    return cfg


__all__ = [
    "Config",
    "SimConfig",
    "PropertyInitConfig",
    "AgentInitConfig",
    "AgentConfig",
    "CreditConfig",
    "ValuationConfig",
    "ExpectationsConfig",
    "MarketConfig",
    "load_config",
]
