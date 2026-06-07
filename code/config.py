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
from dataclasses import dataclass, fields
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
    target_ownership_rate: float = 0.65
    inst_ownership_share: float = 0.10
    seed: int = 42
    n_steps: int = 30
    ownership_mode: str = "emergent"  # "emergent" (default) | "target" (diagnostic only)


@dataclass(frozen=True)
class SpatialConfig:
    """2D toroidal grid. n_zones = grid_rows * grid_cols. Both dims must be >= 3
    (a dimension of size 2 collapses the two opposite neighbours onto one zone)."""

    grid_rows: int = 4
    grid_cols: int = 4

    @property
    def n_zones(self) -> int:
        return self.grid_rows * self.grid_cols


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
    wealth_income_mult_low: float = 0.5
    wealth_income_mult_high: float = 2.0
    ltv_dist_low: float = 0.70
    ltv_dist_high: float = 0.85
    landlord_share: float = 0.10
    landlord_portfolio_geom_p: float = 0.6
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
    inst_ltv: float = 0.60


@dataclass(frozen=True)
class CreditConfig:
    mortgage_rate: float = 0.05
    ltv_limit: float = 0.85
    dti_limit: float = 0.35
    loan_term_years: int = 25
    rent_affordability_fraction: float = 0.35
    btl_funding_rate: float = 0.06
    btl_ltv: float = 0.75


@dataclass(frozen=True)
class ValuationConfig:
    rent_income_fraction: float = 0.35
    quality_sensitivity: float = 0.3
    operating_cost_fraction: float = 0.15
    quality_value_scale: float = 1.0


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
    "spatial": ("spatial", SpatialConfig),
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
    spatial: SpatialConfig = SpatialConfig()
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


def _type_name(declared) -> str:
    """Normalise a dataclass field type to its name, robust to whether
    `from __future__ import annotations` makes it a string or a real type."""
    if isinstance(declared, str):
        return declared
    return getattr(declared, "__name__", str(declared))


def _build_section(name: str, cls: type, raw: dict, strict: bool = False) -> object:
    """
    Instantiate a sub-config dataclass from a TOML section dict.

    Validation performed here:
      - Unknown keys raise (typo protection).
      - Values are TYPE-CHECKED against the field's declared type: a numeric
        field rejects non-numbers (and booleans, since TOML true/false is not a
        number); an int field rejects non-integer floats; a str field rejects
        non-strings. This makes the loader fail fast on a wrong TOML type rather
        than blowing up later in arithmetic.
      - Integer-valued floats for int fields (e.g. 25.0) are coerced to int.

    Lenient-by-default behaviour (IMPORTANT):
      - A MISSING key silently falls back to the dataclass default, which mirrors
        config.toml. This supports partial override files. Pass strict=True to
        require every key to be present (errors on any omission).
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

    if strict:
        missing = set(valid) - set(section)
        if missing:
            raise ValueError(
                f"strict mode: [{name}] section is missing key(s) "
                f"{sorted(missing)}."
            )

    coerced = {}
    for key, value in section.items():
        type_name = _type_name(valid[key].type)

        if type_name in ("int", "float"):
            # bool is a subclass of int in Python; TOML true/false is not numeric.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Key '{key}' in [{name}] must be a number, got "
                    f"{type(value).__name__} ({value!r})."
                )
            if type_name == "int" and isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(
                        f"Key '{key}' in [{name}] must be an integer, got {value!r}."
                    )
                value = int(value)
        elif type_name == "str":
            if not isinstance(value, str):
                raise ValueError(
                    f"Key '{key}' in [{name}] must be a string, got "
                    f"{type(value).__name__} ({value!r})."
                )

        coerced[key] = value
    return cls(**coerced)


def _validate(cfg: Config) -> None:
    """Sanity-check parameter ranges; raise ValueError on violation."""
    s, c, m, sp, ai = cfg.sim, cfg.credit, cfg.market, cfg.spatial, cfg.agent_init

    def _pos(label, v):
        if v <= 0:
            raise ValueError(f"{label} must be > 0, got {v}")

    def _frac(label, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{label} must be in [0, 1], got {v}")

    _pos("sim.n_households", s.n_households)
    _pos("sim.n_institutions", s.n_institutions)
    _pos("sim.n_properties", s.n_properties)
    _pos("sim.n_steps", s.n_steps)
    _frac("sim.target_ownership_rate", s.target_ownership_rate)
    _frac("sim.inst_ownership_share", s.inst_ownership_share)
    if s.ownership_mode not in ("emergent", "target"):
        raise ValueError(
            f"sim.ownership_mode must be 'emergent' or 'target', got {s.ownership_mode!r}"
        )

    if s.n_properties <= s.n_households:
        raise ValueError(
            f"sim.n_properties ({s.n_properties}) must exceed sim.n_households "
            f"({s.n_households}) so renters can find rentals."
        )

    # Spatial grid: both dims >= 3 to avoid degenerate (collapsing) neighbours.
    if sp.grid_rows < 3 or sp.grid_cols < 3:
        raise ValueError(
            f"spatial.grid_rows/grid_cols must each be >= 3 (got "
            f"{sp.grid_rows}x{sp.grid_cols}); a dimension of size 2 makes opposite "
            "neighbours collapse onto the same zone."
        )
    if s.n_properties < sp.n_zones:
        raise ValueError(
            f"sim.n_properties ({s.n_properties}) must be >= n_zones "
            f"({sp.n_zones}) so every zone can hold stock."
        )

    # Agent-init distributions.
    if ai.wealth_income_mult_low > ai.wealth_income_mult_high:
        raise ValueError("agent_init.wealth_income_mult_low must be <= _high")
    _pos("agent_init.wealth_income_mult_low", ai.wealth_income_mult_low)
    _frac("agent_init.ltv_dist_low", ai.ltv_dist_low)
    _frac("agent_init.ltv_dist_high", ai.ltv_dist_high)
    if ai.ltv_dist_low > ai.ltv_dist_high:
        raise ValueError("agent_init.ltv_dist_low must be <= ltv_dist_high")
    if ai.ltv_dist_high > c.ltv_limit:
        raise ValueError(
            f"agent_init.ltv_dist_high ({ai.ltv_dist_high}) must be <= "
            f"credit.ltv_limit ({c.ltv_limit}); otherwise origination LTVs are "
            "silently clamped at the cap, distorting the configured distribution."
        )
    _frac("agent_init.landlord_share", ai.landlord_share)
    if not (0.0 < ai.landlord_portfolio_geom_p <= 1.0):
        raise ValueError(
            f"agent_init.landlord_portfolio_geom_p must be in (0, 1], got "
            f"{ai.landlord_portfolio_geom_p}"
        )

    _frac("credit.ltv_limit", c.ltv_limit)
    _frac("credit.dti_limit", c.dti_limit)
    _frac("credit.rent_affordability_fraction", c.rent_affordability_fraction)
    if c.mortgage_rate < 0:
        raise ValueError(f"credit.mortgage_rate must be >= 0, got {c.mortgage_rate}")
    _pos("credit.loan_term_years", c.loan_term_years)
    _frac("credit.btl_ltv", c.btl_ltv)
    if c.btl_funding_rate < 0:
        raise ValueError(f"credit.btl_funding_rate must be >= 0, got {c.btl_funding_rate}")
    # plan §6: landlord buy-to-let funding must cost more than institutional funding.
    if c.btl_funding_rate < ai.inst_funding_rate_high:
        raise ValueError(
            f"credit.btl_funding_rate ({c.btl_funding_rate}) must be >= institutional "
            f"funding (agent_init.inst_funding_rate_high {ai.inst_funding_rate_high}); "
            "plan §6 requires r_f^BTL > r_f."
        )

    _frac("agent.inst_ltv", cfg.agent.inst_ltv)
    _frac("valuation.rent_income_fraction", cfg.valuation.rent_income_fraction)
    _frac("valuation.operating_cost_fraction", cfg.valuation.operating_cost_fraction)
    if cfg.valuation.quality_value_scale < 0:
        raise ValueError(
            f"valuation.quality_value_scale must be >= 0, got "
            f"{cfg.valuation.quality_value_scale}"
        )
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


def load_config(path: str | Path | None = None, *, strict: bool = False) -> Config:
    """
    Load and validate a Config from a TOML file.

    path   : path to a TOML file, or None to use the bundled config.toml.
    strict : if False (default), the file may be PARTIAL — any section or key
             that is absent silently falls back to the dataclass default (which
             mirrors config.toml). This supports small override files for
             experiments. If True, the file must be COMPLETE: every section and
             every key must be present, otherwise a ValueError is raised. Use
             strict=True when the TOML is meant to be the full canonical source
             and a silent omission would be a bug.

    Note on what is ALWAYS caught (both modes): unknown section headers, unknown
    keys, and wrong value types all raise regardless of `strict`. Only *omission*
    is governed by `strict`.
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

    if strict:
        missing_sections = set(_SECTIONS) - set(raw)
        if missing_sections:
            raise ValueError(
                f"strict mode: {toml_path} is missing section(s) "
                f"{sorted(missing_sections)}."
            )

    kwargs = {
        attr: _build_section(section_name, cls, raw, strict=strict)
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
