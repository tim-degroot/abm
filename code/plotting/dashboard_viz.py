"""
Interactive Solara dashboard for the housing-market ABM.

Run from the repository root with:

    uv run solara run visualisation.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig

# --- 1. Make sure the repo root is importable, then load the LOCAL `code` pkg.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_local_code_package():
    """Import this project's ``code`` package explicitly, regardless of whether a
    stdlib ``code`` module has already been imported by the runtime."""
    pkg_dir = os.path.join(_PROJECT_ROOT, "code")
    init_file = os.path.join(pkg_dir, "__init__.py")
    sys.modules.pop("code", None)  # drop any stdlib `code` that got imported first
    spec = importlib.util.spec_from_file_location(
        "code", init_file, submodule_search_locations=[pkg_dir]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["code"] = module
    spec.loader.exec_module(module)


_load_local_code_package()

# Pull everything we need out of the local package while `code` points to it.
from code.core.model import HousingModel
from code.settings.config import Config
from code.settings.policies import EXPERIMENTS, NoPolicy

# --- 2. Restore the genuine stdlib `code` module so Mesa imports cleanly. -----
_std_code_path = os.path.join(sysconfig.get_paths()["stdlib"], "code.py")
_std_spec = importlib.util.spec_from_file_location("code", _std_code_path)
_std_code = importlib.util.module_from_spec(_std_spec)
_std_spec.loader.exec_module(_std_code)
sys.modules["code"] = _std_code

# --- 3. Now Mesa's Solara visualisation imports safely. -----------------------
import solara  # noqa: E402
from mesa.visualization import SolaraViz, Slider, make_plot_component  # noqa: E402
from mesa.visualization.utils import update_counter  # noqa: E402


class VizHousingModel(HousingModel):
    def __init__(
        self,
        seed: int = 42,
        n_households: int = 500,
        n_institutions: int = 5,
        n_properties: int = 625,
        search_radius: int = 1,
        mortgage_rate: float = 0.00308,
        ltv_limit: float = 0.90,
        dti_limit: float = 0.40,
        risk_aversion_mu: float = 1.0,
        experiment: str = "none",
        shock_step: int = 120,
    ):
        base = Config()
        cfg = base.model_copy(
            update={
                "sim": base.sim.model_copy(
                    update={
                        "n_households": int(n_households),
                        "n_institutions": int(n_institutions),
                        "n_properties": int(n_properties),
                        "seed": int(seed),
                    }
                ),
                "spatial": base.spatial.model_copy(update={"search_radius": int(search_radius)}),
                "credit": base.credit.model_copy(
                    update={
                        "mortgage_rate": float(mortgage_rate),
                        "ltv_limit": float(ltv_limit),
                        "dti_limit": float(dti_limit),
                    }
                ),
                "agent_init": base.agent_init.model_copy(
                    update={"risk_aversion_mu": float(risk_aversion_mu)}
                ),
            }
        )

        if experiment and experiment != "none":
            policy = EXPERIMENTS[experiment](step=int(shock_step))
        else:
            policy = NoPolicy()

        super().__init__(config=cfg, policy=policy)


# =============================================================================
# Controls
# =============================================================================
model_params = {
    "seed": Slider("Random seed", value=42, min=0, max=9999, step=1, dtype=int),
    "experiment": {
        "type": "Select",
        "value": "none",
        "values": ["none"] + list(EXPERIMENTS.keys()),
        "label": "Scheduled experiment (shock)",
    },
    "shock_step": Slider("Shock step (month)", value=120, min=12, max=600, step=12, dtype=int),
    "n_households": Slider("Households", value=500, min=100, max=800, step=50, dtype=int),
    "n_institutions": Slider("Institutions", value=5, min=1, max=20, step=1, dtype=int),
    "n_properties": Slider("Dwellings", value=625, min=200, max=900, step=25, dtype=int),
    "search_radius": Slider("Search radius (zones)", value=1, min=0, max=3, step=1, dtype=int),
    "mortgage_rate": Slider(
        "Mortgage rate (monthly)", value=0.00308, min=0.001, max=0.010, step=0.0005, dtype=float
    ),
    "ltv_limit": Slider("LTV limit", value=0.90, min=0.50, max=1.00, step=0.05, dtype=float),
    "dti_limit": Slider("DTI limit", value=0.40, min=0.20, max=0.60, step=0.05, dtype=float),
    "risk_aversion_mu": Slider(
        "Risk-aversion \u03bc (log-mean)", value=1.0, min=-2.0, max=2.0, step=0.25, dtype=float
    ),
}


# =============================================================================
# Plot components (each reads named columns from model.datacollector)
# =============================================================================
plot_sale_price = make_plot_component("avg_sale_price")
plot_rent = make_plot_component("avg_rent")
plot_volume = make_plot_component("transaction_volume")
plot_gini = make_plot_component("household_net_worth_gini")

# Marginal pricer: share of transactions won by each group.
plot_marginal_pricer = make_plot_component(
    {
        "owner_occupier_share": "tab:blue",
        "landlord_share": "tab:orange",
        "institution_share": "tab:green",
    }
)

# Ownership distribution: share of the housing stock held by each group.
plot_ownership = make_plot_component(
    {
        "owner_occupier_ownership_share": "tab:blue",
        "landlord_ownership_share": "tab:orange",
        "institutional_ownership_share": "tab:green",
    }
)


@solara.component
def Headline(model):
    """Compact live readout of the current month's headline figures."""
    update_counter.get()  # subscribe to the step counter so this re-renders each step
    df = model.datacollector.get_model_vars_dataframe()

    def fmt(col, kind="num"):
        if df.empty or col not in df.columns:
            return "\u2013"
        val = df[col].iloc[-1]
        if val != val:  # NaN
            return "\u2013"
        if kind == "money":
            return f"\u00a3{val:,.0f}"
        if kind == "pct":
            return f"{100 * val:.1f}%"
        return f"{val:,.2f}"

    solara.Markdown(f"""
        **Month {model.steps}** &nbsp;|&nbsp;
        Avg sale price: **{fmt('avg_sale_price', 'money')}** &nbsp;|&nbsp;
        Avg rent: **{fmt('avg_rent', 'money')}** &nbsp;|&nbsp;
        Owner-occupier stock: **{fmt('owner_occupier_ownership_share', 'pct')}** &nbsp;|&nbsp;
        Institutional stock: **{fmt('institutional_ownership_share', 'pct')}** &nbsp;|&nbsp;
        Gini: **{fmt('household_net_worth_gini')}**
        """)


# =============================================================================
# Page (Solara looks for a module-level `page`)
# =============================================================================
_model = VizHousingModel()

page = SolaraViz(
    _model,
    components=[
        Headline,
        plot_sale_price,
        plot_rent,
        plot_marginal_pricer,
        plot_ownership,
        plot_volume,
        plot_gini,
    ],
    model_params=model_params,
    name="Who Sets the Price? \u2014 Housing-Market ABM",
    play_interval=200,
)

# Allow `python visualisation.py` to fail gracefully with a hint.
if __name__ == "__main__":
    print("Launch the dashboard with:  solara run visualisation.py")
