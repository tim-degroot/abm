"""Mesa-style interactive visualization for the housing market ABM.

Run from the `code/` directory with:

```bash
solara run viz.py
```

The app shows a spatial agent layer plus summary charts for prices, rents,
ownership, and market share metrics.
"""

from __future__ import annotations

from dataclasses import replace

import solara
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle

from agents import HouseholdAgent, InstitutionalAgent
from config import Config
from model import HousingModel


class HousingVizModel(HousingModel):
    """Small wrapper that exposes a slider-friendly constructor for Mesa viz."""

    def __init__(
        self,
        seed: int = 42,
        n_households: int = 100,
        n_institutions: int = 5,
        n_properties: int = 120,
        target_ownership_rate: float = 0.65,
        ownership_mode: str = "emergent",
        wealth_income_mult_low: float = 0.8,
        wealth_income_mult_high: float = 3.0,
        base_price: float = 200_000.0,
        initial_rent_yield: float = 0.045,
        debug_bid_logging: bool = False,
    ):
        cfg = Config()
        cfg = replace(
            cfg,
            sim=replace(
                cfg.sim,
                seed=int(seed),
                n_households=int(n_households),
                n_institutions=int(n_institutions),
                n_properties=int(n_properties),
                target_ownership_rate=float(target_ownership_rate),
                ownership_mode=str(ownership_mode).strip().lower(),
            ),
            agent_init=replace(
                cfg.agent_init,
                wealth_income_mult_low=float(wealth_income_mult_low),
                wealth_income_mult_high=float(wealth_income_mult_high),
            ),
            property_init=replace(
                cfg.property_init,
                base_price=float(base_price),
            ),
            market=replace(
                cfg.market,
                initial_rent_yield=float(initial_rent_yield),
            ),
        )
        super().__init__(config=cfg, debug_bid_logging=debug_bid_logging)

        # The base model already exposes the real toroidal house grid; keep the
        # visualization wrapper thin and avoid any synthetic layout.


def agent_portrayal(agent):
    if isinstance(agent, HouseholdAgent):
        if agent.is_owner_occupier and agent.is_landlord:
            color = "#9467bd"
            marker = "D"
        elif agent.is_owner_occupier:
            color = "#2ca02c"
            marker = "o"
        elif agent.is_landlord:
            color = "#ff7f0e"
            marker = "s"
        else:
            color = "#1f77b4"
            marker = "o"

        size = min(95, 32 + 8 * len(agent.owned_properties))
        return AgentPortrayalStyle(
            x=agent.pos[0],
            y=agent.pos[1],
            color=color,
            marker=marker,
            size=size,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=2,
        )

    return AgentPortrayalStyle(x=agent.pos[0], y=agent.pos[1], color="#7f7f7f")


def house_status_portrayal(layer):
    if layer.name != "house_status":
        return None

    return PropertyLayerStyle(
        colormap=[
            "#d9d9d9",  # 0 unowned / vacant
            "#4daf4a",  # 1 owner-occupied, not landlord
            "#984ea3",  # 2 owner-occupied landlord
            "#ff7f00",  # 3 household-owned rental occupied by tenant
            "#377eb8",  # 4 institution-owned rental occupied by tenant
            "#fdbf6f",  # 5 vacant owned / listed for rent
            "#e41a1c",  # 6 listed for sale / distressed
        ],
        alpha=0.72,
        colorbar=False,
        vmin=0,
        vmax=6,
    )


def model_summary(model):
    df = model.datacollector.get_model_vars_dataframe()
    latest = df.iloc[-1] if not df.empty else None

    households = [a for a in model.agents if isinstance(a, HouseholdAgent)]
    institutions = [a for a in model.agents if isinstance(a, InstitutionalAgent)]
    owner_households = sum(1 for a in households if a.owned_properties)
    inst_stock = sum(len(a.portfolio) for a in institutions)

    if latest is None:
        return solara.Markdown("No data collected yet.")

    return solara.Markdown(
        "\n".join(
            [
                f"**Step:** {int(model.steps)}",
                f"**Household ownership rate:** {owner_households / max(len(households), 1):.2f}",
                f"**Institutional stock share:** {inst_stock / max(len(model.properties), 1):.2f}",
                (
                    f"**Avg sale price:** {latest['avg_sale_price']:.0f}"
                    if latest["avg_sale_price"] == latest["avg_sale_price"]
                    else "**Avg sale price:** N/A"
                ),
                (
                    f"**Avg rent:** {latest['avg_rent']:.0f}"
                    if latest["avg_rent"] == latest["avg_rent"]
                    else "**Avg rent:** N/A"
                ),
                f"**Transaction volume:** {int(latest['transaction_volume'])}",
                (
                    f"**Marginal pricer share:** {latest['household_marginal_pricer_share']:.2f}"
                    if latest["household_marginal_pricer_share"]
                    == latest["household_marginal_pricer_share"]
                    else "**Marginal pricer share:** N/A"
                ),
                "**Board colors:** gray vacant, green owner-occupied, purple owner-landlord, orange household rental, blue institutional rental, gold vacant rent, red sale listing",
            ]
        )
    )


def legend_panel(model):
    return solara.Markdown("""
### Board Legend

- <span style="color:#4daf4a">■</span> Owner-occupied, not landlord
- <span style="color:#984ea3">■</span> Owner-occupied landlord
- <span style="color:#ff7f00">■</span> Household-owned rental occupied by tenant
- <span style="color:#377eb8">■</span> Institution-owned rental occupied by tenant
- <span style="color:#fdbf6f">■</span> Vacant owned / listed for rent
- <span style="color:#e41a1c">■</span> Listed for sale or distressed
- <span style="color:#d9d9d9">■</span> Vacant / unowned
""")


model = HousingVizModel()
renderer = (
    SpaceRenderer(model, backend="matplotlib")
    .setup_propertylayer(house_status_portrayal)
    .setup_agents(agent_portrayal)
    .render()
)

PricePlot = make_plot_component(
    {"avg_sale_price": "tab:blue", "avg_rent": "tab:orange"}
)
OwnershipPlot = make_plot_component(
    {
        "ownership_rate": "tab:green",
        "institutional_ownership_share": "tab:red",
        "household_ownership_share_of_stock": "tab:purple",
    }
)
MarketPlot = make_plot_component(
    {
        "transaction_volume": "tab:blue",
        "rental_transaction_volume": "tab:orange",
        "household_marginal_pricer_share": "tab:green",
    }
)
NetWorthPlot = make_plot_component({"total_household_net_worth": "tab:cyan"})

model_params = {
    "seed": {"type": "InputText", "value": 42, "label": "Seed"},
    "n_households": Slider("Households", 100, 50, 200, 10),
    "n_institutions": Slider("Institutions", 5, 1, 10, 1),
    "n_properties": Slider("Properties", 120, 80, 200, 10),
    "target_ownership_rate": Slider("Target ownership", 0.65, 0.3, 0.9, 0.05),
    "ownership_mode": {
        "type": "InputText",
        "value": "emergent",
        "label": "Ownership mode (emergent/target)",
    },
    "wealth_income_mult_low": Slider("Wealth/income low", 0.8, 0.2, 2.0, 0.1),
    "wealth_income_mult_high": Slider("Wealth/income high", 3.0, 1.0, 6.0, 0.1),
    "base_price": Slider("Base price", 200_000.0, 120_000.0, 350_000.0, 10_000.0),
    "initial_rent_yield": Slider("Initial rent yield", 0.045, 0.02, 0.07, 0.0025),
}

page = SolaraViz(
    model,
    renderer,
    components=[
        model_summary,
        legend_panel,
        PricePlot,
        OwnershipPlot,
        MarketPlot,
        NetWorthPlot,
    ],
    model_params=model_params,
    name="Housing Market ABM",
)

page  # noqa: E305
