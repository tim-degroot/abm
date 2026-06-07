"""Run a larger model configuration to smooth markets and produce a long-run CSV.

Placed in `tools/` to keep top-level clean; top-level `run_large.py` remains a lightweight wrapper.
"""

from config import Config, SimConfig, load_config
from model import HousingModel
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os


def run_large(
    steps=500, out_csv="model_timeseries_large.csv", out_png="report_summary_large.png"
):
    # Start from the bundled config but override sim settings for a larger run
    base = load_config()
    sim_cfg = SimConfig(
        n_households=1000,
        n_institutions=50,
        n_properties=1200,
        target_ownership_rate=base.sim.target_ownership_rate,
        inst_ownership_share=base.sim.inst_ownership_share,
        seed=base.sim.seed,
        n_steps=steps,
        ownership_mode=base.sim.ownership_mode,
    )
    cfg = Config(
        sim=sim_cfg,
        spatial=base.spatial,
        property_init=base.property_init,
        agent_init=base.agent_init,
        agent=base.agent,
        credit=base.credit,
        valuation=base.valuation,
        expectations=base.expectations,
        market=base.market,
        debug=base.debug,
    )

    # For stability, use a mild smoothing on estimated values so single outliers
    # don't immediately explode mark-to-market. Set alpha=0.25.
    # (This is a structural improvement, not a clamp.)
    cfg = Config(
        sim=cfg.sim,
        spatial=cfg.spatial,
        property_init=cfg.property_init,
        agent_init=cfg.agent_init,
        agent=cfg.agent,
        credit=cfg.credit,
        valuation=cfg.valuation,
        expectations=cfg.expectations,
        market=type(cfg.market)(
            **{**cfg.market.__dict__, "estimated_value_smooth_alpha": 0.25}
        ),
        debug=cfg.debug,
    )

    m = HousingModel(config=cfg)
    for _ in range(steps):
        m.step()

    df = m.datacollector.get_model_vars_dataframe()
    df.index.name = "step"
    # Ensure results are written into `results/` by default
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(results_dir, out_csv)
    if not os.path.isabs(out_png):
        out_png = os.path.join(results_dir, out_png)

    df.to_csv(out_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()
    if "avg_sale_price" in df.columns:
        df["avg_sale_price"].ffill().plot(ax=ax[0], title="Average Sale Price")
    if "avg_rent" in df.columns:
        df["avg_rent"].ffill().plot(ax=ax[1], title="Average Rent")
    if "ownership_rate" in df.columns:
        df["ownership_rate"].plot(ax=ax[2], title="Household Ownership Rate")
    if "institutional_ownership_share" in df.columns:
        df["institutional_ownership_share"].plot(
            ax=ax[3], title="Institutional Ownership Share"
        )
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"Wrote {out_csv} and {out_png}")


if __name__ == "__main__":
    run_large()
