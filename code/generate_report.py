"""Run a longer simulation and produce summary charts for calibration inspection.

Produces `report_summary.png` and `model_timeseries.csv` in the working directory.

Usage: run from the `code` directory with the project virtualenv activated.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from model import HousingModel
from config import load_config


def run_and_plot(
    steps=200, out_png="report_summary.png", out_csv="model_timeseries.csv"
):
    cfg = load_config()
    # Ensure results go into `results/` by default
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    if not os.path.isabs(out_png):
        out_png = os.path.join(results_dir, out_png)
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(results_dir, out_csv)
    # create model with default config; do not enable bid logging here
    m = HousingModel(config=cfg)
    # Collect per-step zone metrics while stepping the model
    zone_rows = []
    for i in range(steps):
        m.step()

        # compute per-zone aggregates after this step
        n_zones = cfg.spatial.n_zones
        for zone in range(n_zones):
            props = [p for p in m.properties if p.zone == zone]
            if not props:
                continue
            avg_est_val = float(sum(p.estimated_value for p in props) / len(props))
            ownership_rate = float(
                sum(1 for p in props if p.owner_id is not None) / len(props)
            )
            inst_share = (
                float(
                    sum(
                        1
                        for p in props
                        if p.owner_id is not None
                        and m._property_map[p.id].owner_id is not None
                        and getattr(
                            m._agent_map.get(m._property_map[p.id].owner_id),
                            "is_institution",
                            False,
                        )
                    )
                    / sum(1 for p in props if p.owner_id is not None)
                )
                if any(p.owner_id is not None for p in props)
                else 0.0
            )

            # Transactions in this step for this zone
            txns = [
                t
                for t in m.this_step_transactions
                if m._property_map[t.property_id].zone == zone
            ]
            txn_vol = len(txns)
            avg_txn_price = (
                float(sum(t.price for t in txns) / len(txns)) if txns else float("nan")
            )

            zone_rows.append(
                {
                    "step": i,
                    "zone": zone,
                    "avg_estimated_value": avg_est_val,
                    "ownership_rate": ownership_rate,
                    "institutional_ownership_share": inst_share,
                    "transaction_volume": txn_vol,
                    "avg_transaction_price": avg_txn_price,
                }
            )

    df = m.datacollector.get_model_vars_dataframe()
    df.index.name = "step"

    # The model produces sparse price/rent series when there are few
    # transactions in some steps. For reporting we forward-fill missing
    # averages and backfill remaining gaps with the series median so plots
    # are continuous and easier to inspect.
    df_filled = df.copy()
    for col in ("avg_sale_price", "avg_rent"):
        if col in df_filled.columns:
            med = (
                float(df_filled[col].median(skipna=True))
                if df_filled[col].notna().any()
                else 0.0
            )
            df_filled[col + "_filled"] = df_filled[col].ffill().fillna(med)

    df_filled.to_csv(out_csv)

    # Persist per-zone timeseries
    zone_df = pd.DataFrame(zone_rows)
    zone_out = os.path.join(results_dir, "zone_timeseries.csv")
    zone_df.to_csv(zone_out, index=False)
    print(f"Wrote zone-level timeseries to {zone_out}")

    # Plot key series if present
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    if "avg_sale_price_filled" in df_filled.columns:
        df_filled["avg_sale_price_filled"].plot(
            ax=ax[0], title="Average Sale Price (filled)"
        )
    elif "avg_sale_price" in df.columns:
        df["avg_sale_price"].plot(ax=ax[0], title="Average Sale Price")
    else:
        ax[0].text(0.5, 0.5, "avg_sale_price not collected", ha="center")

    if "avg_rent_filled" in df_filled.columns:
        df_filled["avg_rent_filled"].plot(ax=ax[1], title="Average Rent (filled)")
    elif "avg_rent" in df.columns:
        df["avg_rent"].plot(ax=ax[1], title="Average Rent")
    else:
        ax[1].text(0.5, 0.5, "avg_rent not collected", ha="center")

    if "ownership_rate" in df_filled.columns:
        df_filled["ownership_rate"].plot(ax=ax[2], title="Household Ownership Rate")
    else:
        ax[2].text(0.5, 0.5, "ownership_rate not collected", ha="center")

    # Stock-share breakdown. These should sum to 1.0 when all three are present.
    if {
        "household_ownership_share_of_stock",
        "institutional_ownership_share",
    }.issubset(df_filled.columns):
        df_filled["household_ownership_share_of_stock"].plot(
            ax=ax[3], title="Housing Stock Shares"
        )
        df_filled["institutional_ownership_share"].plot(ax=ax[3])
        ax[3].legend(
            [
                "household stock share",
                "institutional stock share",
            ]
        )
    elif "institutional_ownership_share" in df_filled.columns:
        df_filled["institutional_ownership_share"].plot(
            ax=ax[3], title="Institutional Ownership Share"
        )
    elif "inst_property_count" in df_filled.columns:
        df_filled["inst_property_count"].plot(
            ax=ax[3], title="Institutional Property Count"
        )
    else:
        ax[3].text(0.5, 0.5, "Institutional data not collected", ha="center")

    plt.tight_layout()
    fig.savefig(out_png)
    print(f"Wrote {out_png} and {out_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    p.add_argument("--out", type=str, default="report_summary.png", help="Output PNG")
    p.add_argument("--csv", type=str, default="model_timeseries.csv", help="Output CSV")
    args = p.parse_args()
    run_and_plot(steps=args.steps, out_png=args.out, out_csv=args.csv)
