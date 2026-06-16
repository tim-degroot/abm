import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_summary(df: pd.DataFrame, out_png: str):
    matplotlib.use("Agg")

    df_filled = df.copy()
    for col in ("avg_sale_price", "avg_rent"):
        if col in df_filled.columns:
            med = (
                float(df_filled[col].median(skipna=True))
                if df_filled[col].notna().any()
                else 0.0
            )
            df_filled[col + "_filled"] = df_filled[col].ffill().fillna(med)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    if "avg_sale_price_filled" in df_filled.columns:
        df_filled["avg_sale_price_filled"].plot(ax=ax[0], title="Average Sale Price (filled)")
    elif "avg_sale_price" in df.columns:
        df["avg_sale_price"].ffill().plot(ax=ax[0], title="Average Sale Price")
    else:
        ax[0].text(0.5, 0.5, "avg_sale_price not collected", ha="center")

    if "avg_rent_filled" in df_filled.columns:
        df_filled["avg_rent_filled"].plot(ax=ax[1], title="Average Rent (filled)")
    elif "avg_rent" in df.columns:
        df["avg_rent"].ffill().plot(ax=ax[1], title="Average Rent")
    else:
        ax[1].text(0.5, 0.5, "avg_rent not collected", ha="center")

    if "ownership_rate" in df_filled.columns:
        df_filled["ownership_rate"].plot(ax=ax[2], title="Household Ownership Rate")
    else:
        ax[2].text(0.5, 0.5, "ownership_rate not collected", ha="center")

    if {"household_ownership_share_of_stock", "institutional_ownership_share"}.issubset(
        df_filled.columns
    ):
        df_filled["household_ownership_share_of_stock"].plot(
            ax=ax[3], title="Housing Stock Shares"
        )
        df_filled["institutional_ownership_share"].plot(ax=ax[3])
        ax[3].legend(["household stock share", "institutional stock share"])
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
    plt.close(fig)
