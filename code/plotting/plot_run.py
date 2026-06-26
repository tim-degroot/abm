"""
Plotting utility for a single simulation run.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def _stacked_share_plot(ax, df, columns, title, colors, window=12):
    smoothed = df[list(columns)].rolling(window, min_periods=1, center=True).mean()
    smoothed.plot.area(ax=ax, stacked=True, color=colors, alpha=0.8)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.legend(loc="upper left")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left")


def plot_summary(df: pd.DataFrame, out_png: str):
    matplotlib.use("Agg")

    df_filled = df.copy()
    for col in ("avg_sale_price", "avg_rent"):
        if col in df_filled.columns:
            med = float(df_filled[col].median(skipna=True)) if df_filled[col].notna().any() else 0.0
            df_filled[col + "_filled"] = df_filled[col].ffill().fillna(med)

    cols = 2
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
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

    stock_cols = []
    for c in (
        "owner_occupier_ownership_share",
        "landlord_ownership_share",
        "institutional_ownership_share",
    ):
        if c in df_filled.columns:
            stock_cols.append(c)
    if len(stock_cols) == 3:
        _stacked_share_plot(
            ax[3],
            df_filled,
            stock_cols,
            "Housing Stock Ownership by Kind",
            colors=["#4c72b0", "#dd8452", "#55a868"],
        )
    elif "institutional_ownership_share" in df_filled.columns:
        df_filled["institutional_ownership_share"].plot(
            ax=ax[3], title="Institutional Ownership Share"
        )
    else:
        ax[3].text(0.5, 0.5, "Stock share data not collected", ha="center")

    mp_count_cols = ["owner_occupier_share", "landlord_share", "institution_share"]
    if all(c in df_filled.columns for c in mp_count_cols):
        _stacked_share_plot(
            ax[4],
            df_filled,
            mp_count_cols,
            "Marginal Pricer Share (count)",
            colors=["#4c72b0", "#dd8452", "#55a868"],
        )
    elif "institutional_ownership_share" in df_filled.columns:
        df_filled["institutional_ownership_share"].plot(
            ax=ax[4], title="Institutional Ownership Share"
        )
    else:
        ax[4].text(0.5, 0.5, "Marginal pricer data not collected", ha="center")

    mp_value_cols = [
        "owner_occupier_value_share",
        "landlord_value_share",
        "institution_value_share",
    ]
    if all(c in df_filled.columns for c in mp_value_cols):
        _stacked_share_plot(
            ax[5],
            df_filled,
            mp_value_cols,
            "Marginal Pricer Share (value)",
            colors=["#4c72b0", "#dd8452", "#55a868"],
        )
    else:
        ax[5].text(0.5, 0.5, "Marginal pricer value data not collected", ha="center")

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
