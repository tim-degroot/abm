"""
Paired credit-shock analysis wrapper.

Runs the housing ABM under a credit-shock policy and its matched baseline across
N_RUNS replicates, then saves one response figure per policy to results/credit_shocks/.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from code.settings.config import Config
from code.core.model import HousingModel
from code.settings.policies import EXPERIMENTS, NoPolicy

plt.switch_backend("Agg")  # for headless servers

# Configuration

N_RUNS = 20
SHOCK_STEP = 240
PRE_SHOCK_MONTHS = 60
POST_SHOCK_MONTHS = 120
ROLLING_WINDOW = 12
WORKERS = 16

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "policy_analysis"

POLICIES_TO_RUN = [
    "rate-up",
    "rate-down",
    "ltv-tighten",
    "ltv-loosen",
    "recession-easing-crunch",
    "boom-credit-expansion",
    "recession-credit-crunch",
]

TOTAL_STEPS = SHOCK_STEP + POST_SHOCK_MONTHS
# Extended responses mirror the scenario questions: prices, volatility, composition, and wealth.
PERCENT_METRICS = [
    "sale_price",
    "rent",
    "avg_household_net_worth",
]
POINT_METRICS = [
    "price_volatility",
    "owner_occupier",
    "private_landlord",
    "institution",
    "owner_occupier_value",
    "private_landlord_value",
    "institution_value",
    "owner_occupier_stock",
    "private_landlord_stock",
    "institution_stock",
    "household_net_worth_gini",
]
METRICS = PERCENT_METRICS + POINT_METRICS
SHARES = {
    "owner_occupier_share": "owner_occupier",
    "landlord_share": "private_landlord",
    "institution_share": "institution",
}
VALUE_SHARES = {
    "owner_occupier_value_share": "owner_occupier_value",
    "landlord_value_share": "private_landlord_value",
    "institution_value_share": "institution_value",
}
STOCK_SHARES = {
    "owner_occupier_ownership_share": "owner_occupier_stock",
    "landlord_ownership_share": "private_landlord_stock",
    "institutional_ownership_share": "institution_stock",
}
LEVELS = {
    "total_household_net_worth": "total_household_net_worth",
    "household_net_worth_gini": "household_net_worth_gini",
}

# Simulation


def rolling_metrics(data):
    """Trailing 12-month price, rent, volatility, wealth, and ownership/winner shares."""

    volume = data["transaction_volume"].fillna(0.0)

    rolling_volume = volume.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()

    rolling_volume = rolling_volume.where(rolling_volume > 0)

    result = pd.DataFrame({"month": data["month"]})

    transaction_value = data["avg_sale_price"].fillna(0.0) * volume
    rolling_value = transaction_value.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
    rolling_value = rolling_value.where(rolling_value > 0)

    result["sale_price"] = (
        transaction_value.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum() / rolling_volume
    )

    log_returns = np.log(result["sale_price"] / result["sale_price"].shift(1))
    result["price_volatility"] = log_returns.rolling(
        ROLLING_WINDOW, min_periods=ROLLING_WINDOW
    ).std()

    result["rent"] = data["avg_rent"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    for source, target in SHARES.items():
        winner_count = data[source].fillna(0.0) * volume
        result[target] = (
            winner_count.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum() / rolling_volume
        )

    for source, target in VALUE_SHARES.items():
        winner_value = data[source].fillna(0.0) * transaction_value
        result[target] = (
            winner_value.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum() / rolling_value
        )

    for source, target in STOCK_SHARES.items():
        result[target] = data[source].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    for source, target in LEVELS.items():
        result[target] = data[source].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()

    return result


def run_model(seed, policy, initial=None, initial_macro=None):
    """Run one model with matched pre-shock credit and macro state."""
    config = Config()

    if initial is not None:
        credit = config.credit.model_copy(update=initial)
        config = config.model_copy(update={"credit": credit})
    if initial_macro is not None:
        macro = config.macro.model_copy(update={"initial_state": initial_macro})
        config = config.model_copy(update={"macro": macro})

    sim = config.sim.model_copy(update={"seed": seed, "n_steps": TOTAL_STEPS})
    config = config.model_copy(update={"sim": sim})

    model = HousingModel(config=config, policy=policy)
    for _ in range(TOTAL_STEPS):
        model.step()

    data = model.datacollector.get_model_vars_dataframe().rename_axis("month").reset_index()
    return rolling_metrics(data)


def run_policy_seed(args):
    """Run a matched baseline and shock path for one (seed, policy_name)."""
    seed, policy_name = args
    policy = EXPERIMENTS[policy_name](step=SHOCK_STEP)
    initial = getattr(policy, "initial", None)
    initial_macro = getattr(policy, "initial_macro", None)
    baseline = run_model(seed, NoPolicy(), initial=initial, initial_macro=initial_macro)
    shocked = run_model(seed, policy, initial=initial, initial_macro=initial_macro)
    paired = baseline.merge(shocked, on="month", suffixes=("_base", "_shock"))
    response = pd.DataFrame(
        {
            "policy": policy_name,
            "seed": seed,
            "event_month": paired["month"] - SHOCK_STEP,
        }
    )

    for metric in PERCENT_METRICS:
        response[metric] = 100 * (paired[f"{metric}_shock"] / paired[f"{metric}_base"] - 1)
    for metric in POINT_METRICS:
        response[metric] = 100 * (paired[f"{metric}_shock"] - paired[f"{metric}_base"])

    return response[response["event_month"].between(-PRE_SHOCK_MONTHS, POST_SHOCK_MONTHS)]


# Summary and plotting


def summarise(responses):
    """Pointwise means and approximate 95% confidence intervals."""
    long_data = responses.melt(
        id_vars=["policy", "seed", "event_month"],
        value_vars=METRICS,
        var_name="metric",
        value_name="response",
    )
    summary = (
        long_data.groupby(["policy", "event_month", "metric"])["response"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    half_width = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - half_width
    summary["upper"] = summary["mean"] + half_width
    return summary


def plot_policy(policy_name, summary):
    """Save one response figure for one policy."""
    policy_data = summary[summary["policy"] == policy_name]
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    def plot_band(axis, metric, label=None, alpha=0.2):
        values = policy_data[policy_data["metric"] == metric].sort_values("event_month")
        line = axis.plot(values["event_month"], values["mean"], label=label)[0]
        axis.fill_between(
            values["event_month"],
            values["lower"],
            values["upper"],
            color=line.get_color(),
            alpha=alpha,
        )

    plot_band(axes[0], "sale_price")
    axes[0].set_ylabel("Sale price response (%)")

    plot_band(axes[1], "rent")
    axes[1].set_ylabel("Rent response (%)")

    plot_band(axes[2], "owner_occupier", "Owner-occupier", 0.12)
    plot_band(axes[2], "private_landlord", "Private landlord", 0.12)
    plot_band(axes[2], "institution", "Institution", 0.12)
    axes[2].set_ylabel("Winner-share response (pp)")
    axes[2].set_xlabel("Months relative to permanent policy shift")
    axes[2].legend()

    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlim(-PRE_SHOCK_MONTHS, POST_SHOCK_MONTHS)
        axis.grid(alpha=0.25)

    figure.suptitle(policy_name.replace("-", " ").title())
    figure.tight_layout()

    name = policy_name.replace("-", "_") + "_response"
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    plt.close(figure)


def plot_comparison(policy_names, summary):
    """N×1 grid comparing winner-share responses across policies."""
    n_policies = len(policy_names)
    fig, axes = plt.subplots(n_policies, 1, figsize=(9, 3 * n_policies + 1), sharex=True)
    if n_policies == 1:
        axes = [axes]

    metrics = {
        "owner_occupier": "Owner-occupier",
        "private_landlord": "Private landlord",
        "institution": "Institution",
    }

    for ax, policy in zip(axes, policy_names):
        policy_data = summary[summary["policy"] == policy]

        for metric, label in metrics.items():
            values = policy_data[policy_data["metric"] == metric].sort_values("event_month")
            line = ax.plot(values["event_month"], values["mean"], label=label)[0]
            ax.fill_between(
                values["event_month"].to_numpy(),
                values["lower"].to_numpy(),
                values["upper"].to_numpy(),
                color=line.get_color(),
                alpha=0.12,
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlim(-PRE_SHOCK_MONTHS, POST_SHOCK_MONTHS)
        ax.set_ylabel("Winner-share response (pp)")
        ax.set_title(policy.replace("-", " ").title())
        ax.grid(alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("Months relative to permanent policy shift")
    fig.tight_layout()
    path = OUTPUT_DIR / "winner_share_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Paired credit-shock analysis.")
    parser.add_argument(
        "--replot", action="store_true", help="Skip model runs; re-plot from existing responses.csv"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.replot:
        csv_path = OUTPUT_DIR / "responses.csv"
        if not csv_path.exists():
            raise SystemExit(f"{csv_path} not found. Run without --replot first.")
        responses = pd.read_csv(csv_path)
        print(f"Loaded {len(responses)} rows from {csv_path}")
    else:
        items = [(s, p) for s in range(1, N_RUNS + 1) for p in POLICIES_TO_RUN]
        print(
            f"Running {len(items)} items ({N_RUNS} seeds × {len(POLICIES_TO_RUN)} policies) on {min(WORKERS, len(items))} workers"
        )
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(items))) as executor:
            futures = {executor.submit(run_policy_seed, item): item for item in items}
            results = []
            for future in tqdm(
                as_completed(futures),
                total=len(items),
                desc="Items",
                unit="item",
            ):
                results.append(future.result())
        responses = pd.concat(results, ignore_index=True)
        responses.to_csv(OUTPUT_DIR / "responses.csv", index=False)
        print(f"Saved results in {OUTPUT_DIR}/responses.csv")

    summary = summarise(responses)
    for policy_name in POLICIES_TO_RUN:
        plot_policy(policy_name, summary)
        print(f"Saved plots for {policy_name} in {OUTPUT_DIR}")
    plot_comparison(POLICIES_TO_RUN, summary)


if __name__ == "__main__":
    main()
