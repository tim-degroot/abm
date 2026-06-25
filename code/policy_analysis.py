"""
Paired credit-shock analysis wrapper.

Runs the housing ABM under a credit-shock policy and its matched baseline across
N_RUNS stochastic replicates, then saves one response figure per policy to
results/credit_shocks/.

Run from the repo root:
    uv run code/policy_analysis.py

or if project environment already active:
    python code/policy_analysis.py

Configure the run by editing the configuration block below: choose the policies (POLICIES_TO_RUN), number of
stochastic replications (N_RUNS), shock month (SHOCK_STEP), plotting horizon, etc.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from model import HousingModel
from policies import EXPERIMENTS, NoPolicy

plt.switch_backend("Agg")  # for headless servers

# Configuration 

N_RUNS = 20
SHOCK_STEP = 240
PRE_SHOCK_MONTHS = 60
POST_SHOCK_MONTHS = 120
ROLLING_WINDOW = 12
WORKERS = 16

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "credit_shocks"

POLICIES_TO_RUN = [
    "rate-up",
    # "rate-down",
    # "ltv-tighten",
    # "ltv-loosen",
]

TOTAL_STEPS = SHOCK_STEP + POST_SHOCK_MONTHS
METRICS = [
    "sale_price",
    "rent",
    "owner_occupier",
    "private_landlord",
    "institution",
]
SHARES = {
    "owner_occupier_share": "owner_occupier",
    "landlord_share": "private_landlord",
    "institution_share": "institution",
}

# Simulation

def rolling_metrics(data):
    """Trailing 12-month price, rent, and winning-bidder shares."""

    volume = data["transaction_volume"].fillna(0.0)
    
    rolling_volume = volume.rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
    
    rolling_volume = rolling_volume.where(rolling_volume > 0)

    result = pd.DataFrame({"month": data["month"]})

    transaction_value = data["avg_sale_price"].fillna(0.0) * volume
    
    result["sale_price"] = transaction_value.rolling(
        ROLLING_WINDOW, min_periods=ROLLING_WINDOW
    ).sum() / rolling_volume

    result["rent"] = data["avg_rent"].rolling(
        ROLLING_WINDOW, min_periods=ROLLING_WINDOW
    ).mean()

    for source, target in SHARES.items():
        winner_count = data[source].fillna(0.0) * volume
        result[target] = winner_count.rolling(
            ROLLING_WINDOW, min_periods=ROLLING_WINDOW
        ).sum() / rolling_volume

    return result


def run_model(seed, policy, initial=None):
    """Run one model and retain the metrics used in the figure."""
    config = Config()

    if initial is not None:
        credit = config.credit.model_copy(update=initial)
        config = config.model_copy(update={"credit": credit})

    sim = config.sim.model_copy(update={"seed": seed, "n_steps": TOTAL_STEPS})
    config = config.model_copy(update={"sim": sim})

    model = HousingModel(config=config, policy=policy)
    for _ in range(TOTAL_STEPS):
        model.step()

    data = model.datacollector.get_model_vars_dataframe().rename_axis("month").reset_index()
    return rolling_metrics(data)


def run_seed(seed):
    """Run a matched baseline and policy path for each selected policy."""
    responses = []

    for policy_name in POLICIES_TO_RUN:
        policy = EXPERIMENTS[policy_name](step=SHOCK_STEP)
        initial = getattr(policy, "initial", None)
        baseline = run_model(seed, NoPolicy(), initial=initial)
        shocked = run_model(seed, policy, initial=initial)
        paired = baseline.merge(shocked, on="month", suffixes=("_base", "_shock"))
        response = pd.DataFrame(
            {
                "policy": policy_name,
                "seed": seed,
                "event_month": paired["month"] - SHOCK_STEP,
            }
        )

        for metric in ("sale_price", "rent"):
            response[metric] = 100 * (
                paired[f"{metric}_shock"] / paired[f"{metric}_base"] - 1
            )
        for metric in ("owner_occupier", "private_landlord", "institution"):
            response[metric] = 100 * (
                paired[f"{metric}_shock"] - paired[f"{metric}_base"]
            )

        responses.append(
            response[
                response["event_month"].between(
                    -PRE_SHOCK_MONTHS, POST_SHOCK_MONTHS
                )
            ]
        )

    return pd.concat(responses, ignore_index=True)


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = range(1, N_RUNS + 1)

    print(f"Running {N_RUNS} paired replications for: {', '.join(POLICIES_TO_RUN)}")
    with ProcessPoolExecutor(max_workers=min(WORKERS, N_RUNS)) as executor:
        responses = pd.concat(executor.map(run_seed, seeds), ignore_index=True)

    summary = summarise(responses)
    for policy_name in POLICIES_TO_RUN:
        plot_policy(policy_name, summary)
        print(f"Saved plots for {policy_name} in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
