from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path("results/credit_shocks/responses.csv")
OUTPUT_PNG = Path("results/credit_shocks/rate_winner_share_comparison.png")

POLICIES = ("rate-up", "rate-down")
METRICS = {
    "owner_occupier": "Owner-occupier",
    "private_landlord": "Private landlord",
    "institution": "Institution",
}

PRE_SHOCK_MONTHS = 60
POST_SHOCK_MONTHS = 120


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["policy"].isin(POLICIES)]

    summary = (
        df.melt(
            id_vars=["policy", "seed", "event_month"],
            value_vars=METRICS.keys(),
            var_name="metric",
            value_name="response",
        )
        .groupby(["policy", "event_month", "metric"])["response"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    ci = 1.96 * summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - ci
    summary["upper"] = summary["mean"] + ci

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for ax, policy in zip(axes, POLICIES):
        policy_data = summary[summary["policy"] == policy]

        for metric, label in METRICS.items():
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
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
