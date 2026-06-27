"""
Global sensitivity analysis figure: dynamic grid of Sobol indices.

Usage
-----
    uv run -m code.sensitivity.analysis
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "settings", "sensitivity_config.yaml")


def main():
    with open(_CONFIG_PATH) as f:
        sa_cfg = yaml.safe_load(f)

    out_dir = os.path.join(_PROJECT_ROOT, sa_cfg["output"]["dir"])
    indices_path = os.path.join(out_dir, "sobol_indices.csv")
    out_path = os.path.join(out_dir, sa_cfg["output"].get("plot", "sobol_indices.png"))

    if not os.path.exists(indices_path):
        raise SystemExit(
            f"{indices_path} not found. Run the SA pipeline first."
        )

    param_names = [p["name"] for p in sa_cfg["parameters"]]
    responses = [(r["name"], r.get("description", r["name"])) for r in sa_cfg["responses"]]

    sobol_df = pd.read_csv(indices_path)

    n_resp = len(responses)
    n_cols = 2
    n_rows = math.ceil(n_resp / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
    fig.suptitle("Global Sensitivity Analysis", fontsize=16, fontweight="bold")

    x = np.arange(len(param_names))
    width = 0.35

    for ax, (resp_name, resp_label) in zip(axes.flat, responses):
        sub = sobol_df[sobol_df["response"] == resp_name]
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(resp_label)
            continue

        s1 = sub["S1"].values
        s1_c = sub["S1_conf"].values
        st = sub["ST"].values
        st_c = sub["ST_conf"].values

        ax.bar(x - width / 2, s1, width, yerr=s1_c, label="1st order", capsize=3)
        ax.bar(x + width / 2, st, width, yerr=st_c, label="Total order", capsize=3)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=30, ha="right")
        ax.set_title(resp_label)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.flat[n_resp:]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
