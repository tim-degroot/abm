"""
Global sensitivity analysis figure: 2x2 grid of Sobol indices.

Usage
-----
    uv run python sensitivity/analysis.py

Reads results/sensitivity/sobol_indices.csv (produced by
``--aggregate``) and config.yaml; saves a figure to
results/sensitivity/global_sa.png.
"""

import os
import sys as _sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")
for _p in (_PROJECT_ROOT, _CODE_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "sensitivity", "config.yaml")
_INDICES_PATH = os.path.join(_PROJECT_ROOT, "results", "sensitivity", "sobol_indices.csv")
_OUT_PATH = os.path.join(_PROJECT_ROOT, "results", "sensitivity", "global_sa.png")

# Responses to show in the 2x2 grid (top-left → bottom-right)
RESPONSES = [
    "oo_share",
    "inst_share",
    "landlord_share",
    "price_volatility",
]
RESPONSE_LABELS = {
    "oo_share": "Owner-Occupier Share",
    "inst_share": "Institutional Share",
    "landlord_share": "Landlord Share",
    "price_volatility": "Price Volatility",
}
PARAM_LABELS = {
    "n_households": "Households",
    "search_radius": "Search Radius",
    "mortgage_rate": "Mortgage Rate",
    "ltv_limit": "LTV Limit",
    "dti_limit": "DTI Limit",
}


def main():
    if not os.path.exists(_INDICES_PATH):
        raise SystemExit(
            f"{_INDICES_PATH} not found. Run ``uv run python sensitivity/main.py --aggregate`` first."
        )

    with open(_CONFIG_PATH) as f:
        sa_cfg = yaml.safe_load(f)
    param_names = [p["name"] for p in sa_cfg["parameters"]]

    sobol_df = pd.read_csv(_INDICES_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Global Sensitivity Analysis", fontsize=16, fontweight="bold")

    x = np.arange(len(param_names))
    width = 0.35

    for ax, resp in zip(axes.flat, RESPONSES):
        sub = sobol_df[sobol_df["response"] == resp]
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(RESPONSE_LABELS.get(resp, resp))
            continue

        s1 = sub["S1"].values
        s1_c = sub["S1_conf"].values
        st = sub["ST"].values
        st_c = sub["ST_conf"].values

        ax.bar(x - width / 2, s1, width, yerr=s1_c, label="1st order", capsize=3)
        ax.bar(x + width / 2, st, width, yerr=st_c, label="Total order", capsize=3)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_LABELS.get(n, n) for n in param_names], rotation=30, ha="right")
        ax.set_title(RESPONSE_LABELS.get(resp, resp))
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    fig.savefig(_OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved: {_OUT_PATH}")


if __name__ == "__main__":
    main()
