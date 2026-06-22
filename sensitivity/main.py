"""Sobol global sensitivity analysis harness.

Usage:
    uv run sensitivity/main.py
    uv run python -m sensitivity
"""

import os
import sys as _sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import get_context
from tqdm import tqdm

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")
for _p in (_PROJECT_ROOT, _CODE_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "sensitivity", "config.yaml")


def load_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_problem(sa_cfg):
    """Build SALib problem dict from sa_config.yaml parameters."""
    params = sa_cfg["parameters"]
    return {
        "num_vars": len(params),
        "names": [p["name"] for p in params],
        "bounds": [[0.0, 1.0]] * len(params),
    }


def map_params(param_values, sa_cfg):
    """Map SALib [0,1] samples to actual parameter values."""
    rows = []
    for row in param_values:
        mapped = {}
        for x, p in zip(row, sa_cfg["parameters"]):
            low, high = p["bounds"]
            dist = p.get("distribution", "uniform")
            if dist == "uniform":
                mapped[p["name"]] = low + x * (high - low)
            elif dist == "log-uniform":
                mapped[p["name"]] = np.exp(
                    np.log(low) + x * (np.log(high) - np.log(low))
                )
            elif dist == "int":
                mapped[p["name"]] = int(round(low + x * (high - low)))
            else:
                raise ValueError(f"Unknown distribution: {dist}")
        rows.append(mapped)
    return rows


def build_config(params, sa_cfg):
    """Build a Config with overridden parameter values."""
    from code.config import Config

    cfg = Config()
    section_overrides = {}
    for p in sa_cfg["parameters"]:
        section, field = p["path"].split(".")
        if section not in section_overrides:
            section_overrides[section] = {}
        section_overrides[section][field] = params[p["name"]]
    steps = sa_cfg.get("steps", cfg.sim.n_steps)
    if "sim" not in section_overrides:
        section_overrides["sim"] = {}
    section_overrides["sim"]["n_steps"] = steps
    updates = {}
    for section, fields in section_overrides.items():
        section_obj = getattr(cfg, section)
        updates[section] = section_obj.model_copy(update=fields)
    return cfg.model_copy(update=updates)


def compute_responses(df, sa_cfg):
    """Reduce full-run DataFrame to scalar response values."""
    results = {}
    for r in sa_cfg["responses"]:
        name = r["name"]
        if r.get("custom"):
            if name == "max_drawdown":
                prices = df["avg_sale_price"].dropna()
                if len(prices) > 0:
                    peak = prices.expanding().max()
                    dd = (peak - prices) / peak
                    results[name] = float(dd.max()) if float(dd.max()) > 0 else 0.0
                else:
                    results[name] = 0.0
            else:
                raise ValueError(f"Unknown custom response: {name}")
        else:
            series = df[r["metric"]]
            tail = r.get("tail")
            if tail:
                series = series.tail(tail)
            reduce = r.get("reduce", "nanmean")
            if reduce == "nanmean":
                results[name] = (
                    float(series.mean(skipna=True))
                    if not series.isna().all()
                    else np.nan
                )
            elif reduce == "nanstd":
                results[name] = (
                    float(series.std(skipna=True))
                    if not series.isna().all()
                    else np.nan
                )
            elif reduce == "last":
                results[name] = (
                    float(series.iloc[-1]) if not series.empty else np.nan
                )
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
    return results


def run_single(args):
    """Run one model instance and return scalar responses."""
    params, sa_cfg = args
    from code.model import HousingModel

    try:
        config = build_config(params, sa_cfg)
        model = HousingModel(config=config)
        for _ in range(config.sim.n_steps):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        return compute_responses(df, sa_cfg)
    except Exception as e:
        print(f"  FAILED: {params} — {e}")
        return {r["name"]: float("nan") for r in sa_cfg["responses"]}


def plot_sobol(sobol_df, sa_cfg, out_dir):
    """Grouped bar chart: 1st and total order per response."""
    responses = [r["name"] for r in sa_cfg["responses"]]
    param_names = [p["name"] for p in sa_cfg["parameters"]]
    n = len(responses)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    x = np.arange(len(param_names))
    width = 0.35

    for ax, resp in zip(axes, responses):
        sub = sobol_df[sobol_df["response"] == resp]
        s1 = sub["S1"].values
        s1_c = sub["S1_conf"].values
        st = sub["ST"].values
        st_c = sub["ST_conf"].values

        ax.bar(x - width / 2, s1, width, yerr=s1_c, label="1st order", capsize=3)
        ax.bar(
            x + width / 2, st, width, yerr=st_c, label="Total order", capsize=3
        )
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=30, ha="right")
        ax.set_title(resp)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, sa_cfg["output"]["plot"]), dpi=150)
    plt.close()


def main():
    sa_cfg = load_config()

    n_cores = sa_cfg["parallel"]["n_cores"]
    if n_cores == -1 or n_cores is None:
        n_cores = os.cpu_count()

    N = sa_cfg["sobol"]["N"]
    seed = sa_cfg["sobol"]["seed"]
    k = len(sa_cfg["parameters"])
    total_runs = N * (2 * k + 2)

    problem = build_problem(sa_cfg)

    param_values = sobol_sample.sample(problem, N, seed=seed)
    param_rows = map_params(param_values, sa_cfg)

    print(f"SA: k={k}, N={N}, total_runs={total_runs}, cores={n_cores}")

    ctx = get_context("spawn")
    with ctx.Pool(n_cores) as pool:
        all_results = []
        with tqdm(total=len(param_rows), desc="Running simulations") as pbar:
            for r in pool.imap_unordered(run_single, [(r, sa_cfg) for r in param_rows]):
                all_results.append(r)
                pbar.update()

    param_df = pd.DataFrame(param_rows)
    response_df = pd.DataFrame(all_results)
    full_df = pd.concat([param_df, response_df], axis=1)

    os.makedirs(sa_cfg["output"]["dir"], exist_ok=True)
    out_csv = os.path.join(sa_cfg["output"]["dir"], sa_cfg["output"]["data"])
    full_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    sobol_rows = []
    for r in sa_cfg["responses"]:
        name = r["name"]
        Y = full_df[name].values
        nan_count = np.isnan(Y).sum()
        if nan_count > 0:
            print(f"  {name}: {nan_count}/{len(Y)} NaN values — filling with 0")
            Y = np.nan_to_num(Y, nan=0.0)
        try:
            Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
            for i, p in enumerate(sa_cfg["parameters"]):
                sobol_rows.append(
                    {
                        "response": name,
                        "parameter": p["name"],
                        "S1": Si["S1"][i],
                        "S1_conf": Si["S1_conf"][i],
                        "ST": Si["ST"][i],
                        "ST_conf": Si["ST_conf"][i],
                    }
                )
        except Exception as e:
            print(f"  Error analyzing '{name}': {e}")

    sobol_df = pd.DataFrame(sobol_rows)
    sobol_csv = os.path.join(sa_cfg["output"]["dir"], "sobol_indices.csv")
    sobol_df.to_csv(sobol_csv, index=False)
    print(f"Saved: {sobol_csv}")

    plot_sobol(sobol_df, sa_cfg, sa_cfg["output"]["dir"])
    print(f"Plot saved: {os.path.join(sa_cfg['output']['dir'], sa_cfg['output']['plot'])}")
    print("Done.")


if __name__ == "__main__":
    main()
