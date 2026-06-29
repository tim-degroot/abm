"""
Sobol global sensitivity analysis with stochastic replicates.
"""

import glob
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
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "settings", "sensitivity_config.yaml")


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
                mapped[p["name"]] = np.exp(np.log(low) + x * (np.log(high) - np.log(low)))
            elif dist == "int":
                mapped[p["name"]] = int(round(low + x * (high - low)))
            else:
                raise ValueError(f"Unknown distribution: {dist}")
        rows.append(mapped)
    return rows


def build_config(params, sa_cfg, seed=None):
    """Build a Config with overridden parameter values and optional seed."""
    from code.settings.config import Config

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
    if seed is not None:
        section_overrides["sim"]["seed"] = seed
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
        custom = r.get("custom")
        if custom:
            if custom == "returns_std":
                series = df[r["metric"]].dropna()
                if len(series) < 2:
                    results[name] = np.nan
                else:
                    log_returns = np.log(series.iloc[1:].values / series.iloc[:-1].values)
                    log_returns = log_returns[np.isfinite(log_returns)]
                    results[name] = (
                        float(np.std(log_returns, ddof=1)) if len(log_returns) > 0 else np.nan
                    )
            else:
                raise ValueError(f"Unknown custom response: {custom}")
        else:
            series = df[r["metric"]]
            tail = r.get("tail")
            if tail:
                series = series.tail(tail)
            reduce = r.get("reduce", "nanmean")
            if reduce == "nanmean":
                results[name] = (
                    float(series.mean(skipna=True)) if not series.isna().all() else np.nan
                )
            elif reduce == "nanstd":
                results[name] = (
                    float(series.std(skipna=True)) if not series.isna().all() else np.nan
                )
            elif reduce == "last":
                results[name] = float(series.iloc[-1]) if not series.empty else np.nan
            else:
                raise ValueError(f"Unknown reduce: {reduce}")
    return results


def run_single(args):
    """Run one model instance and return sample_id + scalar responses."""
    params, model_seed, sa_cfg = args
    from code.core.model import HousingModel

    sample_id = params.get("sample_id", -1)
    try:
        config = build_config(params, sa_cfg, seed=model_seed)
        model = HousingModel(config=config)
        for _ in range(config.sim.n_steps):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        result = compute_responses(df, sa_cfg)
        result["sample_id"] = sample_id
        return result
    except Exception as e:
        print(f"  FAILED: {params} — {e}")
        result = {r["name"]: float("nan") for r in sa_cfg["responses"]}
        result["sample_id"] = sample_id
        return result


# =========================================================================
# Command: generate
# =========================================================================


def cmd_generate(args):
    """Generate Saltelli samples and save to CSV."""
    sa_cfg = load_config()
    N = args.N if args.N is not None else sa_cfg["sobol"]["N"]
    seed = args.sobol_seed if args.sobol_seed is not None else sa_cfg["sobol"]["seed"]
    k = len(sa_cfg["parameters"])
    total_runs = N * (2 * k + 2)

    problem = build_problem(sa_cfg)
    param_values = sobol_sample.sample(problem, N, seed=seed)
    param_rows = map_params(param_values, sa_cfg)

    df = pd.DataFrame(param_rows)
    df.insert(0, "sample_id", range(len(df)))

    out_dir = sa_cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sobol_samples.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} samples (N={N}, k={k}, total={total_runs})")
    print(f"Saved: {out_path}")


# =========================================================================
# Command: evaluate
# =========================================================================


def cmd_evaluate(args):
    """Evaluate all parameter sets with one model seed.

    Reads the samples file generated by ``generate``, runs the model for
    every parameter set with ``args.model_seed``, and writes responses to
    ``{output.dir}/seed_{model_seed}.csv``.
    """
    sa_cfg = load_config()
    model_seed = args.model_seed
    n_cores = args.cores if args.cores is not None else sa_cfg["parallel"]["n_cores"]
    if n_cores == -1 or n_cores is None:
        n_cores = os.cpu_count()

    samples_path = os.path.join(sa_cfg["output"]["dir"], "sobol_samples.csv")
    if not os.path.exists(samples_path):
        raise SystemExit(
            f"Samples file not found at {samples_path}. "
            f"Run ``python -m sensitivity generate --N <N>`` first."
        )

    samples_df = pd.read_csv(samples_path)
    sample_rows = samples_df.to_dict("records")
    k = len(sa_cfg["parameters"])

    print(
        f"Evaluating {len(sample_rows)} samples with model_seed={model_seed} " f"on {n_cores} cores"
    )

    ctx = get_context("fork")
    with ctx.Pool(n_cores) as pool:
        all_results = []
        with tqdm(total=len(sample_rows), desc=f"seed={model_seed}", file=_sys.stderr) as pbar:
            for r in pool.imap_unordered(
                run_single, [(r, model_seed, sa_cfg) for r in sample_rows]
            ):
                all_results.append(r)
                pbar.update()

    response_df = pd.DataFrame(all_results)
    full_df = samples_df.merge(response_df, on="sample_id", how="left")

    out_dir = sa_cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"seed_{model_seed}.csv")
    full_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# =========================================================================
# Command: aggregate
# =========================================================================


def _read_seed_files(out_dir):
    """Read all ``seed_*.csv`` files and return a concatenated DataFrame."""
    pattern = os.path.join(out_dir, "seed_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            f"No seed files found at {out_dir}/seed_*.csv. "
            f"Run ``python -m sensitivity evaluate --model-seed <S>`` first."
        )
    print(f"Aggregating {len(files)} seed files: {[os.path.basename(f) for f in files]}")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _compute_sobol_indices(problem, response_name, Y, sa_cfg):
    """Run SALib Sobol analysis on one response vector."""
    Y = np.asarray(Y, dtype=float)
    nan_count = int(np.isnan(Y).sum())
    if nan_count > 0:
        nan_frac = nan_count / len(Y)
        if nan_frac > 0.5:
            print(f"  {response_name}: {nan_count}/{len(Y)} NaN ({nan_frac:.0%}) — skipping")
            return []
        print(f"  {response_name}: {nan_count}/{len(Y)} NaN — filling with 0")
        Y = np.nan_to_num(Y, nan=0.0)

    if np.nanvar(Y) < 1e-12 or np.allclose(Y, Y[0]):
        print(f"  {response_name}: near-constant output — skipping analysis")
        return []

    try:
        Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
        rows = []
        for i, p in enumerate(sa_cfg["parameters"]):
            rows.append(
                {
                    "response": response_name,
                    "parameter": p["name"],
                    "S1": Si["S1"][i],
                    "S1_conf": Si["S1_conf"][i],
                    "ST": Si["ST"][i],
                    "ST_conf": Si["ST_conf"][i],
                }
            )
        return rows
    except Exception as e:
        print(f"  Error analyzing '{response_name}': {e}")
        return []


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
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(resp)
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
        ax.set_title(resp)
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, sa_cfg["output"]["plot"]), dpi=150)
    plt.close()


def cmd_aggregate(args):
    """For each stochastic replicate, compute Sobol indices;
    then aggregate mean and confidence intervals across replicates."""
    sa_cfg = load_config()
    out_dir = sa_cfg["output"]["dir"]

    combined = _read_seed_files(out_dir)

    param_names = [p["name"] for p in sa_cfg["parameters"]]
    response_names = [r["name"] for r in sa_cfg["responses"]]

    # Save averaged responses for reference
    avg_responses = combined.groupby("sample_id")[response_names].mean(skipna=True)
    param_cols = combined.groupby("sample_id")[param_names].first()
    avg_df = pd.concat([param_cols, avg_responses], axis=1)
    avg_path = os.path.join(out_dir, "responses_avg.csv")
    avg_df.to_csv(avg_path)
    print(f"Saved: {avg_path}")

    # Compute Sobol indices per seed, then aggregate across seeds
    problem = build_problem(sa_cfg)

    pattern = os.path.join(out_dir, "seed_*.csv")
    seed_files = sorted(glob.glob(pattern))
    print(f"Computing Sobol indices per seed ({len(seed_files)} seeds)...")

    from scipy.stats import t as t_dist

    all_seed_rows = []
    for f in seed_files:
        seed_df = pd.read_csv(f).sort_values("sample_id")
        seed_rows = []
        for r in sa_cfg["responses"]:
            name = r["name"]
            Y = seed_df[name].values
            seed_rows.extend(_compute_sobol_indices(problem, name, Y, sa_cfg))
        seed_label = os.path.basename(f).replace(".csv", "")
        for row in seed_rows:
            row["seed"] = seed_label
        all_seed_rows.extend(seed_rows)

    if not all_seed_rows:
        print("No Sobol indices computed for any seed — nothing to aggregate.")
        return

    seed_sobol_df = pd.DataFrame(all_seed_rows)

    # Aggregate across seeds: mean ± t_{0.975, n-1} × SEM
    sobol_rows = []
    for (resp, param), group in seed_sobol_df.groupby(["response", "parameter"]):
        s1_vals = group["S1"].values
        st_vals = group["ST"].values
        n = len(s1_vals)
        if n < 2:
            print(f"  {resp}/{param}: only {n} seed(s) — setting CI to 0")
            s1_conf, st_conf = 0.0, 0.0
        else:
            t_val = t_dist.ppf(0.975, n - 1)
            s1_conf = float(t_val * np.std(s1_vals, ddof=1) / np.sqrt(n))
            st_conf = float(t_val * np.std(st_vals, ddof=1) / np.sqrt(n))
        sobol_rows.append({
            "response": resp,
            "parameter": param,
            "S1": float(np.mean(s1_vals)),
            "S1_conf": s1_conf,
            "ST": float(np.mean(st_vals)),
            "ST_conf": st_conf,
        })

    sobol_df = pd.DataFrame(sobol_rows)
    sobol_path = os.path.join(out_dir, "sobol_indices.csv")
    sobol_df.to_csv(sobol_path, index=False)
    print(f"Saved: {sobol_path}")

    plot_sobol(sobol_df, sa_cfg, out_dir)
    print(f"Plot saved: {os.path.join(out_dir, sa_cfg['output']['plot'])}")
    print("Done.")


# =========================================================================
# CLI entry point
# =========================================================================


def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Sobol sensitivity analysis with stochastic replicates."
    )
    p.add_argument("--generate", action="store_true", help="Generate Saltelli samples")
    p.add_argument("--N", type=int, default=None, help="Base sample size (overrides config)")
    p.add_argument(
        "--sobol-seed", type=int, default=None, help="Saltelli RNG seed (overrides config)"
    )

    p.add_argument("--evaluate", action="store_true", help="Run model evaluations for one seed")
    p.add_argument("--model-seed", type=int, default=None, help="Model RNG seed for this replicate")
    p.add_argument("--cores", type=int, default=None, help="Worker count (overrides config)")

    p.add_argument(
        "--aggregate", action="store_true", help="Aggregate seeds and compute Sobol indices"
    )
    return p.parse_args(argv)


def main():
    args = _parse_args()
    if args.generate:
        cmd_generate(args)
    elif args.evaluate:
        if args.model_seed is None:
            raise SystemExit("--evaluate requires --model-seed <int>")
        cmd_evaluate(args)
    elif args.aggregate:
        cmd_aggregate(args)
    else:
        print("Nothing to do. Pass --generate, --evaluate, or --aggregate.")


if __name__ == "__main__":
    main()
