"""
Entry point for the Housing Market ABM.

Run with:
    pip install mesa numpy
    python run.py

Executes a baseline simulation and prints per-step metrics to console.
Outputs a CSV of collected data to housing_abm_results.csv.
"""

import os
import sys
import argparse

# Ensure the housing_abm directory is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import load_config
from model import HousingModel


def run_simulation(config=None, n_steps=None, verbose=True):
    """
    Run the baseline simulation.

    config  : Config instance (defaults to the bundled config.toml)
    n_steps : number of model steps to execute (defaults to config.sim.n_steps)
    verbose : print per-step summary to console
    """
    cfg = config if config is not None else load_config()
    if n_steps is None:
        n_steps = cfg.sim.n_steps

    print("=" * 60)
    print("Housing Market ABM — Baseline Simulation")
    print("=" * 60)
    print(f"  Steps:        {n_steps}")
    print(f"  Households:   {cfg.sim.n_households}")
    print(f"  Institutions: {cfg.sim.n_institutions}")
    print(f"  Properties:   {cfg.sim.n_properties}")
    print(
        f"  Zones:        {cfg.spatial.n_zones} "
        f"({cfg.spatial.grid_rows}x{cfg.spatial.grid_cols} torus)"
    )
    print(f"  Tenure init:  {cfg.sim.ownership_mode}")
    print(f"  Seed:         {cfg.sim.seed}")
    print("=" * 60)

    model = HousingModel(config=cfg)

    for step in range(n_steps):
        model.step()

        if verbose:
            state = model.get_model_state()
            df = model.datacollector.get_model_vars_dataframe()
            latest = df.iloc[-1]

            avg_price = latest["avg_sale_price"]
            vol = latest["transaction_volume"]
            own_rate = latest["ownership_rate"]
            inst_share = latest["institutional_ownership_share"]
            mp_share = latest["household_marginal_pricer_share"]
            avg_r = latest["avg_rent"]
            macro_state = getattr(model, "current_macro_state", "Neutral")

            price_str = f"{avg_price:,.0f}" if avg_price == avg_price else "N/A"
            mp_str = f"{mp_share:.2f}" if mp_share == mp_share else "N/A"
            rent_str = f"{avg_r:,.0f}" if avg_r == avg_r else "N/A"

            print(
                f"Step {step+1:>3} | "
                f"AvgPrice: {price_str:>12} | "
                f"Vol: {vol:>5} | "
                f"OwnRate: {own_rate:>5.2f} | "
                f"InstShare: {inst_share:>5.2f} | "
                f"HH_MP: {mp_str:>6} | "
                f"AvgRent: {rent_str:>10} | "
                f"Macro: {macro_state}"
            )

    print("=" * 60)
    print("Simulation complete.")

    # Write results to CSV under `results/` so artifacts are colocated.
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "housing_abm_results.csv")
    df = model.datacollector.get_model_vars_dataframe()
    df.to_csv(output_path)
    print(f"Results written to: {output_path}")

    return model


def run_credit_shock_experiment(config=None, n_steps=480):
    """
    Demonstrates the marginal-pricer mechanism by tightening credit at step 240 (20 years).

    Household WTP falls as mortgage_rate rises.
    Institutional WTP is anchored to yields (less affected).
    Observe shift in household_marginal_pricer_share.

    The shock values below are experiment-specific overrides (intentional
    deviations from the baseline), not baseline assumptions, so they live here
    rather than in config.toml. The loan term is taken from config so only the
    shocked dimensions differ from baseline.
    """
    cfg = config if config is not None else load_config()

    print("\n" + "=" * 60)
    print("Experiment: Credit Tightening Shock (step 20)")
    print("=" * 60)

    from policies import NoPolicy
    from credit import CreditEnvironment

    class CreditShockPolicy(NoPolicy):
        """Tightens mortgage rate at step 240 (20 years into monthly run)."""

        def on_step_start(self, model):
            if model.steps == 240:
                model.credit = CreditEnvironment(
                    mortgage_rate=0.006667,  # up from baseline 0.004167 (8% p.a.)
                    ltv_limit=0.80,  # down from baseline 0.85
                    dti_limit=0.30,  # down from baseline 0.35
                    loan_term_months=cfg.credit.loan_term_months,
                )
                print(
                    "  [SHOCK] Credit tightened at step 240: rate=8% p.a. (0.006667/mo), LTV=80%, DTI=30%"
                )

    model = HousingModel(config=cfg, policy=CreditShockPolicy())

    pre_mp = []
    post_mp = []

    for step in range(n_steps):
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        mp = df.iloc[-1]["household_marginal_pricer_share"]
        if mp == mp:  # not NaN
            if step < 240:
                pre_mp.append(mp)
            else:
                post_mp.append(mp)

    pre_avg = sum(pre_mp) / len(pre_mp) if pre_mp else float("nan")
    post_avg = sum(post_mp) / len(post_mp) if post_mp else float("nan")

    print(f"\n  Pre-shock HH marginal pricer share (avg):  {pre_avg:.3f}")
    print(f"  Post-shock HH marginal pricer share (avg): {post_avg:.3f}")

    direction = (
        "↓ institutions gaining" if post_avg < pre_avg else "↑ households dominant"
    )
    print(f"  Direction: {direction}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Housing Market ABM")
    parser.add_argument("--steps", "-s", type=int, help="number of simulation steps to run")
    parser.add_argument("--seed", type=int, help="override RNG seed in config")
    args = parser.parse_args()

    cfg = load_config()
    if args.seed is not None:
        try:
            cfg.sim.seed = int(args.seed)
        except Exception:
            pass

    # Baseline run (allow --steps override)
    model = run_simulation(config=cfg, n_steps=args.steps, verbose=True)

    # Marginal-pricer experiment
    run_credit_shock_experiment(config=cfg, n_steps=40)
