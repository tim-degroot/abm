"""
Entry point for the Housing Market ABM.

Run with:
    pip install mesa numpy
    python run.py

Executes a baseline simulation and prints per-step metrics to console.
Outputs a CSV of collected data to housing_abm_results.csv.
"""

import csv
import os
import sys

# Ensure the housing_abm directory is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from model import HousingModel


def run_simulation(n_steps=30, seed=42, verbose=True):
    """
    Run the baseline simulation.

    n_steps : number of model steps to execute
    seed    : random seed
    verbose : print per-step summary to console
    """
    print("=" * 60)
    print("Housing Market ABM — Baseline Simulation")
    print("=" * 60)
    print(f"  Steps:        {n_steps}")
    print(f"  Households:   100")
    print(f"  Institutions: 5")
    print(f"  Properties:   120")
    print(f"  Zones:        10")
    print(f"  Seed:         {seed}")
    print("=" * 60)

    model = HousingModel(
        n_households=100,
        n_institutions=5,
        n_properties=120,
        n_zones=10,
        seed=seed,
    )

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

            price_str = (
                f"{avg_price:>10,.0f}" if avg_price == avg_price else "       N/A"
            )
            mp_str = f"{mp_share:.2f}" if mp_share == mp_share else "  N/A"
            rent_str = f"{avg_r:>8,.0f}" if avg_r == avg_r else "     N/A"

            print(
                f"Step {step+1:>3} | "
                f"AvgPrice: {price_str} | "
                f"Vol: {vol:>3} | "
                f"OwnRate: {own_rate:.2f} | "
                f"InstShare: {inst_share:.2f} | "
                f"HH_MP: {mp_str} | "
                f"AvgRent: {rent_str}"
            )

    print("=" * 60)
    print("Simulation complete.")

    # Write results to CSV
    output_path = os.path.join(os.path.dirname(__file__), "housing_abm_results.csv")
    df = model.datacollector.get_model_vars_dataframe()
    df.to_csv(output_path)
    print(f"Results written to: {output_path}")

    return model


def run_credit_shock_experiment(n_steps=40, seed=42):
    """
    Demonstrates the marginal-pricer mechanism by tightening credit at step 20.

    Household WTP falls as mortgage_rate rises.
    Institutional WTP is anchored to yields (less affected).
    Observe shift in household_marginal_pricer_share.
    """
    print("\n" + "=" * 60)
    print("Experiment: Credit Tightening Shock (step 20)")
    print("=" * 60)

    from policies import NoPolicy
    from credit import CreditEnvironment

    class CreditShockPolicy(NoPolicy):
        """Tightens mortgage rate at step 20."""

        def on_step_start(self, model):
            if model.steps == 20:
                model.credit = CreditEnvironment(
                    mortgage_rate=0.08,  # up from 0.05
                    ltv_limit=0.80,  # down from 0.85
                    dti_limit=0.30,  # down from 0.35
                )
                print(
                    "  [SHOCK] Credit tightened at step 20: rate=8%, LTV=80%, DTI=30%"
                )

    model = HousingModel(
        n_households=100,
        n_institutions=5,
        n_properties=120,
        n_zones=10,
        seed=seed,
        policy=CreditShockPolicy(),
    )

    pre_mp = []
    post_mp = []

    for step in range(n_steps):
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        mp = df.iloc[-1]["household_marginal_pricer_share"]
        if mp == mp:  # not NaN
            if step < 20:
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
    # Baseline run
    model = run_simulation(n_steps=30, seed=42, verbose=True)

    # Marginal-pricer experiment
    run_credit_shock_experiment(n_steps=40, seed=42)
