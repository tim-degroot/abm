"""Entry point for the housing market ABM."""

import argparse
import os
import sys

import pandas as pd

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the housing market ABM.")
    parser.add_argument("--steps", "-s", type=int, help="Number of simulation steps")
    parser.add_argument("--experiment", action="store_true", help="Run credit-tightening shock experiment")
    parser.add_argument("--zone-metrics", action="store_true", help="Collect per-zone metrics")
    return parser.parse_args(argv)


def main():
    from config import Config
    from model import HousingModel
    from policies import CreditShockPolicy
    from metrics import collect_zone_metrics
    from plotting import plot_summary

    args = _parse_args(sys.argv[1:])

    cfg = Config()
    steps = args.steps if args.steps is not None else cfg.sim.n_steps

    policy = CreditShockPolicy() if args.experiment else None
    model = HousingModel(config=cfg, policy=policy)

    label = "Experiment: Credit Tightening Shock" if args.experiment else "Baseline"
    print("=" * 60)
    print(f"Housing Market ABM — {label}")
    print("=" * 60)
    print(f"  Steps:        {steps}")
    print(f"  Households:   {cfg.sim.n_households}")
    print(f"  Institutions: {cfg.sim.n_institutions}")
    print(f"  Properties:   {cfg.sim.n_properties}")
    print(f"  Seed:         {cfg.sim.seed}")
    print("=" * 60)

    zone_rows = []
    for step in range(steps):
        model.step()
        if args.zone_metrics:
            zone_rows.extend(collect_zone_metrics(model))

        df = model.datacollector.get_model_vars_dataframe()
        latest = df.iloc[-1]

        avg_price = latest["avg_sale_price"]
        vol = latest["transaction_volume"]
        own_rate = latest["ownership_rate"]
        inst_share = latest["institutional_ownership_share"]
        avg_r = latest["avg_rent"]

        price_str = f"{avg_price:,.0f}" if pd.notna(avg_price) else "N/A"
        rent_str = f"{avg_r:,.0f}" if pd.notna(avg_r) else "N/A"

        oo = latest.get("owner_occupier_share")
        ll = latest.get("landlord_share")
        inst = latest.get("institution_share")
        oo_str = f"{oo:.2f}" if pd.notna(oo) else "N/A"
        ll_str = f"{ll:.2f}" if pd.notna(ll) else "N/A"
        inst_str = f"{inst:.2f}" if pd.notna(inst) else "N/A"

        print(
            f"Step {step + 1:>3} | "
            f"AvgPrice: {price_str:>8} | "
            f"Vol: {vol:>5} | "
            f"OwnRate: {own_rate:>5.2f} | "
            f"InstShare: {inst_share:>5.2f} | "
            f"OO: {oo_str:>5} LL: {ll_str:>5} Inst: {inst_str:>5} | "
            f"AvgRent: {rent_str:>5}"
        )

    print("=" * 60)
    print("Simulation complete.")

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(_RESULTS_DIR, "housing_abm_results.csv")
    model.datacollector.get_model_vars_dataframe().to_csv(out_csv)
    print(f"Results written to: {out_csv}")

    if zone_rows:
        zone_out = os.path.join(_RESULTS_DIR, "zone_timeseries.csv")
        pd.DataFrame(zone_rows).to_csv(zone_out, index=False)
        print(f"Zone metrics written to: {zone_out}")

    out_png = os.path.join(_RESULTS_DIR, "results_summary.png")
    plot_summary(model.datacollector.get_model_vars_dataframe(), out_png)
    print(f"Chart saved to: {out_png}")


if __name__ == "__main__":
    main()
