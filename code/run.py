"""
Entry point for the Housing Market ABM.

Subcommands:
    baseline    Run baseline simulation (per-step console output + CSV)
    experiment  Run credit-tightening shock experiment
    report      Run with per-zone metrics and produce summary charts
    large       Run large-scale configuration with summary charts

Usage:
    uv run python run.py baseline --steps 100
    uv run python run.py experiment --steps 480
    uv run python run.py report --steps 200
    uv run python run.py large --steps 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_results_dir():
    os.makedirs(_RESULTS_DIR, exist_ok=True)


def _resolve_path(name):
    """Resolve *name* under the results directory unless it is already absolute."""
    return name if os.path.isabs(name) else os.path.join(_RESULTS_DIR, name)


# ---------------------------------------------------------------------------
# baseline  —  console-printing simulation, writes CSV
# ---------------------------------------------------------------------------


def run_baseline(args):
    from config import Config
    from model import HousingModel

    cfg = Config()
    if args.seed is not None:
        try:
            cfg.sim.seed = args.seed
        except Exception:
            pass

    n_steps = args.steps if args.steps is not None else cfg.sim.n_steps

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
            f"Step {step + 1:>3} | "
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

    _ensure_results_dir()
    out = _resolve_path("housing_abm_results.csv")
    model.datacollector.get_model_vars_dataframe().to_csv(out)
    print(f"Results written to: {out}")

    return model


# ---------------------------------------------------------------------------
# experiment  —  credit-tightening shock at step 240
# ---------------------------------------------------------------------------


def run_experiment(args):
    from config import Config
    from model import HousingModel
    from policies import NoPolicy
    from credit import CreditEnvironment

    cfg = Config()
    n_steps = args.steps if args.steps is not None else 480

    print("\n" + "=" * 60)
    print("Experiment: Credit Tightening Shock (step 240)")
    print("=" * 60)

    class CreditShockPolicy(NoPolicy):
        def on_step_start(self, model):
            if model.steps == 240:
                model.credit = CreditEnvironment(
                    mortgage_rate=0.006667,
                    ltv_limit=0.80,
                    dti_limit=0.30,
                    loan_term_months=cfg.credit.loan_term_months,
                )
                print(
                    "  [SHOCK] Credit tightened at step 240: "
                    "rate=8% p.a. (0.006667/mo), LTV=80%, DTI=30%"
                )

    model = HousingModel(config=cfg, policy=CreditShockPolicy())

    pre_mp = []
    post_mp = []

    for step in range(n_steps):
        model.step()
        df = model.datacollector.get_model_vars_dataframe()
        mp = df.iloc[-1]["household_marginal_pricer_share"]
        if mp == mp:
            if step < 240:
                pre_mp.append(mp)
            else:
                post_mp.append(mp)

    pre_avg = sum(pre_mp) / len(pre_mp) if pre_mp else float("nan")
    post_avg = sum(post_mp) / len(post_mp) if post_mp else float("nan")

    print(f"\n  Pre-shock HH marginal pricer share (avg):  {pre_avg:.3f}")
    print(f"  Post-shock HH marginal pricer share (avg): {post_avg:.3f}")

    direction = (
        "\u2193 institutions gaining"
        if post_avg < pre_avg
        else "\u2191 households dominant"
    )
    print(f"  Direction: {direction}")
    print("=" * 60)

    return model


# ---------------------------------------------------------------------------
# report  —  longer run + per-zone metrics + summary charts
# ---------------------------------------------------------------------------


def run_report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from model import HousingModel
    from config import Config

    cfg = Config()
    steps = args.steps
    out_png = _resolve_path(args.out)
    out_csv = _resolve_path(args.csv)

    _ensure_results_dir()

    m = HousingModel(config=cfg)

    zone_rows = []
    for i in range(steps):
        m.step()

        n_zones = cfg.spatial.n_zones
        for zone in range(n_zones):
            props = [p for p in m.properties if p.zone == zone]
            if not props:
                continue
            avg_est_val = float(sum(p.estimated_value for p in props) / len(props))
            ownership_rate = float(
                sum(1 for p in props if p.owner_id is not None) / len(props)
            )
            inst_share = (
                float(
                    sum(
                        1
                        for p in props
                        if p.owner_id is not None
                        and m._property_map[p.id].owner_id is not None
                        and getattr(
                            m._agent_map.get(m._property_map[p.id].owner_id),
                            "is_institution",
                            False,
                        )
                    )
                    / sum(1 for p in props if p.owner_id is not None)
                )
                if any(p.owner_id is not None for p in props)
                else 0.0
            )
            txns = [
                t
                for t in m.this_step_transactions
                if m._property_map[t.property_id].zone == zone
            ]
            txn_vol = len(txns)
            avg_txn_price = (
                float(sum(t.price for t in txns) / len(txns)) if txns else float("nan")
            )

            zone_rows.append(
                {
                    "step": i,
                    "zone": zone,
                    "avg_estimated_value": avg_est_val,
                    "ownership_rate": ownership_rate,
                    "institutional_ownership_share": inst_share,
                    "transaction_volume": txn_vol,
                    "avg_transaction_price": avg_txn_price,
                }
            )

    df = m.datacollector.get_model_vars_dataframe()
    df.index.name = "step"

    df_filled = df.copy()
    for col in ("avg_sale_price", "avg_rent"):
        if col in df_filled.columns:
            med = (
                float(df_filled[col].median(skipna=True))
                if df_filled[col].notna().any()
                else 0.0
            )
            df_filled[col + "_filled"] = df_filled[col].ffill().fillna(med)

    df_filled.to_csv(out_csv)

    zone_df = pd.DataFrame(zone_rows)
    zone_out = _resolve_path("zone_timeseries.csv")
    zone_df.to_csv(zone_out, index=False)
    print(f"Wrote zone-level timeseries to {zone_out}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    if "avg_sale_price_filled" in df_filled.columns:
        df_filled["avg_sale_price_filled"].plot(
            ax=ax[0], title="Average Sale Price (filled)"
        )
    elif "avg_sale_price" in df.columns:
        df["avg_sale_price"].plot(ax=ax[0], title="Average Sale Price")
    else:
        ax[0].text(0.5, 0.5, "avg_sale_price not collected", ha="center")

    if "avg_rent_filled" in df_filled.columns:
        df_filled["avg_rent_filled"].plot(ax=ax[1], title="Average Rent (filled)")
    elif "avg_rent" in df.columns:
        df["avg_rent"].plot(ax=ax[1], title="Average Rent")
    else:
        ax[1].text(0.5, 0.5, "avg_rent not collected", ha="center")

    if "ownership_rate" in df_filled.columns:
        df_filled["ownership_rate"].plot(ax=ax[2], title="Household Ownership Rate")
    else:
        ax[2].text(0.5, 0.5, "ownership_rate not collected", ha="center")

    if {
        "household_ownership_share_of_stock",
        "institutional_ownership_share",
    }.issubset(df_filled.columns):
        df_filled["household_ownership_share_of_stock"].plot(
            ax=ax[3], title="Housing Stock Shares"
        )
        df_filled["institutional_ownership_share"].plot(ax=ax[3])
        ax[3].legend(["household stock share", "institutional stock share"])
    elif "institutional_ownership_share" in df_filled.columns:
        df_filled["institutional_ownership_share"].plot(
            ax=ax[3], title="Institutional Ownership Share"
        )
    elif "inst_property_count" in df_filled.columns:
        df_filled["inst_property_count"].plot(
            ax=ax[3], title="Institutional Property Count"
        )
    else:
        ax[3].text(0.5, 0.5, "Institutional data not collected", ha="center")

    plt.tight_layout()
    fig.savefig(out_png)
    print(f"Wrote {out_png} and {out_csv}")


# ---------------------------------------------------------------------------
# large  —  large-scale run with summary charts
# ---------------------------------------------------------------------------


def run_large(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from config import Config, SimConfig
    from model import HousingModel

    base = Config()
    steps = args.steps
    out_csv = _resolve_path(args.csv)
    out_png = _resolve_path(args.out)

    sim_cfg = SimConfig(
        n_households=1000,
        n_institutions=50,
        n_properties=1200,
        seed=base.sim.seed,
        n_steps=steps,
    )
    cfg = Config(
        sim=sim_cfg,
        spatial=base.spatial,
        property_init=base.property_init,
        agent_init=base.agent_init,
        credit=base.credit,
        valuation=base.valuation,
        expectations=base.expectations,
    )

    _ensure_results_dir()

    m = HousingModel(config=cfg)
    for _ in range(steps):
        m.step()

    df = m.datacollector.get_model_vars_dataframe()
    df.index.name = "step"
    df.to_csv(out_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()
    if "avg_sale_price" in df.columns:
        df["avg_sale_price"].ffill().plot(ax=ax[0], title="Average Sale Price")
    if "avg_rent" in df.columns:
        df["avg_rent"].ffill().plot(ax=ax[1], title="Average Rent")
    if "ownership_rate" in df.columns:
        df["ownership_rate"].plot(ax=ax[2], title="Household Ownership Rate")
    if "institutional_ownership_share" in df.columns:
        df["institutional_ownership_share"].plot(
            ax=ax[3], title="Institutional Ownership Share"
        )
    plt.tight_layout()
    fig.savefig(out_png)
    print(f"Wrote {out_csv} and {out_png}")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def _add_common_args(parser):
    parser.add_argument("--steps", "-s", type=int, help="number of simulation steps")


def main():
    parser = argparse.ArgumentParser(description="Housing Market ABM entry point")
    sub = parser.add_subparsers(dest="command", required=True)

    # baseline
    p = sub.add_parser("baseline", help="Run baseline simulation")
    _add_common_args(p)
    p.add_argument("--seed", type=int, help="override RNG seed")
    p.set_defaults(func=run_baseline)

    # experiment
    p = sub.add_parser("experiment", help="Run credit-tightening shock experiment")
    _add_common_args(p)
    p.set_defaults(func=run_experiment)

    # report
    p = sub.add_parser("report", help="Run with per-zone metrics and summary charts")
    _add_common_args(p)
    p.add_argument("--out", default="report_summary.png", help="Output PNG filename")
    p.add_argument("--csv", default="model_timeseries.csv", help="Output CSV filename")
    p.set_defaults(func=run_report)

    # large
    p = sub.add_parser(
        "large", help="Run large-scale configuration with summary charts"
    )
    _add_common_args(p)
    p.add_argument(
        "--out", default="report_summary_large.png", help="Output PNG filename"
    )
    p.add_argument(
        "--csv", default="model_timeseries_large.csv", help="Output CSV filename"
    )
    p.set_defaults(func=run_large)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
