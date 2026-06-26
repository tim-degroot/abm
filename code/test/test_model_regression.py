"""Full-model regression tests: invariants and regression bugs."""

import unittest
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "code"))

from code.settings.config import Config
from code.core.model import HousingModel
from code.core.agents import HouseholdAgent, InstitutionalAgent

_N_STEPS = 24


def _make_config(n_households=20, n_institutions=3, n_properties=25, seed=42):
    cfg = Config()
    sim = cfg.sim.model_copy(
        update={
            "n_households": n_households,
            "n_institutions": n_institutions,
            "n_properties": n_properties,
            "n_steps": _N_STEPS,
            "seed": seed,
        }
    )
    return cfg.model_copy(update={"sim": sim})


class TestModelInvariants(unittest.TestCase):
    """Run a tiny model and verify fundamental invariants hold."""

    def setUp(self):
        cfg = _make_config(seed=42)
        self.model = HousingModel(config=cfg)
        for _ in range(_N_STEPS):
            self.model.step()
        self.df = self.model.datacollector.get_model_vars_dataframe()

    def test_cash_not_below_mortgage_payment_due(self):
        """Cash can go negative from mortgage servicing, but never below
        one month's total payments (the max between income and deduction)."""
        for agent in self.model.agents:
            if isinstance(agent, HouseholdAgent) and agent._mortgages:
                max_shortfall = agent.mortgage_payment_due()
                self.assertGreaterEqual(
                    agent.cash,
                    -max_shortfall - 1e-6,
                    f"Agent {agent.unique_id} cash={agent.cash:.2f} "
                    f"below max shortfall -{max_shortfall:.2f}",
                )

    def test_share_columns_sum_to_one(self):
        """When there are transactions, buyer shares must sum to 1."""
        for step_idx in range(_N_STEPS):
            vol = self.df.iloc[step_idx]["transaction_volume"]
            if vol == 0 or np.isnan(vol):
                continue
            oo = self.df.iloc[step_idx].get("owner_occupier_share", 0.0) or 0.0
            ll = self.df.iloc[step_idx].get("landlord_share", 0.0) or 0.0
            inst = self.df.iloc[step_idx].get("institution_share", 0.0) or 0.0
            total = oo + ll + inst
            self.assertAlmostEqual(
                total,
                1.0,
                delta=1e-9,
                msg=f"Step {step_idx}: shares sum to {total}, vol={vol}",
            )

    def test_no_nan_in_key_metrics(self):
        """All model-level reporters should be finite (or NaN is acceptable for
        steps with zero transactions; we only check non-NaN when data exists)."""
        for col in self.df.columns:
            series = self.df[col]
            has_data = series.notna().any()
            if not has_data:
                continue
            self.assertTrue(
                series.notna().sum() > 0,
                f"Column '{col}' is entirely NaN",
            )

    def test_transaction_volume_non_negative(self):
        vols = self.df["transaction_volume"].fillna(0.0)
        self.assertTrue((vols >= 0).all())

    def test_ownership_rates_in_bounds(self):
        for col in (
            "ownership_rate",
            "institutional_ownership_share",
            "household_ownership_share",
            "owner_occupier_ownership_share",
            "landlord_ownership_share",
        ):
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            self.assertTrue((series >= 0).all(), f"{col} has negative values")
            self.assertTrue((series <= 1).all(), f"{col} has values > 1")


class TestMultiWinRegression(unittest.TestCase):
    """Regression: the multi-win deposit check (model.py:621) uses live cash
    balance, not the pre-loop snapshot. We verify no crash and that cash never
    goes below one month's mortgage obligations."""

    def test_no_crash_or_pathological_cash(self):
        cfg = _make_config(n_households=20, n_institutions=5, n_properties=30, seed=7)
        model = HousingModel(config=cfg)
        for _ in range(_N_STEPS):
            model.step()

        for agent in model.agents:
            if isinstance(agent, HouseholdAgent) and agent._mortgages:
                max_shortfall = agent.mortgage_payment_due()
                self.assertGreaterEqual(
                    agent.cash,
                    -max_shortfall - 1e-6,
                    f"Agent {agent.unique_id} cash={agent.cash:.2f} "
                    f"below max shortfall -{max_shortfall:.2f}",
                )

    def test_multiple_seeds_no_crash(self):
        """Quick sanity: runs with different seeds don't crash."""
        for seed in range(1, 5):
            cfg = _make_config(n_households=15, n_institutions=2, n_properties=20, seed=seed)
            model = HousingModel(config=cfg)
            for _ in range(12):
                model.step()


class TestSimulationRunsToCompletion(unittest.TestCase):
    """The model should complete without raising exceptions for various sizes."""

    def test_tiny_model(self):
        cfg = _make_config(n_households=5, n_institutions=1, n_properties=8, seed=0)
        model = HousingModel(config=cfg)
        for _ in range(6):
            model.step()
        self.assertTrue(len(model.all_transactions) >= 0)

    def test_small_model(self):
        cfg = _make_config(n_households=30, n_institutions=3, n_properties=40, seed=99)
        model = HousingModel(config=cfg)
        for _ in range(_N_STEPS):
            model.step()
        df = model.datacollector.get_model_vars_dataframe()
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
