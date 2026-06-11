import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from agents import HouseholdAgent
from config import Config, SimConfig, SpatialConfig
from model import HousingModel


class TestHouseholdAccounting(unittest.TestCase):
    def test_net_worth_subtracts_mortgage_debt(self):
        # Testing:
        # loan adjusted wealth = cash + house value - mortgage debt
        cfg = Config(
            sim=SimConfig(
                n_households=20,
                n_institutions=2,
                n_properties=30,
                ownership_mode="target",
            ),
            spatial=SpatialConfig(grid_rows=3, grid_cols=3),
        )
        model = HousingModel(config=cfg)

        for h in [a for a in model.agents if isinstance(a, HouseholdAgent)]:
            expected_debt = sum(
                model.credit.outstanding_principal(orig_price, ltv, steps_held)
                for orig_price, ltv, steps_held in h._mortgages.values()
            )

            self.assertAlmostEqual(h.mortgage_debt, expected_debt)
            self.assertAlmostEqual(h.housing_equity, h.gross_housing_assets - h.mortgage_debt)
            self.assertAlmostEqual(h.net_worth, h.cash + h.housing_equity)


if __name__ == "__main__":
    unittest.main()