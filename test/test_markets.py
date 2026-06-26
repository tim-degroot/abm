import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.markets import OwnershipMarket, RentalMarket, Transaction, RentalTransaction


class TestMarketEmpty(unittest.TestCase):
    def test_clear_returns_empty_list_when_nothing_listed(self):
        market = OwnershipMarket(step=0)
        self.assertEqual(market.resolve(), [])

    def test_clear_returns_empty_list_when_no_bids(self):
        market = OwnershipMarket(step=0)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        self.assertEqual(market.resolve(), [])


class TestOwnershipMarketSingleBidder(unittest.TestCase):
    def test_winner_pays_max_own_bid_reservation(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=50_000)
        market.submit_bid(property_id=1, bidder_id=20, amount=100_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 100_000)

    def test_single_bid_pays_reservation_when_bid_above_reserve(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=75_000)
        market.submit_bid(property_id=1, bidder_id=20, amount=100_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 100_000)


class TestOwnershipMarketVickreyPricing(unittest.TestCase):
    def test_winner_pays_second_highest_bid(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=1, bidder_id=20, amount=200_000)
        market.submit_bid(property_id=1, bidder_id=21, amount=150_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 150_000)

    def test_winner_pays_reservation_when_second_bid_below_reserve(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=180_000)
        market.submit_bid(property_id=1, bidder_id=20, amount=200_000)
        market.submit_bid(property_id=1, bidder_id=21, amount=100_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 180_000)

    def test_winner_pays_own_bid_if_only_bidder(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=1, bidder_id=20, amount=90_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 90_000)


class TestBaseMarketReservation(unittest.TestCase):
    def test_bid_below_reservation_no_transaction(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=100_000)
        market.submit_bid(property_id=1, bidder_id=20, amount=50_000)
        self.assertEqual(market.resolve(), [])

    def test_bid_equal_to_reservation_succeeds(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=100_000)
        market.submit_bid(property_id=1, bidder_id=20, amount=100_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].price, 100_000)


class TestBaseMarketBidFiltering(unittest.TestCase):
    def test_zero_bid_silently_dropped(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=1, bidder_id=20, amount=0)
        self.assertEqual(market.resolve(), [])

    def test_negative_bid_silently_dropped(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=1, bidder_id=20, amount=-100)
        self.assertEqual(market.resolve(), [])

    def test_bid_on_unlisted_property_dropped(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=99, bidder_id=20, amount=100_000)
        market.submit_bid(property_id=1, bidder_id=21, amount=50_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].buyer_id, 21)

    def test_mixed_valid_and_invalid_bids(self):
        market = OwnershipMarket(step=1)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(property_id=1, bidder_id=20, amount=0)
        market.submit_bid(property_id=1, bidder_id=21, amount=80_000)
        market.submit_bid(property_id=1, bidder_id=22, amount=-1)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].buyer_id, 21)
        self.assertEqual(txns[0].price, 80_000)


class TestOwnershipMarket(unittest.TestCase):
    def test_returns_transaction_record(self):
        market = OwnershipMarket(step=5)
        market.list_property(property_id=1, owner_id=10, reservation=0)
        market.submit_bid(
            property_id=1,
            bidder_id=20,
            amount=100_000,
            bidder_type="household",
            purpose="buy",
        )
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        txn = txns[0]
        self.assertIsInstance(txn, Transaction)
        self.assertEqual(txn.step, 5)
        self.assertEqual(txn.property_id, 1)
        self.assertEqual(txn.buyer_id, 20)
        self.assertEqual(txn.seller_id, 10)
        self.assertEqual(txn.price, 100_000)
        self.assertEqual(txn.winning_bid, 100_000)
        self.assertEqual(txn.buyer_type, "household")
        self.assertEqual(txn.purpose, "buy")


class TestRentalMarket(unittest.TestCase):
    def test_returns_rental_transaction_record(self):
        market = RentalMarket(step=3)
        market.list_property(property_id=1, owner_id=10)
        market.submit_bid(property_id=1, bidder_id=20, amount=1_500)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        txn = txns[0]
        self.assertIsInstance(txn, RentalTransaction)
        self.assertEqual(txn.step, 3)
        self.assertEqual(txn.property_id, 1)
        self.assertEqual(txn.tenant_id, 20)
        self.assertEqual(txn.landlord_id, 10)
        self.assertEqual(txn.monthly_rent, 1_500)

    def test_tenant_dedup_keeps_highest_rent_bid(self):
        market = RentalMarket(step=1)
        market.list_property(property_id=1, owner_id=10)
        market.list_property(property_id=2, owner_id=11)
        market.submit_bid(property_id=1, bidder_id=20, amount=2_000)
        market.submit_bid(property_id=2, bidder_id=20, amount=1_500)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].property_id, 1)
        self.assertEqual(txns[0].tenant_id, 20)
        self.assertEqual(txns[0].monthly_rent, 2_000)

    def test_tenant_dedup_reversed_order(self):
        market = RentalMarket(step=1)
        market.list_property(property_id=1, owner_id=10)
        market.list_property(property_id=2, owner_id=11)
        market.submit_bid(property_id=1, bidder_id=20, amount=1_000)
        market.submit_bid(property_id=2, bidder_id=20, amount=2_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 1)
        self.assertEqual(txns[0].property_id, 2)
        self.assertEqual(txns[0].monthly_rent, 2_000)

    def test_different_tenants_both_win(self):
        market = RentalMarket(step=1)
        market.list_property(property_id=1, owner_id=10)
        market.list_property(property_id=2, owner_id=11)
        market.submit_bid(property_id=1, bidder_id=20, amount=2_000)
        market.submit_bid(property_id=2, bidder_id=21, amount=1_500)
        txns = market.resolve()
        self.assertEqual(len(txns), 2)

    def test_tenant_dedup_respects_sort_order(self):
        market = RentalMarket(step=1)
        market.list_property(property_id=1, owner_id=10)
        market.list_property(property_id=2, owner_id=11)
        market.list_property(property_id=3, owner_id=12)
        market.submit_bid(property_id=1, bidder_id=20, amount=1_000)
        market.submit_bid(property_id=2, bidder_id=20, amount=3_000)
        market.submit_bid(property_id=3, bidder_id=21, amount=2_000)
        txns = market.resolve()
        self.assertEqual(len(txns), 2)
        tenant_ids = {t.tenant_id for t in txns}
        self.assertEqual(tenant_ids, {20, 21})


if __name__ == "__main__":
    unittest.main()
