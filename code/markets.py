"""
Market layer.

Matching and auction logic lives here, not in agents.
This module is designed to be replaced wholesale in future commits.

Ownership Market:
  - Agents submit bids (bidder_id, property_id, amount)
  - Vickrey auction per property: highest bidder wins, pays second-highest price
  - Seller's reservation price must be met
  - Returns list of Transaction records

Rental Market:
  - Structurally analogous to ownership market
  - Tenants submit rent bids; landlords list properties with reservation rent
  - Highest rent bidder wins, pays second-highest rent
  - Returns list of RentalTransaction records
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Transaction records
# ---------------------------------------------------------------------------


@dataclass
class Transaction:
    """Records a completed ownership transfer."""

    step: int
    property_id: int
    buyer_id: int
    seller_id: int
    price: float  # Vickrey price (second-highest bid)
    winning_bid: float  # Highest bid submitted
    buyer_type: str  # 'household' or 'institution'


@dataclass
class RentalTransaction:
    """Records a completed rental agreement."""

    step: int
    property_id: int
    tenant_id: int
    landlord_id: int
    monthly_rent: float  # Vickrey rent (second-highest bid)
    winning_rent_bid: float  # Highest rent bid
    tenant_type: str  # 'household'


# ---------------------------------------------------------------------------
# Ownership market
# ---------------------------------------------------------------------------


class OwnershipMarket:
    """
    Runs a Vickrey (second-price sealed-bid) auction for each listed property.

    Usage each step:
        market = OwnershipMarket(step)
        market.list_property(property_id, seller_id, reservation_price)
        market.submit_bid(property_id, bidder_id, amount, bidder_type)
        transactions = market.clear()
    """

    def __init__(self, step):
        self.step = step
        # property_id -> {'seller_id': ..., 'reservation': ..., 'bids': []}
        self._listings = {}

    def list_property(self, property_id, seller_id, reservation_price):
        """Register a property for sale this period."""
        self._listings[property_id] = {
            "seller_id": seller_id,
            "reservation": reservation_price,
            "bids": [],
        }

    def submit_bid(self, property_id, bidder_id, amount, bidder_type):
        """
        Submit a bid for a listed property.

        bidder_type: 'household' or 'institution'
        """
        if property_id not in self._listings:
            return  # property not listed; bid silently dropped
        self._listings[property_id]["bids"].append(
            {"bidder_id": bidder_id, "amount": amount, "bidder_type": bidder_type}
        )

    def clear(self):
        """
        Run Vickrey auction for all listed properties.

        Returns list of Transaction objects for successful sales.
        Unsold properties are not returned (caller marks them unsold).
        """
        transactions = []

        for property_id, listing in self._listings.items():
            reservation = listing["reservation"]
            bids = listing["bids"]

            if not bids:
                continue

            # Sort descending by bid amount
            sorted_bids = sorted(bids, key=lambda b: b["amount"], reverse=True)
            top_bid = sorted_bids[0]

            # Must beat reservation price
            if top_bid["amount"] < reservation:
                continue

            # Vickrey price: second-highest bid, floored at reservation
            if len(sorted_bids) >= 2:
                price = max(sorted_bids[1]["amount"], reservation)
            else:
                price = reservation

            transactions.append(
                Transaction(
                    step=self.step,
                    property_id=property_id,
                    buyer_id=top_bid["bidder_id"],
                    seller_id=listing["seller_id"],
                    price=price,
                    winning_bid=top_bid["amount"],
                    buyer_type=top_bid["bidder_type"],
                )
            )

        return transactions


# ---------------------------------------------------------------------------
# Rental market
# ---------------------------------------------------------------------------


class RentalMarket:
    """
    Runs a Vickrey rent auction for each listed rental property.

    Usage each step:
        market = RentalMarket(step)
        market.list_property(property_id, landlord_id, reservation_rent)
        market.submit_bid(property_id, tenant_id, monthly_rent_bid)
        transactions = market.clear()
    """

    def __init__(self, step):
        self.step = step
        self._listings = {}

    def list_property(self, property_id, landlord_id, reservation_rent):
        """Register a property for rent this period."""
        self._listings[property_id] = {
            "landlord_id": landlord_id,
            "reservation": reservation_rent,
            "bids": [],
        }

    def submit_bid(self, property_id, tenant_id, monthly_rent_bid):
        """Submit a rent bid for a listed rental property."""
        if property_id not in self._listings:
            return
        self._listings[property_id]["bids"].append(
            {"tenant_id": tenant_id, "amount": monthly_rent_bid}
        )

    def clear(self):
        """
        Run Vickrey rent auction for all listed rentals.

        Returns list of RentalTransaction objects.
        """
        transactions = []

        for property_id, listing in self._listings.items():
            reservation = listing["reservation"]
            bids = listing["bids"]

            if not bids:
                continue

            sorted_bids = sorted(bids, key=lambda b: b["amount"], reverse=True)
            top_bid = sorted_bids[0]

            if top_bid["amount"] < reservation:
                continue

            if len(sorted_bids) >= 2:
                rent = max(sorted_bids[1]["amount"], reservation)
            else:
                rent = reservation

            transactions.append(
                RentalTransaction(
                    step=self.step,
                    property_id=property_id,
                    tenant_id=top_bid["tenant_id"],
                    landlord_id=listing["landlord_id"],
                    monthly_rent=rent,
                    winning_rent_bid=top_bid["amount"],
                    tenant_type="household",
                )
            )

        return transactions
