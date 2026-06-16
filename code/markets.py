"""Vickrey (second-price sealed-bid) auction markets for ownership and rentals."""

from dataclasses import dataclass


@dataclass
class Transaction:
    """Completed ownership transfer."""

    step: int
    property_id: int
    buyer_id: int
    seller_id: int
    price: float
    winning_bid: float
    buyer_type: str
    origination_ltv: float | None = None


@dataclass
class RentalTransaction:
    """Completed rental agreement."""

    step: int
    property_id: int
    tenant_id: int
    landlord_id: int
    monthly_rent: float
    winning_rent_bid: float


class BaseMarket:
    """
    Vickrey auction for a single step.
    Usage:
        market = BaseMarket(step)
        market.list_property(pid, owner_id, reservation)
        market.submit_bid(pid, bidder_id, amount, ...)
        txns = market.resolve()
    """

    def __init__(self, step: int):
        self.step = step
        self._listings: dict = {}

    def list_property(self, property_id: int, owner_id: int, reservation: float) -> None:
        """Register a property with a minimum reservation price."""
        self._listings[property_id] = {
            "owner_id": owner_id,
            "reservation": reservation,
            "bids": [],
        }

    def submit_bid(
        self,
        property_id: int,
        bidder_id: int,
        amount: float,
        bidder_type: str | None = None,
        origination_ltv: float | None = None,
    ) -> None:
        """Record a bid. Silently dropped if amount <= 0 or property unknown."""
        if property_id not in self._listings or amount <= 0:
            return
        self._listings[property_id]["bids"].append(
            {
                "bidder_id": bidder_id,
                "amount": amount,
                "bidder_type": bidder_type,
                "origination_ltv": origination_ltv,
            }
        )

    def resolve(self) -> list:
        """Settle all auctions."""
        transactions = []
        for property_id, listing in self._listings.items():
            reservation = listing["reservation"]
            bids = listing["bids"]
            if not bids:
                continue

            sorted_bids = sorted(bids, key=lambda b: b["amount"], reverse=True)
            top_bid = sorted_bids[0]

            if top_bid["amount"] <= 0 or top_bid["amount"] < reservation:
                continue

            if len(sorted_bids) >= 2:
                price = max(sorted_bids[1]["amount"], reservation)
            else:
                price = max(top_bid["amount"], reservation)

            transactions.append(self._create_transaction(property_id, listing, top_bid, price))

        return transactions

    def _create_transaction(self, property_id: int, listing: dict, top_bid: dict, price: float):
        """Override in subclass to return the correct record type."""
        raise NotImplementedError


class OwnershipMarket(BaseMarket):
    """Vickrey auction producing Transaction records for ownership."""

    def _create_transaction(
        self, property_id: int, listing: dict, top_bid: dict, price: float
    ) -> Transaction:
        return Transaction(
            step=self.step,
            property_id=property_id,
            buyer_id=top_bid["bidder_id"],
            seller_id=listing["owner_id"],
            price=price,
            winning_bid=top_bid["amount"],
            buyer_type=top_bid["bidder_type"],
            origination_ltv=top_bid.get("origination_ltv"),
        )


class RentalMarket(BaseMarket):
    """Vickrey auction producing RentalTransaction records, with tenant dedup."""

    def _create_transaction(
        self, property_id: int, listing: dict, top_bid: dict, price: float
    ) -> RentalTransaction:
        return RentalTransaction(
            step=self.step,
            property_id=property_id,
            tenant_id=top_bid["bidder_id"],
            landlord_id=listing["owner_id"],
            monthly_rent=price,
            winning_rent_bid=top_bid["amount"],
        )

    def resolve(self):
        """Settle auctions, then keep only each tenant's highest-rent win."""
        transactions = super().resolve()
        winners = []
        winning_tenants = set()
        for txn in sorted(transactions, key=lambda t: (-t.winning_rent_bid, t.property_id)):
            if txn.tenant_id in winning_tenants:
                continue
            winners.append(txn)
            winning_tenants.add(txn.tenant_id)
        return winners
