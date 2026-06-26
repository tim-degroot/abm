"""
Sealed-bid second-price (Vickrey) auction markets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Transaction:
    step: int
    property_id: int
    buyer_id: int
    seller_id: int
    price: float
    winning_bid: float
    buyer_type: str
    purpose: str


@dataclass
class RentalTransaction:
    step: int
    property_id: int
    tenant_id: int
    landlord_id: int
    monthly_rent: float
    winning_rent_bid: float


class OwnershipMarket:
    """Vickrey auction for ownership transfers, with seller reservation prices."""

    def __init__(self, step: int):
        self.step = step
        self._listings: dict = {}

    def list_property(self, property_id: int, owner_id: int, reservation: float) -> None:
        self._listings[property_id] = {
            "owner_id": owner_id,
            "reservation": reservation,
            "bids": [],
        }

    def submit_bid(self, property_id, bidder_id, amount, bidder_type=None, purpose="buy") -> None:
        if property_id not in self._listings or amount <= 0:
            return
        self._listings[property_id]["bids"].append(
            {
                "bidder_id": bidder_id,
                "amount": amount,
                "bidder_type": bidder_type,
                "purpose": purpose,
            }
        )

    def resolve(self) -> list[Transaction]:
        transactions = []
        for pid, listing in self._listings.items():
            # process them in order
            reservation = listing["reservation"]
            bids = listing["bids"]
            if not bids:
                continue
            ordered = sorted(bids, key=lambda b: b["amount"], reverse=True)
            top = ordered[0]
            if top["amount"] < reservation:
                continue  # best bid below the seller's reservation; no sale
            second = ordered[1]["amount"] if len(ordered) >= 2 else top["amount"]
            price = max(second, reservation)
            price = min(price, top["amount"])  # Vickrey: never above the winning bid
            transactions.append(
                Transaction(
                    step=self.step,
                    property_id=pid,
                    buyer_id=top["bidder_id"],
                    seller_id=listing["owner_id"],
                    price=price,
                    winning_bid=top["amount"],
                    buyer_type=top["bidder_type"],
                    purpose=top.get("purpose", "buy"),
                )
            )
        return transactions


class RentalMarket:
    """Vickrey rental auction; no reservation; sequential tenant de-duplication."""

    def __init__(self, step: int):
        self.step = step
        self._listings: dict = {}

    def list_property(self, property_id: int, owner_id: int) -> None:
        self._listings[property_id] = {"owner_id": owner_id, "bids": []}

    def submit_bid(self, property_id, bidder_id, amount) -> None:
        if property_id not in self._listings or amount <= 0:
            return
        self._listings[property_id]["bids"].append({"bidder_id": bidder_id, "amount": amount})

    def resolve(self) -> list[RentalTransaction]:
        transactions = []
        assigned: set[int] = set()
        for pid, listing in sorted(
            self._listings.items(),
            key=lambda item: max((b["amount"] for b in item[1]["bids"]), default=0),
            reverse=True,
        ):
            bids = [b for b in listing["bids"] if b["bidder_id"] not in assigned]
            if not bids:
                continue
            ordered = sorted(bids, key=lambda b: b["amount"], reverse=True)
            top = ordered[0]
            second = ordered[1]["amount"] if len(ordered) >= 2 else top["amount"]
            rent = min(top["amount"], second)  # second-price, no reservation
            transactions.append(
                RentalTransaction(
                    step=self.step,
                    property_id=pid,
                    tenant_id=top["bidder_id"],
                    landlord_id=listing["owner_id"],
                    monthly_rent=rent,
                    winning_rent_bid=top["amount"],
                )
            )
            assigned.add(top["bidder_id"])
        return transactions


__all__ = ["Transaction", "RentalTransaction", "OwnershipMarket", "RentalMarket"]
