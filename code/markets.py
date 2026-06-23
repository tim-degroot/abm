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

            assert price <= top_bid["amount"], (
                f"Vickrey invariant violated: price={price} > "
                f"top_bid={top_bid['amount']} for property {property_id}"
            )
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

    def resolve(self, rng=None):
        """
        Sequential rental clearing. Households may bid on multiple feasible rentals. Once a household wins
        one rental, it is removed from later rental auctions in the same clearing
        round.
        """
        property_ids = sorted(
            self._listings.keys(),
            key=lambda pid: max(
                (b["amount"] for b in self._listings[pid]["bids"]),
                default=0,
            ),
            reverse=True,
        )

        if rng is not None and len(property_ids) > 1:
            if hasattr(rng, "permutation"):
                order = rng.permutation(len(property_ids))
                property_ids = [property_ids[int(i)] for i in order]
            elif hasattr(rng, "shuffle"):
                rng.shuffle(property_ids)

        transactions = []
        assigned_tenants = set()

        for property_id in property_ids:
            listing = self._listings[property_id]
            reservation = listing["reservation"]

            bids = [
                b for b in listing["bids"]
                if b["bidder_id"] not in assigned_tenants
            ]
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

            transactions.append(
                self._create_transaction(property_id, listing, top_bid, price)
            )
            assigned_tenants.add(top_bid["bidder_id"])

        return transactions
