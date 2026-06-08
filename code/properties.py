"""
Properties are passive state containers.

They carry no behavioral logic. All state changes are performed
by the market layer or model after resolving transactions.
"""

from dataclasses import dataclass


@dataclass
class Property:
    """
    A single dwelling unit.

    purchase_anchor_price:
        The price at which the *current owner* acquired this property.
        Reset to transaction price on each sale.
        Used for loss-aversion and gain/loss accounting.

    estimated_value:
        Mark-to-market estimate updated each period from transaction
        history. Used for net-worth accounting. Distinct from anchor.

    current_rent:
        Current monthly rent charged for the property if it is rented out.
        None means the property is not currently let.

    owner_id of None means unowned (bank-owned / vacant stock).
    occupant_id of None means physically unoccupied.
    Ownership and occupancy are independent: a landlord owns but does
    not occupy; a renter occupies but does not own.
    """

    id: int
    zone: int
    quality: float  # standardised; mean 0, sd 1 across stock
    owner_id: int | None  # unique_id of owning agent, or None
    purchase_anchor_price: float
    estimated_value: float = 0.0  # mark-to-market; set after init
    current_rent: float | None = None
    grid_coord: tuple[int, int] | None = None

    # Occupancy (who lives here, regardless of ownership)
    occupant_id: int | None = None
    # Length of the CURRENT rental tenancy, in periods (quarters). Aged each
    # step by the model and reset to 0 whenever a new tenant moves in. Used to
    # enforce a minimum lease term before a tenancy can turn over normally.
    tenancy_quarters: int = 0

    # Market listing state — only listed_for_sale and listed_for_rent matter;
    # the auction reservation price is passed directly to the market layer.
    listed_for_sale: bool = False
    listed_for_rent: bool = False

    def __repr__(self):
        return (
            f"Property(id={self.id}, zone={self.zone}, "
            f"quality={self.quality:.2f}, owner={self.owner_id}, "
            f"occupant={self.occupant_id})"
        )
