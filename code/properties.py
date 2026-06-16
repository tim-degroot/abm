<<<<<<< HEAD
=======
"""
Properties are passive state containers.
"""

>>>>>>> 8627f7a (madonna)
from dataclasses import dataclass


@dataclass
class Property:
<<<<<<< HEAD
    id: int
    zone: int
    quality: float
    owner_id: int | None
=======
    """
    purchase_anchor_price:
        The price at which the *current owner* acquired this property.

    estimated_value:
        Mark-to-market estimate updated each period from transaction
        history. Used for net-worth accounting.

    current_rent:
        Current monthly rent charged for the property if it is rented out.
    """

    id: int
    zone: int
    quality: float  # standardised; mean 0, sd 1 across stock
    owner_id: int
>>>>>>> 8627f7a (madonna)
    purchase_anchor_price: float
    estimated_value: float = 0.0
    current_rent: float | None = None
<<<<<<< HEAD
    grid_coord: tuple[int, int] | None = None
=======
    grid_coord: tuple[int, int]
>>>>>>> 8627f7a (madonna)
    occupant_id: int | None = None
    tenancy_months: int = 0
    listed_for_sale: bool = False
    listed_for_rent: bool = False

    def __repr__(self):
        return (
            f"Property(id={self.id}, zone={self.zone}, "
            f"quality={self.quality:.2f}, owner={self.owner_id}, "
            f"occupant={self.occupant_id})"
        )
