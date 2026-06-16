"""
Properties are passive state containers.
"""

from dataclasses import dataclass


@dataclass
class Property:
    id: int
    zone: int
    quality: float
    owner_id: int | None
    purchase_anchor_price: float
    estimated_value: float = 0.0
    current_rent: float | None = None
    grid_coord: tuple[int, int] | None = None
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
