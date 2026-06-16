"""Data-collector reporters passed to Mesa's DataCollector.

All functions take the model as their argument.
"""

import numpy as np
from agents import HouseholdAgent, InstitutionalAgent


def avg_sale_price(model):
    """Mean transaction price this step."""
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    return sum(t.price for t in txns) / len(txns)


def transaction_volume(model):
    """Number of ownership transactions this step."""
    return len(model.this_step_transactions)


def ownership_rate(model):
    """Fraction of households that own at least one property."""
    households = [a for a in model.agents if isinstance(a, HouseholdAgent)]
    if not households:
        return np.nan
    owners = sum(1 for h in households if len(h.owned_properties) > 0)
    return owners / len(households)


def institutional_ownership_share(model):
    """Fraction of total housing stock owned by institutions."""
    total = len(model.properties)
    if total == 0:
        return np.nan
    inst_owned = sum(len(a.portfolio) for a in model.agents if isinstance(a, InstitutionalAgent))
    return inst_owned / total


def household_ownership_share(model):
    """Fraction of total housing stock owned by households."""
    total = len(model.properties)
    if total == 0:
        return np.nan
    hh_owned = sum(len(a.owned_properties) for a in model.agents if isinstance(a, HouseholdAgent))
    return hh_owned / total


def avg_rent(model):
    """Mean monthly rent across currently rented properties."""
    rents = [
        p.current_rent
        for p in model.properties
        if p.occupant_id is not None
        and p.owner_id is not None
        and p.occupant_id != p.owner_id
        and p.current_rent is not None
    ]
    if not rents:
        return np.nan
    return sum(rents) / len(rents)


def rental_transaction_volume(model):
    """Number of rental transactions this step."""
    return len(model.this_step_rental_transactions)


def household_marginal_pricer_share(model):
    """Share of this step's ownership transactions won by households."""
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    hh_wins = sum(1 for t in txns if t.buyer_type == "household")
    return hh_wins / len(txns)


def total_household_net_worth(model):
    """Sum of household net worth."""
    return sum(a.net_worth for a in model.agents if isinstance(a, HouseholdAgent))


def price_to_rent_ratio(model):
    """Average estimated value / average current rent."""
    rents = [
        p.current_rent
        for p in model.properties
        if p.current_rent is not None and p.current_rent > 0
    ]
    values = [
        p.estimated_value
        for p in model.properties
        if p.estimated_value is not None and p.estimated_value > 0
    ]
    if not rents or not values:
        return np.nan
    avg_value = sum(values) / len(values)
    avg_rent = sum(rents) / len(rents)
    if avg_rent <= 0:
        return np.nan
    return avg_value / avg_rent


def avg_loan_to_value(model):
    """Average LTV of newly purchased properties this step."""
    txns = getattr(model, "this_step_transactions", [])
    if not txns:
        return np.nan
    ltvs = []
    for t in txns:
        buyer = model._agent_map.get(t.buyer_id)
        if buyer is None:
            continue
        mortgage = getattr(buyer, "_mortgages", {}).get(t.property_id)
        if mortgage is not None and t.price > 0:
            ltvs.append(mortgage[1])
    if not ltvs:
        return np.nan
    return sum(ltvs) / len(ltvs)


def vacancy_rate(model):
    """Fraction of housing stock with no occupant."""
    total = len(model.properties)
    if total == 0:
        return np.nan
    vacant = sum(1 for p in model.properties if p.occupant_id is None)
    return vacant / total


MODEL_REPORTERS = {
    "avg_sale_price": avg_sale_price,
    "transaction_volume": transaction_volume,
    "ownership_rate": ownership_rate,
    "institutional_ownership_share": institutional_ownership_share,
    "household_ownership_share": household_ownership_share,
    "avg_rent": avg_rent,
    "rental_transaction_volume": rental_transaction_volume,
    "household_marginal_pricer_share": household_marginal_pricer_share,
    "total_household_net_worth": total_household_net_worth,
    "price_to_rent_ratio": price_to_rent_ratio,
    "avg_loan_to_value": avg_loan_to_value,
    "vacancy_rate": vacancy_rate,
}
