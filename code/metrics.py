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


def owner_occupier_ownership_share(model):
    """Fraction of total housing stock owned by owner-occupier households."""
    total = len(model.properties)
    if total == 0:
        return np.nan
    count = 0
    for p in model.properties:
        owner = model._agent_map.get(p.owner_id)
        if isinstance(owner, HouseholdAgent) and owner.home_property == p.id:
            count += 1
    return count / total


def landlord_ownership_share(model):
    """Fraction of total housing stock owned by landlord households."""
    total = len(model.properties)
    if total == 0:
        return np.nan
    count = 0
    for p in model.properties:
        owner = model._agent_map.get(p.owner_id)
        if isinstance(owner, HouseholdAgent) and owner.home_property != p.id:
            count += 1
    return count / total


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


def _classify_buyer(txn, model):
    buyer = model._agent_map[txn.buyer_id]
    if isinstance(buyer, InstitutionalAgent):
        return "institution"
    return "owner-occupier" if buyer.home_property == txn.property_id else "landlord"


def owner_occupier_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    return sum(1 for t in txns if _classify_buyer(t, model) == "owner-occupier") / len(txns)


def landlord_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    return sum(1 for t in txns if _classify_buyer(t, model) == "landlord") / len(txns)


def institution_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    return sum(1 for t in txns if _classify_buyer(t, model) == "institution") / len(txns)


def owner_occupier_value_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    total = sum(t.price for t in txns)
    if total <= 0:
        return np.nan
    return sum(t.price for t in txns if _classify_buyer(t, model) == "owner-occupier") / total


def landlord_value_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    total = sum(t.price for t in txns)
    if total <= 0:
        return np.nan
    return sum(t.price for t in txns if _classify_buyer(t, model) == "landlord") / total


def institution_value_share(model):
    txns = model.this_step_transactions
    if not txns:
        return np.nan
    total = sum(t.price for t in txns)
    if total <= 0:
        return np.nan
    return sum(t.price for t in txns if _classify_buyer(t, model) == "institution") / total


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


def collect_zone_metrics(model) -> list[dict]:
    rows = []
    for zone in range(model.n_zones):
        props = [p for p in model.properties if p.zone == zone]
        if not props:
            continue
        avg_est_val = float(sum(p.estimated_value for p in props) / len(props))
        ownership_rate_val = float(
            sum(1 for p in props if p.owner_id is not None) / len(props)
        )
        owned = [p for p in props if p.owner_id is not None]
        inst_share = (
            float(
                sum(
                    1
                    for p in owned
                    if isinstance(model._agent_map.get(p.owner_id), InstitutionalAgent)
                )
                / len(owned)
            )
            if owned
            else 0.0
        )
        txns = [
            t
            for t in model.this_step_transactions
            if model._property_map[t.property_id].zone == zone
        ]
        txn_vol = len(txns)
        avg_txn_price = (
            float(sum(t.price for t in txns) / len(txns)) if txns else float("nan")
        )
        rows.append(
            {
                "step": int(model.steps),
                "zone": zone,
                "avg_estimated_value": avg_est_val,
                "ownership_rate": ownership_rate_val,
                "institutional_ownership_share": inst_share,
                "transaction_volume": txn_vol,
                "avg_transaction_price": avg_txn_price,
            }
        )
    return rows


MODEL_REPORTERS = {
    "avg_sale_price": avg_sale_price,
    "transaction_volume": transaction_volume,
    "ownership_rate": ownership_rate,
    "institutional_ownership_share": institutional_ownership_share,
    "household_ownership_share": household_ownership_share,
    "owner_occupier_ownership_share": owner_occupier_ownership_share,
    "landlord_ownership_share": landlord_ownership_share,
    "avg_rent": avg_rent,
    "rental_transaction_volume": rental_transaction_volume,
    "owner_occupier_share": owner_occupier_share,
    "landlord_share": landlord_share,
    "institution_share": institution_share,
    "owner_occupier_value_share": owner_occupier_value_share,
    "landlord_value_share": landlord_value_share,
    "institution_value_share": institution_value_share,
    "total_household_net_worth": total_household_net_worth,
    "price_to_rent_ratio": price_to_rent_ratio,
    "avg_loan_to_value": avg_loan_to_value,
    "vacancy_rate": vacancy_rate,
}
