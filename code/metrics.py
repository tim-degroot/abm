"""
Metrics and data collection.

Defines the reporter functions passed to Mesa's DataCollector.
All functions take the model as their argument.

Primary outputs required by the research question:
  - average sale price
  - transaction volume
  - ownership rate
  - institutional ownership share
  - average bid / average winning bid
  - marginal-pricer identity (household vs institution)
"""


def avg_sale_price(model):
    """Mean transaction price this step. NaN if no transactions."""
    txns = model.this_step_transactions
    if not txns:
        return float("nan")
    return sum(t.price for t in txns) / len(txns)


def transaction_volume(model):
    """Number of ownership transactions this step."""
    return len(model.this_step_transactions)


def ownership_rate(model):
    """
    Fraction of households that own at least one property.
    """
    from agents import HouseholdAgent

    households = [a for a in model.agents if isinstance(a, HouseholdAgent)]
    if not households:
        return float("nan")
    owners = sum(1 for h in households if len(h.owned_properties) > 0)
    return owners / len(households)


def institutional_ownership_share(model):
    """
    Fraction of total housing stock owned by institutions.
    """
    from agents import InstitutionalAgent

    total = len(model.properties)
    if total == 0:
        return float("nan")
    inst_owned = sum(
        len(a.portfolio) for a in model.agents if isinstance(a, InstitutionalAgent)
    )
    return inst_owned / total


def household_ownership_share_of_stock(model):
    """
    Fraction of total housing stock owned by households (all household owners' portfolios).
    This should complement `institutional_ownership_share` when the model keeps
    all stock owned at every time step.
    """
    from agents import HouseholdAgent

    total = len(model.properties)
    if total == 0:
        return float("nan")
    hh_owned = 0
    for a in model.agents:
        if isinstance(a, HouseholdAgent):
            hh_owned += len(getattr(a, "owned_properties", []))
    return hh_owned / total


def avg_winning_bid(model):
    """Mean winning (highest) bid submitted this step."""
    txns = model.this_step_transactions
    if not txns:
        return float("nan")
    return sum(t.winning_bid for t in txns) / len(txns)


def household_marginal_pricer_share(model):
    """
    Share of this step's transactions where the winning bidder was a household.

    This is the primary marginal-pricer metric.
    A value near 1.0 indicates household-dominated pricing.
    A value near 0.0 indicates institution-dominated pricing.
    """
    txns = model.this_step_transactions
    if not txns:
        return float("nan")
    hh_wins = sum(1 for t in txns if t.buyer_type == "household")
    return hh_wins / len(txns)


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
        return float("nan")
    return sum(rents) / len(rents)


def rental_transaction_volume(model):
    """Number of rental transactions this step."""
    return len(model.this_step_rental_transactions)


def total_household_net_worth(model):
    """Sum of household net worth."""
    from agents import HouseholdAgent

    return sum(a.net_worth for a in model.agents if isinstance(a, HouseholdAgent))


def debug_rental_listed(model):
    return getattr(model, "_debug_counts", {}).get("rental_listed", 0)


def debug_ownership_listed(model):
    return getattr(model, "_debug_counts", {}).get("ownership_listed", 0)


def debug_rental_bids_submitted(model):
    return getattr(model, "_debug_counts", {}).get("rental_bids_submitted", 0)


def debug_ownership_bids_submitted(model):
    return getattr(model, "_debug_counts", {}).get("ownership_bids_submitted", 0)


def debug_ownership_bids_filtered(model):
    return getattr(model, "_debug_counts", {}).get("ownership_bids_filtered", 0)


def debug_avg_ownership_bid(model):
    samples = getattr(model, "_debug_counts", {}).get("ownership_bid_samples", [])
    if not samples:
        return float("nan")
    return float(sum(samples) / len(samples))


# Mapping passed to Mesa DataCollector
MODEL_REPORTERS = {
    "avg_sale_price": avg_sale_price,
    "transaction_volume": transaction_volume,
    "ownership_rate": ownership_rate,
    "institutional_ownership_share": institutional_ownership_share,
    "avg_winning_bid": avg_winning_bid,
    "household_marginal_pricer_share": household_marginal_pricer_share,
    "avg_rent": avg_rent,
    "rental_transaction_volume": rental_transaction_volume,
    "total_household_net_worth": total_household_net_worth,
    "debug_rental_listed": debug_rental_listed,
    "debug_ownership_listed": debug_ownership_listed,
    "debug_rental_bids_submitted": debug_rental_bids_submitted,
    "debug_ownership_bids_submitted": debug_ownership_bids_submitted,
    "debug_ownership_bids_filtered": debug_ownership_bids_filtered,
    "debug_avg_ownership_bid": debug_avg_ownership_bid,
    "household_ownership_share_of_stock": household_ownership_share_of_stock,
}
