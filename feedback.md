# Feedback on Project

Conceptually, this is strong project. Framing cycles in terms of shifts in MPI is well aligned with both empirical housing research and recent ABM work that couples mortgage, buy‑to‑let, and rental markets. The focus on structural heterogeneity (utility, financing, constraints, information) rather than ad‑hoc heuristics is a nice contribution. Scope and some core mechanisms, however, need sharpening.

## Emergent phenomenon.
You implicitly target: (i) price and rent cycles, (ii) shifts in marginal pricer type over the cycle, and (iii) changing tenure composition when institutions displace owner‑occupiers. Make those explicit as the emergent phenomena you want to reproduce, not just “housing‑market dynamics” in general.

## Bounded rationality.
You mention heterogeneous information sets and bounded local consideration sets; that is a genuine bounded‑rational element. Be explicit that agents use myopic or adaptive expectations (e.g. extrapolating recent rents/prices), have limited search over a subset of properties, and update rules that do not compute full intertemporal expected utility. Right now the description reads almost as fully rational expected‑profit maximisation.

## Game theory – implicit, could be clearer.
You correctly frame the market as an aggregative strategic interaction: each agent’s payoff depends on the common price, which itself is determined by all bids. To visibly satisfy the “game theory” requirement, I’d:

- Describe the double‑auction / clearing mechanism as a repeated game where each class plays a best‑response bidding strategy to others’ bids; or
- Add a simple 2×2 pricing/entry game between (say) landlords and institutions on a local segment (high/low leverage vs patient/impatient capital).

Right now “marginal pricer” gives the right intuition but not an explicit game structure.

## Risk aversion.
You state that classes differ in risk preferences, but you do not say how this enters behaviour. Spell out that e.g. owner‑occupiers have high CRRA risk aversion and therefore: (i) cap leverage more aggressively, (ii) shade bids more under uncertainty, or (iii) exit earlier when volatility rises. Institutions can be less risk‑averse and focus on portfolio variance vs yield. Without a concrete parameter entering the bid rule, “risk preference” is just a label.

- Price and rent formation. You say prices and rents emerge, but the clearing mechanism isn’t specified. Will you use a double auction, market maker, or simple excess‑demand price adjustment? This choice strongly affects cycle amplitude and who becomes marginal pricer. It should be explicit and probably as simple as possible.
- Credit constraints as key driver. Since your core hypothesis is about credit conditions shifting marginal pricers, be specific about how credit enters: maximum LTVs, debt‑service‑to‑income caps, class‑specific funding spreads, etc. Existing ABM work shows these details matter for the sign and size of shocks.
- Scope. Four agent classes, spatial zones, two linked markets, and calibration “to real‑world data” is ambitious. You might want a clean baseline with three classes (owner‑occupier, small landlord, institution) and stylised credit regimes, then treat full calibration and rental detail as later extensions.

## Suggested Literature

- Bezemer & co‑authors, “Roof or real estate? An agent‑based model of housing affordability” – directly parallels your multi‑class setup.
- Recent JASSS housing ABM with mortgage/buy‑to‑let interactions.
- BoE ABM of heterogeneous housing markets and macroprudential policy.
- Empirical and theoretical work on marginal buyers/pricers in housing demand
