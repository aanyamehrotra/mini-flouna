"""Filter food options, rank matches, and explain the recommendation."""

from collections import Counter

from app.schemas import FoodOption, Preferences, RankedOption, RecommendationResponse, UnderstoodRequest


def rejection_reasons(option: FoodOption, preferences: Preferences) -> list[str]:
    reasons = []
    if preferences.dish and preferences.dish.casefold() not in option.dish_name.casefold():
        reasons.append("dish")
    if preferences.cuisine and preferences.cuisine.casefold() != option.cuisine.casefold():
        reasons.append("cuisine")
    if preferences.location and preferences.location.casefold() != option.location.casefold():
        reasons.append("location")
    diet = preferences.dietary_preference
    if diet and not (
        option.dietary_type == diet
        or diet == "vegetarian" and option.dietary_type == "vegan"
    ):
        reasons.append("dietary preference")
    budget = preferences.budget
    if budget:
        within_budget = (
            option.price_inr < budget.amount
            if budget.operator == "lt" else option.price_inr <= budget.amount
        )
        if not within_budget:
            reasons.append("budget")

    delivery = preferences.delivery_limit
    if delivery:
        within_time = (
            option.delivery_minutes < delivery.minutes
            if delivery.operator == "lt" else option.delivery_minutes <= delivery.minutes
        )
        if not within_time:
            reasons.append("delivery time")
    if preferences.min_rating is not None and option.rating < preferences.min_rating:
        reasons.append("minimum restaurant rating")
    return reasons


def score_option(option: FoodOption, preferences: Preferences) -> RankedOption:
    # Fixed ranges keep an option's score independent of other catalogue entries.
    weights = {
        "price": (0.0, 0.0, 1.0),
        "delivery": (0.0, 1.0, 0.0),
        "rating": (1.0, 0.0, 0.0),
    }.get(preferences.priority)
    if weights is None:
        weights = (0.80, 0.15, 0.05) if preferences.prefer_high_rating else (0.60, 0.25, 0.15)
    components = (
        weights[0] * option.rating / 5,
        weights[1] * max(0, 1 - option.delivery_minutes / 60),
        weights[2] * max(0, 1 - option.price_inr / 500),
    )
    return RankedOption(
        option_id=option.id,
        score=round(sum(components), 8),
        rating_component=round(components[0], 8),
        delivery_component=round(components[1], 8),
        price_component=round(components[2], 8),
    )


def recommend(request: UnderstoodRequest, options: list[FoodOption]) -> RecommendationResponse:
    preferences = request.preferences
    if request.intent not in {"discovery", "decision"}:
        raise ValueError("This baseline supports Discovery and Decision requests only.")
    if preferences.unsupported_constraints:
        constraints = ", ".join(preferences.unsupported_constraints)
        return RecommendationResponse(
            status="unsupported_constraint", understood_request=request,
            explanation=f"The mock catalogue cannot verify or support: {constraints}. No option is recommended.",
            clarification_question="Would you like to revise these requirements?",
        )
    if not any((preferences.domain, preferences.dish, preferences.cuisine, preferences.dietary_preference)):
        return RecommendationResponse(
            status="clarification", understood_request=request,
            explanation="There is not enough information to identify a supported food request.",
            clarification_question="What food or cuisine would you like?",
        )

    eligible = []
    rejected = Counter()
    for option in options:
        reasons = rejection_reasons(option, preferences)
        if reasons:
            rejected.update(reasons)
        else:
            eligible.append(option)
    if not eligible:
        summary = "; ".join(f"{label}: {count} excluded" for label, count in sorted(rejected.items()))
        return RecommendationResponse(
            status="no_match", understood_request=request,
            explanation="No catalogue option satisfies every explicit constraint. "
                        + (f"Counts can overlap: {summary}." if summary else "The catalogue is empty."),
            clarification_question="Which requirement, if any, would you like to change?",
        )

    scores = {option.id: score_option(option, preferences) for option in eligible}
    # Unclipped values preserve cheapest/fastest ordering beyond reference ranges.
    def sort_key(option: FoodOption) -> tuple:
        primary = {
            "price": option.price_inr,
            "delivery": option.delivery_minutes,
            "rating": -option.rating,
        }.get(preferences.priority, -scores[option.id].score)
        return primary, option.id

    eligible.sort(key=sort_key)
    ranking = [scores[option.id] for option in eligible]
    warnings = []
    if preferences.location is None:
        warnings.append("No delivery locality was provided; results span mock locality scenarios.")
    if request.intent == "discovery":
        return RecommendationResponse(
            status="options_found", understood_request=request, options=eligible, ranking=ranking,
            explanation=f"Found {len(eligible)} options satisfying all extracted constraints, ordered by the documented ranking policy.",
            warnings=warnings,
        )

    winner = eligible[0]
    facts = [f"{winner.dish_name} from {winner.restaurant_name} in {winner.location}"]
    if preferences.dietary_preference:
        facts.append(f"matches your {preferences.dietary_preference.replace('_', ' ')} requirement")
    if preferences.budget:
        comparison = "below" if preferences.budget.operator == "lt" else "at most"
        facts.append(f"costs ₹{winner.price_inr:g}, {comparison} your ₹{preferences.budget.amount:g} limit")
    else:
        facts.append(f"costs ₹{winner.price_inr:g}")
    facts.append(f"has a mock restaurant rating of {winner.rating:g}/5")
    facts.append(f"has an estimated delivery time of {winner.delivery_minutes} minutes")
    if preferences.delivery_limit:
        comparison = "below" if preferences.delivery_limit.operator == "lt" else "within"
        facts.append(f"is {comparison} your {preferences.delivery_limit.minutes}-minute delivery limit")
    if preferences.min_rating is not None:
        facts.append(f"meets your minimum restaurant rating of {preferences.min_rating:g}")
    policy = f"explicit {preferences.priority} priority" if preferences.priority else "documented weighted scoring policy"
    explanation = "; ".join(facts) + f". Ranked first among {len(eligible)} eligible options using the {policy}."
    if len(eligible) > 1:
        runner_up = eligible[1]
        explanation += (
            f" Runner-up: {runner_up.dish_name} from {runner_up.restaurant_name}"
            f" in {runner_up.location} (₹{runner_up.price_inr:g}, restaurant rating"
            f" {runner_up.rating:g}/5, estimated {runner_up.delivery_minutes} minutes)."
        )
    return RecommendationResponse(
        status="recommended", understood_request=request, recommendation=winner,
        options=eligible, ranking=ranking, explanation=explanation, warnings=warnings,
    )
