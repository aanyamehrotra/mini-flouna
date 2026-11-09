"""Verify constraints, ranking and failure behaviour end to end."""

import pytest

from app.catalogue import load_catalogue
from app.pipeline import process_request
from app.recommendation import recommend, rejection_reasons, score_option
from app.schemas import Budget, DeliveryLimit, Preferences, UnderstoodRequest


def decide(text):
    return process_request(text, intent="decision")


def test_assignment_example_recommends_a_better_fit_than_cheapest():
    response = decide(
        "Highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"
    )
    assert response.status == "recommended"
    assert response.recommendation.id == "food_002"
    assert response.recommendation.price_inr > min(o.price_inr for o in response.options)
    assert all(not rejection_reasons(o, response.understood_request.preferences) for o in response.options)
    assert "₹270" in response.explanation
    assert "4.7/5" in response.explanation
    assert "24 minutes" in response.explanation
    assert "vegetarian requirement" in response.explanation


def test_exact_budget_boundary_is_excluded_only_for_strict_limit():
    text = "vegetarian biryani in Indiranagar "
    strict = decide(text + "under ₹300")
    inclusive = decide(text + "at most ₹300")
    assert "food_003" not in {o.id for o in strict.options}
    assert "food_003" in {o.id for o in inclusive.options}


def test_exact_delivery_boundary():
    text = "vegetarian biryani in Koramangala "
    strict = decide(text + "under 30 minutes")
    inclusive = decide(text + "within 30 minutes")
    assert "food_012" not in {o.id for o in strict.options}
    assert "food_012" in {o.id for o in inclusive.options}


def test_vegan_item_can_satisfy_vegetarian_but_not_reverse():
    options = {o.id: o for o in load_catalogue()}
    assert rejection_reasons(options["food_006"], Preferences(dietary_preference="vegetarian")) == []
    assert rejection_reasons(options["food_002"], Preferences(dietary_preference="vegan")) == ["dietary preference"]


def test_non_vegetarian_request_excludes_vegetarian_items():
    response = decide("non-vegetarian biryani in Indiranagar")
    assert {o.id for o in response.options} == {"food_007"}


def test_no_match_does_not_relax_constraints():
    response = decide("vegan biryani under ₹50 within 10 minutes")
    assert response.status == "no_match"
    assert response.recommendation is None
    assert response.options == []
    assert response.understood_request.preferences.budget.amount == 50
    assert "budget" in response.explanation
    assert response.clarification_question


@pytest.mark.parametrize("text", ["organic biryani", "pizza without onions", "not vegetarian biryani"])
def test_unsupported_requirements_block_recommendation(text):
    response = decide(text)
    assert response.status == "unsupported_constraint"
    assert response.recommendation is None
    assert response.options == []


def test_unknown_location_does_not_match_known_localities():
    response = decide("Find biryani in Delhi")
    assert response.status == "no_match"
    assert "location" in response.explanation


def test_missing_budget_does_not_block_valid_request():
    response = decide("vegetarian pizza")
    assert response.status == "recommended"
    assert response.understood_request.preferences.budget is None
    assert "your ₹" not in response.explanation
    assert response.warnings


@pytest.mark.parametrize("text", ["Get me something", "pizza budget 300", "pizza and biryani"])
def test_unclear_requests_ask_instead_of_guessing(text):
    response = decide(text)
    assert response.status == "clarification"
    assert response.recommendation is None
    assert response.clarification_question


def test_discovery_lists_options_without_selecting_winner():
    response = process_request("vegetarian pizza", intent="discovery")
    assert response.status == "options_found"
    assert response.recommendation is None
    assert response.options


@pytest.mark.parametrize("priority,field", [("cheapest", "price_inr"), ("fastest", "delivery_minutes")])
def test_explicit_priority_is_respected(priority, field):
    response = decide(f"{priority} vegetarian biryani")
    assert getattr(response.recommendation, field) == min(getattr(o, field) for o in response.options)


def test_highest_rated_priority_is_respected():
    response = decide("highest rated vegetarian biryani")
    assert response.recommendation.rating == max(o.rating for o in response.options)


def test_numeric_rating_is_a_hard_constraint():
    response = decide("biryani rated at least 4.7")
    assert response.options
    assert all(o.rating >= 4.7 for o in response.options)


def test_scores_and_tie_breaking_are_reproducible():
    request = UnderstoodRequest(intent="decision", preferences=Preferences(domain="food"))
    options = load_catalogue()
    first = recommend(request, options)
    second = recommend(request, list(reversed(options)))
    assert first == second
    for entry in first.ranking:
        assert entry.score == pytest.approx(
            entry.rating_component + entry.delivery_component + entry.price_component
        )


def test_equal_options_are_ordered_by_stable_id():
    option = load_catalogue()[0]
    duplicate = option.model_copy(update={"id": "aaa"})
    request = UnderstoodRequest(intent="decision", preferences=Preferences(domain="food"))
    assert recommend(request, [option, duplicate]).recommendation.id == "aaa"


def test_default_score_uses_documented_fixed_ranges():
    option = load_catalogue()[1]
    score = score_option(option, Preferences())
    expected = .60 * (4.7 / 5) + .25 * (1 - 24 / 60) + .15 * (1 - 270 / 500)
    assert score.score == pytest.approx(expected)


def test_all_constraints_are_checked_on_direct_structured_requests():
    p = Preferences(domain="food", dish="biryani", cuisine="Indian", location="Indiranagar",
                    dietary_preference="vegetarian", budget=Budget(amount=300, operator="lt"),
                    delivery_limit=DeliveryLimit(minutes=30, operator="lte"), min_rating=4.5)
    result = recommend(UnderstoodRequest(intent="decision", preferences=p), load_catalogue())
    assert {o.id for o in result.options} == {"food_002", "food_006"}


def test_baseline_does_not_pretend_to_handle_other_intents():
    with pytest.raises(ValueError, match="baseline"):
        process_request("Order biryani", intent="action")
