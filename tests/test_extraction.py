"""Check extraction against hand-written expected preferences."""

import pytest

from app.extraction import extract_preferences


def test_assignment_example_extracts_only_explicit_values():
    result = extract_preferences(
        "I want a highly rated vegetarian biryani under ₹300 that can arrive within 30 minutes."
    )
    p = result.preferences
    assert result.questions == []
    assert p.dish == "biryani"
    assert p.dietary_preference == "vegetarian"
    assert p.budget.model_dump() == {"amount": 300, "operator": "lt", "currency": "INR"}
    assert p.delivery_limit.model_dump() == {"minutes": 30, "operator": "lte"}
    assert p.prefer_high_rating is True
    assert p.min_rating is None
    assert p.location is None


@pytest.mark.parametrize("phrase,operator", [
    ("under Rs. 300", "lt"), ("below INR 300", "lt"),
    ("up to 300 rupees", "lte"), ("₹300 or less", "lte"),
    ("no more than ₹300", "lte"),
])
def test_budget_bound_variants(phrase, operator):
    result = extract_preferences(f"Find biryani {phrase}")
    assert not result.questions
    assert result.preferences.budget.amount == 300
    assert result.preferences.budget.operator == operator


def test_delivery_is_not_mistaken_for_budget():
    result = extract_preferences("Find biryani under 30 minutes")
    assert result.preferences.budget is None
    assert result.preferences.delivery_limit.minutes == 30
    assert result.preferences.delivery_limit.operator == "lt"


def test_hour_conversion():
    result = extract_preferences("Find pizza within 0.5 hours")
    assert result.preferences.delivery_limit.minutes == 30


def test_non_vegetarian_does_not_match_vegetarian():
    result = extract_preferences("non-vegetarian biryani in Indiranagar")
    assert not result.questions
    assert result.preferences.dietary_preference == "non_vegetarian"


def test_veg_alias_preserves_explicit_diet():
    result = extract_preferences("veg biryani")
    assert result.preferences.dietary_preference == "vegetarian"
    assert result.preferences.dish == "biryani"


def test_specific_dish_is_not_reduced_to_generic_dish():
    result = extract_preferences("Find chicken biryani")
    assert result.preferences.dish == "chicken biryani"
    assert result.preferences.dietary_preference is None


def test_unknown_locality_is_preserved():
    assert extract_preferences("Find biryani in Delhi under ₹300").preferences.location == "delhi"


def test_missing_budget_remains_null():
    result = extract_preferences("Show vegetarian pizzas")
    assert not result.questions
    assert result.preferences.budget is None
    assert result.preferences.dish == "pizza"


def test_thousands_separator_is_not_truncated():
    assert extract_preferences("pizza under ₹1,000").preferences.budget.amount == 1000


def test_seconds_are_not_mistaken_for_budget():
    result = extract_preferences("pizza under 30 seconds")
    assert result.preferences.budget is None
    assert result.preferences.unsupported_constraints


@pytest.mark.parametrize("text", [
    "biryani under ₹0", "pizza under ₹200 or under ₹300",
    "pizza budget 300", "pizza within 0 minutes", "pizza rated 6",
    "pizza and biryani", "vegan non-vegetarian pizza",
    "pizza in Indiranagar or Whitefield", "fastest cheapest pizza",
])
def test_conflicting_or_ambiguous_preferences_need_clarification(text):
    assert extract_preferences(text).questions


@pytest.mark.parametrize("text", [
    "organic biryani", "gluten-free pizza", "not spicy noodles",
    "pizza without onions", "not vegetarian biryani", "pizza under $10",
    "pizza near me", "biryani for 3 people", "pizza no onions",
    "I don't want vegetarian pizza",
    "pizza under 500 calories",
])
def test_unsupported_requirements_are_retained(text):
    assert extract_preferences(text).preferences.unsupported_constraints


def test_minimum_numeric_rating():
    result = extract_preferences("Find biryani rated at least 4.5")
    assert result.preferences.min_rating == 4.5
    assert not result.questions
