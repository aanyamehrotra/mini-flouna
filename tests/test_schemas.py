"""Check input boundaries and prevent unsupported structured values."""

import pytest
from pydantic import ValidationError

from app.schemas import Budget, DeliveryLimit, Preferences, UnderstoodRequest, UserRequest


def test_missing_preferences_remain_unknown():
    preferences = Preferences().model_dump()
    assert preferences.pop("unsupported_constraints") == []
    assert all(value is None for value in preferences.values())


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_invalid_request_text_is_rejected(text):
    with pytest.raises(ValidationError):
        UserRequest(text=text)


def test_request_text_is_trimmed():
    assert UserRequest(text="  Find biryani  ").text == "Find biryani"


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(amount):
    with pytest.raises(ValidationError):
        Budget(amount=amount, operator="lt")


def test_strict_and_inclusive_bounds_are_preserved():
    assert Budget(amount=300, operator="lt").operator == "lt"
    assert Budget(amount=300, operator="lte").operator == "lte"
    assert DeliveryLimit(minutes=30, operator="lte").minutes == 30


@pytest.mark.parametrize("minutes", [0, -1, 2.5, True])
def test_invalid_delivery_duration_is_rejected(minutes):
    with pytest.raises(ValidationError):
        DeliveryLimit(minutes=minutes, operator="lte")


@pytest.mark.parametrize("rating", [-0.1, 5.1, float("nan")])
def test_invalid_rating_is_rejected(rating):
    with pytest.raises(ValidationError):
        Preferences(min_rating=rating)


def test_qualitative_rating_does_not_invent_a_numeric_threshold():
    preferences = Preferences(prefer_high_rating=True)
    assert preferences.min_rating is None


def test_unknown_fields_and_intents_are_rejected():
    with pytest.raises(ValidationError):
        Preferences(organic=True)
    with pytest.raises(ValidationError):
        UnderstoodRequest(intent="purchase", preferences=Preferences())


def test_unsupported_constraint_survives_json_round_trip():
    request = UnderstoodRequest(
        intent="discovery",
        preferences=Preferences(unsupported_constraints=["organic"]),
    )
    assert UnderstoodRequest.model_validate_json(request.model_dump_json()) == request
