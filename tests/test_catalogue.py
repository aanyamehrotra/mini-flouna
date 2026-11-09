"""Check catalogue integrity and useful recommendation scenarios."""

import json

import pytest
from pydantic import ValidationError

from app.catalogue import load_catalogue


def test_mock_catalogue_has_varied_valid_options():
    options = load_catalogue()
    assert len(options) == 30
    assert len({option.id for option in options}) == 30
    assert {option.dietary_type for option in options} == {
        "vegetarian", "vegan", "non_vegetarian",
    }
    assert len({option.location for option in options}) == 3


def test_catalogue_supports_price_rating_and_boundary_scenarios():
    options = {option.id: option for option in load_catalogue()}
    cheap, highly_rated = options["food_001"], options["food_002"]
    assert cheap.price_inr < highly_rated.price_inr < 300
    assert cheap.rating < highly_rated.rating
    assert highly_rated.delivery_minutes <= 30
    assert options["food_003"].price_inr == 300
    assert options["food_004"].delivery_minutes > 30
    assert options["food_005"].price_inr > 300


def test_duplicate_ids_are_rejected(tmp_path):
    record = load_catalogue()[0].model_dump()
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_catalogue(path)


def test_empty_catalogue_is_rejected(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_catalogue(path)


def test_invalid_record_is_rejected(tmp_path):
    record = load_catalogue()[0].model_dump()
    record["rating"] = 6
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_catalogue(path)
