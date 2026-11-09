"""Verify metric definitions with independent toy cases, not test-set targets."""

import hashlib

import pytest

from app.dataset import LabelledRequest
from app.schemas import Preferences
from scripts.evaluate import check_data_hashes, normalize_field, preference_metrics
from scripts.train import intent_metrics


def annotated(preferences):
    return LabelledRequest(id="toy", group_id="toy", text="Choose pizza", intent="decision",
                           tags=["toy"], rationale="Synthetic metric check", expected_preferences=preferences)


def test_false_action_rate_uses_actual_non_action_denominator():
    result = intent_metrics(
        ["action", "decision", "information", "clarification"],
        ["action", "action", "information", "information"],
    )
    assert result["false_action_count"] == 1
    assert result["false_action_rate"] == pytest.approx(1 / 3)
    assert result["classification_report"]["action"]["precision"] == .5
    assert result["classification_report"]["action"]["recall"] == 1
    assert sum(sum(row) for row in result["confusion_matrix"]) == 4


def test_missing_fields_do_not_hide_present_field_errors():
    gold = Preferences(domain="food", dish="pizza")
    result = preference_metrics([(annotated(gold), Preferences(domain="food"))])
    assert result["overall"]["present_total"] == 2
    assert result["overall"]["present_accuracy"] == .5
    assert result["overall"]["missing_accuracy"] == 1
    assert result["exact_request_accuracy"] == 0
    assert result["errors"][0]["differences"] == {"dish": {"expected": "pizza", "predicted": None}}


def test_hallucinated_optional_preference_is_counted_as_an_error():
    gold = Preferences(domain="food", dish="pizza")
    prediction = Preferences(domain="food", dish="pizza", dietary_preference="vegetarian")
    result = preference_metrics([(annotated(gold), prediction)])
    assert result["overall"]["present_accuracy"] == 1
    assert result["overall"]["missing_accuracy"] < 1
    assert result["exact_requests"] == 0


def test_unsupported_categories_ignore_wording_but_not_extra_categories():
    gold = Preferences(unsupported_constraints=["spice level"])
    predicted = Preferences(unsupported_constraints=["spice level: not spicy", "spice level: mild"])
    assert preference_metrics([(annotated(gold), predicted)])["exact_requests"] == 1
    predicted.unsupported_constraints.append("food exclusions: onions")
    assert preference_metrics([(annotated(gold), predicted)])["exact_requests"] == 0


def test_case_and_whitespace_normalization_is_limited_to_text():
    assert normalize_field("location", "  Whitefield  ") == "whitefield"
    assert normalize_field("prefer_high_rating", False) is False


def test_empty_annotation_set_has_no_invented_accuracy():
    result = preference_metrics([])
    assert result["exact_request_accuracy"] is None
    assert result["overall"]["accuracy"] is None


def test_data_changes_cannot_silently_reuse_frozen_model_metadata(tmp_path):
    files = {"dataset_sha256": "requests.jsonl", "splits_sha256": "splits.json"}
    metadata = {}
    for key, filename in files.items():
        path = tmp_path / filename
        path.write_text("original", encoding="utf-8")
        metadata[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    check_data_hashes(metadata, tmp_path)
    (tmp_path / "requests.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="same frozen dataset"):
        check_data_hashes(metadata, tmp_path)
