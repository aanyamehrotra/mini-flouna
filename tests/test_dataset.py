"""Check label integrity and prevent evaluation leakage without scoring a model."""

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from app.dataset import INTENTS, load_requests, load_splits, validate_splits
from scripts.prepare_splits import make_splits


def test_dataset_size_labels_and_annotation_coverage():
    examples = load_requests()
    assert len(examples) == 150
    assert Counter(e.intent for e in examples) == {intent: 30 for intent in INTENTS}
    assert len({e.group_id for e in examples}) == 50
    annotated = [e for e in examples if e.expected_preferences is not None]
    assert len(annotated) == 60
    assert {e.intent for e in annotated} == {"discovery", "decision"}
    assert any("action_word_non_action" in e.tags and e.intent != "action" for e in examples)
    assert any("missing_context" in e.tags for e in examples)


def test_fixed_splits_are_balanced_and_reproducible():
    examples = load_requests()
    splits = load_splits(examples)
    assert splits == make_splits(list(reversed(examples)), seed=42)
    by_id = {e.id: e for e in examples}
    for name, per_class in (("train", 18), ("validation", 6), ("test", 6)):
        assert Counter(by_id[i].intent for i in getattr(splits, name)) == {
            intent: per_class for intent in INTENTS
        }


def test_duplicate_split_assignments_are_rejected():
    examples = load_requests()
    splits = load_splits(examples)
    splits.test.append(splits.train[0])
    with pytest.raises(ValueError, match="more than once"):
        validate_splits(examples, splits)


def test_missing_or_unknown_request_ids_are_rejected():
    examples = load_requests()
    splits = load_splits(examples)
    splits.test[0] = "unknown_request"
    with pytest.raises(ValueError, match="exactly"):
        validate_splits(examples, splits)


def test_paraphrase_leakage_is_rejected_even_if_counts_stay_balanced():
    examples = load_requests()
    splits = load_splits(examples)
    by_id = {e.id: e for e in examples}
    train_id = next(i for i in splits.train if by_id[i].intent == "discovery")
    test_id = next(i for i in splits.test if by_id[i].intent == "discovery")
    splits.train[splits.train.index(train_id)] = test_id
    splits.test[splits.test.index(test_id)] = train_id
    with pytest.raises(ValueError, match="paraphrases"):
        validate_splits(examples, splits)


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_text", "invalid_intent", "mixed_group"])
def test_invalid_dataset_rows_are_rejected(tmp_path, change):
    rows = [e.model_dump() for e in load_requests()[:2]]
    if change == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif change == "duplicate_text":
        rows[1]["text"] = "  " + rows[0]["text"].upper() + "  "
    elif change == "invalid_intent":
        rows[1]["intent"] = "purchase"
    else:
        rows[1]["intent"] = "action"
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises((ValueError, ValidationError)):
        load_requests(path)


def test_unsupported_currency_gold_does_not_invent_inr_budget():
    rows = [e for e in load_requests() if "unsupported_currency" in e.tags]
    assert len(rows) == 3
    assert all(e.expected_preferences.budget is None for e in rows)
    assert all(e.expected_preferences.unsupported_constraints == ["non-INR currency"] for e in rows)
