"""Test authorization boundaries independently of statistical model accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.intent import IntentClassifier, has_explicit_action_request, load_classifier
from app.pipeline import process_request
from scripts.train import train_model


@pytest.mark.parametrize("text", [
    "Order a pizza", "Please place an order for biryani",
    "Could you buy one dosa for me?", "Confirm my order MOCK-1",
    "Cancel order MOCK-2", "Pay for mock order MOCK-3",
    "Add food_001 to my cart", "Remove food_002 from the basket",
    "Book a table for two", "Can you make a dinner reservation?",
])
def test_explicit_execution_requests_are_recognized(text):
    assert has_explicit_action_request(text)


@pytest.mark.parametrize("text", [
    "What should I order?", "How do I place an order?",
    "Can this app place orders?", "Which order would be best?",
    "Do not order anything; just show vegetarian pizza",
    "Don't place an order; compare these options",
    "Please do not buy food", "Could you not order anything?",
    "Explain how to cancel an order", "Show meals I could order",
    "Recommend pizza without placing an order",
    'What does the phrase "order a pizza" mean?',
])
def test_mentions_questions_quotes_and_prohibitions_are_not_authorization(text):
    assert not has_explicit_action_request(text)


class StubModel:
    classes_ = np.array(["action", "clarification", "decision", "discovery", "information"])

    def __init__(self, probabilities, known_text=True):
        self.probabilities = probabilities
        self.named_steps = {"tfidf": SimpleNamespace(
            transform=lambda texts: SimpleNamespace(nnz=int(known_text))
        )}

    def predict_proba(self, texts):
        return np.array([self.probabilities])


def test_action_prediction_requires_explicit_authorization():
    classifier = IntentClassifier(StubModel([.60, .05, .30, .03, .02]), .25)
    prediction = classifier.predict("Which meal would suit my preferences?")
    assert prediction.intent == "decision"
    assert prediction.details.raw_intent == "action"
    assert prediction.details.reason == "action_not_authorized"


def test_clear_execution_request_overrides_wrong_model_label():
    classifier = IntentClassifier(StubModel([.05, .05, .05, .05, .80]), .25)
    prediction = classifier.predict("Please order a pizza")
    assert prediction.intent == "action"
    assert prediction.details.raw_intent == "information"
    assert prediction.details.reason == "explicit_action_request"


def test_low_evidence_and_unknown_vocabulary_produce_clarification():
    uncertain = IntentClassifier(StubModel([.20, .20, .20, .20, .20]), .25)
    assert uncertain.predict("Something unclear").intent == "clarification"
    unknown = IntentClassifier(StubModel([.01, .01, .01, .01, .96], known_text=False), .25)
    assert unknown.predict("unknownvocabulary").intent == "clarification"


def test_explicit_listing_is_not_rewritten_as_a_choice_by_the_model():
    classifier = IntentClassifier(StubModel([.01, .01, .95, .02, .01]), .25)
    result = process_request("Show vegetarian pizza", classifier=classifier)
    assert result.status == "options_found"
    assert result.recommendation is None
    assert result.intent_details.reason == "explicit_non_action_request"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("intent-model")
    path = directory / "intent.joblib"
    metadata = train_model(path, directory / "validation.json")
    return load_classifier(path), metadata


def test_training_and_loading_preserve_a_real_classifier(trained):
    classifier, metadata = trained
    assert metadata["training_count"] == 90
    assert metadata["validation_count"] == 30
    assert metadata["test_evaluated"] is False
    assert len(metadata["candidates"]) == 9
    assert set(classifier.model.classes_) == {"action", "clarification", "decision", "discovery", "information"}
    assert classifier.predict("Please order biryani").intent == "action"


def test_tfidf_is_fitted_on_training_text_only(trained):
    from app.dataset import load_requests, load_splits

    classifier, _ = trained
    examples = load_requests()
    splits = load_splits(examples)
    vectorizer = classifier.model.named_steps["tfidf"]
    analyze = vectorizer.build_analyzer()
    training_terms = {term for e in examples if e.id in splits.train for term in analyze(e.text)}
    validation_terms = {term for e in examples if e.id in splits.validation for term in analyze(e.text)}
    assert validation_terms - training_terms
    assert not (validation_terms - training_terms) & set(vectorizer.vocabulary_)
    assert "explicit_action" not in vectorizer.vocabulary_


def test_missing_model_has_actionable_setup_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m scripts.train"):
        load_classifier(tmp_path / "absent.joblib")


@pytest.mark.parametrize("intent,text,status", [
    ("information", "Does the price include taxes?", "information"),
    ("discovery", "Show vegetarian pizza", "options_found"),
    ("decision", "Recommend vegetarian pizza", "recommended"),
    ("clarification", "That other one", "clarification"),
    ("action", "Please order a pizza", "unsupported_action"),
])
def test_all_intent_routes_have_distinct_behaviour(intent, text, status):
    result = process_request(text, intent=intent)
    assert result.status == status
    assert result.understood_request.intent == intent
    if intent not in {"decision", "discovery"}:
        assert result.recommendation is None
        assert result.options == []


def test_manual_action_override_cannot_authorize_a_question():
    with pytest.raises(ValueError, match="explicit execution"):
        process_request("What should I order?", intent="action")


def test_automatic_pipeline_uses_the_classifier(trained):
    classifier, _ = trained
    result = process_request("Please order biryani", classifier=classifier)
    assert result.status == "unsupported_action"
    assert result.intent_details.reason == "explicit_action_request"
    assert result.recommendation is None


def test_information_topics_do_not_become_user_preferences():
    result = process_request("Explain vegan versus vegetarian", intent="information")
    assert result.status == "information"
    assert result.understood_request.preferences.dietary_preference is None


def test_extraction_conflict_updates_final_intent_to_clarification():
    result = process_request("Recommend vegan non-vegetarian biryani", intent="decision")
    assert result.status == "clarification"
    assert result.understood_request.intent == "clarification"


def test_unspecified_required_diet_is_not_ignored():
    result = process_request("Recommend food for my special diet", intent="decision")
    assert result.status == "clarification"
    assert result.recommendation is None
    assert "dietary restrictions" in result.clarification_question


def test_unresolved_locality_is_not_treated_as_a_factual_question():
    result = process_request("Pizza in Indiranagar or Whitefield, undecided", intent="information")
    assert result.status == "clarification"


def test_ambiguous_unit_does_not_invent_a_budget():
    result = process_request("Food under 20; it might be time or money", intent="discovery")
    assert result.status == "clarification"
    assert result.understood_request.preferences.budget is None


@pytest.mark.parametrize("text", ["Pizza under $10", "Pizza under 10 dollars", "Pizza below USD 10"])
def test_foreign_currency_never_becomes_an_inr_budget(text):
    result = process_request(text, intent="decision")
    assert result.status == "unsupported_constraint"
    assert result.understood_request.preferences.budget is None
