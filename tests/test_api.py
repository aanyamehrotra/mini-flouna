"""Exercise the HTTP contract with a real, locally trained classifier."""

import pytest
from fastapi.testclient import TestClient

from app.dataset import load_requests, load_splits
from app.intent import IntentClassifier
from app.main import app, get_classifier
from app.schemas import RecommendationResponse
from scripts.train import make_model


@pytest.fixture(scope="module")
def classifier():
    examples = load_requests()
    splits = load_splits(examples)
    rows = [row for row in examples if row.id in splits.train]
    model = make_model(4.0)
    model.fit([row.text for row in rows], [row.intent for row in rows])
    return IntentClassifier(model, min_probability=0.25)


@pytest.fixture
def client(classifier):
    app.dependency_overrides[get_classifier] = lambda: classifier
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.parametrize("text,status,intent", [
    ("Show vegetarian pizza", "options_found", "discovery"),
    ("Recommend vegetarian pizza under ₹300", "recommended", "decision"),
    ("How do I place an order?", "information", "information"),
    ("Please order a pizza", "unsupported_action", "action"),
    ("Help me", "clarification", "clarification"),
    ("Recommend vegetarian biryani under ₹50", "no_match", "decision"),
    ("Recommend organic pizza", "unsupported_constraint", "decision"),
])
def test_all_domain_outcomes_are_valid_structured_responses(client, text, status, intent):
    response = client.post("/recommend", json={"text": text})
    assert response.status_code == 200
    result = RecommendationResponse.model_validate(response.json())
    assert result.status == status
    assert result.understood_request.intent == intent
    assert result.intent_details is not None
    if status != "recommended":
        assert result.recommendation is None


def test_recommendation_respects_constraints_and_explains_actual_data(client):
    response = client.post("/recommend", json={
        "text": "Recommend highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"
    }).json()
    assert response["recommendation"]["id"] == "food_002"
    assert all(item["price_inr"] < 300 and item["delivery_minutes"] <= 30 for item in response["options"])
    assert "₹270" in response["explanation"]
    assert "4.7/5" in response["explanation"]


@pytest.mark.parametrize("payload", [
    {}, {"text": ""}, {"text": "   "}, {"text": None},
    {"text": 123}, {"text": ["pizza"]}, {"text": "x" * 2001},
    {"text": "Show pizza", "intent": "action"},
])
def test_invalid_payloads_are_rejected(client, payload):
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_malformed_json_is_rejected(client):
    response = client.post("/recommend", content='{ "text":', headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_missing_model_returns_setup_instruction(monkeypatch):
    def missing_model():
        raise FileNotFoundError("Train the local intent model first: python -m scripts.train")

    monkeypatch.setattr("app.main.load_classifier", missing_model)
    with TestClient(app) as client:
        response = client.post("/recommend", json={"text": "Show pizza"})
        assert response.status_code == 503
        assert "python -m scripts.train" in response.json()["detail"]
        assert client.get("/health").json() == {"status": "ok"}


def test_docs_expose_examples_and_schemas(client):
    assert client.get("/").url.path == "/docs"
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/recommend"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]
    assert set(examples) == {"discovery", "decision", "information", "action", "clarification", "no_match", "unsupported"}
    assert "RecommendationResponse" in schema["components"]["schemas"]
    assert {"200", "422", "503"} <= set(operation["responses"])


def test_repeated_http_requests_are_reproducible(client):
    request = {"text": "Recommend vegetarian pizza"}
    assert client.post("/recommend", json=request).json() == client.post("/recommend", json=request).json()
