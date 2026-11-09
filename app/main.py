"""Thin HTTP interface over the existing request-processing pipeline."""

from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.intent import IntentClassifier, load_classifier
from app.pipeline import process_request
from app.schemas import RecommendationResponse, UserRequest


app = FastAPI(
    title="Mini FLOUNA",
    version="0.1.0",
    description="Understand English food requests and explain recommendations from a fictional catalogue. No transactions are executed.",
)

EXAMPLES = {
    "discovery": {"summary": "Find available options", "value": {"text": "Show vegetarian pizza"}},
    "decision": {"summary": "Choose with multiple constraints", "value": {
        "text": "Recommend highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"
    }},
    "information": {"summary": "Ask about ordering capabilities", "value": {"text": "How do I place an order?"}},
    "action": {"summary": "Identify an unsupported action", "value": {"text": "Please order a pizza"}},
    "clarification": {"summary": "Ask for missing context", "value": {"text": "Help me"}},
    "no_match": {"summary": "No option meets the budget", "value": {"text": "Recommend vegetarian biryani under ₹50"}},
    "unsupported": {"summary": "Cannot verify organic status", "value": {"text": "Recommend organic pizza"}},
}


def get_classifier() -> IntentClassifier:
    try:
        return load_classifier()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Check that the API process is running")
def health() -> dict[str, str]:
    """Liveness only; this does not check whether the local model is trained."""
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Understand a request and return its appropriate response",
    responses={503: {"description": "Local model is missing; run python -m scripts.train."}},
)
def recommend_request(
    payload: Annotated[UserRequest, Body(openapi_examples=EXAMPLES)],
    classifier: Annotated[IntentClassifier, Depends(get_classifier)],
) -> RecommendationResponse:
    """All valid domain outcomes use HTTP 200; inspect `status` in the JSON body."""
    return process_request(payload.text, classifier=classifier)
