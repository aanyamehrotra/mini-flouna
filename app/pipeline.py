"""Baseline orchestration; callers supply intent until the classifier is added."""

from typing import Literal

from app.catalogue import load_catalogue
from app.extraction import extract_preferences
from app.recommendation import recommend
from app.schemas import RecommendationResponse, UnderstoodRequest


def process_request(
    text: str, *, intent: Literal["discovery", "decision"],
) -> RecommendationResponse:
    if intent not in {"discovery", "decision"}:
        raise ValueError("The baseline requires an explicit Discovery or Decision intent.")
    extracted = extract_preferences(text)
    request = UnderstoodRequest(intent=intent, preferences=extracted.preferences)
    if extracted.questions:
        return RecommendationResponse(
            status="clarification", understood_request=request,
            explanation="Some explicit preferences need clarification before options can be evaluated.",
            clarification_question=" ".join(extracted.questions),
        )
    return recommend(request, load_catalogue())
