"""Classify, extract, route, and explain a single natural-language request."""

import re

from app.catalogue import load_catalogue
from app.extraction import extract_preferences
from app.information import answer_information
from app.intent import IntentClassifier, has_explicit_action_request, load_classifier
from app.recommendation import recommend
from app.schemas import Intent, Preferences, RecommendationResponse, UnderstoodRequest, UserRequest


def process_request(
    text: str, *, intent: Intent | None = None, classifier: IntentClassifier | None = None,
) -> RecommendationResponse:
    text = UserRequest(text=text).text
    details = None
    if intent is None:
        prediction = (classifier or load_classifier()).predict(text)
        intent, details = prediction.intent, prediction.details
    elif intent == "action" and not has_explicit_action_request(text):
        raise ValueError("Action requires an explicit execution request, even with a manual override.")
    extracted = extract_preferences(text)
    explicit_uncertainty = re.search(r"\b(?:undecided|unspecified|not told|not decided|cannot choose)\b", text.casefold())
    if intent == "information" and extracted.questions and explicit_uncertainty:
        intent = "clarification"
        if details:
            details.reason = "unresolved_preferences"
    if intent == "information":
        return RecommendationResponse(
            status="information", intent_details=details,
            understood_request=UnderstoodRequest(intent=intent, preferences=Preferences()),
            explanation=answer_information(text),
        )
    request = UnderstoodRequest(intent=intent, preferences=extracted.preferences)
    if intent == "action":
        return RecommendationResponse(
            status="unsupported_action", understood_request=request, intent_details=details,
            explanation="An explicit execution request was detected. This prototype cannot place orders, modify carts, reserve tables, or make payments. Nothing was executed.",
        )
    if extracted.questions or intent == "clarification":
        request.intent = "clarification"
        if details and extracted.questions:
            details.reason = "unresolved_preferences"
        question = " ".join(extracted.questions)
        if not question:
            question = (
                "Please restate the food request with the missing option or prior preferences included."
                if re.search(r"\b(?:that|other|earlier|before|previous|same)\b", text.casefold())
                else "What food would you like, and are there any required preferences I should use?"
            )
        return RecommendationResponse(
            status="clarification", understood_request=request, intent_details=details,
            explanation="The request needs clarification before options can be evaluated.",
            clarification_question=question,
        )
    response = recommend(request, load_catalogue())
    response.intent_details = details
    if response.status == "clarification":
        response.understood_request.intent = "clarification"
        if details:
            details.reason = "unresolved_preferences"
    return response
