"""Lightweight text classification with an explicit Action authorization gate."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from app.schemas import Intent, IntentDetails, UserRequest


DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "artifacts" / "intent_model.joblib"


def explicit_non_action_intent(text: str) -> Intent | None:
    """Respect clear list-versus-choice instructions despite vocabulary overlap."""
    text = text.casefold()
    if re.search(
        r"^(?:please\s+)?(?:recommend|choose|pick|help me choose)\b"
        r"|\bwhat\s+(?:\w+\s+){0,5}should i\s+(?:order|choose|pick)\b", text,
    ):
        return "decision"
    if not re.search(r"\bhow\b", text) and re.search(
        r"(?:^|[;.!?]|\bjust\b|\bonly\b)\s*(?:(?:please|can you|could you)\s+)*"
        r"(?:show|list|find(?!\s+out\b)|search)\b", text,
    ):
        return "discovery"
    return None


def has_explicit_action_request(text: str) -> bool:
    text = text.casefold().replace("’", "'")
    # Conservatively suppress execution intent when a prohibition is present.
    if re.search(
        r"\b(?:do not|don't|never|not to)\s+(?:place|order|buy|purchase|pay|confirm|cancel|book|reserve|check out)\b"
        r"|\bwithout\s+(?:doing|ordering|placing|buying|paying)\b"
        r"|\b(?:ordering|payment|checkout)\s+is\s+not\s+requested\b", text,
    ):
        return False
    prefix = r"^(?:(?:please|can you|could you|would you|go ahead and|i want you to|i would like you to)\s+)*"
    operations = (
        r"order\b", r"place\b.*\border\b", r"(?:buy|purchase)\b",
        r"(?:check out|checkout|proceed to checkout|complete checkout)\b",
        r"(?:cancel|confirm|repeat)\b.*\border\b", r"submit confirmation\b",
        r"pay\b", r"(?:make|complete)\b.*\bpayment\b", r"reorder\b",
        r"(?:add|put|remove|delete|take)\b.*\b(?:cart|basket)\b",
        r"(?:reserve|book)\b.*\btable\b", r"make\b.*\breservation\b",
    )
    for clause in re.split(r"[.!?;]|\bbut\b", text):
        if re.search(prefix + "(?:" + "|".join(operations) + ")", clause.strip()):
            return True
    return False


@dataclass
class IntentPrediction:
    intent: Intent
    details: IntentDetails


@dataclass
class IntentClassifier:
    model: Pipeline
    min_probability: float

    def predict(self, text: str) -> IntentPrediction:
        text = UserRequest(text=text).text
        probabilities = self.model.predict_proba([text])[0]
        ordered = sorted(zip(self.model.classes_, probabilities), key=lambda pair: -pair[1])
        raw_intent, raw_probability = ordered[0]
        details = IntentDetails(
            raw_intent=str(raw_intent), raw_probability=float(raw_probability),
            clarification_threshold=self.min_probability, reason="model_prediction",
        )
        if has_explicit_action_request(text):
            details.reason = "explicit_action_request"
            return IntentPrediction("action", details)
        explicit_intent = explicit_non_action_intent(text)
        if explicit_intent:
            details.reason = "explicit_non_action_request"
            return IntentPrediction(explicit_intent, details)
        if raw_intent == "action":
            details.reason = "action_not_authorized"
            intent, probability = next(pair for pair in ordered if pair[0] != "action")
        else:
            intent, probability = raw_intent, raw_probability
        if probability < self.min_probability or self.model.named_steps["tfidf"].transform([text]).nnz == 0:
            details.reason = "low_model_evidence"
            return IntentPrediction("clarification", details)
        return IntentPrediction(str(intent), details)


@lru_cache(maxsize=1)
def load_classifier(path: Path = DEFAULT_MODEL) -> IntentClassifier:
    if not path.is_file():
        raise FileNotFoundError("Train the local intent model first: python -m scripts.train")
    bundle = joblib.load(path)
    return IntentClassifier(model=bundle["model"], min_probability=bundle["min_probability"])
