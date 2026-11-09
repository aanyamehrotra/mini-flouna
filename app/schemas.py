"""Shared data contracts; unknown user preferences stay None."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Intent = Literal["information", "discovery", "decision", "action", "clarification"]
DietaryType = Literal["vegetarian", "vegan", "non_vegetarian"]
UpperBound = Literal["lt", "lte"]


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class UserRequest(DataModel):
    text: Text = Field(max_length=2000)


class Budget(DataModel):
    amount: float = Field(gt=0)
    operator: UpperBound
    currency: Literal["INR"] = "INR"


class DeliveryLimit(DataModel):
    minutes: int = Field(gt=0, strict=True)
    operator: UpperBound


class Preferences(DataModel):
    domain: Literal["food"] | None = None
    dish: Text | None = None
    cuisine: Text | None = None
    dietary_preference: DietaryType | None = None
    location: Text | None = None
    budget: Budget | None = None
    delivery_limit: DeliveryLimit | None = None
    min_rating: float | None = Field(default=None, ge=0, le=5)
    prefer_high_rating: bool | None = None
    priority: Literal["rating", "price", "delivery"] | None = None
    unsupported_constraints: list[Text] = Field(default_factory=list)


class UnderstoodRequest(DataModel):
    intent: Intent
    preferences: Preferences


class IntentDetails(DataModel):
    raw_intent: Intent
    raw_probability: float = Field(ge=0, le=1)
    clarification_threshold: float = Field(ge=0, le=1)
    reason: Literal[
        "model_prediction", "explicit_action_request", "action_not_authorized",
        "low_model_evidence", "unresolved_preferences",
        "explicit_non_action_request",
    ]


class FoodOption(DataModel):
    id: Text
    restaurant_name: Text
    dish_name: Text
    cuisine: Text
    dietary_type: DietaryType
    price_inr: float = Field(gt=0)
    rating: float = Field(ge=0, le=5)
    delivery_minutes: int = Field(gt=0, strict=True)
    location: Text


class RankedOption(DataModel):
    option_id: Text
    score: float = Field(ge=0, le=1)
    rating_component: float = Field(ge=0, le=1)
    delivery_component: float = Field(ge=0, le=1)
    price_component: float = Field(ge=0, le=1)


class RecommendationResponse(DataModel):
    status: Literal[
        "recommended", "options_found", "information", "clarification",
        "no_match", "unsupported_action", "unsupported_constraint",
    ]
    understood_request: UnderstoodRequest
    intent_details: IntentDetails | None = None
    recommendation: FoodOption | None = None
    options: list[FoodOption] = Field(default_factory=list)
    ranking: list[RankedOption] = Field(default_factory=list)
    explanation: Text
    clarification_question: Text | None = None
    warnings: list[Text] = Field(default_factory=list)
