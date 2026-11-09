"""Small, deliberately bounded English preference extractor."""

import re
from dataclasses import dataclass

from app.schemas import Budget, DeliveryLimit, Preferences, UserRequest


@dataclass
class ExtractionResult:
    preferences: Preferences
    questions: list[str]


# Longest dish names are matched first so "chicken biryani" is not just "biryani".
DISHES = (
    "vegetable fried rice", "paneer rice bowl", "vegetable biryani",
    "chicken biryani", "mutton biryani", "vegetable noodles",
    "chicken noodles", "margherita pizza", "vegetable pizza",
    "paneer pizza", "chickpea salad", "egg fried rice", "masala dosa",
    "plain dosa", "fried rice", "biryani", "noodles", "pizza", "dosa",
    "salad", "sushi", "burger", "tacos",
)
CUISINES = ("south indian", "indian", "italian", "chinese", "continental")
LOCATIONS = ("indiranagar", "koramangala", "whitefield")
MONEY = r"(?:₹\s*|rs\.?\s*|inr\s*)?"
NUMBER = r"\d+(?:\.\d+)?"


def _longest_matches(text: str, values: tuple[str, ...]) -> list[str]:
    matches = []
    for value in sorted(values, key=len, reverse=True):
        if re.search(rf"\b{re.escape(value)}\b", text):
            matches.append(value)
            text = re.sub(rf"\b{re.escape(value)}\b", " ", text)
    return matches


def extract_preferences(text: str) -> ExtractionResult:
    text = UserRequest(text=text).text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\b(pizzas|burgers|salads|dosas|biryanis)\b", lambda m: {
        "pizzas": "pizza", "burgers": "burger", "salads": "salad",
        "dosas": "dosa", "biryanis": "biryani",
    }[m.group(0)], text)
    text = text.replace("non-vegetarian", "non vegetarian").replace("non-veg", "non veg")
    text = re.sub(r"\bveg\s+biryani\b", "vegetarian biryani", text)
    preferences = Preferences()
    questions: list[str] = []

    dishes = _longest_matches(text, DISHES)
    cuisines = _longest_matches(text, CUISINES)
    locations = _longest_matches(text, LOCATIONS)
    for values, field, question in (
        (dishes, "dish", "Which single dish would you like?"),
        (cuisines, "cuisine", "Which single cuisine would you like?"),
        (locations, "location", "Which delivery locality should I use?"),
    ):
        if len(values) == 1:
            setattr(preferences, field, values[0])
        elif len(values) > 1:
            questions.append(question)

    # Preserve explicit unfamiliar localities rather than inventing a known one.
    if not locations:
        location = re.search(
            r"\b(?:in|near)\s+([a-z]+(?:\s+[a-z]+){0,2}?)"
            r"(?=\s+(?:under|below|within|with|for|rated|that|and)\b|[,.!?]|$)",
            text,
        )
        if location:
            preferences.location = location.group(1).strip()

    dietary_text = text
    diets = []
    for pattern, label in (
        (r"\bnon\s+(?:vegetarian|veg)\b", "non_vegetarian"),
        (r"\bvegan\b", "vegan"),
        (r"\b(?:vegetarian|veg)\b", "vegetarian"),
    ):
        if re.search(pattern, dietary_text):
            diets.append(label)
            dietary_text = re.sub(pattern, " ", dietary_text)
    if len(diets) == 1:
        preferences.dietary_preference = diets[0]
    elif len(diets) > 1:
        questions.append("Which single dietary requirement should I use?")
    if not diets and re.search(r"\b(?:dietary restrictions?|food restrictions?|special diet)\b", text):
        questions.append("What exact dietary restrictions should I apply?")

    budgets = []
    for match in re.finditer(
        rf"\b(under|below|less than|at most|up to|no more than)\s*{MONEY}({NUMBER})"
        rf"(?:\s*(?:rupees|inr))?(?![\d.])", text,
    ):
        # A bound followed by a time unit belongs to delivery, not price.
        if re.match(r"\s*(?:minutes?|mins?|hours?|seconds?|secs?|calories|kcal)\b", text[match.end():]):
            continue
        operator = "lt" if match.group(1) in {"under", "below", "less than"} else "lte"
        budgets.append((float(match.group(2)), operator))
    for match in re.finditer(rf"{MONEY}({NUMBER})\s*(?:rupees|inr)?\s+or less\b", text):
        if not re.match(r"\s*(?:minutes?|mins?)\b", text[match.end():]):
            budgets.append((float(match.group(1)), "lte"))
    budgets = list(dict.fromkeys(budgets))
    if len(budgets) == 1 and budgets[0][0] > 0:
        preferences.budget = Budget(amount=budgets[0][0], operator=budgets[0][1])
    elif budgets:
        questions.append("What single positive maximum budget should I use?")
    elif re.search(r"₹|\brs\.?\s*\d|\binr\b|\bbudget\b|\brupees\b", text):
        questions.append("What is your maximum budget in INR, and is the limit inclusive?")
    if re.search(r"\b(?:forgot to say the unit|might be time or money|not said what the limit measures)\b", text):
        preferences.budget = None
        questions.append("Does the numeric limit refer to price or delivery time, and what unit should I use?")

    times = []
    for match in re.finditer(
        rf"\b(within|under|below|less than|at most|up to|no more than)\s+"
        rf"({NUMBER})\s*(minutes?|mins?|hours?)\b", text,
    ):
        minutes = float(match.group(2)) * (60 if match.group(3).startswith("hour") else 1)
        operator = "lt" if match.group(1) in {"under", "below", "less than"} else "lte"
        times.append((minutes, operator))
    times = list(dict.fromkeys(times))
    if len(times) == 1 and times[0][0] > 0 and times[0][0].is_integer():
        preferences.delivery_limit = DeliveryLimit(minutes=int(times[0][0]), operator=times[0][1])
    elif times or re.search(r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?)\b", text):
        questions.append("What single positive delivery limit in whole minutes should I use?")

    ratings = re.findall(rf"\b(?:rated|rating)\s+(?:at least\s+|above\s+)?({NUMBER})(\+)?", text)
    if "above" in text and re.search(r"\b(?:rated|rating)\s+above\b", text):
        questions.append("Should the minimum rating be inclusive? Please use 'rated at least'.")
    elif len(set(ratings)) == 1 and 0 <= float(ratings[0][0]) <= 5:
        preferences.min_rating = float(ratings[0][0])
    elif ratings:
        questions.append("What single minimum rating between 0 and 5 should I use?")
    if re.search(r"\b(?:highly rated|top rated|best rated)\b", text):
        preferences.prefer_high_rating = True

    priorities = []
    for pattern, priority in (
        (r"\b(?:cheapest|lowest price)\b", "price"),
        (r"\b(?:fastest|quickest)\b", "delivery"),
        (r"\b(?:highest rated|best rated|prioritize rating)\b", "rating"),
    ):
        if re.search(pattern, text):
            priorities.append(priority)
    if len(priorities) == 1:
        preferences.priority = priorities[0]
    elif priorities:
        questions.append("Should I prioritize rating, price, or delivery speed?")

    for pattern, label in (
        (r"\borganic\b", "organic status"),
        (r"\b(?:calories|kcal|protein|low.fat)\b", "nutrition information"),
        (r"\b(?:allerg\w*|nut.free|gluten.free|dairy.free|peanut\w*)\b", "allergen requirements"),
        (r"\b(?:spicy|spice|mild)\b", "spice level"),
        (r"\b(?:discount|offer|coupon|cashback)\b", "offers"),
        (r"\b(?:in stock|available now)\b", "live availability"),
        (r"[$€£]|\b(?:usd|eur|gbp|dollars?|euros?)\b", "non-INR currency"),
        (r"\b(?:near me|nearby)\b", "current user location"),
        (r"\b\d+\s*(?:seconds?|secs?)\b", "delivery units other than minutes or hours"),
        (r"\b(?:without|exclude|avoid|except)\s+\w+(?:\s+\w+)?", "food exclusions"),
        (r"\bno\s+(?!more\b)\w+", "negative requirement"),
        (r"\b(?:don't|do not)\s+(?:want|like|include)\s+\w+", "negative requirement"),
        (r"\b(?:for\s+\d+\s+(?:people|persons)|\d+\s+(?:servings|portions))\b", "multiple servings"),
        (r"\b(?:without|exclude|avoid|except|not|no)\s+(?!more\b)(?:\w+\s+)?(?:vegetarian|veg|vegan|chicken|meat|egg|paneer|biryani|pizza|dosa|noodles)\b", "excluded food or dietary preference"),
        (r"\b(?:over|above|at least|more than)\s*(?:₹|rs\.?|inr)\s*\d", "minimum spending requirement"),
    ):
        match = re.search(pattern, text)
        if match:
            preferences.unsupported_constraints.append(f"{label}: {match.group(0)}")
            if label == "non-INR currency":
                preferences.budget = None

    if dishes or cuisines or diets or re.search(r"\b(?:food|meal|restaurant|eat|dinner|lunch|breakfast)\b", text):
        preferences.domain = "food"
    return ExtractionResult(preferences, questions)
