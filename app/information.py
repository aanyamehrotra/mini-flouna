"""Small factual FAQ about the prototype, rather than an open-ended knowledge bot."""

import re


def answer_information(text: str) -> str:
    text = text.casefold()
    if re.search(r"\b(?:score|ranking|weights)\b", text):
        return (
            "The default score weights restaurant rating by 0.60, delivery speed by 0.25, "
            "and price by 0.15 after filtering explicit constraints. Rating uses a 0–5 scale; "
            "delivery and price use fixed 60-minute and ₹500 reference ranges. "
            "Highly rated requests use 0.80/0.15/0.05; an explicit price, speed, or rating priority ranks by that attribute."
        )
    if re.search(r"\b(?:rating|ratings)\b", text):
        return "Catalogue ratings are fictional restaurant ratings on a 0–5 scale, not dish-specific ratings."
    if re.search(r"\b(?:delivery|arrival)\b", text):
        return "Delivery times are fictional estimates for each catalogue locality, not guaranteed arrival times or live availability."
    if re.search(r"\b(?:price|price_inr|costs|taxes|fees)\b", text):
        return "price_inr is the total fictional payable price for one serving, including modelled fees and taxes. No additional charges are added."
    if re.search(r"\b(?:vegan|vegetarian|non-vegetarian)\b", text):
        return (
            "In this catalogue, vegan items contain no animal products, vegetarian items may contain dairy, "
            "and egg-containing items are labelled non-vegetarian. Vegan items can satisfy a vegetarian request. "
            "These mock labels do not establish allergen safety or preparation practices."
        )
    if re.search(r"\b(?:order|orders|ordering|payment|checkout)\b", text):
        return "This prototype can understand requests and recommend mock food options. It cannot place orders, modify carts, make payments, or execute other transactions."
    return (
        "This prototype supports a local FAQ about catalogue prices, ratings, delivery estimates, "
        "dietary labels, ranking, and ordering capabilities. It does not have a general food knowledge base."
    )
