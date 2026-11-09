"""Streamlit interface for the request-processing pipeline."""

import hashlib
import json
from pathlib import Path

import streamlit as st

from app.dataset import load_requests, load_splits
from app.intent import IntentClassifier
from app.pipeline import process_request
from app.schemas import RecommendationResponse
from scripts.train import make_model


ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Choose a meal": "Recommend highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes",
    "Browse options": "Show vegetarian pizza",
    "Ask a question": "How do I place an order?",
    "Missing details": "Help me",
    "Nothing fits": "Recommend vegetarian biryani under ₹50",
    "Unavailable information": "Recommend organic pizza",
    "An action request": "Please order a pizza",
    "Write my own": "",
}


@st.cache_resource
def demo_classifier() -> IntentClassifier:
    """Fit the already-selected configuration once; do not tune or use test data."""
    metadata = json.loads((ROOT / "reports" / "validation.json").read_text())
    for filename, key in (("requests.jsonl", "dataset_sha256"), ("splits.json", "splits_sha256")):
        if hashlib.sha256((ROOT / "data" / filename).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError("The demo data differs from the selected model configuration.")
    examples = load_requests()
    splits = load_splits(examples)
    training = [row for row in examples if row.id in splits.train]
    model = make_model(metadata["selected_C"])
    model.fit([row.text for row in training], [row.intent for row in training])
    return IntentClassifier(model, metadata["clarification_threshold"])


def show_result(result: RecommendationResponse) -> None:
    st.divider()
    st.caption(f"Request type: {result.understood_request.intent.title()}")
    if result.recommendation:
        option = result.recommendation
        st.subheader(option.dish_name)
        st.write(f"{option.restaurant_name} · {option.location}")
        price, rating, delivery = st.columns(3)
        price.metric("Price", f"₹{option.price_inr:g}")
        rating.metric("Restaurant rating", f"{option.rating:g}/5")
        delivery.metric("Estimated delivery", f"{option.delivery_minutes} min")
        st.write(result.explanation)
    else:
        headings = {
            "options_found": "Here are your options",
            "information": "Here's what we know",
            "clarification": "A little more detail, please",
            "no_match": "No matching option yet",
            "unsupported_constraint": "We can't verify that requirement",
            "unsupported_action": "Ordering isn't supported in this demo",
        }
        st.subheader(headings[result.status])
        st.write(result.explanation)
    if result.clarification_question:
        st.info(result.clarification_question)
    for warning in result.warnings:
        st.caption(warning)
    if result.options:
        st.dataframe([
            {"Restaurant": option.restaurant_name, "Dish": option.dish_name,
             "Location": option.location, "Price (INR)": option.price_inr,
             "Restaurant rating": option.rating, "Delivery (min)": option.delivery_minutes}
            for option in result.options
        ], hide_index=True, width="stretch")
    with st.expander("What we understood"):
        st.json(result.understood_request.preferences.model_dump())
    with st.expander("Full response and ranking"):
        st.json(result.model_dump())
    st.download_button("Download result as JSON", result.model_dump_json(indent=2),
                       file_name="flouna-result.json", mime="application/json", on_click="ignore")


def main() -> None:
    st.set_page_config(page_title="Mini FLOUNA", page_icon="🍃", layout="centered")
    st.title("Mini FLOUNA")
    st.write("Tell us what sounds good. We'll find options and explain the fit.")
    st.caption("A food recommendation demo with fictional options. No orders or payments are made.")
    example = st.selectbox("Try an example", list(EXAMPLES))
    with st.form("request_form"):
        text = st.text_area("Your request", value=EXAMPLES[example], key=f"request_{example}",
                            height=110, max_chars=2000, placeholder="Vegetarian biryani under ₹300 within 30 minutes…")
        submitted = st.form_submit_button("Get result", type="primary")
    if submitted:
        if not text.strip():
            st.warning("Please enter a food request first.")
        else:
            try:
                with st.spinner("Finding the fit…"):
                    result = process_request(text, classifier=demo_classifier())
                show_result(result)
            except (ValueError, FileNotFoundError):
                st.error("The demo could not load its data. Please check the repository setup.")
    st.caption("English requests · INR prices · Mock catalogue · Single-turn understanding")


if __name__ == "__main__":
    main()
