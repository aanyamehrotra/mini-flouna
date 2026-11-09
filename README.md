# Mini FLOUNA

A food request-understanding and recommendation system built for the FLOUNA ML Intern assignment.

**[Open the live Streamlit app](https://mini-flouna-kvokv5gr9m899aqnzdfhvr.streamlit.app/)** · [Sample inputs and outputs](examples/sample_io.json) · [Evaluation report](reports/README.md)

Enter a request such as “Recommend highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes.” The system extracts the preferences, checks a fictional food catalogue, ranks matching options, and explains its choice. This example selects Garden Pot's ₹270 biryani, rated 4.7, with an estimated delivery time of 24 minutes.

## What it does

- Identifies Information, Discovery, Decision, Action, and Clarification requests.
- Extracts dish, cuisine, dietary preference, location, budget, delivery limit, and rating preference into validated JSON. Unstated values remain `null`.
- Filters explicit constraints before ranking. Vegan options may satisfy vegetarian requests; vegetarian options cannot automatically satisfy vegan requests.
- Lists options for Discovery and selects a best fit for Decision. The cheapest option does not automatically win.
- Explains recommendations using catalogue facts and exposes score contributions.
- Asks for missing essential details, reports no matches, and identifies requirements that the catalogue cannot verify.
- Recognizes explicit Action requests but performs no ordering, payments, cart changes, or reservations. “What should I order?” is never treated as ordering permission.

The catalogue has 30 fictional food items across Indiranagar, Koramangala, and Whitefield. Prices are total mock prices for one serving; ratings describe restaurants; delivery times are estimates. [Catalogue definitions and source](data/README.md).

## Run locally

Python 3.11 or newer is required. Local testing used Python 3.14.2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m streamlit run streamlit_app.py
```

On Windows, activate with `.venv\Scripts\activate`. Open http://localhost:8501. Installing `requirements.txt` alone is sufficient to run the app; development requirements also include testing and plotting tools.

Streamlit builds and caches the small model from the 90 training requests using the configuration already selected on validation data. It checks data hashes and does not tune on startup or use test requests for fitting. No API keys, uploaded model file, or separately running API are needed.

### API and command line

Train a local model artifact first:

```bash
python -m scripts.train
python -m app "Recommend vegetarian pizza under ₹300"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for seven interactive examples, or send a request:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"text":"Recommend vegetarian pizza under ₹300"}'
```

`POST /recommend` accepts only `text` and returns the intent, extracted preferences, matching options, recommendation, score contributions, explanation, and any clarification question. Valid outcomes use HTTP 200 with a JSON `status`: `recommended`, `options_found`, `information`, `clarification`, `no_match`, `unsupported_action`, or `unsupported_constraint`. Invalid payloads return 422; an untrained API model returns 503 with setup instructions. `GET /health` checks process liveness.

The CLI also supports `--intent` for manual debugging; the HTTP interface always classifies automatically. Restart a running API after retraining its cached model.

## Approach and code structure

```text
Request → intent classification → explicit preference extraction
        → clarification or intent-specific response
        → hard filters → ranking → factual explanation
```

The classifier uses TF-IDF word unigrams/bigrams and Logistic Regression. This is fast and reproducible for the small dataset. Short rules preserve clear list/choice instructions, require explicit evidence for Action, and handle known ambiguity. Extraction uses regular expressions and dictionaries. Recommendations use deterministic scoring and template explanations. No runtime LLM or production service is used.

| Location | Responsibility |
| --- | --- |
| `streamlit_app.py` | Browser interface and cached demo model |
| `app/main.py`, `app/__main__.py` | API and CLI entry points |
| `app/pipeline.py`, `app/intent.py`, `app/extraction.py` | Request understanding and routing |
| `app/recommendation.py`, `app/information.py` | Filtering, ranking, explanations, and a small catalogue FAQ |
| `app/schemas.py`, `app/catalogue.py`, `app/dataset.py` | Data validation and loading |
| `scripts/` | Split verification, model training, and evaluation |
| `data/`, `tests/`, `reports/`, `examples/` | Datasets, checks, evaluation evidence, and captured outputs |

### Ranking policy

After enforcing all supported explicit constraints, the default score is:

```text
0.60 × (restaurant rating / 5)
+ 0.25 × max(0, 1 - delivery minutes / 60)
+ 0.15 × max(0, 1 - price INR / 500)
```

“Highly rated” changes the weights to 0.80 / 0.15 / 0.05 without inventing a minimum rating. An explicit cheapest, fastest, or highest-rated priority sorts by that attribute. Stable IDs break ties. The reference ranges and weights are system policy, not inferred user preferences; scores are not confidence probabilities.

“Under ₹300” is strict; “₹300 or less” is inclusive. Explicit minimum ratings are inclusive. If no item meets all constraints, the system asks which requirement to change. Unsupported attributes such as organic status, allergens, or spice level block recommendations rather than being silently ignored.

## Data and evaluation

There are 150 synthetic labelled requests: 30 per intent, grouped into 50 paraphrase families. Sixty requests also have gold preference annotations. The fixed split contains 90 training, 30 validation, and 30 test requests; related paraphrases stay together. Labels, tags, and rationales are never classifier features. [Dataset creation, annotation rules, and limitations](data/REQUESTS.md).

```bash
python -m scripts.prepare_splits
python -m scripts.train
python -m scripts.evaluate
python -m pytest -q
```

Training fits TF-IDF only on training text. Nine C/clarification-threshold combinations are compared on validation data; the selected configuration is C=4 and threshold 0.25. Evaluation loads the saved model and refuses changed dataset/split hashes.

| Test result | Raw classifier | Full pipeline |
| --- | ---: | ---: |
| Accuracy | 63.3% (19/30) | 93.3% (28/30) |
| Macro F1 | 0.569 | 0.933 |
| False Actions among 24 non-Action requests | 1 | 0 |

Preference extraction exactly matches 11/12 annotated test requests, with 97.2% accuracy on explicitly present fields. These results come from a small synthetic benchmark; zero observed false Actions does not establish a zero real-world error rate.

[Per-class precision, recall, F1 and errors](reports/test_metrics.json) · [Every test prediction](reports/test_predictions.jsonl) · [Confusion matrix](reports/test_confusion_matrix.png) · [Validation selection results](reports/validation.json)

The tests cover all seven minimum assignment cases: discovery/decision, missing preferences, clarification, non-Action ordering language, a best fit that is not cheapest, multiple constraints, and unavailable catalogue information. They also check numeric boundaries, negation, empty inputs, deterministic ranking, data leakage, API errors, and all Streamlit outcomes.

## Limitations and improvements

English, INR, single-turn requests, and a bounded food vocabulary are supported. The Information route is a small prototype/catalogue FAQ. Arbitrary ingredients, all wording variations, complex negation, live availability, and multi-turn references are not fully understood. Model probabilities are uncalibrated.

The frozen test has two intent errors: a biryani-definition request gets Clarification, and an unresolved “other one” reference gets Information. One extraction case adds an extra unsupported exclusion label to “not spicy noodles.” These failures remain in the report. The readability cleanup preserves predictions; it adds no test-specific rules.

With more time, the priority would be independently human-labelled requests, more paraphrase groups, better context and negation handling, and calibrated uncertainty, evaluated on a new untouched test set.

## Libraries and AI assistance

Python, scikit-learn, joblib, Pydantic, FastAPI, Uvicorn, Streamlit, pytest, HTTPX/TestClient, Matplotlib, NumPy, and SciPy are used. Direct dependency ranges are in the requirements files. `requirements-evaluated.txt` records the original numerical benchmark environment and predates Streamlit; core versions and hashes are also recorded in the reports.

OpenAI Codex assisted with planning, implementation, synthetic data and label creation, tests, evaluation, and documentation. Labels were AI-authored and AI-reviewed, with no independent human annotation study. No real customer data, private company data, production credentials, external AI API, or production ONDC service is used.

## Demo and deployment

Use the [public Streamlit app](https://mini-flouna-kvokv5gr9m899aqnzdfhvr.streamlit.app/) and [short demo walkthrough](docs/DEMO.md). A prerecorded video is not included; the hosted app provides the live demo requested by the assignment.

The Community Cloud deployment uses repository `aanyamehrotra/mini-flouna`, branch `main`, entry point `streamlit_app.py`, and root `requirements.txt`, with no secrets. The UI was tested with Streamlit 1.67.0. The Streamlit and API documentation interfaces may load display assets from public CDNs; inference and ranking use local resources.
