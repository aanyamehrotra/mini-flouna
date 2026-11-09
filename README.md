# Mini FLOUNA

An explainable food recommendation system being built for the FLOUNA ML Intern technical assignment.

The goal is to understand a natural-language request, extract explicitly stated preferences, evaluate mock food options, and explain a best-fit recommendation.

Start with the [live demo guide](docs/DEMO.md), [design decisions](docs/DESIGN.md), and [complete sample inputs and outputs](examples/sample_io.json). The project runs locally with no API keys.

For a simple browser interface, run `python -m streamlit run streamlit_app.py`. See [Streamlit setup and deployment](docs/STREAMLIT.md) for a shareable Community Cloud deployment. This interface uses the same frozen processing pipeline; it needs no separately running API.

## Current status

The project includes Streamlit and FastAPI interfaces, automatic intent classification, five intent routes, shared schemas, 30 fictional food options, reproducible recommendations, a command-line demo, and 150 labelled requests with frozen splits. Training, validation, and final test evaluation are implemented.

## Approach

1. Classify requests as Information, Discovery, Decision, Action, or Clarification using TF-IDF and Logistic Regression.
2. Extract explicit preferences using regular expressions and small keyword dictionaries. Missing values remain unknown.
3. Filter mock food items against hard constraints, then rank eligible items using a documented scoring policy.
4. Generate explanations directly from the extracted preferences and catalogue data.
5. Expose the pipeline through FastAPI and evaluate it on labelled requests.

The initial scope is English food requests, INR prices, a local mock catalogue, and one request at a time. Ordering and payment execution are outside the scope. Discovery and Decision requests must never be treated as permission to act. Unsupported constraints and requests with no matching options must be reported explicitly.

## Setup

Use Python 3.11 or newer and run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

For runtime dependencies only, install `requirements.txt`. The development requirements include runtime dependencies plus testing and evaluation tools. Dependency ranges are bounded; exact versions used for the final evaluation will be recorded with the results.

## Project structure

```text
app/                  Application package
data/                 Mock catalogue and labelled requests
scripts/              Training and evaluation scripts
tests/                Behaviour and edge-case tests
reports/              Evaluation metrics and confusion matrix
examples/             Captured complete sample inputs and outputs
docs/                 Demo guide, design rationale, and tool disclosure
requirements.txt      Runtime dependencies
requirements-dev.txt  Development and evaluation dependencies
requirements-evaluated.txt  Exact package snapshot from the evaluated environment
```

See [data/README.md](data/README.md) for catalogue provenance, field definitions, dietary conventions, supported locations, and constraint semantics. Modules will be added when their functionality is implemented.

## Run the current checks

```bash
python -m pytest -q
```

These checks cover schemas, catalogue integrity, extraction, ranking, intent safeguards, and the HTTP interface. They are not held-out ML evaluation results. API tests train a temporary model from the training split, so they do not need an existing local model artifact.

## Train and try the system

Train once from the repository root:

```bash
python -m scripts.train
```

This fits TF-IDF and Logistic Regression on the 90 training requests, selects regularization and a clarification threshold using only the 30 validation requests, saves the model locally to `artifacts/intent_model.joblib`, and writes `reports/validation.json`. The binary model is ignored by Git; a fresh checkout needs this training step. Only load locally generated trusted model artifacts. Restart a running app after retraining its cached model.

Automatic intent examples:

```bash
python -m app "Show vegetarian pizza"
python -m app "Recommend vegetarian pizza under ₹300"
python -m app "How do I place an order?"
python -m app "Please order a pizza"
python -m app "Help me"
```

These demonstrate Discovery, Decision, Information, unsupported Action, and Clarification respectively. No ordering, payment, cart modification, or reservation operation is implemented. Information responses use a small factual FAQ about the prototype and catalogue; general food knowledge questions receive an explicit scope limitation. Information topics are not treated as user preferences.

You can still use a manual intent override for debugging:

```bash
python -m app --intent decision "Highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"
python -m app --intent discovery "Show vegetarian pizzas"
python -m app --intent decision "Organic biryani"
```

The first example recommends Garden Pot's ₹270 vegetable biryani over the cheaper option. The second lists eligible options without selecting a winner. The third reports an unsupported organic requirement and returns no recommendation. Output is structured JSON including extracted preferences, eligible options, weighted score contributions, explanations, and any clarification question or warnings.

`--intent` is a manual debug input, not a prediction; it is not required for normal usage. Even a manual Action override requires an explicit execution request and returns `unsupported_action`.

## Run the API

After installing dependencies and training the model:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open [interactive API documentation](http://127.0.0.1:8000/docs). Expand `POST /recommend`, select one of the seven examples, click **Try it out**, then **Execute**. Alternative reference documentation is available at [ReDoc](http://127.0.0.1:8000/redoc). The root URL redirects to the interactive documentation. The documentation UI loads its display assets from a public CDN; request processing uses only local data and the local model.

Send a request from a terminal:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"text":"Recommend highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"}'
```

Selected response fields for that example:

```json
{
  "status": "recommended",
  "recommendation": {
    "id": "food_002",
    "restaurant_name": "Garden Pot",
    "dish_name": "Vegetable Biryani",
    "price_inr": 270.0,
    "rating": 4.7,
    "delivery_minutes": 24
  }
}
```

The full response also includes extracted preferences, intent details, eligible options, score contributions, explanation, and warnings. HTTP requests accept only `text`; the command-line debugging override is not exposed through the API.

| Endpoint or HTTP status | Meaning |
| --- | --- |
| `GET /health` | Liveness check returning `{"status":"ok"}`; it does not verify model readiness |
| `POST /recommend` → `200` | Request processed; inspect the JSON `status` for the outcome |
| `422` | Invalid JSON or request fields: missing/blank/non-string text, more than 2,000 characters, or unknown fields |
| `503` | Model has not been trained; run `python -m scripts.train` |

Valid domain outcomes are `recommended`, `options_found`, `information`, `clarification`, `no_match`, `unsupported_action`, and `unsupported_constraint`. Clarification, no-match, and unsupported outcomes are ordinary processed requests, not HTTP failures. The API adds no ordering or payment endpoint and never executes an Action request.

## Intent policy

The statistical baseline is word TF-IDF unigrams/bigrams with Logistic Regression. Stop words are retained so negation is not deliberately removed. Only request text is a feature. Three C values (1, 4, 10) and three clarification thresholds (0.25, 0.35, 0.45) are compared. Selection minimizes false Actions in the final validation pipeline, then maximizes macro F1 and accuracy, with deterministic ties.

The hybrid policy adds small, explicit checks:

- Anchored execution requests, including polite “Can you order…” forms, produce Action; procedural questions, capability questions, and isolated mentions do not authorize Action.
- Known prohibitions conservatively suppress Action. This may miss a genuine second action in a mixed request; it is not a general negation parser.
- Clear list/show/find instructions preserve Discovery, while clear recommendation/choice instructions preserve Decision.
- A raw model Action prediction without explicit execution evidence is replaced by a sufficiently supported non-Action class or Clarification.
- Low model evidence, unknown vocabulary, conflicting preferences, and detected missing mandatory details lead to clarification. Information definitions comparing dietary labels are not treated as contradictory user diets.

JSON `intent_details` exposes the raw model label, its probability, the clarification threshold, and the policy reason. The raw probability is uncalibrated; it is neither guaranteed confidence nor necessarily the probability of the final policy label. It is absent in manual debug mode.

## Extraction and ranking policy

The extractor supports a small English vocabulary of dishes and cuisines, explicit localities, vegetarian/vegan/non-vegetarian labels, numeric upper price and delivery bounds, numeric minimum ratings, and rating/price/speed priorities. “Under” and “below” are strict; “at most”, “up to”, and “within” are inclusive. Delivery bounds support minutes and numeric hours convertible to whole minutes. Unsupported preferences are retained as matched phrases and block recommendations. Conflicting supported values trigger clarification.

Missing budget, rating, diet, and location remain `null`. Missing optional fields do not block otherwise usable food requests. Without a locality, results span mock locality scenarios and include a warning; they do not establish real delivery availability. Cuisine and locality comparisons are exact ignoring case; dish phrases match catalogue dish names. Vegan items can satisfy vegetarian requests.

Every explicit supported constraint is applied before scoring. Empty results produce `no_match`, overlapping exclusion counts, and a question asking which requirement to change. Constraints are never silently relaxed.

Default score:

```text
0.60 × (restaurant rating / 5)
+ 0.25 × max(0, 1 - delivery minutes / 60)
+ 0.15 × max(0, 1 - price INR / 500)
```

The 60-minute and ₹500 reference ranges are fixed design choices, not inferred user constraints. For “highly rated”, weights become 0.80 / 0.15 / 0.05. Explicit cheapest, fastest, or highest-rated priorities rank directly by the requested attribute and use that attribute alone for the displayed score. Direct ordering preserves price/speed differences beyond the clipped score ranges. Ties use stable option IDs. A score is a ranking utility, not a probability or confidence estimate.

The `ranking` array exposes weighted contributions, aligned with `options`. Decision explanations include the selected record's facts, requested limits, ranking policy, and runner-up facts when available. Discovery lists results without implying permission to order.

## Current limitations

Extraction and intent safeguards are deliberately bounded and do not understand unrestricted English, all synonyms, spelled-out numbers, arbitrary ingredient combinations, or every negation construction. Known unsupported organic, allergen, spice, offers, live availability, exclusions, and multi-serving requests are blocked; this is not a guarantee that every unsupported phrase can be detected. Foreign currency never creates an INR budget. All catalogue data and delivery estimates are synthetic. Model probabilities are uncalibrated and validation metrics are not evidence of production reliability.

## Evaluation

The dataset contains 150 synthetic labelled requests across five intents, including ambiguous wording, negation, missing preferences, and difficult non-Action examples. Sixty also have gold preference annotations. The fixed split is 90 training / 30 validation / 30 test requests, with related paraphrases kept together. See [the dataset card](data/REQUESTS.md) for labelling rules, provenance, limitations, and split verification. Fit text features on training data only.

Run final evaluation after training:

```bash
python -m scripts.evaluate
```

The frozen test split produces 93.3% final-pipeline intent accuracy (28/30), macro F1 0.933, and zero false Actions among 24 non-Action requests. The raw classifier achieves 63.3% accuracy with one false Action; this difference shows the contribution of explicit policies and routing. On 12 preference-annotated test requests, extraction matches 11 exactly, with 97.2% accuracy on explicitly present fields. These are small, synthetic benchmark results.

See [the evaluation report](reports/README.md), [complete test metrics](reports/test_metrics.json), [per-request predictions](reports/test_predictions.jsonl), and [confusion matrix](reports/test_confusion_matrix.png). Errors remain documented; no runtime rules, model parameters, labels, or splits were changed after observing test results.

The [validation report](reports/validation.json) separately records all candidate scores and validation failures. The selected model uses C=4 and a 0.25 clarification threshold. Raw model validation accuracy is 80%; final pipeline validation accuracy is 29/30 (96.7%), with zero false Actions among 24 non-Action validation requests. Safeguards were refined using validation failures, so validation numbers are tuning results rather than final performance estimates.

## AI assistance and data sources

OpenAI Codex assisted with the project plan, repository setup, schemas, synthetic data authorship and annotation, split preparation, extraction, ranking, intent classification and safeguards, training and validation, command-line demo, API integration, documentation, and tests. Provenance and limitations are documented in `data/README.md` and `data/REQUESTS.md`. Labels were AI-authored and AI-reviewed, not independently human-annotated. The runtime uses no external AI API or production service. Further AI usage will be documented as components are added.

See [the complete libraries, models, data, and AI disclosure](docs/TOOLS.md). Exact installed versions are recorded in `requirements-evaluated.txt`; use the bounded requirements files for normal setup.

## Submission materials

- Source code and setup: this repository and README.
- Dataset and generation method: `data/options.json`, `data/requests.jsonl`, `data/splits.json`, and their documentation.
- Evaluation: `reports/README.md`, JSON metrics and per-request predictions, and the PNG confusion matrix.
- Samples: `examples/sample_io.json` and seven API documentation examples.
- Tests: run `python -m pytest -q`; the Streamlit addition includes presentation-layer tests alongside the 166 original checks.
- Live demo: follow `docs/DEMO.md` to present the interactive local API; no prerecorded video or public deployment is included.
- Limitations and improvements: this README, `reports/README.md`, and `docs/DESIGN.md`.
- Libraries, models, APIs, datasets, and AI assistance: `docs/TOOLS.md` and the requirements files.

The Streamlit interface, deployment setup, and its tests were also created with OpenAI Codex assistance. The Streamlit layer does not change the frozen runtime or test labels.
