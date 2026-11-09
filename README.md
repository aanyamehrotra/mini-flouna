# Mini FLOUNA

An explainable food recommendation system being built for the FLOUNA ML Intern technical assignment.

The goal is to understand a natural-language request, extract explicitly stated preferences, evaluate mock food options, and explain a best-fit recommendation.

## Current status

The project includes shared schemas, 30 fictional food options, rule-based preference extraction, constraint filtering, reproducible ranking, factual explanations, a command-line demo, and 150 labelled requests with frozen evaluation splits. Automated intent classification, the API, and evaluation results will be added in subsequent commits. The baseline requires the caller to supply Discovery or Decision intent explicitly.

## Planned approach

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
requirements.txt      Runtime dependencies
requirements-dev.txt  Development and evaluation dependencies
```

See [data/README.md](data/README.md) for catalogue provenance, field definitions, dietary conventions, supported locations, and constraint semantics. Modules will be added when their functionality is implemented.

## Run the current checks

```bash
python -m pytest -q
```

These checks cover schemas, catalogue integrity, extraction, ranking, and baseline request behaviour. They are not held-out ML evaluation results.

## Try the baseline

```bash
python -m app --intent decision "Highly rated vegetarian biryani in Indiranagar under ₹300 within 30 minutes"
python -m app --intent discovery "Show vegetarian pizzas"
python -m app --intent decision "Organic biryani"
```

The first example recommends Garden Pot's ₹270 vegetable biryani over the cheaper option. The second lists eligible options without selecting a winner. The third reports an unsupported organic requirement and returns no recommendation. Output is structured JSON including extracted preferences, eligible options, weighted score contributions, explanations, and any clarification question or warnings.

`--intent` is a manual demo input, not a prediction. The baseline accepts only Discovery and Decision; other intents will be handled when the intent component is implemented. It never executes orders.

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

Extraction is deliberately rule-based and does not understand unrestricted English, all synonyms, spelled-out numbers, arbitrary ingredient combinations, or every negation construction. Known unsupported organic, allergen, spice, offers, live availability, exclusions, and multi-serving requests are blocked; this is not a guarantee that every unsupported phrase can be detected. All data and delivery estimates are synthetic. Supported language and policy examples are tested, but generalization needs the later labelled dataset and evaluation.

## Planned evaluation

The dataset contains 150 synthetic labelled requests across five intents, including ambiguous wording, negation, missing preferences, and difficult non-Action examples. Sixty also have gold preference annotations. The fixed split is 90 training / 30 validation / 30 test requests, with related paraphrases kept together. See [the dataset card](data/REQUESTS.md) for labelling rules, provenance, limitations, and split verification. Fit text features on training data only.

Report accuracy, per-class precision/recall/F1, macro F1, a confusion matrix, false Action predictions, and extraction accuracy. Tests will also check constraint boundaries, no-match responses, unsupported information, reproducibility, and cases where the cheapest option is not the best fit.

## AI assistance and data sources

OpenAI Codex assisted with the project plan, repository setup, schemas, catalogue creation and loading, extraction rules, ranking implementation, command-line demo, labelled request authorship and annotation, split preparation, documentation, and tests. All catalogue records and requests are synthetic; provenance and limitations are documented in `data/README.md` and `data/REQUESTS.md`. Labels were AI-authored and AI-reviewed, not independently human-annotated. No trained model, external API integration, or ML evaluation results are included yet. Further AI usage will be documented as components are added.
