# Mini FLOUNA

An explainable food recommendation system being built for the FLOUNA ML Intern technical assignment.

The goal is to understand a natural-language request, extract explicitly stated preferences, evaluate mock food options, and explain a best-fit recommendation.

## Current status

The project now includes shared request/response schemas, a validated catalogue of 30 fictional food options, and tests for data integrity and input boundaries. Request processing, the API, labelled requests, and evaluation results will be added in subsequent commits. There is no running recommendation service yet.

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

These checks validate the schemas and mock catalogue. They do not measure intent classification or recommendation quality yet.

## Planned evaluation

Target at least 100 reviewed labelled requests covering all five intents, including ambiguous wording, negation, missing preferences, and difficult non-Action examples. Keep related paraphrases in the same split and fit text features on training data only.

Report accuracy, per-class precision/recall/F1, macro F1, a confusion matrix, false Action predictions, and extraction accuracy. Tests will also check constraint boundaries, no-match responses, unsupported information, reproducibility, and cases where the cheapest option is not the best fit.

## AI assistance and data sources

OpenAI Codex assisted with the project plan, repository setup, schemas, catalogue loader, tests, and synthetic catalogue creation. All catalogue records are fictional; their definitions and limitations are documented in `data/README.md`. No trained model, labelled request dataset, external API integration, or evaluation results are included yet. Further data generation, augmentation, and AI usage will be documented as those components are added.
