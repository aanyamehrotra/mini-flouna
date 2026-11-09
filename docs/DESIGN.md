# Design decisions

The system separates language understanding from option evaluation so a classification error cannot directly execute a transaction. The only output is data and explanations.

```text
Request text
  → TF-IDF classifier and explicit intent checks
  → Preference extraction and clarification checks
  → Intent-specific response
  → For Discovery or Decision: hard filters → deterministic ranking → explanation
```

## Source map

| Module | Responsibility |
| --- | --- |
| `app/main.py` | HTTP validation, documentation examples, model dependency, and API routes |
| `app/pipeline.py` | Coordinates intent, extraction, clarification, and response routes |
| `app/intent.py` | Loads the classifier and applies explicit Action/list/choice checks |
| `app/extraction.py` | Extracts supported explicit values and identifies known unresolved requirements |
| `app/recommendation.py` | Enforces constraints, scores eligible options, and explains actual data |
| `app/information.py` | Answers a small set of prototype/catalogue FAQ topics |
| `app/schemas.py` | Shared validated JSON contracts |
| `app/catalogue.py`, `app/dataset.py` | Validate mock records, gold labels, and split integrity |
| `scripts/train.py` | Fits training-only features and selects a model using validation data |
| `scripts/evaluate.py` | Evaluates the frozen artifact and produces auditable test reports |

## Why these choices

TF-IDF and Logistic Regression provide a small, reproducible statistical baseline that is practical for 90 training requests. This data does not justify a neural-model training project. Word bigrams preserve some local context; retaining stop words avoids deliberately dropping negation. Raw model performance is reported separately so the contribution of rules is visible.

Rules handle explicit numeric values and small dictionaries. Missing values remain unknown, unsupported requirements block a recommendation, and detected mandatory ambiguity produces a question. These rules are transparent but have bounded coverage; a larger rule collection is not automatically a better language model.

Hard constraints precede scoring, so a high restaurant rating cannot compensate for violating a dietary restriction, budget, delivery limit, or other supported explicit requirement. Fixed normalization ranges make a candidate's score independent of unrelated catalogue entries. Weights are documented system policy, not learned user preferences. Explicit cheapest/fastest/highest-rated priorities override the balanced policy; stable IDs break ties.

Explanations use templates populated from the selected record and extracted requirements. No runtime LLM is needed to invent or paraphrase supporting facts. JSON files are sufficient for 30 catalogue items and 150 requests; a database, vector store, orchestration framework, or separate frontend would add little to this assignment.

## Limits and next improvements

The benchmark has only ten paraphrase groups per intent and is AI-authored. First improve evidence: collect independently human-labelled requests, increase group diversity, measure annotator agreement, and retain a new untouched test set before changing the tested runtime.

Then address the observed missing-context/definition intent failures and the extra unsupported-exclusion label. Evaluate uncertainty calibration and broader negation/indirect-request handling against the larger data rather than adding exceptions for individual final-test phrases. Ingredient/allergen guarantees, real availability, multi-turn memory, and transaction execution remain outside this prototype's scope.
