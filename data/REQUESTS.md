# Labelled request dataset

Version 1 contains 150 synthetic English requests: 30 each for Information, Discovery, Decision, Action, and Clarification. There are 50 paraphrase groups with three requests per group. Sixty requests (all Discovery and Decision examples) also have complete expected preference objects. The other 90 have `expected_preferences: null`, meaning extraction is not annotated, not that every preference is absent.

## Authorship and annotation

OpenAI Codex authored the requests, paraphrases, tags, rationales, and gold preference objects and checked them against the rules below. Labels were specified directly, not generated from the classifier or extractor's predictions. No real user logs, private data, external dataset, or external generation API was used. Fictional order IDs and restaurant names refer only to mock scenarios. There has been no independent human annotation or inter-annotator agreement study.

`requests.jsonl` is the canonical versioned source, with one JSON object per line. It is not regenerated during training. Each record contains `id`, `group_id`, `text`, `intent`, `tags`, `rationale`, and optional `expected_preferences`. Only `text` may be used as the classifier input. IDs, group IDs, tags, rationales, gold preferences, and split membership would leak labels and must not be features.

## Labelling rules

1. **Clarification**: the request lacks essential domain/context, explicitly leaves a required value unresolved, or demands contradictory mandatory preferences. Missing optional budget or locality alone does not imply Clarification.
2. **Action**: explicitly asks the system to execute an order, purchase, payment, reservation, checkout, or cart/order modification. Polite questions such as “Can you place my order?” are Action requests. Execution is still unsupported by this prototype.
3. **Information**: asks for definitions, facts, procedures, field meanings, system explanations, or capabilities. “How do I place an order?” and “Can this app place orders?” do not authorize execution.
4. **Decision**: requests a best choice, recommendation, comparison to choose, or an explicit ranking priority used to pick an option. “What should I order?” is Decision, not Action.
5. **Discovery**: asks to find, see, or list options without requesting execution or a single best choice. An expression of wanting food is not ordering permission.

For multiple clauses, use the requested outcome, not an isolated action word. Explicitly negated ordering with a request to list options is Discovery. An unsupported attribute or an infeasible budget does not by itself make the intent unclear: a clear Discovery/Decision request retains that intent, while the pipeline may return `unsupported_constraint` or `no_match`.

These are annotation conventions, not universal definitions of overlapping intent categories. Mixed-intent conversations and multi-turn context are outside this dataset's scope.

## Preference annotations

Annotate explicit entities and constraints, leaving unstated scalar values `null`. A named dish such as chicken biryani is an entity, not an independently stated dietary preference. Qualitative “highly rated” is not a numeric rating threshold. “Highest rated” is a ranking priority. Budget and delivery bounds distinguish `lt` from `lte`; minimum ratings are inclusive.

Unsupported constraints use semantic category labels (`organic status`, `spice level`, `allergen requirements`, `food exclusions`, `non-INR currency`). The response extractor may append matched text after a colon. Extraction evaluation should compare these category sets, not incidental explanation wording. The original request preserves the literal constraint. Unsupported foreign-currency amounts do not create an INR budget.

The 60 annotations are gold expectations, not promises that the current parser handles every wording. Do not rewrite labels to match implementation errors. Field accuracy must include both missing and present values; report present-value performance separately so many `null` fields do not hide extraction errors.

## Frozen evaluation splits

`splits.json` assigns every request ID exactly once. It uses seed 42 and shuffles sorted groups independently within each intent, assigning 60% of groups to training, 20% to validation, and 20% to testing. With three examples per group, this gives:

| Split | Requests | Per intent | Groups | Preference annotations |
| --- | ---: | ---: | ---: | ---: |
| Training | 90 | 18 | 30 | 36 |
| Validation | 30 | 6 | 10 | 12 |
| Test | 30 | 6 | 10 | 12 |

Related paraphrases stay in one split. Groups may share broad vocabulary and food topics across splits; group separation prevents direct paraphrase leakage but does not establish real-world independence.

Verify the saved assignments without modifying them:

```bash
python -m scripts.prepare_splits
```

The explicit `--write` option regenerates the manifest. It is intended for deliberate dataset version changes, not searching for an easier test split. Freeze the data before model fitting, fit TF-IDF on training text only, and use validation data for model and threshold choices. Evaluate the final policy on the saved test split and report its failures. Do not tune on test predictions.

## Coverage and limitations

The dataset includes 12 non-Action examples mentioning ordering/capabilities, polite Action requests, negation, incomplete references, conflicting constraints, missing optional preferences, unavailable attributes, numeric bounds, foreign currency, and no-match scenarios.

It is small, balanced by design, AI-authored, and not representative of natural intent frequencies. Each test class has only six examples from two groups, so metrics will be unstable. Existing extraction code and unit-test scenarios predate these annotations; this is not an independently collected extraction benchmark. No model was trained and no prediction-based tuning was performed when preparing this version. Generalization claims must remain limited to this synthetic benchmark.
