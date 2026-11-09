# Validation results

`validation.json` records model selection on the frozen 90-request training and 30-request validation splits. It includes data hashes, dependency versions, all nine candidate configurations, raw-model metrics, guarded-intent metrics, final-pipeline metrics, per-class precision/recall/F1, confusion matrices (rows are actual, columns predicted, using `class_order`), and final-pipeline validation errors.

## Reproduce

```bash
python -m scripts.prepare_splits
python -m scripts.train
```

The selected configuration is word TF-IDF unigrams/bigrams with sublinear term frequency and Logistic Regression C=4, seed 42, and a 0.25 clarification threshold. The feature vocabulary is fitted on training text only. The selected model remains fitted on 90 requests; validation data is not added to its fitting set. Train using the recorded library versions for the closest numerical reproduction.

| Component | Validation accuracy | Macro F1 | False Actions |
| --- | ---: | ---: | ---: |
| Raw statistical model | 80.0% | 0.802 | 0 |
| Model with explicit intent checks | 86.7% | 0.856 | 0 |
| Full routing pipeline | 96.7% (29/30) | 0.966 | 0 |

There are 24 actual non-Action requests in validation; the false Action rate is 0/24 for these results. Zero observed errors on such a small sample does not establish a zero real-world error rate.

## What validation changed

The initial Action-gated classifier confused some explicit list requests with decisions and a speed-oriented choice request with information. Small list/choice checks were added. Pipeline checks also prevent known missing mandatory dietary details from being ignored, allow explicit unresolved locality constraints to trigger clarification even after an Information prediction, and avoid interpreting known ambiguous units or foreign currency as INR budgets.

These are documented policy choices refined using validation examples. Candidate selection uses the final pipeline's validation predictions. This report is therefore a tuning result, not an independent final performance estimate. Unit regression tests exercise these policies separately from the frozen test set.

## Remaining error and limits

“Which ordering actions can the prototype perform?” is labelled Information but the final pipeline asks for clarification. This failure remains recorded rather than changing the gold label or adding a phrase-specific exception.

The dataset is small, synthetic, balanced, and AI-authored. Six validation examples per class come from only two paraphrase groups. Checks are conservative and bounded: complex mixed intents, negation, indirect action requests, and new wording can still fail. The Information route is a catalogue/prototype FAQ, not a general knowledge model. Numeric probabilities are not calibrated.

`test_evaluated` is false. No predictions or extraction metrics have been computed for the 30-request final test split in this commit. The later evaluation should report the raw model and full pipeline separately, include extraction performance, a rendered confusion matrix, and failure analysis, without tuning on test results.
