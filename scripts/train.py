"""Select a tiny intent model using training and validation data only."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from app.dataset import DATA_DIRECTORY, INTENTS, load_requests, load_splits
from app.intent import DEFAULT_MODEL, IntentClassifier
from app.pipeline import process_request


def make_model(regularization: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(C=regularization, max_iter=1000, random_state=42)),
    ])


def intent_metrics(truth: list[str], predicted: list[str]) -> dict:
    false_actions = sum(actual != "action" and prediction == "action" for actual, prediction in zip(truth, predicted))
    non_actions = sum(actual != "action" for actual in truth)
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=INTENTS, average="macro", zero_division=0)),
        "false_action_count": false_actions,
        "false_action_rate": false_actions / non_actions if non_actions else None,
        "classification_report": classification_report(truth, predicted, labels=INTENTS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=INTENTS).tolist(),
        "class_order": list(INTENTS),
    }


def train_model(output_path: Path = DEFAULT_MODEL, report_path: Path | None = None) -> dict:
    examples = load_requests()
    splits = load_splits(examples)
    by_id = {example.id: example for example in examples}
    train = [by_id[i] for i in splits.train]
    validation = [by_id[i] for i in splits.validation]
    truth = [row.intent for row in validation]
    candidates = []
    selected = None
    best_key = None
    # C is inverse regularization strength. Keep the search intentionally small.
    for regularization in (1.0, 4.0, 10.0):
        model = make_model(regularization)
        model.fit([row.text for row in train], [row.intent for row in train])
        for threshold in (0.25, 0.35, 0.45):
            classifier = IntentClassifier(model, threshold)
            predictions = [process_request(row.text, classifier=classifier).understood_request.intent for row in validation]
            metrics = intent_metrics(truth, predictions)
            candidates.append({
                "C": regularization, "clarification_threshold": threshold,
                **{key: metrics[key] for key in ("accuracy", "macro_f1", "false_action_count")},
            })
            key = (metrics["false_action_count"], -metrics["macro_f1"], -metrics["accuracy"], regularization, threshold)
            if best_key is None or key < best_key:
                best_key = key
                selected = (classifier, regularization, metrics, predictions)
    classifier, regularization, metrics, predictions = selected
    raw_predictions = classifier.model.predict([row.text for row in validation]).tolist()
    guarded_predictions = [classifier.predict(row.text).intent for row in validation]
    metadata = {
        "seed": 42, "training_count": len(train), "validation_count": len(validation),
        "test_evaluated": False,
        "dataset_sha256": hashlib.sha256((DATA_DIRECTORY / "requests.jsonl").read_bytes()).hexdigest(),
        "splits_sha256": hashlib.sha256((DATA_DIRECTORY / "splits.json").read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "scipy", "joblib")},
        "selected_C": regularization, "clarification_threshold": classifier.min_probability,
        "selection_policy": "Minimize final-pipeline validation false Actions, then maximize macro F1 and accuracy; prefer smaller C and threshold on ties.",
        "features": "Word TF-IDF unigrams and bigrams, sublinear term frequency; training text only; no stop-word removal.",
        "candidates": candidates, "raw_model_validation": intent_metrics(truth, raw_predictions),
        "guarded_intent_validation": intent_metrics(truth, guarded_predictions),
        "pipeline_validation": metrics,
        "validation_errors": [
            {"id": row.id, "text": row.text, "expected": row.intent, "predicted": prediction}
            for row, prediction in zip(validation, predictions) if row.intent != prediction
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": classifier.model, "min_probability": classifier.min_probability, "metadata": metadata}, output_path)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DATA_DIRECTORY.parent / "reports" / "validation.json")
    args = parser.parse_args()
    metadata = train_model(args.output, args.report)
    metrics = metadata["pipeline_validation"]
    print(f"Trained on {metadata['training_count']} requests; validated on {metadata['validation_count']}.")
    print(f"Selected C={metadata['selected_C']}, clarification threshold={metadata['clarification_threshold']}.")
    print(f"Validation accuracy={metrics['accuracy']:.3f}, macro F1={metrics['macro_f1']:.3f}, false Actions={metrics['false_action_count']}.")
    print("Test predictions were not computed. Restart a running app after retraining its cached model.")


if __name__ == "__main__":
    main()
