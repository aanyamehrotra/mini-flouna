"""Evaluate a frozen model without fitting or selecting any parameters."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import joblib

from app.dataset import DATA_DIRECTORY, INTENTS, LabelledRequest, load_requests, load_splits
from app.extraction import extract_preferences
from app.intent import DEFAULT_MODEL, IntentClassifier
from app.pipeline import process_request
from app.schemas import Preferences
from scripts.train import intent_metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data_hashes(metadata: dict, directory: Path = DATA_DIRECTORY) -> None:
    for key, filename in (("dataset_sha256", "requests.jsonl"), ("splits_sha256", "splits.json")):
        if metadata.get(key) != sha256(directory / filename):
            raise ValueError(f"{filename} differs from the model's training metadata. Evaluation requires the same frozen dataset and splits.")


def normalize_field(field: str, value):
    if field == "unsupported_constraints":
        return sorted({item.split(":", 1)[0].strip().casefold() for item in value})
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return value


def preference_metrics(pairs: list[tuple[LabelledRequest, Preferences]]) -> dict:
    fields = {
        field: {"correct": 0, "total": 0, "present_correct": 0, "present_total": 0,
                "missing_correct": 0, "missing_total": 0}
        for field in Preferences.model_fields
    }
    errors = []
    exact_requests = 0
    empty_baseline_correct = 0
    empty = Preferences().model_dump()
    for row, predicted in pairs:
        if row.expected_preferences is None:
            raise ValueError("Preference scoring requires an annotated request.")
        gold = row.expected_preferences.model_dump()
        actual = predicted.model_dump()
        differences = {}
        for field, expected in gold.items():
            matched = normalize_field(field, expected) == normalize_field(field, actual[field])
            present = expected is not None and expected != []
            category = "present" if present else "missing"
            counts = fields[field]
            counts["total"] += 1
            counts["correct"] += int(matched)
            counts[f"{category}_total"] += 1
            counts[f"{category}_correct"] += int(matched)
            empty_baseline_correct += int(normalize_field(field, expected) == normalize_field(field, empty[field]))
            if not matched:
                differences[field] = {"expected": expected, "predicted": actual[field]}
        if differences:
            errors.append({"id": row.id, "text": row.text, "differences": differences})
        else:
            exact_requests += 1
    totals = {name: sum(counts[name] for counts in fields.values()) for name in (
        "correct", "total", "present_correct", "present_total", "missing_correct", "missing_total",
    )}
    for counts in [*fields.values(), totals]:
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else None
        for category in ("present", "missing"):
            counts[f"{category}_accuracy"] = (
                counts[f"{category}_correct"] / counts[f"{category}_total"]
                if counts[f"{category}_total"] else None
            )
    return {
        "annotated_requests": len(pairs), "exact_requests": exact_requests,
        "exact_request_accuracy": exact_requests / len(pairs) if pairs else None,
        "overall": totals, "per_field": fields,
        "empty_preferences_baseline_accuracy": empty_baseline_correct / totals["total"] if totals["total"] else None,
        "errors": errors,
    }


def plot_confusion_matrices(raw: dict, pipeline: dict, output: Path, split: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mini-flouna-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    maximum = max(max(row) for metrics in (raw, pipeline) for row in metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    labels = [label.title() for label in INTENTS]
    for axis, metrics, title in zip(axes, (raw, pipeline), ("Statistical model", "Full pipeline")):
        matrix = np.array(metrics["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_xticks(range(5), labels, rotation=35, ha="right")
        axis.set_yticks(range(5), labels)
        axis.set_xlabel("Predicted intent")
        axis.set_ylabel("Actual intent")
        axis.set_title(f"{title}\nAccuracy {metrics['accuracy']:.1%} | Macro F1 {metrics['macro_f1']:.3f}")
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, str(count), ha="center", va="center",
                      color="white" if count > maximum / 2 else "black", fontsize=12)
    fig.suptitle(f"Mini FLOUNA {split} split — counts, six examples per class", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def evaluate(model_path: Path, output_directory: Path, split: str = "test") -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation must use validation or test, never the training split.")
    examples = load_requests()
    splits = load_splits(examples)
    bundle = joblib.load(model_path)
    metadata = bundle["metadata"]
    check_data_hashes(metadata)
    classifier = IntentClassifier(bundle["model"], bundle["min_probability"])
    by_id = {row.id: row for row in examples}
    rows = [by_id[identifier] for identifier in getattr(splits, split)]
    predictions = []
    standalone_pairs = []
    pipeline_pairs = []
    extraction_clarifications = 0
    for row in rows:
        guarded = classifier.predict(row.text)
        result = process_request(row.text, classifier=classifier)
        predictions.append({
            "id": row.id, "group_id": row.group_id, "text": row.text,
            "expected_intent": row.intent, "raw_intent": guarded.details.raw_intent,
            "guarded_intent": guarded.intent, "final_intent": result.understood_request.intent,
            "status": result.status, "intent_details": result.intent_details.model_dump(),
            "recommendation_id": result.recommendation.id if result.recommendation else None,
            "expected_preferences": row.expected_preferences.model_dump() if row.expected_preferences else None,
            "predicted_preferences": result.understood_request.preferences.model_dump(),
        })
        if row.expected_preferences is not None:
            extracted = extract_preferences(row.text)
            extraction_clarifications += bool(extracted.questions)
            standalone_pairs.append((row, extracted.preferences))
            pipeline_pairs.append((row, result.understood_request.preferences))
    truth = [row.intent for row in rows]
    report = {
        "split": split, "request_count": len(rows), "test_evaluated": split == "test",
        "model_sha256": sha256(model_path), "dataset_sha256": metadata["dataset_sha256"],
        "splits_sha256": metadata["splits_sha256"], "model_selection": {
            key: metadata[key] for key in ("seed", "training_count", "validation_count", "selected_C", "clarification_threshold", "versions")
        },
        "python_version": platform.python_version(),
        "runtime_sha256": hashlib.sha256(b"".join(
            path.name.encode() + b"\0" + path.read_bytes()
            for path in sorted((DATA_DIRECTORY.parent / "app").glob("*.py"))
        )).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=DATA_DIRECTORY.parent, text=True
        ).strip(),
        "raw_model": intent_metrics(truth, [p["raw_intent"] for p in predictions]),
        "guarded_intent": intent_metrics(truth, [p["guarded_intent"] for p in predictions]),
        "full_pipeline": intent_metrics(truth, [p["final_intent"] for p in predictions]),
        "status_counts": dict(sorted(Counter(p["status"] for p in predictions).items())),
        "standalone_extraction": preference_metrics(standalone_pairs),
        "pipeline_extraction": preference_metrics(pipeline_pairs),
        "extraction_clarifications_on_annotated_requests": extraction_clarifications,
        "intent_errors": [p for p in predictions if p["expected_intent"] != p["final_intent"]],
        "policy": "No fitting, model selection, threshold changes, or runtime rule changes are performed by this evaluator.",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"{split}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_directory / f"{split}_predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8"
    )
    plot_confusion_matrices(report["raw_model"], report["full_pipeline"], output_directory / f"{split}_confusion_matrix.png", split)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIRECTORY.parent / "reports")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    report = evaluate(args.model, args.output_dir, args.split)
    metrics = report["full_pipeline"]
    print(f"{args.split}: {report['request_count']} requests; accuracy={metrics['accuracy']:.3f}, macro F1={metrics['macro_f1']:.3f}, false Actions={metrics['false_action_count']}.")
    extraction = report["standalone_extraction"]
    print(f"Standalone extraction: {extraction['exact_requests']}/{extraction['annotated_requests']} exact requests; present-field accuracy={extraction['overall']['present_accuracy']:.3f}.")


if __name__ == "__main__":
    main()
