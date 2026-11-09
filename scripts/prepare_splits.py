"""Generate or verify fixed, class-balanced splits without separating paraphrases."""

import argparse
import random
from collections import Counter

from app.dataset import (
    DATA_DIRECTORY, INTENTS, SPLIT_NAMES, DatasetSplits, LabelledRequest,
    load_requests, load_splits, validate_splits,
)


def make_splits(examples: list[LabelledRequest], seed: int = 42) -> DatasetSplits:
    rng = random.Random(seed)
    assignments = {name: [] for name in SPLIT_NAMES}
    for intent in INTENTS:
        groups = sorted({example.group_id for example in examples if example.intent == intent})
        if len(groups) < 5:
            raise ValueError("At least five paraphrase groups per intent are needed.")
        rng.shuffle(groups)
        train_end = int(len(groups) * 0.6)
        validation_end = train_end + int(len(groups) * 0.2)
        partitions = (groups[:train_end], groups[train_end:validation_end], groups[validation_end:])
        for name, partition in zip(SPLIT_NAMES, partitions):
            assignments[name].extend(sorted(
                example.id for example in examples if example.group_id in partition
            ))
    splits = DatasetSplits(seed=seed, **assignments)
    validate_splits(examples, splits)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Explicitly regenerate the split manifest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_requests()
    expected = make_splits(examples, seed=args.seed)
    path = DATA_DIRECTORY / "splits.json"
    if args.write:
        path.write_text(expected.model_dump_json(indent=2) + "\n", encoding="utf-8")
    else:
        if load_splits(examples, path) != expected:
            raise ValueError("Saved splits differ from deterministic regeneration. Do not change test splits to improve results.")
    by_id = {example.id: example for example in examples}
    for name in SPLIT_NAMES:
        identifiers = getattr(expected, name)
        counts = dict(sorted(Counter(by_id[identifier].intent for identifier in identifiers).items()))
        print(f"{name}: {len(identifiers)} requests; {counts}")


if __name__ == "__main__":
    main()
