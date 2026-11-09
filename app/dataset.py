"""Read labelled examples and verify that evaluation splits do not leak groups."""

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field

from app.schemas import DataModel, Intent, Preferences, Text


DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
SPLIT_NAMES = ("train", "validation", "test")
INTENTS = ("information", "discovery", "decision", "action", "clarification")


class LabelledRequest(DataModel):
    id: Text
    group_id: Text
    text: Annotated[Text, Field(max_length=2000)]
    intent: Intent
    tags: list[Text] = Field(min_length=1)
    rationale: Text
    expected_preferences: Preferences | None = None


class DatasetSplits(DataModel):
    seed: int
    train: list[Text]
    validation: list[Text]
    test: list[Text]


def load_requests(path: Path = DATA_DIRECTORY / "requests.jsonl") -> list[LabelledRequest]:
    examples = [
        LabelledRequest.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not examples:
        raise ValueError("The labelled dataset must not be empty.")
    ids = [example.id for example in examples]
    texts = [" ".join(example.text.casefold().split()) for example in examples]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise ValueError("Request IDs and normalized texts must be unique.")
    group_intents: dict[str, str] = {}
    for example in examples:
        previous = group_intents.setdefault(example.group_id, example.intent)
        if previous != example.intent:
            raise ValueError("A paraphrase group must have a single intent label.")
    return examples


def validate_splits(examples: list[LabelledRequest], splits: DatasetSplits) -> None:
    by_id = {example.id: example for example in examples}
    assigned = [identifier for name in SPLIT_NAMES for identifier in getattr(splits, name)]
    if len(assigned) != len(set(assigned)):
        raise ValueError("A request must not appear in multiple splits or more than once.")
    if set(assigned) != set(by_id):
        raise ValueError("Splits must cover exactly the labelled request IDs.")
    group_splits: dict[str, str] = {}
    for name in SPLIT_NAMES:
        identifiers = getattr(splits, name)
        if {by_id[identifier].intent for identifier in identifiers} != set(INTENTS):
            raise ValueError("Every split must contain all five intent classes.")
        for identifier in identifiers:
            group = by_id[identifier].group_id
            previous = group_splits.setdefault(group, name)
            if previous != name:
                raise ValueError("Related paraphrases must stay in the same split.")


def load_splits(
    examples: list[LabelledRequest], path: Path = DATA_DIRECTORY / "splits.json",
) -> DatasetSplits:
    splits = DatasetSplits.model_validate_json(path.read_text(encoding="utf-8"))
    validate_splits(examples, splits)
    return splits
