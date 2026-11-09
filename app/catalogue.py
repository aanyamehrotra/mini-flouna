"""Load and validate the local mock food catalogue."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from app.schemas import FoodOption


DEFAULT_CATALOGUE = Path(__file__).resolve().parent.parent / "data" / "options.json"


def load_catalogue(path: Path = DEFAULT_CATALOGUE) -> list[FoodOption]:
    options = TypeAdapter(list[FoodOption]).validate_python(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if not options:
        raise ValueError("The catalogue must contain at least one food option.")
    ids = [option.id for option in options]
    if len(ids) != len(set(ids)):
        raise ValueError("Food option IDs must be unique.")
    return options
