"""Run a request with automatic intent classification or a manual debug override."""

import argparse

from app.pipeline import process_request
from app.dataset import INTENTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="English food request")
    parser.add_argument("--intent", choices=INTENTS, help="Optional manual intent override for debugging")
    args = parser.parse_args()
    try:
        response = process_request(args.text, intent=args.intent)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
