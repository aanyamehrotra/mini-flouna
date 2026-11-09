"""Run a baseline request without an API or trained intent classifier."""

import argparse

from app.pipeline import process_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="English food request")
    parser.add_argument("--intent", choices=["discovery", "decision"], required=True)
    args = parser.parse_args()
    print(process_request(args.text, intent=args.intent).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
