from __future__ import annotations

import argparse
import json
from pathlib import Path

from coral_thesis.config import load_config
from coral_thesis.pipeline import CoralPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="coral-thesis V2 command line tools")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to a YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-config", help="Print the resolved configuration.")
    subparsers.add_parser("validate-config", help="Validate configuration paths and values.")
    subparsers.add_parser("bootstrap", help="Create workspace artifact directories.")
    subparsers.add_parser(
        "describe-pipeline",
        help="Print a high-level description of the current V2 pipeline.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config))
    pipeline = CoralPipeline(config)

    if args.command == "show-config":
        print(json.dumps(config.to_dict(), indent=2))
        return

    if args.command == "validate-config":
        issues = config.validate()
        if issues:
            print("Config validation failed:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("Config validation passed.")
        return

    if args.command == "bootstrap":
        pipeline.bootstrap()
        print(f"Workspace initialized at {config.project_root}")
        return

    if args.command == "describe-pipeline":
        print(pipeline.describe())
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

