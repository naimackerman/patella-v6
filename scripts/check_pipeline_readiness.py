"""Audit KOA-TriFQ implementation and study-stage readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.pipeline_readiness import assess_pipeline_readiness, format_readiness_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root to audit.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    report = assess_pipeline_readiness(args.project_root)
    if args.format == "json":
        print(json.dumps(report, indent=2))
        return
    print(format_readiness_report(report))


if __name__ == "__main__":
    main()
