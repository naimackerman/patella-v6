"""Select manual-only or self-trained JSN checkpoint based on reviewed-mask evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSN evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _score(summary: dict, primary_metric: str) -> float:
    value = summary.get(primary_metric)
    if value is None:
        raise ValueError(f"Metric '{primary_metric}' not found in {summary.get('checkpoint')}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-json", required=True)
    parser.add_argument("--selftrain-json", required=True)
    parser.add_argument("--output", default="checkpoints/jsn_segmenter_selected.txt")
    parser.add_argument("--summary-output", default="results/jsn_selected_checkpoint.json")
    parser.add_argument("--primary-metric", default="mjsw_mae")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    args = parser.parse_args()

    manual = _load_summary(Path(args.manual_json))
    selftrain = _load_summary(Path(args.selftrain_json))
    manual_score = _score(manual, args.primary_metric)
    selftrain_score = _score(selftrain, args.primary_metric)
    use_selftrain = selftrain_score < manual_score if args.mode == "min" else selftrain_score > manual_score
    selected = selftrain if use_selftrain else manual
    selected_name = "selftrain" if use_selftrain else "manual"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(selected["checkpoint"]) + "\n", encoding="utf-8")

    summary = {
        "selected": selected_name,
        "selected_checkpoint": selected["checkpoint"],
        "primary_metric": args.primary_metric,
        "mode": args.mode,
        "manual_score": manual_score,
        "selftrain_score": selftrain_score,
        "manual_checkpoint": manual["checkpoint"],
        "selftrain_checkpoint": selftrain["checkpoint"],
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Selected checkpoint path written to {output}")


if __name__ == "__main__":
    main()
