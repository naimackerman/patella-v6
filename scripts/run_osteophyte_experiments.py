"""Run or print the recommended osteophyte improvement experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "features" / "jsn_v9_measurement_fix"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


EXPERIMENTS = OrderedDict(
    {
        "e1_noflip_manual": {
            "description": "Disable horizontal flips while keeping the full manual training set.",
            "train_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
            ],
        },
        "e2_mediumconf_manual": {
            "description": "Train on medium/high-confidence manual labels and evaluate on the full manual split.",
            "train_overrides": [
                "training.label_mode=manual",
                "training.annotation_confidence.min_train=medium",
                "training.annotation_confidence.min_eval=low",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "training.annotation_confidence.min_eval=low",
            ],
        },
        "e3_noflip_mediumconf_classweight": {
            "description": "Combine no-flip augmentation, medium-confidence training labels, and per-head class-balanced ordinal loss.",
            "train_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_train=medium",
                "training.annotation_confidence.min_eval=low",
                "training.osteophyte_class_balance.enabled=true",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_eval=low",
            ],
        },
        "e4_lightaug_mediumconf_classweight": {
            "description": "The most conservative quality-oriented run: no flip, lighter geometry, medium-confidence training labels, and per-head class weighting.",
            "train_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "preprocessing.augmentation.rotation_limit=5",
                "preprocessing.augmentation.translate_pct=0.02",
                "preprocessing.augmentation.scale_range=[0.95,1.05]",
                "training.annotation_confidence.min_train=medium",
                "training.annotation_confidence.min_eval=low",
                "training.osteophyte_class_balance.enabled=true",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_eval=low",
            ],
        },
        "e5_noflip_lowconf_classweight_balancedsampler": {
            "description": "Use all manual labels, keep no-flip augmentation, enable per-head class weighting, and balance multitask batches by per-site rarity.",
            "train_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_train=low",
                "training.annotation_confidence.min_eval=low",
                "training.osteophyte_class_balance.enabled=true",
                "training.osteophyte_sampling.strategy=mean_class_balance",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_eval=low",
            ],
        },
        "e6_noflip_lowconf_classweight_confcapsampler": {
            "description": "Use all manual labels, but damp low-confidence rare cases with confidence-aware capped multitask sampling.",
            "train_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_train=low",
                "training.annotation_confidence.min_eval=low",
                "training.osteophyte_class_balance.enabled=true",
                "training.osteophyte_sampling.strategy=mean_class_balance",
                "training.osteophyte_sampling.use_confidence_weights=true",
                "training.osteophyte_sampling.confidence_power=0.5",
                "training.osteophyte_sampling.max_weight_ratio_to_median=1.75",
            ],
            "eval_overrides": [
                "training.label_mode=manual",
                "preprocessing.augmentation.horizontal_flip_p=0.0",
                "training.annotation_confidence.min_eval=low",
            ],
        },
    }
)


def _python_executable() -> str:
    if DEFAULT_PYTHON.exists():
        return str(DEFAULT_PYTHON)
    return sys.executable


def _experiment_paths(name: str) -> tuple[Path, Path]:
    base = PROJECT_ROOT / "results" / "osteophyte_experiments" / name
    return PROJECT_ROOT / "checkpoints" / "osteophyte_experiments" / name, base


def _base_overrides(feature_dir: Path) -> list[str]:
    return [
        "+model=se_resnet50",
        f"feature_dir={feature_dir}",
    ]


def _build_command(script_name: str, overrides: list[str]) -> list[str]:
    return [_python_executable(), str(PROJECT_ROOT / "scripts" / script_name), *overrides]


def _format_command(cmd: list[str]) -> str:
    env_prefix = f"PYTHONPATH={shlex.quote(str(PROJECT_ROOT))} PROJECT_ROOT={shlex.quote(str(PROJECT_ROOT))}"
    return f"{env_prefix} {shlex.join(cmd)}"


def _run_command(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _load_summary(result_dir: Path) -> dict[str, object]:
    eval_path = result_dir / "osteophyte_evaluation.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    per_site = {}
    test_kappas = []
    for site_name, site_metrics in data.items():
        test_metrics = site_metrics.get("test", {})
        value = float(test_metrics.get("kappa", float("nan")))
        per_site[site_name] = value
        if value == value:
            test_kappas.append(value)
    mean_test_kappa = float(sum(test_kappas) / len(test_kappas)) if test_kappas else float("nan")
    return {
        "mean_test_kappa": mean_test_kappa,
        "per_site_test_kappa": per_site,
        "evaluation_path": str(eval_path),
    }


def _print_summary_table(summaries: dict[str, dict[str, object]]) -> None:
    if not summaries:
        return
    print("\nSummary")
    print("-" * 78)
    print(f"{'Experiment':<36} {'Mean Test Kappa':>16} {'LF':>8} {'LT':>8}")
    print("-" * 78)
    ranked = sorted(
        summaries.items(),
        key=lambda item: item[1].get("mean_test_kappa", float("-inf")),
        reverse=True,
    )
    for name, summary in ranked:
        per_site = summary["per_site_test_kappa"]
        print(
            f"{name:<36} "
            f"{summary['mean_test_kappa']:>16.4f} "
            f"{per_site.get('lateral_femur', float('nan')):>8.4f} "
            f"{per_site.get('lateral_tibia', float('nan')):>8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["e1_noflip_manual", "e2_mediumconf_manual", "e3_noflip_mediumconf_classweight", "e4_lightaug_mediumconf_classweight"],
        help="Experiment names to print or run. Use 'all' for every preset.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the train/evaluate commands instead of just printing them.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DEFAULT_FEATURE_DIR,
        help="Feature directory containing the ROI crops to use.",
    )
    args = parser.parse_args()

    selected = list(EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}. Available: {list(EXPERIMENTS.keys())}")

    summaries: dict[str, dict[str, object]] = {}
    for name in selected:
        spec = EXPERIMENTS[name]
        checkpoint_root, result_root = _experiment_paths(name)
        common = _base_overrides(args.feature_dir)
        train_cmd = _build_command(
            "train_osteophyte_grader.py",
            common
            + spec["train_overrides"]
            + [f"checkpoint_dir={checkpoint_root}", f"output_dir={PROJECT_ROOT / 'outputs' / 'osteophyte_experiments' / name}"],
        )
        eval_cmd = _build_command(
            "evaluate_osteophyte_grader.py",
            common
            + spec["eval_overrides"]
            + [f"checkpoint_dir={checkpoint_root}", f"result_dir={result_root}"],
        )

        print(f"\n{name}")
        print(spec["description"])
        print("Train:")
        print(_format_command(train_cmd))
        print("Eval:")
        print(_format_command(eval_cmd))

        if not args.execute:
            continue

        _run_command(train_cmd)
        _run_command(eval_cmd)
        summaries[name] = _load_summary(result_root)
        print(
            f"Result: mean_test_kappa={summaries[name]['mean_test_kappa']:.4f} "
            f"evaluation={summaries[name]['evaluation_path']}"
        )

    _print_summary_table(summaries)


if __name__ == "__main__":
    main()
