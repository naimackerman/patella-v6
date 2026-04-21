"""Build a compact reproducibility report from pipeline outputs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.visualization import plot_confusion_matrix, plot_roc_curves


KL_CLASS_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _copy_if_exists(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(source.read_bytes())


def _build_summary(run_root: Path) -> dict[str, Any]:
    result_root = run_root / "results"
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "result_root": str(result_root),
    }

    jsn = _load_json(result_root / "jsn" / "jsn_evaluation.json")
    if jsn is not None:
        summary["jsn"] = {
            "checkpoint": jsn.get("checkpoint"),
            "dice_mean": jsn.get("dice_mean"),
            "dice_medial_mean": jsn.get("dice_medial_mean"),
            "dice_lateral_mean": jsn.get("dice_lateral_mean"),
            "hausdorff95_mean": jsn.get("hausdorff95_mean"),
            "mjsw_mae": jsn.get("mjsw_mae"),
            "mjsw_icc_medial": jsn.get("mjsw_icc_medial"),
            "mjsw_icc_lateral": jsn.get("mjsw_icc_lateral"),
            "mjsw_icc_mean": jsn.get("mjsw_icc_mean"),
        }

    osp = _load_json(result_root / "stage2b_manual_mixed" / "osteophyte_evaluation.json")
    if osp is not None:
        summary["osteophyte"] = osp

    scl = _load_json(result_root / "stage2c_manual" / "sclerosis_evaluation.json")
    if scl is not None:
        summary["sclerosis"] = scl

    path_a = _load_json(result_root / "path_a" / "xgboost_metrics.json")
    if path_a is not None:
        summary["path_a"] = path_a

    path_b = _load_json(result_root / "kl_hybrid_evaluation" / "kl_hybrid_evaluation.json")
    if path_b is not None:
        summary["path_b"] = path_b

    return summary


def _write_stage_tables(report_dir: Path, summary: dict[str, Any]) -> None:
    module_rows: list[dict[str, Any]] = []

    jsn = summary.get("jsn")
    if jsn:
        module_rows.append({
            "module": "jsn",
            "primary_metric": "dice_mean",
            "primary_value": jsn.get("dice_mean"),
            "secondary_metric": "mjsw_mae",
            "secondary_value": jsn.get("mjsw_mae"),
            "tertiary_metric": "mjsw_icc_mean",
            "tertiary_value": jsn.get("mjsw_icc_mean"),
        })

    osp = summary.get("osteophyte", {})
    if osp:
        for site, payload in osp.items():
            test_metrics = payload.get("test", {})
            module_rows.append({
                "module": f"osteophyte::{site}",
                "primary_metric": "kappa",
                "primary_value": test_metrics.get("kappa"),
                "secondary_metric": "balanced_accuracy",
                "secondary_value": test_metrics.get("balanced_accuracy"),
                "tertiary_metric": "auc_macro",
                "tertiary_value": test_metrics.get("auc_macro"),
            })

    scl = summary.get("sclerosis", {})
    if scl:
        hybrid_test = scl.get("hybrid", {}).get("test", {})
        module_rows.append({
            "module": "sclerosis::hybrid",
            "primary_metric": "accuracy",
            "primary_value": hybrid_test.get("accuracy"),
            "secondary_metric": "auc_macro",
            "secondary_value": hybrid_test.get("auc_macro"),
            "tertiary_metric": "kl_correlation",
            "tertiary_value": hybrid_test.get("kl_correlation"),
        })

    _write_csv(
        report_dir / "stage_module_metrics.csv",
        module_rows,
        ["module", "primary_metric", "primary_value", "secondary_metric", "secondary_value", "tertiary_metric", "tertiary_value"],
    )

    osp_rows: list[dict[str, Any]] = []
    if osp:
        for site, payload in osp.items():
            test_metrics = payload.get("test", {})
            osp_rows.append({
                "site": site,
                "checkpoint_mode": payload.get("checkpoint_mode"),
                "checkpoint": payload.get("checkpoint"),
                "test_kappa": test_metrics.get("kappa"),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_auc_macro": test_metrics.get("auc_macro"),
            })
    _write_csv(
        report_dir / "osteophyte_site_metrics.csv",
        osp_rows,
        ["site", "checkpoint_mode", "checkpoint", "test_kappa", "test_balanced_accuracy", "test_auc_macro"],
    )

    path_rows: list[dict[str, Any]] = []
    path_a = summary.get("path_a", {}).get("test", {})
    if path_a:
        path_rows.append({
            "path": "path_a_xgboost",
            "qwk": path_a.get("qwk"),
            "accuracy": path_a.get("accuracy"),
            "f1_macro": path_a.get("f1_macro"),
            "auc_macro": path_a.get("auc_macro"),
        })
    path_b = summary.get("path_b", {}).get("test", {})
    if path_b:
        path_rows.append({
            "path": "path_b_hybrid",
            "qwk": path_b.get("qwk"),
            "accuracy": path_b.get("accuracy"),
            "f1_macro": path_b.get("f1_macro"),
            "auc_macro": path_b.get("auc_macro"),
        })
    _write_csv(
        report_dir / "kl_path_comparison.csv",
        path_rows,
        ["path", "qwk", "accuracy", "f1_macro", "auc_macro"],
    )

    per_class_rows: list[dict[str, Any]] = []
    if path_a:
        for class_name in KL_CLASS_NAMES:
            per_class_rows.append({
                "path": "path_a_xgboost",
                "class_name": class_name,
                "f1": path_a.get(f"f1_{class_name}"),
                "precision": path_a.get(f"precision_{class_name}"),
                "recall": path_a.get(f"recall_{class_name}"),
            })
    if path_b:
        for class_name in KL_CLASS_NAMES:
            per_class_rows.append({
                "path": "path_b_hybrid",
                "class_name": class_name,
                "f1": path_b.get(f"f1_{class_name}"),
                "precision": path_b.get(f"precision_{class_name}"),
                "recall": path_b.get(f"recall_{class_name}"),
            })
    _write_csv(
        report_dir / "kl_per_class_metrics.csv",
        per_class_rows,
        ["path", "class_name", "f1", "precision", "recall"],
    )


def _write_markdown(report_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Reproducibility Report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
    ]

    jsn = summary.get("jsn")
    if jsn:
        lines.extend([
            "## Stage 2A JSN",
            "",
            f"- Checkpoint: `{jsn.get('checkpoint')}`",
            f"- Dice mean: `{jsn.get('dice_mean'):.4f}`",
            f"- mJSW MAE: `{jsn.get('mjsw_mae'):.4f}`",
            f"- Mean ICC: `{jsn.get('mjsw_icc_mean'):.4f}`",
            "",
        ])

    osp = summary.get("osteophyte", {})
    if osp:
        lines.extend(["## Stage 2B Osteophyte", ""])
        for site, payload in osp.items():
            test_metrics = payload.get("test", {})
            lines.append(
                f"- `{site}`: kappa `{test_metrics.get('kappa', float('nan')):.4f}`, "
                f"balanced accuracy `{test_metrics.get('balanced_accuracy', float('nan')):.4f}`, "
                f"AUC `{test_metrics.get('auc_macro', float('nan')):.4f}`"
            )
        lines.append("")

    scl = summary.get("sclerosis", {})
    if scl:
        hybrid_test = scl.get("hybrid", {}).get("test", {})
        lines.extend([
            "## Stage 2C Sclerosis",
            "",
            f"- Accuracy: `{hybrid_test.get('accuracy', float('nan')):.4f}`",
            f"- AUC macro: `{hybrid_test.get('auc_macro', float('nan')):.4f}`",
            f"- KL correlation: `{hybrid_test.get('kl_correlation', float('nan')):.4f}`",
            "",
        ])

    path_a = summary.get("path_a", {}).get("test", {})
    path_b = summary.get("path_b", {}).get("test", {})
    if path_a or path_b:
        lines.extend(["## KL Pipeline", ""])
        if path_a:
            lines.append(
                f"- Path A XGBoost: QWK `{path_a.get('qwk', float('nan')):.4f}`, "
                f"accuracy `{path_a.get('accuracy', float('nan')):.4f}`"
            )
        if path_b:
            lines.append(
                f"- Path B Hybrid: QWK `{path_b.get('qwk', float('nan')):.4f}`, "
                f"accuracy `{path_b.get('accuracy', float('nan')):.4f}`, "
                f"AUC `{path_b.get('auc_macro', float('nan')):.4f}`"
            )
        if path_a and path_b:
            winner = "Path B Hybrid" if float(path_b.get("qwk", -1)) >= float(path_a.get("qwk", -1)) else "Path A XGBoost"
            lines.append(f"- Preferred final KL path: `{winner}`")
        lines.append("")

    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _build_plots(report_dir: Path, run_root: Path) -> None:
    result_root = run_root / "results"

    # Copy Path A plots/XAI if present.
    for name in [
        "xgboost_confusion_matrix.png",
        "xgboost_roc_curves.png",
        "shap_importance.png",
        "shap_waterfall_sample.png",
    ]:
        _copy_if_exists(result_root / "path_a" / name, report_dir / name)
        _copy_if_exists(result_root / "pipeline_evaluation" / name, report_dir / name)

    # Generate Path B plots from the saved predictions.
    path_b_npz = result_root / "kl_hybrid_evaluation" / "hybrid_test_predictions.npz"
    if path_b_npz.exists():
        payload = np.load(path_b_npz, allow_pickle=True)
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        y_prob = payload["y_prob"]
        plot_confusion_matrix(
            y_true,
            y_pred,
            KL_CLASS_NAMES,
            str(report_dir / "hybrid_confusion_matrix.png"),
        )
        plot_roc_curves(
            y_true,
            y_prob,
            KL_CLASS_NAMES,
            str(report_dir / "hybrid_roc_curves.png"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a consolidated reproducibility report.")
    parser.add_argument("--run-root", required=True, help="Root directory for one pipeline run (contains results/, outputs/, checkpoints/, features/).")
    parser.add_argument("--report-dir", default=None, help="Optional explicit report output directory.")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else (run_root / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(run_root)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_stage_tables(report_dir, summary)
    _write_markdown(report_dir, summary)
    _build_plots(report_dir, run_root)

    print(f"Saved report to {report_dir}")


if __name__ == "__main__":
    main()
