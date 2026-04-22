"""Benchmark classical classifiers for binary sclerosis texture features."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.annotation_confidence import confidence_at_least
from src.utils.annotation_paths import MANUAL_SOURCES
from src.utils.seed import seed_everything
from src.utils.sclerosis_labels import map_sclerosis_grades


SIDE_NAMES = {0: "medial", 1: "lateral"}


def _load_manual_split(scl_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(scl_dir / f"{split}_sclerosis_data.npz", allow_pickle=True)
    sources = np.asarray(data["label_sources"]).astype(str)
    keep = np.isin(sources, list(MANUAL_SOURCES))
    if "confidence_levels" in data.files:
        confidences = np.asarray(data["confidence_levels"]).astype(str)
        keep &= np.asarray([confidence_at_least(conf, "low") for conf in confidences], dtype=bool)
    if not keep.any():
        raise ValueError(f"No manual sclerosis rows found for split={split} in {scl_dir}")
    x = np.asarray(data["texture_features"], dtype=np.float64)[keep]
    y = map_sclerosis_grades(np.asarray(data["grades"], dtype=np.int64)[keep], "binary_present")
    if "side_ids" in data.files:
        sides = np.asarray(data["side_ids"], dtype=np.int64)[keep]
    else:
        image_ids = np.asarray(data["image_ids"]).astype(str)[keep]
        sides = np.asarray([0 if image_id.endswith("_medial") else 1 for image_id in image_ids], dtype=np.int64)
    return x, y, sides


def _add_side_onehot(x: np.ndarray, sides: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return x
    return np.concatenate([x, np.eye(2, dtype=np.float64)[sides]], axis=1)


def _candidate_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "logreg_l2": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")),
        ]),
        "logreg_l1": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", penalty="l1")),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=seed,
            n_jobs=2,
        )
    except Exception as exc:
        print(f"Skipping XGBoost: {type(exc).__name__}: {exc}")
    return models


def _positive_probs(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        return np.asarray(probs[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = np.asarray(model.predict(x), dtype=np.float64)
    return preds


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y_pred = (probs >= float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_none": float(precision[0]),
        "precision_present": float(precision[1]),
        "recall_none": float(recall[0]),
        "recall_present": float(recall[1]),
        "f1_none": float(f1[0]),
        "f1_present": float(f1[1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        result["auc"] = float(roc_auc_score(y_true, probs))
    return result


def _threshold_grid(probs: np.ndarray) -> np.ndarray:
    base = np.linspace(0.05, 0.95, 181, dtype=np.float64)
    return np.unique(np.concatenate([base, probs]))


def _select_threshold(y_true: np.ndarray, probs: np.ndarray, objective: str) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = _metrics_at_threshold(y_true, probs, best_threshold)
    for threshold in _threshold_grid(probs):
        metrics = _metrics_at_threshold(y_true, probs, float(threshold))
        score = float(metrics.get(objective, metrics["f1_macro"]))
        best_score = float(best_metrics.get(objective, best_metrics["f1_macro"]))
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _side_threshold_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    sides: np.ndarray,
    objective: str,
) -> tuple[dict[str, float], dict, dict[str, dict]]:
    thresholds = {}
    val_by_side = {}
    preds = np.zeros_like(y_true)
    for side_value, side_name in SIDE_NAMES.items():
        mask = sides == side_value
        if not mask.any() or len(np.unique(y_true[mask])) < 2:
            threshold = 0.5
            metrics = _metrics_at_threshold(y_true[mask], probs[mask], threshold)
        else:
            threshold, metrics = _select_threshold(y_true[mask], probs[mask], objective)
        thresholds[side_name] = float(threshold)
        val_by_side[side_name] = metrics
        preds[mask] = (probs[mask] >= threshold).astype(np.int64)
    combined_probs = preds.astype(np.float64)
    combined = _metrics_at_threshold(y_true, combined_probs, 0.5)
    combined["thresholds"] = thresholds
    if len(np.unique(y_true)) > 1:
        combined["auc"] = float(roc_auc_score(y_true, probs))
    return thresholds, combined, val_by_side


def _apply_side_thresholds(y_true: np.ndarray, probs: np.ndarray, sides: np.ndarray, thresholds: dict[str, float]) -> dict:
    preds = np.zeros_like(y_true)
    for side_value, side_name in SIDE_NAMES.items():
        mask = sides == side_value
        preds[mask] = (probs[mask] >= float(thresholds[side_name])).astype(np.int64)
    metrics = _metrics_at_threshold(y_true, preds.astype(np.float64), 0.5)
    metrics["thresholds"] = {name: float(value) for name, value in thresholds.items()}
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, probs))
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    scl_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    benchmark_cfg = getattr(cfg.training, "sclerosis_texture_benchmark", {})
    objective = str(getattr(benchmark_cfg, "threshold_objective", "f1_macro"))
    add_side_feature = bool(getattr(benchmark_cfg, "add_side_feature", True))

    x_train, y_train, side_train = _load_manual_split(scl_dir, "train")
    x_val, y_val, side_val = _load_manual_split(scl_dir, "val")
    x_test, y_test, side_test = _load_manual_split(scl_dir, "test")
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    side_trainval = np.concatenate([side_train, side_val], axis=0)

    x_train_m = _add_side_onehot(x_train, side_train, add_side_feature)
    x_val_m = _add_side_onehot(x_val, side_val, add_side_feature)
    x_test_m = _add_side_onehot(x_test, side_test, add_side_feature)
    x_trainval_m = _add_side_onehot(x_trainval, side_trainval, add_side_feature)

    results = []
    detailed = {
        "objective": objective,
        "add_side_feature": add_side_feature,
        "splits": {
            "train": {"num_samples": int(len(y_train)), "class_counts": np.bincount(y_train, minlength=2).tolist()},
            "val": {"num_samples": int(len(y_val)), "class_counts": np.bincount(y_val, minlength=2).tolist()},
            "test": {"num_samples": int(len(y_test)), "class_counts": np.bincount(y_test, minlength=2).tolist()},
        },
        "models": {},
    }

    for name, model in _candidate_models(int(cfg.seed)).items():
        print(f"Training {name}...")
        model.fit(x_train_m, y_train)
        val_probs = _positive_probs(model, x_val_m)
        test_probs = _positive_probs(model, x_test_m)
        threshold, val_metrics = _select_threshold(y_val, val_probs, objective)
        _, val_side_metrics, val_by_side = _side_threshold_metrics(y_val, val_probs, side_val, objective)

        model_trainval = _candidate_models(int(cfg.seed))[name]
        model_trainval.fit(x_trainval_m, y_trainval)
        test_probs_refit = _positive_probs(model_trainval, x_test_m)

        test_global = _metrics_at_threshold(y_test, test_probs, threshold)
        test_side = _apply_side_thresholds(
            y_test,
            test_probs,
            side_test,
            {side: float(value) for side, value in val_side_metrics["thresholds"].items()},
        )
        test_refit_global = _metrics_at_threshold(y_test, test_probs_refit, threshold)

        detailed["models"][name] = {
            "threshold": float(threshold),
            "val": val_metrics,
            "val_side_specific": val_side_metrics,
            "val_by_side": val_by_side,
            "test": test_global,
            "test_side_specific": test_side,
            "test_refit_trainval_same_threshold": test_refit_global,
        }
        results.append({
            "model": name,
            "threshold": float(threshold),
            "val_f1_macro": val_metrics["f1_macro"],
            "val_auc": val_metrics.get("auc", np.nan),
            "test_f1_macro": test_global["f1_macro"],
            "test_auc": test_global.get("auc", np.nan),
            "test_balanced_accuracy": test_global["balanced_accuracy"],
            "test_recall_present": test_global["recall_present"],
            "test_recall_none": test_global["recall_none"],
            "test_side_f1_macro": test_side["f1_macro"],
            "test_refit_f1_macro": test_refit_global["f1_macro"],
            "test_refit_auc": test_refit_global.get("auc", np.nan),
        })

    summary = pd.DataFrame(results).sort_values(
        ["test_f1_macro", "test_auc", "test_balanced_accuracy"],
        ascending=False,
    )
    summary_path = result_dir / "sclerosis_texture_benchmark_summary.csv"
    json_path = result_dir / "sclerosis_texture_benchmark.json"
    summary.to_csv(summary_path, index=False)
    detailed["ranking"] = summary.to_dict(orient="records")
    json_path.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Saved benchmark summary to {summary_path}")
    print(f"Saved benchmark details to {json_path}")


if __name__ == "__main__":
    main()
