"""Evaluate an aggregated feature directory with the transparent XGBoost path."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.features.feature_aggregator import FeatureAggregator
from src.models.kl_xgboost import KLXGBoostClassifier
from src.utils.metrics import per_class_metrics, quadratic_weighted_kappa
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    model_cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "xgboost.yaml"
    return OmegaConf.load(model_cfg_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)

    feature_dir = Path(cfg.feature_dir) / "aggregated"
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(feature_dir / "train_features.npz", allow_pickle=True)
    test_data = np.load(feature_dir / "test_features.npz", allow_pickle=True)

    X_train, y_train = train_data["features"], train_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    aggregator = FeatureAggregator()
    aggregator.fit_normalizer(X_train)
    X_train = aggregator.normalize(X_train)
    X_test = aggregator.normalize(X_test)

    model = KLXGBoostClassifier(model_cfg)
    cv_scores = model.fit_cv(X_train, y_train, n_folds=model_cfg.cv_folds)
    preds, probs = model.predict(X_test)
    qwk = quadratic_weighted_kappa(y_test, preds)
    metrics = per_class_metrics(y_test, preds, cfg.data.class_names)

    summary = {
        "feature_dir": str(feature_dir),
        "cv_qwk_mean": float(cv_scores["qwk_mean"]),
        "cv_qwk_std": float(cv_scores["qwk_std"]),
        "test_qwk": float(qwk),
        "test_accuracy": float(metrics["accuracy"]),
        "per_class_f1": {
            cls_name: float(metrics.get(f"f1_{cls_name}", 0.0))
            for cls_name in cfg.data.class_names
        },
    }
    out_path = result_dir / "xgboost_baseline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
