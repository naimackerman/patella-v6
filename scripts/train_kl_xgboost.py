"""Train XGBoost classifier on extracted feature vectors (Path A)."""

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.features.feature_aggregator import FeatureAggregator
from src.models.kl_xgboost import KLXGBoostClassifier
from src.xai.shap_explainer import SHAPExplainer
from src.utils.metrics import quadratic_weighted_kappa, per_class_metrics
from src.utils.visualization import plot_confusion_matrix
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

    feature_dir = Path(cfg.feature_dir)
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load extracted features (from extract_all_features.py)
    train_data = np.load(feature_dir / "aggregated" / "train_features.npz")
    val_data = np.load(feature_dir / "aggregated" / "val_features.npz")
    test_data = np.load(feature_dir / "aggregated" / "test_features.npz")

    X_train, y_train = train_data["features"], train_data["labels"]
    X_val, y_val = val_data["features"], val_data["labels"]
    X_test, y_test = test_data["features"], test_data["labels"]

    # Normalize features
    aggregator = FeatureAggregator()
    aggregator.fit_normalizer(X_train)
    X_train = aggregator.normalize(X_train)
    X_val = aggregator.normalize(X_val)
    X_test = aggregator.normalize(X_test)

    # Train with cross-validation
    classifier = KLXGBoostClassifier(model_cfg)
    cv_scores = classifier.fit_cv(X_train, y_train, n_folds=model_cfg.cv_folds)
    print(f"\nCV QWK: {cv_scores['qwk_mean']:.4f} +/- {cv_scores['qwk_std']:.4f}")

    # Evaluate on test set
    preds, probs = classifier.predict(X_test)
    qwk = quadratic_weighted_kappa(y_test, preds)
    metrics = per_class_metrics(y_test, preds, cfg.data.class_names)
    print(f"Test QWK: {qwk:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    summary = {
        "cv": cv_scores,
        "test": {
            "qwk": float(qwk),
            **{key: float(value) for key, value in metrics.items()},
        },
    }
    (result_dir / "xgboost_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez(
        str(result_dir / "xgboost_predictions.npz"),
        image_ids=test_data["image_ids"],
        y_true=y_test,
        y_pred=preds,
        y_prob=probs,
    )

    # Save model
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    classifier.save(str(ckpt_dir / "kl_xgboost.ubj"))

    # Confusion matrix
    plot_confusion_matrix(
        y_test, preds, cfg.data.class_names,
        str(result_dir / "xgboost_confusion_matrix.png"),
    )

    # SHAP analysis
    feature_names = aggregator.get_feature_names()
    explainer = SHAPExplainer(classifier.model, feature_names)
    explainer.global_importance(X_test, str(result_dir / "shap_importance.png"))
    explainer.waterfall(X_test[0], str(result_dir / "shap_waterfall_sample.png"))


if __name__ == "__main__":
    main()
