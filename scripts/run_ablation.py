"""Ablation studies: evaluate XGBoost with different feature subsets."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.features.feature_aggregator import FeatureAggregator
from src.models.kl_xgboost import KLXGBoostClassifier
from src.utils.metrics import quadratic_weighted_kappa, per_class_metrics
from src.utils.visualization import plot_confusion_matrix
from src.utils.seed import seed_everything


ABLATION_CONFIGS = {
    "jsn_only": {
        "description": "JSN features only (22 dims)",
        "feature_range": (0, 22),
    },
    "jsn_osp": {
        "description": "JSN + Osteophyte features (32 dims)",
        "feature_range": (0, 32),
    },
    "full": {
        "description": "Full feature set (50 dims)",
        "feature_range": (0, 50),
    },
    "osp_only": {
        "description": "Osteophyte features only (10 dims)",
        "feature_range": (22, 32),
    },
    "scl_only": {
        "description": "Sclerosis features only (18 dims)",
        "feature_range": (32, 50),
    },
    "osp_scl": {
        "description": "Osteophyte + Sclerosis features (28 dims)",
        "feature_range": (22, 50),
    },
}


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    model_cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "xgboost.yaml"
    return OmegaConf.load(model_cfg_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)

    result_dir = Path(cfg.result_dir) / "ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(cfg.feature_dir) / "aggregated"

    # Load features
    train_data = np.load(feature_dir / "train_features.npz")
    val_data = np.load(feature_dir / "val_features.npz")
    test_data = np.load(feature_dir / "test_features.npz")

    X_train_full, y_train = train_data["features"], train_data["labels"]
    X_val_full, y_val = val_data["features"], val_data["labels"]
    X_test_full, y_test = test_data["features"], test_data["labels"]

    aggregator = FeatureAggregator()
    feature_names = aggregator.get_feature_names()

    results_table = []

    for name, config in ABLATION_CONFIGS.items():
        start, end = config["feature_range"]
        desc = config["description"]

        print(f"\n{'='*60}")
        print(f"Ablation: {desc}")
        print(f"{'='*60}")

        X_train = X_train_full[:, start:end]
        X_val = X_val_full[:, start:end]
        X_test = X_test_full[:, start:end]

        # Normalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        # Train XGBoost
        classifier = KLXGBoostClassifier(model_cfg)
        cv_scores = classifier.fit_cv(X_train, y_train, n_folds=model_cfg.cv_folds)
        print(f"CV QWK: {cv_scores['qwk_mean']:.4f} +/- {cv_scores['qwk_std']:.4f}")

        # Evaluate on test set
        preds, probs = classifier.predict(X_test)
        qwk = quadratic_weighted_kappa(y_test, preds)
        metrics = per_class_metrics(y_test, preds, cfg.data.class_names)

        print(f"Test QWK:      {qwk:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")

        plot_confusion_matrix(
            y_test, preds, cfg.data.class_names,
            str(result_dir / f"cm_{name}.png"),
        )

        results_table.append({
            "ablation": name,
            "description": desc,
            "n_features": end - start,
            "cv_qwk_mean": cv_scores["qwk_mean"],
            "cv_qwk_std": cv_scores["qwk_std"],
            "test_qwk": qwk,
            "test_accuracy": metrics["accuracy"],
        })

    # Print summary table
    print(f"\n{'='*80}")
    print("Ablation Study Results Summary")
    print(f"{'='*80}")
    print(f"{'Ablation':<15} {'Dims':>5} {'CV QWK':>12} {'Test QWK':>10} {'Test Acc':>10}")
    print(f"{'-'*55}")

    for r in results_table:
        cv_str = f"{r['cv_qwk_mean']:.4f}+/-{r['cv_qwk_std']:.4f}"
        print(f"{r['ablation']:<15} {r['n_features']:>5} {cv_str:>12} "
              f"{r['test_qwk']:>10.4f} {r['test_accuracy']:>10.4f}")

    # Save results
    np.savez(
        str(result_dir / "ablation_results.npz"),
        **{r["ablation"]: np.array([r["test_qwk"], r["test_accuracy"]])
           for r in results_table},
    )
    print(f"\nResults saved to: {result_dir}")


if __name__ == "__main__":
    main()
