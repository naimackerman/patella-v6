"""Train Random Forest and MLP baselines on KOA-TriFQ feature vectors."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from src.utils.metrics import quadratic_weighted_kappa
from src.utils.seed import seed_everything


def _evaluate_classifier(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    result = {
        "qwk": float(quadratic_weighted_kappa(y_test, preds)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1, 2, 3, 4]).tolist(),
    }
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_test)
        try:
            result["auc_macro_ovr"] = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    feature_dir = Path(cfg.feature_dir) / "aggregated"
    train = np.load(feature_dir / "train_features.npz", allow_pickle=True)
    test = np.load(feature_dir / "test_features.npz", allow_pickle=True)
    stats = np.load(feature_dir / "normalizer_stats.npz")

    mean = stats["mean"]
    std = stats["std"]
    std[std < 1e-8] = 1.0
    x_train = (train["features"] - mean) / std
    x_test = (test["features"] - mean) / std
    y_train = train["labels"]
    y_test = test["labels"]

    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=cfg.seed,
    )

    summary = {
        "random_forest": _evaluate_classifier(rf, x_train, y_train, x_test, y_test),
        "mlp": _evaluate_classifier(mlp, x_train, y_train, x_test, y_test),
    }
    out_path = Path(cfg.result_dir) / "kl_feature_baselines.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved KL baseline comparison to {out_path}")


if __name__ == "__main__":
    main()
