"""XGBoost classifier for KL grade prediction (Path A - Transparent)."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb
from omegaconf import DictConfig


class KLXGBoostClassifier:
    """XGBoost-based KL grade classifier with SHAP explainability.

    This is the transparent classification path that operates on the
    50-dim extracted feature vector.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = xgb.XGBClassifier(
            objective=cfg.objective,
            num_class=cfg.num_class,
            max_depth=cfg.max_depth,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            eval_metric=cfg.eval_metric,
            tree_method="hist",
            random_state=42,
        )
        self.cv_scores = None

    @staticmethod
    def resolve_model_path(path: str | Path) -> Path:
        """Resolve a user path to the actual XGBoost model artifact path."""
        candidate = Path(path)
        suffix = candidate.suffix.lower()

        if suffix in {".ubj", ".json"}:
            if candidate.exists():
                return candidate
            alternate_suffix = ".json" if suffix == ".ubj" else ".ubj"
            alternate = candidate.with_suffix(alternate_suffix)
            if alternate.exists():
                return alternate
            return candidate

        if candidate.exists():
            return candidate

        ubj_path = candidate.with_suffix(".ubj")
        json_path = candidate.with_suffix(".json")
        if ubj_path.exists():
            return ubj_path
        if json_path.exists():
            return json_path
        return ubj_path

    @staticmethod
    def resolve_metrics_path(path: str) -> Path:
        """Resolve the companion metrics metadata path."""
        model_path = KLXGBoostClassifier.resolve_model_path(path)
        return model_path.with_name(f"{model_path.stem}_cv_metrics.json")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the model.

        Args:
            X: (N, 50) training feature matrix.
            y: (N,) training KL grade labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
        """
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=True,
        )

    def fit_cv(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict[str, float]:
        """Train with stratified k-fold cross-validation.

        Returns:
            Dict with mean and std of QWK across folds.
        """
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_model = xgb.XGBClassifier(
                objective=self.cfg.objective,
                num_class=self.cfg.num_class,
                max_depth=self.cfg.max_depth,
                n_estimators=self.cfg.n_estimators,
                learning_rate=self.cfg.learning_rate,
                reg_alpha=self.cfg.reg_alpha,
                reg_lambda=self.cfg.reg_lambda,
                eval_metric=self.cfg.eval_metric,
                tree_method="hist",
                random_state=42,
            )
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            preds = fold_model.predict(X_val)
            qwk = cohen_kappa_score(y_val, preds, weights="quadratic")
            fold_scores.append(qwk)
            print(f"Fold {fold + 1}: QWK = {qwk:.4f}")

        # Train final model on all data
        self.model.fit(X, y, verbose=False)

        self.cv_scores = {
            "qwk_mean": float(np.mean(fold_scores)),
            "qwk_std": float(np.std(fold_scores)),
            "qwk_per_fold": fold_scores,
        }
        return self.cv_scores

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict KL grades and probabilities.

        Returns:
            (predictions, probabilities) - (N,) grades and (N, 5) class probs.
        """
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        return preds, probs

    def save(self, path: str):
        """Save model to file."""
        model_path = self.resolve_model_path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        if self.cv_scores:
            meta_path = self.resolve_metrics_path(path)
            with open(meta_path, "w") as f:
                json.dump(self.cv_scores, f, indent=2)

    def load(self, path: str):
        """Load model from file."""
        model_path = self.resolve_model_path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")
        self.model.load_model(model_path)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_
