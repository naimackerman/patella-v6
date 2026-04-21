"""SHAP explanation wrapper for XGBoost KL grade classifier."""

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


class SHAPExplainer:
    """SHAP-based explainability for XGBoost feature-based classifier."""

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Args:
            model: Trained XGBoost model (from KLXGBoostClassifier).
            feature_names: List of 50 feature names for plots.
        """
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def compute_shap_values(self, X: np.ndarray):
        """Compute SHAP values for a feature matrix.

        Args:
            X: (N, 50) feature matrix.

        Returns:
            SHAP values array.
        """
        return self.explainer.shap_values(X)

    def global_importance(self, X: np.ndarray, save_path: str, max_display: int = 20):
        """Generate and save global feature importance bar chart.

        Args:
            X: (N, 50) feature matrix (typically full training set).
            save_path: Path to save the plot.
            max_display: Number of top features to display.
        """
        shap_values = self.compute_shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def waterfall(self, X_single: np.ndarray, save_path: str, predicted_class: int = None):
        """Generate and save per-sample SHAP waterfall plot.

        Args:
            X_single: (1, 50) or (50,) single sample feature vector.
            save_path: Path to save the plot.
            predicted_class: Which class to explain. Defaults to predicted class.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_values = self.explainer(X_single)

        # For multi-output models (e.g. XGBoost multiclass), pick the predicted class
        if predicted_class is None:
            # Default to the class with highest base value
            if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
                predicted_class = int(shap_values.values[0].sum(axis=0).argmax())
            else:
                predicted_class = 0

        if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            sv = shap_values[:, :, predicted_class]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
