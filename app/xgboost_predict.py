"""Subprocess-safe XGBoost prediction helper for the Gradio app."""

from __future__ import annotations

import json
import sys

import numpy as np
from omegaconf import OmegaConf

from src.models.kl_xgboost import KLXGBoostClassifier


def main() -> None:
    payload = json.loads(sys.stdin.read())
    model_path = payload["model_path"]
    features = np.asarray(payload["features"], dtype=np.float64).reshape(1, -1)

    cfg = OmegaConf.load("configs/model/xgboost.yaml")
    model = KLXGBoostClassifier(cfg)
    model.load(model_path)
    pred, prob = model.predict(features)

    print(json.dumps({
        "grade": int(pred[0]),
        "probabilities": prob[0].astype(float).tolist(),
    }))


if __name__ == "__main__":
    main()
