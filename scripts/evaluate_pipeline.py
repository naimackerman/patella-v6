"""End-to-end evaluation of xrAI-OA pipeline on the test set."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.models.kl_xgboost import KLXGBoostClassifier
from src.utils.checkpoints import checkpoint_score, load_checkpoint
from src.utils.metrics import quadratic_weighted_kappa, per_class_metrics
from src.utils.visualization import plot_confusion_matrix, plot_roc_curves
from src.utils.seed import seed_everything


def _load_model_cfg(model_name: str):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model" / f"{model_name}.yaml"
    return OmegaConf.load(config_path)


def _requested_model_name(cfg: DictConfig) -> str:
    if "model" in cfg and cfg.model is not None:
        model_name = getattr(cfg.model, "name", None)
        if model_name is not None:
            return str(model_name).lower()
    return ""


def _checkpoint_feature_dim(checkpoint: dict) -> int | None:
    model_cfg = checkpoint.get("hyper_parameters", {}).get("model")
    if model_cfg is None:
        return None
    value = (
        model_cfg.get("feature_dim")
        if isinstance(model_cfg, dict)
        else getattr(model_cfg, "feature_dim", None)
    )
    return int(value) if value is not None else None


def _find_best_compatible_hybrid_checkpoint(
    checkpoint_dir: Path,
    expected_feature_dim: int,
) -> Path | None:
    for monitor in ("val_qwk", "val/qwk"):
        candidates = []
        for checkpoint_path in sorted(checkpoint_dir.glob("*.ckpt")):
            score = checkpoint_score(checkpoint_path, monitor=monitor)
            if score == float("-inf"):
                continue
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
            feature_dim = _checkpoint_feature_dim(checkpoint)
            if feature_dim is not None and feature_dim != expected_feature_dim:
                continue
            candidates.append((score, checkpoint_path))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    requested_model_name = _requested_model_name(cfg)
    evaluate_path_a = ("xgboost" in requested_model_name) or ("convnext" not in requested_model_name and "hybrid" not in requested_model_name)
    evaluate_path_b = ("convnext" in requested_model_name) or ("hybrid" in requested_model_name) or requested_model_name == ""

    result_dir = Path(cfg.result_dir) / "pipeline_evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(cfg.feature_dir) / "aggregated"

    # Load test features
    test_data = np.load(feature_dir / "test_features.npz")
    X_test = test_data["features"]
    y_test = test_data["labels"]
    image_ids = test_data["image_ids"]

    # Normalize using training stats
    norm_stats = np.load(feature_dir / "normalizer_stats.npz")
    mean, std = norm_stats["mean"], norm_stats["std"]
    std[std < 1e-8] = 1.0
    X_test_norm = (X_test - mean) / std

    # --- Path A: XGBoost ---
    xgb_path = KLXGBoostClassifier.resolve_model_path(Path(cfg.checkpoint_dir) / "kl_xgboost")
    if evaluate_path_a and xgb_path.exists():
        print("\n" + "=" * 60)
        print("Path A: XGBoost Evaluation")
        print("=" * 60)

        xgb = KLXGBoostClassifier(_load_model_cfg("xgboost"))
        xgb.load(str(xgb_path))

        preds_xgb, probs_xgb = xgb.predict(X_test_norm)
        qwk_xgb = quadratic_weighted_kappa(y_test, preds_xgb)
        metrics_xgb = per_class_metrics(y_test, preds_xgb, cfg.data.class_names)

        print(f"XGBoost QWK:      {qwk_xgb:.4f}")
        print(f"XGBoost Accuracy: {metrics_xgb['accuracy']:.4f}")

        for cls_name in cfg.data.class_names:
            f1 = metrics_xgb.get(f"f1_{cls_name}", 0)
            print(f"  {cls_name}: F1={f1:.4f}")

        plot_confusion_matrix(
            y_test, preds_xgb, cfg.data.class_names,
            str(result_dir / "xgboost_confusion_matrix.png"),
        )
        plot_roc_curves(
            y_test, probs_xgb, cfg.data.class_names,
            str(result_dir / "xgboost_roc_curves.png"),
        )

        # Save per-image predictions
        np.savez(
            str(result_dir / "xgboost_predictions.npz"),
            image_ids=image_ids,
            y_true=y_test,
            y_pred=preds_xgb,
            y_prob=probs_xgb,
        )
    elif evaluate_path_a:
        print("XGBoost model not found, skipping Path A evaluation.")

    # --- Path B: Hybrid ConvNeXt ---
    hybrid_ckpt_dir = Path(cfg.checkpoint_dir) / "kl_hybrid"
    hybrid_ckpt = None
    if evaluate_path_b and hybrid_ckpt_dir.exists():
        hybrid_ckpt = _find_best_compatible_hybrid_checkpoint(hybrid_ckpt_dir, X_test.shape[1])

    if hybrid_ckpt is not None:
        print("\n" + "=" * 60)
        print("Path B: Hybrid ConvNeXt Evaluation")
        print("=" * 60)

        from src.models.kl_hybrid import HybridKLClassifier
        from src.data.kl_dataset import KLHybridDataset
        from src.data.transforms import get_eval_transforms
        from src.utils.checkpoints import extract_model_state_dict
        from src.utils.device import get_device, clear_memory
        from torch.utils.data import DataLoader
        import torch

        device = get_device()
        checkpoint = load_checkpoint(hybrid_ckpt, map_location=device)
        module = HybridKLClassifier(_load_model_cfg("convnext_hybrid"))
        module.load_state_dict(extract_model_state_dict(checkpoint))
        module.to(device)
        module.eval()

        val_transform = get_eval_transforms(cfg)
        test_ds = KLHybridDataset(
            cfg.data.root,
            "test",
            str(feature_dir / "test_features.npz"),
            val_transform,
        )
        test_loader = DataLoader(
            test_ds, batch_size=4, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
        )
        hybrid_image_ids = np.asarray([Path(path).stem for path in test_ds.image_ds.samples])

        all_preds = []
        all_probs = []
        all_labels = []

        for images, features, labels in tqdm(test_loader, desc="Hybrid inference"):
            images = images.to(device)
            features_tensor = features.to(device)

            with torch.no_grad():
                logits = module(images, features_tensor)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

        preds_hybrid = np.array(all_preds)
        probs_hybrid = np.vstack(all_probs)
        labels_hybrid = np.array(all_labels)

        qwk_hybrid = quadratic_weighted_kappa(labels_hybrid, preds_hybrid)
        metrics_hybrid = per_class_metrics(labels_hybrid, preds_hybrid, cfg.data.class_names)

        print(f"Hybrid QWK:      {qwk_hybrid:.4f}")
        print(f"Hybrid Accuracy: {metrics_hybrid['accuracy']:.4f}")

        for cls_name in cfg.data.class_names:
            f1 = metrics_hybrid.get(f"f1_{cls_name}", 0)
            print(f"  {cls_name}: F1={f1:.4f}")

        plot_confusion_matrix(
            labels_hybrid, preds_hybrid, cfg.data.class_names,
            str(result_dir / "hybrid_confusion_matrix.png"),
        )
        plot_roc_curves(
            labels_hybrid, probs_hybrid, cfg.data.class_names,
            str(result_dir / "hybrid_roc_curves.png"),
        )

        np.savez(
            str(result_dir / "hybrid_predictions.npz"),
            image_ids=hybrid_image_ids,
            y_true=labels_hybrid,
            y_pred=preds_hybrid,
            y_prob=probs_hybrid,
        )

        clear_memory()
    elif evaluate_path_b:
        print("Hybrid model not found, skipping Path B evaluation.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Test set size: {len(y_test)} images")
    if evaluate_path_a and xgb_path.exists():
        print(f"Path A (XGBoost) QWK: {qwk_xgb:.4f}")
    if evaluate_path_b and hybrid_ckpt is not None:
        print(f"Path B (Hybrid)  QWK: {qwk_hybrid:.4f}")
    print(f"Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
