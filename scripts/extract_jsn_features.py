"""Extract JSN features from all images using trained segmentation model."""

from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.transforms import get_eval_transforms
from src.features.jsw_computation import (
    compute_all_jsn_features,
    get_jsn_measurement_kwargs,
    jsn_features_to_vector,
)
from src.features.bootstrap_heuristics import estimate_jsn_features
from src.models.jsn_segmenter import create_jsn_segmenter
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device, clear_memory
from src.utils.seed import seed_everything


def _resolve_selected_jsn_checkpoint(cfg: DictConfig) -> Path | None:
    selected_file_raw = getattr(
        cfg,
        "jsn_selected_checkpoint_file",
        Path(cfg.checkpoint_dir) / "jsn_segmenter_selected.txt",
    )
    if selected_file_raw in (None, "", "null", "None"):
        return None
    selected_file = Path(str(selected_file_raw))
    if selected_file.exists():
        selected = selected_file.read_text(encoding="utf-8").strip()
        if selected:
            selected_path = Path(selected)
            if selected_path.exists():
                return selected_path
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()
    measurement_kwargs = get_jsn_measurement_kwargs(cfg)

    # Load trained model
    explicit_ckpt = getattr(cfg, "checkpoint_path", None)
    if explicit_ckpt:
        ckpt_path = Path(str(explicit_ckpt))
    else:
        ckpt_path = _resolve_selected_jsn_checkpoint(cfg)
        if ckpt_path is None:
            checkpoint_subdir = str(getattr(cfg, "jsn_checkpoint_subdir", "jsn_segmenter"))
            ckpt_dir = Path(cfg.checkpoint_dir) / checkpoint_subdir
            monitor = getattr(cfg, "checkpoint_monitor", None)
            if monitor in (None, "", "null", "None"):
                monitor = "val_dice"
            monitor = str(monitor)
            mode = str(getattr(cfg, "checkpoint_mode", "max"))
            ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor=monitor, mode=mode) if ckpt_dir.exists() else None
    model = None
    if ckpt_path is not None:
        model = create_jsn_segmenter(cfg.model)
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        state_dict = extract_model_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    transform = get_eval_transforms(cfg)
    output_dir = Path(cfg.feature_dir) / "jsn"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = output_dir / "masks"
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_mask_dir = Path(cfg.annotation_dir) / "jsn_masks"
    label_mode = getattr(cfg.training, "label_mode", "manual")

    # Compute reference mJSW from KL-0 training images
    # (first pass on KL-0 subset to get median mJSW)
    data_root = Path(cfg.data.root)
    all_features = {}
    feature_vectors_by_split = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            for img_path in tqdm(sorted(grade_dir.glob("*.png")),
                                 desc=f"JSN {split}/{grade_dir.name}"):
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image_id = img_path.stem
                is_left = image_id.upper().endswith("L")
                reviewed_mask_path = reviewed_mask_dir / split / f"{image_id}.png"

                if str(label_mode).lower() in {"manual", "expanded", "auto"} and reviewed_mask_path.exists():
                    pred_mask = cv2.imread(str(reviewed_mask_path), cv2.IMREAD_GRAYSCALE)
                    features = compute_all_jsn_features(pred_mask, **measurement_kwargs)
                    np.save(str(mask_output_dir / f"{image_id}_mask.npy"), pred_mask.astype(np.uint8))
                elif model is not None:
                    transformed = transform(image=image)
                    img_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        logits = model(img_tensor)
                        pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
                    features = compute_all_jsn_features(pred_mask, **measurement_kwargs)
                    np.save(str(mask_output_dir / f"{image_id}_mask.npy"), pred_mask.astype(np.uint8))
                else:
                    features = estimate_jsn_features(image, is_left=is_left)

                all_features[image_id] = {
                    "features": features,
                    "grade": int(grade_dir.name),
                    "split": split,
                }

        clear_memory()

    # Compute reference mJSW from KL-0 training images
    kl0_medial = [f["features"]["mJSW_medial"] for f in all_features.values()
                  if f["grade"] == 0 and f["split"] == "train"]
    kl0_lateral = [f["features"]["mJSW_lateral"] for f in all_features.values()
                   if f["grade"] == 0 and f["split"] == "train"]

    ref_medial = float(np.median(kl0_medial)) if kl0_medial else 15.0
    ref_lateral = float(np.median(kl0_lateral)) if kl0_lateral else 15.0
    print(f"Reference mJSW - Medial: {ref_medial:.2f}, Lateral: {ref_lateral:.2f}")

    # Save calibrated reference values for inference
    import json
    ref_path = output_dir / "reference_mjsw.json"
    with open(ref_path, "w") as f:
        json.dump({
            "reference_mjsw_medial": ref_medial,
            "reference_mjsw_lateral": ref_lateral,
            "num_kl0_medial": len(kl0_medial),
            "num_kl0_lateral": len(kl0_lateral),
        }, f, indent=2)
    print(f"Saved reference mJSW to {ref_path}")

    # Recompute JSN rates with proper reference and save vectors
    for image_id, data in all_features.items():
        raw = data["features"]
        raw["jsn_rate_medial"] = 100.0 * (1.0 - raw["mJSW_medial"] / max(ref_medial, 1e-6))
        raw["jsn_rate_lateral"] = 100.0 * (1.0 - raw["mJSW_lateral"] / max(ref_lateral, 1e-6))
        raw["jsn_rate_medial"] = float(np.clip(raw["jsn_rate_medial"], 0, 100))
        raw["jsn_rate_lateral"] = float(np.clip(raw["jsn_rate_lateral"], 0, 100))
        split = data["split"]
        feature_vectors_by_split[split][image_id] = jsn_features_to_vector(raw)

    for split, feature_vectors in feature_vectors_by_split.items():
        np.savez(str(output_dir / f"{split}_jsn_features.npz"), **feature_vectors)
        print(f"Saved JSN features for {len(feature_vectors)} images ({split}) to {output_dir}")


if __name__ == "__main__":
    main()
