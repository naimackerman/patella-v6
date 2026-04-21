"""Extract osteophyte features from all ROI patches using trained grader."""

from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.transforms import get_eval_transforms
from src.features.bootstrap_heuristics import ROI_SITES, heuristic_osteophyte_features
from src.models.osteophyte_grader import OsteophyteGrader
from src.utils.checkpoints import extract_model_state_dict, load_checkpoint, resolve_osteophyte_checkpoint_paths
from src.utils.device import get_device, clear_memory
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    ckpt_dir = Path(cfg.checkpoint_dir) / "osteophyte"
    override_cfg = getattr(cfg.training, "osteophyte_checkpoint_overrides", {})
    selection_cfg = getattr(cfg.training, "osteophyte_checkpoint_selection", {})
    override_paths = {
        str(site_name): str(path_value)
        for site_name, path_value in dict(override_cfg).items()
        if path_value is not None
    } if override_cfg is not None else {}
    prefer_refined = bool(selection_cfg.get("prefer_refined", True))
    force_multitask_sites = [str(site_name) for site_name in selection_cfg.get("force_multitask_sites", [])]
    force_refined_sites = [str(site_name) for site_name in selection_cfg.get("force_refined_sites", [])]
    resolved_ckpts = resolve_osteophyte_checkpoint_paths(
        ckpt_dir,
        ROI_SITES,
        prefer_refined=prefer_refined,
        force_multitask_sites=force_multitask_sites,
        force_refined_sites=force_refined_sites,
        override_paths_by_site=override_paths,
    )
    model_cache: dict[str, OsteophyteGrader] = {}

    def load_model(ckpt_path: Path) -> OsteophyteGrader:
        cache_key = str(ckpt_path)
        if cache_key not in model_cache:
            from omegaconf import OmegaConf
            model_cfg = cfg.get("model", None)
            if model_cfg is None:
                model_cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "model" / "se_resnet50.yaml")
            checkpoint = load_checkpoint(ckpt_path, map_location=device)
            state_dict = extract_model_state_dict(checkpoint)

            # Detect old bare-Linear heads (e.g. heads.medial_femur.weight)
            # vs new Sequential heads (e.g. heads.medial_femur.1.weight)
            has_old_heads = any(
                k.startswith("heads.") and k.count(".") == 2
                for k in state_dict
            )
            if has_old_heads:
                model_cfg = OmegaConf.merge(model_cfg, {"use_mlp_heads": False})

            model = OsteophyteGrader(model_cfg)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_cache[cache_key] = model
        return model_cache[cache_key]

    transform = get_eval_transforms(cfg)
    roi_dir = Path(str(getattr(cfg, "osteophyte_roi_dir", Path(cfg.feature_dir) / "rois")))
    output_dir = Path(cfg.feature_dir) / "osteophyte"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg.data.root)

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        all_features = {}

        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue

            for img_path in tqdm(sorted(grade_dir.glob("*.png")),
                                 desc=f"Osteophyte {split}/{grade_dir.name}"):
                image_id = img_path.stem

                # Load 4 ROI patches for this image
                roi_images = {}
                roi_tensors = {}
                for site in ROI_SITES:
                    roi_path = roi_dir / split / f"{image_id}_{site}.png"
                    if roi_path.exists():
                        roi_img = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
                    else:
                        roi_img = np.zeros((140, 140), dtype=np.uint8)
                    roi_images[site] = roi_img

                    transformed = transform(image=roi_img)
                    roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float()
                    roi_tensors[site] = roi_tensor.to(device)

                if len(resolved_ckpts) == len(ROI_SITES):
                    feature_grades = {}
                    with torch.no_grad():
                        for site in ROI_SITES:
                            ckpt_meta = resolved_ckpts.get(site)
                            if ckpt_meta is None:
                                continue
                            model = load_model(Path(ckpt_meta["path"]))
                            logits = model.forward_single(roi_tensors[site], site)
                            abbrev = {"medial_femur": "mf", "lateral_femur": "lf",
                                      "medial_tibia": "mt", "lateral_tibia": "lt"}[site]
                            feature_grades[f"osp_grade_{abbrev}"] = float(logits.argmax(dim=1).item())
                    mf = feature_grades["osp_grade_mf"]
                    lf = feature_grades["osp_grade_lf"]
                    mt = feature_grades["osp_grade_mt"]
                    lt = feature_grades["osp_grade_lt"]
                    features = {
                        **feature_grades,
                        "osp_sum": mf + lf + mt + lt,
                        "osp_max": max(mf, lf, mt, lt),
                        "osp_medial_sum": mf + mt,
                        "osp_lateral_sum": lf + lt,
                        "osp_femoral_sum": mf + lf,
                        "osp_tibial_sum": mt + lt,
                    }
                else:
                    features = heuristic_osteophyte_features(roi_images)

                all_features[image_id] = np.array([
                    features["osp_grade_mf"], features["osp_grade_lf"],
                    features["osp_grade_mt"], features["osp_grade_lt"],
                    features["osp_sum"], features["osp_max"],
                    features["osp_medial_sum"], features["osp_lateral_sum"],
                    features["osp_femoral_sum"], features["osp_tibial_sum"],
                ], dtype=np.float64)

        np.savez(
            str(output_dir / f"{split}_osteophyte_features.npz"),
            **all_features,
        )
        print(f"Saved osteophyte features for {len(all_features)} images ({split})")
        clear_memory()


if __name__ == "__main__":
    main()
