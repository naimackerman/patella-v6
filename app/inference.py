"""Full xrAI-OA pipeline inference on a single image."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.utils.checkpoints import (
    extract_model_state_dict,
    find_best_lightning_checkpoint,
    load_checkpoint,
    resolve_osteophyte_checkpoint_paths,
)
from src.utils.device import clear_memory, get_device


class TriFQPipeline:
    """End-to-end inference using trained models when available."""

    def __init__(self, checkpoint_dir: str, config=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or self._build_runtime_config()
        self.device = get_device()
        self._roi_detector = None
        self._jsn_model = None
        self._osteophyte_models = None
        self._sclerosis_model = None
        self._xgb_model = None
        self._hybrid_model = None

    def run(
        self,
        image_path: str,
        show_jsn: bool = True,
        show_jsn_medial: bool = True,
        show_jsn_lateral: bool = True,
        show_osteophytes: bool = True,
        show_sclerosis: bool = True,
        show_kl_badge: bool = True,
        display_preprocessing: str = "raw",
        distance_units: str = "px",
        pixel_spacing_mm: Optional[float] = None,
        kl_path: str = "auto",
    ) -> Dict:
        from src.data.transforms import get_eval_transforms
        from src.features.bootstrap_heuristics import (
            ROI_SITES,
            build_sclerosis_features,
            estimate_jsn_features,
            extract_geometric_subchondral_roi,
            extract_geometric_subchondral_roi_with_boxes,
            heuristic_osteophyte_features,
        )
        from src.features.feature_aggregator import FeatureAggregator
        from src.features.jsw_computation import compute_jsn_measurements
        from src.features.kneel_landmarks import (
            KNEELLandmarkDetector,
            estimate_jsn_features_from_landmarks,
            extract_kneel_subchondral_rois,
            extract_kneel_subchondral_rois_with_boxes,
        )
        from src.features.subchondral_roi import extract_subchondral_roi_with_boxes
        from src.features.texture_features import extract_all_texture_features
        from src.models.kl_hybrid import HybridKLClassifier
        from src.models.osteophyte_grader import OsteophyteGrader
        from src.models.roi_detector import ROIDetector
        from src.models.sclerosis_classifier import SclerosisClassifier
        from src.models.jsn_segmenter import create_jsn_segmenter
        from src.xai.feature_heatmaps import (
            compose_heatmap_canvas,
            extract_osteophyte_gradcam,
            extract_sclerosis_gradcam,
        )
        from src.xai.overlay_renderer import generate_xai_overlay
        from src.xai.report_generator import generate_clinical_report

        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        display_image = self._preprocess_display_image(image, display_preprocessing)

        image_id = image_path.stem
        is_left = image_id.upper().endswith("L")
        roi_strategy = getattr(self.config.preprocessing, "roi_strategy", "auto").lower()
        results = {"image_path": str(image_path)}
        bbox_by_site = {}
        landmark_detector = KNEELLandmarkDetector.from_preprocessing_cfg(self.config.preprocessing)

        detector = self._load_roi_detector(ROIDetector) if roi_strategy in {"auto", "detector"} else None
        landmark_state = None
        if detector is not None:
            detections = detector.predict(image)
            if len(detections) >= 3:
                rois = detector.extract_rois(image, detections)
                for det in detections:
                    x1, y1, x2, y2 = det["bbox_xyxy"]
                    bbox_by_site[det["class_name"]] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            else:
                rois, bbox_by_site, landmark_state = self._fallback_rois(
                    image=image,
                    is_left=is_left,
                    roi_strategy=roi_strategy,
                    landmark_detector=landmark_detector,
                )
        else:
            rois, bbox_by_site, landmark_state = self._fallback_rois(
                image=image,
                is_left=is_left,
                roi_strategy=roi_strategy,
                landmark_detector=landmark_detector,
            )
        clear_memory()

        transform = get_eval_transforms(self.config)
        jsn_model = self._load_jsn_model(create_jsn_segmenter)
        pred_mask = None
        measurement_pairs = []
        measurement_pairs_medial = []
        measurement_pairs_lateral = []
        ref_mjsw = self._load_reference_mjsw()
        if jsn_model is not None:
            transformed = transform(image=image)
            image_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                pred_mask = jsn_model(image_tensor).argmax(dim=1).squeeze().cpu().numpy()
            jsn_measurements = compute_jsn_measurements(
                pred_mask,
                reference_mjsw_medial=ref_mjsw["medial"],
                reference_mjsw_lateral=ref_mjsw["lateral"],
                **{
                    "internal_profile_points_per_compartment": int(
                        getattr(self.config.preprocessing.jsn_measurement, "internal_profile_points_per_compartment", 64)
                    ),
                    "contour_smoothing_window": int(
                        getattr(self.config.preprocessing.jsn_measurement, "contour_smoothing_window", 5)
                    ),
                },
            )
            jsn_features = {k: v for k, v in jsn_measurements.items() if k not in {"measurement_pairs", "measurement_pairs_lateral", "measurement_pairs_medial"}}
            measurement_pairs = jsn_measurements["measurement_pairs"]
            measurement_pairs_medial = jsn_measurements.get("measurement_pairs_medial", [])
            measurement_pairs_lateral = jsn_measurements.get("measurement_pairs_lateral", [])
        else:
            if landmark_state is None and roi_strategy in {"auto", "landmark"}:
                try:
                    landmark_state = landmark_detector.predict(image, is_left=is_left)
                except Exception:
                    landmark_state = None
            if landmark_state is not None:
                jsn_features = estimate_jsn_features_from_landmarks(landmark_state)
            else:
                jsn_features = estimate_jsn_features(image, is_left=is_left)
        results["jsn_features"] = jsn_features
        clear_memory()

        osteophyte_models = self._load_osteophyte_models(OsteophyteGrader)
        osp_heatmaps = []
        osteophyte_margin_boxes = self._osteophyte_margin_boxes_from_mask(pred_mask, image.shape) if pred_mask is not None else {}
        if osteophyte_models is not None and len(osteophyte_models) == len(ROI_SITES):
            osp_features = {}
            site_map = {
                "medial_femur": "mf",
                "lateral_femur": "lf",
                "medial_tibia": "mt",
                "lateral_tibia": "lt",
            }
            for site in ROI_SITES:
                roi_image = rois.get(site, np.zeros((140, 140), dtype=np.uint8))
                transformed = transform(image=roi_image)
                roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    logits = osteophyte_models[site].forward_single(roi_tensor, site)
                predicted_grade = int(logits.argmax(dim=1).item())
                osp_features[f"osp_grade_{site_map[site]}"] = float(predicted_grade)
                if show_osteophytes and site in bbox_by_site:
                    heatmap = extract_osteophyte_gradcam(
                        osteophyte_models[site],
                        site=site,
                        image_tensor=roi_tensor,
                        target_class=predicted_grade,
                    )
                    focus_bbox = osteophyte_margin_boxes.get(site) or self._osteophyte_focus_bbox(
                        site=site,
                        roi_bbox=bbox_by_site[site],
                        heatmap=None,
                    )
                    osp_heatmaps.append({
                        "site": site,
                        "bbox": bbox_by_site[site],
                        "focus_bbox": focus_bbox,
                        "grade": predicted_grade,
                        "heatmap": heatmap,
                    })

            mf = osp_features["osp_grade_mf"]
            lf = osp_features["osp_grade_lf"]
            mt = osp_features["osp_grade_mt"]
            lt = osp_features["osp_grade_lt"]
            osp_features.update({
                "osp_sum": mf + lf + mt + lt,
                "osp_max": max(mf, lf, mt, lt),
                "osp_medial_sum": mf + mt,
                "osp_lateral_sum": lf + lt,
                "osp_femoral_sum": mf + lf,
                "osp_tibial_sum": mt + lt,
            })
        else:
            osp_features = heuristic_osteophyte_features(rois)
        results["osp_features"] = osp_features
        clear_memory()

        sclerosis_model = self._load_sclerosis_model(SclerosisClassifier)
        medial_roi = lateral_roi = None
        medial_box = lateral_box = None
        if pred_mask is not None:
            medial_roi, lateral_roi, medial_box, lateral_box = extract_subchondral_roi_with_boxes(
                pred_mask,
                image,
                is_left=is_left,
            )
        if medial_roi is None or lateral_roi is None:
            if landmark_state is not None:
                medial_roi, lateral_roi, medial_box, lateral_box = extract_kneel_subchondral_rois_with_boxes(
                    image,
                    landmark_state,
                )
            else:
                medial_roi, lateral_roi, medial_box, lateral_box = extract_geometric_subchondral_roi_with_boxes(
                    image,
                    is_left=is_left,
                )

        scl_features = build_sclerosis_features(medial_roi, lateral_roi)
        scl_heatmap_items = []
        if sclerosis_model is not None:
            predicted_grades = {}
            for side, roi, box in [
                ("medial", medial_roi, medial_box),
                ("lateral", lateral_roi, lateral_box),
            ]:
                if roi is None:
                    continue
                transformed = transform(image=roi)
                roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(self.device)
                texture_tensor = torch.tensor(
                    extract_all_texture_features(roi),
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = sclerosis_model(roi_tensor, texture_tensor)
                predicted_grade = int(logits.argmax(dim=1).item())
                predicted_grades[side] = predicted_grade
                if show_sclerosis and box is not None:
                    heatmap = extract_sclerosis_gradcam(
                        sclerosis_model,
                        image_tensor=roi_tensor,
                        texture_tensor=texture_tensor,
                        target_class=predicted_grade,
                    )
                    if heatmap is not None:
                        scl_heatmap_items.append((heatmap, box))
            if "medial" in predicted_grades:
                scl_features["scl_grade_medial"] = predicted_grades["medial"]
            if "lateral" in predicted_grades:
                scl_features["scl_grade_lateral"] = predicted_grades["lateral"]
        results["scl_features"] = scl_features

        aggregator = FeatureAggregator()
        feature_vector = aggregator.aggregate(jsn_features, osp_features, scl_features)
        results["feature_vector"] = feature_vector

        norm_path = Path(self.config.feature_dir) / "aggregated" / "normalizer_stats.npz"
        if norm_path.exists():
            norm_stats = np.load(norm_path)
            mean, std = norm_stats["mean"], norm_stats["std"]
            std[std < 1e-8] = 1.0
            feature_vector_norm = (feature_vector - mean) / std
        else:
            feature_vector_norm = feature_vector

        kl_predictions = {
            "heuristic": self._heuristic_kl_prediction(jsn_features, osp_features, scl_features),
        }

        if self._xgboost_runtime_supported():
            from src.models.kl_xgboost import KLXGBoostClassifier

            xgb_prediction = self._predict_xgboost(KLXGBoostClassifier, feature_vector_norm)
            if xgb_prediction is not None:
                probabilities = xgb_prediction["probabilities"]
                kl_predictions["xgboost"] = {
                    "grade": int(xgb_prediction["grade"]),
                    "confidence": float(max(probabilities)),
                    "probabilities": probabilities,
                }

        hybrid_model = self._load_hybrid_model(HybridKLClassifier)
        if hybrid_model is not None:
            transformed_full = transform(image=image)
            full_image_tensor = (
                torch.from_numpy(transformed_full["image"])
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            feature_tensor = torch.tensor(
                np.asarray([feature_vector_norm], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                logits = hybrid_model(full_image_tensor, feature_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            kl_predictions["hybrid"] = {
                "grade": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                "probabilities": probs.tolist(),
            }

        kl_pred, kl_path_used = self._select_kl_prediction(kl_path, kl_predictions)

        results["kl_grade"] = kl_pred["grade"]
        results["kl_confidence"] = kl_pred["confidence"]
        results["kl_probabilities"] = kl_pred["probabilities"]
        results["kl_path_used"] = kl_path_used
        results["kl_predictions"] = kl_predictions

        jsn_viz = {
            "measurement_pairs": measurement_pairs,
            "measurement_pairs_medial": measurement_pairs_medial,
            "measurement_pairs_lateral": measurement_pairs_lateral,
            **jsn_features,
        }
        site_grade_keys = {
            "medial_femur": "osp_grade_mf",
            "lateral_femur": "osp_grade_lf",
            "medial_tibia": "osp_grade_mt",
            "lateral_tibia": "osp_grade_lt",
        }
        osp_viz = {
            "detections": [
                (site, int(round(float(osp_features[grade_key]))), self._resolve_osteophyte_display_bbox(site, bbox_by_site[site], osp_heatmaps))
                for site, grade_key in site_grade_keys.items()
                if site in bbox_by_site
            ],
            "heatmaps": osp_heatmaps,
        }
        scl_canvas = compose_heatmap_canvas(image.shape, scl_heatmap_items) if show_sclerosis else None
        scl_viz = {
            "heatmap": scl_canvas,
            "roi_extent": [0, image.shape[1], image.shape[0], 0],
        }
        results["jsn_viz"] = jsn_viz
        results["osp_viz"] = osp_viz
        results["scl_viz"] = scl_viz
        results["kl_pred"] = kl_pred

        overlay_fig = generate_xai_overlay(
            display_image,
            jsn_viz,
            osp_viz,
            scl_viz,
            kl_pred,
            show_jsn=show_jsn,
            show_jsn_medial=show_jsn_medial,
            show_jsn_lateral=show_jsn_lateral,
            show_osteophytes=show_osteophytes,
            show_sclerosis=show_sclerosis,
            show_kl_badge=show_kl_badge,
            distance_units=distance_units,
            pixel_spacing_mm=pixel_spacing_mm if pixel_spacing_mm is not None else self._display_pixel_spacing_mm(),
        )
        results["overlay_figure"] = overlay_fig
        results["clinical_report"] = generate_clinical_report(jsn_features, osp_features, scl_features, kl_pred)
        return results

    def render_overlay_from_results(
        self,
        image_path: str,
        results: Dict,
        show_jsn: bool = True,
        show_jsn_medial: bool = True,
        show_jsn_lateral: bool = True,
        show_osteophytes: bool = True,
        show_sclerosis: bool = True,
        show_kl_badge: bool = True,
        display_preprocessing: str = "raw",
        distance_units: str = "px",
        pixel_spacing_mm: Optional[float] = None,
    ):
        """Redraw the clinical overlay without recomputing model predictions."""
        from src.xai.overlay_renderer import generate_xai_overlay

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        display_image = self._preprocess_display_image(image, display_preprocessing)
        return generate_xai_overlay(
            display_image,
            results.get("jsn_viz", {}),
            results.get("osp_viz", {}),
            results.get("scl_viz", {}),
            results.get("kl_pred", {
                "grade": results.get("kl_grade", "?"),
                "confidence": results.get("kl_confidence", 0.0),
                "probabilities": results.get("kl_probabilities", []),
            }),
            show_jsn=show_jsn,
            show_jsn_medial=show_jsn_medial,
            show_jsn_lateral=show_jsn_lateral,
            show_osteophytes=show_osteophytes,
            show_sclerosis=show_sclerosis,
            show_kl_badge=show_kl_badge,
            distance_units=distance_units,
            pixel_spacing_mm=pixel_spacing_mm if pixel_spacing_mm is not None else self._display_pixel_spacing_mm(),
        )

    def _load_roi_detector(self, detector_cls):
        if self._roi_detector is not None:
            return self._roi_detector

        model_path = self.checkpoint_dir / "roi_detector" / "weights" / "best.pt"
        if model_path.exists():
            self._roi_detector = detector_cls(str(model_path))
        return self._roi_detector

    @staticmethod
    def _fallback_rois(image, is_left: bool, roi_strategy: str, landmark_detector):
        from src.models.roi_detector import ROIDetector

        try_landmark = roi_strategy in {"auto", "landmark"}
        if try_landmark:
            try:
                landmark_boxes_xyxy = ROIDetector.landmark_boxes(
                    image,
                    is_left=is_left,
                    landmark_detector=landmark_detector,
                )
                rois = ROIDetector.crop_from_boxes(image, landmark_boxes_xyxy)
                bbox_by_site = {
                    site: (x1, y1, x2 - x1, y2 - y1)
                    for site, (x1, y1, x2, y2) in landmark_boxes_xyxy.items()
                    if site != "joint_space"
                }
                landmark_state = landmark_detector.predict(image, is_left=is_left)
                return rois, bbox_by_site, landmark_state
            except Exception:
                pass

        geometric_boxes = ROIDetector.geometric_boxes(image.shape[:2], is_left=is_left)
        rois = ROIDetector.crop_from_boxes(image, geometric_boxes)
        bbox_by_site = {
            site: (x1, y1, x2 - x1, y2 - y1)
            for site, (x1, y1, x2, y2) in geometric_boxes.items()
            if site != "joint_space"
        }
        return rois, bbox_by_site, None

    def _load_jsn_model(self, factory):
        if self._jsn_model is not None:
            return self._jsn_model

        ckpt_dir = self.checkpoint_dir / "jsn_segmenter"
        ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor="val/dice") if ckpt_dir.exists() else None
        if ckpt_path is None:
            return None

        model = factory(self._get_stage_cfg("jsn"))
        checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        model.to(self.device)
        model.eval()
        self._jsn_model = model
        return self._jsn_model

    def _load_osteophyte_models(self, model_cls):
        if self._osteophyte_models is not None:
            return self._osteophyte_models

        ckpt_dir = self.checkpoint_dir / "osteophyte"
        if not ckpt_dir.exists():
            return None

        models = {}
        training_cfg = getattr(self.config, "training", None)
        override_cfg = getattr(training_cfg, "osteophyte_checkpoint_overrides", {})
        override_paths = {
            str(site_name): str(path_value)
            for site_name, path_value in dict(override_cfg).items()
            if path_value is not None
        } if override_cfg is not None else {}
        resolved_ckpts = resolve_osteophyte_checkpoint_paths(
            ckpt_dir,
            ("medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"),
            override_paths_by_site=override_paths,
        )
        if len(resolved_ckpts) != 4:
            return None

        model_cache = {}
        for site, ckpt_meta in resolved_ckpts.items():
            ckpt_path = str(ckpt_meta["path"])
            if ckpt_path not in model_cache:
                model = model_cls(self._get_stage_cfg("osteophyte"))
                checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
                model.load_state_dict(extract_model_state_dict(checkpoint))
                model.to(self.device)
                model.eval()
                model_cache[ckpt_path] = model
            models[site] = model_cache[ckpt_path]

        self._osteophyte_models = models
        return self._osteophyte_models

    def _load_sclerosis_model(self, model_cls):
        if self._sclerosis_model is not None:
            return self._sclerosis_model

        checkpoint_candidates = [
            (self.checkpoint_dir / "sclerosis_binary_texture_only" / "sclerosis", "val_auc_macro"),
            (self.checkpoint_dir / "sclerosis" / "sclerosis", "val_auc_macro"),
            (self.checkpoint_dir / "sclerosis", "val/accuracy"),
            (self.checkpoint_dir / "sclerosis", "val_f1_macro"),
        ]
        ckpt_path = None
        for ckpt_dir, monitor in checkpoint_candidates:
            if not ckpt_dir.exists():
                continue
            ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor=monitor)
            if ckpt_path is not None:
                break
        if ckpt_path is None:
            return None

        checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
        model_cfg = self._checkpoint_model_cfg("sclerosis", checkpoint)
        model = model_cls(model_cfg)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        model.to(self.device)
        model.eval()
        self._sclerosis_model = model
        return self._sclerosis_model

    def _load_xgboost_model(self, model_cls):
        if self._xgb_model is not None:
            return self._xgb_model

        model_path = model_cls.resolve_model_path(self.checkpoint_dir / "kl_xgboost")
        if not model_path.exists():
            return None

        model = model_cls(self._get_stage_cfg("xgboost"))
        model.load(str(model_path))
        self._xgb_model = model
        return self._xgb_model

    def _predict_xgboost(self, model_cls, feature_vector_norm: np.ndarray) -> Optional[Dict]:
        model_path = model_cls.resolve_model_path(self.checkpoint_dir / "kl_xgboost")
        if not model_path.exists():
            return None

        # Keep XGBoost in a short-lived subprocess for app stability. Some macOS
        # Python/native-library combinations can segfault in xgboost.load_model.
        helper = Path(__file__).with_name("xgboost_predict.py")
        payload = json.dumps({
            "model_path": str(model_path),
            "features": np.asarray(feature_vector_norm, dtype=np.float64).reshape(-1).tolist(),
        })
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        try:
            completed = subprocess.run(
                [sys.executable, str(helper)],
                input=payload,
                text=True,
                capture_output=True,
                check=True,
                cwd=str(repo_root),
                env=env,
                timeout=60,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return json.loads(completed.stdout)

    @staticmethod
    def _preprocess_display_image(image: np.ndarray, mode: str) -> np.ndarray:
        normalized = str(mode or "raw").strip().lower()
        if normalized in {"raw", "none"}:
            return image
        if normalized == "clahe":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image.astype(np.uint8))
        if normalized in {"clip", "histogram_clip"}:
            low, high = np.percentile(image, [1, 99])
            clipped = np.clip(image.astype(np.float32), low, high)
            if high > low:
                clipped = (clipped - low) / (high - low) * 255.0
            return clipped.astype(np.uint8)
        if normalized in {"clip_clahe", "histogram_clip_clahe"}:
            clipped = TriFQPipeline._preprocess_display_image(image, "clip")
            return TriFQPipeline._preprocess_display_image(clipped, "clahe")
        return image

    @staticmethod
    def _osteophyte_margin_boxes_from_mask(
        pred_mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Estimate marginal osteophyte display boxes from JSN compartment anatomy.

        These boxes are visualization anchors for OARSI site predictions. They
        deliberately target the peripheral femoral/tibial joint margins instead
        of the larger classifier ROI crops.
        """
        from src.features.jsw_computation import extract_contours

        h, w = image_shape[:2]
        compartments = {
            "medial": 1,
            "lateral": 2,
        }
        contours = {}
        centers = {}
        for side, class_id in compartments.items():
            femoral, tibial = extract_contours(pred_mask, class_id=class_id)
            if femoral is None or tibial is None:
                continue
            all_points = np.vstack([femoral, tibial])
            contours[side] = {"femur": femoral, "tibia": tibial}
            centers[side] = float(np.mean(all_points[:, 0]))

        if not contours:
            return {}

        boxes: dict[str, tuple[int, int, int, int]] = {}
        for side, side_contours in contours.items():
            other_centers = [value for key, value in centers.items() if key != side]
            if other_centers:
                outer_is_left = centers[side] < float(np.mean(other_centers))
            else:
                outer_is_left = side == "lateral"

            for bone_key, site in [("femur", f"{side}_femur"), ("tibia", f"{side}_tibia")]:
                contour = side_contours[bone_key]
                x_values = contour[:, 0]
                target_x = float(np.min(x_values) if outer_is_left else np.max(x_values))
                margin_band = max(3.0, 0.08 * float(np.ptp(x_values) if np.ptp(x_values) > 0 else w))
                candidates = contour[np.abs(contour[:, 0] - target_x) <= margin_band]
                if len(candidates) == 0:
                    candidates = contour
                if bone_key == "femur":
                    # Femoral marginal osteophytes are anchored at the distal
                    # femoral articular margin adjacent to the joint space.
                    point = candidates[np.argmax(candidates[:, 1])]
                else:
                    # Tibial marginal osteophytes are anchored at the proximal
                    # tibial plateau margin adjacent to the joint space.
                    point = candidates[np.argmin(candidates[:, 1])]

                comp_width = max(1.0, float(np.ptp(x_values)))
                box_size = int(max(24, min(48, round(comp_width * 0.24))))
                cx, cy = float(point[0]), float(point[1])
                if bone_key == "femur":
                    cy -= box_size * 0.70
                else:
                    cy += box_size * 0.85
                x1 = int(round(cx - box_size / 2))
                y1 = int(round(cy - box_size / 2))
                x1 = max(0, min(w - box_size, x1))
                y1 = max(0, min(h - box_size, y1))
                boxes[site] = (x1, y1, box_size, box_size)
        return boxes

    @staticmethod
    def _osteophyte_focus_bbox(
        site: str,
        roi_bbox: tuple[int, int, int, int],
        heatmap: Optional[np.ndarray],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = map(int, roi_bbox)
        focus_w = max(12, int(w * 0.13))
        focus_h = max(12, int(h * 0.13))
        if heatmap is not None and heatmap.size > 0:
            hm = np.asarray(heatmap, dtype=np.float32)
            threshold = max(float(np.percentile(hm, 96)), float(hm.max()) * 0.72)
            ys, xs = np.where(hm >= threshold)
            if xs.size > 0 and ys.size > 0:
                weights = hm[ys, xs]
                if float(weights.sum()) > 1e-8:
                    cx_hm = float(np.average(xs, weights=weights))
                    cy_hm = float(np.average(ys, weights=weights))
                else:
                    cx_hm = float(np.mean(xs))
                    cy_hm = float(np.mean(ys))
                cx = int(round(x + cx_hm / max(hm.shape[1] - 1, 1) * w))
                cy = int(round(y + cy_hm / max(hm.shape[0] - 1, 1) * h))
                return (
                    max(0, cx - focus_w // 2),
                    max(0, cy - focus_h // 2),
                    focus_w,
                    focus_h,
                )

        box_w = focus_w
        box_h = focus_h
        if "femur" in site:
            fy = y + int(h * 0.67)
        else:
            fy = y + int(h * 0.08)
        if "medial" in site:
            fx = x + int(w * 0.72)
        else:
            fx = x + int(w * 0.08)
        return (fx, fy, box_w, box_h)

    @staticmethod
    def _resolve_osteophyte_display_bbox(site: str, roi_bbox: tuple[int, int, int, int], heatmap_items: list[dict]) -> tuple[int, int, int, int]:
        for item in heatmap_items:
            if item.get("site") == site and item.get("focus_bbox") is not None:
                return item["focus_bbox"]
        return TriFQPipeline._osteophyte_focus_bbox(site, roi_bbox, None)

    @staticmethod
    def _xgboost_runtime_supported() -> bool:
        """Avoid known native xgboost crashes in the local Python 3.13 app runtime."""
        return sys.version_info < (3, 13)

    def _display_pixel_spacing_mm(self) -> Optional[float]:
        value = getattr(self.config.preprocessing, "display_pixel_spacing_mm", None)
        if value in (None, "", "null", "None"):
            return None
        try:
            spacing = float(value)
        except (TypeError, ValueError):
            return None
        return spacing if spacing > 0 else None

    def _load_hybrid_model(self, model_cls):
        if self._hybrid_model is not None:
            return self._hybrid_model

        ckpt_dir = self.checkpoint_dir / "kl_hybrid"
        ckpt_path = None
        if ckpt_dir.exists():
            for monitor in ("val_qwk", "val/qwk"):
                ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor=monitor)
                if ckpt_path is not None:
                    break
        if ckpt_path is None:
            return None

        checkpoint = load_checkpoint(ckpt_path, map_location=self.device)
        model_cfg = self._checkpoint_model_cfg("hybrid", checkpoint)
        model = model_cls(model_cfg)
        model.load_state_dict(extract_model_state_dict(checkpoint))
        model.to(self.device)
        model.eval()
        self._hybrid_model = model
        return self._hybrid_model

    @staticmethod
    def _select_kl_prediction(requested_path: str, predictions: Dict[str, Dict]) -> tuple[Dict, str]:
        normalized = str(requested_path or "auto").strip().lower()
        if normalized in predictions:
            return predictions[normalized], normalized
        if normalized == "xgboost" and "xgboost" not in predictions:
            available = ", ".join(predictions.keys())
            raise ValueError(
                "XGBoost is unavailable in this Python runtime. "
                f"Available KL paths: {available}. Use Python 3.11/3.12 for Path A XGBoost."
            )
        if normalized not in {"auto", "", "default"}:
            raise ValueError(
                "Unsupported kl_path. Expected one of: auto, xgboost, hybrid, heuristic."
            )
        for candidate in ("xgboost", "hybrid", "heuristic"):
            if candidate in predictions:
                return predictions[candidate], candidate
        raise RuntimeError("No KL prediction path is available.")

    @staticmethod
    def _heuristic_kl_prediction(jsn: Dict, osp: Dict, scl: Dict) -> Dict:
        severity = (
            0.40 * max(jsn.get("jsn_rate_medial", 0.0), jsn.get("jsn_rate_lateral", 0.0)) / 100.0
            + 0.35 * float(osp.get("osp_sum", 0.0)) / 12.0
            + 0.25 * float(scl.get("scl_grade_medial", 0) + scl.get("scl_grade_lateral", 0)) / 4.0
        )
        thresholds = [0.15, 0.30, 0.48, 0.68]
        grade = int(np.digitize(severity, thresholds))

        logits = np.array([
            -abs(severity - 0.05),
            -abs(severity - 0.22),
            -abs(severity - 0.40),
            -abs(severity - 0.60),
            -abs(severity - 0.82),
        ])
        exp_logits = np.exp(logits - logits.max())
        probabilities = (exp_logits / exp_logits.sum()).tolist()
        return {
            "grade": grade,
            "confidence": float(max(probabilities)),
            "probabilities": probabilities,
        }

    def _load_reference_mjsw(self) -> dict:
        """Load calibrated mJSW reference values from JSN feature extraction."""
        import json

        ref_path = Path(self.config.feature_dir) / "jsn" / "reference_mjsw.json"
        if ref_path.exists():
            with open(ref_path) as f:
                data = json.load(f)
            return {
                "medial": data.get("reference_mjsw_medial", 15.0),
                "lateral": data.get("reference_mjsw_lateral", 15.0),
            }
        return {"medial": 15.0, "lateral": 15.0}

    def _build_runtime_config(self):
        project_root = Path(__file__).resolve().parents[1]
        preprocessing = OmegaConf.load(project_root / "configs" / "preprocessing" / "default.yaml")
        training = OmegaConf.load(project_root / "configs" / "training" / "default.yaml")
        models = OmegaConf.create({
            "jsn": OmegaConf.load(project_root / "configs" / "model" / "unetpp.yaml"),
            "osteophyte": OmegaConf.load(project_root / "configs" / "model" / "se_resnet50.yaml"),
            "sclerosis": OmegaConf.load(project_root / "configs" / "model" / "sclerosis_hybrid.yaml"),
            "xgboost": OmegaConf.load(project_root / "configs" / "model" / "xgboost.yaml"),
            "hybrid": OmegaConf.load(project_root / "configs" / "model" / "convnext_hybrid.yaml"),
        })
        return OmegaConf.create({
            "preprocessing": preprocessing,
            "training": training,
            "feature_dir": str(project_root / "features"),
            "models": models,
        })

    def _get_stage_cfg(self, stage_name: str):
        if hasattr(self.config, "models"):
            return getattr(self.config.models, stage_name)
        if hasattr(self.config, "model"):
            return self.config.model
        raise AttributeError(f"Missing model configuration for stage: {stage_name}")

    def _checkpoint_model_cfg(self, stage_name: str, checkpoint: dict):
        cfg = OmegaConf.create(OmegaConf.to_container(self._get_stage_cfg(stage_name), resolve=True))
        checkpoint_model_cfg = checkpoint.get("hyper_parameters", {}).get("model")
        if checkpoint_model_cfg:
            cfg = OmegaConf.merge(cfg, checkpoint_model_cfg)
        if "pretrained" in cfg:
            cfg.pretrained = False
        return cfg
