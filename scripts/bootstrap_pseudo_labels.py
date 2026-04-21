"""Bootstrap an image-only baseline without leaking KL labels into features."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.features.bootstrap_heuristics import (
    ROI_SITES,
    build_sclerosis_features,
    estimate_jsn_features,
    extract_geometric_subchondral_roi,
    fit_osteophyte_calibrator,
    fit_osteophyte_calibrators_from_reviewed_scores,
    fit_sclerosis_calibrator,
    fit_sclerosis_calibrators_from_reviewed_scores,
    heuristic_osteophyte_features,
    osteophyte_roi_score,
    sclerosis_roi_score,
)
from src.features.feature_aggregator import FeatureAggregator
from src.features.kneel_landmarks import (
    KNEELLandmarkDetector,
    estimate_jsn_features_from_landmarks,
    extract_kneel_subchondral_rois,
)
from src.models.roi_detector import ROIDetector
from src.utils.manifest_paths import to_repo_local_manifest_path
from src.utils.seed import seed_everything


CONFIDENCE_ORDER = ("low", "medium", "high")


def scan_dataset(data_root: Path, project_root: Path | None = None):
    """Scan dataset and collect image_ids, KL grades, and paths per split."""
    data_root = Path(data_root).resolve()
    project_root = Path(project_root).resolve() if project_root is not None else None
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        records = []
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = int(grade_dir.name)
            for img_path in sorted(grade_dir.glob("*.png")):
                image_id = img_path.stem
                records.append({
                    "image_id": image_id,
                    "grade": grade,
                    "path": str(img_path),
                    "manifest_path": (
                        to_repo_local_manifest_path(img_path, project_root=project_root, data_root=data_root)
                        if project_root is not None
                        else str(img_path)
                    ),
                    "is_left": image_id.upper().endswith("L"),
                })
        splits[split] = records
        print(f"  {split}: {len(records)} images")
    return splits


def extract_bootstrap_rois(splits: dict, output_dir: Path, roi_strategy: str):
    """Extract ROI patches for all images and save to disk."""
    print(f"\n[1/5] Extracting {roi_strategy} ROIs...")
    for split, records in splits.items():
        for rec in tqdm(records, desc=f"ROIs {split}"):
            image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            rois = _select_bootstrap_rois(image, rec["is_left"], roi_strategy)
            for roi_name, roi_patch in rois.items():
                save_path = output_dir / split / f"{rec['image_id']}_{roi_name}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), roi_patch)
    print("  Done.")


def _confidence_penalty(confidence: str, levels: int = 1) -> str:
    confidence = str(confidence).strip().lower()
    if confidence not in CONFIDENCE_ORDER:
        confidence = "medium"
    idx = CONFIDENCE_ORDER.index(confidence)
    return CONFIDENCE_ORDER[max(0, idx - max(int(levels), 0))]


def _fit_reviewed_heuristic_priors(
    splits: dict,
    feature_dir: Path,
    annotation_dir: Path,
    roi_strategy: str,
):
    osp_path = annotation_dir / "osteophyte_labels_reviewed.csv"
    scl_path = annotation_dir / "sclerosis_labels_reviewed.csv"
    if not osp_path.exists() or not scl_path.exists():
        return None, None, None

    osp_df = pd.read_csv(osp_path)
    scl_df = pd.read_csv(scl_path)
    if "label_source" in osp_df.columns:
        osp_df = osp_df[osp_df["label_source"] == "manual_review"].copy()
    if "label_source" in scl_df.columns:
        scl_df = scl_df[scl_df["label_source"] == "manual_review"].copy()
    osp_df["image_id"] = osp_df["image_id"].astype(str).str.replace(".png", "", regex=False)
    scl_df["image_id"] = scl_df["image_id"].astype(str).str.replace(".png", "", regex=False)

    train_records = {rec["image_id"]: rec for rec in splits.get("train", [])}
    osp_labeled_scores = {site: [] for site in ROI_SITES}
    scl_labeled_scores = {"medial": [], "lateral": []}
    osp_sum_by_kl: dict[int, list[float]] = {}
    scl_sum_by_kl: dict[int, list[float]] = {}

    for row in osp_df.itertuples(index=False):
        rec = train_records.get(str(row.image_id))
        if rec is None:
            continue
        site_sum = 0.0
        for site, short in (("medial_femur", "mf"), ("lateral_femur", "lf"), ("medial_tibia", "mt"), ("lateral_tibia", "lt")):
            roi_path = feature_dir / "rois" / "train" / f"{rec['image_id']}_{site}.png"
            roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
            score = osteophyte_roi_score(roi, site)
            label = getattr(row, f"osp_{short}")
            if pd.notna(label):
                osp_labeled_scores[site].append((float(score), int(label)))
                site_sum += float(label)
        osp_sum_by_kl.setdefault(int(rec["grade"]), []).append(site_sum)

    for row in scl_df.itertuples(index=False):
        rec = train_records.get(str(row.image_id))
        if rec is None:
            continue
        image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        medial_roi, lateral_roi = _select_subchondral_rois(image, rec["is_left"], roi_strategy)
        score_med = sclerosis_roi_score(medial_roi)
        score_lat = sclerosis_roi_score(lateral_roi)
        if pd.notna(row.scl_medial):
            scl_labeled_scores["medial"].append((float(score_med), int(row.scl_medial)))
        if pd.notna(row.scl_lateral):
            scl_labeled_scores["lateral"].append((float(score_lat), int(row.scl_lateral)))
        scl_sum = float(row.scl_medial) + float(row.scl_lateral)
        scl_sum_by_kl.setdefault(int(rec["grade"]), []).append(scl_sum)

    osp_calibrators = fit_osteophyte_calibrators_from_reviewed_scores(osp_labeled_scores)
    scl_calibrators = fit_sclerosis_calibrators_from_reviewed_scores(scl_labeled_scores)
    kl_priors = {
        "osteophyte_sum_by_kl": {
            int(kl): {
                "q10": float(np.quantile(values, 0.10)),
                "q90": float(np.quantile(values, 0.90)),
            }
            for kl, values in osp_sum_by_kl.items()
            if values
        },
        "sclerosis_sum_by_kl": {
            int(kl): {
                "q10": float(np.quantile(values, 0.10)),
                "q90": float(np.quantile(values, 0.90)),
            }
            for kl, values in scl_sum_by_kl.items()
            if values
        },
    }
    return osp_calibrators, scl_calibrators, kl_priors


def fit_heuristic_calibrators(splits: dict, feature_dir: Path, annotation_dir: Path, roi_strategy: str):
    """Fit unsupervised discretizers for osteophyte and sclerosis suggestions."""
    print("\n[2/5] Fitting heuristic calibrators from training images...")
    supervised_osp, supervised_scl, kl_priors = _fit_reviewed_heuristic_priors(
        splits,
        feature_dir,
        annotation_dir,
        roi_strategy,
    )
    if supervised_osp is not None and supervised_scl is not None:
        print("  Using reviewed-label calibrated thresholds from manual train labels.")
        for site in ROI_SITES:
            print(f"  Osteophyte {site} thresholds: {supervised_osp[site].thresholds.tolist()}")
        for side in ("medial", "lateral"):
            print(f"  Sclerosis {side} thresholds: {supervised_scl[side].thresholds.tolist()}")
        return supervised_osp, supervised_scl, kl_priors

    train_records = splits.get("train", [])

    osp_rois = []
    scl_rois = []
    for rec in tqdm(train_records, desc="Calibrators"):
        image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        for site in ROI_SITES:
            roi_path = feature_dir / "rois" / "train" / f"{rec['image_id']}_{site}.png"
            if roi_path.exists():
                roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
                if roi is not None:
                    osp_rois.append((site, roi))

        medial_roi, lateral_roi = _select_subchondral_rois(image, rec["is_left"], roi_strategy)
        if medial_roi is not None:
            scl_rois.append(medial_roi)
        if lateral_roi is not None:
            scl_rois.append(lateral_roi)

    osp_calibrator = fit_osteophyte_calibrator(osp_rois)
    scl_calibrator = fit_sclerosis_calibrator(scl_rois)
    print(f"  Osteophyte thresholds: {osp_calibrator.thresholds.tolist()}")
    print(f"  Sclerosis thresholds: {scl_calibrator.thresholds.tolist()}")
    return osp_calibrator, scl_calibrator, None


def save_annotation_suggestions(
    splits: dict,
    feature_dir: Path,
    annotation_dir: Path,
    osp_calibrator,
    scl_calibrator,
    kl_priors,
    roi_strategy: str,
):
    """Create heuristic label suggestions for manual review."""
    print("\n[3/5] Generating heuristic annotation suggestions...")
    osteophyte_records = []
    sclerosis_records = []

    for split, records in splits.items():
        for rec in tqdm(records, desc=f"Suggestions {split}"):
            image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            roi_images = {}
            osp_scores = {}
            for site in ROI_SITES:
                roi_path = feature_dir / "rois" / split / f"{rec['image_id']}_{site}.png"
                roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
                roi_images[site] = roi
                osp_scores[site] = osteophyte_roi_score(roi, site)

            osp_features = heuristic_osteophyte_features(roi_images, calibrator=osp_calibrator)
            osp_sum = float(osp_features["osp_sum"])
            osp_penalty = 0
            if kl_priors is not None:
                bounds = kl_priors.get("osteophyte_sum_by_kl", {}).get(int(rec["grade"]))
                if bounds is not None and (osp_sum < bounds["q10"] or osp_sum > bounds["q90"]):
                    osp_penalty = 1
            osp_conf = {}
            for site, short in (("medial_femur", "mf"), ("lateral_femur", "lf"), ("medial_tibia", "mt"), ("lateral_tibia", "lt")):
                score = osp_scores[site]
                site_calibrator = osp_calibrator[site] if isinstance(osp_calibrator, dict) else osp_calibrator
                osp_conf[short] = _confidence_penalty(site_calibrator.confidence(score), osp_penalty)
            osteophyte_records.append({
                "image_id": rec["image_id"],
                "osp_mf": int(round(osp_features["osp_grade_mf"])),
                "osp_lf": int(round(osp_features["osp_grade_lf"])),
                "osp_mt": int(round(osp_features["osp_grade_mt"])),
                "osp_lt": int(round(osp_features["osp_grade_lt"])),
                "pseudo_label": True,
                "split": split,
                "label_source": "heuristic_image_only",
                "needs_review": True,
                "confidence_mf": osp_conf["mf"],
                "confidence_lf": osp_conf["lf"],
                "confidence_mt": osp_conf["mt"],
                "confidence_lt": osp_conf["lt"],
                "score_mf": osp_scores["medial_femur"],
                "score_lf": osp_scores["lateral_femur"],
                "score_mt": osp_scores["medial_tibia"],
                "score_lt": osp_scores["lateral_tibia"],
            })

            medial_roi, lateral_roi = _select_subchondral_rois(image, rec["is_left"], roi_strategy)
            scl_features = build_sclerosis_features(medial_roi, lateral_roi, calibrator=scl_calibrator)
            scl_sum = float(scl_features["scl_grade_medial"] + scl_features["scl_grade_lateral"])
            scl_penalty = 0
            if kl_priors is not None:
                bounds = kl_priors.get("sclerosis_sum_by_kl", {}).get(int(rec["grade"]))
                if bounds is not None and (scl_sum < bounds["q10"] or scl_sum > bounds["q90"]):
                    scl_penalty = 1
            med_calibrator = scl_calibrator["medial"] if isinstance(scl_calibrator, dict) else scl_calibrator
            lat_calibrator = scl_calibrator["lateral"] if isinstance(scl_calibrator, dict) else scl_calibrator
            sclerosis_records.append({
                "image_id": rec["image_id"],
                "scl_medial": int(scl_features["scl_grade_medial"]),
                "scl_lateral": int(scl_features["scl_grade_lateral"]),
                "pseudo_label": True,
                "split": split,
                "label_source": "heuristic_image_only",
                "needs_review": True,
                "scl_confidence_med": _confidence_penalty(med_calibrator.confidence(sclerosis_roi_score(medial_roi)), scl_penalty),
                "scl_confidence_lat": _confidence_penalty(lat_calibrator.confidence(sclerosis_roi_score(lateral_roi)), scl_penalty),
                "score_medial": sclerosis_roi_score(medial_roi),
                "score_lateral": sclerosis_roi_score(lateral_roi),
            })

    annotation_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(osteophyte_records).to_csv(annotation_dir / "osteophyte_labels.csv", index=False)
    pd.DataFrame(sclerosis_records).to_csv(annotation_dir / "sclerosis_labels.csv", index=False)
    with open(annotation_dir / "heuristic_calibration.json", "w") as f:
        json.dump(
            {
                "osteophyte": {
                    site: cal.to_dict()
                    for site, cal in osp_calibrator.items()
                } if isinstance(osp_calibrator, dict) else osp_calibrator.to_dict(),
                "sclerosis": {
                    side: cal.to_dict()
                    for side, cal in scl_calibrator.items()
                } if isinstance(scl_calibrator, dict) else scl_calibrator.to_dict(),
                "kl_priors": kl_priors or {},
            },
            f,
            indent=2,
        )
    print(f"  -> {annotation_dir / 'osteophyte_labels.csv'}")
    print(f"  -> {annotation_dir / 'sclerosis_labels.csv'}")


def extract_all_bootstrap_features(
    splits: dict,
    feature_dir: Path,
    osp_calibrator,
    scl_calibrator,
    roi_strategy: str,
):
    """Aggregate image-only heuristic features into 50-dim vectors."""
    print("\n[4/5] Extracting image-only bootstrap features...")

    aggregator = FeatureAggregator()
    agg_dir = feature_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    jsn_features_all = {}
    for split, records in splits.items():
        for rec in tqdm(records, desc=f"JSN {split}"):
            image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            jsn = _select_jsn_features(image, rec["is_left"], roi_strategy)
            jsn_features_all[rec["image_id"]] = {
                "features": jsn,
                "grade": rec["grade"],
                "split": split,
            }

    kl0_med = [v["features"]["mJSW_medial"] for v in jsn_features_all.values() if v["grade"] == 0 and v["split"] == "train"]
    kl0_lat = [v["features"]["mJSW_lateral"] for v in jsn_features_all.values() if v["grade"] == 0 and v["split"] == "train"]
    ref_med = float(np.median(kl0_med)) if kl0_med else 15.0
    ref_lat = float(np.median(kl0_lat)) if kl0_lat else 15.0
    print(f"  Reference mJSW — Medial: {ref_med:.2f}, Lateral: {ref_lat:.2f}")

    for split, records in splits.items():
        features_list = []
        labels_list = []
        image_ids = []

        for rec in tqdm(records, desc=f"Aggregate {split}"):
            image = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image_id = rec["image_id"]
            jsn_feats = jsn_features_all[image_id]["features"]
            jsn_feats["jsn_rate_medial"] = float(np.clip(100.0 * (1.0 - jsn_feats["mJSW_medial"] / max(ref_med, 1e-6)), 0, 100))
            jsn_feats["jsn_rate_lateral"] = float(np.clip(100.0 * (1.0 - jsn_feats["mJSW_lateral"] / max(ref_lat, 1e-6)), 0, 100))

            roi_images = {}
            for site in ROI_SITES:
                roi_path = feature_dir / "rois" / split / f"{image_id}_{site}.png"
                roi_images[site] = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
            osp_feats = heuristic_osteophyte_features(roi_images, calibrator=osp_calibrator)

            medial_roi, lateral_roi = _select_subchondral_rois(image, rec["is_left"], roi_strategy)
            scl_feats = build_sclerosis_features(medial_roi, lateral_roi, calibrator=scl_calibrator)

            vec = aggregator.aggregate(jsn_feats, osp_feats, scl_feats)
            features_list.append(vec)
            labels_list.append(rec["grade"])
            image_ids.append(image_id)

        features = np.array(features_list, dtype=np.float64)
        labels = np.array(labels_list, dtype=np.int64)
        np.savez(
            str(agg_dir / f"{split}_features.npz"),
            features=features,
            labels=labels,
            image_ids=image_ids,
        )
        print(f"  Saved {split}: {features.shape[0]} samples, {features.shape[1]} dims")

    train_data = np.load(agg_dir / "train_features.npz")
    aggregator.fit_normalizer(train_data["features"])
    np.savez(
        str(agg_dir / "normalizer_stats.npz"),
        mean=aggregator.train_mean,
        std=aggregator.train_std,
    )
    print("  Saved normalizer stats.")


def save_annotation_manifests(splits: dict, annotation_dir: Path):
    """Create stratified manual-review manifests that match the research plan."""
    print("\n[5/5] Creating annotation manifests...")
    manifest_dir = annotation_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = _sample_per_grade_and_split(
        splits,
        per_grade_by_split={"train": 70, "val": 15, "test": 15},
    )
    jsn_rows = _sample_per_grade_and_split(
        splits,
        per_grade_by_split={"train": 56, "val": 12, "test": 12},
    )

    pd.DataFrame(feature_rows).to_csv(manifest_dir / "feature_grading_manifest.csv", index=False)
    pd.DataFrame(jsn_rows).to_csv(manifest_dir / "jsn_contour_manifest.csv", index=False)
    pd.DataFrame(feature_rows[:50]).to_csv(manifest_dir / "inter_rater_subset.csv", index=False)
    print(f"  -> {manifest_dir / 'feature_grading_manifest.csv'}")
    print(f"  -> {manifest_dir / 'jsn_contour_manifest.csv'}")


def _sample_per_grade_and_split(splits: dict, per_grade_by_split: dict[str, int]) -> list[dict]:
    records = []
    for grade in range(5):
        used_image_ids = set()
        for split, quota in per_grade_by_split.items():
            if quota <= 0:
                continue
            grade_rows = [
                {
                    "image_id": rec["image_id"],
                    "kl_grade": rec["grade"],
                    "split": split,
                    "path": rec.get("manifest_path", rec["path"]),
                    "is_left": rec["is_left"],
                }
                for rec in splits.get(split, [])
                if rec["grade"] == grade
                and rec["image_id"] not in used_image_ids
            ]
            selected = grade_rows[:quota]
            records.extend(selected)
            used_image_ids.update(row["image_id"] for row in selected)
    return records


def _select_bootstrap_rois(image: np.ndarray, is_left: bool, roi_strategy: str) -> dict:
    strategy = (roi_strategy or "geometric").lower()
    if strategy == "landmark":
        return ROIDetector.landmark_rois(image, is_left=is_left)
    return ROIDetector.geometric_rois(image, is_left=is_left)


def _select_subchondral_rois(image: np.ndarray, is_left: bool, roi_strategy: str):
    strategy = (roi_strategy or "geometric").lower()
    if strategy == "landmark":
        landmarks = KNEELLandmarkDetector().predict(image, is_left=is_left)
        return extract_kneel_subchondral_rois(image, landmarks)
    return extract_geometric_subchondral_roi(image, is_left=is_left)


def _select_jsn_features(image: np.ndarray, is_left: bool, roi_strategy: str) -> dict:
    strategy = (roi_strategy or "geometric").lower()
    if strategy == "landmark":
        landmarks = KNEELLandmarkDetector().predict(image, is_left=is_left)
        return estimate_jsn_features_from_landmarks(landmarks)
    return estimate_jsn_features(image, is_left=is_left)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    data_root = Path(cfg.data.root)
    project_root = Path(cfg.project_root)
    feature_dir = Path(cfg.feature_dir)
    annotation_dir = Path(cfg.annotation_dir)
    roi_strategy = getattr(cfg.preprocessing, "roi_strategy", "geometric")

    print("=" * 60)
    print("xrAI-OA: Bootstrap Without KL Leakage")
    print("=" * 60)
    print(f"Data root:   {data_root}")
    print(f"Project root: {project_root}")
    print(f"Feature dir: {feature_dir}")
    print(f"ROI strategy: {roi_strategy}")

    print("\nScanning dataset...")
    splits = scan_dataset(data_root, project_root=project_root)

    extract_bootstrap_rois(splits, feature_dir / "rois", roi_strategy)
    osp_calibrator, scl_calibrator, kl_priors = fit_heuristic_calibrators(splits, feature_dir, annotation_dir, roi_strategy)
    save_annotation_suggestions(splits, feature_dir, annotation_dir, osp_calibrator, scl_calibrator, kl_priors, roi_strategy)
    extract_all_bootstrap_features(splits, feature_dir, osp_calibrator, scl_calibrator, roi_strategy)
    save_annotation_manifests(splits, annotation_dir)

    print("\n" + "=" * 60)
    print("Bootstrap complete.")
    print("=" * 60)
    print(
        """
  Outputs created:
  - feature ROIs under features/rois/<split>/
  - heuristic label suggestions in annotations/
  - 50-dim aggregated bootstrap features in features/aggregated/
  - manual review manifests in annotations/manifests/

  Recommended next steps:
  1. Review/edit annotations/osteophyte_labels.csv
  2. Review/edit annotations/sclerosis_labels.csv
  3. Start JSN contour annotation from annotations/manifests/jsn_contour_manifest.csv
  4. Re-train stage models after reviewed labels are available
"""
    )


if __name__ == "__main__":
    main()
