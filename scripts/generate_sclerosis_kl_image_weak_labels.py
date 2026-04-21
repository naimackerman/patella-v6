"""Generate KL-guided image weak labels for 3-class sclerosis experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.features.bootstrap_heuristics import (
    fit_sclerosis_calibrators_from_reviewed_scores,
    sclerosis_roi_score,
)
from src.utils.annotation_paths import MANUAL_SOURCES, resolve_annotation_csv
from src.utils.seed import seed_everything


OUTPUT_COLUMNS = [
    "image_id",
    "split",
    "pseudo_label",
    "needs_review",
    "label_source",
    "scl_confidence_med",
    "scl_medial",
    "label_source_medial",
    "scl_confidence_lat",
    "scl_lateral",
    "label_source_lateral",
]


def _build_kl_lookup(data_root: Path) -> dict[str, tuple[int, str]]:
    lookup: dict[str, tuple[int, str]] = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for grade_dir in split_dir.iterdir():
            if not grade_dir.is_dir():
                continue
            try:
                kl_grade = int(grade_dir.name)
            except ValueError:
                continue
            for image_path in grade_dir.glob("*.png"):
                lookup[image_path.stem] = (kl_grade, split)
    return lookup


def _base_image_id(image_id: str) -> tuple[str, str]:
    value = str(image_id)
    if value.endswith("_medial"):
        return value[:-7], "medial"
    if value.endswith("_lateral"):
        return value[:-8], "lateral"
    base, side = value.rsplit("_", 1)
    return base, side


def _confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(confidence), 0)


def _weak_grade(kl_grade: int, image_grade: int, include_kl2: bool, require_image_agreement: bool) -> int | None:
    if kl_grade <= 1:
        if require_image_agreement and image_grade > 1:
            return None
        return 0
    if kl_grade == 2:
        if not include_kl2:
            return None
        return int(np.clip(image_grade, 0, 1))
    if kl_grade == 3:
        if require_image_agreement and image_grade == 0:
            return None
        return max(1, min(2, int(image_grade)))
    if require_image_agreement and image_grade == 0:
        return None
    return 2


def _load_manual_ids(annotation_dir: Path) -> set[str]:
    manual_path = resolve_annotation_csv(
        annotation_dir,
        "sclerosis_labels",
        mode="manual",
        allow_bootstrap_fallback=False,
    )
    manual_df = pd.read_csv(manual_path)
    if "label_source" in manual_df.columns:
        manual_df = manual_df[manual_df["label_source"].isin(MANUAL_SOURCES)].copy()
    return {str(image_id).replace(".png", "") for image_id in manual_df["image_id"].tolist()}


def _fit_calibrators(data) -> dict:
    labeled_scores: dict[str, list[tuple[float, int]]] = {"medial": [], "lateral": []}
    sources = np.asarray(data["label_sources"]).astype(str)
    for roi_path, image_id, grade, source in zip(
        data["roi_paths"],
        data["image_ids"],
        data["grades"],
        sources,
    ):
        if source not in MANUAL_SOURCES:
            continue
        _, side = _base_image_id(str(image_id))
        if side not in labeled_scores:
            continue
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue
        labeled_scores[side].append((sclerosis_roi_score(roi), int(grade)))
    return fit_sclerosis_calibrators_from_reviewed_scores(labeled_scores)


def _merge_expanded(annotation_dir: Path, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    manual_path = resolve_annotation_csv(
        annotation_dir,
        "sclerosis_labels",
        mode="manual",
        allow_bootstrap_fallback=False,
    )
    manual_df = pd.read_csv(manual_path)
    if "label_source" in manual_df.columns:
        manual_df = manual_df[manual_df["label_source"].isin(MANUAL_SOURCES)].copy()
    manual_df["image_id"] = manual_df["image_id"].astype(str).str.replace(".png", "", regex=False)
    if not pseudo_df.empty:
        pseudo_df = pseudo_df.copy()
        pseudo_df["image_id"] = pseudo_df["image_id"].astype(str).str.replace(".png", "", regex=False)
        pseudo_df = pseudo_df[~pseudo_df["image_id"].isin(set(manual_df["image_id"]))]
    merged = pd.concat([manual_df, pseudo_df], ignore_index=True)
    out_path = annotation_dir / "sclerosis_labels_expanded.csv"
    merged.to_csv(out_path, index=False)
    return merged


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    annotation_dir = Path(cfg.annotation_dir)
    sclerosis_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    data_path = sclerosis_dir / "train_sclerosis_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sclerosis train data: {data_path}")

    weak_cfg = getattr(cfg.training, "sclerosis_kl_image_weak", {})
    target_rows = int(getattr(weak_cfg, "target_rows", 1500))
    include_kl2 = bool(getattr(weak_cfg, "include_kl2", False))
    require_image_agreement = bool(getattr(weak_cfg, "require_image_agreement", True))

    data = np.load(data_path, allow_pickle=True)
    calibrators = _fit_calibrators(data)
    manual_ids = _load_manual_ids(annotation_dir)
    kl_lookup = _build_kl_lookup(Path(cfg.data.root))

    rows_by_image: dict[str, dict[str, object]] = {}
    priorities: dict[str, tuple[int, int, str]] = {}
    side_counts = defaultdict(int)

    for roi_path, image_id in zip(data["roi_paths"], data["image_ids"]):
        base_id, side = _base_image_id(str(image_id))
        if base_id in manual_ids or side not in {"medial", "lateral"}:
            continue
        kl_info = kl_lookup.get(base_id)
        if kl_info is None:
            continue
        kl_grade, split = kl_info
        if split != "train":
            continue

        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue
        score = sclerosis_roi_score(roi)
        calibrator = calibrators.get(side)
        image_grade = calibrator.grade(score) if calibrator is not None else min(2, int(round(score * 2.0)))
        weak_grade = _weak_grade(
            int(kl_grade),
            int(image_grade),
            include_kl2=include_kl2,
            require_image_agreement=require_image_agreement,
        )
        if weak_grade is None:
            continue

        confidence = calibrator.confidence(score) if calibrator is not None else "medium"
        row = rows_by_image.setdefault(
            base_id,
            {
                "image_id": base_id,
                "split": "train",
                "pseudo_label": True,
                "needs_review": False,
                "label_source": "kl_image_weak",
                "scl_confidence_med": np.nan,
                "scl_medial": np.nan,
                "label_source_medial": "low_confidence_skip",
                "scl_confidence_lat": np.nan,
                "scl_lateral": np.nan,
                "label_source_lateral": "low_confidence_skip",
            },
        )
        suffix = "med" if side == "medial" else "lat"
        row[f"scl_confidence_{suffix}"] = confidence
        row[f"scl_{side}"] = int(weak_grade)
        row[f"label_source_{side}"] = "kl_image_weak"
        side_counts[(side, int(weak_grade))] += 1
        current_priority = priorities.get(base_id, (0, 0, base_id))
        priorities[base_id] = (
            max(current_priority[0], _confidence_rank(confidence)),
            max(current_priority[1], int(kl_grade)),
            base_id,
        )

    candidates = list(rows_by_image.values())
    candidates.sort(key=lambda row: priorities[str(row["image_id"])], reverse=True)
    if target_rows > 0:
        candidates = candidates[:target_rows]
    pseudo_df = pd.DataFrame(candidates, columns=OUTPUT_COLUMNS)

    out_dir = annotation_dir / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = out_dir / "sclerosis_labels_kl_image_weak.csv"
    pseudo_df.to_csv(pseudo_path, index=False)
    merged = _merge_expanded(annotation_dir, pseudo_df)

    print(f"Saved KL/image weak sclerosis labels: {pseudo_path} ({len(pseudo_df)} image rows)")
    print(f"Saved expanded sclerosis labels: {annotation_dir / 'sclerosis_labels_expanded.csv'} ({len(merged)} rows)")
    print(f"Manual IDs excluded: {len(manual_ids)}")
    print(f"Side/grade accepted counts before row cap: {dict(side_counts)}")


if __name__ == "__main__":
    main()
