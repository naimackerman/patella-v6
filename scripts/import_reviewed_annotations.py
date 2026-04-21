"""Import reviewed annotation outputs into training-ready CSV and mask files."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data.annotation_converter import cvat_coco_to_masks


REQUIRED_FEATURE_REVIEW_COLUMNS = {
    "image_id",
    "split",
    "final_osp_mf",
    "final_osp_lf",
    "final_osp_mt",
    "final_osp_lt",
    "final_scl_medial",
    "final_scl_lateral",
}


def _final_only(row: pd.Series, final_col: str):
    final_value = row.get(final_col)
    if pd.notna(final_value) and str(final_value).strip() != "":
        return int(final_value), "manual_review"
    return None, "missing"


def _validate_feature_review_sheet(review: pd.DataFrame, review_csv: Path):
    missing = sorted(REQUIRED_FEATURE_REVIEW_COLUMNS - set(review.columns))
    if missing:
        raise ValueError(f"Reviewed feature sheet is missing required columns {missing}: {review_csv}")


def _import_feature_reviews(annotation_dir: Path):
    review_csv = annotation_dir / "packages" / "feature_grading" / "feature_review_template.csv"
    if not review_csv.exists():
        print(f"Skipping feature-review import; not found: {review_csv}")
        return

    review = pd.read_csv(review_csv)
    _validate_feature_review_sheet(review, review_csv)
    osp_rows = []
    scl_rows = []

    for _, row in review.iterrows():
        image_id = str(row["image_id"]).replace(".png", "")

        osp_values = {}
        osp_sources = []
        for short in ("mf", "lf", "mt", "lt"):
            value, source = _final_only(row, f"final_osp_{short}")
            osp_values[f"osp_{short}"] = value
            osp_values[f"label_source_{short}"] = source
            osp_sources.append(source)

        scl_values = {}
        scl_sources = []
        for side in ("medial", "lateral"):
            value, source = _final_only(row, f"final_scl_{side}")
            scl_values[f"scl_{side}"] = value
            scl_values[f"label_source_{side}"] = source
            scl_sources.append(source)

        osp_rows.append({
            "image_id": image_id,
            **osp_values,
            "confidence_mf": row.get("confidence_mf", ""),
            "confidence_lf": row.get("confidence_lf", ""),
            "confidence_mt": row.get("confidence_mt", ""),
            "confidence_lt": row.get("confidence_lt", ""),
            "notes": row.get("notes", ""),
            "split": row.get("split", ""),
            "pseudo_label": False,
            "needs_review": not any(source == "manual_review" for source in osp_sources),
            "label_source": "manual_review" if any(source == "manual_review" for source in osp_sources) else "missing",
        })
        scl_rows.append({
            "image_id": image_id,
            **scl_values,
            "scl_confidence_med": row.get("scl_confidence_med", ""),
            "scl_confidence_lat": row.get("scl_confidence_lat", ""),
            "notes": row.get("notes", ""),
            "split": row.get("split", ""),
            "pseudo_label": False,
            "needs_review": not any(source == "manual_review" for source in scl_sources),
            "label_source": "manual_review" if any(source == "manual_review" for source in scl_sources) else "missing",
        })

    osp_df = pd.DataFrame(osp_rows)
    scl_df = pd.DataFrame(scl_rows)
    osp_reviewed = osp_df[osp_df["label_source"] == "manual_review"].copy()
    scl_reviewed = scl_df[scl_df["label_source"] == "manual_review"].copy()

    osp_df.to_csv(annotation_dir / "osteophyte_labels_import_audit.csv", index=False)
    scl_df.to_csv(annotation_dir / "sclerosis_labels_import_audit.csv", index=False)
    osp_reviewed.to_csv(annotation_dir / "osteophyte_labels_reviewed.csv", index=False)
    scl_reviewed.to_csv(annotation_dir / "sclerosis_labels_reviewed.csv", index=False)

    print(f"Imported osteophyte reviews: {len(osp_reviewed)} manual rows")
    print(f"Imported sclerosis reviews:  {len(scl_reviewed)} manual rows")


def _load_jsn_manifest(annotation_dir: Path) -> pd.DataFrame:
    manifest_path = annotation_dir / "manifests" / "jsn_contour_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"JSN contour manifest is required for split-aware mask import: {manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"JSN contour manifest is missing required columns {missing}: {manifest_path}")
    manifest["image_id"] = manifest["image_id"].astype(str).str.replace(".png", "", regex=False)
    return manifest


def _normalize_jsn_export_image_id(name: str, manifest: pd.DataFrame) -> str:
    stem = Path(str(name)).stem
    if stem in set(manifest["image_id"].tolist()):
        return stem

    for col in ("local_image_path", "preprocessed_local_image_path", "prefixed_local_image_path"):
        if col in manifest.columns:
            matched = manifest[manifest[col].astype(str).map(lambda p: Path(p).stem == stem)]
            if not matched.empty:
                return str(matched.iloc[0]["image_id"])

    parts = stem.split("_", 2)
    if len(parts) == 3:
        candidate = parts[2]
        if candidate in set(manifest["image_id"].tolist()):
            return candidate
    return stem


def _find_jsn_export_json(annotation_dir: Path) -> Path | None:
    explicit = annotation_dir / "reviewed" / "jsn_cvat_export.json"
    if explicit.exists():
        return explicit

    roboflow_root = annotation_dir / "packages" / "jsn_contours"
    candidates = sorted(roboflow_root.rglob("_annotations.coco.json"))
    return candidates[0] if candidates else None


def _validate_jsn_export(coco_json: Path):
    obj = json.loads(coco_json.read_text(encoding="utf-8"))
    cat_map = {cat["id"]: cat["name"] for cat in obj.get("categories", [])}
    by_img = defaultdict(list)
    img_name_map = {}

    for img in obj.get("images", []):
        original = (img.get("extra", {}) or {}).get("name") or img.get("file_name", "")
        img_name_map[img["id"]] = original

    for ann in obj.get("annotations", []):
        by_img[ann["image_id"]].append(cat_map.get(ann["category_id"], str(ann["category_id"])))

    bad = []
    for img in obj.get("images", []):
        names = Counter(by_img.get(img["id"], []))
        if names.get("femoral_surface", 0) != 1 or names.get("tibial_surface", 0) != 1:
            bad.append(
                (
                    img_name_map.get(img["id"], img.get("file_name", "")),
                    dict(names),
                )
            )

    if bad:
        preview = "; ".join(f"{name}: {counts}" for name, counts in bad[:10])
        raise ValueError(
            f"JSN export contains {len(bad)} image(s) without exactly one femoral_surface and one tibial_surface. "
            f"Examples: {preview}"
        )


def _import_jsn_masks(annotation_dir: Path):
    coco_json = _find_jsn_export_json(annotation_dir)
    if coco_json is None:
        print("Skipping JSN mask import; no reviewed or Roboflow COCO export found.")
        return
    _validate_jsn_export(coco_json)

    manifest = _load_jsn_manifest(annotation_dir)
    split_lookup = dict(zip(manifest["image_id"], manifest["split"]))

    output_dir = annotation_dir / "jsn_masks"
    temp_dir = output_dir / "_flat_import"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    generated = cvat_coco_to_masks(str(coco_json), str(temp_dir), img_shape=(224, 224))

    if not generated:
        raise ValueError(f"No JSN masks were generated from reviewed export: {coco_json}")

    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing_in_manifest = []
    for generated_path in map(Path, generated):
        image_id = _normalize_jsn_export_image_id(generated_path.stem.replace(".png", ""), manifest)
        split = split_lookup.get(image_id)
        if split is None:
            missing_in_manifest.append(image_id)
            continue
        shutil.move(str(generated_path), str(output_dir / split / f"{image_id}.png"))
        moved += 1

    shutil.rmtree(temp_dir, ignore_errors=True)

    if missing_in_manifest:
        raise ValueError(
            "Reviewed JSN masks were generated for image_ids not present in jsn_contour_manifest.csv: "
            + ", ".join(sorted(missing_in_manifest)[:20])
        )

    print(f"Generated {moved} JSN mask files under {output_dir}/{{train,val,test}}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    annotation_dir = Path(cfg.annotation_dir)
    _import_feature_reviews(annotation_dir)
    _import_jsn_masks(annotation_dir)


if __name__ == "__main__":
    main()
