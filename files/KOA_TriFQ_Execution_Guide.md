# KOA-TriFQ Execution Guide

## Purpose

This guide describes the corrected implementation flow for the real KOA-TriFQ study. It separates:

- bootstrap baselines built from image-only heuristics
- reviewed manual annotations for the main study
- semi-supervised expansion after the first supervised models are trained

Do not report the bootstrap-only results as the main research result.

Before a run, audit stage readiness:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/check_pipeline_readiness.py
```

## 1. Bootstrap Setup

Generate heuristic suggestions, ROI crops, aggregated features, and balanced annotation manifests:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/bootstrap_pseudo_labels.py
PYTHONPATH=. ./.venv/bin/python scripts/refresh_annotation_manifests.py
PYTHONPATH=. ./.venv/bin/python scripts/prepare_annotation_workspace.py
```

Outputs:

- `annotations/osteophyte_labels.csv`
- `annotations/sclerosis_labels.csv`
- `annotations/manifests/feature_grading_manifest.csv`
- `annotations/manifests/jsn_contour_manifest.csv`
- `annotations/packages/feature_grading/feature_review_template.csv`
- `annotations/packages/jsn_contours/jsn_contour_manifest.csv`

## 2. Stage 1 ROI Annotations

After reviewed ROI box annotations are available under `annotations/reviewed/`, prepare the YOLO dataset:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/prepare_roi_yolo_dataset.py
PYTHONPATH=. ./.venv/bin/python scripts/train_roi_detector.py +model=yolov8
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_roi_detector.py +model=yolov8
PYTHONPATH=. ./.venv/bin/python scripts/compare_roi_backends.py preprocessing.landmark_backend=kneel_repo
PYTHONPATH=. ./.venv/bin/python scripts/extract_rois.py
```

The KNEEL backend expects the external microservice to be running.

## 3. Manual Annotation Work

### Feature grading package

Edit:

- `annotations/packages/feature_grading/feature_review_template.csv`

Fill:

- `final_osp_mf`, `final_osp_lf`, `final_osp_mt`, `final_osp_lt`
- `final_scl_medial`, `final_scl_lateral`
- confidence columns using `high` / `medium` / `low`
- `notes`

The training code now supports confidence-aware filtering/weighting for reviewed osteophyte and sclerosis labels. Defaults keep all reviewed labels (`min_train=low`, `min_eval=low`) while down-weighting lower-confidence rows during training.

### JSN contours package

Import images from:

- `annotations/packages/jsn_contours/images/`

Export the reviewed CVAT COCO JSON to:

- `annotations/reviewed/jsn_cvat_export.json`

## 3. Import Reviewed Annotations

Convert reviewed annotations into training-ready files:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

Outputs:

- `annotations/osteophyte_labels_reviewed.csv`
- `annotations/sclerosis_labels_reviewed.csv`
- `annotations/jsn_masks/train/*.png`
- `annotations/jsn_masks/val/*.png`
- `annotations/jsn_masks/test/*.png`

## 5. Initial Supervised Training

Use reviewed labels only:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_jsn_segmenter.py training.label_mode=manual
PYTHONPATH=. ./.venv/bin/python scripts/train_osteophyte_grader.py training.label_mode=manual +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/train_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/extract_osteophyte_features.py +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/extract_all_features.py
PYTHONPATH=. ./.venv/bin/python scripts/train_kl_xgboost.py +model=xgboost
PYTHONPATH=. ./.venv/bin/python scripts/train_kl_hybrid.py +model=convnext_hybrid
```

For Stage 2C, the current research-aligned default is a deeper, contour-following
subchondral band below the tibial plateau, configured under
`preprocessing.sclerosis_roi`. Keep that extractor fixed when comparing
sclerosis runs so the ROI definition remains consistent.

The current Stage 2C default is a compartment-aware multitask setup: one shared
hybrid feature extractor is trained on all sclerosis ROIs, but medial and
lateral predictions are produced by separate classifier heads and optimized with
ordinal supervision. This keeps the task aligned with per-compartment
sclerosis grading while preserving more training signal than two fully separate
models.

## 6. Semi-Supervised Expansion

After the first supervised osteophyte and sclerosis models are trained, generate high-confidence pseudo-label expansions:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/generate_feature_pseudolabels.py training.pseudo_confidence_threshold_osteophyte=0.90 training.pseudo_confidence_threshold_sclerosis=0.90
```

Outputs:

- `annotations/pseudo/osteophyte_labels_high_conf.csv`
- `annotations/pseudo/sclerosis_labels_high_conf.csv`
- `annotations/osteophyte_labels_expanded.csv`
- `annotations/sclerosis_labels_expanded.csv`

Then retrain the feature models with expanded labels:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_osteophyte_grader.py training.label_mode=expanded +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=expanded +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/train_sclerosis.py training.label_mode=expanded +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/extract_osteophyte_features.py +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=expanded +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/extract_all_features.py
PYTHONPATH=. ./.venv/bin/python scripts/train_kl_xgboost.py +model=xgboost
PYTHONPATH=. ./.venv/bin/python scripts/train_kl_hybrid.py +model=convnext_hybrid
```

## 7. Evaluation

Run evaluation only after the staged models and aggregated features are ready:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_jsn_segmenter.py +model=unetpp +checkpoint_monitor=val_mjsw_mae +checkpoint_mode=min
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_osteophyte_grader.py training.label_mode=manual +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/run_kl_feature_baselines.py
PYTHONPATH=. ./.venv/bin/python scripts/run_ablation.py +model=xgboost
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_pipeline.py +model=xgboost
```

For the osteophyte follow-up runs that target the current lateral-site failure
modes, use the preset runner:

```bash
./.venv/bin/python scripts/run_osteophyte_experiments.py --experiments all
```

Then execute the strongest preset explicitly, for example:

```bash
./.venv/bin/python scripts/run_osteophyte_experiments.py --experiments e3_noflip_mediumconf_classweight --execute
./.venv/bin/python scripts/train_osteophyte_site_refiners.py training.label_mode=manual preprocessing.augmentation.horizontal_flip_p=0.0 training.osteophyte_class_balance.enabled=true training.osteophyte_refinement.sites='[lateral_femur,lateral_tibia]'
```

For Stage 2A, prefer the JSN checkpoint selected by `val_mjsw_mae` rather than
`val_dice` when reporting or exporting downstream JSN features. The segmentation
module is optimized to produce reliable `mJSW`, and the final measurement rule
uses interior weight-bearing contour distances instead of letting the
intercondylar notch or compartment endpoints define the minimum.

## 8. Reporting and Deployment

```bash
PYTHONPATH=. ./.venv/bin/python scripts/export_pdf_report.py image_path=/absolute/path/to/image.png
python app/gradio_app.py
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## 9. Interpretation Rules

- `training.label_mode=manual`: use reviewed labels only when available
- `training.label_mode=expanded`: use reviewed labels plus high-confidence model pseudo-labels
- `training.allow_bootstrap_fallback=false` by default: `manual` and `expanded` now fail fast if reviewed inputs are missing
- no training script should use KL-derived pseudo feature labels for the main study
- the untouched test split remains for final reporting only
