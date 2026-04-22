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

The current Stage 2C default is a conservative shared-head hybrid setup: one
hybrid feature extractor is trained on all medial/lateral sclerosis ROIs, side
identity is supplied as a covariate, and class/sampling weights are capped to
avoid collapse toward the `significant` class. This keeps the model aligned with
per-compartment sclerosis grading while preserving training signal from the
small manual-label set.

For Stage 2C reporting, run a texture-only baseline before accepting the hybrid
model as the main sclerosis result:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid model.input_mode=texture_only model.use_side_specific_heads=false training.sclerosis_strategy=shared training.sclerosis_sampling.max_weight_ratio_to_median=2.0
```

Then compare it against the default hybrid run. Report the hybrid model only if
it improves macro-F1 and preserves useful recall for the `none` class.

To train the same sclerosis model as a binary `none` versus `present` endpoint,
add `training.sclerosis_label_scheme=binary_present` and use separate output
paths so binary checkpoints do not mix with 3-class severity checkpoints:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid training.sclerosis_label_scheme=binary_present model.input_mode=hybrid model.use_side_specific_heads=false training.sclerosis_strategy=multitask_heads training.sclerosis_dev_pool.enabled=true training.sclerosis_dev_pool.n_splits=5 training.sclerosis_dev_pool.holdout_fold=0 training.sclerosis_sampling.max_weight_ratio_to_median=2.0 training.sclerosis_sampling.multiplier_power=0.75 training.learning_rate=3.0e-5 training.weight_decay=1.0e-4 training.max_epochs=100 training.early_stopping.patience=20 sclerosis_output_dir=features/sclerosis_manual_teacher checkpoint_dir=checkpoints/sclerosis_binary_present output_dir=outputs/sclerosis_binary_present
```

Keep `training.sclerosis_label_scheme=severity` or omit the override for the
original 3-class `none/mild/significant` run. If using the staged runner,
pass `SCLEROSIS_LABEL_SCHEME=binary_present` and a separate
`SCLEROSIS_CHECKPOINT_DIR`, for example:

```bash
SCLEROSIS_LABEL_SCHEME=binary_present SCLEROSIS_CHECKPOINT_DIR=checkpoints/sclerosis_binary_present ./scripts/run_main_study_pipeline.sh ...
```

For a 3-class weak-supervision experiment, generate KL/image-guided sclerosis
labels separately from the manual-label baseline:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/generate_sclerosis_kl_image_weak_labels.py training.label_mode=manual sclerosis_output_dir=features/sclerosis_manual_teacher
```

This writes `annotations/pseudo/sclerosis_labels_kl_image_weak.csv` and merges
it into `annotations/sclerosis_labels_expanded.csv` with `label_source=kl_image_weak`.
Use it only with `training.label_mode=expanded`; final evaluation should still
use `training.label_mode=manual`.

For the current downstream KL pipeline, use the manual-label binary texture-only
sclerosis checkpoint with the validation-tuned global threshold. This writes
binary `scl_grade_medial` and `scl_grade_lateral` features (`0=none`,
`1=present`) while preserving the continuous texture features:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid training.sclerosis_label_scheme=binary_present training.sclerosis_binary_threshold=0.4227197766304016 training.sclerosis_apply_classifier_in_manual=true training.sclerosis_force_model_predictions=true model.input_mode=texture_only model.use_side_specific_heads=false checkpoint_path=checkpoints/sclerosis_binary_texture_only/sclerosis/scl-auc-epoch=044-val_auc_macro=0.6730.ckpt sclerosis_standardizer_dir=features/sclerosis_manual_teacher sclerosis_output_dir=features/sclerosis
```

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

For the binary sclerosis branch, keep pseudo-label fine-tuning as a separate
experiment from the manual-only baseline. Generate binary pseudo labels with
the manual teacher checkpoint, train the expanded model in its own checkpoint
directory, and down-weight `high_conf_model` rows so pseudo labels add coverage
without replacing reviewed labels:

```bash
SCLEROSIS_LABEL_SCHEME=binary_present SCLEROSIS_CHECKPOINT_DIR=checkpoints/sclerosis_binary_texture_pseudo SCLEROSIS_PSEUDO_TEACHER_CKPT=checkpoints/sclerosis_binary_texture_only/sclerosis/scl-auc-epoch=044-val_auc_macro=0.6730.ckpt SCLEROSIS_STANDARDIZER_DIR=features/sclerosis_expanded_teacher SCLEROSIS_EVAL_DIR=features/sclerosis_expanded_teacher SCLEROSIS_PSEUDO_TARGET_ROWS=4000 SCLEROSIS_PSEUDO_MIN_CONFIDENCE=0.75 SCLEROSIS_PSEUDO_WEIGHT=0.35 ./scripts/run_main_study_pipeline.sh --skip-stages 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,22,24,25,26,27,28,30,31,32,33,34
```

Accept this pseudo-fine-tuned branch only if manual-label validation/test
metrics improve over the manual binary texture-only baseline.

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
