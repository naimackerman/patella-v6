# KOA-TriFQ

KOA-TriFQ is a staged knee OA research pipeline built around three feature groups:

- JSN: contour annotation, segmentation, mJSW, and derived joint-space features
- Osteophytes: per-site ROI grading
- Sclerosis: subchondral ROI texture and CNN-based grading

The current repo is set up for a reproducible `patella-v6` workflow:

- manifests store repo-local image paths such as `KneeXrayData/ClsKLData/...`
- the main pipeline starts by cleaning duplicate numbered files
- annotation packages use the dataset in place by default instead of copying image folders
- copied package images, when explicitly enabled, use specific filenames of the form `split_grade_imageid.png`

## Setup

Create a virtual environment and install the project requirements:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

By default the repo expects the dataset at:

```bash
./KneeXrayData/ClsKLData/kneeKL224
```

If your dataset lives elsewhere, keep the repo unchanged and point the run to it:

```bash
export DATA_ROOT=/absolute/path/to/KneeXrayData
```

Useful preflight command:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/check_pipeline_readiness.py --project-root .
```

## .env For W&B

If you want repo-local logging secrets, create `.env` from the example file:

```bash
cp .env.example .env
```

Typical `.env`:

```dotenv
WANDB_API_KEY=your_wandb_api_key_here
```

Optional values:

```dotenv
WANDB_RUN_ID=
WANDB_RESUME=allow
DATA_ROOT=/absolute/path/to/KneeXrayData
```

The shell runners load `.env` automatically, and the Lightning logger also checks the repo `.env` directly for W&B when training scripts are run outside those shell wrappers.

## Main Entry Points

Bootstrap-only baseline:

```bash
./scripts/run_initial_experiment.sh
```

Main study pipeline:

```bash
./scripts/run_main_study_pipeline.sh
```

Recommended next command for the current repo state, when `feature_review_template.csv` is already reviewed and the JSN export is already available under `annotations/packages/jsn_contours/jsn-new.coco-segmentation/`:

```bash
./scripts/run_main_study_pipeline.sh --skip-stages 3,6,7,8,9,10
```

The main-study runner covers 34 stages:

1. clean numbered duplicate files
2. audit pipeline readiness
3. build bootstrap pseudo-label suggestions
4. refresh annotation manifests
5. prepare annotation workspace packages
6. prepare ROI YOLO dataset from reviewed boxes
7. train ROI detector
8. evaluate ROI detector
9. compare ROI backends
10. extract ROIs with trained detector or fallback
11. import reviewed annotations
12. train JSN segmenter
13. extract JSN features and masks
14. extract osteophyte ROI patches with CLAHE and JSN-guided boxes
15. train osteophyte grader from manual labels
16. extract sclerosis features for the manual teacher
17. train sclerosis classifier from manual labels
18. generate high-confidence pseudo labels
19. train expanded osteophyte grader
20. extract sclerosis features for expanded training
21. train expanded sclerosis classifier
22. extract osteophyte features
23. re-extract final sclerosis features with the trained classifier
24. aggregate all features
25. train KL XGBoost classifier
26. train KL hybrid classifier
27. evaluate JSN segmenter
28. evaluate osteophyte grader
29. evaluate sclerosis classifier
30. run KL feature baselines
31. run KL ablation studies
32. evaluate the end-to-end pipeline
33. build reproducibility report
34. audit pipeline readiness after the run

Skip stages when you need to resume or isolate part of the pipeline:

```bash
./scripts/run_main_study_pipeline.sh --skip-stage 3
./scripts/run_main_study_pipeline.sh --skip-stages 6,7,8,9,10
```

Optional JSN self-training branch:

```bash
ENABLE_JSN_SELF_TRAINING=1 ./scripts/run_main_study_pipeline.sh --skip-stages 3,6,7,8,9,10
```

By default, stage 13 uses the manual-reviewed JSN checkpoint only. With `ENABLE_JSN_SELF_TRAINING=1`, the runner generates high-confidence pseudo-masks, trains a separate self-trained JSN checkpoint, evaluates manual-only and self-trained checkpoints on reviewed test masks, writes the best checkpoint to `checkpoints/jsn_segmenter_selected.txt`, then re-extracts JSN features with that selected checkpoint.

## Portable Paths And Annotation Packages

`configs/config.yaml` now resolves `project_root` from the active repo and `data_root` from `DATA_ROOT` or `./KneeXrayData`. The annotation manifests created by:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/bootstrap_pseudo_labels.py
PYTHONPATH=. ./.venv/bin/python scripts/refresh_annotation_manifests.py
```

write portable paths such as:

```text
KneeXrayData/ClsKLData/kneeKL224/train/0/9001695L.png
```

That makes the repo reproducible across devices without editing CSVs.

Package preparation:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/prepare_annotation_workspace.py
```

Default behavior:

- `annotations/packages/feature_grading/feature_review_template.csv` points `local_image_path` at the local dataset
- `annotations/packages/jsn_contours/jsn_contour_manifest.csv` points `local_image_path` at the local dataset
- `annotations/packages/feature_grading/images` is not created
- `annotations/packages/jsn_contours/images*` are not created

If you explicitly need copied image folders for annotation tooling:

```bash
ANNOTATION_PACKAGE_COPY_IMAGES=1 PYTHONPATH=. ./.venv/bin/python scripts/prepare_annotation_workspace.py
```

When copy mode is enabled, copied files are named with explicit metadata:

```text
train_0_9001695L.png
val_3_9804376R.png
```

This avoids collisions from generic filenames.

## Annotation Workflow

After stage 5, review these files:

- `annotations/packages/feature_grading/feature_review_template.csv`
- `annotations/packages/jsn_contours/jsn_contour_manifest.csv`

For feature grading:

- fill the `final_*` columns directly in `feature_review_template.csv`
- keep `image_id` and `split` unchanged

For JSN contours:

- annotate from `jsn_contour_manifest.csv`
- export reviewed contours as COCO JSON
- place the reviewed export at `annotations/reviewed/jsn_cvat_export.json`

Import reviewed annotations back into training artifacts:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

This generates:

- `annotations/osteophyte_labels_reviewed.csv`
- `annotations/sclerosis_labels_reviewed.csv`
- `annotations/jsn_masks/{train,val,test}/*.png`

## Main-Study Outputs

The pipeline writes the main artifacts to:

- `annotations/`
- `features/`
- `checkpoints/`
- `results/`
- `report/`

Important directories used by the runner:

- osteophyte ROIs: `features/rois_osteophyte_clahe_full`
- JSN manual checkpoint: `checkpoints/jsn_segmenter`
- JSN self-trained checkpoint: `checkpoints/jsn_segmenter_selftrain`
- JSN selected checkpoint pointer: `checkpoints/jsn_segmenter_selected.txt`
- JSN pseudo-masks: `annotations/jsn_pseudo_masks`
- sclerosis manual teacher features: `features/sclerosis_manual_teacher`
- sclerosis expanded teacher features: `features/sclerosis_expanded_teacher`
- final sclerosis features: `features/sclerosis`
- JSN manual evaluation: `results/jsn_manual`
- JSN self-trained evaluation: `results/jsn_selftrain`
- osteophyte evaluation results: `results/osteophyte_main_manual`
- reproducibility report: `report/`

## Duplicate Cleanup

The repo now includes:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/clean_duplicate_suffix_files.py --project-root . --remove
```

It removes exact duplicate files named like:

```text
train_sclerosis 2.py
bootstrap_pseudo_labels 2.py
```

Only exact safe copies are removed. Conflicting files are reported and left untouched.

## Optional ROI Detector Branch

Stages 6 to 10 of `run_main_study_pipeline.sh` run only when reviewed ROI annotations are present under `annotations/reviewed/`, for example:

- `roi_boxes.csv`
- `roi_annotations.csv`
- `roi_cvat_export.json`
- `roi_boxes_coco.json`

If these files are absent, the pipeline skips the ROI-detector branch and continues with the fallback study path.

## Reproducibility Notes

Useful environment overrides:

- `PYTHON_BIN`: choose the Python interpreter used by the shell runner
- `DATA_ROOT`: point to an external `KneeXrayData` directory
- `ANNOTATION_PACKAGE_COPY_IMAGES=1`: recreate copied annotation image folders
- `ENABLE_JSN_SELF_TRAINING=1`: run the separate JSN self-training branch and use the better JSN checkpoint downstream
- `OSTEOPHYTE_REPRO_CKPT`: override the warm-start osteophyte checkpoint
- `SCLEROSIS_TEACHER_CKPT`: override the teacher checkpoint used for pseudo-label stages

To regenerate the reproducibility summary without rerunning training:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_repro_report.py --run-root .
```
