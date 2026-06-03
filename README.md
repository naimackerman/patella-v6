# Knee-xRAI

> **Knee-xRAI** — An Explainable AI Framework for Automatic Kellgren–Lawrence Grading of Knee Osteoarthritis.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Research-lightgrey)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter)](notebooks/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)](Dockerfile)

---

## Overview

Knee-xRAI is a modular pipeline for knee osteoarthritis (OA) severity grading using the Kellgren–Lawrence (KL) scale. It combines three complementary radiographic feature groups:

| Feature Group | Description |
|---|---|
| **JSN** | Contour annotation, segmentation, minimum joint-space width (mJSW), and derived joint-space features |
| **Osteophytes** | Per-site region-of-interest (ROI) detection and grading |
| **Sclerosis** | Subchondral ROI texture features and CNN-based grading |

The final KL classifier fuses all three groups via an XGBoost / hybrid ensemble trained on manual and pseudo-labelled data.

The repository is configured for a reproducible **`patella-v6`** workflow:

- Manifests store repo-local image paths (e.g. `KneeXrayData/ClsKLData/...`)
- The main pipeline begins by cleaning duplicate numbered files
- Annotation packages reference the dataset in-place by default
- Copied package images (when explicitly enabled) use collision-safe filenames: `split_grade_imageid.png`

---

## Repository Structure

```
knee-xrai/
├── KneeXrayData/          # Dataset (kneeKL224 split by KL grade)
├── annotations/           # Manifests, reviewed labels, masks
├── app/                   # Demo / inference app
├── checkpoints/           # Saved model checkpoints
├── configs/               # YAML configuration
├── docs/                  # Extended documentation
├── features/              # Extracted feature CSVs
├── notebooks/             # Exploratory & analysis notebooks
├── paper_figures/         # Figures used in the manuscript
├── report/                # Auto-generated reproducibility report
├── results/               # Evaluation outputs
├── scripts/               # Pipeline entry-point scripts
├── src/                   # Core source modules
├── tests/                 # Unit and integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Prerequisites

- Python 3.8+
- Git LFS (for large data files)
- (Optional) Docker & Docker Compose

---

## Setup

### 1 — Clone and create a virtual environment

```bash
git clone https://github.com/ois-lab/knee-xrai.git
cd knee-xrai
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### 2 — Dataset location

By default the pipeline expects the dataset at:

```
./KneeXrayData/ClsKLData/kneeKL224
```

If your dataset lives elsewhere, export the override before running any script:

```bash
export DATA_ROOT=/absolute/path/to/KneeXrayData
```

### 3 — Preflight check

```bash
PYTHONPATH=. ./.venv/bin/python scripts/check_pipeline_readiness.py --project-root .
```

---

## Configuration

### W&B Logging

Copy the example environment file and add your Weights & Biases API key:

```bash
cp .env.example .env
```

Minimal `.env`:

```dotenv
WANDB_API_KEY=your_wandb_api_key_here
```

Optional overrides:

```dotenv
WANDB_RUN_ID=
WANDB_RESUME=allow
DATA_ROOT=/absolute/path/to/KneeXrayData
```

Shell runners load `.env` automatically. The Lightning logger also reads it directly when training scripts are invoked outside the shell wrappers.

---

## Running the Pipeline

### Bootstrap baseline (quick start)

```bash
./scripts/run_initial_experiment.sh
```

### Full main-study pipeline

```bash
./scripts/run_main_study_pipeline.sh
```

### Resume from a checkpoint

If `feature_review_template.csv` has already been reviewed and the JSN export is available under `annotations/packages/jsn_contours/jsn-new.coco-segmentation/`, skip the completed stages:

```bash
./scripts/run_main_study_pipeline.sh --skip-stages 3,6,7,8,9,10
```

Skip a single stage:

```bash
./scripts/run_main_study_pipeline.sh --skip-stage 3
```

---

## Pipeline Stages

The main-study runner executes **34 stages**:

| # | Stage |
|---|---|
| 1 | Clean numbered duplicate files |
| 2 | Audit pipeline readiness |
| 3 | Build bootstrap pseudo-label suggestions |
| 4 | Refresh annotation manifests |
| 5 | Prepare annotation workspace packages |
| 6 | Prepare ROI YOLO dataset from reviewed boxes |
| 7 | Train ROI detector |
| 8 | Evaluate ROI detector |
| 9 | Compare ROI backends |
| 10 | Extract ROIs with trained detector or fallback |
| 11 | Import reviewed annotations |
| 12 | Train JSN segmenter |
| 13 | Extract JSN features and masks |
| 14 | Extract osteophyte ROI patches (CLAHE + JSN-guided boxes) |
| 15 | Train osteophyte grader from manual labels |
| 16 | Extract sclerosis features (manual teacher) |
| 17 | Train sclerosis classifier from manual labels |
| 18 | Generate high-confidence pseudo-labels |
| 19 | Train expanded osteophyte grader |
| 20 | Extract sclerosis features for expanded training |
| 21 | Train expanded sclerosis classifier |
| 22 | Extract osteophyte features |
| 23 | Re-extract final sclerosis features |
| 24 | Aggregate all features |
| 25 | Train KL XGBoost classifier |
| 26 | Train KL hybrid classifier |
| 27 | Evaluate JSN segmenter |
| 28 | Evaluate osteophyte grader |
| 29 | Evaluate sclerosis classifier |
| 30 | Run KL feature baselines |
| 31 | Run KL ablation studies |
| 32 | Evaluate end-to-end pipeline |
| 33 | Build reproducibility report |
| 34 | Final pipeline readiness audit |

---

## Annotation Workflow

After **Stage 5**, two files require human review:

```
annotations/packages/feature_grading/feature_review_template.csv
annotations/packages/jsn_contours/jsn_contour_manifest.csv
```

**Feature grading:**
- Fill the `final_*` columns directly in `feature_review_template.csv`
- Keep `image_id` and `split` unchanged

**JSN contours:**
- Annotate from `jsn_contour_manifest.csv`
- Export reviewed contours as COCO JSON to `annotations/reviewed/jsn_cvat_export.json`

Import back into training artifacts:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

This generates:

```
annotations/osteophyte_labels_reviewed.csv
annotations/sclerosis_labels_reviewed.csv
annotations/jsn_masks/{train,val,test}/*.png
```

---

## Annotation Packages & Portable Paths

`configs/config.yaml` resolves `project_root` from the active repo and `data_root` from `DATA_ROOT` or `./KneeXrayData`.

Regenerate manifests:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/bootstrap_pseudo_labels.py
PYTHONPATH=. ./.venv/bin/python scripts/refresh_annotation_manifests.py
```

Manifests write portable paths such as:

```
KneeXrayData/ClsKLData/kneeKL224/train/0/9001695L.png
```

This ensures reproducibility across machines without editing CSV files.

**Prepare annotation workspace:**

```bash
PYTHONPATH=. ./.venv/bin/python scripts/prepare_annotation_workspace.py
```

By default, image folders are *not* copied — manifests point directly at the local dataset. To enable image copying for annotation tooling (files named `split_grade_imageid.png`):

```bash
ANNOTATION_PACKAGE_COPY_IMAGES=1 PYTHONPATH=. ./.venv/bin/python scripts/prepare_annotation_workspace.py
```

---

## Key Outputs

| Path | Contents |
|---|---|
| `annotations/` | Labels, masks, manifests |
| `features/` | Aggregated feature CSVs |
| `checkpoints/` | Trained model weights |
| `results/` | Per-module evaluation metrics |
| `report/` | Reproducibility summary |

Notable directories used by the runner:

```
features/rois_osteophyte_clahe_full        # Osteophyte ROI patches
checkpoints/jsn_segmenter                  # JSN manual checkpoint
checkpoints/jsn_segmenter_selftrain        # JSN self-trained checkpoint
checkpoints/jsn_segmenter_selected.txt     # Pointer to best JSN checkpoint
annotations/jsn_pseudo_masks               # JSN pseudo-masks
features/sclerosis_manual_teacher          # Sclerosis teacher features (manual)
features/sclerosis_expanded_teacher        # Sclerosis teacher features (expanded)
features/sclerosis                         # Final sclerosis features
results/jsn_manual                         # JSN manual evaluation
results/jsn_selftrain                      # JSN self-trained evaluation
results/osteophyte_main_manual             # Osteophyte evaluation
report/                                    # Reproducibility report
```

---

## Optional Features

### JSN Self-Training Branch

By default, Stage 13 uses only the manually reviewed JSN checkpoint. Enable self-training to select the best checkpoint automatically:

```bash
ENABLE_JSN_SELF_TRAINING=1 ./scripts/run_main_study_pipeline.sh --skip-stages 3,6,7,8,9,10
```

With this flag the runner:
1. Generates high-confidence pseudo-masks
2. Trains a self-trained JSN checkpoint in parallel
3. Evaluates both checkpoints on reviewed test masks
4. Writes the winner to `checkpoints/jsn_segmenter_selected.txt`
5. Re-extracts JSN features using the selected checkpoint

### Optional ROI Detector Branch (Stages 6–10)

Stages 6–10 run only when reviewed ROI annotations are present under `annotations/reviewed/`:

```
roi_boxes.csv  |  roi_annotations.csv  |  roi_cvat_export.json  |  roi_boxes_coco.json
```

If any of these files are absent, the pipeline skips to the fallback study path automatically.

---

## Reproducibility

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PYTHON_BIN` | system python | Python interpreter used by shell runner |
| `DATA_ROOT` | `./KneeXrayData` | Path to external dataset |
| `ANNOTATION_PACKAGE_COPY_IMAGES` | `0` | Recreate copied annotation image folders |
| `ENABLE_JSN_SELF_TRAINING` | `0` | Run JSN self-training branch |
| `OSTEOPHYTE_REPRO_CKPT` | — | Override warm-start osteophyte checkpoint |
| `SCLEROSIS_TEACHER_CKPT` | — | Override teacher checkpoint for pseudo-label stages |

### Regenerate the reproducibility report only

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_repro_report.py --run-root .
```

### Clean duplicate files

```bash
PYTHONPATH=. ./.venv/bin/python scripts/clean_duplicate_suffix_files.py --project-root . --remove
```

Only exact safe copies (e.g. `train_sclerosis 2.py`) are removed. Conflicting files are reported and left untouched.

---

## Docker

```bash
docker compose up
```

See `Dockerfile` and `docker-compose.yml` for build details.

---

## Citation

```bash

@article{irfan2026knee,
  title={Knee-xRAI: An Explainable AI Framework for Automatic Kellgren-Lawrence Grading of Knee Osteoarthritis},
  author={Irfan, Azmul A and Khatim, Nur Ahmad and Irfan, Alfan Alfian and Zaki, Achmad and Suwarsono, Erike A and Arief, Mansur M},
  journal={arXiv preprint arXiv:2604.23435},
  year={2026}
}

```

---


## License

This project is for research purposes. See [LICENSE](LICENSE) for details.
