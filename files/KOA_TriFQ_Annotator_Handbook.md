# KOA-TriFQ Annotator Handbook

## Purpose

This handbook consolidates the annotation instructions needed for the KOA-TriFQ study implementation. It is aligned with:

- `files/KOA_TriFQ_Research_Plan.md`
- `files/KOA_TriFQ_Annotation_and_Timeline.md`
- `files/KOA_TriFQ_Execution_Guide.md`

Use this as the operational guide for preparing reviewed annotations that the codebase can import directly.

## What Needs Annotation

There are 3 annotation tasks relevant to the main-study path:

1. JSN contour annotation
2. Osteophyte grading at 4 anatomical ROI sites
3. Sclerosis grading at 2 compartments

There is also an optional Stage 1 ROI box annotation task if you want to train the YOLOv8-m ROI detector.

## 1. JSN Annotation

### Goal

Annotate joint-space boundaries so the system can build segmentation masks and compute:

- `mJSW_medial`
- `mJSW_lateral`
- `jsn_rate_medial`
- `jsn_rate_lateral`
- `jsw_profile`

### Source of truth in repo

- Research basis: `files/KOA_TriFQ_Research_Plan.md`
- Detailed protocol: `files/KOA_TriFQ_Annotation_and_Timeline.md`
- Package manifest: `annotations/packages/jsn_contours/jsn_contour_manifest.csv`
- Package note: `annotations/packages/jsn_contours/README.md`

### Images to annotate

Use the images listed in:

- `annotations/packages/jsn_contours/jsn_contour_manifest.csv`

The linked image files are under:

- `annotations/packages/jsn_contours/images/`

### What to draw

For each image, annotate the tibiofemoral joint-space boundaries so the import script can reconstruct a 3-class mask.

Required structures:

- `femoral_surface`
- `tibial_surface`

These should represent the articular boundary lines spanning the lateral and medial compartments across the visible tibiofemoral joint.

### Output format expected by implementation

Export the reviewed CVAT file to:

- `annotations/reviewed/jsn_cvat_export.json`

The import path is:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

That will generate split-aware masks under:

- `annotations/jsn_masks/train/`
- `annotations/jsn_masks/val/`
- `annotations/jsn_masks/test/`

### Annotation notes

- Follow the visible femoral and tibial joint margins, not bone shafts.
- Keep contours smooth and anatomically faithful.
- Cover both compartments when visible.
- If one compartment is poorly visible, still trace the most plausible visible boundary rather than leaving random gaps.

## 2. Osteophyte Annotation

### Goal

Grade osteophytes at 4 sites according to OARSI-style severity.

Required sites:

- Medial femur: `mf`
- Lateral femur: `lf`
- Medial tibia: `mt`
- Lateral tibia: `lt`

### Source of truth in repo

- Research basis: `files/KOA_TriFQ_Research_Plan.md`
- Detailed protocol: `files/KOA_TriFQ_Annotation_and_Timeline.md`
- Review sheet: `annotations/packages/feature_grading/feature_review_template.csv`

### Images to review

Use:

- `annotations/packages/feature_grading/feature_review_template.csv`

The linked review images are under:

- `annotations/packages/feature_grading/images/`

### What to fill

Fill these columns in the CSV:

- `final_osp_mf`
- `final_osp_lf`
- `final_osp_mt`
- `final_osp_lt`

Optional confidence/notes fields:

- `confidence_mf`
- `confidence_lf`
- `confidence_mt`
- `confidence_lt`
- `notes`

### Grade scale

Use the 4-level osteophyte grade:

- `0`: absent
- `1`: small / doubtful
- `2`: definite / moderate
- `3`: large / severe

### Site definitions

- `mf`: medial femoral margin at the joint line
- `lf`: lateral femoral margin at the joint line
- `mt`: medial tibial margin at the joint line
- `lt`: lateral tibial margin at the joint line

### Annotation notes

- Grade each site independently.
- Use the marginal outgrowth at the joint edge, not diffuse sclerosis or shape changes elsewhere.
- Do not infer the grade from KL label alone.
- If uncertain, assign the best anatomical judgment and record the uncertainty in the confidence field or `notes`.

### Output format expected by implementation

Keep the completed review file as:

- `annotations/packages/feature_grading/feature_review_template.csv`

Then import it with:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

That will generate:

- `annotations/osteophyte_labels_reviewed.csv`

## 3. Sclerosis Annotation

### Goal

Grade subchondral sclerosis severity in the 2 tibial compartments.

Required compartments:

- Medial
- Lateral

### Source of truth in repo

- Research basis: `files/KOA_TriFQ_Research_Plan.md`
- Detailed protocol: `files/KOA_TriFQ_Annotation_and_Timeline.md`
- Review sheet: `annotations/packages/feature_grading/feature_review_template.csv`

### What to fill

Fill these columns in the same review CSV:

- `final_scl_medial`
- `final_scl_lateral`

Optional confidence/notes fields:

- `scl_confidence_med`
- `scl_confidence_lat`
- `notes`

### Grade scale

Use the 3-level sclerosis grade:

- `0`: none / minimal
- `1`: mild
- `2`: significant

### Region definition

Grade sclerosis in the subchondral bone immediately below the tibial plateau in each compartment.

Focus on:

- increased bone density
- thicker/brighter subchondral bone appearance
- compartment-specific severity

Do not use:

- osteophytes as a substitute for sclerosis
- generalized exposure differences alone

### Output format expected by implementation

After review, import with:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

That will generate:

- `annotations/sclerosis_labels_reviewed.csv`

## 4. Optional ROI Box Annotation

### Goal

Train the Stage 1 ROI detector with 5 boxes per image.

### Required classes

- `joint_space`
- `medial_femur`
- `lateral_femur`
- `medial_tibia`
- `lateral_tibia`

### Expected reviewed location

Place reviewed ROI annotations under:

- `annotations/reviewed/`

Supported inputs in the implementation:

- `roi_boxes.csv`
- `roi_annotations.csv`
- `roi_cvat_export.json`
- `roi_boxes_coco.json`

### Conversion path

```bash
PYTHONPATH=. ./.venv/bin/python scripts/prepare_roi_yolo_dataset.py
PYTHONPATH=. ./.venv/bin/python scripts/train_roi_detector.py +model=yolov8
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_roi_detector.py +model=yolov8
```

If ROI annotation is not available, the repo can still run with:

- KNEEL landmark backend
- geometric fallback

## 5. Practical Workflow

### JSN

1. Open the images from `annotations/packages/jsn_contours/images/`
2. Annotate `femoral_surface` and `tibial_surface`
3. Export to `annotations/reviewed/jsn_cvat_export.json`
4. Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

### Osteophyte and Sclerosis

1. Open `annotations/packages/feature_grading/feature_review_template.csv`
2. Review the linked images in `annotations/packages/feature_grading/images/`
3. Fill:
   - osteophyte: `final_osp_mf`, `final_osp_lf`, `final_osp_mt`, `final_osp_lt`
   - sclerosis: `final_scl_medial`, `final_scl_lateral`
4. Run:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/import_reviewed_annotations.py
```

## 6. After Annotation

Main-study training sequence:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/train_jsn_segmenter.py training.label_mode=manual +model=unetpp
PYTHONPATH=. ./.venv/bin/python scripts/train_osteophyte_grader.py training.label_mode=manual +model=se_resnet50
PYTHONPATH=. ./.venv/bin/python scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid
PYTHONPATH=. ./.venv/bin/python scripts/train_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid
```

## 7. Important Rules

- `manual` mode now means reviewed/manual labels only
- bootstrap suggestions are not manual truth
- do not report bootstrap-only labels as the main-study annotation set
- keep the test split untouched for final evaluation

## 8. Key Files

- `files/KOA_TriFQ_Research_Plan.md`
- `files/KOA_TriFQ_Annotation_and_Timeline.md`
- `files/KOA_TriFQ_Execution_Guide.md`
- `annotations/packages/feature_grading/feature_review_template.csv`
- `annotations/packages/jsn_contours/jsn_contour_manifest.csv`
