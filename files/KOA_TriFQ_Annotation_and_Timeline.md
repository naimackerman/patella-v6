# KOA-TriFQ Supplement: Annotation Protocols & Detailed Day-by-Day Timeline

---

## Part A: Revised Osteophyte Strategy (Without OAI OARSI Access)

### The Challenge

The shashwatwork Kaggle dataset provides **only KL grade labels (0–4)** — no per-compartment OARSI osteophyte grades, no JSN sub-scores, no sclerosis annotations. This means we cannot directly train a per-ROI osteophyte classifier using supervised labels from the dataset alone.

### The Solution: Three-Tier Labeling Strategy

We use a combination of **KL-grade-derived weak pseudo-labels**, **manual expert annotation of a targeted subset**, and **semi-supervised refinement**.

#### Tier 1: KL-Grade-Derived Pseudo-Labels (All 9,786 images, automated)

Based on Ko et al. (2025, Osteoarthritis and Cartilage Open, DOI: 10.1016/j.ocarto.2025.100566), who validated OARSI-to-KL criteria on 10,804 knees with sensitivity/specificity ≥ 0.92, we can derive conservative **lower-bound pseudo-labels** for each KL grade:

| KL Grade | Implied JSN (OARSI) | Implied Osteophyte Sum | Implied Sclerosis |
|----------|---------------------|----------------------|-------------------|
| **KL 0** | JSN = 0 | Sum osteophyte = 0 | None |
| **KL 1** | JSN ≤ 1 | Sum osteophyte ≤ 1 | None |
| **KL 2** | JSN ≤ 1 | Sum osteophyte ≥ 1 (definite) | Possible/None |
| **KL 3** | JSN = 2 (definite) | Sum osteophyte ≥ 2 (moderate, multiple) | Some sclerosis |
| **KL 4** | JSN = 3 (marked) | Sum osteophyte ≥ 3 (large) | Severe sclerosis |

**How to use these**: For the per-ROI osteophyte classifier, we can generate **weak binary labels** (osteophyte present vs absent per ROI) using the following rule:
- KL 0–1: label all 4 ROIs as "likely absent" (grade 0)
- KL 2: label as "likely present somewhere" — requires manual verification for site-specific assignment
- KL 3–4: label as "definitely present in at least 2 sites" — still requires manual ROI-level assignment

This gives us **strong negative examples** (KL 0–1, ~4,694 images where osteophytes are absent or minimal) and **strong positive examples** (KL 3–4, ~1,328 images where osteophytes are definitely present), but requires manual annotation for the ambiguous middle (KL 2) and for site-specific localization.

#### Tier 2: Manual Expert Annotation (500 targeted images)

Select a stratified subset of **500 images** (100 per KL grade) for full manual annotation. This subset receives:
- Per-ROI osteophyte grades (0–3) at 4 sites
- JSN severity (medial/lateral, 0–3)
- Sclerosis grade (medial/lateral, 0–2)

This forms the **gold-standard training set** for the feature-specific classifiers.

#### Tier 3: Semi-Supervised Expansion

1. Train initial models on the 500 manually annotated images
2. Apply to the remaining ~9,286 images to generate pseudo-predictions
3. Select high-confidence predictions (softmax > 0.90) as pseudo-labels
4. Retrain with the expanded labeled set
5. Repeat for 2–3 iterations until convergence

This approach is validated by the Cleanlab framework's success on this exact dataset (MDPI Diagnostics, 2025) and follows standard teacher-student semi-supervised learning protocols.

---

## Part B: Complete Annotation Protocol

### B.1 Annotation Tool Setup

**Recommended Tool**: CVAT (Computer Vision Annotation Tool)

```bash
# Install CVAT locally via Docker
git clone https://github.com/opencv/cvat.git
cd cvat
docker compose up -d
# Access at http://localhost:8080
```

**Alternative for lighter setup**: Label Studio (pip install label-studio)

**Pre-annotation acceleration**: Use MedSAM (Segment Anything for Medical Images) to generate initial mask proposals that annotators can refine:
```python
# Generate mask proposals with MedSAM
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
predictor = SamPredictor(sam)
predictor.set_image(xray_image)
# Provide bounding box prompt for joint space region
masks, scores, _ = predictor.predict(box=np.array([x1, y1, x2, y2]))
```

### B.2 Annotation Protocol: Joint Space Boundaries (for JSN Quantification)

**Goal**: Annotate the femoral condyle inferior surface and tibial plateau superior surface as polygon contours.

**Number of images**: 300–500 images, stratified: 60–100 per KL grade

**Annotation steps per image**:

1. **Identify the tibiofemoral joint space** — the radiolucent (dark) band between the femoral condyle and tibial plateau

2. **Trace the femoral condyle inferior surface** (the upper boundary of the joint space):
   - Start from the most medial visible point of the medial femoral condyle
   - Trace along the articular surface moving laterally
   - Follow the curved bone-cartilage interface faithfully
   - End at the most lateral visible point of the lateral femoral condyle
   - Use **polyline** annotation type (not polygon)
   - Place points approximately every 5–8 pixels along the contour
   - Place extra points at high-curvature regions (intercondylar notch, condyle edges)

3. **Trace the tibial plateau superior surface** (the lower boundary of the joint space):
   - Start from the most medial point of the medial tibial plateau
   - Trace along the plateau surface laterally
   - Navigate around the tibial spines (intercondylar eminence) — trace the articular surface, not the spine tips
   - End at the most lateral point of the lateral tibial plateau

4. **Quality checks**:
   - Contours should NOT cross each other
   - Both contours should span the full mediolateral extent of the joint
   - In severe JSN (KL 3–4), where bone-on-bone contact exists, femoral and tibial contours may touch or nearly touch — this is correct
   - Mark any image where the joint space is completely obliterated as "bone-on-bone" flag

**Output format**: Save as COCO JSON format with two annotation classes:
- Class 1: `femoral_surface` (polyline)
- Class 2: `tibial_surface` (polyline)

**Conversion to mask**: For U-Net++ training, convert paired polylines to a filled mask:
```python
def contours_to_mask(femoral_pts, tibial_pts, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    # Fill region between femoral (upper) and tibial (lower) contours
    for x in range(img_shape[1]):
        y_fem = interpolate_y(femoral_pts, x)
        y_tib = interpolate_y(tibial_pts, x)
        if y_fem is not None and y_tib is not None:
            # Determine medial vs lateral based on x position relative to midline
            compartment = 1 if x > midline else 2  # 1=medial, 2=lateral
            mask[int(y_fem):int(y_tib), x] = compartment
    return mask
```

**Inter-annotator agreement**: Have 2 annotators independently trace 50 images. Compute boundary agreement as mean point-to-curve distance (target: < 2 pixels).

### B.3 Annotation Protocol: Osteophyte Grading (per ROI)

**Goal**: Grade osteophyte severity at 4 anatomical sites using the OARSI atlas criteria (0–3 scale).

**Number of images**: 500 images, stratified: 100 per KL grade

**OARSI Osteophyte Grading Criteria** (reference atlas: Altman & Gold, 2007):

| Grade | Description | Visual Cues on X-ray |
|-------|-------------|---------------------|
| **0** | No osteophyte | Smooth bone margins, no bony protrusions |
| **1** | Small/doubtful | Tiny lip or point of new bone at the joint margin, may be difficult to distinguish from normal anatomy. Size: < 2mm equivalent |
| **2** | Definite/moderate | Clear bony protrusion at the joint margin, easily visible. Size: 2–5mm equivalent. Well-defined triangular or shelf-like shape |
| **3** | Large/severe | Large bony outgrowth extending significantly from the joint margin. Size: > 5mm equivalent. May alter bone contour substantially |

**Four anatomical ROI sites** (annotate each independently):

1. **Medial Femur (MF)**: The medial margin of the medial femoral condyle. Look for bony protrusions extending medially from the condyle edge.

2. **Lateral Femur (LF)**: The lateral margin of the lateral femoral condyle. Look for bony protrusions extending laterally. NOTE: The popliteal groove can mimic an osteophyte — distinguish by its smooth, concave contour vs the irregular, pointed shape of true osteophytes.

3. **Medial Tibia (MT)**: The medial margin of the medial tibial plateau. Look for bony lips or shelves extending medially below the joint line.

4. **Lateral Tibia (LT)**: The lateral margin of the lateral tibial plateau. Look for bony outgrowths at the tibial edge. NOTE: This is the most consistently visible osteophyte site and often the easiest to grade.

**Annotation steps per image**:

1. Open the image in CVAT/Label Studio
2. For each of the 4 ROI sites:
   a. Visually inspect the bone margin at that site
   b. Compare against the OARSI reference atlas examples
   c. Assign a grade (0, 1, 2, or 3) as an **image-level tag** (not a spatial annotation)
   d. If grade ≥ 1, optionally draw a **bounding box** around the osteophyte for Grad-CAM validation
3. Record confidence level for each grading decision: `high`, `medium`, or `low`
4. Flag any images where view quality prevents reliable assessment

**Output format**: CSV file with columns:
```
image_id, osp_mf, osp_lf, osp_mt, osp_lt, confidence_mf, confidence_lf, confidence_mt, confidence_lt, notes
KL0_001.png, 0, 0, 0, 0, high, high, high, high, ""
KL3_042.png, 2, 1, 3, 2, high, low, high, high, "LF difficult due to overlap"
```

**Critical guidance for the radiologist annotator**:
- Use the OARSI Knee OA Atlas as reference (Altman RD, Gold GE. Atlas of individual radiographic features in osteoarthritis, revised. Osteoarthritis and Cartilage, 2007;15:A1-A56)
- At 224×224 pixel resolution, grade 1 (doubtful) osteophytes may be genuinely invisible — grade what you can see, and mark lower confidence where resolution is insufficient (`medium` or `low`)
- For KL 0 images, spend minimal time — most should be all-zeros
- For KL 3–4 images, all 4 sites typically have grade ≥ 1; focus on distinguishing 2 vs 3
- If tibial spines are prominent, do NOT count them as osteophytes — they are normal anatomical variants

### B.4 Annotation Protocol: Subchondral Sclerosis Grading

**Goal**: Grade sclerosis severity in medial and lateral subchondral tibial bone.

**Number of images**: 500 images (same set as osteophyte annotation)

**Sclerosis Grading Criteria** (adapted from Kim et al., 2025, Sensors):

| Grade | Description | Visual Cues on X-ray |
|-------|-------------|---------------------|
| **0 (None/Minimal)** | Normal subchondral bone density | Trabecular pattern visible through subchondral plate, bone appears uniformly gray |
| **1 (Mild)** | Subtle increase in bone density | Slight whitening/brightening of subchondral region, trabecular pattern partially obscured but still somewhat visible |
| **2 (Significant)** | Marked increase in bone density | Distinctly white/bright subchondral band, trabecular pattern completely obscured, sharp demarcation between sclerotic and normal bone |

**Assessment regions** (2 per image):
- **Medial subchondral region**: The zone of tibial bone immediately below the medial tibial articular surface, approximately 5–10mm deep
- **Lateral subchondral region**: The corresponding zone below the lateral tibial articular surface

**Annotation steps per image**:

1. Adjust your display brightness/contrast to clearly visualize bone density differences
2. Compare the subchondral bone region (just below the joint surface) against the mid-shaft tibial bone density:
   - If the subchondral region looks the same density → Grade 0
   - If noticeably brighter but trabecular texture still partly visible → Grade 1
   - If significantly brighter with a dense white band → Grade 2
3. Grade medial and lateral regions independently
4. Look for asymmetry — in medial-compartment OA, medial sclerosis typically exceeds lateral

**Expected distribution by KL grade** (validation check):
- KL 0–1: ~95% should be grade 0 both sides
- KL 2: ~60% grade 0, ~35% grade 1, ~5% grade 2
- KL 3: ~10% grade 0, ~50% grade 1, ~40% grade 2
- KL 4: ~5% grade 0, ~25% grade 1, ~70% grade 2

If your annotations deviate substantially from these distributions, review your calibration against the reference examples.

**Output format**: Append to the same CSV:
```
image_id, ..., scl_medial, scl_lateral, scl_confidence_med, scl_confidence_lat
```

Recommended confidence rubric:
- `high`: clear visual evidence and confident compartment assignment
- `medium`: plausible grade but mild ambiguity from overlap, low resolution, or marginal visibility
- `low`: major ambiguity; keep the label if clinically defensible, but allow downstream sensitivity analysis or down-weighting

### B.5 Annotation Workflow Summary

**Total annotation effort for 500 images**:
- Joint space contours (300–500 images): ~3–5 minutes per image = 15–42 hours
- Osteophyte grading (500 images): ~1–2 minutes per image = 8–17 hours
- Sclerosis grading (500 images): ~1 minute per image = 8 hours
- **Total estimated time: 31–67 hours** (spread across ~2–4 weeks with breaks)

**Recommended annotation schedule**:
- Session 1 (calibration): Annotate 20 images together with radiologist, discuss ambiguous cases, establish consensus criteria
- Sessions 2–5: Annotator works independently, 100–120 images per session
- Session 6 (quality check): Review 50 randomly selected annotations with radiologist, measure inter-annotator agreement
- Sessions 7–10: Complete remaining images with any calibration adjustments

### B.6 How to Handle 224×224 Resolution Limitations

The Kaggle dataset's 224×224 pixel images are downsampled from OAI originals (~2048×2560). This causes:

1. **Grade 1 osteophytes may be invisible** — a 1–2mm osteophyte maps to ~1–2 pixels. Annotators should mark these as "uncertain" rather than guessing.

2. **Fine trabecular texture for sclerosis is partially lost** — LBP and GLCM features will capture coarser texture patterns. The fractal dimension and intensity-based features are more robust to downsampling.

3. **Joint space boundary annotation is less precise** — contour points have ±1 pixel uncertainty. This translates to ±1–2 pixel uncertainty in mJSW measurement, acceptable given that the entire joint space spans ~10–30 pixels at this resolution.

**Mitigation**: In your paper, explicitly discuss these resolution limitations and recommend that clinical deployment use higher-resolution inputs. Position the 224×224 work as a proof-of-concept that the framework architecture works, with expected performance improvement at higher resolution.

---

## Part C: Detailed Day-by-Day Timeline (24 Weeks)

### Phase 1: Data Preparation (Weeks 1–3, 15 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **1** | Download shashwatwork Kaggle dataset; set up project repository (Git + DVC); install PyTorch + MONAI + SMP environments | Repository initialized, dataset downloaded |
| **2** | Exploratory Data Analysis: compute class distribution, visualize samples from each KL grade, check for duplicates/corrupted images, measure image statistics (mean, std, min/max pixel values) | EDA notebook with distribution plots |
| **3** | Implement preprocessing pipeline: CLAHE, histogram clipping, normalization; test augmentation transforms; implement data loading with PyTorch DataLoader and weighted sampler | `preprocessing.py`, `dataset.py` modules |
| **4** | Set up experiment tracking (W&B), Hydra configs for hyperparameters, PyTorch Lightning training template; define project directory structure | `configs/`, `train.py`, W&B integration |
| **5** | Install and configure CVAT annotation tool; create annotation project with label schema (joint space contours, osteophyte grades, sclerosis grades) | CVAT running, project configured |
| **6** | Annotation calibration session with radiologist: annotate 20 images together, establish consensus for ambiguous cases, create reference image gallery | Annotation protocol finalized |
| **7** | Begin joint space contour annotation (batch 1: 100 images — 20 per KL grade) | 100 annotated contours |
| **8** | Continue joint space contour annotation (batch 2: 100 images) | 200 annotated contours |
| **9** | Continue joint space contour annotation (batch 3: 100 images) | 300 annotated contours |
| **10** | Begin osteophyte + sclerosis grading annotation (batch 1: 125 images) | 125 graded images |
| **11** | Continue osteophyte + sclerosis annotation (batch 2: 125 images) | 250 graded images |
| **12** | Continue osteophyte + sclerosis annotation (batch 3: 125 images) | 375 graded images |
| **13** | Complete osteophyte + sclerosis annotation (batch 4: 125 images); generate pseudo-labels for remaining images using KL-grade rules | 500 graded images + pseudo-labels |
| **14** | Inter-annotator agreement check: re-annotate 50 images, compute Cohen's κ for osteophytes, ICC for sclerosis, point-to-curve distance for contours | Agreement report |
| **15** | Convert all annotations to training format: COCO masks for JSN, CSV for features; implement train/val/test split (70/15/15 stratified); finalize data pipeline | Training-ready data |

### Phase 2: Stage 1 — ROI Detection (Weeks 3–5, 10 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **16** | Prepare YOLOv8 dataset: convert 300 annotated joint space images to YOLO format with 5 bounding box classes (joint_roi, mf, lf, mt, lt) | YOLO dataset YAML |
| **17** | Train YOLOv8-m: initial training run (50 epochs), monitor mAP on validation set | First model checkpoint |
| **18** | Hyperparameter tuning: try different image sizes (224, 416), augmentation settings, confidence thresholds | Comparison table |
| **19** | Extended training (100 epochs) with best config; evaluate on held-out test set | Best YOLOv8 model |
| **20** | Implement KNEEL landmark detection as alternative approach: download pre-trained model, test on 50 images, compare with YOLO-based ROI extraction | KNEEL vs YOLO comparison |
| **21** | Select best ROI approach; implement post-processing (confidence filtering, NMS, ROI normalization) | `roi_detector.py` module |
| **22** | Run ROI detection on ALL 9,786 images; save cropped ROIs (joint space patches + 4 osteophyte ROI patches per image) | Cropped ROI dataset |
| **23** | Quality audit: manually check 200 random detections for accuracy, flag failures | QA report, error analysis |
| **24** | Fix edge cases (missed detections, wrong laterality); implement fallback for failed detections | Robust ROI pipeline |
| **25** | Document Stage 1 results: mAP, detection rate, failure modes; freeze ROI detector | Stage 1 complete |

### Phase 3: Stage 2A — JSN Module (Weeks 5–8, 15 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **26** | Prepare segmentation dataset: convert contour annotations to 3-class masks (background/medial/lateral); split into train/val/test | Segmentation dataset |
| **27** | Implement U-Net++ model with EfficientNet-B4 encoder using SMP; implement DiceCE loss | `jsn_segmentation.py` |
| **28** | Training run 1: 100 epochs, lr=1e-4, batch=16; monitor Dice score per class | First segmentation model |
| **29** | Training run 2: experiment with alternative encoders (ResNet-50, EfficientNet-B7); compare Dice scores | Encoder comparison |
| **30** | Training run 3: add MedSAM-initialized weights experiment; test attention gates | Advanced architectures |
| **31** | Select best model; train for 200 epochs with cosine annealing; early stopping on val Dice | Best segmentation model |
| **32** | Evaluate on test set: compute Dice, Hausdorff95, per-class metrics; visualize predictions overlaid on images | Segmentation evaluation |
| **33** | Implement contour extraction from predicted masks: Canny edges, cv2.findContours, contour smoothing | `contour_extraction.py` |
| **34** | Implement mJSW computation: contour-based pairwise distances with local mediolateral matching, interior weight-bearing minimum search, medial/lateral partitioning, multi-point JSW profile (16 points) | `jsw_computation.py` |
| **35** | Implement JSN rate computation: compute median mJSW from KL-0 subset, apply formula | `jsn_rate.py` |
| **36** | Validate mJSW against manual measurements: correlate automated mJSW with annotated contour distances on 100 test images using the same interior compartment measurement rule | mJSW validation report |
| **37** | Run full JSN pipeline on all 9,786 images; save feature vectors | JSN features for all images |
| **38** | Error analysis: identify failure modes (bone-on-bone, poor segmentation, incorrect compartment assignment) | Error catalog |
| **39** | Implement JSN visualization: distance lines overlaid on X-ray image using the same contour-distance rule used for mJSW computation | JSN overlay renderer |
| **40** | Document Stage 2A: Dice scores, mJSW MAE, JSN rate distributions by KL grade; write methods section draft | Stage 2A complete |

### Phase 4: Stage 2B — Osteophyte Module (Weeks 8–11, 15 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **41** | Prepare per-ROI classification dataset: extract 4 ROI patches per image using Stage 1 detector; organize by site and grade | Per-ROI patch dataset |
| **42** | Implement SE-ResNet-50 multi-task model with 4 classification heads; implement ordinal cross-entropy loss | `osteophyte_grader.py` |
| **43** | Training run 1: train on 500 manually annotated images only, 100 epochs | Baseline model |
| **44** | Generate pseudo-labels for KL 0–1 (all grade-0) and KL 3–4 (using model predictions + KL-grade constraints) | Pseudo-label set |
| **45** | Training run 2: semi-supervised — retrain with expanded dataset (500 manual + high-confidence pseudo-labels) | Semi-supervised model v1 |
| **46** | Training run 3: second round of pseudo-label generation + retraining | Semi-supervised model v2 |
| **47** | Evaluate on test set: Cohen's κ per site, balanced accuracy, confusion matrices | Per-site evaluation |
| **48** | Implement Grad-CAM extraction from SE-ResNet-50 last conv layer for each ROI | `gradcam_osteophyte.py` |
| **49** | Validate Grad-CAM: overlay heatmaps on 50 test images, check if highlighted regions correspond to annotated osteophyte locations | Grad-CAM validation report |
| **50** | Experiment with alternative architecture: EfficientNet-B2 backbone comparison | Architecture comparison |
| **51** | Implement osteophyte visualization: severity-colored bounding boxes + Grad-CAM overlay at 4 sites | Osteophyte overlay renderer |
| **52** | Run full osteophyte pipeline on all 9,786 images; save grade predictions + Grad-CAM maps | Osteophyte features for all |
| **53** | Analyze osteophyte grade distributions by KL grade; validate against Ko et al. (2025) expected patterns | Distribution validation |
| **54** | Error analysis and model refinement for difficult cases (grade 1 vs 0, lateral femur site) | Error analysis report |
| **55** | Document Stage 2B: per-site κ, semi-supervised improvement curves; write methods section draft | Stage 2B complete |

### Phase 5: Stage 2C — Sclerosis Module (Weeks 11–15, 20 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **56** | Implement automatic subchondral ROI extraction using Stage 2A segmentation masks: define region 5–10px below tibial surface | `subchondral_roi.py` |
| **57** | Extract subchondral ROI patches for all annotated images; visualize to verify ROI placement accuracy | Verified ROI patches |
| **58** | Implement LBP feature extraction: multi-scale (radius 1, 2, 3), uniform patterns; compute histograms | `lbp_features.py` |
| **59** | Implement GLCM/Haralick feature extraction: contrast, dissimilarity, homogeneity, energy, correlation at multiple distances and angles | `glcm_features.py` |
| **60** | Implement fractal dimension computation (box-counting method) and intensity statistics (mean, std, skewness, kurtosis, entropy) | `texture_features.py` |
| **61** | Feature analysis on annotated set: correlate each texture feature with sclerosis grade and KL grade; identify most discriminative features | Feature importance analysis |
| **62** | Train baseline ML classifiers on texture features alone: SVM, Random Forest, XGBoost; 5-fold CV | Texture-only baselines |
| **63** | Implement EfficientNet-B0 CNN branch for subchondral ROI patches; train on 500 annotated images | CNN-only baseline |
| **64** | Implement hybrid model: CNN (EfficientNet-B0) + texture MLP fusion; implement training loop | `sclerosis_classifier.py` |
| **65** | Training run 1: hybrid model, 100 epochs, lr=1e-4 | Hybrid model v1 |
| **66** | Hyperparameter tuning: fusion strategy (early vs late), dropout rates, texture feature selection | Tuning results |
| **67** | Training run 2: best config, 200 epochs with cosine annealing | Best hybrid model |
| **68** | Evaluate on test set: accuracy, per-class AUC, confusion matrix; compare texture-only vs CNN-only vs hybrid | Comparative evaluation |
| **69** | Semi-supervised expansion: generate pseudo-labels using KL-grade constraints (KL 0–1 → grade 0, KL 4 → grade ≥ 1) | Expanded training set |
| **70** | Retrain hybrid model with semi-supervised data; evaluate improvement | Semi-supervised model |
| **71** | Implement sclerosis heatmap generation: use LBP/intensity spatial mapping to create pixel-wise density overlay | `sclerosis_heatmap.py` |
| **72** | Validate heatmap: overlay on 50 test images, verify that bright regions align with radiologist-identified sclerosis zones | Heatmap validation |
| **73** | Run full sclerosis pipeline on all 9,786 images; save grade predictions + texture features + heatmaps | Sclerosis features for all |
| **74** | Statistical validation: compute correlation of sclerosis scores with KL grade (expected r ≥ 0.65) | Correlation analysis |
| **75** | Document Stage 2C: accuracy metrics, feature importance, semi-supervised gains; write methods section draft | Stage 2C complete |

### Phase 6: Stage 3–4 — Integration & Classification (Weeks 15–18, 15 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **76** | Build feature aggregation pipeline: combine JSN (22 dims) + osteophyte (10 dims) + sclerosis (18 dims) into 50-dim vector per image | `feature_aggregator.py` |
| **77** | Feature analysis: PCA visualization, feature correlation matrix, ANOVA per KL grade, feature importance ranking | Feature engineering notebook |
| **78** | Train XGBoost classifier (Path A): 5-fold stratified CV, grid search for max_depth, n_estimators, learning_rate | XGBoost baseline |
| **79** | Train MLP classifier comparison: 2-layer MLP on feature vector; train Random Forest comparison | ML classifier comparison |
| **80** | Implement SHAP explanations for XGBoost: global feature importance + per-sample waterfall plots | `shap_explanation.py` |
| **81** | Implement ConvNeXt-Small image encoder for Hybrid Path B; freeze backbone initially | Hybrid encoder setup |
| **82** | Implement feature concatenation layer: ConvNeXt image features (768d) + extracted features (50d) → fusion MLP | `hybrid_classifier.py` |
| **83** | Training run 1: Hybrid model, 100 epochs, lr=1e-4, ordinal soft-label loss | Hybrid model v1 |
| **84** | Training run 2: unfreeze backbone, fine-tune end-to-end with lower lr (1e-5) for 50 epochs | Hybrid model v2 |
| **85** | Evaluate both paths on test set: QWK, accuracy, per-class F1, confusion matrix | Path A vs Path B evaluation |
| **86** | **Ablation study 1**: JSN-only → KL grade | Ablation results |
| **87** | **Ablation study 2**: JSN + osteophytes → KL grade | Ablation results |
| **88** | **Ablation study 3**: Full pipeline (JSN + osteophytes + sclerosis) → KL grade; compute improvement over each ablation | Complete ablation table |
| **89** | Comparison with baselines: train end-to-end ConvNeXt (no features), compare QWK | Baseline comparison |
| **90** | Document Stage 3–4: classification results, ablation analysis, SHAP analysis; write results section draft | Stage 3–4 complete |

### Phase 7: Stage 5 — XAI Visualization (Weeks 18–20, 10 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **91** | Implement multi-layer overlay renderer: combine JSN lines (blue), osteophyte boxes (severity-colored), sclerosis heatmap (yellow-red) on single image | `xai_overlay.py` |
| **92** | Implement template-based clinical narrative generator: structured report from feature values | `report_generator.py` |
| **93** | Design and implement interactive Gradio demo: upload image → full pipeline → annotated image + report | `gradio_app.py` |
| **94** | Polish Gradio UI: add toggle switches for each overlay layer, confidence displays, grade distribution bar chart | Refined demo |
| **95** | Generate XAI outputs for 100 test images; compile into a visual atlas for radiologist evaluation | XAI atlas |
| **96** | Radiologist evaluation session: present 100 annotated images, collect ratings on overlay accuracy and clinical utility | Clinician feedback |
| **97** | Iterate on visualization based on feedback: adjust colors, label placement, report wording | Refined visualization |
| **98** | Implement PDF report export: generate single-page clinical report with annotated image + findings table | PDF report generator |
| **99** | Implement ONNX export for all models; benchmark inference time (target: < 5 seconds per image on GPU) | ONNX models + benchmark |
| **100** | Docker containerization: package full pipeline into deployable container | Dockerfile + docker-compose |

### Phase 8: Paper Writing & Deployment (Weeks 20–24, 20 working days)

| Day | Activities | Deliverables |
|-----|-----------|-------------|
| **101** | Write Introduction section: problem statement, motivation, contribution claims | Draft intro |
| **102** | Write Related Work section: organize by feature (JSN, osteophyte, sclerosis, XAI) | Draft related work |
| **103** | Compile and polish Methods section from phase drafts: dataset, preprocessing, each stage architecture | Draft methods |
| **104** | Write Methods (continued): training details, evaluation metrics, ablation design | Complete methods |
| **105** | Write Results section: compile all tables, figures, statistical tests | Draft results |
| **106** | Create all figures: architecture diagram, overlay examples, confusion matrices, ROC curves, SHAP plots, ablation bar charts | Publication figures |
| **107** | Write Discussion section: comparison with SOTA, contribution analysis, limitations | Draft discussion |
| **108** | Write Conclusion + Future Work | Draft conclusion |
| **109** | Write Abstract; compile References (BibTeX) | Complete first draft |
| **110** | Internal review round 1: self-edit for clarity, consistency, logic flow | Revised draft |
| **111** | Co-author/advisor review; collect feedback | Feedback incorporated |
| **112** | Statistical review: verify all reported metrics, confidence intervals, p-values | Verified statistics |
| **113** | Revise manuscript based on all feedback | Second draft |
| **114** | Prepare supplementary materials: additional visualizations, full results tables, code availability statement | Supplementary PDF |
| **115** | Final manuscript polish: formatting per target journal guidelines, figure resolution check | Camera-ready draft |
| **116** | Deploy Gradio demo on HuggingFace Spaces | Public demo URL |
| **117** | Clean and release code on GitHub: README, requirements.txt, example notebooks | GitHub repository |
| **118** | Prepare cover letter; select target journal (suggested: Computers in Biology and Medicine, Artificial Intelligence in Medicine, or IEEE JBHI) | Submission package |
| **119** | Final proofread; check all cross-references, figure numbering, table formatting | Final manuscript |
| **120** | Submit manuscript; archive all code, data splits, model weights, experiment logs | Submission confirmation |

---

## Part D: Target Journal Recommendations

| Journal | Impact Factor | Typical Review Time | Fit |
|---------|--------------|-------------------|-----|
| **Computers in Biology and Medicine** (Elsevier) | ~7.0 | 8–12 weeks | Excellent — accepts feature engineering + DL + clinical validation papers for KOA |
| **Artificial Intelligence in Medicine** (Elsevier) | ~7.5 | 10–14 weeks | Excellent — XAI focus aligns well; values clinical decision support |
| **IEEE J. Biomedical and Health Informatics** | ~7.7 | 12–16 weeks | Strong — multi-module systems; values engineering contribution |
| **Medical Image Analysis** (Elsevier) | ~10.7 | 12–20 weeks | High bar — would need exceptional results; segmentation focus |
| **Scientific Reports** (Nature) | ~3.8 | 6–10 weeks | Good fallback — broad scope, fast review; several KOA DL papers published here |
