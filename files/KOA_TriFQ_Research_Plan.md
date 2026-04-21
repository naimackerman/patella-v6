# KOA-TriFQ: Tri-Feature Quantification Framework for Explainable Knee Osteoarthritis Diagnosis

## A Step-by-Step Research Plan for Implementation

---

## 1. Framework Overview and Research Contribution

### 1.1 Problem Statement

Current deep learning approaches for KL grade classification treat the problem as end-to-end image classification, producing a single grade prediction without explaining *which* radiographic features drove the decision (Pi et al., 2023, Scientific Reports, DOI: 10.1038/s41598-023-50210-4). Clinicians assess KOA severity by evaluating three cardinal radiographic features — Joint Space Narrowing (JSN), Osteophytes, and Subchondral Sclerosis — yet existing AI systems either ignore this decomposition entirely or address only a subset. MediAI-OA (Yoon et al., 2023, BMC Musculoskeletal Disorders, DOI: 10.1186/s12891-023-06951-4) quantifies JSN and detects osteophytes but **cannot assess sclerosis**. No published system integrates all three quantified features with multi-layer XAI visualization on a single radiograph.

### 1.2 Proposed Solution: KOA-TriFQ

KOA-TriFQ is a **modular, feature-decomposed pipeline** that:

1. **Quantifies** JSN (via segmentation → contour → distance measurement), Osteophytes (via per-ROI OARSI grading), and Sclerosis (via hybrid texture analysis + CNN classification)
2. **Classifies** KL grade (0–4) using the extracted feature vector through both a transparent feature-based path and a hybrid deep learning path
3. **Visualizes** all three features as color-coded overlays on the original X-ray with a template-based clinical narrative report

### 1.3 Key Novelty Claims

1. **First system integrating all three quantified radiographic features** (JSN + Osteophytes + Sclerosis) for KL grading — advancing beyond MediAI-OA's two-feature approach
2. **Sclerosis quantification module** using hybrid texture analysis (LBP, GLCM, fractal dimension) validated against bone microarchitecture literature (Hirvasniemi et al., 2016, r = 0.86 radiograph-to-microCT correlation)
3. **Multi-layer XAI visualization** overlaying JSN distance annotations, osteophyte severity contours, and sclerosis density heatmaps on a single image
4. **Dual classification path** — transparent feature-based (XGBoost) alongside hybrid deep learning (ConvNeXt + features) — enabling clinician-interpretable and high-accuracy modes
5. **Clinical decision support report** with template-based natural language descriptions of each feature, suitable for non-specialist physicians in primary care settings

---

## 2. Critical Design Decision: Why Hybrid (Not Pure Segmentation, Detection, or Regression)

### 2.1 The Decision Matrix

Based on the research evidence, I recommend a **task-specific hybrid approach** where each feature uses the method best suited to its clinical assessment pattern:

| Feature | Recommended Approach | Justification |
|---------|---------------------|---------------|
| **JSN** | Segmentation → Measurement | JSN is inherently a continuous spatial measurement (mm). Segmenting the joint space boundary and computing distances is the clinically faithful approach. TransUNet achieves Dice = 0.889 (Guo et al., 2025). |
| **Osteophytes** | Per-ROI Classification (OARSI 0–3) | Radiologists grade osteophytes per anatomical site on a discrete ordinal scale. Per-ROI classification with SE-ResNet matches this workflow and achieves κ = 0.84 (Tiulpin & Saarakkala, 2020). |
| **Sclerosis** | Texture Analysis + CNN Classification | Sclerosis manifests as increased bone density, which correlates with pixel intensity and texture patterns. No segmentation ground truth exists, but texture features (LBP AUC = 0.840, Bayramoglu et al., 2020) and CNN classification (accuracy = 84.7%, Kim et al., 2025) are validated. |

### 2.2 Why NOT Pure Segmentation for Everything

Pixel-level segmentation for osteophytes achieves only Dice = 0.64 (Pan et al., 2024), far below clinical utility thresholds. Osteophyte boundaries are ambiguous in 2D projections; marginal bone outgrowths blend with normal cortex. Per-ROI classification avoids this problem entirely by matching the OARSI grading paradigm that radiologists already use. Sclerosis has no pixel-level ground truth available in any public dataset, making segmentation infeasible without extensive manual annotation.

### 2.3 Why NOT Pure Object Detection (YOLO/Faster R-CNN) for Everything

While YOLOv8 excels at joint ROI localization (used in Stage 1), applying bounding-box detection to osteophytes and sclerosis is clinically misaligned. Osteophyte severity depends on size and morphology, not merely location. Sclerosis is a diffuse regional phenomenon, not a discrete object. YOLO-based osteophyte detection has **no peer-reviewed validation** on knee radiographs (Park et al., 2025 applied YOLO to cervical spine only). Furthermore, bounding boxes provide poor XAI value — a colored contour or heatmap is far more informative.

### 2.4 Why NOT Pure End-to-End Deep Learning

End-to-end models (ConvNeXt at κ = 0.880, Chavoshi et al., 2025) achieve the highest raw accuracy but provide no decomposed explanation. The "Beyond XAI" paper in your collection (Weber et al., 2024) documents that post-hoc methods like Grad-CAM face an "Interpretation Gap" — heatmaps may not faithfully represent the model's actual decision process. By extracting quantified features first and then classifying, KOA-TriFQ provides **mechanistic transparency**: each feature maps directly to a clinical finding, and the KL grade is a deterministic function of these findings.

---

## 3. Dataset Strategy

### 3.1 Primary Dataset: Shashwatwork Kaggle KOA Dataset

**Source**: Derivative of OAI data, curated by Pingjun Chen (Mendeley DOI: 10.17632/56rmx5bjcr.1)
**Contents**: ~9,786 PNG images, 224×224 pixels, grayscale, pre-cropped knee ROIs
**Labels**: KL grades 0–4 via folder structure (train/val/test split provided)
**Distribution**: Grade 0: 3,218 (39.4%) | Grade 1: 1,476 (18.1%) | Grade 2: 2,142 (26.2%) | Grade 3: 1,079 (13.2%) | Grade 4: 249 (3.1%)

### 3.2 Supplementary Data Requirements

The Kaggle dataset has **no feature-level annotations**. The implementation requires:

1. **For JSN quantification**: Manual annotation of ~300–500 images with joint space boundary contours (medial and lateral femoral/tibial surfaces). Alternatively, use KNEEL (Tiulpin et al., 2019) pre-trained landmark models to bootstrap boundary detection, then fine-tune.

2. **For osteophyte grading**: Access the **OAI OARSI grading data** (publicly available via NDA application, free for researchers), which provides per-compartment osteophyte grades (0–3) for medial femur, lateral femur, medial tibia, and lateral tibia. Transfer these labels to the Kaggle images via OAI participant IDs.

3. **For sclerosis analysis**: Use a **semi-supervised approach** — define subchondral ROIs automatically from the segmented joint space boundary, extract texture features, and validate against KL grade correlation (sclerosis is definitionally present at KL ≥ 3). For graded classification, manually annotate ~500 images with sclerosis severity (none/mild/significant) following Kim et al.'s (2025) protocol.

### 3.3 Preprocessing Pipeline

```
Step 1: Load PNG images (224×224, grayscale)
Step 2: CLAHE enhancement (clip_limit=3.0, tile_grid_size=8×8)
Step 3: Histogram clipping at 5th/99th percentiles
Step 4: Normalize to [0,1] then apply ImageNet mean/std for transfer learning
Step 5: Augmentation (training only):
        - Random rotation: ±10°
        - Horizontal flip: p=0.5
        - Brightness/contrast jitter: ±15%
        - Random affine: scale=[0.9, 1.1], translate=5%
Step 6: Class imbalance handling:
        - Weighted sampler (inverse-frequency weights)
        - Ordinal soft-labelling for KL classification (Gaussian smoothing σ=0.5)
```

---

## 4. Stage-by-Stage Implementation Plan

### Stage 1: Preprocessing and ROI Detection

**Objective**: Detect the knee joint region and extract 5 ROIs (1 joint space region + 4 osteophyte sites).

**Architecture**: YOLOv8-m (medium) from Ultralytics

**Implementation Steps**:

1. **Prepare ROI annotations**: Annotate ~500 images with 5 bounding boxes each using CVAT:
   - Joint space ROI: region encompassing the tibiofemoral articulation
   - Medial Femur (MF), Lateral Femur (LF), Medial Tibia (MT), Lateral Tibia (LT) osteophyte ROIs
   
2. **Train YOLOv8-m**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8m.pt')  # pretrained on COCO
   model.train(data='koa_roi.yaml', epochs=100, imgsz=224, batch=32)
   ```

3. **Post-processing**: Apply confidence threshold ≥ 0.75 (following Chen et al., JHE 2021, who found 95% preservation at this threshold). Normalize ROI positions relative to image dimensions for consistency.

**Evaluation metrics**: mAP@0.5, mAP@0.5:0.95, IoU ≥ 0.75 detection rate

**Alternative approach if annotation is limited**: Use KNEEL landmark detection (pre-trained PyTorch model available) to detect femoral condyle corners and tibial plateau edges, then compute ROIs geometrically from landmarks. This avoids bounding box annotation entirely.

---

### Stage 2A: Joint Space Narrowing Quantification

**Objective**: Segment the tibiofemoral joint space and compute mJSW, JSN rate, and multi-point JSW profile.

**Architecture**: U-Net++ with EfficientNet-B4 encoder (via segmentation_models_pytorch)

**Sub-step 2A.1: Joint Space Segmentation**

```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",
    encoder_weights="imagenet",
    in_channels=1,        # grayscale input
    classes=3,            # background, medial joint space, lateral joint space
    activation=None       # raw logits for DiceCE loss
)
```

- **Loss function**: DiceCELoss (MONAI implementation) — combines Dice loss for shape accuracy with cross-entropy for pixel classification stability
- **Training**: 200 epochs, Adam optimizer, lr=1e-4, cosine annealing scheduler, batch_size=16
- **Ground truth**: Manually annotated joint space masks (~300–500 images), supplemented with MedSAM-assisted pre-labeling

**Sub-step 2A.2: Contour Extraction and mJSW Computation**

```python
import numpy as np
from scipy.spatial.distance import cdist

def compute_jsw_profile(mask, n_points=16):
    """
    Extract femoral and tibial contours from the segmentation mask,
    compute a 16-point JSW profile, and derive mJSW from a denser
    internal profile.
    """
    femoral_contour, tibial_contour = extract_boundaries(mask)

    # Resample along the compartment mediolateral axis
    dense_x = np.linspace(x_min, x_max, 64)
    femoral_dense = resample_contour(femoral_contour, dense_x)
    tibial_dense = resample_contour(tibial_contour, dense_x)

    # Restrict mJSW measurement to the interior weight-bearing region so
    # intercondylar-notch and compartment-edge artifacts do not dominate
    interior_x = trim_compartment_edges(dense_x, trim_fraction=0.10)

    dense_profile = []
    for x in interior_x:
        femoral_point = point_at_x(femoral_dense, x)
        tibial_candidates = local_window(
            tibial_dense,
            center_x=x,
            window_fraction=0.10,
        )
        distances = cdist([femoral_point], tibial_candidates, metric="euclidean")
        dense_profile.append(distances.min())

    # Export a 16-point profile for the feature vector, but compute mJSW
    # from the denser internal profile for better measurement fidelity.
    jsw_profile = sample_profile(dense_profile, n_points=16)
    mJSW_lateral = min(dense_profile[: len(dense_profile) // 2])
    mJSW_medial = min(dense_profile[len(dense_profile) // 2 :])
    return mJSW_medial, mJSW_lateral, jsw_profile
```

Implementation note: the intercondylar notch and compartment endpoints are
still annotated as part of the femoral/tibial surface contours, but they are
not allowed to define `mJSW`. This follows the same rationale as fixed-location
or central weight-bearing JSW methods in the radiographic OA literature: the
reported width should come from the compartment interior rather than from a
spurious notch- or edge-driven minimum.

**Sub-step 2A.3: JSN Rate Computation**

Following Yoon et al. (2023, MediAI-OA):

```
JSN_rate(x) = 100 × (1 - mJSW_x / median(mJSW_KL0))
```

Where `median(mJSW_KL0)` is computed from the KL grade 0 subset as the population reference.

**Output features**:
- `mJSW_medial`: minimum joint space width, medial compartment (pixels)
- `mJSW_lateral`: minimum joint space width, lateral compartment (pixels)
- `jsw_profile`: 16-point JSW measurement vector
- `jsn_rate_medial`: JSN rate as percentage (0% = normal, 100% = bone-on-bone)
- `jsn_rate_lateral`: JSN rate as percentage
- `jsw_ratio`: medial-to-lateral JSW ratio (asymmetry indicator)

**Evaluation metrics**: Dice coefficient (target ≥ 0.85), Hausdorff distance (95th percentile), MAE of mJSW vs manual measurement, ICC

---

### Stage 2B: Osteophyte Grading

**Objective**: Classify osteophyte severity (OARSI grade 0–3) at 4 anatomical sites with Grad-CAM localization.

**Architecture**: SE-ResNet-50 with multi-task heads (following Tiulpin & Saarakkala, 2020, Diagnostics, DOI: 10.3390/diagnostics10110932)

**Implementation Steps**:

1. **Crop 4 ROI patches** from each image (MF, LF, MT, LT) using Stage 1 detections
2. **Resize** each patch to 140×140 pixels with bilinear interpolation
3. **Apply preprocessing**: histogram clipping (5th/99th percentile), horizontal alignment

```python
import timm

class OsteophyteGrader(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone: SE-ResNet-50
        self.backbone = timm.create_model('seresnet50', pretrained=True, 
                                           num_classes=0, in_chans=1)
        feature_dim = self.backbone.num_features  # 2048
        
        # 4 separate classification heads (one per ROI site)
        self.head_mf = nn.Linear(feature_dim, 4)  # OARSI 0-3
        self.head_lf = nn.Linear(feature_dim, 4)
        self.head_mt = nn.Linear(feature_dim, 4)
        self.head_lt = nn.Linear(feature_dim, 4)
    
    def forward(self, x_mf, x_lf, x_mt, x_lt):
        f_mf = self.backbone(x_mf)
        f_lf = self.backbone(x_lf)
        f_mt = self.backbone(x_mt)
        f_lt = self.backbone(x_lt)
        
        return (self.head_mf(f_mf), self.head_lf(f_lf),
                self.head_mt(f_mt), self.head_lt(f_lt))
```

4. **Loss function**: Ordinal cross-entropy (penalize distant misclassifications more than adjacent ones)
5. **Grad-CAM extraction**: From the last convolutional layer of SE-ResNet-50, generate heatmaps for each ROI to localize osteophyte evidence

**Output features**:
- `osp_grade_mf`: OARSI grade for medial femur (0–3)
- `osp_grade_lf`: OARSI grade for lateral femur (0–3)
- `osp_grade_mt`: OARSI grade for medial tibia (0–3)
- `osp_grade_lt`: OARSI grade for lateral tibia (0–3)
- `osp_sum`: Sum of all osteophyte grades (0–12)
- `osp_max`: Maximum grade across sites
- `osp_gradcam_maps`: 4 heatmap arrays for visualization

**Evaluation metrics**: Cohen's κ per site (target ≥ 0.80), multi-class AUC, balanced accuracy

---

### Stage 2C: Subchondral Sclerosis Quantification

**Objective**: Quantify sclerosis severity in the subchondral bone region using hybrid texture analysis and CNN classification.

This is the **most novel module** — sclerosis is the least studied of the three features, and no integrated system has previously included it.

**Architecture**: Hybrid pipeline combining handcrafted texture features + EfficientNet-B0 classification

**Sub-step 2C.1: Automatic Subchondral ROI Definition**

Using the joint space segmentation from Stage 2A, define the subchondral bone region:

```python
def extract_subchondral_roi(mask, image, depth_mm=10, offset_pct=0.10):
    """
    Extract subchondral bone ROI below the tibial plateau surface.
    depth_mm: depth into bone (in pixels, approximately 10mm)
    offset_pct: lateral offset to avoid osteophyte contamination
    """
    # Get tibial surface contour from segmentation mask
    tibial_surface = get_tibial_surface(mask)
    
    # Define ROI: region immediately below tibial surface
    # Use a contour-following band below the tibial plateau rather than
    # a single flat strip, then resize the band for CNN input.
    # Medial subchondral ROI
    medial_roi = crop_below_surface(image, tibial_surface, 
                                      side='medial', depth=depth_mm,
                                      lateral_offset=offset_pct)
    # Lateral subchondral ROI  
    lateral_roi = crop_below_surface(image, tibial_surface,
                                       side='lateral', depth=depth_mm,
                                       lateral_offset=offset_pct)
    
    return medial_roi, lateral_roi
```

Implementation note: on the normalized `224x224` images, the practical extractor
should preserve a deeper subchondral band (roughly 5–10 mm / about 10% of image
height) and avoid letting the notch/edge dominate the ROI geometry. This keeps
the extracted patch closer to the weight-bearing subchondral bone texture that
the sclerosis stage is meant to quantify.

**Sub-step 2C.2: Texture Feature Extraction**

Following Bayramoglu et al. (2020, Osteoarthritis and Cartilage):

```python
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from radiomics import featureextractor

def extract_sclerosis_features(roi_patch):
    """Extract multi-scale texture features from subchondral ROI."""
    features = {}
    
    # 1. Local Binary Patterns (LBP) — best single predictor (AUC=0.840)
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(roi_patch, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points+2, density=True)
        features[f'lbp_r{radius}'] = hist
    
    # 2. GLCM / Haralick features
    glcm = graycomatrix(roi_patch, distances=[1,2,3], 
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features[f'glcm_{prop}'] = graycoprops(glcm, prop).mean()
    
    # 3. Fractal dimension (box-counting method)
    features['fractal_dim'] = compute_fractal_dimension(roi_patch)
    
    # 4. Intensity statistics (proxy for bone density)
    features['intensity_mean'] = roi_patch.mean()
    features['intensity_std'] = roi_patch.std()
    features['intensity_skewness'] = scipy.stats.skew(roi_patch.flatten())
    features['intensity_kurtosis'] = scipy.stats.kurtosis(roi_patch.flatten())
    
    # 5. Shannon entropy
    hist = np.histogram(roi_patch, bins=64, density=True)[0]
    features['entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
    
    return features
```

**Sub-step 2C.3: CNN-based Sclerosis Classification**

```python
import timm

class SclerosisClassifier(nn.Module):
    def __init__(self, texture_feat_dim=64):
        super().__init__()
        # CNN branch: EfficientNet-B0 on subchondral ROI patches
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True,
                                      num_classes=0, in_chans=1)
        cnn_dim = self.cnn.num_features  # 1280
        
        # Texture feature branch
        self.texture_mlp = nn.Sequential(
            nn.Linear(texture_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)  # none/mild/significant
        )
    
    def forward(self, roi_image, texture_features):
        cnn_feat = self.cnn(roi_image)
        tex_feat = self.texture_mlp(texture_features)
        fused = torch.cat([cnn_feat, tex_feat], dim=1)
        return self.classifier(fused)
```

**Output features**:
- `scl_grade_medial`: Sclerosis grade for medial subchondral region (0=none, 1=mild, 2=significant)
- `scl_grade_lateral`: Sclerosis grade for lateral subchondral region
- `scl_intensity_medial`: Mean pixel intensity of medial subchondral ROI
- `scl_intensity_lateral`: Mean pixel intensity of lateral subchondral ROI
- `scl_lbp_vector`: LBP histogram features (for density heatmap generation)
- `scl_fractal_dim`: Fractal dimension (trabecular complexity metric)

**Evaluation metrics**: Multi-class accuracy (target ≥ 82%), AUC per class, confusion matrix, correlation with KL grade (expected r ≥ 0.65)

---

### Stage 3: Feature Aggregation

**Objective**: Construct a structured feature vector from all three quantification modules.

**Feature Vector Composition** (total: ~50 dimensions):

```python
feature_vector = {
    # JSN features (22 dims)
    'mJSW_medial': float,          # 1
    'mJSW_lateral': float,         # 1
    'jsw_profile': float[16],      # 16
    'jsn_rate_medial': float,      # 1
    'jsn_rate_lateral': float,     # 1
    'jsw_ratio': float,            # 1
    'jsw_asymmetry': float,        # 1
    
    # Osteophyte features (10 dims)
    'osp_grade_mf': int,           # 1
    'osp_grade_lf': int,           # 1
    'osp_grade_mt': int,           # 1
    'osp_grade_lt': int,           # 1
    'osp_sum': int,                # 1
    'osp_max': int,                # 1
    'osp_medial_sum': int,         # 1
    'osp_lateral_sum': int,        # 1
    'osp_femoral_sum': int,        # 1
    'osp_tibial_sum': int,         # 1
    
    # Sclerosis features (18 dims)
    'scl_grade_medial': int,       # 1
    'scl_grade_lateral': int,      # 1
    'scl_intensity_medial': float, # 1
    'scl_intensity_lateral': float,# 1
    'scl_fractal_dim_med': float,  # 1
    'scl_fractal_dim_lat': float,  # 1
    'scl_glcm_contrast_med': float,# 1
    'scl_glcm_homog_med': float,   # 1
    'scl_glcm_energy_med': float,  # 1
    'scl_lbp_entropy_med': float,  # 1
    # ... (mirrored for lateral)
}
```

---

### Stage 4: KL Grade Classification (Dual Path)

**Objective**: Classify KL grade 0–4 using the aggregated feature vector, with two complementary paths.

#### Path A: Feature-Based (Transparent) — XGBoost

```python
import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    max_depth=6,
    n_estimators=300,
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight='balanced',
    eval_metric='mlogloss'
)
```

This path is **fully interpretable** — SHAP values show exactly which features (e.g., "medial JSN rate = 72%", "osteophyte sum = 7") drove the KL grade prediction. This follows the KOALA system precedent (Nehrer et al., 2021, FDA-cleared 510(k) K192109).

#### Path B: Hybrid Deep (High Accuracy) — ConvNeXt + Feature Concatenation

```python
class HybridKLClassifier(nn.Module):
    def __init__(self, feature_dim=50):
        super().__init__()
        # Image branch: ConvNeXt-Small on full knee image
        self.image_encoder = timm.create_model('convnext_small', pretrained=True,
                                                 num_classes=0, in_chans=1)
        img_dim = self.image_encoder.num_features  # 768
        
        # Feature branch: MLP on extracted features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        # Fusion + ordinal classification
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5)  # KL grades 0-4
        )
    
    def forward(self, image, features):
        img_feat = self.image_encoder(image)
        ext_feat = self.feature_encoder(features)
        fused = torch.cat([img_feat, ext_feat], dim=1)
        return self.classifier(fused)
```

**Loss function**: CORN (Conditional Ordinal Regression Network) loss or soft-label cross-entropy with Gaussian smoothing — both respect the ordinal nature of KL grades.

**Evaluation metrics**:
- **Primary**: Quadratic Weighted Kappa (QWK) — target ≥ 0.82
- **Secondary**: Multi-class accuracy, per-class F1, AUC (one-vs-rest), confusion matrix
- **Comparison**: Both paths evaluated on identical test set; report which performs better and where they disagree

---

### Stage 5: Explainable Visualization and Clinical Report

**Objective**: Generate a multi-layer annotated X-ray and structured clinical narrative.

#### 5.1 Multi-Layer Overlay Design

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_xai_overlay(image, jsn_results, osp_results, scl_results, kl_pred):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    
    # Layer 1: JSN — Blue distance lines
    for i, (p1, p2, dist) in enumerate(jsn_results['measurement_pairs']):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5, alpha=0.8)
        midx, midy = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        ax.annotate(f'{dist:.1f}px', (midx, midy), color='cyan', fontsize=8)
    
    # Layer 2: Osteophytes — Severity-colored boxes at 4 ROIs
    severity_colors = {0: 'none', 1: 'green', 2: 'yellow', 3: 'red'}
    for site, grade, bbox in osp_results['detections']:
        if grade > 0:
            color = severity_colors[grade]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                      linewidth=2, edgecolor=color, 
                                      facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1]-5, f'{site}: Grade {grade}', 
                    color=color, fontsize=9, weight='bold')
    
    # Layer 3: Sclerosis — Semi-transparent density heatmap
    sclerosis_heatmap = scl_results['heatmap']
    ax.imshow(sclerosis_heatmap, cmap='YlOrRd', alpha=0.3, 
              extent=scl_results['roi_extent'])
    
    # KL Grade badge
    ax.text(10, 20, f'KL Grade: {kl_pred["grade"]}', 
            fontsize=14, weight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    return fig
```

#### 5.2 Template-Based Clinical Narrative

```python
def generate_clinical_report(jsn, osp, scl, kl_pred):
    """Generate structured clinical narrative from quantified features."""
    
    report = f"""
AUTOMATED KOA ASSESSMENT REPORT
================================
KL Grade: {kl_pred['grade']} (Confidence: {kl_pred['confidence']:.1%})
Grade distribution: {format_distribution(kl_pred['probabilities'])}

JOINT SPACE NARROWING:
- Medial compartment: mJSW = {jsn['mJSW_medial']:.1f} px 
  (JSN rate: {jsn['jsn_rate_medial']:.0f}%)
  → {'Normal' if jsn['jsn_rate_medial'] < 20 else 'Narrowed' if jsn['jsn_rate_medial'] < 50 else 'Severely narrowed'}
- Lateral compartment: mJSW = {jsn['mJSW_lateral']:.1f} px 
  (JSN rate: {jsn['jsn_rate_lateral']:.0f}%)
  → {'Normal' if jsn['jsn_rate_lateral'] < 20 else 'Narrowed' if jsn['jsn_rate_lateral'] < 50 else 'Severely narrowed'}
- Medial-lateral ratio: {jsn['jsw_ratio']:.2f} 
  ({'Symmetric' if 0.8 < jsn['jsw_ratio'] < 1.2 else 'Asymmetric — predominantly medial' if jsn['jsw_ratio'] < 0.8 else 'Asymmetric — predominantly lateral'})

OSTEOPHYTES:
- Medial femur: OARSI Grade {osp['mf']} ({grade_descriptor(osp['mf'])})
- Lateral femur: OARSI Grade {osp['lf']} ({grade_descriptor(osp['lf'])})
- Medial tibia: OARSI Grade {osp['mt']} ({grade_descriptor(osp['mt'])})
- Lateral tibia: OARSI Grade {osp['lt']} ({grade_descriptor(osp['lt'])})
- Total osteophyte burden: {osp['sum']}/12

SUBCHONDRAL SCLEROSIS:
- Medial tibial plateau: {scl_descriptor(scl['grade_medial'])}
  (Bone density index: {scl['intensity_medial']:.0f}, 
   Trabecular complexity: FD={scl['fractal_dim_med']:.3f})
- Lateral tibial plateau: {scl_descriptor(scl['grade_lateral'])}

CLINICAL IMPRESSION:
{generate_impression(jsn, osp, scl, kl_pred)}

NOTE: This is an AI-assisted assessment. All findings should be 
confirmed by a qualified radiologist before clinical decision-making.
"""
    return report
```

---

## 5. Evaluation Plan

### 5.1 Component-Level Evaluation

| Module | Primary Metric | Target | Benchmark |
|--------|---------------|--------|-----------|
| ROI Detection (YOLOv8) | mAP@0.5 | ≥ 0.90 | Chen et al., 2021: 0.85 |
| JSN Segmentation (U-Net++) | Dice coefficient | ≥ 0.85 | TransUNet: 0.889 |
| mJSW Measurement | MAE (pixels) | ≤ 3.0 px | Mulford et al., 2025: 0.85mm |
| Osteophyte Grading (SE-ResNet) | Cohen's κ per site | ≥ 0.80 | Tiulpin, 2020: 0.79–0.84 |
| Sclerosis Classification | Accuracy / AUC | ≥ 82% / ≥ 0.90 | Kim et al., 2025: 84.7% |
| KL Grade (XGBoost path) | QWK | ≥ 0.78 | KOALA: κ = 0.768 |
| KL Grade (Hybrid path) | QWK | ≥ 0.84 | ConvNeXt SOTA: κ = 0.880 |

### 5.2 System-Level Evaluation

1. **End-to-end accuracy**: Compare full pipeline KL grade prediction against ground truth
2. **Ablation study**: Remove each feature module independently to measure contribution:
   - JSN only → KL grade
   - JSN + Osteophytes → KL grade
   - JSN + Osteophytes + Sclerosis → KL grade (full pipeline)
3. **Comparison with baselines**:
   - End-to-end ConvNeXt (no feature decomposition)
   - MediAI-OA reproduction (2 features only)
   - Feature-based only vs. hybrid
4. **XAI evaluation**: Qualitative assessment by 2–3 radiologists rating overlay quality and clinical report accuracy on 100 test cases

### 5.3 Cross-Validation Strategy

5-fold stratified cross-validation on the training set, with the provided test set held out for final evaluation. Report mean ± standard deviation across folds.

---

## 6. Implementation Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Data Preparation** | Weeks 1–3 | Download dataset, EDA, annotation setup (CVAT), OAI data access application |
| **Phase 2: Stage 1 — ROI Detection** | Weeks 3–5 | Annotate ROIs, train YOLOv8, validate detection |
| **Phase 3: Stage 2A — JSN Module** | Weeks 5–8 | Annotate joint space masks, train U-Net++, implement mJSW computation |
| **Phase 4: Stage 2B — Osteophyte Module** | Weeks 8–11 | Map OAI OARSI labels, train SE-ResNet, implement Grad-CAM |
| **Phase 5: Stage 2C — Sclerosis Module** | Weeks 11–15 | Annotate sclerosis grades, implement texture extraction, train hybrid classifier |
| **Phase 6: Stage 3–4 — Integration & Classification** | Weeks 15–18 | Build feature aggregation, train XGBoost + hybrid classifier, ablation studies |
| **Phase 7: Stage 5 — XAI Visualization** | Weeks 18–20 | Implement overlay renderer, template report generator, clinical evaluation |
| **Phase 8: Paper Writing & Deployment** | Weeks 20–24 | Write journal manuscript, build Gradio demo, Docker containerization |

---

## 7. Technology Stack Summary

| Component | Library | Version |
|-----------|---------|---------|
| Framework | PyTorch | ≥ 2.0 |
| Medical imaging | MONAI | ≥ 1.3 |
| Segmentation | segmentation_models_pytorch | ≥ 0.5 |
| Detection | Ultralytics YOLOv8 | ≥ 8.1 |
| Classification backbones | timm | ≥ 1.0 |
| Texture analysis | scikit-image, pyradiomics | latest |
| ML classification | XGBoost, scikit-learn | latest |
| XAI | pytorch-grad-cam | ≥ 1.5 |
| Experiment tracking | Weights & Biases | latest |
| Configuration | Hydra | ≥ 1.3 |
| Training framework | PyTorch Lightning | ≥ 2.0 |
| Annotation | CVAT + MedSAM | latest |
| Deployment | Gradio + FastAPI + Docker | latest |
| Model export | ONNX Runtime | latest |

---

## 8. References

1. Pi, S-W. et al. (2023). "Ensemble deep-learning networks for automated osteoarthritis grading in knee X-ray images." Scientific Reports, 13:22887. DOI: 10.1038/s41598-023-50210-4
2. Yoon, J-H. et al. (2023). "Assessment of a novel deep learning-based software developed for automatic feature extraction and grading of radiographic knee osteoarthritis." BMC Musculoskeletal Disorders, 24:869. DOI: 10.1186/s12891-023-06951-4
3. Tiulpin, A. & Saarakkala, S. (2020). "Automatic Grading of Individual Knee Osteoarthritis Features in Plain Radiographs Using Deep Convolutional Neural Networks." Diagnostics, 10(11):932. DOI: 10.3390/diagnostics10110932
4. Chavoshi, M. et al. (2025). "Comparative Evaluation of Deep Learning and Foundation Model Embeddings for Osteoarthritis Feature Classification in Knee Radiographs." Journal of Imaging Informatics in Medicine. DOI: 10.1007/s10278-025-01636-x
5. Guo, Z. et al. (2025). "Predicting joint space changes in knee osteoarthritis over 6 years: a combined model of TransUNet and XGBoost." Quantitative Imaging in Medicine and Surgery. DOI: 10.21037/qims-24-1397
6. Bayramoglu, N. et al. (2020). "Adaptive segmentation of knee radiographs for selecting the optimal ROI in texture analysis." Osteoarthritis and Cartilage. DOI: 10.1016/j.joca.2020.03.013
7. Kim, J. et al. (2025). "Classification of Grades of Subchondral Sclerosis from Knee Radiographic Images Using Artificial Intelligence." Sensors, 25(8):2535. PMC12031447
8. Tiulpin, A. et al. (2019). "KNEEL: Knee Anatomical Landmark Localization Using Hourglass Networks." ICCVW. DOI: 10.1109/ICCVW.2019.00046
9. Nehrer, S. et al. (2021). "Automated Knee Osteoarthritis Assessment Increases Physicians' Agreement Rate and Accuracy." Cartilage, 13(1_suppl):957s–965s. DOI: 10.1177/1947603519888793
10. Hirvasniemi, J. et al. (2016). "Correlation of Subchondral Bone Density and Structure from Plain Radiographs with Micro Computed Tomography Ex Vivo." Annals of Biomedical Engineering. DOI: 10.1007/s10439-015-1452-y
11. Pan, Y. et al. (2024). "Automatic knee osteoarthritis severity grading based on X-ray images using a hierarchical classification method." Arthritis Research & Therapy. DOI: 10.1186/s13075-024-03416-4
12. Daneshmand, A. et al. (2024). "Deep learning based detection of osteophytes in radiographs and magnetic resonance imagings of the knee using 2D and 3D morphology." Journal of Orthopaedic Research. DOI: 10.1002/jor.25800
13. Karim, M.R. et al. (2021). "DeepKneeExplainer: Explainable Knee Osteoarthritis Diagnosis From Radiographs and Magnetic Resonance Imaging." IEEE Access. DOI: 10.1109/ACCESS.2021.3062493
14. Teoh, Y.X. et al. (2024). "Deciphering Knee Osteoarthritis Diagnostic Features With Explainable Artificial Intelligence: A Systematic Review." IEEE Access. DOI: 10.1109/ACCESS.2024.3439096
15. Ko, S. et al. (2025). "Association of radiographic structure deformity phenotypes of knee OA: Proposing a modification of Kellgren-Lawrence grade." Osteoarthritis and Cartilage Open. DOI: 10.1016/j.ocarto.2025.100566
16. Schiphof, D. et al. (2008). "Differences in descriptions of Kellgren and Lawrence grades of knee osteoarthritis." Annals of the Rheumatic Diseases, 67:1034-1036. DOI: 10.1136/ard.2007.079020
17. Weber, R.O. et al. (2024). "XAI is in trouble." AI Magazine, 45(3):300-316.
18. Mulford, K. et al. (2025). "A Deep Learning Tool for Minimum Joint Space Width Calculation on Antero-posterior Knee Radiographs." Journal of Arthroplasty. DOI: 10.1016/j.arth.2025.00065
19. Alkhatatbeh, M. et al. (2026). "Cross-Institutional Five-Class Kellgren-Lawrence Grading of Knee Osteoarthritis via Multitask Deep Learning." Annals of the NY Academy of Sciences. DOI: 10.1111/nyas.70254
20. Chen, J. et al. (2021). "Knee OA severity assessment using visual transformer." Journal of Healthcare Engineering. DOI: 10.1155/2021/5586529
21. Abdullah, S.S. & Rajasekaran, M.P. (2022). "Automatic detection and classification of knee osteoarthritis using deep learning approach." Radiologia Medica, 127:398-406.
22. Fatema, K. et al. (2023). "KOA Grade Classification Using Distance Features." Heliyon, 9:e21703.
23. Abed, I.S. et al. (2026). "Towards Reliable Osteoarthritis Classification: Fine-Tuned CNNs, Vision Transformers, and Ensemble Learning." International Journal of Technology, 17(1):301-321.
