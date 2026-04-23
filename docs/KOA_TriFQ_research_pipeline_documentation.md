# KOA-TriFQ Research Pipeline Documentation

Generated: 2026-04-23

This document records the complete implemented research pipeline for KOA-TriFQ, a tri-feature quantification framework for explainable knee osteoarthritis assessment from frontal knee radiographs. It is intended as the primary technical and scientific reference for writing the later manuscript. It documents the clinical rationale, implementation design, experimental stages, final results, limitations, and interpretation logic needed to convert the engineering work into a defensible research paper.

The implementation is in `/Users/naim/Desktop/patella-v6`. The main execution script is `scripts/run_main_study_pipeline.sh`, and the final application is in `app/gradio_app.py` and `app/inference.py`.

## 1. Research Thesis

The central research thesis is that radiographic knee osteoarthritis (KOA) severity should not be treated only as a black-box image classification problem. The Kellgren-Lawrence (KL) grade is a global ordinal label, but it is clinically derived from visible structural findings, especially joint space narrowing (JSN), osteophytes, and subchondral sclerosis. KOA-TriFQ operationalizes this observation by decomposing the radiograph into three quantified feature families and then using those features for KL prediction and explanation.

The work therefore has two linked objectives:

1. Build feature-level quantification modules for JSN, osteophytes, and sclerosis.
2. Use the quantified features to support two KL prediction paths:
   - Path A: a transparent feature-based XGBoost classifier.
   - Path B: a hybrid ConvNeXt classifier that combines the full image with the 50-dimensional feature vector.

The intended contribution is not simply higher KL accuracy. The intended contribution is a clinically structured, interpretable pipeline in which every intermediate module maps to a radiographic concept that clinicians already use.

## 2. Research Background and Design Rationale

### 2.1 Why feature decomposition is clinically meaningful

Radiographic OA assessment is feature-based. A radiograph does not show cartilage directly; cartilage loss is inferred from narrowing of the joint space between bony margins. Radiographs also show osteophytes and subchondral bone change. The literature describes radiographically visible OA changes as including joint space narrowing, osteophytes, and subchondral sclerosis, with JSW used as an indirect marker of cartilage loss and progression. See the knee JSW literature in [Location specific radiographic joint space width for osteoarthritis progression](https://pmc.ncbi.nlm.nih.gov/articles/PMC3138121/) and the dataset/background statement in [Minimum joint space width and tibial cartilage morphology in healthy knees](https://link.springer.com/article/10.1186/1471-2474-9-119).

The KL system remains widely used, but the literature also notes variability and ambiguity in its descriptions, especially around how osteophytes and JSN are weighted. The review [Classifications in Brief: Kellgren-Lawrence Classification of Osteoarthritis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/) is useful for framing why an explicit, feature-decomposed approach may be more transparent than a single end-to-end KL output.

The OARSI atlas formalizes individual radiographic feature grading, including osteophytes and JSN, across compartments and severity levels. This supports the decision to treat osteophytes as site-specific ordinal grades rather than as a single global binary label. See [Atlas of individual radiographic features in osteoarthritis, revised](https://www.sciencedirect.com/science/article/pii/S1063458406003281).

### 2.2 Why JSN is implemented as segmentation plus measurement

JSN is a spatial measurement problem. The clinically meaningful quantity is not only whether narrowing exists, but where the femoral and tibial margins are and how close they are. Therefore, KOA-TriFQ segments medial and lateral joint-space regions and derives contour-based minimum joint space width (mJSW), JSN rate, and a 16-point JSW profile.

This is aligned with radiographic JSW research, which typically delineates femoral and tibial margins and computes minimum or location-specific JSW. It is also sensitive to radiographic acquisition conditions: knee flexion, beam angle, and foot rotation can influence JSW measurement, as shown in radiographic positioning work such as [Ravaud et al., 1996](https://academic.oup.com/rheumatology/article/35/8/761/1782848). This limitation is important for discussion because this implementation uses resized PNGs rather than DICOM-calibrated radiographs.

### 2.3 Why osteophytes are implemented as site-specific ordinal grading

Osteophytes are marginal bony outgrowths. In clinical grading, they are commonly scored by site and severity rather than segmented pixel-by-pixel. The implementation therefore trains a site-aware SE-ResNet-50 osteophyte classifier for four anatomical sites:

- medial femur
- lateral femur
- medial tibia
- lateral tibia

The visualization in the application now displays anatomy-guided marginal focus boxes rather than the entire classifier ROI. This is important because the classifier is trained with site-level grades, not pixel-level spur masks. The box is therefore an explanatory site anchor, not a claim of exact osteophyte boundary segmentation.

The use of SE-ResNet is justified by the Squeeze-and-Excitation mechanism, which recalibrates channel-wise responses and improves representational power with modest overhead. See [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html).

### 2.4 Why sclerosis is implemented as binary texture/CNN classification

Subchondral sclerosis is visually diffuse. It is not naturally a bounding-box object and, in this project, there is no reliable pixel-level sclerosis segmentation ground truth. The implemented approach extracts subchondral bone ROIs guided by the JSN segmentation and combines:

- intensity statistics
- GLCM texture features
- LBP texture entropy
- fractal dimension
- a CNN branch for learned ROI representation

The final selected formulation is binary sclerosis presence (`none` versus `present`) rather than three-class severity. This was an empirically motivated decision. The three-class setting was unstable with the available manual labels, while the binary formulation produced more meaningful thresholded validation behavior and clearer use in the downstream KL feature vector.

### 2.5 Why two KL paths are preserved

The project intentionally keeps two KL paths.

Path A, XGBoost, is the transparent path. XGBoost is strong for tabular features and can be interpreted with feature importance and SHAP. See [XGBoost: A Scalable Tree Boosting System](https://www.kdd.org/kdd2016/subtopic/view/xgboost-a-scalable-tree-boosting-system/670/) and [A Unified Approach to Interpreting Model Predictions](https://huggingface.co/papers/1705.07874).

Path B, ConvNeXt hybrid, is the performance-oriented path. It combines full-image representation learning with the structured 50-dimensional feature vector. ConvNeXt is a modern convolutional backbone designed to compete with transformer-era architectures while preserving convolutional inductive biases. See [A ConvNet for the 2020s](https://collaborate.princeton.edu/en/publications/a-convnet-for-the-2020s/).

This dual-path design is central to the manuscript argument: Path A demonstrates transparent clinical feature reasoning; Path B demonstrates how structured features can be fused with image representation for stronger predictive performance.

## 3. Dataset and Label Strategy

### 3.1 Image dataset

The image dataset is the knee osteoarthritis severity grading dataset derived from OAI data and published on Mendeley: [Knee Osteoarthritis Severity Grading Dataset](https://data.mendeley.com/datasets/56rmx5bjcr/1). The local data path is:

`KneeXrayData/ClsKLData/kneeKL224`

The dataset is organized by KL grade folders and split into train, validation, and test sets. The implemented feature aggregation stage produced:

| Split | Samples | Feature shape | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 5,778 | 5,778 x 50 | 2,286 | 1,046 | 1,516 | 757 | 173 |
| Validation | 826 | 826 x 50 | 328 | 153 | 212 | 106 | 27 |
| Test | 1,656 | 1,656 x 50 | 639 | 296 | 447 | 223 | 51 |

The class distribution is ordinal and imbalanced. KL0 is common; KL4 is rare. This imbalance should be discussed because it affects both model learning and per-class F1 interpretation, especially for KL1 and KL4.

### 3.2 Manual feature labels

The feature-level annotations are stored under `annotations/`. The major reviewed label files are:

- `annotations/osteophyte_labels_reviewed.csv`
- `annotations/sclerosis_labels_reviewed.csv`
- JSN masks and contour packages under `annotations/packages/` and `annotations/jsn_masks`

The manual feature labels were used as the primary source of ground truth for JSN evaluation, osteophyte grading, and sclerosis training/evaluation. High-confidence pseudo-labels were tested for expansion, but were not accepted blindly. Where pseudo-labels failed to improve held-out manual generalization, the pipeline retained the manual-teacher result.

## 4. Pipeline Overview

The implemented pipeline contains 34 stages:

| Stage | Purpose | Main script |
|---:|---|---|
| 1 | Clean numbered duplicate files | `scripts/clean_duplicate_suffix_files.py` |
| 2 | Preflight readiness audit | `scripts/check_pipeline_readiness.py` |
| 3 | Bootstrap pseudo-label suggestions | `scripts/bootstrap_pseudo_labels.py` |
| 4 | Refresh annotation manifests | `scripts/refresh_annotation_manifests.py` |
| 5 | Prepare annotation workspaces | `scripts/prepare_annotation_workspace.py` |
| 6 | Prepare ROI detector dataset | `scripts/prepare_roi_yolo_dataset.py` |
| 7 | Train ROI detector | `scripts/train_roi_detector.py` |
| 8 | Evaluate ROI detector | `scripts/evaluate_roi_detector.py` |
| 9 | Compare ROI backends | `scripts/compare_roi_backends.py` |
| 10 | Extract ROIs | `scripts/extract_rois.py` |
| 11 | Import reviewed annotations | `scripts/import_reviewed_annotations.py` |
| 12 | Train JSN segmenter | `scripts/train_jsn_segmenter.py` |
| 13 | Extract JSN features and masks | `scripts/extract_jsn_features.py` |
| 14 | Extract CLAHE osteophyte ROI patches | `scripts/extract_roi_with_fullimage_clahe.py` |
| 15 | Train manual osteophyte grader | `scripts/train_osteophyte_grader.py` |
| 16 | Extract sclerosis manual-teacher features | `scripts/extract_sclerosis_features.py` |
| 17 | Train manual sclerosis classifier | `scripts/train_sclerosis.py` |
| 18 | Generate high-confidence pseudo-label expansions | `scripts/generate_feature_pseudolabels.py` |
| 19 | Train expanded osteophyte grader | `scripts/train_osteophyte_grader.py` |
| 20 | Extract expanded sclerosis features | `scripts/extract_sclerosis_features.py` |
| 21 | Train expanded sclerosis classifier | `scripts/train_sclerosis.py` |
| 21t | Tune binary sclerosis threshold | `scripts/evaluate_sclerosis_thresholds.py` |
| 22 | Extract osteophyte features | `scripts/extract_osteophyte_features.py` |
| 23 | Re-extract final sclerosis features | `scripts/extract_sclerosis_features.py` |
| 24 | Aggregate all features | `scripts/extract_all_features.py` |
| 25 | Train KL XGBoost classifier | `scripts/train_kl_xgboost.py` |
| 26 | Train KL hybrid classifier | `scripts/train_kl_hybrid.py` |
| 27 | Evaluate JSN segmenter | `scripts/evaluate_jsn_segmenter.py` |
| 28 | Evaluate osteophyte grader | `scripts/evaluate_osteophyte_grader.py` |
| 29 | Evaluate sclerosis classifier | `scripts/evaluate_sclerosis.py` |
| 30 | Run KL feature baselines | `scripts/run_kl_feature_baselines.py` |
| 31 | Run KL ablation studies | `scripts/run_ablation.py` |
| 32 | Evaluate end-to-end pipeline | `scripts/evaluate_pipeline.py` |
| 33 | Build reproducibility report | `scripts/build_repro_report.py` |
| 34 | Post-run readiness audit | `scripts/check_pipeline_readiness.py` |

The main executable is:

```bash
./scripts/run_main_study_pipeline.sh
```

Stages can be skipped with `--skip-stages`.

## 5. Module Methods

### 5.1 Preprocessing

The default preprocessing configuration is in `configs/preprocessing/default.yaml`.

Core preprocessing:

- CLAHE with clip limit 3.0 and tile grid 8 x 8.
- Histogram clipping at the 5th and 99th percentiles.
- Single-channel normalization with mean 0.485 and standard deviation 0.229.
- Training augmentation: rotation up to 10 degrees, brightness/contrast jitter, affine scaling 0.9 to 1.1, and translation up to 5 percent.

Important implementation note: osteophyte reproducibility ROI extraction uses full-image CLAHE before extracting 140 x 140 site patches. During osteophyte model training, ROI-level CLAHE and histogram clipping are disabled to avoid double enhancement.

### 5.2 JSN segmentation and measurement

Configuration:

- Model: U-Net++ with EfficientNet-B4 encoder.
- Config: `configs/model/unetpp.yaml`.
- Classes: background, medial joint space, lateral joint space.
- Evaluation checkpoint: `checkpoints/jsn_segmenter/jsn-epoch=086-val_dice=0.9012.ckpt`.

The model choice follows the medical segmentation rationale of UNet++, which uses nested dense skip pathways to reduce the semantic gap between encoder and decoder features. See [UNet++](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/). EfficientNet is used as the encoder because compound scaling provides strong representational efficiency. See [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html).

Measurement algorithm:

1. Segment medial and lateral joint-space regions.
2. Extract femoral and tibial boundary contours per compartment.
3. Resample contours along the mediolateral axis.
4. Compute Euclidean femur-tibia distances with a local x-window.
5. Trim compartment endpoints to reduce false minima from edge artifacts and the intercondylar notch.
6. Compute mJSW, 16-point JSW profile, JSN rate, JSW ratio, and asymmetry.

The JSN feature vector is 22-dimensional:

- 2 mJSW features: medial and lateral.
- 16 JSW profile samples.
- 2 JSN rates: medial and lateral.
- 1 JSW ratio.
- 1 JSW asymmetry.

### 5.3 Osteophyte grading

Configuration:

- Model: SE-ResNet-50 with four site heads.
- Config: `configs/model/se_resnet50.yaml`.
- ROI size: 140 x 140.
- Classes: OARSI-like ordinal grades 0 to 3.
- Sites: medial femur, lateral femur, medial tibia, lateral tibia.

The model uses a shared SE-ResNet-50 backbone and site-specific heads. The shared backbone lets all sites benefit from common radiographic morphology, while separate heads allow site-specific grading differences. The refinement step was applied selectively to lateral femur and medial tibia because validation behavior suggested these sites needed additional specialization.

The osteophyte feature vector is 10-dimensional:

- 4 site grades: MF, LF, MT, LT.
- total osteophyte burden.
- maximum osteophyte grade.
- medial sum, lateral sum.
- femoral sum, tibial sum.

Visualization:

The app displays marginal anatomy-guided focus boxes for nonzero osteophyte grades. These are not exact osteophyte segmentations. They are site-level interpretability anchors because the model was trained from site-level grades, not pixel masks.

### 5.4 Sclerosis quantification

Configuration:

- Model: hybrid CNN plus texture branch.
- Config: `configs/model/sclerosis_hybrid.yaml`.
- Final label scheme: `binary_present`.
- Selected threshold: 0.4227197766304016.
- Selected checkpoint: `checkpoints/sclerosis_binary_texture_only/sclerosis/scl-auc-epoch=044-val_auc_macro=0.6730.ckpt`.

The final sclerosis implementation is binary. The three-class severity formulation was tested but was unstable because the manual dataset was small and because sclerosis is more diffuse and less sharply defined than JSN or osteophytes. The binary formulation better matched the available evidence and produced more useful downstream features.

The sclerosis features are extracted from JSN-guided subchondral ROIs. The 18-dimensional sclerosis vector contains:

- 2 predicted side grades: medial and lateral.
- 2 mean intensity features.
- 2 fractal dimension features.
- 10 GLCM features, five per side.
- 2 LBP entropy features.

Pseudo-label fine-tuning was tested. The strict pseudo-label setting did not improve held-out manual test generalization over the manual/binary texture teacher. Therefore, the final pipeline uses the manual teacher with tuned threshold.

### 5.5 Feature aggregation

The feature aggregation implementation is in `src/features/feature_aggregator.py`.

The final KL feature vector has 50 dimensions:

| Feature family | Dimensions | Summary |
|---|---:|---|
| JSN | 22 | mJSW, 16-point JSW profile, JSN rates, ratio, asymmetry |
| Osteophyte | 10 | four site grades and burden summaries |
| Sclerosis | 18 | side grades, intensity, fractal, GLCM, LBP entropy |
| Total | 50 | full tri-feature vector |

Training statistics are fit on the train split and saved as normalizer artifacts. KL classifiers consume the normalized feature matrix.

### 5.6 KL Path A: XGBoost

Configuration:

- Config: `configs/model/xgboost.yaml`.
- Objective: `multi:softprob`.
- Classes: KL0 to KL4.
- Trees: 300 estimators.
- Depth: 6.
- Learning rate: 0.05.
- Cross-validation: 5 folds.

This path provides the main transparent tabular model. Its outputs are compatible with SHAP-based feature attribution, and its ablation behavior directly tests the contribution of JSN, osteophyte, and sclerosis feature families.

### 5.7 KL Path B: ConvNeXt hybrid

Configuration:

- Config: `configs/model/convnext_hybrid.yaml`.
- Backbone: ConvNeXt-Small.
- Input: full grayscale knee radiograph plus normalized 50-dimensional feature vector.
- Fusion hidden dimension: 256.
- Loss: soft-label cross entropy.
- Final evaluated checkpoint: `checkpoints/kl_hybrid/hybrid-epoch=072-val_qwk=0.8053.ckpt`.

The hybrid path is the performance path. It preserves the structured clinical feature vector while allowing the network to use radiographic context not fully captured by handcrafted feature extraction.

### 5.8 Explainability and application layer

The final app is implemented in `app/gradio_app.py` and `app/inference.py`.

The application displays:

- input radiograph
- annotated overlay
- KL probability distribution
- structured clinical report
- layer toggles for JSN, osteophytes, sclerosis, and KL badge
- display-only preprocessing choices: raw, CLAHE, histogram clipping, clip plus CLAHE
- JSN label units in px or optionally converted display mm/px

Important distinction: display preprocessing is for visualization only. After the latest UI patch, changing CLAHE or display clipping redraws the cached overlay but does not rerun the assessment. This is necessary because display enhancement should not change the prediction.

## 6. Results

### 6.1 JSN module

| Model setting | Dice mean | Hausdorff95 mean | mJSW MAE | mJSW ICC mean |
|---|---:|---:|---:|---:|
| Manual-only JSN checkpoint | 0.8904 | 1.4734 | 1.7020 | 0.8699 |
| Self-trained JSN checkpoint | 0.8894 | 1.6060 | 1.7217 | 0.8645 |
| Selected downstream result | 0.8909 | 1.4964 | 1.7041 | 0.8674 |

Interpretation:

The manual-only JSN checkpoint was selected for downstream use because it had the lower mJSW MAE compared with the self-training branch. Self-training was useful to test but did not improve the measurement criterion. This is scientifically important because the downstream KL features depend not only on segmentation overlap but on measurement fidelity.

### 6.2 Osteophyte module

Manual-label main framework:

| Site | Checkpoint mode | Validation kappa | Test kappa | Test balanced accuracy | Test AUC macro |
|---|---|---:|---:|---:|---:|
| Medial femur | multitask | 0.5707 | 0.5828 | 0.4427 | 0.6872 |
| Lateral femur | refined site | 0.6660 | 0.1048 | 0.3228 | 0.5623 |
| Medial tibia | refined site | 0.5200 | 0.4706 | 0.4539 | 0.7105 |
| Lateral tibia | multitask | 0.6652 | 0.3665 | 0.4551 | 0.6773 |
| Mean | mixed | - | 0.3812 | - | - |

Expanded pseudo-label framework:

| Site | Checkpoint mode | Validation kappa | Test kappa | Test balanced accuracy | Test AUC macro |
|---|---|---:|---:|---:|---:|
| Medial femur | multitask | 0.6418 | 0.6354 | 0.4406 | 0.6821 |
| Lateral femur | refined site | 0.6597 | 0.1186 | 0.3326 | 0.5649 |
| Medial tibia | refined site | 0.5330 | 0.4869 | 0.4420 | 0.7049 |
| Lateral tibia | multitask | 0.5839 | 0.2756 | 0.4178 | 0.6716 |
| Mean | mixed | - | 0.3791 | - | - |

Interpretation:

Osteophyte grading is partially successful but heterogeneous by site. Medial femur and medial tibia generalize better than lateral femur. The lateral femur result should be discussed transparently as a weakness, likely reflecting limited manual labels, class imbalance, site ambiguity, and the difficulty of mapping OARSI-like labels to small 224 x 224 crops. Pseudo-label expansion improved some site validation metrics but did not substantially improve mean held-out test kappa. For the paper, this should be framed as an important empirical finding rather than hidden.

### 6.3 Sclerosis module

Final tuned binary threshold:

| Split | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Validation | 0.4227 | 0.6267 | 0.6234 | 0.6212 | 0.6709 | [[38, 25], [31, 56]] |
| Test | 0.4227 | 0.5800 | 0.5786 | 0.5785 | 0.6114 | [[39, 31], [32, 48]] |

Texture benchmark summary:

| Model | Validation F1 macro | Test F1 macro | Test AUC |
|---|---:|---:|---:|
| Random forest | 0.6052 | 0.5852 | 0.5941 |
| Extra trees | 0.6288 | 0.5696 | 0.6204 |
| Logistic regression L2 | 0.6027 | 0.5400 | 0.5532 |
| XGBoost | 0.5867 | 0.5354 | 0.5696 |
| SVM RBF | 0.5768 | 0.5348 | 0.5996 |
| Logistic regression L1 | 0.6288 | 0.5023 | 0.5598 |

Interpretation:

Sclerosis is the weakest and most exploratory module. This is expected because it is diffuse, weakly localized, and manually labeled on a smaller sample. The final thresholded binary classifier is usable as a structured feature but should not be overclaimed as a mature standalone diagnostic model. The strongest paper framing is that sclerosis was integrated as a quantifiable third radiographic feature, but larger manual labels and DICOM-calibrated radiographs are needed for stronger independent validation.

### 6.4 KL Path A: XGBoost feature classifier

| Metric | Value |
|---|---:|
| Cross-validation QWK | 0.6467 +/- 0.0109 |
| Test QWK | 0.6294 |
| Test accuracy | 0.5399 |
| Test F1 macro | 0.5238 |
| Test F1 weighted | 0.4986 |

Per-class test F1:

| Class | F1 |
|---|---:|
| KL0 | 0.6407 |
| KL1 | 0.0817 |
| KL2 | 0.4291 |
| KL3 | 0.7294 |
| KL4 | 0.7379 |

Interpretation:

The transparent feature-based path performs reasonably for severe disease and non-OA, but KL1 remains poor. This is clinically plausible because KL1 is a borderline/doubtful category and often has weak radiographic signal. The XGBoost path is still valuable because it enables feature-family ablation and SHAP interpretation.

### 6.5 KL Path B: hybrid ConvNeXt plus features

| Split | QWK | Accuracy | F1 macro | AUC macro |
|---|---:|---:|---:|---:|
| Validation | 0.7966 | 0.5993 | 0.6210 | 0.8879 |
| Test | 0.8436 | 0.6636 | 0.6693 | 0.9017 |

Interpretation:

The hybrid model substantially outperforms Path A on test QWK and F1 macro. This supports the hypothesis that structured features are valuable, but full-image representation still carries additional information not captured by the feature vector. For the paper, the correct claim is not that Path B is more interpretable than Path A. The correct claim is that Path B preserves feature-aware fusion while improving predictive performance.

### 6.6 KL baselines

| Model | Test QWK | Accuracy | F1 macro | AUC macro OVR |
|---|---:|---:|---:|---:|
| Random forest | 0.6348 | 0.5405 | 0.5078 | 0.8093 |
| MLP | 0.5932 | 0.4764 | 0.4988 | 0.7726 |

The random forest baseline was slightly above XGBoost in test QWK but below the hybrid model. It should be included as a tabular baseline in the manuscript.

### 6.7 Feature ablation

| Feature set | Test QWK | Test accuracy |
|---|---:|---:|
| JSN only | 0.6103 | 0.5242 |
| JSN + osteophyte | 0.6286 | 0.5386 |
| Full JSN + osteophyte + sclerosis | 0.6294 | 0.5399 |
| Osteophyte only | 0.4078 | 0.4312 |
| Sclerosis only | 0.2396 | 0.3865 |
| Osteophyte + sclerosis | 0.4263 | 0.4275 |

Interpretation:

JSN is the dominant tabular predictor. Osteophyte features add modest signal over JSN alone. Sclerosis adds only a small additional gain in the current data. This result should not be interpreted as sclerosis being clinically irrelevant. It means that in this implementation, with limited sclerosis labels and 224 x 224 PNG inputs, the sclerosis signal is weak relative to JSN and osteophyte. This is a key discussion point and a direction for future work.

## 7. Research Decisions and Their Justification

### 7.1 Keeping JSN as the measurement anchor

The ablation confirms that JSN is the strongest feature family for KL prediction. This matches clinical intuition because cartilage loss is a major driver of radiographic OA severity. It also validates the segmentation-plus-measurement design.

### 7.2 Keeping osteophyte grading despite mixed site performance

Osteophyte grading is necessary for clinical alignment with KL and OARSI-style feature assessment. Even when site-level performance is heterogeneous, the osteophyte burden features improve the tabular KL model over JSN-only. The correct framing is that osteophyte grading contributes useful burden information but remains limited by small manual site labels and site-specific ambiguity.

### 7.3 Keeping sclerosis as a constrained exploratory feature

Sclerosis alone has weak KL predictive value in this dataset, but the project goal is tri-feature quantification, not only maximizing KL accuracy. Sclerosis should be presented as a feasibility module whose current role is to complete the tri-feature clinical representation and whose independent performance needs more data.

### 7.4 Rejecting pseudo-label expansion when it harms generalization

Pseudo-labeling was tested for osteophytes and sclerosis. The project does not simply assume that more pseudo-labeled data is better. The manual held-out test results guided model selection. This is a scientifically stronger decision because it avoids teacher-bias amplification.

### 7.5 Separating display preprocessing from inference

The app supports CLAHE and histogram clipping as display options only. This was explicitly patched so changing display preprocessing does not rerun inference or alter the assessment result. This distinction matters for clinical trust: visualization controls should not change the diagnosis.

## 8. Explainability Claims

The strongest explainability claim is not "the model is fully interpretable." The stronger and more defensible claim is:

KOA-TriFQ produces a structured, clinically decomposed explanation by reporting the measured JSN profile, site-level osteophyte grades, sclerosis features, and KL probabilities together.

The system provides three layers of explanation:

1. Feature-level explanation: the 50-dimensional vector maps to clinical concepts.
2. Spatial explanation: overlay shows closest JSN areas, marginal osteophyte site anchors, and sclerosis heatmap.
3. Model-level explanation: Path A can be interpreted using XGBoost feature importance and SHAP.

Grad-CAM is useful as a supporting visualization for CNN outputs, but it should not be overclaimed as exact lesion localization. The literature positions Grad-CAM as coarse localization for CNN decision regions, not as pixel-perfect pathology segmentation. See [Grad-CAM](https://mlanthology.org/iccv/2017/selvaraju2017iccv-gradcam/).

## 9. Limitations

1. The dataset uses resized 224 x 224 PNG images. Physical mm calibration is not available unless original DICOM pixel spacing or calibration metadata are restored. Therefore, JSW in this implementation is primarily pixel-based. Optional mm display conversion is a user-specified visualization scale, not a validated physical measurement.

2. KL labels are global ordinal labels. They are not perfect surrogates for individual radiographic features. A knee can have inconsistent combinations of JSN, osteophytes, and sclerosis.

3. Feature-level manual labels are limited. This especially affects osteophyte site generalization and sclerosis classification.

4. Sclerosis remains the weakest module. The current binary classifier is suitable for feature integration but not sufficient as an independent clinical sclerosis detector.

5. Osteophyte visualization boxes are anatomy-guided site anchors, not true spur segmentations. Exact osteophyte localization would require pixel-level or bounding-box spur annotations.

6. The hybrid KL model outperforms the feature-only model, but its image branch is less transparent. Therefore, Path B should be presented as a performance path, while Path A remains the primary interpretable feature path.

7. External validation is not yet performed. The pipeline should be tested on a separate institution or another OAI-derived split before clinical claims.

## 10. Suggested Manuscript Structure

### Title candidate

KOA-TriFQ: Tri-Feature Quantification and Explainable Hybrid Classification for Radiographic Knee Osteoarthritis Severity Assessment

### Abstract skeleton

Background: KL grading is widely used but single-output AI classifiers provide limited feature-level explanation.

Methods: Present JSN segmentation and contour measurement, site-level osteophyte grading, sclerosis texture/CNN classification, feature aggregation, XGBoost Path A, ConvNeXt-feature hybrid Path B, and XAI dashboard.

Results: Report JSN Dice 0.8909 and mJSW ICC 0.8674, XGBoost test QWK 0.6294, hybrid test QWK 0.8436, and ablation showing JSN dominance with incremental osteophyte/sclerosis contribution.

Conclusion: KOA-TriFQ demonstrates a clinically structured, feature-decomposed framework for explainable KOA assessment, with high hybrid KL performance and transparent feature pathway, while highlighting that sclerosis and osteophyte localization require further annotation.

### Methods sections

1. Dataset and preprocessing.
2. Annotation strategy.
3. JSN segmentation and mJSW computation.
4. Osteophyte ROI extraction and ordinal grading.
5. Sclerosis ROI extraction and binary classification.
6. Feature aggregation.
7. KL classification paths.
8. Explainability interface.
9. Statistical analysis and metrics.

### Results sections

1. Dataset and feature matrix.
2. JSN segmentation/measurement performance.
3. Osteophyte grading performance.
4. Sclerosis classification and threshold tuning.
5. KL Path A and Path B performance.
6. Ablation study.
7. Qualitative dashboard examples.

### Discussion sections

1. Why JSN dominates KL prediction.
2. How osteophyte features add clinically meaningful burden information.
3. Why sclerosis is feasible but still data-limited.
4. Interpretability versus performance tradeoff between Path A and Path B.
5. Limitations of resized PNG and lack of physical calibration.
6. Future work: larger feature labels, DICOM calibration, true osteophyte localization labels, external validation.

## 11. Reproducibility Artifacts

Key result files:

- `results/jsn_evaluation.json`
- `results/jsn_manual/jsn_evaluation.json`
- `results/jsn_selftrain/jsn_evaluation.json`
- `results/jsn_selected_checkpoint.json`
- `results/osteophyte_main_manual/osteophyte_evaluation.json`
- `results/osteophyte_expanded/osteophyte_evaluation.json`
- `results/sclerosis_threshold_tuning/sclerosis_threshold_evaluation.json`
- `results/sclerosis_evaluation.json`
- `results/xgboost_metrics.json`
- `results/kl_hybrid_evaluation/kl_hybrid_evaluation.json`
- `results/kl_feature_baselines.json`
- `results/ablation/ablation_results.npz`
- `results/pipeline_evaluation/`

Key feature files:

- `features/aggregated/train_features.npz`
- `features/aggregated/val_features.npz`
- `features/aggregated/test_features.npz`
- `features/jsn/`
- `features/rois_osteophyte_clahe_full/`
- `features/osteophyte/`
- `features/sclerosis/`

Key checkpoints:

- JSN: `checkpoints/jsn_segmenter/jsn-epoch=086-val_dice=0.9012.ckpt`
- Osteophyte: `checkpoints/osteophyte/`
- Sclerosis: `checkpoints/sclerosis_binary_texture_only/sclerosis/scl-auc-epoch=044-val_auc_macro=0.6730.ckpt`
- XGBoost: `checkpoints/kl_xgboost.ubj`
- Hybrid KL: `checkpoints/kl_hybrid/hybrid-epoch=072-val_qwk=0.8053.ckpt`

Application command:

```bash
cd /Users/naim/Desktop/patella-v6
source .venv311/bin/activate
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. python app/gradio_app.py
```

## 12. Reference Links Used for Method Rationale

- Dataset: [Knee Osteoarthritis Severity Grading Dataset](https://data.mendeley.com/datasets/56rmx5bjcr/1)
- KL background: [Classifications in Brief: Kellgren-Lawrence Classification of Osteoarthritis](https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/)
- OARSI feature atlas: [Atlas of individual radiographic features in osteoarthritis, revised](https://www.sciencedirect.com/science/article/pii/S1063458406003281)
- JSW/mJSW measurement: [Location specific radiographic joint space width for osteoarthritis progression](https://pmc.ncbi.nlm.nih.gov/articles/PMC3138121/)
- Knee JSW background: [Minimum joint space width and tibial cartilage morphology in healthy knees](https://link.springer.com/article/10.1186/1471-2474-9-119)
- Radiographic acquisition limitation: [Ravaud et al., knee JSW measurement and positioning](https://academic.oup.com/rheumatology/article/35/8/761/1782848)
- Radiographic surrogate limitation: [Why radiography should no longer be considered a surrogate outcome measure for cartilage](https://arthritis-research.biomedcentral.com/articles/10.1186/ar3488)
- UNet++: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/)
- EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html)
- Squeeze-and-Excitation: [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)
- ConvNeXt: [A ConvNet for the 2020s](https://collaborate.princeton.edu/en/publications/a-convnet-for-the-2020s/)
- XGBoost: [XGBoost: A Scalable Tree Boosting System](https://www.kdd.org/kdd2016/subtopic/view/xgboost-a-scalable-tree-boosting-system/670/)
- SHAP: [A Unified Approach to Interpreting Model Predictions](https://huggingface.co/papers/1705.07874)
- Grad-CAM: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://mlanthology.org/iccv/2017/selvaraju2017iccv-gradcam/)

