#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL_STEPS=34

# Load repo-local secrets and overrides such as WANDB_API_KEY or DATA_ROOT.
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/load_env.sh"
load_repo_env "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_main_study_pipeline.sh [options]

Options:
  --skip-stage N         Skip a single numbered stage. Can be repeated.
  --skip-stages LIST     Skip multiple stages, comma-separated (e.g. 6,7,8).
  -h, --help             Show this help message.

Stages:
  1  Clean numbered duplicate files
  2  Audit pipeline readiness (preflight)
  3  Build bootstrap pseudo-label suggestions
  4  Refresh annotation manifests
  5  Prepare annotation workspace packages
  6  Prepare ROI YOLO dataset from reviewed boxes
  7  Train ROI detector
  8  Evaluate ROI detector
  9  Compare ROI backends
  10 Extract ROIs with trained detector or fallback
  11 Import reviewed annotations
  12 Train JSN segmenter
  13 Extract JSN features and masks
  14 Extract osteophyte ROI patches with CLAHE and JSN-guided boxes
  15 Train osteophyte grader (manual)
  16 Extract sclerosis features for manual teacher
  17 Train sclerosis classifier (manual)
  18 Generate high-confidence pseudo labels
  19 Train osteophyte grader (expanded)
  20 Extract sclerosis features for expanded training
  21 Train sclerosis classifier (expanded)
  22 Extract osteophyte features
  23 Re-extract sclerosis features with trained classifier
  24 Aggregate all features
  25 Train KL XGBoost classifier
  26 Train KL hybrid classifier
  27 Evaluate JSN segmenter
  28 Evaluate osteophyte grader
  29 Evaluate sclerosis classifier
  30 Run KL feature baselines
  31 Run KL ablation studies
  32 Evaluate end-to-end pipeline
  33 Build reproducibility report
  34 Audit pipeline readiness (post-run)
EOF
}

SKIP_STAGES=()

add_skip_stage() {
  local stage="$1"
  if [[ ! "${stage}" =~ ^[0-9]+$ ]] || (( stage < 1 || stage > TOTAL_STEPS )); then
    echo "Invalid stage number to skip: ${stage}" >&2
    exit 1
  fi
  local existing
  for existing in "${SKIP_STAGES[@]-}"; do
    [[ "${existing}" == "${stage}" ]] && return
  done
  SKIP_STAGES+=("${stage}")
}

parse_skip_stage_list() {
  local raw_list="$1"
  local item
  IFS=',' read -ra items <<< "${raw_list}"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    add_skip_stage "${item}"
  done
}

stage_is_skipped() {
  local stage="$1"
  local existing
  for existing in "${SKIP_STAGES[@]-}"; do
    [[ "${existing}" == "${stage}" ]] && return 0
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stage)
      [[ $# -lt 2 ]] && { echo "--skip-stage requires a stage number." >&2; exit 1; }
      add_skip_stage "$2"
      shift 2
      ;;
    --skip-stages)
      [[ $# -lt 2 ]] && { echo "--skip-stages requires a comma-separated list." >&2; exit 1; }
      parse_skip_stage_list "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    echo "${ROOT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "Unable to find a Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
}

resolve_data_root() {
  local candidates=()
  if [[ -n "${DATA_ROOT:-}" ]]; then
    candidates+=("${DATA_ROOT}")
  fi
  candidates+=("${ROOT_DIR}/KneeXrayData")

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}/ClsKLData/kneeKL224" ]]; then
      echo "${candidate}"
      return
    fi
  done

  echo "Unable to find KneeXrayData under this repo. Set DATA_ROOT=/path/to/KneeXrayData." >&2
  exit 1
}

resolve_optional_osteophyte_repro_checkpoint() {
  local checkpoint_dir="$1"
  if [[ -n "${OSTEOPHYTE_REPRO_CKPT:-}" ]]; then
    if [[ -f "${OSTEOPHYTE_REPRO_CKPT}" ]]; then
      echo "${OSTEOPHYTE_REPRO_CKPT}"
      return 0
    fi
    echo "Configured OSTEOPHYTE_REPRO_CKPT does not exist: ${OSTEOPHYTE_REPRO_CKPT}" >&2
    return 1
  fi

  local resolved
  resolved="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; sys.path.insert(0, sys.argv[2]); from src.utils.checkpoints import find_best_lightning_checkpoint; p = find_best_lightning_checkpoint(Path(sys.argv[1]), pattern="hybrid-*.ckpt", monitor="val_kappa_mean"); print("" if p is None else p)' "${checkpoint_dir}" "${ROOT_DIR}")"
  if [[ -n "${resolved}" && -f "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  local preferred="${checkpoint_dir}/hybrid-epoch=078-val_kappa_mean=0.8264.ckpt"
  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  return 1
}

resolve_optional_sclerosis_teacher_checkpoint() {
  local checkpoint_dir="$1"
  local checkpoint_monitor="${SCLEROSIS_CHECKPOINT_MONITOR:-val_f1_macro}"
  local checkpoint_mode="${SCLEROSIS_CHECKPOINT_MODE:-max}"
  if [[ -n "${SCLEROSIS_TEACHER_CKPT:-}" ]]; then
    if [[ -f "${SCLEROSIS_TEACHER_CKPT}" ]]; then
      echo "${SCLEROSIS_TEACHER_CKPT}"
      return 0
    fi
    echo "Configured SCLEROSIS_TEACHER_CKPT does not exist: ${SCLEROSIS_TEACHER_CKPT}" >&2
    return 1
  fi

  local resolved
  resolved="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; sys.path.insert(0, sys.argv[4]); from src.utils.checkpoints import find_best_lightning_checkpoint; p = find_best_lightning_checkpoint(Path(sys.argv[1]), pattern="scl-*.ckpt", monitor=sys.argv[2], mode=sys.argv[3]); print("" if p is None else p)' "${checkpoint_dir}" "${checkpoint_monitor}" "${checkpoint_mode}" "${ROOT_DIR}")"
  if [[ -n "${resolved}" && -f "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  local preferred_binary="${checkpoint_dir}/scl-auc-epoch=044-val_auc_macro=0.6730.ckpt"
  if [[ "${SCLEROSIS_LABEL_SCHEME:-}" == "binary_present" && -f "${preferred_binary}" ]]; then
    echo "${preferred_binary}"
    return 0
  fi

  local preferred="${checkpoint_dir}/scl-f1-epoch=036-val_f1_macro=0.3662.ckpt"
  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
    return 0
  fi

  return 1
}

reviewed_roi_annotations_available() {
  local reviewed_dir="${ROOT_DIR}/annotations/reviewed"
  local candidate
  for candidate in \
    "${reviewed_dir}/roi_boxes.csv" \
    "${reviewed_dir}/roi_annotations.csv" \
    "${reviewed_dir}/roi_cvat_export.json" \
    "${reviewed_dir}/roi_boxes_coco.json"; do
    [[ -f "${candidate}" ]] && return 0
  done
  return 1
}

print_skip_message() {
  local step="$1"
  local label="$2"
  local reason="${3:-}"
  echo "[${step}/${TOTAL_STEPS}] Skipping ${label}"
  [[ -n "${reason}" ]] && echo "${reason}"
  echo
}

run_step() {
  local step="$1"
  local label="$2"
  shift 2

  if stage_is_skipped "${step}"; then
    print_skip_message "${step}" "${label}"
    return
  fi

  echo "[${step}/${TOTAL_STEPS}] ${label}"
  "${PYTHON_BIN}" "$@"
  echo
}

run_optional_roi_step() {
  local step="$1"
  local label="$2"
  shift 2

  if stage_is_skipped "${step}"; then
    print_skip_message "${step}" "${label}"
    return
  fi
  if ! reviewed_roi_annotations_available; then
    print_skip_message \
      "${step}" \
      "${label}" \
      "Reviewed ROI annotations not found under ${ROOT_DIR}/annotations/reviewed; skipping ROI-detector-specific stage."
    return
  fi

  echo "[${step}/${TOTAL_STEPS}] ${label}"
  "${PYTHON_BIN}" "$@"
  echo
}

reset_osteophyte_roi_outputs() {
  local roi_dir="$1"

  mkdir -p "${roi_dir}"
  rm -rf \
    "${roi_dir}/train" \
    "${roi_dir}/val" \
    "${roi_dir}/test" \
    "${roi_dir}/roi_extraction_audit.csv" \
    "${roi_dir}/roi_extraction_failures.csv"
}

reset_sclerosis_outputs() {
  local scl_dir="$1"

  mkdir -p "${scl_dir}"
  rm -rf \
    "${scl_dir}/patches" \
    "${scl_dir}/train_sclerosis_data.npz" \
    "${scl_dir}/val_sclerosis_data.npz" \
    "${scl_dir}/test_sclerosis_data.npz" \
    "${scl_dir}/train_sclerosis_features.npz" \
    "${scl_dir}/val_sclerosis_features.npz" \
    "${scl_dir}/test_sclerosis_features.npz" \
    "${scl_dir}/train_sclerosis_lbp_histograms.npz" \
    "${scl_dir}/val_sclerosis_lbp_histograms.npz" \
    "${scl_dir}/test_sclerosis_lbp_histograms.npz" \
    "${scl_dir}/train_sclerosis_progress.csv" \
    "${scl_dir}/val_sclerosis_progress.csv" \
    "${scl_dir}/test_sclerosis_progress.csv" \
    "${scl_dir}/sclerosis_extraction_failures.csv" \
    "${scl_dir}/sclerosis_extraction_failures_remaining.csv"
}

PYTHON_BIN="$(resolve_python_bin)"
DATA_ROOT="$(resolve_data_root)"
DATA_DIR="${DATA_ROOT}/ClsKLData/kneeKL224"
FEATURE_REVIEW_TEMPLATE="${ROOT_DIR}/annotations/packages/feature_grading/feature_review_template.csv"
OSTEOPHYTE_REPRO_CKPT_DIR="${OSTEOPHYTE_REPRO_CKPT_DIR:-${ROOT_DIR}/checkpoints/clahe_fullimage_ordinal}"
OSTEOPHYTE_REPRO_ROI_DIR="${OSTEOPHYTE_REPRO_ROI_DIR:-${ROOT_DIR}/features/rois_osteophyte_clahe_full}"
OSTEOPHYTE_RESULTS_DIR="${OSTEOPHYTE_RESULTS_DIR:-${ROOT_DIR}/results/osteophyte_main_manual}"
SCLEROSIS_MANUAL_DIR="${SCLEROSIS_MANUAL_DIR:-${ROOT_DIR}/features/sclerosis_manual_teacher}"
SCLEROSIS_EXPANDED_DIR="${SCLEROSIS_EXPANDED_DIR:-${ROOT_DIR}/features/sclerosis_expanded_teacher}"
SCLEROSIS_FINAL_DIR="${SCLEROSIS_FINAL_DIR:-${ROOT_DIR}/features/sclerosis}"
SCLEROSIS_STANDARDIZER_DIR="${SCLEROSIS_STANDARDIZER_DIR:-}"
SCLEROSIS_LABEL_SCHEME="${SCLEROSIS_LABEL_SCHEME:-severity}"
SCLEROSIS_BINARY_THRESHOLD="${SCLEROSIS_BINARY_THRESHOLD:-}"
SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL="${SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL:-}"
SCLEROSIS_FORCE_MODEL_PREDICTIONS="${SCLEROSIS_FORCE_MODEL_PREDICTIONS:-}"
SCLEROSIS_CHECKPOINT_DIR="${SCLEROSIS_CHECKPOINT_DIR:-}"
SCLEROSIS_CHECKPOINT_MONITOR="${SCLEROSIS_CHECKPOINT_MONITOR:-}"
SCLEROSIS_CHECKPOINT_MODE="${SCLEROSIS_CHECKPOINT_MODE:-max}"
SCLEROSIS_INPUT_MODE="${SCLEROSIS_INPUT_MODE:-}"
SCLEROSIS_STRATEGY="${SCLEROSIS_STRATEGY:-}"
SCLEROSIS_FINAL_LABEL_MODE="${SCLEROSIS_FINAL_LABEL_MODE:-}"
SCLEROSIS_PSEUDO_CONFIDENCE="${SCLEROSIS_PSEUDO_CONFIDENCE:-0.90}"
SCLEROSIS_PSEUDO_MIN_CONFIDENCE="${SCLEROSIS_PSEUDO_MIN_CONFIDENCE:-0.75}"
SCLEROSIS_PSEUDO_TARGET_ROWS="${SCLEROSIS_PSEUDO_TARGET_ROWS:-250}"
SCLEROSIS_PSEUDO_WEIGHT="${SCLEROSIS_PSEUDO_WEIGHT:-0.35}"
SCLEROSIS_PSEUDO_TEACHER_CKPT="${SCLEROSIS_PSEUDO_TEACHER_CKPT:-}"
SCLEROSIS_EVAL_DIR="${SCLEROSIS_EVAL_DIR:-}"
OSTEOPHYTE_WARM_START_CHECKPOINT=""
SCLEROSIS_TEACHER_CHECKPOINT=""

if [[ -z "${SCLEROSIS_INPUT_MODE}" ]]; then
  if [[ "${SCLEROSIS_LABEL_SCHEME}" == "binary_present" ]]; then
    SCLEROSIS_INPUT_MODE="texture_only"
  else
    SCLEROSIS_INPUT_MODE="hybrid"
  fi
fi

if [[ -z "${SCLEROSIS_STRATEGY}" ]]; then
  if [[ "${SCLEROSIS_LABEL_SCHEME}" == "binary_present" ]]; then
    SCLEROSIS_STRATEGY="shared"
  else
    SCLEROSIS_STRATEGY="multitask_heads"
  fi
fi

if [[ -z "${SCLEROSIS_CHECKPOINT_DIR}" ]]; then
  if [[ "${SCLEROSIS_LABEL_SCHEME}" == "binary_present" ]]; then
    SCLEROSIS_CHECKPOINT_DIR="${ROOT_DIR}/checkpoints/sclerosis_binary_texture_only"
  else
    SCLEROSIS_CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
  fi
fi

if [[ "${SCLEROSIS_LABEL_SCHEME}" == "binary_present" ]]; then
  SCLEROSIS_BINARY_THRESHOLD="${SCLEROSIS_BINARY_THRESHOLD:-0.4227197766304016}"
  SCLEROSIS_STANDARDIZER_DIR="${SCLEROSIS_STANDARDIZER_DIR:-${SCLEROSIS_MANUAL_DIR}}"
  SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL="${SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL:-1}"
  SCLEROSIS_FORCE_MODEL_PREDICTIONS="${SCLEROSIS_FORCE_MODEL_PREDICTIONS:-1}"
  SCLEROSIS_FINAL_LABEL_MODE="${SCLEROSIS_FINAL_LABEL_MODE:-manual}"
  SCLEROSIS_CHECKPOINT_MONITOR="${SCLEROSIS_CHECKPOINT_MONITOR:-val_auc_macro}"
else
  SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL="${SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL:-0}"
  SCLEROSIS_FORCE_MODEL_PREDICTIONS="${SCLEROSIS_FORCE_MODEL_PREDICTIONS:-0}"
  SCLEROSIS_FINAL_LABEL_MODE="${SCLEROSIS_FINAL_LABEL_MODE:-expanded}"
  SCLEROSIS_CHECKPOINT_MONITOR="${SCLEROSIS_CHECKPOINT_MONITOR:-val_f1_macro}"
fi
SCLEROSIS_EVAL_DIR="${SCLEROSIS_EVAL_DIR:-${SCLEROSIS_STANDARDIZER_DIR:-${SCLEROSIS_MANUAL_DIR}}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset directory: ${DATA_DIR}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

export PROJECT_ROOT="${ROOT_DIR}"
export DATA_ROOT
export PYTHONPATH="${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export ANNOTATION_PACKAGE_COPY_IMAGES="${ANNOTATION_PACKAGE_COPY_IMAGES:-0}"
export ENABLE_JSN_SELF_TRAINING="${ENABLE_JSN_SELF_TRAINING:-0}"

echo "Preparing pipeline environment..."
echo "Project root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Data root: ${DATA_ROOT}"
echo "Annotation package image copies: ${ANNOTATION_PACKAGE_COPY_IMAGES}"
echo "JSN self-training branch: ${ENABLE_JSN_SELF_TRAINING}"
echo "Sclerosis label scheme: ${SCLEROSIS_LABEL_SCHEME}"
echo "Sclerosis input mode: ${SCLEROSIS_INPUT_MODE}"
echo "Sclerosis strategy: ${SCLEROSIS_STRATEGY}"
echo "Sclerosis binary threshold: ${SCLEROSIS_BINARY_THRESHOLD:-argmax/default}"
echo "Sclerosis standardizer dir: ${SCLEROSIS_STANDARDIZER_DIR:-same as output dir}"
echo "Sclerosis checkpoint root: ${SCLEROSIS_CHECKPOINT_DIR}"
echo "Sclerosis checkpoint monitor: ${SCLEROSIS_CHECKPOINT_MONITOR} (${SCLEROSIS_CHECKPOINT_MODE})"
echo "Sclerosis final label mode: ${SCLEROSIS_FINAL_LABEL_MODE}"
echo "Sclerosis eval feature dir: ${SCLEROSIS_EVAL_DIR}"
echo "Sclerosis pseudo confidence: ${SCLEROSIS_PSEUDO_CONFIDENCE} (min ${SCLEROSIS_PSEUDO_MIN_CONFIDENCE}, target rows ${SCLEROSIS_PSEUDO_TARGET_ROWS})"
echo "Sclerosis pseudo source weight: ${SCLEROSIS_PSEUDO_WEIGHT}"
echo

if [[ "${OSTEOPHYTE_DISABLE_WARM_START:-0}" != "1" ]]; then
  echo "Resolving osteophyte warm-start checkpoint..."
  if OSTEOPHYTE_WARM_START_CHECKPOINT="$(resolve_optional_osteophyte_repro_checkpoint "${OSTEOPHYTE_REPRO_CKPT_DIR}")"; then
    echo "Using reproduced osteophyte checkpoint as warm-start: ${OSTEOPHYTE_WARM_START_CHECKPOINT}"
    echo
  else
    echo "No reproduced osteophyte checkpoint resolved; stage 15 will train from scratch."
    echo
  fi
fi

echo "Resolving sclerosis teacher checkpoint..."
if SCLEROSIS_TEACHER_CHECKPOINT="$(resolve_optional_sclerosis_teacher_checkpoint "${SCLEROSIS_CHECKPOINT_DIR}/sclerosis")"; then
  if [[ -n "${SCLEROSIS_TEACHER_CKPT:-}" ]]; then
    echo "Using pinned sclerosis teacher checkpoint: ${SCLEROSIS_TEACHER_CHECKPOINT}"
  else
    echo "Resolved sclerosis checkpoint for reference: ${SCLEROSIS_TEACHER_CHECKPOINT}"
    echo "Final extraction will re-resolve the best checkpoint at runtime unless SCLEROSIS_TEACHER_CKPT is set."
  fi
  echo
else
  echo "No pinned sclerosis teacher checkpoint resolved; stages 18, 20, and 23 will auto-pick from checkpoints."
  echo
fi
if [[ -z "${SCLEROSIS_PSEUDO_TEACHER_CKPT}" && -n "${SCLEROSIS_TEACHER_CHECKPOINT}" ]]; then
  SCLEROSIS_PSEUDO_TEACHER_CKPT="${SCLEROSIS_TEACHER_CHECKPOINT}"
fi

run_step 1 "Cleaning numbered duplicate files" \
  "${ROOT_DIR}/scripts/clean_duplicate_suffix_files.py" \
  --project-root "${ROOT_DIR}" \
  --remove

run_step 2 "Auditing pipeline readiness (preflight)" \
  "${ROOT_DIR}/scripts/check_pipeline_readiness.py" \
  --project-root "${ROOT_DIR}"

run_step 3 "Building bootstrap pseudo-label suggestions" \
  "${ROOT_DIR}/scripts/bootstrap_pseudo_labels.py"

run_step 4 "Refreshing annotation manifests" \
  "${ROOT_DIR}/scripts/refresh_annotation_manifests.py"

if [[ "${FORCE_REBUILD_ANNOTATION_PACKAGES:-0}" == "1" || ! -f "${FEATURE_REVIEW_TEMPLATE}" ]]; then
  run_step 5 "Preparing annotation workspace packages" \
    "${ROOT_DIR}/scripts/prepare_annotation_workspace.py"
else
  print_skip_message \
    5 \
    "Preparing annotation workspace packages" \
    "Existing reviewed feature sheet detected at ${FEATURE_REVIEW_TEMPLATE}. Set FORCE_REBUILD_ANNOTATION_PACKAGES=1 to regenerate annotation packages."
fi

run_optional_roi_step 6 "Preparing ROI YOLO dataset from reviewed boxes" \
  "${ROOT_DIR}/scripts/prepare_roi_yolo_dataset.py"

run_optional_roi_step 7 "Training ROI detector" \
  "${ROOT_DIR}/scripts/train_roi_detector.py" \
  +model=yolov8

run_optional_roi_step 8 "Evaluating ROI detector" \
  "${ROOT_DIR}/scripts/evaluate_roi_detector.py" \
  +model=yolov8

run_optional_roi_step 9 "Comparing ROI backends" \
  "${ROOT_DIR}/scripts/compare_roi_backends.py" \
  +model=yolov8 \
  preprocessing.landmark_backend=kneel_repo

run_optional_roi_step 10 "Extracting ROIs with trained detector or fallback" \
  "${ROOT_DIR}/scripts/extract_rois.py"

run_step 11 "Importing reviewed annotations" \
  "${ROOT_DIR}/scripts/import_reviewed_annotations.py"

run_step 12 "Training JSN segmenter" \
  "${ROOT_DIR}/scripts/train_jsn_segmenter.py" \
  training.label_mode=manual \
  +model=unetpp

run_step 13 "Extracting JSN features and masks" \
  "${ROOT_DIR}/scripts/extract_jsn_features.py" \
  training.label_mode=manual \
  +model=unetpp \
  jsn_checkpoint_subdir=jsn_segmenter \
  jsn_selected_checkpoint_file=null

if stage_is_skipped 13; then
  print_skip_message 13 "JSN self-training comparison" "Stage 13 was skipped, so the optional JSN self-training branch was not run."
elif [[ "${ENABLE_JSN_SELF_TRAINING}" == "1" ]]; then
  echo "[13a/${TOTAL_STEPS}] Generating high-confidence JSN pseudo-masks"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/generate_jsn_pseudo_masks.py" \
    +model=unetpp \
    jsn_checkpoint_subdir=jsn_segmenter \
    checkpoint_monitor=val_mjsw_mae \
    checkpoint_mode=min
  echo

  echo "[13b/${TOTAL_STEPS}] Training JSN segmenter with manual masks plus pseudo-masks"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/train_jsn_segmenter.py" \
    training.label_mode=manual \
    +model=unetpp \
    jsn_checkpoint_subdir=jsn_segmenter_selftrain \
    jsn_logger_name=jsn_segmenter_selftrain \
    training.jsn_self_training.enabled=true
  echo

  echo "[13c/${TOTAL_STEPS}] Evaluating manual-only JSN checkpoint"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/evaluate_jsn_segmenter.py" \
    +model=unetpp \
    jsn_checkpoint_subdir=jsn_segmenter \
    checkpoint_monitor=val_mjsw_mae \
    checkpoint_mode=min \
    result_dir="${ROOT_DIR}/results/jsn_manual"
  echo

  echo "[13d/${TOTAL_STEPS}] Evaluating self-trained JSN checkpoint"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/evaluate_jsn_segmenter.py" \
    +model=unetpp \
    jsn_checkpoint_subdir=jsn_segmenter_selftrain \
    checkpoint_monitor=val_mjsw_mae \
    checkpoint_mode=min \
    result_dir="${ROOT_DIR}/results/jsn_selftrain"
  echo

  echo "[13e/${TOTAL_STEPS}] Selecting best JSN checkpoint for downstream features"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/select_best_jsn_checkpoint.py" \
    --manual-json "${ROOT_DIR}/results/jsn_manual/jsn_evaluation.json" \
    --selftrain-json "${ROOT_DIR}/results/jsn_selftrain/jsn_evaluation.json" \
    --output "${ROOT_DIR}/checkpoints/jsn_segmenter_selected.txt" \
    --summary-output "${ROOT_DIR}/results/jsn_selected_checkpoint.json" \
    --primary-metric mjsw_mae \
    --mode min
  echo

  echo "[13f/${TOTAL_STEPS}] Re-extracting JSN features and masks with selected checkpoint"
  SELECTED_JSN_CHECKPOINT="$(<"${ROOT_DIR}/checkpoints/jsn_segmenter_selected.txt")"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/extract_jsn_features.py" \
    training.label_mode=manual \
    +model=unetpp \
    "checkpoint_path='${SELECTED_JSN_CHECKPOINT}'"
  echo
else
  echo "[13a/${TOTAL_STEPS}] JSN self-training branch disabled"
  echo "Set ENABLE_JSN_SELF_TRAINING=1 to train and compare manual-only vs self-trained JSN checkpoints."
  echo
fi

if stage_is_skipped 14; then
  print_skip_message 14 "Extracting osteophyte ROI patches with CLAHE and JSN-guided boxes"
else
  echo "[14/${TOTAL_STEPS}] Resetting osteophyte ROI outputs"
  echo "Removing prior ROI crops/logs from ${OSTEOPHYTE_REPRO_ROI_DIR}"
  reset_osteophyte_roi_outputs "${OSTEOPHYTE_REPRO_ROI_DIR}"
  echo

  echo "[14/${TOTAL_STEPS}] Extracting osteophyte ROI patches with CLAHE and JSN-guided boxes"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/extract_roi_with_fullimage_clahe.py" \
    --input-dir "${DATA_DIR}" \
    --output-dir "${OSTEOPHYTE_REPRO_ROI_DIR}" \
    --scan-all-images \
    --clahe-clip 3.0 \
    --clahe-tile 8 \
    --osteophyte-roi-size 140 \
    --reviewed-jsn-mask-dir "${ROOT_DIR}/annotations/jsn_masks" \
    --predicted-jsn-mask-dir "${ROOT_DIR}/features/jsn/masks" \
    --landmark-backend heuristic \
    --skip-landmark-if-no-jsn \
    --progress-every 25
  echo
fi

STAGE15_ARGS=(
  "${ROOT_DIR}/scripts/train_osteophyte_grader.py"
  training.label_mode=manual
  +model=se_resnet50
  preprocessing.augmentation.horizontal_flip_p=0.0
  preprocessing.clahe=null
  preprocessing.histogram_clip=null
  training.scheduler=reduce_on_plateau
  training.scheduler_params.mode=max
  training.scheduler_params.factor=0.5
  training.scheduler_params.patience=10
  training.layer_wise_lr.enabled=true
  training.layer_wise_lr.backbone_ratio=0.1
  training.osteophyte_class_balance.enabled=true
  training.osteophyte_sampling.strategy=mean_class_balance
  training.osteophyte_refinement.sites='[lateral_femur,medial_tibia]'
  osteophyte_roi_dir="${OSTEOPHYTE_REPRO_ROI_DIR}"
)
if [[ -n "${OSTEOPHYTE_WARM_START_CHECKPOINT}" ]]; then
  STAGE15_ARGS+=(
    "training.osteophyte_warm_start_checkpoint='${OSTEOPHYTE_WARM_START_CHECKPOINT}'"
    training.osteophyte_force_retrain_multitask=true
  )
fi

SCLEROSIS_MODEL_ARGS=(
  +model=sclerosis_hybrid
  "training.sclerosis_strategy=${SCLEROSIS_STRATEGY}"
  "model.input_mode=${SCLEROSIS_INPUT_MODE}"
  model.use_side_specific_heads=false
  model.dropout_cnn=0.3
  model.dropout_fusion=0.4
)

SCLEROSIS_TRAINING_ARGS=(
  "training.sclerosis_label_scheme=${SCLEROSIS_LABEL_SCHEME}"
  training.scheduler=cosine
  training.weight_decay=1.0e-4
  training.early_stopping.patience=20
  training.accumulate_grad_batches=4
  training.log_every_n_steps=5
  training.sclerosis_backbone_freeze_epochs=0
)
if [[ "${SCLEROSIS_LABEL_SCHEME}" == "binary_present" ]]; then
  SCLEROSIS_TRAINING_ARGS+=(
    training.learning_rate=1.0e-4
    training.max_epochs=100
    training.sclerosis_use_ordinal_loss=false
    training.sclerosis_sampling.multiplier_power=0.5
    training.sclerosis_sampling.max_weight_ratio_to_median=1.5
    "training.sclerosis_source_weights.high_conf_model=${SCLEROSIS_PSEUDO_WEIGHT}"
  )
else
  SCLEROSIS_TRAINING_ARGS+=(
    training.learning_rate=3.0e-5
    training.sclerosis_sampling.multiplier_power=0.75
    training.sclerosis_sampling.max_weight_ratio_to_median=2.0
    training.sclerosis_ordinal_weight=0.1
  )
fi

SCLEROSIS_DEVPOOL_ARGS=(
  training.sclerosis_dev_pool.enabled=true
  training.sclerosis_dev_pool.n_splits=5
  training.sclerosis_dev_pool.holdout_fold=0
)

SCLEROSIS_STAGE17_TUNED_ARGS=(
  training.early_stopping.patience=24
  model.dropout_fusion=0.4
)

SCLEROSIS_PSEUDO_ARGS=(
  "training.sclerosis_primary_monitor=${SCLEROSIS_CHECKPOINT_MONITOR}"
  "training.sclerosis_primary_mode=${SCLEROSIS_CHECKPOINT_MODE}"
  training.pseudo_confidence_threshold_osteophyte=0.90
  "training.pseudo_confidence_threshold_sclerosis=${SCLEROSIS_PSEUDO_CONFIDENCE}"
  "training.pseudo_confidence_threshold_sclerosis_min=${SCLEROSIS_PSEUDO_MIN_CONFIDENCE}"
  "training.pseudo_target_rows_sclerosis=${SCLEROSIS_PSEUDO_TARGET_ROWS}"
  "training.sclerosis_roi_source_filter.pseudo=[jsn_guided]"
)
if [[ -n "${SCLEROSIS_BINARY_THRESHOLD}" ]]; then
  SCLEROSIS_PSEUDO_ARGS+=(
    "training.sclerosis_binary_threshold=${SCLEROSIS_BINARY_THRESHOLD}"
  )
fi
if [[ -n "${SCLEROSIS_PSEUDO_TEACHER_CKPT}" ]]; then
  SCLEROSIS_PSEUDO_ARGS+=(
    "checkpoint_path='${SCLEROSIS_PSEUDO_TEACHER_CKPT}'"
  )
fi

SCLEROSIS_EXTRACTOR_TEACHER_ARGS=(
  "checkpoint_dir=${SCLEROSIS_CHECKPOINT_DIR}"
  "checkpoint_monitor=${SCLEROSIS_CHECKPOINT_MONITOR}"
  "checkpoint_mode=${SCLEROSIS_CHECKPOINT_MODE}"
)
if [[ -n "${SCLEROSIS_TEACHER_CKPT:-}" && -n "${SCLEROSIS_TEACHER_CHECKPOINT}" ]]; then
  SCLEROSIS_EXTRACTOR_TEACHER_ARGS+=(
    "checkpoint_path='${SCLEROSIS_TEACHER_CHECKPOINT}'"
  )
fi

SCLEROSIS_FINAL_EXTRACTOR_ARGS=("${SCLEROSIS_EXTRACTOR_TEACHER_ARGS[@]}")
if [[ -n "${SCLEROSIS_BINARY_THRESHOLD}" ]]; then
  SCLEROSIS_FINAL_EXTRACTOR_ARGS+=(
    "training.sclerosis_binary_threshold=${SCLEROSIS_BINARY_THRESHOLD}"
  )
fi
if [[ -n "${SCLEROSIS_STANDARDIZER_DIR}" ]]; then
  SCLEROSIS_FINAL_EXTRACTOR_ARGS+=(
    "sclerosis_standardizer_dir=${SCLEROSIS_STANDARDIZER_DIR}"
  )
fi
if [[ "${SCLEROSIS_APPLY_CLASSIFIER_IN_MANUAL}" == "1" ]]; then
  SCLEROSIS_FINAL_EXTRACTOR_ARGS+=(
    training.sclerosis_apply_classifier_in_manual=true
  )
fi
if [[ "${SCLEROSIS_FORCE_MODEL_PREDICTIONS}" == "1" ]]; then
  SCLEROSIS_FINAL_EXTRACTOR_ARGS+=(
    training.sclerosis_force_model_predictions=true
  )
fi

run_step 15 "Training osteophyte grader (manual-label main framework)" \
  "${STAGE15_ARGS[@]}"

if stage_is_skipped 16; then
  print_skip_message 16 "Extracting sclerosis features for manual teacher"
else
  if [[ "${SKIP_SCLEROSIS_RESET:-0}" == "1" ]]; then
    print_skip_message 16 "Resetting sclerosis feature outputs" "SKIP_SCLEROSIS_RESET=1"
  else
    echo "[16/${TOTAL_STEPS}] Resetting sclerosis feature outputs"
    echo "Removing prior sclerosis patches/features from ${SCLEROSIS_MANUAL_DIR}"
    reset_sclerosis_outputs "${SCLEROSIS_MANUAL_DIR}"
    echo
  fi

  echo "[16/${TOTAL_STEPS}] Extracting sclerosis features for manual teacher"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/extract_sclerosis_features.py" \
    training.label_mode=manual \
    sclerosis_output_dir="${SCLEROSIS_MANUAL_DIR}" \
    "${SCLEROSIS_MODEL_ARGS[@]}"
  echo
fi

run_step 17 "Training sclerosis classifier (manual teacher)" \
  "${ROOT_DIR}/scripts/train_sclerosis.py" \
  training.label_mode=manual \
  sclerosis_output_dir="${SCLEROSIS_MANUAL_DIR}" \
  checkpoint_dir="${SCLEROSIS_CHECKPOINT_DIR}" \
  "${SCLEROSIS_MODEL_ARGS[@]}" \
  "${SCLEROSIS_DEVPOOL_ARGS[@]}" \
  "${SCLEROSIS_TRAINING_ARGS[@]}" \
  "${SCLEROSIS_STAGE17_TUNED_ARGS[@]}"

run_step 18 "Generating high-confidence pseudo-label expansions" \
  "${ROOT_DIR}/scripts/generate_feature_pseudolabels.py" \
  preprocessing.clahe=null \
  preprocessing.histogram_clip=null \
  sclerosis_output_dir="${SCLEROSIS_MANUAL_DIR}" \
  "${SCLEROSIS_MODEL_ARGS[@]}" \
  "${SCLEROSIS_PSEUDO_ARGS[@]}" \
  osteophyte_roi_dir="${OSTEOPHYTE_REPRO_ROI_DIR}"

STAGE19_ARGS=(
  "${ROOT_DIR}/scripts/train_osteophyte_grader.py"
  training.label_mode=expanded
  +model=se_resnet50
  preprocessing.augmentation.horizontal_flip_p=0.0
  preprocessing.clahe=null
  preprocessing.histogram_clip=null
  training.scheduler=reduce_on_plateau
  training.scheduler_params.mode=max
  training.scheduler_params.factor=0.5
  training.scheduler_params.patience=10
  training.layer_wise_lr.enabled=true
  training.layer_wise_lr.backbone_ratio=0.1
  training.osteophyte_class_balance.enabled=true
  training.osteophyte_sampling.strategy=mean_class_balance
  training.osteophyte_refinement.sites='[lateral_femur,medial_tibia]'
  training.osteophyte_force_retrain_multitask=true
  osteophyte_roi_dir="${OSTEOPHYTE_REPRO_ROI_DIR}"
)
if [[ -d "${ROOT_DIR}/checkpoints/osteophyte" ]]; then
  EXPANDED_MANUAL_BASE_CKPT="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sys; sys.path.insert(0, sys.argv[2]); from src.utils.checkpoints import find_best_lightning_checkpoint; p = find_best_lightning_checkpoint(Path(sys.argv[1]), pattern="osp-multitask-*.ckpt", monitor="val_kappa_mean"); print("" if p is None else p)' "${ROOT_DIR}/checkpoints/osteophyte" "${ROOT_DIR}")"
  if [[ -n "${EXPANDED_MANUAL_BASE_CKPT}" && -f "${EXPANDED_MANUAL_BASE_CKPT}" ]]; then
    STAGE19_ARGS+=(
      "training.osteophyte_warm_start_checkpoint='${EXPANDED_MANUAL_BASE_CKPT}'"
    )
  elif [[ -n "${OSTEOPHYTE_WARM_START_CHECKPOINT}" ]]; then
    STAGE19_ARGS+=(
      "training.osteophyte_warm_start_checkpoint='${OSTEOPHYTE_WARM_START_CHECKPOINT}'"
    )
  fi
elif [[ -n "${OSTEOPHYTE_WARM_START_CHECKPOINT}" ]]; then
  STAGE19_ARGS+=(
    "training.osteophyte_warm_start_checkpoint='${OSTEOPHYTE_WARM_START_CHECKPOINT}'"
  )
fi

run_step 19 "Training osteophyte grader (expanded manual + high-confidence)" \
  "${STAGE19_ARGS[@]}"

if stage_is_skipped 20; then
  print_skip_message 20 "Extracting sclerosis features for expanded training"
else
  if [[ "${SKIP_SCLEROSIS_RESET:-0}" == "1" ]]; then
    print_skip_message 20 "Resetting sclerosis feature outputs" "SKIP_SCLEROSIS_RESET=1"
  else
    echo "[20/${TOTAL_STEPS}] Resetting sclerosis feature outputs"
    echo "Removing prior sclerosis patches/features from ${SCLEROSIS_EXPANDED_DIR}"
    reset_sclerosis_outputs "${SCLEROSIS_EXPANDED_DIR}"
    echo
  fi

  echo "[20/${TOTAL_STEPS}] Extracting sclerosis features for expanded training"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/extract_sclerosis_features.py" \
    training.label_mode=expanded \
    sclerosis_output_dir="${SCLEROSIS_EXPANDED_DIR}" \
    "${SCLEROSIS_EXTRACTOR_TEACHER_ARGS[@]}" \
    "${SCLEROSIS_MODEL_ARGS[@]}"
  echo
fi

run_step 21 "Training sclerosis classifier (expanded manual + high-confidence)" \
  "${ROOT_DIR}/scripts/train_sclerosis.py" \
  training.label_mode=expanded \
  sclerosis_output_dir="${SCLEROSIS_EXPANDED_DIR}" \
  checkpoint_dir="${SCLEROSIS_CHECKPOINT_DIR}" \
  "${SCLEROSIS_MODEL_ARGS[@]}" \
  "${SCLEROSIS_DEVPOOL_ARGS[@]}" \
  "${SCLEROSIS_TRAINING_ARGS[@]}" \
  "${SCLEROSIS_STAGE17_TUNED_ARGS[@]}"

run_step 22 "Extracting osteophyte features" \
  "${ROOT_DIR}/scripts/extract_osteophyte_features.py" \
  +model=se_resnet50 \
  osteophyte_roi_dir="${OSTEOPHYTE_REPRO_ROI_DIR}"

if stage_is_skipped 23; then
  print_skip_message 23 "Re-extracting sclerosis features with trained classifier"
else
  if [[ "${SKIP_SCLEROSIS_RESET:-0}" == "1" ]]; then
    print_skip_message 23 "Resetting sclerosis feature outputs" "SKIP_SCLEROSIS_RESET=1"
  else
    echo "[23/${TOTAL_STEPS}] Resetting sclerosis feature outputs"
    echo "Removing prior sclerosis patches/features from ${SCLEROSIS_FINAL_DIR}"
    reset_sclerosis_outputs "${SCLEROSIS_FINAL_DIR}"
    echo
  fi

  echo "[23/${TOTAL_STEPS}] Re-extracting sclerosis features with trained classifier"
  "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/extract_sclerosis_features.py" \
    "training.label_mode=${SCLEROSIS_FINAL_LABEL_MODE}" \
    sclerosis_output_dir="${SCLEROSIS_FINAL_DIR}" \
    "${SCLEROSIS_FINAL_EXTRACTOR_ARGS[@]}" \
    "${SCLEROSIS_MODEL_ARGS[@]}"
  echo
fi

run_step 24 "Aggregating all features" \
  "${ROOT_DIR}/scripts/extract_all_features.py"

run_step 25 "Training KL XGBoost classifier" \
  "${ROOT_DIR}/scripts/train_kl_xgboost.py" \
  +model=xgboost

run_step 26 "Training KL hybrid classifier" \
  "${ROOT_DIR}/scripts/train_kl_hybrid.py" \
  +model=convnext_hybrid

run_step 27 "Evaluating JSN segmenter" \
  "${ROOT_DIR}/scripts/evaluate_jsn_segmenter.py" \
  +model=unetpp \
  checkpoint_monitor=val_mjsw_mae \
  checkpoint_mode=min

run_step 28 "Evaluating osteophyte grader (manual-label main framework)" \
  "${ROOT_DIR}/scripts/evaluate_osteophyte_grader.py" \
  training.label_mode=manual \
  +model=se_resnet50 \
  preprocessing.clahe=null \
  preprocessing.histogram_clip=null \
  osteophyte_roi_dir="${OSTEOPHYTE_REPRO_ROI_DIR}" \
  result_dir="${OSTEOPHYTE_RESULTS_DIR}"

run_step 29 "Evaluating sclerosis classifier" \
  "${ROOT_DIR}/scripts/evaluate_sclerosis.py" \
  training.label_mode=manual \
  "training.sclerosis_label_scheme=${SCLEROSIS_LABEL_SCHEME}" \
  sclerosis_output_dir="${SCLEROSIS_EVAL_DIR}" \
  checkpoint_dir="${SCLEROSIS_CHECKPOINT_DIR}" \
  "checkpoint_monitor=${SCLEROSIS_CHECKPOINT_MONITOR}" \
  "checkpoint_mode=${SCLEROSIS_CHECKPOINT_MODE}" \
  "${SCLEROSIS_MODEL_ARGS[@]}"

run_step 30 "Running KL feature baselines" \
  "${ROOT_DIR}/scripts/run_kl_feature_baselines.py"

run_step 31 "Running KL ablation studies" \
  "${ROOT_DIR}/scripts/run_ablation.py" \
  +model=xgboost

run_step 32 "Evaluating end-to-end pipeline" \
  "${ROOT_DIR}/scripts/evaluate_pipeline.py" \
  +model=xgboost

run_step 33 "Building reproducibility report" \
  "${ROOT_DIR}/scripts/build_repro_report.py" \
  --run-root "${ROOT_DIR}"

run_step 34 "Auditing pipeline readiness (post-run)" \
  "${ROOT_DIR}/scripts/check_pipeline_readiness.py" \
  --project-root "${ROOT_DIR}"

echo "Main-study pipeline complete."
echo "Project root: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Data root: ${DATA_ROOT}"
echo "Osteophyte ROI dir: ${OSTEOPHYTE_REPRO_ROI_DIR}"
echo "Sclerosis manual dir: ${SCLEROSIS_MANUAL_DIR}"
echo "Sclerosis expanded dir: ${SCLEROSIS_EXPANDED_DIR}"
echo "Sclerosis final dir: ${SCLEROSIS_FINAL_DIR}"
echo "Osteophyte results: ${OSTEOPHYTE_RESULTS_DIR}"
echo "Reproducibility report: ${ROOT_DIR}/report"
echo "Results root: ${ROOT_DIR}/results"
