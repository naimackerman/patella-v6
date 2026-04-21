#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

# Load repo-local secrets and overrides such as WANDB_API_KEY or DATA_ROOT.
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/load_env.sh"
load_repo_env "${ROOT_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python at ${PYTHON_BIN}" >&2
  exit 1
fi

export PROJECT_ROOT="${ROOT_DIR}"
export DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/KneeXrayData}"
export PYTHONPATH="${ROOT_DIR}"

DATA_DIR="${DATA_ROOT}/ClsKLData/kneeKL224"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing dataset directory: ${DATA_DIR}" >&2
  exit 1
fi

echo "[1/6] Building bootstrap suggestions and aggregated features"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/bootstrap_pseudo_labels.py"

echo "[2/6] Refreshing annotation manifests"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/refresh_annotation_manifests.py"

echo "[3/6] Preparing annotation workspace packages"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_annotation_workspace.py"

echo "[4/6] Creating provisional non-clinician review sheet"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prefill_feature_review.py"

echo "[5/6] Evaluating geometric bootstrap baseline"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_xgboost_baseline.py" \
  +model=xgboost \
  result_dir="${ROOT_DIR}/results_geometric_eval"

echo "[6/6] Evaluating landmark bootstrap baseline"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_landmark_xgboost_baseline.py" \
  +model=xgboost \
  feature_dir="${ROOT_DIR}/features_landmark_fast" \
  result_dir="${ROOT_DIR}/results_landmark_fast"

echo
echo "Initial experiment complete."
echo "Geometric summary: ${ROOT_DIR}/results_geometric_eval/xgboost_baseline_summary.json"
echo "Landmark summary:  ${ROOT_DIR}/results_landmark_fast/landmark_xgboost_summary.json"
echo "Provisional review: ${ROOT_DIR}/annotations/packages/feature_grading/feature_review_provisional.csv"
