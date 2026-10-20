#!/usr/bin/env bash
set -euo pipefail

# Run paper baseline (demos_occlusion_batch.py) for ALL scene json files in a directory.
#
# Usage:
#   bash run_paper_json_dir.sh JSON_DIR OUT_ROOT
#   bash run_paper_json_dir.sh JSON_DIR H W OUT_ROOT
#   bash run_paper_json_dir.sh JSON_DIR H W OUT_ROOT NUM_IMAGES
#
# Examples:
#   bash run_paper_json_dir.sh /media/jiang/ssd/zly/Occlusion/src/json_paper /media/jiang/ssd/zly/Occlusion/outputs/paper_ds500
#   bash run_paper_json_dir.sh /media/jiang/ssd/zly/Occlusion/src/json_paper 1024 1024 /media/jiang/ssd/zly/Occlusion/outputs/paper_ds500 16
#
# Notes:
# - This runner expects paper-format bboxes: [x0,x1,y0,y1] (convert your dataset first).
# - By default uses NUM_IMAGES=16 (override with PAPER_NUM_IMAGES env or last arg).
# - Safe to resume: it skips a scene if OUT_ROOT/<scene_stem>/scene_used.json exists.

if [[ $# -eq 2 ]]; then
  JSON_DIR="$1"
  OUT_ROOT="$2"
  H=""
  W=""
  NUM_IMAGES="${PAPER_NUM_IMAGES:-16}"
elif [[ $# -eq 4 ]]; then
  JSON_DIR="$1"
  H="$2"
  W="$3"
  OUT_ROOT="$4"
  NUM_IMAGES="${PAPER_NUM_IMAGES:-16}"
elif [[ $# -eq 5 ]]; then
  JSON_DIR="$1"
  H="$2"
  W="$3"
  OUT_ROOT="$4"
  NUM_IMAGES="$5"
else
  echo "Usage: bash $0 JSON_DIR OUT_ROOT"
  echo "   or: bash $0 JSON_DIR H W OUT_ROOT [NUM_IMAGES]"
  echo "   or: PAPER_NUM_IMAGES=16 bash $0 JSON_DIR H W OUT_ROOT"
  exit 1
fi

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "[ERROR] JSON_DIR does not exist: ${JSON_DIR}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${OCCLUSION_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Try to locate demos_occlusion_batch.py
BATCH_PY="${SCRIPT_DIR}/demos_occlusion_batch.py"
if [[ ! -f "${BATCH_PY}" ]]; then
  BATCH_PY="${ROOT}/src/demos_occlusion_batch.py"
fi
if [[ ! -f "${BATCH_PY}" ]]; then
  echo "[ERROR] Cannot find demos_occlusion_batch.py"
  echo "  Tried: ${SCRIPT_DIR}/demos_occlusion_batch.py"
  echo "  Tried: ${ROOT}/src/demos_occlusion_batch.py"
  exit 1
fi

# Reduce memory if needed
export LARENDER_BATCH_SIZE="${LARENDER_BATCH_SIZE:-2}"

shopt -s nullglob
mapfile -t scenes < <(ls -1 "${JSON_DIR}"/*.json 2>/dev/null | sort)
if [[ ${#scenes[@]} -eq 0 ]]; then
  echo "[ERROR] No .json files found in: ${JSON_DIR}"
  exit 1
fi

echo "[INFO] JSON_DIR=${JSON_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] BATCH_PY=${BATCH_PY}"
echo "[INFO] NUM_IMAGES=${NUM_IMAGES}"
echo "[INFO] LARENDER_BATCH_SIZE=${LARENDER_BATCH_SIZE}"

ARGS_COMMON=( "--disable_dp" "--num_images" "${NUM_IMAGES}" "--out_root" "${OUT_ROOT}" )
if [[ -n "${H}" ]]; then
  ARGS_COMMON+=( "--height" "${H}" )
fi
if [[ -n "${W}" ]]; then
  ARGS_COMMON+=( "--width" "${W}" )
fi

for scene_path in "${scenes[@]}"; do
  stem="$(basename "${scene_path}" .json)"
  out_dir="${OUT_ROOT}/${stem}"
  if [[ -f "${out_dir}/scene_used.json" ]]; then
    echo "[SKIP] ${stem} (already done)"
    continue
  fi
  echo -e "\n=== Paper baseline: ${stem} ==="
  python "${BATCH_PY}" --scene "${scene_path}" "${ARGS_COMMON[@]}"
done

echo -e "\nDone. Outputs in: ${OUT_ROOT}"
