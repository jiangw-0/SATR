\
#!/usr/bin/env bash
set -euo pipefail

# Run all scene JSON files in a directory.
#
# Usage:
#   bash run_dataset_json_dir.sh JSON_DIR OUT_DIR
#   bash run_dataset_json_dir.sh JSON_DIR WIDTH HEIGHT OUT_DIR
#
# Examples:
#   bash run_dataset_json_dir.sh /media/jiang/ssd/zly/Occlusion/src/json /media/jiang/ssd/zly/Occlusion/outputs/ds500
#   bash run_dataset_json_dir.sh /media/jiang/ssd/zly/Occlusion/src/json 1024 1024 /media/jiang/ssd/zly/Occlusion/outputs/ds500
#
# Notes:
# - Forces best-of-16 by default via OCCLUSION_K=16 (override with env OCCLUSION_K).
# - Uses demos_occlusion_sam_attn.py in the same folder as this script, otherwise falls back to $ROOT/src.

if [[ $# -ge 4 && "$2" =~ ^[0-9]+$ && "$3" =~ ^[0-9]+$ ]]; then
  JSON_DIR="$1"
  export OCCLUSION_WIDTH="$2"
  export OCCLUSION_HEIGHT="$3"
  OUT_ROOT="$4"
else
  JSON_DIR="${1:-}"
  OUT_ROOT="${2:-}"
fi

if [[ -z "${JSON_DIR}" || -z "${OUT_ROOT}" ]]; then
  echo "Usage: bash run_dataset_json_dir.sh JSON_DIR [WIDTH HEIGHT] OUT_DIR"
  exit 1
fi

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "[ERROR] JSON_DIR does not exist: ${JSON_DIR}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${OCCLUSION_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

DEMO_PY="${SCRIPT_DIR}/demos_occlusion_sam_attn.py"
if [[ ! -f "${DEMO_PY}" ]]; then
  DEMO_PY="${ROOT}/src/demos_occlusion_sam_attn.py"
fi
if [[ ! -f "${DEMO_PY}" ]]; then
  echo "[ERROR] Cannot find demos_occlusion_sam_attn.py"
  echo "  Tried: ${SCRIPT_DIR}/demos_occlusion_sam_attn.py"
  echo "  Tried: ${ROOT}/src/demos_occlusion_sam_attn.py"
  exit 1
fi

# Force fair best-of-K by default
export OCCLUSION_K="${OCCLUSION_K:-16}"

# Allow natural ordering (001, 002, ..., 100) if your filenames are zero-padded.
shopt -s nullglob
mapfile -t scenes < <(ls -1 "${JSON_DIR}"/*.json 2>/dev/null | sort)
if [[ ${#scenes[@]} -eq 0 ]]; then
  echo "[ERROR] No .json files found in: ${JSON_DIR}"
  exit 1
fi

echo "[INFO] JSON_DIR=${JSON_DIR}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] DEMO_PY=${DEMO_PY}"
echo "[INFO] OCCLUSION_K=${OCCLUSION_K}"
if [[ -n "${OCCLUSION_WIDTH:-}" && -n "${OCCLUSION_HEIGHT:-}" ]]; then
  echo "[INFO] OCCLUSION_WIDTH=${OCCLUSION_WIDTH}, OCCLUSION_HEIGHT=${OCCLUSION_HEIGHT}"
fi

for scene_path in "${scenes[@]}"; do
  s="$(basename "${scene_path}")"
  name="${s%.json}"
  out_dir="${OUT_ROOT}/${name}"
  # Safe to resume: skip if already done (set FORCE_RERUN=1 to disable)
  if [[ "${FORCE_RERUN:-0}" != "1" ]]; then
    if [[ -f "${out_dir}/scene_run_summary.json" || -f "${out_dir}/refine_best.jpg" || -f "${out_dir}/refine_best.png" ]]; then
      echo "[SKIP] ${name} (already done)"
      continue
    fi
  fi
  echo -e "\n=== Running ${s} -> ${out_dir} ==="
  python "${DEMO_PY}" --scene "${scene_path}" --out "${out_dir}"
done

echo -e "\nDone. Outputs in: ${OUT_ROOT}"
