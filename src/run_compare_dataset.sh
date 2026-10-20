\
#!/usr/bin/env bash
set -euo pipefail

# Orchestrator: run OUR method and PAPER baseline on a directory of scenes, then evaluate both.
#
# Defaults assume your layout:
#   ROOT is inferred from this script location (SATR/src/run_compare_dataset.sh)
#   code in $ROOT/src
#   ours scenes in $ROOT/json (bbox=[y0,y1,x0,x1])
#   paper scenes in $ROOT/json_paper (bbox=[x0,x1,y0,y1])
#
# Usage:
#   bash run_compare_dataset.sh SAM_CKPT
#
# Optional env overrides:
#   OCCLUSION_ROOT           (default: auto-detected project root)
#   JSON_DIR                 (default: $ROOT/json)
#   JSON_PAPER_DIR           (default: $ROOT/json_paper)
#   OUT_OURS                 (default: $ROOT/outputs/ours_ds)
#   OUT_PAPER                (default: $ROOT/outputs/paper_ds)
#   EVAL_OUT                 (default: $ROOT/outputs/eval_ds)
#   H, W                     (default: 1024, 1024)
#   OCCLUSION_K              (default: 16)
#   PAPER_NUM_IMAGES         (default: 16)
#   LARENDER_BATCH_SIZE      (default: 2)
#   REORDER_BY_PAIRS         (default: 1)  # for json->json_paper conversion
#   FORCE_CONVERT            (default: 0)  # force re-generate json_paper
#   SKIP_DONE_OURS           (default: 1)  # skip ours scene if refine_best already exists
#   SKIP_DONE_PAPER          (default: 1)  # run_paper script already skips by scene_used.json
#
# Files expected under $ROOT/src:
#   - demos_occlusion_sam_attn.py            (ours demo)
#   - demos_occlusion_batch.py               (paper baseline demo)
#   - eval_occlusion_v24_structv3.py         (released evaluator with bbox_format/touch/struct)
#   - convert_json_bbox_to_paper.py          (provided converter)
#   - run_paper_json_dir.sh                  (provided runner for paper baseline)
#
# Notes:
# - To use another evaluator, set EVAL_PY to its path.
# - This script evaluates:
#     ours:     refine_*.jpg  and refine_best.jpg
#     paper:    *.png
#   and writes per-scene mean/min summaries + a merged comparison CSV.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 SAM_CKPT"
  exit 1
fi
SAM_CKPT="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OCCLUSION_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SRC="${ROOT}/src"

JSON_DIR="${JSON_DIR:-${ROOT}/json}"
JSON_PAPER_DIR="${JSON_PAPER_DIR:-${ROOT}/json_paper}"

OUT_OURS="${OUT_OURS:-${ROOT}/outputs/ours_ds}"
OUT_PAPER="${OUT_PAPER:-${ROOT}/outputs/paper_ds}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/outputs/eval_ds}"

H="${H:-1024}"
W="${W:-1024}"

export OCCLUSION_K="${OCCLUSION_K:-16}"
export PAPER_NUM_IMAGES="${PAPER_NUM_IMAGES:-16}"
export LARENDER_BATCH_SIZE="${LARENDER_BATCH_SIZE:-2}"

REORDER_BY_PAIRS="${REORDER_BY_PAIRS:-1}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
SKIP_DONE_OURS="${SKIP_DONE_OURS:-1}"

# Locate scripts
EVAL_PY="${EVAL_PY:-${SRC}/eval_occlusion_v24_structv3.py}"
CONVERT_PY="${CONVERT_PY:-${SRC}/convert_json_bbox_to_paper.py}"
RUN_PAPER_SH="${RUN_PAPER_SH:-${SRC}/run_paper_json_dir.sh}"
OURS_DEMO_PY="${OURS_DEMO_PY:-${SRC}/demos_occlusion_sam_attn.py}"

if [[ ! -d "${JSON_DIR}" ]]; then
  echo "[ERROR] JSON_DIR not found: ${JSON_DIR}"
  exit 1
fi
mkdir -p "${OUT_OURS}" "${OUT_PAPER}" "${EVAL_OUT}"

if [[ ! -f "${OURS_DEMO_PY}" ]]; then
  echo "[ERROR] OURS_DEMO_PY not found: ${OURS_DEMO_PY}"
  exit 1
fi
if [[ ! -f "${RUN_PAPER_SH}" ]]; then
  echo "[ERROR] RUN_PAPER_SH not found: ${RUN_PAPER_SH}"
  exit 1
fi
if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "[ERROR] CONVERT_PY not found: ${CONVERT_PY}"
  exit 1
fi
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] EVAL_PY not found: ${EVAL_PY}"
  echo "  Expected eval_occlusion_v24_structv3.py in ${SRC}, or set EVAL_PY=/path/to/another_evaluator.py"
  exit 1
fi

echo "[INFO] ROOT=${ROOT}"
echo "[INFO] JSON_DIR=${JSON_DIR}"
echo "[INFO] JSON_PAPER_DIR=${JSON_PAPER_DIR}"
echo "[INFO] OUT_OURS=${OUT_OURS}"
echo "[INFO] OUT_PAPER=${OUT_PAPER}"
echo "[INFO] EVAL_OUT=${EVAL_OUT}"
echo "[INFO] H=${H}, W=${W}"
echo "[INFO] OCCLUSION_K=${OCCLUSION_K}, PAPER_NUM_IMAGES=${PAPER_NUM_IMAGES}"
echo "[INFO] LARENDER_BATCH_SIZE=${LARENDER_BATCH_SIZE}"

# 1) Convert json -> json_paper (bbox swap) if needed
need_convert=0
if [[ "${FORCE_CONVERT}" == "1" ]]; then
  need_convert=1
elif [[ ! -d "${JSON_PAPER_DIR}" ]]; then
  need_convert=1
else
  # If counts mismatch, convert again
  n_src="$(ls -1 "${JSON_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')"
  n_paper="$(ls -1 "${JSON_PAPER_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${n_paper}" -lt "${n_src}" ]]; then
    need_convert=1
  fi
fi

if [[ "${need_convert}" == "1" ]]; then
  echo -e "\n=== [1/4] Converting scenes to paper bbox format ==="
  conv_args=( "--in_dir" "${JSON_DIR}" "--out_dir" "${JSON_PAPER_DIR}" "--overwrite" )
  if [[ "${REORDER_BY_PAIRS}" == "1" ]]; then
    conv_args+=( "--reorder_by_pairs" )
  fi
  python "${CONVERT_PY}" "${conv_args[@]}"
else
  echo -e "\n=== [1/4] Convert skipped (json_paper seems up to date) ==="
fi

# 2) Run OUR method on JSON_DIR
echo -e "\n=== [2/4] Running OUR method on ${JSON_DIR} ==="
shopt -s nullglob
mapfile -t scenes < <(ls -1 "${JSON_DIR}"/*.json 2>/dev/null | sort)
if [[ ${#scenes[@]} -eq 0 ]]; then
  echo "[ERROR] No .json found in ${JSON_DIR}"
  exit 1
fi

for scene_path in "${scenes[@]}"; do
  stem="$(basename "${scene_path}" .json)"
  out_dir="${OUT_OURS}/${stem}"
  if [[ "${SKIP_DONE_OURS}" == "1" && -f "${out_dir}/refine_best.jpg" ]]; then
    echo "[SKIP] ours ${stem} (refine_best exists)"
    continue
  fi
  mkdir -p "${out_dir}"
  echo -e "\n--- OURS: ${stem} ---"
  OCCLUSION_WIDTH="${W}" OCCLUSION_HEIGHT="${H}" python "${OURS_DEMO_PY}" --scene "${scene_path}" --out "${out_dir}"
done

# 3) Run PAPER baseline on JSON_PAPER_DIR
echo -e "\n=== [3/4] Running PAPER baseline on ${JSON_PAPER_DIR} ==="
bash "${RUN_PAPER_SH}" "${JSON_PAPER_DIR}" "${H}" "${W}" "${OUT_PAPER}" "${PAPER_NUM_IMAGES}"

# 4) Evaluate both
echo -e "\n=== [4/4] Evaluating outputs (scalar50) ==="
pushd "${EVAL_OUT}" >/dev/null

# Optional touch/struct weights: keep 0 by default for compatibility, enable via env.
W_TOUCH="${W_TOUCH:-0.0}"
W_STRUCT="${W_STRUCT:-0.0}"
TOUCH_MAX_DEFAULT="${TOUCH_MAX_DEFAULT:-0.06}"
TOUCH_DILATE_K="${TOUCH_DILATE_K:-11}"
HOLES_MIN_DEFAULT="${HOLES_MIN_DEFAULT:-1}"

common_eval_args=( "--sam_ckpt" "${SAM_CKPT}" "--scalar_mode" "scalar50" )

echo "[INFO] Eval extras: W_TOUCH=${W_TOUCH}, W_STRUCT=${W_STRUCT}"
extra_eval_args=()
# The released v24 evaluator supports these arguments.
extra_eval_args+=( "--w_touch" "${W_TOUCH}" "--touch_max_default" "${TOUCH_MAX_DEFAULT}" "--touch_dilate_k" "${TOUCH_DILATE_K}" )
extra_eval_args+=( "--w_struct" "${W_STRUCT}" "--holes_min_default" "${HOLES_MIN_DEFAULT}" )

# OUR candidates
python "${EVAL_PY}" \
  --images_root "${OUT_OURS}" \
  --scenes_dir "${JSON_DIR}" \
  --bbox_format "y1y2x1x2" \
  --image_glob "refine_*.jpg" \
  --out_csv "ours_candidates.csv" \
  --out_json "ours_candidates.json" \
  "${common_eval_args[@]}" \
  "${extra_eval_args[@]}"

# OUR refine_best
python "${EVAL_PY}" \
  --images_root "${OUT_OURS}" \
  --scenes_dir "${JSON_DIR}" \
  --bbox_format "y1y2x1x2" \
  --image_glob "refine_best.jpg" \
  --out_csv "ours_best.csv" \
  --out_json "ours_best.json" \
  "${common_eval_args[@]}" \
  "${extra_eval_args[@]}"

# PAPER candidates (*.png)
python "${EVAL_PY}" \
  --images_root "${OUT_PAPER}" \
  --scenes_dir "${JSON_PAPER_DIR}" \
  --bbox_format "x0x1y0y1" \
  --image_glob "*.png" \
  --out_csv "paper_candidates.csv" \
  --out_json "paper_candidates.json" \
  "${common_eval_args[@]}" \
  "${extra_eval_args[@]}"

# Summaries (per-scene mean/min) and merged comparison
python - <<'PY'
import pandas as pd

def summarize(path: str, scalar_col="scalar"):
    df = pd.read_csv(path)
    # guard
    if "scene" not in df.columns:
        raise ValueError(f"Missing 'scene' column in {path}")
    if scalar_col not in df.columns:
        # fallback: pick first numeric col
        scalar_col = [c for c in df.columns if c.startswith("scalar")][0]
    g = df.groupby("scene", as_index=False).agg(
        n=("image","count"),
        mean_scalar=(scalar_col,"mean"),
        min_scalar=(scalar_col,"min"),
        mean_pair_pen=("pair_pen_mean","mean") if "pair_pen_mean" in df.columns else (scalar_col,"mean"),
        mean_quality=("quality_pen","mean") if "quality_pen" in df.columns else (scalar_col,"mean"),
        mean_depth=("depth_order_acc","mean") if "depth_order_acc" in df.columns else (scalar_col,"mean"),
    )
    return g

ours_c = summarize("ours_candidates.csv")
ours_b = summarize("ours_best.csv")
paper_c = summarize("paper_candidates.csv")

ours_c.to_csv("summary_ours_candidates.csv", index=False)
ours_b.to_csv("summary_ours_best.csv", index=False)
paper_c.to_csv("summary_paper_candidates.csv", index=False)

# merged compare: use paper min as best-of-16 baseline
m = ours_b.merge(paper_c[["scene","mean_scalar","min_scalar"]], on="scene", how="outer", suffixes=("_ours_best","_paper"))
m = m.rename(columns={"mean_scalar_paper":"paper_mean", "min_scalar_paper":"paper_min"})
# also include ours candidate min (true best-of-K) for analysis
m2 = m.merge(ours_c[["scene","min_scalar"]], on="scene", how="left")
m2 = m2.rename(columns={"min_scalar":"ours_min"})
m2.to_csv("compare_ours_vs_paper.csv", index=False)

print("[OK] Wrote summaries:")
print("  summary_ours_candidates.csv")
print("  summary_ours_best.csv")
print("  summary_paper_candidates.csv")
print("  compare_ours_vs_paper.csv")
PY

popd >/dev/null

echo -e "\nAll done."
echo " - OUR outputs:   ${OUT_OURS}"
echo " - PAPER outputs: ${OUT_PAPER}"
echo " - EVAL outputs:  ${EVAL_OUT}"
echo "Key file: ${EVAL_OUT}/compare_ours_vs_paper.csv"
