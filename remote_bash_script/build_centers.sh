#!/bin/bash
# Merge the 1,281,167 per-image ImageNet latents into one flat fp16 array.
#
# CPU-only, no GPU needed. Writes into the latent directory itself:
#   centers_fp16.npy / centers_index.txt / centers_labels.npy / centers_meta.json
# and inserts a delimited section into that directory's README.md.
#
# Values are stored RAW -- the 0.18215 SD scale is applied at load time, not here.
#
#   ./build_centers.sh                  # production, all latents
#   ./build_centers.sh 50000            # development subset
#   WORKERS=32 ./build_centers.sh       # override worker count

set -euo pipefail

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd /scratch/project/prj-02-visual-ai/hkzhang/Ground-Truth-Marginal-Flow

LATENT_ROOT="${LATENT_ROOT:-/scratch/project/prj-02-visual-ai/hkzhang/ILSVRC/latents_8_mean_fp16/train}"
WORKERS="${WORKERS:-16}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-200}"
LIMIT="${1:-}"

echo "════════════════════════════════════════════════════════"
echo " Task        : build consolidated GMM centers array"
echo " Latent root : $LATENT_ROOT"
echo " Out dir     : $(dirname "$LATENT_ROOT")"
echo " Workers     : $WORKERS"
echo " Subset      : ${LIMIT:-all}"
echo "════════════════════════════════════════════════════════"

ARGS=(--latent-root "$LATENT_ROOT" --workers "$WORKERS" --verify-samples "$VERIFY_SAMPLES")
[ -n "$LIMIT" ] && ARGS+=(--limit "$LIMIT")

python build_centers.py "${ARGS[@]}"
