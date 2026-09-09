#!/bin/bash
# Reference w_avg(t) on the ImageNet latent mixture -- the production run.
#
# Query states are sharded across ranks; the CENTRES ARE REPLICATED, not sharded
# (1.31 GB of 140 GB), so no rank ever needs another rank's components and the
# only thing exchanged is one per-query vector per timepoint.
#
#   ./run_wavg_8xh200.sh                       # 8 GPUs, production config
#   NUM_GPUS=4 ./run_wavg_8xh200.sh            # override the rank count
#   CONFIG=configs/other.yaml ./run_wavg_8xh200.sh

set -euo pipefail

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate SiT
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/wavg_imagenet.yaml}"
# The config is the record of the run, so the GPU count comes from it unless the
# caller says otherwise -- not from whatever the node happens to have.
NUM_GPUS="${NUM_GPUS:-$(python -c "
import sys; sys.path.insert(0, '.')
from gtmf.config import Config
print(Config.load('$CONFIG').get('distributed.num_gpus', default=1))
")}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"   # avoid a stale port

echo "════════════════════════════════════════════════════════"
echo " Task        : reference w_avg(t)"
echo " Config      : $CONFIG"
echo " GPUs        : $NUM_GPUS"
echo " Master port : $MASTER_PORT"
echo " Started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"

torchrun --standalone --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    wavg.py --config "$CONFIG" "$@"
