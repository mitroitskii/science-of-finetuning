#! /bin/bash

set -x

SUBSET_SIZE=50000

BASE_DATASET=science-of-finetuning/fineweb-1m-sample
REASONING_DATASET=koyena/OpenR1-Math-220k-formatted
BASE_SPLIT=validation
REASONING_SPLIT=test
BATCH_SIZE=1
MAX_LENGTH=5000

FLAGS="--crosscoder-path /disk/u/troitskiid/data/checkpoints/DeepScaleR-1.5B-crosscoder-L15-k100-lr1e-04-local-shuffling-CCLoss/ae.pt \
    --base-dataset $BASE_DATASET \
    --reasoning-dataset $REASONING_DATASET \
    --max-length $MAX_LENGTH \
    --dataset-base-split $BASE_SPLIT \
    --dataset-reasoning-split $REASONING_SPLIT \
    --batch-size $BATCH_SIZE \
    --layer 15 \
    --subset-size $SUBSET_SIZE \
    --base-model Qwen/Qwen2.5-Math-1.5B \
    --reasoning-model agentica-org/DeepScaleR-1.5B-Preview \
    --device cuda \
    --results-dir stats"

additional_flags=$@

python scripts/compute_latent_statistics_reasoning.py $FLAGS $additional_flags