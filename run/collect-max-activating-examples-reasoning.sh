#!/bin/bash

CROSSCODER_PATH="/share/u/troitskiid/projects/science-of-finetuning/checkpoints/DeepScaleR-1.5B-crosscoder-L15-k100-lr1e-04-local-shuffling-CCLoss/checkpoint_90000.pt"
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
REASONING_MODEL="agentica-org/DeepScaleR-1.5B-Preview"

python scripts/collect_max_activating_examples_reasoning.py $CROSSCODER_PATH \
    --base-model $BASE_MODEL \
    --ft-model $REASONING_MODEL \
    --layer 15 \
    --cc-device cuda:1 \
    --base-device cuda:0 \
    --ft-device cuda:1 \
    --model-batch-size 1 \
    --crosscoder-batch-size 1024 \
    --workers 16 \
    --seq-len 5000 \
    --n 100 \