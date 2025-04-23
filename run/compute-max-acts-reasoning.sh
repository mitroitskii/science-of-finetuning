#! /bin/bash

ACTIVATIONS_PATH="/share/u/troitskiid/data/activations"
CROSSCODER_PATH="/share/u/troitskiid/projects/science-of-finetuning/checkpoints/DeepScaleR-1.5B-crosscoder-L15-k100-lr1e-04-local-shuffling-CCLoss/checkpoint_90000.pt"
BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
REASONING_MODEL="agentica-org/DeepScaleR-1.5B-Preview"

python scripts/compute_max_activation_reasoning.py \
    --activation-cache-path $ACTIVATIONS_PATH \
    --dataset lmsys-chat-1m-chat-formatted \
    --crosscoder $CROSSCODER_PATH \
    --base-model $BASE_MODEL \
    --reasoning-model $REASONING_MODEL \
    --layer 15 \
    --batch-size 2048 \
    --num-workers 16 \
    --device cuda:1 


python scripts/compute_max_activation_reasoning.py \
    --activation-cache-path $ACTIVATIONS_PATH \
    --dataset fineweb-1m-sample \
    --crosscoder $CROSSCODER_PATH \
    --base-model $BASE_MODEL \
    --reasoning-model $REASONING_MODEL \
    --layer 15 \
    --batch-size 2048 \
    --num-workers 16 \
    --device cuda:1 
