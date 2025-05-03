#!/bin/bash

# Default values
LAYER=15
MODEL_BATCH_SIZE=4
CROSCODER_BATCH_SIZE=8192
WORKERS=48
SEQ_LEN=5000
N=50
TOTAL_TOKENS=10_000_000
CHECKPOINT_EVERY=250

CC_DEVICE="cuda:7"
BASE_DEVICE="cuda:0"
FT_DEVICE="cuda:1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --layer)
      LAYER="$2"
      shift 2
      ;;
    --cc-device)
      CC_DEVICE="$2"
      shift 2
      ;;
    --base-device)
      BASE_DEVICE="$2"
      shift 2
      ;;
    --ft-device)
      FT_DEVICE="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

BASE_MODEL="meta-llama/Llama-3.1-8B"
REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
CROSSCODER_PATH="/disk/u/troitskiid/data/checkpoints/L1-Crosscoder/L${LAYER}R/cc_weights.pt"
# TODO: add dir for diffed features vs attributed features
SAVE_PATH="/disk/u/troitskiid/data/max_activating_examples/L1-Crosscoder"

python scripts/collect_max_activating_examples_reasoning.py $CROSSCODER_PATH \
    --base-model $BASE_MODEL \
    --ft-model $REASONING_MODEL \
    --layer $LAYER \
    --model-batch-size $MODEL_BATCH_SIZE \
    --crosscoder-batch-size $CROSCODER_BATCH_SIZE \
    --total-tokens $TOTAL_TOKENS \
    --seq-len $SEQ_LEN \
    --n $N \
    --checkpoint-every $CHECKPOINT_EVERY \
    --save-path $SAVE_PATH \
    --cc-device $CC_DEVICE \
    --base-device $BASE_DEVICE \
    --ft-device $FT_DEVICE \
    --workers $WORKERS \