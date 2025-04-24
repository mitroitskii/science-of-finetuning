#! /bin/bash

set -x

# Define datasets and other constants
DATASET=koyena/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-formatted # reasoning 
# DATASET=science-of-finetuning/fineweb-1m-sample # fineweb
ACTIVATION_STORE_DIR=~/data/activations
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

BASE_MODEL=meta-llama/Llama-3.1-8B
TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Llama-8B  
CONTEXT_LEN=5000
LAYERS="5 11 17 23 29" # 32 total layers in llama-3.1-8b 0 - 5 - 11 - 17 - 23 - 29 - 31
SHARD_SIZE=1_000_000

# Initialize variables for command-line arguments
BATCH_SIZE=64
DEVICE_MAP="auto"
SPLIT_ARG="train"
OTHER_FLAGS=""
REASONING_ONLY=false
BASE_ONLY=false
WANDB_ENTITY=tentative

# Parse the command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --split)
            SPLIT_ARG="$2"
            shift 2
            ;;
        --wandb-entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --reasoning-only)
            REASONING_ONLY=true
            shift
            ;;
        --base-only)
            BASE_ONLY=true
            shift
            ;;
        --device-map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate that both arguments were supplied
if [ -z "$SPLIT_ARG" ]; then
    echo "Usage: $0 --split <train|validation> --batch-size <batch-size>"
    exit 1
fi

# Validate split argument and set corresponding values
if [ "$SPLIT_ARG" == "train" ]; then
    SPLIT="train"
    N_TOKS=200_000_000
elif [ "$SPLIT_ARG" == "validation" ]; then
    SPLIT="validation"
    N_TOKS=2_000_000
else
    echo "Error: --split must be either 'train' or 'validation'"
    exit 1
fi

# Build common flags using the updated variables
COMMON_FLAGS="--wandb --overwrite 
--tokenizer $TOKENIZER \
--wandb-entity $WANDB_ENTITY \
--device-map $DEVICE_MAP \
--disable-multiprocessing \
--dtype bfloat16 \
--store-tokens \
--batch-size $BATCH_SIZE \
--context-len $CONTEXT_LEN \
--layers $LAYERS \
--dataset $DATASET \
--dataset-split $SPLIT \
--shard-size $SHARD_SIZE \
--activation-store-dir $ACTIVATION_STORE_DIR \
--max-tokens $N_TOKS" 

 
# Run activation collection based on flags
if [ "$BASE_ONLY" = true ] && [ "$REASONING_ONLY" = true ]; then
    echo "Error: Cannot specify both --base-only and --reasoning-only"
    exit 1
fi

if [ "$BASE_ONLY" = true ]; then
    python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL $OTHER_FLAGS
elif [ "$REASONING_ONLY" = true ]; then
    python scripts/collect_activations.py $COMMON_FLAGS --model $REASONING_MODEL $OTHER_FLAGS
else
    # Run both models if no specific flag is set
    python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL $OTHER_FLAGS
    python scripts/collect_activations.py $COMMON_FLAGS --model $REASONING_MODEL $OTHER_FLAGS
fi
