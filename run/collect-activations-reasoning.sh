#! /bin/bash

set -x

# NOTE on batch size:
# The optimal batch size seems to be just 1 and ran on a single GPU
# - to check, ran with different sizes and see how long it takes to cache and store 1 share
# Notice, that total time to run is NOT AS PREDICTED BY THE PROGRESS BAR
# Instead, it is num_of_shards * time_per_shard

# NOTE ⚠️:
# Do NOT cache with different batch sizes for the same dataset
# This will lead to different shard sizes and different token counts


# NOTE on training several layers at once:
# tokens.pt is only saved for the last layer on the list
# need to manually copy the tokens.pt file to the other layer dirs


# Define dataset constants
FINEWEB_DATASET="science-of-finetuning/fineweb-1m-sample"
FINEWEB_INPUT_COLUMN="text"
REASONING_DATASET="koyena/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-formatted"
REASONING_INPUT_COLUMN="message_llama_chat_template"

# Default to fineweb
DATASET_TYPE="fineweb"
DATASET=$FINEWEB_DATASET
INPUT_COLUMN=$FINEWEB_INPUT_COLUMN

ACTIVATION_STORE_DIR=~/data/activations
REASONING_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
BASE_MODEL=meta-llama/Llama-3.1-8B
TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Llama-8B  
CONTEXT_LEN=5000
# LAYERS="5 11 17 23 29" # 32 total layers in llama-3.1-8b 0 - 5 - 11 - 17 - 23 - 29 - 31
LAYERS="7 15 23"
SHARD_SIZE=10_000_000 # 10 mins to store on disk # around 100GB per layer

# Initialize variables for command-line arguments
BATCH_SIZE=1
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
        --dataset-type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set dataset based on dataset type
if [ "$DATASET_TYPE" == "reasoning" ]; then
    DATASET=$REASONING_DATASET
    INPUT_COLUMN=$REASONING_INPUT_COLUMN
elif [ "$DATASET_TYPE" == "fineweb" ]; then
    DATASET=$FINEWEB_DATASET
    INPUT_COLUMN=$FINEWEB_INPUT_COLUMN
else
    echo "Error: Unknown dataset type '$DATASET_TYPE'. Must be 'fineweb' or 'reasoning'."
    exit 1
fi

# Validate that split argument was supplied
if [ -z "$SPLIT_ARG" ]; then
    echo "Usage: $0 --split <train|validation> [--dataset-type <fineweb|reasoning>] [--batch-size <batch-size>]"
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
--text-column $INPUT_COLUMN \
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
# else
#     # Run both models if no specific flag is set
#     python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL $OTHER_FLAGS
#     python scripts/collect_activations.py $COMMON_FLAGS --model $REASONING_MODEL $OTHER_FLAGS
fi