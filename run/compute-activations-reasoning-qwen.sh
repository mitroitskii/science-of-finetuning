#! /bin/bash

set -x

# Define datasets and other constants
DATASET=koyena/OpenR1-Math-220k-formatted
ACTIVATION_STORE_DIR=~/data/activations
REASONING_MODEL=agentica-org/DeepScaleR-1.5B-Preview
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B
CONTEXT_LEN=5000
LAYERS=15
SHARD_SIZE=5_000_000
# Initialize variables for command-line arguments
BATCH_SIZE=1
SPLIT_ARG=""
OTHER_FLAGS=""
INPUT_COLUMN=message_in_chat_template
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
        --input-column)
            INPUT_COLUMN="$2"
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --split <train|test> [--reasoning-only] [--base-only]"
            exit 1
            ;;
    esac
done

# Validate that both arguments were supplied
if [ -z "$SPLIT_ARG" ]; then
    echo "Usage: $0 --split <train|test>"
    exit 1
fi

# Validate split argument and set corresponding values
if [ "$SPLIT_ARG" == "train" ]; then
    SPLIT="train"
    N_TOKS=200_000_000
elif [ "$SPLIT_ARG" == "test" ]; then
    SPLIT="test"
    N_TOKS=20_000_000
else
    echo "Error: --split must be either 'train' or 'test'"
    exit 1
fi

# Build common flags using the updated variables
COMMON_FLAGS="--wandb --overwrite 
--wandb-entity $WANDB_ENTITY \
--dtype float32 \
--disable-multiprocessing \
--store-tokens \
--batch-size $BATCH_SIZE \
--context-len $CONTEXT_LEN \
--layers $LAYERS \
--dataset $DATASET \
--dataset-split $SPLIT \
--shard-size $SHARD_SIZE \
--activation-store-dir $ACTIVATION_STORE_DIR \
--text-column $INPUT_COLUMN \
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
