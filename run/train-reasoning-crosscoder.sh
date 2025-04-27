#! /bin/bash

set -x

DATASTORE="$HOME/data"
ACTIVATION_DIR="$DATASTORE/activations/"

BASE_MODEL="meta-llama/Llama-3.1-8B"
REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# Model configuration
NUM_TOKENS=400_000_000
NUM_VALIDATION_TOKENS=1_000_000
LAYER=15
EXPANSION_FACTOR=32 
 # for 8B model
BATCH_SIZE=2048
MU=3.6e-2
LR=1e-4
EPOCHS=1
VALIDATE_EVERY_N_STEPS=10_000 # total steps for len 400_000_000 and bs 2048 is ~195000
DEVICE="cuda" # Default device

# Parse command line arguments to check for custom mu value and layer
custom_mu=false
custom_layer=false
custom_device=false
custom_batch_size=false
for arg in "$@"; do
    if [[ $arg == --mu* ]]; then
        custom_mu=true
    fi
    if [[ $arg == --layer* ]]; then
        custom_layer=true
    fi
    if [[ $arg == --device* ]]; then
        custom_device=true
    fi
    if [[ $arg == --batch-size* ]]; then
        custom_batch_size=true
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--type batch-top-k \
--base-model $BASE_MODEL \
--reasoning-model $REASONING_MODEL \
--text-column message_llama_chat_template \
--layer $LAYER \
--lr $LR \
--validate-every-n-steps $VALIDATE_EVERY_N_STEPS \
--epochs $EPOCHS \
--num-tokens $NUM_TOKENS \
--num-validation-tokens $NUM_VALIDATION_TOKENS \
--same-init-for-all-layers \
--init-with-transpose \
--local-shuffling"

# Only add default mu if not provided in command line arguments
if [ "$custom_mu" = false ]; then
    FLAGS="$FLAGS --mu $MU"
fi

# Only add default device if not provided in command line arguments
if [ "$custom_device" = false ]; then
    FLAGS="$FLAGS --device $DEVICE"
fi

# Only add default batch size if not provided in command line arguments
if [ "$custom_batch_size" = false ]; then
    FLAGS="$FLAGS --batch-size $BATCH_SIZE"
fi

additional_flags=$@

python scripts/train_reasoning_crosscoder.py $FLAGS $additional_flags