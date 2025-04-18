#! /bin/bash

set -x

DATASTORE="$HOME/data"
ACTIVATION_DIR="$DATASTORE/activations/"

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
REASONING_MODEL="agentic/DeepScaleR-1.5B-Preview"

# Model configuration
NUM_SAMPLES=300_000_000
NUM_VALIDATION_SAMPLES=10_000_000
LAYER=15
EXPANSION_FACTOR=32
BATCH_SIZE=2048
MU=3.6e-2
LR=1e-4
EPOCHS=2
VALIDATE_EVERY_N_STEPS=15_000
# Parse command line arguments to check for custom mu value
custom_mu=false
for arg in "$@"; do
    if [[ $arg == --mu* ]]; then
        custom_mu=true
        break
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--type batch-top-k \
--base-model $BASE_MODEL \
--reasoning-model $REASONING_MODEL \
--text-column message_in_chat_template \
--layer $LAYER \
--batch-size $BATCH_SIZE \
--lr $LR \
--validate-every-n-steps $VALIDATE_EVERY_N_STEPS \
--epochs $EPOCHS \
--num-samples $NUM_SAMPLES \
--num-validation-samples $NUM_VALIDATION_SAMPLES \
--same-init-for-all-layers \
--init-with-transpose \
--local-shuffling"

# Only add default mu if not provided in command line arguments
if [ "$custom_mu" = false ]; then
    FLAGS="$FLAGS --mu $MU"
fi

additional_flags=$@

python scripts/train_reasoning_crosscoder.py $FLAGS $additional_flags