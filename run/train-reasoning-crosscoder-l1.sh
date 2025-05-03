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
BATCH_SIZE=2048
MU=4e-2
LR=1e-4
EPOCHS=1
VALIDATE_EVERY_N_STEPS=10_000 # total steps for len 400_000_000 and bs 2048 is ~195000
DEVICE="cuda" # Default device
RUN_NAME="" # Default run name (empty)

# Parse command line arguments to check for custom mu value, layer, expansion factor, and other parameters
custom_mu=false
custom_layer=false
custom_device=false
custom_batch_size=false
custom_lr=false
custom_expansion_factor=false
custom_run_name=false
custom_validate_every_n_steps=false
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
    if [[ $arg == --lr* ]]; then
        custom_lr=true
    fi
    if [[ $arg == --exp* ]]; then
        custom_expansion_factor=true
    fi
    if [[ $arg == --run-name* ]]; then
        custom_run_name=true
    fi
    if [[ $arg == --validate-every-n-steps* ]]; then
        custom_validate_every_n_steps=true
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--type relu \
--base-model $BASE_MODEL \
--reasoning-model $REASONING_MODEL \
--text-column message_llama_chat_template \
--layer $LAYER \
--epochs $EPOCHS \
--num-tokens $NUM_TOKENS \
--num-validation-tokens $NUM_VALIDATION_TOKENS \
--validate-every-n-steps $VALIDATE_EVERY_N_STEPS \
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

# Only add default learning rate if not provided in command line arguments
if [ "$custom_lr" = false ]; then
    FLAGS="$FLAGS --lr $LR"
fi

# Only add default expansion factor if not provided in command line arguments
if [ "$custom_expansion_factor" = false ]; then
    FLAGS="$FLAGS --expansion-factor $EXPANSION_FACTOR"
fi

# Only add default run name if not provided in command line arguments and not empty
if [ "$custom_run_name" = false ] && [ ! -z "$RUN_NAME" ]; then
    FLAGS="$FLAGS --run-name $RUN_NAME"
fi

# Only add default validate every n steps if not provided in command line arguments
if [ "$custom_validate_every_n_steps" = false ]; then
    FLAGS="$FLAGS --validate-every-n-steps $VALIDATE_EVERY_N_STEPS"
fi

additional_flags=$@

python scripts/train_reasoning_crosscoder.py $FLAGS $additional_flags