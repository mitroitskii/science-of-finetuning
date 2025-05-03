#!/bin/bash

# NOTE: couldn't run this time because I cached my activations without the BOS token; make sure to cache with BOS for this to work

BASE_MODEL="meta-llama/Llama-3.1-8B"
REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER=15
DICTIONARY_MODEL="/disk/u/troitskiid/data/checkpoints/L1-Crosscoder/L${LAYER}R/cc_weights.pt"
ACTIVATION_STORE_DIR="/disk/u/troitskiid/data/activations"
LATENT_ACTIVATIONS_DIR="/disk/u/troitskiidf/data/latent_activations/"

python scripts/compute_latent_activations_reasoning.py \
    --base-model $BASE_MODEL \
    --ft-model $REASONING_MODEL \
    --layer $LAYER \
    --dictionary-model $DICTIONARY_MODEL \
    --activation-store-dir $ACTIVATION_STORE_DIR \
    --latent-activations-dir $LATENT_ACTIVATIONS_DIR \
    --split validation