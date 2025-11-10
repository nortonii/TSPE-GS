#!/bin/bash

# "computer" "computer" "computer" "computer" "computer" "computer" "computer" "computer" "computer"

SCENES=("computer" "foodtray" "hotfood" "bottleship" "blood" "capsule" "gumball" "smallbuild")

DATASET_DIR=~/dataset/trans_light

OUTPUT_DIR=./eval_translight

for scene in "${SCENES[@]}"; do
    echo "==============================="
    echo "Training scene: $scene"
    echo "==============================="

    SCENE_INPUT="$DATASET_DIR/$scene"
    SCENE_OUTPUT="$OUTPUT_DIR/$scene"

    mkdir -p "$SCENE_OUTPUT"

    # CUDA_VISIBLE_DEVICES=2  python train.py -s "$SCENE_INPUT" -m "$SCENE_OUTPUT" \
                                # --use_decoupled_appearance

    # python render.py -s "$SCENE_INPUT" -m "$SCENE_OUTPUT"
    CUDA_VISIBLE_DEVICES=1 python mesh_extract_opa_hotfix.py -s "$SCENE_INPUT" -m "$SCENE_OUTPUT"
    echo "Finished training: $scene"
    echo
done