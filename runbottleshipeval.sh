#!/bin/bash
set -e

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

    python odata_eval/eval.py --input_path "$SCENE_OUTPUT/recon_all.ply" --gt_path "$DATASET_DIR/$scene/$scene.ply"

    echo "Finished training: $scene"
    echo
done
# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_old.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 
#CUDA_VISIBLE_DEVICES=1 python mesh_extract_opa_hotfix.py -s eval_dtu/scan24/ -m ~/dataset/DTU/scan24/