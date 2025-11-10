#!/bin/bash

# "blood" "boatship" "capsule" "computer" "foodcube" "foodtray" "gumball" "hotfood" "smallbuild"
# 定义场景列表
SCENES=("boatship")

# 数据集根目录
DATASET_DIR=~/dataset/trans_light

# 评估输出目录
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

# python odata_eval/eval.py  --input_path ./eval_translight/blood/recon_opa.ply --gt_path ~/dataset/trans_light/blood/blood.ply 

python odata_eval/eval.py  --input_path ./eval_translight/boatship/recon.ply --gt_path ~/dataset/trans_light/boatship/boatship.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/capsule/recon_opa.ply --gt_path ~/dataset/trans_light/capsule/capsule.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/computer/recon_opa.ply --gt_path ~/dataset/trans_light/computer/computer.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/foodtray/recon_opa.ply --gt_path ~/dataset/trans_light/foodtray/foodtray.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/gumball/recon_opa.ply --gt_path ~/dataset/trans_light/gumball/gumball.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/hotfood/recon_opa.ply --gt_path ~/dataset/trans_light/hotfood/hotfood.ply 

# python odata_eval/eval.py  --input_path ./eval_translight/smallbuild/recon_opa.ply --gt_path ~/dataset/trans_light/smallbuild/smallbuild.ply 
CUDA_VISIBLE_DEVICES=1 python mesh_extract_opa_hotfix.py -s eval_dtu/scan24/ -m ~/dataset/DTU/scan24/