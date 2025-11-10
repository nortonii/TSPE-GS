#!/bin/bash

DATASET_PATH="dataset/DTU"
OUTPUT_FOLDER="eval/"

scenes=(
  "scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69"
  "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122"
)

for scene in "${scenes[@]}"
do
  echo "Evaluating scene $scene"
  python evaluate_dtu_mesh.py -s "$DATASET_PATH/$scene" -m "$OUTPUT_FOLDER/$scene"
done