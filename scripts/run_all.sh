#!/bin/bash
set -e

echo "Step 1: Data preparation"
python src/data_prep.py

echo "Step 2: Training"
python src/train.py --use_wandb --output_dir outputs/my_model --packing_train True

echo "Step 3: Evaluation (base vs fine‑tuned)"
python src/evaluate.py --adapter_path outputs/my_model

echo "Step 4: Merge and upload (optional)"
python src/merge_and_upload.py --adapter_path outputs/my_model --push --hub_repo_id your-username/your-model

echo "Step 5: Inference benchmark"
python src/inference_benchmark.py --model outputs/my_model --is_adapter

echo "All done!"