#!/bin/bash

set -x
# Prepare repository and environment
cd /mnt/bn/tiktok-mm-5/aiic/users/tianyu/MagicData
pip install -r requirements.txt

export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_gpt.yaml --split yimi --mode three_shot_same_test --model_name gpt-4o-2024-11-20  --output_dir /mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data --batch_size 1 --use_accel