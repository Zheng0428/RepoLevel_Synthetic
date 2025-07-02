#!/bin/bash

set -x
# Prepare repository and environment
cd /mnt/bn/tiktok-mm-5/aiic/users/tianyu/MagicData
pip install -r requirements.txt

export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_gpt.yaml --split 9_ready_train_gpt4o --mode default --model_name gcp-claude4-sonnet  --output_dir /mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/model_output_v1 --batch_size 1 --use_accel