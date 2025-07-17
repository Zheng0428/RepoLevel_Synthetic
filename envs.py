# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

"""Defines the environment variables that are used by the agentless module."""

import os
import shutil

# Whether or not to enable the thinking mode (yes or no)
# THINKING = os.environ.get("THINKING", "no") == "yes"
# If enabled, what's the enclosing tag for fetching the answer from output
ANSWER_START_TAG = os.environ.get("ANSWER_START_TAG", "<solution>")
ANSWER_END_TAG = os.environ.get("ANSWER_END_TAG", "</solution>")
# Where to put temporary generated files during processing & reranking
PLAYGROUND_DIR = os.getenv("PLAYGROUND_DIR", "/opt/tiger/Github-Repo/playground")
# if os.path.exists(PLAYGROUND_DIR):
#     shutil.rmtree(PLAYGROUND_DIR)
os.makedirs(PLAYGROUND_DIR, exist_ok=True)
# Preprocessed structure information for each SWE-Bench problem
# Please download it from the original Agentless repository
# https://github.com/OpenAutoCoder/Agentless/tree/main
PROJECT_FILE_LOC = os.environ.get("PROJECT_FILE_LOC", "/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-verified/repo_structure")
PATCH_FILES_JSON = os.environ.get("PATCH_FILRES", "/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-verified/verified/swe-bench-verified-patch-files.json")

# TESTBED_DIR = os.environ.get("TESTBED_DIR", "/opt/tiger/expr/testbed")

EXPR_PATH = os.getenv("EXPR_PATH", "/opt/tiger/sweb")
ENV_DIR=f"{EXPR_PATH}/conda/"
REPO_COMMIT_DIR=f"{EXPR_PATH}/repo/"
# TEST_PATCH_PATH="/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra/${TASK_ID}/test_patch.diff"
# TEST_PATCH_DIR="/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-bench-extra/"

import time
# Create a timestamp string in a readable format  
time_string = time.strftime("%Y-%m-%d")  # Format: 2025-04-05_14-30-45  
# Define the log path with the timestamp  
RUN_NAME = os.environ.get('RUN_NAME','sweb_eval')
ORIGIN_DEFAULT_PATH = os.environ.get('ORIGIN_DEFAULT_PATH', '/opt/tiger/expr/repo_commit')
DEFAULT_PATH = "/opt/tiger/expr/true_repo_commit"
NEW_DEFAULT_PATH = "/opt/tiger/expr/synthetic_repo_commit"
GENERATE_DATA_PATH='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data'
LOG_PATH='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_Synthetic/data'
OUTPUT_DATA_PATH='/mnt/bn/tiktok-mm-5/aiic/users/tianyu/RepoLevel_BugFix_yimi/model_output_v1'
TEST_PATCH_PATH='/opt/tiger/expr/test_patch/'



# Define the API endpoint and API key
BASE_URL = "https://search-va.byteintl.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi" #http://maas.byteintl.net/gateway/v1/chat/completions
API_KEY = "54nhP5uBXv7iWgHJ4bWMD90Nwkn09BXN"  # Replace with your actual API key
MODEL = "gpt-4o-2024-11-20" # gcp-claude37-sonnet/gemini-2.5-pro-preview-05-06/gpt-4o-2024-11-20
MAX_TOKENS = 16000
SYSTEM_PROMPT = "You are a very helpful assistant."
