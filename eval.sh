#!/bin/bash
# 用于在Firejail中执行补丁测试的脚本
# 注意所有source data都是readonly的，不允许写入

set -xuo pipefail

# 参数检查
if [ $# -lt 3 ]; then
    show_usage
fi

# 参数设置
TESTBED_PATH="$1" #临时repo路径
TEST_COMMAND="$2" # pytest xxx.py
TEST_PATCH_PATH="$3" # xx/xxx.diff -> test.py
PATCH_PATH="${4:-}" # 如果提供了第4个参数，使用它；否则设为空字符串   !!! repair code  测有bug和没bug状态



# 进入测试目录
cd  "$TESTBED_PATH" || exit 1

echo ">>>>> Applying test patch"
git apply --verbose $TEST_PATCH_PATH
echo ">>>>> Applied test patch successfully"



# exam PATCH_PATH exists
if [ -z "$PATCH_PATH" ]; then
    echo "No patch file provided, skipping patch application."
else
    if [ ! -f "$PATCH_PATH" ]; then
        echo "ERROR: Patch file not found: $PATCH_PATH"
        exit 1
    fi
fi
# if PATCH_PATH exists，apply patch
#apply prediction patch
if [ -n "$PATCH_PATH" ] && [ -f "$PATCH_PATH" ]; then  
    echo ">>>>> Applying prediction patch"
    git apply --verbose $PATCH_PATH

echo ">>>>> Prediction patch applied successfully"
fi

# 执行测试
: ">>>>> Start Test Output"
$TEST_COMMAND
: '>>>>> End Test Output'

