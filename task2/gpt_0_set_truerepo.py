import json
import re
import os # Import os for file existence checks
import json5, simplejson
import subprocess
from utils import fake_git_repo
from envs import DEFAULT_PATH, TRUE_DEFAULT_PATH
from envs import GENERATE_DATA_PATH, OUTPUT_DATA_PATH
from typing import Dict, List, Optional
from dataclasses import dataclass
from temp_testbed import TempTestbed, get_all_filenames
import tempfile


def generate_true_repo(data):
    """
    根据GPT生成的bug数据，创建临时环境并生成patch
    
    Args:
        data: 原始数据，包含repo、patch等信息
        result: 解析后的GPT响应
        
    Returns:
        dict: 包含gpt_patch和gpt_test_patch的数据
    """
    # 获取基本信息
    repo_name = data.get('repo', '').replace('/', '__') + '__' + data.get('base_commit', '')[:6]
    original_patch = data.get('patch', '')
    
    # 设置repo路径
    source_testbed = os.path.join(DEFAULT_PATH, repo_name)
    
    # 设置正确的repo路径
    true_testbed = os.path.join(TRUE_DEFAULT_PATH, repo_name)
    
    if not os.path.exists(source_testbed):
        print(f"Source testbed not found: {source_testbed}")
        return data
    
    # 获取原始patch修改的文件
    original_patch_files = get_all_filenames(original_patch)
    modified_files = original_patch_files["modified"] + original_patch_files["added"]
    
    
    try:
        with TempTestbed(source_testbed=source_testbed, copy_files=modified_files) as temp_testbed:
            temp_dir = temp_testbed.temp_dir
            # 第一步：应用原始patch，让repo变成正确状态
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as original_patch_file:
                original_patch_file.write(original_patch)
                original_patch_file_path = original_patch_file.name
            
            try:
                # 应用原始patch
                patch_cmd = f'cd {temp_dir} && git apply --whitespace=nowarn {original_patch_file_path}'
                result_apply = subprocess.run(patch_cmd, shell=True, capture_output=True, text=True)
                
                if result_apply.returncode != 0:
                    print(f"Failed to apply original patch: {result_apply.stderr}")
                    return data
                
                # 第二步：读取修复后的文件内容（正确状态）
                fixed_file_contents = {}
                for bug_file in result.buggy_files:
                    bug_file_path = os.path.join(temp_dir, bug_file.file_path)
                    if os.path.exists(bug_file_path):
                        with open(bug_file_path, 'r', encoding='utf-8') as f:
                            fixed_file_contents[bug_file.file_path] = f.read()
                    else:
                        print(f"Warning: File not found after applying original patch: {bug_file.file_path}")
                        fixed_file_contents[bug_file.file_path] = ""
                
                # 第三步：使用fake_git_repo生成从bug到fix的patch
                bug_file_paths = []
                bug_contents = []
                fixed_contents = []
                files_exist_list = []
                
                for bug_file in result.buggy_files:
                    bug_file_paths.append(bug_file.file_path)
                    bug_contents.append(bug_file.code)
                    fixed_contents.append(fixed_file_contents.get(bug_file.file_path, ""))
                    
                    # 检查文件是否存在于临时目录中
                    bug_file_full_path = os.path.join(temp_dir, bug_file.file_path)
                    files_exist_list.append(os.path.exists(bug_file_full_path))
                
                # 生成从bug到fix的patch (bug->fix)
                # 如果所有文件都存在，则files_exist=True，否则False
                all_files_exist = all(files_exist_list)
                gpt_patch = fake_git_repo(
                    file_pathes=bug_file_paths,
                    old_contents=bug_contents,
                    new_contents=fixed_contents,
                    files_exist=all_files_exist
                )

                gpt_reverse_patch = fake_git_repo(
                    file_pathes=bug_file_paths,
                    old_contents=fixed_contents,
                    new_contents=bug_contents,
                    files_exist=all_files_exist
                )
                
                # 第四步：生成unittest patch (新建文件)，使用解决冲突后的路径
                gpt_test_patch = fake_git_repo(
                    file_pathes=[resolved_unittest_path],
                    old_contents=[""],
                    new_contents=[result.unittest_code],
                    files_exist=False
                )
                
                # 更新数据
                data['gpt_patch'] = gpt_patch
                data['gpt_reverse_patch'] = gpt_reverse_patch
                data['gpt_test_patch'] = gpt_test_patch
                data['gpt_problem_statement'] = result.problem_statement
                
            finally:
                # 清理临时patch文件
                os.unlink(original_patch_file_path)
                
    except Exception as e:
        print(f"Error processing bug data: {str(e)}")
        
    return data

# --- Modified process_jsonl function ---
def process_jsonl(input_path, output_path):
    default_path = "/opt/tiger/expr/repo_commit"
    num_processed_lines_in_file = 0
    # --- Loop through each input file path provided ---
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}. Skipping.")
        input_path = input_path.replace(".tmp", "") # Remove the.tmp extension if it exists
    aggregated_data = []  # List to store aggregated results
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        # --- Process each line in the current file ---
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                generate_true_repo(data)     
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                continue
            
        print(f"Finished processing {num_processed_lines_in_file} lines from {input_path}.")

# --- Main execution block ---
if __name__ == "__main__":
    # --- Define Input and Output Paths ---
    # MODIFIED: Define a LIST of input files
    input_jsonl_files = f'{OUTPUT_DATA_PATH}/gpt-4o-2024-11-20_yimi_prompt_generate_bug_v1.jsonl.tmp'
    process_jsonl(input_jsonl_files)
    