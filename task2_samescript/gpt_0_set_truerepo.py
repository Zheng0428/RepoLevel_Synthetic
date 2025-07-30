import json
import os
import subprocess
import shutil
from envs import DEFAULT_PATH, ORIGIN_DEFAULT_PATH
from envs import GENERATE_DATA_PATH, TRUE_PROJECT_FILE_LOC
from temp_testbed import get_all_filenames
import tempfile
from utils import get_project_structure_from_scratch
import argparse

def generate_true_repo(data):
    """
    应用原始patch并保存到true_testbed路径
    
    Args:
        data: 原始数据，包含repo、patch等信息
        
    Returns:
        dict: 原始数据
    """
    # 获取基本信息
    repo_name = data.get('repo', '').replace('/', '__') + '__' + data.get('base_commit', '')[:6]
    original_patch = data.get('patch', '')
    
    if not original_patch:
        print(f"No patch found for repo: {repo_name}")
        return data
    
    # 设置repo路径
    source_testbed = os.path.join(ORIGIN_DEFAULT_PATH, repo_name)
    true_testbed = os.path.join(DEFAULT_PATH, repo_name)
    
    if not os.path.exists(source_testbed):
        print(f"Source testbed not found: {source_testbed}")
        return data
    
    try:
        # 确保true_testbed目录存在
        os.makedirs(DEFAULT_PATH, exist_ok=True)
        
        # 如果true_testbed已存在，先删除
        if os.path.exists(true_testbed):
            shutil.rmtree(true_testbed)
        
        # 复制source_testbed到true_testbed
        shutil.copytree(source_testbed, true_testbed)
        print(f"Copied {source_testbed} to {true_testbed}")
        
        # 应用原始patch
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
            patch_file.write(original_patch)
            patch_file_path = patch_file.name
        
        try:
            patch_cmd = f'cd {true_testbed} && git apply --whitespace=nowarn {patch_file_path}'
            result_apply = subprocess.run(patch_cmd, shell=True, capture_output=True, text=True)
            
            if result_apply.returncode != 0:
                print(f"Failed to apply patch to {true_testbed}: {result_apply.stderr}")
                return data
            
            print(f"Successfully applied patch to {true_testbed}")
            
        finally:
            # 清理临时patch文件
            os.unlink(patch_file_path)
            
    except Exception as e:
        print(f"Error processing repo {repo_name}: {str(e)}")
        
    return data


def process_jsonl(input_path, args):
    """处理JSONL文件中的每一行数据"""
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found: {input_path}. Skipping.")
        input_path = input_path.replace(".tmp", "")  # Remove the .tmp extension if it exists
        if not os.path.exists(input_path):
            print(f"Warning: Input file still not found: {input_path}. Skipping.")
            return
    
    num_processed_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                generate_true_repo(data)
                num_processed_lines += 1
                if args.save_structure:
                    repo_playground = os.path.join(DEFAULT_PATH, data.get('repo', '').replace('/', '__') + '__' + data.get('base_commit', '')[:6])
                    structure = get_project_structure_from_scratch(data['repo'], data['base_commit'], data['instance_id'], repo_playground)
                    structure_file_path = f"{TRUE_PROJECT_FILE_LOC}/{data['instance_id']}.json"
                    os.makedirs(TRUE_PROJECT_FILE_LOC, exist_ok=True)
                    with open(structure_file_path, "w") as f_out_structure:
                        json.dump(structure, f_out_structure, indent=4)
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}")
                continue
    
    print(f"Finished processing {num_processed_lines} lines from {input_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process project structure')
    parser.add_argument('--save-structure', action='store_true', 
                       help='Whether to save project structure to files')
    args = parser.parse_args()
    
    # 定义输入文件路径
    input_jsonl_file = f'{GENERATE_DATA_PATH}/gpt-4o-2024-11-20_yimi_three_shot_same_test.jsonl.tmp'
    process_jsonl(input_jsonl_file, args)
    