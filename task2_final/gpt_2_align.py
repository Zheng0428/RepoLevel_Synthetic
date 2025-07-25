import json
from envs import DEFAULT_PATH, ORIGIN_DEFAULT_PATH, NEW_DEFAULT_PATH
import os, re
from utils import get_project_structure_from_scratch
import random
from envs import GENERATE_DATA_PATH, OUTPUT_DATA_PATH, TEST_PATCH_PATH
from temp_testbed import get_all_filenames
import subprocess
import argparse
import ast
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
"""
重新生成一个instance_id
"""


def apply_patches_and_create_buggy_repo(entry, repo_playground, new_repo_playground):
    """
    应用patches并创建包含bug的repo
    
    Args:
        entry: 包含patch信息的数据条目
        repo_playground: 原始repo路径
        new_repo_playground: 新repo路径
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 检查目标文件夹是否存在，如果存在则删除
        if os.path.exists(new_repo_playground):
            print(f"Target directory {new_repo_playground} already exists, removing it...")
            subprocess.run(f"rm -rf {new_repo_playground}", shell=True, check=True)
        
        # 复制原始repo
        subprocess.run(f"cp -r {repo_playground} {new_repo_playground}", shell=True, check=True)
        
        # 获取patches
        origin_patch = entry.get('origin_patch', '')
        patch = entry.get('patch', '')
        
        if not origin_patch or not patch:
            print(f"Warning: Missing patches for {entry.get('instance_id', 'unknown')}")
            return False
        
        # 第一步：应用origin_patch，让repo变成正确状态
        # with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as origin_patch_file:
        #     origin_patch_file.write(origin_patch)
        #     origin_patch_file_path = origin_patch_file.name
        
        # 第二步：创建patch文件用于反向应用
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as patch_file:
            patch_file.write(patch)
            patch_file_path = patch_file.name
        
        try:
            # 应用origin_patch
            # origin_patch_cmd = f'cd {new_repo_playground} && git apply --whitespace=nowarn {origin_patch_file_path}'
            # result_origin = subprocess.run(origin_patch_cmd, shell=True, capture_output=True, text=True)
            
            # if result_origin.returncode != 0:
            #     print(f"Failed to apply origin_patch for {entry.get('instance_id', 'unknown')}: {result_origin.stderr}")
            #     return False
                
            # 反向应用patch，获得错误的repo
            reverse_patch_cmd = f'cd {new_repo_playground} && git apply --reverse --whitespace=nowarn {patch_file_path}'
            result_reverse = subprocess.run(reverse_patch_cmd, shell=True, capture_output=True, text=True)
            
            if result_reverse.returncode != 0:
                print(f"Failed to apply reverse patch for {entry.get('instance_id', 'unknown')}: {result_reverse.stderr}")
                return False
                
            print(f"Successfully applied patches for {entry.get('instance_id', 'unknown')}")
            return True
            
        finally:
            # 清理临时patch文件
            os.unlink(patch_file_path)
            
    except Exception as e:
        print(f"Error applying patches for {entry.get('instance_id', 'unknown')}: {str(e)}")
        return False






def process_data_entry(data, args, save_structure_dir):
    """
    处理单个数据条目的函数，用于多线程处理
    
    Args:
        data: 单个数据条目
        args: 命令行参数
        save_structure_dir: 保存结构的目录
        
    Returns:
        tuple: (success, new_data, error_message)
    """
    try:
        random.seed(data['instance_id'])
        new_data = {}
        new_data['instance_id'] = data['instance_id']
        new_data['repo_base_name'] = os.path.join(data.get('repo').replace('/', '__') + '__' + data.get('base_commit')[:6])
        new_data['origin_instance_id'] = data['instance_id']
        new_data['origin_patch'] = data['patch']
        new_data['patch'] = data['gpt_patch']
        new_data['reserve_patch'] = data['gpt_reverse_patch']
        new_data['repo'] = data['repo']
        new_data['base_commit'] = ''.join(random.choices('0123456789abcdef', k=40))
        new_data['instance_id'] = new_data['repo'].replace("/", "__")+"__"+ new_data['base_commit'][:6]
        new_data['hints_text'] = data['hints_text']
        new_data['test_patch'] = data['test_patch']
        
        if os.path.exists(TEST_PATCH_PATH+new_data['instance_id']+'__test_patch.diff'):
            os.remove(TEST_PATCH_PATH+new_data['instance_id']+'__test_patch.diff')
        
        new_data['problem_statement'] = data['gpt_problem_statement']
        new_data['version'] = data['version']
        new_data['environment_setup_commit'] = data['environment_setup_commit']
        new_data['FAIL_TO_PASS'] = data['FAIL_TO_PASS']
        new_data['PASS_TO_PASS'] = data['PASS_TO_PASS']
        test_patch_files = get_all_filenames(data['gpt_patch'])
        new_data['modified_files'] = test_patch_files["modified"]+test_patch_files['added']+test_patch_files["removed"]
        new_data['extra_related_files'] = data['noise_files']
        new_data['noise_files'] = data['noise_files']

        repo_name = new_data['repo']
        commit_id = new_data['base_commit']
        instance_id = new_data['instance_id']
        
        repo_playground = os.path.join(DEFAULT_PATH, new_data['repo_base_name'])
        new_repo_playground = os.path.join(NEW_DEFAULT_PATH, new_data['instance_id'])
        
        # 应用patches并创建buggy repo
        success = apply_patches_and_create_buggy_repo(new_data, repo_playground, new_repo_playground)
        
        if not success:
            print(f"Skipping {instance_id} due to patch application failure")
            return False, None, f"Patch application failed for {instance_id}"
        
        # 生成项目结构
        if args.save_structure:
            structure = get_project_structure_from_scratch(repo_name, commit_id, instance_id, new_repo_playground)
            structure_file_path = f"{save_structure_dir}/{new_data['instance_id']}.json"
            os.makedirs(save_structure_dir, exist_ok=True)
            with open(structure_file_path, "w") as f_out_structure:
                json.dump(structure, f_out_structure, indent=4)
        
        print(f"Successfully processed {instance_id}")
        return True, new_data, None
        
    except Exception as e:
        error_msg = f"Error processing {data.get('instance_id', 'unknown')}: {str(e)}"
        print(error_msg)
        return False, None, error_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process project structure')
    parser.add_argument('--save-structure', action='store_true', 
                       help='Whether to save project structure to files')
    parser.add_argument('--max-workers', type=int, default=32,
                       help='Maximum number of worker threads (default: 4)')
    args = parser.parse_args()
    
    dataset = []
    input_file = f'{GENERATE_DATA_PATH}/gpt_2_finish_bug_gpt4o.jsonl'
    output_jsonl_file = f'{GENERATE_DATA_PATH}/gpt_3_seg_bug_success_with_noise_gpt4o.jsonl'
    
    if not os.path.exists(TEST_PATCH_PATH):
        os.makedirs(TEST_PATCH_PATH)
    
    # 读取数据集
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            dataset.append(entry)
    
    # save_structure_dir = f'{GENERATE_DATA_PATH}/structure'
    save_structure_dir = '/mnt/bn/tiktok-mm-5/aiic/users/yiming/data/swe-verified/repo_structure'
    os.makedirs(save_structure_dir, exist_ok=True)
    
    # 强制设置save_structure为True
    args.save_structure = True
    
    print(f"Processing {len(dataset)} entries with {args.max_workers} workers...")
    
    # 收集成功处理的结果
    successful_results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_data = {
            executor.submit(process_data_entry, data, args, save_structure_dir): data 
            for data in dataset
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_data):
            data = future_to_data[future]
            try:
                success, result, error_msg = future.result()
                
                if success and result:
                    successful_results.append(result)
                else:
                    failed_count += 1
                    if error_msg:
                        print(f"Failed: {error_msg}")
                        
            except Exception as exc:
                failed_count += 1
                print(f"Generated an exception for {data.get('instance_id', 'unknown')}: {exc}")
    
    # 最后一次性写入所有成功的结果到output_jsonl_file
    print(f"Writing {len(successful_results)} successful results to {output_jsonl_file}...")
    with open(output_jsonl_file, 'w') as f_out:
        for new_data in successful_results:
            f_out.write(json.dumps(new_data) + '\n')
    
    print(f"\nProcessing completed!")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(dataset)}")
        